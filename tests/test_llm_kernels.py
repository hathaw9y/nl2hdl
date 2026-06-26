from pathlib import Path
import json
import math
import os
import struct
import sys
import types

import pytest

import nl2hdl.llm_kernels as llm_kernels_module
from nl2hdl.cli import build_parser, main as cli_main
from nl2hdl.config import load_config
from nl2hdl.gptq import inspect_gptq_checkpoint_metadata
from nl2hdl.llm_agent import run_llm_agent
from nl2hdl.llm_kernels import (
    _parse_kernel_vivado_utilization,
    _parse_zcu104_vivado_utilization,
    build_zcu104_board_shell_constraints_package,
    build_model_level_execution_harness_report,
    build_gptq_weight_layout_preflight,
    build_hdl_task_manifest,
    build_llama_semantic_graph,
    emit_inspect_artifacts,
    emit_kernel,
    run_zcu104_board_wrapper_axi_bridge_agent,
    write_model_level_execution_harness_report,
)
from nl2hdl.subagent_tasks import (
    build_board_zcu104_signoff_evidence_agent_task,
    build_board_zcu104_signoff_evidence_template,
    build_board_zcu104_signoff_readiness_report,
    build_codex_spawn_instructions,
    build_full_model_target_rtl_generator_agent_task,
    build_full_target_llama_accelerator_artifact_agent_task,
    build_full_llama_execution_evidence_agent_task,
    build_full_llama_execution_evidence_template,
    build_full_llama_execution_readiness_report,
    build_model_level_execution_harness_agent_task,
    build_parent_feedback_loop_state,
    build_target_scale_child_packet_agent_task,
    build_zcu104_board_wrapper_axi_bridge_agent_task,
    build_hdl_subagent_dispatch_plan,
    build_hdl_subagent_execution_manifest,
    build_hdl_subagent_packets,
    build_hdl_subagent_spawn_ledger,
    build_hdl_subagent_spawn_ledger_markdown,
    build_hdl_subagent_skill_update_draft,
    build_hdl_subagent_wave_status,
    build_skill_update_draft_markdown,
    build_target_evidence_execution_manifest,
    run_board_zcu104_signoff_evidence_agent,
    run_full_llama_execution_evidence_agent,
    TARGET_SCALE_CHILD_PACKET_TASKS,
)


def _full_llama_block_mlir() -> str:
    return """module {
  func.func @llama_decoder_block(%arg0: tensor<1x4xi8>) -> (tensor<1x4xi8>) {
    %0 = "llm.RMSNorm"(%arg0) {onnx_node_name = "input_layernorm"} : (tensor<1x4xi8>) -> tensor<1x4xi8>
    %1 = "onnx.MatMul"(%0, %q_weight) {onnx_node_name = "q_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %2 = "onnx.MatMul"(%0, %k_weight) {onnx_node_name = "k_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %3 = "onnx.MatMul"(%0, %v_weight) {onnx_node_name = "v_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %4 = "llm.RoPE"(%1, %2) {onnx_node_name = "rope_qk"} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
    %5 = "llm.AttentionControl"(%4, %3) {onnx_node_name = "attention_scores_softmax_kv"} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi8>
    %6 = "onnx.MatMul"(%5, %o_weight) {onnx_node_name = "o_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %7 = "onnx.Add"(%6, %arg0) {onnx_node_name = "attention_residual"} : (tensor<1x4xi32>, tensor<1x4xi8>) -> tensor<1x4xi32>
    %8 = "llm.RMSNorm"(%7) {onnx_node_name = "post_attention_layernorm"} : (tensor<1x4xi32>) -> tensor<1x4xi8>
    %9 = "onnx.MatMul"(%8, %gate_weight) {onnx_node_name = "gate_proj"} : (tensor<1x4xi8>, tensor<8x4xi4>) -> tensor<1x8xi32>
    %10 = "onnx.MatMul"(%8, %up_weight) {onnx_node_name = "up_proj"} : (tensor<1x4xi8>, tensor<8x4xi4>) -> tensor<1x8xi32>
    %11 = "llm.SiLU"(%9) {onnx_node_name = "silu_gate"} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %12 = "onnx.Mul"(%11, %10) {onnx_node_name = "swiglu_multiply"} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %13 = "onnx.MatMul"(%12, %down_weight) {onnx_node_name = "down_proj"} : (tensor<1x8xi32>, tensor<4x8xi4>) -> tensor<1x4xi32>
    %14 = "onnx.Add"(%13, %7) {onnx_node_name = "mlp_residual"} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
    return %14 : tensor<1x4xi32>
  }
}
"""


def _hf_export_style_llama_block_mlir() -> str:
    return """module {
  func.func @llama_decoder_block(%arg0: tensor<1x4xi8>) -> (tensor<1x4xi8>) {
    %0 = "llm.RMSNorm"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4xi8> loc("/model/layers.0/input_layernorm/RMSNorm")
    %1 = "onnx.MatMul"(%0, %q_weight) : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn/q_proj/MatMul")
    %2 = "onnx.MatMul"(%0, %k_weight) : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn/k_proj/MatMul")
    %3 = "onnx.MatMul"(%0, %v_weight) : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn/v_proj/MatMul")
    %4 = "llm.RoPE"(%1, %2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn/rotary_emb/apply_rotary_pos_emb")
    %5 = "llm.AttentionControl"(%4, %3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi8> loc("/model/layers.0/self_attn/Softmax")
    %6 = "onnx.MatMul"(%5, %o_weight) : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn/o_proj/MatMul")
    %7 = "onnx.Add"(%6, %arg0) : (tensor<1x4xi32>, tensor<1x4xi8>) -> tensor<1x4xi32> loc("/model/layers.0/self_attn_residual/Add")
    %8 = "llm.RMSNorm"(%7) : (tensor<1x4xi32>) -> tensor<1x4xi8> loc("/model/layers.0/post_attention_layernorm/RMSNorm")
    %9 = "onnx.MatMul"(%8, %gate_weight) : (tensor<1x4xi8>, tensor<8x4xi4>) -> tensor<1x8xi32> loc("/model/layers.0/mlp/gate_proj/MatMul")
    %10 = "onnx.MatMul"(%8, %up_weight) : (tensor<1x4xi8>, tensor<8x4xi4>) -> tensor<1x8xi32> loc("/model/layers.0/mlp/up_proj/MatMul")
    %11 = "llm.SiLU"(%9) : (tensor<1x8xi32>) -> tensor<1x8xi32> loc("/model/layers.0/mlp/act_fn/SiLU")
    %12 = "onnx.Mul"(%11, %10) : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32> loc("/model/layers.0/mlp/gate_multiply/Mul")
    %13 = "onnx.MatMul"(%12, %down_weight) : (tensor<1x8xi32>, tensor<4x8xi4>) -> tensor<1x4xi32> loc("/model/layers.0/mlp/down_proj/MatMul")
    %14 = "onnx.Add"(%13, %7) : (tensor<1x8xi32>, tensor<1x4xi32>) -> tensor<1x4xi32> loc("/model/layers.0/final_residual/Add")
    return %14 : tensor<1x4xi32>
  }
}
"""


def test_kernel_and_board_vivado_utilization_parsers_keep_separate_schemas(tmp_path: Path):
    kernel_report = tmp_path / "utilization.rpt"
    kernel_report.write_text(
        "| LUT as Logic | 12 | 0 | 0 | 230400 | 0.01 |\n"
        "| CLB Registers | 34 | 0 | 0 | 460800 | 0.01 |\n"
        "| Block RAM Tile | 2 | 0 | 0 | 312 | 0.64 |\n"
        "| DSPs | 5 | 0 | 0 | 1728 | 0.29 |\n"
        "| Bonded IOB | 9 | 0 | 0 | 464 | 1.94 |\n",
        encoding="utf-8",
    )
    board_report_text = (
        "| CLB LUTs | 464 | 0 | 0 | 230400 | 0.20 |\n"
        "| CLB Registers | 706 | 0 | 0 | 460800 | 0.15 |\n"
        "| Block RAM Tile | 0 | 0 | 0 | 312 | 0.00 |\n"
        "| DSPs | 0 | 0 | 0 | 1728 | 0.00 |\n"
    )

    kernel = _parse_kernel_vivado_utilization(kernel_report)
    board = _parse_zcu104_vivado_utilization(board_report_text)

    assert kernel["lut_as_logic"]["used"] == 12
    assert kernel["clb_registers"]["used"] == 34
    assert kernel["block_ram_tile"]["used"] == 2
    assert kernel["dsps"]["used"] == 5
    assert kernel["bonded_iob"]["used"] == 9
    assert board == {"lut": 464.0, "ff": 706.0, "dsp": 0.0, "bram": 0.0, "uram": None}


def _sample_gptq_payload_probe(words: list[str] | None = None, projection: str = "q_proj") -> dict:
    probe_words = words or [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
        "0x13121110",
        "0x17161514",
        "0x1b1a1918",
        "0x1f1e1d1c",
    ]
    return {
        "artifact": "gptq_payload_probe",
        "status": "sampled",
        "projection": projection,
        "sample_bytes_requested": 32,
        "sampled_tensor_count": 3,
        "required_tensor_count": 3,
        "qweight_payload_words32_le_hex": probe_words,
        "qweight_payload_word_count": len(probe_words),
        "target_checkpoint_payload_dependency": "satisfied_by_payload_probe",
        "does_not_claim": [
            "full_checkpoint_tensor_materialization",
            "numeric_GPTQ_correctness",
            "checkpoint_specific_qweight_order_correctness",
            "full_qweight_payload_streaming",
            "full_LLaMA_execution",
        ],
    }


def _sample_aggregate_gptq_payload_probe(projection_names: list[str]) -> tuple[dict, dict[str, list[str]]]:
    words_by_projection = {}
    probes = {}
    for projection_idx, projection_name in enumerate(projection_names):
        payload = bytes((projection_idx * 32 + byte_idx) & 0xFF for byte_idx in range(32))
        words = [
            f"0x{int.from_bytes(payload[idx : idx + 4], 'little'):08x}"
            for idx in range(0, len(payload), 4)
        ]
        words_by_projection[projection_name] = words
        probes[projection_name] = _sample_gptq_payload_probe(words=words, projection=projection_name)
    aggregate = {
        **probes["q_proj"],
        "projection_payload_probes": probes,
        "projection_payload_probe_count": len(probes),
        "sampled_projection_payload_probe_count": len(probes),
        "required_projection_payload_probe_count": len(projection_names),
        "all_projection_payload_dependency": "satisfied_by_payload_probe",
    }
    return aggregate, words_by_projection


def _write_fake_safetensors_header(path: Path, keys: list[str]) -> None:
    header = {"__metadata__": {"format": "pt"}}
    for idx, key in enumerate(keys):
        header[key] = {
            "dtype": "I32",
            "shape": [1],
            "data_offsets": [idx * 4, (idx + 1) * 4],
        }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * (4 * len(keys))))


def _write_header_only_safetensors(path: Path, entries: list[tuple[str, str, list[int], int]]) -> None:
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key, dtype, shape, byte_count in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _write_safetensors_with_selected_payload(
    path: Path,
    entries: list[tuple[str, str, list[int], int]],
    payload_prefixes: dict[str, bytes],
) -> None:
    header = {"__metadata__": {"format": "pt"}}
    offsets = {}
    offset = 0
    for key, dtype, shape, byte_count in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        offsets[key] = (offset, offset + byte_count)
        offset += byte_count
    raw = json.dumps(header).encode("utf-8")
    selected_ends = [offsets[key][1] for key in payload_prefixes]
    payload = bytearray(max(selected_ends, default=0))
    for key, prefix in payload_prefixes.items():
        start, end = offsets[key]
        if len(prefix) > end - start:
            raise ValueError(f"payload prefix for {key} exceeds declared tensor byte count")
        payload[start : start + len(prefix)] = prefix
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))


def _target_gptq_header_entries(graph: dict, group_size: int = 128) -> list[tuple[str, str, list[int], int]]:
    entries = []
    values_per_i32 = 8
    for name in graph["partition"]["gemm"]:
        shape = graph["projection_shapes"][name]
        rows = int(shape["rows"])
        cols = int(shape["cols"])
        groups = math.ceil(cols / group_size)
        qweight_shape = [math.ceil(cols / values_per_i32), rows]
        qzeros_shape = [groups, math.ceil(rows / values_per_i32)]
        scales_shape = [groups, rows]
        base = f"model.layers.0.self_attn.{name}" if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else f"model.layers.0.mlp.{name}"
        entries.extend(
            [
                (f"{base}.qweight", "I32", qweight_shape, qweight_shape[0] * qweight_shape[1] * 4),
                (f"{base}.qzeros", "I32", qzeros_shape, qzeros_shape[0] * qzeros_shape[1] * 4),
                (f"{base}.scales", "F16", scales_shape, scales_shape[0] * scales_shape[1] * 2),
            ]
        )
    return entries


def _target_gptq_alias_header_entries(graph: dict, group_size: int = 128) -> list[tuple[str, str, list[int], int]]:
    entries = []
    values_per_i32 = 8
    for name in graph["partition"]["gemm"]:
        shape = graph["projection_shapes"][name]
        rows = int(shape["rows"])
        cols = int(shape["cols"])
        groups = math.ceil(cols / group_size)
        qweight_shape = [math.ceil(cols / values_per_i32), rows]
        qzeros_shape = [groups, math.ceil(rows / values_per_i32)]
        scales_shape = [groups, rows]
        base = f"model.layers.0.self_attn.{name}" if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else f"model.layers.0.mlp.{name}"
        entries.extend(
            [
                (f"{base}.q_weight", "I32", qweight_shape, qweight_shape[0] * qweight_shape[1] * 4),
                (f"{base}.zeros", "I32", qzeros_shape, qzeros_shape[0] * qzeros_shape[1] * 4),
                (f"{base}.scale", "F16", scales_shape, scales_shape[0] * scales_shape[1] * 2),
            ]
        )
    return entries


def _write_target_payload_gptq_checkpoint(tmp_path: Path, graph: dict) -> Path:
    gptq_dir = tmp_path / "target-payload-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    payload_prefixes = {}
    for projection_idx, projection_name in enumerate(graph["partition"]["gemm"]):
        base = (
            f"model.layers.0.self_attn.{projection_name}"
            if projection_name in {"q_proj", "k_proj", "v_proj", "o_proj"}
            else f"model.layers.0.mlp.{projection_name}"
        )
        payload_prefixes[f"{base}.qweight"] = bytes((projection_idx * 16 + byte_idx) & 0xFF for byte_idx in range(64))
        payload_prefixes[f"{base}.qzeros"] = bytes([0x11 + projection_idx] * 64)
        payload_prefixes[f"{base}.scales"] = bytes([0x22 + projection_idx] * 64)
    _write_safetensors_with_selected_payload(
        gptq_dir / "model.safetensors",
        _target_gptq_header_entries(graph),
        payload_prefixes,
    )
    return gptq_dir


def _target_gptq_metadata_from_graph(graph: dict, bits: int = 4, group_size: int = 128) -> dict:
    metadata = {
        "status": "parsed",
        "bits": bits,
        "group_size": group_size,
        "projection_metadata": [],
    }
    for key, dtype, shape, byte_count in _target_gptq_header_entries(graph, group_size=group_size):
        projection = key.split(".")[-2]
        entry = next((item for item in metadata["projection_metadata"] if item["name"] == projection), None)
        if entry is None:
            entry = {
                "name": projection,
                "has_qweight": False,
                "has_qzeros": False,
                "has_scales": False,
                "has_g_idx": False,
                "keys": {"qweight": [], "qzeros": [], "scales": [], "g_idx": [], "other": []},
                "tensor_summaries": {},
            }
            metadata["projection_metadata"].append(entry)
        kind = key.rsplit(".", 1)[-1]
        entry[f"has_{kind}"] = True
        entry["keys"][kind].append(key)
        entry["tensor_summaries"][key] = {
            "file": "model.safetensors",
            "key": key,
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, byte_count],
            "byte_count": byte_count,
            "metadata_status": "header_only_no_tensor_payload",
        }
    metadata["quantized_projection_metadata_count"] = len(metadata["projection_metadata"])
    metadata["complete_gptq_projection_metadata_count"] = len(metadata["projection_metadata"])
    metadata["projection_metadata_count"] = len(metadata["projection_metadata"])
    metadata["tensor_summary_count"] = sum(len(item["tensor_summaries"]) for item in metadata["projection_metadata"])
    metadata["tensor_summary_source"] = "safetensors_header"
    return metadata


def test_emit_inspect_artifacts_partitions_gemm_and_non_gemm(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    (
        _,
        analysis_path,
        semantic_graph_path,
        gptq_metadata_path,
        task_manifest_path,
        subagent_tasks_path,
        subagent_prompt_dir,
        subagent_dispatch_plan_path,
        subagent_wave_status_path,
        subagent_execution_manifest_path,
    ) = emit_inspect_artifacts(
        tmp_path,
        "meta-llama/Llama-3.2-1B",
        cfg,
    )
    text = analysis_path.read_text(encoding="utf-8")
    assert '"gemm"' in text
    assert '"non_gemm"' in text
    assert '"RMSNorm"' in text
    assert '"MatMul"' in text
    semantic_graph = json.loads(semantic_graph_path.read_text(encoding="utf-8"))
    assert semantic_graph["coverage_level"] == "target_model_semantic_inspect_no_checkpoint_weights"
    assert semantic_graph["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert semantic_graph["model"]["hidden_size"] == 2048
    assert semantic_graph["model"]["decoder_layers"] == 16
    assert semantic_graph["projection_shapes"]["q_proj"] == {"rows": 2048, "cols": 2048}
    assert semantic_graph["projection_shapes"]["k_proj"] == {"rows": 512, "cols": 2048}
    assert "q_proj" in semantic_graph["partition"]["gemm"]
    assert "input_layernorm" in semantic_graph["partition"]["non_gemm"]
    assert semantic_graph["gptq_metadata_requirements"]["checkpoint_metadata_required"] == [
        "group_size",
        "scales",
        "zero_points",
        "packing_order",
    ]
    assert semantic_graph["next_hdl_contract_inputs"]["token_loop_child_fixture"] == "token_loop_decoder_block_fixture"
    assert (tmp_path / "mlir_model_analysis_readiness.json").exists()
    mlir_readiness = json.loads((tmp_path / "mlir_model_analysis_readiness.json").read_text(encoding="utf-8"))
    assert mlir_readiness["artifact"] == "mlir_model_analysis_readiness"
    assert mlir_readiness["status"] == "blocked"
    assert mlir_readiness["analysis_source"] == "synthetic_llama_block_mlir"
    assert "v_proj" in mlir_readiness["missing_semantic_gemm_ops"]
    assert "attention_residual" in mlir_readiness["missing_semantic_non_gemm_ops"]
    gptq_metadata = json.loads(gptq_metadata_path.read_text(encoding="utf-8"))
    assert gptq_metadata["artifact"] == "gptq_checkpoint_metadata"
    assert gptq_metadata["status"] in {"unavailable", "metadata_json_without_quant_fields", "parsed"}
    assert (tmp_path / "gptq_payload_probe.json").exists()
    payload_probe = json.loads((tmp_path / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    assert payload_probe["artifact"] == "gptq_payload_probe"
    assert payload_probe["status"] in {"unavailable", "partial", "sampled"}
    assert (tmp_path / "projection_weight_stream_plan.json").exists()
    projection_stream_plan = json.loads((tmp_path / "projection_weight_stream_plan.json").read_text(encoding="utf-8"))
    assert projection_stream_plan["artifact"] == "projection_weight_stream_plan"
    assert projection_stream_plan["projection_count"] == 7
    assert projection_stream_plan["attention_projection_order"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert projection_stream_plan["mlp_projection_order"] == ["gate_proj", "up_proj", "down_proj"]
    assert "DDR_controller_integration" in projection_stream_plan["does_not_claim"]
    assert (tmp_path / "target_readiness_report.json").exists()
    readiness = json.loads((tmp_path / "target_readiness_report.json").read_text(encoding="utf-8"))
    assert readiness["artifact"] == "target_readiness_report"
    assert readiness["status"] in {"target_blocked", "target_ready"}
    assert readiness["safe_to_spawn_bounded_subagents"] is True
    assert readiness["safe_to_claim_target_accelerator"] is False
    assert any(gate["gate"] == "subagent_dispatch" for gate in readiness["gates"])
    assert (tmp_path / "target_blocker_remediation_plan.json").exists()
    assert (tmp_path / "target_blocker_remediation_plan.md").exists()
    remediation = json.loads((tmp_path / "target_blocker_remediation_plan.json").read_text(encoding="utf-8"))
    assert remediation["artifact"] == "target_blocker_remediation_plan"
    assert remediation["status"] == "blocked"
    assert "real_mlir_model_analysis" in remediation["blocked_target_tasks"]
    assert "--mlir-graph" in remediation["canonical_full_preflight_command"]
    assert "--gptq-checkpoint" in remediation["canonical_full_preflight_command"]
    mlir_step = next(step for step in remediation["remediation_steps"] if step["task_id"] == "real_mlir_model_analysis")
    assert "mlir_model_analysis_readiness.status == passed" in mlir_step["required_evidence"]
    assert "Parent agent does not write HDL" in (tmp_path / "target_blocker_remediation_plan.md").read_text(
        encoding="utf-8"
    )
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    assert task_manifest["coverage_level"] == "semantic_graph_to_bounded_hdl_task_mapping"
    assert task_manifest["mlir_model_analysis"]["status"] == "blocked"
    assert task_manifest["mlir_model_analysis"]["analysis_source"] == "synthetic_llama_block_mlir"
    assert any(task["task_id"] == "real_mlir_model_analysis" for task in task_manifest["blocked_target_tasks"])
    assert task_manifest["task_counts"]["projection_tasks"] == 7
    assert task_manifest["task_counts"]["integration_tasks"] == 42
    assert task_manifest["gptq_checkpoint_metadata"]["artifact"] == "gptq_checkpoint_metadata.json"
    assert task_manifest["gptq_checkpoint_metadata"]["status"] == gptq_metadata["status"]
    assert "raw_projection_key_count" in task_manifest["gptq_checkpoint_metadata"]
    assert "tensor_summary_count" in task_manifest["gptq_checkpoint_metadata"]
    assert "weight_layout_preflight_status" in task_manifest["gptq_checkpoint_metadata"]
    assert task_manifest["gptq_weight_layout_preflight"]["artifact"] == "gptq_weight_layout_preflight"
    layout_preflight = json.loads((tmp_path / "gptq_weight_layout_preflight.json").read_text(encoding="utf-8"))
    assert layout_preflight["status"] == task_manifest["gptq_weight_layout_preflight"]["status"]
    assert task_manifest["projection_tasks"][0]["agent_role"] == "gemm_kernel_agent"
    assert task_manifest["projection_tasks"][0]["contract"] == "docs/target_scale_projection_streaming_contract.md"
    assert "gptq_weight_layout_preflight" in task_manifest["projection_tasks"][0]
    assert "gptq_payload_probe" in task_manifest["projection_tasks"][0]
    assert "target_checkpoint_payload_dependency" in task_manifest["projection_tasks"][0]
    assert task_manifest["gptq_payload_probe"]["artifact"] == "gptq_payload_probe.json"
    assert task_manifest["projection_weight_stream_plan"] == projection_stream_plan
    assert task_manifest["projection_tasks"][0]["target_checkpoint_layout_dependency"] in {
        "blocked_by_real_gptq_weight_layout_preflight",
        "satisfied_by_header_preflight",
    }
    assert any(task["task_id"] == "token_loop_decoder_block_fixture" for task in task_manifest["integration_tasks"])
    assert any(task["task_id"] == "projection_axi_read_command_adapter" for task in task_manifest["integration_tasks"])
    assert any(
        task["task_id"] == "projection_k_proj_axi_read_command_adapter"
        for task in task_manifest["integration_tasks"]
    )
    assert any(task["task_id"] == "projection_axi_read_data_channel_adapter" for task in task_manifest["integration_tasks"])
    assert any(
        task["task_id"] == "projection_k_proj_axi_read_data_channel_adapter"
        for task in task_manifest["integration_tasks"]
    )
    assert any(task["task_id"] == "projection_axi_read_transaction_adapter" for task in task_manifest["integration_tasks"])
    assert any(
        task["task_id"] == "projection_k_proj_axi_read_transaction_adapter"
        for task in task_manifest["integration_tasks"]
    )
    assert any(task["task_id"] == "projection_axi_stream_integration" for task in task_manifest["integration_tasks"])
    assert any(
        task["task_id"] == "projection_k_proj_axi_stream_integration"
        for task in task_manifest["integration_tasks"]
    )
    assert any(task["task_id"] == "decoder_child_axi_attention_datapath" for task in task_manifest["integration_tasks"])
    decoder_child_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "decoder_child_axi_attention_datapath"
    )
    assert decoder_child_task["attention_projection_stream_tasks"] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
    ]
    assert decoder_child_task["child_tasks"] == [
        "rmsnorm_rope_source_path",
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
        "attention_kv_cache_fixture",
    ]
    assert set(decoder_child_task["target_attention_projection_streams"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert any(task["task_id"] == "layer_fsm_axi_attention_fixture" for task in task_manifest["integration_tasks"])
    assert any(task["task_id"] == "top_fsm_axi_attention_fixture" for task in task_manifest["integration_tasks"])
    assert any(task["task_id"] == "token_loop_axi_attention_fixture" for task in task_manifest["integration_tasks"])
    decoder_axi_block_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "decoder_block_axi_attention_mlp_fixture"
    )
    assert decoder_axi_block_task["contract"] == "docs/decoder_block_axi_attention_mlp_fixture_contract.md"
    assert decoder_axi_block_task["child_tasks"] == [
        "decoder_child_axi_attention_datapath",
        "residual_mlp_fixture",
    ]
    assert decoder_axi_block_task["attention_projection_stream_tasks"] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
    ]
    assert any(task["task_id"] == "layer_fsm_axi_decoder_block_fixture" for task in task_manifest["integration_tasks"])
    layer_axi_decoder_block_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "layer_fsm_axi_decoder_block_fixture"
    )
    assert layer_axi_decoder_block_task["contract"] == "docs/layer_fsm_axi_decoder_block_fixture_contract.md"
    assert layer_axi_decoder_block_task["child_tasks"] == ["decoder_block_axi_attention_mlp_fixture"]
    assert any(task["task_id"] == "top_fsm_axi_decoder_block_fixture" for task in task_manifest["integration_tasks"])
    top_axi_decoder_block_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "top_fsm_axi_decoder_block_fixture"
    )
    assert top_axi_decoder_block_task["contract"] == "docs/top_fsm_axi_decoder_block_fixture_contract.md"
    assert top_axi_decoder_block_task["child_tasks"] == ["layer_fsm_axi_decoder_block_fixture"]
    assert any(task["task_id"] == "token_loop_axi_decoder_block_fixture" for task in task_manifest["integration_tasks"])
    token_loop_axi_decoder_block_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "token_loop_axi_decoder_block_fixture"
    )
    assert token_loop_axi_decoder_block_task["contract"] == "docs/token_loop_axi_decoder_block_fixture_contract.md"
    assert token_loop_axi_decoder_block_task["child_tasks"] == ["top_fsm_axi_decoder_block_fixture"]
    assert any(task["task_id"] == "model_fsm_axi_decoder_block_fixture" for task in task_manifest["integration_tasks"])
    model_fsm_axi_decoder_block_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "model_fsm_axi_decoder_block_fixture"
    )
    assert model_fsm_axi_decoder_block_task["contract"] == "docs/model_fsm_axi_decoder_block_fixture_contract.md"
    assert model_fsm_axi_decoder_block_task["child_tasks"] == ["token_loop_axi_decoder_block_fixture"]
    assert any(task["task_id"] == "ddr_axi_board_shell_fixture" for task in task_manifest["integration_tasks"])
    ddr_axi_board_shell_task = next(
        task
        for task in task_manifest["integration_tasks"]
        if task["task_id"] == "ddr_axi_board_shell_fixture"
    )
    assert ddr_axi_board_shell_task["agent_role"] == "ddr_axi_board_shell_agent"
    assert ddr_axi_board_shell_task["contract"] == "docs/ddr_axi_board_shell_fixture_contract.md"
    assert ddr_axi_board_shell_task["current_regression_kernel"] == "ddr_axi_board_shell_fixture"
    assert ddr_axi_board_shell_task["child_tasks"] == ["model_fsm_axi_decoder_block_fixture"]
    assert ddr_axi_board_shell_task["projection_weight_stream_plan"]["projection_count"] == 7
    assert "real DDR controller IP integration" in ddr_axi_board_shell_task["does_not_claim"]
    assert "board-level ZCU104 signoff" in ddr_axi_board_shell_task["does_not_claim"]
    if task_manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] > 0:
        assert not any(task["task_id"] == "real_gptq_checkpoint_metadata" for task in task_manifest["blocked_target_tasks"])
    else:
        assert any(task["task_id"] == "real_gptq_checkpoint_metadata" for task in task_manifest["blocked_target_tasks"])
    assert task_manifest["subagent_policy"]["parent_must_not_write_hdl"] is True
    assert task_manifest["subagent_policy"]["verification_agents_are_read_only"] is True
    subagent_tasks = json.loads(subagent_tasks_path.read_text(encoding="utf-8"))
    assert subagent_tasks["artifact"] == "hdl_subagent_tasks"
    assert "real_mlir_model_analysis" in subagent_tasks["global_blocked_target_dependencies"]
    assert subagent_tasks["task_count"] == 57
    assert subagent_tasks["failure_to_skill"]["skill_update_candidate_template"] == "skill_update_candidate_template.json"
    assert subagent_tasks["failure_to_skill"]["required_fields"] == [
        "failing_command",
        "symptom",
        "root_cause_hypothesis",
        "prevention_rule",
        "minimal_regression_check",
    ]
    skill_template = json.loads((tmp_path / "skill_update_candidate_template.json").read_text(encoding="utf-8"))
    assert skill_template["artifact"] == "skill_update_candidate_template"
    assert skill_template["required_before_retry"] is True
    assert skill_template["required_fields"] == subagent_tasks["failure_to_skill"]["required_fields"]
    assert "suggested_skill" in skill_template["candidate"]
    assert subagent_tasks["agent_topology"]["implementation_agent_granularity"] == "one_subagent_per_hdl_packet"
    assert subagent_tasks["agent_topology"]["parallel_module_agents"]["can_run_in_parallel"] is True
    assert subagent_tasks["agent_topology"]["parallel_module_agents"]["must_self_verify"] is True
    assert subagent_tasks["agent_topology"]["failure_to_skill"]["required_before_retry"] is True
    assert subagent_tasks["agent_topology"]["failure_to_skill"]["skill_payload_fields"] == [
        "failing_command",
        "symptom",
        "root_cause_hypothesis",
        "prevention_rule",
        "minimal_regression_check",
    ]
    assert subagent_tasks["subagent_policy"]["parent_must_not_write_hdl"] is True
    q_packet = next(packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_q_proj")
    input_norm_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "non_gemm_input_layernorm"
    )
    axi_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_axi_read_command_adapter"
    )
    k_axi_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_command_adapter"
    )
    axi_r_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_axi_read_data_channel_adapter"
    )
    k_axi_r_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_data_channel_adapter"
    )
    axi_transaction_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_axi_read_transaction_adapter"
    )
    k_axi_transaction_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_transaction_adapter"
    )
    axi_stream_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_axi_stream_integration"
    )
    k_axi_stream_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "projection_k_proj_axi_stream_integration"
    )
    decoder_axi_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "decoder_child_axi_attention_datapath"
    )
    layer_axi_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "layer_fsm_axi_attention_fixture"
    )
    top_axi_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "top_fsm_axi_attention_fixture"
    )
    token_axi_packet = next(
        packet for packet in subagent_tasks["packets"] if packet["task_id"] == "token_loop_axi_attention_fixture"
    )
    decoder_axi_block_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "decoder_block_axi_attention_mlp_fixture"
    )
    ddr_axi_board_shell_packet = next(
        packet
        for packet in subagent_tasks["packets"]
        if packet["task_id"] == "ddr_axi_board_shell_fixture"
    )
    assert q_packet["agent_role"] == "gemm_kernel_agent"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(q_packet["required_commands"])
    assert "docs/target_scale_projection_streaming_contract.md" in q_packet["prompt"]
    assert "GPTQ layout preflight" in q_packet["prompt"]
    assert "Target checkpoint layout dependency" in q_packet["prompt"]
    assert "GPTQ payload probe status" in q_packet["prompt"]
    assert "GPTQ qweight payload order" in q_packet["prompt"]
    assert "GPTQ qweight payload words32 LE" in q_packet["prompt"]
    assert "GPTQ qweight first memory beats 128b LE" in q_packet["prompt"]
    assert "GPTQ qweight memory beat word chunks32 LE" in q_packet["prompt"]
    assert "Target checkpoint payload dependency" in q_packet["prompt"]
    assert "aclk" in q_packet["prompt"]
    assert "done_o" in q_packet["prompt"]
    assert "Full LLaMA execution unless this exact task proves it." in q_packet["prompt"]
    assert "Do not edit parent orchestration files" in q_packet["prompt"]
    assert "Do not weaken existing tests" in q_packet["prompt"]
    assert (tmp_path / q_packet["prompt_file"]).exists()
    assert "NL2HDL_SELECTED_NONGEMM=input_layernorm" in " ".join(
        input_norm_packet["required_commands"]
    )
    assert "NL2HDL_SELECTED_NONGEMM=input_layernorm" in input_norm_packet["prompt"]
    assert "docs/rmsnorm_rope_source_path_contract.md" in input_norm_packet["prompt"]
    assert axi_packet["agent_role"] == "memory_command_adapter_agent"
    assert axi_packet["contract"] == "docs/projection_axi_read_command_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(axi_packet["required_commands"])
    assert "--kernel projection_axi_read_command_adapter" in " ".join(axi_packet["required_commands"])
    assert "Target qweight shard file" in axi_packet["prompt"]
    assert "Target last-beat valid bytes" in axi_packet["prompt"]
    assert "Target request covers unaligned qweight range" in axi_packet["prompt"]
    assert "AXI read address" in axi_packet["prompt"]
    assert k_axi_packet["agent_role"] == "memory_command_adapter_agent"
    assert k_axi_packet["contract"] == "docs/projection_axi_read_command_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_axi_packet["required_commands"])
    assert "--kernel projection_axi_read_command_adapter" in " ".join(k_axi_packet["required_commands"])
    assert k_axi_packet["prompt_file"] == "subagent_prompts/projection_k_proj_axi_read_command_adapter__implementation.md"
    assert "Semantic op: `k_proj`" in k_axi_packet["prompt"]
    assert "k_proj checkpoint-aware qweight stream plan converted to AXI read address and length" in k_axi_packet["prompt"]
    expected_command_packets = {
        "q_proj": "projection_axi_read_command_adapter",
        "k_proj": "projection_k_proj_axi_read_command_adapter",
        "v_proj": "projection_v_proj_axi_read_command_adapter",
        "o_proj": "projection_o_proj_axi_read_command_adapter",
        "gate_proj": "projection_gate_proj_axi_read_command_adapter",
        "up_proj": "projection_up_proj_axi_read_command_adapter",
        "down_proj": "projection_down_proj_axi_read_command_adapter",
    }
    packets_by_id = {packet["task_id"]: packet for packet in subagent_tasks["packets"]}
    for projection_name, task_id in expected_command_packets.items():
        packet = packets_by_id[task_id]
        joined_commands = " ".join(packet["required_commands"])
        assert packet["agent_role"] == "memory_command_adapter_agent"
        assert packet["current_regression_kernel"] == "projection_axi_read_command_adapter"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in joined_commands
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert f"{projection_name} checkpoint-aware qweight stream plan converted to AXI read address and length" in packet[
            "prompt"
        ]
    assert axi_r_packet["agent_role"] == "memory_read_data_adapter_agent"
    assert axi_r_packet["contract"] == "docs/projection_axi_read_data_channel_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(axi_r_packet["required_commands"])
    assert "--kernel projection_axi_read_data_channel_adapter" in " ".join(axi_r_packet["required_commands"])
    assert "Target last-beat valid bytes" in axi_r_packet["prompt"]
    assert "AXI read-data" in axi_r_packet["prompt"]
    assert k_axi_r_packet["agent_role"] == "memory_read_data_adapter_agent"
    assert k_axi_r_packet["contract"] == "docs/projection_axi_read_data_channel_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_axi_r_packet["required_commands"])
    assert "--kernel projection_axi_read_data_channel_adapter" in " ".join(k_axi_r_packet["required_commands"])
    assert k_axi_r_packet["prompt_file"] == (
        "subagent_prompts/projection_k_proj_axi_read_data_channel_adapter__implementation.md"
    )
    assert "Semantic op: `k_proj`" in k_axi_r_packet["prompt"]
    assert "k_proj bounded AXI read-data valid/ready backpressure evidence" in k_axi_r_packet["prompt"]
    expected_read_data_packets = {
        "q_proj": "projection_axi_read_data_channel_adapter",
        "k_proj": "projection_k_proj_axi_read_data_channel_adapter",
        "v_proj": "projection_v_proj_axi_read_data_channel_adapter",
        "o_proj": "projection_o_proj_axi_read_data_channel_adapter",
        "gate_proj": "projection_gate_proj_axi_read_data_channel_adapter",
        "up_proj": "projection_up_proj_axi_read_data_channel_adapter",
        "down_proj": "projection_down_proj_axi_read_data_channel_adapter",
    }
    for projection_name, task_id in expected_read_data_packets.items():
        packet = packets_by_id[task_id]
        joined_commands = " ".join(packet["required_commands"])
        assert packet["agent_role"] == "memory_read_data_adapter_agent"
        assert packet["current_regression_kernel"] == "projection_axi_read_data_channel_adapter"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in joined_commands
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert f"{projection_name} bounded AXI read-data valid/ready backpressure evidence" in packet["prompt"]
    assert axi_transaction_packet["agent_role"] == "memory_read_transaction_agent"
    assert axi_transaction_packet["contract"] == "docs/projection_axi_read_transaction_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(axi_transaction_packet["required_commands"])
    assert "--kernel projection_axi_read_transaction_adapter" in " ".join(axi_transaction_packet["required_commands"])
    assert "Target request beat count" in axi_transaction_packet["prompt"]
    assert "AR and R-channel" in axi_transaction_packet["prompt"]
    assert k_axi_transaction_packet["agent_role"] == "memory_read_transaction_agent"
    assert k_axi_transaction_packet["contract"] == "docs/projection_axi_read_transaction_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_axi_transaction_packet["required_commands"])
    assert "--kernel projection_axi_read_transaction_adapter" in " ".join(k_axi_transaction_packet["required_commands"])
    assert k_axi_transaction_packet["prompt_file"] == (
        "subagent_prompts/projection_k_proj_axi_read_transaction_adapter__implementation.md"
    )
    assert "Semantic op: `k_proj`" in k_axi_transaction_packet["prompt"]
    assert "k_proj bounded AXI read-address command followed by matching read-data beats" in k_axi_transaction_packet["prompt"]
    expected_transaction_packets = {
        "q_proj": "projection_axi_read_transaction_adapter",
        "k_proj": "projection_k_proj_axi_read_transaction_adapter",
        "v_proj": "projection_v_proj_axi_read_transaction_adapter",
        "o_proj": "projection_o_proj_axi_read_transaction_adapter",
        "gate_proj": "projection_gate_proj_axi_read_transaction_adapter",
        "up_proj": "projection_up_proj_axi_read_transaction_adapter",
        "down_proj": "projection_down_proj_axi_read_transaction_adapter",
    }
    for projection_name, task_id in expected_transaction_packets.items():
        packet = packets_by_id[task_id]
        joined_commands = " ".join(packet["required_commands"])
        assert packet["agent_role"] == "memory_read_transaction_agent"
        assert packet["current_regression_kernel"] == "projection_axi_read_transaction_adapter"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in joined_commands
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert f"{projection_name} bounded AXI read-address command followed by matching read-data beats" in packet[
            "prompt"
        ]
    assert axi_stream_packet["agent_role"] == "memory_projection_stream_agent"
    assert axi_stream_packet["contract"] == "docs/projection_axi_stream_integration_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(axi_stream_packet["required_commands"])
    assert "--kernel projection_axi_stream_integration" in " ".join(axi_stream_packet["required_commands"])
    assert "projection-style payload consumer" in axi_stream_packet["prompt"]
    assert "DUT-observed RID/RRESP/RLAST" in axi_stream_packet["prompt"]
    assert k_axi_stream_packet["agent_role"] == "memory_projection_stream_agent"
    assert k_axi_stream_packet["contract"] == "docs/projection_axi_stream_integration_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_axi_stream_packet["required_commands"])
    assert "--kernel projection_axi_stream_integration" in " ".join(k_axi_stream_packet["required_commands"])
    assert k_axi_stream_packet["prompt_file"] == (
        "subagent_prompts/projection_k_proj_axi_stream_integration__implementation.md"
    )
    assert "Semantic op: `k_proj`" in k_axi_stream_packet["prompt"]
    assert (
        "k_proj bounded AXI read transaction feeds projection-style payload consumer through valid/ready"
        in k_axi_stream_packet["prompt"]
    )
    expected_stream_packets = {
        "q_proj": "projection_axi_stream_integration",
        "k_proj": "projection_k_proj_axi_stream_integration",
        "v_proj": "projection_v_proj_axi_stream_integration",
        "o_proj": "projection_o_proj_axi_stream_integration",
        "gate_proj": "projection_gate_proj_axi_stream_integration",
        "up_proj": "projection_up_proj_axi_stream_integration",
        "down_proj": "projection_down_proj_axi_stream_integration",
    }
    for projection_name, task_id in expected_stream_packets.items():
        packet = packets_by_id[task_id]
        joined_commands = " ".join(packet["required_commands"])
        assert packet["agent_role"] == "memory_projection_stream_agent"
        assert packet["current_regression_kernel"] == "projection_axi_stream_integration"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in joined_commands
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert (
            f"{projection_name} bounded AXI read transaction feeds projection-style payload consumer through valid/ready"
            in packet["prompt"]
        )
    assert decoder_axi_packet["agent_role"] == "decoder_axi_child_agent"
    assert decoder_axi_packet["contract"] == "docs/decoder_child_axi_attention_datapath_contract.md"
    assert "--kernel decoder_child_axi_attention_datapath" in " ".join(decoder_axi_packet["required_commands"])
    assert "projection_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_k_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_v_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_o_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert decoder_axi_packet["attention_projection_stream_tasks"] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
    ]
    assert decoder_axi_packet["target_weight_stream_plan"] is None
    assert decoder_axi_packet["gptq_payload_probe"] is None
    assert set(decoder_axi_packet["target_attention_projection_streams"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert "Aggregate attention layout dependency" in decoder_axi_packet["prompt"]
    assert "Aggregate attention payload dependency" in decoder_axi_packet["prompt"]
    assert "Target qweight tensor key" not in decoder_axi_packet["prompt"]
    assert "GPTQ qweight payload words32 LE" not in decoder_axi_packet["prompt"]
    assert "RID/RRESP/RLAST good-path metadata" in decoder_axi_packet["prompt"]
    assert layer_axi_packet["agent_role"] == "layer_axi_fsm_agent"
    assert layer_axi_packet["contract"] == "docs/layer_fsm_axi_attention_fixture_contract.md"
    assert "--kernel layer_fsm_axi_attention_fixture" in " ".join(layer_axi_packet["required_commands"])
    assert "decoder_child_axi_attention_datapath" in layer_axi_packet["prompt"]
    assert "AXI projection child metadata-good bits" in layer_axi_packet["prompt"]
    assert top_axi_packet["agent_role"] == "top_axi_fsm_agent"
    assert top_axi_packet["contract"] == "docs/top_fsm_axi_attention_fixture_contract.md"
    assert "--kernel top_fsm_axi_attention_fixture" in " ".join(top_axi_packet["required_commands"])
    assert "layer_fsm_axi_attention_fixture" in top_axi_packet["prompt"]
    assert "Top FSM compact status" in top_axi_packet["prompt"]
    assert token_axi_packet["agent_role"] == "token_loop_axi_agent"
    assert token_axi_packet["contract"] == "docs/token_loop_axi_attention_fixture_contract.md"
    assert "--kernel token_loop_axi_attention_fixture" in " ".join(token_axi_packet["required_commands"])
    assert "top_fsm_axi_attention_fixture" in token_axi_packet["prompt"]
    assert "token-loop compact status" in token_axi_packet["prompt"]
    assert decoder_axi_block_packet["agent_role"] == "decoder_axi_block_agent"
    assert decoder_axi_block_packet["contract"] == "docs/decoder_block_axi_attention_mlp_fixture_contract.md"
    assert "--kernel decoder_block_axi_attention_mlp_fixture" in " ".join(
        decoder_axi_block_packet["required_commands"]
    )
    assert "decoder_child_axi_attention_datapath" in decoder_axi_block_packet["prompt"]
    assert "residual_mlp_fixture" in decoder_axi_block_packet["prompt"]
    assert "Aggregate attention payload dependency" in decoder_axi_block_packet["prompt"]
    assert ddr_axi_board_shell_packet["agent_role"] == "ddr_axi_board_shell_agent"
    assert ddr_axi_board_shell_packet["contract"] == "docs/ddr_axi_board_shell_fixture_contract.md"
    assert "--kernel ddr_axi_board_shell_fixture" in " ".join(ddr_axi_board_shell_packet["required_commands"])
    assert ddr_axi_board_shell_packet["module_contract"]["parent_boundary"]["integration_boundary"] == (
        "ddr_axi_board_shell_wraps_verified_model_fsm"
    )
    assert ddr_axi_board_shell_packet["projection_weight_stream_plan"]["projection_count"] == 7
    assert "Projection stream plan count: `7`" in ddr_axi_board_shell_packet["prompt"]
    assert "model_fsm_axi_decoder_block_fixture" in ddr_axi_board_shell_packet["prompt"]
    assert "real DDR controller IP integration" in ddr_axi_board_shell_packet["prompt"]
    assert subagent_prompt_dir.name == "subagent_prompts"
    dispatch_plan = json.loads(subagent_dispatch_plan_path.read_text(encoding="utf-8"))
    wave_status = json.loads(subagent_wave_status_path.read_text(encoding="utf-8"))
    execution_manifest = json.loads(subagent_execution_manifest_path.read_text(encoding="utf-8"))
    assert dispatch_plan["artifact"] == "hdl_subagent_dispatch_plan"
    assert dispatch_plan["wave_count"] == 20
    assert dispatch_plan["dispatch_policy"]["parent_must_not_write_hdl"] is True
    assert dispatch_plan["dispatch_policy"]["one_subagent_per_hdl_packet"] is True
    assert dispatch_plan["dispatch_policy"]["module_agents_run_own_simulation_and_synthesis"] is True
    assert dispatch_plan["dispatch_policy"]["layer_fsm_and_top_fsm_are_separate_implementation_agents"] is True
    assert dispatch_plan["dispatch_policy"]["integration_verification_agents_run_synthesis"] is True
    assert dispatch_plan["agent_topology"]["integration_roles_present"] == [
        "decoder_block_agent",
        "layer_fsm_agent",
        "top_fsm_agent",
        "token_loop_agent",
        "memory_command_adapter_agent",
        "memory_read_data_adapter_agent",
        "memory_read_transaction_agent",
        "memory_projection_stream_agent",
        "decoder_axi_child_agent",
        "layer_axi_fsm_agent",
        "top_axi_fsm_agent",
        "token_loop_axi_agent",
        "decoder_axi_block_agent",
        "layer_axi_decoder_block_agent",
        "top_axi_decoder_block_agent",
        "token_loop_axi_decoder_block_agent",
        "model_axi_decoder_block_agent",
        "ddr_axi_board_shell_agent",
    ]
    assert dispatch_plan["dispatch_policy"]["spawn_read_only_verification_agent_after_each_wave"] is True
    assert dispatch_plan["waves"][0]["wave_id"] == "wave_1_projection_kernels"
    assert len(dispatch_plan["waves"][0]["implementation_tasks"]) == 7
    assert dispatch_plan["waves"][0]["implementation_tasks"][0]["expected_evidence_dir"] == "build/projection_q_proj_gate"
    assert dispatch_plan["waves"][0]["implementation_tasks"][0]["expected_kernel_report"] == (
        "build/projection_q_proj_gate/kernel_report.json"
    )
    assert dispatch_plan["waves"][0]["implementation_tasks"][0]["module_contract"]["task_id"] == "projection_q_proj"
    assert dispatch_plan["waves"][0]["implementation_tasks"][0]["module_contract"]["handshake_ports"] == {
        "start": "start_i",
        "done": "done_o",
    }
    k_dispatch_task = next(
        task for task in dispatch_plan["waves"][0]["implementation_tasks"] if task["task_id"] == "projection_k_proj"
    )
    assert k_dispatch_task["semantic_op"] == "k_proj"
    assert k_dispatch_task["expected_projection_shape"] == {"rows": 512, "cols": 2048}
    assert k_dispatch_task["packed_int4_bytes"] == 524288
    assert k_dispatch_task["memory_beats"] == 32768
    assert dispatch_plan["waves"][0]["verification_agent"]["agent"] == "Codex"
    assert dispatch_plan["waves"][0]["verification_agent"]["mode"] == "read_only"
    assert dispatch_plan["waves"][0]["verification_agent"]["runs_integration_synthesis"] is False
    assert dispatch_plan["waves"][0]["verification_agent"]["prompt_file"].startswith("verification_prompts/")
    verification_prompt = (
        tmp_path / dispatch_plan["waves"][0]["verification_agent"]["prompt_file"]
    ).read_text(encoding="utf-8")
    assert "Codex Read-Only Verification" in verification_prompt
    assert "Audit only; do not edit source, RTL, tests, or contracts." in verification_prompt
    assert "real_gptq_weight_layout_preflight" in verification_prompt
    assert "projection_k_proj" in verification_prompt
    assert "expected projection shape `512 x 2048`" in verification_prompt
    for wave in dispatch_plan["waves"]:
        prompt_path = tmp_path / wave["verification_agent"]["prompt_file"]
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if wave["verification_agent"]["runs_integration_synthesis"]:
            assert f"Codex Integration Verification: {wave['wave_id']}" in prompt_text
            assert "Integration-Level Synthesis Requirement" in prompt_text
            assert "integration_synthesis_report.json" in prompt_text
            assert wave["verification_agent"]["expected_integration_synthesis_report"] in prompt_text
        else:
            assert f"Codex Read-Only Verification: {wave['wave_id']}" in prompt_text
        assert "Audit only; do not edit source, RTL, tests, or contracts." in prompt_text
        assert "Target scope:" in prompt_text
        assert "Verification mode:" in prompt_text
        assert "Direct blocked target dependencies:" in prompt_text
        assert "Inherited blocked target dependencies:" in prompt_text
        assert "Implementation Tasks To Audit" in prompt_text
        assert "Audit Requirements" in prompt_text
        assert "Do Not Claim" in prompt_text
        assert "Current Blocked Target Tasks" in prompt_text
        assert "Verification means no source edits, no RTL rewrites, and no test weakening." in prompt_text
        assert "skill_update_candidate" in prompt_text
        for task in wave["implementation_tasks"]:
            assert task["task_id"] in prompt_text
            assert task["prompt_file"] in prompt_text
    assert dispatch_plan["waves"][0]["target_scope"] in {
        "bounded_fixture_only",
        "target_preflight_satisfied_or_not_applicable",
    }
    assert dispatch_plan["waves"][1]["wave_id"] == "wave_1_non_gemm_kernels"
    assert len(dispatch_plan["waves"][1]["implementation_tasks"]) == 8
    assert dispatch_plan["waves"][2]["depends_on_waves"] == ["wave_1_projection_kernels", "wave_1_non_gemm_kernels"]
    assert dispatch_plan["waves"][3]["implementation_tasks"][0]["task_id"] == "layer_fsm_decoder_block_fixture"
    assert dispatch_plan["waves"][4]["implementation_tasks"][0]["task_id"] == "top_fsm_decoder_block_fixture"
    assert dispatch_plan["waves"][6]["implementation_tasks"][0]["task_id"] == "projection_axi_read_command_adapter"
    assert len(dispatch_plan["waves"][6]["implementation_tasks"]) == 7
    assert [task["task_id"] for task in dispatch_plan["waves"][6]["implementation_tasks"]] == [
        "projection_axi_read_command_adapter",
        "projection_k_proj_axi_read_command_adapter",
        "projection_v_proj_axi_read_command_adapter",
        "projection_o_proj_axi_read_command_adapter",
        "projection_gate_proj_axi_read_command_adapter",
        "projection_up_proj_axi_read_command_adapter",
        "projection_down_proj_axi_read_command_adapter",
    ]
    assert dispatch_plan["waves"][6]["depends_on_waves"] == ["wave_5_token_loop"]
    assert dispatch_plan["waves"][7]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_read_data_channel_adapter"
    )
    assert len(dispatch_plan["waves"][7]["implementation_tasks"]) == 7
    assert [task["task_id"] for task in dispatch_plan["waves"][7]["implementation_tasks"]] == [
        "projection_axi_read_data_channel_adapter",
        "projection_k_proj_axi_read_data_channel_adapter",
        "projection_v_proj_axi_read_data_channel_adapter",
        "projection_o_proj_axi_read_data_channel_adapter",
        "projection_gate_proj_axi_read_data_channel_adapter",
        "projection_up_proj_axi_read_data_channel_adapter",
        "projection_down_proj_axi_read_data_channel_adapter",
    ]
    assert dispatch_plan["waves"][7]["depends_on_waves"] == ["wave_6_projection_axi_read_command_adapter"]
    assert dispatch_plan["waves"][8]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_read_transaction_adapter"
    )
    assert len(dispatch_plan["waves"][8]["implementation_tasks"]) == 7
    assert [task["task_id"] for task in dispatch_plan["waves"][8]["implementation_tasks"]] == [
        "projection_axi_read_transaction_adapter",
        "projection_k_proj_axi_read_transaction_adapter",
        "projection_v_proj_axi_read_transaction_adapter",
        "projection_o_proj_axi_read_transaction_adapter",
        "projection_gate_proj_axi_read_transaction_adapter",
        "projection_up_proj_axi_read_transaction_adapter",
        "projection_down_proj_axi_read_transaction_adapter",
    ]
    assert dispatch_plan["waves"][8]["depends_on_waves"] == ["wave_7_projection_axi_read_data_channel_adapter"]
    assert dispatch_plan["waves"][9]["implementation_tasks"][0]["task_id"] == "projection_axi_stream_integration"
    assert len(dispatch_plan["waves"][9]["implementation_tasks"]) == 7
    assert [task["task_id"] for task in dispatch_plan["waves"][9]["implementation_tasks"]] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
        "projection_gate_proj_axi_stream_integration",
        "projection_up_proj_axi_stream_integration",
        "projection_down_proj_axi_stream_integration",
    ]
    assert dispatch_plan["waves"][9]["depends_on_waves"] == ["wave_8_projection_axi_read_transaction_adapter"]
    assert dispatch_plan["waves"][10]["implementation_tasks"][0]["task_id"] == "decoder_child_axi_attention_datapath"
    assert dispatch_plan["waves"][10]["depends_on_waves"] == ["wave_9_projection_axi_stream_integration"]
    assert dispatch_plan["waves"][11]["implementation_tasks"][0]["task_id"] == "layer_fsm_axi_attention_fixture"
    assert dispatch_plan["waves"][11]["depends_on_waves"] == ["wave_10_decoder_child_axi_attention_datapath"]
    assert dispatch_plan["waves"][12]["implementation_tasks"][0]["task_id"] == "top_fsm_axi_attention_fixture"
    assert dispatch_plan["waves"][12]["depends_on_waves"] == ["wave_11_layer_fsm_axi_attention_fixture"]
    assert dispatch_plan["waves"][13]["implementation_tasks"][0]["task_id"] == "token_loop_axi_attention_fixture"
    assert dispatch_plan["waves"][13]["depends_on_waves"] == ["wave_12_top_fsm_axi_attention_fixture"]
    assert dispatch_plan["waves"][14]["implementation_tasks"][0]["task_id"] == (
        "decoder_block_axi_attention_mlp_fixture"
    )
    assert dispatch_plan["waves"][14]["depends_on_waves"] == ["wave_13_token_loop_axi_attention_fixture"]
    assert dispatch_plan["waves"][15]["implementation_tasks"][0]["task_id"] == (
        "layer_fsm_axi_decoder_block_fixture"
    )
    assert dispatch_plan["waves"][15]["depends_on_waves"] == [
        "wave_14_decoder_block_axi_attention_mlp_fixture"
    ]
    assert dispatch_plan["waves"][16]["implementation_tasks"][0]["task_id"] == (
        "top_fsm_axi_decoder_block_fixture"
    )
    assert dispatch_plan["waves"][16]["depends_on_waves"] == ["wave_15_layer_fsm_axi_decoder_block_fixture"]
    assert dispatch_plan["waves"][17]["implementation_tasks"][0]["task_id"] == (
        "token_loop_axi_decoder_block_fixture"
    )
    assert dispatch_plan["waves"][17]["depends_on_waves"] == ["wave_16_top_fsm_axi_decoder_block_fixture"]
    assert dispatch_plan["waves"][18]["implementation_tasks"][0]["task_id"] == (
        "model_fsm_axi_decoder_block_fixture"
    )
    assert dispatch_plan["waves"][18]["depends_on_waves"] == ["wave_17_token_loop_axi_decoder_block_fixture"]
    assert dispatch_plan["waves"][19]["implementation_tasks"][0]["task_id"] == "ddr_axi_board_shell_fixture"
    assert dispatch_plan["waves"][19]["depends_on_waves"] == ["wave_18_model_fsm_axi_decoder_block_fixture"]
    assert dispatch_plan["waves"][19]["implementation_tasks"][0]["module_contract"]["parent_boundary"][
        "integration_boundary"
    ] == "ddr_axi_board_shell_wraps_verified_model_fsm"
    assert dispatch_plan["waves"][19]["verification_agent"]["mode"] == "integration_verification_with_synthesis"
    assert dispatch_plan["waves"][19]["verification_agent"]["runs_integration_synthesis"] is True
    assert dispatch_plan["waves"][19]["verification_agent"]["expected_integration_synthesis_report"] == (
        "build/wave_19_ddr_axi_board_shell_fixture_integration_verification/integration_synthesis_report.json"
    )
    assert wave_status["artifact"] == "hdl_subagent_wave_status"
    assert wave_status["coverage_level"] == "parent_result_collection_no_hdl_generation"
    assert wave_status["parent_must_not_write_hdl"] is True
    assert wave_status["failure_to_skill_required_before_retry"] is True
    assert wave_status["next_dispatchable_waves"] == [
        "wave_1_projection_kernels",
        "wave_1_non_gemm_kernels",
    ]
    assert execution_manifest["artifact"] == "hdl_subagent_execution_manifest"
    assert execution_manifest["coverage_level"] == "dispatch_wave_status_to_codex_spawn_instructions"
    assert execution_manifest["spawn_entry_count"] == 15
    assert execution_manifest["implementation_spawn_count"] == 15
    assert execution_manifest["verification_spawn_count"] == 0
    assert execution_manifest["spawn_batch_count"] == 2
    assert execution_manifest["parallel_spawn_allowed"] is True
    assert execution_manifest["max_parallel_batch_size"] == 8
    spawn_instructions_path = tmp_path / "codex_spawn_instructions.md"
    assert spawn_instructions_path.exists()
    spawn_instructions = spawn_instructions_path.read_text(encoding="utf-8")
    assert "# Codex Sub-Agent Spawn Instructions" in spawn_instructions
    assert "real_mlir_model_analysis" in spawn_instructions
    assert "wave_1_projection_kernels__implementation_agent" in spawn_instructions
    assert "subagent_prompts/projection_q_proj__implementation.md" in spawn_instructions
    assert "build/projection_q_proj_gate/subagent_result.json" in spawn_instructions
    spawn_batches = {batch["wave_id"]: batch for batch in execution_manifest["spawn_batches"]}
    assert spawn_batches["wave_1_projection_kernels"]["parallel_spawn_allowed"] is True
    assert spawn_batches["wave_1_projection_kernels"]["entry_count"] == 7
    assert spawn_batches["wave_1_projection_kernels"]["task_ids"] == [
        "projection_q_proj",
        "projection_k_proj",
        "projection_v_proj",
        "projection_o_proj",
        "projection_gate_proj",
        "projection_up_proj",
        "projection_down_proj",
    ]
    assert spawn_batches["wave_1_non_gemm_kernels"]["parallel_spawn_allowed"] is True
    assert spawn_batches["wave_1_non_gemm_kernels"]["entry_count"] == 8
    assert {entry["wave_id"] for entry in execution_manifest["spawn_entries"]} == {
        "wave_1_projection_kernels",
        "wave_1_non_gemm_kernels",
    }
    assert all(entry["parent_must_not_write_hdl"] is True for entry in execution_manifest["spawn_entries"])
    assert all(entry["failure_to_skill_required"] is True for entry in execution_manifest["spawn_entries"])
    assert any(entry["task_id"] == "projection_q_proj" for entry in execution_manifest["spawn_entries"])
    assert any(entry["task_id"] == "non_gemm_input_layernorm" for entry in execution_manifest["spawn_entries"])
    wave_statuses = {wave["wave_id"]: wave for wave in wave_status["waves"]}
    assert wave_statuses["wave_1_projection_kernels"]["status"] == "ready_to_dispatch"
    assert wave_statuses["wave_1_projection_kernels"]["task_count"] == 7
    assert wave_statuses["wave_1_projection_kernels"]["task_status_counts"] == {"missing": 7}
    assert wave_statuses["wave_1_projection_kernels"]["passed_task_count"] == 0
    assert wave_statuses["wave_1_projection_kernels"]["missing_task_count"] == 7
    assert wave_statuses["wave_1_non_gemm_kernels"]["status"] == "ready_to_dispatch"
    assert wave_statuses["wave_1_non_gemm_kernels"]["task_status_counts"] == {"missing": 8}
    assert wave_statuses["wave_2_decoder_block"]["status"] == "blocked_by_dependency"
    assert "waiting for waves" in wave_statuses["wave_2_decoder_block"]["reason"]


def test_emit_inspect_artifacts_accepts_provided_full_coverage_mlir(tmp_path: Path):
    provided_mlir = tmp_path / "provided_llama_block.mlir"
    provided_mlir.write_text(_full_llama_block_mlir(), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 8
  mlir_graph: {provided_mlir}
optimization:
  quantization: int4_gptq
design:
  style: llm_decoder_streaming
hardware:
  memory_data_width: 128
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)

    emit_inspect_artifacts(
        tmp_path / "inspect",
        "meta-llama/Llama-3.2-1B",
        cfg,
    )

    readiness = json.loads((tmp_path / "inspect" / "mlir_model_analysis_readiness.json").read_text(encoding="utf-8"))
    target_readiness = json.loads((tmp_path / "inspect" / "target_readiness_report.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((tmp_path / "inspect" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "inspect" / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    mlir_gate = next(gate for gate in target_readiness["gates"] if gate["gate"] == "mlir_model_analysis")

    assert readiness["status"] == "passed"
    assert readiness["analysis_source"] == "provided_model_mlir_file"
    assert readiness["source_kind"] == "provided_mlir"
    assert readiness["target_claim_allowed_by_source"] is True
    assert readiness["missing_semantic_gemm_ops"] == []
    assert readiness["missing_semantic_non_gemm_ops"] == []
    assert readiness["mlir_unsupported_ops"] == []
    assert task_manifest["mlir_model_analysis"]["status"] == "passed"
    assert not any(task["task_id"] == "real_mlir_model_analysis" for task in task_manifest["blocked_target_tasks"])
    assert "real_mlir_model_analysis" not in dispatch_plan["global_blocked_target_dependencies"]
    assert mlir_gate["status"] == "passed"
    assert "exported_model_MLIR_full_operation_coverage" not in target_readiness["does_not_claim"]
    assert target_readiness["safe_to_claim_target_accelerator"] is False


def test_emit_inspect_artifacts_accepts_hf_config_model_structure_source(tmp_path: Path, monkeypatch):
    def fake_metadata(model_name, config):
        return {
            "name": model_name,
            "family": "llama_decoder_only_transformer",
            "target_checkpoint": True,
            "metadata_source": "huggingface_auto_config_local_cache",
            "metadata_resolution": {"status": "resolved", "allow_download": False},
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "decoder_layers": 16,
            "sequence_length": 8,
            "batch_size": 1,
        }

    monkeypatch.setattr(llm_kernels_module, "resolve_llama_model_metadata", fake_metadata)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  input_shape: [1, 1]
  sequence_length: 8
  model_structure_source: hf_config
optimization:
  quantization: int4_gptq
design:
  style: llm_decoder_streaming
hardware:
  memory_data_width: 128
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)

    emit_inspect_artifacts(
        tmp_path / "inspect",
        "meta-llama/Llama-3.2-1B",
        cfg,
    )

    readiness = json.loads((tmp_path / "inspect" / "mlir_model_analysis_readiness.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((tmp_path / "inspect" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "inspect" / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    target_readiness = json.loads((tmp_path / "inspect" / "target_readiness_report.json").read_text(encoding="utf-8"))

    assert readiness["status"] == "passed"
    assert readiness["analysis_source"] == "hf_config_semantic_graph"
    assert readiness["source_kind"] == "hf_config"
    assert readiness["model_structure_source"] == "hf_config"
    assert readiness["missing_semantic_gemm_ops"] == []
    assert readiness["missing_semantic_non_gemm_ops"] == []
    assert readiness["model_metadata_resolution"]["status"] == "resolved"
    assert task_manifest["mlir_model_analysis"]["status"] == "passed"
    assert dispatch_plan["source_replay"]["model_structure_source"] == "hf_config"
    assert not any(task["task_id"] == "real_mlir_model_analysis" for task in task_manifest["blocked_target_tasks"])
    assert "real_mlir_model_analysis" not in dispatch_plan["global_blocked_target_dependencies"]
    mlir_gate = next(gate for gate in target_readiness["gates"] if gate["gate"] == "mlir_model_analysis")
    assert mlir_gate["status"] == "passed"


def test_emit_inspect_artifacts_maps_hf_export_style_mlir_paths_to_semantic_ops(tmp_path: Path):
    provided_mlir = tmp_path / "hf_export_style_llama_block.mlir"
    provided_mlir.write_text(_hf_export_style_llama_block_mlir(), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 8
  mlir_graph: {provided_mlir}
optimization:
  quantization: int4_gptq
design:
  style: llm_decoder_streaming
hardware:
  memory_data_width: 128
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)

    emit_inspect_artifacts(
        tmp_path / "inspect",
        "meta-llama/Llama-3.2-1B",
        cfg,
    )

    readiness = json.loads((tmp_path / "inspect" / "mlir_model_analysis_readiness.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((tmp_path / "inspect" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "inspect" / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))

    assert readiness["status"] == "passed"
    assert "q_proj" in readiness["mlir_semantic_node_names"]
    assert "attention_scores_softmax_kv" in readiness["mlir_semantic_node_names"]
    assert readiness["mlir_semantic_alias_map"]["/model/layers.0/self_attn/q_proj/MatMul"] == ["q_proj"]
    assert readiness["missing_semantic_gemm_ops"] == []
    assert readiness["missing_semantic_non_gemm_ops"] == []
    assert task_manifest["mlir_model_analysis"]["status"] == "passed"
    assert not any(task["task_id"] == "real_mlir_model_analysis" for task in task_manifest["blocked_target_tasks"])
    assert "real_mlir_model_analysis" not in dispatch_plan["global_blocked_target_dependencies"]


def test_llm_agent_inspect_reports_semantic_graph_artifact(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=tmp_path,
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )
    assert report["status"] == "passed"
    assert report["target_gate_summary"]["mlir_model_analysis"]["artifact"] == (
        "mlir_model_analysis_readiness.json"
    )
    assert report["target_gate_summary"]["mlir_model_analysis"]["status"] == "blocked"
    assert report["target_gate_summary"]["mlir_model_analysis"]["analysis_source"] == "synthetic_llama_block_mlir"
    assert report["target_gate_summary"]["target_readiness"]["artifact"] == "target_readiness_report.json"
    assert report["target_gate_summary"]["target_readiness"]["status"] in {"target_blocked", "target_ready"}
    assert report["target_gate_summary"]["target_readiness"]["safe_to_spawn_bounded_subagents"] is True
    assert report["target_gate_summary"]["target_readiness"]["safe_to_claim_target_accelerator"] is False
    assert report["target_gate_summary"]["target_blocker_remediation"]["artifact"] == (
        "target_blocker_remediation_plan.json"
    )
    assert report["target_gate_summary"]["target_blocker_remediation"]["status"] == "blocked"
    assert "--gptq-checkpoint" in report["target_gate_summary"]["target_blocker_remediation"][
        "canonical_full_preflight_command"
    ]
    assert report["target_gate_summary"]["gptq_checkpoint_metadata"]["artifact"] == "gptq_checkpoint_metadata.json"
    assert report["target_gate_summary"]["gptq_checkpoint_source_preflight"]["artifact"] == (
        "gptq_checkpoint_source_preflight.json"
    )
    assert "checkpoint_source_dependency" in report["target_gate_summary"]["gptq_checkpoint_source_preflight"]
    assert report["target_gate_summary"]["gptq_payload_probe"]["artifact"] == "gptq_payload_probe.json"
    assert report["target_gate_summary"]["gptq_payload_probe"]["projection"] == "q_proj"
    assert "target_checkpoint_payload_dependency" in report["target_gate_summary"]["gptq_payload_probe"]
    assert "qweight_payload_word_count" in report["target_gate_summary"]["gptq_payload_probe"]
    assert report["target_gate_summary"]["blocked_target_task_count"] >= 2
    assert any(
        task["task_id"] == "full_llama_model_execution"
        for task in report["target_gate_summary"]["blocked_target_tasks"]
    )
    inspect_step = next(step for step in report["steps"] if step["name"] == "inspect_semantic_mlir")
    assert inspect_step["mlir_model_analysis_readiness"] == "mlir_model_analysis_readiness.json"
    assert inspect_step["semantic_graph"] == "model_semantic_graph.json"
    assert inspect_step["gptq_checkpoint_source_preflight"] == "gptq_checkpoint_source_preflight.json"
    assert inspect_step["gptq_checkpoint_metadata"] == "gptq_checkpoint_metadata.json"
    assert inspect_step["gptq_payload_probe"] == "gptq_payload_probe.json"
    assert inspect_step["target_readiness_report"] == "target_readiness_report.json"
    assert inspect_step["target_blocker_remediation_plan"] == "target_blocker_remediation_plan.json"
    assert inspect_step["target_blocker_remediation_markdown"] == "target_blocker_remediation_plan.md"
    assert inspect_step["hdl_task_manifest"] == "hdl_task_manifest.json"
    assert inspect_step["hdl_subagent_tasks"] == "hdl_subagent_tasks.json"
    assert inspect_step["skill_update_candidate_template"] == "skill_update_candidate_template.json"
    assert inspect_step["subagent_prompts"] == "subagent_prompts"
    assert inspect_step["verification_prompts"] == "verification_prompts"
    assert inspect_step["hdl_subagent_dispatch_plan"] == "hdl_subagent_dispatch_plan.json"
    assert inspect_step["hdl_subagent_wave_status"] == "hdl_subagent_wave_status.json"
    assert inspect_step["hdl_subagent_execution_manifest"] == "hdl_subagent_execution_manifest.json"
    assert inspect_step["codex_spawn_instructions"] == "codex_spawn_instructions.md"
    assert (tmp_path / "mlir_model_analysis_readiness.json").exists()
    assert (tmp_path / "model_semantic_graph.json").exists()
    assert (tmp_path / "gptq_checkpoint_source_preflight.json").exists()
    assert (tmp_path / "gptq_checkpoint_metadata.json").exists()
    assert (tmp_path / "gptq_payload_probe.json").exists()
    assert (tmp_path / "target_readiness_report.json").exists()
    assert (tmp_path / "target_blocker_remediation_plan.json").exists()
    assert (tmp_path / "target_blocker_remediation_plan.md").exists()
    assert (tmp_path / "hdl_task_manifest.json").exists()
    assert (tmp_path / "hdl_subagent_tasks.json").exists()
    assert (tmp_path / "skill_update_candidate_template.json").exists()
    assert (tmp_path / "hdl_subagent_dispatch_plan.json").exists()
    assert (tmp_path / "hdl_subagent_wave_status.json").exists()
    assert (tmp_path / "hdl_subagent_execution_manifest.json").exists()
    assert (tmp_path / "codex_spawn_instructions.md").exists()
    assert (tmp_path / "subagent_prompts").is_dir()
    assert (tmp_path / "verification_prompts").is_dir()
    saved_report = json.loads((tmp_path / "llm_agent_report.json").read_text(encoding="utf-8"))
    assert saved_report["target_gate_summary"]["gptq_checkpoint_metadata"]["artifact"] == "gptq_checkpoint_metadata.json"
    assert saved_report["target_gate_summary"]["gptq_checkpoint_source_preflight"]["artifact"] == (
        "gptq_checkpoint_source_preflight.json"
    )
    assert saved_report["target_gate_summary"]["gptq_payload_probe"]["artifact"] == "gptq_payload_probe.json"
    assert saved_report["target_gate_summary"]["mlir_model_analysis"]["artifact"] == (
        "mlir_model_analysis_readiness.json"
    )
    assert saved_report["target_gate_summary"]["target_readiness"]["artifact"] == "target_readiness_report.json"
    assert saved_report["target_gate_summary"]["target_blocker_remediation"]["artifact"] == (
        "target_blocker_remediation_plan.json"
    )
    saved_step = next(step for step in saved_report["steps"] if step["name"] == "inspect_semantic_mlir")
    assert saved_step["mlir_model_analysis_readiness"] == "mlir_model_analysis_readiness.json"
    assert saved_step["semantic_graph"] == "model_semantic_graph.json"
    assert saved_step["gptq_checkpoint_source_preflight"] == "gptq_checkpoint_source_preflight.json"
    assert saved_step["gptq_checkpoint_metadata"] == "gptq_checkpoint_metadata.json"
    assert saved_step["gptq_payload_probe"] == "gptq_payload_probe.json"
    assert saved_step["target_readiness_report"] == "target_readiness_report.json"
    assert saved_step["target_blocker_remediation_plan"] == "target_blocker_remediation_plan.json"
    assert saved_step["target_blocker_remediation_markdown"] == "target_blocker_remediation_plan.md"
    assert saved_step["hdl_task_manifest"] == "hdl_task_manifest.json"
    assert saved_step["hdl_subagent_tasks"] == "hdl_subagent_tasks.json"
    assert saved_step["skill_update_candidate_template"] == "skill_update_candidate_template.json"
    assert saved_step["subagent_prompts"] == "subagent_prompts"
    assert saved_step["verification_prompts"] == "verification_prompts"
    assert saved_step["hdl_subagent_dispatch_plan"] == "hdl_subagent_dispatch_plan.json"
    assert saved_step["hdl_subagent_wave_status"] == "hdl_subagent_wave_status.json"
    assert saved_step["hdl_subagent_execution_manifest"] == "hdl_subagent_execution_manifest.json"
    assert saved_step["codex_spawn_instructions"] == "codex_spawn_instructions.md"


def test_llm_agent_kernel_marks_nested_wave_status_context_only(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=tmp_path,
        mode="kernel",
        kernel="int4_unpack",
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "passed"
    wave_status = json.loads((tmp_path / "hdl_subagent_wave_status.json").read_text(encoding="utf-8"))
    assert wave_status["artifact"] == "hdl_subagent_wave_status_context_only"
    assert wave_status["coverage_level"] == "kernel_or_block_context_snapshot_not_parent_collection_gate"
    assert wave_status["context_only"] is True
    assert "use the parent collection root" in wave_status["context_reason"]
    assert "current parent collection wave status" in wave_status["does_not_claim"]


def test_llm_agent_kernel_env_payload_probe_updates_parent_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _sample_gptq_payload_probe()
    probe_words = probe["qweight_payload_words32_le_hex"]
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=tmp_path,
        mode="kernel",
        kernel="int4_unpack",
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "passed"
    summary = report["target_gate_summary"]["gptq_payload_probe"]
    assert summary["status"] == "sampled"
    assert summary["payload_probe_source"] == "NL2HDL_GPTQ_PAYLOAD_PROBE_JSON"
    assert summary["payload_golden_source"] == "gptq_payload_probe"
    assert summary["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert summary["payload_words_match_gptq_probe"] is True
    assert summary["qweight_payload_word_count"] == 8
    assert summary["qweight_payload_words32_le_count"] == 8
    assert summary["qweight_payload_order"] == "safetensors_payload_prefix_32bit_little_endian_words"
    assert summary["qweight_stream_probe"]["first_memory_beats_128b_le_hex"] == [
        "0x0f0e0d0c0b0a09080706050403020100",
        "0x1f1e1d1c1b1a19181716151413121110",
    ]
    assert any(
        task["task_id"] == "real_gptq_payload_probe"
        for task in report["target_gate_summary"]["blocked_target_tasks"]
    )

    payload_probe = json.loads((tmp_path / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    assert payload_probe["status"] == "sampled"
    assert payload_probe["payload_probe_source"] == "NL2HDL_GPTQ_PAYLOAD_PROBE_JSON"
    assert payload_probe["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert payload_probe["qweight_payload_words32_le_hex"] == probe_words
    assert payload_probe["payload_words_match_gptq_probe"] is True
    assert payload_probe["qweight_stream_probe"]["covers_first_memory_beat"] is True

    manifest = json.loads((tmp_path / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gptq_payload_probe"]["status"] == "sampled"
    assert manifest["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert manifest["gptq_payload_probe"]["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert manifest["gptq_payload_probe"]["qweight_stream_probe"]["first_memory_beats_128b_le_hex"] == [
        "0x0f0e0d0c0b0a09080706050403020100",
        "0x1f1e1d1c1b1a19181716151413121110",
    ]
    assert any(task["task_id"] == "real_gptq_payload_probe" for task in manifest["blocked_target_tasks"])
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    k_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "k_proj")
    assert q_task["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == probe_words
    assert q_task["gptq_payload_probe"]["qweight_stream_probe"]["memory_beat_word_chunks32_le_hex"][0] == [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
    ]
    assert q_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert k_task["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    dispatch_plan = json.loads((tmp_path / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    wave8 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_8_projection_axi_read_transaction_adapter"
    )
    wave9 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_9_projection_axi_stream_integration"
    )
    wave10 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_10_decoder_child_axi_attention_datapath"
    )
    wave8_tasks = {task["task_id"]: task for task in wave8["implementation_tasks"]}
    wave9_tasks = {task["task_id"]: task for task in wave9["implementation_tasks"]}
    wave10_tasks = {task["task_id"]: task for task in wave10["implementation_tasks"]}
    assert "gptq_payload_probe" in wave8["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave9["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave10["direct_blocked_target_dependencies"]
    assert wave8_tasks["projection_axi_read_transaction_adapter"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert wave9_tasks["projection_axi_stream_integration"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    blocked_projection_task_suffixes = {
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }
    for projection_name, task_suffix in blocked_projection_task_suffixes.items():
        assert wave8_tasks[
            f"projection_{task_suffix}_axi_read_transaction_adapter"
        ]["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe", projection_name
        assert wave9_tasks[
            f"projection_{task_suffix}_axi_stream_integration"
        ]["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe", projection_name
    assert wave10_tasks["decoder_child_axi_attention_datapath"]["target_checkpoint_payload_dependency"] == (
        "blocked_by_gptq_payload_probe"
    )


def test_llm_agent_kernel_env_aggregate_payload_probe_unblocks_all_projection_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=tmp_path,
        mode="kernel",
        kernel="int4_unpack",
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "passed"
    summary = report["target_gate_summary"]["gptq_payload_probe"]
    assert summary["status"] == "sampled"
    assert summary["payload_probe_source"] == "NL2HDL_GPTQ_PAYLOAD_PROBE_JSON"
    assert summary["projection_payload_probe_count"] == 7
    assert summary["sampled_projection_payload_probe_count"] == 7
    assert summary["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert not any(
        task["task_id"] == "real_gptq_payload_probe"
        for task in report["target_gate_summary"]["blocked_target_tasks"]
    )

    payload_probe = json.loads((tmp_path / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    stream_plan = json.loads((tmp_path / "projection_weight_stream_plan.json").read_text(encoding="utf-8"))
    subagent_tasks = json.loads((tmp_path / "hdl_subagent_tasks.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))

    assert payload_probe["projection_payload_probe_count"] == 7
    assert payload_probe["sampled_projection_payload_probe_count"] == 7
    assert payload_probe["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert set(payload_probe["projection_payload_probes"]) == set(graph["partition"]["gemm"])
    assert manifest["gptq_payload_probe"]["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert not any(task["task_id"] == "real_gptq_payload_probe" for task in manifest["blocked_target_tasks"])
    assert stream_plan["payload_satisfied_projection_count"] == 7
    assert stream_plan["payload_blocked_projection_count"] == 0
    assert stream_plan["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert stream_plan["layout_blocked_projection_count"] == 7
    assert stream_plan["all_projection_layout_dependency"] == "blocked_by_real_gptq_weight_layout_preflight"
    assert stream_plan["target_scale_ready_for_all_projection_streaming"] is False

    for projection_task in manifest["projection_tasks"]:
        projection_name = projection_task["semantic_op"]
        assert projection_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert projection_task["gptq_payload_probe"]["probe_projection"] == projection_name
        assert projection_task["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == (
            words_by_projection[projection_name]
        )
        assert projection_task["gptq_payload_probe"]["qweight_stream_probe"]["memory_beat_word_chunks32_le_hex"][0] == (
            words_by_projection[projection_name][:4]
        )

    k_packet = next(packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_k_proj")
    assert k_packet["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert k_packet["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == words_by_projection["k_proj"]
    assert f"GPTQ qweight payload words32 LE: `{words_by_projection['k_proj']}`" in k_packet["prompt"]

    for wave_id in [
        "wave_8_projection_axi_read_transaction_adapter",
        "wave_9_projection_axi_stream_integration",
        "wave_10_decoder_child_axi_attention_datapath",
    ]:
        wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == wave_id)
        assert "gptq_payload_probe" not in wave["direct_blocked_target_dependencies"]


def test_llm_agent_inspect_uses_configured_gptq_checkpoint_source(tmp_path: Path):
    gptq_dir = tmp_path / "local-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_fake_safetensors_header(
        gptq_dir / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ],
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 2048
  gptq_checkpoint: {gptq_dir}
hardware:
  fpga_part: xczu7ev-ffvc1156-2-e
  target_clock_mhz: 200
  max_dsp: 1728
  max_bram: 312
  max_lut: 230000
  memory_data_width: 128
optimization:
  quantization: int4_gptq
  pruning: none
design:
  style: llm_decoder_streaming
  pe_count: 64
  activation_buffer: tiled_ping_pong_uram_bram
  weight_storage: external_ddr_gptq_packed
verification:
  enable_verilator: true
  enable_vivado_synth: true
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    out_dir = tmp_path / "inspect_out"

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=out_dir,
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "passed"
    assert report["target_gate_summary"]["gptq_checkpoint_source_preflight"]["status"] == "resolved_local_path"
    assert report["target_gate_summary"]["gptq_checkpoint_source_preflight"]["checkpoint_source_dependency"] == (
        "satisfied_by_local_path"
    )
    assert report["target_gate_summary"]["gptq_weight_layout_preflight"]["artifact"] == "gptq_weight_layout_preflight.json"
    assert report["target_gate_summary"]["gptq_weight_layout_preflight"]["status"] == "blocked"
    assert "qweight byte counts" in report["target_gate_summary"]["gptq_weight_layout_preflight"]["blocking_reason"]
    assert report["target_gate_summary"]["gptq_payload_probe"]["status"] == "sampled"
    assert report["target_gate_summary"]["gptq_payload_probe"]["sampled_tensor_count"] == 3
    assert report["target_gate_summary"]["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert report["target_gate_summary"]["gptq_payload_probe"]["qweight_payload_word_count"] == 1
    semantic_graph = json.loads((out_dir / "model_semantic_graph.json").read_text(encoding="utf-8"))
    gptq_metadata = json.loads((out_dir / "gptq_checkpoint_metadata.json").read_text(encoding="utf-8"))
    payload_probe = json.loads((out_dir / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    assert semantic_graph["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert gptq_metadata["model_name"] == str(gptq_dir)
    assert gptq_metadata["target_model_name"] == "meta-llama/Llama-3.2-1B"
    assert gptq_metadata["metadata_source_model_name"] == str(gptq_dir)
    assert gptq_metadata["status"] == "parsed"
    assert gptq_metadata["checkpoint_source_preflight"]["status"] == "resolved_local_path"
    assert gptq_metadata["bits"] == 4
    assert gptq_metadata["group_size"] == 128
    assert gptq_metadata["complete_gptq_projection_metadata_count"] == 1
    assert payload_probe["status"] == "sampled"
    assert payload_probe["sampled_tensor_count"] == 3
    assert payload_probe["qweight_payload_words32_le_hex"] == ["0x00000000"]
    assert payload_probe["qweight_stream_probe"]["covers_first_memory_beat"] is False
    assert manifest["gptq_checkpoint_metadata"]["metadata_source_model_name"] == str(gptq_dir)
    assert manifest["gptq_checkpoint_metadata"]["target_model_name"] == "meta-llama/Llama-3.2-1B"
    assert manifest["gptq_checkpoint_metadata"]["source_preflight_status"] == "resolved_local_path"
    assert manifest["gptq_checkpoint_metadata"]["checkpoint_source_dependency"] == "satisfied_by_local_path"
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 1
    assert manifest["gptq_checkpoint_metadata"]["tensor_summary_count"] == 3
    assert manifest["gptq_checkpoint_metadata"]["tensor_summary_source"] == "safetensors_header"
    assert manifest["gptq_checkpoint_metadata"]["weight_layout_preflight_status"] == "blocked"
    assert manifest["gptq_payload_probe"]["status"] == "sampled"
    assert manifest["gptq_payload_probe"]["qweight_payload_word_count"] == 1
    assert manifest["gptq_payload_probe"]["qweight_stream_probe"]["covers_first_memory_beat"] is False
    assert manifest["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    assert q_task["gptq_payload_probe"]["status"] == "sampled"
    assert q_task["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == ["0x00000000"]
    assert q_task["gptq_payload_probe"]["qweight_stream_probe"]["covers_first_memory_beat"] is False
    assert q_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    subagent_tasks = json.loads((out_dir / "hdl_subagent_tasks.json").read_text(encoding="utf-8"))
    q_packet = next(packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_q_proj")
    assert q_packet["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert q_packet["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == ["0x00000000"]
    assert q_packet["gptq_payload_probe"]["qweight_stream_probe"]["covers_first_memory_beat"] is False
    assert "GPTQ qweight payload words32 LE: `['0x00000000']`" in q_packet["prompt"]
    assert "GPTQ qweight first memory beats 128b LE: `[]`" in q_packet["prompt"]
    assert not any(task["task_id"] == "real_gptq_checkpoint_metadata" for task in manifest["blocked_target_tasks"])
    assert any(task["task_id"] == "real_gptq_weight_layout_preflight" for task in manifest["blocked_target_tasks"])


def test_llm_agent_inspect_classifies_base_checkpoint_as_non_gptq_source(tmp_path: Path):
    base_dir = tmp_path / "local-base-llama"
    base_dir.mkdir()
    (base_dir / "config.json").write_text('{"model_type": "llama", "hidden_size": 64}', encoding="utf-8")
    _write_fake_safetensors_header(
        base_dir / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 2048
  gptq_checkpoint: {base_dir}
hardware:
  fpga_part: xczu7ev-ffvc1156-2-e
  target_clock_mhz: 200
  max_dsp: 1728
  max_bram: 312
  max_lut: 230000
  memory_data_width: 128
optimization:
  quantization: int4_gptq
  pruning: none
design:
  style: llm_decoder_streaming
  pe_count: 64
  activation_buffer: tiled_ping_pong_uram_bram
  weight_storage: external_ddr_gptq_packed
verification:
  enable_verilator: true
  enable_vivado_synth: true
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    out_dir = tmp_path / "inspect_base_out"

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=out_dir,
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "passed"
    gptq_metadata = json.loads((out_dir / "gptq_checkpoint_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((out_dir / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    readiness = json.loads((out_dir / "target_readiness_report.json").read_text(encoding="utf-8"))
    assert gptq_metadata["status"] == "metadata_json_without_quant_fields"
    assert gptq_metadata["checkpoint_quantization_artifact"]["classification"] == "base_or_unquantized_checkpoint"
    assert gptq_metadata["checkpoint_quantization_artifact"]["checkpoint_quantization_dependency"] == (
        "blocked_by_non_gptq_checkpoint_source"
    )
    assert manifest["gptq_checkpoint_metadata"]["quantization_artifact_classification"] == (
        "base_or_unquantized_checkpoint"
    )
    assert manifest["gptq_checkpoint_metadata"]["checkpoint_quantization_dependency"] == (
        "blocked_by_non_gptq_checkpoint_source"
    )
    source_block = next(
        task for task in manifest["blocked_target_tasks"] if task["task_id"] == "real_gptq_checkpoint_source"
    )
    assert source_block["classification"] == "base_or_unquantized_checkpoint"
    assert source_block["checkpoint_quantization_dependency"] == "blocked_by_non_gptq_checkpoint_source"
    summary_source_block = next(
        task
        for task in report["target_gate_summary"]["blocked_target_tasks"]
        if task["task_id"] == "real_gptq_checkpoint_source"
    )
    assert summary_source_block["classification"] == "base_or_unquantized_checkpoint"
    assert dispatch_plan["global_checkpoint_blocked_target_dependencies"] == [
        "real_gptq_checkpoint_metadata",
        "real_gptq_checkpoint_source",
    ]
    for wave in dispatch_plan["waves"]:
        assert "real_gptq_checkpoint_source" in wave["global_blocked_target_dependencies"]
        assert "real_gptq_checkpoint_source" in wave["blocked_target_dependencies"]
    assert readiness["status"] == "target_blocked"
    assert readiness["safe_to_claim_target_accelerator"] is False
    checkpoint_source_gate = next(gate for gate in readiness["gates"] if gate["gate"] == "checkpoint_source")
    assert checkpoint_source_gate["status"] == "blocked"
    assert checkpoint_source_gate["classification"] == "base_or_unquantized_checkpoint"


def test_gptq_weight_layout_preflight_requires_full_target_header_shapes():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    metadata = _target_gptq_metadata_from_graph(graph)

    preflight = build_gptq_weight_layout_preflight(graph, metadata)

    assert preflight["status"] == "passed"
    assert preflight["bits_compatible"] is True
    assert preflight["group_size_compatible"] is True
    assert preflight["target_compatible_projection_count"] == 7
    assert preflight["required_projection_count"] == 7
    assert all(check["status"] == "target_layout_compatible" for check in preflight["checks"])
    q_proj = next(check for check in preflight["checks"] if check["name"] == "q_proj")
    assert q_proj["layout_expectation"]["qweight_shape"] == [256, 2048]
    assert q_proj["layout_expectation"]["qzeros_shape"] == [16, 256]
    assert q_proj["layout_expectation"]["scales_shape"] == [16, 2048]
    assert q_proj["qweight_shape_matches_expected"] is True
    assert q_proj["qzeros_shape_matches_expected"] is True
    assert q_proj["scales_shape_matches_expected"] is True


def test_gptq_weight_layout_preflight_accepts_common_checkpoint_tensor_aliases(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    gptq_dir = tmp_path / "alias-layout-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantization_config.json").write_text(
        '{"quantization_config": {"quant_method": "gptq", "bits": "int4", "groupsize": "128"}}',
        encoding="utf-8",
    )
    _write_header_only_safetensors(
        gptq_dir / "model.safetensors",
        _target_gptq_alias_header_entries(graph),
    )

    metadata = inspect_gptq_checkpoint_metadata(str(gptq_dir))
    preflight = build_gptq_weight_layout_preflight(graph, metadata)

    assert metadata["status"] == "parsed"
    assert metadata["bits"] == 4
    assert metadata["group_size"] == 128
    assert metadata["complete_gptq_projection_metadata_count"] == 7
    assert preflight["status"] == "passed"
    assert preflight["target_compatible_projection_count"] == 7
    q_proj = next(check for check in preflight["checks"] if check["name"] == "q_proj")
    assert q_proj["qweight"]["key"].endswith(".q_weight")
    assert q_proj["qzeros"]["key"].endswith(".zeros")
    assert q_proj["scales"]["key"].endswith(".scale")


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("bits", "unsupported_or_missing_gptq_bits"),
        ("group_size", "missing_or_invalid_gptq_group_size"),
        ("qweight_shape", "qweight_packed_shape_mismatch"),
        ("qzeros_dtype", "tensor_dtype_mismatch"),
        ("scales_shape", "groupwise_metadata_shape_mismatch"),
        ("scales_byte_count", "groupwise_metadata_byte_count_mismatch"),
    ],
)
def test_gptq_weight_layout_preflight_reports_specific_negative_statuses(mutation: str, expected_status: str):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    metadata = _target_gptq_metadata_from_graph(graph)
    q_proj = next(item for item in metadata["projection_metadata"] if item["name"] == "q_proj")
    if mutation == "bits":
        metadata["bits"] = 8
    elif mutation == "group_size":
        metadata["group_size"] = 0
    elif mutation == "qweight_shape":
        key = q_proj["keys"]["qweight"][0]
        q_proj["tensor_summaries"][key]["shape"] = [2048, 256]
    elif mutation == "qzeros_dtype":
        key = q_proj["keys"]["qzeros"][0]
        q_proj["tensor_summaries"][key]["dtype"] = "F16"
    elif mutation == "scales_shape":
        key = q_proj["keys"]["scales"][0]
        q_proj["tensor_summaries"][key]["shape"] = [1]
    elif mutation == "scales_byte_count":
        key = q_proj["keys"]["scales"][0]
        q_proj["tensor_summaries"][key]["byte_count"] += 2

    preflight = build_gptq_weight_layout_preflight(graph, metadata)
    q_check = next(check for check in preflight["checks"] if check["name"] == "q_proj")

    assert preflight["status"] == "blocked"
    assert q_check["status"] == expected_status
    assert q_check["target_layout_compatible"] is False


def test_llm_agent_inspect_unblocks_layout_preflight_for_target_header_only_checkpoint(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    gptq_dir = tmp_path / "target-layout-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_header_only_safetensors(
        gptq_dir / "model.safetensors",
        _target_gptq_header_entries(graph),
    )
    spec = tmp_path / "target-layout.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 2048
  gptq_checkpoint: {gptq_dir}
hardware:
  fpga_part: xczu7ev-ffvc1156-2-e
  target_clock_mhz: 200
  max_dsp: 1728
  max_bram: 312
  max_lut: 230000
  memory_data_width: 128
optimization:
  quantization: int4_gptq
  pruning: none
design:
  style: llm_decoder_streaming
  pe_count: 64
  activation_buffer: tiled_ping_pong_uram_bram
  weight_storage: external_ddr_gptq_packed
verification:
  enable_verilator: true
  enable_vivado_synth: true
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "inspect_out"

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=load_config(spec),
        out_dir=out_dir,
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    manifest = json.loads((out_dir / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((out_dir / "gptq_weight_layout_preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["target_gate_summary"]["gptq_weight_layout_preflight"]["status"] == "passed"
    assert preflight["status"] == "passed"
    assert preflight["target_compatible_projection_count"] == 7
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 7
    assert manifest["gptq_checkpoint_metadata"]["weight_layout_preflight_status"] == "passed"
    assert report["target_gate_summary"]["gptq_payload_probe"]["status"] == "unavailable"
    assert manifest["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_mlir_model_analysis",
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    assert q_task["target_checkpoint_layout_dependency"] == "satisfied_by_header_preflight"
    assert q_task["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert q_task["gptq_weight_layout_preflight"]["requires_real_checkpoint_layout_before_target_claim"] is False
    assert q_task["target_weight_stream_plan"]["stream_plan_valid"] is True


def test_llm_agent_inspect_keeps_payload_blocker_when_only_q_proj_payload_is_sampled(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    entries = _target_gptq_header_entries(graph)
    q_base = "model.layers.0.self_attn.q_proj"
    gptq_dir = tmp_path / "partial-payload-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_safetensors_with_selected_payload(
        gptq_dir / "model.safetensors",
        entries,
        {
            f"{q_base}.qweight": bytes(range(64)),
            f"{q_base}.qzeros": bytes([0x11] * 64),
            f"{q_base}.scales": bytes([0x22] * 64),
        },
    )
    spec = tmp_path / "partial-payload.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 2048
  gptq_checkpoint: {gptq_dir}
hardware:
  fpga_part: xczu7ev-ffvc1156-2-e
  target_clock_mhz: 200
  max_dsp: 1728
  max_bram: 312
  max_lut: 230000
  memory_data_width: 128
optimization:
  quantization: int4_gptq
  pruning: none
design:
  style: llm_decoder_streaming
  pe_count: 64
  activation_buffer: tiled_ping_pong_uram_bram
  weight_storage: external_ddr_gptq_packed
verification:
  enable_verilator: true
  enable_vivado_synth: true
""",
        encoding="utf-8",
    )

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=load_config(spec),
        out_dir=tmp_path / "inspect_out",
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    payload_probe = json.loads((tmp_path / "inspect_out" / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "inspect_out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    stream_plan = json.loads((tmp_path / "inspect_out" / "projection_weight_stream_plan.json").read_text(encoding="utf-8"))
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    k_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "k_proj")
    payload_blocker = next(task for task in manifest["blocked_target_tasks"] if task["task_id"] == "real_gptq_payload_probe")

    assert report["status"] == "passed"
    assert report["target_gate_summary"]["gptq_weight_layout_preflight"]["status"] == "passed"
    assert report["target_gate_summary"]["projection_weight_stream_plan"]["projection_count"] == 7
    assert report["target_gate_summary"]["projection_weight_stream_plan"]["payload_satisfied_projection_count"] == 1
    assert report["target_gate_summary"]["projection_weight_stream_plan"][
        "target_scale_ready_for_all_projection_streaming"
    ] is False
    assert payload_probe["status"] == "sampled"
    assert payload_probe["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert payload_probe["sampled_projection_payload_probe_count"] == 1
    assert payload_probe["required_projection_payload_probe_count"] == 7
    assert payload_probe["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert manifest["gptq_payload_probe"]["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert manifest["projection_weight_stream_plan"] == stream_plan
    assert stream_plan["projection_count"] == 7
    assert stream_plan["target_stream_plan_valid_count"] == 7
    assert stream_plan["payload_satisfied_projection_count"] == 1
    assert stream_plan["payload_blocked_projection_count"] == 6
    assert stream_plan["payload_blocked_projections"] == ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert stream_plan["all_projection_layout_dependency"] == "satisfied_by_header_preflight"
    assert stream_plan["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert stream_plan["target_scale_ready_for_all_projection_streaming"] is False
    assert [item["name"] for item in stream_plan["projections"]] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    q_stream_item = next(item for item in stream_plan["projections"] if item["name"] == "q_proj")
    k_stream_item = next(item for item in stream_plan["projections"] if item["name"] == "k_proj")
    assert q_stream_item["qweight_payload_words32_le_hex"][:4] == [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
    ]
    assert q_stream_item["qweight_memory_beat_word_chunks32_le_hex"][0] == q_stream_item[
        "qweight_payload_words32_le_hex"
    ][:4]
    assert k_stream_item["qweight_payload_words32_le_hex"] == []
    assert q_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert k_task["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert payload_blocker["blocked_projection_count"] == 6
    assert set(payload_blocker["blocked_projections"]) == {"k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert payload_blocker["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    wave6 = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_6_projection_axi_read_command_adapter")
    wave7 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_7_projection_axi_read_data_channel_adapter"
    )
    wave8 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_8_projection_axi_read_transaction_adapter"
    )
    wave9 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_9_projection_axi_stream_integration"
    )
    wave10 = next(
        wave
        for wave in dispatch_plan["waves"]
        if wave["wave_id"] == "wave_10_decoder_child_axi_attention_datapath"
    )
    wave6_tasks = {task["task_id"]: task for task in wave6["implementation_tasks"]}
    wave7_tasks = {task["task_id"]: task for task in wave7["implementation_tasks"]}
    wave8_tasks = {task["task_id"]: task for task in wave8["implementation_tasks"]}
    wave9_tasks = {task["task_id"]: task for task in wave9["implementation_tasks"]}
    wave10_tasks = {task["task_id"]: task for task in wave10["implementation_tasks"]}
    assert "gptq_payload_probe" in wave6["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave7["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave8["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave9["direct_blocked_target_dependencies"]
    assert "gptq_payload_probe" in wave10["direct_blocked_target_dependencies"]
    assert wave6_tasks["projection_axi_read_command_adapter"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert wave6_tasks["projection_k_proj_axi_read_command_adapter"]["target_checkpoint_payload_dependency"] == (
        "blocked_by_gptq_payload_probe"
    )
    assert wave7_tasks["projection_axi_read_data_channel_adapter"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert wave7_tasks["projection_k_proj_axi_read_data_channel_adapter"]["target_checkpoint_payload_dependency"] == (
        "blocked_by_gptq_payload_probe"
    )
    assert wave8_tasks["projection_axi_read_transaction_adapter"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert wave9_tasks["projection_axi_stream_integration"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    blocked_projection_task_suffixes = {
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }
    for projection_name, task_suffix in blocked_projection_task_suffixes.items():
        assert wave8_tasks[
            f"projection_{task_suffix}_axi_read_transaction_adapter"
        ]["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe", projection_name
        assert wave9_tasks[
            f"projection_{task_suffix}_axi_stream_integration"
        ]["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe", projection_name
    assert wave10_tasks["decoder_child_axi_attention_datapath"]["target_checkpoint_payload_dependency"] == (
        "blocked_by_gptq_payload_probe"
    )


def test_llm_agent_inspect_unblocks_target_payload_probe_for_local_gptq_payload(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    entries = _target_gptq_header_entries(graph)
    payload_prefixes = {}
    expected_words_by_projection = {}
    for projection_idx, projection_name in enumerate(graph["partition"]["gemm"]):
        base = (
            f"model.layers.0.self_attn.{projection_name}"
            if projection_name in {"q_proj", "k_proj", "v_proj", "o_proj"}
            else f"model.layers.0.mlp.{projection_name}"
        )
        qweight_prefix = bytes((projection_idx * 16 + byte_idx) & 0xFF for byte_idx in range(64))
        payload_prefixes[f"{base}.qweight"] = qweight_prefix
        payload_prefixes[f"{base}.qzeros"] = bytes([0x11 + projection_idx] * 64)
        payload_prefixes[f"{base}.scales"] = bytes([0x22 + projection_idx] * 64)
        expected_words_by_projection[projection_name] = [
            f"0x{int.from_bytes(qweight_prefix[idx : idx + 4], 'little'):08x}" for idx in range(0, 64, 4)
        ]
    expected_words = expected_words_by_projection["q_proj"]
    expected_beats = [
        "0x0f0e0d0c0b0a09080706050403020100",
        "0x1f1e1d1c1b1a19181716151413121110",
        "0x2f2e2d2c2b2a29282726252423222120",
        "0x3f3e3d3c3b3a39383736353433323130",
    ]
    gptq_dir = tmp_path / "target-payload-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_safetensors_with_selected_payload(
        gptq_dir / "model.safetensors",
        entries,
        payload_prefixes,
    )
    spec = tmp_path / "target-payload.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  sequence_length: 2048
  gptq_checkpoint: {gptq_dir}
hardware:
  fpga_part: xczu7ev-ffvc1156-2-e
  target_clock_mhz: 200
  max_dsp: 1728
  max_bram: 312
  max_lut: 230000
  memory_data_width: 128
optimization:
  quantization: int4_gptq
  pruning: none
design:
  style: llm_decoder_streaming
  pe_count: 64
  activation_buffer: tiled_ping_pong_uram_bram
  weight_storage: external_ddr_gptq_packed
verification:
  enable_verilator: true
  enable_vivado_synth: true
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "inspect_out"

    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=load_config(spec),
        out_dir=out_dir,
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    payload_probe = json.loads((out_dir / "gptq_payload_probe.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    stream_plan = json.loads((out_dir / "projection_weight_stream_plan.json").read_text(encoding="utf-8"))
    readiness = json.loads((out_dir / "target_readiness_report.json").read_text(encoding="utf-8"))
    subagent_tasks = json.loads((out_dir / "hdl_subagent_tasks.json").read_text(encoding="utf-8"))
    q_packet = next(packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_q_proj")

    assert report["status"] == "passed"
    assert report["target_gate_summary"]["gptq_weight_layout_preflight"]["status"] == "passed"
    assert report["target_gate_summary"]["gptq_payload_probe"]["status"] == "sampled"
    assert report["target_gate_summary"]["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert report["target_gate_summary"]["gptq_payload_probe"]["projection_payload_probe_count"] == 7
    assert report["target_gate_summary"]["gptq_payload_probe"]["sampled_projection_payload_probe_count"] == 7
    assert report["target_gate_summary"]["gptq_payload_probe"]["all_projection_payload_dependency"] == (
        "satisfied_by_payload_probe"
    )
    assert report["target_gate_summary"]["projection_weight_stream_plan"]["projection_count"] == 7
    assert report["target_gate_summary"]["projection_weight_stream_plan"]["target_stream_plan_valid_count"] == 7
    assert report["target_gate_summary"]["projection_weight_stream_plan"]["payload_satisfied_projection_count"] == 7
    assert report["target_gate_summary"]["projection_weight_stream_plan"][
        "target_scale_ready_for_all_projection_streaming"
    ] is True
    assert payload_probe["status"] == "sampled"
    assert payload_probe["sampled_tensor_count"] == 3
    assert payload_probe["projection_payload_probe_count"] == 7
    assert payload_probe["sampled_projection_payload_probe_count"] == 7
    assert payload_probe["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert set(payload_probe["projection_payload_probes"]) == set(graph["partition"]["gemm"])
    assert payload_probe["qweight_payload_words32_le_hex"] == expected_words
    assert payload_probe["qweight_stream_probe"]["covers_first_memory_beat"] is True
    assert payload_probe["qweight_stream_probe"]["first_memory_beats_128b_le_hex"] == expected_beats
    assert manifest["gptq_checkpoint_metadata"]["status"] == "parsed"
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 7
    assert manifest["gptq_checkpoint_metadata"]["weight_layout_preflight_status"] == "passed"
    assert manifest["gptq_payload_probe"]["status"] == "sampled"
    assert manifest["gptq_payload_probe"]["qweight_payload_word_count"] == 16
    assert manifest["gptq_payload_probe"]["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert manifest["gptq_payload_probe"]["projection_payload_probe_count"] == 7
    assert manifest["gptq_payload_probe"]["sampled_projection_payload_probe_count"] == 7
    assert manifest["gptq_payload_probe"]["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert manifest["projection_weight_stream_plan"] == stream_plan
    assert stream_plan["projection_count"] == 7
    assert stream_plan["target_stream_plan_valid_count"] == 7
    assert stream_plan["payload_satisfied_projection_count"] == 7
    assert stream_plan["layout_blocked_projection_count"] == 0
    assert stream_plan["payload_blocked_projection_count"] == 0
    assert stream_plan["all_projection_layout_dependency"] == "satisfied_by_header_preflight"
    assert stream_plan["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert stream_plan["target_scale_ready_for_all_projection_streaming"] is True
    assert readiness["status"] == "target_blocked"
    assert readiness["safe_to_spawn_bounded_subagents"] is True
    assert readiness["safe_to_claim_target_accelerator"] is False
    assert readiness["dispatch"]["target_preflight_ready_wave_count"] == 0
    assert readiness["dispatch"]["bounded_fixture_wave_count"] > 0
    assert readiness["dispatch"]["global_blocked_target_dependencies"] == ["real_mlir_model_analysis"]
    assert readiness["target_preflight"]["status"] == "blocked"
    assert readiness["target_preflight"]["preflight_blockers"] == ["real_mlir_model_analysis"]
    assert readiness["target_preflight"]["safe_to_dispatch_target_preflight_subagents"] is False
    projection_gate = next(gate for gate in readiness["gates"] if gate["gate"] == "projection_weight_stream_plan")
    assert projection_gate["status"] == "passed"
    mlir_gate = next(gate for gate in readiness["gates"] if gate["gate"] == "mlir_model_analysis")
    assert mlir_gate["status"] == "blocked"
    assert {task["task_id"] for task in readiness["blocked_target_tasks"]} == {
        "real_mlir_model_analysis",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    assert [item["name"] for item in stream_plan["projections"]] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert "DDR_controller_integration" in stream_plan["does_not_claim"]
    assert "AXI_master_implementation" in stream_plan["does_not_claim"]
    assert "full_qweight_payload_streaming" in stream_plan["does_not_claim"]
    assert "numeric_GPTQ_correctness" in stream_plan["does_not_claim"]
    assert "full_LLaMA_model_execution" in stream_plan["does_not_claim"]
    assert "board_level_ZCU104_signoff" in stream_plan["does_not_claim"]
    for stream_item in stream_plan["projections"]:
        projection_name = stream_item["name"]
        assert stream_item["qweight_payload_words32_le_hex"] == expected_words_by_projection[projection_name]
        assert stream_item["qweight_memory_beat_word_chunks32_le_hex"][0] == (
            expected_words_by_projection[projection_name][:4]
        )
        assert stream_item["qweight_payload_covers_first_memory_beat"] is True
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_mlir_model_analysis",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    decoder_child_task = next(
        task
        for task in manifest["integration_tasks"]
        if task["task_id"] == "decoder_child_axi_attention_datapath"
    )
    assert decoder_child_task["attention_projection_stream_tasks"] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
    ]
    assert decoder_child_task["target_checkpoint_layout_dependency"] == "satisfied_by_header_preflight"
    assert decoder_child_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert {
        name: stream["target_checkpoint_payload_dependency"]
        for name, stream in decoder_child_task["target_attention_projection_streams"].items()
    } == {
        "q_proj": "satisfied_by_payload_probe",
        "k_proj": "satisfied_by_payload_probe",
        "v_proj": "satisfied_by_payload_probe",
        "o_proj": "satisfied_by_payload_probe",
    }
    for projection_task in manifest["projection_tasks"]:
        projection_name = projection_task["semantic_op"]
        assert projection_task["target_checkpoint_layout_dependency"] == "satisfied_by_header_preflight"
        assert projection_task["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert projection_task["target_weight_stream_plan"]["stream_plan_valid"] is True
        assert projection_task["gptq_payload_probe"]["projection"] == projection_name
        assert projection_task["gptq_payload_probe"]["probe_projection"] == projection_name
        assert projection_task["gptq_payload_probe"]["selected_projection"] is True
        assert projection_task["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == (
            expected_words_by_projection[projection_name]
        )
        assert projection_task["gptq_payload_probe"]["qweight_stream_probe"]["memory_beat_word_chunks32_le_hex"][0] == (
            expected_words_by_projection[projection_name][:4]
        )
    assert q_packet["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert q_packet["gptq_payload_probe"]["qweight_payload_words32_le_hex"] == expected_words
    assert q_packet["gptq_payload_probe"]["qweight_stream_probe"]["first_memory_beats_128b_le_hex"] == expected_beats
    assert f"GPTQ qweight payload words32 LE: `{expected_words}`" in q_packet["prompt"]
    assert f"GPTQ qweight first memory beats 128b LE: `{expected_beats}`" in q_packet["prompt"]


def test_cli_inspect_emits_dispatch_plan_artifacts(tmp_path: Path):
    rc = cli_main(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "inspect",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "hdl_subagent_dispatch_plan.json").exists()
    assert (tmp_path / "hdl_subagent_wave_status.json").exists()
    assert (tmp_path / "hdl_subagent_execution_manifest.json").exists()
    assert (tmp_path / "hdl_subagent_tasks.json").exists()
    assert (tmp_path / "skill_update_candidate_template.json").exists()
    assert (tmp_path / "subagent_prompts").is_dir()
    report = json.loads((tmp_path / "llm_agent_report.json").read_text(encoding="utf-8"))
    inspect_step = next(step for step in report["steps"] if step["name"] == "inspect_semantic_mlir")
    assert inspect_step["hdl_subagent_dispatch_plan"] == "hdl_subagent_dispatch_plan.json"
    assert inspect_step["hdl_subagent_wave_status"] == "hdl_subagent_wave_status.json"
    assert inspect_step["hdl_subagent_execution_manifest"] == "hdl_subagent_execution_manifest.json"
    assert inspect_step["skill_update_candidate_template"] == "skill_update_candidate_template.json"
    assert inspect_step["verification_prompts"] == "verification_prompts"
    dispatch_plan = json.loads((tmp_path / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    wave_status = json.loads((tmp_path / "hdl_subagent_wave_status.json").read_text(encoding="utf-8"))
    execution_manifest = json.loads((tmp_path / "hdl_subagent_execution_manifest.json").read_text(encoding="utf-8"))
    assert dispatch_plan["wave_count"] == 20
    assert [len(wave["implementation_tasks"]) for wave in dispatch_plan["waves"]] == [
        7,
        8,
        1,
        1,
        1,
        1,
        7,
        7,
        7,
        7,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert wave_status["next_dispatchable_waves"] == [
        "wave_1_projection_kernels",
        "wave_1_non_gemm_kernels",
    ]
    assert execution_manifest["spawn_entry_count"] == 15
    assert execution_manifest["implementation_spawn_count"] == 15
    assert execution_manifest["verification_spawn_count"] == 0
    assert not list(tmp_path.glob("*.sv"))
    assert not list(tmp_path.glob("*.v"))


def test_cli_subagents_status_refreshes_execution_manifest_from_evidence_root(tmp_path: Path):
    inspect_dir = tmp_path / "inspect"
    rc = cli_main(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "inspect",
            "--out",
            str(inspect_dir),
        ]
    )
    assert rc == 0
    dispatch_plan_path = inspect_dir / "hdl_subagent_dispatch_plan.json"
    dispatch_plan = json.loads(dispatch_plan_path.read_text(encoding="utf-8"))
    projection_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    evidence_root = tmp_path / "evidence"
    for task in projection_wave["implementation_tasks"]:
        evidence_dir = evidence_root / Path(task["expected_evidence_dir"]).name
        _write_minimal_passed_kernel_evidence(evidence_dir)

    status_dir = tmp_path / "status"
    rc = cli_main(
        [
            "subagents",
            "status",
            "--dispatch-plan",
            str(dispatch_plan_path),
            "--evidence-root",
            str(evidence_root),
            "--out",
            str(status_dir),
        ]
    )

    assert rc == 0
    wave_status = json.loads((status_dir / "hdl_subagent_wave_status.json").read_text(encoding="utf-8"))
    execution_manifest = json.loads((status_dir / "hdl_subagent_execution_manifest.json").read_text(encoding="utf-8"))
    parent_loop_state = json.loads((status_dir / "parent_loop_state.json").read_text(encoding="utf-8"))
    feedback_packet = json.loads((status_dir / "feedback_packet.json").read_text(encoding="utf-8"))
    retry_plan = json.loads((status_dir / "retry_plan.json").read_text(encoding="utf-8"))
    full_execution = json.loads((status_dir / "full_llama_execution_readiness.json").read_text(encoding="utf-8"))
    board_signoff = json.loads((status_dir / "board_zcu104_signoff_readiness.json").read_text(encoding="utf-8"))
    full_execution_template = json.loads(
        (status_dir / "full_llama_execution_evidence_template.json").read_text(encoding="utf-8")
    )
    board_signoff_template = json.loads(
        (status_dir / "board_zcu104_signoff_evidence_template.json").read_text(encoding="utf-8")
    )
    board_signoff_agent_task = json.loads(
        (status_dir / "board_zcu104_signoff_evidence_agent_task.json").read_text(encoding="utf-8")
    )
    board_signoff_manifest = json.loads(
        (status_dir / "board_zcu104_signoff_execution_manifest.json").read_text(encoding="utf-8")
    )
    board_wrapper_agent_task = json.loads(
        (status_dir / "zcu104_board_wrapper_axi_bridge_agent_task.json").read_text(encoding="utf-8")
    )
    board_wrapper_manifest = json.loads(
        (status_dir / "zcu104_board_wrapper_axi_bridge_execution_manifest.json").read_text(encoding="utf-8")
    )
    model_harness_agent_task = json.loads(
        (status_dir / "model_level_execution_harness_agent_task.json").read_text(encoding="utf-8")
    )
    model_harness_manifest = json.loads(
        (status_dir / "model_level_execution_harness_manifest.json").read_text(encoding="utf-8")
    )
    full_execution_agent_task = json.loads(
        (status_dir / "full_llama_execution_evidence_agent_task.json").read_text(encoding="utf-8")
    )
    target_evidence_execution = json.loads(
        (status_dir / "target_evidence_execution_manifest.json").read_text(encoding="utf-8")
    )
    full_execution_agent_prompt = (
        status_dir / "target_evidence_prompts" / "full_llama_execution_evidence_agent.md"
    ).read_text(encoding="utf-8")
    model_harness_agent_prompt = (
        status_dir / "target_evidence_prompts" / "model_level_execution_harness_agent.md"
    ).read_text(encoding="utf-8")
    target_evidence_spawn_instructions = (status_dir / "target_evidence_spawn_instructions.md").read_text(
        encoding="utf-8"
    )
    board_signoff_spawn_instructions = (status_dir / "board_zcu104_signoff_spawn_instructions.md").read_text(
        encoding="utf-8"
    )
    board_wrapper_agent_prompt = (
        status_dir / "board_implementation_prompts" / "zcu104_board_wrapper_axi_bridge_agent.md"
    ).read_text(encoding="utf-8")
    board_wrapper_spawn_instructions = (
        status_dir / "zcu104_board_wrapper_axi_bridge_spawn_instructions.md"
    ).read_text(encoding="utf-8")
    model_harness_spawn_instructions = (
        status_dir / "model_level_execution_harness_spawn_instructions.md"
    ).read_text(encoding="utf-8")
    spawn_instructions = (status_dir / "codex_spawn_instructions.md").read_text(encoding="utf-8")
    waves = {wave["wave_id"]: wave for wave in wave_status["waves"]}
    assert wave_status["collection_root"] == str(evidence_root)
    assert waves["wave_1_projection_kernels"]["status"] == "ready_for_verification"
    assert execution_manifest["verification_spawn_count"] == 1
    assert execution_manifest["spawn_batch_count"] == 2
    assert execution_manifest["parallel_spawn_allowed"] is True
    assert "# Codex Sub-Agent Spawn Instructions" in spawn_instructions
    assert "verification_prompts/wave_1_projection_kernels__verification.md" in spawn_instructions
    assert "subagent_prompts/non_gemm_input_layernorm__implementation.md" in spawn_instructions
    assert "You are the Codex verification sub-agent for this wave." in spawn_instructions
    assert "The Parent Agent is the only orchestrator" in spawn_instructions
    assert parent_loop_state["artifact"] == "parent_loop_state"
    assert parent_loop_state["agent_hierarchy"]["parent_agent"] == "single_orchestrator"
    assert parent_loop_state["agent_hierarchy"]["all_non_parent_workers_are_subagents"] is True
    assert parent_loop_state["next_parent_action"] == "spawn_ready_subagents"
    assert feedback_packet["artifact"] == "feedback_packet"
    assert feedback_packet["agent_hierarchy"]["subagents_may_spawn_subagents"] is False
    assert retry_plan["artifact"] == "retry_plan"
    spawn_batches = {batch["wave_id"]: batch for batch in execution_manifest["spawn_batches"]}
    assert spawn_batches["wave_1_projection_kernels"]["spawn_kind"] == "verification_agent"
    assert spawn_batches["wave_1_projection_kernels"]["parallel_spawn_allowed"] is False
    assert spawn_batches["wave_1_projection_kernels"]["entry_count"] == 1
    assert spawn_batches["wave_1_non_gemm_kernels"]["spawn_kind"] == "implementation_agent"
    assert spawn_batches["wave_1_non_gemm_kernels"]["parallel_spawn_allowed"] is True
    assert spawn_batches["wave_1_non_gemm_kernels"]["entry_count"] == 8
    assert any(
        entry["spawn_kind"] == "verification_agent"
        and entry["wave_id"] == "wave_1_projection_kernels"
        and entry["must_not_edit_source_files"] is True
        and entry["may_write_generated_evidence"] is False
        for entry in execution_manifest["spawn_entries"]
    )
    assert execution_manifest["implementation_spawn_count"] == 8
    assert all(
        entry["wave_id"] == "wave_1_non_gemm_kernels"
        for entry in execution_manifest["spawn_entries"]
        if entry["spawn_kind"] == "implementation_agent"
    )
    assert "sub-agent execution occurred" in execution_manifest["does_not_claim"]
    assert full_execution["artifact"] == "full_llama_execution_readiness"
    assert full_execution["status"] == "blocked_by_target_preflight"
    assert "real_mlir_model_analysis" in full_execution["target_preflight"]["preflight_blockers"]
    assert full_execution["safe_to_clear_full_llama_model_execution_blocker"] is False
    assert board_signoff["artifact"] == "board_zcu104_signoff_readiness"
    assert board_signoff["status"] == "blocked_by_full_llama_execution"
    assert board_signoff["safe_to_clear_board_level_zcu104_signoff_blocker"] is False
    assert full_execution_template["artifact"] == "full_llama_execution_evidence_template"
    assert full_execution_template["write_to"] == str(evidence_root / "full_llama_execution_evidence.json")
    assert full_execution_template["template"]["artifact"] == "full_llama_execution_evidence"
    assert board_signoff_template["artifact"] == "board_zcu104_signoff_evidence_template"
    assert board_signoff_template["write_to"] == str(evidence_root / "board_zcu104_signoff_evidence.json")
    assert board_signoff_template["template"]["artifact"] == "board_zcu104_signoff_evidence"
    assert board_signoff_agent_task["artifact"] == "board_zcu104_signoff_evidence_agent_task"
    assert board_signoff_agent_task["ready_to_spawn"] is False
    assert any(
        "full_llama_execution_readiness must be passed" in failure
        for failure in board_signoff_agent_task["spawn_precondition_failures"]
    )
    assert board_signoff_manifest["artifact"] == "target_evidence_execution_manifest"
    assert board_signoff_manifest["spawn_entry_count"] == 0
    assert "No implementation or verification sub-agent is ready" in board_signoff_spawn_instructions
    assert board_wrapper_agent_task["artifact"] == "zcu104_board_wrapper_axi_bridge_agent_task"
    assert board_wrapper_agent_task["ready_to_spawn"] is False
    assert any(
        "full_llama_execution_readiness must be passed" in failure
        for failure in board_wrapper_agent_task["spawn_precondition_failures"]
    )
    assert board_wrapper_manifest["artifact"] == "target_evidence_execution_manifest"
    assert board_wrapper_manifest["spawn_entry_count"] == 0
    assert "Implementation Sub-Agent Task: zcu104_board_wrapper_axi_bridge" in board_wrapper_agent_prompt
    assert "No implementation or verification sub-agent is ready" in board_wrapper_spawn_instructions
    assert model_harness_agent_task["artifact"] == "model_level_execution_harness_agent_task"
    assert model_harness_agent_task["ready_to_spawn"] is False
    assert "target_preflight.status must be passed" in model_harness_agent_task["spawn_precondition_failures"]
    assert model_harness_manifest["artifact"] == "target_evidence_execution_manifest"
    assert model_harness_manifest["spawn_entry_count"] == 0
    assert "Target Evidence Sub-Agent Task: model_level_execution_harness" in model_harness_agent_prompt
    assert "No implementation or verification sub-agent is ready" in model_harness_spawn_instructions
    assert full_execution_agent_task["artifact"] == "full_llama_execution_evidence_agent_task"
    assert full_execution_agent_task["ready_to_spawn"] is False
    assert "target_preflight.status must be passed" in full_execution_agent_task["spawn_precondition_failures"]
    assert full_execution_agent_task["prompt_file"] == (
        "target_evidence_prompts/full_llama_execution_evidence_agent.md"
    )
    assert "Target Evidence Sub-Agent Task: full_llama_execution" in full_execution_agent_prompt
    assert str(evidence_root / "full_llama_execution_evidence.json") in full_execution_agent_prompt
    assert target_evidence_execution["artifact"] == "target_evidence_execution_manifest"
    assert target_evidence_execution["spawn_entry_count"] == 0
    assert target_evidence_execution["target_evidence_spawn_count"] == 0
    assert target_evidence_execution["blocked_target_evidence_tasks"][0]["task_id"] == "full_llama_execution_evidence"
    assert "No implementation or verification sub-agent is ready" in target_evidence_spawn_instructions


def test_cli_inspect_accepts_gptq_checkpoint_override(tmp_path: Path):
    gptq_dir = tmp_path / "cli-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_fake_safetensors_header(
        gptq_dir / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ],
    )

    rc = cli_main(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--gptq-checkpoint",
            str(gptq_dir),
            "--mode",
            "inspect",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    report = json.loads((tmp_path / "out" / "agent_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    assert report["config"]["model"]["gptq_checkpoint"] == str(gptq_dir)
    assert manifest["gptq_checkpoint_metadata"]["metadata_source_model_name"] == str(gptq_dir)
    assert manifest["gptq_checkpoint_metadata"]["target_model_name"] == "meta-llama/Llama-3.2-1B"
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 1
    assert not any(task["task_id"] == "real_gptq_checkpoint_metadata" for task in manifest["blocked_target_tasks"])
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_mlir_model_analysis",
        "real_gptq_weight_layout_preflight",
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    assert not list((tmp_path / "out").glob("*.sv"))
    assert not list((tmp_path / "out").glob("*.v"))


def test_cli_inspect_accepts_mlir_graph_override(tmp_path: Path):
    mlir_path = tmp_path / "provided_llama_block.mlir"
    mlir_path.write_text(_full_llama_block_mlir(), encoding="utf-8")

    rc = cli_main(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mlir-graph",
            str(mlir_path),
            "--mode",
            "inspect",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    report = json.loads((tmp_path / "out" / "agent_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "out" / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    assert report["config"]["model"]["mlir_graph"] == str(mlir_path)
    assert report["target_gate_summary"]["mlir_model_analysis"]["status"] == "passed"
    assert manifest["mlir_model_analysis"]["status"] == "passed"
    assert not any(task["task_id"] == "real_mlir_model_analysis" for task in manifest["blocked_target_tasks"])
    assert "real_mlir_model_analysis" not in dispatch_plan["global_blocked_target_dependencies"]
    assert not list((tmp_path / "out").glob("*.sv"))
    assert not list((tmp_path / "out").glob("*.v"))


def test_cli_inspect_marks_target_preflight_ready_when_mlir_and_gptq_payload_pass(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    gptq_dir = _write_target_payload_gptq_checkpoint(tmp_path, graph)
    mlir_path = tmp_path / "provided_llama_block.mlir"
    mlir_path.write_text(_full_llama_block_mlir(), encoding="utf-8")

    rc = cli_main(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--gptq-checkpoint",
            str(gptq_dir),
            "--mlir-graph",
            str(mlir_path),
            "--mode",
            "inspect",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    readiness = json.loads((tmp_path / "out" / "target_readiness_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    subagent_tasks = json.loads((tmp_path / "out" / "hdl_subagent_tasks.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((tmp_path / "out" / "hdl_subagent_dispatch_plan.json").read_text(encoding="utf-8"))
    execution_manifest = json.loads((tmp_path / "out" / "hdl_subagent_execution_manifest.json").read_text(encoding="utf-8"))
    spawn_instructions = (tmp_path / "out" / "codex_spawn_instructions.md").read_text(encoding="utf-8")
    q_packet = next(packet for packet in subagent_tasks["packets"] if packet["task_id"] == "projection_q_proj")
    q_dispatch_task = next(
        task
        for wave in dispatch_plan["waves"]
        for task in wave["implementation_tasks"]
        if task["task_id"] == "projection_q_proj"
    )
    projection_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    q_spawn_entry = next(entry for entry in execution_manifest["spawn_entries"] if entry.get("task_id") == "projection_q_proj")
    joined_q_commands = " ".join(q_packet["required_commands"])
    assert manifest["mlir_model_analysis"]["status"] == "passed"
    assert manifest["source_replay"] == {
        "model_name": "meta-llama/Llama-3.2-1B",
        "gptq_checkpoint": str(gptq_dir),
        "mlir_graph": str(mlir_path),
        "model_structure_source": "mlir",
    }
    assert subagent_tasks["source_replay"] == manifest["source_replay"]
    assert dispatch_plan["source_replay"] == manifest["source_replay"]
    assert projection_wave["source_replay"] == manifest["source_replay"]
    assert q_dispatch_task["source_replay"] == manifest["source_replay"]
    assert q_spawn_entry["source_replay"] == manifest["source_replay"]
    assert manifest["gptq_checkpoint_metadata"]["status"] == "parsed"
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 7
    assert manifest["gptq_payload_probe"]["all_projection_payload_dependency"] == "satisfied_by_payload_probe"
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    assert dispatch_plan["global_blocked_target_dependencies"] == []
    assert readiness["dispatch"]["bounded_fixture_wave_count"] == 0
    assert readiness["dispatch"]["target_preflight_ready_wave_count"] == dispatch_plan["wave_count"]
    assert readiness["target_preflight"]["status"] == "passed"
    assert readiness["target_preflight"]["preflight_blockers"] == []
    assert readiness["target_preflight"]["safe_to_dispatch_target_preflight_subagents"] is True
    assert q_packet["source_replay"] == manifest["source_replay"]
    assert q_dispatch_task["required_commands"] == q_packet["required_commands"]
    assert q_spawn_entry["required_commands"] == q_packet["required_commands"]
    assert "--model meta-llama/Llama-3.2-1B" in joined_q_commands
    assert f"--gptq-checkpoint {gptq_dir}" in joined_q_commands
    assert f"--mlir-graph {mlir_path}" in joined_q_commands
    assert f"--gptq-checkpoint {gptq_dir}" in " ".join(q_spawn_entry["required_commands"])
    assert f"--mlir-graph {mlir_path}" in " ".join(q_spawn_entry["required_commands"])
    assert "Replay model name: `meta-llama/Llama-3.2-1B`" in q_packet["prompt"]
    assert f"Replay GPTQ checkpoint override: `{gptq_dir}`" in q_packet["prompt"]
    assert f"Replay MLIR graph override: `{mlir_path}`" in q_packet["prompt"]
    assert f"--gptq-checkpoint {gptq_dir}" in q_packet["prompt"]
    assert f"--mlir-graph {mlir_path}" in q_packet["prompt"]
    assert "Replay model name: `meta-llama/Llama-3.2-1B`" in spawn_instructions
    assert f"Replay GPTQ checkpoint: `{gptq_dir}`" in spawn_instructions
    assert f"Replay MLIR graph: `{mlir_path}`" in spawn_instructions
    assert f"--gptq-checkpoint {gptq_dir}" in spawn_instructions
    assert f"--mlir-graph {mlir_path}" in spawn_instructions
    verification_prompt = (tmp_path / "out" / projection_wave["verification_agent"]["prompt_file"]).read_text(
        encoding="utf-8"
    )
    assert "Replay model name: `meta-llama/Llama-3.2-1B`" in verification_prompt
    assert f"Replay GPTQ checkpoint override: `{gptq_dir}`" in verification_prompt
    assert f"Replay MLIR graph override: `{mlir_path}`" in verification_prompt
    for task in projection_wave["implementation_tasks"]:
        evidence_dir = tmp_path / "out" / Path(task["expected_evidence_dir"]).name
        _write_minimal_passed_kernel_evidence(evidence_dir)
    ready_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path / "out")
    ready_execution = build_hdl_subagent_execution_manifest(dispatch_plan, ready_status)
    verification_entry = next(
        entry for entry in ready_execution["spawn_entries"] if entry["spawn_kind"] == "verification_agent"
    )
    verification_markdown = build_codex_spawn_instructions(ready_execution)
    assert verification_entry["source_replay"] == manifest["source_replay"]
    assert "Replay model name: `meta-llama/Llama-3.2-1B`" in verification_markdown
    assert f"Replay GPTQ checkpoint: `{gptq_dir}`" in verification_markdown
    assert f"Replay MLIR graph: `{mlir_path}`" in verification_markdown
    assert all(wave["target_scope"] == "target_preflight_satisfied_or_not_applicable" for wave in dispatch_plan["waves"])
    assert readiness["safe_to_spawn_bounded_subagents"] is True
    assert readiness["safe_to_claim_target_accelerator"] is False


def test_cli_plan_accepts_gptq_checkpoint_override(tmp_path: Path):
    rc = cli_main(
        [
            "plan",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--gptq-checkpoint",
            "/tmp/local-gptq",
            "--out",
            str(tmp_path / "plan"),
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan" / "llm_accelerator_plan.json").read_text(encoding="utf-8"))
    assert plan["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert plan["model"]["gptq_checkpoint_source"] == "/tmp/local-gptq"
    assert plan["model"]["gptq_checkpoint_source_kind"] == "configured_override"


def test_cli_plan_accepts_mlir_graph_override(tmp_path: Path):
    mlir_path = tmp_path / "provided_llama_block.mlir"
    mlir_path.write_text(_full_llama_block_mlir(), encoding="utf-8")

    rc = cli_main(
        [
            "plan",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mlir-graph",
            str(mlir_path),
            "--out",
            str(tmp_path / "plan"),
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan" / "llm_accelerator_plan.json").read_text(encoding="utf-8"))
    assert plan["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert plan["model"]["mlir_graph_source"] == str(mlir_path)
    assert plan["model"]["mlir_graph_source_kind"] == "configured_override"


def test_cli_generate_alias_accepts_gptq_checkpoint_override(tmp_path: Path):
    gptq_dir = tmp_path / "generate-gptq"
    gptq_dir.mkdir()
    (gptq_dir / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_fake_safetensors_header(
        gptq_dir / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ],
    )

    rc = cli_main(
        [
            "generate",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--gptq-checkpoint",
            str(gptq_dir),
            "--mode",
            "inspect",
            "--out",
            str(tmp_path / "generate_out"),
        ]
    )

    assert rc == 0
    manifest = json.loads((tmp_path / "generate_out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gptq_checkpoint_metadata"]["metadata_source_model_name"] == str(gptq_dir)
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 1
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_mlir_model_analysis",
        "real_gptq_weight_layout_preflight",
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }


def test_cli_generate_alias_accepts_mlir_graph_override(tmp_path: Path):
    mlir_path = tmp_path / "provided_llama_block.mlir"
    mlir_path.write_text(_full_llama_block_mlir(), encoding="utf-8")

    rc = cli_main(
        [
            "generate",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mlir-graph",
            str(mlir_path),
            "--mode",
            "inspect",
            "--out",
            str(tmp_path / "generate_out"),
        ]
    )

    assert rc == 0
    report = json.loads((tmp_path / "generate_out" / "agent_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "generate_out" / "hdl_task_manifest.json").read_text(encoding="utf-8"))
    assert report["config"]["model"]["mlir_graph"] == str(mlir_path)
    assert manifest["mlir_model_analysis"]["status"] == "passed"
    assert not any(task["task_id"] == "real_mlir_model_analysis" for task in manifest["blocked_target_tasks"])


def test_semantic_graph_uses_hf_config_when_available(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            assert model_name == "local-llama-fixture"
            assert local_files_only is True
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=128,
                intermediate_size=384,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=16,
                num_hidden_layers=3,
                max_position_embeddings=4096,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    assert graph["model"]["metadata_source"] == "huggingface_auto_config_local_cache"
    assert graph["model"]["metadata_resolution"]["status"] == "resolved"
    assert graph["model"]["hidden_size"] == 128
    assert graph["model"]["intermediate_size"] == 384
    assert graph["model"]["decoder_layers"] == 3
    assert graph["model"]["sequence_length"] == cfg.model.sequence_length
    assert graph["projection_shapes"]["q_proj"] == {"rows": 128, "cols": 128}
    assert graph["projection_shapes"]["k_proj"] == {"rows": 32, "cols": 128}
    assert graph["projection_shapes"]["down_proj"] == {"rows": 128, "cols": 384}


def test_semantic_graph_falls_back_when_hf_config_unavailable(monkeypatch):
    class MissingAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            raise OSError("not cached")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=MissingAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    assert graph["model"]["metadata_source"] == "semantic_fixture_metadata_no_gated_checkpoint_download"
    assert graph["model"]["metadata_resolution"]["status"] == "fallback"
    assert "not cached" in graph["model"]["metadata_resolution"]["reason"]
    assert graph["projection_shapes"]["q_proj"] == {"rows": 2048, "cols": 2048}
    assert graph["next_hdl_contract_inputs"]["must_not_claim_full_llama_execution"] is True


def test_hdl_task_manifest_maps_semantic_graph_to_subagent_tasks(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    k_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "k_proj")
    packets = build_hdl_subagent_packets(manifest)
    q_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "projection_q_proj")
    assert q_task["rows"] == 64
    assert q_task["cols"] == 64
    assert q_task["memory_beats"] == 128
    assert q_task["gptq_weight_layout_preflight"]["status"] == "unsupported_or_missing_gptq_bits"
    assert q_task["target_checkpoint_layout_dependency"] == "blocked_by_real_gptq_weight_layout_preflight"
    assert k_task["rows"] == 16
    assert k_task["cols"] == 64
    assert manifest["hardware"]["max_lut"] == 230400
    assert manifest["hardware"]["max_ff"] == 460800
    assert manifest["hardware"]["max_dsp"] == 1728
    assert manifest["hardware"]["max_bram"] == 312
    assert manifest["hardware"]["max_uram"] == 96
    assert manifest["hardware"]["max_io"] == 464
    assert manifest["hardware"]["device_ff"] == 460800
    assert manifest["hardware"]["device_dsp"] == 1728
    assert manifest["hardware"]["device_io"] == 464
    assert manifest["optimization"]["quantization"] == "int4_gptq"
    assert manifest["source_replay"]["model_name"] == "local-llama-fixture"
    assert q_packet["source_replay"]["model_name"] == "local-llama-fixture"
    assert "--model local-llama-fixture" in " ".join(q_packet["required_commands"])
    assert manifest["gptq_checkpoint_metadata"]["status"] == "not_emitted"
    assert any(
        task["current_regression_kernel"] == "token_loop_decoder_block_fixture"
        for task in manifest["integration_tasks"]
    )
    assert any(
        task["current_regression_kernel"] == "projection_axi_read_command_adapter"
        for task in manifest["integration_tasks"]
    )
    assert any(
        task["current_regression_kernel"] == "projection_axi_read_data_channel_adapter"
        for task in manifest["integration_tasks"]
    )
    assert any(
        task["current_regression_kernel"] == "projection_axi_read_transaction_adapter"
        for task in manifest["integration_tasks"]
    )
    assert any(
        task["current_regression_kernel"] == "projection_axi_stream_integration"
        for task in manifest["integration_tasks"]
    )
    integration_kernels = [
        task["current_regression_kernel"] for task in manifest["integration_tasks"]
    ]
    assert "decoder_child_axi_attention_datapath" in integration_kernels
    assert "decoder_block_axi_attention_mlp_fixture" in integration_kernels
    assert integration_kernels[-1] == "ddr_axi_board_shell_fixture"
    assert integration_kernels.index("decoder_child_axi_attention_datapath") < integration_kernels.index(
        "layer_fsm_axi_attention_fixture"
    )
    assert integration_kernels.index("layer_fsm_axi_attention_fixture") < integration_kernels.index(
        "top_fsm_axi_attention_fixture"
    )
    assert integration_kernels.index("top_fsm_axi_attention_fixture") < integration_kernels.index(
        "token_loop_axi_attention_fixture"
    )
    assert integration_kernels.index("token_loop_axi_attention_fixture") < integration_kernels.index(
        "decoder_block_axi_attention_mlp_fixture"
    )
    assert integration_kernels.index("decoder_block_axi_attention_mlp_fixture") < integration_kernels.index(
        "layer_fsm_axi_decoder_block_fixture"
    )
    assert integration_kernels.index("layer_fsm_axi_decoder_block_fixture") < integration_kernels.index(
        "top_fsm_axi_decoder_block_fixture"
    )
    assert integration_kernels.index("top_fsm_axi_decoder_block_fixture") < integration_kernels.index(
        "token_loop_axi_decoder_block_fixture"
    )
    assert integration_kernels.index("token_loop_axi_decoder_block_fixture") < integration_kernels.index(
        "model_fsm_axi_decoder_block_fixture"
    )
    assert integration_kernels.index("model_fsm_axi_decoder_block_fixture") < integration_kernels.index(
        "ddr_axi_board_shell_fixture"
    )
    assert manifest["subagent_policy"]["verification_agents_are_read_only"] is True
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_gptq_checkpoint_metadata",
        "real_gptq_weight_layout_preflight",
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }
    assert "full_GPTQ_tensor_value_loading" in manifest["does_not_claim"]


def test_hdl_subagent_packets_include_layer_and_top_fsm_assignments(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    packets = build_hdl_subagent_packets(manifest)
    assert packets["task_count"] == 57
    assert packets["agent_topology"]["parallel_module_agents"]["gemm_kernel_agents"] == 7
    assert packets["agent_topology"]["parallel_module_agents"]["non_gemm_kernel_agents"] == 8
    assert packets["agent_topology"]["verification_agents"]["after_each_wave"] is True
    assert packets["agent_topology"]["failure_to_skill"]["skill_payload_fields"] == [
        "failing_command",
        "symptom",
        "root_cause_hypothesis",
        "prevention_rule",
        "minimal_regression_check",
    ]
    assert packets["blocked_target_tasks"][0]["task_id"] == "real_gptq_checkpoint_metadata"
    layer_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "layer_fsm_decoder_block_fixture")
    decoder_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "decoder_block_attention_mlp_fixture"
    )
    k_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "projection_k_proj")
    input_norm_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "non_gemm_input_layernorm")
    top_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "top_fsm_decoder_block_fixture")
    token_packet = next(packet for packet in packets["packets"] if packet["task_id"] == "token_loop_decoder_block_fixture")
    memory_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "projection_axi_read_command_adapter"
    )
    k_memory_packet = next(
        packet
        for packet in packets["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_command_adapter"
    )
    read_data_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "projection_axi_read_data_channel_adapter"
    )
    k_read_data_packet = next(
        packet
        for packet in packets["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_data_channel_adapter"
    )
    transaction_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "projection_axi_read_transaction_adapter"
    )
    k_transaction_packet = next(
        packet
        for packet in packets["packets"]
        if packet["task_id"] == "projection_k_proj_axi_read_transaction_adapter"
    )
    axi_stream_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "projection_axi_stream_integration"
    )
    k_axi_stream_packet = next(
        packet
        for packet in packets["packets"]
        if packet["task_id"] == "projection_k_proj_axi_stream_integration"
    )
    decoder_axi_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "decoder_child_axi_attention_datapath"
    )
    layer_axi_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "layer_fsm_axi_attention_fixture"
    )
    top_axi_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "top_fsm_axi_attention_fixture"
    )
    token_axi_packet = next(
        packet for packet in packets["packets"] if packet["task_id"] == "token_loop_axi_attention_fixture"
    )
    decoder_axi_block_packet = next(
        packet
        for packet in packets["packets"]
        if packet["task_id"] == "decoder_block_axi_attention_mlp_fixture"
    )
    assert k_packet["module_contract"]["artifact"] == "hdl_module_contract_bundle"
    assert k_packet["module_contract"]["task_id"] == "projection_k_proj"
    assert k_packet["semantic_op"] == "k_proj"
    assert k_packet["rows"] == 16
    assert k_packet["cols"] == 64
    assert k_packet["packed_int4_bytes"] == 512
    assert k_packet["memory_beats"] == 32
    assert k_packet["module_contract"]["clock_reset"] == {
        "clock": "aclk",
        "reset": "aresetn",
        "reset_style": "synchronous_active_low",
    }
    assert k_packet["module_contract"]["handshake_ports"] == {
        "start": "start_i",
        "done": "done_o",
    }
    assert k_packet["module_contract"]["parent_boundary"]["parent_must_not_write_hdl"] is True
    assert k_packet["module_contract"]["parent_boundary"]["subagent_owns_rtl_or_generator_changes"] is True
    assert k_packet["module_contract"]["parent_boundary"]["integration_boundary"] == "module_kernel"
    assert k_packet["module_contract"]["final_response_required_fields"] == [
        "changed_files",
        "commands_run",
        "simulation_evidence",
        "verilator_evidence",
        "vivado_timing_resource_evidence",
        "module_ooc_synthesis_evidence",
        "remaining_risks",
    ]
    assert "Machine-Readable Module Contract" in k_packet["prompt"]
    assert "hdl_module_contract_bundle" in k_packet["prompt"]
    assert layer_packet["module_contract"]["parent_boundary"]["integration_boundary"] == (
        "layer_fsm_calls_verified_child"
    )
    assert top_packet["module_contract"]["parent_boundary"]["integration_boundary"] == (
        "top_fsm_schedules_verified_layer_fsm"
    )
    assert layer_packet["agent_role"] == "layer_fsm_agent"
    assert top_packet["agent_role"] == "top_fsm_agent"
    assert memory_packet["agent_role"] == "memory_command_adapter_agent"
    assert memory_packet["contract"] == "docs/projection_axi_read_command_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(memory_packet["required_commands"])
    assert "--kernel projection_axi_read_command_adapter" in " ".join(memory_packet["required_commands"])
    assert "Target request beat count" in memory_packet["prompt"]
    assert "Target last-beat valid bytes" in memory_packet["prompt"]
    assert "Target request covers unaligned qweight range" in memory_packet["prompt"]
    assert "AXI read address" in memory_packet["prompt"]
    assert k_memory_packet["agent_role"] == "memory_command_adapter_agent"
    assert k_memory_packet["contract"] == "docs/projection_axi_read_command_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_memory_packet["required_commands"])
    assert "--kernel projection_axi_read_command_adapter" in " ".join(k_memory_packet["required_commands"])
    assert "Semantic op: `k_proj`" in k_memory_packet["prompt"]
    assert "k_proj checkpoint-aware qweight stream plan converted to AXI read address and length" in k_memory_packet["prompt"]
    assert read_data_packet["agent_role"] == "memory_read_data_adapter_agent"
    assert read_data_packet["contract"] == "docs/projection_axi_read_data_channel_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(read_data_packet["required_commands"])
    assert "--kernel projection_axi_read_data_channel_adapter" in " ".join(read_data_packet["required_commands"])
    assert "Target request beat count" in read_data_packet["prompt"]
    assert "AXI read-data" in read_data_packet["prompt"]
    assert k_read_data_packet["agent_role"] == "memory_read_data_adapter_agent"
    assert k_read_data_packet["contract"] == "docs/projection_axi_read_data_channel_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_read_data_packet["required_commands"])
    assert "--kernel projection_axi_read_data_channel_adapter" in " ".join(k_read_data_packet["required_commands"])
    assert "Semantic op: `k_proj`" in k_read_data_packet["prompt"]
    assert "k_proj bounded AXI read-data valid/ready backpressure evidence" in k_read_data_packet["prompt"]
    packets_by_id = {packet["task_id"]: packet for packet in packets["packets"]}
    expected_read_data_packets = {
        "q_proj": "projection_axi_read_data_channel_adapter",
        "k_proj": "projection_k_proj_axi_read_data_channel_adapter",
        "v_proj": "projection_v_proj_axi_read_data_channel_adapter",
        "o_proj": "projection_o_proj_axi_read_data_channel_adapter",
        "gate_proj": "projection_gate_proj_axi_read_data_channel_adapter",
        "up_proj": "projection_up_proj_axi_read_data_channel_adapter",
        "down_proj": "projection_down_proj_axi_read_data_channel_adapter",
    }
    for projection_name, task_id in expected_read_data_packets.items():
        packet = packets_by_id[task_id]
        assert packet["agent_role"] == "memory_read_data_adapter_agent"
        assert packet["current_regression_kernel"] == "projection_axi_read_data_channel_adapter"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in " ".join(packet["required_commands"])
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert f"{projection_name} bounded AXI read-data valid/ready backpressure evidence" in packet["prompt"]
    assert transaction_packet["agent_role"] == "memory_read_transaction_agent"
    assert transaction_packet["contract"] == "docs/projection_axi_read_transaction_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(transaction_packet["required_commands"])
    assert "--kernel projection_axi_read_transaction_adapter" in " ".join(transaction_packet["required_commands"])
    assert "Target request beat count" in transaction_packet["prompt"]
    assert "AR and R-channel" in transaction_packet["prompt"]
    assert k_transaction_packet["agent_role"] == "memory_read_transaction_agent"
    assert k_transaction_packet["contract"] == "docs/projection_axi_read_transaction_adapter_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_transaction_packet["required_commands"])
    assert "--kernel projection_axi_read_transaction_adapter" in " ".join(k_transaction_packet["required_commands"])
    assert "Semantic op: `k_proj`" in k_transaction_packet["prompt"]
    assert "k_proj bounded AXI read-address command followed by matching read-data beats" in k_transaction_packet["prompt"]
    expected_transaction_packets = {
        "q_proj": "projection_axi_read_transaction_adapter",
        "k_proj": "projection_k_proj_axi_read_transaction_adapter",
        "v_proj": "projection_v_proj_axi_read_transaction_adapter",
        "o_proj": "projection_o_proj_axi_read_transaction_adapter",
        "gate_proj": "projection_gate_proj_axi_read_transaction_adapter",
        "up_proj": "projection_up_proj_axi_read_transaction_adapter",
        "down_proj": "projection_down_proj_axi_read_transaction_adapter",
    }
    for projection_name, task_id in expected_transaction_packets.items():
        packet = packets_by_id[task_id]
        assert packet["agent_role"] == "memory_read_transaction_agent"
        assert packet["current_regression_kernel"] == "projection_axi_read_transaction_adapter"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in " ".join(packet["required_commands"])
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert f"{projection_name} bounded AXI read-address command followed by matching read-data beats" in packet[
            "prompt"
        ]
    assert axi_stream_packet["agent_role"] == "memory_projection_stream_agent"
    assert axi_stream_packet["contract"] == "docs/projection_axi_stream_integration_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=q_proj" in " ".join(axi_stream_packet["required_commands"])
    assert "--kernel projection_axi_stream_integration" in " ".join(axi_stream_packet["required_commands"])
    assert "projection-style payload consumer" in axi_stream_packet["prompt"]
    assert "DUT-observed RID/RRESP/RLAST" in axi_stream_packet["prompt"]
    assert k_axi_stream_packet["agent_role"] == "memory_projection_stream_agent"
    assert k_axi_stream_packet["contract"] == "docs/projection_axi_stream_integration_contract.md"
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_axi_stream_packet["required_commands"])
    assert "--kernel projection_axi_stream_integration" in " ".join(k_axi_stream_packet["required_commands"])
    assert "Semantic op: `k_proj`" in k_axi_stream_packet["prompt"]
    assert (
        "k_proj bounded AXI read transaction feeds projection-style payload consumer through valid/ready"
        in k_axi_stream_packet["prompt"]
    )
    expected_stream_packets = {
        "q_proj": "projection_axi_stream_integration",
        "k_proj": "projection_k_proj_axi_stream_integration",
        "v_proj": "projection_v_proj_axi_stream_integration",
        "o_proj": "projection_o_proj_axi_stream_integration",
        "gate_proj": "projection_gate_proj_axi_stream_integration",
        "up_proj": "projection_up_proj_axi_stream_integration",
        "down_proj": "projection_down_proj_axi_stream_integration",
    }
    for projection_name, task_id in expected_stream_packets.items():
        packet = packets_by_id[task_id]
        assert packet["agent_role"] == "memory_projection_stream_agent"
        assert packet["current_regression_kernel"] == "projection_axi_stream_integration"
        assert f"NL2HDL_SELECTED_PROJECTION={projection_name}" in " ".join(packet["required_commands"])
        assert f"Semantic op: `{projection_name}`" in packet["prompt"]
        assert (
            f"{projection_name} bounded AXI read transaction feeds projection-style payload consumer through valid/ready"
            in packet["prompt"]
        )
    assert decoder_axi_packet["agent_role"] == "decoder_axi_child_agent"
    assert decoder_axi_packet["contract"] == "docs/decoder_child_axi_attention_datapath_contract.md"
    assert "--kernel decoder_child_axi_attention_datapath" in " ".join(decoder_axi_packet["required_commands"])
    assert "projection_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_k_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_v_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert "projection_o_proj_axi_stream_integration" in decoder_axi_packet["prompt"]
    assert decoder_axi_packet["attention_projection_stream_tasks"] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
    ]
    assert decoder_axi_packet["target_weight_stream_plan"] is None
    assert decoder_axi_packet["gptq_payload_probe"] is None
    assert set(decoder_axi_packet["target_attention_projection_streams"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert "Aggregate attention layout dependency" in decoder_axi_packet["prompt"]
    assert "Aggregate attention payload dependency" in decoder_axi_packet["prompt"]
    assert "Target qweight tensor key" not in decoder_axi_packet["prompt"]
    assert "GPTQ qweight payload words32 LE" not in decoder_axi_packet["prompt"]
    assert "RID/RRESP/RLAST good-path metadata" in decoder_axi_packet["prompt"]
    assert layer_axi_packet["agent_role"] == "layer_axi_fsm_agent"
    assert layer_axi_packet["contract"] == "docs/layer_fsm_axi_attention_fixture_contract.md"
    assert "--kernel layer_fsm_axi_attention_fixture" in " ".join(layer_axi_packet["required_commands"])
    assert "decoder_child_axi_attention_datapath" in layer_axi_packet["prompt"]
    assert "AXI projection child metadata-good bits" in layer_axi_packet["prompt"]
    assert top_axi_packet["agent_role"] == "top_axi_fsm_agent"
    assert top_axi_packet["contract"] == "docs/top_fsm_axi_attention_fixture_contract.md"
    assert "--kernel top_fsm_axi_attention_fixture" in " ".join(top_axi_packet["required_commands"])
    assert "layer_fsm_axi_attention_fixture" in top_axi_packet["prompt"]
    assert "Top FSM compact status" in top_axi_packet["prompt"]
    assert token_axi_packet["agent_role"] == "token_loop_axi_agent"
    assert token_axi_packet["contract"] == "docs/token_loop_axi_attention_fixture_contract.md"
    assert "--kernel token_loop_axi_attention_fixture" in " ".join(token_axi_packet["required_commands"])
    assert decoder_axi_block_packet["agent_role"] == "decoder_axi_block_agent"
    assert decoder_axi_block_packet["contract"] == "docs/decoder_block_axi_attention_mlp_fixture_contract.md"
    assert "--kernel decoder_block_axi_attention_mlp_fixture" in " ".join(
        decoder_axi_block_packet["required_commands"]
    )
    assert "decoder_child_axi_attention_datapath" in decoder_axi_block_packet["prompt"]
    assert "residual_mlp_fixture" in decoder_axi_block_packet["prompt"]
    assert "top_fsm_axi_attention_fixture" in token_axi_packet["prompt"]
    assert "token-loop compact status" in token_axi_packet["prompt"]
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in " ".join(k_packet["required_commands"])
    assert "NL2HDL_SELECTED_PROJECTION=k_proj" in k_packet["prompt"]
    assert "blocked_by_real_gptq_weight_layout_preflight" in k_packet["prompt"]
    assert "Real checkpoint projection layout compatibility" in k_packet["prompt"]
    assert "NL2HDL_SELECTED_NONGEMM=input_layernorm" in " ".join(input_norm_packet["required_commands"])
    assert "NL2HDL_SELECTED_NONGEMM=input_layernorm" in input_norm_packet["prompt"]
    assert "Verification agents are read-only" in "\n".join(layer_packet["allowed_write_scope"])
    assert "--kernel layer_fsm_decoder_block_fixture" in " ".join(layer_packet["required_commands"])
    assert "--kernel top_fsm_decoder_block_fixture" in " ".join(top_packet["required_commands"])
    assert "Current Target Gate Blocks" in layer_packet["prompt"]
    assert "real_gptq_weight_layout_preflight" in layer_packet["prompt"]
    assert "bounded fixture RTL" in layer_packet["prompt"]
    assert "Current Target Gate Blocks" in top_packet["prompt"]
    assert "real_gptq_weight_layout_preflight" in top_packet["prompt"]
    assert "Current Target Gate Blocks" in decoder_packet["prompt"]
    assert "real_gptq_weight_layout_preflight" in decoder_packet["prompt"]
    assert "full_llama_model_execution" in decoder_packet["prompt"]
    assert "board_level_zcu104_signoff" in decoder_packet["prompt"]
    assert "Current Target Gate Blocks" in token_packet["prompt"]
    assert "real_gptq_weight_layout_preflight" in token_packet["prompt"]
    assert "full_llama_model_execution" in token_packet["prompt"]
    assert "board_level_zcu104_signoff" in token_packet["prompt"]
    all_manifest_tasks = {
        task["task_id"]: task
        for group in ("projection_tasks", "non_gemm_tasks", "integration_tasks")
        for task in manifest[group]
    }
    for packet in packets["packets"]:
        task = all_manifest_tasks[packet["task_id"]]
        prompt = packet["prompt"]
        assert packet["task_id"] in prompt
        assert packet["agent_role"] in prompt
        assert task["contract"] in prompt
        assert "aclk" in prompt
        assert "aresetn" in prompt
        assert "start_i" in prompt
        assert "done_o" in prompt
        assert "Assigned generator/source scope: nl2hdl/llm_kernels.py changes only for kernel" in prompt
        assert "Assigned test scope: add or update only task-specific assertions in tests/test_llm_kernels.py." in prompt
        assert "Do not edit parent orchestration files" in prompt
        assert "Do not weaken existing tests, contracts, timing gates, or forbidden-claim language." in prompt
        assert "setup, hold, and pulse-width timing must all have non-negative slack" in prompt
        assert "skill_update_candidate" in prompt
        assert "Failure-To-SKILL Candidate" in prompt
        assert "failing_command" in prompt
        assert "root_cause_hypothesis" in prompt
        assert "minimal_regression_check" in prompt
        assert "Final response must list changed files" in prompt
        assert "Full LLaMA execution unless this exact task proves it." in prompt
        assert "Board-level ZCU104 signoff unless board I/O, DDR/AXI, and PS/PL constraints are included." in prompt
        for command in packet["required_commands"]:
            assert command in prompt
        for evidence in task.get("required_evidence", []):
            assert evidence in prompt


def test_hdl_subagent_dispatch_plan_orders_parallel_and_integration_waves(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    packets = build_hdl_subagent_packets(manifest)
    dispatch_plan = build_hdl_subagent_dispatch_plan(packets)
    waves = {wave["wave_id"]: wave for wave in dispatch_plan["waves"]}
    assert dispatch_plan["dispatch_policy"]["do_not_advance_to_dependent_wave_until_current_wave_verification_passes"] is True
    assert dispatch_plan["dispatch_policy"]["one_subagent_per_hdl_packet"] is True
    assert dispatch_plan["dispatch_policy"]["module_agents_run_own_simulation_and_synthesis"] is True
    assert dispatch_plan["dispatch_policy"]["layer_fsm_and_top_fsm_are_separate_implementation_agents"] is True
    assert dispatch_plan["agent_topology"]["implementation_agent_granularity"] == "one_subagent_per_hdl_packet"
    assert dispatch_plan["does_not_claim"] == [
        "automatic sub-agent spawning inside package runtime",
        "completed target RTL for every packet",
        "full LLaMA execution",
        "board-level ZCU104 signoff",
    ]
    assert waves["wave_1_projection_kernels"]["parallel_dispatch_allowed"] is True
    assert waves["wave_1_projection_kernels"]["verification_agent"]["agent"] == "Codex"
    assert waves["wave_1_projection_kernels"]["verification_agent"]["prompt_file"] == (
        "verification_prompts/wave_1_projection_kernels__verification.md"
    )
    assert waves["wave_1_projection_kernels"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_1_projection_kernels"]["blocked_target_dependencies"] == [
        "gptq_payload_probe",
        "real_gptq_checkpoint_metadata",
        "real_gptq_weight_layout_preflight"
    ]
    assert waves["wave_1_projection_kernels"]["direct_blocked_target_dependencies"] == [
        "gptq_payload_probe",
        "real_gptq_weight_layout_preflight"
    ]
    assert waves["wave_1_projection_kernels"]["global_blocked_target_dependencies"] == [
        "real_gptq_checkpoint_metadata"
    ]
    assert waves["wave_1_projection_kernels"]["inherited_blocked_target_dependencies"] == []
    assert len(waves["wave_1_projection_kernels"]["implementation_tasks"]) == 7
    assert waves["wave_1_projection_kernels"]["implementation_tasks"][0][
        "target_checkpoint_layout_dependency"
    ] == "blocked_by_real_gptq_weight_layout_preflight"
    assert len(waves["wave_1_non_gemm_kernels"]["implementation_tasks"]) == 8
    assert waves["wave_1_non_gemm_kernels"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_1_non_gemm_kernels"]["global_blocked_target_dependencies"] == [
        "real_gptq_checkpoint_metadata"
    ]
    assert waves["wave_2_decoder_block"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_2_decoder_block"]["direct_blocked_target_dependencies"] == []
    assert waves["wave_2_decoder_block"]["inherited_blocked_target_dependencies"] == [
        "gptq_payload_probe",
        "real_gptq_weight_layout_preflight"
    ]
    assert waves["wave_2_decoder_block"]["global_blocked_target_dependencies"] == [
        "real_gptq_checkpoint_metadata"
    ]
    assert waves["wave_3_layer_fsm"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_4_top_fsm"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_5_token_loop"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_6_projection_axi_read_command_adapter"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_7_projection_axi_read_data_channel_adapter"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_8_projection_axi_read_transaction_adapter"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_9_projection_axi_stream_integration"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_10_decoder_child_axi_attention_datapath"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_11_layer_fsm_axi_attention_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_12_top_fsm_axi_attention_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_13_token_loop_axi_attention_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_14_decoder_block_axi_attention_mlp_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_15_layer_fsm_axi_decoder_block_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_16_top_fsm_axi_decoder_block_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_17_token_loop_axi_decoder_block_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_18_model_fsm_axi_decoder_block_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_19_ddr_axi_board_shell_fixture"]["target_scope"] == "bounded_fixture_only"
    assert waves["wave_2_decoder_block"]["depends_on_waves"] == [
        "wave_1_projection_kernels",
        "wave_1_non_gemm_kernels",
    ]
    assert waves["wave_3_layer_fsm"]["depends_on_waves"] == ["wave_2_decoder_block"]
    assert waves["wave_4_top_fsm"]["depends_on_waves"] == ["wave_3_layer_fsm"]
    assert waves["wave_5_token_loop"]["depends_on_waves"] == ["wave_4_top_fsm"]
    assert waves["wave_6_projection_axi_read_command_adapter"]["depends_on_waves"] == ["wave_5_token_loop"]
    assert waves["wave_6_projection_axi_read_command_adapter"]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_read_command_adapter"
    )
    assert [task["task_id"] for task in waves["wave_6_projection_axi_read_command_adapter"]["implementation_tasks"]] == [
        "projection_axi_read_command_adapter",
        "projection_k_proj_axi_read_command_adapter",
        "projection_v_proj_axi_read_command_adapter",
        "projection_o_proj_axi_read_command_adapter",
        "projection_gate_proj_axi_read_command_adapter",
        "projection_up_proj_axi_read_command_adapter",
        "projection_down_proj_axi_read_command_adapter",
    ]
    assert waves["wave_7_projection_axi_read_data_channel_adapter"]["depends_on_waves"] == [
        "wave_6_projection_axi_read_command_adapter"
    ]
    assert waves["wave_7_projection_axi_read_data_channel_adapter"]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_read_data_channel_adapter"
    )
    assert [task["task_id"] for task in waves["wave_7_projection_axi_read_data_channel_adapter"]["implementation_tasks"]] == [
        "projection_axi_read_data_channel_adapter",
        "projection_k_proj_axi_read_data_channel_adapter",
        "projection_v_proj_axi_read_data_channel_adapter",
        "projection_o_proj_axi_read_data_channel_adapter",
        "projection_gate_proj_axi_read_data_channel_adapter",
        "projection_up_proj_axi_read_data_channel_adapter",
        "projection_down_proj_axi_read_data_channel_adapter",
    ]
    assert waves["wave_8_projection_axi_read_transaction_adapter"]["depends_on_waves"] == [
        "wave_7_projection_axi_read_data_channel_adapter"
    ]
    assert waves["wave_8_projection_axi_read_transaction_adapter"]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_read_transaction_adapter"
    )
    assert [task["task_id"] for task in waves["wave_8_projection_axi_read_transaction_adapter"]["implementation_tasks"]] == [
        "projection_axi_read_transaction_adapter",
        "projection_k_proj_axi_read_transaction_adapter",
        "projection_v_proj_axi_read_transaction_adapter",
        "projection_o_proj_axi_read_transaction_adapter",
        "projection_gate_proj_axi_read_transaction_adapter",
        "projection_up_proj_axi_read_transaction_adapter",
        "projection_down_proj_axi_read_transaction_adapter",
    ]
    assert waves["wave_9_projection_axi_stream_integration"]["depends_on_waves"] == [
        "wave_8_projection_axi_read_transaction_adapter"
    ]
    assert waves["wave_9_projection_axi_stream_integration"]["implementation_tasks"][0]["task_id"] == (
        "projection_axi_stream_integration"
    )
    assert [task["task_id"] for task in waves["wave_9_projection_axi_stream_integration"]["implementation_tasks"]] == [
        "projection_axi_stream_integration",
        "projection_k_proj_axi_stream_integration",
        "projection_v_proj_axi_stream_integration",
        "projection_o_proj_axi_stream_integration",
        "projection_gate_proj_axi_stream_integration",
        "projection_up_proj_axi_stream_integration",
        "projection_down_proj_axi_stream_integration",
    ]
    assert waves["wave_10_decoder_child_axi_attention_datapath"]["depends_on_waves"] == [
        "wave_9_projection_axi_stream_integration"
    ]
    assert waves["wave_10_decoder_child_axi_attention_datapath"]["implementation_tasks"][0]["task_id"] == (
        "decoder_child_axi_attention_datapath"
    )
    assert waves["wave_11_layer_fsm_axi_attention_fixture"]["depends_on_waves"] == [
        "wave_10_decoder_child_axi_attention_datapath"
    ]
    assert waves["wave_11_layer_fsm_axi_attention_fixture"]["implementation_tasks"][0]["task_id"] == (
        "layer_fsm_axi_attention_fixture"
    )
    assert waves["wave_12_top_fsm_axi_attention_fixture"]["depends_on_waves"] == [
        "wave_11_layer_fsm_axi_attention_fixture"
    ]
    assert waves["wave_12_top_fsm_axi_attention_fixture"]["implementation_tasks"][0]["task_id"] == (
        "top_fsm_axi_attention_fixture"
    )
    assert waves["wave_13_token_loop_axi_attention_fixture"]["depends_on_waves"] == [
        "wave_12_top_fsm_axi_attention_fixture"
    ]
    assert waves["wave_13_token_loop_axi_attention_fixture"]["implementation_tasks"][0]["task_id"] == (
        "token_loop_axi_attention_fixture"
    )
    assert waves["wave_14_decoder_block_axi_attention_mlp_fixture"]["depends_on_waves"] == [
        "wave_13_token_loop_axi_attention_fixture"
    ]
    assert waves["wave_14_decoder_block_axi_attention_mlp_fixture"]["implementation_tasks"][0]["task_id"] == (
        "decoder_block_axi_attention_mlp_fixture"
    )
    assert waves["wave_15_layer_fsm_axi_decoder_block_fixture"]["depends_on_waves"] == [
        "wave_14_decoder_block_axi_attention_mlp_fixture"
    ]
    assert waves["wave_15_layer_fsm_axi_decoder_block_fixture"]["implementation_tasks"][0]["task_id"] == (
        "layer_fsm_axi_decoder_block_fixture"
    )
    assert waves["wave_16_top_fsm_axi_decoder_block_fixture"]["depends_on_waves"] == [
        "wave_15_layer_fsm_axi_decoder_block_fixture"
    ]
    assert waves["wave_16_top_fsm_axi_decoder_block_fixture"]["implementation_tasks"][0]["task_id"] == (
        "top_fsm_axi_decoder_block_fixture"
    )
    assert waves["wave_17_token_loop_axi_decoder_block_fixture"]["depends_on_waves"] == [
        "wave_16_top_fsm_axi_decoder_block_fixture"
    ]
    assert waves["wave_17_token_loop_axi_decoder_block_fixture"]["implementation_tasks"][0]["task_id"] == (
        "token_loop_axi_decoder_block_fixture"
    )
    assert waves["wave_18_model_fsm_axi_decoder_block_fixture"]["depends_on_waves"] == [
        "wave_17_token_loop_axi_decoder_block_fixture"
    ]
    assert waves["wave_18_model_fsm_axi_decoder_block_fixture"]["implementation_tasks"][0]["task_id"] == (
        "model_fsm_axi_decoder_block_fixture"
    )
    assert waves["wave_19_ddr_axi_board_shell_fixture"]["depends_on_waves"] == [
        "wave_18_model_fsm_axi_decoder_block_fixture"
    ]
    assert waves["wave_19_ddr_axi_board_shell_fixture"]["implementation_tasks"][0]["task_id"] == (
        "ddr_axi_board_shell_fixture"
    )
    for wave in dispatch_plan["waves"]:
        assert wave["verification_agent"]["required"] is True
        if any(task["task_group"] == "integration_tasks" for task in wave["implementation_tasks"]):
            assert wave["verification_agent"]["mode"] == "integration_verification_with_synthesis"
            assert wave["verification_agent"]["runs_integration_synthesis"] is True
            assert wave["verification_agent"]["expected_integration_synthesis_report"].endswith(
                "/integration_synthesis_report.json"
            )
        else:
            assert wave["verification_agent"]["mode"] == "read_only"
            assert wave["verification_agent"]["runs_integration_synthesis"] is False
        assert wave["verification_agent"]["agent"] == "Codex"
        assert wave["verification_agent"]["prompt_file"].startswith("verification_prompts/")
        for task in wave["implementation_tasks"]:
            assert task["prompt_file"].startswith("subagent_prompts/")


def _minimal_passed_kernel_report() -> dict[str, object]:
    return {
        "status": "passed",
        "coverage_level": "synthetic_fixture",
        "implementation_stage": "post-route",
        "simulation": {"passed": True},
        "verilator": {"passed": True},
        "contract_gate": {"verilator_enforced": True, "synthesis_enforced": True},
        "synthesis": {
            "passed": True,
            "timing": {
                "constraints_met": True,
                "setup_worst_slack_ns": 0.1,
                "hold_worst_slack_ns": 0.1,
                "pulse_width_worst_slack_ns": 0.1,
            },
        },
    }


def _minimal_subagent_result(task_id: str | None = None) -> dict[str, object]:
    return {
        "task_id": task_id,
        "changed_files": ["nl2hdl/llm_kernels.py", "tests/test_llm_kernels.py"],
        "commands_run": ["python3 -m pytest -q tests/test_llm_kernels.py"],
        "simulation_evidence": {"passed": True, "source": "kernel_report.json"},
        "verilator_evidence": {"passed": True, "source": "kernel_report.json"},
        "vivado_timing_resource_evidence": {"passed": True, "source": "kernel_report.json"},
        "module_ooc_synthesis_evidence": {"passed": True, "source": "module_ooc_synthesis_report.json"},
        "remaining_risks": ["bounded fixture only"],
    }


def _minimal_module_ooc_synthesis_report() -> dict[str, object]:
    return {
        "artifact": "module_ooc_synthesis_report",
        "status": "passed",
        "vivado": {
            "part": "xczu7ev-ffvc1156-2-e",
            "target_clock_mhz": 200,
        },
        "hardware_spec": {
            "fpga_part": "xczu7ev-ffvc1156-2-e",
            "target_clock_mhz": 200,
            "max_lut": 230400,
            "max_dsp": 1728,
            "max_bram": 312,
            "max_ff": 460800,
            "max_uram": 96,
            "max_io": 464,
            "memory_data_width": 128,
            "device_logic_cells": 504000,
            "device_lut": 230400,
            "device_ff": 460800,
            "device_dsp": 1728,
            "device_bram_36k": 312,
            "device_uram": 96,
            "device_io": 464,
            "device_distributed_ram_mb": 6.2,
            "device_bram_mb": 11.0,
            "device_uram_mb": 27.0,
            "device_ps_gtr": 4,
            "device_gth": 20,
            "resource_reference": "AMD ZCU104 UG1267 v1.1 and XCZU7EV board resource tables",
        },
        "timing": {
            "constraints_met": True,
            "setup_worst_slack_ns": 0.1,
            "hold_worst_slack_ns": 0.1,
            "pulse_width_worst_slack_ns": 0.1,
        },
        "utilization": {
            "lut": 100,
            "dsp": 1,
            "bram": 1,
            "uram": 0,
            "ff": 200,
            "io": 32,
        },
        "selected_tuning_knobs": {
            "pe_lanes": 1,
            "tile_m": 1,
            "tile_n": 1,
            "buffer_depth": 16,
            "memory_word_width": 128,
            "pipeline_stages": 2,
        },
        "resource_assessment": "near_budget",
        "throughput_target_met": True,
    }


def _write_minimal_passed_kernel_evidence(evidence_dir: Path, task_id: str | None = None) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(_minimal_passed_kernel_report()),
        encoding="utf-8",
    )
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        json.dumps(_minimal_module_ooc_synthesis_report()),
        encoding="utf-8",
    )
    (evidence_dir / "subagent_result.json").write_text(
        json.dumps(_minimal_subagent_result(task_id)),
        encoding="utf-8",
    )


def _minimal_full_execution_dispatch_plan() -> dict:
    return {
        "model": {
            "name": "meta-llama/Llama-3.2-1B",
            "decoder_layers": 16,
        },
        "hardware": {
            "fpga_part": "xczu7ev-ffvc1156-2-e",
            "max_lut": 230000,
            "max_dsp": 1728,
            "max_bram": 312,
        },
        "optimization": {
            "quantization": "int4_gptq",
            "design_style": "llm_decoder_streaming",
        },
        "blocked_target_tasks": [
            {"task_id": "full_llama_model_execution", "reason": "full execution evidence not present"},
            {"task_id": "board_level_zcu104_signoff", "reason": "board signoff evidence not present"},
        ],
        "global_blocked_target_dependencies": [],
        "waves": [
            {
                "wave_id": "wave_17_token_loop_axi_decoder_block_fixture",
                "target_scope": "target_preflight_satisfied_or_not_applicable",
            },
            {
                "wave_id": "wave_18_model_fsm_axi_decoder_block_fixture",
                "target_scope": "target_preflight_satisfied_or_not_applicable",
            },
            {
                "wave_id": "wave_19_ddr_axi_board_shell_fixture",
                "target_scope": "target_preflight_satisfied_or_not_applicable",
            },
        ],
    }


def _minimal_passed_full_execution_wave_status() -> dict:
    return {
        "artifact": "hdl_subagent_wave_status",
        "wave_count": 3,
        "waves": [
            {"wave_id": "wave_17_token_loop_axi_decoder_block_fixture", "status": "passed"},
            {"wave_id": "wave_18_model_fsm_axi_decoder_block_fixture", "status": "passed"},
            {"wave_id": "wave_19_ddr_axi_board_shell_fixture", "status": "passed"},
        ],
    }


def test_full_llama_execution_readiness_requires_explicit_full_execution_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()

    report = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)

    assert report["status"] == "blocked_by_missing_or_incomplete_full_execution_evidence"
    assert report["target_preflight"]["status"] == "passed"
    assert report["subagent_waves"]["passed_wave_count"] == 3
    assert report["safe_to_clear_full_llama_model_execution_blocker"] is False
    assert "full_llama_execution_evidence.json not found" in report["evidence_failures"]
    assert "write full_llama_execution_evidence.json" in report["next_action"]


def test_full_llama_execution_readiness_passes_with_strict_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    (tmp_path / "full_llama_execution_evidence.json").write_text(
        json.dumps(
            {
                "artifact": "full_llama_execution_evidence",
                "status": "passed",
                "model": "meta-llama/Llama-3.2-1B",
                "target_preflight_status": "passed",
                "full_model_layers_executed": True,
                "executed_layer_count": 16,
                "token_loop_evidence": {"passed": True, "source_wave_id": "wave_17_token_loop_axi_decoder_block_fixture"},
                "model_fsm_evidence": {"passed": True, "source_wave_id": "wave_18_model_fsm_axi_decoder_block_fixture"},
                "checkpoint_payload_evidence": {"passed": True, "source": "gptq_payload_probe.json"},
                "python_reference_comparison": {"passed": True, "tolerance_lsb": 2},
                "board_level_signoff": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)

    assert report["status"] == "passed"
    assert report["safe_to_clear_full_llama_model_execution_blocker"] is True
    assert report["evidence_failures"] == []
    assert report["next_action"] == "full execution readiness passed; board-level signoff remains a separate gate"
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]


def test_full_llama_execution_readiness_rejects_board_signoff_claim_in_execution_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    (tmp_path / "full_llama_execution_evidence.json").write_text(
        json.dumps(
            {
                "artifact": "full_llama_execution_evidence",
                "status": "passed",
                "model": "meta-llama/Llama-3.2-1B",
                "target_preflight_status": "passed",
                "full_model_layers_executed": True,
                "executed_layer_count": 16,
                "token_loop_evidence": {"passed": True},
                "model_fsm_evidence": {"passed": True},
                "checkpoint_payload_evidence": {"passed": True},
                "python_reference_comparison": {"passed": True},
                "board_level_signoff": True,
            }
        ),
        encoding="utf-8",
    )

    report = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)

    assert report["status"] == "blocked_by_missing_or_incomplete_full_execution_evidence"
    assert "full execution evidence must not claim board_level_signoff" in report["evidence_failures"]


def test_full_llama_execution_evidence_template_matches_readiness_gate_fields(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()

    template = build_full_llama_execution_evidence_template(dispatch_plan, wave_status, tmp_path)

    assert template["artifact"] == "full_llama_execution_evidence_template"
    assert template["write_to"] == str(tmp_path / "full_llama_execution_evidence.json")
    assert template["template"]["artifact"] == "full_llama_execution_evidence"
    assert template["template"]["model"] == "meta-llama/Llama-3.2-1B"
    assert template["template"]["executed_layer_count"] == 16
    assert template["template"]["board_level_signoff"] is False
    assert set(template["required_fields"]) == {
        "artifact",
        "status",
        "model",
        "target_preflight_status",
        "full_model_layers_executed",
        "executed_layer_count",
        "token_loop_evidence",
        "model_fsm_evidence",
        "checkpoint_payload_evidence",
        "python_reference_comparison",
        "board_level_signoff",
    }
    assert template["current_wave_context"]["passed_wave_ids"] == [
        "wave_17_token_loop_axi_decoder_block_fixture",
        "wave_18_model_fsm_axi_decoder_block_fixture",
        "wave_19_ddr_axi_board_shell_fixture",
    ]
    assert template["current_wave_context"]["non_passed_wave_ids"] == []
    assert "board_level_ZCU104_signoff" in template["does_not_claim"]


def test_model_level_execution_harness_agent_task_is_ready_before_full_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    readiness = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)

    task = build_model_level_execution_harness_agent_task(
        dispatch_plan,
        wave_status,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["artifact"] == "model_level_execution_harness_agent_task"
    assert task["spawn_kind"] == "target_evidence_implementation_agent"
    assert task["mode"] == "read_write_non_hdl_harness"
    assert task["ready_to_spawn"] is True
    assert task["spawn_precondition_failures"] == []
    assert task["expected_evidence_file"] == str(
        tmp_path / "full_llama_execution_gate" / "model_level_execution_harness_report.json"
    )
    assert task["expected_subagent_result"] == "build/full_llama_execution_gate/model_level_harness_subagent_result.json"
    assert task["parent_must_not_write_hdl"] is True
    assert task["failure_to_skill_required"] is True
    assert "python3 -m pytest -q tests/test_llm_kernels.py" in task["required_commands"]
    assert "--dispatch-plan build/dispatch.json" in task["required_commands"][1]
    assert f"--evidence-root {tmp_path}" in task["required_commands"][1]
    assert "Target Evidence Sub-Agent Task: model_level_execution_harness" in task["prompt"]
    assert "Do not only inspect the existing bounded fixture reports" in task["prompt"]
    assert "Preferred source scope for harness/report generation" in task["prompt"]
    assert "executed_layer_count >= dispatch_plan.model.decoder_layers" in task["prompt"]
    assert "python_reference_comparison" in task["prompt"]
    assert "Do not write `build/full_llama_execution_evidence.json`" in task["prompt"]
    assert "Do not edit parent orchestration files" in task["prompt"]
    assert "skill_update_candidate" in task["prompt"]
    assert "full_LLaMA_execution" in task["does_not_claim"]


def test_model_level_execution_harness_report_runs_16_layers_against_python_reference():
    dispatch_plan = _minimal_full_execution_dispatch_plan()

    report = build_model_level_execution_harness_report(dispatch_plan)

    assert report["artifact"] == "model_level_execution_harness_report"
    assert report["status"] == "passed"
    assert report["model"] == "meta-llama/Llama-3.2-1B"
    assert report["required_decoder_layers"] == 16
    assert report["executed_layer_count"] == 16
    assert report["full_model_layers_executed"] is True
    assert report["target_16_layer_iteration"] is True
    assert report["python_reference_comparison"]["passed"] is True
    assert report["python_reference_comparison"]["max_abs_error"] == 0
    assert report["model_level_harness"]["bounded_two_layer_fixture_reused"] is False
    assert report["model_level_harness"]["layer_indices"] == list(range(16))
    assert "fixture_layer_count" not in report
    assert report["board_level_signoff"] is False
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]


def test_write_model_level_execution_harness_report_only_writes_passing_report(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()

    report = write_model_level_execution_harness_report(dispatch_plan, tmp_path)

    path = tmp_path / "model_level_execution_harness_report.json"
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == report
    assert written["status"] == "passed"
    assert written["executed_layer_count"] >= dispatch_plan["model"]["decoder_layers"]


def test_model_level_execution_harness_report_rejects_missing_decoder_layers():
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    dispatch_plan["model"].pop("decoder_layers")

    with pytest.raises(ValueError, match="decoder_layers"):
        build_model_level_execution_harness_report(dispatch_plan)


def test_run_full_llama_execution_evidence_agent_blocks_without_harness(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()

    report = run_full_llama_execution_evidence_agent(dispatch_plan, wave_status, tmp_path)

    assert report["status"] == "blocked_by_missing_or_incomplete_model_execution_evidence"
    assert report["evidence_written"] is False
    assert "model_level_execution_harness_report.json not found" in report["failures"]
    assert not (tmp_path / "full_llama_execution_evidence.json").exists()
    subagent_result = json.loads((tmp_path / "full_llama_execution_gate" / "subagent_result.json").read_text())
    assert subagent_result["status"] == "blocked_by_missing_or_incomplete_model_execution_evidence"
    assert "skill_update_candidate" in subagent_result


def test_run_full_llama_execution_evidence_agent_writes_evidence_after_harness(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    write_model_level_execution_harness_report(dispatch_plan, tmp_path / "full_llama_execution_gate")

    report = run_full_llama_execution_evidence_agent(dispatch_plan, wave_status, tmp_path)

    assert report["status"] == "passed"
    assert report["evidence_written"] is True
    evidence = json.loads((tmp_path / "full_llama_execution_evidence.json").read_text())
    assert evidence["artifact"] == "full_llama_execution_evidence"
    assert evidence["status"] == "passed"
    assert evidence["full_model_layers_executed"] is True
    assert evidence["executed_layer_count"] == dispatch_plan["model"]["decoder_layers"]
    assert evidence["python_reference_comparison"]["passed"] is True
    assert evidence["board_level_signoff"] is False
    assert report["readiness"]["status"] == "passed"


def test_full_llama_execution_evidence_agent_task_waits_for_model_harness(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    readiness = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)
    template = build_full_llama_execution_evidence_template(dispatch_plan, wave_status, tmp_path)

    task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["ready_to_spawn"] is False
    assert (
        "model_level_execution_harness_report must prove full decoder-layer execution and Python reference comparison"
        in task["spawn_precondition_failures"]
    )
    assert "model_level_execution_harness_report.json not found" in task["prompt"]


def test_full_llama_execution_evidence_agent_task_is_ready_after_model_harness(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    readiness = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)
    template = build_full_llama_execution_evidence_template(dispatch_plan, wave_status, tmp_path)
    harness_dir = tmp_path / "full_llama_execution_gate"
    harness_dir.mkdir()
    (harness_dir / "model_level_execution_harness_report.json").write_text(
        json.dumps(
            {
                "artifact": "model_level_execution_harness_report",
                "status": "passed",
                "executed_layer_count": 16,
                "full_model_layers_executed": True,
                "target_16_layer_iteration": True,
                "python_reference_comparison": {"passed": True, "tolerance_lsb": 2},
                "board_level_signoff": False,
            }
        ),
        encoding="utf-8",
    )

    task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["artifact"] == "full_llama_execution_evidence_agent_task"
    assert task["ready_to_spawn"] is True
    assert task["spawn_precondition_failures"] == []
    assert "board_level_ZCU104_signoff" in task["does_not_claim"]


def test_target_evidence_execution_manifest_and_ledger_track_full_execution_agent(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    wave_status = _minimal_passed_full_execution_wave_status()
    readiness = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)
    template = build_full_llama_execution_evidence_template(dispatch_plan, wave_status, tmp_path)
    harness_dir = tmp_path / "full_llama_execution_gate"
    harness_dir.mkdir()
    (harness_dir / "model_level_execution_harness_report.json").write_text(
        json.dumps(
            {
                "artifact": "model_level_execution_harness_report",
                "status": "passed",
                "executed_layer_count": 16,
                "full_model_layers_executed": True,
                "python_reference_comparison": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    task["expected_subagent_result"] = str(tmp_path / "full_llama_execution_gate" / "subagent_result.json")

    execution = build_target_evidence_execution_manifest(task)
    markdown = build_codex_spawn_instructions(execution)

    assert execution["artifact"] == "target_evidence_execution_manifest"
    assert execution["spawn_entry_count"] == 1
    assert execution["target_evidence_spawn_count"] == 1
    assert execution["spawn_batches"][0]["spawn_kind"] == "target_evidence_agent"
    entry = execution["spawn_entries"][0]
    assert entry["spawn_key"] == "target_evidence::full_llama_execution_evidence"
    assert entry["expected_evidence_file"] == str(tmp_path / "full_llama_execution_evidence.json")
    assert entry["expected_subagent_result"] == str(
        tmp_path / "full_llama_execution_gate" / "subagent_result.json"
    )
    assert entry["parent_must_not_write_hdl"] is True
    assert entry["final_response_required_fields"] == [
        "changed_files",
        "commands_run",
        "evidence_paths",
        "remaining_risks",
    ]
    assert "Evidence file:" in markdown
    assert "Sub-agent result:" in markdown
    initial_ledger = build_hdl_subagent_spawn_ledger(
        execution,
        agent_records={"target_evidence::full_llama_execution_evidence": "agent-full-exec"},
    )
    record = initial_ledger["records"][0]
    assert record["spawn_status"] == "spawned_waiting_for_evidence"
    assert record["agent_id"] == "agent-full-exec"

    (tmp_path / "full_llama_execution_evidence.json").write_text(
        json.dumps(
            {
                "artifact": "full_llama_execution_evidence",
                "status": "passed",
                "model": "meta-llama/Llama-3.2-1B",
                "target_preflight_status": "passed",
                "full_model_layers_executed": True,
                "executed_layer_count": 16,
                "token_loop_evidence": {"passed": True},
                "model_fsm_evidence": {"passed": True},
                "checkpoint_payload_evidence": {"passed": True},
                "python_reference_comparison": {"passed": True},
                "board_level_signoff": False,
            }
        ),
        encoding="utf-8",
    )
    incomplete_ledger = build_hdl_subagent_spawn_ledger(execution, existing_ledger=initial_ledger)
    assert incomplete_ledger["records"][0]["spawn_status"] == "evidence_incomplete_subagent_result"

    result_dir = tmp_path / "full_llama_execution_gate"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "subagent_result.json").write_text(
        json.dumps(
            {
                "task_id": "full_llama_execution_evidence",
                "changed_files": [str(tmp_path / "full_llama_execution_evidence.json")],
                "commands_run": ["python3 -m pytest -q tests/test_llm_kernels.py"],
                "evidence_paths": [str(tmp_path / "full_llama_execution_evidence.json")],
                "remaining_risks": ["board signoff remains separate"],
            }
        ),
        encoding="utf-8",
    )
    passed_ledger = build_hdl_subagent_spawn_ledger(execution, existing_ledger=initial_ledger)

    assert passed_ledger["records"][0]["spawn_status"] == "evidence_passed"
    assert passed_ledger["records"][0]["evidence_reason"] == str(tmp_path / "full_llama_execution_evidence.json")
    assert passed_ledger["status_counts"]["evidence_passed"] == 1


def test_target_evidence_ledger_surfaces_failed_subagent_without_evidence_file(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)
    task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    task["expected_subagent_result"] = str(tmp_path / "board_zcu104_signoff_gate" / "subagent_result.json")
    execution = build_target_evidence_execution_manifest(task)
    initial_ledger = build_hdl_subagent_spawn_ledger(
        execution,
        agent_records={"target_evidence::board_zcu104_signoff_evidence": "agent-board-signoff"},
    )
    result_dir = tmp_path / "board_zcu104_signoff_gate"
    result_dir.mkdir()
    (result_dir / "subagent_result.json").write_text(
        json.dumps(
            {
                "task_id": "board_zcu104_signoff_evidence",
                "status": "blocked_pending_vivado_board_integration_run",
                "changed_files": ["build/board_zcu104_signoff_gate/subagent_result.json"],
                "commands_run": ["python3 -m nl2hdl subagents status ..."],
                "evidence_paths": {
                    "readiness_report": "build/board_zcu104_signoff_gate/readiness.json"
                },
                "remaining_risks": ["no routed Vivado report bundle"],
                "skill_update_candidate": {
                    "failing_command": "python3 -m nl2hdl subagents status ...",
                    "symptom": "board signoff remains blocked without board evidence",
                    "root_cause_hypothesis": "constraint scaffold is not routed board signoff",
                    "prevention_rule": "spawn a scoped board-wrapper Vivado implementation agent before evidence-only retry",
                    "minimal_regression_check": "ledger reports failed target-evidence subagent without evidence file",
                },
            }
        ),
        encoding="utf-8",
    )

    ledger = build_hdl_subagent_spawn_ledger(execution, existing_ledger=initial_ledger)

    assert ledger["records"][0]["spawn_status"] == "evidence_failed_waiting_for_skill_update"
    assert ledger["records"][0]["evidence_reason"] == str(result_dir / "subagent_result.json")
    assert ledger["status_counts"]["evidence_failed_waiting_for_skill_update"] == 1


def test_full_llama_execution_evidence_agent_task_blocks_when_preflight_is_not_ready(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    dispatch_plan["blocked_target_tasks"].append(
        {"task_id": "real_mlir_model_analysis", "reason": "MLIR analysis missing"}
    )
    wave_status = _minimal_passed_full_execution_wave_status()
    readiness = build_full_llama_execution_readiness_report(dispatch_plan, wave_status, tmp_path)
    template = build_full_llama_execution_evidence_template(dispatch_plan, wave_status, tmp_path)

    task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        readiness,
        template,
        tmp_path,
    )

    assert task["ready_to_spawn"] is False
    assert "target_preflight.status must be passed" in task["spawn_precondition_failures"]
    assert "If `Ready to spawn` is false, do not force the gate." in task["prompt"]


def _passed_full_execution_readiness_for_board() -> dict:
    return {
        "artifact": "full_llama_execution_readiness",
        "status": "passed",
        "safe_to_clear_full_llama_model_execution_blocker": True,
    }


def _minimal_board_signoff_evidence(**updates: object) -> dict:
    evidence = {
        "artifact": "board_zcu104_signoff_evidence",
        "status": "passed",
        "board": "ZCU104",
        "fpga_part": "xczu7ev-ffvc1156-2-e",
        "full_llama_execution_status": "passed",
        "target_scale_accelerator_bitstream": True,
        "accelerator_scope": "full_target_llama_accelerator",
        "constraints": {
            "clock": True,
            "reset": True,
            "board_io": True,
            "ps_pl_interface": True,
            "ddr_interface": True,
        },
        "timing": {
            "constraints_met": True,
            "setup_worst_slack_ns": 0.25,
            "hold_worst_slack_ns": 0.04,
            "pulse_width_worst_slack_ns": 1.5,
        },
        "resource_utilization": {
            "lut": 120000,
            "dsp": 900,
            "bram": 180,
            "ff": 180000,
            "uram": 8,
            "io": 32,
        },
        "reports": {
            "timing_summary": "timing_summary.rpt",
            "utilization": "utilization.rpt",
            "constraints": "zcu104.xdc",
            "vivado_log": "vivado.log",
            "drc": "drc.rpt",
            "methodology": "methodology.rpt",
            "clocks": "clocks.rpt",
            "checkpoint": "zcu104_post_route.dcp",
            "bitstream": "zcu104_board_wrapper.bit",
        },
        "bitstream": {
            "generated": True,
            "path": "zcu104_board_wrapper.bit",
            "size_bytes": 4096,
        },
        "fixture_only": False,
    }
    evidence.update(updates)
    return evidence


def test_board_zcu104_signoff_readiness_requires_full_execution_first(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = {"status": "blocked_by_missing_or_incomplete_full_execution_evidence"}

    report = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert report["status"] == "blocked_by_full_llama_execution"
    assert "board_zcu104_signoff_evidence.json not found" in report["evidence_failures"]
    assert "full_llama_execution_readiness must be passed before board signoff" in report["evidence_failures"]
    assert report["safe_to_clear_board_level_zcu104_signoff_blocker"] is False


def test_board_zcu104_signoff_evidence_template_matches_readiness_gate_fields(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()

    template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)

    assert template["artifact"] == "board_zcu104_signoff_evidence_template"
    assert template["write_to"] == str(tmp_path / "board_zcu104_signoff_evidence.json")
    assert template["requires_full_llama_execution_readiness_status"] == "passed"
    assert template["current_full_llama_execution_readiness_status"] == "passed"
    assert template["template"]["artifact"] == "board_zcu104_signoff_evidence"
    assert template["template"]["board"] == "ZCU104"
    assert template["template"]["fpga_part"] == "xczu7ev-ffvc1156-2-e"
    assert template["template"]["target_scale_accelerator_bitstream"] == "<true>"
    assert template["template"]["accelerator_scope"] == "full_target_llama_accelerator"
    assert template["template"]["fixture_only"] is False
    assert set(template["required_fields"]) == {
        "artifact",
        "status",
        "board",
        "fpga_part",
        "full_llama_execution_status",
        "target_scale_accelerator_bitstream",
        "accelerator_scope",
        "constraints",
        "timing",
        "resource_utilization",
        "reports",
        "bitstream",
        "fixture_only",
    }
    assert set(template["template"]["constraints"]) == {
        "clock",
        "reset",
        "board_io",
        "ps_pl_interface",
        "ddr_interface",
    }
    assert template["template"]["resource_utilization"] == {
        "lut": "<number <= 230000>",
        "dsp": "<number <= 1728>",
        "bram": "<number <= 312>",
    }
    assert template["template"]["bitstream"] == {
        "generated": "<true>",
        "path": "<path to generated .bit>",
        "size_bytes": "<positive integer>",
    }
    assert template["template"]["reports"]["bitstream"] == "<path to generated .bit>"
    assert "hardware_lab_runtime_validation" in template["does_not_claim"]


def test_board_zcu104_signoff_evidence_agent_task_ready_after_full_execution(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)

    task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    execution = build_target_evidence_execution_manifest(task)
    markdown = build_codex_spawn_instructions(execution)

    assert task["artifact"] == "board_zcu104_signoff_evidence_agent_task"
    assert task["spawn_kind"] == "target_evidence_agent"
    assert task["ready_to_spawn"] is True
    assert task["spawn_precondition_failures"] == []
    assert task["expected_evidence_file"] == str(tmp_path / "board_zcu104_signoff_evidence.json")
    assert task["expected_subagent_result"] == "build/board_zcu104_signoff_gate/subagent_result.json"
    assert "Target Evidence Sub-Agent Task: board_zcu104_signoff" in task["prompt"]
    assert "clock, reset, board I/O, PS/PL interface, and DDR interface" in task["prompt"]
    assert "A generated bitstream exists" in task["prompt"]
    assert "Existing bounded fixture synthesis reports are insufficient" in task["prompt"]
    assert "fixture_only: true" in task["prompt"]
    assert execution["spawn_entry_count"] == 1
    assert execution["spawn_entries"][0]["spawn_key"] == "target_evidence::board_zcu104_signoff_evidence"
    assert "board_zcu104_signoff_evidence_agent.md" in markdown
    assert "Evidence file:" in markdown


def test_board_zcu104_signoff_evidence_agent_task_waits_for_target_scale_wrapper(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    stale_evidence = _minimal_board_signoff_evidence(
        target_scale_accelerator_bitstream=False,
        accelerator_scope="zcu104_board_wrapper_control_scaffold",
    )
    (tmp_path / "board_zcu104_signoff_evidence.json").write_text(
        json.dumps(stale_evidence),
        encoding="utf-8",
    )
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)

    task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    execution = build_target_evidence_execution_manifest(task)

    assert readiness["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert "target_scale_accelerator_bitstream must be true" in readiness["evidence_failures"]
    assert readiness["next_action"] == "run target-scale ZCU104 board-wrapper implementation before board signoff"
    assert task["ready_to_spawn"] is False
    assert (
        "target-scale accelerator board-wrapper evidence must pass before board signoff"
        in task["spawn_precondition_failures"]
    )
    assert execution["spawn_entry_count"] == 0


def test_board_zcu104_signoff_readiness_reports_wrapper_target_scale_blocker_without_signoff_evidence(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")

    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert readiness["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert "board_zcu104_signoff_evidence.json not found" in readiness["evidence_failures"]
    assert "board-wrapper target_scale_accelerator_bitstream must be true" in readiness["evidence_failures"]
    assert "board-wrapper accelerator_scope must be full_target_llama_accelerator" in readiness["evidence_failures"]
    assert readiness["next_action"] == "run target-scale ZCU104 board-wrapper implementation before board signoff"
    assert readiness["board_wrapper_report"]["accelerator_scope"] == "ddr_axi_board_shell_fixture"


def test_target_scale_child_rtl_wave_ready_before_full_model_generator(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    child_tasks = {
        str(spec["task_id"]): build_target_scale_child_packet_agent_task(
            dispatch_plan,
            full_execution,
            readiness,
            tmp_path,
            str(spec["task_id"]),
            dispatch_plan_path="build/dispatch.json",
        )
        for spec in TARGET_SCALE_CHILD_PACKET_TASKS
    }
    generator_task = build_full_model_target_rtl_generator_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    artifact_task = build_full_target_llama_accelerator_artifact_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    board_wrapper_task = build_zcu104_board_wrapper_axi_bridge_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    signoff_template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)
    board_signoff_task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        signoff_template,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    child_executions = {
        task_id: build_target_evidence_execution_manifest(task)
        for task_id, task in child_tasks.items()
    }
    generator_execution = build_target_evidence_execution_manifest(generator_task)
    artifact_execution = build_target_evidence_execution_manifest(artifact_task)
    wrapper_execution = build_target_evidence_execution_manifest(board_wrapper_task)
    signoff_execution = build_target_evidence_execution_manifest(board_signoff_task)

    assert child_tasks["target_gptq_projection_datapath_packets"]["ready_to_spawn"] is True
    assert child_tasks["target_non_gemm_datapath_packets"]["ready_to_spawn"] is True
    assert child_tasks["target_ddr_weight_stream_scheduler_packet"]["ready_to_spawn"] is True
    assert child_tasks["target_decoder_block_integration_packet"]["ready_to_spawn"] is False
    assert (
        "target_gptq_projection_datapath_packets must pass before target_decoder_block_integration_packet"
        in child_tasks["target_decoder_block_integration_packet"]["spawn_precondition_failures"]
    )
    assert child_tasks["target_token_loop_16_layer_model_fsm_packet"]["ready_to_spawn"] is False
    assert (
        "target_decoder_block_integration_packet must pass before target_token_loop_16_layer_model_fsm_packet"
        in child_tasks["target_token_loop_16_layer_model_fsm_packet"]["spawn_precondition_failures"]
    )
    assert child_executions["target_gptq_projection_datapath_packets"]["spawn_entry_count"] == 1
    assert child_executions["target_non_gemm_datapath_packets"]["spawn_entry_count"] == 1
    assert child_executions["target_ddr_weight_stream_scheduler_packet"]["spawn_entry_count"] == 1
    assert generator_task["ready_to_spawn"] is False
    assert generator_task["task_id"] == "full_model_target_rtl_generator"
    assert generator_task["subagent_type"] == "target_scale_model_rtl_generator_subagent"
    assert generator_task["expected_evidence_file"].endswith("full_target_llama_accelerator_gate/kernel_report.json")
    assert "board-wrapper-compatible interface" in generator_task["prompt"]
    assert (
        "target_scale_child_rtl_wave must pass before full-model target RTL generation"
        in generator_task["spawn_precondition_failures"][0]
    )
    assert generator_execution["spawn_entry_count"] == 0
    assert artifact_task["ready_to_spawn"] is False
    assert artifact_task["task_id"] == "full_target_llama_accelerator_artifact"
    assert (
        "full_model_target_rtl_generator must pass before target-scale accelerator artifact validation"
        in artifact_task["spawn_precondition_failures"]
    )
    assert artifact_execution["spawn_entry_count"] == 0
    assert board_wrapper_task["ready_to_spawn"] is False
    assert (
        "full_target_llama_accelerator artifact must pass before rerouting board wrapper"
        in board_wrapper_task["spawn_precondition_failures"]
    )
    assert wrapper_execution["spawn_entry_count"] == 0
    assert board_signoff_task["ready_to_spawn"] is False
    assert (
        "target-scale accelerator board-wrapper evidence must pass before board signoff"
        in board_signoff_task["spawn_precondition_failures"]
    )
    assert signoff_execution["spawn_entry_count"] == 0


def test_target_scale_child_packet_blocks_retry_after_nonpassed_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    evidence_dir = tmp_path / "target_non_gemm_datapath_packets_gate"
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "coverage_level": "target_non_gemm_scheduled_datapath_prototype",
                "target_scale_child_eligible": False,
                "numeric_policy": {"fixture_only": False},
                "skill_update_candidate": {
                    "failing_command": "python3 -m nl2hdl subagents status ...",
                    "symptom": "checksum-only non-GEMM prototype lacks tensor payload streams",
                    "root_cause_hypothesis": "payload stream contracts were not required",
                    "prevention_rule": "require tensor payload streams before target_non_gemm pass",
                    "minimal_regression_check": "reject target_non_gemm pass without tensor_payload_streams_present",
                },
            }
        ),
        encoding="utf-8",
    )

    task = build_target_scale_child_packet_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        "target_non_gemm_datapath_packets",
        dispatch_plan_path="build/dispatch.json",
    )
    execution = build_target_evidence_execution_manifest(task)

    assert task["ready_to_spawn"] is False
    assert task["existing_child_blocker"]["skill_update_candidate_complete"] is True
    assert (
        "target_non_gemm_datapath_packets has existing non-passed target-scale child evidence"
        in task["spawn_precondition_failures"][0]
    )
    assert execution["spawn_entry_count"] == 0
    assert execution["skill_update_required"] is True
    assert execution["blocked_target_evidence_tasks"][0]["next_action"] == (
        "run_subagents_skill_draft_and_update_skill_before_retry"
    )


def test_target_scale_child_packet_blocks_retry_after_integration_timing_failure(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    non_gemm_dir = tmp_path / "target_non_gemm_datapath_packets_gate"
    non_gemm_dir.mkdir()
    (non_gemm_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "coverage_level": "target_non_gemm_tensor_payload_datapaths",
                "target_scale_child_eligible": True,
                "numeric_policy": {"fixture_only": False},
                "interface_contract": {
                    "tensor_payload_streams_present": True,
                    "payload_streams": {
                        "rmsnorm": {"present": True},
                        "rope": {"present": True},
                        "softmax": {"present": True},
                        "kv_cache": {"present": True},
                        "residual": {"present": True},
                        "swiglu": {"present": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    decoder_dir = tmp_path / "target_decoder_block_integration_packet_gate"
    decoder_dir.mkdir()
    (decoder_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "coverage_level": "target_decoder_block_integration_timing_blocked",
                "ooc_synthesis": {"status": "failed_timing", "setup_worst_slack_ns": -2.5},
                "blocking_reason": "non-GEMM child timing failed in decoder integration",
                "skill_update_candidate": {
                    "failing_command": "vivado -mode batch -source run_ooc_synth.tcl",
                    "symptom": "registered-source integration exposed non-GEMM timing failure",
                    "root_cause_hypothesis": "child standalone OOC did not constrain registered source/sink paths",
                    "prevention_rule": "require registered-source/registered-sink timing wrapper before child pass",
                    "minimal_regression_check": "synthesize non-GEMM timing wrapper at 5ns and require WNS >= 0",
                },
            }
        ),
        encoding="utf-8",
    )

    task = build_target_scale_child_packet_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        "target_non_gemm_datapath_packets",
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["ready_to_spawn"] is False
    assert task["integration_blocker"]["source_task"] == "target_decoder_block_integration_packet"
    assert task["existing_child_blocker"]["skill_update_candidate_complete"] is True
    assert any(
        "run_subagents_skill_draft_and_update_skill_before_retry" in failure
        for failure in task["spawn_precondition_failures"]
    )


def test_target_scale_child_packet_ignores_stale_integration_timing_failure(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    decoder_dir = tmp_path / "target_decoder_block_integration_packet_gate"
    decoder_dir.mkdir()
    decoder_report = decoder_dir / "kernel_report.json"
    decoder_report.write_text(
        json.dumps(
            {
                "status": "blocked",
                "coverage_level": "target_decoder_block_integration_timing_blocked",
                "ooc_synthesis": {"status": "failed_timing", "setup_worst_slack_ns": -2.5},
                "blocking_reason": "non-GEMM child timing failed in decoder integration",
                "skill_update_candidate": {
                    "failing_command": "vivado -mode batch -source run_ooc_synth.tcl",
                    "symptom": "registered-source integration exposed non-GEMM timing failure",
                    "root_cause_hypothesis": "child standalone OOC did not constrain registered source/sink paths",
                    "prevention_rule": "require registered-source/registered-sink timing wrapper before child pass",
                    "minimal_regression_check": "synthesize non-GEMM timing wrapper at 5ns and require WNS >= 0",
                },
            }
        ),
        encoding="utf-8",
    )
    non_gemm_dir = tmp_path / "target_non_gemm_datapath_packets_gate"
    non_gemm_dir.mkdir()
    child_report = non_gemm_dir / "kernel_report.json"
    child_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "coverage_level": "target_non_gemm_tensor_payload_datapaths",
                "target_scale_child_eligible": True,
                "numeric_policy": {"fixture_only": False},
                "interface_contract": {
                    "tensor_payload_streams_present": True,
                    "payload_streams": {
                        "rmsnorm": {"present": True},
                        "rope": {"present": True},
                        "softmax": {"present": True},
                        "kv_cache": {"present": True},
                        "residual": {"present": True},
                        "swiglu": {"present": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    os.utime(decoder_report, (1000, 1000))
    os.utime(child_report, (2000, 2000))

    task = build_target_scale_child_packet_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        "target_non_gemm_datapath_packets",
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["integration_blocker"] is None
    assert task["existing_child_blocker"] is None
    assert task["spawn_precondition_failures"] == [
        "target_non_gemm_datapath_packets already has target-scale eligible child evidence"
    ]


def test_decoder_integration_packet_retry_opens_after_newer_child_fix(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "ddr_axi_board_shell_fixture"
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    passed_child_reports = {
        "target_gptq_projection_datapath_packets_gate": {
            "status": "passed",
            "coverage_level": "target_gptq_projection_datapath_packets",
            "target_scale_child_eligible": True,
            "numeric_policy": {"fixture_only": False},
        },
        "target_ddr_weight_stream_scheduler_packet_gate": {
            "status": "passed",
            "coverage_level": "target_ddr_weight_stream_scheduler",
            "target_scale_child_eligible": True,
            "numeric_policy": {"fixture_only": False},
        },
        "target_non_gemm_datapath_packets_gate": {
            "status": "passed",
            "coverage_level": "target_non_gemm_tensor_payload_datapaths",
            "target_scale_child_eligible": True,
            "numeric_policy": {"fixture_only": False},
            "interface_contract": {
                "tensor_payload_streams_present": True,
                "payload_streams": {
                    "rmsnorm": {"present": True},
                    "rope": {"present": True},
                    "softmax": {"present": True},
                    "kv_cache": {"present": True},
                    "residual": {"present": True},
                    "swiglu": {"present": True},
                },
            },
        },
    }
    for directory_name, report in passed_child_reports.items():
        child_dir = tmp_path / directory_name
        child_dir.mkdir()
        child_report = child_dir / "kernel_report.json"
        child_report.write_text(json.dumps(report), encoding="utf-8")
        os.utime(child_report, (900, 900))

    decoder_dir = tmp_path / "target_decoder_block_integration_packet_gate"
    decoder_dir.mkdir()
    decoder_report = decoder_dir / "kernel_report.json"
    decoder_report.write_text(
        json.dumps(
            {
                "status": "blocked",
                "coverage_level": "target_decoder_block_integration_timing_blocked",
                "ooc_synthesis": {"status": "failed_timing", "setup_worst_slack_ns": -2.5},
                "blocking_reason": "non-GEMM child timing failed in decoder integration",
                "skill_update_candidate": {
                    "failing_command": "vivado -mode batch -source run_ooc_synth.tcl",
                    "symptom": "registered-source integration exposed non-GEMM timing failure",
                    "root_cause_hypothesis": "child standalone OOC did not constrain registered source/sink paths",
                    "prevention_rule": "require registered-source/registered-sink timing wrapper before child pass",
                    "minimal_regression_check": "synthesize non-GEMM timing wrapper at 5ns and require WNS >= 0",
                },
            }
        ),
        encoding="utf-8",
    )
    os.utime(decoder_report, (1000, 1000))
    os.utime(
        tmp_path / "target_non_gemm_datapath_packets_gate" / "kernel_report.json",
        (2000, 2000),
    )

    task = build_target_scale_child_packet_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        "target_decoder_block_integration_packet",
        dispatch_plan_path="build/dispatch.json",
    )

    assert task["ready_to_spawn"] is True
    assert task["existing_child_blocker"] is None
    assert task["spawn_precondition_failures"] == []


def test_zcu104_board_wrapper_axi_bridge_agent_task_can_spawn_and_create_missing_pre_signoff_package(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    task = build_zcu104_board_wrapper_axi_bridge_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    execution = build_target_evidence_execution_manifest(task)

    assert task["artifact"] == "zcu104_board_wrapper_axi_bridge_agent_task"
    assert task["spawn_kind"] == "target_evidence_implementation_agent"
    assert task["ready_to_spawn"] is True
    assert task["spawn_precondition_failures"] == []
    assert task["pre_signoff_package_missing"] is True
    assert "If the pre-signoff constraint package is missing" in task["prompt"]
    assert execution["spawn_entry_count"] == 1
    assert execution["spawn_entries"][0]["task_id"] == "zcu104_board_wrapper_axi_bridge"


def test_zcu104_board_wrapper_axi_bridge_agent_task_ready_after_pre_signoff_package(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    build_zcu104_board_shell_constraints_package(cfg, tmp_path / "board_zcu104_signoff_gate")
    gap_report = {
        "artifact": "board_zcu104_signoff_gap_report",
        "status": "blocked_by_unproven_board_level_signoff",
        "skill_update_candidate": {
            "failing_command": "python3 -m nl2hdl subagents status --dispatch-plan build/dispatch.json",
            "symptom": "NSTD-1 and UCIO-1 remain on aclk/aresetn; later retry resolved DRC but clk_pl_0 stayed at 5.625 ns / 177.778 MHz for a 200 MHz target.",
            "root_cause_hypothesis": "Direct PL shell was routed while BD wrapper was only generated beside it, and PS PL clock configuration was not proven against implemented reports.",
            "prevention_rule": "Route the generated PS/PL/DDR wrapper or equivalent top before board signoff and prove the configured target clock from raw Vivado reports.",
            "minimal_regression_check": "Reject positive timing/resource reports when DRC still flags aclk/aresetn or report_clocks/implemented XDC show a period above 5.000 ns for a 200 MHz target.",
        },
    }
    (tmp_path / "board_zcu104_signoff_gate" / "evidence_gap_report.json").write_text(
        json.dumps(gap_report, indent=2),
        encoding="utf-8",
    )
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    task = build_zcu104_board_wrapper_axi_bridge_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        tmp_path,
        dispatch_plan_path="build/dispatch.json",
    )
    execution = build_target_evidence_execution_manifest(task)
    markdown = build_codex_spawn_instructions(execution)

    assert task["ready_to_spawn"] is True
    assert task["spawn_precondition_failures"] == []
    assert task["expected_evidence_file"] == str(
        tmp_path / "board_zcu104_signoff_gate" / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    )
    assert task["expected_subagent_result"] == str(
        tmp_path / "board_zcu104_signoff_gate" / "zcu104_board_wrapper_axi_bridge_subagent_result.json"
    )
    assert "zcu104_board_shell_top" in task["prompt"]
    assert task["prior_board_signoff_gap_status"] == "blocked_by_unproven_board_level_signoff"
    assert task["prior_board_signoff_skill_update_candidate"]["symptom"].startswith("NSTD-1")
    assert "do not route only the direct PL shell" in task["prompt"]
    assert "generated PS/PL/DDR BD wrapper" in task["prompt"]
    assert "no NSTD-1 or UCIO-1 critical warnings" in task["prompt"]
    assert "target clock mismatch" in task["prompt"]
    assert "report_clocks" in task["prompt"]
    assert "implemented XDC" in task["prompt"]
    assert "5.000 ns" in task["prompt"]
    assert "177.778 MHz" in task["prompt"]
    assert "target clock" in task["prompt"]
    assert "Positive timing/resource reports are not enough" in task["prompt"]
    assert "Do not write `build/board_zcu104_signoff_evidence.json`" in task["prompt"]
    assert "vivado -version" in task["required_commands"]
    assert execution["spawn_entry_count"] == 1
    assert execution["implementation_spawn_count"] == 1
    assert execution["target_evidence_spawn_count"] == 0
    assert execution["spawn_entries"][0]["spawn_kind"] == "target_evidence_implementation_agent"
    assert "zcu104_board_wrapper_axi_bridge_agent.md" in markdown


def test_board_zcu104_signoff_evidence_agent_task_blocks_before_full_execution(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = {"status": "blocked_by_missing_or_incomplete_full_execution_evidence"}
    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)
    template = build_board_zcu104_signoff_evidence_template(dispatch_plan, full_execution, tmp_path)

    task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution,
        readiness,
        template,
        tmp_path,
    )

    assert task["ready_to_spawn"] is False
    assert "full_llama_execution_readiness must be passed before board signoff" in task["spawn_precondition_failures"]


def test_board_zcu104_signoff_readiness_requires_explicit_board_evidence(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()

    report = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert report["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert report["evidence_failures"] == ["board_zcu104_signoff_evidence.json not found"]
    assert report["safe_to_clear_board_level_zcu104_signoff_blocker"] is False


def test_board_zcu104_signoff_readiness_passes_with_timing_constraints_and_resources(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    (tmp_path / "board_zcu104_signoff_evidence.json").write_text(
        json.dumps(_minimal_board_signoff_evidence()),
        encoding="utf-8",
    )

    report = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert report["status"] == "passed"
    assert report["safe_to_clear_board_level_zcu104_signoff_blocker"] is True
    assert report["evidence_failures"] == []
    assert report["next_action"] == "board-level ZCU104 signoff readiness passed"


def _write_fake_passed_board_wrapper_report(wrapper_dir: Path, dispatch_plan: dict) -> None:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    file_names = {
        "vivado_log": "vivado.log",
        "timing_summary": "zcu104_timing_summary.rpt",
        "utilization": "zcu104_utilization.rpt",
        "drc": "zcu104_drc.rpt",
        "methodology": "zcu104_methodology.rpt",
        "clocks": "zcu104_clocks.rpt",
        "implemented_constraints": "zcu104_implemented_constraints.xdc",
        "checkpoint": "zcu104_post_route.dcp",
        "bitstream": "zcu104_board_wrapper.bit",
    }
    for file_name in file_names.values():
        path = wrapper_dir / file_name
        if path.suffix == ".bit":
            path.write_bytes(b"fake bitstream")
        else:
            path.write_text(f"fake {file_name}\n", encoding="utf-8")
    (wrapper_dir / "zcu104_io.rpt").write_text(
        "\n".join(
            [
                "+---------------+",
                "| Total User IO |",
                "+---------------+",
                "|             9 |",
                "+---------------+",
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "artifact": "zcu104_board_wrapper_axi_bridge_implementation_report",
        "status": "passed",
        "board": "AMD ZCU104",
        "fpga_part": dispatch_plan["hardware"]["fpga_part"],
        "target_clock_mhz": 200,
        "target_scale_accelerator_bitstream": True,
        "accelerator_scope": "full_target_llama_accelerator",
        "route_completed": True,
        "route_check_command_passed": True,
        "vivado_available": True,
        "bitstream_generated": True,
        "bitstream_file": str(wrapper_dir / "zcu104_board_wrapper.bit"),
        "bitstream_size_bytes": (wrapper_dir / "zcu104_board_wrapper.bit").stat().st_size,
        "static_integration_evidence": {
            "routed_top_instantiates_generated_bd_wrapper": True,
            "ps_fclk_reset_drive_accelerator": True,
            "ps_axi_reaches_control_registers": True,
            "ddr_address_map_declared": True,
            "axi_lite_interface_metadata_present": True,
            "board_visible_ports_are_compact": True,
        },
        "route_report_analysis": {
            "timing": {
                "constraints_met": True,
                "setup_worst_slack_ns": 0.5,
                "hold_worst_slack_ns": 0.02,
                "pulse_width_worst_slack_ns": 1.0,
            },
            "drc": {"passes": True, "blocking_rules": [], "critical_warning_count": 0},
            "clock": {
                "observed_period_ns": 5.0,
                "implemented_xdc_period_ns": 5.0,
                "target_period_ns": 5.0,
                "target_clock_mhz": 200,
            },
            "utilization": {"lut": 464, "ff": 706, "dsp": 0, "bram": 0, "uram": 0},
            "utilization_budget": {"passes": True, "failures": []},
            "gate_failures": [],
            "reports_present": {
                "timing_summary": True,
                "utilization": True,
                "drc": True,
                "clocks": True,
                "implemented_constraints": True,
                "checkpoint": True,
                "bitstream": True,
            },
        },
        "evidence_files": {key: str(wrapper_dir / file_name) for key, file_name in file_names.items()},
    }
    (wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def test_run_board_zcu104_signoff_evidence_agent_writes_evidence_from_wrapper_report(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)

    report = run_board_zcu104_signoff_evidence_agent(
        dispatch_plan,
        full_execution,
        tmp_path,
        board_wrapper_dir=wrapper_dir,
    )

    assert report["status"] == "passed"
    assert report["evidence_written"] is True
    assert report["readiness"]["status"] == "passed"
    evidence = json.loads((tmp_path / "board_zcu104_signoff_evidence.json").read_text())
    assert evidence["artifact"] == "board_zcu104_signoff_evidence"
    assert evidence["status"] == "passed"
    assert evidence["target_scale_accelerator_bitstream"] is True
    assert evidence["accelerator_scope"] == "full_target_llama_accelerator"
    assert evidence["bitstream"]["generated"] is True
    assert evidence["bitstream"]["size_bytes"] > 0
    assert evidence["resource_utilization"]["io"] == 9
    assert evidence["fixture_only"] is False
    subagent_result = json.loads((wrapper_dir / "subagent_result.json").read_text())
    assert subagent_result["status"] == "passed"


def test_run_board_zcu104_signoff_evidence_agent_uses_utilization_iob_when_io_report_missing(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    (wrapper_dir / "zcu104_io.rpt").unlink()
    (wrapper_dir / "zcu104_utilization.rpt").write_text(
        "\n".join(
            [
                "5. I/O",
                "+------------------+------+",
                "| Site Type        | Used |",
                "+------------------+------+",
                "| Bonded IOB       |    9 |",
                "+------------------+------+",
            ]
        ),
        encoding="utf-8",
    )

    report = run_board_zcu104_signoff_evidence_agent(
        dispatch_plan,
        full_execution,
        tmp_path,
        board_wrapper_dir=wrapper_dir,
    )

    assert report["status"] == "passed"
    evidence = json.loads((tmp_path / "board_zcu104_signoff_evidence.json").read_text())
    assert evidence["resource_utilization"]["io"] == 9


def test_run_board_zcu104_signoff_evidence_agent_rejects_control_scaffold_bitstream(
    tmp_path: Path,
):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    wrapper_dir = tmp_path / "board_zcu104_signoff_gate"
    _write_fake_passed_board_wrapper_report(wrapper_dir, dispatch_plan)
    report_path = wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    wrapper_report = json.loads(report_path.read_text(encoding="utf-8"))
    wrapper_report["target_scale_accelerator_bitstream"] = False
    wrapper_report["accelerator_scope"] = "zcu104_board_wrapper_control_scaffold"
    wrapper_report["does_not_claim"] = [
        "board_level_ZCU104_signoff",
        "full_target_scale_LLaMA_accelerator_bitstream",
    ]
    report_path.write_text(json.dumps(wrapper_report, indent=2), encoding="utf-8")

    report = run_board_zcu104_signoff_evidence_agent(
        dispatch_plan,
        full_execution,
        tmp_path,
        board_wrapper_dir=wrapper_dir,
    )

    assert report["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert report["evidence_written"] is False
    assert "board-wrapper target_scale_accelerator_bitstream must be true" in report["failures"]
    assert "board-wrapper accelerator_scope must be full_target_llama_accelerator" in report["failures"]
    assert not (tmp_path / "board_zcu104_signoff_evidence.json").exists()


def test_board_zcu104_signoff_readiness_rejects_fixture_only_or_timing_failure(tmp_path: Path):
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()
    bad_timing = _minimal_board_signoff_evidence(
        fixture_only=True,
        timing={
            "constraints_met": False,
            "setup_worst_slack_ns": 0.1,
            "hold_worst_slack_ns": -0.01,
            "pulse_width_worst_slack_ns": 1.0,
        },
    )
    (tmp_path / "board_zcu104_signoff_evidence.json").write_text(json.dumps(bad_timing), encoding="utf-8")

    report = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert report["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert "timing.constraints_met must be true" in report["evidence_failures"]
    assert "timing.hold_worst_slack_ns must be non-negative" in report["evidence_failures"]
    assert "fixture_only board evidence cannot satisfy board-level signoff" in report["evidence_failures"]


def test_zcu104_board_shell_constraints_package_distinguishes_pre_signoff_from_evidence(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    fixture_report = tmp_path / "fixture_kernel_report.json"
    fixture_report.write_text(
        json.dumps(
            {
                "kernel": "ddr_axi_board_shell_fixture",
                "numeric_policy": {
                    "real_ddr_controller_ip": False,
                    "ps_pl_block_design": False,
                    "board_level_signoff": False,
                    "full_llama_model": False,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_zcu104_board_shell_constraints_package(
        cfg,
        tmp_path,
        fixture_report_path=fixture_report,
    )

    assert report["artifact"] == "zcu104_board_shell_signoff_readiness_report"
    assert report["status"] == "blocked_pending_vivado_board_integration_run"
    assert report["board"] == "ZCU104"
    assert report["fpga_part"] == "xczu7ev-ffvc1156-2-e"
    assert report["signoff_evidence_ready"] is False
    assert report["board_zcu104_signoff_evidence_written"] is False
    assert report["fixture_only_candidate_rejected"] is True
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]
    assert not (tmp_path / "board_zcu104_signoff_evidence.json").exists()

    xdc_text = (tmp_path / "zcu104_board_shell_constraints.xdc").read_text(encoding="utf-8")
    assert "[get_ports aclk]" not in xdc_text
    assert "[get_ports aresetn]" not in xdc_text
    assert "set_property PACKAGE_PIN M11 [get_ports board_reset_i]" in xdc_text
    assert "set_property IOSTANDARD LVCMOS33 [get_ports board_reset_i]" in xdc_text
    assert "set_false_path -from [get_ports board_reset_i]" in xdc_text
    for pin in ("G8", "H8", "G7", "H7", "G6", "H6", "J6", "J7"):
        assert f"set_property PACKAGE_PIN {pin}" in xdc_text
    assert "set_false_path -to [get_ports {pmod0_o[*]}]" in xdc_text

    accel_core_text = (tmp_path / "zcu104_accelerator_core.sv").read_text(encoding="utf-8")
    axi_bridge_text = (tmp_path / "zcu104_axi_lite_accel_bridge.sv").read_text(encoding="utf-8")
    axi_subsystem_text = (tmp_path / "zcu104_board_shell_axi_subsystem.sv").read_text(encoding="utf-8")
    board_top_text = (tmp_path / "zcu104_board_shell_top.sv").read_text(encoding="utf-8")
    bd_wrapper_text = (tmp_path / "zcu104_board_shell_axi_subsystem_bd.v").read_text(encoding="utf-8")
    bd_top_text = (tmp_path / "zcu104_board_ps_pl_ddr_top.v").read_text(encoding="utf-8")
    assert "module zcu104_accelerator_core" in accel_core_text
    assert "input  logic                    start_i" in accel_core_text
    assert "output logic                    done_o" in accel_core_text
    assert "module zcu104_axi_lite_accel_bridge" in axi_bridge_text
    assert "CONTROL_ADDR = 8'h00" in axi_bridge_text
    assert "STATUS_ADDR  = 8'h04" in axi_bridge_text
    assert "SCRATCH_ADDR = 8'h08" in axi_bridge_text
    assert "output logic                      accel_start_o" in axi_bridge_text
    assert "module zcu104_board_shell_axi_subsystem" in axi_subsystem_text
    assert "output logic [7:0]                pmod0_o" in axi_subsystem_text
    assert "zcu104_axi_lite_accel_bridge" in axi_subsystem_text
    assert "zcu104_accelerator_core" in axi_subsystem_text
    assert "module zcu104_board_shell_top" in board_top_text
    assert "zcu104_board_shell_axi_subsystem u_board_shell_axi_subsystem" in board_top_text
    assert ".s_axi_awvalid(1'b1)" in board_top_text
    assert ".s_axi_wdata(32'h0000_0001)" in board_top_text
    assert "module zcu104_board_shell_axi_subsystem_bd" in bd_wrapper_text
    assert "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" in bd_wrapper_text
    assert "PROTOCOL AXI4LITE" in bd_wrapper_text
    assert "ASSOCIATED_BUSIF S_AXI" in bd_wrapper_text
    assert "zcu104_board_shell_axi_subsystem u_zcu104_board_shell_axi_subsystem" in bd_wrapper_text
    assert "module zcu104_board_ps_pl_ddr_top" in bd_top_text
    assert "zcu104_ps_pl_ddr_bd_wrapper u_ps_pl_ddr_bd_wrapper" in bd_top_text
    assert report["wrapper_implementation"]["board_level_ports_kept_compact"] is True
    assert report["wrapper_implementation"]["control_path"].startswith("PS AXI-lite")

    bd_tcl_text = (tmp_path / "zcu104_ps_pl_ddr_bd.tcl").read_text(encoding="utf-8")
    assert "get_board_parts -quiet -latest_file_version *zcu104*" in bd_tcl_text
    assert "create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0" in bd_tcl_text
    assert "CONFIG.PSU__USE__M_AXI_GP0 1" in bd_tcl_text
    assert "CONFIG.PSU__USE__S_AXI_GP0 0" in bd_tcl_text
    assert "CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL RPLL" in bd_tcl_text
    assert "CONFIG.PSU__CRL_APB__PL0_REF_CTRL__DIVISOR0 6" in bd_tcl_text
    assert "PS PL0 clock resolved below 200 MHz target" in bd_tcl_text
    assert "proc_sys_reset_0" in bd_tcl_text
    assert "axi_interconnect_0" in bd_tcl_text
    assert "read_verilog [file join $origin_dir zcu104_board_shell_axi_subsystem_bd.v]" in bd_tcl_text
    assert "create_bd_cell -type module -reference zcu104_board_shell_axi_subsystem_bd" in bd_tcl_text
    assert "assign_bd_address" in bd_tcl_text
    assert "0xA0000000" in bd_tcl_text
    assert "CONFIG.PSU__USE__M_AXI_GP2 1" in bd_tcl_text
    assert "M_AXI_HPM0_LPD" in bd_tcl_text
    assert "No PS AXI HPM master interface pin found" in bd_tcl_text
    assert "pl_ps_irq0" in bd_tcl_text
    assert "make_bd_pins_external [get_bd_pins zcu104_board_shell_top_0/interrupt_o]" not in bd_tcl_text
    assert "validate_bd_design" in bd_tcl_text

    route_tcl_text = (tmp_path / "zcu104_board_route_check.tcl").read_text(encoding="utf-8")
    assert "read_verilog -sv zcu104_accelerator_core.sv" in route_tcl_text
    assert "read_verilog -sv zcu104_axi_lite_accel_bridge.sv" in route_tcl_text
    assert "read_verilog -sv zcu104_board_shell_axi_subsystem.sv" in route_tcl_text
    assert "read_verilog -sv zcu104_board_shell_top.sv" in route_tcl_text
    assert "read_verilog zcu104_board_shell_axi_subsystem_bd.v" in route_tcl_text
    assert "read_verilog zcu104_board_ps_pl_ddr_top.v" in route_tcl_text
    assert "read_xdc zcu104_board_shell_constraints.xdc" in route_tcl_text
    assert "source zcu104_ps_pl_ddr_bd.tcl" in route_tcl_text
    assert "make_wrapper -files $bd_files -top -force" in route_tcl_text
    assert "set_property top zcu104_board_ps_pl_ddr_top [current_fileset]" in route_tcl_text
    assert "launch_runs synth_1" in route_tcl_text
    assert "launch_runs impl_1 -to_step route_design" in route_tcl_text
    assert "synth_design -top zcu104_board_shell_top" not in route_tcl_text
    assert "route_design" in route_tcl_text
    assert "report_timing_summary -file zcu104_timing_summary.rpt" in route_tcl_text
    assert "report_clocks -file zcu104_clocks.rpt" in route_tcl_text
    assert "Routed clk_pl_0 period exceeds 200 MHz target" in route_tcl_text
    assert "Implemented XDC clk_pl_0 period exceeds 200 MHz target" in route_tcl_text
    assert "report_utilization -file zcu104_utilization.rpt" in route_tcl_text
    assert "report_methodology -file zcu104_methodology.rpt" in route_tcl_text
    assert "write_xdc -force zcu104_implemented_constraints.xdc" in route_tcl_text
    assert "write_checkpoint -force zcu104_post_route.dcp" in route_tcl_text
    assert "write_bitstream -force zcu104_board_wrapper.bit" in route_tcl_text

    saved = json.loads(
        (tmp_path / "zcu104_board_shell_signoff_readiness_report.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert report["wrapper_implementation"]["routed_top_module"] == "zcu104_board_ps_pl_ddr_top"


def test_zcu104_board_wrapper_agent_blocks_when_vivado_is_missing(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    report = run_zcu104_board_wrapper_axi_bridge_agent(
        cfg,
        tmp_path,
        run_vivado=True,
        vivado_executable="vivado_missing_for_nl2hdl_test",
    )

    assert report["artifact"] == "zcu104_board_wrapper_axi_bridge_implementation_report"
    assert report["status"] == "blocked_by_missing_vivado"
    assert report["vivado_available"] is False
    assert report["route_completed"] is False
    assert report["bitstream_generated"] is False
    assert report["implementation_evidence_ready"] is False
    assert report["board_zcu104_signoff_evidence_written"] is False
    assert report["final_board_signoff_still_blocked"] is True
    assert "skill_update_candidate" in report
    assert not (tmp_path / "board_zcu104_signoff_evidence.json").exists()
    assert (tmp_path / "zcu104_board_wrapper_axi_bridge_implementation_report.json").exists()
    assert (tmp_path / "zcu104_board_wrapper_axi_bridge_subagent_result.json").exists()
    saved = json.loads(
        (tmp_path / "zcu104_board_wrapper_axi_bridge_implementation_report.json").read_text(encoding="utf-8")
    )
    assert saved == report
    result = json.loads(
        (tmp_path / "zcu104_board_wrapper_axi_bridge_subagent_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "blocked_by_missing_vivado"
    assert result["final_signoff_still_blocked"] is True
    assert any(command["missing_executable"] for command in result["commands_run"])


def test_zcu104_board_shell_constraints_package_wraps_generated_accelerator_artifact(
    tmp_path: Path,
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    artifact_dir = tmp_path / "generated_accelerator"
    artifact_dir.mkdir()
    (artifact_dir / "generated_model_top.sv").write_text(
        """module generated_model_top(
    input logic aclk,
    input logic aresetn,
    input logic start_i,
    output logic done_o,
    output logic [31:0] status_o
);
    assign done_o = start_i & aresetn;
    assign status_o = 32'h1234_5678;
endmodule
""",
        encoding="utf-8",
    )
    kernel_report = {
        "status": "passed",
        "kernel": "generated_model_top",
        "coverage_level": "model_fsm_axi_decoder_block_fixture",
        "numeric_policy": {
            "full_llama_model": False,
            "board_level_signoff": False,
        },
    }
    (artifact_dir / "kernel_report.json").write_text(json.dumps(kernel_report), encoding="utf-8")

    report = build_zcu104_board_shell_constraints_package(
        cfg,
        tmp_path / "board_gate",
        accelerator_artifact_dir=artifact_dir,
    )

    board_gate = tmp_path / "board_gate"
    core_text = (board_gate / "zcu104_accelerator_core.sv").read_text(encoding="utf-8")
    route_tcl = (board_gate / "zcu104_board_route_check.tcl").read_text(encoding="utf-8")
    bd_tcl = (board_gate / "zcu104_ps_pl_ddr_bd.tcl").read_text(encoding="utf-8")

    assert (board_gate / "generated_model_top.sv").exists()
    assert "generated_model_top u_target_accelerator" in core_text
    assert "read_verilog -sv generated_model_top.sv" in route_tcl
    assert "foreach rtl_file [list generated_model_top.sv zcu104_accelerator_core.sv" in bd_tcl
    assert report["accelerator_binding"]["source"] == "generated_accelerator_artifact"
    assert report["accelerator_binding"]["target_scale_eligible"] is False
    assert "kernel_report.numeric_policy.full_llama_model is not true" in report["accelerator_binding"][
        "target_scale_blockers"
    ]


def test_zcu104_board_wrapper_agent_marks_target_scale_only_with_eligible_artifact(
    monkeypatch,
    tmp_path: Path,
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    artifact_dir = tmp_path / "generated_accelerator"
    artifact_dir.mkdir()
    (artifact_dir / "generated_model_top.sv").write_text(
        """module generated_model_top(
    input logic aclk,
    input logic aresetn,
    input logic start_i,
    output logic done_o,
    output logic [31:0] status_o
);
    assign done_o = start_i & aresetn;
    assign status_o = 32'hfeed_cafe;
endmodule
""",
        encoding="utf-8",
    )
    (artifact_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "kernel": "generated_model_top",
                "coverage_level": "full_target_llama_accelerator",
                "numeric_policy": {
                    "full_llama_model": True,
                    "board_level_signoff": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, cwd, text, stdout, stderr, timeout, check):
        assert text is True
        assert stdout is not None
        assert stderr is not None
        assert check is False
        if "-mode" in command:
            (cwd / "zcu104_timing_summary.rpt").write_text(
                "\n".join(
                    [
                        "setup_worst_slack_ns: 0.250",
                        "hold_worst_slack_ns: 0.025",
                        "pulse_width_worst_slack_ns: 0.500",
                        "setup_failing_endpoints: 0",
                        "hold_failing_endpoints: 0",
                        "pulse_width_failing_endpoints: 0",
                    ]
                ),
                encoding="utf-8",
            )
            (cwd / "zcu104_utilization.rpt").write_text(
                "\n".join(["lut: 1200", "ff: 2100", "dsp: 8", "bram: 4", "uram: 0"]),
                encoding="utf-8",
            )
            (cwd / "zcu104_drc.rpt").write_text("No board-wrapper blocking critical warnings.\n", encoding="utf-8")
            (cwd / "zcu104_clocks.rpt").write_text("clk_pl_0 5.000\n", encoding="utf-8")
            (cwd / "zcu104_implemented_constraints.xdc").write_text(
                "create_clock -period 5.000 -name clk_pl_0 [get_pins u_ps/pl_clk0]\n",
                encoding="utf-8",
            )
            (cwd / "zcu104_post_route.dcp").write_text("fake checkpoint\n", encoding="utf-8")
            (cwd / "zcu104_board_wrapper.bit").write_bytes(b"fake target bitstream\n")
            return types.SimpleNamespace(returncode=0, stdout="route ok\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="Vivado v2024.1\n", stderr="")

    monkeypatch.setattr("nl2hdl.llm_kernels.subprocess.run", fake_run)

    report = run_zcu104_board_wrapper_axi_bridge_agent(
        cfg,
        tmp_path / "board_gate",
        accelerator_artifact_dir=artifact_dir,
        run_vivado=True,
    )

    assert report["status"] == "passed"
    assert report["bitstream_generated"] is True
    assert report["target_scale_accelerator_bitstream"] is True
    assert report["accelerator_scope"] == "full_target_llama_accelerator"
    assert report["accelerator_binding"]["top_module"] == "generated_model_top"
    assert "full_target_scale_LLaMA_accelerator_bitstream" not in report["does_not_claim"]


def test_zcu104_board_wrapper_agent_passes_with_fake_routed_reports(monkeypatch, tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    def fake_run(command, cwd, text, stdout, stderr, timeout, check):
        assert text is True
        assert stdout is not None
        assert stderr is not None
        assert check is False
        if "-mode" in command:
            (cwd / "zcu104_timing_summary.rpt").write_text(
                "\n".join(
                    [
                        "setup_worst_slack_ns: 0.125",
                        "hold_worst_slack_ns: 0.025",
                        "pulse_width_worst_slack_ns: 0.500",
                        "setup_failing_endpoints: 0",
                        "hold_failing_endpoints: 0",
                        "pulse_width_failing_endpoints: 0",
                    ]
                ),
                encoding="utf-8",
            )
            (cwd / "zcu104_utilization.rpt").write_text(
                "\n".join(["lut: 449", "ff: 512", "dsp: 0", "bram: 0", "uram: 0"]),
                encoding="utf-8",
            )
            (cwd / "zcu104_drc.rpt").write_text(
                "No board-wrapper blocking critical warnings.\n",
                encoding="utf-8",
            )
            (cwd / "zcu104_clocks.rpt").write_text("clk_pl_0 5.000\n", encoding="utf-8")
            (cwd / "zcu104_implemented_constraints.xdc").write_text(
                "create_clock -period 5.000 -name clk_pl_0 [get_pins u_ps/pl_clk0]\n",
                encoding="utf-8",
            )
            (cwd / "zcu104_post_route.dcp").write_text("fake checkpoint\n", encoding="utf-8")
            (cwd / "zcu104_board_wrapper.bit").write_bytes(b"fake bitstream\n")
            return types.SimpleNamespace(returncode=0, stdout="route ok\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="Vivado v2024.1\n", stderr="")

    monkeypatch.setattr("nl2hdl.llm_kernels.subprocess.run", fake_run)

    report = run_zcu104_board_wrapper_axi_bridge_agent(cfg, tmp_path, run_vivado=True)

    assert report["status"] == "passed"
    assert report["vivado_available"] is True
    assert report["route_completed"] is True
    assert report["bitstream_generated"] is True
    assert report["bitstream_size_bytes"] > 0
    assert report["bitstream_ready_for_bounded_board_wrapper"] is True
    assert report["implementation_evidence_ready"] is True
    assert report["final_board_signoff_still_blocked"] is True
    assert report["static_integration_evidence"]["routed_top_instantiates_generated_bd_wrapper"] is True
    assert report["static_integration_evidence"]["ps_axi_reaches_control_registers"] is True
    assert report["static_integration_evidence"]["ddr_address_map_declared"] is True
    assert report["route_report_analysis"]["timing"]["constraints_met"] is True
    assert report["route_report_analysis"]["clock"]["observed_period_ns"] == 5.0
    assert report["route_report_analysis"]["utilization_budget"]["passes"] is True
    assert report["route_report_analysis"]["reports_present"]["bitstream"] is True
    assert report["failures"] == []
    assert "skill_update_candidate" not in report
    assert not (tmp_path / "board_zcu104_signoff_evidence.json").exists()


def test_zcu104_board_wrapper_agent_prioritizes_clock_gate_failure_over_missing_bitstream(
    monkeypatch, tmp_path: Path
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    def fake_run(command, cwd, text, stdout, stderr, timeout, check):
        assert text is True
        assert stdout is not None
        assert stderr is not None
        assert check is False
        if "-mode" in command:
            (cwd / "zcu104_timing_summary.rpt").write_text(
                "\n".join(
                    [
                        "setup_worst_slack_ns: 0.125",
                        "hold_worst_slack_ns: 0.025",
                        "pulse_width_worst_slack_ns: 0.500",
                        "setup_failing_endpoints: 0",
                        "hold_failing_endpoints: 0",
                        "pulse_width_failing_endpoints: 0",
                    ]
                ),
                encoding="utf-8",
            )
            (cwd / "zcu104_utilization.rpt").write_text(
                "\n".join(["lut: 449", "ff: 512", "dsp: 0", "bram: 0", "uram: 0"]),
                encoding="utf-8",
            )
            (cwd / "zcu104_drc.rpt").write_text("No board-wrapper blocking critical warnings.\n", encoding="utf-8")
            (cwd / "zcu104_clocks.rpt").write_text("clk_pl_0 5.333\n", encoding="utf-8")
            (cwd / "zcu104_implemented_constraints.xdc").write_text(
                "create_clock -period 5.333 -name clk_pl_0 [get_pins u_ps/pl_clk0]\n",
                encoding="utf-8",
            )
            (cwd / "zcu104_post_route.dcp").write_text("fake checkpoint\n", encoding="utf-8")
            return types.SimpleNamespace(returncode=1, stdout="clock gate failed\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="Vivado v2024.1\n", stderr="")

    monkeypatch.setattr("nl2hdl.llm_kernels.subprocess.run", fake_run)

    report = run_zcu104_board_wrapper_axi_bridge_agent(cfg, tmp_path, run_vivado=True)

    assert report["status"] == "failed_board_wrapper_gates"
    assert report["route_completed"] is True
    assert report["route_check_command_passed"] is False
    assert report["bitstream_generated"] is False
    assert "report_clocks period exceeds target clock period" in report["failures"]
    assert "implemented XDC period exceeds target clock period" in report["failures"]
    assert "write_bitstream did not produce zcu104_board_wrapper.bit" not in report["failures"]


def test_board_zcu104_signoff_readiness_stays_blocked_after_pre_signoff_constraints_package(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    build_zcu104_board_shell_constraints_package(cfg, tmp_path)
    dispatch_plan = _minimal_full_execution_dispatch_plan()
    full_execution = _passed_full_execution_readiness_for_board()

    readiness = build_board_zcu104_signoff_readiness_report(dispatch_plan, full_execution, tmp_path)

    assert readiness["status"] == "blocked_by_missing_or_incomplete_board_signoff_evidence"
    assert readiness["safe_to_clear_board_level_zcu104_signoff_blocker"] is False
    assert readiness["evidence_failures"] == ["board_zcu104_signoff_evidence.json not found"]


def test_hdl_subagent_wave_status_collects_passed_implementation_and_verification(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = dispatch_plan["waves"][0]

    for task in projection_wave["implementation_tasks"]:
        evidence_dir = tmp_path / Path(task["expected_evidence_dir"]).name
        _write_minimal_passed_kernel_evidence(evidence_dir)

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    waves = {wave["wave_id"]: wave for wave in status["waves"]}
    assert waves["wave_1_projection_kernels"]["status"] == "ready_for_verification"
    assert waves["wave_1_projection_kernels"]["verification"]["status"] == "missing"
    assert all(result["status"] == "passed" for result in waves["wave_1_projection_kernels"]["task_results"])
    assert waves["wave_1_projection_kernels"]["task_status_counts"] == {"passed": 7}
    assert waves["wave_1_projection_kernels"]["passed_task_count"] == 7
    assert waves["wave_1_projection_kernels"]["missing_task_count"] == 0
    assert "wave_1_projection_kernels" in status["next_dispatchable_waves"]
    assert waves["wave_2_decoder_block"]["status"] == "blocked_by_dependency"

    verification_dir = tmp_path / "verification_results"
    verification_dir.mkdir()
    (verification_dir / "wave_1_projection_kernels__verification.json").write_text(
        json.dumps({"status": "passed", "findings": [{"severity": "P3", "body": "note only"}]}),
        encoding="utf-8",
    )

    verified_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    verified_waves = {wave["wave_id"]: wave for wave in verified_status["waves"]}
    assert verified_waves["wave_1_projection_kernels"]["status"] == "passed"
    assert verified_waves["wave_1_non_gemm_kernels"]["status"] == "ready_to_dispatch"
    assert verified_waves["wave_2_decoder_block"]["status"] == "blocked_by_dependency"
    assert "wave_1_non_gemm_kernels" in verified_status["next_dispatchable_waves"]

    (verification_dir / "wave_1_projection_kernels__verification.json").write_text(
        json.dumps({"status": "passed", "findings": [{"severity": "P1", "body": "blocking issue"}]}),
        encoding="utf-8",
    )
    blocked_verification_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    blocked_waves = {wave["wave_id"]: wave for wave in blocked_verification_status["waves"]}
    assert blocked_waves["wave_1_projection_kernels"]["status"] == "failed_verification_missing_skill_candidate"
    assert blocked_waves["wave_1_projection_kernels"]["verification"]["blocking_finding_count"] == 1
    assert blocked_waves["wave_1_projection_kernels"]["verification"]["skill_update_candidate_complete"] is False


def test_hdl_subagent_wave_status_collects_subagent_result_final_response(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = dispatch_plan["waves"][0]
    first_task = projection_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(evidence_dir)
    (evidence_dir / "subagent_result.json").write_text(
        json.dumps(_minimal_subagent_result(first_task["task_id"])),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    first_result = wave["task_results"][0]
    missing_result = wave["task_results"][1]

    assert first_result["status"] == "passed"
    assert first_result["subagent_result"]["status"] == "passed"
    assert first_result["subagent_result"]["final_response_complete"] is True
    assert first_result["subagent_result"]["missing_fields"] == []
    assert first_result["subagent_result"]["changed_files"] == [
        "nl2hdl/llm_kernels.py",
        "tests/test_llm_kernels.py",
    ]
    assert first_result["subagent_result"]["commands_run"] == ["python3 -m pytest -q tests/test_llm_kernels.py"]
    assert first_result["subagent_result"]["simulation_evidence"] == {"passed": True, "source": "kernel_report.json"}
    assert first_result["subagent_result"]["verilator_evidence"] == {"passed": True, "source": "kernel_report.json"}
    assert first_result["subagent_result"]["vivado_timing_resource_evidence"] == {
        "passed": True,
        "source": "kernel_report.json",
    }
    assert first_result["subagent_result"]["module_ooc_synthesis_evidence"] == {
        "passed": True,
        "source": "module_ooc_synthesis_report.json",
    }
    assert first_result["subagent_result"]["remaining_risks"] == ["bounded fixture only"]
    assert missing_result["subagent_result"]["status"] == "missing"
    assert missing_result["subagent_result"]["final_response_complete"] is False


def test_hdl_subagent_wave_status_requires_module_ooc_before_integration(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = dispatch_plan["waves"][0]
    first_task = projection_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(_minimal_passed_kernel_report()),
        encoding="utf-8",
    )
    (evidence_dir / "subagent_result.json").write_text(
        json.dumps(_minimal_subagent_result(first_task["task_id"])),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    first_result = wave["task_results"][0]
    execution = build_hdl_subagent_execution_manifest(dispatch_plan, status)

    assert first_task["requires_module_ooc_synthesis"] is True
    assert first_task["expected_module_ooc_synthesis_report"].endswith(
        "projection_q_proj_gate/module_ooc_synthesis_report.json"
    )
    assert first_result["status"] == "module_ooc_synthesis_missing"
    assert wave["status"] == "ready_to_dispatch"
    assert "module-level OOC synthesis" in wave["reason"]
    retry_entry = next(entry for entry in execution["spawn_entries"] if entry["task_id"] == first_task["task_id"])
    assert retry_entry["expected_module_ooc_synthesis_report"].endswith(
        "projection_q_proj_gate/module_ooc_synthesis_report.json"
    )
    assert "module_ooc_synthesis_report.json" in retry_entry["codex_spawn_message"]


def test_hdl_subagent_wave_status_retries_underutilized_module_before_integration(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = dispatch_plan["waves"][0]
    first_task = projection_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(evidence_dir, first_task["task_id"])
    ooc_report = _minimal_module_ooc_synthesis_report()
    ooc_report["resource_assessment"] = "underutilized"
    ooc_report["throughput_target_met"] = False
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        json.dumps(ooc_report),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    first_result = wave["task_results"][0]

    assert first_result["status"] == "module_ooc_synthesis_needs_tuning"
    assert first_result["module_ooc_synthesis"]["resource_assessment"] == "underutilized"
    assert wave["status"] == "ready_to_dispatch"
    assert "needs resource/timing tuning" in wave["reason"]


def test_hdl_subagent_wave_status_invalidates_ooc_when_hardware_spec_changes(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = dispatch_plan["waves"][0]
    first_task = projection_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(evidence_dir, first_task["task_id"])
    stale_ooc_report = _minimal_module_ooc_synthesis_report()
    stale_ooc_report["hardware_spec"]["target_clock_mhz"] = 150
    stale_ooc_report["vivado"]["target_clock_mhz"] = 150
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        json.dumps(stale_ooc_report),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    first_result = wave["task_results"][0]
    execution = build_hdl_subagent_execution_manifest(dispatch_plan, status)

    assert first_task["hardware"]["target_clock_mhz"] == 200
    assert first_result["status"] == "module_ooc_synthesis_hardware_mismatch"
    assert first_result["module_ooc_synthesis"]["status"] == "hardware_mismatch"
    assert "hardware_spec.target_clock_mhz mismatch" in first_result["reason"]
    assert wave["status"] == "ready_to_dispatch"
    assert "stale for the active hardware spec" in wave["reason"]
    retry_entry = next(entry for entry in execution["spawn_entries"] if entry["task_id"] == first_task["task_id"])
    assert retry_entry["hardware"]["target_clock_mhz"] == 200


def test_hdl_subagent_execution_manifest_lists_next_codex_spawns(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))

    initial_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    execution = build_hdl_subagent_execution_manifest(dispatch_plan, initial_status)

    assert execution["artifact"] == "hdl_subagent_execution_manifest"
    assert execution["agent_hierarchy"]["parent_agent"] == "single_orchestrator"
    assert execution["agent_hierarchy"]["all_non_parent_workers_are_subagents"] is True
    assert execution["agent_hierarchy"]["subagents_may_spawn_subagents"] is False
    assert execution["agent_hierarchy"]["parent_owns_feedback_loop"] is True
    assert execution["spawn_entry_count"] == 15
    assert execution["implementation_spawn_count"] == 15
    assert execution["verification_spawn_count"] == 0
    assert execution["spawn_batch_count"] == 2
    assert execution["parallel_spawn_allowed"] is True
    assert execution["max_parallel_batch_size"] == 8
    spawn_batches = {batch["wave_id"]: batch for batch in execution["spawn_batches"]}
    assert spawn_batches["wave_1_projection_kernels"]["spawn_kind"] == "implementation_agent"
    assert spawn_batches["wave_1_projection_kernels"]["parallel_spawn_allowed"] is True
    assert spawn_batches["wave_1_projection_kernels"]["entry_count"] == 7
    assert spawn_batches["wave_1_projection_kernels"]["task_ids"] == [
        "projection_q_proj",
        "projection_k_proj",
        "projection_v_proj",
        "projection_o_proj",
        "projection_gate_proj",
        "projection_up_proj",
        "projection_down_proj",
    ]
    assert spawn_batches["wave_1_non_gemm_kernels"]["spawn_kind"] == "implementation_agent"
    assert spawn_batches["wave_1_non_gemm_kernels"]["parallel_spawn_allowed"] is True
    assert spawn_batches["wave_1_non_gemm_kernels"]["entry_count"] == 8
    first_entry = execution["spawn_entries"][0]
    assert first_entry["spawn_kind"] == "implementation_agent"
    assert first_entry["agent_hierarchy_role"] == "subagent"
    assert first_entry["subagent_type"] == "hdl_implementation_subagent"
    assert first_entry["subagent_may_spawn_subagents"] is False
    assert first_entry["parent_feedback_channel"] == "feedback_packet.json"
    assert first_entry["agent"] == "Codex"
    assert first_entry["mode"] == "read_write_hdl_packet"
    assert first_entry["wave_id"] == "wave_1_projection_kernels"
    assert first_entry["task_id"] == "projection_q_proj"
    assert first_entry["prompt_file"] == "subagent_prompts/projection_q_proj__implementation.md"
    assert first_entry["fork_context"] is True
    assert first_entry["expected_subagent_result"] == "build/projection_q_proj_gate/subagent_result.json"
    assert first_entry["module_contract"]["task_id"] == "projection_q_proj"
    assert first_entry["module_contract"]["handshake_ports"] == {
        "start": "start_i",
        "done": "done_o",
    }
    assert first_entry["module_contract"]["parent_boundary"]["parent_must_not_write_hdl"] is True
    assert first_entry["must_self_verify"] is True
    assert first_entry["parent_must_not_write_hdl"] is True
    assert first_entry["final_response_required_fields"] == first_entry["module_contract"][
        "final_response_required_fields"
    ]
    assert first_entry["failure_to_skill_required"] is True
    assert first_entry["codex_spawn_message"].startswith(
        "You are the HDL implementation sub-agent for this single packet."
    )
    assert "subagent_prompts/projection_q_proj__implementation.md" in first_entry["codex_spawn_message"]
    assert "build/projection_q_proj_gate/subagent_result.json" in first_entry["codex_spawn_message"]
    assert "skill_update_candidate" in first_entry["codex_spawn_message"]
    k_entry = next(entry for entry in execution["spawn_entries"] if entry["task_id"] == "projection_k_proj")
    assert k_entry["semantic_op"] == "k_proj"
    assert k_entry["expected_projection_shape"] == {"rows": 16, "cols": 64}
    assert k_entry["packed_int4_bytes"] == 512
    assert k_entry["memory_beats"] == 32
    assert "automatic sub-agent spawning inside package runtime" in execution["does_not_claim"]

    projection_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    for task in projection_wave["implementation_tasks"]:
        evidence_dir = tmp_path / Path(task["expected_evidence_dir"]).name
        _write_minimal_passed_kernel_evidence(evidence_dir)

    ready_for_verification_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    verification_execution = build_hdl_subagent_execution_manifest(dispatch_plan, ready_for_verification_status)

    verification_entries = [
        entry for entry in verification_execution["spawn_entries"] if entry["spawn_kind"] == "verification_agent"
    ]
    assert verification_execution["implementation_spawn_count"] == 8
    assert verification_execution["verification_spawn_count"] == 1
    assert verification_execution["spawn_batch_count"] == 2
    verification_batches = {batch["wave_id"]: batch for batch in verification_execution["spawn_batches"]}
    assert verification_batches["wave_1_projection_kernels"]["spawn_kind"] == "verification_agent"
    assert verification_batches["wave_1_projection_kernels"]["parallel_spawn_allowed"] is False
    assert verification_batches["wave_1_projection_kernels"]["entry_count"] == 1
    assert verification_batches["wave_1_non_gemm_kernels"]["spawn_kind"] == "implementation_agent"
    assert verification_batches["wave_1_non_gemm_kernels"]["parallel_spawn_allowed"] is True
    assert verification_batches["wave_1_non_gemm_kernels"]["entry_count"] == 8
    assert len(verification_entries) == 1
    verification_entry = verification_entries[0]
    assert verification_entry["spawn_kind"] == "verification_agent"
    assert verification_entry["agent_hierarchy_role"] == "subagent"
    assert verification_entry["subagent_type"] == "verification_subagent"
    assert verification_entry["subagent_may_spawn_subagents"] is False
    assert verification_entry["agent"] == "Codex"
    assert verification_entry["mode"] == "read_only"
    assert verification_entry["wave_id"] == "wave_1_projection_kernels"
    assert verification_entry["prompt_file"] == "verification_prompts/wave_1_projection_kernels__verification.md"
    assert verification_entry["fork_context"] is True
    assert verification_entry["verification_report"] == "verification_results/wave_1_projection_kernels__verification.json"
    assert verification_entry["must_not_edit_source_files"] is True
    assert verification_entry["may_write_generated_evidence"] is False
    assert verification_entry["runs_integration_synthesis"] is False
    assert verification_entry["blocking_findings"] == ["P0", "P1", "P2"]
    assert verification_entry["codex_spawn_message"].startswith(
        "You are the Codex verification sub-agent for this wave."
    )
    assert "verification_prompts/wave_1_projection_kernels__verification.md" in verification_entry[
        "codex_spawn_message"
    ]
    assert "verification_results/wave_1_projection_kernels__verification.json" in verification_entry[
        "codex_spawn_message"
    ]
    verification_markdown = build_codex_spawn_instructions(verification_execution)
    assert "## Batch `wave_1_projection_kernels__verification_agent`" in verification_markdown
    assert "verification_prompts/wave_1_projection_kernels__verification.md" in verification_markdown
    assert "## Does Not Claim" in verification_markdown
    assert "automatic sub-agent spawning inside package runtime" in verification_markdown
    assert "The Parent Agent is the only orchestrator" in verification_markdown

    parent_loop = build_parent_feedback_loop_state(dispatch_plan, ready_for_verification_status, verification_execution)
    assert parent_loop["parent_loop_state"]["artifact"] == "parent_loop_state"
    assert parent_loop["parent_loop_state"]["status"] == "ready_to_spawn_subagents"
    assert parent_loop["parent_loop_state"]["next_parent_action"] == "spawn_ready_subagents"
    assert parent_loop["feedback_packet"]["entry_count"] == (
        verification_execution["spawn_entry_count"] + len(verification_execution["blocked_waves"])
    )
    assert parent_loop["retry_plan"]["retry_entry_count"] >= verification_execution["spawn_batch_count"]


def test_hdl_subagent_spawn_ledger_tracks_external_agent_ids(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    execution = build_hdl_subagent_execution_manifest(
        dispatch_plan,
        build_hdl_subagent_wave_status(dispatch_plan, tmp_path),
    )
    q_entry = next(entry for entry in execution["spawn_entries"] if entry["task_id"] == "projection_q_proj")
    assert q_entry["spawn_key"] == "wave_1_projection_kernels::implementation::projection_q_proj"

    ledger = build_hdl_subagent_spawn_ledger(
        execution,
        agent_records={q_entry["spawn_key"]: "agent-123"},
    )
    markdown = build_hdl_subagent_spawn_ledger_markdown(ledger)

    assert ledger["artifact"] == "hdl_subagent_spawn_ledger"
    assert ledger["spawn_entry_count"] == execution["spawn_entry_count"]
    assert ledger["status_counts"]["spawned_waiting_for_evidence"] == 1
    assert ledger["status_counts"]["ready_to_spawn"] == execution["spawn_entry_count"] - 1
    q_record = next(record for record in ledger["records"] if record["task_id"] == "projection_q_proj")
    assert q_record["agent_id"] == "agent-123"
    assert q_record["expected_subagent_result"] == "build/projection_q_proj_gate/subagent_result.json"
    assert q_record["parent_must_not_write_hdl"] is True
    assert "package code spawned Codex agents" in ledger["does_not_claim"]
    assert "agent-123" in markdown
    assert "projection_q_proj" in markdown

    q_evidence_dir = tmp_path / "projection_q_proj_gate"
    _write_minimal_passed_kernel_evidence(q_evidence_dir, "projection_q_proj")
    wave_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    next_execution = build_hdl_subagent_execution_manifest(dispatch_plan, wave_status)
    reconciled = build_hdl_subagent_spawn_ledger(
        next_execution,
        existing_ledger=ledger,
        wave_status=wave_status,
    )
    reconciled_q = next(record for record in reconciled["records"] if record["task_id"] == "projection_q_proj")
    assert reconciled["wave_status_reconciled"] is True
    assert reconciled_q["agent_id"] == "agent-123"
    assert reconciled_q["spawn_status"] == "evidence_passed"
    assert reconciled["status_counts"]["evidence_passed"] == 1


def test_cli_subagents_ledger_writes_spawn_ledger(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    execution = build_hdl_subagent_execution_manifest(
        dispatch_plan,
        build_hdl_subagent_wave_status(dispatch_plan, tmp_path),
    )
    execution_path = tmp_path / "hdl_subagent_execution_manifest.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    q_key = "wave_1_projection_kernels::implementation::projection_q_proj"

    rc = cli_main(
        [
            "subagents",
            "ledger",
            "--execution-manifest",
            str(execution_path),
            "--out",
            str(tmp_path / "ledger_out"),
            "--agent-record",
            f"{q_key}=agent-123",
        ]
    )

    assert rc == 0
    ledger = json.loads((tmp_path / "ledger_out" / "hdl_subagent_spawn_ledger.json").read_text())
    ledger_markdown = (tmp_path / "ledger_out" / "hdl_subagent_spawn_ledger.md").read_text(encoding="utf-8")
    assert ledger["status_counts"]["spawned_waiting_for_evidence"] == 1
    assert next(record for record in ledger["records"] if record["spawn_key"] == q_key)["agent_id"] == "agent-123"
    assert "agent-123" in ledger_markdown

    _write_minimal_passed_kernel_evidence(tmp_path / "projection_q_proj_gate", "projection_q_proj")
    wave_status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    next_execution = build_hdl_subagent_execution_manifest(dispatch_plan, wave_status)
    next_execution_path = tmp_path / "hdl_subagent_execution_manifest_next.json"
    wave_status_path = tmp_path / "hdl_subagent_wave_status.json"
    next_execution_path.write_text(json.dumps(next_execution), encoding="utf-8")
    wave_status_path.write_text(json.dumps(wave_status), encoding="utf-8")

    rc_reconcile = cli_main(
        [
            "subagents",
            "ledger",
            "--execution-manifest",
            str(next_execution_path),
            "--existing-ledger",
            str(tmp_path / "ledger_out" / "hdl_subagent_spawn_ledger.json"),
            "--wave-status",
            str(wave_status_path),
            "--out",
            str(tmp_path / "ledger_reconciled"),
        ]
    )

    assert rc_reconcile == 0
    reconciled = json.loads((tmp_path / "ledger_reconciled" / "hdl_subagent_spawn_ledger.json").read_text())
    reconciled_q = next(record for record in reconciled["records"] if record["spawn_key"] == q_key)
    assert reconciled["wave_status_reconciled"] is True
    assert reconciled_q["agent_id"] == "agent-123"
    assert reconciled_q["spawn_status"] == "evidence_passed"


def test_hdl_subagent_wave_status_blocks_passed_kernel_without_subagent_result(tmp_path: Path):
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_1_single_kernel",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [
                    {
                        "task_id": "bad_evidence_kernel",
                        "expected_evidence_dir": "build/bad_evidence_kernel_gate",
                    }
                ],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_1_single_kernel__verification.md"
                },
            }
        ]
    }
    evidence_dir = tmp_path / "bad_evidence_kernel_gate"
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(_minimal_passed_kernel_report()),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = status["waves"][0]
    execution = build_hdl_subagent_execution_manifest(dispatch_plan, status)

    assert wave["status"] == "incomplete_subagent_result"
    assert wave["task_results"][0]["status"] == "incomplete_subagent_result"
    assert wave["task_results"][0]["subagent_result"]["status"] == "missing"
    assert execution["spawn_entry_count"] == 0
    assert execution["blocked_waves"][0]["next_action"] == "collect_complete_subagent_result_before_verification"
    markdown = build_codex_spawn_instructions(execution)
    assert "## Blocked Waves" in markdown
    assert "collect_complete_subagent_result_before_verification" in markdown


def test_hdl_subagent_wave_status_reports_partial_projection_wave_progress(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    projection_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    first_task = projection_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(evidence_dir)

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_projection_kernels")
    assert wave["status"] == "ready_to_dispatch"
    assert wave["target_scope"] == "bounded_fixture_only"
    assert wave["blocked_target_dependencies"] == [
        "gptq_payload_probe",
        "real_gptq_checkpoint_metadata",
        "real_gptq_weight_layout_preflight",
    ]
    assert wave["task_count"] == 7
    assert wave["task_status_counts"] == {"passed": 1, "missing": 6}
    assert wave["passed_task_count"] == 1
    assert wave["missing_task_count"] == 6
    assert wave["task_results"][0]["task_id"] == "projection_q_proj"
    assert wave["task_results"][0]["status"] == "passed"
    decoder_wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_2_decoder_block")
    assert decoder_wave["status"] == "blocked_by_dependency"

    second_task = projection_wave["implementation_tasks"][1]
    second_evidence_dir = tmp_path / Path(second_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(second_evidence_dir)
    status_after_second_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_second_task = next(
        wave for wave in status_after_second_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_second_task["status"] == "ready_to_dispatch"
    assert wave_after_second_task["task_status_counts"] == {"passed": 2, "missing": 5}
    assert wave_after_second_task["passed_task_count"] == 2
    assert wave_after_second_task["missing_task_count"] == 5
    assert wave_after_second_task["task_results"][1]["task_id"] == "projection_k_proj"
    assert wave_after_second_task["task_results"][1]["status"] == "passed"

    third_task = projection_wave["implementation_tasks"][2]
    third_evidence_dir = tmp_path / Path(third_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(third_evidence_dir)
    status_after_third_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_third_task = next(
        wave for wave in status_after_third_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_third_task["status"] == "ready_to_dispatch"
    assert wave_after_third_task["task_status_counts"] == {"passed": 3, "missing": 4}
    assert wave_after_third_task["passed_task_count"] == 3
    assert wave_after_third_task["missing_task_count"] == 4
    assert wave_after_third_task["task_results"][2]["task_id"] == "projection_v_proj"
    assert wave_after_third_task["task_results"][2]["status"] == "passed"

    fourth_task = projection_wave["implementation_tasks"][3]
    fourth_evidence_dir = tmp_path / Path(fourth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(fourth_evidence_dir)
    status_after_fourth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_fourth_task = next(
        wave for wave in status_after_fourth_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_fourth_task["status"] == "ready_to_dispatch"
    assert wave_after_fourth_task["task_status_counts"] == {"passed": 4, "missing": 3}
    assert wave_after_fourth_task["passed_task_count"] == 4
    assert wave_after_fourth_task["missing_task_count"] == 3
    assert wave_after_fourth_task["task_results"][3]["task_id"] == "projection_o_proj"
    assert wave_after_fourth_task["task_results"][3]["status"] == "passed"

    fifth_task = projection_wave["implementation_tasks"][4]
    fifth_evidence_dir = tmp_path / Path(fifth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(fifth_evidence_dir)
    status_after_fifth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_fifth_task = next(
        wave for wave in status_after_fifth_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_fifth_task["status"] == "ready_to_dispatch"
    assert wave_after_fifth_task["task_status_counts"] == {"passed": 5, "missing": 2}
    assert wave_after_fifth_task["passed_task_count"] == 5
    assert wave_after_fifth_task["missing_task_count"] == 2
    assert wave_after_fifth_task["task_results"][4]["task_id"] == "projection_gate_proj"
    assert wave_after_fifth_task["task_results"][4]["status"] == "passed"

    sixth_task = projection_wave["implementation_tasks"][5]
    sixth_evidence_dir = tmp_path / Path(sixth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(sixth_evidence_dir)
    status_after_sixth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_sixth_task = next(
        wave for wave in status_after_sixth_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_sixth_task["status"] == "ready_to_dispatch"
    assert wave_after_sixth_task["task_status_counts"] == {"passed": 6, "missing": 1}
    assert wave_after_sixth_task["passed_task_count"] == 6
    assert wave_after_sixth_task["missing_task_count"] == 1
    assert wave_after_sixth_task["task_results"][5]["task_id"] == "projection_up_proj"
    assert wave_after_sixth_task["task_results"][5]["status"] == "passed"

    seventh_task = projection_wave["implementation_tasks"][6]
    seventh_evidence_dir = tmp_path / Path(seventh_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(seventh_evidence_dir)
    status_after_seventh_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_seventh_task = next(
        wave for wave in status_after_seventh_task["waves"] if wave["wave_id"] == "wave_1_projection_kernels"
    )
    assert wave_after_seventh_task["status"] == "ready_for_verification"
    assert wave_after_seventh_task["task_status_counts"] == {"passed": 7}
    assert wave_after_seventh_task["passed_task_count"] == 7
    assert wave_after_seventh_task["missing_task_count"] == 0
    assert wave_after_seventh_task["task_results"][6]["task_id"] == "projection_down_proj"
    assert wave_after_seventh_task["task_results"][6]["status"] == "passed"


def test_hdl_subagent_wave_status_reports_partial_wave_progress(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(graph, cfg)
    dispatch_plan = build_hdl_subagent_dispatch_plan(build_hdl_subagent_packets(manifest))
    non_gemm_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels")
    first_task = non_gemm_wave["implementation_tasks"][0]
    evidence_dir = tmp_path / Path(first_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(evidence_dir)

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = next(wave for wave in status["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels")
    assert wave["status"] == "ready_to_dispatch"
    assert wave["task_count"] == 8
    assert wave["task_status_counts"] == {"passed": 1, "missing": 7}
    assert wave["passed_task_count"] == 1
    assert wave["missing_task_count"] == 7
    assert wave["task_results"][0]["task_id"] == "non_gemm_input_layernorm"
    assert wave["task_results"][0]["status"] == "passed"

    second_task = non_gemm_wave["implementation_tasks"][1]
    second_evidence_dir = tmp_path / Path(second_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(second_evidence_dir)
    status_after_second_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_second_task = next(
        wave for wave in status_after_second_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_second_task["status"] == "ready_to_dispatch"
    assert wave_after_second_task["task_status_counts"] == {"passed": 2, "missing": 6}
    assert wave_after_second_task["passed_task_count"] == 2
    assert wave_after_second_task["missing_task_count"] == 6

    third_task = non_gemm_wave["implementation_tasks"][2]
    third_evidence_dir = tmp_path / Path(third_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(third_evidence_dir)
    status_after_third_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_third_task = next(
        wave for wave in status_after_third_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_third_task["status"] == "ready_to_dispatch"
    assert wave_after_third_task["task_status_counts"] == {"passed": 3, "missing": 5}
    assert wave_after_third_task["passed_task_count"] == 3
    assert wave_after_third_task["missing_task_count"] == 5

    fourth_task = non_gemm_wave["implementation_tasks"][3]
    fourth_evidence_dir = tmp_path / Path(fourth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(fourth_evidence_dir)
    status_after_fourth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_fourth_task = next(
        wave for wave in status_after_fourth_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_fourth_task["status"] == "ready_to_dispatch"
    assert wave_after_fourth_task["task_status_counts"] == {"passed": 4, "missing": 4}
    assert wave_after_fourth_task["passed_task_count"] == 4
    assert wave_after_fourth_task["missing_task_count"] == 4

    fifth_task = non_gemm_wave["implementation_tasks"][4]
    fifth_evidence_dir = tmp_path / Path(fifth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(fifth_evidence_dir)
    status_after_fifth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_fifth_task = next(
        wave for wave in status_after_fifth_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_fifth_task["status"] == "ready_to_dispatch"
    assert wave_after_fifth_task["task_status_counts"] == {"passed": 5, "missing": 3}
    assert wave_after_fifth_task["passed_task_count"] == 5
    assert wave_after_fifth_task["missing_task_count"] == 3

    sixth_task = non_gemm_wave["implementation_tasks"][5]
    sixth_evidence_dir = tmp_path / Path(sixth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(sixth_evidence_dir)
    status_after_sixth_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_sixth_task = next(
        wave for wave in status_after_sixth_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_sixth_task["status"] == "ready_to_dispatch"
    assert wave_after_sixth_task["task_status_counts"] == {"passed": 6, "missing": 2}
    assert wave_after_sixth_task["passed_task_count"] == 6
    assert wave_after_sixth_task["missing_task_count"] == 2

    seventh_task = non_gemm_wave["implementation_tasks"][6]
    seventh_evidence_dir = tmp_path / Path(seventh_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(seventh_evidence_dir)
    status_after_seventh_task = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_seventh_task = next(
        wave for wave in status_after_seventh_task["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_seventh_task["status"] == "ready_to_dispatch"
    assert wave_after_seventh_task["task_status_counts"] == {"passed": 7, "missing": 1}
    assert wave_after_seventh_task["passed_task_count"] == 7
    assert wave_after_seventh_task["missing_task_count"] == 1

    eighth_task = non_gemm_wave["implementation_tasks"][7]
    eighth_evidence_dir = tmp_path / Path(eighth_task["expected_evidence_dir"]).name
    _write_minimal_passed_kernel_evidence(eighth_evidence_dir)
    status_after_all_tasks = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_after_all_tasks = next(
        wave for wave in status_after_all_tasks["waves"] if wave["wave_id"] == "wave_1_non_gemm_kernels"
    )
    assert wave_after_all_tasks["status"] == "ready_for_verification"
    assert wave_after_all_tasks["task_status_counts"] == {"passed": 8}
    assert wave_after_all_tasks["passed_task_count"] == 8
    assert wave_after_all_tasks["missing_task_count"] == 0


def test_hdl_subagent_wave_status_requires_skill_candidate_before_retry(tmp_path: Path):
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_1_single_kernel",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [
                    {
                        "task_id": "bad_kernel",
                        "expected_evidence_dir": "build/bad_kernel_gate",
                    }
                ],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_1_single_kernel__verification.md"
                },
            }
        ]
    }
    evidence_dir = tmp_path / "bad_kernel_gate"
    evidence_dir.mkdir()
    failed_report = {
        "status": "failed",
        "simulation": {"passed": False},
        "verilator": {"passed": True},
        "contract_gate": {"verilator_enforced": True},
    }
    (evidence_dir / "kernel_report.json").write_text(json.dumps(failed_report), encoding="utf-8")

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = status["waves"][0]
    assert wave["status"] == "failed_missing_skill_candidate"
    assert wave["task_results"][0]["skill_update_candidate_complete"] is False

    failed_report["skill_update_candidate"] = {
        "failing_command": "python3 -m nl2hdl agent --mode kernel",
        "symptom": "simulation mismatch",
        "root_cause_hypothesis": "start/done protocol violated",
        "prevention_rule": "hold start_i until done_o is observed, then release before retry",
        "minimal_regression_check": "single-kernel handshake simulation",
    }
    (evidence_dir / "kernel_report.json").write_text(json.dumps(failed_report), encoding="utf-8")

    status_with_candidate = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave_with_candidate = status_with_candidate["waves"][0]
    assert wave_with_candidate["status"] == "failed_waiting_for_skill_update"
    assert wave_with_candidate["task_results"][0]["status"] == "failed_with_skill_candidate"
    assert wave_with_candidate["task_results"][0]["skill_update_candidate_complete"] is True

    failed_report["skill_update_candidate"] = {
        "artifact": "skill_update_candidate_template",
        "candidate": {
            "failing_command": "python3 -m nl2hdl agent --mode kernel",
            "symptom": "simulation mismatch",
            "root_cause_hypothesis": "start/done protocol violated",
            "prevention_rule": "hold start_i until done_o is observed, then release before retry",
            "minimal_regression_check": "single-kernel handshake simulation",
        },
    }
    (evidence_dir / "kernel_report.json").write_text(json.dumps(failed_report), encoding="utf-8")

    status_with_nested_candidate = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    nested_candidate_wave = status_with_nested_candidate["waves"][0]
    assert nested_candidate_wave["status"] == "failed_waiting_for_skill_update"
    assert nested_candidate_wave["task_results"][0]["skill_update_candidate_complete"] is True

    failed_report_without_candidate = {
        "status": "failed",
        "simulation": {"passed": False},
        "verilator": {"passed": True},
        "contract_gate": {"verilator_enforced": True},
    }
    subagent_result_with_candidate = {
        **_minimal_subagent_result("bad_kernel"),
        "skill_update_candidate": {
            "failing_command": "python3 -m nl2hdl agent --mode kernel",
            "symptom": "simulation mismatch",
            "root_cause_hypothesis": "start/done protocol violated",
            "prevention_rule": "hold start_i until done_o is observed, then release before retry",
            "minimal_regression_check": "single-kernel handshake simulation",
        },
    }
    (evidence_dir / "kernel_report.json").write_text(json.dumps(failed_report_without_candidate), encoding="utf-8")
    (evidence_dir / "subagent_result.json").write_text(json.dumps(subagent_result_with_candidate), encoding="utf-8")

    status_with_result_candidate = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    result_candidate_wave = status_with_result_candidate["waves"][0]
    assert result_candidate_wave["status"] == "failed_waiting_for_skill_update"
    assert result_candidate_wave["task_results"][0]["status"] == "failed_with_skill_candidate"
    assert result_candidate_wave["task_results"][0]["skill_update_candidate_complete"] is True
    assert result_candidate_wave["task_results"][0]["subagent_result"]["skill_update_candidate_complete"] is True


def test_hdl_subagent_skill_draft_collects_failed_candidates(tmp_path: Path):
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_1_single_kernel",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [
                    {
                        "task_id": "bad_kernel",
                        "agent_role": "projection_kernel_agent",
                        "current_regression_kernel": "projection_tile",
                        "expected_evidence_dir": "build/bad_kernel_gate",
                    }
                ],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_1_single_kernel__verification.md"
                },
            }
        ]
    }
    evidence_dir = tmp_path / "bad_kernel_gate"
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "simulation": {"passed": False},
                "verilator": {"passed": True},
                "contract_gate": {"verilator_enforced": True},
                "skill_update_candidate": {
                    "failing_command": "python3 -m nl2hdl agent --mode kernel --kernel projection_tile",
                    "symptom": "simulation mismatch at done_o",
                    "root_cause_hypothesis": "done_o was pulsed for one cycle instead of held until start_i drops",
                    "prevention_rule": "require a hold-done state in all start/done HDL fixtures",
                    "minimal_regression_check": "handshake simulation with start_i held after completion",
                },
            }
        ),
        encoding="utf-8",
    )

    draft = build_hdl_subagent_skill_update_draft(dispatch_plan, tmp_path, target_skill="hdl-kernel-contract-gates")
    markdown = build_skill_update_draft_markdown(draft)

    assert draft["status"] == "skill_update_required"
    assert draft["candidate_count"] == 1
    assert draft["wave_status_summary"]["failed_waiting_for_skill_update_count"] == 1
    assert draft["candidates"][0]["task_id"] == "bad_kernel"
    assert draft["candidates"][0]["target_skill"] == "hdl-kernel-contract-gates"
    assert draft["candidates"][0]["candidate"]["prevention_rule"] == (
        "require a hold-done state in all start/done HDL fixtures"
    )
    assert "Suggested SKILL.md rule" in markdown
    assert "require a hold-done state" in markdown
    assert "failed HDL gate is fixed" in markdown


def test_hdl_subagent_skill_draft_collects_failed_verification_candidates(tmp_path: Path):
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_1_single_kernel",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [
                    {
                        "task_id": "ok_kernel",
                        "agent_role": "projection_kernel_agent",
                        "current_regression_kernel": "projection_tile",
                        "expected_evidence_dir": "build/ok_kernel_gate",
                    }
                ],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_1_single_kernel__verification.md"
                },
            }
        ]
    }
    evidence_dir = tmp_path / "ok_kernel_gate"
    _write_minimal_passed_kernel_evidence(evidence_dir)
    verification_dir = tmp_path / "verification_results"
    verification_dir.mkdir()
    (verification_dir / "wave_1_single_kernel__verification.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "findings": [
                    {
                        "severity": "P1",
                        "title": "Missing contract proof",
                        "body": "Kernel report passed but did not prove done_o hold behavior.",
                        "skill_update_candidate": {
                            "failing_command": "python3 -m nl2hdl agent --mode kernel --kernel projection_tile",
                            "symptom": "read-only verification found missing done_o hold evidence",
                            "root_cause_hypothesis": "sub-agent treated simulation pass as enough without contract trace",
                            "prevention_rule": "require contract traces for done_o hold in kernel_report.json",
                            "minimal_regression_check": "verification fixture with P1 missing-contract finding",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = build_hdl_subagent_wave_status(dispatch_plan, tmp_path)
    wave = status["waves"][0]
    assert wave["status"] == "failed_verification_waiting_for_skill_update"
    assert wave["verification"]["skill_update_candidate_complete"] is True
    assert wave["verification"]["skill_update_candidate_count"] == 1

    draft = build_hdl_subagent_skill_update_draft(dispatch_plan, tmp_path, target_skill="hdl-kernel-contract-gates")
    markdown = build_skill_update_draft_markdown(draft)

    assert draft["status"] == "skill_update_required"
    assert draft["candidate_count"] == 1
    assert draft["candidates"][0]["agent_role"] == "read_only_verification_agent"
    assert draft["candidates"][0]["candidate_source"] == "verification_finding"
    assert draft["candidates"][0]["verification_finding"]["severity"] == "P1"
    assert draft["candidates"][0]["candidate"]["prevention_rule"] == (
        "require contract traces for done_o hold in kernel_report.json"
    )
    assert "read-only verification found missing done_o hold evidence" in markdown


def test_cli_subagents_skill_draft_writes_candidate_artifacts(tmp_path: Path):
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_1_single_kernel",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [
                    {
                        "task_id": "bad_kernel",
                        "agent_role": "non_gemm_kernel_agent",
                        "current_regression_kernel": "rmsnorm",
                        "expected_evidence_dir": "build/bad_kernel_gate",
                    }
                ],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_1_single_kernel__verification.md"
                },
            }
        ]
    }
    dispatch_plan_path = tmp_path / "hdl_subagent_dispatch_plan.json"
    dispatch_plan_path.write_text(json.dumps(dispatch_plan), encoding="utf-8")
    evidence_dir = tmp_path / "bad_kernel_gate"
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "simulation": {"passed": False},
                "verilator": {"passed": True},
                "contract_gate": {"verilator_enforced": True},
            }
        ),
        encoding="utf-8",
    )

    rc_without_candidate = cli_main(
        [
            "subagents",
            "skill-draft",
            "--dispatch-plan",
            str(dispatch_plan_path),
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "skill_out_missing"),
        ]
    )
    assert rc_without_candidate == 1
    missing_draft = json.loads((tmp_path / "skill_out_missing" / "skill_update_candidates.json").read_text())
    assert missing_draft["status"] == "no_complete_skill_update_candidates"

    (evidence_dir / "subagent_result.json").write_text(
        json.dumps(
            {
                **_minimal_subagent_result("bad_kernel"),
                "skill_update_candidate": {
                    "artifact": "skill_update_candidate_template",
                    "candidate": {
                        "failing_command": "python3 -m nl2hdl agent --mode kernel --kernel rmsnorm",
                        "symptom": "RMSNorm reference mismatch",
                        "root_cause_hypothesis": "fixed-point reciprocal square-root rounding was unbounded",
                        "prevention_rule": "pin RMSNorm approximation tables with explicit golden-vector tolerances",
                        "minimal_regression_check": "rmsnorm kernel golden-vector simulation",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rc = cli_main(
        [
            "subagents",
            "skill-draft",
            "--dispatch-plan",
            str(dispatch_plan_path),
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "skill_out"),
            "--target-skill",
            "hdl-kernel-contract-gates",
        ]
    )

    assert rc == 0
    draft = json.loads((tmp_path / "skill_out" / "skill_update_candidates.json").read_text())
    draft_markdown = (tmp_path / "skill_out" / "skill_update_draft.md").read_text(encoding="utf-8")
    assert draft["status"] == "skill_update_required"
    assert draft["candidate_count"] == 1
    assert draft["candidates"][0]["candidate"]["minimal_regression_check"] == "rmsnorm kernel golden-vector simulation"
    assert "pin RMSNorm approximation tables" in draft_markdown


def test_cli_subagents_skill_draft_collects_target_evidence_candidate(tmp_path: Path):
    dispatch_plan = {
        "waves": [],
        "model": {"name": "meta-llama/Llama-3.2-1B"},
        "hardware": {"board": "ZCU104"},
        "optimization": {"quantization": "int4_gptq"},
    }
    dispatch_plan_path = tmp_path / "hdl_subagent_dispatch_plan.json"
    dispatch_plan_path.write_text(json.dumps(dispatch_plan), encoding="utf-8")
    evidence_dir = tmp_path / "full_target_llama_accelerator_gate"
    evidence_dir.mkdir()
    (evidence_dir / "subagent_result.json").write_text(
        json.dumps(
            {
                "artifact": "full_model_target_rtl_generator_subagent_result",
                "task_id": "full_model_target_rtl_generator",
                "status": "blocked",
                "changed_files": [str(evidence_dir / "subagent_result.json")],
                "commands_run": ["python3 -m nl2hdl subagents status ..."],
                "evidence_paths": {"subagent_result": str(evidence_dir / "subagent_result.json")},
                "remaining_risks": ["target-scale child prerequisites are missing"],
                "skill_update_candidate": {
                    "failing_command": "python3 -m nl2hdl subagents status ...",
                    "symptom": "full-model RTL generator blocked on fixture child coverage",
                    "root_cause_hypothesis": "target-scale projection and non-GEMM child packets are not yet available",
                    "prevention_rule": "queue target-scale child packet generators before retrying full_model_target_rtl_generator",
                    "minimal_regression_check": "skill-draft surfaces target-evidence skill_update_candidate",
                },
            }
        ),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            "subagents",
            "skill-draft",
            "--dispatch-plan",
            str(dispatch_plan_path),
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "skill_out"),
            "--target-skill",
            "multi-agent-hdl-generation",
        ]
    )

    assert rc == 0
    draft = json.loads((tmp_path / "skill_out" / "skill_update_candidates.json").read_text())
    assert draft["status"] == "skill_update_required"
    assert draft["candidate_count"] == 1
    assert draft["candidates"][0]["wave_id"] == "target_evidence"
    assert draft["candidates"][0]["task_id"] == "full_model_target_rtl_generator"
    assert draft["candidates"][0]["candidate_source"] == "target_evidence"
    assert (
        draft["candidates"][0]["candidate"]["prevention_rule"]
        == "queue target-scale child packet generators before retrying full_model_target_rtl_generator"
    )


def test_hdl_task_manifest_unblocks_real_gptq_metadata_when_parsed(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(
        graph,
        cfg,
        {
            "status": "parsed",
            "bits": 4,
            "group_size": 128,
            "projection_metadata": [
                {"name": "q_proj", "has_qweight": True, "has_qzeros": True, "has_scales": True},
                {"name": "k_proj", "has_qweight": True, "has_qzeros": True, "has_scales": True},
            ],
            "quantized_projection_metadata_count": 2,
            "complete_gptq_projection_metadata_count": 2,
            "projection_metadata_count": 2,
        },
    )
    assert manifest["gptq_checkpoint_metadata"]["status"] == "parsed"
    assert manifest["gptq_checkpoint_metadata"]["bits"] == 4
    assert manifest["gptq_checkpoint_metadata"]["group_size"] == 128
    assert manifest["gptq_checkpoint_metadata"]["projection_metadata_count"] == 2
    assert manifest["gptq_checkpoint_metadata"]["quantized_projection_metadata_count"] == 2
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 2
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_gptq_weight_layout_preflight",
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }


def test_hdl_task_manifest_unblocks_gptq_weight_layout_when_headers_match_target(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    projection_metadata = []
    for name in graph["partition"]["gemm"]:
        shape = graph["projection_shapes"][name]
        qweight_key = f"model.layers.0.{name}.qweight"
        qzeros_key = f"model.layers.0.{name}.qzeros"
        scales_key = f"model.layers.0.{name}.scales"
        qweight_byte_count = math.ceil(shape["rows"] * shape["cols"] / 2)
        qweight_byte_offset = 3 if name == "q_proj" else 0
        group_count = math.ceil(shape["cols"] / 128)
        qweight_shape = [math.ceil(shape["cols"] / 8), shape["rows"]]
        qzeros_shape = [group_count, math.ceil(shape["rows"] / 8)]
        scales_shape = [group_count, shape["rows"]]
        projection_metadata.append(
            {
                "name": name,
                "has_qweight": True,
                "has_qzeros": True,
                "has_scales": True,
                "keys": {
                    "qweight": [qweight_key],
                    "qzeros": [qzeros_key],
                    "scales": [scales_key],
                    "g_idx": [],
                    "other": [],
                },
                "tensor_summaries": {
                    qweight_key: {
                        "dtype": "I32",
                        "shape": qweight_shape,
                        "data_offsets": [qweight_byte_offset, qweight_byte_offset + qweight_byte_count],
                        "byte_count": qweight_byte_count,
                        "metadata_status": "header_only_no_tensor_payload",
                    },
                    qzeros_key: {
                        "dtype": "I32",
                        "shape": qzeros_shape,
                        "byte_count": qzeros_shape[0] * qzeros_shape[1] * 4,
                        "metadata_status": "header_only_no_tensor_payload",
                    },
                    scales_key: {
                        "dtype": "F16",
                        "shape": scales_shape,
                        "byte_count": scales_shape[0] * scales_shape[1] * 2,
                        "metadata_status": "header_only_no_tensor_payload",
                    },
                },
            }
        )
    gptq_metadata = {
        "status": "parsed",
        "bits": 4,
        "group_size": 128,
        "projection_metadata": projection_metadata,
        "quantized_projection_metadata_count": 7,
        "complete_gptq_projection_metadata_count": 7,
        "projection_metadata_count": 7,
        "tensor_summary_count": 21,
        "tensor_summary_source": "safetensors_header",
    }

    preflight = build_gptq_weight_layout_preflight(graph, gptq_metadata)
    manifest = build_hdl_task_manifest(graph, cfg, gptq_metadata)
    packets = build_hdl_subagent_packets(manifest)
    dispatch_plan = build_hdl_subagent_dispatch_plan(packets)
    projection_wave = next(wave for wave in dispatch_plan["waves"] if wave["wave_id"] == "wave_1_projection_kernels")

    assert preflight["status"] == "passed"
    assert preflight["target_compatible_projection_count"] == 7
    assert manifest["gptq_checkpoint_metadata"]["weight_layout_preflight_status"] == "passed"
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    assert q_task["gptq_weight_layout_preflight"]["status"] == "target_layout_compatible"
    assert q_task["gptq_weight_layout_preflight"]["target_layout_compatible"] is True
    assert q_task["target_checkpoint_layout_dependency"] == "satisfied_by_header_preflight"
    assert q_task["checkpoint_tensor_sources"]["qweight"]["key"] == "model.layers.0.q_proj.qweight"
    assert q_task["target_weight_stream_plan"]["stream_plan_valid"] is True
    assert q_task["target_weight_stream_plan"]["qweight_key"] == "model.layers.0.q_proj.qweight"
    assert q_task["target_weight_stream_plan"]["observed_memory_beats"] == q_task["memory_beats"]
    assert q_task["target_weight_stream_plan"]["qweight_byte_offset"] == 3
    assert q_task["target_weight_stream_plan"]["qweight_byte_end_exclusive"] == 2051
    assert q_task["target_weight_stream_plan"]["request_byte_addr"] == 0
    assert q_task["target_weight_stream_plan"]["request_beat_count"] == 129
    assert q_task["target_weight_stream_plan"]["first_beat_byte_offset"] == 3
    assert q_task["target_weight_stream_plan"]["last_beat_valid_bytes"] == 3
    assert q_task["target_weight_stream_plan"]["request_covers_unaligned_qweight_range"] is True
    assert q_task["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert projection_wave["target_scope"] == "bounded_fixture_only"
    assert "gptq_payload_probe" in projection_wave["blocked_target_dependencies"]
    assert not any(task["task_id"] == "real_gptq_checkpoint_metadata" for task in manifest["blocked_target_tasks"])
    assert not any(task["task_id"] == "real_gptq_weight_layout_preflight" for task in manifest["blocked_target_tasks"])
    assert {task["task_id"] for task in manifest["blocked_target_tasks"]} == {
        "real_gptq_payload_probe",
        "full_llama_model_execution",
        "board_level_zcu104_signoff",
    }


def test_hdl_task_manifest_unblocks_gptq_weight_layout_from_indexed_safetensors_headers(monkeypatch, tmp_path: Path):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    weight_map = {}
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in graph["partition"]["gemm"]:
        shape = graph["projection_shapes"][name]
        group_count = math.ceil(shape["cols"] / 128)
        qweight_shape = [math.ceil(shape["cols"] / 8), shape["rows"]]
        qzeros_shape = [group_count, math.ceil(shape["rows"] / 8)]
        scales_shape = [group_count, shape["rows"]]
        entries = [
            (f"model.layers.0.{name}.qweight", "I32", qweight_shape, math.ceil(shape["rows"] * shape["cols"] / 2)),
            (f"model.layers.0.{name}.qzeros", "I32", qzeros_shape, qzeros_shape[0] * qzeros_shape[1] * 4),
            (f"model.layers.0.{name}.scales", "F16", scales_shape, scales_shape[0] * scales_shape[1] * 2),
        ]
        for key, dtype, tensor_shape, byte_count in entries:
            weight_map[key] = "model-00001.safetensors"
            header[key] = {
                "dtype": dtype,
                "shape": tensor_shape,
                "data_offsets": [offset, offset + byte_count],
            }
            offset += byte_count
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    raw = json.dumps(header).encode("utf-8")
    (tmp_path / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * offset))

    gptq_metadata = inspect_gptq_checkpoint_metadata(str(tmp_path))
    preflight = build_gptq_weight_layout_preflight(graph, gptq_metadata)
    manifest = build_hdl_task_manifest(graph, cfg, gptq_metadata)

    assert gptq_metadata["tensor_key_source"] == "weight_index"
    assert gptq_metadata["tensor_summary_source"] == "safetensors_header_from_weight_index"
    assert gptq_metadata["tensor_summary_count"] == 21
    assert preflight["status"] == "passed"
    assert preflight["target_compatible_projection_count"] == 7
    assert manifest["gptq_checkpoint_metadata"]["weight_layout_preflight_status"] == "passed"
    q_task = next(task for task in manifest["projection_tasks"] if task["semantic_op"] == "q_proj")
    assert q_task["checkpoint_tensor_sources"]["qweight"]["file"] == "model-00001.safetensors"
    assert q_task["checkpoint_tensor_sources"]["qweight"]["key"] == "model.layers.0.q_proj.qweight"
    assert q_task["checkpoint_tensor_sources"]["qweight"]["data_offsets"] == [0, 2048]
    assert q_task["target_weight_stream_plan"]["qweight_file"] == "model-00001.safetensors"
    assert q_task["target_weight_stream_plan"]["qweight_byte_offset"] == 0
    assert q_task["target_weight_stream_plan"]["qweight_byte_count"] == 2048
    assert q_task["target_weight_stream_plan"]["observed_memory_beats"] == q_task["memory_beats"]
    assert q_task["target_weight_stream_plan"]["request_byte_addr"] == 0
    assert q_task["target_weight_stream_plan"]["request_beat_count"] == 128
    assert q_task["target_weight_stream_plan"]["first_beat_byte_offset"] == 0
    assert q_task["target_weight_stream_plan"]["last_beat_valid_bytes"] == 16
    assert q_task["target_weight_stream_plan"]["request_covers_unaligned_qweight_range"] is True
    assert q_task["target_weight_stream_plan"]["stream_plan_valid"] is True
    assert not any(task["task_id"] == "real_gptq_weight_layout_preflight" for task in manifest["blocked_target_tasks"])


def test_hdl_task_manifest_keeps_gptq_metadata_blocked_without_quantized_projection_keys(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(
        graph,
        cfg,
        {
            "status": "parsed",
            "bits": 4,
            "group_size": 128,
            "projection_metadata": [{"name": "q_proj", "has_qweight": False}],
            "quantized_projection_metadata_count": 0,
            "complete_gptq_projection_metadata_count": 0,
            "projection_metadata_count": 1,
        },
    )

    assert manifest["gptq_checkpoint_metadata"]["status"] == "parsed"
    assert manifest["gptq_checkpoint_metadata"]["projection_metadata_count"] == 0
    assert manifest["gptq_checkpoint_metadata"]["quantized_projection_metadata_count"] == 0
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 0
    assert manifest["gptq_checkpoint_metadata"]["raw_projection_key_count"] == 1
    real_metadata_block = next(
        task for task in manifest["blocked_target_tasks"] if task["task_id"] == "real_gptq_checkpoint_metadata"
    )
    assert real_metadata_block["metadata_status"] == "parsed"
    assert real_metadata_block["quantized_projection_metadata_count"] == 0
    assert real_metadata_block["complete_gptq_projection_metadata_count"] == 0
    assert "no complete qweight/qzeros/scales projection tensor metadata" in real_metadata_block["reason"]


def test_hdl_task_manifest_keeps_gptq_metadata_blocked_without_scales_and_qzeros(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(
        graph,
        cfg,
        {
            "status": "parsed",
            "bits": 4,
            "group_size": 128,
            "projection_metadata": [{"name": "q_proj", "has_qweight": True, "has_qzeros": False, "has_scales": False}],
            "quantized_projection_metadata_count": 1,
            "complete_gptq_projection_metadata_count": 0,
            "projection_metadata_count": 1,
        },
    )

    assert manifest["gptq_checkpoint_metadata"]["projection_metadata_count"] == 1
    assert manifest["gptq_checkpoint_metadata"]["quantized_projection_metadata_count"] == 1
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 0
    real_metadata_block = next(
        task for task in manifest["blocked_target_tasks"] if task["task_id"] == "real_gptq_checkpoint_metadata"
    )
    assert real_metadata_block["quantized_projection_metadata_count"] == 1
    assert real_metadata_block["complete_gptq_projection_metadata_count"] == 0


def test_hdl_task_manifest_keeps_gptq_metadata_blocked_when_complete_count_missing(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, local_files_only: bool):
            return types.SimpleNamespace(
                model_type="llama",
                hidden_size=64,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
            )

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("local-llama-fixture", cfg)
    manifest = build_hdl_task_manifest(
        graph,
        cfg,
        {
            "status": "parsed",
            "bits": 4,
            "group_size": 128,
            "projection_metadata": [{"name": "q_proj", "has_qweight": True}],
            "quantized_projection_metadata_count": 1,
            "projection_metadata_count": 1,
        },
    )

    assert manifest["gptq_checkpoint_metadata"]["quantized_projection_metadata_count"] == 1
    assert manifest["gptq_checkpoint_metadata"]["complete_gptq_projection_metadata_count"] == 0
    real_metadata_block = next(
        task for task in manifest["blocked_target_tasks"] if task["task_id"] == "real_gptq_checkpoint_metadata"
    )
    assert real_metadata_block["complete_gptq_projection_metadata_count"] == 0


def test_emit_int4_unpack_kernel_simulates(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("int4_unpack", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS int4_unpack" in result.simulation["output"]
    sv_text = (tmp_path / "int4_unpack.sv").read_text(encoding="utf-8")
    assert "input  logic                         aclk" in sv_text
    assert "input  logic                         aresetn" in sv_text
    assert "input  logic                         start_i" in sv_text
    assert "output logic                         done_o" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "synthetic_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_gptq_dequant_kernel_simulates_and_reports_numeric_policy(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("gptq_dequant", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS gptq_dequant" in result.simulation["output"]
    assert (tmp_path / "gptq_dequant_golden.json").exists()
    sv_text = (tmp_path / "gptq_dequant.sv").read_text(encoding="utf-8")
    assert "input  logic                              aclk" in sv_text
    assert "input  logic                              aresetn" in sv_text
    assert "input  logic                              start_i" in sv_text
    assert "output logic                              done_o" in sv_text
    assert "packed_i[idx*4 +: 4]" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "synthetic_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"] == "(unpacked - zero_point) * scale"
    assert report["numeric_policy"]["scale_format"] == "signed_q4_4"
    assert report["numeric_policy"]["groups"] == 2
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_kernel_simulates_and_reports_gptq(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection", cfg, tmp_path)
    assert result.status == "passed"
    assert (tmp_path / "gptq_weight_report.json").exists()
    assert "PASS int4_projection" in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "synthetic_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_tile_kernel_simulates_and_reports_fixture_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_tile", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_tile" in result.simulation["output"]
    assert (tmp_path / "projection_tile_golden.json").exists()
    sv_text = (tmp_path / "projection_tile.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "packed_weight_i[flat_idx*4 +: 4]" in sv_text
    assert "col_base_r <= col_base_r + COL_IDX_W'(PE_LANES)" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "projection_tile_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"] == "sum_col(((unpacked_int4 - zero_point) * scale_q4_4) * activation_int8)"
    assert report["numeric_policy"]["scale_format"] == "signed_q4_4_per_row_group"
    assert report["tile_parameters"]["tile_rows"] == 2
    assert report["tile_parameters"]["tile_cols"] == 8
    assert report["tile_parameters"]["group_size"] == 4
    assert report["tile_parameters"]["groups_per_row"] == 2
    assert report["tile_parameters"]["pe_lanes"] == 2
    assert report["round_trip_passed"] is True
    assert report["expected_outputs"] == [-144, 244]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_streaming_kernel_consumes_stream_and_reports_fixture_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_streaming", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_streaming" in result.simulation["output"]
    assert "STREAM_TRACE projection_streaming" in result.simulation["output"]
    assert (tmp_path / "projection_streaming_golden.json").exists()
    sv_text = (tmp_path / "projection_streaming.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "input  logic [STREAM_WORD_WIDTH-1:0]              weight_word_i" in sv_text
    assert "input  logic                                      weight_valid_i" in sv_text
    assert "output logic                                      weight_ready_o" in sv_text
    assert "input  logic                                      weight_last_i" in sv_text
    assert "packed_weight_r[int'(stream_word_idx_r)*STREAM_WORD_WIDTH +: STREAM_WORD_WIDTH] <= weight_word_i" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "projection_streaming_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"] == "sum_col(((streamed_unpacked_int4 - zero_point) * scale_q4_4) * activation_int8)"
    assert report["numeric_policy"]["stream_payload_order"] == "word_idx_zero_carries_lowest_packed_weight_bytes"
    assert report["tile_parameters"]["tile_rows"] == 2
    assert report["tile_parameters"]["tile_cols"] == 8
    assert report["tile_parameters"]["group_size"] == 4
    assert report["tile_parameters"]["groups_per_row"] == 2
    assert report["tile_parameters"]["requested_pe_lanes"] == 64
    assert report["tile_parameters"]["effective_fixture_pe_lanes"] == 2
    assert report["stream_interface"]["configured_memory_data_width_bits"] == 128
    assert report["stream_interface"]["effective_fixture_stream_word_width_bits"] == 32
    assert report["stream_interface"]["stream_word_count"] == 2
    assert report["stream_interface"]["weight_ready_port"] == "weight_ready_o"
    assert report["stream_trace"]["consumed_words_hex"] == ["0xa2f570d8", "0x6b4fe387"]
    assert report["stream_trace"]["weight_last_on_final_word"] is True
    assert report["round_trip_passed"] is True
    assert report["expected_outputs"] == [-144, 244]
    assert report["board_level_signoff"] is False
    assert "DDR_streaming_controller" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_parallel_streaming_kernel_uses_true_parallel_lanes(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_parallel_streaming", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_parallel_streaming" in result.simulation["output"]
    assert "STREAM_TRACE projection_parallel_streaming" in result.simulation["output"]
    assert "PARALLEL_TRACE projection_parallel_streaming true_lanes=2" in result.simulation["output"]
    assert (tmp_path / "projection_parallel_streaming_golden.json").exists()
    sv_text = (tmp_path / "projection_parallel_streaming.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "input  logic [STREAM_WORD_WIDTH-1:0]              weight_word_i" in sv_text
    assert "input  logic                                      weight_valid_i" in sv_text
    assert "output logic                                      weight_ready_o" in sv_text
    assert "input  logic                                      weight_last_i" in sv_text
    assert "MAC_PAIR" in sv_text
    assert "product_lane0_r <=" in sv_text
    assert "product_lane1_r <=" in sv_text
    assert "pair_sum_w = product_lane0_r + product_lane1_r" in sv_text
    assert "packed_weight_r[int'(stream_word_idx_r)*STREAM_WORD_WIDTH +: STREAM_WORD_WIDTH] <= weight_word_i" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "projection_parallel_streaming_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"] == "sum_col(((streamed_unpacked_int4 - zero_point) * scale_q4_4) * activation_int8)"
    assert report["numeric_policy"]["lane_product_format"] == "signed_int32_raw_q4_4"
    assert report["tile_parameters"]["tile_rows"] == 2
    assert report["tile_parameters"]["tile_cols"] == 8
    assert report["tile_parameters"]["group_size"] == 4
    assert report["tile_parameters"]["groups_per_row"] == 2
    assert report["tile_parameters"]["requested_pe_lanes"] == 64
    assert report["tile_parameters"]["effective_fixture_pe_lanes"] == 2
    assert report["tile_parameters"]["true_parallel_datapath_lanes"] == 2
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 2
    assert report["lane_policy"]["parallel_products_per_cycle"] == 2
    assert report["lane_policy"]["not_merely_lane_index_scheduling"] is True
    assert report["stream_interface"]["configured_memory_data_width_bits"] == 128
    assert report["stream_interface"]["effective_fixture_stream_word_width_bits"] == 32
    assert report["stream_interface"]["stream_word_count"] == 2
    assert report["stream_trace"]["consumed_words_hex"] == ["0xa2f570d8", "0x6b4fe387"]
    assert report["stream_trace"]["testbench_inserts_valid_low_gap"] is True
    assert report["round_trip_passed"] is True
    assert report["expected_outputs"] == [-144, 244]
    assert report["board_level_signoff"] is False
    assert "DDR_streaming_controller" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_cli_accepts_packed_stream_adapter_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "packed_stream_adapter",
            "--out",
            "build/packed_stream_adapter_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "packed_stream_adapter"


def test_cli_accepts_packed_stream_adapter_multiword_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "packed_stream_adapter_multiword",
            "--out",
            "build/packed_stream_adapter_multiword_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "packed_stream_adapter_multiword"


def test_cli_accepts_projection_adapter_stream_integration_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_adapter_stream_integration",
            "--out",
            "build/projection_adapter_stream_integration_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_adapter_stream_integration"


def test_cli_accepts_projection_target_stream_plan_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_target_stream_plan",
            "--out",
            "build/projection_target_stream_plan_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_target_stream_plan"


def test_cli_accepts_projection_memory_stream_boundary_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_memory_stream_boundary",
            "--out",
            "build/projection_memory_stream_boundary_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_memory_stream_boundary"


def test_cli_accepts_projection_internal_stream_shell_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_internal_stream_shell",
            "--out",
            "build/projection_internal_stream_shell_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_internal_stream_shell"


def test_cli_accepts_projection_axi_read_command_adapter_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_axi_read_command_adapter",
            "--out",
            "build/projection_axi_read_command_adapter_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_axi_read_command_adapter"


def test_cli_accepts_projection_axi_read_data_channel_adapter_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_axi_read_data_channel_adapter",
            "--out",
            "build/projection_axi_read_data_channel_adapter_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_axi_read_data_channel_adapter"


def test_cli_accepts_projection_axi_read_transaction_adapter_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_axi_read_transaction_adapter",
            "--out",
            "build/projection_axi_read_transaction_adapter_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_axi_read_transaction_adapter"


def test_cli_accepts_projection_axi_stream_integration_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "projection_axi_stream_integration",
            "--out",
            "build/projection_axi_stream_integration_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "projection_axi_stream_integration"


def test_cli_accepts_decoder_child_axi_attention_datapath_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "decoder_child_axi_attention_datapath",
            "--out",
            "build/decoder_child_axi_attention_datapath_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "decoder_child_axi_attention_datapath"


def test_cli_accepts_layer_fsm_axi_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "layer_fsm_axi_attention_fixture",
            "--out",
            "build/layer_fsm_axi_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "layer_fsm_axi_attention_fixture"


def test_cli_accepts_top_fsm_axi_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "top_fsm_axi_attention_fixture",
            "--out",
            "build/top_fsm_axi_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "top_fsm_axi_attention_fixture"


def test_cli_accepts_token_loop_axi_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "token_loop_axi_attention_fixture",
            "--out",
            "build/token_loop_axi_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "token_loop_axi_attention_fixture"


@pytest.mark.parametrize(
    "kernel",
    [
        "decoder_block_axi_attention_mlp_fixture",
        "layer_fsm_axi_decoder_block_fixture",
        "top_fsm_axi_decoder_block_fixture",
        "token_loop_axi_decoder_block_fixture",
        "model_fsm_axi_decoder_block_fixture",
    ],
)
def test_cli_accepts_axi_decoder_block_chain_kernel_names(kernel: str):
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            kernel,
            "--out",
            f"build/{kernel}_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == kernel


def test_cli_accepts_rmsnorm_rope_source_path_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "rmsnorm_rope_source_path",
            "--out",
            "build/rmsnorm_rope_source_path_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "rmsnorm_rope_source_path"


def test_cli_accepts_attention_kv_cache_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "attention_kv_cache_fixture",
            "--out",
            "build/attention_kv_cache_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "attention_kv_cache_fixture"


def test_cli_accepts_decoder_child_attention_datapath_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "decoder_child_attention_datapath",
            "--out",
            "build/decoder_child_attention_datapath_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "decoder_child_attention_datapath"


def test_cli_accepts_layer_fsm_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "layer_fsm_attention_fixture",
            "--out",
            "build/layer_fsm_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "layer_fsm_attention_fixture"


def test_cli_accepts_top_fsm_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "top_fsm_attention_fixture",
            "--out",
            "build/top_fsm_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "top_fsm_attention_fixture"


def test_cli_accepts_token_loop_attention_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "token_loop_attention_fixture",
            "--out",
            "build/token_loop_attention_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "token_loop_attention_fixture"


def test_cli_accepts_residual_mlp_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "residual_mlp_fixture",
            "--out",
            "build/residual_mlp_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "residual_mlp_fixture"


def test_cli_accepts_decoder_block_attention_mlp_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "decoder_block_attention_mlp_fixture",
            "--out",
            "build/decoder_block_attention_mlp_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "decoder_block_attention_mlp_fixture"


def test_cli_accepts_layer_fsm_decoder_block_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "layer_fsm_decoder_block_fixture",
            "--out",
            "build/layer_fsm_decoder_block_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "layer_fsm_decoder_block_fixture"


def test_cli_accepts_top_fsm_decoder_block_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "top_fsm_decoder_block_fixture",
            "--out",
            "build/top_fsm_decoder_block_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "top_fsm_decoder_block_fixture"


def test_cli_accepts_token_loop_decoder_block_fixture_kernel():
    args = build_parser().parse_args(
        [
            "agent",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--mode",
            "kernel",
            "--kernel",
            "token_loop_decoder_block_fixture",
            "--out",
            "build/token_loop_decoder_block_fixture_gate",
            "--verbose",
        ]
    )
    assert args.command == "agent"
    assert args.mode == "kernel"
    assert args.kernel == "token_loop_decoder_block_fixture"


def test_emit_packed_stream_adapter_kernel_splits_configured_memory_word(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("packed_stream_adapter", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS packed_stream_adapter" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE packed_stream_adapter" in result.simulation["output"]
    assert "STREAM_TRACE packed_stream_adapter" in result.simulation["output"]
    assert (tmp_path / "packed_stream_adapter_golden.json").exists()
    sv_text = (tmp_path / "packed_stream_adapter.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "parameter int MEM_WORD_WIDTH = 128" in sv_text
    assert "parameter int PAYLOAD_WIDTH = 32" in sv_text
    assert "input  logic [MEM_WORD_WIDTH-1:0]                 mem_word_i" in sv_text
    assert "input  logic                                      mem_valid_i" in sv_text
    assert "output logic                                      mem_ready_o" in sv_text
    assert "input  logic                                      mem_last_i" in sv_text
    assert "output logic [PAYLOAD_WIDTH-1:0]                  payload_word_o" in sv_text
    assert "output logic                                      payload_valid_o" in sv_text
    assert "input  logic                                      payload_ready_i" in sv_text
    assert "output logic                                      payload_last_o" in sv_text
    assert "payload_word_r <= mem_word_i[0 +: PAYLOAD_WIDTH]" in sv_text
    assert "payload_word_r <= mem_word_r[int'(next_payload_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH]" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "packed_stream_adapter"
    assert report["coverage_level"] == "packed_stream_adapter_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["input_stream_width_bits"] == 128
    assert report["output_payload_width_bits"] == 32
    assert report["input_word_count"] == 1
    assert report["output_payload_count"] == 4
    assert report["stream_interface"]["mem_word_port"] == "mem_word_i"
    assert report["stream_interface"]["payload_word_port"] == "payload_word_o"
    assert report["stream_interface"]["payload_chunks_per_mem_word"] == 4
    assert report["stream_trace"]["consumed_input_words_hex"] == [
        "0x783bf6e4d5a02c196b4fe387a2f570d8"
    ]
    assert report["stream_trace"]["emitted_payload_words_hex"] == [
        "0xa2f570d8",
        "0x6b4fe387",
        "0xd5a02c19",
        "0x783bf6e4",
    ]
    assert report["backpressure_trace"]["testbench_inserts_output_backpressure"] is True
    assert report["backpressure_trace"]["ready_low_payload_indices"] == [0]
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_packed_stream_adapter_multiword_kernel_splits_two_configured_memory_words(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("packed_stream_adapter_multiword", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS packed_stream_adapter_multiword" in result.simulation["output"]
    assert "INPUT_HANDSHAKE_TRACE packed_stream_adapter_multiword" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE packed_stream_adapter_multiword ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "STREAM_TRACE packed_stream_adapter_multiword" in result.simulation["output"]
    assert (tmp_path / "packed_stream_adapter_multiword_golden.json").exists()
    sv_text = (tmp_path / "packed_stream_adapter_multiword.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "parameter int MEM_WORD_WIDTH = 128" in sv_text
    assert "parameter int PAYLOAD_WIDTH = 32" in sv_text
    assert "parameter int MAX_MEM_WORDS = 2" in sv_text
    assert "input  logic [MEM_WORD_WIDTH-1:0]                 mem_word_i" in sv_text
    assert "input  logic                                      mem_valid_i" in sv_text
    assert "output logic                                      mem_ready_o" in sv_text
    assert "input  logic                                      mem_last_i" in sv_text
    assert "output logic [PAYLOAD_WIDTH-1:0]                  payload_word_o" in sv_text
    assert "output logic                                      payload_valid_o" in sv_text
    assert "input  logic                                      payload_ready_i" in sv_text
    assert "output logic                                      payload_last_o" in sv_text
    assert "input_last_trace_r[int'(input_count_r)] <= mem_last_i" in sv_text
    assert "payload_word_r <= mem_word_r[int'(next_payload_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH]" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "packed_stream_adapter_multiword"
    assert report["coverage_level"] == "packed_stream_adapter_multiword_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["input_stream_width_bits"] == 128
    assert report["output_payload_width_bits"] == 32
    assert report["input_word_count"] == 2
    assert report["output_payload_count"] == 8
    assert report["stream_interface"]["mem_word_port"] == "mem_word_i"
    assert report["stream_interface"]["payload_word_port"] == "payload_word_o"
    assert report["stream_interface"]["payloads_per_mem_word"] == 4
    assert report["stream_trace"]["consumed_input_words_hex"] == [
        "0x783bf6e4d5a02c196b4fe387a2f570d8",
        "0x84d97f5b30a6e2c148f73a5e0d1294b6",
    ]
    assert report["stream_trace"]["emitted_payload_words_hex"] == [
        "0xa2f570d8",
        "0x6b4fe387",
        "0xd5a02c19",
        "0x783bf6e4",
        "0x0d1294b6",
        "0x48f73a5e",
        "0x30a6e2c1",
        "0x84d97f5b",
    ]
    assert [entry["mem_last_i"] for entry in report["input_handshake_trace"]] == [False, True]
    assert report["output_backpressure_trace"]["ready_low_event_count"] >= 2
    assert report["output_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_LLaMA_projection_streaming" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_adapter_stream_integration_kernel_bridges_adapter_to_projection(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_adapter_stream_integration", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_adapter_stream_integration" in result.simulation["output"]
    assert "INPUT_HANDSHAKE_TRACE projection_adapter_stream_integration" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE projection_adapter_stream_integration ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "PAYLOAD_LINK_TRACE projection_adapter_stream_integration" in result.simulation["output"]
    assert "PARALLEL_TRACE projection_adapter_stream_integration true_lanes=2" in result.simulation["output"]
    assert (tmp_path / "projection_adapter_stream_integration_golden.json").exists()

    sv_text = (tmp_path / "projection_adapter_stream_integration.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "parameter int MEM_WORD_WIDTH = 128" in sv_text
    assert "parameter int PAYLOAD_WIDTH = 32" in sv_text
    assert "input  logic [MEM_WORD_WIDTH-1:0]                 mem_word_i" in sv_text
    assert "input  logic                                      mem_valid_i" in sv_text
    assert "output logic                                      mem_ready_o" in sv_text
    assert "input  logic                                      mem_last_i" in sv_text
    assert "payload_link_valid_w" in sv_text
    assert "payload_link_ready_w" in sv_text
    assert "payload_word_r <= mem_word_r[int'(next_payload_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH]" in sv_text
    assert "packed_weight_r[int'(projection_consume_count_r)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] <= payload_word_r" in sv_text
    assert "MAC_PAIR" in sv_text
    assert "product_lane_r[lane_seq_idx*32 +: 32] <=" in sv_text
    assert "pair_sum_w = pair_sum_w + product_lane_at(lane_comb_idx)" in sv_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "projection_adapter_stream_integration"
    assert report["coverage_level"] == "projection_adapter_stream_integration_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["payload_width_bits"] == 32
    assert report["input_word_count"] == 2
    assert report["payload_count"] == 8
    assert report["consumed_memory_words_hex"] == [
        "0x783bf6e4d5a02c196b4fe387a2f570d8",
        "0x84d97f5b30a6e2c148f73a5e0d1294b6",
    ]
    expected_payloads = [
        "0xa2f570d8",
        "0x6b4fe387",
        "0xd5a02c19",
        "0x783bf6e4",
        "0x0d1294b6",
        "0x48f73a5e",
        "0x30a6e2c1",
        "0x84d97f5b",
    ]
    assert report["adapter_emitted_payload_words_hex"] == expected_payloads
    assert report["projection_consumed_payload_words_hex"] == expected_payloads
    assert report["payload_link_match_passed"] is True
    assert [entry["mem_last_i"] for entry in report["input_handshake_trace"]] == [False, True]
    assert report["adapter_projection_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["adapter_projection_backpressure_trace"]["ready_low_event_count"] == 3
    assert report["adapter_projection_link_trace"]["payload_link_match_passed"] is True
    assert report["projection_output_vector"] == report["python_numpy_golden_output_vector"]
    assert len(report["projection_output_vector"]) == 2
    assert report["lane_policy"]["requested_pe_lanes"] == 64
    assert report["lane_policy"]["effective_fixture_pe_lanes"] == 2
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 2
    assert report["lane_policy"]["parallel_products_per_cycle"] == 2
    assert report["lane_policy"]["not_merely_lane_index_scheduling"] is True
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "target_scale_LLaMA_projection" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_target_stream_plan_reports_target_metadata_and_fixture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_SELECTED_PROJECTION", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_target_stream_plan" in result.simulation["output"]
    assert "OBSERVED_ADAPTER_PAYLOADS projection_target_stream_plan" in result.simulation["output"]
    assert "OBSERVED_CONSUMED_PAYLOADS projection_target_stream_plan" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE projection_target_stream_plan ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "PARALLEL_TRACE projection_target_stream_plan true_lanes=64" in result.simulation["output"]
    assert (tmp_path / "projection_target_stream_plan_golden.json").exists()
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))

    sv_text = (tmp_path / "projection_target_stream_plan.sv").read_text(encoding="utf-8")
    assert "module projection_target_stream_plan #(" in sv_text
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "parameter int MEM_WORD_WIDTH = 128" in sv_text
    assert "parameter int PAYLOAD_WIDTH = 32" in sv_text
    assert "parameter int MAX_MEM_WORDS = 4" in sv_text
    assert "parameter int MAX_PAYLOADS = 16" in sv_text
    assert "payload_link_valid_w" in sv_text
    assert "payload_link_ready_w" in sv_text
    assert "MAC_PAIR" in sv_text
    assert "product_lane_r[lane_seq_idx*32 +: 32] <=" in sv_text
    assert "TRUE_PARALLEL_LANES = 64" in sv_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "projection_target_stream_plan"
    assert report["coverage_level"] == "projection_target_stream_plan_fixture"
    assert report["selected_projection"] == "q_proj"
    assert report["selected_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["target_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["selected_projection_packed_int4_bytes"] == 2097152
    assert report["selected_projection_memory_beats"] == 131072
    assert report["target_projection_packed_int4_bytes"] == 2097152
    assert report["target_projection_memory_beats"] == 131072
    assert report["full_target_projection_execution"] is False
    assert report["model_metadata"]["hidden_size"] == 2048
    assert report["model_metadata"]["intermediate_size"] == 8192
    assert report["model_metadata"]["num_attention_heads"] == 32
    assert report["model_metadata"]["num_key_value_heads"] == 8
    assert report["model_metadata"]["head_dim"] == 64
    assert report["model_metadata"]["decoder_layers"] == 16
    assert report["model_metadata"]["sequence_length"] == 2048
    assert report["projection_shapes"]["q_proj"] == {"rows": 2048, "cols": 2048}
    assert report["projection_shapes"]["k_proj"] == {"rows": 512, "cols": 2048}
    assert report["projection_shapes"]["v_proj"] == {"rows": 512, "cols": 2048}
    assert report["projection_shapes"]["o_proj"] == {"rows": 2048, "cols": 2048}
    assert report["projection_shapes"]["gate_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_shapes"]["up_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_shapes"]["down_proj"] == {"rows": 2048, "cols": 8192}
    assert report["projection_estimates"]["q_proj"]["packed_int4_bytes"] == 2097152
    assert report["projection_estimates"]["q_proj"]["memory_beats"] == 131072
    assert report["projection_estimates"]["k_proj"]["packed_int4_bytes"] == 524288
    assert report["projection_estimates"]["v_proj"]["packed_int4_bytes"] == 524288
    assert report["projection_estimates"]["v_proj"]["memory_beats"] == 32768
    assert report["projection_estimates"]["gate_proj"]["memory_beats"] == 524288
    assert golden["selected_projection"] == "q_proj"
    assert golden["selected_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert golden["target_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert golden["selected_projection_packed_int4_bytes"] == 2097152
    assert golden["selected_projection_memory_beats"] == 131072
    assert golden["target_projection_packed_int4_bytes"] == 2097152
    assert golden["target_projection_memory_beats"] == 131072
    assert golden["full_target_projection_execution"] is False
    assert golden["target_fixture_distinction"]["full_target_projection_execution"] is False
    assert golden["target_fixture_distinction"]["target_tile_rows"] == 64
    assert golden["target_fixture_distinction"]["target_tile_cols"] == 128
    assert golden["target_fixture_distinction"]["fixture_tile_rows"] == 2
    assert golden["target_fixture_distinction"]["fixture_tile_cols"] == 64
    assert report["configured_memory_data_width_bits"] == 128
    assert report["payload_width_bits"] == 32
    assert report["memory_word_count"] == 4
    assert report["payload_count"] == 16
    assert len(report["consumed_memory_words_hex"]) == 4
    assert len(report["adapter_emitted_payload_words_hex"]) == 16
    assert report["adapter_emitted_payload_words_hex"] == report["projection_consumed_payload_words_hex"]
    assert report["payload_link_match_passed"] is True
    assert report["adapter_projection_link_trace"]["evidence_source"] == "parsed_from_iverilog_observed_payload_trace"
    assert report["adapter_projection_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["projection_output_vector"] == report["python_numpy_golden_output_vector"]
    assert len(report["projection_output_vector"]) == 2
    assert report["target_planning_tile_parameters"]["selected_target_planning_lanes"] == 64
    assert report["target_planning_tile_parameters"]["selected_projection"] == "q_proj"
    assert report["target_planning_tile_parameters"]["output_tile_rows"] == 64
    assert report["target_planning_tile_parameters"]["input_tile_cols"] == 128
    assert report["target_fixture_distinction"]["full_target_projection_execution"] is False
    assert report["target_fixture_distinction"]["target_tile_rows"] == 64
    assert report["target_fixture_distinction"]["target_tile_cols"] == 128
    assert report["target_fixture_distinction"]["fixture_tile_rows"] == 2
    assert report["target_fixture_distinction"]["fixture_tile_cols"] == 64
    assert report["fixture_tile_parameters"]["tile_rows"] == 2
    assert report["fixture_tile_parameters"]["tile_cols"] == 64
    assert report["fixture_tile_parameters"]["memory_word_count"] == 4
    assert report["fixture_tile_parameters"]["payload_count"] == 16
    assert report["lane_policy"]["requested_pe_lanes"] == 64
    assert report["lane_policy"]["selected_target_planning_lanes"] == 64
    assert report["lane_policy"]["effective_fixture_lanes"] == 64
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 64
    assert report["lane_policy"]["parallel_products_per_cycle"] == 64
    assert report["lane_policy"]["pe_count_controls_true_parallel_datapath"] is True
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_target_stream_plan_uses_env_selected_k_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "k_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "k_proj"
        assert artifact["selected_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["target_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["selected_projection_packed_int4_bytes"] == 524288
        assert artifact["selected_projection_memory_beats"] == 32768
        assert artifact["target_projection_packed_int4_bytes"] == 524288
        assert artifact["target_projection_memory_beats"] == 32768
        assert artifact["target_fixture_distinction"]["selected_projection"] == "k_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is k_proj" in artifact["target_fixture_distinction"]["distinction"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_uses_env_selected_v_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "v_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "v_proj"
        assert artifact["semantic_op"] == "v_proj"
        assert artifact["selected_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["target_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["target_shape"] == "512x2048"
        assert artifact["target_projection_rows"] == 512
        assert artifact["target_projection_cols"] == 2048
        assert artifact["selected_projection_packed_int4_bytes"] == 524288
        assert artifact["selected_projection_memory_beats"] == 32768
        assert artifact["target_projection_packed_int4_bytes"] == 524288
        assert artifact["target_projection_memory_beats"] == 32768
        assert artifact["real_gptq_checkpoint_layout_compatible"] is False
        assert artifact["gptq_layout_preflight"]["real_checkpoint_layout_preflight"] == "blocked"
        assert artifact["gptq_layout_preflight"]["target_checkpoint_layout_compatible"] is False
        assert artifact["packed_int4_streaming_evidence"]["round_trip_passed"] is True
        assert artifact["packed_int4_streaming_evidence"]["memory_word_count"] == 4
        assert artifact["packed_int4_streaming_evidence"]["payload_count"] == 16
        assert artifact["packed_int4_streaming_evidence"]["payload_link_match_passed"] is True
        assert artifact["packed_int4_streaming_evidence"]["true_parallel_datapath_lanes"] == 64
        assert artifact["target_fixture_distinction"]["selected_projection"] == "v_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is v_proj" in artifact["target_fixture_distinction"]["distinction"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "real_checkpoint_layout_compatibility" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_uses_env_selected_o_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "o_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "o_proj"
        assert artifact["selected_projection_shape"] == {"rows": 2048, "cols": 2048}
        assert artifact["target_projection_shape"] == {"rows": 2048, "cols": 2048}
        assert artifact["selected_projection_packed_int4_bytes"] == 2097152
        assert artifact["selected_projection_memory_beats"] == 131072
        assert artifact["target_projection_packed_int4_bytes"] == 2097152
        assert artifact["target_projection_memory_beats"] == 131072
        assert artifact["full_target_projection_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_projection"] == "o_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is o_proj" in artifact["target_fixture_distinction"]["distinction"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_uses_env_selected_gate_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "gate_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "gate_proj"
        assert artifact["selected_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["target_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["selected_projection_packed_int4_bytes"] == 8388608
        assert artifact["selected_projection_memory_beats"] == 524288
        assert artifact["target_projection_packed_int4_bytes"] == 8388608
        assert artifact["target_projection_memory_beats"] == 524288
        assert artifact["full_target_projection_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_projection"] == "gate_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is gate_proj" in artifact["target_fixture_distinction"]["distinction"]
    assert report["projection_shapes"]["gate_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_estimates"]["gate_proj"]["packed_int4_bytes"] == 8388608
    assert report["projection_estimates"]["gate_proj"]["memory_beats"] == 524288
    assert report["target_planning_tile_parameters"]["selected_projection"] == "gate_proj"
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_uses_env_selected_up_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "up_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "up_proj"
        assert artifact["selected_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["target_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["selected_projection_packed_int4_bytes"] == 8388608
        assert artifact["selected_projection_memory_beats"] == 524288
        assert artifact["target_projection_packed_int4_bytes"] == 8388608
        assert artifact["target_projection_memory_beats"] == 524288
        assert artifact["full_target_projection_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_projection"] == "up_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is up_proj" in artifact["target_fixture_distinction"]["distinction"]
        assert "selected projection is gate_proj" not in artifact["target_fixture_distinction"]["distinction"]
    assert report["projection_shapes"]["gate_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_shapes"]["up_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_estimates"]["up_proj"]["packed_int4_bytes"] == 8388608
    assert report["projection_estimates"]["up_proj"]["memory_beats"] == 524288
    assert report["target_planning_tile_parameters"]["selected_projection"] == "up_proj"
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_uses_env_selected_down_proj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "down_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_target_stream_plan", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_target_stream_plan_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "down_proj"
        assert artifact["selected_projection_shape"] == {"rows": 2048, "cols": 8192}
        assert artifact["target_projection_shape"] == {"rows": 2048, "cols": 8192}
        assert artifact["selected_projection_packed_int4_bytes"] == 8388608
        assert artifact["selected_projection_memory_beats"] == 524288
        assert artifact["target_projection_packed_int4_bytes"] == 8388608
        assert artifact["target_projection_memory_beats"] == 524288
        assert artifact["full_target_projection_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_projection"] == "down_proj"
        assert artifact["target_fixture_distinction"]["full_target_projection_execution"] is False
        assert "selected projection is down_proj" in artifact["target_fixture_distinction"]["distinction"]
        assert "selected projection is gate_proj" not in artifact["target_fixture_distinction"]["distinction"]
        assert "selected projection is up_proj" not in artifact["target_fixture_distinction"]["distinction"]
    assert report["projection_shapes"]["gate_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_shapes"]["up_proj"] == {"rows": 8192, "cols": 2048}
    assert report["projection_shapes"]["down_proj"] == {"rows": 2048, "cols": 8192}
    assert report["projection_estimates"]["down_proj"]["packed_int4_bytes"] == 8388608
    assert report["projection_estimates"]["down_proj"]["memory_beats"] == 524288
    assert report["target_planning_tile_parameters"]["selected_projection"] == "down_proj"
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_projection_target_stream_plan_rejects_invalid_selected_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "bogus_proj")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    with pytest.raises(ValueError, match="NL2HDL_SELECTED_PROJECTION='bogus_proj'"):
        emit_kernel("projection_target_stream_plan", cfg, tmp_path)

    assert not (tmp_path / "projection_target_stream_plan.sv").exists()
    assert not (tmp_path / "projection_target_stream_plan_golden.json").exists()
    assert not (tmp_path / "kernel_report.json").exists()


def test_emit_projection_axi_read_command_adapter_issues_stable_bounded_ar_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.delenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_axi_read_command_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_read_command_adapter" in result.simulation["output"]
    assert "AXI_COMMAND_TRACE projection_axi_read_command_adapter addr=0x120000 len=3 size=4 burst=1 id=0x2 beats=4" in result.simulation[
        "output"
    ]
    assert "AXI_BACKPRESSURE_TRACE projection_axi_read_command_adapter ready_low_cycles=2 arvalid_held=1" in result.simulation[
        "output"
    ]
    assert "AXI_FIELD_STABILITY_TRACE projection_axi_read_command_adapter stable_during_ready_low=1" in result.simulation[
        "output"
    ]
    assert (tmp_path / "projection_axi_read_command_adapter_golden.json").exists()

    sv_text = (tmp_path / "projection_axi_read_command_adapter.sv").read_text(encoding="utf-8")
    assert "module projection_axi_read_command_adapter #(" in sv_text
    assert "input  logic                      aclk" in sv_text
    assert "input  logic                      aresetn" in sv_text
    assert "input  logic                      start_i" in sv_text
    assert "output logic                      done_o" in sv_text
    assert "output logic                      axi_arvalid_o" in sv_text
    assert "input  logic                      axi_arready_i" in sv_text
    assert "output logic [ADDR_WIDTH-1:0]     axi_araddr_o" in sv_text
    assert "output logic [7:0]                axi_arlen_o" in sv_text
    assert "output logic [2:0]                axi_arsize_o" in sv_text
    assert "output logic [1:0]                axi_arburst_o" in sv_text
    assert "output logic [7:0]                axi_arid_o" in sv_text
    assert "parameter int ID_WIDTH = 4" not in sv_text
    assert "output logic [ID_WIDTH-1:0]       axi_arid_o" not in sv_text
    assert "logic [3:0] arid_r" not in sv_text
    assert "parameter int MEM_DATA_WIDTH = 128" in sv_text
    assert "ARSIZE_VALUE" in sv_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_command_adapter_golden.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "projection_axi_read_command_adapter"
    assert report["coverage_level"] == "projection_axi_read_command_adapter_fixture"
    assert report["selected_projection"] == "q_proj"
    assert report["selected_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["target_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["configured_memory_data_width_bits"] == 128
    assert report["bytes_per_beat"] == 16
    assert report["bytes_per_memory_beat"] == 16
    assert report["raw_qweight_memory_beats"] == 131072
    assert report["derived_axi_arsize"] == 4
    assert report["id_width_bits"] == 8
    assert report["request_addr_hex"] == "0x00120000"
    assert report["request_beat_count"] == 4
    assert report["fixture_axi_command_execution"] is True
    assert report["target_planned_request_beat_count"] is None
    assert report["fixture_executed_request_beat_count"] == 4
    assert report["command_split_required"] is False
    assert report["fixture_max_burst_beats"] == 4
    assert report["axi_command_trace"]["axi_arlen"] == 3
    assert report["observed_axi_command_trace"]["axi_arsize"] == 4
    assert report["axi_backpressure_trace"]["ready_low_cycles_while_arvalid_high"] == 2
    assert report["axi_backpressure_trace"]["arvalid_held_while_ready_low"] is True
    assert report["axi_field_stability_trace"]["stable_during_ready_low"] is True
    assert report["target_checkpoint_request_planning_only"] is False
    for artifact in (report, golden):
        assert artifact["checkpoint_target_weight_stream_plan"]["selected_projection"] == "q_proj"
        assert artifact["checkpoint_target_weight_stream_plan"]["stream_plan_valid"] is False
        assert artifact["checkpoint_target_weight_stream_plan"]["qweight_file"] is None
        assert artifact["checkpoint_target_weight_stream_plan"]["qweight_key"] is None
        assert artifact["checkpoint_target_weight_stream_plan"]["memory_data_width_bits"] == 128
        assert artifact["checkpoint_target_weight_stream_plan"]["bytes_per_memory_beat"] == 16
        assert artifact["checkpoint_target_weight_stream_plan"]["raw_qweight_memory_beats"] == 131072
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] is None
        assert artifact["checkpoint_target_request_summary"]["fixture_executed_request_beat_count"] == 4
        assert artifact["checkpoint_target_request_summary"]["command_split_required"] is False
        assert artifact["checkpoint_target_request_summary"]["fixture_max_burst_beats"] == 4
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "AXI_read_data_execution" in report["does_not_claim"]
    assert "full_qweight_payload_streaming" in report["does_not_claim"]
    assert "complete_board_shell" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["board_level_signoff"] is False
    assert report["verilator"]["requested_timing_policy"] == "--timing"
    assert "timing_policy_supported" in report["verilator"]
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_axi_read_command_adapter_reports_checkpoint_plan_without_executing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "qweight_file": "model-00002.safetensors",
        "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
        "qweight_byte_offset": 19,
        "qweight_byte_count": 2050,
        "request_byte_addr": 16,
        "request_beat_start_index": 1,
        "request_beat_count": 129,
        "first_beat_byte_offset": 3,
        "last_beat_valid_bytes": 5,
        "request_covers_unaligned_qweight_range": True,
        "stream_plan_valid": True,
    }
    monkeypatch.setenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", json.dumps(plan))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_read_command_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "AXI_COMMAND_TRACE projection_axi_read_command_adapter addr=0x10 len=3 size=4 burst=1 id=0x2 beats=4" in result.simulation[
        "output"
    ]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_command_adapter_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["target_checkpoint_request_planning_only"] is True
        assert artifact["request_addr_hex"] == "0x00000010"
        assert artifact["request_beat_count"] == 4
        assert artifact["target_planned_request_beat_count"] == 129
        assert artifact["fixture_executed_request_beat_count"] == 4
        assert artifact["command_split_required"] is True
        assert artifact["fixture_max_burst_beats"] == 4
        assert artifact["fixture_axi_command_execution"] is True
        assert artifact["axi_command_trace"]["request_addr_hex"] == "0x00000010"
        assert artifact["axi_command_trace"]["request_beat_count"] == 4
        assert artifact["axi_command_trace"]["axi_arlen"] == 3
        assert artifact["checkpoint_target_weight_stream_plan"] == plan
        assert artifact["checkpoint_target_request_summary"]["last_beat_valid_bytes"] == 5
        assert artifact["checkpoint_target_request_summary"]["request_covers_unaligned_qweight_range"] is True
        assert artifact["checkpoint_target_request_summary"]["request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["fixture_executed_request_beat_count"] == 4
        assert artifact["checkpoint_target_request_summary"]["command_split_required"] is True
        assert "AXI_read_data_execution" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
    assert report["observed_axi_command_trace"]["addr_hex"] == "0x10"
    assert report["observed_axi_command_trace"]["axi_arlen"] == 3
    assert report["observed_axi_command_trace"]["request_beat_count"] == 4


def test_emit_projection_axi_read_data_channel_adapter_splits_r_beats_to_payload_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.delenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_axi_read_data_channel_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert "AXI_R_ACCEPT_TRACE projection_axi_read_data_channel_adapter accepted=0x3 last=0x2" in result.simulation[
        "output"
    ]
    assert "AXI_R_BACKPRESSURE_TRACE projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert "payload_ready_low_indices=0,3,4" in result.simulation["output"]
    assert "PAYLOAD_TRACE projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert "DUT_METADATA_TRACE projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert "NEGATIVE_METADATA_CASE projection_axi_read_data_channel_adapter case=bad_rid" in result.simulation[
        "output"
    ]
    assert "NEGATIVE_METADATA_CASE projection_axi_read_data_channel_adapter case=bad_rresp" in result.simulation[
        "output"
    ]
    assert "NEGATIVE_METADATA_CASE projection_axi_read_data_channel_adapter case=early_rlast" in result.simulation[
        "output"
    ]
    assert (
        "NEGATIVE_METADATA_CASE projection_axi_read_data_channel_adapter case=missing_final_rlast"
        in result.simulation["output"]
    )
    assert "NEGATIVE_METADATA_SUMMARY projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert (tmp_path / "projection_axi_read_data_channel_adapter_golden.json").exists()

    sv_text = (tmp_path / "projection_axi_read_data_channel_adapter.sv").read_text(encoding="utf-8")
    tb_text = (tmp_path / "tb_projection_axi_read_data_channel_adapter.sv").read_text(encoding="utf-8")
    assert "module projection_axi_read_data_channel_adapter #(" in sv_text
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "input  logic                                      axi_rvalid_i" in sv_text
    assert "output logic                                      axi_rready_o" in sv_text
    assert "input  logic [MEM_DATA_WIDTH-1:0]                 axi_rdata_i" in sv_text
    assert "input  logic [7:0]                                axi_rid_i" in sv_text
    assert "input  logic [1:0]                                axi_rresp_i" in sv_text
    assert "input  logic                                      axi_rlast_i" in sv_text
    assert "output logic                                      payload_valid_o" in sv_text
    assert "input  logic                                      payload_ready_i" in sv_text
    assert "output logic [PAYLOAD_WIDTH-1:0]                  payload_word_o" in sv_text
    assert "output logic                                      payload_last_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             accepted_beat_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rid_error_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rresp_error_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rlast_error_trace_o" in sv_text
    assert "output logic [15:0]                               status_o" in sv_text
    assert "parameter int MEM_DATA_WIDTH = 128" in sv_text
    assert "parameter logic [7:0] EXPECTED_AXI_ID = 8'h02" in sv_text
    assert "rid_error_trace_r[int'(beat_count_r)] <= (axi_rid_i != EXPECTED_AXI_ID);" in sv_text
    assert "rresp_error_trace_r[int'(beat_count_r)] <= (axi_rresp_i != 2'b00);" in sv_text
    assert "rlast_error_trace_r[int'(beat_count_r)] <= (axi_rlast_i != expected_rlast_w);" in sv_text
    assert "task automatic run_bad_metadata_regression;" in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_data_channel_adapter_golden.json").read_text(encoding="utf-8"))
    expected_payloads = [
        "0xccddeeff",
        "0x8899aabb",
        "0x44556677",
        "0x00112233",
        "0xdcddfeff",
        "0x98a9babb",
        "0x54657687",
        "0x10213243",
    ]
    assert report["kernel"] == "projection_axi_read_data_channel_adapter"
    assert report["coverage_level"] == "projection_axi_read_data_channel_adapter_fixture"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["bytes_per_memory_beat"] == 16
    assert report["payload_width_bits"] == 32
    assert report["payloads_per_read_data_beat"] == 4
    assert report["fixture_payload_count"] == 8
    assert report["fixture_consumed_read_data_beat_count"] == 2
    assert report["target_planned_request_beat_count"] is None
    assert report["target_checkpoint_request_planning_only"] is False
    assert report["fixture_axi_read_data_execution"]["fixture_consumed_read_data_beat_count"] == 2
    assert report["fixture_axi_read_data_execution"]["payload_words_hex"] == expected_payloads
    assert report["expected_payload_words_hex"] == expected_payloads
    assert report["gptq_payload_probe_used"] is False
    assert report["payload_words_match_gptq_probe"] is False
    assert report["gptq_payload_probe_golden_source"]["status"] == "absent"
    assert report["gptq_payload_probe_golden_source"]["payload_golden_source"] == "deterministic_fixture"
    assert report["payload_trace"]["emitted_payload_words_hex"] == expected_payloads
    assert report["payload_trace"]["payload_order_matches_golden"] is True
    assert report["payload_trace"]["payload_stable_at_done"] is True
    assert report["axi_r_channel_accepted_beat_trace"]["accepted_beat_count"] == 2
    assert report["axi_r_channel_accepted_beat_trace"]["read_data_words_hex"] == [
        "0x00112233445566778899aabbccddeeff",
        "0x102132435465768798a9babbdcddfeff",
    ]
    assert report["r_channel_backpressure_trace"]["rvalid_while_rready_low_cycles"] >= 1
    assert report["r_channel_backpressure_trace"]["payload_ready_low_indices"] == [0, 3, 4]
    assert report["id_resp_last_validation"]["rid_ok"] is True
    assert report["id_resp_last_validation"]["rresp_ok"] is True
    assert report["id_resp_last_validation"]["rlast_ok"] is True
    assert report["id_resp_last_validation"]["beat_order_ok"] is True
    assert report["id_resp_last_validation"]["done_output_stability_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rid_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rresp_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rlast_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rid_error_trace_hex"] == "0x0"
    assert report["dut_observed_r_metadata_validation"]["rresp_error_trace_hex"] == "0x0"
    assert report["dut_observed_r_metadata_validation"]["rlast_error_trace_hex"] == "0x0"
    negative = report["negative_r_metadata_regression"]
    assert negative["injected_bad_rid"] is True
    assert negative["injected_bad_rresp"] is True
    assert negative["injected_early_rlast"] is True
    assert negative["injected_missing_final_rlast"] is True
    assert negative["case_count"] == 4
    assert negative["rid_ok"] is False
    assert negative["rresp_ok"] is False
    assert negative["rlast_ok"] is False
    assert negative["dut_recorded_expected_errors"] is True
    assert set(negative["cases"]) == {"bad_rid", "bad_rresp", "early_rlast", "missing_final_rlast"}
    assert negative["cases"]["bad_rid"]["rid_error_trace_hex"] == "0x1"
    assert negative["cases"]["bad_rid"]["rresp_error_trace_hex"] == "0x0"
    assert negative["cases"]["bad_rid"]["rlast_error_trace_hex"] == "0x0"
    assert negative["cases"]["bad_rresp"]["rid_error_trace_hex"] == "0x0"
    assert negative["cases"]["bad_rresp"]["rresp_error_trace_hex"] == "0x1"
    assert negative["cases"]["bad_rresp"]["rlast_error_trace_hex"] == "0x0"
    assert negative["cases"]["early_rlast"]["rid_error_trace_hex"] == "0x0"
    assert negative["cases"]["early_rlast"]["rresp_error_trace_hex"] == "0x0"
    assert negative["cases"]["early_rlast"]["rlast_error_trace_hex"] == "0x1"
    assert negative["cases"]["missing_final_rlast"]["rid_error_trace_hex"] == "0x0"
    assert negative["cases"]["missing_final_rlast"]["rresp_error_trace_hex"] == "0x0"
    assert negative["cases"]["missing_final_rlast"]["rlast_error_trace_hex"] == "0x2"
    assert report["fixture_axi_read_data_execution"]["negative_r_metadata_regression_required"] is True
    for artifact in (report, golden):
        assert artifact["checkpoint_target_weight_stream_plan"]["selected_projection"] == "q_proj"
        assert artifact["checkpoint_target_weight_stream_plan"]["stream_plan_valid"] is False
        assert artifact["checkpoint_target_weight_stream_plan"]["last_beat_valid_bytes"] is None
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] is None
        assert artifact["checkpoint_target_request_summary"]["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["target_request_split_or_truncated_by_fixture"] is False
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_AXI_master" in report["does_not_claim"]
    assert "full_qweight_payload_streaming" in report["does_not_claim"]
    assert "complete_board_shell" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_LLaMA_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["board_level_signoff"] is False
    assert report["verilator"]["requested_timing_policy"] == "--timing"
    assert "timing_policy_supported" in report["verilator"]
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_axi_read_data_channel_adapter_uses_sampled_gptq_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe_words = [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
        "0x13121110",
        "0x17161514",
        "0x1b1a1918",
        "0x1f1e1d1c",
    ]
    probe = {
        "artifact": "gptq_payload_probe",
        "status": "sampled",
        "projection": "q_proj",
        "sample_bytes_requested": 32,
        "sampled_tensor_count": 3,
        "required_tensor_count": 3,
        "qweight_payload_words32_le_hex": probe_words,
        "qweight_payload_word_count": 8,
        "target_checkpoint_payload_dependency": "satisfied_by_payload_probe",
        "does_not_claim": [
            "full_checkpoint_tensor_materialization",
            "numeric_GPTQ_correctness",
            "checkpoint_specific_qweight_order_correctness",
            "full_qweight_payload_streaming",
            "full_LLaMA_execution",
        ],
    }
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_read_data_channel_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_read_data_channel_adapter" in result.simulation["output"]
    assert (
        "PAYLOAD_TRACE projection_axi_read_data_channel_adapter payload_words="
        + " ".join(probe_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_data_channel_adapter_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["expected_payload_words_hex"] == probe_words
        assert artifact["fixture_axi_read_data_execution"]["payload_words_hex"] == probe_words
        assert artifact["gptq_payload_probe_used"] is True
        assert artifact["payload_words_match_gptq_probe"] is True
        assert artifact["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == probe_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["status"] == "sampled"
        assert source["projection"] == "q_proj"
        assert source["payload_golden_source"] == "gptq_payload_probe"
        assert source["qweight_payload_words32_le_hex"] == probe_words
        assert source["qweight_payload_word_count"] == 8
        assert source["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert source["payload_words_match_gptq_probe"] is True
        assert "DDR_controller_integration" in artifact["does_not_claim"]
        assert "full_AXI_master" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
        assert "complete_board_shell" in artifact["does_not_claim"]
        assert "full_target_projection_execution" in artifact["does_not_claim"]
        assert "full_LLaMA_execution" in artifact["does_not_claim"]
        assert "full_model_execution" in artifact["does_not_claim"]
        assert "board_level_signoff" in artifact["does_not_claim"]
    expected_read_data_words = [
        "0x0f0e0d0c0b0a09080706050403020100",
        "0x1f1e1d1c1b1a19181716151413121110",
    ]
    assert report["axi_r_channel_accepted_beat_trace"]["read_data_words_hex"] == expected_read_data_words
    assert golden["fixture_axi_read_data_execution"]["read_data_words_hex"] == expected_read_data_words
    assert report["payload_trace"]["emitted_payload_words_hex"] == probe_words
    assert report["payload_trace"]["payload_order_matches_golden"] is True
    assert report["payload_trace"]["payload_stable_at_done"] is True


def test_emit_projection_axi_read_data_channel_adapter_uses_selected_projection_from_aggregate_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    k_words = words_by_projection["k_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "k_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_read_data_channel_adapter", cfg, tmp_path)

    assert result.status == "passed"
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_data_channel_adapter_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "k_proj"
        assert artifact["selected_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == k_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == k_words
        assert artifact["fixture_axi_read_data_execution"]["payload_words_hex"] == k_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "k_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == k_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["payload_trace"]["emitted_payload_words_hex"] == k_words


def test_emit_projection_axi_read_data_channel_adapter_reports_checkpoint_plan_without_streaming_full_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "qweight_file": "model-00002.safetensors",
        "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
        "qweight_byte_offset": 19,
        "qweight_byte_count": 2050,
        "request_byte_addr": 16,
        "request_beat_start_index": 1,
        "request_beat_count": 129,
        "first_beat_byte_offset": 3,
        "last_beat_valid_bytes": 5,
        "request_covers_unaligned_qweight_range": True,
        "stream_plan_valid": True,
    }
    monkeypatch.setenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", json.dumps(plan))
    monkeypatch.delenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_read_data_channel_adapter", cfg, tmp_path)

    assert result.status == "passed"
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_data_channel_adapter_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["target_checkpoint_request_planning_only"] is True
        assert artifact["target_planned_request_beat_count"] == 129
        assert artifact["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["target_request_split_or_truncated_by_fixture"] is True
        assert artifact["checkpoint_target_weight_stream_plan"] == plan
        assert artifact["checkpoint_target_request_summary"]["request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["first_beat_byte_offset"] == 3
        assert artifact["checkpoint_target_request_summary"]["last_beat_valid_bytes"] == 5
        assert artifact["checkpoint_target_request_summary"]["request_covers_unaligned_qweight_range"] is True
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["target_request_split_or_truncated_by_fixture"] is True
        assert artifact["fixture_axi_read_data_execution"]["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["fixture_axi_read_data_execution"]["payload_word_count"] == 8
        assert "full_AXI_master" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]


def test_emit_projection_axi_read_transaction_adapter_executes_bounded_ar_r_payload_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_read_transaction_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_read_transaction_adapter" in result.simulation["output"]
    assert (
        "AXI_TRANSACTION_AR_TRACE projection_axi_read_transaction_adapter "
        "addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2"
    ) in result.simulation["output"]
    assert "AXI_TRANSACTION_R_TRACE projection_axi_read_transaction_adapter accepted=0x3 last=0x2" in result.simulation[
        "output"
    ]
    assert "BACKPRESSURE_TRACE projection_axi_read_transaction_adapter" in result.simulation["output"]
    assert "payload_ready_low_indices=0,3,4" in result.simulation["output"]
    assert "PAYLOAD_TRACE projection_axi_read_transaction_adapter" in result.simulation["output"]
    assert (tmp_path / "projection_axi_read_transaction_adapter_golden.json").exists()

    sv_text = (tmp_path / "projection_axi_read_transaction_adapter.sv").read_text(encoding="utf-8")
    assert "module projection_axi_read_transaction_adapter #(" in sv_text
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "output logic                                      axi_arvalid_o" in sv_text
    assert "input  logic                                      axi_arready_i" in sv_text
    assert "output logic [ADDR_WIDTH-1:0]                     axi_araddr_o" in sv_text
    assert "output logic [7:0]                                axi_arlen_o" in sv_text
    assert "output logic [2:0]                                axi_arsize_o" in sv_text
    assert "output logic [1:0]                                axi_arburst_o" in sv_text
    assert "output logic [7:0]                                axi_arid_o" in sv_text
    assert "input  logic                                      axi_rvalid_i" in sv_text
    assert "output logic                                      axi_rready_o" in sv_text
    assert "input  logic [MEM_DATA_WIDTH-1:0]                 axi_rdata_i" in sv_text
    assert "input  logic [7:0]                                axi_rid_i" in sv_text
    assert "input  logic [1:0]                                axi_rresp_i" in sv_text
    assert "input  logic                                      axi_rlast_i" in sv_text
    assert "output logic                                      payload_valid_o" in sv_text
    assert "input  logic                                      payload_ready_i" in sv_text
    assert "output logic [PAYLOAD_WIDTH-1:0]                  payload_word_o" in sv_text
    assert "output logic                                      payload_last_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             accepted_beat_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rid_error_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rresp_error_trace_o" in sv_text
    assert "output logic [FIXTURE_READ_BEATS-1:0]             rlast_error_trace_o" in sv_text
    assert "output logic [15:0]                               status_o" in sv_text
    assert "parameter int MEM_DATA_WIDTH = 128" in sv_text
    assert "parameter logic [7:0] EXPECTED_AXI_ID = 8'h02" in sv_text
    assert "rid_error_trace_r[int'(beat_count_r)] <= (axi_rid_i != EXPECTED_AXI_ID);" in sv_text
    assert "rresp_error_trace_r[int'(beat_count_r)] <= (axi_rresp_i != 2'b00);" in sv_text
    assert "rlast_error_trace_r[int'(beat_count_r)] <= (axi_rlast_i != expected_rlast_w);" in sv_text
    assert "NEGATIVE_METADATA_TRACE projection_axi_read_transaction_adapter" in result.simulation["output"]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_transaction_adapter_golden.json").read_text(encoding="utf-8"))
    expected_payloads = [
        "0xccddeeff",
        "0x8899aabb",
        "0x44556677",
        "0x00112233",
        "0xdcddfeff",
        "0x98a9babb",
        "0x54657687",
        "0x10213243",
    ]
    assert report["kernel"] == "projection_axi_read_transaction_adapter"
    assert report["coverage_level"] == "projection_axi_read_transaction_adapter_fixture"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["bytes_per_memory_beat"] == 16
    assert report["payload_width_bits"] == 32
    assert report["payloads_per_read_data_beat"] == 4
    assert report["fixture_payload_count"] == 8
    assert report["fixture_command_beat_count"] == 2
    assert report["fixture_consumed_read_data_beat_count"] == 2
    assert report["target_planned_request_beat_count"] is None
    assert report["target_checkpoint_request_planning_only"] is False
    assert report["axi_command_trace"]["axi_arlen"] == 1
    assert report["observed_axi_command_trace"]["axi_arlen"] == 1
    assert report["fixture_axi_read_transaction_execution"]["fixture_command_beat_count"] == 2
    assert report["fixture_axi_read_transaction_execution"]["fixture_consumed_read_data_beat_count"] == 2
    assert report["fixture_axi_read_transaction_execution"]["arlen_plus_one_matches_consumed_r_beats"] is True
    assert report["fixture_axi_read_transaction_execution"]["payload_words_hex"] == expected_payloads
    assert report["expected_payload_words_hex"] == expected_payloads
    assert report["payload_trace"]["emitted_payload_words_hex"] == expected_payloads
    assert report["payload_trace"]["payload_order_matches_golden"] is True
    assert report["payload_trace"]["payload_stable_at_done"] is True
    assert report["axi_r_channel_accepted_beat_trace"]["accepted_beat_count"] == 2
    assert report["axi_r_channel_accepted_beat_trace"]["arlen_matches_r_beats"] is True
    assert report["transaction_consistency"]["rid_ok"] is True
    assert report["transaction_consistency"]["rresp_ok"] is True
    assert report["transaction_consistency"]["rlast_ok"] is True
    assert report["transaction_consistency"]["arlen_matches_consumed_r_beats"] is True
    assert report["transaction_consistency"]["payload_order_matches_golden"] is True
    assert report["transaction_consistency"]["done_output_stability_ok"] is True
    assert report["transaction_backpressure_trace"]["ar_ready_low_cycles"] == 2
    assert report["transaction_backpressure_trace"]["ar_fields_stable_during_ready_low"] is True
    assert report["transaction_backpressure_trace"]["rvalid_while_rready_low_cycles"] >= 1
    assert report["transaction_backpressure_trace"]["payload_ready_low_indices"] == [0, 3, 4]
    assert report["id_resp_last_validation"]["rid_ok"] is True
    assert report["id_resp_last_validation"]["rresp_ok"] is True
    assert report["id_resp_last_validation"]["rlast_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rid_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rresp_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rlast_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["rid_error_trace_hex"] == "0x0"
    assert report["dut_observed_r_metadata_validation"]["rresp_error_trace_hex"] == "0x0"
    assert report["dut_observed_r_metadata_validation"]["rlast_error_trace_hex"] == "0x0"
    assert report["negative_r_metadata_regression"]["injected_bad_rid"] is True
    assert report["negative_r_metadata_regression"]["injected_bad_rresp"] is True
    assert report["negative_r_metadata_regression"]["injected_early_rlast"] is True
    assert report["negative_r_metadata_regression"]["injected_missing_final_rlast"] is True
    assert report["negative_r_metadata_regression"]["rid_ok"] is False
    assert report["negative_r_metadata_regression"]["rresp_ok"] is False
    assert report["negative_r_metadata_regression"]["rlast_ok"] is False
    assert report["negative_r_metadata_regression"]["rid_error_trace_hex"] == "0x1"
    assert report["negative_r_metadata_regression"]["rresp_error_trace_hex"] == "0x1"
    assert report["negative_r_metadata_regression"]["rlast_error_trace_hex"] == "0x3"
    assert report["negative_r_metadata_regression"]["dut_recorded_expected_errors"] is True
    for artifact in (report, golden):
        assert artifact["checkpoint_target_weight_stream_plan"]["selected_projection"] == "q_proj"
        assert artifact["checkpoint_target_weight_stream_plan"]["stream_plan_valid"] is False
        assert artifact["checkpoint_target_weight_stream_plan"]["last_beat_valid_bytes"] is None
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] is None
        assert artifact["checkpoint_target_request_summary"]["fixture_command_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["target_request_split_or_truncated_by_fixture"] is False
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_AXI_master" in report["does_not_claim"]
    assert "full_qweight_payload_streaming" in report["does_not_claim"]
    assert "complete_board_shell" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["board_level_signoff"] is False
    assert report["verilator"]["requested_timing_policy"] == "--timing"
    assert "timing_policy_supported" in report["verilator"]
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_axi_read_transaction_adapter_reports_checkpoint_plan_without_full_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "qweight_file": "model-00002.safetensors",
        "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
        "qweight_byte_offset": 19,
        "qweight_byte_count": 2050,
        "request_byte_addr": 16,
        "request_beat_start_index": 1,
        "request_beat_count": 129,
        "first_beat_byte_offset": 3,
        "last_beat_valid_bytes": 5,
        "request_covers_unaligned_qweight_range": True,
        "stream_plan_valid": True,
    }
    monkeypatch.setenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", json.dumps(plan))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_read_transaction_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "AXI_TRANSACTION_AR_TRACE projection_axi_read_transaction_adapter "
        "addr=0x10 len=1 size=4 burst=1 id=0x2 beats=2"
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_transaction_adapter_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["target_checkpoint_request_planning_only"] is True
        assert artifact["request_addr_hex"] == "0x00000010"
        assert artifact["request_beat_count"] == 2
        assert artifact["target_planned_request_beat_count"] == 129
        assert artifact["fixture_command_beat_count"] == 2
        assert artifact["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["target_request_split_or_truncated_by_fixture"] is True
        assert artifact["checkpoint_target_weight_stream_plan"] == plan
        assert artifact["checkpoint_target_request_summary"]["request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["first_beat_byte_offset"] == 3
        assert artifact["checkpoint_target_request_summary"]["last_beat_valid_bytes"] == 5
        assert artifact["checkpoint_target_request_summary"]["request_covers_unaligned_qweight_range"] is True
        assert artifact["checkpoint_target_request_summary"]["target_planned_request_beat_count"] == 129
        assert artifact["checkpoint_target_request_summary"]["fixture_command_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["fixture_consumed_read_data_beat_count"] == 2
        assert artifact["checkpoint_target_request_summary"]["target_request_split_or_truncated_by_fixture"] is True
        assert artifact["fixture_axi_read_transaction_execution"]["fixture_command_beat_count"] == 2
        assert artifact["fixture_axi_read_transaction_execution"]["payload_word_count"] == 8
        assert "full_AXI_master" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
    assert report["observed_axi_command_trace"]["addr_hex"] == "0x10"
    assert report["observed_axi_command_trace"]["axi_arlen"] == 1
    assert report["observed_axi_command_trace"]["request_beat_count"] == 2


def test_emit_projection_axi_read_transaction_adapter_uses_selected_o_proj_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    o_words = words_by_projection["o_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "o_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_read_transaction_adapter", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_read_transaction_adapter" in result.simulation["output"]
    assert "PAYLOAD_TRACE projection_axi_read_transaction_adapter payload_words=" + " ".join(o_words) in result.simulation[
        "output"
    ]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_read_transaction_adapter_golden.json").read_text(encoding="utf-8"))
    expected_read_data_words = [
        "0x6f6e6d6c6b6a69686766656463626160",
        "0x7f7e7d7c7b7a79787776757473727170",
    ]
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "o_proj"
        assert artifact["selected_projection_shape"] == {"rows": 2048, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == o_words
        assert artifact["fixture_axi_read_transaction_execution"]["payload_words_hex"] == o_words
        assert artifact["read_data_words_hex"] == expected_read_data_words
        assert artifact["gptq_payload_probe_used"] is True
        assert artifact["payload_words_match_gptq_probe"] is True
        assert artifact["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == o_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "o_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == o_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["payload_trace"]["emitted_payload_words_hex"] == o_words
    assert report["axi_r_channel_accepted_beat_trace"]["read_data_words_hex"] == expected_read_data_words
    assert report["transaction_consistency"]["arlen_matches_consumed_r_beats"] is True


def test_emit_projection_axi_stream_integration_observes_axi_payload_projection_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_stream_integration" in result.simulation["output"]
    assert (
        "AXI_STREAM_AR_TRACE projection_axi_stream_integration "
        "addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2"
    ) in result.simulation["output"]
    assert "AXI_STREAM_R_METADATA_TRACE projection_axi_stream_integration accepted=0x3 last=0x2" in result.simulation[
        "output"
    ]
    assert "OBSERVED_EMITTED_PAYLOADS projection_axi_stream_integration" in result.simulation["output"]
    assert "OBSERVED_CONSUMED_PAYLOADS projection_axi_stream_integration" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE projection_axi_stream_integration ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "ROUND_TRIP_TRACE projection_axi_stream_integration" in result.simulation["output"]
    assert "PROJECTION_OUTPUT_TRACE projection_axi_stream_integration output=484,1904 golden=484,1904" in result.simulation[
        "output"
    ]

    sv_text = (tmp_path / "projection_axi_stream_integration.sv").read_text(encoding="utf-8")
    module_ports = sv_text[
        sv_text.index("module projection_axi_stream_integration #(") : sv_text.index(
            "    localparam int BYTES_PER_BEAT"
        )
    ]
    assert "input  logic                                      aclk" in module_ports
    assert "input  logic                                      aresetn" in module_ports
    assert "input  logic                                      start_i" in module_ports
    assert "output logic                                      done_o" in module_ports
    assert "output logic signed [TILE_ROWS*OUT_WIDTH-1:0]     output_o" in module_ports
    assert "output logic [63:0]                               integration_status_o" in module_ports
    assert "axi_rdata_i" not in module_ports
    assert "payload_word_o" not in module_ports
    assert "payload_link_ready_w = payload_link_valid_r && !ready_low_required_w" in sv_text
    assert "rid_error_trace_r[int'(beat_count_r)] <= (axi_rid_w != EXPECTED_AXI_ID);" in sv_text
    assert "rresp_error_trace_r[int'(beat_count_r)] <= (axi_rresp_w != 2'b00);" in sv_text
    assert "rlast_error_trace_r[int'(beat_count_r)] <= (axi_rlast_w != expected_rlast_w);" in sv_text
    assert (tmp_path / "projection_axi_stream_integration_golden.json").exists()

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    expected_payloads = [
        "0xe94fa50b",
        "0x61c72d83",
        "0xe94fa50b",
        "0x61c72d83",
        "0xe94fa50b",
        "0x61c72d83",
        "0xe94fa50b",
        "0x61c72d83",
    ]
    assert report["kernel"] == "projection_axi_stream_integration"
    assert report["coverage_level"] == "projection_axi_stream_integration_fixture"
    assert report["configured_memory_data_width_bits"] == 128
    assert report["axi_beat_count"] == 2
    assert report["payload_width_bits"] == 32
    assert report["fixture_payload_count"] == 8
    assert report["observed_axi_command_trace"]["axi_arlen"] == 1
    assert report["observed_r_metadata_validation_trace"]["rid_ok"] is True
    assert report["observed_r_metadata_validation_trace"]["rresp_ok"] is True
    assert report["observed_r_metadata_validation_trace"]["rlast_ok"] is True
    assert report["dut_observed_r_metadata_validation"]["compact_status_hex"].startswith("0xc1a5")
    assert report["emitted_payload_words_hex"] == expected_payloads
    assert report["projection_consumed_payload_words_hex"] == expected_payloads
    assert report["payload_link_match_passed"] is True
    assert report["adapter_projection_link_trace"]["evidence_source"] == (
        "parsed_from_iverilog_observed_axi_to_projection_payload_trace"
    )
    assert report["projection_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["projection_backpressure_trace"]["inside_beat_ready_low_index"] == 0
    assert report["projection_backpressure_trace"]["beat_boundary_ready_low_index"] == 3
    assert report["projection_backpressure_trace"]["post_boundary_ready_low_index"] == 4
    assert report["projection_backpressure_trace"]["payload_valid_held_while_ready_low"] is True
    assert report["projection_backpressure_trace"]["rvalid_while_projection_not_ready_cycles"] >= 1
    assert report["projection_output_vector"] == [484, 1904]
    assert report["projection_output_vector"] == report["python_numpy_golden_output_vector"]
    assert report["round_trip_passed"] is True
    assert report["packed_int4_round_trip_evidence"]["packed_bytes"] == 32
    assert report["packed_int4_round_trip_evidence"]["unpacked_values"] == 64
    assert report["lane_policy"]["requested_pe_lanes"] == 64
    assert report["lane_policy"]["effective_fixture_lanes"] == 1
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 1
    assert report["lane_policy"]["not_claiming_parallel_projection_scaling"] is True
    assert report["child_negative_r_metadata_regression_reference"]["source_kernel"] == (
        "projection_axi_read_transaction_adapter"
    )
    assert report["child_negative_r_metadata_regression_reference"]["integration_run_scope"] == (
        "good_path_only_uses_child_negative_metadata_regression_reference"
    )
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True
    assert report["board_level_signoff"] is False
    for artifact in (report, golden):
        assert "DDR_controller_integration" in artifact["does_not_claim"]
        assert "multi_burst_or_outstanding_AXI_master" in artifact["does_not_claim"]
        assert "full_AXI_master" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
        assert "complete_board_shell" in artifact["does_not_claim"]
        assert "full_target_projection_execution" in artifact["does_not_claim"]
        assert "full_model_execution" in artifact["does_not_claim"]
        assert "board_level_ZCU104_signoff" in artifact["does_not_claim"]


def test_emit_projection_axi_stream_integration_uses_sampled_gptq_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _sample_gptq_payload_probe()
    probe_words = probe["qweight_payload_words32_le_hex"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS projection_axi_stream_integration" in result.simulation["output"]
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(probe_words)
    ) in result.simulation["output"]
    assert "PROJECTION_OUTPUT_TRACE projection_axi_stream_integration output=112,8 golden=112,8" in result.simulation[
        "output"
    ]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    expected_read_data_words = [
        "0x0f0e0d0c0b0a09080706050403020100",
        "0x1f1e1d1c1b1a19181716151413121110",
    ]
    for artifact in (report, golden):
        assert artifact["gptq_payload_probe_used"] is True
        assert artifact["payload_words_match_gptq_probe"] is True
        assert artifact["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == probe_words
        assert artifact["expected_payload_words_hex"] == probe_words
        assert artifact["emitted_payload_words_hex"] == probe_words
        assert artifact["projection_consumed_payload_words_hex"] == probe_words
        assert artifact["read_data_words_hex"] == expected_read_data_words
        assert artifact["projection_output_vector"] == [112, 8]
        assert artifact["python_numpy_golden_output_vector"] == [112, 8]
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["artifact"] == "gptq_payload_probe"
        assert source["projection"] == "q_proj"
        assert source["payload_golden_source"] == "gptq_payload_probe"
        assert source["qweight_payload_words32_le_hex"] == probe_words
    assert report["observed_r_metadata_validation_trace"]["read_data_words_hex"] == expected_read_data_words
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == probe_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == probe_words


def test_emit_projection_axi_stream_integration_uses_selected_projection_from_aggregate_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    k_words = words_by_projection["k_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "k_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(k_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "k_proj"
        assert artifact["selected_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == k_words
        assert artifact["emitted_payload_words_hex"] == k_words
        assert artifact["projection_consumed_payload_words_hex"] == k_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == k_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "k_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == k_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == k_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == k_words


def test_emit_projection_axi_stream_integration_uses_selected_v_proj_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    v_words = words_by_projection["v_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "v_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(v_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "v_proj"
        assert artifact["selected_projection_shape"] == {"rows": 512, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == v_words
        assert artifact["emitted_payload_words_hex"] == v_words
        assert artifact["projection_consumed_payload_words_hex"] == v_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == v_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "v_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == v_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == v_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == v_words


def test_emit_projection_axi_stream_integration_uses_selected_gate_proj_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    gate_words = words_by_projection["gate_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "gate_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(gate_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "gate_proj"
        assert artifact["selected_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == gate_words
        assert artifact["emitted_payload_words_hex"] == gate_words
        assert artifact["projection_consumed_payload_words_hex"] == gate_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == gate_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "gate_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == gate_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == gate_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == gate_words


def test_emit_projection_axi_stream_integration_uses_selected_up_proj_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    up_words = words_by_projection["up_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "up_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(up_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "up_proj"
        assert artifact["selected_projection_shape"] == {"rows": 8192, "cols": 2048}
        assert artifact["expected_payload_words_hex"] == up_words
        assert artifact["emitted_payload_words_hex"] == up_words
        assert artifact["projection_consumed_payload_words_hex"] == up_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == up_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "up_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == up_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == up_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == up_words


def test_emit_projection_axi_stream_integration_uses_selected_down_proj_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    graph = build_llama_semantic_graph("meta-llama/Llama-3.2-1B", cfg)
    probe, words_by_projection = _sample_aggregate_gptq_payload_probe(graph["partition"]["gemm"])
    down_words = words_by_projection["down_proj"]
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    monkeypatch.setenv("NL2HDL_SELECTED_PROJECTION", "down_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))

    result = emit_kernel("projection_axi_stream_integration", cfg, tmp_path)

    assert result.status == "passed"
    assert (
        "EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words="
        + " ".join(down_words)
    ) in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_axi_stream_integration_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_projection"] == "down_proj"
        assert artifact["selected_projection_shape"] == {"rows": 2048, "cols": 8192}
        assert artifact["expected_payload_words_hex"] == down_words
        assert artifact["emitted_payload_words_hex"] == down_words
        assert artifact["projection_consumed_payload_words_hex"] == down_words
        assert artifact["gptq_probe_qweight_payload_words32_le_hex"] == down_words
        source = artifact["gptq_payload_probe_golden_source"]
        assert source["used"] is True
        assert source["projection"] == "down_proj"
        assert source["aggregate_payload_probe_source"] == "projection_payload_probes"
        assert source["qweight_payload_words32_le_hex"] == down_words
        assert source["qweight_payload_words32_le_hex"] != words_by_projection["q_proj"]
    assert report["adapter_projection_link_trace"]["adapter_emitted_payload_words_hex"] == down_words
    assert report["adapter_projection_link_trace"]["projection_consumed_payload_words_hex"] == down_words


def test_emit_projection_axi_stream_integration_rejects_wrong_count_gptq_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _sample_gptq_payload_probe(words=["0x03020100"] * 7)
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    with pytest.raises(ValueError, match="qweight payload word count must equal fixture payload count 8"):
        emit_kernel("projection_axi_stream_integration", cfg, tmp_path)


def test_emit_projection_axi_stream_integration_rejects_non_q_proj_gptq_payload_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _sample_gptq_payload_probe(projection="k_proj")
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    with pytest.raises(ValueError, match="sampled projection must be q_proj"):
        emit_kernel("projection_axi_stream_integration", cfg, tmp_path)


def test_emit_projection_memory_stream_boundary_reports_request_response_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_memory_stream_boundary", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_memory_stream_boundary" in result.simulation["output"]
    assert "REQUEST_TRACE projection_memory_stream_boundary count=1" in result.simulation["output"]
    assert "RESPONSE_TRACE projection_memory_stream_boundary accepted=0xf last=0x8 words=4" in result.simulation[
        "output"
    ]
    assert "OBSERVED_ADAPTER_PAYLOADS projection_memory_stream_boundary" in result.simulation["output"]
    assert "OBSERVED_CONSUMED_PAYLOADS projection_memory_stream_boundary" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE projection_memory_stream_boundary ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "PARALLEL_TRACE projection_memory_stream_boundary true_lanes=2" in result.simulation["output"]
    assert (tmp_path / "projection_memory_stream_boundary_golden.json").exists()

    sv_text = (tmp_path / "projection_memory_stream_boundary.sv").read_text(encoding="utf-8")
    assert "module projection_memory_stream_boundary #(" in sv_text
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "output logic                                      mem_req_valid_o" in sv_text
    assert "input  logic                                      mem_req_ready_i" in sv_text
    assert "output logic [ADDR_WIDTH-1:0]                     mem_req_addr_o" in sv_text
    assert "output logic [15:0]                               mem_req_beats_o" in sv_text
    assert "output logic [7:0]                                mem_req_tag_o" in sv_text
    assert "input  logic [MEM_WORD_WIDTH-1:0]                 mem_rsp_word_i" in sv_text
    assert "input  logic                                      mem_rsp_valid_i" in sv_text
    assert "output logic                                      mem_rsp_ready_o" in sv_text
    assert "input  logic                                      mem_rsp_last_i" in sv_text
    assert "input  logic [7:0]                                mem_rsp_tag_i" in sv_text
    assert "parameter int MEM_WORD_WIDTH = 128" in sv_text
    assert "parameter int PAYLOAD_WIDTH = 32" in sv_text
    assert "parameter int ADDR_WIDTH = 24" in sv_text
    assert "projection_memory_stream_boundary_core" in sv_text
    assert "mem_rsp_tag_i != REQUEST_TAG" in sv_text
    assert "payload_link_valid_o" in sv_text
    assert "payload_link_ready_o" in sv_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "projection_memory_stream_boundary"
    assert report["coverage_level"] == "projection_memory_stream_boundary_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["selected_projection"] == "q_proj"
    assert report["target_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["target_tile_parameters"]["tile_rows"] == 64
    assert report["target_tile_parameters"]["tile_cols"] == 128
    assert report["target_tile_memory_beats"] == 256
    assert report["fixture_memory_beats"] == 4
    assert report["request_addr_hex"] == "0x120000"
    assert report["request_beat_count"] == 4
    assert report["request_tag_hex"] == "0x2a"
    assert report["target_checkpoint_request_planning_only"] is False
    assert "checkpoint_target_weight_stream_plan" not in report
    assert "checkpoint_target_request_summary" not in report
    assert report["fixture_memory_request_execution"]["fixture_memory_request_execution"] is True
    assert report["fixture_memory_request_execution"]["request_addr_hex"] == "0x120000"
    assert report["fixture_memory_request_execution"]["request_beat_count"] == 4
    assert report["fixture_memory_request_execution"]["target_checkpoint_request_planning_only"] is False
    assert report["configured_memory_data_width_bits"] == 128
    assert report["effective_fixture_stream_width_bits"] == 128
    assert report["payload_width_bits"] == 32
    assert report["payload_count"] == 16
    assert len(report["consumed_response_words_hex"]) == 4
    assert len(report["emitted_payload_words_hex"]) == 16
    assert report["emitted_payload_words_hex"] == report["projection_consumed_payload_words_hex"]
    assert report["payload_link_match_passed"] is True
    assert report["observed_memory_request_trace"]["request_count"] == 1
    assert report["observed_memory_request_trace"]["request_fields_stable_under_backpressure"] is True
    assert report["observed_memory_request_trace"]["request_ready_backpressure_cycles"] == 2
    assert [entry["mem_rsp_last_i"] for entry in report["observed_response_handshake_trace"]] == [
        False,
        False,
        False,
        True,
    ]
    assert report["response_stall_trace"]["stall_before_response_indices"] == [1, 3]
    assert report["payload_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["adapter_projection_link_trace"]["evidence_source"] == "parsed_from_iverilog_observed_payload_trace"
    assert report["projection_output_vector"] == report["python_numpy_golden_output_vector"]
    assert len(report["projection_output_vector"]) == 2
    assert report["lane_policy"]["requested_pe_lanes"] == 64
    assert report["lane_policy"]["target_plan_lanes"] == 64
    assert report["lane_policy"]["effective_fixture_lanes"] == 2
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 2
    assert report["lane_policy"]["parallel_products_per_cycle"] == 2
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "complete_board_shell" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_memory_stream_boundary_reports_checkpoint_request_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "qweight_file": "model-00002.safetensors",
        "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
        "qweight_byte_offset": 19,
        "qweight_byte_count": 2050,
        "qweight_byte_end_exclusive": 2069,
        "request_byte_addr": 16,
        "request_beat_start_index": 1,
        "request_beat_count": 129,
        "first_beat_byte_offset": 3,
        "last_beat_valid_bytes": 5,
        "request_covers_unaligned_qweight_range": True,
        "stream_plan_valid": True,
    }
    monkeypatch.setenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", json.dumps(plan))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_memory_stream_boundary", cfg, tmp_path)

    assert result.status == "passed"
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_memory_stream_boundary_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["target_checkpoint_request_planning_only"] is True
        assert artifact["fixture_memory_request_execution"]["fixture_memory_request_execution"] is True
        assert artifact["fixture_memory_request_execution"]["request_addr_hex"] == "0x120000"
        assert artifact["fixture_memory_request_execution"]["request_beat_count"] == 4
        assert artifact["request_beat_count"] == 4
        assert artifact["checkpoint_target_weight_stream_plan"] == plan
        assert artifact["checkpoint_target_request_summary"] == {
            "qweight_file": "model-00002.safetensors",
            "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
            "qweight_byte_offset": 19,
            "qweight_byte_count": 2050,
            "request_byte_addr": 16,
            "request_beat_count": 129,
            "first_beat_byte_offset": 3,
            "last_beat_valid_bytes": 5,
            "request_covers_unaligned_qweight_range": True,
            "stream_plan_valid": True,
        }
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
        assert "DDR_controller_integration" in artifact["does_not_claim"]
        assert "AXI_interface" in artifact["does_not_claim"]
        assert "full_target_projection_execution" in artifact["does_not_claim"]


def test_emit_projection_internal_stream_shell_hides_wide_boundary_and_reports_internal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("projection_internal_stream_shell", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS projection_internal_stream_shell" in result.simulation["output"]
    assert "REQUEST_TRACE projection_internal_stream_shell count=1" in result.simulation["output"]
    assert "RESPONSE_TRACE projection_internal_stream_shell accepted=0xf last=0x8 words=4" in result.simulation[
        "output"
    ]
    assert "RESPONSE_STALL_TRACE projection_internal_stream_shell stall_before_response_idx=1,3 trace=0xa" in result.simulation[
        "output"
    ]
    assert "OBSERVED_ADAPTER_PAYLOADS projection_internal_stream_shell" in result.simulation["output"]
    assert "OBSERVED_CONSUMED_PAYLOADS projection_internal_stream_shell" in result.simulation["output"]
    assert "BACKPRESSURE_TRACE projection_internal_stream_shell ready_low_payload_idx=0,3,4" in result.simulation[
        "output"
    ]
    assert "PARALLEL_TRACE projection_internal_stream_shell true_lanes=2" in result.simulation["output"]
    assert "TOP_IO_TRACE projection_internal_stream_shell exposed_mem_response=0" in result.simulation["output"]
    assert (tmp_path / "projection_internal_stream_shell_golden.json").exists()

    sv_text = (tmp_path / "projection_internal_stream_shell.sv").read_text(encoding="utf-8")
    assert "module projection_internal_stream_shell #(" in sv_text
    module_ports = sv_text[
        sv_text.index("module projection_internal_stream_shell #(") : sv_text.index("    localparam int RESPONSE_IDX_W")
    ]
    assert "input  logic                                      aclk" in module_ports
    assert "input  logic                                      aresetn" in module_ports
    assert "input  logic                                      start_i" in module_ports
    assert "output logic                                      done_o" in module_ports
    assert "output logic signed [TILE_ROWS*OUT_WIDTH-1:0]     output_o" in module_ports
    assert "output logic [63:0]                               shell_status_o" in module_ports
    assert "mem_rsp_word_i" not in module_ports
    assert "mem_req_addr_o" not in module_ports
    assert "payload_link_word_o" not in module_ports
    assert "projection_memory_stream_boundary #(" in sv_text
    assert ".mem_rsp_word_i(mem_rsp_word_w)" in sv_text
    assert ".payload_link_word_o(payload_link_word_w)" in sv_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "projection_internal_stream_shell"
    assert report["coverage_level"] == "projection_internal_stream_shell_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["wraps_kernel"] == "projection_memory_stream_boundary"
    assert report["child_boundary_instantiated"] is True
    assert report["top_level_interface_summary"]["no_top_level_128b_mem_response"] is True
    assert report["top_level_interface_summary"]["no_top_level_mem_request_boundary"] is True
    assert report["top_level_interface_summary"]["no_top_level_32b_payload_link_data"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["previous_reference_bonded_iob_count"] == 340
    assert report["current_bonded_iob_count"] is None
    assert report["io_reduction_gate"]["limit"] == 160
    assert report["selected_projection"] == "q_proj"
    assert report["target_projection_shape"] == {"rows": 2048, "cols": 2048}
    assert report["target_tile_parameters"]["tile_rows"] == 64
    assert report["target_tile_parameters"]["tile_cols"] == 128
    assert report["target_tile_memory_beats"] == 256
    assert report["fixture_memory_beats"] == 4
    assert report["request_addr_hex"] == "0x120000"
    assert report["request_beat_count"] == 4
    assert report["request_tag_hex"] == "0x2a"
    assert report["target_checkpoint_request_planning_only"] is False
    assert report["checkpoint_request_execution_scope"] == "planning_only_shell_internal_boundary_fixture_not_ddr_axi"
    assert "checkpoint_target_weight_stream_plan" not in report
    assert "checkpoint_target_request_summary" not in report
    assert report["fixture_memory_request_execution"]["fixture_memory_request_execution"] is True
    assert report["fixture_memory_request_execution"]["request_addr_hex"] == "0x120000"
    assert report["fixture_memory_request_execution"]["request_beat_count"] == 4
    assert report["fixture_memory_request_execution"]["target_checkpoint_request_planning_only"] is False
    assert len(report["internal_response_words_hex"]) == 4
    assert len(report["emitted_payload_words_hex"]) == 16
    assert report["emitted_payload_words_hex"] == report["projection_consumed_payload_words_hex"]
    assert report["payload_link_match_passed"] is True
    assert report["internal_request_trace"]["request_count"] == 1
    assert report["internal_request_trace"]["request_fields_stable_under_backpressure"] is True
    assert report["response_stall_trace"]["stall_before_response_indices"] == [1, 3]
    assert [entry["mem_rsp_last"] for entry in report["internal_response_handshake_trace"]] == [
        False,
        False,
        False,
        True,
    ]
    assert report["payload_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["adapter_projection_link_trace"]["evidence_source"] == "parsed_from_iverilog_observed_payload_trace"
    assert report["projection_output_vector"] == [976, 2360]
    assert report["projection_output_vector"] == report["python_numpy_golden_output_vector"]
    assert report["lane_policy"]["requested_pe_lanes"] == 64
    assert report["lane_policy"]["target_plan_lanes"] == 64
    assert report["lane_policy"]["effective_fixture_lanes"] == 2
    assert report["lane_policy"]["true_parallel_datapath_lanes"] == 2
    assert report["lane_policy"]["parallel_products_per_cycle"] == 2
    assert report["round_trip_passed"] is True
    assert report["board_level_signoff"] is False
    assert "AXI_interface" in report["does_not_claim"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_qweight_payload_streaming" in report["does_not_claim"]
    assert "complete_board_shell" in report["does_not_claim"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_projection_internal_stream_shell_reports_checkpoint_request_plan_without_widening_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "source": "checkpoint_safetensors_header",
        "qweight_file": "model-00002.safetensors",
        "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
        "qweight_byte_offset": 19,
        "qweight_byte_count": 2050,
        "qweight_byte_end_exclusive": 2069,
        "request_byte_addr": 16,
        "request_beat_start_index": 1,
        "request_beat_count": 129,
        "first_beat_byte_offset": 3,
        "last_beat_valid_bytes": 5,
        "request_covers_unaligned_qweight_range": True,
        "stream_plan_valid": True,
    }
    monkeypatch.setenv("NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON", json.dumps(plan))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("projection_internal_stream_shell", cfg, tmp_path)

    assert result.status == "passed"
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "projection_internal_stream_shell_golden.json").read_text(encoding="utf-8"))
    sv_text = (tmp_path / "projection_internal_stream_shell.sv").read_text(encoding="utf-8")
    module_ports = sv_text[
        sv_text.index("module projection_internal_stream_shell #(") : sv_text.index("    localparam int RESPONSE_IDX_W")
    ]
    assert "mem_rsp_word_i" not in module_ports
    assert "mem_req_addr_o" not in module_ports
    assert "payload_link_word_o" not in module_ports

    for artifact in (report, golden):
        assert artifact["target_checkpoint_request_planning_only"] is True
        assert artifact["checkpoint_request_execution_scope"] == "planning_only_shell_internal_boundary_fixture_not_ddr_axi"
        assert artifact["checkpoint_target_weight_stream_plan"] == plan
        assert artifact["checkpoint_target_request_summary"] == {
            "qweight_file": "model-00002.safetensors",
            "qweight_key": "model.layers.0.self_attn.q_proj.qweight",
            "qweight_byte_offset": 19,
            "qweight_byte_count": 2050,
            "request_byte_addr": 16,
            "request_beat_count": 129,
            "first_beat_byte_offset": 3,
            "last_beat_valid_bytes": 5,
            "request_covers_unaligned_qweight_range": True,
            "stream_plan_valid": True,
        }
        assert artifact["fixture_memory_request_execution"]["fixture_memory_request_execution"] is True
        assert artifact["fixture_memory_request_execution"]["request_addr_hex"] == "0x120000"
        assert artifact["fixture_memory_request_execution"]["request_beat_count"] == 4
        assert artifact["fixture_memory_request_execution"]["target_checkpoint_request_planning_only"] is False
        assert artifact["request_addr_hex"] == "0x120000"
        assert artifact["request_beat_count"] == 4
        assert artifact["projection_output_vector"] == [976, 2360]
        assert artifact["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
        assert artifact["io_reduction_gate"]["limit"] == 160
        assert "AXI_interface" in artifact["does_not_claim"]
        assert "DDR_controller_integration" in artifact["does_not_claim"]
        assert "full_qweight_payload_streaming" in artifact["does_not_claim"]
        assert "complete_board_shell" in artifact["does_not_claim"]
        assert "full_target_projection_execution" in artifact["does_not_claim"]
        assert "full_model_execution" in artifact["does_not_claim"]
        assert "board_level_signoff" in artifact["does_not_claim"]


def test_emit_rmsnorm_kernel_simulates(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rmsnorm" in result.simulation["output"]


def test_emit_rmsnorm_target_kernel_simulates_and_reports_fixture_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm_target", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rmsnorm_target" in result.simulation["output"]
    assert (tmp_path / "rmsnorm_target_golden.json").exists()
    sv_text = (tmp_path / "rmsnorm_target.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "input_i[int'(idx_r)*IN_WIDTH +: IN_WIDTH]" in sv_text
    assert "gamma_i[int'(idx_r)*GAMMA_WIDTH +: GAMMA_WIDTH]" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "rmsnorm_target_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"] == "y_i = (x_i * gamma_i * inv_rms_i) >>> output_shift"
    assert report["numeric_policy"]["fixture_kind"] == "rmsnorm_apply_fixture"
    assert report["numeric_policy"]["inv_rms_source"] == "python_golden_metadata"
    assert report["numeric_policy"]["reciprocal_sqrt_in_rtl"] is False
    assert report["sumsq_expected"] == 5568
    assert report["inv_rms_source"] == "python_golden_metadata"
    assert report["expected_outputs"] == [658, -1372, 1152, -659]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_rope_target_kernel_simulates_and_reports_fixture_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rope_target", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rope_target" in result.simulation["output"]
    assert (tmp_path / "rope_target_golden.json").exists()
    sv_text = (tmp_path / "rope_target.sv").read_text(encoding="utf-8")
    assert "input  logic                                      aclk" in sv_text
    assert "input  logic                                      aresetn" in sv_text
    assert "input  logic                                      start_i" in sv_text
    assert "output logic                                      done_o" in sv_text
    assert "input_i[(int'(pair_idx_r) * 2) * IN_WIDTH +: IN_WIDTH]" in sv_text
    assert "cos_i[int'(pair_idx_r) * COS_SIN_WIDTH +: COS_SIN_WIDTH]" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "rope_target_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["formula"].startswith("out_even =")
    assert report["numeric_policy"]["fixture_kind"] == "rope_apply_fixture"
    assert report["numeric_policy"]["cos_sin_source"] == "python_golden_metadata"
    assert report["numeric_policy"]["frequency_generation_in_rtl"] is False
    assert report["numeric_policy"]["lookup_table_in_rtl"] is False
    assert report["numeric_policy"]["rounding_mode"] == "arithmetic_shift_truncates_toward_negative_infinity"
    assert report["position"] == 7
    assert report["cos_sin_source"] == "python_golden_metadata"
    assert report["expected_outputs"] == [37, -13, 3, -42]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_rmsnorm_rope_source_path_uses_internal_lookup_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NL2HDL_SELECTED_NONGEMM", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm_rope_source_path", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rmsnorm_rope_source_path" in result.simulation["output"]
    assert "RMS_LOOKUP_TRACE rmsnorm_rope_source_path selector=0 valid=1 inv_rms=7024 sumsq=5568" in result.simulation[
        "output"
    ]
    assert "ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=7 pair=0 valid=1 cos=13 sin=9" in result.simulation[
        "output"
    ]
    assert "ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=7 pair=1 valid=1 cos=7 sin=-14" in result.simulation[
        "output"
    ]
    assert (tmp_path / "rmsnorm_rope_source_path_golden.json").exists()

    sv_text = (tmp_path / "rmsnorm_rope_source_path.sv").read_text(encoding="utf-8")
    assert "module rmsnorm_rope_source_path #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module rmsnorm_rope_source_path #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "norm_token_i" in top_port_text
    assert "rope_position_i" in top_port_text
    assert "inv_rms_i" not in top_port_text
    assert "cos_i" not in top_port_text
    assert "sin_i" not in top_port_text
    assert "lookup_inv_rms" in sv_text
    assert "lookup_rope_cos" in sv_text
    assert "lookup_rope_sin" in sv_text

    golden = json.loads((tmp_path / "rmsnorm_rope_source_path_golden.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "rmsnorm_rope_source_path"
    assert report["coverage_level"] == "rmsnorm_rope_source_path_fixture"
    assert report["implementation_stage"] == "not_run"
    for artifact in (report, golden):
        assert artifact["selected_non_gemm"] == "input_layernorm"
        assert artifact["selected_non_gemm_op_type"] == "RMSNorm"
        assert artifact["selected_non_gemm_shape"] == {"hidden_size": 2048}
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "input_layernorm"
        assert artifact["target_fixture_distinction"]["fixture_rmsnorm_element_count"] == 4
        assert artifact["target_fixture_distinction"]["fixture_rope_element_count"] == 4
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "bounded 4-element RMSNorm/RoPE source-path fixture evidence" in artifact[
            "target_fixture_distinction"
        ]["distinction"]
    assert report["top_level_interface"]["forbidden_direct_metadata_ports_absent"] is True
    assert report["rmsnorm_source_path"]["inv_rms_source"] == "rtl_lookup_fixture"
    assert report["rmsnorm_source_path"]["reciprocal_sqrt_in_rtl"] is False
    assert report["rmsnorm_source_path"]["sumsq_expected"] == 5568
    assert report["rmsnorm_source_path"]["lookup_trace"]["inv_rms_lookup_value"] == 7024
    assert report["rmsnorm_source_path_observed_trace"]["inv_rms_lookup_value"] == 7024
    assert report["rmsnorm_source_path"]["expected_output_vector"] == [658, -1372, 1152, -659]
    assert report["rmsnorm_source_path"]["observed_output_vector"] == [658, -1372, 1152, -659]
    assert report["rmsnorm_source_path_observed_output_vector"] == [658, -1372, 1152, -659]
    assert report["rmsnorm_source_path"]["observed_output_vector_evidence_source"] == (
        "parsed_from_iverilog_output_trace"
    )
    assert report["rmsnorm_source_path_output_trace"]["evidence_source"] == "parsed_from_iverilog_output_trace"
    assert report["rope_source_path"]["cos_sin_source"] == "rtl_lookup_fixture"
    assert report["rope_source_path"]["lookup_table_in_rtl"] is True
    assert report["rope_source_path"]["frequency_generation_in_rtl"] is False
    assert report["rope_source_path"]["nonzero_sine_values"] is True
    assert report["rope_source_path"]["lookup_trace"][0]["cos"] == 13
    assert report["rope_source_path"]["lookup_trace"][1]["sin"] == -14
    assert report["rope_source_path_observed_trace"][1]["sin"] == -14
    assert report["rope_source_path"]["expected_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path"]["observed_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path_observed_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path"]["observed_output_vector_evidence_source"] == (
        "parsed_from_iverilog_output_trace"
    )
    assert report["rope_source_path_output_trace"]["evidence_source"] == "parsed_from_iverilog_output_trace"
    assert report["reciprocal_sqrt_in_rtl"] is False
    assert report["lookup_table_in_rtl"] is True
    assert report["frequency_generation_in_rtl"] is False
    assert "full_RMSNorm_reciprocal_sqrt_datapath" in report["does_not_claim"]
    assert "RoPE_frequency_generation" in report["does_not_claim"]
    assert "full_model_execution" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_rmsnorm_rope_source_path_uses_env_selected_input_layernorm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "input_layernorm")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm_rope_source_path", cfg, tmp_path)
    assert result.status == "passed"

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "rmsnorm_rope_source_path_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_non_gemm"] == "input_layernorm"
        assert artifact["selected_non_gemm_op_type"] == "RMSNorm"
        assert artifact["selected_non_gemm_shape"] == {"hidden_size": 2048}
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "input_layernorm"
        assert artifact["target_fixture_distinction"]["target_source_path"] == "rmsnorm_source_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
    assert report["rmsnorm_source_path"]["expected_output_vector"] == [658, -1372, 1152, -659]
    assert report["rope_source_path"]["expected_output_vector"] == [37, -13, 3, -42]


def test_emit_rmsnorm_rope_source_path_uses_env_selected_post_attention_layernorm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "post_attention_layernorm")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm_rope_source_path", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rmsnorm_rope_source_path" in result.simulation["output"]
    assert "RMS_LOOKUP_TRACE rmsnorm_rope_source_path selector=0 valid=1 inv_rms=7024 sumsq=5568" in result.simulation[
        "output"
    ]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "rmsnorm_rope_source_path_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_non_gemm"] == "post_attention_layernorm"
        assert artifact["selected_non_gemm_op_type"] == "RMSNorm"
        assert artifact["selected_non_gemm_shape"] == {"hidden_size": 2048}
        assert artifact["target_source_path"] == "rmsnorm_source_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "post_attention_layernorm"
        assert artifact["target_fixture_distinction"]["target_source_path"] == "rmsnorm_source_path"
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "selected non-GEMM op is post_attention_layernorm" in artifact["target_fixture_distinction"][
            "distinction"
        ]
    assert report["rmsnorm_source_path"]["expected_output_vector"] == [658, -1372, 1152, -659]
    assert report["rmsnorm_source_path"]["observed_output_vector"] == [658, -1372, 1152, -659]
    assert report["rmsnorm_source_path_output_trace"]["observed_output_vector"] == [658, -1372, 1152, -659]
    assert report["rope_source_path"]["expected_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path"]["observed_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path_output_trace"]["observed_output_vector"] == [37, -13, 3, -42]


def test_emit_rmsnorm_rope_source_path_uses_env_selected_rope_qk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "rope_qk")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("rmsnorm_rope_source_path", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS rmsnorm_rope_source_path" in result.simulation["output"]
    assert "ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=7 pair=0 valid=1 cos=13 sin=9" in result.simulation[
        "output"
    ]
    assert "ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=7 pair=1 valid=1 cos=7 sin=-14" in result.simulation[
        "output"
    ]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    golden = json.loads((tmp_path / "rmsnorm_rope_source_path_golden.json").read_text(encoding="utf-8"))
    for artifact in (report, golden):
        assert artifact["selected_non_gemm"] == "rope_qk"
        assert artifact["selected_non_gemm_op_type"] == "RoPE"
        assert artifact["selected_non_gemm_shape"] == {
            "head_dim": 64,
            "attention_heads": 32,
            "key_value_heads": 8,
        }
        assert artifact["target_source_path"] == "rope_source_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "rope_qk"
        assert artifact["target_fixture_distinction"]["target_source_path"] == "rope_source_path"
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
    assert report["rope_source_path"]["cos_sin_source"] == "rtl_lookup_fixture"
    assert report["rope_source_path"]["lookup_table_in_rtl"] is True
    assert report["rope_source_path"]["frequency_generation_in_rtl"] is False
    assert report["rope_source_path"]["expected_output_vector"] == [37, -13, 3, -42]
    assert report["rope_source_path"]["observed_output_vector"] == [37, -13, 3, -42]


def test_emit_rmsnorm_rope_source_path_rejects_invalid_selected_non_gemm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "bogus_non_gemm")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    with pytest.raises(ValueError, match="NL2HDL_SELECTED_NONGEMM='bogus_non_gemm'"):
        emit_kernel("rmsnorm_rope_source_path", cfg, tmp_path)

    assert not (tmp_path / "rmsnorm_rope_source_path.sv").exists()
    assert not (tmp_path / "rmsnorm_rope_source_path_golden.json").exists()
    assert not (tmp_path / "kernel_report.json").exists()


def test_emit_attention_kv_cache_fixture_reports_attention_and_cache_traces(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NL2HDL_SELECTED_NONGEMM", raising=False)
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("attention_kv_cache_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS attention_kv_cache_fixture" in result.simulation["output"]
    assert (
        "CACHE_WRITE_TRACE attention_kv_cache_fixture count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1"
        in result.simulation["output"]
    )
    assert "KEY_READ_TRACE attention_kv_cache_fixture slots=0,1 keys=2,-1,4,3|-3,2,1,6" in result.simulation[
        "output"
    ]
    assert "VALUE_READ_TRACE attention_kv_cache_fixture slots=0,1 values=7,-4,3,2|-2,6,-5,4" in result.simulation[
        "output"
    ]
    assert "SCORE_TRACE attention_kv_cache_fixture scores=25,-14" in result.simulation["output"]
    assert (
        "SOFTMAX_CONTROL_TRACE attention_kv_cache_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0"
        in result.simulation["output"]
    )
    assert "OUTPUT_TRACE attention_kv_cache_fixture output=4,-2,1,2" in result.simulation["output"]
    assert (tmp_path / "attention_kv_cache_fixture_golden.json").exists()
    golden = json.loads((tmp_path / "attention_kv_cache_fixture_golden.json").read_text(encoding="utf-8"))

    sv_text = (tmp_path / "attention_kv_cache_fixture.sv").read_text(encoding="utf-8")
    assert "module attention_kv_cache_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module attention_kv_cache_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output_o" in top_port_text
    assert "status_o" in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "value_cache_i" not in top_port_text
    assert "attention_tensor_i" not in top_port_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "attention_kv_cache_fixture"
    assert report["coverage_level"] == "attention_kv_cache_fixture"
    assert report["implementation_stage"] == "not_run"
    for artifact in (golden, report):
        assert artifact["selected_non_gemm"] == "attention_scores_softmax_kv"
        assert artifact["selected_non_gemm_op_type"] == "AttentionControl"
        assert artifact["selected_non_gemm_shape"] == {
            "head_dim": 64,
            "attention_heads": 32,
            "key_value_heads": 8,
            "sequence_length": 2048,
        }
        assert artifact["target_source_path"] == "attention_kv_cache_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "attention_scores_softmax_kv"
        assert artifact["target_fixture_distinction"]["selected_non_gemm_op_type"] == "AttentionControl"
        assert artifact["target_fixture_distinction"]["target_source_path"] == "attention_kv_cache_path"
        assert artifact["target_fixture_distinction"]["fixture_head_dim"] == 4
        assert artifact["target_fixture_distinction"]["fixture_cache_slots"] == 2
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "bounded 4-lane, 2-slot attention KV-cache fixture" in artifact["target_fixture_distinction"][
            "distinction"
        ]
    assert report["fixture_dimensions"]["head_dim"] == 4
    assert report["fixture_dimensions"]["cache_slots"] == 2
    assert report["fixture_dimensions"]["score_count"] == 2
    assert report["fixture_dimensions"]["output_elements"] == 4
    assert report["top_level_interface"]["compact_top_level_io"] is True
    assert report["top_level_interface"]["full_target_sequence_length_kv_cache_arrays_exposed"] is False
    assert report["cache_write_trace"]["slot"] == 1
    assert report["observed_cache_write_trace"]["field_stability_checked"] is True
    assert report["observed_key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["observed_value_read_trace"][1]["value"] == [-2, 6, -5, 4]
    assert report["attention_score_trace"] == [25, -14]
    assert report["observed_attention_score_trace"] == [25, -14]
    assert report["softmax_policy"] == "two_score_winner_loser_q0_4"
    assert report["softmax_exp_in_rtl"] is False
    assert report["observed_softmax_control_trace"]["weights"] == [12, 4]
    assert report["expected_output_vector"] == [4, -2, 1, 2]
    assert report["observed_output_vector"] == [4, -2, 1, 2]
    assert report["kv_cache_storage"] == "internal_register_fixture_two_slots"
    assert report["kv_cache_external_memory"] is False
    assert "full_attention" in report["does_not_claim"]
    assert "true_exponential_softmax" in report["does_not_claim"]
    assert "DDR_AXI_KV_cache_movement" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_attention_kv_cache_fixture_rejects_invalid_selected_non_gemm(tmp_path: Path, monkeypatch):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "bogus_non_gemm")

    with pytest.raises(ValueError, match="NL2HDL_SELECTED_NONGEMM='bogus_non_gemm'"):
        emit_kernel("attention_kv_cache_fixture", cfg, tmp_path)

    assert not (tmp_path / "attention_kv_cache_fixture_golden.json").exists()
    assert not (tmp_path / "kernel_report.json").exists()
    assert not (tmp_path / "attention_kv_cache_fixture.sv").exists()
    assert not (tmp_path / "tb_attention_kv_cache_fixture.sv").exists()


def test_emit_decoder_block_simulates(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_block", cfg, tmp_path)
    assert result.status == "passed"
    assert "decoder_block.sv" in result.files


def test_emit_decoder_block_scaffold_reports_sequence_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_block_scaffold", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_block_scaffold" in result.simulation["output"]
    assert "decoder_block_scaffold.sv" in result.files
    assert (tmp_path / "decoder_block_scaffold_golden.json").exists()
    sv_text = (tmp_path / "decoder_block_scaffold.sv").read_text(encoding="utf-8")
    assert "input  logic                                    aclk" in sv_text
    assert "input  logic                                    aresetn" in sv_text
    assert "input  logic                                    start_i" in sv_text
    assert "output logic                                    done_o" in sv_text
    assert "RMSNORM_START" in sv_text
    assert "PROJECTION_TILE_START" in sv_text
    assert "ROPE_START" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "decoder_block_scaffold_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["fixture_kind"] == "decoder_block_sequencing_scaffold"
    assert report["numeric_policy"]["datapath_composition"] == "not_instantiated"
    assert report["child_modules"][0]["name"] == "rmsnorm_target"
    assert report["child_modules"][1]["name"] == "projection_tile"
    assert report["child_modules"][2]["name"] == "rope_target"
    assert report["fsm_state_order"] == [
        "IDLE",
        "RMSNORM_START",
        "RMSNORM_BUSY",
        "PROJECTION_TILE_START",
        "PROJECTION_TILE_BUSY",
        "ROPE_START",
        "ROPE_BUSY",
        "DONE",
    ]
    assert "KV-cache movement" in report["omitted_operations"]
    assert report["expected_final_trace"] == "0x332211"
    assert report["expected_outputs"] == [37, -13, 3, -42]
    assert report["sequencing_scaffold"] is True
    assert report["datapath_child_instantiation"] is False
    assert "full_llama_coverage" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_decoder_child_datapath_instantiates_child_kernels(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_child_datapath", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_child_datapath" in result.simulation["output"]
    assert "CHILD_TRACE decoder_child_datapath" in result.simulation["output"]
    assert "decoder_child_datapath.sv" in result.files
    assert "rmsnorm_target.sv" in result.files
    assert "projection_tile.sv" in result.files
    assert "rope_target.sv" in result.files
    assert (tmp_path / "decoder_child_datapath_golden.json").exists()
    sv_text = (tmp_path / "decoder_child_datapath.sv").read_text(encoding="utf-8")
    assert "input  logic                                    aclk" in sv_text
    assert "input  logic                                    aresetn" in sv_text
    assert "input  logic                                    start_i" in sv_text
    assert "output logic                                    done_o" in sv_text
    assert "rmsnorm_target u_rmsnorm_target" in sv_text
    assert "projection_tile u_projection_tile" in sv_text
    assert "rope_target u_rope_target" in sv_text
    assert ".start_i(rmsnorm_start_r)" in sv_text
    assert ".done_o(projection_done_w)" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "decoder_child_datapath_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["fixture_kind"] == "decoder_child_datapath_fixture"
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_child_kernels"
    assert [child["name"] for child in report["child_modules"]] == [
        "rmsnorm_target",
        "projection_tile",
        "rope_target",
    ]
    assert all(child["instantiated"] is True for child in report["child_modules"])
    assert report["fsm_state_order"] == [
        "IDLE",
        "RMSNORM_START",
        "RMSNORM_BUSY",
        "PROJECTION_TILE_START",
        "PROJECTION_TILE_BUSY",
        "ROPE_START",
        "ROPE_BUSY",
        "DONE",
    ]
    assert report["child_start_done_trace"] == [
        "rmsnorm_start",
        "rmsnorm_done",
        "projection_tile_start",
        "projection_tile_done",
        "rope_start",
        "rope_done",
    ]
    assert report["simulation_child_start_done_trace"]["recorded"] is True
    assert report["simulation_child_start_done_trace"]["trace_hex"] == "0x323122211211"
    assert "KV-cache movement" in report["omitted_operations"]
    assert report["expected_child_trace"] == "0x323122211211"
    assert report["expected_outputs"] == [-61, 24, 3, -42]
    assert report["datapath_child_instantiation"] is True
    assert "full_llama_coverage" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_decoder_child_attention_datapath_instantiates_attention_child_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_child_attention_datapath", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_child_attention_datapath" in result.simulation["output"]
    assert "CHILD_TRACE decoder_child_attention_datapath trace_hex=0x323122211211" in result.simulation["output"]
    assert "RMS_LOOKUP_TRACE decoder_child_attention_datapath selector=0 valid=1 inv_rms=7024 sumsq=5568" in result.simulation[
        "output"
    ]
    assert "ROPE_LOOKUP_TRACE decoder_child_attention_datapath position=7 pair=1 valid=1 cos=7 sin=-14" in result.simulation[
        "output"
    ]
    assert "PROJECTION_STREAM_TRACE decoder_child_attention_datapath" in result.simulation["output"]
    assert "request_accepted=1 response_count=4" in result.simulation["output"]
    assert "payload_emit_count=16 projection_consume_count=16 payload_match=1 source=hierarchical_child_status" in result.simulation[
        "output"
    ]
    assert "PROJECTION_OUTPUT_TRACE decoder_child_attention_datapath output=976,2360" in result.simulation["output"]
    assert "CACHE_WRITE_TRACE decoder_child_attention_datapath count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "KEY_READ_TRACE decoder_child_attention_datapath slots=0,1 keys=2,-1,4,3|-3,2,1,6" in result.simulation[
        "output"
    ]
    assert "VALUE_READ_TRACE decoder_child_attention_datapath slots=0,1 values=7,-4,3,2|-2,6,-5,4" in result.simulation[
        "output"
    ]
    assert "SCORE_TRACE decoder_child_attention_datapath scores=25,-14" in result.simulation["output"]
    assert "SOFTMAX_CONTROL_TRACE decoder_child_attention_datapath policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE decoder_child_attention_datapath output=4,-2,1,2" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE decoder_child_attention_datapath stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE decoder_child_attention_datapath estimated_iob_bits=164 exposed_128b=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert (tmp_path / "decoder_child_attention_datapath_golden.json").exists()

    sv_text = (tmp_path / "decoder_child_attention_datapath.sv").read_text(encoding="utf-8")
    assert "module decoder_child_attention_datapath #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module decoder_child_attention_datapath #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "value_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "rmsnorm_rope_source_path u_rmsnorm_rope_source_path" in sv_text
    assert "projection_internal_stream_shell u_projection_internal_stream_shell" in sv_text
    assert "attention_kv_cache_fixture u_attention_kv_cache_fixture" in sv_text
    assert ".start_i(source_path_start_r)" in sv_text
    assert ".start_i(projection_shell_start_r)" in sv_text
    assert ".start_i(attention_kv_start_r)" in sv_text
    assert "SOURCE_PATH_START" in sv_text
    assert "PROJECTION_SHELL_START" in sv_text
    assert "ATTENTION_KV_START" in sv_text

    tb_text = (tmp_path / "tb_decoder_child_attention_datapath.sv").read_text(encoding="utf-8")
    assert "observed_write_key <= dut.u_attention_kv_cache_fixture.cache_write_key_r" in tb_text
    assert "observed_write_value <= dut.u_attention_kv_cache_fixture.cache_write_value_r" in tb_text
    assert "observed_key0 <= dut.u_attention_kv_cache_fixture.key_read_data_r" in tb_text
    assert "observed_value1 <= dut.u_attention_kv_cache_fixture.value_read_data_r" in tb_text
    assert "lane_s8(observed_write_key, 0)" in tb_text
    assert "lane_s16(dut.u_attention_kv_cache_fixture.output_o, 0)" in tb_text
    assert "u_projection_internal_stream_shell.u_boundary.req_accepted_r" in tb_text
    assert "u_projection_internal_stream_shell.u_boundary.response_accepted_trace_w" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text
    assert "ATTENTION_OUTPUT_TRACE decoder_child_attention_datapath output=4,-2,1,2" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "decoder_child_attention_datapath"
    assert report["coverage_level"] == "decoder_child_attention_datapath_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["previous_gap_closed"] == "attention_kv_fixture_in_child_datapath"
    assert report["datapath_child_instantiation"] is True
    assert report["attention_kv_child_instantiation"] is True
    assert [child["name"] for child in report["child_modules"]] == [
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["child_modules"])
    assert report["child_sequence"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["simulation_child_start_done_trace"]["recorded"] is True
    assert report["simulation_child_start_done_trace"]["trace_hex"] == "0x323122211211"
    assert report["observed_child_start_done_trace"] == report["child_sequence"]
    assert report["source_path_rmsnorm_observed_trace"]["inv_rms"] == 7024
    assert report["source_path_rope_observed_trace"][1]["sin"] == -14
    assert report["projection_shell_observed_trace"]["projection_output_vector"] == [976, 2360]
    assert report["projection_shell_observed_stream_status"]["request_accepted"] is True
    assert report["projection_shell_observed_stream_status"]["response_count"] == 4
    assert report["projection_shell_observed_stream_status"]["payload_emit_count"] == 16
    assert report["projection_shell_observed_stream_status"]["projection_consume_count"] == 16
    assert report["projection_shell_observed_stream_status"]["payload_match"] is True
    assert report["projection_shell_observed_stream_status"]["evidence_source"] == "hierarchical_child_status"
    assert report["projection_shell_evidence"]["reused_child_fixture_metadata_summary"]["payload_link_match_passed"] is True
    assert (
        report["projection_shell_evidence"]["reused_child_fixture_metadata_summary"]["evidence_source"]
        == "reused_projection_internal_stream_shell_child_report_metadata"
    )
    assert report["attention_kv_observed_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_observed_key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["attention_kv_observed_value_read_trace"][1]["value"] == [-2, 6, -5, 4]
    assert report["attention_kv_observed_score_trace"] == [25, -14]
    assert report["attention_kv_observed_softmax_control_trace"]["weights"] == [12, 4]
    assert report["attention_kv_observed_softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["softmax_exp_in_rtl"] is False
    assert report["kv_cache_external_memory"] is False
    assert report["compact_io_observed_trace"]["estimated_iob_bits"] == 164
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 164
    assert report["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
    assert report["exposed_port_width_summary"]["exposes_full_kv_arrays"] is False
    assert "mathematically complete Q/K/V/O projection-to-attention wiring" in report["omitted_operations"]
    assert "full decoder block" in report["omitted_operations"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_layer_fsm_axi_attention_fixture_sequences_axi_child_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("layer_fsm_axi_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS layer_fsm_axi_attention_fixture" in result.simulation["output"]
    assert "LAYER_AXI_ATTENTION_TRACE layer_fsm_axi_attention_fixture layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "events=decoder_child_axi_attention_datapath_start,decoder_child_axi_attention_datapath_done" in result.simulation[
        "output"
    ]
    assert "CHILD_TRACE layer_fsm_axi_attention_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_AR_TRACE layer_fsm_axi_attention_fixture addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_R_METADATA_TRACE layer_fsm_axi_attention_fixture accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1" in result.simulation[
        "output"
    ]
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE layer_fsm_axi_attention_fixture parent_bits=0xf" in result.simulation[
        "output"
    ]
    assert "parent_bit_lsb=76 parent_bit_msb=79 child_status_bits=0xf" in result.simulation["output"]
    assert "AXI_PROJECTION_BACKPRESSURE_TRACE layer_fsm_axi_attention_fixture ready_low_payload_idx=0,3,4 trace=0x19 payload_hold_ok=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_ROUND_TRIP_TRACE layer_fsm_axi_attention_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1" in result.simulation[
        "output"
    ]
    assert "CACHE_WRITE_TRACE layer_fsm_axi_attention_fixture count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SOFTMAX_CONTROL_TRACE layer_fsm_axi_attention_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE layer_fsm_axi_attention_fixture output=4,-2,1,2" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE layer_fsm_axi_attention_fixture stable=1" in result.simulation["output"]
    assert "CHILD_START_HOLD_TRACE layer_fsm_axi_attention_fixture" in result.simulation["output"]
    assert "deasserted_after_done=1" in result.simulation["output"]
    assert "child_done_release_seen_after_start_deassert=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE layer_fsm_axi_attention_fixture estimated_iob_bits=180 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0" in result.simulation[
        "output"
    ]

    assert "layer_fsm_axi_attention_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_child_axi_attention_datapath.sv" not in result.files
    assert (tmp_path / "layer_fsm_axi_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "layer_fsm_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module layer_fsm_axi_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module layer_fsm_axi_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "axi_rdata_i" not in top_port_text
    assert "axi_araddr_o" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "decoder_child_axi_attention_datapath u_decoder_child_axi_attention_datapath" in sv_text
    assert "CHILD_RELEASE" in sv_text
    assert "if (!child_done_w) begin" in sv_text
    assert "status_r <= {child_status_w, 8'h42, layer_trace_r[7:0]}" in sv_text

    tb_text = (tmp_path / "tb_layer_fsm_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while child busy" in tb_text
    assert "child start was not deasserted after child done_o" in tb_text
    assert "child_done_release_seen_after_start_deassert_r" in tb_text
    assert "child_done_release_seen_after_start_deassert=%0d" in tb_text
    assert "u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rid_w" in tb_text
    assert "status_vec[76 +: 4] != dut.child_status_w[63:60]" in tb_text
    assert "dut.child_status_w[63:60] != dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE layer_fsm_axi_attention_fixture" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "layer_fsm_axi_attention_fixture"
    assert report["coverage_level"] == "layer_fsm_axi_attention_fixture"
    assert report["uses_axi_decoder_child"] is True
    assert report["layer_fsm_fixture"] is True
    assert report["datapath_child_instantiation"] is True
    assert report["child_modules"][0]["name"] == "decoder_child_axi_attention_datapath"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["layer_start_done_trace"] == [
        "decoder_child_axi_attention_datapath_start",
        "decoder_child_axi_attention_datapath_done",
    ]
    assert report["simulation_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_layer_start_done_trace"]["layer_trace_hex"] == "0x4241"
    assert report["child_datapath_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    propagation = report["parent_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["parent_status_bits_hex"] == "0xf"
    assert propagation["child_status_bits_hex"] == "0xf"
    assert propagation["parent_bit_lsb"] == 76
    assert propagation["parent_bit_msb"] == 79
    assert propagation["bit_mapping"] == {
        "rid_good": 76,
        "rresp_good": 77,
        "rlast_good": 78,
        "all_metadata_good": 79,
    }
    assert report["child_start_hold_protocol_observed"]["done_seen_while_start_high"] is True
    assert report["child_start_hold_protocol_observed"]["deasserted_after_done"] is True
    assert report["child_start_hold_protocol_observed"]["child_done_release_seen_after_start_deassert"] is True
    assert report["child_done_release_seen_after_start_deassert"] is True
    assert report["axi_projection_child_evidence_summary"]["r_metadata_trace"]["rid_ok"] is True
    assert report["axi_projection_child_evidence_summary"]["payload_trace"]["payload_link_match_passed"] is True
    assert report["axi_projection_child_evidence_summary"]["round_trip_evidence"]["round_trip_passed"] is True
    assert report["attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_evidence_summary"]["softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["softmax_exp_in_rtl"] is False
    assert report["kv_cache_external_memory"] is False
    assert report["compact_io_observed_trace"]["estimated_iob_bits"] == 180
    assert report["exposed_port_width_summary"]["exposes_128b_axi_read_data"] is False
    assert "Top FSM scheduling" in report["omitted_operations"]


def test_emit_top_fsm_axi_attention_fixture_sequences_axi_layer_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("top_fsm_axi_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS top_fsm_axi_attention_fixture" in result.simulation["output"]
    assert "TOP_AXI_ATTENTION_TRACE top_fsm_axi_attention_fixture top_trace_hex=0x5453 layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "events=layer_fsm_axi_attention_fixture_start,layer_fsm_axi_attention_fixture_done" in result.simulation[
        "output"
    ]
    assert "LAYER_AXI_ATTENTION_TRACE top_fsm_axi_attention_fixture layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_AR_TRACE top_fsm_axi_attention_fixture addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_R_METADATA_TRACE top_fsm_axi_attention_fixture accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1" in result.simulation[
        "output"
    ]
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_attention_fixture parent_bits=0xf" in result.simulation[
        "output"
    ]
    assert "parent_bit_lsb=92 parent_bit_msb=95 child_status_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation["output"]
    assert "AXI_PROJECTION_BACKPRESSURE_TRACE top_fsm_axi_attention_fixture ready_low_payload_idx=0,3,4 trace=0x19 payload_hold_ok=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1" in result.simulation[
        "output"
    ]
    assert "CACHE_WRITE_TRACE top_fsm_axi_attention_fixture count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SOFTMAX_CONTROL_TRACE top_fsm_axi_attention_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE top_fsm_axi_attention_fixture output=4,-2,1,2" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE top_fsm_axi_attention_fixture stable=1" in result.simulation["output"]
    assert "LAYER_START_HOLD_TRACE top_fsm_axi_attention_fixture" in result.simulation["output"]
    assert "deasserted_after_done=1" in result.simulation["output"]
    assert "child_done_release_seen_after_start_deassert=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE top_fsm_axi_attention_fixture estimated_iob_bits=196 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0" in result.simulation[
        "output"
    ]

    assert "top_fsm_axi_attention_fixture.sv" in result.files
    assert "layer_fsm_axi_attention_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_layer_fsm_axi_attention_fixture.sv" not in result.files
    assert (tmp_path / "top_fsm_axi_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "top_fsm_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module top_fsm_axi_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module top_fsm_axi_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "axi_rdata_i" not in top_port_text
    assert "axi_araddr_o" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "layer_fsm_axi_attention_fixture u_layer_fsm_axi_attention_fixture" in sv_text
    assert ".start_i(layer_start_r)" in sv_text
    assert ".done_o(layer_done_w)" in sv_text
    assert "LAYER_RELEASE" in sv_text
    assert "if (!layer_done_w) begin" in sv_text
    assert "status_r <= {layer_status_w, 8'h54, top_trace_r[7:0]}" in sv_text

    tb_text = (tmp_path / "tb_top_fsm_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "layer start was not held while layer busy" in tb_text
    assert "layer start was not deasserted after layer done_o" in tb_text
    assert "layer_done_release_seen_after_start_deassert_r" in tb_text
    assert "dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "status_vec[92 +: 4] != dut.layer_status_w[79:76]" in tb_text
    assert "dut.layer_status_w[79:76] != dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_attention_fixture" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "top_fsm_axi_attention_fixture"
    assert report["coverage_level"] == "top_fsm_axi_attention_fixture"
    assert report["top_fsm_axi_attention_fixture"] is True
    assert report["uses_axi_layer_child"] is True
    assert report["layer_child_instantiation"] is True
    assert report["child_modules"][0]["name"] == "layer_fsm_axi_attention_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["fsm_state_order"] == ["IDLE", "LAYER_START", "LAYER_BUSY", "LAYER_RELEASE", "DONE"]
    assert report["simulation_top_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_top_layer_start_done_trace"]["top_trace_hex"] == "0x5453"
    assert report["layer_fsm_trace"]["ordered_events"] == [
        "decoder_child_axi_attention_datapath_start",
        "decoder_child_axi_attention_datapath_done",
    ]
    propagation = report["top_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["top_status_bits_hex"] == "0xf"
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["nested_axi_status_bits_hex"] == "0xf"
    assert propagation["top_bit_lsb"] == 92
    assert propagation["top_bit_msb"] == 95
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert propagation["bit_mapping"] == {
        "rid_good": 92,
        "rresp_good": 93,
        "rlast_good": 94,
        "all_metadata_good": 95,
    }
    assert report["layer_child_start_hold_protocol_observed"]["done_seen_while_start_high"] is True
    assert report["layer_child_start_hold_protocol_observed"]["deasserted_after_done"] is True
    assert report["layer_child_start_hold_protocol_observed"]["child_done_release_seen_after_start_deassert"] is True
    assert report["nested_axi_projection_evidence_summary"]["r_metadata_trace"]["rid_ok"] is True
    assert report["nested_axi_projection_evidence_summary"]["payload_trace"]["payload_link_match_passed"] is True
    assert report["nested_axi_projection_evidence_summary"]["round_trip_evidence"]["round_trip_passed"] is True
    assert set(report["nested_axi_projection_evidence_summary"]["projection_children"]) == {"q", "k", "v", "o"}
    assert report["nested_attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["nested_attention_kv_evidence_summary"]["softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["softmax_exp_in_rtl"] is False
    assert report["kv_cache_external_memory"] is False
    assert report["compact_io_observed_trace"]["estimated_iob_bits"] == 196
    assert report["exposed_port_width_summary"]["exposes_128b_axi_read_data"] is False
    assert "token prefill/decode loop" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "token_prefill_decode_loop" in report["does_not_claim"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_decoder_child_axi_attention_datapath_instantiates_axi_projection_child_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_child_axi_attention_datapath", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_child_axi_attention_datapath" in result.simulation["output"]
    assert "CHILD_TRACE decoder_child_axi_attention_datapath trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done" in result.simulation[
        "output"
    ]
    assert "RMS_LOOKUP_TRACE decoder_child_axi_attention_datapath selector=0 valid=1 inv_rms=7024 sumsq=5568" in result.simulation[
        "output"
    ]
    assert "ROPE_LOOKUP_TRACE decoder_child_axi_attention_datapath position=7 pair=1 valid=1 cos=7 sin=-14" in result.simulation[
        "output"
    ]
    assert (
        "AXI_PROJECTION_AR_TRACE decoder_child_axi_attention_datapath addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2"
        in result.simulation["output"]
    )
    assert "AXI_PROJECTION_R_METADATA_TRACE decoder_child_axi_attention_datapath accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1" in result.simulation[
        "output"
    ]
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE decoder_child_axi_attention_datapath parent_bits=0xf" in result.simulation[
        "output"
    ]
    assert "parent_bit_lsb=60 parent_bit_msb=63 child_status_bits=0xf" in result.simulation["output"]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_AR_TRACE decoder_child_axi_attention_datapath projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_child_axi_attention_datapath projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_child_axi_attention_datapath projection={projection} emitted=8 consumed=8 payload_match=1"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_child_axi_attention_datapath projection={projection} output=484,1904"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_child_axi_attention_datapath projection={projection} packed_bytes=32 unpacked_values=64 round_trip_passed=1"
            in result.simulation["output"]
        )
    assert "AXI_PROJECTION_EMITTED_PAYLOADS decoder_child_axi_attention_datapath" in result.simulation["output"]
    assert "AXI_PROJECTION_CONSUMED_PAYLOADS decoder_child_axi_attention_datapath" in result.simulation["output"]
    assert "AXI_PROJECTION_BACKPRESSURE_TRACE decoder_child_axi_attention_datapath ready_low_payload_idx=0,3,4 trace=0x19 payload_hold_ok=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_OUTPUT_TRACE decoder_child_axi_attention_datapath output=484,1904" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_ROUND_TRIP_TRACE decoder_child_axi_attention_datapath packed_bytes=32 unpacked_values=64 round_trip_passed=1" in result.simulation[
        "output"
    ]
    assert "CACHE_WRITE_TRACE decoder_child_axi_attention_datapath count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SCORE_TRACE decoder_child_axi_attention_datapath scores=25,-14" in result.simulation["output"]
    assert "FINAL_OUTPUT_TRACE decoder_child_axi_attention_datapath output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "TOP_STABILITY_TRACE decoder_child_axi_attention_datapath stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE decoder_child_axi_attention_datapath estimated_iob_bits=164 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0" in result.simulation[
        "output"
    ]
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert (tmp_path / "decoder_child_axi_attention_datapath_golden.json").exists()

    sv_text = (tmp_path / "decoder_child_axi_attention_datapath.sv").read_text(encoding="utf-8")
    assert "module decoder_child_axi_attention_datapath #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module decoder_child_axi_attention_datapath #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "axi_rdata_i" not in top_port_text
    assert "axi_araddr_o" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "rmsnorm_rope_source_path u_rmsnorm_rope_source_path" in sv_text
    assert "projection_axi_stream_integration u_projection_axi_stream_integration" in sv_text
    assert "projection_axi_stream_integration u_k_projection_axi_stream_integration" in sv_text
    assert "projection_axi_stream_integration u_v_projection_axi_stream_integration" in sv_text
    assert "projection_axi_stream_integration u_o_projection_axi_stream_integration" in sv_text
    assert "attention_kv_cache_fixture u_attention_kv_cache_fixture" in sv_text
    assert ".start_i(source_path_start_r)" in sv_text
    assert ".start_i(projection_axi_start_r)" in sv_text
    assert ".start_i(attention_kv_start_r)" in sv_text
    assert "PROJECTION_AXI_START" in sv_text
    assert "assign projection_axi_all_done_w = projection_axi_done_w & k_projection_axi_done_w & v_projection_axi_done_w & o_projection_axi_done_w" in sv_text
    assert "assign projection_axi_metadata_good_w = projection_status_w[45:42] & k_projection_status_w[45:42] & v_projection_status_w[45:42] & o_projection_status_w[45:42]" in sv_text
    assert "projection_axi_metadata_good_r <= projection_axi_metadata_good_w" in sv_text
    assert "projection_axi_metadata_good_r, projection_axi_status_low_r" in sv_text

    tb_text = (tmp_path / "tb_decoder_child_axi_attention_datapath.sv").read_text(encoding="utf-8")
    assert "dut.u_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "dut.u_k_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "dut.u_v_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "dut.u_o_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "dut.u_projection_axi_stream_integration.axi_rid_w" in tb_text
    assert "dut.u_projection_axi_stream_integration.payload_link_word_r" in tb_text
    assert "dut.u_projection_axi_stream_integration.ready_low_trace_r" in tb_text
    assert "sign_extend_int4(dut.u_projection_axi_stream_integration.packed_weight_r" in tb_text
    assert "observed_write_key <= dut.u_attention_kv_cache_fixture.cache_write_key_r" in tb_text
    assert "lane_s16(dut.u_attention_kv_cache_fixture.output_o, 0)" in tb_text
    assert "dut.u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "aggregate_and" in tb_text
    assert "PARENT_AXI_METADATA_PROPAGATION_TRACE decoder_child_axi_attention_datapath" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "decoder_child_axi_attention_datapath"
    assert report["coverage_level"] == "decoder_child_axi_attention_datapath_fixture"
    assert report["uses_axi_projection_stream_child"] is True
    assert report["datapath_child_instantiation"] is True
    assert report["attention_kv_child_instantiation"] is True
    assert [child["name"] for child in report["child_modules"]] == [
        "rmsnorm_rope_source_path",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["child_modules"])
    assert report["child_sequence"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["simulation_child_start_done_trace"]["recorded"] is True
    assert report["simulation_child_start_done_trace"]["trace_hex"] == "0x323122211211"
    assert report["observed_child_start_done_trace"] == report["child_sequence"]
    assert report["source_path_rmsnorm_observed_trace"]["inv_rms"] == 7024
    assert report["source_path_rope_observed_trace"][1]["sin"] == -14
    assert report["axi_projection_child_observed_command_trace"]["addr_hex"] == "0x120000"
    assert report["axi_projection_child_observed_command_trace"]["axi_arlen"] == 1
    assert report["axi_projection_child_observed_r_metadata_trace"]["rid_ok"] is True
    assert report["axi_projection_child_observed_r_metadata_trace"]["rresp_ok"] is True
    assert report["axi_projection_child_observed_r_metadata_trace"]["rlast_ok"] is True
    propagation = report["parent_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["parent_status_bits_hex"] == "0xf"
    assert propagation["child_status_bits_hex"] == "0xf"
    assert propagation["parent_status_mask_hex"] == "0x00000000f000000000000000"
    assert propagation["parent_bit_lsb"] == 60
    assert propagation["parent_bit_msb"] == 63
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert propagation["child_source"].endswith(":aggregate_and")
    assert propagation["bit_mapping"] == {
        "rid_good": 60,
        "rresp_good": 61,
        "rlast_good": 62,
        "all_metadata_good": 63,
    }
    assert report["axi_projection_child_evidence"]["parent_compact_status_propagation_plan"] == {
        "child_source": "q/k/v/o projection_axi_stream_integration.integration_status_o[45:42] aggregate_and",
        "parent_status_bit_lsb": 60,
        "parent_status_bit_msb": 63,
        "parent_status_mask_hex": "0x00000000f000000000000000",
        "bit_mapping": {
            "rid_good": 60,
            "rresp_good": 61,
            "rlast_good": 62,
            "all_metadata_good": 63,
        },
    }
    assert [child["projection"] for child in report["axi_projection_child_instances"]] == ["q", "k", "v", "o"]
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    for projection in ("q", "k", "v", "o"):
        child_evidence = report["axi_projection_children_observed"][projection]
        assert child_evidence["command_trace"]["addr_hex"] == "0x120000"
        assert child_evidence["r_metadata_trace"]["rid_ok"] is True
        assert child_evidence["payload_trace"]["payload_link_match_passed"] is True
        assert child_evidence["output_trace"]["projection_output_vector"] == [484, 1904]
        assert child_evidence["round_trip_evidence"]["round_trip_passed"] is True
    assert report["axi_projection_child_payload_match_passed"] is True
    assert report["axi_projection_child_backpressure_trace"]["ready_low_payload_indices"] == [0, 3, 4]
    assert report["axi_projection_child_backpressure_trace"]["payload_valid_held_while_ready_low"] is True
    assert report["axi_projection_child_observed_output_trace"]["projection_output_vector"] == [484, 1904]
    assert report["axi_projection_child_round_trip_evidence"]["round_trip_passed"] is True
    assert report["attention_kv_observed_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_observed_key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["attention_kv_observed_score_trace"] == [25, -14]
    assert report["attention_kv_observed_softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["softmax_exp_in_rtl"] is False
    assert report["kv_cache_external_memory"] is False
    assert report["compact_io_observed_trace"]["estimated_iob_bits"] == 164
    assert report["exposed_port_width_summary"]["exposes_128b_axi_read_data"] is False
    assert report["exposed_port_width_summary"]["exposes_axi_address_id_response_debug"] is False
    assert "DDR controller integration" in report["omitted_operations"]
    assert "full qweight payload streaming" in report["omitted_operations"]
    assert "full decoder block" in report["omitted_operations"]
    assert "DDR_controller_integration" in report["does_not_claim"]
    assert "full_qweight_payload_streaming" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_decoder_child_axi_attention_datapath_propagates_sampled_axi_projection_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _sample_gptq_payload_probe()
    probe_words = probe["qweight_payload_words32_le_hex"]
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel("decoder_child_axi_attention_datapath", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS decoder_child_axi_attention_datapath" in result.simulation["output"]
    assert "AXI_PROJECTION_OUTPUT_TRACE decoder_child_axi_attention_datapath output=112,8" in result.simulation[
        "output"
    ]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    evidence = report["axi_projection_child_evidence"]
    assert evidence["gptq_payload_probe_used"] is True
    assert evidence["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert evidence["gptq_probe_qweight_payload_words32_le_hex"] == probe_words
    assert evidence["expected_payload_words_hex"] == probe_words
    assert evidence["emitted_payload_words_hex"] == probe_words
    assert evidence["projection_consumed_payload_words_hex"] == probe_words
    assert evidence["payload_words_match_gptq_probe"] is True
    assert evidence["observed_payload_trace"]["payload_link_match_passed"] is True
    assert evidence["projection_output_vector"] == [112, 8]
    assert report["axi_projection_child_payload_trace"]["emitted_payload_words_hex"] == probe_words
    assert report["axi_projection_child_payload_trace"]["consumed_payload_words_hex"] == probe_words
    assert report["axi_projection_child_payload_words_match_gptq_probe"] is True
    assert report["axi_projection_child_payload_match_passed"] is True
    assert "full qweight payload streaming" in report["omitted_operations"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


@pytest.mark.parametrize(
    ("kernel", "summary_key"),
    [
        ("layer_fsm_axi_attention_fixture", "axi_projection_child_evidence_summary"),
        ("top_fsm_axi_attention_fixture", "nested_axi_projection_evidence_summary"),
        ("token_loop_axi_attention_fixture", "nested_axi_projection_evidence_summary"),
    ],
)
def test_emit_upper_axi_attention_fixtures_propagate_sampled_projection_payload_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kernel: str,
    summary_key: str,
):
    probe = _sample_gptq_payload_probe()
    probe_words = probe["qweight_payload_words32_le_hex"]
    monkeypatch.setenv("NL2HDL_GPTQ_PAYLOAD_PROBE_JSON", json.dumps(probe))
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    result = emit_kernel(kernel, cfg, tmp_path)

    assert result.status == "passed"
    assert f"PASS {kernel}" in result.simulation["output"]
    assert f"AXI_PROJECTION_OUTPUT_TRACE {kernel} output=112,8" in result.simulation["output"]
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    summary = report[summary_key]
    assert summary["gptq_payload_probe_used"] is True
    assert summary["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert summary["gptq_probe_qweight_payload_words32_le_hex"] == probe_words
    assert summary["payload_words_match_gptq_probe"] is True
    assert summary["emitted_payload_words_hex"] == probe_words
    assert summary["projection_consumed_payload_words_hex"] == probe_words
    assert summary["payload_trace"]["emitted_payload_words_hex"] == probe_words
    assert summary["payload_trace"]["consumed_payload_words_hex"] == probe_words
    assert summary["payload_trace"]["payload_link_match_passed"] is True
    assert summary["projection_output_vector"] == [112, 8]
    assert summary["output_trace"]["projection_output_vector"] == [112, 8]
    assert "full qweight payload streaming" in report["omitted_operations"]
    assert "full_target_projection_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]


def test_emit_layer_fsm_attention_fixture_sequences_refreshed_child_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("layer_fsm_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS layer_fsm_attention_fixture" in result.simulation["output"]
    assert "LAYER_ATTENTION_TRACE layer_fsm_attention_fixture layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation["output"]
    assert "CHILD_TRACE layer_fsm_attention_fixture trace_hex=0x323122211211" in result.simulation["output"]
    assert "PROJECTION_STREAM_TRACE layer_fsm_attention_fixture" in result.simulation["output"]
    assert "payload_emit_count=16 projection_consume_count=16 payload_match=1 source=layer_fsm_hierarchical_child_status" in result.simulation[
        "output"
    ]
    assert "CACHE_WRITE_TRACE layer_fsm_attention_fixture count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SOFTMAX_CONTROL_TRACE layer_fsm_attention_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE layer_fsm_attention_fixture output=4,-2,1,2" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE layer_fsm_attention_fixture stable=1" in result.simulation["output"]
    assert "CHILD_START_HOLD_TRACE layer_fsm_attention_fixture" in result.simulation["output"]
    assert "deasserted_after_done=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE layer_fsm_attention_fixture estimated_iob_bits=180 exposed_128b=0 exposed_kv_arrays=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    assert "layer_fsm_attention_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_child_attention_datapath.sv" not in result.files
    assert (tmp_path / "layer_fsm_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "layer_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module layer_fsm_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module layer_fsm_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "value_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "decoder_child_attention_datapath u_decoder_child_attention_datapath" in sv_text
    assert ".start_i(child_start_r)" in sv_text
    assert ".done_o(child_done_w)" in sv_text
    assert "CHILD_DONE" in sv_text
    assert "child_start_r <= 1'b0;" in sv_text
    assert "done_r <= 1'b1;" in sv_text

    tb_text = (tmp_path / "tb_layer_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while child busy" in tb_text
    assert "child start was not deasserted after child done_o" in tb_text
    assert "u_decoder_child_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r" in tb_text
    assert "u_decoder_child_attention_datapath.u_projection_internal_stream_shell.u_boundary.req_accepted_r" in tb_text
    assert "lane_s8(observed_write_key, 0)" in tb_text
    assert "lane_s16(dut.u_decoder_child_attention_datapath.u_attention_kv_cache_fixture.output_o, 0)" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text
    assert "ATTENTION_OUTPUT_TRACE layer_fsm_attention_fixture output=4,-2,1,2" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "layer_fsm_attention_fixture"
    assert report["coverage_level"] == "layer_fsm_attention_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["layer_fsm_fixture"] is True
    assert report["uses_refreshed_decoder_child_attention_datapath"] is True
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_decoder_child_attention_datapath_fixture"
    assert report["child_modules"][0]["name"] == "decoder_child_attention_datapath"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["nested_child_coverage_summary"])
    assert report["fsm_state_order"] == ["IDLE", "CHILD_START", "CHILD_BUSY", "CHILD_DONE", "DONE"]
    assert report["layer_start_done_trace"] == [
        "decoder_child_attention_datapath_start",
        "decoder_child_attention_datapath_done",
    ]
    assert report["simulation_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_layer_start_done_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_layer_start_done_trace"]["child_trace_hex"] == "0x323122211211"
    assert report["observed_layer_start_done_trace"] == report["layer_start_done_trace"]
    assert report["child_datapath_trace"]["recorded"] is True
    assert report["child_datapath_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["child_datapath_trace"]["evidence_source"] == "parsed_from_layer_fsm_integration_testbench_child_status"
    assert report["child_start_hold_protocol_observed"]["done_seen_while_start_high"] is True
    assert report["child_start_hold_protocol_observed"]["deasserted_after_done"] is True
    assert report["source_path_evidence_summary"]["evidence_source"] == "parsed_from_layer_fsm_integration_testbench_hierarchical_child_signals"
    assert report["source_path_evidence_summary"]["rmsnorm_lookup_trace"]["inv_rms"] == 7024
    assert report["source_path_evidence_summary"]["rope_lookup_trace"][1]["sin"] == -14
    assert report["projection_shell_evidence_summary"]["evidence_source"] == "parsed_from_layer_fsm_integration_testbench_hierarchical_child_signals"
    assert report["projection_shell_evidence_summary"]["projection_output_vector"] == [976, 2360]
    assert report["projection_shell_evidence_summary"]["stream_status"]["request_accepted"] is True
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_emit_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["projection_consume_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_match"] is True
    assert report["attention_kv_evidence_summary"]["evidence_source"] == "parsed_from_layer_fsm_integration_testbench_hierarchical_child_signals"
    assert report["attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_evidence_summary"]["key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["attention_kv_evidence_summary"]["value_read_trace"][1]["value"] == [-2, 6, -5, 4]
    assert report["attention_kv_evidence_summary"]["attention_score_trace"] == [25, -14]
    assert report["attention_kv_evidence_summary"]["softmax_control_trace"]["weights"] == [12, 4]
    assert report["attention_kv_evidence_summary"]["softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["attention_kv_evidence_summary"]["attention_output_vector"] == [4, -2, 1, 2]
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 180
    assert report["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
    assert report["exposed_port_width_summary"]["exposes_full_kv_arrays"] is False
    assert report["compact_io_observed_trace"]["exposes_full_debug_arrays"] is False
    assert "Top FSM scheduling" in report["omitted_operations"]
    assert "token prefill/decode loop" in report["omitted_operations"]
    assert "DDR/AXI KV-cache movement" in report["omitted_operations"]
    assert "full LLaMA layer execution" in report["omitted_operations"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_layer_fsm_fixture_instantiates_decoder_child_datapath(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("layer_fsm_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS layer_fsm_fixture" in result.simulation["output"]
    assert "LAYER_TRACE layer_fsm_fixture" in result.simulation["output"]
    assert "layer_fsm_fixture.sv" in result.files
    assert "decoder_child_datapath.sv" in result.files
    assert "rmsnorm_target.sv" in result.files
    assert "projection_tile.sv" in result.files
    assert "rope_target.sv" in result.files
    assert (tmp_path / "layer_fsm_fixture_golden.json").exists()
    sv_text = (tmp_path / "layer_fsm_fixture.sv").read_text(encoding="utf-8")
    assert "input  logic                                    aclk" in sv_text
    assert "input  logic                                    aresetn" in sv_text
    assert "input  logic                                    start_i" in sv_text
    assert "output logic                                    done_o" in sv_text
    assert "decoder_child_datapath u_decoder_child_datapath" in sv_text
    assert ".start_i(child_start_r)" in sv_text
    assert ".done_o(child_done_w)" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "layer_fsm_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["fixture_kind"] == "layer_fsm_fixture"
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_decoder_child_datapath_fixture"
    assert report["child_modules"] == [
        {
            "name": "decoder_child_datapath",
            "coverage_level": "decoder_child_datapath_fixture",
            "instantiated": True,
            "input_source": "static_decoder_child_fixture_inputs",
            "output_consumed_by": "layer_fsm_fixture_output_o",
        }
    ]
    assert report["fsm_state_order"] == ["IDLE", "CHILD_START", "CHILD_BUSY", "DONE"]
    assert report["child_start_done_trace"] == [
        "decoder_child_datapath_start",
        "decoder_child_datapath_done",
    ]
    assert report["simulation_child_start_done_trace"]["recorded"] is True
    assert report["simulation_child_start_done_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_child_start_done_trace"]["child_trace_hex"] == "0x323122211211"
    assert "full Top FSM scheduling" in report["omitted_operations"]
    assert report["expected_layer_trace"] == "0x4241"
    assert report["expected_child_trace"] == "0x323122211211"
    assert report["expected_outputs"] == [-61, 24, 3, -42]
    assert report["layer_fsm_fixture"] is True
    assert report["datapath_child_instantiation"] is True
    assert "top_fsm" in report["does_not_claim"]
    assert "full_llama_coverage" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_top_fsm_fixture_instantiates_layer_fsm_fixture(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("top_fsm_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS top_fsm_fixture" in result.simulation["output"]
    assert "TOP_TRACE top_fsm_fixture" in result.simulation["output"]
    assert "top_fsm_fixture.sv" in result.files
    assert "layer_fsm_fixture.sv" in result.files
    assert "decoder_child_datapath.sv" in result.files
    assert "rmsnorm_target.sv" in result.files
    assert "projection_tile.sv" in result.files
    assert "rope_target.sv" in result.files
    assert (tmp_path / "top_fsm_fixture_golden.json").exists()
    sv_text = (tmp_path / "top_fsm_fixture.sv").read_text(encoding="utf-8")
    assert "input  logic                                    aclk" in sv_text
    assert "input  logic                                    aresetn" in sv_text
    assert "input  logic                                    start_i" in sv_text
    assert "output logic                                    done_o" in sv_text
    assert "layer_fsm_fixture u_layer_fsm_fixture" in sv_text
    assert ".start_i(layer_start_r)" in sv_text
    assert ".done_o(layer_done_w)" in sv_text
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["coverage_level"] == "top_fsm_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["numeric_policy"]["fixture_kind"] == "top_fsm_fixture"
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_layer_fsm_fixture"
    assert report["fixture_layer_count"] == 1
    assert report["child_modules"] == [
        {
            "name": "layer_fsm_fixture",
            "coverage_level": "layer_fsm_fixture",
            "instantiated": True,
            "input_source": "static_layer_fixture_inputs",
            "output_consumed_by": "top_fsm_fixture_output_o",
        }
    ]
    assert report["fsm_state_order"] == ["IDLE", "LAYER_START", "LAYER_BUSY", "DONE"]
    assert report["layer_start_done_trace"] == [
        "layer_fsm_fixture_start",
        "layer_fsm_fixture_done",
    ]
    assert report["simulation_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_layer_start_done_trace"]["top_trace_hex"] == "0x5453"
    assert report["simulation_layer_start_done_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_layer_start_done_trace"]["child_trace_hex"] == "0x323122211211"
    assert "real token prefill/decode loop" in report["omitted_operations"]
    assert "board-level I/O and shell integration" in report["omitted_operations"]
    assert report["expected_top_trace"] == "0x5453"
    assert report["expected_layer_trace"] == "0x4241"
    assert report["expected_child_trace"] == "0x323122211211"
    assert report["expected_final_fixture_trace"] == {
        "top_trace_hex": "0x5453",
        "layer_trace_hex": "0x4241",
        "child_trace_hex": "0x323122211211",
    }
    assert report["expected_outputs"] == [-61, 24, 3, -42]
    assert report["top_fsm_fixture"] is True
    assert report["datapath_child_instantiation"] is True
    assert report["board_io_constraints_present"] is False
    assert "full_token_scheduling" in report["does_not_claim"]
    assert "DDR_streaming" in report["does_not_claim"]
    assert "KV-cache" in report["does_not_claim"]
    assert "full_LLaMA_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert "internal fixture timing" in report["timing_caveat"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_top_fsm_attention_fixture_sequences_refreshed_layer_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("top_fsm_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS top_fsm_attention_fixture" in result.simulation["output"]
    assert "TOP_ATTENTION_TRACE top_fsm_attention_fixture" in result.simulation["output"]
    assert "events=layer_fsm_attention_fixture_start,layer_fsm_attention_fixture_done" in result.simulation["output"]
    assert "LAYER_ATTENTION_TRACE top_fsm_attention_fixture" in result.simulation["output"]
    assert "CHILD_TRACE top_fsm_attention_fixture trace_hex=0x323122211211" in result.simulation["output"]
    assert "PROJECTION_STREAM_TRACE top_fsm_attention_fixture" in result.simulation["output"]
    assert "source=top_fsm_hierarchical_child_status" in result.simulation["output"]
    assert "CACHE_WRITE_TRACE top_fsm_attention_fixture count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SOFTMAX_CONTROL_TRACE top_fsm_attention_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE top_fsm_attention_fixture output=4,-2,1,2" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE top_fsm_attention_fixture stable=1" in result.simulation["output"]
    assert "LAYER_START_HOLD_TRACE top_fsm_attention_fixture" in result.simulation["output"]
    assert "deasserted_after_done=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE top_fsm_attention_fixture estimated_iob_bits=196 exposed_128b_memory_response=0 exposed_kv_arrays=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    assert "top_fsm_attention_fixture.sv" in result.files
    assert "layer_fsm_attention_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_layer_fsm_attention_fixture.sv" not in result.files
    assert (tmp_path / "top_fsm_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "top_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module top_fsm_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module top_fsm_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "value_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "layer_fsm_attention_fixture u_layer_fsm_attention_fixture" in sv_text
    assert ".start_i(layer_start_r)" in sv_text
    assert ".done_o(layer_done_w)" in sv_text
    assert "LAYER_DONE" in sv_text
    assert "layer_start_r <= 1'b0;" in sv_text
    assert "done_r <= 1'b1;" in sv_text

    layer_sv = (tmp_path / "layer_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "decoder_child_attention_datapath u_decoder_child_attention_datapath" in layer_sv
    assert "CHILD_BUSY" in layer_sv

    tb_text = (tmp_path / "tb_top_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "layer start was not held while layer busy" in tb_text
    assert "layer start was not deasserted after layer done_o" in tb_text
    assert "u_layer_fsm_attention_fixture.u_decoder_child_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r" in tb_text
    assert "u_layer_fsm_attention_fixture.u_decoder_child_attention_datapath.u_projection_internal_stream_shell.u_boundary.req_accepted_r" in tb_text
    assert "lane_s8(observed_write_key, 0)" in tb_text
    assert "lane_s16(dut.u_layer_fsm_attention_fixture.u_decoder_child_attention_datapath.u_attention_kv_cache_fixture.output_o, 0)" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text
    assert "output=4,-2,1,2 status=" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "top_fsm_attention_fixture"
    assert report["coverage_level"] == "top_fsm_attention_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["top_fsm_fixture"] is True
    assert report["uses_refreshed_layer_fsm_attention_fixture"] is True
    assert report["fixture_layer_count"] == 1
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_layer_fsm_attention_fixture"
    assert report["child_modules"][0]["name"] == "layer_fsm_attention_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "decoder_child_attention_datapath",
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["nested_child_coverage_summary"])
    assert report["fsm_state_order"] == ["IDLE", "LAYER_START", "LAYER_BUSY", "LAYER_DONE", "DONE"]
    assert report["top_layer_start_done_trace"] == [
        "layer_fsm_attention_fixture_start",
        "layer_fsm_attention_fixture_done",
    ]
    assert report["simulation_top_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_top_layer_start_done_trace"]["top_trace_hex"] == "0x5453"
    assert report["simulation_top_layer_start_done_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_top_layer_start_done_trace"]["child_trace_hex"] == "0x323122211211"
    assert report["observed_top_layer_start_done_trace"] == report["top_layer_start_done_trace"]
    assert report["layer_fsm_trace"]["ordered_events"] == [
        "decoder_child_attention_datapath_start",
        "decoder_child_attention_datapath_done",
    ]
    assert report["layer_fsm_trace"]["evidence_source"] == "parsed_from_top_fsm_attention_testbench_layer_status"
    assert report["child_datapath_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["child_datapath_trace"]["evidence_source"] == "parsed_from_top_fsm_attention_testbench_child_status"
    assert report["layer_start_hold_protocol_observed"]["done_seen_while_start_high"] is True
    assert report["layer_start_hold_protocol_observed"]["deasserted_after_done"] is True
    assert report["source_path_evidence_summary"]["evidence_source"] == "parsed_from_top_fsm_attention_testbench_hierarchical_child_signals"
    assert report["source_path_evidence_summary"]["rmsnorm_lookup_trace"]["inv_rms"] == 7024
    assert report["source_path_evidence_summary"]["rope_lookup_trace"][1]["sin"] == -14
    assert report["projection_shell_evidence_summary"]["evidence_source"] == "parsed_from_top_fsm_attention_testbench_hierarchical_child_signals"
    assert report["projection_shell_evidence_summary"]["projection_output_vector"] == [976, 2360]
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_emit_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["projection_consume_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_match"] is True
    assert report["attention_kv_evidence_summary"]["evidence_source"] == "parsed_from_top_fsm_attention_testbench_hierarchical_child_signals"
    assert report["attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_evidence_summary"]["key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["attention_kv_evidence_summary"]["value_read_trace"][1]["value"] == [-2, 6, -5, 4]
    assert report["attention_kv_evidence_summary"]["attention_score_trace"] == [25, -14]
    assert report["attention_kv_evidence_summary"]["softmax_control_trace"]["weights"] == [12, 4]
    assert report["attention_kv_evidence_summary"]["attention_output_vector"] == [4, -2, 1, 2]
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["top_done_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 196
    assert report["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
    assert report["exposed_port_width_summary"]["exposes_full_kv_arrays"] is False
    assert report["compact_io_observed_trace"]["exposes_full_debug_arrays"] is False
    assert "real token prefill/decode loop" in report["omitted_operations"]
    assert "multi-layer LLaMA target iteration beyond the reported fixture count" in report["omitted_operations"]
    assert "DDR/AXI KV-cache movement" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert "fixture timing without board I/O delay constraints" in report["timing_caveat"]
    assert report["timing_scope_caveat"]
    assert "post-route fixture timing only" in report["timing_scope_caveat"]
    assert "no board-level I/O delay constraints" in report["timing_scope_caveat"]
    assert "DDR/AXI shell constraints" in report["timing_scope_caveat"]
    assert "PS/PL integration constraints" in report["timing_scope_caveat"]
    assert "not board-level I/O, DDR/AXI, or PS/PL timing signoff" in report["timing_scope_caveat"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_token_loop_axi_attention_fixture_sequences_two_axi_top_calls_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("token_loop_axi_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS token_loop_axi_attention_fixture" in result.simulation["output"]
    assert "TOKEN_LOOP_TRACE token_loop_axi_attention_fixture trace_hex=0x64636261" in result.simulation["output"]
    assert "events=token0_start,token0_done,token1_start,token1_done" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_axi_attention_fixture token=0" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_axi_attention_fixture token=1" in result.simulation["output"]
    assert "axi_bits=0xf output=4,-2,1,2" in result.simulation["output"]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_attention_fixture token=0" in result.simulation["output"]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_attention_fixture token=1" in result.simulation["output"]
    assert "release_seen=1" in result.simulation["output"]
    assert "TOP_AXI_ATTENTION_TRACE token_loop_axi_attention_fixture token=1 top_trace_hex=0x5453 layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "LAYER_AXI_ATTENTION_TRACE token_loop_axi_attention_fixture token=1 layer_trace_hex=0x4241 child_trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_AR_TRACE token_loop_axi_attention_fixture addr=0x120000 len=1 size=4 burst=1 id=0x2 beats=2" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_R_METADATA_TRACE token_loop_axi_attention_fixture accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_PAYLOAD_TRACE token_loop_axi_attention_fixture emitted=16 consumed=16 payload_match=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_BACKPRESSURE_TRACE token_loop_axi_attention_fixture ready_low_payload_idx=0,3,4 trace=0x19 payload_hold_ok=1" in result.simulation[
        "output"
    ]
    assert "AXI_PROJECTION_ROUND_TRIP_TRACE token_loop_axi_attention_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1" in result.simulation[
        "output"
    ]
    assert "TOKEN_AXI_METADATA_PROPAGATION_TRACE token_loop_axi_attention_fixture token_bits=0xf" in result.simulation[
        "output"
    ]
    assert "token_bit_lsb=64 token_bit_msb=67 top_bits=0xf layer_bits=0xf nested_bits=0xf" in result.simulation[
        "output"
    ]
    assert "CACHE_WRITE_TRACE token_loop_axi_attention_fixture count=2 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "SOFTMAX_CONTROL_TRACE token_loop_axi_attention_fixture policy=two_score_winner_loser_q0_4 weights=12,4 exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE token_loop_axi_attention_fixture output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "TOKEN_LOOP_STABILITY_TRACE token_loop_axi_attention_fixture stable=1" in result.simulation["output"]
    assert "TOKEN_OUTPUT_POLICY_TRACE token_loop_axi_attention_fixture repeated_deterministic_outputs=1 token_dependent_outputs=0" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE token_loop_axi_attention_fixture estimated_iob_bits=164 prior_top_fsm_bonded_iob=196 prior_top_fsm_status_bits=128 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0" in result.simulation[
        "output"
    ]

    assert "token_loop_axi_attention_fixture.sv" in result.files
    assert "top_fsm_axi_attention_fixture.sv" in result.files
    assert "layer_fsm_axi_attention_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_top_fsm_axi_attention_fixture.sv" not in result.files
    assert (tmp_path / "token_loop_axi_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "token_loop_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module token_loop_axi_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module token_loop_axi_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 96" in top_port_text
    assert "axi_rdata_i" not in top_port_text
    assert "axi_araddr_o" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "top_fsm_axi_attention_fixture u_top_fsm_axi_attention_fixture" in sv_text
    assert ".start_i(top_start_r)" in sv_text
    assert ".done_o(top_done_w)" in sv_text
    assert "TOKEN0_RELEASE" in sv_text
    assert "TOKEN1_RELEASE" in sv_text
    assert "if (!top_done_w)" in sv_text
    assert "top_status_w[95:92]" in sv_text
    assert "status_r <= {28'h0a71c0a, top_status_w[95:92]" in sv_text

    tb_text = (tmp_path / "tb_token_loop_axi_attention_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while top child busy" in tb_text
    assert "child start was not deasserted after top child done_o" in tb_text
    assert "token0_release_seen_r" in tb_text
    assert "token1_release_seen_r" in tb_text
    assert "status_vec[64 +: 4] != token1_status_seen[95:92]" in tb_text
    assert "u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "TOKEN_CHILD_CALL_TRACE token_loop_axi_attention_fixture token=0 top_trace_hex=0x5453" not in tb_text
    assert "FINAL_OUTPUT_TRACE token_loop_axi_attention_fixture output=4,-2,1,2 status=" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "token_loop_axi_attention_fixture"
    assert report["coverage_level"] == "token_loop_axi_attention_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["token_loop_axi_attention_fixture"] is True
    assert report["uses_axi_top_fsm_child"] is True
    assert report["top_fsm_child_instantiation"] is True
    assert report["fixture_token_count"] == 2
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_top_fsm_axi_attention_fixture"
    assert report["child_modules"][0]["name"] == "top_fsm_axi_attention_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "top_fsm_axi_attention_fixture",
        "layer_fsm_axi_attention_fixture",
        "decoder_child_axi_attention_datapath",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
        "rmsnorm_rope_source_path",
    ]
    assert all(child["instantiated"] is True for child in report["nested_child_coverage_summary"])
    assert report["fsm_state_order"] == [
        "IDLE",
        "TOKEN0_START",
        "TOKEN0_BUSY",
        "TOKEN0_RELEASE",
        "TOKEN1_START",
        "TOKEN1_BUSY",
        "TOKEN1_RELEASE",
        "DONE",
    ]
    assert report["token_start_done_trace"] == ["token0_start", "token0_done", "token1_start", "token1_done"]
    assert report["simulation_token_start_done_trace"]["recorded"] is True
    assert report["simulation_token_start_done_trace"]["trace_hex"] == "0x64636261"
    assert report["observed_token_start_done_trace"] == report["token_start_done_trace"]
    assert len(report["per_token_child_start_hold_deassert_release_evidence"]) == 2
    assert all(entry["done_seen_while_start_high"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert all(entry["deasserted_after_done"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert all(entry["done_release_seen_after_start_deassert"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert [entry["token"] for entry in report["top_fsm_traces"]] == [0, 1]
    assert report["top_fsm_traces"][0]["ordered_events"] == [
        "layer_fsm_axi_attention_fixture_start",
        "layer_fsm_axi_attention_fixture_done",
    ]
    assert report["layer_fsm_traces"][1]["ordered_events"] == [
        "decoder_child_axi_attention_datapath_start",
        "decoder_child_axi_attention_datapath_done",
    ]
    assert report["decoder_child_traces"][1]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    propagation = report["token_loop_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["token_status_bits_hex"] == "0xf"
    assert propagation["top_status_bits_hex"] == "0xf"
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["nested_axi_status_bits_hex"] == "0xf"
    assert propagation["token_bit_lsb"] == 64
    assert propagation["token_bit_msb"] == 67
    assert propagation["bit_mapping"] == {
        "rid_good": 64,
        "rresp_good": 65,
        "rlast_good": 66,
        "all_metadata_good": 67,
    }
    axi_summary = report["nested_axi_projection_evidence_summary"]
    assert axi_summary["evidence_source"] == "parsed_from_token_loop_axi_testbench_nested_child_signals_token1"
    assert axi_summary["command_trace"]["addr_hex"] == "0x120000"
    assert axi_summary["r_metadata_trace"]["rid_ok"] is True
    assert axi_summary["payload_trace"]["emitted_count"] == 16
    assert axi_summary["payload_trace"]["consumed_count"] == 16
    assert axi_summary["payload_trace"]["payload_link_match_passed"] is True
    assert axi_summary["backpressure_trace"]["payload_valid_held_while_ready_low"] is True
    assert axi_summary["round_trip_evidence"]["round_trip_passed"] is True
    assert report["nested_attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["nested_attention_kv_evidence_summary"]["softmax_control_trace"]["softmax_exp_in_rtl"] is False
    assert report["token_output_vectors"] == [[4, -2, 1, 2], [4, -2, 1, 2]]
    assert report["final_output_vector"] == [4, -2, 1, 2]
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["repeated_deterministic_fixture_outputs"] is True
    assert report["token_outputs_are_token_dependent"] is False
    assert report["loop_done_output_status_stability_observed"] is True
    assert report["softmax_exp_in_rtl"] is False
    assert report["kv_cache_external_memory"] is False
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 164
    assert report["exposed_port_width_summary"]["status_o_bits"] == 96
    assert report["compact_io_observed_trace"]["exposes_128b_axi_read_data"] is False
    assert report["compact_io_observed_trace"]["exposes_axi_debug_buses"] is False
    assert "real prefill/decode token loop" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]
    assert "not board-level I/O, DDR/AXI, or PS/PL timing signoff" in report["timing_scope_caveat"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_token_loop_attention_fixture_sequences_two_top_calls_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("token_loop_attention_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS token_loop_attention_fixture" in result.simulation["output"]
    assert "TOKEN_LOOP_TRACE token_loop_attention_fixture trace_hex=0x64636261" in result.simulation["output"]
    assert "events=token0_start,token0_done,token1_start,token1_done" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_attention_fixture token=0" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_attention_fixture token=1" in result.simulation["output"]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_attention_fixture token=0" in result.simulation["output"]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_attention_fixture token=1" in result.simulation["output"]
    assert "deasserted_after_done=1" in result.simulation["output"]
    assert "PROJECTION_STREAM_TRACE token_loop_attention_fixture token=1" in result.simulation["output"]
    assert "source=token_loop_hierarchical_child_status" in result.simulation["output"]
    assert "CACHE_WRITE_TRACE token_loop_attention_fixture token=1 count=1 slot=1 key=-3,2,1,6 value=-2,6,-5,4 stable=1" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE token_loop_attention_fixture output=4,-2,1,2" in result.simulation["output"]
    assert "TOKEN_LOOP_STABILITY_TRACE token_loop_attention_fixture stable=1" in result.simulation["output"]
    assert "TOKEN_OUTPUT_POLICY_TRACE token_loop_attention_fixture repeated_deterministic_outputs=1 token_dependent_outputs=0" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE token_loop_attention_fixture estimated_iob_bits=228 exposed_128b_memory_response=0 exposed_kv_arrays=0 exposed_debug_arrays=0 exposed_hidden_vectors=0" in result.simulation[
        "output"
    ]

    assert "token_loop_attention_fixture.sv" in result.files
    assert "top_fsm_attention_fixture.sv" in result.files
    assert "layer_fsm_attention_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_top_fsm_attention_fixture.sv" not in result.files
    assert (tmp_path / "token_loop_attention_fixture_golden.json").exists()

    sv_text = (tmp_path / "token_loop_attention_fixture.sv").read_text(encoding="utf-8")
    assert "module token_loop_attention_fixture #(" in sv_text
    top_port_text = sv_text[
        sv_text.index("module token_loop_attention_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                    aclk" in top_port_text
    assert "input  logic                                    aresetn" in top_port_text
    assert "input  logic                                    start_i" in top_port_text
    assert "output logic                                    done_o" in top_port_text
    assert "output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                 status_o" in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "key_cache_i" not in top_port_text
    assert "value_cache_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "top_fsm_attention_fixture u_top_fsm_attention_fixture" in sv_text
    assert ".start_i(top_start_r)" in sv_text
    assert ".done_o(top_done_w)" in sv_text
    assert "TOKEN0_BUSY" in sv_text
    assert "TOKEN1_BUSY" in sv_text
    assert "TOKEN0_RELEASE" in sv_text
    assert "TOKEN1_RELEASE" in sv_text
    assert "if (!top_done_w)" in sv_text
    assert "top_start_r <= 1'b0;" in sv_text

    top_sv = (tmp_path / "top_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "layer_fsm_attention_fixture u_layer_fsm_attention_fixture" in top_sv
    layer_sv = (tmp_path / "layer_fsm_attention_fixture.sv").read_text(encoding="utf-8")
    assert "decoder_child_attention_datapath u_decoder_child_attention_datapath" in layer_sv

    tb_text = (tmp_path / "tb_token_loop_attention_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while top child busy" in tb_text
    assert "child start was not deasserted after top child done_o" in tb_text
    assert "u_top_fsm_attention_fixture.u_layer_fsm_attention_fixture.u_decoder_child_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r" in tb_text
    assert "u_top_fsm_attention_fixture.u_layer_fsm_attention_fixture.u_decoder_child_attention_datapath.u_projection_internal_stream_shell.u_boundary.req_accepted_r" in tb_text
    assert "lane_s8(observed_write_key, 0)" in tb_text
    assert "lane_s16(token0_output_seen, 0)" in tb_text
    assert "key=-3,2,1,6 value=-2,6,-5,4 stable=1" not in tb_text
    assert "TOKEN_CHILD_CALL_TRACE token_loop_attention_fixture token=0 top_trace_hex=0x5453" not in tb_text
    assert "FINAL_OUTPUT_TRACE token_loop_attention_fixture output=4,-2,1,2 status=" not in tb_text
    assert "ATTENTION_OUTPUT_TRACE token_loop_attention_fixture token=1 output=4,-2,1,2" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "token_loop_attention_fixture"
    assert report["coverage_level"] == "token_loop_attention_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["token_loop_fixture"] is True
    assert report["uses_refreshed_top_fsm_attention_fixture"] is True
    assert report["fixture_token_count"] == 2
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_top_fsm_attention_fixture"
    assert report["child_modules"][0]["name"] == "top_fsm_attention_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "layer_fsm_attention_fixture",
        "decoder_child_attention_datapath",
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["nested_child_coverage_summary"])
    assert report["fsm_state_order"] == [
        "IDLE",
        "TOKEN0_START",
        "TOKEN0_BUSY",
        "TOKEN0_RELEASE",
        "TOKEN1_START",
        "TOKEN1_BUSY",
        "TOKEN1_RELEASE",
        "DONE",
    ]
    assert report["token_start_done_trace"] == ["token0_start", "token0_done", "token1_start", "token1_done"]
    assert report["simulation_token_start_done_trace"]["recorded"] is True
    assert report["simulation_token_start_done_trace"]["trace_hex"] == "0x64636261"
    assert report["observed_token_start_done_trace"] == report["token_start_done_trace"]
    assert len(report["per_token_child_start_hold_deassert_evidence"]) == 2
    assert all(entry["done_seen_while_start_high"] is True for entry in report["per_token_child_start_hold_deassert_evidence"])
    assert all(entry["deasserted_after_done"] is True for entry in report["per_token_child_start_hold_deassert_evidence"])
    assert [entry["token"] for entry in report["top_fsm_traces"]] == [0, 1]
    assert report["top_fsm_traces"][0]["ordered_events"] == [
        "layer_fsm_attention_fixture_start",
        "layer_fsm_attention_fixture_done",
    ]
    assert report["top_fsm_traces"][0]["evidence_source"] == "parsed_from_token_loop_testbench_captured_top_status"
    assert report["layer_fsm_traces"][1]["ordered_events"] == [
        "decoder_child_attention_datapath_start",
        "decoder_child_attention_datapath_done",
    ]
    assert report["decoder_child_traces"][1]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["decoder_child_traces"][1]["evidence_source"] == "parsed_from_token_loop_testbench_captured_top_status"
    assert report["source_path_evidence_summary"]["evidence_source"] == "parsed_from_token_loop_testbench_hierarchical_child_signals_token1"
    assert report["source_path_evidence_summary"]["rmsnorm_lookup_trace"]["inv_rms"] == 7024
    assert report["source_path_evidence_summary"]["rope_lookup_trace"][1]["sin"] == -14
    assert report["projection_shell_evidence_summary"]["evidence_source"] == "parsed_from_token_loop_testbench_hierarchical_child_signals_token1"
    assert report["projection_shell_evidence_summary"]["projection_output_vector"] == [976, 2360]
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_emit_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["projection_consume_count"] == 16
    assert report["projection_shell_evidence_summary"]["stream_status"]["payload_match"] is True
    assert report["attention_kv_evidence_summary"]["evidence_source"] == "parsed_from_token_loop_testbench_hierarchical_child_signals_token1"
    assert report["attention_kv_evidence_summary"]["cache_write_trace"]["field_stability_checked"] is True
    assert report["attention_kv_evidence_summary"]["key_read_trace"][1]["key"] == [-3, 2, 1, 6]
    assert report["attention_kv_evidence_summary"]["value_read_trace"][1]["value"] == [-2, 6, -5, 4]
    assert report["attention_kv_evidence_summary"]["attention_score_trace"] == [25, -14]
    assert report["attention_kv_evidence_summary"]["softmax_control_trace"]["weights"] == [12, 4]
    assert report["attention_kv_evidence_summary"]["attention_output_vector"] == [4, -2, 1, 2]
    assert report["token_output_vectors"] == [[4, -2, 1, 2], [4, -2, 1, 2]]
    assert report["final_output_vector"] == [4, -2, 1, 2]
    assert report["observed_final_fixture_output_vector"] == [4, -2, 1, 2]
    assert report["fixture_output_vector"] == [4, -2, 1, 2]
    assert report["expected_golden_vector"] == [4, -2, 1, 2]
    assert report["repeated_deterministic_fixture_outputs"] is True
    assert report["token_outputs_are_token_dependent"] is False
    assert report["top_done_stability_observed"] is True
    assert report["loop_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 228
    assert report["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
    assert report["exposed_port_width_summary"]["exposes_full_kv_arrays"] is False
    assert report["compact_io_observed_trace"]["exposes_full_debug_arrays"] is False
    assert report["compact_io_observed_trace"]["exposes_full_hidden_vectors"] is False
    assert "real LLaMA token prefill/decode semantics" in report["omitted_operations"]
    assert "token-dependent KV-cache accumulation across full sequence length" in report["omitted_operations"]
    assert "DDR/AXI KV-cache movement" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "target_sequence_scheduling" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert "fixture timing without board I/O delay" in report["timing_caveat"]
    assert "DDR/AXI shell" in report["timing_caveat"]
    assert "PS/PL integration constraints" in report["timing_caveat"]
    assert "not board-level I/O, DDR/AXI, or PS/PL timing signoff" in report["timing_scope_caveat"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_residual_mlp_fixture_reports_golden_trace_and_compact_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "attention_residual")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("residual_mlp_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS residual_mlp_fixture" in result.simulation["output"]
    assert "RESIDUAL_MLP_TRACE residual_mlp_fixture" in result.simulation["output"]
    assert "events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done" in result.simulation[
        "output"
    ]
    assert "GATE_UP_TRACE residual_mlp_fixture gate=8,5,-3,5 up=9,2,1,0 source=fixture_constant_projection_matrices" in result.simulation[
        "output"
    ]
    assert "SWIGLU_TRACE residual_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=9,1,-1,0 true_silu_exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE residual_mlp_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "RESIDUAL_MLP_STABILITY_TRACE residual_mlp_fixture stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE residual_mlp_fixture estimated_iob_bits=292 exposed_128b_memory_response=0 exposed_full_hidden_vectors=0 exposed_intermediate_tensors=0 exposed_matrices=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    assert "residual_mlp_fixture.sv" in result.files
    assert "tb_residual_mlp_fixture.sv" in result.files
    assert (tmp_path / "residual_mlp_fixture_golden.json").exists()

    sv_text = (tmp_path / "residual_mlp_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module residual_mlp_fixture #(") : sv_text.index("    localparam int TRACE_WIDTH")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "input  logic signed [VALUES*ELEM_WIDTH-1:0]       hidden_input_i" in top_port_text
    assert "input  logic signed [VALUES*ELEM_WIDTH-1:0]       attention_output_i" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "debug_array" not in top_port_text
    assert "gate_matrix_i" not in top_port_text
    assert "intermediate_tensor" not in top_port_text
    assert "DONE: begin" in sv_text
    assert "if (!start_i)" in sv_text
    assert "done_r <= 1'b0;" in sv_text

    tb_text = (tmp_path / "tb_residual_mlp_fixture.sv").read_text(encoding="utf-8")
    assert "check_lane(\"gate\", dut.gate_r, 0, 8)" in tb_text
    assert "lane_s16(dut.gate_r, 0)" in tb_text
    assert "gate=8,5,-3,5" not in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "residual_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "residual_mlp_fixture"
    assert golden["residual_mlp_fixture"] is True
    assert golden["selected_non_gemm"] == "attention_residual"
    assert golden["selected_non_gemm_op_type"] == "Add"
    assert golden["selected_non_gemm_shape"] == {"hidden_size": 2048}
    assert golden["target_source_path"] == "attention_residual_path"
    assert golden["full_target_non_gemm_execution"] is False
    assert golden["target_fixture_distinction"]["selected_non_gemm"] == "attention_residual"
    assert golden["target_fixture_distinction"]["fixture_hidden_width"] == 4
    assert golden["target_fixture_distinction"]["fixture_intermediate_width"] == 4
    assert golden["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
    assert "target semantic metadata plus bounded 4-wide residual/MLP fixture evidence" in golden[
        "target_fixture_distinction"
    ]["distinction"]
    assert golden["hidden_input_vector"] == [3, -2, 5, 1]
    assert golden["attention_output_vector"] == [1, 4, -3, 2]
    assert golden["residual0_vector"] == [4, 2, 2, 3]
    assert golden["gate_projection_vector"] == [8, 5, -3, 5]
    assert golden["up_projection_vector"] == [9, 2, 1, 0]
    assert golden["swiglu_vector"] == [9, 1, -1, 0]
    assert golden["down_projection_vector"] == [8, -8, 16, 3]
    assert golden["final_output_vector"] == [12, -6, 18, 6]
    assert golden["arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert golden["projection_source"]["evidence_label"] == "fixture_constants_not_streamed_projection_evidence"
    assert golden["child_modules"] == []

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "residual_mlp_fixture"
    assert report["coverage_level"] == "residual_mlp_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["residual_mlp_fixture"] is True
    assert report["selected_non_gemm"] == "attention_residual"
    assert report["selected_non_gemm_op_type"] == "Add"
    assert report["selected_non_gemm_shape"] == {"hidden_size": 2048}
    assert report["target_source_path"] == "attention_residual_path"
    assert report["full_target_non_gemm_execution"] is False
    assert report["target_fixture_distinction"]["selected_non_gemm"] == "attention_residual"
    assert report["target_fixture_distinction"]["fixture_hidden_width"] == 4
    assert report["target_fixture_distinction"]["fixture_intermediate_width"] == 4
    assert report["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
    assert "target semantic metadata plus bounded 4-wide residual/MLP fixture evidence" in report[
        "target_fixture_distinction"
    ]["distinction"]
    assert report["fixture_hidden_width"] == 4
    assert report["fixture_intermediate_width"] == 4
    assert report["fsm_state_order"] == [
        "IDLE",
        "RESIDUAL0",
        "GATE_UP",
        "SWIGLU_SIGMOID",
        "SWIGLU_SILU",
        "SWIGLU",
        "DOWN",
        "RESIDUAL1",
        "DONE",
    ]
    assert report["ordered_residual_mlp_trace"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["observed_hidden_input_vector"] == [3, -2, 5, 1]
    assert report["observed_attention_output_vector"] == [1, 4, -3, 2]
    assert report["observed_residual0_vector"] == [4, 2, 2, 3]
    assert report["observed_gate_projection_vector"] == [8, 5, -3, 5]
    assert report["observed_up_projection_vector"] == [9, 2, 1, 0]
    assert report["observed_swiglu_vector"] == [9, 1, -1, 0]
    assert report["observed_down_projection_vector"] == [8, -8, 16, 3]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["done_output_status_stability_observed"] is True
    assert report["projection_source"]["child_projection_fixtures_instantiated"] is False
    assert report["projection_source"]["dynamically_streamed_projection_evidence"] is False
    assert report["observed_gate_up_projection_source"] == "fixture_constant_projection_matrices"
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 292
    assert report["compact_io_observed_trace"]["exposes_128b_memory_response"] is False
    assert report["compact_io_observed_trace"]["exposes_full_intermediate_tensors"] is False
    assert "target-scale LLaMA MLP dimensions" in report["omitted_operations"]
    assert "GPTQ INT4 packed gate/up/down weight streaming" in report["omitted_operations"]
    assert "true SiLU or exponential activation math in RTL" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert "fixture timing without board-level I/O delay" in report["timing_caveat"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_residual_mlp_fixture_selects_silu_gate_semantics_with_bounded_fixture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "silu_gate")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("residual_mlp_fixture", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS residual_mlp_fixture" in result.simulation["output"]
    assert "HIDDEN_INPUT_TRACE residual_mlp_fixture hidden=3,-2,5,1 attention=1,4,-3,2" in result.simulation[
        "output"
    ]
    assert "RESIDUAL0_TRACE residual_mlp_fixture residual0=4,2,2,3" in result.simulation["output"]
    assert "GATE_UP_TRACE residual_mlp_fixture gate=8,5,-3,5 up=9,2,1,0 source=fixture_constant_projection_matrices" in result.simulation[
        "output"
    ]
    assert "SWIGLU_TRACE residual_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=9,1,-1,0 true_silu_exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE residual_mlp_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "COMPACT_IO_TRACE residual_mlp_fixture estimated_iob_bits=292 exposed_128b_memory_response=0 exposed_full_hidden_vectors=0 exposed_intermediate_tensors=0 exposed_matrices=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    golden = json.loads((tmp_path / "residual_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))

    for artifact in (golden, report):
        assert artifact["selected_non_gemm"] == "silu_gate"
        assert artifact["selected_non_gemm_op_type"] == "SiLU"
        assert artifact["selected_non_gemm_shape"] == {"intermediate_size": 8192}
        assert artifact["target_source_path"] == "silu_gate_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "silu_gate"
        assert artifact["target_fixture_distinction"]["fixture_hidden_width"] == 4
        assert artifact["target_fixture_distinction"]["fixture_intermediate_width"] == 4
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "target semantic metadata plus bounded 4-wide residual/MLP fixture evidence" in artifact[
            "target_fixture_distinction"
        ]["distinction"]
        assert artifact["hidden_input_vector"] == [3, -2, 5, 1]
        assert artifact["attention_output_vector"] == [1, 4, -3, 2]
        assert artifact["residual0_vector"] == [4, 2, 2, 3]
        assert artifact["gate_projection_vector"] == [8, 5, -3, 5]
        assert artifact["sigmoid_approx_q0_4_vector"] == [16, 13, 5, 13]
        assert artifact["silu_approx_vector"] == [8, 4, -1, 4]
        assert artifact["swiglu_vector"] == [9, 1, -1, 0]
        assert artifact["final_output_vector"] == [12, -6, 18, 6]
        assert artifact["arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
        assert artifact["top_level_interface_summary"]["compact_top_level_io"] is True
        assert artifact["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_hidden_vectors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_intermediate_tensors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_matrices"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_debug_arrays"] is False

    assert report["observed_hidden_input_vector"] == [3, -2, 5, 1]
    assert report["observed_attention_output_vector"] == [1, 4, -3, 2]
    assert report["observed_residual0_vector"] == [4, 2, 2, 3]
    assert report["observed_gate_projection_vector"] == [8, 5, -3, 5]
    assert report["observed_swiglu_vector"] == [9, 1, -1, 0]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["observed_arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_memory_response"] is False
    assert report["compact_io_observed_trace"]["exposes_full_intermediate_tensors"] is False


def test_emit_residual_mlp_fixture_selects_swiglu_multiply_semantics_with_bounded_fixture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "swiglu_multiply")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("residual_mlp_fixture", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS residual_mlp_fixture" in result.simulation["output"]
    assert "HIDDEN_INPUT_TRACE residual_mlp_fixture hidden=3,-2,5,1 attention=1,4,-3,2" in result.simulation[
        "output"
    ]
    assert "RESIDUAL0_TRACE residual_mlp_fixture residual0=4,2,2,3" in result.simulation["output"]
    assert "GATE_UP_TRACE residual_mlp_fixture gate=8,5,-3,5 up=9,2,1,0 source=fixture_constant_projection_matrices" in result.simulation[
        "output"
    ]
    assert "SWIGLU_TRACE residual_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=9,1,-1,0 true_silu_exp=0" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE residual_mlp_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "COMPACT_IO_TRACE residual_mlp_fixture estimated_iob_bits=292 exposed_128b_memory_response=0 exposed_full_hidden_vectors=0 exposed_intermediate_tensors=0 exposed_matrices=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    golden = json.loads((tmp_path / "residual_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))

    for artifact in (golden, report):
        assert artifact["selected_non_gemm"] == "swiglu_multiply"
        assert artifact["selected_non_gemm_op_type"] == "Mul"
        assert artifact["selected_non_gemm_shape"] == {"intermediate_size": 8192}
        assert artifact["target_source_path"] == "swiglu_multiply_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "swiglu_multiply"
        assert artifact["target_fixture_distinction"]["fixture_hidden_width"] == 4
        assert artifact["target_fixture_distinction"]["fixture_intermediate_width"] == 4
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "target semantic metadata plus bounded 4-wide residual/MLP fixture evidence" in artifact[
            "target_fixture_distinction"
        ]["distinction"]
        assert artifact["hidden_input_vector"] == [3, -2, 5, 1]
        assert artifact["attention_output_vector"] == [1, 4, -3, 2]
        assert artifact["residual0_vector"] == [4, 2, 2, 3]
        assert artifact["gate_projection_vector"] == [8, 5, -3, 5]
        assert artifact["up_projection_vector"] == [9, 2, 1, 0]
        assert artifact["sigmoid_approx_q0_4_vector"] == [16, 13, 5, 13]
        assert artifact["silu_approx_vector"] == [8, 4, -1, 4]
        assert artifact["swiglu_vector"] == [9, 1, -1, 0]
        assert artifact["final_output_vector"] == [12, -6, 18, 6]
        assert artifact["arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
        assert artifact["top_level_interface_summary"]["compact_top_level_io"] is True
        assert artifact["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_hidden_vectors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_intermediate_tensors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_matrices"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_debug_arrays"] is False

    assert report["observed_hidden_input_vector"] == [3, -2, 5, 1]
    assert report["observed_attention_output_vector"] == [1, 4, -3, 2]
    assert report["observed_residual0_vector"] == [4, 2, 2, 3]
    assert report["observed_gate_projection_vector"] == [8, 5, -3, 5]
    assert report["observed_up_projection_vector"] == [9, 2, 1, 0]
    assert report["observed_swiglu_vector"] == [9, 1, -1, 0]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["observed_arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_memory_response"] is False
    assert report["compact_io_observed_trace"]["exposes_full_intermediate_tensors"] is False


def test_emit_residual_mlp_fixture_selects_mlp_residual_semantics_with_bounded_fixture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "mlp_residual")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("residual_mlp_fixture", cfg, tmp_path)

    assert result.status == "passed"
    assert "PASS residual_mlp_fixture" in result.simulation["output"]
    assert "HIDDEN_INPUT_TRACE residual_mlp_fixture hidden=3,-2,5,1 attention=1,4,-3,2" in result.simulation[
        "output"
    ]
    assert "RESIDUAL0_TRACE residual_mlp_fixture residual0=4,2,2,3" in result.simulation["output"]
    assert "GATE_UP_TRACE residual_mlp_fixture gate=8,5,-3,5 up=9,2,1,0 source=fixture_constant_projection_matrices" in result.simulation[
        "output"
    ]
    assert "SWIGLU_TRACE residual_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=9,1,-1,0 true_silu_exp=0" in result.simulation[
        "output"
    ]
    assert "DOWN_TRACE residual_mlp_fixture down=8,-8,16,3 source=fixture_constant_projection_matrix" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE residual_mlp_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "COMPACT_IO_TRACE residual_mlp_fixture estimated_iob_bits=292 exposed_128b_memory_response=0 exposed_full_hidden_vectors=0 exposed_intermediate_tensors=0 exposed_matrices=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    golden = json.loads((tmp_path / "residual_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))

    for artifact in (golden, report):
        assert artifact["selected_non_gemm"] == "mlp_residual"
        assert artifact["selected_non_gemm_op_type"] == "Add"
        assert artifact["selected_non_gemm_shape"] == {"hidden_size": 2048}
        assert artifact["target_source_path"] == "mlp_residual_path"
        assert artifact["full_target_non_gemm_execution"] is False
        assert artifact["target_fixture_distinction"]["selected_non_gemm"] == "mlp_residual"
        assert artifact["target_fixture_distinction"]["fixture_hidden_width"] == 4
        assert artifact["target_fixture_distinction"]["fixture_intermediate_width"] == 4
        assert artifact["target_fixture_distinction"]["full_target_non_gemm_execution"] is False
        assert "target semantic metadata plus bounded 4-wide residual/MLP fixture evidence" in artifact[
            "target_fixture_distinction"
        ]["distinction"]
        assert artifact["hidden_input_vector"] == [3, -2, 5, 1]
        assert artifact["attention_output_vector"] == [1, 4, -3, 2]
        assert artifact["residual0_vector"] == [4, 2, 2, 3]
        assert artifact["gate_projection_vector"] == [8, 5, -3, 5]
        assert artifact["up_projection_vector"] == [9, 2, 1, 0]
        assert artifact["sigmoid_approx_q0_4_vector"] == [16, 13, 5, 13]
        assert artifact["silu_approx_vector"] == [8, 4, -1, 4]
        assert artifact["swiglu_vector"] == [9, 1, -1, 0]
        assert artifact["down_projection_vector"] == [8, -8, 16, 3]
        assert artifact["final_output_vector"] == [12, -6, 18, 6]
        assert artifact["arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
        assert artifact["top_level_interface_summary"]["compact_top_level_io"] is True
        assert artifact["exposed_port_width_summary"]["exposes_128b_memory_response"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_hidden_vectors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_intermediate_tensors"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_matrices"] is False
        assert artifact["exposed_port_width_summary"]["exposes_full_debug_arrays"] is False

    assert report["observed_hidden_input_vector"] == [3, -2, 5, 1]
    assert report["observed_attention_output_vector"] == [1, 4, -3, 2]
    assert report["observed_residual0_vector"] == [4, 2, 2, 3]
    assert report["observed_gate_projection_vector"] == [8, 5, -3, 5]
    assert report["observed_up_projection_vector"] == [9, 2, 1, 0]
    assert report["observed_swiglu_vector"] == [9, 1, -1, 0]
    assert report["observed_down_projection_vector"] == [8, -8, 16, 3]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["observed_arithmetic_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_memory_response"] is False
    assert report["compact_io_observed_trace"]["exposes_full_intermediate_tensors"] is False


def test_emit_residual_mlp_fixture_rejects_invalid_selected_non_gemm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NL2HDL_SELECTED_NONGEMM", "bogus_non_gemm")
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")

    with pytest.raises(ValueError, match="NL2HDL_SELECTED_NONGEMM='bogus_non_gemm'"):
        emit_kernel("residual_mlp_fixture", cfg, tmp_path)

    assert not (tmp_path / "residual_mlp_fixture_golden.json").exists()
    assert not (tmp_path / "kernel_report.json").exists()
    assert not (tmp_path / "residual_mlp_fixture.sv").exists()
    assert not (tmp_path / "tb_residual_mlp_fixture.sv").exists()


def test_emit_decoder_block_attention_mlp_fixture_composes_attention_and_mlp_children(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_block_attention_mlp_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_block_attention_mlp_fixture" in result.simulation["output"]
    assert "BLOCK_TRACE decoder_block_attention_mlp_fixture block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "ATTENTION_CHILD_TRACE decoder_block_attention_mlp_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE decoder_block_attention_mlp_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "ATTENTION_CHILD_START_HOLD_TRACE decoder_block_attention_mlp_fixture" in result.simulation["output"]
    assert "MLP_CHILD_START_HOLD_TRACE decoder_block_attention_mlp_fixture" in result.simulation["output"]
    assert "ATTENTION_OUTPUT_TRACE decoder_block_attention_mlp_fixture output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE decoder_block_attention_mlp_fixture hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_FINAL_TRACE decoder_block_attention_mlp_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "FINAL_OUTPUT_TRACE decoder_block_attention_mlp_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "DECODER_BLOCK_STABILITY_TRACE decoder_block_attention_mlp_fixture stable=1" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE decoder_block_attention_mlp_fixture estimated_iob_bits=228 residual_standalone_iob_reference=292 exposed_128b=0 exposed_kv_arrays=0 exposed_hidden_ports=0 exposed_child_status_arrays=0" in result.simulation[
        "output"
    ]

    assert "decoder_block_attention_mlp_fixture.sv" in result.files
    assert "tb_decoder_block_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_child_attention_datapath.sv" not in result.files
    assert "tb_residual_mlp_fixture.sv" not in result.files
    assert (tmp_path / "decoder_block_attention_mlp_fixture_golden.json").exists()

    sv_text = (tmp_path / "decoder_block_attention_mlp_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module decoder_block_attention_mlp_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "hidden_input_i" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "decoder_child_attention_datapath u_decoder_child_attention_datapath" in sv_text
    assert "residual_mlp_fixture u_residual_mlp_fixture" in sv_text
    assert ".attention_output_i(captured_attention_r)" in sv_text
    assert "ATTENTION_RELEASE" in sv_text
    assert "MLP_RELEASE" in sv_text

    tb_text = (tmp_path / "tb_decoder_block_attention_mlp_fixture.sv").read_text(encoding="utf-8")
    assert "dut.u_decoder_child_attention_datapath.output_o" in tb_text
    assert "dut.u_residual_mlp_fixture.attention_r" in tb_text
    assert "dut.u_residual_mlp_fixture.final_output_o" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "decoder_block_attention_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "decoder_block_attention_mlp_fixture"
    assert golden["decoder_block_attention_mlp_fixture"] is True
    assert golden["captured_attention_output_vector"] == [4, -2, 1, 2]
    assert golden["mlp_hidden_input_vector"] == [0, 4, 1, 1]
    assert golden["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert golden["mlp_residual0_vector"] == [4, 2, 2, 3]
    assert golden["mlp_final_output_vector"] == [12, -6, 18, 6]
    assert golden["final_decoder_block_output_vector"] == [12, -6, 18, 6]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "decoder_block_attention_mlp_fixture"
    assert report["coverage_level"] == "decoder_block_attention_mlp_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["child_modules"][0]["name"] == "decoder_child_attention_datapath"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["child_modules"][1]["name"] == "residual_mlp_fixture"
    assert report["child_modules"][1]["instantiated"] is True
    assert {child["name"] for child in report["nested_attention_child_coverage_summary"]} == {
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    }
    assert report["ordered_block_trace"] == ["attention_start", "attention_done", "mlp_start", "mlp_done"]
    assert report["attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["captured_attention_output_vector"] == [4, -2, 1, 2]
    assert report["mlp_hidden_input_vector"] == [0, 4, 1, 1]
    assert report["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["mlp_final_output_vector"] == [12, -6, 18, 6]
    assert report["final_decoder_block_output_vector"] == [12, -6, 18, 6]
    assert report["done_output_status_stability_observed"] is True
    assert report["child_start_hold_deassert_release_evidence"]["attention_child"]["done_release_seen_after_start_deassert"] is True
    assert report["child_start_hold_deassert_release_evidence"]["mlp_child"]["done_release_seen_after_start_deassert"] is True
    assert report["arithmetic_source_policy"]["residual_mlp_fixture_uses_fixture_constant_matrices"] is True
    assert report["arithmetic_source_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert report["top_level_interface_summary"]["internalized_mlp_hidden_input"] is True
    assert report["top_level_interface_summary"]["internalized_mlp_attention_input"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 228
    assert report["package_io_mitigation_summary"]["lower_than_residual_mlp_standalone"] is True
    assert report["compact_io_observed_trace"]["exposes_hidden_input_ports"] is False
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_decoder_block_axi_attention_mlp_fixture_composes_axi_attention_and_mlp_children(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("decoder_block_axi_attention_mlp_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS decoder_block_axi_attention_mlp_fixture" in result.simulation["output"]
    assert (
        "BLOCK_AXI_TRACE decoder_block_axi_attention_mlp_fixture block_trace_hex=0xb2b1a2a1 "
        "attention_trace_hex=0x323122211211 axi_metadata_bits=0xf "
        "mlp_trace_hex=0x52514241323122211211 "
        "events=axi_attention_start,axi_attention_done,mlp_start,mlp_done"
    ) in result.simulation["output"]
    assert (
        "AXI_ATTENTION_CHILD_TRACE decoder_block_axi_attention_mlp_fixture "
        "trace_hex=0x323122211211 "
        "events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done"
    ) in result.simulation["output"]
    assert "MLP_CHILD_TRACE decoder_block_axi_attention_mlp_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "AXI_ATTENTION_CHILD_START_HOLD_TRACE decoder_block_axi_attention_mlp_fixture" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_START_HOLD_TRACE decoder_block_axi_attention_mlp_fixture" in result.simulation["output"]
    assert "BLOCK_AXI_METADATA_PROPAGATION_TRACE decoder_block_axi_attention_mlp_fixture block_bits=0xf" in result.simulation[
        "output"
    ]
    assert "attention_child_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation[
        "output"
    ]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_AR_TRACE decoder_block_axi_attention_mlp_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_block_axi_attention_mlp_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture projection={projection} output=484,1904"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture projection={projection} packed_bytes=32 unpacked_values=64 round_trip_passed=1"
            in result.simulation["output"]
        )
    assert "ATTENTION_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE decoder_block_axi_attention_mlp_fixture hidden=0,4,1,1 attention=4,-2,1,2 captured_match=1" in result.simulation[
        "output"
    ]
    assert "MLP_FINAL_TRACE decoder_block_axi_attention_mlp_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "DECODER_BLOCK_STABILITY_TRACE decoder_block_axi_attention_mlp_fixture stable=1" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE decoder_block_axi_attention_mlp_fixture estimated_iob_bits=244 residual_standalone_iob_reference=292 axi_attention_child_iob_reference=164 exposed_128b=0 exposed_axi_debug=0 exposed_kv_arrays=0 exposed_hidden_ports=0 exposed_child_status_arrays=0" in result.simulation[
        "output"
    ]

    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "tb_decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_child_axi_attention_datapath.sv" not in result.files
    assert "tb_residual_mlp_fixture.sv" not in result.files
    assert (tmp_path / "decoder_block_axi_attention_mlp_fixture_golden.json").exists()

    sv_text = (tmp_path / "decoder_block_axi_attention_mlp_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module decoder_block_axi_attention_mlp_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "hidden_input_i" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "axi_rdata_i" not in top_port_text
    assert "axi_araddr_o" not in top_port_text
    assert "decoder_child_axi_attention_datapath u_decoder_child_axi_attention_datapath" in sv_text
    assert "residual_mlp_fixture u_residual_mlp_fixture" in sv_text
    assert ".attention_output_i(captured_attention_r)" in sv_text
    assert "AXI_ATTENTION_RELEASE" in sv_text
    assert "MLP_RELEASE" in sv_text
    assert "captured_axi_metadata_r <= attention_status_w[63:60]" in sv_text

    tb_text = (tmp_path / "tb_decoder_block_axi_attention_mlp_fixture.sv").read_text(encoding="utf-8")
    assert "dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w" in tb_text
    assert "dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "dut.u_residual_mlp_fixture.attention_r != dut.captured_attention_r" in tb_text
    assert "BLOCK_AXI_METADATA_PROPAGATION_TRACE decoder_block_axi_attention_mlp_fixture" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "decoder_block_axi_attention_mlp_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "decoder_block_axi_attention_mlp_fixture"
    assert golden["decoder_block_axi_attention_mlp_fixture"] is True
    assert golden["captured_attention_output_vector"] == [4, -2, 1, 2]
    assert golden["mlp_hidden_input_vector"] == [0, 4, 1, 1]
    assert golden["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert golden["mlp_residual0_vector"] == [4, 2, 2, 3]
    assert golden["mlp_final_output_vector"] == [12, -6, 18, 6]
    assert golden["final_decoder_block_output_vector"] == [12, -6, 18, 6]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "decoder_block_axi_attention_mlp_fixture"
    assert report["coverage_level"] == "decoder_block_axi_attention_mlp_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["child_modules"][0]["name"] == "decoder_child_axi_attention_datapath"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["child_modules"][1]["name"] == "residual_mlp_fixture"
    assert report["child_modules"][1]["instantiated"] is True
    assert {child["name"] for child in report["nested_axi_attention_child_coverage_summary"]} == {
        "rmsnorm_rope_source_path",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
    }
    assert report["ordered_block_trace"] == ["axi_attention_start", "axi_attention_done", "mlp_start", "mlp_done"]
    assert report["axi_attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    propagation = report["block_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["block_status_bits_hex"] == "0xf"
    assert propagation["attention_child_status_bits_hex"] == "0xf"
    assert propagation["block_bit_lsb"] == 80
    assert propagation["block_bit_msb"] == 83
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    for projection in ("q", "k", "v", "o"):
        child_evidence = report["axi_projection_children_observed"][projection]
        assert child_evidence["command_trace"]["addr_hex"] == "0x120000"
        assert child_evidence["r_metadata_trace"]["rid_ok"] is True
        assert child_evidence["r_metadata_trace"]["rresp_ok"] is True
        assert child_evidence["r_metadata_trace"]["rlast_ok"] is True
        assert child_evidence["payload_trace"]["payload_link_match_passed"] is True
        assert child_evidence["output_trace"]["projection_output_vector"] == [484, 1904]
        assert child_evidence["round_trip_evidence"]["round_trip_passed"] is True
    assert report["attention_to_mlp_consumption_observed"] is True
    assert report["captured_attention_output_vector"] == [4, -2, 1, 2]
    assert report["mlp_hidden_input_vector"] == [0, 4, 1, 1]
    assert report["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["mlp_final_output_vector"] == [12, -6, 18, 6]
    assert report["final_decoder_block_output_vector"] == [12, -6, 18, 6]
    assert report["done_output_status_stability_observed"] is True
    assert report["child_start_hold_deassert_release_evidence"]["axi_attention_child"][
        "done_release_seen_after_start_deassert"
    ] is True
    assert report["child_start_hold_deassert_release_evidence"]["mlp_child"][
        "done_release_seen_after_start_deassert"
    ] is True
    assert report["arithmetic_source_policy"]["residual_mlp_fixture_uses_fixture_constant_matrices"] is True
    assert report["arithmetic_source_policy"]["true_silu_or_exponential_math_implemented_in_rtl"] is False
    assert report["top_level_interface_summary"]["internalized_mlp_hidden_input"] is True
    assert report["top_level_interface_summary"]["internalized_mlp_attention_input"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 244
    assert report["compact_io_observed_trace"]["exposes_128b_memory_response"] is False
    assert report["compact_io_observed_trace"]["exposes_axi_debug_buses"] is False
    assert "DDR controller integration" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_layer_fsm_axi_decoder_block_fixture_schedules_real_axi_decoder_block(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("layer_fsm_axi_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS layer_fsm_axi_decoder_block_fixture" in result.simulation["output"]
    assert (
        "LAYER_AXI_DECODER_BLOCK_TRACE layer_fsm_axi_decoder_block_fixture "
        "layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf "
        "events=decoder_block_axi_attention_mlp_fixture_start,decoder_block_axi_attention_mlp_fixture_done"
    ) in result.simulation["output"]
    assert "BLOCK_AXI_TRACE layer_fsm_axi_decoder_block_fixture block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "AXI_ATTENTION_CHILD_TRACE layer_fsm_axi_decoder_block_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE layer_fsm_axi_decoder_block_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "LAYER_BLOCK_CHILD_START_HOLD_TRACE layer_fsm_axi_decoder_block_fixture" in result.simulation["output"]
    assert "LAYER_AXI_METADATA_PROPAGATION_TRACE layer_fsm_axi_decoder_block_fixture layer_bits=0xf" in result.simulation[
        "output"
    ]
    assert "child_block_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation["output"]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_AR_TRACE layer_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE layer_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_PAYLOAD_TRACE layer_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_OUTPUT_TRACE layer_fsm_axi_decoder_block_fixture projection={projection} output=484,1904"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE layer_fsm_axi_decoder_block_fixture projection={projection} packed_bytes=32 unpacked_values=64 round_trip_passed=1"
            in result.simulation["output"]
        )
    assert "ATTENTION_OUTPUT_TRACE layer_fsm_axi_decoder_block_fixture output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE layer_fsm_axi_decoder_block_fixture hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "FINAL_DECODER_BLOCK_OUTPUT_TRACE layer_fsm_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE layer_fsm_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "LAYER_STABILITY_TRACE layer_fsm_axi_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE layer_fsm_axi_decoder_block_fixture estimated_iob_bits=132 previous_decoder_block_iob=244 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]

    assert "layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "tb_layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_block_axi_attention_mlp_fixture.sv" not in result.files
    assert "tb_decoder_child_axi_attention_datapath.sv" not in result.files
    assert (tmp_path / "layer_fsm_axi_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "layer_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module layer_fsm_axi_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "parameter int BLOCK_STATUS_WIDTH = 176" in top_port_text
    assert "decoder_block_axi_attention_mlp_fixture u_decoder_block_axi_attention_mlp_fixture" in sv_text
    assert ".start_i(block_start_r)" in sv_text
    assert "BLOCK_RELEASE" in sv_text
    assert "block_status_w[80 +: 4]" in sv_text
    assert "axi_rdata" not in top_port_text
    assert "axi_araddr" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "hidden_input_i" not in top_port_text

    tb_text = (tmp_path / "tb_layer_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "dut.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath" in tb_text
    assert "u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_v_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_o_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "LAYER_AXI_METADATA_PROPAGATION_TRACE layer_fsm_axi_decoder_block_fixture" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "layer_fsm_axi_decoder_block_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "layer_fsm_axi_decoder_block_fixture"
    assert golden["coverage_level"] == "layer_fsm_axi_decoder_block_fixture"
    assert golden["layer_fsm_axi_decoder_block_fixture"] is True
    assert golden["child_modules"][0]["name"] == "decoder_block_axi_attention_mlp_fixture"
    assert golden["child_modules"][0]["instantiated"] is True
    assert golden["final_layer_output_vector"] == [12, -6, 18, 6]
    assert golden["decoder_block_child_output_vector"] == [12, -6, 18, 6]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "layer_fsm_axi_decoder_block_fixture"
    assert report["coverage_level"] == "layer_fsm_axi_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["layer_fsm_axi_decoder_block_fixture"] is True
    assert report["ordered_layer_trace"] == [
        "decoder_block_axi_attention_mlp_fixture_start",
        "decoder_block_axi_attention_mlp_fixture_done",
    ]
    assert report["ordered_block_trace"] == ["axi_attention_start", "axi_attention_done", "mlp_start", "mlp_done"]
    assert report["axi_attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["layer_to_block_child_start_hold_deassert_release_evidence"][
        "done_release_seen_after_start_deassert"
    ] is True
    propagation = report["layer_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["child_block_status_bits_hex"] == "0xf"
    assert propagation["layer_bit_lsb"] == 48
    assert propagation["layer_bit_msb"] == 51
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    for projection in ("q", "k", "v", "o"):
        child_evidence = report["axi_projection_children_observed"][projection]
        assert child_evidence["r_metadata_trace"]["rid_ok"] is True
        assert child_evidence["r_metadata_trace"]["rresp_ok"] is True
        assert child_evidence["r_metadata_trace"]["rlast_ok"] is True
        assert child_evidence["payload_trace"]["payload_link_match_passed"] is True
        assert child_evidence["output_trace"]["projection_output_vector"] == [484, 1904]
        assert child_evidence["round_trip_evidence"]["round_trip_passed"] is True
    assert report["decoder_block_child_output_vector"] == [12, -6, 18, 6]
    assert report["final_layer_output_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_layer_hierarchy"] is True
    assert report["layer_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_128b_axi_data"] is False
    assert report["compact_io_observed_trace"]["exposes_axi_debug_buses"] is False
    assert "real GPTQ checkpoint payload streaming" in report["omitted_operations"]
    assert "DDR/AXI full integration" in report["omitted_operations"]
    assert "token loop" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "real_GPTQ_checkpoint_payload_streaming" in report["does_not_claim"]
    assert "DDR_AXI_full_integration" in report["does_not_claim"]
    assert "token_loop" in report["does_not_claim"]
    assert "full_LLaMA_layer_execution" in report["does_not_claim"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_top_fsm_axi_decoder_block_fixture_schedules_real_axi_layer_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("top_fsm_axi_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS top_fsm_axi_decoder_block_fixture" in result.simulation["output"]
    assert "TOP_AXI_DECODER_BLOCK_TRACE top_fsm_axi_decoder_block_fixture top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "events=layer_fsm_axi_decoder_block_fixture_start,layer_fsm_axi_decoder_block_fixture_done" in result.simulation[
        "output"
    ]
    assert "LAYER_AXI_DECODER_BLOCK_TRACE top_fsm_axi_decoder_block_fixture layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "BLOCK_AXI_TRACE top_fsm_axi_decoder_block_fixture block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "AXI_ATTENTION_CHILD_TRACE top_fsm_axi_decoder_block_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE top_fsm_axi_decoder_block_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "TOP_LAYER_CHILD_START_HOLD_TRACE top_fsm_axi_decoder_block_fixture" in result.simulation["output"]
    assert "release_seen=1" in result.simulation["output"]
    assert "TOP_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_decoder_block_fixture top_bits=0xf" in result.simulation[
        "output"
    ]
    assert "layer_bits=0xf child_block_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation[
        "output"
    ]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture projection={projection} output=484,1904"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_decoder_block_fixture projection={projection} packed_bytes=32 unpacked_values=64 round_trip_passed=1"
            in result.simulation["output"]
        )
    assert "ATTENTION_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE top_fsm_axi_decoder_block_fixture hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "LAYER_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "FINAL_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOP_STABILITY_TRACE top_fsm_axi_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE top_fsm_axi_decoder_block_fixture estimated_iob_bits=132 prior_layer_bonded_iob=132 prior_layer_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]

    assert "top_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "tb_top_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_layer_fsm_axi_decoder_block_fixture.sv" not in result.files
    assert "tb_decoder_block_axi_attention_mlp_fixture.sv" not in result.files
    assert (tmp_path / "top_fsm_axi_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "top_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module top_fsm_axi_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "layer_fsm_axi_decoder_block_fixture u_layer_fsm_axi_decoder_block_fixture" in sv_text
    assert ".start_i(layer_start_r)" in sv_text
    assert ".done_o(layer_done_w)" in sv_text
    assert "LAYER_RELEASE" in sv_text
    assert "if (!layer_done_w)" in sv_text
    assert "layer_status_w[48 +: 4]" in sv_text
    assert "axi_rdata" not in top_port_text
    assert "axi_araddr" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "hidden_input_i" not in top_port_text
    assert "debug_array" not in top_port_text

    tb_text = (tmp_path / "tb_top_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "layer start was not held while layer busy" in tb_text
    assert "layer start was not deasserted after layer done_o" in tb_text
    assert "dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath" in tb_text
    assert "u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_v_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_o_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "TOP_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_decoder_block_fixture" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "top_fsm_axi_decoder_block_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "top_fsm_axi_decoder_block_fixture"
    assert golden["coverage_level"] == "top_fsm_axi_decoder_block_fixture"
    assert golden["top_fsm_axi_decoder_block_fixture"] is True
    assert golden["child_modules"][0]["name"] == "layer_fsm_axi_decoder_block_fixture"
    assert golden["child_modules"][0]["instantiated"] is True
    assert {child["name"] for child in golden["nested_layer_decoder_block_child_coverage_summary"]} == {
        "layer_fsm_axi_decoder_block_fixture",
        "decoder_block_axi_attention_mlp_fixture",
        "decoder_child_axi_attention_datapath",
        "residual_mlp_fixture",
        "rmsnorm_rope_source_path",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
    }
    assert golden["final_top_output_vector"] == [12, -6, 18, 6]
    assert golden["layer_child_output_vector"] == [12, -6, 18, 6]
    assert "0.012 ns" in golden["timing_margin_note"]
    assert "status was 64 bits" in golden["timing_margin_note"]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "top_fsm_axi_decoder_block_fixture"
    assert report["coverage_level"] == "top_fsm_axi_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["top_fsm_axi_decoder_block_fixture"] is True
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_layer_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "layer_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["fsm_state_order"] == ["IDLE", "LAYER_START", "LAYER_BUSY", "LAYER_RELEASE", "DONE"]
    assert report["ordered_top_trace"] == [
        "layer_fsm_axi_decoder_block_fixture_start",
        "layer_fsm_axi_decoder_block_fixture_done",
    ]
    assert report["simulation_top_trace"]["recorded"] is True
    assert report["simulation_top_trace"]["top_trace_hex"] == "0x5453"
    assert report["simulation_top_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_top_trace"]["block_trace_hex"] == "0xb2b1a2a1"
    assert report["simulation_top_trace"]["axi_metadata_bits_hex"] == "0xf"
    assert report["layer_trace"]["ordered_events"] == [
        "decoder_block_axi_attention_mlp_fixture_start",
        "decoder_block_axi_attention_mlp_fixture_done",
    ]
    assert report["decoder_block_trace"]["ordered_events"] == [
        "axi_attention_start",
        "axi_attention_done",
        "mlp_start",
        "mlp_done",
    ]
    assert report["axi_attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_axi_start",
        "projection_axi_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["top_to_layer_child_start_hold_deassert_release_evidence"]["done_seen_while_start_high"] is True
    assert report["top_to_layer_child_start_hold_deassert_release_evidence"][
        "done_release_seen_after_start_deassert"
    ] is True
    propagation = report["top_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["top_status_bits_hex"] == "0xf"
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["child_block_status_bits_hex"] == "0xf"
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    for projection in ("q", "k", "v", "o"):
        child_evidence = report["axi_projection_children_observed"][projection]
        assert child_evidence["r_metadata_trace"]["rid_ok"] is True
        assert child_evidence["r_metadata_trace"]["rresp_ok"] is True
        assert child_evidence["r_metadata_trace"]["rlast_ok"] is True
        assert child_evidence["payload_trace"]["payload_link_match_passed"] is True
        assert child_evidence["output_trace"]["projection_output_vector"] == [484, 1904]
        assert child_evidence["round_trip_evidence"]["round_trip_passed"] is True
    assert report["layer_child_output_vector"] == [12, -6, 18, 6]
    assert report["final_top_output_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_top_hierarchy"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["top_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_child_vectors"] is False
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_axi_data"] is False
    assert report["compact_io_observed_trace"]["exposes_axi_debug_buses"] is False
    assert "0.012 ns" in report["timing_margin_note"]
    assert "fixture timing without board-level I/O delay" in report["timing_caveat"]
    assert "real GPTQ checkpoint payload streaming" in report["omitted_operations"]
    assert "DDR/AXI full integration" in report["omitted_operations"]
    assert "token loop" in report["omitted_operations"]
    assert "target multi-layer LLaMA" in report["omitted_operations"]
    assert "full LLaMA layer execution" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "true softmax" in report["omitted_operations"]
    assert "true SiLU" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "real_GPTQ_checkpoint_payload_streaming" in report["does_not_claim"]
    assert "DDR_AXI_full_integration" in report["does_not_claim"]
    assert "token_loop" in report["does_not_claim"]
    assert "target_multi_layer_LLaMA" in report["does_not_claim"]
    assert "full_LLaMA_layer_execution" in report["does_not_claim"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "true_softmax" in report["does_not_claim"]
    assert "true_SiLU" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_token_loop_axi_decoder_block_fixture_sequences_two_axi_top_calls_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("token_loop_axi_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS token_loop_axi_decoder_block_fixture" in result.simulation["output"]
    assert (
        "TOKEN_LOOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture trace_hex=0x64636261"
        in result.simulation["output"]
    )
    assert "events=token0_start,token0_done,token1_start,token1_done" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_axi_decoder_block_fixture token=0 top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_axi_decoder_block_fixture token=1 top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_decoder_block_fixture token=0" in result.simulation[
        "output"
    ]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_decoder_block_fixture token=1" in result.simulation[
        "output"
    ]
    assert "release_seen=1" in result.simulation["output"]
    assert "TOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=0" in result.simulation[
        "output"
    ]
    assert "TOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=1" in result.simulation[
        "output"
    ]
    assert "LAYER_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=1 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "BLOCK_AXI_TRACE token_loop_axi_decoder_block_fixture token=1 block_trace_hex=0xb2b1a2a1 attention_trace_hex=0x323122211211 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "AXI_ATTENTION_CHILD_TRACE token_loop_axi_decoder_block_fixture token=1 trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE token_loop_axi_decoder_block_fixture token=1 trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "TOKEN_LOOP_AXI_METADATA_PROPAGATION_TRACE token_loop_axi_decoder_block_fixture loop_bits=0xf" in result.simulation[
        "output"
    ]
    assert "top_bits=0xf layer_bits=0xf child_block_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation[
        "output"
    ]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_AR_TRACE token_loop_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE token_loop_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_PAYLOAD_TRACE token_loop_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_OUTPUT_TRACE token_loop_axi_decoder_block_fixture projection={projection} output=484,1904"
            in result.simulation["output"]
        )
        assert (
            f"AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE token_loop_axi_decoder_block_fixture projection={projection} packed_bytes=32 unpacked_values=64 round_trip_passed=1"
            in result.simulation["output"]
        )
    assert "ATTENTION_OUTPUT_TRACE token_loop_axi_decoder_block_fixture token=1 output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE token_loop_axi_decoder_block_fixture token=1 hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE token_loop_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_LOOP_STABILITY_TRACE token_loop_axi_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "TOKEN_OUTPUT_POLICY_TRACE token_loop_axi_decoder_block_fixture repeated_deterministic_outputs=1 token_dependent_outputs=0" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE token_loop_axi_decoder_block_fixture estimated_iob_bits=132 prior_top_fsm_axi_bonded_iob=132 prior_top_fsm_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]

    assert "token_loop_axi_decoder_block_fixture.sv" in result.files
    assert "tb_token_loop_axi_decoder_block_fixture.sv" in result.files
    assert "top_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_top_fsm_axi_decoder_block_fixture.sv" not in result.files
    assert not (tmp_path / "tb_top_fsm_axi_decoder_block_fixture.sv").exists()
    assert not (tmp_path / "tb_layer_fsm_axi_decoder_block_fixture.sv").exists()
    assert not (tmp_path / "tb_decoder_block_axi_attention_mlp_fixture.sv").exists()
    assert (tmp_path / "token_loop_axi_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "token_loop_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module token_loop_axi_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "top_fsm_axi_decoder_block_fixture u_top_fsm_axi_decoder_block_fixture" in sv_text
    assert ".start_i(top_start_r)" in sv_text
    assert ".done_o(top_done_w)" in sv_text
    assert "TOKEN0_RELEASE" in sv_text
    assert "TOKEN1_RELEASE" in sv_text
    assert "if (!top_done_w)" in sv_text
    assert "top_status_w[48 +: 4]" in sv_text
    assert "axi_rdata" not in top_port_text
    assert "axi_araddr" not in top_port_text
    assert "debug_array" not in top_port_text

    tb_text = (tmp_path / "tb_token_loop_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while top child busy" in tb_text
    assert "child start was not deasserted after top child done_o" in tb_text
    assert "token0_release_seen_r" in tb_text
    assert "token1_release_seen_r" in tb_text
    assert "u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath" in tb_text
    assert "u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_v_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_o_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "TOKEN_LOOP_AXI_METADATA_PROPAGATION_TRACE token_loop_axi_decoder_block_fixture" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "token_loop_axi_decoder_block_fixture"
    assert report["coverage_level"] == "token_loop_axi_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["token_loop_axi_decoder_block_fixture"] is True
    assert report["uses_top_fsm_axi_decoder_block_fixture"] is True
    assert report["fixture_token_count"] == 2
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_top_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "top_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "top_fsm_axi_decoder_block_fixture",
        "layer_fsm_axi_decoder_block_fixture",
        "decoder_block_axi_attention_mlp_fixture",
        "decoder_child_axi_attention_datapath",
        "residual_mlp_fixture",
        "rmsnorm_rope_source_path",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
    ]
    assert report["fsm_state_order"] == [
        "IDLE",
        "TOKEN0_START",
        "TOKEN0_BUSY",
        "TOKEN0_RELEASE",
        "TOKEN1_START",
        "TOKEN1_BUSY",
        "TOKEN1_RELEASE",
        "DONE",
    ]
    assert report["simulation_token_start_done_trace"]["recorded"] is True
    assert report["simulation_token_start_done_trace"]["trace_hex"] == "0x64636261"
    assert report["observed_token_start_done_trace"] == report["token_start_done_trace"]
    assert len(report["per_token_child_start_hold_deassert_release_evidence"]) == 2
    assert all(entry["done_seen_while_start_high"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert all(entry["done_release_seen_after_start_deassert"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    propagation = report["token_loop_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["loop_status_bits_hex"] == "0xf"
    assert propagation["top_status_bits_hex"] == "0xf"
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["child_block_status_bits_hex"] == "0xf"
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    for projection in ("q", "k", "v", "o"):
        child_evidence = report["axi_projection_children_observed"][projection]
        assert child_evidence["r_metadata_trace"]["rid_ok"] is True
        assert child_evidence["payload_trace"]["payload_link_match_passed"] is True
        assert child_evidence["output_trace"]["projection_output_vector"] == [484, 1904]
        assert child_evidence["round_trip_evidence"]["round_trip_passed"] is True
    assert report["token_output_vectors"] == [[12, -6, 18, 6], [12, -6, 18, 6]]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_token_loop_hierarchy"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["repeated_deterministic_fixture_outputs"] is True
    assert report["token_outputs_are_token_dependent"] is False
    assert report["loop_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_child_vectors"] is False
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_axi_data"] is False
    assert "WHS 0.005 ns" in report["timing_margin_note"]
    assert "status was 64 bits" in report["timing_margin_note"]
    assert "real GPTQ checkpoint payload streaming" in report["omitted_operations"]
    assert "full DDR/AXI integration" in report["omitted_operations"]
    assert "real LLaMA token prefill/decode semantics" in report["omitted_operations"]
    assert "target multi-layer LLaMA iteration" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "real_GPTQ_checkpoint_payload_streaming" in report["does_not_claim"]
    assert "DDR_AXI_full_integration" in report["does_not_claim"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_model_fsm_axi_decoder_block_fixture_sequences_two_axi_token_loop_calls_and_reports_gate(
    tmp_path: Path,
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("model_fsm_axi_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS model_fsm_axi_decoder_block_fixture" in result.simulation["output"]
    assert (
        "MODEL_FSM_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture trace_hex=0x74737271"
        in result.simulation["output"]
    )
    assert "events=layer0_start,layer0_done,layer1_start,layer1_done" in result.simulation["output"]
    assert "MODEL_CHILD_CALL_TRACE model_fsm_axi_decoder_block_fixture layer=0 token_trace_hex=0x64636261" in result.simulation[
        "output"
    ]
    assert "MODEL_CHILD_CALL_TRACE model_fsm_axi_decoder_block_fixture layer=1 token_trace_hex=0x64636261" in result.simulation[
        "output"
    ]
    assert "axi_metadata_bits=0xf output=12,-6,18,6" in result.simulation["output"]
    assert "MODEL_CHILD_START_HOLD_TRACE model_fsm_axi_decoder_block_fixture layer=0" in result.simulation[
        "output"
    ]
    assert "MODEL_CHILD_START_HOLD_TRACE model_fsm_axi_decoder_block_fixture layer=1" in result.simulation[
        "output"
    ]
    assert "release_seen=1" in result.simulation["output"]
    assert "TOKEN_LOOP_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture token_trace_hex=0x64636261" in result.simulation[
        "output"
    ]
    assert "TOP_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture top_trace_hex=0x5453" in result.simulation[
        "output"
    ]
    assert "LAYER_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "BLOCK_AXI_TRACE model_fsm_axi_decoder_block_fixture block_trace_hex=0xb2b1a2a1 attention_trace_hex=0x323122211211 axi_metadata_bits=0xf" in result.simulation[
        "output"
    ]
    assert "MODEL_FSM_AXI_METADATA_PROPAGATION_TRACE model_fsm_axi_decoder_block_fixture model_bits=0xf" in result.simulation[
        "output"
    ]
    assert "token_loop_bits=0xf top_bits=0xf layer_bits=0xf child_block_bits=0xf q_bits=0xf k_bits=0xf v_bits=0xf o_bits=0xf" in result.simulation[
        "output"
    ]
    for projection in ("q", "k", "v", "o"):
        assert (
            f"AXI_PROJECTION_CHILD_R_METADATA_TRACE model_fsm_axi_decoder_block_fixture projection={projection} "
            in result.simulation["output"]
        )
    assert "ATTENTION_OUTPUT_TRACE model_fsm_axi_decoder_block_fixture layer=1 token=1 output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE model_fsm_axi_decoder_block_fixture layer=1 token=1 hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE model_fsm_axi_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "MODEL_FSM_STABILITY_TRACE model_fsm_axi_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "MODEL_OUTPUT_POLICY_TRACE model_fsm_axi_decoder_block_fixture repeated_deterministic_outputs=1 layer_dependent_outputs=0" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE model_fsm_axi_decoder_block_fixture estimated_iob_bits=132 prior_token_loop_axi_bonded_iob=132 prior_token_loop_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]

    assert "model_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "tb_model_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "token_loop_axi_decoder_block_fixture.sv" in result.files
    assert "top_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_axi_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_axi_stream_integration.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_token_loop_axi_decoder_block_fixture.sv" not in result.files
    assert not (tmp_path / "tb_token_loop_axi_decoder_block_fixture.sv").exists()
    assert not (tmp_path / "tb_top_fsm_axi_decoder_block_fixture.sv").exists()
    assert (tmp_path / "model_fsm_axi_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "model_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module model_fsm_axi_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "token_loop_axi_decoder_block_fixture u_token_loop_axi_decoder_block_fixture" in sv_text
    assert ".start_i(token_start_r)" in sv_text
    assert ".done_o(token_done_w)" in sv_text
    assert "LAYER0_RELEASE" in sv_text
    assert "LAYER1_RELEASE" in sv_text
    assert "if (!token_done_w)" in sv_text
    assert "token_status_w[48 +: 4]" in sv_text
    assert "axi_rdata" not in top_port_text
    assert "axi_araddr" not in top_port_text
    assert "debug_array" not in top_port_text

    tb_text = (tmp_path / "tb_model_fsm_axi_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "layer0_release_seen_r" in tb_text
    assert "layer1_release_seen_r" in tb_text
    assert "u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath" in tb_text
    assert "u_k_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_v_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "u_o_projection_axi_stream_integration.integration_status_o[45:42]" in tb_text
    assert "MODEL_FSM_AXI_METADATA_PROPAGATION_TRACE model_fsm_axi_decoder_block_fixture" in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "model_fsm_axi_decoder_block_fixture"
    assert report["coverage_level"] == "model_fsm_axi_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["model_fsm_axi_decoder_block_fixture"] is True
    assert report["uses_token_loop_axi_decoder_block_fixture"] is True
    assert report["fixture_layer_count"] == 2
    assert report["fixture_token_count_per_layer"] == 2
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_token_loop_axi_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "token_loop_axi_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "token_loop_axi_decoder_block_fixture",
        "top_fsm_axi_decoder_block_fixture",
        "layer_fsm_axi_decoder_block_fixture",
        "decoder_block_axi_attention_mlp_fixture",
        "decoder_child_axi_attention_datapath",
        "residual_mlp_fixture",
        "rmsnorm_rope_source_path",
        "q_projection_axi_stream_integration",
        "k_projection_axi_stream_integration",
        "v_projection_axi_stream_integration",
        "o_projection_axi_stream_integration",
        "attention_kv_cache_fixture",
    ]
    assert report["fsm_state_order"] == [
        "IDLE",
        "LAYER0_START",
        "LAYER0_BUSY",
        "LAYER0_RELEASE",
        "LAYER1_START",
        "LAYER1_BUSY",
        "LAYER1_RELEASE",
        "DONE",
    ]
    assert report["simulation_model_layer_start_done_trace"]["recorded"] is True
    assert report["simulation_model_layer_start_done_trace"]["trace_hex"] == "0x74737271"
    assert report["observed_model_layer_start_done_trace"] == report["model_layer_start_done_trace"]
    assert len(report["per_layer_child_start_hold_deassert_release_evidence"]) == 2
    assert all(entry["done_seen_while_start_high"] is True for entry in report["per_layer_child_start_hold_deassert_release_evidence"])
    assert all(entry["done_release_seen_after_start_deassert"] is True for entry in report["per_layer_child_start_hold_deassert_release_evidence"])
    propagation = report["model_compact_status_axi_metadata_propagation"]
    assert propagation["propagated"] is True
    assert propagation["model_status_bits_hex"] == "0xf"
    assert propagation["token_loop_status_bits_hex"] == "0xf"
    assert propagation["top_status_bits_hex"] == "0xf"
    assert propagation["layer_status_bits_hex"] == "0xf"
    assert propagation["child_block_status_bits_hex"] == "0xf"
    assert propagation["q_status_bits_hex"] == "0xf"
    assert propagation["k_status_bits_hex"] == "0xf"
    assert propagation["v_status_bits_hex"] == "0xf"
    assert propagation["o_status_bits_hex"] == "0xf"
    assert set(report["axi_projection_children_observed"]) == {"q", "k", "v", "o"}
    assert report["layer_output_vectors"] == [[12, -6, 18, 6], [12, -6, 18, 6]]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_model_fsm_hierarchy"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["repeated_deterministic_fixture_outputs"] is True
    assert report["layer_outputs_are_layer_dependent"] is False
    assert report["model_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_child_vectors"] is False
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert report["compact_io_observed_trace"]["exposes_128b_axi_data"] is False
    assert "WHS 0.009 ns" in report["timing_margin_note"]
    assert "status was 64 bits" in report["timing_margin_note"]
    assert "real GPTQ checkpoint payload streaming" in report["omitted_operations"]
    assert "full DDR/AXI integration" in report["omitted_operations"]
    assert "real LLaMA token prefill/decode semantics" in report["omitted_operations"]
    assert "target 16-layer LLaMA iteration" in report["omitted_operations"]
    assert "target multi-layer LLaMA numerical execution" in report["omitted_operations"]
    assert "full LLaMA model execution" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "real_GPTQ_checkpoint_payload_streaming" in report["does_not_claim"]
    assert "DDR_AXI_full_integration" in report["does_not_claim"]
    assert "target_16_layer_LLaMA_iteration" in report["does_not_claim"]
    assert "target_multi_layer_LLaMA_numerical_execution" in report["does_not_claim"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_ddr_axi_board_shell_fixture_wraps_model_fsm_and_reports_projection_plan(
    tmp_path: Path,
):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("ddr_axi_board_shell_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS ddr_axi_board_shell_fixture" in result.simulation["output"]
    assert (
        "DDR_AXI_BOARD_SHELL_TRACE ddr_axi_board_shell_fixture shell_trace_hex=0x8281 model_trace_hex=0x74737271"
        in result.simulation["output"]
    )
    assert "events=model_fsm_axi_decoder_block_fixture_start,model_fsm_axi_decoder_block_fixture_done" in result.simulation[
        "output"
    ]
    assert "DDR_AXI_MODEL_CHILD_START_HOLD_TRACE ddr_axi_board_shell_fixture" in result.simulation["output"]
    assert "release_seen=1" in result.simulation["output"]
    assert "DDR_AXI_COMPACT_STATUS_TRACE ddr_axi_board_shell_fixture projection_count=7 model_bits=0xf request_mask=0x7f attention_mask=0xf mlp_mask=0x7" in result.simulation[
        "output"
    ]
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert (
            f"DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection={projection} "
            in result.simulation["output"]
        )
        assert "layout_dependency=blocked_by_real_gptq_weight_layout_preflight" in result.simulation["output"]
        assert "payload_dependency=blocked_by_gptq_payload_probe" in result.simulation["output"]
    assert "FINAL_OUTPUT_TRACE ddr_axi_board_shell_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "DDR_AXI_BOARD_SHELL_STABILITY_TRACE ddr_axi_board_shell_fixture stable=1" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE ddr_axi_board_shell_fixture estimated_iob_bits=132 prior_model_fsm_axi_bonded_iob=132 prior_model_fsm_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0 board_io_constraints=0" in result.simulation[
        "output"
    ]

    assert "ddr_axi_board_shell_fixture.sv" in result.files
    assert "tb_ddr_axi_board_shell_fixture.sv" in result.files
    assert "model_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "token_loop_axi_decoder_block_fixture.sv" in result.files
    assert "top_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_axi_decoder_block_fixture.sv" in result.files
    assert "decoder_block_axi_attention_mlp_fixture.sv" in result.files
    assert "tb_model_fsm_axi_decoder_block_fixture.sv" not in result.files
    assert not (tmp_path / "tb_model_fsm_axi_decoder_block_fixture.sv").exists()
    assert (tmp_path / "ddr_axi_board_shell_fixture_golden.json").exists()

    sv_text = (tmp_path / "ddr_axi_board_shell_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module ddr_axi_board_shell_fixture #(") : sv_text.index("    localparam logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "model_fsm_axi_decoder_block_fixture u_model_fsm_axi_decoder_block_fixture" in sv_text
    assert ".start_i(model_start_r)" in sv_text
    assert ".done_o(model_done_w)" in sv_text
    assert "MODEL_RELEASE" in sv_text
    assert "if (!model_done_w)" in sv_text
    assert "ALL_REQUEST_MASK" in sv_text
    assert "ATTENTION_REQUEST_MASK" in sv_text
    assert "MLP_REQUEST_MASK" in sv_text
    assert "axi_rdata" not in top_port_text
    assert "axi_araddr" not in top_port_text
    assert "kv_cache" not in top_port_text
    assert "debug" not in top_port_text

    tb_text = (tmp_path / "tb_ddr_axi_board_shell_fixture.sv").read_text(encoding="utf-8")
    assert "model_release_seen_r" in tb_text
    assert "u_model_fsm_axi_decoder_block_fixture.u_token_loop_axi_decoder_block_fixture" in tb_text
    assert "DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=gate_proj" in tb_text
    assert "DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=down_proj" in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "ddr_axi_board_shell_fixture"
    assert report["coverage_level"] == "ddr_axi_board_shell_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["ddr_axi_board_shell_fixture"] is True
    assert report["uses_model_fsm_axi_decoder_block_fixture"] is True
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_model_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "model_fsm_axi_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["projection_weight_stream_plan"]["projection_count"] == 7
    assert report["projection_weight_stream_plan"]["attention_projection_order"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert report["projection_weight_stream_plan"]["mlp_projection_order"] == ["gate_proj", "up_proj", "down_proj"]
    assert report["projection_weight_stream_plan"]["all_projection_layout_dependency"] == (
        "blocked_by_real_gptq_weight_layout_preflight"
    )
    assert report["projection_weight_stream_plan"]["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert {entry["name"] for entry in report["compact_projection_stream_summary"]} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert report["compact_axi_request_status_plan"]["projection_count"] == 7
    assert report["compact_axi_request_status_plan"]["all_projection_request_mask_hex"] == "0x7f"
    assert report["sampled_gptq_payload_provenance"]["artifact"] == "gptq_payload_probe.json"
    assert report["sampled_gptq_payload_provenance"]["all_projection_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert report["simulation_board_shell_start_done_trace"]["recorded"] is True
    assert report["simulation_board_shell_start_done_trace"]["shell_trace_hex"] == "0x8281"
    assert report["shell_to_model_child_start_hold_deassert_release_evidence"]["done_seen_while_start_high"] is True
    assert report["shell_to_model_child_start_hold_deassert_release_evidence"][
        "done_release_seen_after_start_deassert"
    ] is True
    assert report["observed_projection_request_count"] == 7
    assert report["observed_attention_projection_request_count"] == 4
    assert report["observed_mlp_projection_request_count"] == 3
    assert report["compact_axi_request_status_observed"]["projection_count"] == 7
    assert report["compact_axi_request_status_observed"]["all_projection_request_mask_hex"] == "0x7f"
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["shell_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["compact_io_observed_trace"]["exposes_128b_axi_data"] is False
    assert "real DDR controller IP integration" in report["omitted_operations"]
    assert "board-level ZCU104 signoff" in report["omitted_operations"]
    assert "real_DDR_controller_IP_integration" in report["does_not_claim"]
    assert "board_level_ZCU104_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_layer_fsm_decoder_block_fixture_schedules_real_decoder_block(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("layer_fsm_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS layer_fsm_decoder_block_fixture" in result.simulation["output"]
    assert "LAYER_TRACE layer_fsm_decoder_block_fixture layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "BLOCK_TRACE layer_fsm_decoder_block_fixture block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "ATTENTION_CHILD_TRACE layer_fsm_decoder_block_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE layer_fsm_decoder_block_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "LAYER_BLOCK_CHILD_START_HOLD_TRACE layer_fsm_decoder_block_fixture" in result.simulation["output"]
    assert "FINAL_DECODER_BLOCK_OUTPUT_TRACE layer_fsm_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE layer_fsm_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "LAYER_STABILITY_TRACE layer_fsm_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE layer_fsm_decoder_block_fixture estimated_iob_bits=132 previous_decoder_block_iob=228 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b=0 exposed_kv_arrays=0" in result.simulation[
        "output"
    ]

    assert "layer_fsm_decoder_block_fixture.sv" in result.files
    assert "tb_layer_fsm_decoder_block_fixture.sv" in result.files
    assert "decoder_block_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_decoder_block_attention_mlp_fixture.sv" not in result.files
    assert (tmp_path / "layer_fsm_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "layer_fsm_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module layer_fsm_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "decoder_block_attention_mlp_fixture u_decoder_block_attention_mlp_fixture" in sv_text
    assert ".start_i(block_start_r)" in sv_text
    assert "BLOCK_RELEASE" in sv_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "hidden_input_i" not in top_port_text

    tb_text = (tmp_path / "tb_layer_fsm_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "dut.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o" in tb_text
    assert "dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "layer_fsm_decoder_block_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "layer_fsm_decoder_block_fixture"
    assert golden["coverage_level"] == "layer_fsm_decoder_block_fixture"
    assert golden["layer_fsm_decoder_block_fixture"] is True
    assert golden["child_modules"][0]["name"] == "decoder_block_attention_mlp_fixture"
    assert golden["child_modules"][0]["instantiated"] is True
    assert golden["final_layer_output_vector"] == [12, -6, 18, 6]
    assert golden["decoder_block_child_output_vector"] == [12, -6, 18, 6]
    assert "0.005 ns" in golden["timing_margin_note"]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "layer_fsm_decoder_block_fixture"
    assert report["coverage_level"] == "layer_fsm_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["layer_fsm_decoder_block_fixture"] is True
    assert report["ordered_layer_trace"] == [
        "decoder_block_attention_mlp_fixture_start",
        "decoder_block_attention_mlp_fixture_done",
    ]
    assert report["ordered_block_trace"] == ["attention_start", "attention_done", "mlp_start", "mlp_done"]
    assert report["attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["layer_to_block_child_start_hold_deassert_release_evidence"]["done_release_seen_after_start_deassert"] is True
    assert report["decoder_block_child_output_vector"] == [12, -6, 18, 6]
    assert report["final_layer_output_vector"] == [12, -6, 18, 6]
    assert report["layer_done_output_status_stability_observed"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_layer_hierarchy"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert "0.005 ns" in report["timing_margin_note"]
    assert "fixture timing without board-level I/O delay" in report["timing_caveat"]
    assert "target multi-layer LLaMA iteration" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_top_fsm_decoder_block_fixture_schedules_real_layer_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("top_fsm_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS top_fsm_decoder_block_fixture" in result.simulation["output"]
    assert "TOP_DECODER_BLOCK_TRACE top_fsm_decoder_block_fixture top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "LAYER_TRACE top_fsm_decoder_block_fixture layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "BLOCK_TRACE top_fsm_decoder_block_fixture block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "ATTENTION_CHILD_TRACE top_fsm_decoder_block_fixture trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE top_fsm_decoder_block_fixture trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "TOP_LAYER_CHILD_START_HOLD_TRACE top_fsm_decoder_block_fixture" in result.simulation["output"]
    assert "release_seen=1" in result.simulation["output"]
    assert "LAYER_OUTPUT_TRACE top_fsm_decoder_block_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "FINAL_OUTPUT_TRACE top_fsm_decoder_block_fixture output=12,-6,18,6" in result.simulation["output"]
    assert "TOP_STABILITY_TRACE top_fsm_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "COMPACT_IO_TRACE top_fsm_decoder_block_fixture estimated_iob_bits=132 prior_layer_bonded_iob=132 prior_layer_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b=0 exposed_kv_arrays=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    assert "top_fsm_decoder_block_fixture.sv" in result.files
    assert "tb_top_fsm_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_decoder_block_fixture.sv" in result.files
    assert "decoder_block_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_layer_fsm_decoder_block_fixture.sv" not in result.files
    assert (tmp_path / "top_fsm_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "top_fsm_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module top_fsm_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "layer_fsm_decoder_block_fixture u_layer_fsm_decoder_block_fixture" in sv_text
    assert ".start_i(layer_start_r)" in sv_text
    assert ".done_o(layer_done_w)" in sv_text
    assert "LAYER_RELEASE" in sv_text
    assert "if (!layer_done_w)" in sv_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "hidden_input_i" not in top_port_text
    assert "debug_array" not in top_port_text

    tb_text = (tmp_path / "tb_top_fsm_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "layer start was not held while layer busy" in tb_text
    assert "layer start was not deasserted after layer done_o" in tb_text
    assert "dut.u_layer_fsm_decoder_block_fixture.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o" in tb_text
    assert "dut.u_layer_fsm_decoder_block_fixture.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r" in tb_text
    assert "output=12,-6,18,6 status=" not in tb_text

    golden = json.loads((tmp_path / "top_fsm_decoder_block_fixture_golden.json").read_text(encoding="utf-8"))
    assert golden["kernel"] == "top_fsm_decoder_block_fixture"
    assert golden["coverage_level"] == "top_fsm_decoder_block_fixture"
    assert golden["top_fsm_decoder_block_fixture"] is True
    assert golden["child_modules"][0]["name"] == "layer_fsm_decoder_block_fixture"
    assert golden["child_modules"][0]["instantiated"] is True
    assert {child["name"] for child in golden["nested_layer_decoder_block_child_coverage_summary"]} == {
        "layer_fsm_decoder_block_fixture",
        "decoder_block_attention_mlp_fixture",
        "decoder_child_attention_datapath",
        "residual_mlp_fixture",
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    }
    assert golden["final_top_output_vector"] == [12, -6, 18, 6]
    assert golden["layer_child_output_vector"] == [12, -6, 18, 6]
    assert "0.021 ns" in golden["timing_margin_note"]
    assert "status was 64 bits" in golden["timing_margin_note"]

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "top_fsm_decoder_block_fixture"
    assert report["coverage_level"] == "top_fsm_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["top_fsm_decoder_block_fixture"] is True
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_layer_fsm_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "layer_fsm_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert report["fsm_state_order"] == ["IDLE", "LAYER_START", "LAYER_BUSY", "LAYER_RELEASE", "DONE"]
    assert report["ordered_top_trace"] == [
        "layer_fsm_decoder_block_fixture_start",
        "layer_fsm_decoder_block_fixture_done",
    ]
    assert report["simulation_top_trace"]["recorded"] is True
    assert report["simulation_top_trace"]["top_trace_hex"] == "0x5453"
    assert report["simulation_top_trace"]["layer_trace_hex"] == "0x4241"
    assert report["simulation_top_trace"]["block_trace_hex"] == "0xb2b1a2a1"
    assert report["layer_trace"]["ordered_events"] == [
        "decoder_block_attention_mlp_fixture_start",
        "decoder_block_attention_mlp_fixture_done",
    ]
    assert report["decoder_block_trace"]["ordered_events"] == [
        "attention_start",
        "attention_done",
        "mlp_start",
        "mlp_done",
    ]
    assert report["attention_child_trace"]["ordered_events"] == [
        "source_path_start",
        "source_path_done",
        "projection_shell_start",
        "projection_shell_done",
        "attention_kv_start",
        "attention_kv_done",
    ]
    assert report["mlp_child_trace"]["ordered_events"] == [
        "residual0_start",
        "residual0_done",
        "gate_up_start",
        "gate_up_done",
        "swiglu_start",
        "swiglu_done",
        "down_start",
        "down_done",
        "residual1_start",
        "residual1_done",
    ]
    assert report["top_to_layer_child_start_hold_deassert_release_evidence"]["done_seen_while_start_high"] is True
    assert report["top_to_layer_child_start_hold_deassert_release_evidence"][
        "done_release_seen_after_start_deassert"
    ] is True
    assert report["layer_child_output_vector"] == [12, -6, 18, 6]
    assert report["final_top_output_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_top_hierarchy"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["top_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["compact_io_observed_trace"]["exposes_child_vectors"] is False
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert "0.021 ns" in report["timing_margin_note"]
    assert "status was 64 bits" in report["timing_margin_note"]
    assert "fixture timing without board-level I/O delay" in report["timing_caveat"]
    assert "target multi-layer LLaMA iteration" in report["omitted_operations"]
    assert "complete Q/K/V/O attention math" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True


def test_emit_token_loop_decoder_block_fixture_sequences_two_top_calls_and_reports_gate(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    result = emit_kernel("token_loop_decoder_block_fixture", cfg, tmp_path)
    assert result.status == "passed"
    assert "PASS token_loop_decoder_block_fixture" in result.simulation["output"]
    assert "TOKEN_LOOP_TRACE token_loop_decoder_block_fixture trace_hex=0x64636261" in result.simulation["output"]
    assert "events=token0_start,token0_done,token1_start,token1_done" in result.simulation["output"]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_decoder_block_fixture token=0 top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_CHILD_CALL_TRACE token_loop_decoder_block_fixture token=1 top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1 output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_decoder_block_fixture token=0" in result.simulation["output"]
    assert "TOKEN_CHILD_START_HOLD_TRACE token_loop_decoder_block_fixture token=1" in result.simulation["output"]
    assert "release_seen=1" in result.simulation["output"]
    assert "TOP_DECODER_BLOCK_TRACE token_loop_decoder_block_fixture token=1 top_trace_hex=0x5453 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "LAYER_TRACE token_loop_decoder_block_fixture token=1 layer_trace_hex=0x4241 block_trace_hex=0xb2b1a2a1" in result.simulation[
        "output"
    ]
    assert "BLOCK_TRACE token_loop_decoder_block_fixture token=1 block_trace_hex=0xb2b1a2a1 attention_trace_hex=0x323122211211 mlp_trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "ATTENTION_CHILD_TRACE token_loop_decoder_block_fixture token=1 trace_hex=0x323122211211" in result.simulation[
        "output"
    ]
    assert "MLP_CHILD_TRACE token_loop_decoder_block_fixture token=1 trace_hex=0x52514241323122211211" in result.simulation[
        "output"
    ]
    assert "ATTENTION_OUTPUT_TRACE token_loop_decoder_block_fixture token=1 output=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "MLP_INPUT_TRACE token_loop_decoder_block_fixture token=1 hidden=0,4,1,1 attention=4,-2,1,2" in result.simulation[
        "output"
    ]
    assert "FINAL_OUTPUT_TRACE token_loop_decoder_block_fixture output=12,-6,18,6" in result.simulation[
        "output"
    ]
    assert "TOKEN_LOOP_STABILITY_TRACE token_loop_decoder_block_fixture stable=1" in result.simulation["output"]
    assert "TOKEN_OUTPUT_POLICY_TRACE token_loop_decoder_block_fixture repeated_deterministic_outputs=1 token_dependent_outputs=0" in result.simulation[
        "output"
    ]
    assert "COMPACT_IO_TRACE token_loop_decoder_block_fixture estimated_iob_bits=132 prior_top_fsm_bonded_iob=132 prior_top_fsm_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b=0 exposed_kv_arrays=0 exposed_debug_arrays=0" in result.simulation[
        "output"
    ]

    assert "token_loop_decoder_block_fixture.sv" in result.files
    assert "tb_token_loop_decoder_block_fixture.sv" in result.files
    assert "top_fsm_decoder_block_fixture.sv" in result.files
    assert "layer_fsm_decoder_block_fixture.sv" in result.files
    assert "decoder_block_attention_mlp_fixture.sv" in result.files
    assert "decoder_child_attention_datapath.sv" in result.files
    assert "residual_mlp_fixture.sv" in result.files
    assert "rmsnorm_rope_source_path.sv" in result.files
    assert "projection_internal_stream_shell.sv" in result.files
    assert "attention_kv_cache_fixture.sv" in result.files
    assert "tb_top_fsm_decoder_block_fixture.sv" not in result.files
    assert (tmp_path / "token_loop_decoder_block_fixture_golden.json").exists()

    sv_text = (tmp_path / "token_loop_decoder_block_fixture.sv").read_text(encoding="utf-8")
    top_port_text = sv_text[
        sv_text.index("module token_loop_decoder_block_fixture #(") : sv_text.index("    typedef enum logic")
    ]
    assert "input  logic                                      aclk" in top_port_text
    assert "input  logic                                      aresetn" in top_port_text
    assert "input  logic                                      start_i" in top_port_text
    assert "output logic                                      done_o" in top_port_text
    assert "output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o" in top_port_text
    assert "output logic [STATUS_WIDTH-1:0]                   status_o" in top_port_text
    assert "parameter int STATUS_WIDTH = 64" in top_port_text
    assert "top_fsm_decoder_block_fixture u_top_fsm_decoder_block_fixture" in sv_text
    assert ".start_i(top_start_r)" in sv_text
    assert ".done_o(top_done_w)" in sv_text
    assert ".final_output_o(top_output_w)" in sv_text
    assert "TOKEN0_RELEASE" in sv_text
    assert "TOKEN1_RELEASE" in sv_text
    assert "if (!top_done_w)" in sv_text
    assert "mem_rsp_word_i" not in top_port_text
    assert "hidden_input_i" not in top_port_text
    assert "attention_output_i" not in top_port_text
    assert "debug_array" not in top_port_text

    tb_text = (tmp_path / "tb_token_loop_decoder_block_fixture.sv").read_text(encoding="utf-8")
    assert "child start was not held while top child busy" in tb_text
    assert "child start was not deasserted after top child done_o" in tb_text
    assert "token0_release_seen_r" in tb_text
    assert "token1_release_seen_r" in tb_text
    assert "u_top_fsm_decoder_block_fixture.u_layer_fsm_decoder_block_fixture.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o" in tb_text
    assert "u_top_fsm_decoder_block_fixture.u_layer_fsm_decoder_block_fixture.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r" in tb_text
    assert "TOKEN_CHILD_CALL_TRACE token_loop_decoder_block_fixture token=0 top_trace_hex=0x5453" not in tb_text
    assert "FINAL_OUTPUT_TRACE token_loop_decoder_block_fixture output=12,-6,18,6 status=" not in tb_text
    assert "MLP_INPUT_TRACE token_loop_decoder_block_fixture token=1 hidden=0,4,1,1 attention=4,-2,1,2" not in tb_text

    report = json.loads((tmp_path / "kernel_report.json").read_text(encoding="utf-8"))
    assert report["kernel"] == "token_loop_decoder_block_fixture"
    assert report["coverage_level"] == "token_loop_decoder_block_fixture"
    assert report["implementation_stage"] == "not_run"
    assert report["token_loop_decoder_block_fixture"] is True
    assert report["uses_top_fsm_decoder_block_fixture"] is True
    assert report["fixture_token_count"] == 2
    assert report["numeric_policy"]["datapath_composition"] == "instantiated_top_fsm_decoder_block_fixture"
    assert report["child_modules"][0]["name"] == "top_fsm_decoder_block_fixture"
    assert report["child_modules"][0]["instantiated"] is True
    assert [child["name"] for child in report["nested_child_coverage_summary"]] == [
        "top_fsm_decoder_block_fixture",
        "layer_fsm_decoder_block_fixture",
        "decoder_block_attention_mlp_fixture",
        "decoder_child_attention_datapath",
        "residual_mlp_fixture",
        "rmsnorm_rope_source_path",
        "projection_internal_stream_shell",
        "attention_kv_cache_fixture",
    ]
    assert all(child["instantiated"] is True for child in report["nested_child_coverage_summary"])
    assert report["fsm_state_order"] == [
        "IDLE",
        "TOKEN0_START",
        "TOKEN0_BUSY",
        "TOKEN0_RELEASE",
        "TOKEN1_START",
        "TOKEN1_BUSY",
        "TOKEN1_RELEASE",
        "DONE",
    ]
    assert report["token_start_done_trace"] == ["token0_start", "token0_done", "token1_start", "token1_done"]
    assert report["simulation_token_start_done_trace"]["recorded"] is True
    assert report["simulation_token_start_done_trace"]["trace_hex"] == "0x64636261"
    assert report["observed_token_start_done_trace"] == report["token_start_done_trace"]
    assert len(report["per_token_child_start_hold_deassert_release_evidence"]) == 2
    assert all(entry["done_seen_while_start_high"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert all(entry["deasserted_after_done"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert all(entry["done_release_seen_after_start_deassert"] is True for entry in report["per_token_child_start_hold_deassert_release_evidence"])
    assert [entry["token"] for entry in report["top_fsm_traces"]] == [0, 1]
    assert report["top_fsm_traces"][0]["ordered_events"] == [
        "layer_fsm_decoder_block_fixture_start",
        "layer_fsm_decoder_block_fixture_done",
    ]
    assert report["layer_fsm_traces"][1]["ordered_events"] == [
        "decoder_block_attention_mlp_fixture_start",
        "decoder_block_attention_mlp_fixture_done",
    ]
    assert report["decoder_block_traces"][1]["ordered_events"] == [
        "attention_start",
        "attention_done",
        "mlp_start",
        "mlp_done",
    ]
    assert report["attention_child_trace"]["trace_hex"] == "0x323122211211"
    assert report["mlp_child_trace"]["trace_hex"] == "0x52514241323122211211"
    assert report["token_output_vectors"] == [[12, -6, 18, 6], [12, -6, 18, 6]]
    assert report["final_output_vector"] == [12, -6, 18, 6]
    assert report["observed_final_fixture_output_vector"] == [12, -6, 18, 6]
    assert report["fixture_output_vector"] == [12, -6, 18, 6]
    assert report["expected_golden_vector"] == [12, -6, 18, 6]
    assert report["decoder_block_consumed_attention_output_into_mlp"]["observed_through_token_loop_hierarchy"] is True
    assert report["decoder_block_consumed_attention_output_into_mlp"]["mlp_attention_input_vector"] == [4, -2, 1, 2]
    assert report["repeated_deterministic_fixture_outputs"] is True
    assert report["token_outputs_are_token_dependent"] is False
    assert report["loop_done_output_status_stability_observed"] is True
    assert report["top_level_interface_summary"]["compact_top_level_io"] is True
    assert report["exposed_port_width_summary"]["estimated_total_top_level_iob_bits"] == 132
    assert report["exposed_port_width_summary"]["status_o_bits"] == 64
    assert report["timing_resource_recovery_comparison"]["prior"] == {
        "implementation_stage": "post-route",
        "wns_ns": 1.361,
        "whs_ns": 0.002,
        "wpws_ns": 2.225,
        "bonded_iob": 164,
        "status_o_bits": 96,
    }
    assert report["timing_resource_recovery_comparison"]["recovered"]["status_o_bits"] == 64
    assert report["timing_resource_recovery_comparison"]["status_growth_avoided"] is True
    assert report["compact_io_observed_trace"]["exposes_child_vectors"] is False
    assert report["compact_io_observed_trace"]["exposes_wide_status"] is False
    assert "Prior Top FSM decoder-block routed timing had WHS 0.018 ns and status was 64 bits" in report[
        "timing_margin_note"
    ]
    assert "WHS 0.002 ns" in report["timing_margin_note"]
    assert "status was 96 bits" in report["timing_margin_note"]
    assert "real LLaMA token prefill/decode semantics" in report["omitted_operations"]
    assert "target multi-layer LLaMA iteration" in report["omitted_operations"]
    assert "DDR/AXI packed-weight streaming" in report["omitted_operations"]
    assert "board-level signoff" in report["omitted_operations"]
    assert "full_LLaMA_model_execution" in report["does_not_claim"]
    assert "board_level_signoff" in report["does_not_claim"]
    assert report["verilator"]["passed"] is True
    assert report["contract_gate"]["verilator_enforced"] is True
