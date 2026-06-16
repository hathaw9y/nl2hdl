import numpy as np
import hashlib
import json
import struct
import sys
import types

from nl2hdl.gptq import (
    GPTQ_CHECKPOINT_ALLOW_PATTERNS,
    build_gptq_checkpoint_source_preflight_report,
    build_gptq_payload_probe_report,
    dequantize_int4,
    inspect_gptq_checkpoint_metadata,
    pack_int4,
    synthetic_projection,
    unpack_int4,
)


def _write_fake_safetensors_header(path, keys):
    header = {"__metadata__": {"format": "pt"}}
    for idx, key in enumerate(keys):
        header[key] = {
            "dtype": "I32",
            "shape": [1],
            "data_offsets": [idx * 4, (idx + 1) * 4],
        }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * (4 * len(keys))))


def _write_safetensors_header(path, entries):
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
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * offset))


def _write_safetensors_payload(path, entries):
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    payload = bytearray()
    for key, dtype, shape, raw_payload in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw_payload)],
        }
        offset += len(raw_payload)
        payload.extend(raw_payload)
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))


def test_pack_unpack_signed_int4_round_trip():
    values = np.array([-8, -1, 0, 7, 3, -4, 5, -6], dtype=np.int8)
    packed = pack_int4(values, signed=True)
    unpacked = unpack_int4(packed, values.size, signed=True)
    np.testing.assert_array_equal(unpacked, values)


def test_dequantize_groupwise_int4():
    values = np.array([1, 2, 3, 4], dtype=np.int8)
    scales = np.array([0.5, 2.0], dtype=np.float32)
    zeros = np.array([0, 1], dtype=np.int32)
    out = dequantize_int4(values, scales, zeros, group_size=2)
    np.testing.assert_allclose(out, np.array([0.5, 1.0, 4.0, 6.0], dtype=np.float32))


def test_synthetic_projection_metadata_shape():
    proj = synthetic_projection()
    unpacked = unpack_int4(proj.qweight, proj.rows * proj.cols, signed=proj.signed)
    assert unpacked.size == proj.rows * proj.cols
    assert proj.group_size == 4


def test_inspect_gptq_checkpoint_metadata_from_local_dir(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        """
{
  "weight_map": {
    "model.layers.0.self_attn.q_proj.qweight": "model-00001.safetensors",
    "model.layers.0.self_attn.q_proj.qzeros": "model-00001.safetensors",
    "model.layers.0.self_attn.q_proj.scales": "model-00001.safetensors",
    "model.layers.0.self_attn.k_proj.qweight": "model-00001.safetensors"
  }
}
""",
        encoding="utf-8",
    )
    report = inspect_gptq_checkpoint_metadata(str(tmp_path))
    assert report["status"] == "parsed"
    assert report["metadata_resolution"]["source"] == "local_path"
    assert report["bits"] == 4
    assert report["group_size"] == 128
    q_proj = next(item for item in report["projection_metadata"] if item["name"] == "q_proj")
    assert q_proj["has_qweight"] is True
    assert q_proj["has_qzeros"] is True
    assert q_proj["has_scales"] is True
    assert q_proj["tensor_summaries"] == {}
    assert report["tensor_summary_count"] == 0
    assert report["tensor_summary_source"] == "not_available"


def test_gptq_checkpoint_source_preflight_resolves_local_path(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128}',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")

    report = build_gptq_checkpoint_source_preflight_report(str(tmp_path))

    assert report["status"] == "resolved_local_path"
    assert report["checkpoint_source_dependency"] == "satisfied_by_local_path"
    assert report["resolved_model_dir"] == str(tmp_path)
    assert report["network_download_probe"]["attempted"] is False
    assert report["artifact_inventory"]["file_count"] == 2
    assert report["artifact_inventory"]["missing_metadata_json"] is False
    assert report["artifact_inventory"]["has_weight_index"] is True
    assert report["expected_file_patterns"] == GPTQ_CHECKPOINT_ALLOW_PATTERNS


def test_gptq_checkpoint_source_preflight_reports_uncached_hf_without_network(monkeypatch):
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_id, local_files_only, allow_patterns):
        assert repo_id == "org/gated-gptq"
        assert local_files_only is True
        assert allow_patterns == GPTQ_CHECKPOINT_ALLOW_PATTERNS
        raise RuntimeError("repo not found in local cache")

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("NL2HDL_ALLOW_HF_SNAPSHOT_DOWNLOAD", raising=False)

    report = build_gptq_checkpoint_source_preflight_report("org/gated-gptq")

    assert report["status"] == "unresolved"
    assert report["checkpoint_source_dependency"] == "blocked_by_checkpoint_source_preflight"
    assert report["huggingface_hub_probe"]["available"] is True
    assert report["local_cache_probe"]["status"] == "unresolved"
    assert "repo not found in local cache" in report["local_cache_probe"]["error"]
    assert report["network_download_probe"]["attempted"] is False
    assert "network snapshot download is disabled" in report["blocking_reason"]
    assert "model.gptq_checkpoint" in report["next_action"]


def test_gptq_checkpoint_source_preflight_reports_gated_network_failure(monkeypatch):
    fake_hub = types.ModuleType("huggingface_hub")
    calls = []

    def fake_snapshot_download(repo_id, local_files_only, allow_patterns):
        calls.append(local_files_only)
        assert repo_id == "org/gated-gptq"
        assert allow_patterns == GPTQ_CHECKPOINT_ALLOW_PATTERNS
        if local_files_only:
            raise RuntimeError("not cached")
        raise PermissionError("gated repository access denied")

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("NL2HDL_ALLOW_HF_SNAPSHOT_DOWNLOAD", "1")

    report = build_gptq_checkpoint_source_preflight_report("org/gated-gptq")

    assert calls == [True, False]
    assert report["status"] == "unresolved"
    assert report["network_snapshot_download_allowed"] is True
    assert report["network_download_probe"]["attempted"] is True
    assert report["network_download_probe"]["status"] == "unresolved"
    assert "gated repository access denied" in report["network_download_probe"]["error"]
    assert "authenticate" in report["next_action"]


def test_inspect_gptq_checkpoint_metadata_reads_indexed_safetensors_headers(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.qweight": "model-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.qzeros": "model-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.scales": "model-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_safetensors_header(
        tmp_path / "model-00001.safetensors",
        [
            ("model.layers.0.self_attn.q_proj.qweight", "I32", [16, 1], 64),
            ("model.layers.0.self_attn.q_proj.qzeros", "I32", [1], 4),
            ("model.layers.0.self_attn.q_proj.scales", "F16", [1], 2),
        ],
    )

    report = inspect_gptq_checkpoint_metadata(str(tmp_path))

    assert report["status"] == "parsed"
    assert report["tensor_key_source"] == "weight_index"
    assert report["indexed_safetensors_files_scanned"] == ["model-00001.safetensors"]
    assert report["indexed_safetensors_header_errors"] == []
    assert report["tensor_summary_source"] == "safetensors_header_from_weight_index"
    assert report["tensor_summary_count"] == 3
    q_proj = next(item for item in report["projection_metadata"] if item["name"] == "q_proj")
    qweight_summary = q_proj["tensor_summaries"]["model.layers.0.self_attn.q_proj.qweight"]
    assert qweight_summary["file"] == "model-00001.safetensors"
    assert qweight_summary["shape"] == [16, 1]
    assert qweight_summary["byte_count"] == 64
    assert qweight_summary["metadata_status"] == "header_only_no_tensor_payload"


def test_inspect_gptq_checkpoint_metadata_from_single_safetensors_header(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": true, "sym": false}',
        encoding="utf-8",
    )
    _write_fake_safetensors_header(
        tmp_path / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.g_idx",
            "model.layers.0.mlp.gate_proj.qweight",
            "model.layers.0.mlp.gate_proj.scales",
            "model.layers.0.mlp.down_proj.qzeros",
        ],
    )

    report = inspect_gptq_checkpoint_metadata(str(tmp_path))

    assert report["status"] == "parsed"
    assert report["bits"] == 4
    assert report["group_size"] == 128
    assert report["tensor_key_source"] == "safetensors_header"
    assert report["index_files"] == []
    assert report["direct_safetensors_files_scanned"] == ["model.safetensors"]
    assert report["direct_safetensors_header_errors"] == []
    assert report["scales_and_zero_points"]["source"] == "tensor_key_presence"
    assert report["projection_metadata_count"] == 3
    assert report["quantized_projection_metadata_count"] == 2
    assert report["complete_gptq_projection_metadata_count"] == 1
    assert report["tensor_summary_source"] == "safetensors_header"
    assert report["tensor_summary_count"] == 7
    q_proj = next(item for item in report["projection_metadata"] if item["name"] == "q_proj")
    assert q_proj["has_qweight"] is True
    assert q_proj["has_qzeros"] is True
    assert q_proj["has_scales"] is True
    assert q_proj["has_g_idx"] is True
    qweight_summary = q_proj["tensor_summaries"]["model.layers.0.self_attn.q_proj.qweight"]
    assert qweight_summary["file"] == "model.safetensors"
    assert qweight_summary["dtype"] == "I32"
    assert qweight_summary["shape"] == [1]
    assert qweight_summary["data_offsets"] == [0, 4]
    assert qweight_summary["byte_count"] == 4
    assert qweight_summary["metadata_status"] == "header_only_no_tensor_payload"
    gate_proj = next(item for item in report["projection_metadata"] if item["name"] == "gate_proj")
    assert gate_proj["has_qweight"] is True
    assert gate_proj["has_scales"] is True
    down_proj = next(item for item in report["projection_metadata"] if item["name"] == "down_proj")
    assert down_proj["has_qzeros"] is True


def test_inspect_gptq_checkpoint_metadata_accepts_common_tensor_and_config_aliases(tmp_path):
    (tmp_path / "quantization_config.json").write_text(
        '{"quantization_config": {"quant_method": "gptq", "bits": "int4", "groupsize": "128", "desc_act": false}}',
        encoding="utf-8",
    )
    _write_safetensors_payload(
        tmp_path / "model.safetensors",
        [
            ("model.layers.0.self_attn.q_proj.q_weight", "I32", [4, 1], bytes(range(16))),
            ("model.layers.0.self_attn.q_proj.zeros", "I32", [1, 1], bytes([0x11, 0x22, 0x33, 0x44])),
            ("model.layers.0.self_attn.q_proj.scale", "F16", [1, 2], bytes([0x55, 0x66, 0x77, 0x88])),
            ("model.layers.0.self_attn.q_proj.gidx", "I32", [1], bytes([0, 0, 0, 0])),
        ],
    )

    report = inspect_gptq_checkpoint_metadata(str(tmp_path))
    payload = build_gptq_payload_probe_report(str(tmp_path), projection_name="q_proj", sample_bytes=16)

    assert report["status"] == "parsed"
    assert report["bits"] == 4
    assert report["group_size"] == 128
    assert report["quant_method"] == "gptq"
    assert report["complete_gptq_projection_metadata_count"] == 1
    q_proj = next(item for item in report["projection_metadata"] if item["name"] == "q_proj")
    assert q_proj["has_qweight"] is True
    assert q_proj["has_qzeros"] is True
    assert q_proj["has_scales"] is True
    assert q_proj["has_g_idx"] is True
    assert q_proj["keys"]["qweight"] == ["model.layers.0.self_attn.q_proj.q_weight"]
    assert q_proj["keys"]["qzeros"] == ["model.layers.0.self_attn.q_proj.zeros"]
    assert q_proj["keys"]["scales"] == ["model.layers.0.self_attn.q_proj.scale"]
    assert q_proj["keys"]["g_idx"] == ["model.layers.0.self_attn.q_proj.gidx"]
    assert payload["status"] == "sampled"
    assert payload["sampled_tensor_count"] == 3
    assert payload["tensors"]["qzeros"]["key"] == "model.layers.0.self_attn.q_proj.zeros"
    assert payload["tensors"]["scales"]["key"] == "model.layers.0.self_attn.q_proj.scale"


def test_inspect_gptq_checkpoint_metadata_does_not_count_plain_safetensors_as_quantized(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 64}', encoding="utf-8")
    _write_fake_safetensors_header(
        tmp_path / "model.safetensors",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )

    report = inspect_gptq_checkpoint_metadata(str(tmp_path))

    assert report["status"] == "metadata_json_without_quant_fields"
    assert report["tensor_key_source"] == "safetensors_header"
    assert report["projection_metadata_count"] == 3
    assert report["quantized_projection_metadata_count"] == 0
    assert report["complete_gptq_projection_metadata_count"] == 0
    assert report["tensor_summary_count"] == 3
    assert report["checkpoint_quantization_artifact"]["classification"] == "base_or_unquantized_checkpoint"
    assert report["checkpoint_quantization_artifact"]["checkpoint_quantization_dependency"] == (
        "blocked_by_non_gptq_checkpoint_source"
    )
    assert report["checkpoint_quantization_artifact"]["has_plain_projection_weights"] is True
    assert "GPTQ INT4 checkpoint" in report["checkpoint_quantization_artifact"]["next_action"]
    assert all(not item["has_qweight"] for item in report["projection_metadata"])


def test_inspect_gptq_checkpoint_metadata_reports_unavailable_for_missing_path():
    report = inspect_gptq_checkpoint_metadata("missing/local-or-hf-model")
    assert report["status"] == "unavailable"
    assert report["checkpoint_quantization_artifact"]["classification"] == "checkpoint_source_unavailable"
    assert report["checkpoint_quantization_artifact"]["checkpoint_quantization_dependency"] == (
        "blocked_by_checkpoint_source_preflight"
    )
    assert "checkpoint_weight_loading" in report["does_not_claim"]


def test_gptq_payload_probe_unavailable_schema_is_stable_for_missing_path():
    report = build_gptq_payload_probe_report("missing/local-or-hf-model", projection_name="q_proj", sample_bytes=8)

    assert report["status"] == "unavailable"
    assert report["sampled_tensor_count"] == 0
    assert report["required_tensor_count"] == 3
    assert report["qweight_payload_words32_le_hex"] == []
    assert report["qweight_payload_word_count"] == 0
    assert report["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert "model_dir" in report["reason"]


def test_gptq_payload_probe_reads_bounded_safetensors_payload_prefix(tmp_path):
    qweight = bytes(range(16))
    qzeros = bytes([0x11, 0x22, 0x33, 0x44])
    scales = bytes([0x55, 0x66, 0x77, 0x88])
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_safetensors_payload(
        tmp_path / "model.safetensors",
        [
            ("model.layers.0.self_attn.q_proj.qweight", "I32", [4, 1], qweight),
            ("model.layers.0.self_attn.q_proj.qzeros", "I32", [1, 1], qzeros),
            ("model.layers.0.self_attn.q_proj.scales", "F16", [1, 2], scales),
        ],
    )

    report = build_gptq_payload_probe_report(str(tmp_path), projection_name="q_proj", sample_bytes=16)

    assert report["status"] == "sampled"
    assert report["target_checkpoint_payload_dependency"] == "satisfied_by_payload_probe"
    assert report["sampled_tensor_count"] == 3
    assert report["required_tensor_count"] == 3
    assert report["does_not_claim"] == [
        "full_checkpoint_tensor_materialization",
        "numeric_GPTQ_correctness",
        "checkpoint_specific_qweight_order_correctness",
        "full_qweight_payload_streaming",
        "full_LLaMA_execution",
    ]
    qweight_probe = report["tensors"]["qweight"]
    assert qweight_probe["status"] == "sampled"
    assert qweight_probe["sample_byte_count"] == 16
    assert qweight_probe["sample_bytes_hex"] == qweight[:16].hex()
    assert qweight_probe["sample_sha256"] == hashlib.sha256(qweight[:16]).hexdigest()
    assert qweight_probe["words32_le_hex"] == [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
    ]
    assert report["qweight_payload_order"] == "safetensors_payload_prefix_32bit_little_endian_words"
    assert report["qweight_payload_words32_le_hex"] == [
        "0x03020100",
        "0x07060504",
        "0x0b0a0908",
        "0x0f0e0d0c",
    ]
    assert report["qweight_payload_word_count"] == 4
    assert report["qweight_stream_probe"]["covers_first_memory_beat"] is True
    assert report["qweight_stream_probe"]["first_memory_beats_128b_le_hex"] == [
        "0x0f0e0d0c0b0a09080706050403020100"
    ]
    assert report["qweight_stream_probe"]["memory_beat_word_chunks32_le_hex"] == [
        ["0x03020100", "0x07060504", "0x0b0a0908", "0x0f0e0d0c"]
    ]
    assert report["tensors"]["qzeros"]["sample_bytes_hex"] == qzeros.hex()
    assert report["tensors"]["scales"]["sample_bytes_hex"] == scales.hex()


def test_gptq_payload_probe_reports_partial_when_tensor_summary_missing(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 128, "desc_act": false, "sym": true}',
        encoding="utf-8",
    )
    _write_safetensors_payload(
        tmp_path / "model.safetensors",
        [
            ("model.layers.0.self_attn.q_proj.qweight", "I32", [4, 1], bytes(range(16))),
        ],
    )

    report = build_gptq_payload_probe_report(str(tmp_path), projection_name="q_proj", sample_bytes=8)

    assert report["status"] == "partial"
    assert report["target_checkpoint_payload_dependency"] == "blocked_by_gptq_payload_probe"
    assert report["sampled_tensor_count"] == 1
    assert report["required_tensor_count"] == 3
    assert report["qweight_payload_words32_le_hex"] == ["0x03020100", "0x07060504"]
    assert report["qweight_payload_word_count"] == 2
    assert report["qweight_stream_probe"]["covers_first_memory_beat"] is False
    assert report["tensors"]["qweight"]["status"] == "sampled"
    assert report["tensors"]["qzeros"]["status"] == "unavailable"
    assert report["tensors"]["scales"]["status"] == "unavailable"
    assert "could not be sampled" in report["reason"]
