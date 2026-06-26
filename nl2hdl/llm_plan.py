from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

from .config import AgentConfig
from .input_clarification import build_input_clarification_report


def build_llm_accelerator_plan(model_name: str, config: AgentConfig) -> dict[str, Any]:
    input_clarification = build_input_clarification_report(config)
    return {
        "goal": "coding_agent_framework_for_model_to_verilog_llm_accelerator",
        "model": {
            "name": model_name,
            "gptq_checkpoint_source": config.model.gptq_checkpoint or model_name,
            "gptq_checkpoint_source_kind": "configured_override" if config.model.gptq_checkpoint else "same_as_model_name",
            "mlir_graph_source": config.model.mlir_graph,
            "mlir_graph_source_kind": "configured_override" if config.model.mlir_graph else "synthetic_or_export_required",
            "model_structure_source": config.model.model_structure_source,
            "target_family": "LLaMA decoder-only transformer",
            "target_instance": "LLaMA-3.2-1B when model_name selects that checkpoint",
            "required_extraction": [
                "token embedding",
                "decoder block count",
                "hidden size",
                "attention head and kv-head count",
                "rope parameters",
                "MLP gate/up/down projections",
                "RMSNorm weights",
                "LM head",
                "GPTQ group size, zero-points, scales, and packing order",
            ],
        },
        "hardware": asdict(config.hardware),
        "optimization": asdict(config.optimization),
        "design": asdict(config.design),
        "input_clarification": input_clarification,
        "zcu104_assumptions": {
            "board": "AMD ZCU104",
            "device": config.hardware.fpga_part,
            "primary_constraint": "1.23B INT4 weights are roughly 615 MB before GPTQ metadata, so weights cannot reside wholly on-chip",
            "memory_strategy": "store GPTQ packed weights in external DDR/host storage and stream tiles into PL compute buffers",
        },
        "agent_pipeline": [
            "Load model config and quantization metadata from Hugging Face/local checkpoint",
            "Export or lower supported model graph fragments to ONNX/MLIR",
            "Parse MLIR into a semantic LLM graph: embedding, decoder blocks, attention, MLP, norms, lm_head",
            "Partition MLIR operations into GEMM and non-GEMM groups before hardware planning",
            "Build a hardware design plan from the semantic graph and ZCU104 resource constraints",
            "Generate module-level SystemVerilog for selected kernels first: INT4 GEMV/GEMM, RMSNorm, RoPE, attention score/value, SwiGLU",
            "Generate top-level token loop controller and AXI interfaces",
            "Run golden-model tests against PyTorch for prefill=small and decode=single-token paths",
            "Run Verilator/XSIM simulation and Vivado synthesis; retry design knobs such as tile sizes, PE lanes, and buffering",
        ],
        "first_rtl_milestones": [
            "MLIR GEMM/non-GEMM partition report for the selected LLaMA checkpoint",
            "INT4 GPTQ unpack/dequant tile reader",
            "INT4xINT8 or INT4xINT16 projection GEMV kernel",
            "RMSNorm kernel",
            "single decoder block skeleton with AXI-stream-like tile handshakes",
            "single-token decode loop for one block, then all blocks",
        ],
        "non_goals_for_current_dense_backend": [
            "full LLaMA generation with the existing dense_layer backend",
            "keeping all 1B parameters in BRAM/URAM",
            "claiming throughput before memory bandwidth and KV-cache models are measured",
        ],
        "acceptance_gates": [
            "MLIR/semantic graph report names every LLaMA submodule and tensor shape",
            "GEMM ops are mapped to tiled INT4 projection kernels and non-GEMM ops are mapped to dedicated RMSNorm/RoPE/softmax/control kernels",
            "GPTQ metadata parser round-trips packed INT4 weights for at least one projection",
            "kernel-level RTL matches PyTorch/NumPy references for deterministic vectors",
            "Vivado synthesis reports timing and resource usage on the ZCU104 part",
            "agent report records attempted design knobs and final effective design",
        ],
    }


def write_llm_accelerator_plan(model_name: str, config: AgentConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_llm_accelerator_plan(model_name, config)
    (out_dir / "llm_accelerator_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out_dir / "input_clarification_questions.json").write_text(
        json.dumps(plan["input_clarification"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "llm_accelerator_plan.md").write_text(_markdown(plan), encoding="utf-8")
    return plan


def _markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# LLM Accelerator Coding-Agent Plan",
        "",
        f"Model: `{plan['model']['name']}`",
        f"GPTQ metadata source: `{plan['model']['gptq_checkpoint_source']}`",
        f"GPTQ metadata source kind: `{plan['model']['gptq_checkpoint_source_kind']}`",
        f"MLIR graph source: `{plan['model']['mlir_graph_source'] or 'not_provided'}`",
        f"MLIR graph source kind: `{plan['model']['mlir_graph_source_kind']}`",
        f"Model structure source: `{plan['model']['model_structure_source']}`",
        f"Target board/device: `{plan['zcu104_assumptions']['board']}` / `{plan['hardware']['fpga_part']}`",
        (
            "Device resources: "
            f"LUT `{plan['hardware'].get('device_lut')}`, "
            f"FF `{plan['hardware'].get('device_ff')}`, "
            f"DSP `{plan['hardware'].get('device_dsp')}`, "
            f"BRAM36 `{plan['hardware'].get('device_bram_36k')}`, "
            f"URAM `{plan['hardware'].get('device_uram')}`, "
            f"I/O `{plan['hardware'].get('device_io')}`"
        ),
        (
            "Active budgets: "
            f"LUT `{plan['hardware'].get('max_lut')}`, "
            f"FF `{plan['hardware'].get('max_ff')}`, "
            f"DSP `{plan['hardware'].get('max_dsp')}`, "
            f"BRAM `{plan['hardware'].get('max_bram')}`, "
            f"URAM `{plan['hardware'].get('max_uram')}`, "
            f"I/O `{plan['hardware'].get('max_io')}`"
        ),
        f"Quantization: `{plan['optimization']['quantization']}`",
        f"Optimization brief: `{plan['optimization'].get('optimization_brief') or 'not_provided'}`",
        f"Design style alias: `{plan['design']['style']}`",
        f"Compute style: `{plan['design']['compute_style']}`",
        f"Execution style: `{plan['design']['execution_style']}`",
        f"Memory style: `{plan['design']['memory_style']}`",
        f"Control style: `{plan['design']['control_style']}`",
        f"Architecture brief: `{plan['design'].get('architecture_brief') or 'not_provided'}`",
        "",
        "## Key Constraint",
        "",
        plan["zcu104_assumptions"]["primary_constraint"],
        "",
        "## Input Clarification",
        "",
        f"Status: `{plan['input_clarification']['status']}`",
        f"Requires user response: `{plan['input_clarification']['requires_user_response']}`",
        f"Question count: `{plan['input_clarification']['question_count']}`",
        "",
        "Questions:",
    ]
    clarification_questions = plan["input_clarification"].get("questions") or []
    if clarification_questions:
        lines.extend(
            [
                f"- `{question['id']}` ({question['scope']}): {question['question']}"
                for question in clarification_questions
            ]
        )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Candidate Directions",
        "",
        "Optimization candidates:",
    ])
    optimization_candidates = plan["optimization"].get("optimization_candidates") or []
    if optimization_candidates:
        lines.extend([f"- `{candidate}`" for candidate in optimization_candidates])
    else:
        lines.append("- not_provided")
    lines.extend(["", "Design candidates:"])
    design_candidates = plan["design"].get("design_candidates") or []
    if design_candidates:
        lines.extend([f"- `{candidate}`" for candidate in design_candidates])
    else:
        lines.append("- not_provided")
    lines.extend([
        "",
        "## Agent Pipeline",
        "",
    ])
    lines.extend([f"- {item}" for item in plan["agent_pipeline"]])
    lines.extend(["", "## First RTL Milestones", ""])
    lines.extend([f"- {item}" for item in plan["first_rtl_milestones"]])
    lines.extend(["", "## Acceptance Gates", ""])
    lines.extend([f"- {item}" for item in plan["acceptance_gates"]])
    lines.extend(
        [
            "",
            "## Source Notes",
            "",
            "- AMD ZCU104 product page: https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu104.html",
            "- AMD UG1267 ZCU104 board guide: https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd",
            "- Meta Llama 3.2 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md",
        ]
    )
    return "\n".join(lines) + "\n"
