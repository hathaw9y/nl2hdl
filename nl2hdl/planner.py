from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

from .config import AgentConfig
from .graph import ModelGraph
from .mlir import MlirAnalysis
from .verify import estimate_resources
from .quant import QuantizedModel


def build_design_decision_report(
    model_name: str,
    config: AgentConfig,
    mlir_analysis: MlirAnalysis,
    graph: ModelGraph,
    qmodel: QuantizedModel,
    planner: str = "heuristic",
    planner_model: str = "gpt-4.1-mini",
) -> dict[str, Any]:
    resources = estimate_resources(qmodel, config.design.pe_count)
    report = {
        "agent_role": "model_inspection_to_direct_systemverilog_accelerator_designer",
        "planner": {
            "requested": planner,
            "model": planner_model,
            "actual": "heuristic",
            "reason": "deterministic local planner selected",
        },
        "model": model_name,
        "model_observations": {
            "mlir_entry": mlir_analysis.entry,
            "mlir_ops": [op.op_type for op in mlir_analysis.ops],
            "input_size": graph.input_size,
            "output_size": graph.output_size,
            "dense_layers": len(graph.layers),
        },
        "accepted_capability": "fixed_shape_dense_dnn",
        "rejected_capabilities": [
            "dynamic_batch",
            "dynamic_sequence_length",
            "attention",
            "convolution",
            "runtime_sparse_pruning",
        ],
        "optimization_decision": {
            "quantization": config.optimization.quantization,
            "pruning": config.optimization.pruning,
            "pruning_threshold": config.optimization.pruning_threshold,
            "integer_contract": "int8 activations and weights, int32 accumulators, per-layer fixed-point requantization",
        },
        "hardware_decision": {
            "style": config.design.style,
            "pe_count": config.design.pe_count,
            "activation_buffer": config.design.activation_buffer,
            "weight_storage": config.design.weight_storage,
            "top_level_protocol": "start/done with packed int8 input/output vectors",
            "clock_mhz": config.hardware.target_clock_mhz,
            "fpga_part": config.hardware.fpga_part,
        },
        "resource_estimate": resources,
        "emitted_rtl": {
            "top": "model_top.sv",
            "layers": [f"dense_layer_{idx}.sv" for idx, _ in enumerate(qmodel.layers)],
            "testbench": "tb_model_top.sv",
            "vivado_tcl": "vivado_synth.tcl",
        },
    }
    llm_plan = _try_llm_planner(report, planner, planner_model)
    if llm_plan is not None:
        report["planner"] = {
            "requested": planner,
            "model": planner_model,
            "actual": "llm",
            "reason": "LLM planner returned a design review",
        }
        report["llm_design_review"] = llm_plan
    elif planner == "llm":
        report["planner"] = {
            "requested": planner,
            "model": planner_model,
            "actual": "heuristic",
            "reason": "LLM planner unavailable; install openai and set OPENAI_API_KEY",
        }
    elif planner == "auto":
        report["planner"] = {
            "requested": planner,
            "model": planner_model,
            "actual": "heuristic",
            "reason": "LLM planner unavailable in auto mode; used deterministic local planner",
        }
    return report


def _try_llm_planner(base_report: dict[str, Any], planner: str, planner_model: str) -> dict[str, Any] | None:
    if planner not in {"llm", "auto"}:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI()
    prompt = {
        "task": "Review this MLIR-derived model analysis and propose a direct SystemVerilog accelerator plan.",
        "constraints": {
            "rtl_backend": "direct SystemVerilog",
            "supported_v1_designs": ["layer_fsm"],
            "must_preserve": ["fixed shapes", "int8 quantization", "start_done_top_protocol"],
            "output_format": "concise JSON design review with risks and accepted architecture",
        },
        "base_report": base_report,
    }
    try:
        response = client.responses.create(
            model=planner_model,
            input=[
                {
                    "role": "system",
                    "content": "You are a hardware coding agent designing small neural-network accelerators from MLIR analysis.",
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
    except Exception:
        return None
    text = getattr(response, "output_text", "")
    if not text:
        return None
    return {"raw_text": text}


def save_design_decision_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
