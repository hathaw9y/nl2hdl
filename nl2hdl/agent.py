from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np

from .config import AgentConfig
from .export import deterministic_input, emit_onnx_mlir, export_model_to_onnx
from .graph import save_graph_summary
from .llm_agent import run_llm_agent
from .mlir import analyze_mlir, save_mlir_analysis
from .parser import UnsupportedModelError, parse_onnx_graph
from .planner import build_design_decision_report, save_design_decision_report
from .pruning import apply_pruning, save_pruning_report
from .quant import quantize_graph, save_quant_report
from .verilog import emit_systemverilog
from .verify import estimate_resources, run_iverilog, run_verilator_lint, run_vivado_synth


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_agent(
    model_name: str,
    config: AgentConfig,
    out_dir: Path,
    planner: str = "heuristic",
    planner_model: str = "gpt-4.1-mini",
    mode: str = "full",
    kernel: str | None = None,
    partition: str = "gemm_non_gemm",
    skip_synth: bool = False,
    keep_intermediates: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": model_name,
        "status": "started",
        "agent_goal": "inspect_model_and_design_direct_systemverilog_accelerator",
        "planner": {"mode": planner, "model": planner_model},
        "config": {
            "model": asdict(config.model),
            "hardware": asdict(config.hardware),
            "optimization": asdict(config.optimization),
            "design": asdict(config.design),
            "verification": asdict(config.verification),
        },
        "steps": [],
    }

    try:
        if (
            config.optimization.quantization != "int8_static"
            or config.optimization.pruning not in {"none", "magnitude_unstructured"}
            or config.design.style != "layer_fsm"
        ):
            llm_report = run_llm_agent(
                model_name=model_name,
                config=config,
                out_dir=out_dir,
                mode=mode,
                kernel=kernel,
                partition=partition,
                skip_synth=skip_synth,
                verbose=False,
            )
            report.update(llm_report)
            if llm_report["status"] == "needs_clarification":
                return report
            if llm_report["status"] != "passed":
                raise UnsupportedModelError(
                    llm_report.get("error", "LLM flow did not complete"),
                    unsupported_ops=[config.optimization.quantization, config.design.style],
                )
            return report
        onnx_path = export_model_to_onnx(model_name, config, out_dir)
        report["steps"].append({"name": "export_onnx", "status": "passed", "path": str(onnx_path)})
        mlir_path = emit_onnx_mlir(onnx_path, out_dir)
        report["steps"].append({"name": "emit_onnx_mlir", "status": "passed", "path": str(mlir_path)})

        mlir_analysis = analyze_mlir(mlir_path)
        save_mlir_analysis(mlir_analysis, out_dir / "mlir_analysis.json")
        report["mlir_analysis"] = mlir_analysis.to_dict()
        if mlir_analysis.entry is None or not mlir_analysis.ops:
            raise UnsupportedModelError("MLIR analysis did not find an entry function and supported operation graph")
        report["steps"].append(
            {
                "name": "analyze_mlir",
                "status": "passed" if not mlir_analysis.unsupported_ops else "unsupported",
                "ops": [op.op_type for op in mlir_analysis.ops],
                "unsupported_ops": list(mlir_analysis.unsupported_ops),
            }
        )
        if mlir_analysis.unsupported_ops:
            raise UnsupportedModelError(
                "unsupported ops in MLIR graph: " + ", ".join(mlir_analysis.unsupported_ops),
                unsupported_ops=list(mlir_analysis.unsupported_ops),
            )

        graph = parse_onnx_graph(onnx_path, model_name)
        save_graph_summary(graph, out_dir / "graph_summary.json")
        report["steps"].append({"name": "analyze_graph", "status": "passed", "layers": len(graph.layers)})

        graph, pruning_report = apply_pruning(graph, config.optimization)
        save_pruning_report(pruning_report, out_dir / "pruning_report.json")
        report["pruning"] = pruning_report
        report["steps"].append({"name": "apply_pruning", "status": "passed", "method": pruning_report["method"]})

        sample_input = np.load(out_dir / "dummy_input.npy")
        qmodel = quantize_graph(graph, sample_input)
        save_quant_report(qmodel, out_dir / "quantization_report.json")
        report["steps"].append({"name": "quantize_int8", "status": "passed"})

        effective_config = config
        attempts: list[dict[str, Any]] = []
        max_attempts = max(1, config.retry_count + 1)
        final_resources: dict[str, int] | None = None
        for attempt_idx in range(max_attempts):
            pe_count = effective_config.design.pe_count
            resources = estimate_resources(qmodel, pe_count)
            final_resources = resources
            if resources["estimated_dsp"] > effective_config.hardware.max_dsp:
                raise RuntimeError(
                    f"estimated DSP use {resources['estimated_dsp']} exceeds budget {effective_config.hardware.max_dsp}"
                )
            if resources["estimated_bram"] > effective_config.hardware.max_bram:
                raise RuntimeError(
                    f"estimated BRAM use {resources['estimated_bram']} exceeds budget {effective_config.hardware.max_bram}"
                )

            generated = emit_systemverilog(qmodel, effective_config, out_dir)
            lint = run_verilator_lint(out_dir, qmodel)
            (out_dir / f"verilator_lint_attempt_{attempt_idx}.log").write_text(lint["output"], encoding="utf-8")
            (out_dir / "verilator_lint.log").write_text(lint["output"], encoding="utf-8")
            if effective_config.verification.enable_verilator and not lint["passed"]:
                attempts.append({"attempt": attempt_idx, "pe_count": pe_count, "lint": "failed"})
                raise RuntimeError("Verilator lint failed; see verilator_lint.log")

            sim = run_iverilog(out_dir, qmodel)
            (out_dir / f"simulation_attempt_{attempt_idx}.log").write_text(sim["output"], encoding="utf-8")
            (out_dir / "simulation.log").write_text(sim["output"], encoding="utf-8")
            if effective_config.verification.enable_verilator and not sim["passed"]:
                attempts.append({"attempt": attempt_idx, "pe_count": pe_count, "lint": "passed", "simulation": "failed"})
                raise RuntimeError("integer RTL simulation failed; see simulation.log")

            synth_status: dict[str, Any] = {"passed": True, "timing": {}}
            if effective_config.verification.enable_vivado_synth and not skip_synth:
                synth_status = run_vivado_synth(out_dir, effective_config.verification.vivado_timeout_sec)
                (out_dir / f"vivado_synth_attempt_{attempt_idx}.log").write_text(
                    synth_status["output"], encoding="utf-8"
                )
                (out_dir / "vivado_synth.log").write_text(synth_status["output"], encoding="utf-8")
            attempt_report = {
                "attempt": attempt_idx,
                "pe_count": pe_count,
                "generated_files": [path.name for path in generated],
                "resource_estimate": resources,
                "lint": "passed",
                "simulation": "passed",
                "vivado_synth": "skipped"
                if not effective_config.verification.enable_vivado_synth or skip_synth
                else ("passed" if synth_status["passed"] else "failed"),
                "timing": synth_status.get("timing", {}),
            }
            attempts.append(attempt_report)
            if synth_status["passed"]:
                break

            next_pe_count = max(1, pe_count // 2)
            if attempt_idx + 1 >= max_attempts or next_pe_count == pe_count:
                report["vivado_timing"] = synth_status.get("timing", {})
                raise RuntimeError("Vivado synthesis failed; see vivado_synth.log")
            effective_config = replace(
                effective_config,
                design=replace(effective_config.design, pe_count=next_pe_count),
            )

        report["synthesis_attempts"] = attempts
        report["effective_design"] = asdict(effective_config.design)
        report["resource_estimate"] = final_resources or {}
        if attempts:
            report["vivado_timing"] = attempts[-1].get("timing", {})
        report["steps"].append(
            {
                "name": "emit_systemverilog_and_verify",
                "status": "passed",
                "attempts": attempts,
            }
        )

        design_report = build_design_decision_report(
            model_name,
            effective_config,
            mlir_analysis,
            graph,
            qmodel,
            planner=planner,
            planner_model=planner_model,
        )
        design_report["requested_design"] = asdict(config.design)
        design_report["effective_design"] = asdict(effective_config.design)
        save_design_decision_report(design_report, out_dir / "design_decision_report.json")
        report["design_decision"] = design_report
        report["steps"].append({"name": "plan_accelerator_design", "status": "passed"})

        report["status"] = "passed"
    except UnsupportedModelError as exc:
        report["status"] = "unsupported_model"
        report["error"] = str(exc)
        report["unsupported_ops"] = exc.unsupported_ops
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    finally:
        if not keep_intermediates:
            cache_dir = out_dir / "__pycache__"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        _write_report(out_dir, report)
        if verbose:
            print(json.dumps(report, indent=2))
    return report
