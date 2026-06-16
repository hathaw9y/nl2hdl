from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import AgentConfig
from .llm_kernels import emit_inspect_artifacts, emit_kernel
from .llm_plan import write_llm_accelerator_plan


def _finalize_report(out_dir: Path, report: dict[str, Any], verbose: bool) -> dict[str, Any]:
    (out_dir / "llm_agent_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def _mark_wave_status_context_only(path: Path, mode: str) -> None:
    if not path.exists():
        return
    status = json.loads(path.read_text(encoding="utf-8"))
    status["artifact"] = "hdl_subagent_wave_status_context_only"
    status["coverage_level"] = "kernel_or_block_context_snapshot_not_parent_collection_gate"
    status["context_only"] = True
    status["context_reason"] = (
        f"`{mode}` mode emits inspect artifacts beside kernel evidence for traceability; "
        "use the parent collection root's hdl_subagent_wave_status.json for wave advancement."
    )
    status["does_not_claim"] = list(
        dict.fromkeys(
            status.get("does_not_claim", [])
            + [
                "current parent collection wave status",
                "permission to advance dependent waves",
            ]
        )
    )
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_llm_agent(
    model_name: str,
    config: AgentConfig,
    out_dir: Path,
    mode: str,
    kernel: str | None,
    partition: str,
    skip_synth: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": model_name,
        "status": "started",
        "mode": mode,
        "kernel": kernel,
        "partition": partition,
        "target": {
            "board": "ZCU104",
            "part": config.hardware.fpga_part,
            "device_lut": config.hardware.device_lut,
            "device_ff": config.hardware.device_ff,
            "device_dsp": config.hardware.device_dsp,
            "device_bram_36k": config.hardware.device_bram_36k,
            "device_uram": config.hardware.device_uram,
            "device_io": config.hardware.device_io,
            "quantization": config.optimization.quantization,
            "optimization_brief": config.optimization.optimization_brief,
            "optimization_candidates": config.optimization.optimization_candidates,
            "optimization_extra_options": config.optimization.extra_options,
            "design_style": config.design.style,
            "compute_style": config.design.compute_style,
            "execution_style": config.design.execution_style,
            "memory_style": config.design.memory_style,
            "control_style": config.design.control_style,
            "architecture_brief": config.design.architecture_brief,
            "design_candidates": config.design.design_candidates,
            "design_extra_options": config.design.extra_options,
        },
        "steps": [],
    }

    try:
        plan = write_llm_accelerator_plan(model_name, config, out_dir)
        report["steps"].append({"name": "write_llm_plan", "status": "passed"})
        input_clarification = plan["input_clarification"]
        report["input_clarification"] = input_clarification
        if input_clarification.get("requires_user_response"):
            report["status"] = "needs_clarification"
            report["error"] = "Parent agent needs clarification before HDL sub-agent dispatch"
            report["steps"].append(
                {
                    "name": "clarify_free_form_input",
                    "status": "needs_clarification",
                    "question_count": input_clarification.get("question_count", 0),
                    "questions_file": "input_clarification_questions.json",
                }
            )
            report["plan_summary"] = {
                "first_rtl_milestones": plan["first_rtl_milestones"],
                "acceptance_gates": plan["acceptance_gates"],
            }
            return _finalize_report(out_dir, report, verbose)

        if partition != "gemm_non_gemm":
            raise ValueError("LLM mode currently supports only --partition gemm_non_gemm")

        (
            mlir_path,
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
            out_dir,
            model_name,
            config,
        )
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        payload_probe_path = out_dir / "gptq_payload_probe.json"
        payload_probe = json.loads(payload_probe_path.read_text(encoding="utf-8")) if payload_probe_path.exists() else {}
        mlir_readiness_path = out_dir / "mlir_model_analysis_readiness.json"
        mlir_readiness = (
            json.loads(mlir_readiness_path.read_text(encoding="utf-8"))
            if mlir_readiness_path.exists()
            else {}
        )
        target_readiness_path = out_dir / "target_readiness_report.json"
        target_readiness = (
            json.loads(target_readiness_path.read_text(encoding="utf-8"))
            if target_readiness_path.exists()
            else {}
        )
        remediation_path = out_dir / "target_blocker_remediation_plan.json"
        remediation = json.loads(remediation_path.read_text(encoding="utf-8")) if remediation_path.exists() else {}
        if mode != "inspect":
            _mark_wave_status_context_only(subagent_wave_status_path, mode)
        report["target_gate_summary"] = {
            "mlir_model_analysis": {
                "artifact": "mlir_model_analysis_readiness.json",
                "status": mlir_readiness.get("status", "not_emitted"),
                "analysis_source": mlir_readiness.get("analysis_source"),
                "source_kind": mlir_readiness.get("source_kind"),
                "target_claim_allowed_by_source": mlir_readiness.get("target_claim_allowed_by_source"),
                "coverage_level": mlir_readiness.get("coverage_level"),
                "missing_semantic_gemm_ops": mlir_readiness.get("missing_semantic_gemm_ops"),
                "missing_semantic_non_gemm_ops": mlir_readiness.get("missing_semantic_non_gemm_ops"),
            },
            "target_readiness": {
                "artifact": "target_readiness_report.json",
                "status": target_readiness.get("status", "not_emitted"),
                "safe_to_spawn_bounded_subagents": target_readiness.get("safe_to_spawn_bounded_subagents"),
                "safe_to_claim_target_accelerator": target_readiness.get("safe_to_claim_target_accelerator"),
                "blocked_target_task_count": target_readiness.get("blocked_target_task_count"),
            },
            "target_blocker_remediation": {
                "artifact": "target_blocker_remediation_plan.json",
                "status": remediation.get("status", "not_emitted"),
                "blocked_target_task_count": remediation.get("blocked_target_task_count"),
                "target_preflight_blockers": remediation.get("target_preflight_blockers"),
                "canonical_full_preflight_command": remediation.get("canonical_full_preflight_command"),
            },
            "gptq_checkpoint_metadata": task_manifest["gptq_checkpoint_metadata"],
            "gptq_checkpoint_source_preflight": {
                "artifact": "gptq_checkpoint_source_preflight.json",
                "status": task_manifest["gptq_checkpoint_metadata"].get("source_preflight_status"),
                "checkpoint_source_dependency": task_manifest["gptq_checkpoint_metadata"].get(
                    "checkpoint_source_dependency"
                ),
                "next_action": task_manifest["gptq_checkpoint_metadata"].get("source_preflight_next_action"),
            },
            "gptq_payload_probe": {
                "artifact": "gptq_payload_probe.json",
                "status": payload_probe.get("status", "not_emitted"),
                "projection": payload_probe.get("projection"),
                "payload_probe_source": payload_probe.get("payload_probe_source"),
                "payload_golden_source": payload_probe.get("payload_golden_source"),
                "sampled_tensor_count": payload_probe.get("sampled_tensor_count"),
                "required_tensor_count": payload_probe.get("required_tensor_count"),
                "sample_bytes_requested": payload_probe.get("sample_bytes_requested"),
                "qweight_payload_order": payload_probe.get("qweight_payload_order"),
                "qweight_payload_word_count": payload_probe.get("qweight_payload_word_count"),
                "qweight_stream_probe": payload_probe.get("qweight_stream_probe"),
                "target_checkpoint_payload_dependency": payload_probe.get("target_checkpoint_payload_dependency"),
                "payload_words_match_gptq_probe": payload_probe.get("payload_words_match_gptq_probe"),
                "qweight_payload_words32_le_count": len(payload_probe.get("qweight_payload_words32_le_hex", []))
                if isinstance(payload_probe.get("qweight_payload_words32_le_hex"), list)
                else 0,
                "projection_payload_probe_count": payload_probe.get("projection_payload_probe_count"),
                "sampled_projection_payload_probe_count": payload_probe.get(
                    "sampled_projection_payload_probe_count"
                ),
                "required_projection_payload_probe_count": payload_probe.get(
                    "required_projection_payload_probe_count"
                ),
                "all_projection_payload_dependency": payload_probe.get("all_projection_payload_dependency"),
            },
            "gptq_weight_layout_preflight": {
                "artifact": "gptq_weight_layout_preflight.json",
                "status": task_manifest["gptq_weight_layout_preflight"]["status"],
                "target_compatible_projection_count": task_manifest["gptq_weight_layout_preflight"][
                    "target_compatible_projection_count"
                ],
                "required_projection_count": task_manifest["gptq_weight_layout_preflight"][
                    "required_projection_count"
                ],
                "blocking_reason": task_manifest["gptq_weight_layout_preflight"]["blocking_reason"],
            },
            "projection_weight_stream_plan": {
                "artifact": "projection_weight_stream_plan.json",
                "projection_count": task_manifest["projection_weight_stream_plan"]["projection_count"],
                "target_stream_plan_valid_count": task_manifest["projection_weight_stream_plan"][
                    "target_stream_plan_valid_count"
                ],
                "payload_satisfied_projection_count": task_manifest["projection_weight_stream_plan"][
                    "payload_satisfied_projection_count"
                ],
                "all_projection_layout_dependency": task_manifest["projection_weight_stream_plan"][
                    "all_projection_layout_dependency"
                ],
                "all_projection_payload_dependency": task_manifest["projection_weight_stream_plan"][
                    "all_projection_payload_dependency"
                ],
                "target_scale_ready_for_all_projection_streaming": task_manifest["projection_weight_stream_plan"][
                    "target_scale_ready_for_all_projection_streaming"
                ],
            },
            "blocked_target_task_count": task_manifest["task_counts"]["blocked_target_tasks"],
            "blocked_target_tasks": [
                {
                    "task_id": task["task_id"],
                    "reason": task["reason"],
                    **({"metadata_status": task["metadata_status"]} if "metadata_status" in task else {}),
                    **({"classification": task["classification"]} if "classification" in task else {}),
                    **(
                        {"checkpoint_quantization_dependency": task["checkpoint_quantization_dependency"]}
                        if "checkpoint_quantization_dependency" in task
                        else {}
                    ),
                }
                for task in task_manifest["blocked_target_tasks"]
            ],
        }
        report["steps"].append(
            {
                "name": "inspect_semantic_mlir",
                "status": "passed",
                "mlir": mlir_path.name,
                "analysis": analysis_path.name,
                "mlir_model_analysis_readiness": "mlir_model_analysis_readiness.json",
                "semantic_graph": semantic_graph_path.name,
                "gptq_checkpoint_source_preflight": "gptq_checkpoint_source_preflight.json",
                "gptq_checkpoint_metadata": gptq_metadata_path.name,
                "gptq_payload_probe": "gptq_payload_probe.json",
                "projection_weight_stream_plan": "projection_weight_stream_plan.json",
                "target_readiness_report": "target_readiness_report.json",
                "target_blocker_remediation_plan": "target_blocker_remediation_plan.json",
                "target_blocker_remediation_markdown": "target_blocker_remediation_plan.md",
                "hdl_task_manifest": task_manifest_path.name,
                "hdl_subagent_tasks": subagent_tasks_path.name,
                "skill_update_candidate_template": "skill_update_candidate_template.json",
                "subagent_prompts": subagent_prompt_dir.name,
                "verification_prompts": "verification_prompts",
                "hdl_subagent_dispatch_plan": subagent_dispatch_plan_path.name,
                "hdl_subagent_wave_status": subagent_wave_status_path.name,
                "hdl_subagent_execution_manifest": subagent_execution_manifest_path.name,
                "codex_spawn_instructions": "codex_spawn_instructions.md",
            }
        )

        if mode == "inspect":
            report["status"] = "passed"
        elif mode == "kernel":
            selected = kernel or "int4_unpack"
            result = emit_kernel(selected, config, out_dir, skip_synth=skip_synth)
            report["steps"].append({"name": f"emit_kernel_{selected}", **result.to_dict()})
            report["status"] = result.status
        elif mode == "block":
            selected = kernel or "decoder_block"
            if selected != "decoder_block":
                raise ValueError("block mode currently supports only --kernel decoder_block")
            result = emit_kernel("decoder_block", config, out_dir, skip_synth=skip_synth)
            report["steps"].append({"name": "emit_decoder_block", **result.to_dict()})
            report["status"] = result.status
        elif mode == "full":
            report["status"] = "unsupported_model"
            report["error"] = (
                "full LLaMA-3.2-1B generation is gated until kernel and decoder-block "
                "milestones pass on ZCU104; use --mode inspect, --mode kernel, or --mode block"
            )
        else:
            raise ValueError(f"unsupported mode: {mode}")

        report["plan_summary"] = {
            "first_rtl_milestones": plan["first_rtl_milestones"],
            "acceptance_gates": plan["acceptance_gates"],
        }
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)

    return _finalize_report(out_dir, report, verbose)
