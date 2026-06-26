from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
import json
import os
import shutil

from .config import AgentConfig, validate_config
from .llm_agent import run_llm_agent
from .llm_kernels import (
    emit_kernel,
    _parse_vivado_timing_summary,
    run_zcu104_board_wrapper_axi_bridge_agent,
    write_model_level_execution_harness_report,
)
from .subagent_tasks import (
    build_board_zcu104_signoff_evidence_agent_task,
    build_board_zcu104_signoff_evidence_template,
    build_board_zcu104_signoff_readiness_report,
    build_codex_spawn_instructions,
    build_full_model_target_rtl_generator_agent_task,
    build_full_target_llama_accelerator_artifact_agent_task,
    build_full_llama_execution_evidence_agent_task,
    build_full_llama_execution_evidence_template,
    build_full_llama_execution_readiness_report,
    build_hdl_subagent_execution_manifest,
    build_hdl_subagent_skill_update_draft,
    build_hdl_subagent_wave_status,
    build_model_level_execution_harness_agent_task,
    build_target_scale_child_packet_agent_task,
    TARGET_SCALE_CHILD_PACKET_TASKS,
    run_board_zcu104_signoff_evidence_agent,
    run_full_llama_execution_evidence_agent,
    build_target_evidence_execution_manifest,
    build_zcu104_board_wrapper_axi_bridge_agent_task,
    write_codex_spawn_instructions,
    write_parent_feedback_loop_state,
)


@dataclass(frozen=True)
class ParentLoopOptions:
    max_iterations: int = 8
    max_subagents_per_iteration: int | None = None
    skip_synth: bool = True
    skip_vivado_route: bool = True
    vivado_executable: str = "vivado"
    backend: str = "local"
    local_verification: bool = False
    verbose: bool = False
    board_wrapper_evidence_dir: Path | None = None
    auto_tune_ooc: bool = False
    max_ooc_tuning_attempts: int = 1


@dataclass(frozen=True)
class ParentState:
    dispatch_plan: dict[str, Any]
    wave_status: dict[str, Any]
    execution_manifest: dict[str, Any]
    full_execution_readiness: dict[str, Any]
    board_signoff_readiness: dict[str, Any]
    target_tasks: dict[str, dict[str, Any]]
    status_paths: dict[str, Path]


def run_parent_loop(
    model_name: str,
    config: AgentConfig,
    out_dir: Path,
    *,
    partition: str = "gemm_non_gemm",
    options: ParentLoopOptions | None = None,
) -> dict[str, Any]:
    """Run the parent-owned feedback loop over collected sub-agent evidence.

    The parent loop never writes HDL itself. In `local` backend mode it invokes
    existing bounded sub-agent executors such as `emit_kernel`, then refreshes
    parent-owned evidence state. Codex-only verification and signoff work is
    queued unless `local_verification` is explicitly enabled for deterministic
    CI smoke coverage.
    """

    options = options or ParentLoopOptions()
    if options.backend not in {"local", "queue"}:
        raise ValueError("parent loop backend must be 'local' or 'queue'")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if options.max_subagents_per_iteration is not None and options.max_subagents_per_iteration <= 0:
        raise ValueError("max_subagents_per_iteration must be positive when provided")
    if options.max_ooc_tuning_attempts < 0:
        raise ValueError("max_ooc_tuning_attempts must be non-negative")

    out_dir.mkdir(parents=True, exist_ok=True)
    inspect_dir = out_dir / "inspect"
    evidence_root = out_dir / "evidence"
    status_dir = out_dir / "status"
    evidence_root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    inspect_report = _ensure_inspect_artifacts(model_name, config, inspect_dir, partition, options.verbose)
    if inspect_report.get("status") == "needs_clarification":
        report = _final_report(
            out_dir,
            status="needs_clarification",
            model_name=model_name,
            config=config,
            inspect_dir=inspect_dir,
            evidence_root=evidence_root,
            status_dir=status_dir,
            iterations=[],
            queue_entries=[],
            reason="Parent clarification gate blocked sub-agent dispatch",
        )
        _write_json(out_dir / "parent_loop_run_report.json", report)
        return report

    dispatch_plan_path = inspect_dir / "hdl_subagent_dispatch_plan.json"
    if not dispatch_plan_path.exists():
        raise FileNotFoundError(f"inspect did not emit dispatch plan: {dispatch_plan_path}")

    evidence_imports: list[dict[str, Any]] = []
    if options.board_wrapper_evidence_dir is not None:
        evidence_imports.append(
            _import_board_wrapper_evidence(
                Path(options.board_wrapper_evidence_dir),
                evidence_root,
                status_dir,
            )
        )

    iterations: list[dict[str, Any]] = []
    queue_entries: list[dict[str, Any]] = []
    executed_total = 0
    queued_total = 0
    last_state: ParentState | None = None
    final_reason = "max_iterations_reached"

    for index in range(options.max_iterations):
        state = _refresh_parent_state(dispatch_plan_path, evidence_root, status_dir)
        last_state = state
        execution_entries = list(state.execution_manifest.get("spawn_entries", []))
        ready_entries = _ready_entries_for_iteration(execution_entries, options.max_subagents_per_iteration)
        iteration_record: dict[str, Any] = {
            "iteration": index + 1,
            "state_before": _manifest_summary(state.execution_manifest),
            "ready_entry_count": len(execution_entries),
            "selected_entry_count": len(ready_entries),
            "executed": [],
            "queued": [],
        }

        if state.execution_manifest.get("skill_update_required"):
            skill_draft = _write_skill_update_draft(state.dispatch_plan, evidence_root, status_dir)
            iteration_record["skill_update_draft"] = skill_draft
            final_reason = "blocked_skill_update_required"
            iterations.append(iteration_record)
            break

        if state.execution_manifest.get("missing_skill_update_candidate"):
            final_reason = "blocked_missing_skill_update_candidate"
            iterations.append(iteration_record)
            break

        if not execution_entries:
            target_actions = _run_or_queue_target_tasks(state, config, evidence_root, status_dir, options)
            iteration_record["target_actions"] = target_actions
            queue_entries.extend(target_actions["queued"])
            queued_total += len(target_actions["queued"])
            if target_actions["executed"]:
                executed_total += len(target_actions["executed"])
                iteration_record["executed"].extend(target_actions["executed"])
                iterations.append(iteration_record)
                continue
            preflight_queue = _target_preflight_queue_entries(state.dispatch_plan, state.full_execution_readiness)
            if preflight_queue:
                iteration_record["queued"].extend(preflight_queue)
                queue_entries.extend(preflight_queue)
                queued_total += len(preflight_queue)
                final_reason = "blocked_by_target_preflight"
                iterations.append(iteration_record)
                break
            final_reason = "idle_or_waiting_for_external_target_evidence"
            iterations.append(iteration_record)
            break

        if not ready_entries:
            final_reason = "max_subagents_per_iteration_prevented_dispatch"
            iterations.append(iteration_record)
            break

        for entry in ready_entries:
            if options.backend == "queue":
                queued = _queue_external_entry(entry, "backend_queue_mode")
                iteration_record["queued"].append(queued)
                queue_entries.append(queued)
                queued_total += 1
                continue
            action = _run_local_or_queue_entry(entry, config, evidence_root, options)
            iteration_record[action["parent_loop_action"]].append(action)
            if action["parent_loop_action"] == "executed":
                executed_total += 1
            else:
                queue_entries.append(action)
                queued_total += 1

        iterations.append(iteration_record)

    final_state = _refresh_parent_state(dispatch_plan_path, evidence_root, status_dir)
    final_queue = _write_parent_queue(status_dir, queue_entries)
    status = _parent_loop_status(final_state, final_reason, executed_total, queued_total)
    report = _final_report(
        out_dir,
        status=status,
        model_name=model_name,
        config=config,
        inspect_dir=inspect_dir,
        evidence_root=evidence_root,
        status_dir=status_dir,
        iterations=iterations,
        queue_entries=queue_entries,
        reason=final_reason,
        final_state=final_state,
        parent_queue=final_queue,
        evidence_imports=evidence_imports,
    )
    _write_json(out_dir / "parent_loop_run_report.json", report)
    return report


def _ensure_inspect_artifacts(
    model_name: str,
    config: AgentConfig,
    inspect_dir: Path,
    partition: str,
    verbose: bool,
) -> dict[str, Any]:
    report_path = inspect_dir / "llm_agent_report.json"
    dispatch_plan_path = inspect_dir / "hdl_subagent_dispatch_plan.json"
    if _inspect_artifacts_match_inputs(inspect_dir, model_name, config, partition):
        return _read_json(report_path)
    return run_llm_agent(
        model_name=model_name,
        config=config,
        out_dir=inspect_dir,
        mode="inspect",
        kernel=None,
        partition=partition,
        skip_synth=True,
        verbose=verbose,
    )


def _inspect_artifacts_match_inputs(
    inspect_dir: Path,
    model_name: str,
    config: AgentConfig,
    partition: str,
) -> bool:
    report_path = inspect_dir / "llm_agent_report.json"
    dispatch_plan_path = inspect_dir / "hdl_subagent_dispatch_plan.json"
    if not report_path.exists() or not dispatch_plan_path.exists():
        return False
    try:
        report = _read_json(report_path)
        dispatch_plan = _read_json(dispatch_plan_path)
    except (json.JSONDecodeError, ValueError):
        return False
    if report.get("model") != model_name or report.get("partition") != partition:
        return False
    source_replay = dispatch_plan.get("source_replay")
    if not isinstance(source_replay, dict):
        return False
    expected_source = {
        "model_name": model_name,
        "gptq_checkpoint": config.model.gptq_checkpoint,
        "mlir_graph": config.model.mlir_graph,
        "model_structure_source": config.model.model_structure_source,
    }
    for key, expected in expected_source.items():
        actual = source_replay.get(key)
        if key == "model_structure_source" and actual is None:
            actual = "mlir"
        if actual != expected:
            return False

    hardware = dispatch_plan.get("hardware")
    if not isinstance(hardware, dict):
        return False
    for key, expected in _hardware_spec_identity(config).items():
        if key in hardware and hardware.get(key) != expected:
            return False

    optimization = dispatch_plan.get("optimization")
    if not isinstance(optimization, dict):
        return False
    expected_optimization = {
        "quantization": config.optimization.quantization,
        "design_style": config.design.style,
        "compute_style": config.design.compute_style,
        "execution_style": config.design.execution_style,
        "memory_style": config.design.memory_style,
        "control_style": config.design.control_style,
        "pe_count": config.design.pe_count,
    }
    for key, expected in expected_optimization.items():
        if optimization.get(key) != expected:
            return False
    return True


def _resolve_existing_import_path(path_value: Any, source_dir: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    candidates = [candidate] if candidate.is_absolute() else [candidate, source_dir / candidate, source_dir / candidate.name]
    for item in candidates:
        if item.exists():
            return item
    return None


def _copy_import_file(source: Path, dest_dir: Path) -> str:
    dest = dest_dir / source.name
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return str(dest)


def _import_board_wrapper_evidence(source_dir: Path, evidence_root: Path, status_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.expanduser()
    dest_dir = evidence_root / "board_zcu104_signoff_gate"
    dest_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    import_report_path = status_dir / "board_wrapper_evidence_import.json"
    source_report_path = source_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    if not source_report_path.exists():
        result = {
            "artifact": "board_wrapper_evidence_import",
            "status": "failed",
            "source_dir": str(source_dir),
            "destination_dir": str(dest_dir),
            "failure": "zcu104_board_wrapper_axi_bridge_implementation_report.json not found",
        }
        _write_json(import_report_path, result)
        return result

    wrapper_report = _read_json(source_report_path)
    source_evidence_files = wrapper_report.get("evidence_files")
    if not isinstance(source_evidence_files, dict):
        source_evidence_files = {}

    copied_files: dict[str, str] = {}
    missing_files: list[str] = []
    for key, value in source_evidence_files.items():
        if key == "implementation_report":
            continue
        resolved = _resolve_existing_import_path(value, source_dir)
        if resolved is None:
            missing_files.append(key)
            continue
        copied_files[key] = _copy_import_file(resolved, dest_dir)

    source_subagent_result = _resolve_existing_import_path(
        source_evidence_files.get("subagent_result")
        or source_dir / "zcu104_board_wrapper_axi_bridge_subagent_result.json",
        source_dir,
    )
    if source_subagent_result is not None:
        copied_files["subagent_result"] = _copy_import_file(source_subagent_result, dest_dir)

    destination_report_path = dest_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    copied_files["implementation_report"] = str(destination_report_path)
    wrapper_report["evidence_files"] = {**source_evidence_files, **copied_files}

    bitstream_path = Path(copied_files["bitstream"]) if copied_files.get("bitstream") else None
    if bitstream_path is not None and bitstream_path.exists():
        wrapper_report["bitstream_file"] = str(bitstream_path)
        wrapper_report["bitstream_size_bytes"] = bitstream_path.stat().st_size
        wrapper_report["bitstream_generated"] = True

    invalidated_board_signoff_evidence: str | None = None
    stale_board_signoff_evidence = evidence_root / "board_zcu104_signoff_evidence.json"
    if stale_board_signoff_evidence.exists():
        invalidated_dir = status_dir / "invalidated_evidence"
        invalidated_dir.mkdir(parents=True, exist_ok=True)
        invalidated_path = invalidated_dir / "board_zcu104_signoff_evidence_after_board_wrapper_import.json"
        shutil.move(str(stale_board_signoff_evidence), str(invalidated_path))
        invalidated_board_signoff_evidence = str(invalidated_path)

    _write_json(destination_report_path, wrapper_report)
    status = "passed" if not missing_files else "incomplete"
    result = {
        "artifact": "board_wrapper_evidence_import",
        "status": status,
        "source_dir": str(source_dir),
        "source_report": str(source_report_path),
        "destination_dir": str(dest_dir),
        "destination_report": str(destination_report_path),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "bitstream_imported": bitstream_path is not None and bitstream_path.exists(),
        "invalidated_board_signoff_evidence": invalidated_board_signoff_evidence,
        "does_not_claim": [
            "full LLaMA execution",
            "board-level ZCU104 signoff",
            "full target-scale LLaMA accelerator bitstream",
        ],
    }
    _write_json(import_report_path, result)
    return result


def _refresh_parent_state(dispatch_plan_path: Path, evidence_root: Path, status_dir: Path) -> ParentState:
    dispatch_plan = _read_json(dispatch_plan_path)
    wave_status = build_hdl_subagent_wave_status(dispatch_plan, evidence_root)
    execution_manifest = build_hdl_subagent_execution_manifest(dispatch_plan, wave_status)

    _write_json(status_dir / "hdl_subagent_wave_status.json", wave_status)
    _write_json(status_dir / "hdl_subagent_execution_manifest.json", execution_manifest)
    write_parent_feedback_loop_state(dispatch_plan, wave_status, execution_manifest, status_dir)
    write_codex_spawn_instructions(execution_manifest, status_dir)

    full_execution_readiness = build_full_llama_execution_readiness_report(
        dispatch_plan,
        wave_status,
        evidence_root,
    )
    board_signoff_readiness = build_board_zcu104_signoff_readiness_report(
        dispatch_plan,
        full_execution_readiness,
        evidence_root,
    )
    full_execution_template = build_full_llama_execution_evidence_template(
        dispatch_plan,
        wave_status,
        evidence_root,
    )
    board_signoff_template = build_board_zcu104_signoff_evidence_template(
        dispatch_plan,
        full_execution_readiness,
        evidence_root,
    )
    model_harness_task = build_model_level_execution_harness_agent_task(
        dispatch_plan,
        wave_status,
        full_execution_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    full_execution_task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        full_execution_readiness,
        full_execution_template,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    board_signoff_task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        board_signoff_template,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    board_wrapper_task = build_zcu104_board_wrapper_axi_bridge_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    target_scale_child_tasks = {
        str(spec["task_id"]): build_target_scale_child_packet_agent_task(
            dispatch_plan,
            full_execution_readiness,
            board_signoff_readiness,
            evidence_root,
            str(spec["task_id"]),
            dispatch_plan_path=str(dispatch_plan_path),
        )
        for spec in TARGET_SCALE_CHILD_PACKET_TASKS
    }
    full_model_target_rtl_task = build_full_model_target_rtl_generator_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    full_target_artifact_task = build_full_target_llama_accelerator_artifact_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )

    target_tasks = {
        "model_level_execution_harness": model_harness_task,
        "full_llama_execution": full_execution_task,
        **target_scale_child_tasks,
        "full_model_target_rtl_generator": full_model_target_rtl_task,
        "full_target_llama_accelerator_artifact": full_target_artifact_task,
        "zcu104_board_wrapper_axi_bridge": board_wrapper_task,
        "board_zcu104_signoff": board_signoff_task,
    }
    target_paths = _write_target_status_files(
        status_dir,
        full_execution_readiness,
        board_signoff_readiness,
        full_execution_template,
        board_signoff_template,
        target_tasks,
    )

    status_paths = {
        "dispatch_plan": dispatch_plan_path,
        "wave_status": status_dir / "hdl_subagent_wave_status.json",
        "execution_manifest": status_dir / "hdl_subagent_execution_manifest.json",
        "parent_loop_state": status_dir / "parent_loop_state.json",
        "feedback_packet": status_dir / "feedback_packet.json",
        "retry_plan": status_dir / "retry_plan.json",
        **target_paths,
    }
    return ParentState(
        dispatch_plan=dispatch_plan,
        wave_status=wave_status,
        execution_manifest=execution_manifest,
        full_execution_readiness=full_execution_readiness,
        board_signoff_readiness=board_signoff_readiness,
        target_tasks=target_tasks,
        status_paths=status_paths,
    )


def _write_target_status_files(
    status_dir: Path,
    full_execution_readiness: dict[str, Any],
    board_signoff_readiness: dict[str, Any],
    full_execution_template: dict[str, Any],
    board_signoff_template: dict[str, Any],
    target_tasks: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    fixed_artifacts = {
        "full_llama_execution_readiness": full_execution_readiness,
        "board_zcu104_signoff_readiness": board_signoff_readiness,
        "full_llama_execution_evidence_template": full_execution_template,
        "board_zcu104_signoff_evidence_template": board_signoff_template,
    }
    for name, payload in fixed_artifacts.items():
        path = status_dir / f"{name}.json"
        _write_json(path, payload)
        paths[name] = path

    for name, task in target_tasks.items():
        task_path = status_dir / f"{name}_agent_task.json"
        prompt_path = status_dir / task["prompt_file"]
        manifest_path = status_dir / f"{name}_execution_manifest.json"
        spawn_path = status_dir / f"{name}_spawn_instructions.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(task["prompt"], encoding="utf-8")
        _write_json(task_path, {key: value for key, value in task.items() if key != "prompt"})
        manifest = build_target_evidence_execution_manifest(task)
        _write_json(manifest_path, manifest)
        spawn_path.write_text(build_codex_spawn_instructions(manifest), encoding="utf-8")
        paths[f"{name}_agent_task"] = task_path
        paths[f"{name}_execution_manifest"] = manifest_path
        paths[f"{name}_spawn_instructions"] = spawn_path
    return paths


def _ready_entries_for_iteration(
    execution_entries: list[dict[str, Any]],
    max_subagents_per_iteration: int | None,
) -> list[dict[str, Any]]:
    if max_subagents_per_iteration is None:
        return execution_entries
    return execution_entries[:max_subagents_per_iteration]


def _run_local_or_queue_entry(
    entry: dict[str, Any],
    config: AgentConfig,
    evidence_root: Path,
    options: ParentLoopOptions,
) -> dict[str, Any]:
    spawn_kind = entry.get("spawn_kind")
    if spawn_kind == "implementation_agent":
        return _run_local_implementation_entry(entry, config, evidence_root, options)
    if spawn_kind == "verification_agent" and options.local_verification:
        if options.skip_synth and entry.get("runs_integration_synthesis"):
            missing_synthesis = _verification_tasks_missing_synthesis(entry, evidence_root)
            if missing_synthesis:
                queued = _queue_external_entry(entry, "integration_synthesis_required_but_skip_synth_enabled")
                queued["missing_synthesis_task_ids"] = missing_synthesis
                queued["next_action"] = (
                    "rerun parent-loop without --skip-synth for these implementation tasks or collect Vivado "
                    "timing/resource evidence before local integration verification"
                )
                return queued
        if not options.skip_synth and entry.get("runs_integration_synthesis"):
            missing_tasks = _verification_tasks_missing_synthesis_tasks(entry, evidence_root)
            if missing_tasks:
                return _run_child_synthesis_then_verification(
                    entry,
                    missing_tasks,
                    config,
                    evidence_root,
                    options,
                )
        return _run_local_verification_entry(entry, config, evidence_root)
    return _queue_external_entry(entry, "requires_codex_subagent_or_local_verification_flag")


def _verification_tasks_missing_synthesis(entry: dict[str, Any], evidence_root: Path) -> list[str]:
    return [
        str(task.get("task_id") or "unknown_task")
        for task in _verification_tasks_missing_synthesis_tasks(entry, evidence_root)
    ]


def _verification_tasks_missing_synthesis_tasks(
    entry: dict[str, Any],
    evidence_root: Path,
) -> list[dict[str, Any]]:
    missing_tasks: list[dict[str, Any]] = []
    implementation_tasks = entry.get("implementation_tasks")
    if not isinstance(implementation_tasks, list):
        return missing_tasks
    for task in implementation_tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "unknown_task")
        evidence_dir = _local_evidence_dir_from_expected(
            task.get("expected_evidence_dir"),
            evidence_root,
            f"{task_id}_gate",
        )
        kernel_report, error = _load_json_object(evidence_dir / "kernel_report.json")
        if error or kernel_report is None:
            continue
        synthesis = kernel_report.get("synthesis")
        if not isinstance(synthesis, dict) or synthesis.get("passed") is not True:
            missing_tasks.append(task)
    return missing_tasks


def _implementation_entry_for_verification_child(
    verification_entry: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "unknown_task")
    return {
        **task,
        "spawn_key": f"{verification_entry.get('spawn_key', verification_entry.get('wave_id'))}::child_synthesis::{task_id}",
        "spawn_kind": "implementation_agent",
        "agent_hierarchy_role": "subagent",
        "subagent_type": task.get("subagent_type", "implementation_subagent"),
        "subagent_may_spawn_subagents": False,
        "parent_feedback_channel": verification_entry.get("parent_feedback_channel", "feedback_packet.json"),
        "wave_id": verification_entry.get("wave_id"),
        "source_replay": task.get("source_replay") or verification_entry.get("source_replay", {}),
        "required_commands": task.get("required_commands", []),
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required": True,
        "rerun_reason": "child_synthesis_required_before_integration_verification",
    }


def _run_child_synthesis_then_verification(
    verification_entry: dict[str, Any],
    missing_tasks: list[dict[str, Any]],
    config: AgentConfig,
    evidence_root: Path,
    options: ParentLoopOptions,
) -> dict[str, Any]:
    child_options = replace(options, skip_synth=False)
    child_actions: list[dict[str, Any]] = []
    for task in missing_tasks:
        child_entry = _implementation_entry_for_verification_child(verification_entry, task)
        child_action = _run_local_implementation_entry(child_entry, config, evidence_root, child_options)
        child_actions.append(child_action)
        if child_action.get("parent_loop_action") != "executed" or child_action.get("status") != "passed":
            queued = _queue_external_entry(
                verification_entry,
                "child_synthesis_rerun_failed_before_integration_verification",
            )
            queued["child_synthesis_actions"] = child_actions
            queued["missing_synthesis_task_ids"] = [
                str(item.get("task_id") or "unknown_task") for item in missing_tasks
            ]
            queued["next_action"] = (
                "inspect child_synthesis_actions, fix the failing implementation sub-agent, then rerun "
                "parent-loop so integration verification can consume Vivado synthesis evidence"
            )
            return queued

    verification_action = _run_local_verification_entry(verification_entry, config, evidence_root)
    verification_action["backend"] = "local_feedback_child_synthesis_then_integration_verification"
    verification_action["child_synthesis_actions"] = child_actions
    verification_action["child_synthesis_task_ids"] = [
        str(task.get("task_id") or "unknown_task") for task in missing_tasks
    ]
    return verification_action


def _run_local_implementation_entry(
    entry: dict[str, Any],
    config: AgentConfig,
    evidence_root: Path,
    options: ParentLoopOptions,
) -> dict[str, Any]:
    kernel = entry.get("current_regression_kernel")
    if not isinstance(kernel, str) or not kernel:
        return _queue_external_entry(entry, "missing_current_regression_kernel")

    evidence_dir = _entry_evidence_dir(entry, evidence_root)
    tuning_blocker = _existing_module_ooc_tuning_blocker(evidence_dir)
    if tuning_blocker is not None:
        tuning_history = _load_ooc_tuning_history(evidence_dir)
        attempt_count = len(tuning_history.get("attempts", []))
        tuning_effectiveness = tuning_blocker.get("tuning_effectiveness")
        if isinstance(tuning_effectiveness, dict) and tuning_effectiveness.get("effective") is False:
            queued = _queue_external_entry(entry, "module_ooc_tuning_ineffective_requires_generator_fix")
            queued["module_ooc_synthesis_report"] = str(evidence_dir / "module_ooc_synthesis_report.json")
            queued["resource_assessment"] = tuning_blocker.get("resource_assessment")
            queued["tuning_recommendation"] = tuning_blocker.get("tuning_recommendation")
            queued["tuning_effectiveness"] = tuning_effectiveness
            if isinstance(tuning_blocker.get("skill_update_candidate"), dict):
                queued["skill_update_candidate"] = tuning_blocker["skill_update_candidate"]
            elif isinstance(tuning_effectiveness.get("skill_update_candidate"), dict):
                queued["skill_update_candidate"] = tuning_effectiveness["skill_update_candidate"]
            queued["auto_tune_ooc"] = options.auto_tune_ooc
            queued["max_ooc_tuning_attempts"] = options.max_ooc_tuning_attempts
            queued["completed_ooc_tuning_attempts"] = attempt_count
            queued["next_action"] = (
                "fix the kernel generator or module packet so tuning knobs change true RTL datapath resources"
            )
            return queued
        if options.auto_tune_ooc and attempt_count < options.max_ooc_tuning_attempts:
            tuned_config, tune_change = _config_with_ooc_tuning(config, tuning_blocker)
            if tune_change.get("changed"):
                archive_dir = _archive_ooc_tuning_attempt(evidence_dir, attempt_count + 1, tuning_blocker)
                with _subagent_env(entry):
                    result = emit_kernel(kernel, tuned_config, evidence_dir, skip_synth=options.skip_synth)
                kernel_report = result.to_dict()

                module_ooc_path = evidence_dir / "module_ooc_synthesis_report.json"
                module_ooc_report: dict[str, Any] | None = None
                if entry.get("requires_module_ooc_synthesis") and result.synthesis and result.synthesis.get("passed"):
                    module_ooc_report = _module_ooc_report_from_kernel_result(entry, tuned_config, kernel_report)
                    effectiveness = _module_ooc_tuning_effectiveness(
                        tuning_blocker,
                        module_ooc_report,
                        tune_change,
                    )
                    module_ooc_report["ooc_tuning_attempt"] = {
                        "attempt": attempt_count + 1,
                        "previous_archive_dir": str(archive_dir),
                        "applied_change": tune_change,
                        "source_tuning_recommendation": tuning_blocker.get("tuning_recommendation"),
                    }
                    module_ooc_report["tuning_effectiveness"] = effectiveness
                    if effectiveness.get("effective") is False:
                        module_ooc_report["skill_update_candidate"] = effectiveness.get("skill_update_candidate")
                    _write_json(module_ooc_path, module_ooc_report)
                _write_json(
                    evidence_dir / "subagent_result.json",
                    _subagent_result_from_kernel_result(
                        entry,
                        kernel_report,
                        module_ooc_path,
                        module_ooc_report=module_ooc_report,
                    ),
                )
                history_after = _append_ooc_tuning_history(
                    evidence_dir,
                    tuning_history,
                    {
                        "attempt": attempt_count + 1,
                        "archive_dir": str(archive_dir),
                        "applied_change": tune_change,
                        "result_status": result.status,
                        "module_ooc_synthesis_report": str(module_ooc_path)
                        if module_ooc_path.exists()
                        else None,
                    },
                )
                return {
                    "parent_loop_action": "executed",
                    "backend": "local_subagent_ooc_auto_tune",
                    "spawn_key": entry.get("spawn_key"),
                    "spawn_kind": entry.get("spawn_kind"),
                    "task_id": entry.get("task_id"),
                    "wave_id": entry.get("wave_id"),
                    "kernel": kernel,
                    "status": result.status,
                    "evidence_dir": str(evidence_dir),
                    "kernel_report": str(evidence_dir / "kernel_report.json"),
                    "subagent_result": str(evidence_dir / "subagent_result.json"),
                    "module_ooc_synthesis_report": str(module_ooc_path) if module_ooc_path.exists() else None,
                    "auto_tune_ooc": True,
                    "tuning_attempt": attempt_count + 1,
                    "applied_change": tune_change,
                    "tuning_history": str(evidence_dir / "ooc_tuning_history" / "tuning_history.json"),
                    "tuning_attempt_count": len(history_after.get("attempts", [])),
                    "does_not_claim": [
                        "parent hand-wrote HDL",
                        "Codex Desktop sub-agent was spawned from package runtime",
                        "resource tuning completed for every module",
                    ],
                }

        queued = _queue_external_entry(entry, "module_ooc_tuning_required_before_local_rerun")
        queued["module_ooc_synthesis_report"] = str(evidence_dir / "module_ooc_synthesis_report.json")
        queued["resource_assessment"] = tuning_blocker.get("resource_assessment")
        queued["tuning_recommendation"] = tuning_blocker.get("tuning_recommendation")
        queued["tuning_effectiveness"] = tuning_blocker.get("tuning_effectiveness")
        if isinstance(tuning_blocker.get("skill_update_candidate"), dict):
            queued["skill_update_candidate"] = tuning_blocker["skill_update_candidate"]
        queued["auto_tune_ooc"] = options.auto_tune_ooc
        queued["max_ooc_tuning_attempts"] = options.max_ooc_tuning_attempts
        queued["completed_ooc_tuning_attempts"] = attempt_count
        if options.auto_tune_ooc and attempt_count < options.max_ooc_tuning_attempts:
            queued["reason"] = "module_ooc_tuning_no_allowed_knob_change"
        queued["next_action"] = "adjust allowed knobs such as pe_count, tile sizes, buffering depth, or memory width before rerun"
        return queued

    if (
        options.skip_synth
        and entry.get("requires_module_ooc_synthesis")
        and (evidence_dir / "kernel_report.json").exists()
        and (evidence_dir / "subagent_result.json").exists()
        and not (evidence_dir / "module_ooc_synthesis_report.json").exists()
    ):
        return _queue_external_entry(entry, "module_ooc_synthesis_required_but_skip_synth_enabled")

    with _subagent_env(entry):
        result = emit_kernel(kernel, config, evidence_dir, skip_synth=options.skip_synth)
    kernel_report = result.to_dict()

    module_ooc_path = evidence_dir / "module_ooc_synthesis_report.json"
    if entry.get("requires_module_ooc_synthesis") and result.synthesis and result.synthesis.get("passed"):
        _write_json(
            module_ooc_path,
            _module_ooc_report_from_kernel_result(entry, config, kernel_report),
        )
    _write_json(
        evidence_dir / "subagent_result.json",
        _subagent_result_from_kernel_result(entry, kernel_report, module_ooc_path),
    )

    return {
        "parent_loop_action": "executed",
        "backend": "local_subagent",
        "spawn_key": entry.get("spawn_key"),
        "spawn_kind": entry.get("spawn_kind"),
        "task_id": entry.get("task_id"),
        "wave_id": entry.get("wave_id"),
        "kernel": kernel,
        "status": result.status,
        "evidence_dir": str(evidence_dir),
        "kernel_report": str(evidence_dir / "kernel_report.json"),
        "subagent_result": str(evidence_dir / "subagent_result.json"),
        "module_ooc_synthesis_report": str(module_ooc_path) if module_ooc_path.exists() else None,
        "skip_synth": options.skip_synth,
        "does_not_claim": [
            "parent hand-wrote HDL",
            "Codex Desktop sub-agent was spawned from package runtime",
        ],
    }


def _existing_module_ooc_tuning_blocker(evidence_dir: Path) -> dict[str, Any] | None:
    report_path = evidence_dir / "module_ooc_synthesis_report.json"
    if not report_path.exists():
        return None
    try:
        report = _read_json(report_path)
    except (json.JSONDecodeError, ValueError):
        return None
    if report.get("status") != "passed":
        return None
    if report.get("resource_assessment") != "underutilized":
        return None
    if report.get("throughput_target_met") is True:
        return None
    kernel_report_path = evidence_dir / "kernel_report.json"
    if kernel_report_path.exists():
        try:
            kernel_report = _read_json(kernel_report_path)
        except (json.JSONDecodeError, ValueError):
            kernel_report = {}
        if _kernel_report_is_fixture_control_scaffold(kernel_report):
            return None
    return report


def _load_ooc_tuning_history(evidence_dir: Path) -> dict[str, Any]:
    path = evidence_dir / "ooc_tuning_history" / "tuning_history.json"
    if not path.exists():
        return {
            "artifact": "module_ooc_tuning_history",
            "attempts": [],
        }
    try:
        history = _read_json(path)
    except (json.JSONDecodeError, ValueError):
        return {
            "artifact": "module_ooc_tuning_history",
            "attempts": [],
            "load_warning": "previous tuning_history.json could not be parsed",
        }
    if not isinstance(history.get("attempts"), list):
        history["attempts"] = []
    return history


def _append_ooc_tuning_history(
    evidence_dir: Path,
    history: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    updated = {
        **history,
        "artifact": "module_ooc_tuning_history",
        "attempts": [*history.get("attempts", []), attempt],
    }
    _write_json(evidence_dir / "ooc_tuning_history" / "tuning_history.json", updated)
    return updated


def _archive_ooc_tuning_attempt(
    evidence_dir: Path,
    attempt: int,
    tuning_blocker: dict[str, Any],
) -> Path:
    history_dir = evidence_dir / "ooc_tuning_history"
    archive_dir = history_dir / f"attempt_{attempt:02d}_before_tuning"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    ignored = {"ooc_tuning_history"}
    for item in evidence_dir.iterdir():
        if item.name in ignored:
            continue
        dest = archive_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    _write_json(
        archive_dir / "source_tuning_blocker.json",
        {
            "artifact": "module_ooc_tuning_source_blocker",
            "tuning_blocker": tuning_blocker,
        },
    )
    return archive_dir


def _config_with_ooc_tuning(config: AgentConfig, tuning_blocker: dict[str, Any]) -> tuple[AgentConfig, dict[str, Any]]:
    recommendation = tuning_blocker.get("tuning_recommendation")
    if not isinstance(recommendation, dict):
        return config, {"changed": False, "reason": "tuning_recommendation missing"}
    suggested = recommendation.get("suggested_next_knobs")
    if not isinstance(suggested, dict):
        return config, {"changed": False, "reason": "suggested_next_knobs missing"}

    config_current = {
        "pe_count": config.design.pe_count,
        "memory_data_width": config.hardware.memory_data_width,
        "activation_buffer": config.design.activation_buffer,
        "weight_storage": config.design.weight_storage,
    }
    report_current = tuning_blocker.get("selected_tuning_knobs")
    if not isinstance(report_current, dict):
        report_current = config_current
    current = {
        "pe_count": int(report_current.get("pe_count", config_current["pe_count"])),
        "memory_data_width": int(report_current.get("memory_data_width", config_current["memory_data_width"])),
        "activation_buffer": str(report_current.get("activation_buffer", config_current["activation_buffer"])),
        "weight_storage": str(report_current.get("weight_storage", config_current["weight_storage"])),
    }
    next_pe = int(suggested.get("pe_count", current["pe_count"]))
    next_mem = int(suggested.get("memory_data_width", current["memory_data_width"]))
    next_activation = str(suggested.get("activation_buffer", current["activation_buffer"]))
    next_weight = str(suggested.get("weight_storage", current["weight_storage"]))

    tuned = replace(
        config,
        design=replace(
            config.design,
            pe_count=next_pe,
            activation_buffer=next_activation,
            weight_storage=next_weight,
        ),
        hardware=replace(config.hardware, memory_data_width=next_mem),
    )
    validate_config(tuned)
    updated = {
        "pe_count": tuned.design.pe_count,
        "memory_data_width": tuned.hardware.memory_data_width,
        "activation_buffer": tuned.design.activation_buffer,
        "weight_storage": tuned.design.weight_storage,
    }
    return tuned, {
        "changed": updated != current,
        "current_knobs": current,
        "updated_knobs": updated,
        "source": "module_ooc_synthesis_report.tuning_recommendation.suggested_next_knobs",
    }


def _module_ooc_tuning_effectiveness(
    previous_report: dict[str, Any],
    current_report: dict[str, Any],
    tune_change: dict[str, Any],
) -> dict[str, Any]:
    previous_utilization = previous_report.get("utilization")
    current_utilization = current_report.get("utilization")
    if not isinstance(previous_utilization, dict) or not isinstance(current_utilization, dict):
        return {
            "effective": None,
            "reason": "previous or current utilization is missing",
        }
    deltas = {}
    for key in sorted(set(previous_utilization) | set(current_utilization)):
        before = previous_utilization.get(key)
        after = current_utilization.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas[key] = after - before
    changed_resources = {key: value for key, value in deltas.items() if abs(value) > 0}
    effective = bool(changed_resources)
    result: dict[str, Any] = {
        "effective": effective,
        "resource_deltas": deltas,
        "changed_resources": changed_resources,
        "applied_knob_change": tune_change,
    }
    if not effective and tune_change.get("changed"):
        result["reason"] = "applied OOC tuning knobs did not change Vivado resource utilization"
        result["skill_update_candidate"] = {
            "failing_command": "parent-loop --auto-tune-ooc local module rerun",
            "symptom": "pe_count or other OOC tuning knobs changed, but post-route LUT/FF/DSP/BRAM/URAM/IO utilization was unchanged",
            "root_cause_hypothesis": "The selected kernel generator records requested parallelism but does not connect the tuned knob to true RTL datapath parallelism or resource allocation.",
            "prevention_rule": "When a sub-agent tunes pe_count or tile lanes, require kernel_report/module_ooc_synthesis_report to compare requested lanes, true parallel datapath lanes, and Vivado resource deltas; do not keep doubling a knob that leaves resources unchanged.",
            "minimal_regression_check": "Run parent-loop --auto-tune-ooc on a projection OOC report and assert unchanged utilization emits tuning_effectiveness.effective=false and a skill_update_candidate.",
        }
    else:
        result["reason"] = "Vivado resource utilization changed after tuning" if effective else "no knob change was applied"
    return result


def _local_evidence_dir_from_expected(expected: Any, evidence_root: Path, fallback_name: str) -> Path:
    if expected:
        candidate = Path(str(expected))
        if candidate.is_absolute():
            return candidate
        return evidence_root / candidate.name
    return evidence_root / fallback_name


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"{path} not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{path} json decode failed: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} JSON root is not an object"
    return data, None


def _integration_resource_utilization(synthesis: dict[str, Any]) -> dict[str, int]:
    resources = synthesis.get("resource_utilization")
    if not isinstance(resources, dict):
        resources = {}
    return {
        "lut": _resource_used(resources, "lut_as_logic"),
        "ff": _resource_used(resources, "clb_registers"),
        "dsp": _resource_used(resources, "dsps"),
        "bram": _resource_used(resources, "block_ram_tile"),
        "uram": _resource_used(resources, "uram"),
        "io": _resource_used(resources, "bonded_iob"),
    }


def _integration_timing_from_synthesis(synthesis: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    timing = synthesis.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    report_name = synthesis.get("timing_report")
    report_path = evidence_dir / str(report_name) if report_name else None
    parsed = None
    if report_path is not None and report_path.exists():
        parsed = _parse_vivado_timing_summary(report_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(parsed, dict):
        return {
            "constraints_met": timing.get("constraints_met") is True,
            "setup_worst_slack_ns": timing.get("setup_worst_slack_ns"),
            "hold_worst_slack_ns": timing.get("hold_worst_slack_ns"),
            "pulse_width_worst_slack_ns": timing.get("pulse_width_worst_slack_ns"),
            "setup_failing_endpoints": None,
            "hold_failing_endpoints": None,
            "pulse_width_failing_endpoints": None,
            "timing_summary": str(report_path) if report_path is not None else None,
            "parse_source": "kernel_report_synthesis_timing",
        }
    return {
        "constraints_met": parsed.get("constraints_met") is True,
        "setup_worst_slack_ns": parsed.get("setup_worst_slack_ns"),
        "hold_worst_slack_ns": parsed.get("hold_worst_slack_ns"),
        "pulse_width_worst_slack_ns": parsed.get("pulse_width_worst_slack_ns"),
        "setup_failing_endpoints": parsed.get("setup_failing_endpoints"),
        "hold_failing_endpoints": parsed.get("hold_failing_endpoints"),
        "pulse_width_failing_endpoints": parsed.get("pulse_width_failing_endpoints"),
        "missing_fields": parsed.get("missing_fields", []),
        "timing_summary": str(report_path),
        "parse_source": "vivado_timing_summary_report",
    }


def _integration_verification_skill_candidate(reason: str) -> dict[str, str]:
    return {
        "failing_command": "local integration verification sub-agent inspected integration Vivado evidence",
        "symptom": reason,
        "root_cause_hypothesis": (
            "The integration verification wave could not prove composed-top simulation, lint, synthesis, "
            "or timing evidence from the collected implementation artifacts."
        ),
        "prevention_rule": (
            "Integration verification sub-agents must write integration_synthesis_report.json from the "
            "composed integration top's Vivado reports, including setup, hold, pulse-width, endpoint counts, "
            "resource use, hardware identity, and residual fixture-scope limits."
        ),
        "minimal_regression_check": (
            "Run parent-loop with --local-verification on a decoder-block integration fixture and assert "
            "wave_2_decoder_block advances only when integration_synthesis_report.json exists and has status passed."
        ),
    }


def _local_integration_synthesis_report(
    entry: dict[str, Any],
    config: AgentConfig,
    evidence_root: Path,
) -> tuple[dict[str, Any], Path]:
    integration_dir = _local_evidence_dir_from_expected(
        entry.get("expected_integration_synthesis_dir"),
        evidence_root,
        f"{entry.get('wave_id', 'unknown')}_integration_verification",
    )
    expected_report = entry.get("expected_integration_synthesis_report")
    report_name = Path(str(expected_report)).name if expected_report else "integration_synthesis_report.json"
    report_path = integration_dir / report_name

    failures: list[str] = []
    task_reports: list[dict[str, Any]] = []
    selected_child_modules: list[str] = []
    aggregate_utilization = {"lut": 0, "ff": 0, "dsp": 0, "bram": 0, "uram": 0, "io": 0}
    timing_records: list[dict[str, Any]] = []
    implementation_tasks = entry.get("implementation_tasks")
    if not isinstance(implementation_tasks, list) or not implementation_tasks:
        failures.append("verification entry does not list implementation_tasks")
        implementation_tasks = []

    for task in implementation_tasks:
        if not isinstance(task, dict):
            failures.append("implementation task entry is not an object")
            continue
        task_id = str(task.get("task_id") or "unknown_task")
        evidence_dir = _local_evidence_dir_from_expected(
            task.get("expected_evidence_dir"),
            evidence_root,
            f"{task_id}_gate",
        )
        kernel_report_path = evidence_dir / "kernel_report.json"
        subagent_result_path = evidence_dir / "subagent_result.json"
        kernel_report, kernel_error = _load_json_object(kernel_report_path)
        subagent_result, subagent_error = _load_json_object(subagent_result_path)
        if kernel_error:
            failures.append(kernel_error)
            continue
        assert kernel_report is not None
        if subagent_error:
            failures.append(subagent_error)
        elif subagent_result is not None and subagent_result.get("status") != "passed":
            failures.append(f"{subagent_result_path} status is {subagent_result.get('status')}")

        simulation = kernel_report.get("simulation") if isinstance(kernel_report.get("simulation"), dict) else {}
        verilator = kernel_report.get("verilator") if isinstance(kernel_report.get("verilator"), dict) else {}
        synthesis = kernel_report.get("synthesis") if isinstance(kernel_report.get("synthesis"), dict) else {}
        if kernel_report.get("status") != "passed":
            failures.append(f"{kernel_report_path} status is {kernel_report.get('status')}")
        if simulation.get("passed") is not True:
            failures.append(f"{task_id} simulation did not pass")
        if verilator.get("passed") is not True:
            failures.append(f"{task_id} Verilator lint did not pass")
        if synthesis.get("passed") is not True:
            failures.append(f"{task_id} Vivado synthesis did not pass")

        timing = _integration_timing_from_synthesis(synthesis, evidence_dir)
        if timing.get("constraints_met") is not True:
            failures.append(f"{task_id} timing constraints were not proven met")
        timing_records.append({"task_id": task_id, **timing})

        utilization = _integration_resource_utilization(synthesis)
        for key, value in utilization.items():
            aggregate_utilization[key] += value
        files = [str(item) for item in kernel_report.get("files", []) if isinstance(item, str)]
        selected_child_modules.extend(item for item in files if item.endswith(".sv") and not item.startswith("tb_"))
        task_reports.append(
            {
                "task_id": task_id,
                "coverage_level": kernel_report.get("coverage_level"),
                "implementation_stage": kernel_report.get("implementation_stage"),
                "evidence_dir": str(evidence_dir),
                "kernel_report": str(kernel_report_path),
                "subagent_result": str(subagent_result_path),
                "simulation_passed": simulation.get("passed") is True,
                "verilator_passed": verilator.get("passed") is True,
                "synthesis_passed": synthesis.get("passed") is True,
                "vivado_command": synthesis.get("cmd"),
                "vivado_log": str(evidence_dir / "vivado.log"),
                "timing_report": timing.get("timing_summary"),
                "utilization_report": str(evidence_dir / str(synthesis.get("utilization_report")))
                if synthesis.get("utilization_report")
                else None,
                "checkpoint": str(evidence_dir / "post_route.dcp")
                if (evidence_dir / "post_route.dcp").exists()
                else None,
                "timing": timing,
                "utilization": utilization,
            }
        )

    status = "passed" if not failures else "failed"
    candidate = _integration_verification_skill_candidate("; ".join(failures)) if failures else None
    report: dict[str, Any] = {
        "artifact": "integration_synthesis_report",
        "status": status,
        "wave_id": entry.get("wave_id"),
        "verification_backend": "local_integration_verification_subagent",
        "evidence_scope": "bounded_integration_fixture",
        "integration_top_modules": [task.get("task_id") for task in implementation_tasks if isinstance(task, dict)],
        "selected_child_module_list": sorted(set(selected_child_modules)),
        "hardware_spec": _hardware_spec_identity(config),
        "selected_child_tuning_knobs": {
            "pe_count": config.design.pe_count,
            "memory_data_width": config.hardware.memory_data_width,
            "activation_buffer": config.design.activation_buffer,
            "weight_storage": config.design.weight_storage,
        },
        "task_reports": task_reports,
        "timing": {
            "records": timing_records,
            "constraints_met": status == "passed"
            and all(record.get("constraints_met") is True for record in timing_records),
            "setup_worst_slack_ns": min(
                (
                    record["setup_worst_slack_ns"]
                    for record in timing_records
                    if isinstance(record.get("setup_worst_slack_ns"), (int, float))
                ),
                default=None,
            ),
            "hold_worst_slack_ns": min(
                (
                    record["hold_worst_slack_ns"]
                    for record in timing_records
                    if isinstance(record.get("hold_worst_slack_ns"), (int, float))
                ),
                default=None,
            ),
            "pulse_width_worst_slack_ns": min(
                (
                    record["pulse_width_worst_slack_ns"]
                    for record in timing_records
                    if isinstance(record.get("pulse_width_worst_slack_ns"), (int, float))
                ),
                default=None,
            ),
            "setup_failing_endpoints": sum(
                int(record.get("setup_failing_endpoints") or 0) for record in timing_records
            ),
            "hold_failing_endpoints": sum(
                int(record.get("hold_failing_endpoints") or 0) for record in timing_records
            ),
            "pulse_width_failing_endpoints": sum(
                int(record.get("pulse_width_failing_endpoints") or 0) for record in timing_records
            ),
        },
        "utilization": aggregate_utilization,
        "resource_assessment": _resource_assessment(aggregate_utilization, config),
        "resource_ratios": _resource_ratios(aggregate_utilization, config),
        "drc": {
            "status": "not_run_for_bounded_integration_fixture",
            "board_level_drc_required_before_board_signoff": True,
        },
        "methodology": {
            "status": "not_run_for_bounded_integration_fixture",
            "board_level_methodology_required_before_board_signoff": True,
        },
        "failures": failures,
        "residual_risks": [
            "bounded integration fixture only",
            "DRC and methodology are deferred to board-wrapper or board-level gates",
            "does not prove full target-scale LLaMA execution",
        ],
        "does_not_claim": [
            "full LLaMA execution",
            "board-level ZCU104 signoff",
            "full target-scale accelerator bitstream",
        ],
    }
    if candidate is not None:
        report["skill_update_candidate"] = candidate
    _write_json(report_path, report)
    return report, report_path


def _run_local_verification_entry(entry: dict[str, Any], config: AgentConfig, evidence_root: Path) -> dict[str, Any]:
    report_path = evidence_root / entry.get("verification_report", "")
    if not report_path.name:
        report_path = evidence_root / "verification_results" / f"{entry.get('wave_id', 'unknown')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runs_integration_synthesis = bool(entry.get("runs_integration_synthesis"))
    integration_report: dict[str, Any] | None = None
    integration_report_path: Path | None = None
    findings: list[dict[str, Any]] = []
    if runs_integration_synthesis:
        integration_report, integration_report_path = _local_integration_synthesis_report(entry, config, evidence_root)
        status = "passed" if integration_report.get("status") == "passed" else "failed"
        if status == "failed":
            candidate = integration_report.get("skill_update_candidate")
            findings.append(
                {
                    "severity": "P1",
                    "title": "Integration synthesis evidence did not pass",
                    "body": "; ".join(integration_report.get("failures", []))
                    or "integration_synthesis_report.json status is failed",
                    "skill_update_candidate": candidate,
                }
            )
        else:
            findings.append(
                {
                    "severity": "P3",
                    "title": "Bounded integration fixture is not board signoff",
                    "body": (
                        "Integration synthesis passed for the bounded fixture; DRC, methodology, "
                        "full model execution, and board-level signoff remain downstream gates."
                    ),
                }
            )
    else:
        status = "passed"
    if runs_integration_synthesis and integration_report_path is None:
        status = "failed"
        findings.append(
            {
                "severity": "P1",
                "title": "Integration synthesis report was not produced",
                "body": "Local integration verification could not create integration_synthesis_report.json.",
                "skill_update_candidate": _integration_verification_skill_candidate(
                    "integration_synthesis_report.json was not produced"
                ),
            }
        )
    does_not_claim = ["Codex Desktop external read-only verification agent audited the code"]
    if not runs_integration_synthesis:
        does_not_claim.append("integration synthesis passed")
    does_not_claim.extend(
        [
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ]
    )
    payload: dict[str, Any] = {
        "artifact": "hdl_subagent_verification_report",
        "status": status,
        "wave_id": entry.get("wave_id"),
        "verification_backend": "local_integration_verification_subagent"
        if runs_integration_synthesis
        else "local_deterministic_smoke",
        "findings": findings,
        "audit_summary": "Integration synthesis evidence passed for a bounded fixture."
        if runs_integration_synthesis and status == "passed"
        else (
            "Integration synthesis evidence failed or was incomplete."
            if runs_integration_synthesis
            else "Implementation evidence JSON was present for parent-loop smoke coverage."
        ),
        "does_not_claim": does_not_claim,
    }
    if integration_report_path is not None:
        payload["integration_synthesis_report"] = str(integration_report_path)
        payload["integration_synthesis_status"] = integration_report.get("status") if integration_report else None
    if status == "failed" and integration_report and isinstance(
        integration_report.get("skill_update_candidate"), dict
    ):
        payload["skill_update_candidate"] = integration_report["skill_update_candidate"]
    _write_json(
        report_path,
        payload,
    )
    return {
        "parent_loop_action": "executed",
        "backend": "local_integration_verification"
        if runs_integration_synthesis
        else "local_verification_smoke",
        "spawn_key": entry.get("spawn_key"),
        "spawn_kind": entry.get("spawn_kind"),
        "wave_id": entry.get("wave_id"),
        "status": status,
        "verification_report": str(report_path),
        "integration_synthesis_report": str(integration_report_path) if integration_report_path else None,
    }


def _run_or_queue_target_tasks(
    state: ParentState,
    config: AgentConfig,
    evidence_root: Path,
    status_dir: Path,
    options: ParentLoopOptions,
) -> dict[str, list[dict[str, Any]]]:
    executed: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []
    for name, task in state.target_tasks.items():
        if not task.get("ready_to_spawn"):
            existing_child_blocker = task.get("existing_child_blocker")
            if isinstance(existing_child_blocker, dict):
                queued.append(
                    {
                        "parent_loop_action": "queued",
                        "target_task": name,
                        "target_wave": task.get("target_wave"),
                        "reason": (
                            "target_scale_child_rtl_packet_skill_update_required"
                            if existing_child_blocker.get("skill_update_candidate_complete") is True
                            else "target_scale_child_rtl_packet_missing_skill_update_candidate"
                        ),
                        "expected_evidence_file": task.get("expected_evidence_file"),
                        "expected_subagent_result": task.get("expected_subagent_result"),
                        "ready_to_spawn": False,
                        "existing_child_blocker": existing_child_blocker,
                        "next_action": existing_child_blocker.get("next_action"),
                    }
                )
            continue
        if name == "model_level_execution_harness" and options.backend == "local":
            out_dir = evidence_root / "full_llama_execution_gate"
            report = write_model_level_execution_harness_report(state.dispatch_plan, out_dir)
            subagent_result_path = out_dir / "model_level_harness_subagent_result.json"
            subagent_result = {
                "artifact": "model_level_execution_harness_subagent_result",
                "task_id": "model_level_execution_harness",
                "status": report["status"],
                "changed_files": [str(out_dir / "model_level_execution_harness_report.json")],
                "commands_run": ["local model-level execution harness backend"],
                "evidence_paths": {
                    "model_level_execution_harness_report": str(
                        out_dir / "model_level_execution_harness_report.json"
                    ),
                },
                "remaining_risks": [
                    "Software harness evidence only; full execution evidence remains a downstream gate.",
                ],
            }
            _write_json(subagent_result_path, subagent_result)
            executed.append(
                {
                    "parent_loop_action": "executed",
                    "backend": "local_model_level_harness_subagent",
                    "target_task": name,
                    "status": report["status"],
                    "evidence_file": str(out_dir / "model_level_execution_harness_report.json"),
                    "subagent_result": str(subagent_result_path),
                }
            )
            continue
        if name == "full_llama_execution" and options.backend == "local":
            report = run_full_llama_execution_evidence_agent(
                state.dispatch_plan,
                state.wave_status,
                evidence_root,
            )
            executed.append(
                {
                    "parent_loop_action": "executed",
                    "backend": "local_full_execution_subagent",
                    "target_task": name,
                    "status": report["status"],
                    "evidence_written": report["evidence_written"],
                    "evidence_file": report["evidence_file"],
                    "subagent_result": report["subagent_result"],
                }
            )
            continue
        if task.get("target_wave") == "target_scale_child_rtl_wave":
            queued.append(
                {
                    "parent_loop_action": "queued",
                    "target_task": name,
                    "target_wave": "target_scale_child_rtl_wave",
                    "reason": "target_scale_child_rtl_packet_requires_codex_implementation_subagent",
                    "prompt_file": str(status_dir / task["prompt_file"]),
                    "expected_evidence_file": task.get("expected_evidence_file"),
                    "expected_subagent_result": task.get("expected_subagent_result"),
                    "ready_to_spawn": task.get("ready_to_spawn"),
                    "depends_on": task.get("depends_on", []),
                    "next_action": (
                        "run target-scale child RTL packet sub-agents before retrying "
                        "full_model_target_rtl_generator or rerouting the ZCU104 board wrapper"
                    ),
                }
            )
            continue
        if name == "full_model_target_rtl_generator":
            queued.append(
                {
                    "parent_loop_action": "queued",
                    "target_task": name,
                    "reason": "full_model_target_rtl_generator_requires_codex_implementation_subagent",
                    "prompt_file": str(status_dir / task["prompt_file"]),
                    "expected_evidence_file": task.get("expected_evidence_file"),
                    "expected_subagent_result": task.get("expected_subagent_result"),
                    "ready_to_spawn": task.get("ready_to_spawn"),
                    "next_action": (
                        "run the full-model target RTL generator sub-agent so it can produce a "
                        "non-fixture target-scale accelerator artifact before any ZCU104 board-wrapper reroute"
                    ),
                }
            )
            continue
        if name == "full_target_llama_accelerator_artifact":
            queued.append(
                {
                    "parent_loop_action": "queued",
                    "target_task": name,
                    "reason": "target_scale_accelerator_artifact_requires_codex_implementation_subagent",
                    "prompt_file": str(status_dir / task["prompt_file"]),
                    "expected_evidence_file": task.get("expected_evidence_file"),
                    "expected_subagent_result": task.get("expected_subagent_result"),
                    "ready_to_spawn": task.get("ready_to_spawn"),
                    "next_action": (
                        "run the target-scale accelerator artifact implementation sub-agent, then rerun "
                        "parent-loop so the ZCU104 board-wrapper can route that artifact"
                    ),
                }
            )
            continue
        if name == "zcu104_board_wrapper_axi_bridge" and options.backend == "local":
            if options.skip_vivado_route:
                queued.append(
                    {
                        "parent_loop_action": "queued",
                        "target_task": name,
                        "reason": "target_scale_board_wrapper_route_required_but_skip_vivado_route_enabled",
                        "prompt_file": str(status_dir / task["prompt_file"]),
                        "expected_evidence_file": task.get("expected_evidence_file"),
                        "expected_subagent_result": task.get("expected_subagent_result"),
                        "ready_to_spawn": task.get("ready_to_spawn"),
                        "next_action": (
                            "rerun parent-loop without --skip-vivado-route or run the board-wrapper "
                            "implementation sub-agent so it can produce a target-scale accelerator bitstream report"
                        ),
                    }
                )
                continue
            out_dir = Path(task.get("expected_evidence_dir") or evidence_root / "board_zcu104_signoff_gate")
            if not out_dir.is_absolute():
                out_dir = evidence_root / Path(out_dir).name
            accelerator_artifact = _select_board_wrapper_accelerator_artifact(evidence_root)
            existing_wrapper = _existing_board_wrapper_report(evidence_root)
            selected_target_eligible = bool(accelerator_artifact and accelerator_artifact.get("target_scale_eligible"))
            if (
                existing_wrapper is not None
                and existing_wrapper.get("status") == "passed"
                and existing_wrapper.get("bitstream_generated") is True
                and existing_wrapper.get("target_scale_accelerator_bitstream") is not True
                and not selected_target_eligible
            ):
                queued.append(
                    {
                        "parent_loop_action": "queued",
                        "target_task": name,
                        "reason": "target_scale_accelerator_artifact_required",
                        "prompt_file": str(status_dir / task["prompt_file"]),
                        "expected_evidence_file": task.get("expected_evidence_file"),
                        "expected_subagent_result": task.get("expected_subagent_result"),
                        "ready_to_spawn": task.get("ready_to_spawn"),
                        "existing_board_wrapper_report": str(
                            evidence_root
                            / "board_zcu104_signoff_gate"
                            / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
                        ),
                        "current_accelerator_artifact": accelerator_artifact,
                        "next_action": (
                            "generate or provide a target-scale non-fixture accelerator artifact, then rerun "
                            "the board-wrapper implementation sub-agent"
                        ),
                    }
                )
                continue
            report = run_zcu104_board_wrapper_axi_bridge_agent(
                config,
                out_dir,
                accelerator_artifact_dir=Path(accelerator_artifact["artifact_dir"]) if accelerator_artifact else None,
                accelerator_top_module=accelerator_artifact["top_module"] if accelerator_artifact else None,
                accelerator_kernel_report_path=Path(accelerator_artifact["kernel_report"]) if accelerator_artifact else None,
                run_vivado=not options.skip_vivado_route,
                vivado_executable=options.vivado_executable,
            )
            executed.append(
                {
                    "parent_loop_action": "executed",
                    "backend": "local_board_wrapper_subagent",
                    "target_task": name,
                    "status": report["status"],
                    "evidence_dir": str(out_dir),
                    "implementation_report": report["evidence_files"]["implementation_report"],
                    "subagent_result": report["evidence_files"]["subagent_result"],
                    "bitstream_generated": report.get("bitstream_generated"),
                    "bitstream_file": report.get("bitstream_file"),
                    "accelerator_artifact": accelerator_artifact,
                    "target_scale_accelerator_bitstream": report.get("target_scale_accelerator_bitstream"),
                    "accelerator_scope": report.get("accelerator_scope"),
                }
            )
            continue
        if name == "board_zcu104_signoff" and options.backend == "local":
            report = run_board_zcu104_signoff_evidence_agent(
                state.dispatch_plan,
                state.full_execution_readiness,
                evidence_root,
                board_wrapper_dir=evidence_root / "board_zcu104_signoff_gate",
            )
            executed.append(
                {
                    "parent_loop_action": "executed",
                    "backend": "local_board_signoff_subagent",
                    "target_task": name,
                    "status": report["status"],
                    "evidence_written": report["evidence_written"],
                    "evidence_file": report["evidence_file"],
                    "subagent_result": report["subagent_result"],
                    "readiness_status": report["readiness"]["status"],
                }
            )
            continue
        queued.append(
            {
                "parent_loop_action": "queued",
                "target_task": name,
                "reason": "target_evidence_requires_codex_or_external_agent",
                "prompt_file": str(status_dir / task["prompt_file"]),
                "expected_evidence_file": task.get("expected_evidence_file"),
                "ready_to_spawn": task.get("ready_to_spawn"),
            }
        )
    return {"executed": executed, "queued": queued}


def _queue_external_entry(entry: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "parent_loop_action": "queued",
        "backend": "external_codex_subagent",
        "reason": reason,
        "spawn_key": entry.get("spawn_key"),
        "spawn_kind": entry.get("spawn_kind"),
        "subagent_type": entry.get("subagent_type"),
        "task_id": entry.get("task_id"),
        "wave_id": entry.get("wave_id"),
        "prompt_file": entry.get("prompt_file"),
        "expected_evidence_dir": entry.get("expected_evidence_dir"),
        "expected_evidence_file": entry.get("expected_evidence_file"),
        "expected_subagent_result": entry.get("expected_subagent_result"),
        "verification_report": entry.get("verification_report"),
    }


def _target_preflight_queue_entries(
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
) -> list[dict[str, Any]]:
    target_preflight = full_execution_readiness.get("target_preflight")
    if not isinstance(target_preflight, dict) or target_preflight.get("status") != "blocked":
        return []
    blocker_ids = target_preflight.get("preflight_blockers")
    if not isinstance(blocker_ids, list) or not blocker_ids:
        return []
    blocked_tasks = {
        task.get("task_id"): task
        for task in dispatch_plan.get("blocked_target_tasks", [])
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    expected_artifacts = {
        "real_mlir_model_analysis": [
            "inspect/model_graph.mlir",
            "inspect/mlir_analysis.json",
            "inspect/mlir_model_analysis_readiness.json",
        ],
        "real_gptq_checkpoint_source": [
            "inspect/gptq_checkpoint_source_preflight.json",
        ],
        "real_gptq_checkpoint_metadata": [
            "inspect/gptq_checkpoint_metadata.json",
        ],
        "real_gptq_weight_layout_preflight": [
            "inspect/gptq_weight_layout_preflight.json",
            "inspect/projection_weight_stream_plan.json",
        ],
        "real_gptq_payload_probe": [
            "inspect/gptq_payload_probe.json",
        ],
    }
    entries = []
    for blocker_id in sorted(str(item) for item in blocker_ids):
        task = blocked_tasks.get(blocker_id, {})
        entries.append(
            {
                "parent_loop_action": "queued",
                "backend": "external_target_preflight_subagent",
                "spawn_kind": "target_preflight_agent",
                "task_id": blocker_id,
                "reason": "target_preflight_blocker",
                "blocking_reason": task.get("reason") or "target preflight evidence is missing or incomplete",
                "expected_artifacts": expected_artifacts.get(blocker_id, []),
                "next_action": (
                    "rerun parent-loop or inspect with --mlir-graph and/or --gptq-checkpoint so inspect artifacts "
                    "are regenerated and target_preflight.status can become passed"
                ),
                "source_replay": dispatch_plan.get("source_replay", {}),
                "does_not_claim": [
                    "preflight sub-agent has executed",
                    "full LLaMA execution",
                    "board-level ZCU104 signoff",
                ],
            }
        )
    return entries


def _entry_evidence_dir(entry: dict[str, Any], evidence_root: Path) -> Path:
    expected = entry.get("expected_evidence_dir") or f"build/{entry.get('task_id', 'unknown')}_gate"
    return evidence_root / Path(str(expected)).name


def _module_ooc_report_from_kernel_result(
    entry: dict[str, Any],
    config: AgentConfig,
    kernel_report: dict[str, Any],
) -> dict[str, Any]:
    synthesis = kernel_report.get("synthesis") if isinstance(kernel_report.get("synthesis"), dict) else {}
    resources = synthesis.get("resource_utilization", {}) if isinstance(synthesis, dict) else {}
    timing = synthesis.get("timing", {}) if isinstance(synthesis, dict) else {}
    utilization = {
        "lut": _resource_used(resources, "lut_as_logic"),
        "dsp": _resource_used(resources, "dsps"),
        "bram": _resource_used(resources, "block_ram_tile"),
        "uram": _resource_used(resources, "uram"),
        "ff": _resource_used(resources, "clb_registers"),
        "io": _resource_used(resources, "bonded_iob"),
    }
    resource_assessment = _resource_assessment(utilization, config)
    if _kernel_report_is_fixture_control_scaffold(kernel_report):
        resource_assessment = "fixture_control_scaffold"
    datapath_parallelism = _datapath_parallelism_from_kernel_report(kernel_report)
    throughput_target = _module_throughput_target(kernel_report, resource_assessment)
    tuning_recommendation = _module_ooc_tuning_recommendation(
        entry,
        config,
        utilization,
        resource_assessment,
        kernel_report,
    )
    return {
        "artifact": "module_ooc_synthesis_report",
        "status": "passed" if synthesis.get("passed") is True else "failed",
        "task_id": entry.get("task_id"),
        "vivado": {
            "part": config.hardware.fpga_part,
            "target_clock_mhz": config.hardware.target_clock_mhz,
            "implementation_stage": kernel_report.get("implementation_stage"),
            "timing_report": synthesis.get("timing_report"),
            "utilization_report": synthesis.get("utilization_report"),
        },
        "hardware_spec": _hardware_spec_identity(config),
        "timing": {
            "constraints_met": timing.get("constraints_met"),
            "setup_worst_slack_ns": timing.get("setup_worst_slack_ns"),
            "hold_worst_slack_ns": timing.get("hold_worst_slack_ns"),
            "pulse_width_worst_slack_ns": timing.get("pulse_width_worst_slack_ns"),
        },
        "utilization": utilization,
        "selected_tuning_knobs": {
            "pe_count": config.design.pe_count,
            "memory_data_width": config.hardware.memory_data_width,
            "activation_buffer": config.design.activation_buffer,
            "weight_storage": config.design.weight_storage,
        },
        "datapath_parallelism": datapath_parallelism,
        "resource_assessment": resource_assessment,
        "resource_ratios": _resource_ratios(utilization, config),
        "throughput_target_met": throughput_target["met"],
        "throughput_target_basis": throughput_target["basis"],
        "tuning_recommendation": tuning_recommendation,
        "source_kernel_report": "kernel_report.json",
    }


def _subagent_result_from_kernel_result(
    entry: dict[str, Any],
    kernel_report: dict[str, Any],
    module_ooc_path: Path,
    *,
    module_ooc_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = kernel_report.get("status", "unknown")
    simulation = kernel_report.get("simulation") if isinstance(kernel_report.get("simulation"), dict) else {}
    verilator = kernel_report.get("verilator") if isinstance(kernel_report.get("verilator"), dict) else {}
    synthesis = kernel_report.get("synthesis") if isinstance(kernel_report.get("synthesis"), dict) else None
    remaining_risks = ["bounded fixture only"]
    if synthesis is None:
        remaining_risks.append("Vivado synthesis not run in this parent loop attempt")
    if module_ooc_path.exists() is False and entry.get("requires_module_ooc_synthesis"):
        remaining_risks.append("module OOC synthesis evidence missing")
    payload = {
        "artifact": "hdl_subagent_result",
        "task_id": entry.get("task_id"),
        "status": status,
        "changed_files": list(kernel_report.get("files", [])),
        "commands_run": [
            "local parent-loop sub-agent backend invoked nl2hdl.llm_kernels.emit_kernel",
            *entry.get("required_commands", []),
        ],
        "simulation_evidence": {"passed": simulation.get("passed") is True, "source": "kernel_report.json"},
        "verilator_evidence": {"passed": verilator.get("passed") is True, "source": "kernel_report.json"},
        "vivado_timing_resource_evidence": {
            "passed": isinstance(synthesis, dict) and synthesis.get("passed") is True,
            "source": "kernel_report.json" if isinstance(synthesis, dict) else "not_run",
        },
        "module_ooc_synthesis_evidence": {
            "passed": module_ooc_path.exists(),
            "source": "module_ooc_synthesis_report.json" if module_ooc_path.exists() else "not_run",
        },
        "remaining_risks": remaining_risks,
        "parent_feedback_channel": "feedback_packet.json",
        "subagent_may_spawn_subagents": False,
    }
    if status != "passed":
        payload["skill_update_candidate"] = {
            "failing_command": "nl2hdl.llm_kernels.emit_kernel",
            "symptom": f"kernel_report status was {status}",
            "root_cause_hypothesis": "The bounded local sub-agent backend failed simulation, lint, or synthesis.",
            "prevention_rule": "Sub-agents must return complete kernel_report.json and subagent_result.json with failure detail before retry.",
            "minimal_regression_check": "Run parent-loop with --max-subagents-per-iteration 1 and assert a failed kernel includes a complete skill_update_candidate.",
        }
    elif isinstance(module_ooc_report, dict) and isinstance(module_ooc_report.get("skill_update_candidate"), dict):
        payload["skill_update_candidate"] = module_ooc_report["skill_update_candidate"]
    return payload


def _datapath_parallelism_from_kernel_report(kernel_report: dict[str, Any]) -> dict[str, Any]:
    lane_policy = kernel_report.get("lane_policy")
    if not isinstance(lane_policy, dict):
        return {
            "available": False,
            "reason": "kernel_report.lane_policy missing",
        }
    fields = [
        "requested_pe_lanes",
        "selected_target_planning_lanes",
        "effective_fixture_lanes",
        "effective_fixture_pe_lanes",
        "true_parallel_datapath_lanes",
        "max_fixture_true_parallel_lanes",
        "parallel_stage",
        "parallel_products_per_cycle",
        "pe_count_controls_true_parallel_datapath",
        "pe_count_headroom_in_fixture",
        "fixture_lanes_saturated_by_tile_cols",
        "not_merely_lane_index_scheduling",
    ]
    return {
        "available": True,
        **{field: lane_policy.get(field) for field in fields if field in lane_policy},
    }


def _module_throughput_target(kernel_report: dict[str, Any], resource_assessment: str) -> dict[str, Any]:
    if resource_assessment == "fixture_control_scaffold":
        return {
            "met": True,
            "basis": (
                "bounded fixture/control scaffold met its packet evidence target; "
                "full target datapath claims remain outside this fixture"
            ),
        }
    if resource_assessment != "underutilized":
        return {
            "met": True,
            "basis": "module is near at least one configured resource budget",
        }
    lane_policy = kernel_report.get("lane_policy")
    if isinstance(lane_policy, dict) and lane_policy.get("pe_count_headroom_in_fixture") is False:
        return {
            "met": True,
            "basis": (
                "bounded module packet uses all fixture true-parallel lanes requested by the selected PE configuration; "
                "full target projection execution remains outside this fixture"
            ),
        }
    return {
        "met": False,
        "basis": "module remains underutilized and no bounded packet throughput completion evidence was found",
    }


def _kernel_report_is_fixture_control_scaffold(kernel_report: dict[str, Any]) -> bool:
    coverage_level = kernel_report.get("coverage_level")
    if coverage_level not in {
        "rmsnorm_rope_source_path_fixture",
        "attention_kv_cache_fixture",
        "residual_mlp_fixture",
    }:
        return False
    does_not_claim = kernel_report.get("does_not_claim")
    if not isinstance(does_not_claim, list):
        return False
    return any("full" in str(item).lower() for item in does_not_claim)


def _resource_used(resources: dict[str, Any], key: str) -> int:
    value = resources.get(key)
    if isinstance(value, dict) and isinstance(value.get("used"), int):
        return int(value["used"])
    if isinstance(value, int):
        return value
    return 0


def _resource_assessment(utilization: dict[str, int], config: AgentConfig) -> str:
    ratios = [
        value
        for value in _resource_ratios(utilization, config).values()
        if isinstance(value, (int, float))
    ]
    if ratios and max(ratios) >= 0.80:
        return "near_budget"
    return "underutilized"


def _resource_ratios(utilization: dict[str, int], config: AgentConfig) -> dict[str, float | None]:
    budgets = {
        "lut": config.hardware.max_lut,
        "dsp": config.hardware.max_dsp,
        "bram": config.hardware.max_bram,
        "uram": config.hardware.max_uram,
        "ff": config.hardware.max_ff,
        "io": config.hardware.max_io,
    }
    return {
        key: (utilization.get(key, 0) / budget if isinstance(budget, int) and budget > 0 else None)
        for key, budget in budgets.items()
    }


def _module_ooc_tuning_recommendation(
    entry: dict[str, Any],
    config: AgentConfig,
    utilization: dict[str, int],
    resource_assessment: str,
    kernel_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratios = _resource_ratios(utilization, config)
    numeric_ratios = {key: value for key, value in ratios.items() if isinstance(value, (int, float))}
    bottleneck = max(numeric_ratios, key=numeric_ratios.get) if numeric_ratios else None
    lane_policy = kernel_report.get("lane_policy") if isinstance(kernel_report, dict) else None
    pe_count_headroom = lane_policy.get("pe_count_headroom_in_fixture") if isinstance(lane_policy, dict) else None
    if resource_assessment == "fixture_control_scaffold":
        reason = "bounded fixture/control scaffold OOC evidence is accepted without resource-saturation tuning"
    elif resource_assessment == "underutilized":
        reason = "module uses less than 80% of every declared resource budget"
    else:
        reason = "module is near at least one declared resource budget"
    recommendation = {
        "required": resource_assessment == "underutilized",
        "reason": reason,
        "current_knobs": {
            "pe_count": config.design.pe_count,
            "memory_data_width": config.hardware.memory_data_width,
            "activation_buffer": config.design.activation_buffer,
            "weight_storage": config.design.weight_storage,
        },
        "resource_ratios": ratios,
        "dominant_resource": bottleneck,
        "allowed_knob_updates": [
            "increase pe_count",
            "increase tile sizes within timing and I/O limits",
            "increase buffering depth when BRAM/URAM budget allows",
            "increase memory_data_width when the hardware spec allows it",
        ],
        "task_id": entry.get("task_id"),
    }
    if resource_assessment == "underutilized":
        if pe_count_headroom is False:
            recommendation["required"] = False
            recommendation["reason"] = (
                "bounded module throughput target is met for the selected PE configuration; "
                "additional resource use requires a larger tile or generator/module-packet revision"
            )
            recommendation["suggested_next_knobs"] = recommendation["current_knobs"]
            recommendation["blocked_knob_reason"] = (
                "kernel_report.lane_policy shows pe_count already reaches the fixture's true parallel lane limit"
            )
            recommendation["next_action"] = (
                "increase target tile size or revise the generator/module packet before retrying pe_count"
            )
        else:
            recommendation["suggested_next_knobs"] = {
                "pe_count": max(config.design.pe_count + 1, config.design.pe_count * 2),
                "memory_data_width": config.hardware.memory_data_width,
                "activation_buffer": config.design.activation_buffer,
                "weight_storage": config.design.weight_storage,
            }
    else:
        recommendation["suggested_next_knobs"] = recommendation["current_knobs"]
    return recommendation


def _hardware_spec_identity(config: AgentConfig) -> dict[str, Any]:
    return {
        "fpga_part": config.hardware.fpga_part,
        "target_clock_mhz": config.hardware.target_clock_mhz,
        "max_lut": config.hardware.max_lut,
        "max_dsp": config.hardware.max_dsp,
        "max_bram": config.hardware.max_bram,
        "max_ff": config.hardware.max_ff,
        "max_uram": config.hardware.max_uram,
        "max_io": config.hardware.max_io,
        "memory_data_width": config.hardware.memory_data_width,
        "device_logic_cells": config.hardware.device_logic_cells,
        "device_lut": config.hardware.device_lut,
        "device_ff": config.hardware.device_ff,
        "device_dsp": config.hardware.device_dsp,
        "device_bram_36k": config.hardware.device_bram_36k,
        "device_uram": config.hardware.device_uram,
        "device_io": config.hardware.device_io,
        "device_distributed_ram_mb": config.hardware.device_distributed_ram_mb,
        "device_bram_mb": config.hardware.device_bram_mb,
        "device_uram_mb": config.hardware.device_uram_mb,
        "device_ps_gtr": config.hardware.device_ps_gtr,
        "device_gth": config.hardware.device_gth,
        "resource_reference": config.hardware.resource_reference,
    }


@contextmanager
def _subagent_env(entry: dict[str, Any]) -> Iterator[None]:
    updates: dict[str, str] = {}
    kernel = str(entry.get("current_regression_kernel") or "")
    semantic_op = entry.get("semantic_op")
    if isinstance(semantic_op, str):
        if semantic_op in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
            updates["NL2HDL_SELECTED_PROJECTION"] = semantic_op
        elif entry.get("task_group") == "non_gemm_tasks" or "rmsnorm" in kernel or "rope" in kernel:
            updates["NL2HDL_SELECTED_NONGEMM"] = semantic_op
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_skill_update_draft(
    dispatch_plan: dict[str, Any],
    evidence_root: Path,
    status_dir: Path,
) -> dict[str, Any]:
    draft = build_hdl_subagent_skill_update_draft(
        dispatch_plan,
        evidence_root,
        target_skill="hdl-kernel-contract-gates",
    )
    json_path = status_dir / "skill_update_candidates.json"
    markdown_path = status_dir / "skill_update_draft.md"
    _write_json(json_path, draft)
    markdown_path.write_text(
        "# Parent Loop Skill Update Draft\n\n"
        f"Candidate count: `{draft.get('candidate_count', 0)}`\n\n"
        "Review the JSON candidate list, update the project skill, sync runtime skills, then rerun the parent loop.\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path), "candidate_count": draft.get("candidate_count", 0)}


def _write_parent_queue(status_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    deduped = _dedupe_queue_entries(entries)
    queue = {
        "artifact": "parent_loop_queue",
        "status": "queued_external_subagents" if deduped else "empty",
        "entry_count": len(deduped),
        "deduped_entry_count": len(entries) - len(deduped),
        "entries": deduped,
        "does_not_claim": [
            "queued sub-agents have already executed",
            "Codex Desktop sub-agents were spawned from package runtime",
        ],
    }
    path = status_dir / "parent_loop_queue.json"
    _write_json(path, queue)
    return {
        "path": str(path),
        "status": queue["status"],
        "entry_count": len(deduped),
        "deduped_entry_count": queue["deduped_entry_count"],
    }


def _queue_entry_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    missing = entry.get("missing_synthesis_task_ids")
    if isinstance(missing, list):
        missing_key = tuple(sorted(str(item) for item in missing))
    else:
        missing_key = None
    return (
        entry.get("parent_loop_action"),
        entry.get("reason"),
        entry.get("spawn_key"),
        entry.get("spawn_kind"),
        entry.get("task_id"),
        entry.get("wave_id"),
        entry.get("target_task"),
        entry.get("expected_evidence_file"),
        missing_key,
    )


def _dedupe_queue_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in entries:
        key = _queue_entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _manifest_summary(execution_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "spawn_entry_count": execution_manifest.get("spawn_entry_count", 0),
        "implementation_spawn_count": execution_manifest.get("implementation_spawn_count", 0),
        "verification_spawn_count": execution_manifest.get("verification_spawn_count", 0),
        "skill_update_required": execution_manifest.get("skill_update_required", False),
        "missing_skill_update_candidate": execution_manifest.get("missing_skill_update_candidate", False),
        "blocked_wave_count": len(execution_manifest.get("blocked_waves", [])),
    }


def _parent_loop_status(
    final_state: ParentState,
    final_reason: str,
    executed_total: int,
    queued_total: int,
) -> str:
    manifest = final_state.execution_manifest
    if final_state.board_signoff_readiness.get("status") == "passed":
        return "passed"
    if (
        final_state.full_execution_readiness.get("status") == "passed"
        and final_state.board_signoff_readiness.get("status") != "passed"
    ):
        return "blocked_board_signoff_evidence"
    if manifest.get("skill_update_required"):
        return "blocked_skill_update_required"
    if manifest.get("missing_skill_update_candidate"):
        return "blocked_missing_skill_update_candidate"
    if queued_total:
        return "queued_external_subagents"
    if manifest.get("spawn_entry_count", 0) > 0:
        return "ready_to_continue"
    if executed_total > 0 and final_reason == "max_iterations_reached":
        return "max_iterations_reached"
    if final_reason.startswith("idle"):
        return "idle_or_waiting_for_target_evidence"
    return "passed"


def _final_report(
    out_dir: Path,
    *,
    status: str,
    model_name: str,
    config: AgentConfig,
    inspect_dir: Path,
    evidence_root: Path,
    status_dir: Path,
    iterations: list[dict[str, Any]],
    queue_entries: list[dict[str, Any]],
    reason: str,
    final_state: ParentState | None = None,
    parent_queue: dict[str, Any] | None = None,
    evidence_imports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bitstream_evidence = _board_wrapper_bitstream_summary(evidence_root)
    does_not_claim = [
        "Codex Desktop multi-agent API was available inside package runtime",
        "full LLaMA execution",
        "full target-scale LLaMA accelerator bitstream",
        "hardware lab programming or on-board validation",
    ]
    if final_state is None or final_state.board_signoff_readiness.get("status") != "passed":
        does_not_claim.append("board-level ZCU104 signoff")
    if not bitstream_evidence.get("generated"):
        does_not_claim.append("board-wrapper bitstream generated")

    report = {
        "artifact": "parent_loop_run_report",
        "status": status,
        "reason": reason,
        "model": model_name,
        "target": {
            "fpga_part": config.hardware.fpga_part,
            "target_clock_mhz": config.hardware.target_clock_mhz,
            "quantization": config.optimization.quantization,
            "design_style": config.design.style,
            "compute_style": config.design.compute_style,
            "execution_style": config.design.execution_style,
            "memory_style": config.design.memory_style,
            "control_style": config.design.control_style,
        },
        "out": str(out_dir),
        "inspect_dir": str(inspect_dir),
        "evidence_root": str(evidence_root),
        "status_dir": str(status_dir),
        "iteration_count": len(iterations),
        "iterations": iterations,
        "queued_external_subagent_count": len(queue_entries),
        "parent_queue": parent_queue,
        "evidence_imports": evidence_imports or [],
        "bitstream_evidence": bitstream_evidence,
        "parent_must_not_write_hdl": True,
        "does_not_claim": does_not_claim,
    }
    if final_state is not None:
        report["final_state"] = {
            "wave_status": str(final_state.status_paths["wave_status"]),
            "execution_manifest": str(final_state.status_paths["execution_manifest"]),
            "parent_loop_state": str(final_state.status_paths["parent_loop_state"]),
            "feedback_packet": str(final_state.status_paths["feedback_packet"]),
            "retry_plan": str(final_state.status_paths["retry_plan"]),
            "next_dispatchable_waves": final_state.wave_status.get("next_dispatchable_waves", []),
            "spawn_entry_count": final_state.execution_manifest.get("spawn_entry_count", 0),
            "blocked_waves": final_state.execution_manifest.get("blocked_waves", []),
            "full_execution_readiness_status": final_state.full_execution_readiness.get("status"),
            "board_signoff_readiness_status": final_state.board_signoff_readiness.get("status"),
            "safe_to_clear_full_llama_model_execution_blocker": final_state.full_execution_readiness.get("status")
            == "passed",
            "safe_to_clear_board_level_zcu104_signoff_blocker": final_state.board_signoff_readiness.get("status")
            == "passed",
        }
    return report


def _board_wrapper_bitstream_summary(evidence_root: Path) -> dict[str, Any]:
    report_path = evidence_root / "board_zcu104_signoff_gate" / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    if not report_path.exists():
        return {
            "artifact": "parent_loop_bitstream_evidence_summary",
            "generated": False,
            "reason": "board_wrapper_report_not_found",
            "expected_board_wrapper_report": str(report_path),
            "scope": "zcu104_board_wrapper_control_scaffold",
            "not_full_llama_accelerator": True,
        }

    wrapper_report = _read_json(report_path)
    bitstream_file = wrapper_report.get("bitstream_file")
    bitstream_path = _resolve_existing_import_path(bitstream_file, report_path.parent) if bitstream_file else None
    route_analysis = wrapper_report.get("route_report_analysis")
    if not isinstance(route_analysis, dict):
        route_analysis = wrapper_report.get("route_analysis")
    if not isinstance(route_analysis, dict):
        route_analysis = {}
    generated = bool(wrapper_report.get("bitstream_generated")) and bitstream_path is not None and bitstream_path.exists()
    size = bitstream_path.stat().st_size if generated and bitstream_path is not None else wrapper_report.get("bitstream_size_bytes")
    target_scale_bitstream = wrapper_report.get("target_scale_accelerator_bitstream") is True
    scope = str(
        wrapper_report.get("accelerator_scope")
        or ("full_target_llama_accelerator" if target_scale_bitstream else "zcu104_board_wrapper_control_scaffold")
    )
    does_not_claim = [
        "hardware lab programming or on-board validation",
    ]
    if not target_scale_bitstream:
        does_not_claim.extend(
            [
                "full LLaMA execution",
                "full target-scale LLaMA accelerator bitstream",
            ]
        )
    return {
        "artifact": "parent_loop_bitstream_evidence_summary",
        "generated": generated,
        "board_wrapper_report": str(report_path),
        "board_wrapper_status": wrapper_report.get("status"),
        "bitstream_file": str(bitstream_path) if bitstream_path is not None else None,
        "bitstream_size_bytes": size,
        "route_completed": wrapper_report.get("route_completed"),
        "route_check_command_passed": wrapper_report.get("route_check_command_passed"),
        "clock": route_analysis.get("clock", {}),
        "timing": route_analysis.get("timing", {}),
        "utilization": route_analysis.get("utilization", {}),
        "gate_failures": route_analysis.get("gate_failures", []),
        "scope": scope,
        "target_scale_accelerator_bitstream": target_scale_bitstream,
        "not_full_llama_accelerator": not target_scale_bitstream,
        "does_not_claim": does_not_claim,
    }


def _select_board_wrapper_accelerator_artifact(evidence_root: Path) -> dict[str, Any] | None:
    candidate_gate_names = [
        "full_target_llama_accelerator_gate",
        "ddr_axi_board_shell_fixture_gate",
        "model_fsm_axi_decoder_block_fixture_gate",
        "token_loop_axi_decoder_block_fixture_gate",
        "top_fsm_axi_decoder_block_fixture_gate",
    ]
    for gate_name in candidate_gate_names:
        artifact_dir = evidence_root / gate_name
        kernel_report_path = artifact_dir / "kernel_report.json"
        if not kernel_report_path.exists():
            continue
        try:
            kernel_report = _read_json(kernel_report_path)
        except (json.JSONDecodeError, ValueError):
            continue
        if kernel_report.get("status") != "passed":
            continue
        top_module = str(kernel_report.get("kernel") or "").strip()
        if not top_module:
            continue
        top_rtl = artifact_dir / f"{top_module}.sv"
        if not top_rtl.exists():
            continue
        numeric_policy = kernel_report.get("numeric_policy", {})
        if not isinstance(numeric_policy, dict):
            numeric_policy = {}
        return {
            "artifact_dir": str(artifact_dir),
            "kernel_report": str(kernel_report_path),
            "top_module": top_module,
            "coverage_level": kernel_report.get("coverage_level"),
            "target_scale_eligible": (
                numeric_policy.get("full_llama_model") is True
                and numeric_policy.get("board_level_signoff") is True
                and "fixture" not in str(kernel_report.get("coverage_level") or "")
            ),
        }
    return None


def _existing_board_wrapper_report(evidence_root: Path) -> dict[str, Any] | None:
    report_path = evidence_root / "board_zcu104_signoff_gate" / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    if not report_path.exists():
        return None
    try:
        return _read_json(report_path)
    except (json.JSONDecodeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
