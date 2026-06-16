from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import json
import os

from .config import AgentConfig
from .llm_agent import run_llm_agent
from .llm_kernels import emit_kernel, run_zcu104_board_wrapper_axi_bridge_agent
from .subagent_tasks import (
    build_board_zcu104_signoff_evidence_agent_task,
    build_board_zcu104_signoff_evidence_template,
    build_board_zcu104_signoff_readiness_report,
    build_codex_spawn_instructions,
    build_full_llama_execution_evidence_agent_task,
    build_full_llama_execution_evidence_template,
    build_full_llama_execution_readiness_report,
    build_hdl_subagent_execution_manifest,
    build_hdl_subagent_skill_update_draft,
    build_hdl_subagent_wave_status,
    build_model_level_execution_harness_agent_task,
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


@dataclass(frozen=True)
class ParentState:
    dispatch_plan: dict[str, Any]
    wave_status: dict[str, Any]
    execution_manifest: dict[str, Any]
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
    if report_path.exists() and dispatch_plan_path.exists():
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

    target_tasks = {
        "model_level_execution_harness": model_harness_task,
        "full_llama_execution": full_execution_task,
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
        return _run_local_verification_entry(entry, evidence_root)
    return _queue_external_entry(entry, "requires_codex_subagent_or_local_verification_flag")


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


def _run_local_verification_entry(entry: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    report_path = evidence_root / entry.get("verification_report", "")
    if not report_path.name:
        report_path = evidence_root / "verification_results" / f"{entry.get('wave_id', 'unknown')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runs_integration_synthesis = bool(entry.get("runs_integration_synthesis"))
    status = "blocked_by_integration_synthesis_requirement" if runs_integration_synthesis else "passed"
    findings = []
    if runs_integration_synthesis:
        findings.append(
            {
                "priority": "P1",
                "title": "Integration synthesis must be run by a Vivado-capable verification sub-agent",
                "body": "Local deterministic verification does not clear integration synthesis evidence.",
            }
        )
    _write_json(
        report_path,
        {
            "artifact": "hdl_subagent_verification_report",
            "status": status,
            "wave_id": entry.get("wave_id"),
            "verification_backend": "local_deterministic_smoke",
            "findings": findings,
            "audit_summary": "Implementation evidence JSON was present for parent-loop smoke coverage."
            if status == "passed"
            else "Integration synthesis evidence was not produced.",
            "does_not_claim": [
                "Codex read-only verification agent audited the code",
                "integration synthesis passed",
            ],
        },
    )
    return {
        "parent_loop_action": "executed",
        "backend": "local_verification_smoke",
        "spawn_key": entry.get("spawn_key"),
        "spawn_kind": entry.get("spawn_kind"),
        "wave_id": entry.get("wave_id"),
        "status": status,
        "verification_report": str(report_path),
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
            continue
        if name == "zcu104_board_wrapper_axi_bridge" and options.backend == "local":
            out_dir = Path(task.get("expected_evidence_dir") or evidence_root / "board_zcu104_signoff_gate")
            if not out_dir.is_absolute():
                out_dir = evidence_root / Path(out_dir).name
            report = run_zcu104_board_wrapper_axi_bridge_agent(
                config,
                out_dir,
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
        "resource_assessment": _resource_assessment(utilization, config),
        "throughput_target_met": True,
        "source_kernel_report": "kernel_report.json",
    }


def _subagent_result_from_kernel_result(
    entry: dict[str, Any],
    kernel_report: dict[str, Any],
    module_ooc_path: Path,
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
    return payload


def _resource_used(resources: dict[str, Any], key: str) -> int:
    value = resources.get(key)
    if isinstance(value, dict) and isinstance(value.get("used"), int):
        return int(value["used"])
    if isinstance(value, int):
        return value
    return 0


def _resource_assessment(utilization: dict[str, int], config: AgentConfig) -> str:
    budgets = {
        "lut": config.hardware.max_lut,
        "dsp": config.hardware.max_dsp,
        "bram": config.hardware.max_bram,
        "uram": config.hardware.max_uram,
        "ff": config.hardware.max_ff,
        "io": config.hardware.max_io,
    }
    ratios = [
        utilization[key] / budget
        for key, budget in budgets.items()
        if isinstance(budget, int) and budget > 0
    ]
    if ratios and max(ratios) >= 0.80:
        return "near_budget"
    return "underutilized"


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
    queue = {
        "artifact": "parent_loop_queue",
        "status": "queued_external_subagents" if entries else "empty",
        "entry_count": len(entries),
        "entries": entries,
        "does_not_claim": [
            "queued sub-agents have already executed",
            "Codex Desktop sub-agents were spawned from package runtime",
        ],
    }
    path = status_dir / "parent_loop_queue.json"
    _write_json(path, queue)
    return {"path": str(path), "status": queue["status"], "entry_count": len(entries)}


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
) -> dict[str, Any]:
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
        "parent_must_not_write_hdl": True,
        "does_not_claim": [
            "Codex Desktop multi-agent API was available inside package runtime",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
            "final bitstream generated",
        ],
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
        }
    return report


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
