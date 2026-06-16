from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import shlex


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


SKILL_UPDATE_CANDIDATE_FIELDS = [
    "failing_command",
    "symptom",
    "root_cause_hypothesis",
    "prevention_rule",
    "minimal_regression_check",
]

TIMING_FIELDS = [
    "setup_worst_slack_ns",
    "hold_worst_slack_ns",
    "pulse_width_worst_slack_ns",
]

SUBAGENT_RESULT_FIELDS = [
    "changed_files",
    "commands_run",
    "simulation_evidence",
    "verilator_evidence",
    "vivado_timing_resource_evidence",
    "module_ooc_synthesis_evidence",
    "remaining_risks",
]

TARGET_EVIDENCE_RESULT_FIELDS = [
    "changed_files",
    "commands_run",
    "evidence_paths",
    "remaining_risks",
]


def _module_contract(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["task_id"]
    role = task.get("agent_role", "")
    kernel = task.get("current_regression_kernel", task_id)
    contract = task.get("contract", "docs/hdl_module_interface_contract.md")
    integration_boundary = "module_kernel"
    if "layer" in role and "fsm" in role:
        integration_boundary = "layer_fsm_calls_verified_child"
    elif "top" in role and "fsm" in role:
        integration_boundary = "top_fsm_schedules_verified_layer_fsm"
    elif "decoder" in role:
        integration_boundary = "decoder_block_composes_verified_children"
    elif "token_loop" in role:
        integration_boundary = "token_loop_schedules_verified_top_fsm"
    elif "ddr_axi_board_shell" in role:
        integration_boundary = "ddr_axi_board_shell_wraps_verified_model_fsm"
    return {
        "artifact": "hdl_module_contract_bundle",
        "task_id": task_id,
        "kernel": kernel,
        "contract_file": contract,
        "clock_reset": {
            "clock": "aclk",
            "reset": "aresetn",
            "reset_style": "synchronous_active_low",
        },
        "handshake_ports": {
            "start": "start_i",
            "done": "done_o",
        },
        "handshake_rules": [
            "start_i is sampled only when the module is idle",
            "done_o remains high until start_i is deasserted",
            "inputs remain stable while the kernel is busy",
            "outputs remain stable whenever done_o is high",
        ],
        "packed_vector_order": "little_element_order_idx_times_width_plus_width",
        "parent_boundary": {
            "parent_must_not_write_hdl": True,
            "subagent_owns_rtl_or_generator_changes": True,
            "integration_boundary": integration_boundary,
        },
        "required_artifacts": [
            "SystemVerilog module generator or concrete .sv",
            "testbench or generated testbench",
            "Python/NumPy golden vector source or report",
            "kernel_report.json",
            "module_ooc_synthesis_report.json for real datapath modules",
            "subagent_result.json",
            "timing_summary.rpt and utilization.rpt when synthesis is requested",
        ],
        "module_ooc_synthesis_gate": {
            "required_for_real_datapath_modules": True,
            "required_fields": [
                "status",
                "vivado",
                "timing",
                "utilization",
                "selected_tuning_knobs",
                "resource_assessment",
            ],
            "allowed_resource_assessments": [
                "underutilized",
                "near_budget",
                "bandwidth_limited",
                "timing_limited",
                "fixture_control_scaffold",
            ],
            "integration_requires_selected_knobs": True,
        },
        "hardware_spec_identity": task.get("hardware", {}),
        "final_response_required_fields": SUBAGENT_RESULT_FIELDS,
        "failure_to_skill_required": True,
    }


def _skill_update_candidate_template(task_id: str | None = None) -> dict[str, Any]:
    return {
        "artifact": "skill_update_candidate_template",
        "required_before_retry": True,
        "use_when": "an HDL implementation sub-agent cannot pass its assigned simulation, lint, synthesis, timing, or evidence gate",
        "required_fields": SKILL_UPDATE_CANDIDATE_FIELDS,
        "candidate": {
            "task_id": task_id or "<task_id>",
            "failing_command": "<exact command that failed>",
            "symptom": "<concise simulator/lint/Vivado/test symptom with log path>",
            "root_cause_hypothesis": "<why this failure likely happened>",
            "prevention_rule": "<reusable rule to add to a SKILL before retry>",
            "minimal_regression_check": "<smallest test/sim/synth check that catches this failure>",
            "evidence_dir": f"build/{_slug(task_id)}_gate" if task_id else "build/<task_id>_gate",
            "suggested_skill": "fpga-vivado-systemverilog or hdl-kernel-contract-gates or hdl-vivado-timing-closure",
        },
    }


def _source_replay_cli_args(task: dict[str, Any]) -> str:
    source_replay = task.get("source_replay", {})
    args = []
    gptq_checkpoint = source_replay.get("gptq_checkpoint")
    if gptq_checkpoint:
        args.extend(["--gptq-checkpoint", str(gptq_checkpoint)])
    mlir_graph = source_replay.get("mlir_graph")
    if mlir_graph:
        args.extend(["--mlir-graph", str(mlir_graph)])
    return " ".join(shlex.quote(arg) for arg in args)


def _required_commands(task: dict[str, Any]) -> list[str]:
    kernel = task.get("current_regression_kernel")
    model_name = shlex.quote(str(task.get("source_replay", {}).get("model_name") or "meta-llama/Llama-3.2-1B"))
    source_args = _source_replay_cli_args(task)
    source_args_text = f" {source_args}" if source_args else ""
    if not kernel:
        return [
            "python3 -m pytest -q",
            (
                f"python3 -m nl2hdl agent --model {model_name} "
                f"--spec examples/zcu104_llama32_1b_gptq.yaml{source_args_text} "
                "--mode inspect --out build/inspect_verify --verbose"
            ),
        ]
    task_prefix = ""
    projection_ops = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (
        task.get("semantic_op") in projection_ops
        and isinstance(kernel, str)
        and kernel.startswith("projection_")
    ):
        task_prefix = f"NL2HDL_SELECTED_PROJECTION={task['semantic_op']} "
    elif task.get("task_group") == "non_gemm_tasks" and task.get("semantic_op"):
        task_prefix = f"NL2HDL_SELECTED_NONGEMM={task['semantic_op']} "
    return [
        "python3 -m pytest -q tests/test_llm_kernels.py -q",
        (
            f"{task_prefix}python3 -m nl2hdl agent --model {model_name} "
            f"--spec examples/zcu104_llama32_1b_gptq.yaml{source_args_text} --mode kernel --kernel {kernel} "
            f"--out build/{_slug(task['task_id'])}_gate --verbose"
        ),
    ]


def _allowed_write_scope(task: dict[str, Any]) -> list[str]:
    task_id = _slug(task["task_id"])
    contract = task.get("contract", "docs/hdl_module_interface_contract.md")
    kernel = task.get("current_regression_kernel", task["task_id"])
    return [
        "Do not edit unrelated files.",
        f"Assigned generator/source scope: nl2hdl/llm_kernels.py changes only for kernel `{kernel}`.",
        "Assigned test scope: add or update only task-specific assertions in tests/test_llm_kernels.py.",
        f"Assigned contract scope: read `{contract}`; edit it only if the task explicitly requires contract clarification.",
        f"Generated evidence should go under build/{task_id}_gate/.",
        "Do not edit parent orchestration files such as nl2hdl/llm_agent.py, nl2hdl/cli.py, nl2hdl/subagent_tasks.py, or manifest/report generators.",
        "Do not weaken existing tests, contracts, timing gates, or forbidden-claim language.",
        "Verification agents are read-only unless explicitly reassigned as implementation agents.",
    ]


def _task_with_group(task: dict[str, Any], group: str) -> dict[str, Any]:
    return {**task, "task_group": group}


def _checkpoint_blocked_target_dependencies(blocked_target_tasks: list[dict[str, Any]]) -> list[str]:
    dependency_task_ids = {
        "real_gptq_checkpoint_source",
        "real_gptq_checkpoint_metadata",
    }
    return sorted(
        {
            task["task_id"]
            for task in blocked_target_tasks
            if task.get("task_id") in dependency_task_ids
        }
    )


def _global_blocked_target_dependencies(blocked_target_tasks: list[dict[str, Any]]) -> list[str]:
    dependency_task_ids = {
        "real_mlir_model_analysis",
        "real_gptq_checkpoint_source",
        "real_gptq_checkpoint_metadata",
    }
    return sorted(
        {
            task["task_id"]
            for task in blocked_target_tasks
            if task.get("task_id") in dependency_task_ids
        }
    )


def _blocked_task_line(blocked: dict[str, Any]) -> str:
    details = []
    if blocked.get("classification") is not None:
        details.append(f"classification `{blocked['classification']}`")
    if blocked.get("checkpoint_quantization_dependency") is not None:
        details.append(f"dependency `{blocked['checkpoint_quantization_dependency']}`")
    if blocked.get("analysis_source") is not None:
        details.append(f"analysis source `{blocked['analysis_source']}`")
    if blocked.get("coverage_level") is not None:
        details.append(f"coverage `{blocked['coverage_level']}`")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"- `{blocked['task_id']}`: {blocked['reason']}{suffix}"


def _prompt_for_task(task: dict[str, Any], manifest: dict[str, Any]) -> str:
    task_id = task["task_id"]
    role = task["agent_role"]
    contract = task.get("contract", "docs/hdl_module_interface_contract.md")
    required_evidence = task.get("required_evidence", [])
    does_not_claim = task.get("does_not_claim", [])
    commands = _required_commands(task)
    prompt = [
        f"# HDL Sub-Agent Task: {task_id}",
        "",
        f"Role: `{role}`.",
        "",
        "You are one HDL implementation sub-agent for exactly this packet. The parent agent coordinates only; you own the assigned RTL/generator work and must verify it before reporting success.",
        "Do not wait for the parent to write RTL. If this packet needs RTL or generator changes, make those changes inside the allowed scope and run the required checks yourself.",
        "",
        "## Target Context",
        "",
        f"- Model: `{manifest['model']['name']}`",
        f"- Replay model name: `{manifest.get('source_replay', {}).get('model_name') or manifest['model']['name']}`",
        f"- FPGA part: `{manifest['hardware']['fpga_part']}`",
        f"- Target clock: `{manifest['hardware']['target_clock_mhz']} MHz`",
        f"- Quantization: `{manifest['optimization']['quantization']}`",
        f"- Optimization brief: `{manifest['optimization'].get('optimization_brief') or 'not_configured'}`",
        f"- Design style alias: `{manifest['optimization']['design_style']}`",
        f"- Compute style: `{manifest['optimization'].get('compute_style', 'not_configured')}`",
        f"- Execution style: `{manifest['optimization'].get('execution_style', 'not_configured')}`",
        f"- Memory style: `{manifest['optimization'].get('memory_style', 'not_configured')}`",
        f"- Control style: `{manifest['optimization'].get('control_style', 'not_configured')}`",
        f"- Architecture brief: `{manifest['optimization'].get('architecture_brief') or 'not_configured'}`",
        f"- Optimization candidates: `{manifest['optimization'].get('optimization_candidates') or []}`",
        f"- Design candidates: `{manifest['optimization'].get('design_candidates') or []}`",
        f"- Contract: `{contract}`",
        f"- Replay GPTQ checkpoint override: `{manifest.get('source_replay', {}).get('gptq_checkpoint') or 'not_configured'}`",
        f"- Replay MLIR graph override: `{manifest.get('source_replay', {}).get('mlir_graph') or 'not_configured'}`",
        "",
        "## Assigned Operation",
        "",
        f"- Task id: `{task_id}`",
        f"- Partition: `{task.get('partition', 'integration')}`",
        f"- Semantic op: `{task.get('semantic_op', task_id)}`",
        f"- Regression kernel: `{task.get('current_regression_kernel', 'not_assigned')}`",
        f"- Status before assignment: `{task.get('status', 'unknown')}`",
    ]
    if "rows" in task and "cols" in task:
        prompt.extend(
            [
                f"- Projection shape: rows `{task['rows']}`, cols `{task['cols']}`",
                f"- Packed INT4 bytes: `{task.get('packed_int4_bytes')}`",
                f"- Memory beats: `{task.get('memory_beats')}`",
            ]
        )
    if "gptq_weight_layout_preflight" in task:
        preflight = task["gptq_weight_layout_preflight"]
        prompt.extend(
            [
                f"- GPTQ layout preflight: `{preflight['status']}`",
                f"- Target layout compatible: `{preflight['target_layout_compatible']}`",
                f"- Expected qweight packed bytes: `{preflight['expected_qweight_int4_packed_bytes']}`",
                f"- Observed qweight byte count: `{preflight['observed_qweight_byte_count']}`",
                f"- Target checkpoint layout dependency: `{task.get('target_checkpoint_layout_dependency')}`",
            ]
        )
    if "target_weight_stream_plan" in task:
        stream_plan = task["target_weight_stream_plan"]
        prompt.extend(
            [
                f"- Target qweight shard file: `{stream_plan.get('qweight_file')}`",
                f"- Target qweight tensor key: `{stream_plan.get('qweight_key')}`",
                f"- Target qweight byte offset: `{stream_plan.get('qweight_byte_offset')}`",
                f"- Target qweight byte count: `{stream_plan.get('qweight_byte_count')}`",
                f"- Target request byte address: `{stream_plan.get('request_byte_addr')}`",
                f"- Target request beat count: `{stream_plan.get('request_beat_count')}`",
                f"- Target first-beat byte offset: `{stream_plan.get('first_beat_byte_offset')}`",
                f"- Target last-beat valid bytes: `{stream_plan.get('last_beat_valid_bytes')}`",
                f"- Target request covers unaligned qweight range: `{stream_plan.get('request_covers_unaligned_qweight_range')}`",
                f"- Target stream plan valid: `{stream_plan.get('stream_plan_valid')}`",
            ]
        )
    if "gptq_payload_probe" in task:
        payload_probe = task["gptq_payload_probe"]
        stream_probe = payload_probe.get("qweight_stream_probe", {})
        prompt.extend(
            [
                f"- GPTQ payload probe status: `{payload_probe.get('status')}`",
                f"- GPTQ payload probe source: `{payload_probe.get('payload_golden_source')}`",
                f"- GPTQ payload selected projection: `{payload_probe.get('selected_projection')}`",
                f"- GPTQ qweight payload order: `{payload_probe.get('qweight_payload_order')}`",
                f"- GPTQ qweight sample bytes: `{payload_probe.get('qweight_sample_byte_count')}`",
                f"- GPTQ qweight payload word count: `{payload_probe.get('qweight_payload_word_count')}`",
                f"- GPTQ qweight payload words32 LE: `{payload_probe.get('qweight_payload_words32_le_hex')}`",
                f"- GPTQ qweight first memory beats 128b LE: `{stream_probe.get('first_memory_beats_128b_le_hex')}`",
                f"- GPTQ qweight memory beat word chunks32 LE: `{stream_probe.get('memory_beat_word_chunks32_le_hex')}`",
                f"- GPTQ qweight covers first memory beat: `{stream_probe.get('covers_first_memory_beat')}`",
                f"- Target checkpoint payload dependency: `{task.get('target_checkpoint_payload_dependency')}`",
            ]
        )
    if "attention_projection_stream_tasks" in task:
        prompt.extend(
            [
                f"- Attention projection order: `{task.get('attention_projection_order', [])}`",
                f"- Attention projection stream tasks: `{task['attention_projection_stream_tasks']}`",
                f"- Aggregate attention layout dependency: `{task.get('target_checkpoint_layout_dependency')}`",
                f"- Aggregate attention payload dependency: `{task.get('target_checkpoint_payload_dependency')}`",
            ]
        )
        for projection_name, stream in task.get("target_attention_projection_streams", {}).items():
            prompt.extend(
                [
                    f"- `{projection_name}` stream task: `{stream.get('stream_task_id')}`",
                    f"- `{projection_name}` layout dependency: `{stream.get('target_checkpoint_layout_dependency')}`",
                    f"- `{projection_name}` payload dependency: `{stream.get('target_checkpoint_payload_dependency')}`",
                ]
            )
    if "projection_weight_stream_plan" in task:
        stream_plan = task["projection_weight_stream_plan"]
        prompt.extend(
            [
                f"- Projection stream plan artifact: `{stream_plan.get('artifact')}`",
                f"- Projection stream plan count: `{stream_plan.get('projection_count')}`",
                f"- Target stream plan valid count: `{stream_plan.get('target_stream_plan_valid_count')}`",
                f"- Payload satisfied projection count: `{stream_plan.get('payload_satisfied_projection_count')}`",
                f"- All projection layout dependency: `{stream_plan.get('all_projection_layout_dependency')}`",
                f"- All projection payload dependency: `{stream_plan.get('all_projection_payload_dependency')}`",
                f"- Target-scale ready for all projection streaming: `{stream_plan.get('target_scale_ready_for_all_projection_streaming')}`",
            ]
        )
    if "child_tasks" in task:
        prompt.append(f"- Child tasks: `{', '.join(task['child_tasks'])}`")
    module_contract = _module_contract(task)
    blocked_target_tasks = manifest.get("blocked_target_tasks", [])
    if blocked_target_tasks:
        prompt.extend(
            [
                "",
                "## Current Target Gate Blocks",
                "",
            ]
        )
        for blocked in blocked_target_tasks:
            prompt.append(_blocked_task_line(blocked))
        prompt.extend(
            [
                "",
                "These blocks apply to target-scale claims even when this packet can still improve bounded fixture RTL.",
            ]
        )
    prompt.extend(
        [
            "",
            "## Required Interface",
            "",
            "- Follow `docs/hdl_module_interface_contract.md`.",
            "- Include common handshake ports: `aclk`, `aresetn`, `start_i`, and `done_o`.",
            "- Outputs must be stable when `done_o` is asserted.",
            "- Packed vectors use little element order: element `idx` is `[idx*WIDTH +: WIDTH]`.",
            "",
            "## Machine-Readable Module Contract",
            "",
            f"- Contract bundle artifact: `{module_contract['artifact']}`",
            f"- Clock/reset: `{module_contract['clock_reset']}`",
            f"- Handshake ports: `{module_contract['handshake_ports']}`",
            f"- Parent boundary: `{module_contract['parent_boundary']}`",
            f"- Final response required fields: `{', '.join(module_contract['final_response_required_fields'])}`",
            "",
            "## Allowed Write Scope",
            "",
        ]
    )
    prompt.extend(f"- {item}" for item in _allowed_write_scope(task))
    prompt.extend(
        [
            "",
            "## Required Evidence",
            "",
        ]
    )
    prompt.extend(f"- {item}" for item in required_evidence)
    prompt.extend(
        [
            "- `kernel_report.json` must state coverage level and implementation stage.",
            "- Real datapath modules must also write `module_ooc_synthesis_report.json` before any integration wave can consume them.",
            "- `module_ooc_synthesis_report.json` must include Vivado part/clock, setup/hold/pulse-width status, LUT/DSP/BRAM/URAM/FF/I/O utilization, selected tuning knobs, and resource assessment.",
            "- If this packet is only a fixture/control scaffold, state the fixture-only OOC waiver explicitly in `kernel_report.json`.",
            "- If Vivado runs, setup, hold, and pulse-width timing must all have non-negative slack and zero failing endpoints.",
            "- Final response must list changed files, commands run, simulation evidence, timing/resource evidence, module OOC synthesis evidence, selected knobs, and remaining risks.",
            "- Also write `subagent_result.json` in your evidence directory with changed files, commands run, simulation evidence, Verilator evidence, Vivado timing/resource evidence, module OOC synthesis evidence, remaining risks, and any `skill_update_candidate`.",
            "- If you cannot pass the gate, preserve the failing evidence and return a reusable `skill_update_candidate` with failing command, symptom, root-cause hypothesis, prevention rule, and minimal regression check.",
            "",
            "## Failure-To-SKILL Candidate",
            "",
            "- If this gate fails, do not hide the failure and do not retry the same pattern blindly.",
            "- Return a `skill_update_candidate` containing these required fields:",
        ]
    )
    prompt.extend(f"  - `{field}`" for field in SKILL_UPDATE_CANDIDATE_FIELDS)
    prompt.extend(
        [
            f"- Evidence directory for this candidate: `build/{_slug(task_id)}_gate/`",
            "- The parent will convert reusable prevention rules into a SKILL before retrying.",
            "",
            "## Do Not Claim",
            "",
        ]
    )
    if does_not_claim:
        prompt.extend(f"- {item}" for item in does_not_claim)
    if "gptq_weight_layout_preflight" in task and task["gptq_weight_layout_preflight"][
        "requires_real_checkpoint_layout_before_target_claim"
    ]:
        prompt.append(
            "- Real checkpoint projection layout compatibility; this packet may improve bounded fixtures only until `real_gptq_weight_layout_preflight` passes."
        )
    if task.get("target_checkpoint_payload_dependency") == "blocked_by_gptq_payload_probe":
        prompt.append(
            "- Real checkpoint payload streaming; use synthetic or bounded fixture payloads only until `gptq_payload_probe.json` reports sampled qweight/qzeros/scales prefixes."
        )
    prompt.extend(
        [
            "- Full LLaMA execution unless this exact task proves it.",
            "- Board-level ZCU104 signoff unless board I/O, DDR/AXI, and PS/PL constraints are included.",
            "",
            "## Required Commands",
            "",
        ]
    )
    prompt.extend(f"- `{command}`" for command in commands)
    return "\n".join(prompt) + "\n"


def _agent_topology(manifest: dict[str, Any], task_count: int | None = None) -> dict[str, Any]:
    projection_count = len(manifest.get("projection_tasks", []))
    non_gemm_count = len(manifest.get("non_gemm_tasks", []))
    integration_roles = list(dict.fromkeys(task["agent_role"] for task in manifest.get("integration_tasks", [])))
    return {
        "parent_agent": {
            "owns": [
                "model and semantic inspection",
                "module interface contracts",
                "dispatch plan generation",
                "independent verification",
                "Skill updates after failed reusable HDL attempts",
            ],
            "must_not": [
                "hand-write Verilog/SystemVerilog kernels",
                "weaken generated evidence requirements",
                "advance dependent waves before read-only verification passes",
            ],
        },
        "implementation_agent_granularity": "one_subagent_per_hdl_packet",
        "parallel_module_agents": {
            "gemm_kernel_agents": projection_count,
            "non_gemm_kernel_agents": non_gemm_count,
            "can_run_in_parallel": True,
            "must_self_verify": True,
        },
        "integration_agents": [
            {
                "agent_role": "decoder_block_agent",
                "owns": "compose passed module fixtures into one decoder-block fixture",
            },
            {
                "agent_role": "layer_fsm_agent",
                "owns": "call a verified decoder-block child inside one layer/block FSM",
            },
            {
                "agent_role": "top_fsm_agent",
                "owns": "schedule verified Layer FSM calls and top-level token/control state",
            },
            {
                "agent_role": "token_loop_agent",
                "owns": "extend Top FSM evidence to bounded prefill/decode token-loop fixtures",
            },
            {
                "agent_role": "token_loop_axi_agent",
                "owns": "extend AXI-aware Top FSM evidence to bounded token-loop fixtures without owning DDR or full model scheduling",
            },
            {
                "agent_role": "memory_command_adapter_agent",
                "owns": "adapt verified projection weight-stream requests into bounded AXI read-command fixtures",
            },
            {
                "agent_role": "memory_read_data_adapter_agent",
                "owns": "consume bounded AXI read-data channel beats and bridge them toward packed projection payload streams",
            },
            {
                "agent_role": "memory_read_transaction_agent",
                "owns": "compose bounded AXI read-address and read-data channel fixtures into one read transaction payload stream",
            },
            {
                "agent_role": "memory_projection_stream_agent",
                "owns": "connect a verified bounded AXI read transaction stream to a projection-style payload consumer fixture",
            },
            {
                "agent_role": "decoder_axi_child_agent",
                "owns": "compose source-path, bounded AXI projection stream, and attention/KV fixtures into one decoder child datapath",
            },
            {
                "agent_role": "layer_axi_fsm_agent",
                "owns": "call the verified AXI-aware decoder child inside one Layer FSM fixture without owning Top FSM scheduling",
            },
            {
                "agent_role": "top_axi_fsm_agent",
                "owns": "schedule verified AXI-aware Layer FSM calls at top level without owning token-loop or DDR policy",
            },
            {
                "agent_role": "decoder_axi_block_agent",
                "owns": "compose verified AXI-aware attention child and residual/MLP fixtures into one decoder-block fixture",
            },
            {
                "agent_role": "layer_axi_decoder_block_agent",
                "owns": "call a verified AXI decoder-block child inside one Layer FSM fixture",
            },
            {
                "agent_role": "top_axi_decoder_block_agent",
                "owns": "schedule verified AXI decoder-block Layer FSM calls without owning token-loop or DDR policy",
            },
            {
                "agent_role": "token_loop_axi_decoder_block_agent",
                "owns": "extend AXI decoder-block Top FSM evidence to bounded token-loop fixtures without claiming full LLaMA decode",
            },
            {
                "agent_role": "model_axi_decoder_block_agent",
                "owns": "schedule verified AXI decoder-block token-loop children at model level without claiming target-scale LLaMA execution",
            },
            {
                "agent_role": "ddr_axi_board_shell_agent",
                "owns": "wrap the verified model FSM child with a bounded ZCU104 DDR/AXI shell fixture without claiming PS/PL or board-level signoff",
            },
        ],
        "integration_roles_present": integration_roles,
        "verification_agents": {
            "after_each_wave": True,
            "default_mode": "read_only",
            "integration_wave_mode": "integration_verification_with_synthesis",
            "source_edit_policy": "no_source_or_rtl_edits",
            "audit": [
                "requirement coverage",
                "common handshake and packed-vector contract",
                "simulation, Verilator, and Vivado evidence",
                "integration-level Vivado synthesis evidence for integration waves",
                "unsafe target-scale or board-level claims",
            ],
        },
        "failure_to_skill": {
            "required_before_retry": True,
            "skill_payload_fields": [
                "failing_command",
                "symptom",
                "root_cause_hypothesis",
                "prevention_rule",
                "minimal_regression_check",
            ],
        },
        "task_count": task_count,
    }


def build_hdl_subagent_packets(manifest: dict[str, Any]) -> dict[str, Any]:
    task_groups = ("projection_tasks", "non_gemm_tasks", "integration_tasks")
    packets = []
    for group in task_groups:
        for task in manifest.get(group, []):
            grouped_task = {
                **_task_with_group(task, group),
                "source_replay": manifest.get("source_replay", {}),
                "hardware": manifest.get("hardware", {}),
            }
            packets.append(
                {
                    "task_id": task["task_id"],
                    "agent_role": task["agent_role"],
                    "task_group": group,
                    "partition": task.get("partition"),
                    "semantic_op": task.get("semantic_op"),
                    "rows": task.get("rows"),
                    "cols": task.get("cols"),
                    "packed_int4_bytes": task.get("packed_int4_bytes"),
                    "memory_beats": task.get("memory_beats"),
                    "contract": task.get("contract"),
                    "current_regression_kernel": task.get("current_regression_kernel"),
                    "source_replay": manifest.get("source_replay", {}),
                    "hardware": manifest.get("hardware", {}),
                    "allowed_write_scope": _allowed_write_scope(grouped_task),
                    "required_commands": _required_commands(grouped_task),
                    "target_checkpoint_layout_dependency": task.get("target_checkpoint_layout_dependency"),
                    "target_checkpoint_payload_dependency": task.get("target_checkpoint_payload_dependency"),
                    "gptq_weight_layout_preflight": task.get("gptq_weight_layout_preflight"),
                    "gptq_payload_probe": task.get("gptq_payload_probe"),
                    "target_weight_stream_plan": task.get("target_weight_stream_plan"),
                    "projection_weight_stream_plan": task.get("projection_weight_stream_plan"),
                    "attention_projection_stream_tasks": task.get("attention_projection_stream_tasks"),
                    "target_attention_projection_streams": task.get("target_attention_projection_streams"),
                    "module_contract": _module_contract(grouped_task),
                    "requires_module_ooc_synthesis": group in {"projection_tasks", "non_gemm_tasks"},
                    "prompt_file": f"subagent_prompts/{_slug(task['task_id'])}__implementation.md",
                    "prompt": _prompt_for_task(grouped_task, manifest),
                }
            )
    return {
        "artifact": "hdl_subagent_tasks",
        "coverage_level": "manifest_to_subagent_prompt_packets",
        "model": manifest["model"],
        "hardware": manifest["hardware"],
        "optimization": manifest["optimization"],
        "input_clarification": manifest.get("input_clarification", {}),
        "source_replay": manifest.get("source_replay", {}),
        "task_count": len(packets),
        "agent_topology": _agent_topology(manifest, len(packets)),
        "subagent_policy": manifest["subagent_policy"],
        "failure_to_skill": {
            "required_before_retry": True,
            "skill_update_candidate_template": "skill_update_candidate_template.json",
            "required_fields": SKILL_UPDATE_CANDIDATE_FIELDS,
        },
        "blocked_target_tasks": manifest["blocked_target_tasks"],
        "global_checkpoint_blocked_target_dependencies": _checkpoint_blocked_target_dependencies(
            manifest["blocked_target_tasks"]
        ),
        "global_blocked_target_dependencies": _global_blocked_target_dependencies(manifest["blocked_target_tasks"]),
        "packets": packets,
    }


def _packet_index(packets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {packet["task_id"]: packet for packet in packets["packets"]}


def _dispatch_wave(
    wave_id: str,
    description: str,
    task_ids: list[str],
    packets_by_id: dict[str, dict[str, Any]],
    depends_on: list[str] | None = None,
    inherited_blocked_target_dependencies: list[str] | None = None,
) -> dict[str, Any]:
    implementation_tasks = []
    blocked_target_dependencies = []
    requires_integration_synthesis_verification = False
    for task_id in task_ids:
        packet = packets_by_id[task_id]
        task_group = packet.get("task_group")
        if task_group == "integration_tasks":
            requires_integration_synthesis_verification = True
        entry = {
            "task_id": task_id,
            "agent_role": packet["agent_role"],
            "task_group": task_group,
            "prompt_file": packet["prompt_file"],
            "current_regression_kernel": packet["current_regression_kernel"],
            "source_replay": packet.get("source_replay", {}),
            "hardware": packet.get("hardware", {}),
            "required_commands": packet.get("required_commands", []),
            "expected_evidence_dir": f"build/{_slug(task_id)}_gate",
            "expected_kernel_report": f"build/{_slug(task_id)}_gate/kernel_report.json",
            "expected_module_ooc_synthesis_report": f"build/{_slug(task_id)}_gate/module_ooc_synthesis_report.json",
            "requires_module_ooc_synthesis": bool(packet.get("requires_module_ooc_synthesis")),
            "module_contract": packet["module_contract"],
        }
        if packet.get("semantic_op") is not None:
            entry["semantic_op"] = packet["semantic_op"]
        if packet.get("partition") is not None:
            entry["partition"] = packet["partition"]
        if packet.get("rows") is not None and packet.get("cols") is not None:
            entry["expected_projection_shape"] = {
                "rows": packet["rows"],
                "cols": packet["cols"],
            }
            entry["packed_int4_bytes"] = packet.get("packed_int4_bytes")
            entry["memory_beats"] = packet.get("memory_beats")
        dependency = packet.get("target_checkpoint_layout_dependency")
        if dependency is not None:
            entry["target_checkpoint_layout_dependency"] = dependency
            if dependency.startswith("blocked_by_"):
                blocked_target_dependencies.append(dependency.removeprefix("blocked_by_"))
        payload_dependency = packet.get("target_checkpoint_payload_dependency")
        if payload_dependency is not None:
            entry["target_checkpoint_payload_dependency"] = payload_dependency
            if payload_dependency.startswith("blocked_by_"):
                blocked_target_dependencies.append(payload_dependency.removeprefix("blocked_by_"))
        implementation_tasks.append(entry)
    direct_blocked_target_dependencies = sorted(set(blocked_target_dependencies))
    inherited_blocked_target_dependencies = sorted(set(inherited_blocked_target_dependencies or []))
    blocked_target_dependencies = sorted(
        set(direct_blocked_target_dependencies + inherited_blocked_target_dependencies)
    )
    source_replay = next(
        (task.get("source_replay", {}) for task in implementation_tasks if task.get("source_replay")),
        {},
    )
    verification_audit_scope = [
        "requirement coverage",
        "generated RTL correctness evidence",
        "simulation and Verilator evidence",
        "Vivado setup/hold/pulse-width timing evidence when synthesis ran",
        "missing tests or unsafe claims",
    ]
    expected_integration_synthesis_dir = None
    expected_integration_synthesis_report = None
    if requires_integration_synthesis_verification:
        expected_integration_synthesis_dir = f"build/{_slug(wave_id)}_integration_verification"
        expected_integration_synthesis_report = (
            f"{expected_integration_synthesis_dir}/integration_synthesis_report.json"
        )
        verification_audit_scope.extend(
            [
                "run or inspect integration-level Vivado synthesis for the composed integration top",
                "confirm integration-level utilization includes child modules, FSM, adapters, and interconnect buffers",
                "confirm integration-level timing/resource evidence matches the active hardware spec and selected child knobs",
            ]
        )
    return {
        "wave_id": wave_id,
        "description": description,
        "source_replay": source_replay,
        "parallel_dispatch_allowed": len(task_ids) > 1,
        "depends_on_waves": depends_on or [],
        "target_scope": "bounded_fixture_only" if blocked_target_dependencies else "target_preflight_satisfied_or_not_applicable",
        "blocked_target_dependencies": blocked_target_dependencies,
        "direct_blocked_target_dependencies": direct_blocked_target_dependencies,
        "inherited_blocked_target_dependencies": inherited_blocked_target_dependencies,
        "implementation_tasks": implementation_tasks,
        "verification_agent": {
            "required": True,
            "mode": "integration_verification_with_synthesis"
            if requires_integration_synthesis_verification
            else "read_only",
            "agent": "Codex",
            "prompt_file": f"verification_prompts/{_slug(wave_id)}__verification.md",
            "source_edit_policy": "no_source_or_rtl_edits",
            "runs_integration_synthesis": requires_integration_synthesis_verification,
            "expected_integration_synthesis_dir": expected_integration_synthesis_dir,
            "expected_integration_synthesis_report": expected_integration_synthesis_report,
            "audit_scope": verification_audit_scope,
        },
    }


def _apply_global_blocked_target_dependencies(
    waves: list[dict[str, Any]],
    global_dependencies: list[str],
) -> list[dict[str, Any]]:
    global_dependencies = sorted(set(global_dependencies))
    if not global_dependencies:
        for wave in waves:
            wave["global_blocked_target_dependencies"] = []
        return waves
    for wave in waves:
        wave["global_blocked_target_dependencies"] = global_dependencies
        wave["blocked_target_dependencies"] = sorted(
            set(wave["blocked_target_dependencies"] + global_dependencies)
        )
        wave["target_scope"] = "bounded_fixture_only"
    return waves


def _verification_prompt_for_wave(wave: dict[str, Any], dispatch_plan: dict[str, Any]) -> str:
    runs_integration_synthesis = bool(wave["verification_agent"].get("runs_integration_synthesis"))
    prompt_title = "Codex Integration Verification" if runs_integration_synthesis else "Codex Read-Only Verification"
    prompt = [
        f"# {prompt_title}: {wave['wave_id']}",
        "",
        "You are the Codex verification sub-agent for this dispatch wave. Audit only; do not edit source, RTL, tests, or contracts.",
        "",
        "## Target Context",
        "",
        f"- Model: `{dispatch_plan['model']['name']}`",
        f"- Replay model name: `{dispatch_plan.get('source_replay', {}).get('model_name') or dispatch_plan['model']['name']}`",
        f"- FPGA part: `{dispatch_plan['hardware']['fpga_part']}`",
        f"- Target clock: `{dispatch_plan['hardware']['target_clock_mhz']} MHz`",
        f"- Quantization: `{dispatch_plan['optimization']['quantization']}`",
        f"- Optimization brief: `{dispatch_plan['optimization'].get('optimization_brief') or 'not_configured'}`",
        f"- Design style alias: `{dispatch_plan['optimization']['design_style']}`",
        f"- Compute style: `{dispatch_plan['optimization'].get('compute_style', 'not_configured')}`",
        f"- Execution style: `{dispatch_plan['optimization'].get('execution_style', 'not_configured')}`",
        f"- Memory style: `{dispatch_plan['optimization'].get('memory_style', 'not_configured')}`",
        f"- Control style: `{dispatch_plan['optimization'].get('control_style', 'not_configured')}`",
        f"- Architecture brief: `{dispatch_plan['optimization'].get('architecture_brief') or 'not_configured'}`",
        f"- Optimization candidates: `{dispatch_plan['optimization'].get('optimization_candidates') or []}`",
        f"- Design candidates: `{dispatch_plan['optimization'].get('design_candidates') or []}`",
        f"- Replay GPTQ checkpoint override: `{dispatch_plan.get('source_replay', {}).get('gptq_checkpoint') or 'not_configured'}`",
        f"- Replay MLIR graph override: `{dispatch_plan.get('source_replay', {}).get('mlir_graph') or 'not_configured'}`",
        "",
        "## Wave Scope",
        "",
        f"- Wave id: `{wave['wave_id']}`",
        f"- Description: {wave['description']}",
        f"- Target scope: `{wave['target_scope']}`",
        f"- Verification mode: `{wave['verification_agent']['mode']}`",
        f"- Direct blocked target dependencies: `{', '.join(wave['direct_blocked_target_dependencies']) or 'none'}`",
        f"- Inherited blocked target dependencies: `{', '.join(wave['inherited_blocked_target_dependencies']) or 'none'}`",
        f"- Global blocked target dependencies: `{', '.join(wave.get('global_blocked_target_dependencies', [])) or 'none'}`",
        "",
        "## Implementation Tasks To Audit",
        "",
    ]
    for task in wave["implementation_tasks"]:
        task_line = (
            f"- `{task['task_id']}`: role `{task['agent_role']}`, "
            f"prompt `{task['prompt_file']}`, kernel `{task['current_regression_kernel']}`"
        )
        if task.get("semantic_op"):
            task_line += f", semantic op `{task['semantic_op']}`"
        if task.get("expected_projection_shape"):
            shape = task["expected_projection_shape"]
            task_line += f", expected projection shape `{shape['rows']} x {shape['cols']}`"
        prompt.append(task_line)
    prompt.extend(
        [
            "",
            "## Audit Requirements",
            "",
        ]
    )
    prompt.extend(f"- {item}" for item in wave["verification_agent"]["audit_scope"])
    if runs_integration_synthesis:
        prompt.extend(
            [
                "",
                "## Integration-Level Synthesis Requirement",
                "",
                "- This is an integration wave, so verification must run or inspect Vivado synthesis for the composed integration top after the implementation agent passes simulation.",
                f"- Write integration synthesis evidence under `{wave['verification_agent']['expected_integration_synthesis_dir']}`.",
                f"- Write `{wave['verification_agent']['expected_integration_synthesis_report']}` with hardware spec identity, selected child knobs, command/log paths, timing, utilization, DRC, methodology, and pass/fail status.",
                "- Integration synthesis must include the generated parent integration module plus selected child modules; child module OOC reports alone are not sufficient.",
                "- If Vivado cannot run in this environment, record the exact command, log path, and blocker as a P1/P2 finding instead of claiming synthesis passed.",
            ]
        )
    prompt.extend(
        [
            "- Confirm implementation evidence matches this wave's target scope.",
            "- Confirm no target-scale claim bypasses blocked target dependencies.",
            "- Confirm HDL agents did not edit parent orchestration files unless explicitly allowed.",
            "- If an implementation gate failed, confirm a `skill_update_candidate` with all required fields was returned before retry.",
            "- Report findings first as P0/P1/P2/P3 with file and line references where possible.",
            "- If no P0/P1/P2 issues are found, say so clearly.",
            "",
            "## Do Not Claim",
            "",
        ]
    )
    prompt.extend(f"- {item}" for item in dispatch_plan["does_not_claim"])
    prompt.extend(
        [
            "",
            "## Current Blocked Target Tasks",
            "",
        ]
    )
    for blocked in dispatch_plan["blocked_target_tasks"]:
        prompt.append(_blocked_task_line(blocked))
    prompt.extend(
        [
            "",
            "Verification means no source edits, no RTL rewrites, and no test weakening. Integration verification may write generated Vivado evidence and the verification JSON only.",
        ]
    )
    return "\n".join(prompt) + "\n"


def build_hdl_subagent_dispatch_plan(packets: dict[str, Any]) -> dict[str, Any]:
    packets_by_id = _packet_index(packets)
    projection_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "projection_tasks"
    ]
    non_gemm_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "non_gemm_tasks"
    ]
    projection_axi_read_command_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "integration_tasks"
        and packet["current_regression_kernel"] == "projection_axi_read_command_adapter"
    ]
    projection_axi_read_data_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "integration_tasks"
        and packet["current_regression_kernel"] == "projection_axi_read_data_channel_adapter"
    ]
    projection_axi_read_transaction_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "integration_tasks"
        and packet["current_regression_kernel"] == "projection_axi_read_transaction_adapter"
    ]
    projection_axi_stream_integration_ids = [
        packet["task_id"]
        for packet in packets["packets"]
        if packet["task_group"] == "integration_tasks"
        and packet["current_regression_kernel"] == "projection_axi_stream_integration"
    ]
    waves = []
    projection_wave = _dispatch_wave(
        "wave_1_projection_kernels",
        "Spawn one GEMM implementation agent per LLaMA projection task.",
        projection_ids,
        packets_by_id,
    )
    waves.append(projection_wave)
    non_gemm_wave = _dispatch_wave(
        "wave_1_non_gemm_kernels",
        "Spawn non-GEMM implementation agents for RMSNorm/RoPE/attention/control/residual/MLP fixtures.",
        non_gemm_ids,
        packets_by_id,
    )
    waves.append(non_gemm_wave)
    inherited_target_dependencies = sorted(
        set(projection_wave["blocked_target_dependencies"] + non_gemm_wave["blocked_target_dependencies"])
    )
    decoder_wave = _dispatch_wave(
        "wave_2_decoder_block",
        "Compose passed projection and non-GEMM fixtures into a single decoder-block fixture.",
        ["decoder_block_attention_mlp_fixture"],
        packets_by_id,
        depends_on=["wave_1_projection_kernels", "wave_1_non_gemm_kernels"],
        inherited_blocked_target_dependencies=inherited_target_dependencies,
    )
    waves.append(decoder_wave)
    layer_wave = _dispatch_wave(
        "wave_3_layer_fsm",
        "Spawn a Layer FSM agent after decoder-block evidence passes.",
        ["layer_fsm_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_2_decoder_block"],
        inherited_blocked_target_dependencies=decoder_wave["blocked_target_dependencies"],
    )
    waves.append(layer_wave)
    top_wave = _dispatch_wave(
        "wave_4_top_fsm",
        "Spawn a Top FSM agent after Layer FSM evidence passes.",
        ["top_fsm_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_3_layer_fsm"],
        inherited_blocked_target_dependencies=layer_wave["blocked_target_dependencies"],
    )
    waves.append(top_wave)
    token_wave = _dispatch_wave(
        "wave_5_token_loop",
        "Spawn a token-loop scheduler agent after Top FSM evidence passes.",
        ["token_loop_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_4_top_fsm"],
        inherited_blocked_target_dependencies=top_wave["blocked_target_dependencies"],
    )
    waves.append(token_wave)
    memory_wave = _dispatch_wave(
        "wave_6_projection_axi_read_command_adapter",
        "Spawn one memory-command adapter agent per projection to map checkpoint-aware qweight stream plans into bounded AXI read-command fixtures.",
        projection_axi_read_command_ids,
        packets_by_id,
        depends_on=["wave_5_token_loop"],
        inherited_blocked_target_dependencies=token_wave["blocked_target_dependencies"],
    )
    waves.append(memory_wave)
    read_data_wave = _dispatch_wave(
        "wave_7_projection_axi_read_data_channel_adapter",
        "Spawn one memory read-data adapter agent per projection to consume bounded AXI R-channel beats and preserve packed-weight payload order.",
        projection_axi_read_data_ids,
        packets_by_id,
        depends_on=["wave_6_projection_axi_read_command_adapter"],
        inherited_blocked_target_dependencies=memory_wave["blocked_target_dependencies"],
    )
    waves.append(read_data_wave)
    read_transaction_wave = _dispatch_wave(
        "wave_8_projection_axi_read_transaction_adapter",
        "Spawn one memory read-transaction agent per projection to compose bounded AXI AR command and R-channel payload evidence.",
        projection_axi_read_transaction_ids,
        packets_by_id,
        depends_on=["wave_7_projection_axi_read_data_channel_adapter"],
        inherited_blocked_target_dependencies=read_data_wave["blocked_target_dependencies"],
    )
    waves.append(read_transaction_wave)
    axi_stream_integration_wave = _dispatch_wave(
        "wave_9_projection_axi_stream_integration",
        "Spawn one AXI-to-projection stream integration agent per projection after bounded read-transaction evidence passes.",
        projection_axi_stream_integration_ids,
        packets_by_id,
        depends_on=["wave_8_projection_axi_read_transaction_adapter"],
        inherited_blocked_target_dependencies=read_transaction_wave["blocked_target_dependencies"],
    )
    waves.append(axi_stream_integration_wave)
    decoder_child_axi_wave = _dispatch_wave(
        "wave_10_decoder_child_axi_attention_datapath",
        "Spawn a decoder-child AXI datapath agent after AXI projection stream evidence passes.",
        ["decoder_child_axi_attention_datapath"],
        packets_by_id,
        depends_on=["wave_9_projection_axi_stream_integration"],
        inherited_blocked_target_dependencies=axi_stream_integration_wave["blocked_target_dependencies"],
    )
    waves.append(decoder_child_axi_wave)
    layer_axi_wave = _dispatch_wave(
        "wave_11_layer_fsm_axi_attention_fixture",
        "Spawn a Layer FSM agent that calls the verified AXI-aware decoder child fixture.",
        ["layer_fsm_axi_attention_fixture"],
        packets_by_id,
        depends_on=["wave_10_decoder_child_axi_attention_datapath"],
        inherited_blocked_target_dependencies=decoder_child_axi_wave["blocked_target_dependencies"],
    )
    waves.append(layer_axi_wave)
    top_axi_wave = _dispatch_wave(
        "wave_12_top_fsm_axi_attention_fixture",
        "Spawn a Top FSM agent that calls the verified AXI-aware Layer FSM fixture.",
        ["top_fsm_axi_attention_fixture"],
        packets_by_id,
        depends_on=["wave_11_layer_fsm_axi_attention_fixture"],
        inherited_blocked_target_dependencies=layer_axi_wave["blocked_target_dependencies"],
    )
    waves.append(top_axi_wave)
    token_axi_wave = _dispatch_wave(
        "wave_13_token_loop_axi_attention_fixture",
        "Spawn a token-loop agent that calls the verified AXI-aware Top FSM fixture.",
        ["token_loop_axi_attention_fixture"],
        packets_by_id,
        depends_on=["wave_12_top_fsm_axi_attention_fixture"],
        inherited_blocked_target_dependencies=top_axi_wave["blocked_target_dependencies"],
    )
    waves.append(token_axi_wave)
    decoder_axi_block_wave = _dispatch_wave(
        "wave_14_decoder_block_axi_attention_mlp_fixture",
        "Spawn a decoder-block agent that composes the verified AXI-aware attention child with the residual/MLP fixture.",
        ["decoder_block_axi_attention_mlp_fixture"],
        packets_by_id,
        depends_on=["wave_13_token_loop_axi_attention_fixture"],
        inherited_blocked_target_dependencies=token_axi_wave["blocked_target_dependencies"],
    )
    waves.append(decoder_axi_block_wave)
    layer_axi_decoder_block_wave = _dispatch_wave(
        "wave_15_layer_fsm_axi_decoder_block_fixture",
        "Spawn a Layer FSM agent that calls the verified AXI decoder-block fixture.",
        ["layer_fsm_axi_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_14_decoder_block_axi_attention_mlp_fixture"],
        inherited_blocked_target_dependencies=decoder_axi_block_wave["blocked_target_dependencies"],
    )
    waves.append(layer_axi_decoder_block_wave)
    top_axi_decoder_block_wave = _dispatch_wave(
        "wave_16_top_fsm_axi_decoder_block_fixture",
        "Spawn a Top FSM agent that calls the verified AXI decoder-block Layer FSM fixture.",
        ["top_fsm_axi_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_15_layer_fsm_axi_decoder_block_fixture"],
        inherited_blocked_target_dependencies=layer_axi_decoder_block_wave["blocked_target_dependencies"],
    )
    waves.append(top_axi_decoder_block_wave)
    token_axi_decoder_block_wave = _dispatch_wave(
        "wave_17_token_loop_axi_decoder_block_fixture",
        "Spawn a token-loop agent that calls the verified AXI decoder-block Top FSM fixture.",
        ["token_loop_axi_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_16_top_fsm_axi_decoder_block_fixture"],
        inherited_blocked_target_dependencies=top_axi_decoder_block_wave["blocked_target_dependencies"],
    )
    waves.append(token_axi_decoder_block_wave)
    model_axi_decoder_block_wave = _dispatch_wave(
        "wave_18_model_fsm_axi_decoder_block_fixture",
        "Spawn a model-level FSM agent that calls the verified AXI decoder-block token-loop fixture.",
        ["model_fsm_axi_decoder_block_fixture"],
        packets_by_id,
        depends_on=["wave_17_token_loop_axi_decoder_block_fixture"],
        inherited_blocked_target_dependencies=token_axi_decoder_block_wave["blocked_target_dependencies"],
    )
    waves.append(model_axi_decoder_block_wave)
    ddr_axi_board_shell_wave = _dispatch_wave(
        "wave_19_ddr_axi_board_shell_fixture",
        "Spawn a DDR/AXI board-shell agent that wraps the verified model FSM child with bounded external-memory request/status metadata.",
        ["ddr_axi_board_shell_fixture"],
        packets_by_id,
        depends_on=["wave_18_model_fsm_axi_decoder_block_fixture"],
        inherited_blocked_target_dependencies=model_axi_decoder_block_wave["blocked_target_dependencies"],
    )
    waves.append(ddr_axi_board_shell_wave)
    global_checkpoint_blocked_target_dependencies = packets.get("global_checkpoint_blocked_target_dependencies", [])
    global_blocked_target_dependencies = packets.get(
        "global_blocked_target_dependencies",
        global_checkpoint_blocked_target_dependencies,
    )
    waves = _apply_global_blocked_target_dependencies(waves, global_blocked_target_dependencies)
    return {
        "artifact": "hdl_subagent_dispatch_plan",
        "coverage_level": "subagent_prompt_packets_to_dispatch_waves",
        "model": packets["model"],
        "hardware": packets["hardware"],
        "optimization": packets["optimization"],
        "input_clarification": packets.get("input_clarification", {}),
        "source_replay": packets.get("source_replay", {}),
        "subagent_policy": packets["subagent_policy"],
        "blocked_target_tasks": packets["blocked_target_tasks"],
        "global_checkpoint_blocked_target_dependencies": global_checkpoint_blocked_target_dependencies,
        "global_blocked_target_dependencies": global_blocked_target_dependencies,
        "dispatch_policy": {
            "parent_must_not_write_hdl": True,
            "one_subagent_per_hdl_packet": True,
            "module_agents_run_own_simulation_and_synthesis": True,
            "spawn_implementation_agents_from_prompt_files": True,
            "spawn_verification_agent_after_each_wave": True,
            "spawn_read_only_verification_agent_after_each_wave": True,
            "integration_verification_agents_run_synthesis": True,
            "layer_fsm_and_top_fsm_are_separate_implementation_agents": True,
            "failed_hdl_attempt_updates_skill_before_retry": True,
            "do_not_advance_to_dependent_wave_until_current_wave_verification_passes": True,
        },
        "agent_topology": _agent_topology(
            {
                "projection_tasks": [
                    packet for packet in packets["packets"] if packet["task_group"] == "projection_tasks"
                ],
                "non_gemm_tasks": [
                    packet for packet in packets["packets"] if packet["task_group"] == "non_gemm_tasks"
                ],
                "integration_tasks": [
                    packet for packet in packets["packets"] if packet["task_group"] == "integration_tasks"
                ],
            },
            len(packets["packets"]),
        ),
        "wave_count": len(waves),
        "waves": waves,
        "does_not_claim": [
            "automatic sub-agent spawning inside package runtime",
            "completed target RTL for every packet",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ],
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_load_error": "json_decode_error", "path": str(path)}
    if not isinstance(data, dict):
        return {"_load_error": "json_root_not_object", "path": str(path)}
    return data


def _candidate_complete(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    payload = candidate.get("candidate", candidate)
    if not isinstance(payload, dict):
        return False
    return all(bool(payload.get(field)) for field in SKILL_UPDATE_CANDIDATE_FIELDS)


def _candidate_payload(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not _candidate_complete(candidate):
        return None
    payload = candidate.get("candidate", candidate) if isinstance(candidate, dict) else None
    if not isinstance(payload, dict):
        return None
    return {field: payload[field] for field in SKILL_UPDATE_CANDIDATE_FIELDS}


def _load_task_skill_update_candidate(evidence_dir: Path) -> dict[str, Any] | None:
    report = _load_json(evidence_dir / "kernel_report.json")
    if isinstance(report, dict):
        payload = _candidate_payload(report.get("skill_update_candidate"))
        if payload is not None:
            return payload
    result = _load_json(evidence_dir / "subagent_result.json")
    if isinstance(result, dict):
        payload = _candidate_payload(result.get("skill_update_candidate"))
        if payload is not None:
            return payload
    return None


def _subagent_result_status(task: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    result_path = evidence_dir / "subagent_result.json"
    result = _load_json(result_path)
    if result is None:
        return {
            "status": "missing",
            "subagent_result": str(result_path),
            "final_response_complete": False,
            "missing_fields": SUBAGENT_RESULT_FIELDS,
            "reason": "subagent_result.json not found",
        }
    if result.get("_load_error"):
        return {
            "status": "failed",
            "subagent_result": str(result_path),
            "final_response_complete": False,
            "missing_fields": SUBAGENT_RESULT_FIELDS,
            "reason": result["_load_error"],
        }
    missing_fields = [
        field for field in SUBAGENT_RESULT_FIELDS if not result.get(field)
    ]
    task_id_matches = result.get("task_id") in {None, task["task_id"]}
    if not task_id_matches:
        missing_fields.append("task_id_matches_expected_task")
    return {
        "status": "passed" if not missing_fields else "incomplete",
        "subagent_result": str(result_path),
        "final_response_complete": not missing_fields,
        "missing_fields": missing_fields,
        "changed_files": result.get("changed_files", []),
        "commands_run": result.get("commands_run", []),
        "simulation_evidence": result.get("simulation_evidence"),
        "verilator_evidence": result.get("verilator_evidence"),
        "vivado_timing_resource_evidence": result.get("vivado_timing_resource_evidence"),
        "module_ooc_synthesis_evidence": result.get("module_ooc_synthesis_evidence"),
        "remaining_risks": result.get("remaining_risks", []),
        "skill_update_candidate_complete": _candidate_complete(result.get("skill_update_candidate")),
    }


def _module_ooc_synthesis_status(
    task: dict[str, Any],
    evidence_dir: Path,
    kernel_report: dict[str, Any],
) -> dict[str, Any]:
    report_path = evidence_dir / "module_ooc_synthesis_report.json"
    waiver = kernel_report.get("module_ooc_synthesis_waiver")
    if isinstance(waiver, dict) and waiver.get("waived") is True:
        classification = waiver.get("classification")
        return {
            "status": "waived" if classification == "fixture_control_scaffold" else "failed",
            "module_ooc_synthesis_report": str(report_path),
            "required": bool(task.get("requires_module_ooc_synthesis")),
            "waiver": waiver,
            "reason": None
            if classification == "fixture_control_scaffold"
            else "module OOC waiver must classify the packet as fixture_control_scaffold",
        }
    report = _load_json(report_path)
    if report is None:
        return {
            "status": "missing",
            "module_ooc_synthesis_report": str(report_path),
            "required": bool(task.get("requires_module_ooc_synthesis")),
            "reason": "module_ooc_synthesis_report.json not found",
        }
    if report.get("_load_error"):
        return {
            "status": "failed",
            "module_ooc_synthesis_report": str(report_path),
            "required": bool(task.get("requires_module_ooc_synthesis")),
            "reason": report["_load_error"],
        }

    failures = []
    hardware = task.get("hardware", {})
    report_hardware = report.get("hardware_spec")
    if not isinstance(report_hardware, dict):
        failures.append("hardware_spec section is missing")
        report_hardware = {}
    hardware_keys = [
        "fpga_part",
        "target_clock_mhz",
        "max_lut",
        "max_dsp",
        "max_bram",
        "max_ff",
        "max_uram",
        "max_io",
        "memory_data_width",
        "device_logic_cells",
        "device_lut",
        "device_ff",
        "device_dsp",
        "device_bram_36k",
        "device_uram",
        "device_io",
        "device_distributed_ram_mb",
        "device_bram_mb",
        "device_uram_mb",
        "device_ps_gtr",
        "device_gth",
        "resource_reference",
    ]
    for key in hardware_keys:
        expected = hardware.get(key)
        if expected is None:
            continue
        observed = report_hardware.get(key)
        if observed != expected:
            failures.append(f"hardware_spec.{key} mismatch: expected {expected}, observed {observed}")
    if report.get("status") != "passed":
        failures.append("module OOC status is not passed")
    vivado = report.get("vivado")
    if not isinstance(vivado, dict):
        failures.append("vivado section is missing")
    else:
        if not vivado.get("part"):
            failures.append("vivado.part is missing")
        elif hardware.get("fpga_part") is not None and vivado.get("part") != hardware.get("fpga_part"):
            failures.append(
                f"vivado.part mismatch: expected {hardware.get('fpga_part')}, observed {vivado.get('part')}"
            )
        if vivado.get("target_clock_mhz") is None and vivado.get("target_clock_period_ns") is None:
            failures.append("vivado target clock is missing")
        elif hardware.get("target_clock_mhz") is not None and vivado.get("target_clock_mhz") is not None:
            if float(vivado.get("target_clock_mhz")) != float(hardware.get("target_clock_mhz")):
                failures.append(
                    "vivado.target_clock_mhz mismatch: "
                    f"expected {hardware.get('target_clock_mhz')}, observed {vivado.get('target_clock_mhz')}"
                )
    timing = report.get("timing")
    if not isinstance(timing, dict):
        failures.append("timing section is missing")
    else:
        if timing.get("constraints_met") is not True:
            failures.append("module OOC timing constraints are not met")
        for field in TIMING_FIELDS:
            value = timing.get(field)
            if value is None or float(value) < 0:
                failures.append(f"module OOC {field} is missing or negative")
    utilization = report.get("utilization")
    if not isinstance(utilization, dict):
        failures.append("utilization section is missing")
    else:
        for field in ("lut", "dsp", "bram", "uram", "ff", "io"):
            if utilization.get(field) is None:
                failures.append(f"utilization.{field} is missing")
    if not isinstance(report.get("selected_tuning_knobs"), dict):
        failures.append("selected_tuning_knobs section is missing")
    allowed_assessments = {
        "underutilized",
        "near_budget",
        "bandwidth_limited",
        "timing_limited",
        "fixture_control_scaffold",
    }
    if report.get("resource_assessment") not in allowed_assessments:
        failures.append("resource_assessment is missing or invalid")
    if report.get("resource_assessment") == "underutilized" and report.get("throughput_target_met") is not True:
        failures.append("underutilized module must either tune further or mark throughput_target_met true")

    hardware_mismatch = any("mismatch" in failure or failure == "hardware_spec section is missing" for failure in failures)
    return {
        "status": "passed" if not failures else "hardware_mismatch" if hardware_mismatch else "failed",
        "module_ooc_synthesis_report": str(report_path),
        "required": bool(task.get("requires_module_ooc_synthesis")),
        "expected_hardware_spec": hardware,
        "observed_hardware_spec": report_hardware,
        "resource_assessment": report.get("resource_assessment"),
        "selected_tuning_knobs": report.get("selected_tuning_knobs"),
        "utilization": utilization if isinstance(utilization, dict) else None,
        "timing": timing if isinstance(timing, dict) else None,
        "reason": None if not failures else "; ".join(failures),
    }


def _task_evidence_dir(task: dict[str, Any], collection_root: Path) -> Path:
    expected = task.get("expected_evidence_dir", f"build/{_slug(task['task_id'])}_gate")
    return collection_root / Path(expected).name


def _task_gate_status(task: dict[str, Any], collection_root: Path) -> dict[str, Any]:
    evidence_dir = _task_evidence_dir(task, collection_root)
    report_path = evidence_dir / "kernel_report.json"
    subagent_result = _subagent_result_status(task, evidence_dir)
    report = _load_json(report_path)
    if report is None:
        return {
            "task_id": task["task_id"],
            "status": "missing",
            "evidence_dir": str(evidence_dir),
            "kernel_report": str(report_path),
            "subagent_result": subagent_result,
            "reason": "kernel_report.json not found",
        }
    if report.get("_load_error"):
        return {
            "task_id": task["task_id"],
            "status": "failed_missing_skill_candidate",
            "evidence_dir": str(evidence_dir),
            "kernel_report": str(report_path),
            "subagent_result": subagent_result,
            "reason": report["_load_error"],
            "skill_update_candidate_complete": False,
        }

    failures = []
    if report.get("status") != "passed":
        failures.append("kernel_report status is not passed")
    if report.get("simulation", {}).get("passed") is not True:
        failures.append("simulation.passed is not true")
    if report.get("verilator", {}).get("passed") is not True:
        failures.append("verilator.passed is not true")
    if report.get("contract_gate", {}).get("verilator_enforced") is not True:
        failures.append("contract_gate.verilator_enforced is not true")
    synthesis = report.get("synthesis")
    timing = synthesis.get("timing", {}) if isinstance(synthesis, dict) else {}
    if isinstance(synthesis, dict):
        if synthesis.get("passed") is not True:
            failures.append("synthesis.passed is not true")
        if timing:
            if timing.get("constraints_met") is not True:
                failures.append("synthesis timing constraints are not met")
            for field in TIMING_FIELDS:
                value = timing.get(field)
                if value is None or float(value) < 0:
                    failures.append(f"{field} is missing or negative")
    module_ooc = None
    if task.get("requires_module_ooc_synthesis"):
        module_ooc = _module_ooc_synthesis_status(task, evidence_dir, report)
        if module_ooc["status"] == "missing":
            return {
                "task_id": task["task_id"],
                "status": "module_ooc_synthesis_missing",
                "evidence_dir": str(evidence_dir),
                "kernel_report": str(report_path),
                "subagent_result": subagent_result,
                "module_ooc_synthesis": module_ooc,
                "reason": module_ooc["reason"],
                "skill_update_candidate_complete": False,
            }
        if module_ooc["status"] == "hardware_mismatch":
            return {
                "task_id": task["task_id"],
                "status": "module_ooc_synthesis_hardware_mismatch",
                "evidence_dir": str(evidence_dir),
                "kernel_report": str(report_path),
                "subagent_result": subagent_result,
                "module_ooc_synthesis": module_ooc,
                "reason": module_ooc["reason"],
                "skill_update_candidate_complete": False,
            }
        if (
            module_ooc["status"] == "failed"
            and "underutilized module" in str(module_ooc.get("reason", ""))
        ):
            return {
                "task_id": task["task_id"],
                "status": "module_ooc_synthesis_needs_tuning",
                "evidence_dir": str(evidence_dir),
                "kernel_report": str(report_path),
                "subagent_result": subagent_result,
                "module_ooc_synthesis": module_ooc,
                "reason": module_ooc["reason"],
                "skill_update_candidate_complete": False,
            }
        if module_ooc["status"] not in {"passed", "waived"}:
            failures.append(module_ooc["reason"] or "module OOC synthesis gate did not pass")

    subagent_payload = _load_json(Path(subagent_result["subagent_result"]))
    candidate = report.get("skill_update_candidate")
    if not candidate and isinstance(subagent_payload, dict):
        candidate = subagent_payload.get("skill_update_candidate")
    if failures:
        complete = _candidate_complete(candidate)
        return {
            "task_id": task["task_id"],
            "status": "failed_with_skill_candidate" if complete else "failed_missing_skill_candidate",
            "evidence_dir": str(evidence_dir),
            "kernel_report": str(report_path),
            "subagent_result": subagent_result,
            "module_ooc_synthesis": module_ooc,
            "reason": "; ".join(failures),
            "skill_update_candidate_complete": complete,
        }
    if subagent_result.get("status") != "passed" or subagent_result.get("final_response_complete") is not True:
        return {
            "task_id": task["task_id"],
            "status": "incomplete_subagent_result",
            "evidence_dir": str(evidence_dir),
            "kernel_report": str(report_path),
            "subagent_result": subagent_result,
            "module_ooc_synthesis": module_ooc,
            "reason": "kernel_report passed but subagent_result.json is missing or incomplete",
            "skill_update_candidate_complete": subagent_result.get("skill_update_candidate_complete", False),
        }

    return {
        "task_id": task["task_id"],
        "status": "passed",
        "evidence_dir": str(evidence_dir),
        "kernel_report": str(report_path),
        "subagent_result": subagent_result,
        "module_ooc_synthesis": module_ooc,
        "coverage_level": report.get("coverage_level"),
        "implementation_stage": report.get("implementation_stage"),
        "synthesis_timing": timing or None,
    }


def _verification_report_path(wave: dict[str, Any], collection_root: Path) -> Path:
    prompt_name = Path(wave["verification_agent"]["prompt_file"]).stem
    return collection_root / "verification_results" / f"{prompt_name}.json"


def _verification_skill_update_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    top_level_candidate = _candidate_payload(report.get("skill_update_candidate"))
    if top_level_candidate is not None:
        candidates.append(
            {
                "source": "verification_report",
                "candidate": top_level_candidate,
            }
        )
    for finding in report.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("severity") not in {"P0", "P1", "P2"}:
            continue
        finding_candidate = _candidate_payload(finding.get("skill_update_candidate"))
        if finding_candidate is None:
            continue
        candidates.append(
            {
                "source": "verification_finding",
                "severity": finding.get("severity"),
                "title": finding.get("title"),
                "body": finding.get("body"),
                "candidate": finding_candidate,
            }
        )
    return candidates


def _verification_status(wave: dict[str, Any], collection_root: Path) -> dict[str, Any]:
    report_path = _verification_report_path(wave, collection_root)
    report = _load_json(report_path)
    if report is None:
        return {
            "status": "missing",
            "verification_report": str(report_path),
            "reason": "read-only Codex verification report not found",
        }
    if report.get("_load_error"):
        return {
            "status": "failed",
            "verification_report": str(report_path),
            "reason": report["_load_error"],
        }
    blocking_findings = []
    for finding in report.get("findings", []):
        if isinstance(finding, dict) and finding.get("severity") in {"P0", "P1", "P2"}:
            blocking_findings.append(finding)
    passed = report.get("status") == "passed" and not blocking_findings
    skill_candidates = _verification_skill_update_candidates(report)
    return {
        "status": "passed" if passed else "failed",
        "verification_report": str(report_path),
        "blocking_finding_count": len(blocking_findings),
        "skill_update_candidate_count": len(skill_candidates),
        "skill_update_candidate_complete": bool(skill_candidates),
        "reason": None if passed else "verification status is not passed or has P0/P1/P2 findings",
    }


def build_hdl_subagent_wave_status(
    dispatch_plan: dict[str, Any],
    collection_root: Path,
) -> dict[str, Any]:
    """Summarize sub-agent evidence without generating or editing HDL."""
    waves = []
    passed_wave_ids: set[str] = set()
    for wave in dispatch_plan["waves"]:
        missing_dependencies = [
            dependency for dependency in wave["depends_on_waves"] if dependency not in passed_wave_ids
        ]
        task_results = [
            _task_gate_status(task, collection_root) for task in wave["implementation_tasks"]
        ]
        task_status_counts: dict[str, int] = {}
        for result in task_results:
            result_status = result["status"]
            task_status_counts[result_status] = task_status_counts.get(result_status, 0) + 1
        if missing_dependencies:
            status = "blocked_by_dependency"
            reason = f"waiting for waves: {', '.join(missing_dependencies)}"
            verification = {"status": "not_applicable_until_dependencies_pass"}
        elif any(result["status"] == "failed_missing_skill_candidate" for result in task_results):
            status = "failed_missing_skill_candidate"
            reason = "at least one failed task lacks a complete skill_update_candidate"
            verification = {"status": "not_started"}
        elif any(result["status"] == "failed_with_skill_candidate" for result in task_results):
            status = "failed_waiting_for_skill_update"
            reason = "at least one task failed and must be converted into a SKILL before retry"
            verification = {"status": "not_started"}
        elif any(result["status"] == "incomplete_subagent_result" for result in task_results):
            status = "incomplete_subagent_result"
            reason = "at least one passed kernel lacks a complete subagent_result.json final-response record"
            verification = {"status": "not_started"}
        elif any(result["status"] == "module_ooc_synthesis_missing" for result in task_results):
            status = "ready_to_dispatch"
            reason = "at least one real datapath module is missing module-level OOC synthesis evidence"
            verification = {"status": "not_started"}
        elif any(result["status"] == "module_ooc_synthesis_hardware_mismatch" for result in task_results):
            status = "ready_to_dispatch"
            reason = "at least one module-level OOC synthesis report is stale for the active hardware spec"
            verification = {"status": "not_started"}
        elif any(result["status"] == "module_ooc_synthesis_needs_tuning" for result in task_results):
            status = "ready_to_dispatch"
            reason = "at least one real datapath module needs resource/timing tuning before integration"
            verification = {"status": "not_started"}
        elif any(result["status"] == "missing" for result in task_results):
            status = "ready_to_dispatch"
            reason = "one or more implementation task results are missing"
            verification = {"status": "not_started"}
        else:
            verification = _verification_status(wave, collection_root)
            if verification["status"] == "passed":
                status = "passed"
                reason = None
                passed_wave_ids.add(wave["wave_id"])
            elif verification["status"] == "missing":
                status = "ready_for_verification"
                reason = "implementation tasks passed; read-only Codex verification is required"
            else:
                if verification.get("skill_update_candidate_complete"):
                    status = "failed_verification_waiting_for_skill_update"
                    reason = "read-only verification failed and must be converted into a SKILL before retry"
                else:
                    status = "failed_verification_missing_skill_candidate"
                    reason = "read-only verification failed without a complete skill_update_candidate"

        waves.append(
            {
                "wave_id": wave["wave_id"],
                "status": status,
                "reason": reason,
                "target_scope": wave["target_scope"],
                "depends_on_waves": wave["depends_on_waves"],
                "blocked_target_dependencies": wave["blocked_target_dependencies"],
                "task_count": len(task_results),
                "task_status_counts": task_status_counts,
                "passed_task_count": task_status_counts.get("passed", 0),
                "missing_task_count": task_status_counts.get("missing", 0),
                "task_results": task_results,
                "verification": verification,
            }
        )

    next_dispatchable = [
        wave["wave_id"] for wave in waves if wave["status"] in {"ready_to_dispatch", "ready_for_verification"}
    ]
    return {
        "artifact": "hdl_subagent_wave_status",
        "coverage_level": "parent_result_collection_no_hdl_generation",
        "collection_root": str(collection_root),
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required_before_retry": True,
        "next_dispatchable_waves": next_dispatchable,
        "wave_count": len(waves),
        "waves": waves,
        "does_not_claim": [
            "automatic sub-agent spawning",
            "sub-agent execution occurred",
            "generated RTL completeness",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ],
    }


def _preflight_blocker_ids(dispatch_plan: dict[str, Any]) -> list[str]:
    preflight_ids = {
        "real_mlir_model_analysis",
        "real_gptq_checkpoint_source",
        "real_gptq_checkpoint_metadata",
        "real_gptq_weight_layout_preflight",
        "real_gptq_payload_probe",
    }
    return sorted(
        {
            task.get("task_id")
            for task in dispatch_plan.get("blocked_target_tasks", [])
            if isinstance(task, dict) and task.get("task_id") in preflight_ids
        }
        | set(dispatch_plan.get("global_blocked_target_dependencies", []))
    )


def _load_full_execution_evidence(collection_root: Path) -> dict[str, Any] | None:
    return _load_json(collection_root / "full_llama_execution_evidence.json")


def _full_execution_evidence_failures(
    evidence: dict[str, Any] | None,
    dispatch_plan: dict[str, Any],
) -> list[str]:
    if evidence is None:
        return ["full_llama_execution_evidence.json not found"]
    if evidence.get("_load_error"):
        return [str(evidence["_load_error"])]
    failures = []
    if evidence.get("artifact") != "full_llama_execution_evidence":
        failures.append("artifact must be full_llama_execution_evidence")
    if evidence.get("status") != "passed":
        failures.append("status must be passed")
    if evidence.get("model") != dispatch_plan.get("model", {}).get("name"):
        failures.append("model must match dispatch plan model")
    if evidence.get("target_preflight_status") != "passed":
        failures.append("target_preflight_status must be passed")
    if evidence.get("full_model_layers_executed") is not True:
        failures.append("full_model_layers_executed must be true")
    decoder_layers = dispatch_plan.get("model", {}).get("decoder_layers")
    if isinstance(decoder_layers, int):
        layer_count = evidence.get("executed_layer_count")
        if not isinstance(layer_count, int) or layer_count < decoder_layers:
            failures.append("executed_layer_count must cover every decoder layer")
    reference = evidence.get("python_reference_comparison")
    if not isinstance(reference, dict) or reference.get("passed") is not True:
        failures.append("python_reference_comparison.passed must be true")
    token_loop = evidence.get("token_loop_evidence")
    if not isinstance(token_loop, dict) or token_loop.get("passed") is not True:
        failures.append("token_loop_evidence.passed must be true")
    model_fsm = evidence.get("model_fsm_evidence")
    if not isinstance(model_fsm, dict) or model_fsm.get("passed") is not True:
        failures.append("model_fsm_evidence.passed must be true")
    checkpoint = evidence.get("checkpoint_payload_evidence")
    if not isinstance(checkpoint, dict) or checkpoint.get("passed") is not True:
        failures.append("checkpoint_payload_evidence.passed must be true")
    if evidence.get("board_level_signoff") is True:
        failures.append("full execution evidence must not claim board_level_signoff")
    return failures


def build_full_llama_execution_readiness_report(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    collection_root: Path,
) -> dict[str, Any]:
    preflight_blockers = _preflight_blocker_ids(dispatch_plan)
    waves = wave_status.get("waves", [])
    non_passed_waves = [
        {"wave_id": wave.get("wave_id"), "status": wave.get("status"), "reason": wave.get("reason")}
        for wave in waves
        if isinstance(wave, dict) and wave.get("status") != "passed"
    ]
    target_scopes = sorted(
        {
            wave.get("target_scope")
            for wave in dispatch_plan.get("waves", [])
            if isinstance(wave, dict) and isinstance(wave.get("target_scope"), str)
        }
    )
    evidence = _load_full_execution_evidence(collection_root)
    evidence_failures = _full_execution_evidence_failures(evidence, dispatch_plan)
    if preflight_blockers:
        status = "blocked_by_target_preflight"
    elif non_passed_waves:
        status = "blocked_by_subagent_wave_status"
    elif evidence_failures:
        status = "blocked_by_missing_or_incomplete_full_execution_evidence"
    else:
        status = "passed"
    return {
        "artifact": "full_llama_execution_readiness",
        "coverage_level": "parent_gate_for_full_model_execution_claim",
        "status": status,
        "collection_root": str(collection_root),
        "model": dispatch_plan.get("model", {}),
        "target": {
            "fpga_part": dispatch_plan.get("hardware", {}).get("fpga_part"),
            "quantization": dispatch_plan.get("optimization", {}).get("quantization"),
            "design_style": dispatch_plan.get("optimization", {}).get("design_style"),
            "compute_style": dispatch_plan.get("optimization", {}).get("compute_style"),
            "execution_style": dispatch_plan.get("optimization", {}).get("execution_style"),
            "memory_style": dispatch_plan.get("optimization", {}).get("memory_style"),
            "control_style": dispatch_plan.get("optimization", {}).get("control_style"),
        },
        "target_preflight": {
            "status": "passed" if not preflight_blockers else "blocked",
            "preflight_blockers": preflight_blockers,
            "dispatch_target_scopes": target_scopes,
        },
        "subagent_waves": {
            "wave_count": wave_status.get("wave_count", len(waves)),
            "passed_wave_count": sum(1 for wave in waves if isinstance(wave, dict) and wave.get("status") == "passed"),
            "non_passed_wave_count": len(non_passed_waves),
            "non_passed_waves": non_passed_waves,
        },
        "required_evidence_file": str(collection_root / "full_llama_execution_evidence.json"),
        "full_execution_evidence": evidence if isinstance(evidence, dict) and not evidence.get("_load_error") else None,
        "evidence_failures": evidence_failures,
        "safe_to_clear_full_llama_model_execution_blocker": status == "passed",
        "next_action": (
            "resolve target preflight blockers before claiming full execution"
            if preflight_blockers
            else "finish/publish all HDL sub-agent wave evidence before claiming full execution"
            if non_passed_waves
            else "write full_llama_execution_evidence.json with full-layer decode and Python reference comparison"
            if evidence_failures
            else "full execution readiness passed; board-level signoff remains a separate gate"
        ),
        "does_not_claim": [
            "board_level_ZCU104_signoff",
            "real_time_LLaMA_inference_performance",
        ],
    }


def build_full_llama_execution_evidence_template(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    collection_root: Path,
) -> dict[str, Any]:
    model = dispatch_plan.get("model", {})
    waves = wave_status.get("waves", [])
    passed_wave_ids = [
        wave.get("wave_id")
        for wave in waves
        if isinstance(wave, dict) and wave.get("status") == "passed"
    ]
    return {
        "artifact": "full_llama_execution_evidence_template",
        "write_to": str(collection_root / "full_llama_execution_evidence.json"),
        "purpose": "Evidence required before clearing full_llama_model_execution.",
        "required_fields": [
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
        ],
        "template": {
            "artifact": "full_llama_execution_evidence",
            "status": "<passed>",
            "model": model.get("name"),
            "target_preflight_status": "<passed>",
            "full_model_layers_executed": "<true>",
            "executed_layer_count": model.get("decoder_layers"),
            "token_loop_evidence": {
                "passed": "<true>",
                "source_wave_id": "wave_17_token_loop_axi_decoder_block_fixture",
                "report": "<path to token-loop/model execution report>",
            },
            "model_fsm_evidence": {
                "passed": "<true>",
                "source_wave_id": "wave_18_model_fsm_axi_decoder_block_fixture",
                "report": "<path to model FSM execution report>",
            },
            "checkpoint_payload_evidence": {
                "passed": "<true>",
                "source": "gptq_payload_probe.json",
                "projection_payloads_verified": "<all required projections>",
            },
            "python_reference_comparison": {
                "passed": "<true>",
                "tolerance_lsb": "<configured tolerance>",
                "reference_artifact": "<path to Python/NumPy reference output>",
                "rtl_artifact": "<path to RTL/model output>",
            },
            "board_level_signoff": False,
        },
        "current_wave_context": {
            "passed_wave_ids": passed_wave_ids,
            "non_passed_wave_ids": [
                wave.get("wave_id")
                for wave in waves
                if isinstance(wave, dict) and wave.get("status") != "passed"
            ],
        },
        "does_not_claim": [
            "board_level_ZCU104_signoff",
            "hardware_lab_runtime_validation",
        ],
    }


def _model_level_harness_report_path(collection_root: Path) -> Path:
    return collection_root / "full_llama_execution_gate" / "model_level_execution_harness_report.json"


def _model_level_harness_failures(
    report: dict[str, Any] | None,
    dispatch_plan: dict[str, Any],
) -> list[str]:
    if report is None:
        return ["model_level_execution_harness_report.json not found"]
    if report.get("_load_error"):
        return [str(report["_load_error"])]
    failures = []
    if report.get("artifact") != "model_level_execution_harness_report":
        failures.append("artifact must be model_level_execution_harness_report")
    if report.get("status") != "passed":
        failures.append("status must be passed")
    decoder_layers = dispatch_plan.get("model", {}).get("decoder_layers")
    executed_layer_count = report.get("executed_layer_count")
    if isinstance(decoder_layers, int):
        if not isinstance(executed_layer_count, int) or executed_layer_count < decoder_layers:
            failures.append("executed_layer_count must cover dispatch_plan.model.decoder_layers")
    if report.get("full_model_layers_executed") is not True and report.get("target_16_layer_iteration") is not True:
        failures.append("full_model_layers_executed or target_16_layer_iteration must be true")
    reference = report.get("python_reference_comparison")
    if not isinstance(reference, dict) or reference.get("passed") is not True:
        failures.append("python_reference_comparison.passed must be true")
    if report.get("full_llama_model") is False:
        failures.append("full_llama_model false cannot satisfy full execution harness")
    fixture_layer_count = report.get("fixture_layer_count")
    if isinstance(fixture_layer_count, int) and isinstance(decoder_layers, int) and fixture_layer_count < decoder_layers:
        failures.append("bounded fixture_layer_count is below decoder layer count")
    if report.get("board_level_signoff") is True:
        failures.append("model-level harness must not claim board_level_signoff")
    return failures


def build_model_level_execution_harness_agent_task(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    collection_root: Path,
    dispatch_plan_path: str | None = None,
) -> dict[str, Any]:
    waves = wave_status.get("waves", [])
    non_passed_waves = [
        wave.get("wave_id")
        for wave in waves
        if isinstance(wave, dict) and wave.get("status") != "passed"
    ]
    target_preflight = full_execution_readiness.get("target_preflight", {})
    harness_path = _model_level_harness_report_path(collection_root)
    harness_report = _load_json(harness_path)
    harness_failures = _model_level_harness_failures(harness_report, dispatch_plan)
    precondition_failures = []
    if target_preflight.get("status") != "passed":
        precondition_failures.append("target_preflight.status must be passed")
    if non_passed_waves:
        precondition_failures.append("all sub-agent waves must be passed before model-level harness evidence")
    if not harness_failures:
        precondition_failures.append("model_level_execution_harness_report already passed; spawn full evidence agent instead")
    ready_to_spawn = not precondition_failures
    model = dispatch_plan.get("model", {})
    hardware = dispatch_plan.get("hardware", {})
    optimization = dispatch_plan.get("optimization", {})
    target_clock_mhz = hardware.get("target_clock_mhz")
    target_clock_period_ns = None
    if isinstance(target_clock_mhz, (int, float)) and target_clock_mhz > 0:
        target_clock_period_ns = 1000.0 / float(target_clock_mhz)
    dispatch_arg = dispatch_plan_path or "<path to hdl_subagent_dispatch_plan.json>"
    status_command = (
        "python3 -m nl2hdl subagents status "
        f"--dispatch-plan {shlex.quote(str(dispatch_arg))} "
        f"--evidence-root {shlex.quote(str(collection_root))} "
        "--out build/full_llama_execution_gate/model_harness_status_after"
    )
    prompt_lines = [
        "# Target Evidence Sub-Agent Task: model_level_execution_harness",
        "",
        "You are the Codex implementation sub-agent for the model-level execution harness required before full LLaMA execution evidence.",
        "Do not only inspect the existing bounded fixture reports. Implement the smallest non-HDL model-level harness/report path that can prove, or explicitly fail to prove, the model-level loop.",
        "Do not write final `full_llama_execution_evidence.json`. Your job is to create the missing harness report only if it proves the model-level loop is no longer a bounded 2-layer fixture.",
        "",
        "## Target Context",
        "",
        f"- Model: `{model.get('name')}`",
        f"- Decoder layers required: `{model.get('decoder_layers')}`",
        f"- FPGA part: `{hardware.get('fpga_part')}`",
        f"- Quantization: `{optimization.get('quantization')}`",
        f"- Design style alias: `{optimization.get('design_style')}`",
        f"- Compute style: `{optimization.get('compute_style')}`",
        f"- Execution style: `{optimization.get('execution_style')}`",
        f"- Memory style: `{optimization.get('memory_style')}`",
        f"- Control style: `{optimization.get('control_style')}`",
        f"- Collection root: `{collection_root}`",
        f"- Dispatch plan: `{dispatch_arg}`",
        "",
        "## Preconditions",
        "",
        f"- Ready to spawn: `{ready_to_spawn}`",
        f"- Target preflight status: `{target_preflight.get('status')}`",
        f"- Non-passed waves: `{non_passed_waves}`",
        f"- Existing harness failures: `{harness_failures}`",
        "",
        "## Required Output",
        "",
        f"- Write `{harness_path}` only after it proves 16-layer model-level execution and Python reference comparison.",
        "- Write `build/full_llama_execution_gate/model_level_harness_subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.",
        "- Do not write `build/full_llama_execution_evidence.json`; that is a downstream target-evidence gate.",
        "",
        "## Harness Report Requirements",
        "",
        "- `artifact == model_level_execution_harness_report`.",
        "- `status == passed`.",
        "- `executed_layer_count >= dispatch_plan.model.decoder_layers`.",
        "- `full_model_layers_executed == true` or `target_16_layer_iteration == true`.",
        "- `python_reference_comparison.passed == true`.",
        "- Bounded fixture fields such as `fixture_layer_count: 2`, `target_16_layer_iteration: false`, or `full_llama_model: false` must not be used as passing evidence.",
        "- `board_level_signoff` must remain false or absent.",
        "",
        "## Allowed Write Scope",
        "",
        "- You may write files under `build/full_llama_execution_gate/`.",
        "- You may add or update focused non-HDL harness code and tests needed to produce this report.",
        "- Preferred source scope for harness/report generation is `nl2hdl/llm_kernels.py` or a small focused helper imported by it.",
        "- Preferred test scope is focused assertions in `tests/test_llm_kernels.py`.",
        "- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.",
        "- Do not modify HDL/SystemVerilog unless a separate HDL packet failure is identified; if so, return a `skill_update_candidate` instead of broad edits.",
        "",
        "## Failure-To-SKILL Candidate",
        "",
        "- If you cannot prove the harness, preserve evidence and return a `skill_update_candidate` with:",
    ]
    prompt_lines.extend(f"  - `{field}`" for field in SKILL_UPDATE_CANDIDATE_FIELDS)
    prompt_lines.extend(
        [
            "",
            "## Required Commands",
            "",
            "- `python3 -m pytest -q tests/test_llm_kernels.py`",
            f"- `{status_command}`",
            "",
            "## Do Not Claim",
            "",
            "- Full LLaMA execution readiness until the downstream `full_llama_execution_evidence.json` gate passes.",
            "- Board-level ZCU104 signoff.",
            "- Real-time LLaMA inference performance.",
            "- Hardware lab runtime validation.",
            "",
        ]
    )
    return {
        "artifact": "model_level_execution_harness_agent_task",
        "task_id": "model_level_execution_harness",
        "spawn_kind": "target_evidence_implementation_agent",
        "agent": "Codex",
        "mode": "read_write_non_hdl_harness",
        "ready_to_spawn": ready_to_spawn,
        "spawn_precondition_failures": precondition_failures,
        "harness_report": str(harness_path),
        "harness_failures": harness_failures,
        "prompt_file": "target_evidence_prompts/model_level_execution_harness_agent.md",
        "expected_evidence_file": str(harness_path),
        "expected_subagent_result": "build/full_llama_execution_gate/model_level_harness_subagent_result.json",
        "source_readiness_status": full_execution_readiness.get("status"),
        "required_commands": [
            "python3 -m pytest -q tests/test_llm_kernels.py",
            status_command,
        ],
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required": True,
        "prompt": "\n".join(prompt_lines),
        "does_not_claim": [
            "full_LLaMA_execution",
            "board_level_ZCU104_signoff",
            "real_time_LLaMA_inference_performance",
            "hardware_lab_runtime_validation",
        ],
    }


def build_full_llama_execution_evidence_agent_task(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    evidence_template: dict[str, Any],
    collection_root: Path,
    dispatch_plan_path: str | None = None,
) -> dict[str, Any]:
    waves = wave_status.get("waves", [])
    non_passed_waves = [
        wave.get("wave_id")
        for wave in waves
        if isinstance(wave, dict) and wave.get("status") != "passed"
    ]
    target_preflight = full_execution_readiness.get("target_preflight", {})
    evidence_failures = full_execution_readiness.get("evidence_failures", [])
    harness_report = _load_json(_model_level_harness_report_path(collection_root))
    harness_failures = _model_level_harness_failures(harness_report, dispatch_plan)
    precondition_failures = []
    if target_preflight.get("status") != "passed":
        precondition_failures.append("target_preflight.status must be passed")
    if non_passed_waves:
        precondition_failures.append("all sub-agent waves must be passed before full execution evidence")
    if "full_llama_execution_evidence.json not found" not in evidence_failures:
        precondition_failures.append("full execution readiness must be waiting on full_llama_execution_evidence.json")
    if harness_failures:
        precondition_failures.append(
            "model_level_execution_harness_report must prove full decoder-layer execution and Python reference comparison"
        )
    ready_to_spawn = not precondition_failures
    model = dispatch_plan.get("model", {})
    hardware = dispatch_plan.get("hardware", {})
    optimization = dispatch_plan.get("optimization", {})
    target_clock_mhz = hardware.get("target_clock_mhz")
    target_clock_period_ns = None
    if isinstance(target_clock_mhz, (int, float)) and target_clock_mhz > 0:
        target_clock_period_ns = 1000.0 / float(target_clock_mhz)
    dispatch_arg = dispatch_plan_path or "<path to hdl_subagent_dispatch_plan.json>"
    status_command = (
        "python3 -m nl2hdl subagents status "
        f"--dispatch-plan {shlex.quote(str(dispatch_arg))} "
        f"--evidence-root {shlex.quote(str(collection_root))} "
        "--out build/full_llama_execution_gate/status_after"
    )
    prompt_lines = [
        "# Target Evidence Sub-Agent Task: full_llama_execution",
        "",
        "You are the Codex target-evidence sub-agent for full LLaMA execution readiness.",
        "The parent agent must not hand-write HDL or fabricate evidence. Your job is to inspect the passed child evidence, run or add the smallest necessary model-level execution harness, and write the required evidence only if it is proven.",
        "",
        "## Target Context",
        "",
        f"- Model: `{model.get('name')}`",
        f"- Decoder layers required: `{model.get('decoder_layers')}`",
        f"- FPGA part: `{hardware.get('fpga_part')}`",
        f"- Quantization: `{optimization.get('quantization')}`",
        f"- Design style alias: `{optimization.get('design_style')}`",
        f"- Compute style: `{optimization.get('compute_style')}`",
        f"- Execution style: `{optimization.get('execution_style')}`",
        f"- Memory style: `{optimization.get('memory_style')}`",
        f"- Control style: `{optimization.get('control_style')}`",
        f"- Collection root: `{collection_root}`",
        f"- Dispatch plan: `{dispatch_arg}`",
        "",
        "## Preconditions",
        "",
        f"- Ready to spawn: `{ready_to_spawn}`",
        f"- Target preflight status: `{target_preflight.get('status')}`",
        f"- Passed wave count: `{full_execution_readiness.get('subagent_waves', {}).get('passed_wave_count')}`",
        f"- Non-passed waves: `{non_passed_waves}`",
        f"- Current evidence failures: `{evidence_failures}`",
        f"- Model-level harness failures: `{harness_failures}`",
        "",
        "If `Ready to spawn` is false, do not force the gate. Return a `skill_update_candidate` or a precise missing-evidence report instead.",
        "",
        "## Required Output",
        "",
        f"- Write `{evidence_template['write_to']}` only after every required field below is backed by concrete artifacts.",
        "- Also write `build/full_llama_execution_gate/subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.",
        "- Do not set `board_level_signoff` to true. Board signoff is a separate downstream gate.",
        "",
        "## Evidence Schema",
        "",
        f"- Required fields: `{', '.join(evidence_template.get('required_fields', []))}`",
        f"- Template artifact: `{evidence_template.get('template', {}).get('artifact')}`",
        f"- Template JSON: `{evidence_template.get('template')}`",
        "",
        "## Evidence That Must Be Proven",
        "",
        "- `target_preflight_status == passed` from parent readiness.",
        "- Every decoder layer required by the dispatch plan is executed or explicitly covered by a model-level loop fixture.",
        "- `wave_17_token_loop_axi_decoder_block_fixture` evidence is consumed for token-loop scheduling.",
        "- `wave_18_model_fsm_axi_decoder_block_fixture` evidence is consumed for model-level FSM scheduling.",
        "- GPTQ checkpoint payload evidence covers every required projection, not only q_proj.",
        "- Python/NumPy or PyTorch reference comparison passes within configured tolerance.",
        "- The produced evidence must remain scoped to full model execution and must not claim ZCU104 board signoff.",
        "",
        "## Allowed Write Scope",
        "",
        "- You may write `build/full_llama_execution_evidence.json` and files under `build/full_llama_execution_gate/`.",
        "- You may add or update focused non-HDL harness/tests only if they are necessary to prove model-level execution evidence.",
        "- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.",
        "- Do not weaken existing tests, evidence gates, blocked-target language, or board-signoff separation.",
        "- Do not rewrite passed child HDL packets unless a new failure is found; if that happens, return a `skill_update_candidate` before retry.",
        "",
        "## Failure-To-SKILL Candidate",
        "",
        "- If you cannot prove full execution, preserve evidence and return a `skill_update_candidate` with:",
    ]
    prompt_lines.extend(f"  - `{field}`" for field in SKILL_UPDATE_CANDIDATE_FIELDS)
    prompt_lines.extend(
        [
            "",
            "## Required Commands",
            "",
            "- `python3 -m pytest -q tests/test_llm_kernels.py`",
            f"- `{status_command}`",
            "",
            "## Do Not Claim",
            "",
            "- Board-level ZCU104 signoff.",
            "- Real-time LLaMA inference performance.",
            "- Hardware lab runtime validation.",
            "- New HDL correctness beyond the already verified child evidence unless you create and verify a separate HDL sub-agent packet.",
            "",
        ]
    )
    return {
        "artifact": "full_llama_execution_evidence_agent_task",
        "task_id": "full_llama_execution_evidence",
        "spawn_kind": "target_evidence_agent",
        "agent": "Codex",
        "mode": "read_write_target_evidence",
        "ready_to_spawn": ready_to_spawn,
        "spawn_precondition_failures": precondition_failures,
        "prompt_file": "target_evidence_prompts/full_llama_execution_evidence_agent.md",
        "expected_evidence_file": str(collection_root / "full_llama_execution_evidence.json"),
        "expected_subagent_result": "build/full_llama_execution_gate/subagent_result.json",
        "source_readiness_status": full_execution_readiness.get("status"),
        "required_commands": [
            "python3 -m pytest -q tests/test_llm_kernels.py",
            status_command,
        ],
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required": True,
        "evidence_template": evidence_template,
        "prompt": "\n".join(prompt_lines),
        "does_not_claim": [
            "board_level_ZCU104_signoff",
            "real_time_LLaMA_inference_performance",
            "hardware_lab_runtime_validation",
        ],
    }


def build_target_evidence_execution_manifest(target_task: dict[str, Any]) -> dict[str, Any]:
    """Describe the next target-evidence Codex agent spawn without spawning it."""
    spawn_entries: list[dict[str, Any]] = []
    blocked_target_evidence_tasks = []
    task_id = target_task.get("task_id", "target_evidence")
    if target_task.get("ready_to_spawn") is True:
        spawn_kind = target_task.get("spawn_kind", "target_evidence_agent")
        entry = {
            "spawn_key": f"target_evidence::{task_id}",
            "spawn_kind": spawn_kind,
            "agent_hierarchy_role": "subagent",
            "subagent_type": (
                "board_wrapper_implementation_subagent"
                if spawn_kind == "target_evidence_implementation_agent"
                else "target_evidence_subagent"
            ),
            "subagent_may_spawn_subagents": False,
            "parent_feedback_channel": "feedback_packet.json",
            "agent": target_task.get("agent", "Codex"),
            "mode": target_task.get("mode", "read_write_target_evidence"),
            "task_id": task_id,
            "prompt_file": target_task.get("prompt_file"),
            "fork_context": True,
            "expected_evidence_file": target_task.get("expected_evidence_file"),
            "expected_subagent_result": target_task.get("expected_subagent_result"),
            "source_readiness_status": target_task.get("source_readiness_status"),
            "required_commands": target_task.get("required_commands", []),
            "parent_must_not_write_hdl": target_task.get("parent_must_not_write_hdl", True),
            "failure_to_skill_required": target_task.get("failure_to_skill_required", True),
            "final_response_required_fields": TARGET_EVIDENCE_RESULT_FIELDS,
            "codex_spawn_message": (
                "You are the target-evidence Codex sub-agent for this single gate. "
                f"Read `{target_task.get('prompt_file')}`, run the required checks, and write "
                f"`{target_task.get('expected_evidence_file')}` plus "
                f"`{target_task.get('expected_subagent_result')}`. "
                "If the gate fails, preserve evidence and return a complete skill_update_candidate."
            ),
        }
        spawn_entries.append(entry)
    else:
        blocked_target_evidence_tasks.append(
            {
                "task_id": task_id,
                "status": "blocked_by_precondition",
                "precondition_failures": target_task.get("spawn_precondition_failures", []),
                "next_action": "resolve preconditions before spawning target-evidence agent",
            }
        )
    spawn_batches = []
    if spawn_entries:
        spawn_batches.append(
            {
                "batch_id": f"{_slug(task_id)}__target_evidence_agent",
                "wave_id": "target_evidence",
                "spawn_kind": spawn_entries[0]["spawn_kind"],
                "parallel_spawn_allowed": False,
                "entry_count": len(spawn_entries),
                "task_ids": [entry["task_id"] for entry in spawn_entries],
                "prompt_files": [entry["prompt_file"] for entry in spawn_entries],
                "spawn_entries": spawn_entries,
            }
        )
    implementation_spawn_count = sum(
        1 for entry in spawn_entries if entry.get("spawn_kind") == "target_evidence_implementation_agent"
    )
    target_evidence_spawn_count = sum(
        1 for entry in spawn_entries if entry.get("spawn_kind") != "target_evidence_implementation_agent"
    )
    return {
        "artifact": "target_evidence_execution_manifest",
        "coverage_level": "target_readiness_gap_to_codex_evidence_agent_spawn",
        "agent_hierarchy": _parent_subagent_hierarchy(),
        "parent_agent_runtime_boundary": (
            "Package code emits target-evidence spawn instructions only; the interactive Codex parent "
            "or an external runner must call the sub-agent, deliver feedback packets, "
            "collect evidence, and decide retries."
        ),
        "spawn_entry_count": len(spawn_entries),
        "implementation_spawn_count": implementation_spawn_count,
        "verification_spawn_count": 0,
        "target_evidence_spawn_count": target_evidence_spawn_count,
        "spawn_batch_count": len(spawn_batches),
        "parallel_spawn_allowed": False,
        "max_parallel_batch_size": 0,
        "skill_update_required": False,
        "missing_skill_update_candidate": False,
        "spawn_batches": spawn_batches,
        "spawn_entries": spawn_entries,
        "blocked_waves": [],
        "blocked_target_evidence_tasks": blocked_target_evidence_tasks,
        "does_not_claim": [
            "sub-agent execution occurred",
            "automatic sub-agent spawning inside package runtime",
            "sub-agents spawned other sub-agents",
            "generated RTL completeness",
            "board-level ZCU104 signoff",
        ],
    }


def _load_board_signoff_evidence(collection_root: Path) -> dict[str, Any] | None:
    return _load_json(collection_root / "board_zcu104_signoff_evidence.json")


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) >= 0.0


def _board_signoff_evidence_failures(
    evidence: dict[str, Any] | None,
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
) -> list[str]:
    full_execution_failure = (
        ["full_llama_execution_readiness must be passed before board signoff"]
        if full_execution_readiness.get("status") != "passed"
        else []
    )
    if evidence is None:
        return ["board_zcu104_signoff_evidence.json not found"] + full_execution_failure
    if evidence.get("_load_error"):
        return [str(evidence["_load_error"])] + full_execution_failure
    failures = []
    hardware = dispatch_plan.get("hardware", {})
    if evidence.get("artifact") != "board_zcu104_signoff_evidence":
        failures.append("artifact must be board_zcu104_signoff_evidence")
    if evidence.get("status") != "passed":
        failures.append("status must be passed")
    if evidence.get("board") != "ZCU104":
        failures.append("board must be ZCU104")
    if evidence.get("fpga_part") != hardware.get("fpga_part"):
        failures.append("fpga_part must match dispatch plan")
    if evidence.get("full_llama_execution_status") != "passed":
        failures.append("full_llama_execution_status must be passed")
    failures.extend(full_execution_failure)
    required_constraints = {
        "clock",
        "reset",
        "board_io",
        "ps_pl_interface",
        "ddr_interface",
    }
    constraints = evidence.get("constraints")
    if not isinstance(constraints, dict):
        failures.append("constraints must be an object")
    else:
        missing = sorted(name for name in required_constraints if constraints.get(name) is not True)
        if missing:
            failures.append("constraints missing or false: " + ", ".join(missing))
    timing = evidence.get("timing")
    if not isinstance(timing, dict):
        failures.append("timing must be an object")
    else:
        if timing.get("constraints_met") is not True:
            failures.append("timing.constraints_met must be true")
        for field in TIMING_FIELDS:
            if not _nonnegative_number(timing.get(field)):
                failures.append(f"timing.{field} must be non-negative")
    resources = evidence.get("resource_utilization")
    if not isinstance(resources, dict):
        failures.append("resource_utilization must be an object")
    else:
        budget_fields = {
            "lut": "max_lut",
            "dsp": "max_dsp",
            "bram": "max_bram",
            "ff": "max_ff",
            "uram": "max_uram",
            "io": "max_io",
        }
        for used_field, max_field in budget_fields.items():
            used = resources.get(used_field)
            limit = hardware.get(max_field)
            if limit is None:
                continue
            if not isinstance(used, (int, float)):
                failures.append(f"resource_utilization.{used_field} must be numeric")
            elif isinstance(limit, (int, float)) and used > limit:
                failures.append(f"resource_utilization.{used_field} exceeds hardware.{max_field}")
    reports = evidence.get("reports")
    if not isinstance(reports, dict):
        failures.append("reports must list generated Vivado artifacts")
    else:
        for name in ("timing_summary", "utilization", "constraints", "vivado_log"):
            if not reports.get(name):
                failures.append(f"reports.{name} must be provided")
    if evidence.get("fixture_only") is True:
        failures.append("fixture_only board evidence cannot satisfy board-level signoff")
    return failures


def build_board_zcu104_signoff_readiness_report(
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    collection_root: Path,
) -> dict[str, Any]:
    evidence = _load_board_signoff_evidence(collection_root)
    evidence_failures = _board_signoff_evidence_failures(evidence, dispatch_plan, full_execution_readiness)
    status = "passed" if not evidence_failures else (
        "blocked_by_full_llama_execution"
        if full_execution_readiness.get("status") != "passed"
        else "blocked_by_missing_or_incomplete_board_signoff_evidence"
    )
    return {
        "artifact": "board_zcu104_signoff_readiness",
        "coverage_level": "parent_gate_for_board_level_zcu104_signoff_claim",
        "status": status,
        "collection_root": str(collection_root),
        "model": dispatch_plan.get("model", {}),
        "target": {
            "board": "ZCU104",
            "fpga_part": dispatch_plan.get("hardware", {}).get("fpga_part"),
            "quantization": dispatch_plan.get("optimization", {}).get("quantization"),
            "design_style": dispatch_plan.get("optimization", {}).get("design_style"),
            "compute_style": dispatch_plan.get("optimization", {}).get("compute_style"),
            "execution_style": dispatch_plan.get("optimization", {}).get("execution_style"),
            "memory_style": dispatch_plan.get("optimization", {}).get("memory_style"),
            "control_style": dispatch_plan.get("optimization", {}).get("control_style"),
        },
        "full_execution_status": full_execution_readiness.get("status"),
        "required_evidence_file": str(collection_root / "board_zcu104_signoff_evidence.json"),
        "board_signoff_evidence": evidence if isinstance(evidence, dict) and not evidence.get("_load_error") else None,
        "evidence_failures": evidence_failures,
        "safe_to_clear_board_level_zcu104_signoff_blocker": status == "passed",
        "next_action": (
            "pass full_llama_execution_readiness before board signoff"
            if full_execution_readiness.get("status") != "passed"
            else "write board_zcu104_signoff_evidence.json with Vivado board constraints, timing, resource, and report evidence"
            if evidence_failures
            else "board-level ZCU104 signoff readiness passed"
        ),
        "does_not_claim": [
            "real_time_LLaMA_inference_performance",
            "hardware_lab_runtime_validation",
        ],
    }


def build_board_zcu104_signoff_evidence_template(
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    collection_root: Path,
) -> dict[str, Any]:
    hardware = dispatch_plan.get("hardware", {})
    resource_template = {
        "lut": f"<number <= {hardware.get('max_lut')}>",
        "dsp": f"<number <= {hardware.get('max_dsp')}>",
        "bram": f"<number <= {hardware.get('max_bram')}>",
    }
    optional_resource_fields = {
        "ff": "max_ff",
        "uram": "max_uram",
        "io": "max_io",
    }
    for used_field, max_field in optional_resource_fields.items():
        if hardware.get(max_field) is not None:
            resource_template[used_field] = f"<number <= {hardware.get(max_field)}>"
    return {
        "artifact": "board_zcu104_signoff_evidence_template",
        "write_to": str(collection_root / "board_zcu104_signoff_evidence.json"),
        "purpose": "Evidence required before clearing board_level_zcu104_signoff.",
        "requires_full_llama_execution_readiness_status": "passed",
        "current_full_llama_execution_readiness_status": full_execution_readiness.get("status"),
        "required_fields": [
            "artifact",
            "status",
            "board",
            "fpga_part",
            "full_llama_execution_status",
            "constraints",
            "timing",
            "resource_utilization",
            "reports",
            "fixture_only",
        ],
        "template": {
            "artifact": "board_zcu104_signoff_evidence",
            "status": "<passed>",
            "board": "ZCU104",
            "fpga_part": hardware.get("fpga_part"),
            "full_llama_execution_status": "<passed>",
            "constraints": {
                "clock": "<true>",
                "reset": "<true>",
                "board_io": "<true>",
                "ps_pl_interface": "<true>",
                "ddr_interface": "<true>",
            },
            "timing": {
                "constraints_met": "<true>",
                "setup_worst_slack_ns": "<non-negative number>",
                "hold_worst_slack_ns": "<non-negative number>",
                "pulse_width_worst_slack_ns": "<non-negative number>",
            },
            "resource_utilization": resource_template,
            "reports": {
                "timing_summary": "<path to timing_summary.rpt>",
                "utilization": "<path to utilization.rpt>",
                "constraints": "<path to ZCU104 XDC/TCL constraints>",
                "vivado_log": "<path to Vivado log>",
            },
            "fixture_only": False,
        },
        "does_not_claim": [
            "real_time_LLaMA_inference_performance",
            "hardware_lab_runtime_validation",
        ],
    }


def build_board_zcu104_signoff_evidence_agent_task(
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    board_signoff_readiness: dict[str, Any],
    evidence_template: dict[str, Any],
    collection_root: Path,
    dispatch_plan_path: str | None = None,
) -> dict[str, Any]:
    evidence_failures = board_signoff_readiness.get("evidence_failures", [])
    precondition_failures = []
    if full_execution_readiness.get("status") != "passed":
        precondition_failures.append("full_llama_execution_readiness must be passed before board signoff")
    if board_signoff_readiness.get("status") == "passed":
        precondition_failures.append("board_zcu104_signoff_readiness already passed")
    if "board_zcu104_signoff_evidence.json not found" not in evidence_failures and board_signoff_readiness.get(
        "status"
    ) != "blocked_by_missing_or_incomplete_board_signoff_evidence":
        precondition_failures.append("board signoff readiness must be waiting on board evidence")
    ready_to_spawn = not precondition_failures
    model = dispatch_plan.get("model", {})
    hardware = dispatch_plan.get("hardware", {})
    optimization = dispatch_plan.get("optimization", {})
    target_clock_mhz = hardware.get("target_clock_mhz")
    target_clock_period_ns = None
    if isinstance(target_clock_mhz, (int, float)) and target_clock_mhz > 0:
        target_clock_period_ns = 1000.0 / float(target_clock_mhz)
    dispatch_arg = dispatch_plan_path or "<path to hdl_subagent_dispatch_plan.json>"
    status_command = (
        "python3 -m nl2hdl subagents status "
        f"--dispatch-plan {shlex.quote(str(dispatch_arg))} "
        f"--evidence-root {shlex.quote(str(collection_root))} "
        "--out build/board_zcu104_signoff_gate/status_after"
    )
    prompt_lines = [
        "# Target Evidence Sub-Agent Task: board_zcu104_signoff",
        "",
        "You are the Codex target-evidence sub-agent for board-level ZCU104 signoff.",
        "The parent agent must not hand-write HDL or fabricate board evidence. Your job is to inspect or run the required Vivado/board-constraint flow and write board evidence only if the gate is actually proven.",
        "",
        "## Target Context",
        "",
        f"- Model: `{model.get('name')}`",
        "- Board: `ZCU104`",
        f"- FPGA part: `{hardware.get('fpga_part')}`",
        f"- Target clock: `{target_clock_mhz} MHz`",
        f"- Target clock period: `{target_clock_period_ns:.3f} ns`" if target_clock_period_ns else "- Target clock period: `<unknown>`",
        f"- Resource budgets: LUT `{hardware.get('max_lut')}`, FF `{hardware.get('max_ff')}`, DSP `{hardware.get('max_dsp')}`, BRAM `{hardware.get('max_bram')}`, URAM `{hardware.get('max_uram')}`",
        f"- Quantization: `{optimization.get('quantization')}`",
        f"- Design style alias: `{optimization.get('design_style')}`",
        f"- Compute style: `{optimization.get('compute_style')}`",
        f"- Execution style: `{optimization.get('execution_style')}`",
        f"- Memory style: `{optimization.get('memory_style')}`",
        f"- Control style: `{optimization.get('control_style')}`",
        f"- Collection root: `{collection_root}`",
        f"- Dispatch plan: `{dispatch_arg}`",
        "",
        "## Preconditions",
        "",
        f"- Ready to spawn: `{ready_to_spawn}`",
        f"- Full execution readiness status: `{full_execution_readiness.get('status')}`",
        f"- Board readiness status: `{board_signoff_readiness.get('status')}`",
        f"- Current board evidence failures: `{evidence_failures}`",
        "",
        "If `Ready to spawn` is false, do not force the gate. Return a precise missing-evidence report or a complete `skill_update_candidate`.",
        "",
        "## Required Output",
        "",
        f"- Write `{evidence_template['write_to']}` only after every required field below is backed by real artifacts.",
        "- Also write `build/board_zcu104_signoff_gate/subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.",
        "- Do not write `fixture_only: true`; fixture-only board evidence cannot satisfy this gate.",
        "",
        "## Evidence Schema",
        "",
        f"- Required fields: `{', '.join(evidence_template.get('required_fields', []))}`",
        f"- Template artifact: `{evidence_template.get('template', {}).get('artifact')}`",
        f"- Template JSON: `{evidence_template.get('template')}`",
        "",
        "## Evidence That Must Be Proven",
        "",
        "- `full_llama_execution_status == passed` from parent readiness.",
        "- Board is `ZCU104` and FPGA part matches `xczu7ev-ffvc1156-2-e`.",
        "- Constraints explicitly cover clock, reset, board I/O, PS/PL interface, and DDR interface.",
        "- Vivado timing constraints are met and setup, hold, and pulse-width worst slack are all non-negative.",
        "- Implemented `report_clocks`, timing summary, and XDC prove the accelerator/PS PL clock period is at or below the configured target period.",
        "- Resource utilization reports numeric LUT/DSP/BRAM usage under configured budgets.",
        "- Reports include timing summary, utilization, constraints, and Vivado log paths.",
        "- Existing bounded fixture synthesis reports are insufficient unless they include the required board I/O, PS/PL, and DDR constraints.",
        "",
        "## Allowed Write Scope",
        "",
        "- You may write `build/board_zcu104_signoff_evidence.json` and files under `build/board_zcu104_signoff_gate/`.",
        "- You may add or update focused non-HDL scripts/tests that parse or validate board signoff evidence.",
        "- If RTL or SystemVerilog changes are necessary, stop and return a `skill_update_candidate`; board signoff should be retried through a scoped HDL/board-shell sub-agent.",
        "- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.",
        "- Do not weaken existing tests, evidence gates, blocked-target language, or full-execution/board-signoff separation.",
        "",
        "## Failure-To-SKILL Candidate",
        "",
        "- If you cannot prove board signoff, preserve evidence and return a `skill_update_candidate` with:",
    ]
    prompt_lines.extend(f"  - `{field}`" for field in SKILL_UPDATE_CANDIDATE_FIELDS)
    prompt_lines.extend(
        [
            "",
            "## Required Commands",
            "",
            "- `python3 -m pytest -q tests/test_llm_kernels.py`",
            f"- `{status_command}`",
            "",
            "## Do Not Claim",
            "",
            "- Real-time LLaMA inference performance.",
            "- Hardware lab runtime validation unless actual hardware-run evidence is present.",
            "- Board-level signoff from bounded fixture-only synthesis reports.",
            "- New HDL correctness beyond already verified child evidence.",
            "",
        ]
    )
    return {
        "artifact": "board_zcu104_signoff_evidence_agent_task",
        "task_id": "board_zcu104_signoff_evidence",
        "spawn_kind": "target_evidence_agent",
        "agent": "Codex",
        "mode": "read_write_target_evidence",
        "ready_to_spawn": ready_to_spawn,
        "spawn_precondition_failures": precondition_failures,
        "prompt_file": "target_evidence_prompts/board_zcu104_signoff_evidence_agent.md",
        "expected_evidence_file": str(collection_root / "board_zcu104_signoff_evidence.json"),
        "expected_subagent_result": "build/board_zcu104_signoff_gate/subagent_result.json",
        "source_readiness_status": board_signoff_readiness.get("status"),
        "required_commands": [
            "python3 -m pytest -q tests/test_llm_kernels.py",
            status_command,
        ],
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required": True,
        "evidence_template": evidence_template,
        "prompt": "\n".join(prompt_lines),
        "does_not_claim": [
            "real_time_LLaMA_inference_performance",
            "hardware_lab_runtime_validation_without_hardware_evidence",
            "board_level_signoff_from_fixture_only_synthesis",
        ],
    }


def build_zcu104_board_wrapper_axi_bridge_agent_task(
    dispatch_plan: dict[str, Any],
    full_execution_readiness: dict[str, Any],
    board_signoff_readiness: dict[str, Any],
    collection_root: Path,
    dispatch_plan_path: str | None = None,
) -> dict[str, Any]:
    board_gate_dir = collection_root / "board_zcu104_signoff_gate"
    pre_signoff_report = board_gate_dir / "zcu104_board_shell_signoff_readiness_report.json"
    signoff_gap_report = board_gate_dir / "evidence_gap_report.json"
    expected_report = board_gate_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
    expected_subagent_result = board_gate_dir / "zcu104_board_wrapper_axi_bridge_subagent_result.json"
    pre_signoff = _load_json(pre_signoff_report) if pre_signoff_report.exists() else None
    signoff_gap = _load_json(signoff_gap_report) if signoff_gap_report.exists() else None
    signoff_gap_status = signoff_gap.get("status") if isinstance(signoff_gap, dict) else None
    signoff_gap_candidate = (
        signoff_gap.get("skill_update_candidate") if isinstance(signoff_gap, dict) else None
    )
    precondition_failures = []
    if full_execution_readiness.get("status") != "passed":
        precondition_failures.append("full_llama_execution_readiness must be passed before board-wrapper implementation")
    if board_signoff_readiness.get("status") == "passed":
        precondition_failures.append("board_zcu104_signoff_readiness already passed")
    if board_signoff_readiness.get("status") != "blocked_by_missing_or_incomplete_board_signoff_evidence":
        precondition_failures.append("board signoff readiness must still be blocked on board evidence")
    if not isinstance(pre_signoff, dict) or pre_signoff.get("status") != "blocked_pending_vivado_board_integration_run":
        precondition_failures.append("zcu104 pre-signoff constraints package must exist before board-wrapper implementation")
    ready_to_spawn = not precondition_failures
    model = dispatch_plan.get("model", {})
    hardware = dispatch_plan.get("hardware", {})
    optimization = dispatch_plan.get("optimization", {})
    target_clock_mhz = hardware.get("target_clock_mhz")
    target_clock_period_ns = None
    if isinstance(target_clock_mhz, (int, float)) and target_clock_mhz > 0:
        target_clock_period_ns = 1000.0 / float(target_clock_mhz)
    dispatch_arg = dispatch_plan_path or "<path to hdl_subagent_dispatch_plan.json>"
    status_command = (
        "python3 -m nl2hdl subagents status "
        f"--dispatch-plan {shlex.quote(str(dispatch_arg))} "
        f"--evidence-root {shlex.quote(str(collection_root))} "
        "--out build/board_zcu104_signoff_gate/status_after_board_wrapper"
    )
    prompt_lines = [
        "# Implementation Sub-Agent Task: zcu104_board_wrapper_axi_bridge",
        "",
        "You are the Codex implementation sub-agent for the next ZCU104 board-signoff step.",
        "The parent agent must not hand-write HDL. Your job is to implement the missing board wrapper, accelerator bridge, Vivado block-design wiring, and routed-report flow needed before any evidence-only signoff agent can run.",
        "",
        "## Target Context",
        "",
        f"- Model: `{model.get('name')}`",
        "- Board: `ZCU104`",
        f"- FPGA part: `{hardware.get('fpga_part')}`",
        f"- Target clock: `{target_clock_mhz} MHz`",
        f"- Target clock period: `{target_clock_period_ns:.3f} ns`" if target_clock_period_ns else "- Target clock period: `<unknown>`",
        f"- Resource budgets: LUT `{hardware.get('max_lut')}`, FF `{hardware.get('max_ff')}`, DSP `{hardware.get('max_dsp')}`, BRAM `{hardware.get('max_bram')}`, URAM `{hardware.get('max_uram')}`",
        f"- Quantization: `{optimization.get('quantization')}`",
        f"- Design style alias: `{optimization.get('design_style')}`",
        f"- Compute style: `{optimization.get('compute_style')}`",
        f"- Execution style: `{optimization.get('execution_style')}`",
        f"- Memory style: `{optimization.get('memory_style')}`",
        f"- Control style: `{optimization.get('control_style')}`",
        f"- Collection root: `{collection_root}`",
        f"- Dispatch plan: `{dispatch_arg}`",
        f"- Pre-signoff report: `{pre_signoff_report}`",
        f"- Prior board-signoff gap report: `{signoff_gap_report}`",
        f"- Prior board-signoff gap status: `{signoff_gap_status}`",
        "",
        "## Preconditions",
        "",
        f"- Ready to spawn: `{ready_to_spawn}`",
        f"- Full execution readiness status: `{full_execution_readiness.get('status')}`",
        f"- Board readiness status: `{board_signoff_readiness.get('status')}`",
        f"- Spawn precondition failures: `{precondition_failures}`",
        "",
        "If `Ready to spawn` is false, do not force implementation. Write the sub-agent result with the missing preconditions and a `skill_update_candidate` if the failure pattern is reusable.",
        "",
        "## Required Implementation",
        "",
        "- Generate or update a compact PL subsystem such as `zcu104_board_shell_top` with board-visible status and a child accelerator start/done path; do not require this direct PL shell to be the final routed board-signoff top.",
        "- Add an accelerator AXI-lite or AXI-stream bridge stub that exposes a realistic PS-controlled register/address-map path without widening board-level ports.",
        "- Update the ZCU104 BD TCL so the Zynq UltraScale+ PS, reset block, wrapper, and AXI interconnect/address map are generated and validated.",
        "- Update the Vivado route-check TCL so it reads all generated HDL/BD/XDC inputs, runs synth/place/route, and writes timing, utilization, methodology/constraints, checkpoint, and Vivado log artifacts.",
        "- If a prior board-signoff gap report exists, treat it as the retry target. In particular, do not route only the direct PL shell with package-level `aclk`/`aresetn` while the PS/PL/DDR BD is generated as a side artifact.",
        "- If the prior board-signoff gap report mentions a target clock mismatch, treat it as a retry target too. Positive timing/resource reports are not enough when Vivado implements `clk_pl_0` at 5.625 ns / 177.778 MHz for a 200 MHz target.",
        "- The corrected routed design must use the generated PS/PL/DDR BD wrapper, or an equivalent top where PS FCLK/reset drive the accelerator internally and PS AXI reaches the accelerator control path.",
        "- The corrected DRC evidence must have no NSTD-1 or UCIO-1 critical warnings. Positive timing/resource reports are not enough if `aclk` or `aresetn` remain unconstrained/default-standard top-level ports.",
        "- The corrected clock evidence must prove the configured target clock from raw `report_clocks`, `report_timing_summary`, and the implemented XDC. For a 200 MHz target, the accelerator/PS PL clock period must be <= 5.000 ns.",
        "- Run Vivado when feasible. If Vivado cannot complete in this environment, preserve the exact command/log failure and do not write board signoff evidence.",
        "",
        "## Required Output",
        "",
        f"- Write `{expected_report}` with status `passed` only if the board-wrapper flow and routed report bundle are actually produced.",
        f"- Always write `{expected_subagent_result}` with changed files, commands run, evidence paths, and remaining risks.",
        "- Do not write `build/board_zcu104_signoff_evidence.json`; that file belongs to a later evidence-only agent after this implementation report passes.",
        "",
        "## Allowed Write Scope",
        "",
        "- You may edit focused generator/source/test files needed for this board wrapper flow, including `nl2hdl/llm_kernels.py` and `tests/test_llm_kernels.py`.",
        "- You may write generated HDL/TCL/XDC/log/report artifacts under `build/board_zcu104_signoff_gate/`.",
        "- You may add focused docs/contracts for the board wrapper if the interface contract would otherwise be ambiguous.",
        "- Do not edit SKILL files, parent orchestration policy, unrelated kernels, or existing passed evidence files.",
        "",
        "## Pass Criteria",
        "",
        "- The implementation report distinguishes scaffold-only, synthesis-only, and routed board-wrapper evidence.",
        "- The implementation report records whether the routed top is the generated PS/PL/DDR wrapper or equivalent, not just the direct PL shell.",
        "- The implementation report records DRC status and explicitly fails if NSTD-1 or UCIO-1 critical warnings remain.",
        "- The implementation report records target clock MHz, observed accelerator clock name, observed clock period/frequency, and fails if the observed clock period is greater than the target period.",
        "- The implementation report records that PS FCLK/reset, PS AXI, and DDR/address-map evidence are present in the implemented hierarchy.",
        "- Reported timing has non-negative setup, hold, and pulse-width slack if route completes.",
        "- Reported LUT/DSP/BRAM usage is numeric and checked against the configured budgets.",
        "- The flow still does not claim hardware lab runtime validation or real-time LLaMA inference.",
        "",
        "## Failure-To-SKILL Candidate",
        "",
        "- If you cannot produce a routed board-wrapper report bundle, preserve logs and return a `skill_update_candidate` with:",
    ]
    prompt_lines.extend(f"  - `{field}`" for field in SKILL_UPDATE_CANDIDATE_FIELDS)
    prompt_lines.extend(
        [
            "",
            "## Required Commands",
            "",
            "- `python3 -m compileall nl2hdl/llm_kernels.py nl2hdl/subagent_tasks.py nl2hdl/cli.py`",
            "- `python3 -m pytest -q tests/test_llm_kernels.py`",
            "- `vivado -version`",
            f"- `{status_command}`",
            "",
            "## Do Not Claim",
            "",
            "- Board-level ZCU104 signoff unless the later evidence-only gate writes `board_zcu104_signoff_evidence.json`.",
            "- Hardware lab runtime validation.",
            "- Real-time LLaMA inference performance.",
            "",
        ]
    )
    return {
        "artifact": "zcu104_board_wrapper_axi_bridge_agent_task",
        "task_id": "zcu104_board_wrapper_axi_bridge",
        "spawn_kind": "target_evidence_implementation_agent",
        "agent": "Codex",
        "mode": "read_write_board_wrapper_vivado_implementation",
        "ready_to_spawn": ready_to_spawn,
        "spawn_precondition_failures": precondition_failures,
        "prompt_file": "board_implementation_prompts/zcu104_board_wrapper_axi_bridge_agent.md",
        "expected_evidence_file": str(expected_report),
        "expected_subagent_result": str(expected_subagent_result),
        "source_readiness_status": board_signoff_readiness.get("status"),
        "prior_board_signoff_gap_report": str(signoff_gap_report),
        "prior_board_signoff_gap_status": signoff_gap_status,
        "prior_board_signoff_skill_update_candidate": signoff_gap_candidate,
        "required_commands": [
            "python3 -m compileall nl2hdl/llm_kernels.py nl2hdl/subagent_tasks.py nl2hdl/cli.py",
            "python3 -m pytest -q tests/test_llm_kernels.py",
            "vivado -version",
            status_command,
        ],
        "parent_must_not_write_hdl": True,
        "failure_to_skill_required": True,
        "prompt": "\n".join(prompt_lines),
        "does_not_claim": [
            "board_level_ZCU104_signoff",
            "hardware_lab_runtime_validation",
            "real_time_LLaMA_inference_performance",
        ],
    }


def build_hdl_subagent_skill_update_draft(
    dispatch_plan: dict[str, Any],
    collection_root: Path,
    target_skill: str = "hdl-kernel-contract-gates",
) -> dict[str, Any]:
    """Collect failed sub-agent lessons that are ready to become Skill updates."""
    wave_status = build_hdl_subagent_wave_status(dispatch_plan, collection_root)
    dispatch_waves = {
        wave["wave_id"]: wave
        for wave in dispatch_plan.get("waves", [])
        if isinstance(wave, dict) and isinstance(wave.get("wave_id"), str)
    }
    candidates: list[dict[str, Any]] = []
    for wave in wave_status.get("waves", []):
        if wave.get("status") == "failed_waiting_for_skill_update":
            dispatch_wave = dispatch_waves.get(wave["wave_id"], {})
            tasks_by_id = {
                task["task_id"]: task
                for task in dispatch_wave.get("implementation_tasks", [])
                if isinstance(task, dict) and isinstance(task.get("task_id"), str)
            }
            for result in wave.get("task_results", []):
                if result.get("status") != "failed_with_skill_candidate":
                    continue
                task_id = result["task_id"]
                task = tasks_by_id.get(task_id, {"task_id": task_id})
                evidence_dir = _task_evidence_dir(task, collection_root)
                candidate = _load_task_skill_update_candidate(evidence_dir)
                if candidate is None:
                    continue
                candidates.append(
                    {
                        "wave_id": wave["wave_id"],
                        "task_id": task_id,
                        "agent_role": task.get("agent_role"),
                        "current_regression_kernel": task.get("current_regression_kernel"),
                        "evidence_dir": str(evidence_dir),
                        "target_skill": target_skill,
                        "candidate": candidate,
                    }
                )
        elif wave.get("status") == "failed_verification_waiting_for_skill_update":
            dispatch_wave = dispatch_waves.get(wave["wave_id"], {})
            if not isinstance(dispatch_wave.get("verification_agent"), dict):
                continue
            report_path = _verification_report_path(dispatch_wave, collection_root)
            report = _load_json(report_path)
            if not isinstance(report, dict) or report.get("_load_error"):
                continue
            for index, candidate_entry in enumerate(_verification_skill_update_candidates(report)):
                candidates.append(
                    {
                        "wave_id": wave["wave_id"],
                        "task_id": f"{wave['wave_id']}__verification_{index}",
                        "agent_role": "read_only_verification_agent",
                        "current_regression_kernel": "verification_audit",
                        "evidence_dir": str(report_path.parent),
                        "verification_report": str(report_path),
                        "target_skill": target_skill,
                        "candidate_source": candidate_entry.get("source"),
                        "verification_finding": {
                            key: candidate_entry.get(key)
                            for key in ("severity", "title", "body")
                            if candidate_entry.get(key) is not None
                        },
                        "candidate": candidate_entry["candidate"],
                    }
                )
    return {
        "artifact": "hdl_subagent_skill_update_draft",
        "coverage_level": "failed_subagent_candidate_to_skill_update_draft",
        "status": "skill_update_required" if candidates else "no_complete_skill_update_candidates",
        "collection_root": str(collection_root),
        "target_skill": target_skill,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "wave_status_summary": {
            "wave_count": wave_status["wave_count"],
            "failed_waiting_for_skill_update_count": sum(
                1 for wave in wave_status["waves"] if wave["status"] == "failed_waiting_for_skill_update"
            ),
            "failed_missing_skill_candidate_count": sum(
                1 for wave in wave_status["waves"] if wave["status"] == "failed_missing_skill_candidate"
            ),
        },
        "next_action": (
            "append_rules_to_target_skill_before_retry"
            if candidates
            else "collect_complete_skill_update_candidate_from_failed_subagent"
        ),
        "does_not_claim": [
            "Skill files were edited automatically",
            "sub-agent retry occurred",
            "failed HDL gate is fixed",
            "generated RTL completeness",
        ],
    }


def build_skill_update_draft_markdown(draft: dict[str, Any]) -> str:
    lines = [
        "# HDL Sub-Agent Failure Skill Update Draft",
        "",
        "This file is parent-owned evidence for the failure-to-SKILL loop.",
        "Review the entries and update the target SKILL.md before retrying the same HDL pattern.",
        "",
        f"- Status: `{draft.get('status')}`",
        f"- Target skill: `{draft.get('target_skill')}`",
        f"- Candidate count: `{draft.get('candidate_count', 0)}`",
        f"- Collection root: `{draft.get('collection_root')}`",
        "",
    ]
    candidates = draft.get("candidates", [])
    if not candidates:
        lines.extend(
            [
                "## No Complete Candidates",
                "",
                "No failed sub-agent evidence contains all required skill update fields yet.",
                "Ask the failed HDL sub-agent to preserve the failing command, symptom, root-cause hypothesis, prevention rule, and minimal regression check.",
                "",
            ]
        )
    for entry in candidates:
        candidate = entry["candidate"]
        lines.extend(
            [
                f"## `{entry['task_id']}`",
                "",
                f"- Wave: `{entry['wave_id']}`",
                f"- Agent role: `{entry.get('agent_role') or 'unknown'}`",
                f"- Kernel: `{entry.get('current_regression_kernel') or 'unknown'}`",
                f"- Evidence dir: `{entry['evidence_dir']}`",
                "",
                "Suggested SKILL.md rule:",
                "",
                (
                    f"- When `{candidate['failing_command']}` fails with `{candidate['symptom']}`, "
                    f"assume `{candidate['root_cause_hypothesis']}` until disproven. "
                    f"Before retrying, enforce: {candidate['prevention_rule']} "
                    f"Regression check: `{candidate['minimal_regression_check']}`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Does Not Claim",
            "",
        ]
    )
    for claim in draft.get("does_not_claim", []):
        lines.append(f"- {claim}")
    lines.append("")
    return "\n".join(lines)


def write_hdl_subagent_skill_update_draft(
    dispatch_plan: dict[str, Any],
    collection_root: Path,
    out_dir: Path,
    target_skill: str = "hdl-kernel-contract-gates",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    draft = build_hdl_subagent_skill_update_draft(dispatch_plan, collection_root, target_skill)
    json_path = out_dir / "skill_update_candidates.json"
    markdown_path = out_dir / "skill_update_draft.md"
    json_path.write_text(json.dumps(draft, indent=2), encoding="utf-8")
    markdown_path.write_text(build_skill_update_draft_markdown(draft), encoding="utf-8")
    return json_path, markdown_path


def _wave_status_by_id(wave_status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        wave["wave_id"]: wave
        for wave in wave_status.get("waves", [])
        if isinstance(wave, dict) and isinstance(wave.get("wave_id"), str)
    }


def _parent_subagent_hierarchy() -> dict[str, Any]:
    return {
        "parent_agent": "single_orchestrator",
        "all_non_parent_workers_are_subagents": True,
        "subagents_may_spawn_subagents": False,
        "parent_owns_feedback_loop": True,
        "parent_feedback_artifacts": [
            "parent_loop_state.json",
            "feedback_packet.json",
            "retry_plan.json",
        ],
        "subagent_result_returns_to": "parent_agent",
    }


def _execution_entry_for_task(wave: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "spawn_key": f"{wave['wave_id']}::implementation::{task['task_id']}",
        "spawn_kind": "implementation_agent",
        "agent_hierarchy_role": "subagent",
        "subagent_type": "hdl_implementation_subagent",
        "subagent_may_spawn_subagents": False,
        "parent_feedback_channel": "feedback_packet.json",
        "agent": "Codex",
        "mode": "read_write_hdl_packet",
        "wave_id": wave["wave_id"],
        "task_id": task["task_id"],
        "agent_role": task["agent_role"],
        "prompt_file": task["prompt_file"],
        "fork_context": True,
        "expected_evidence_dir": task["expected_evidence_dir"],
        "expected_kernel_report": task["expected_kernel_report"],
        "expected_module_ooc_synthesis_report": task.get("expected_module_ooc_synthesis_report"),
        "expected_subagent_result": f"{task['expected_evidence_dir']}/subagent_result.json",
        "current_regression_kernel": task["current_regression_kernel"],
        "source_replay": task.get("source_replay", {}),
        "hardware": task.get("hardware", {}),
        "required_commands": task.get("required_commands", []),
        "wave_blocked_target_dependencies": wave.get("blocked_target_dependencies", []),
        "wave_global_blocked_target_dependencies": wave.get("global_blocked_target_dependencies", []),
        "module_contract": task["module_contract"],
        "must_self_verify": True,
        "parent_must_not_write_hdl": True,
        "final_response_required_fields": SUBAGENT_RESULT_FIELDS,
        "failure_to_skill_required": True,
        "codex_spawn_message": (
            "You are the HDL implementation sub-agent for this single packet. "
            f"Read this execution manifest's sibling prompt file `{task['prompt_file']}`, "
            "edit only the allowed write scope, run the required checks, and write "
            f"`{task['expected_evidence_dir']}/kernel_report.json` plus "
            f"`{task['expected_evidence_dir']}/module_ooc_synthesis_report.json` when this is a real datapath module, and "
            f"`{task['expected_evidence_dir']}/subagent_result.json`. "
            "If the gate fails, preserve evidence and return a complete skill_update_candidate."
        ),
    }
    if task.get("semantic_op") is not None:
        entry["semantic_op"] = task["semantic_op"]
    if task.get("partition") is not None:
        entry["partition"] = task["partition"]
    if task.get("expected_projection_shape") is not None:
        entry["expected_projection_shape"] = task["expected_projection_shape"]
        entry["packed_int4_bytes"] = task.get("packed_int4_bytes")
        entry["memory_beats"] = task.get("memory_beats")
    return entry


def _execution_entry_for_verification(wave: dict[str, Any]) -> dict[str, Any]:
    runs_integration_synthesis = bool(wave["verification_agent"].get("runs_integration_synthesis"))
    entry = {
        "spawn_key": f"{wave['wave_id']}::verification::{wave['wave_id']}",
        "spawn_kind": "verification_agent",
        "agent_hierarchy_role": "subagent",
        "subagent_type": "verification_subagent",
        "subagent_may_spawn_subagents": False,
        "parent_feedback_channel": "feedback_packet.json",
        "agent": "Codex",
        "mode": wave["verification_agent"].get("mode", "read_only"),
        "wave_id": wave["wave_id"],
        "prompt_file": wave["verification_agent"]["prompt_file"],
        "fork_context": True,
        "verification_report": f"verification_results/{_slug(wave['wave_id'])}__verification.json",
        "source_replay": wave.get("source_replay", {}),
        "wave_blocked_target_dependencies": wave.get("blocked_target_dependencies", []),
        "wave_global_blocked_target_dependencies": wave.get("global_blocked_target_dependencies", []),
        "must_not_edit_source_files": True,
        "may_write_generated_evidence": runs_integration_synthesis,
        "runs_integration_synthesis": runs_integration_synthesis,
        "blocking_findings": ["P0", "P1", "P2"],
        "codex_spawn_message": (
            "You are the Codex verification sub-agent for this wave. "
            f"Read this execution manifest's sibling prompt file `{wave['verification_agent']['prompt_file']}`, "
            "do not edit source/RTL/test files, audit the collected implementation evidence, "
            "run or inspect integration synthesis when the prompt requires it, and write "
            f"`verification_results/{_slug(wave['wave_id'])}__verification.json` with findings."
        ),
    }
    if runs_integration_synthesis:
        entry["expected_integration_synthesis_dir"] = wave["verification_agent"][
            "expected_integration_synthesis_dir"
        ]
        entry["expected_integration_synthesis_report"] = wave["verification_agent"][
            "expected_integration_synthesis_report"
        ]
    return entry


def _ledger_records_by_key(existing_ledger: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(existing_ledger, dict):
        return {}
    return {
        record["spawn_key"]: record
        for record in existing_ledger.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("spawn_key"), str)
    }


def _parse_agent_records(records: list[str] | None) -> dict[str, str]:
    parsed = {}
    for item in records or []:
        if "=" not in item:
            raise ValueError(f"agent record must use spawn_key=agent_id: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"agent record must use non-empty spawn_key=agent_id: {item}")
        parsed[key] = value
    return parsed


def _wave_status_task_results(wave_status: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(wave_status, dict):
        return {}
    results = {}
    for wave in wave_status.get("waves", []):
        if not isinstance(wave, dict) or not isinstance(wave.get("wave_id"), str):
            continue
        for result in wave.get("task_results", []):
            if isinstance(result, dict) and isinstance(result.get("task_id"), str):
                results[(wave["wave_id"], result["task_id"])] = result
    return results


def _wave_status_verification_results(wave_status: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(wave_status, dict):
        return {}
    return {
        wave["wave_id"]: wave.get("verification", {})
        for wave in wave_status.get("waves", [])
        if isinstance(wave, dict) and isinstance(wave.get("wave_id"), str)
    }


def _record_status_from_evidence(
    record: dict[str, Any],
    task_results: dict[tuple[str, str], dict[str, Any]],
    verification_results: dict[str, dict[str, Any]],
) -> tuple[str, str | None]:
    spawn_kind = record.get("spawn_kind")
    wave_id = record.get("wave_id")
    if spawn_kind == "implementation_agent" and wave_id and record.get("task_id"):
        result = task_results.get((wave_id, record["task_id"]))
        if not result:
            return record["spawn_status"], None
        status = result.get("status")
        if status == "passed":
            return "evidence_passed", result.get("kernel_report")
        if status == "missing":
            return record["spawn_status"], result.get("reason")
        if status == "incomplete_subagent_result":
            return "evidence_incomplete_subagent_result", result.get("reason")
        if status == "failed_with_skill_candidate":
            return "evidence_failed_waiting_for_skill_update", result.get("reason")
        if status == "failed_missing_skill_candidate":
            return "evidence_failed_missing_skill_candidate", result.get("reason")
        return f"evidence_{status}", result.get("reason") if isinstance(status, str) else None
    if spawn_kind == "verification_agent" and wave_id:
        verification = verification_results.get(wave_id)
        if not verification:
            return record["spawn_status"], None
        status = verification.get("status")
        if status == "passed":
            return "evidence_passed", verification.get("verification_report")
        if status == "missing":
            return record["spawn_status"], verification.get("reason")
        if status == "failed" and verification.get("skill_update_candidate_complete"):
            return "evidence_failed_waiting_for_skill_update", verification.get("reason")
        if status == "failed":
            return "evidence_failed_missing_skill_candidate", verification.get("reason")
        return f"evidence_{status}", verification.get("reason") if isinstance(status, str) else None
    if spawn_kind in {"target_evidence_agent", "target_evidence_implementation_agent"}:
        evidence_file = record.get("expected_evidence_file")
        if not evidence_file:
            return "evidence_incomplete_subagent_result", "expected_evidence_file is missing"
        evidence = _load_json(Path(evidence_file))
        if evidence is None:
            result_file = record.get("expected_subagent_result")
            if result_file:
                result = _load_json(Path(result_file))
                if isinstance(result, dict) and not result.get("_load_error"):
                    if result.get("status") != "passed" and _candidate_complete(result.get("skill_update_candidate")):
                        return "evidence_failed_waiting_for_skill_update", result_file
                    required_fields = ["changed_files", "commands_run", "evidence_paths", "remaining_risks"]
                    if all(result.get(field) for field in required_fields):
                        return "evidence_incomplete_subagent_result", "expected_evidence_file is missing"
            return record["spawn_status"], None
        if evidence.get("_load_error"):
            return "evidence_failed_missing_skill_candidate", str(evidence["_load_error"])
        if evidence.get("status") != "passed":
            if _candidate_complete(evidence.get("skill_update_candidate")):
                return "evidence_failed_waiting_for_skill_update", evidence_file
            return "evidence_failed_missing_skill_candidate", "target evidence status is not passed"
        result_file = record.get("expected_subagent_result")
        if not result_file:
            return "evidence_incomplete_subagent_result", "expected_subagent_result is missing"
        result = _load_json(Path(result_file))
        if result is None:
            return "evidence_incomplete_subagent_result", "target evidence subagent_result.json not found"
        if result.get("_load_error"):
            return "evidence_incomplete_subagent_result", str(result["_load_error"])
        missing_fields = [
            field for field in TARGET_EVIDENCE_RESULT_FIELDS if not result.get(field)
        ]
        if result.get("task_id") not in {None, record.get("task_id")}:
            missing_fields.append("task_id_matches_expected_task")
        if missing_fields:
            if _candidate_complete(result.get("skill_update_candidate")):
                return "evidence_failed_waiting_for_skill_update", result_file
            return "evidence_incomplete_subagent_result", "missing fields: " + ", ".join(missing_fields)
        return "evidence_passed", evidence_file
    return record["spawn_status"], None


def build_hdl_subagent_spawn_ledger(
    execution_manifest: dict[str, Any],
    existing_ledger: dict[str, Any] | None = None,
    agent_records: dict[str, str] | None = None,
    wave_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a parent-owned ledger for externally spawned Codex sub-agents."""
    previous = _ledger_records_by_key(existing_ledger)
    agent_records = agent_records or {}
    task_results = _wave_status_task_results(wave_status)
    verification_results = _wave_status_verification_results(wave_status)
    entries_by_key: dict[str, dict[str, Any]] = {}
    for key, prior in previous.items():
        entries_by_key[key] = dict(prior)
    for entry in execution_manifest.get("spawn_entries", []):
        if not isinstance(entry, dict):
            continue
        spawn_key = entry.get("spawn_key")
        if not isinstance(spawn_key, str):
            label = entry.get("task_id") or entry.get("wave_id") or "unknown"
            spawn_key = f"{entry.get('wave_id', 'unknown')}::{entry.get('spawn_kind', 'unknown')}::{label}"
        merged = dict(previous.get(spawn_key, {}))
        merged.update(entry)
        merged["spawn_key"] = spawn_key
        entries_by_key[spawn_key] = merged
    records = []
    for spawn_key in sorted(entries_by_key):
        entry = entries_by_key[spawn_key]
        spawn_key = entry.get("spawn_key")
        prior = previous.get(spawn_key, {})
        agent_id = agent_records.get(spawn_key, prior.get("agent_id"))
        label = entry.get("task_id") or entry.get("wave_id") or "unknown"
        record = {
            "spawn_key": spawn_key,
            "spawn_status": "spawned_waiting_for_evidence" if agent_id else "ready_to_spawn",
            "agent_id": agent_id,
            "agent_nickname": prior.get("agent_nickname"),
            "spawn_kind": entry.get("spawn_kind"),
            "agent_hierarchy_role": entry.get("agent_hierarchy_role", "subagent"),
            "subagent_type": entry.get("subagent_type"),
            "subagent_may_spawn_subagents": entry.get("subagent_may_spawn_subagents", False),
            "parent_feedback_channel": entry.get("parent_feedback_channel", "feedback_packet.json"),
            "wave_id": entry.get("wave_id"),
            "task_id": entry.get("task_id"),
            "label": label,
            "prompt_file": entry.get("prompt_file"),
            "expected_evidence_dir": entry.get("expected_evidence_dir"),
            "expected_evidence_file": entry.get("expected_evidence_file"),
            "expected_kernel_report": entry.get("expected_kernel_report"),
            "expected_module_ooc_synthesis_report": entry.get("expected_module_ooc_synthesis_report"),
            "expected_subagent_result": entry.get("expected_subagent_result"),
            "verification_report": entry.get("verification_report"),
            "required_commands": entry.get("required_commands", []),
            "parent_must_not_write_hdl": entry.get("parent_must_not_write_hdl", True),
            "failure_to_skill_required": entry.get("failure_to_skill_required", True),
            "codex_spawn_message": entry.get("codex_spawn_message"),
        }
        if prior.get("spawn_status") in {"closed", "completed", "failed", "cancelled"} and spawn_key not in agent_records:
            record["spawn_status"] = prior["spawn_status"]
        reconciled_status, evidence_reason = _record_status_from_evidence(record, task_results, verification_results)
        record["spawn_status"] = reconciled_status
        if evidence_reason:
            record["evidence_reason"] = evidence_reason
        records.append(record)
    status_counts: dict[str, int] = {}
    for record in records:
        status = record["spawn_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "artifact": "hdl_subagent_spawn_ledger",
        "coverage_level": "execution_manifest_to_external_codex_agent_tracking",
        "agent_hierarchy": _parent_subagent_hierarchy(),
        "spawn_entry_count": len(records),
        "status_counts": status_counts,
        "parallel_spawn_allowed": execution_manifest.get("parallel_spawn_allowed", False),
        "max_parallel_batch_size": execution_manifest.get("max_parallel_batch_size", 0),
        "wave_status_reconciled": bool(wave_status),
        "records": records,
        "does_not_claim": [
            "package code spawned Codex agents",
            "sub-agent execution completed",
            "sub-agents spawned other sub-agents",
            "generated RTL completeness",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ],
    }


def build_hdl_subagent_spawn_ledger_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# HDL Sub-Agent Spawn Ledger",
        "",
        "This parent-owned ledger maps execution-manifest entries to external Codex sub-agent ids.",
        "It is bookkeeping only; package code still does not spawn agents.",
        "",
        f"- Spawn entries: `{ledger.get('spawn_entry_count', 0)}`",
        f"- Parallel spawn allowed: `{ledger.get('parallel_spawn_allowed', False)}`",
        f"- Max parallel batch size: `{ledger.get('max_parallel_batch_size', 0)}`",
        f"- Wave status reconciled: `{ledger.get('wave_status_reconciled', False)}`",
        f"- Status counts: `{ledger.get('status_counts', {})}`",
        "",
    ]
    for record in ledger.get("records", []):
        if not isinstance(record, dict):
            continue
        lines.extend(
            [
                f"## `{record.get('label', 'unknown')}`",
                "",
                f"- Spawn key: `{record.get('spawn_key')}`",
                f"- Status: `{record.get('spawn_status')}`",
                f"- Agent id: `{record.get('agent_id') or 'not_recorded'}`",
                f"- Kind: `{record.get('spawn_kind')}`",
                f"- Wave: `{record.get('wave_id')}`",
                f"- Prompt file: `{record.get('prompt_file')}`",
            ]
        )
        if record.get("expected_evidence_dir"):
            lines.append(f"- Evidence dir: `{record['expected_evidence_dir']}`")
        if record.get("expected_evidence_file"):
            lines.append(f"- Evidence file: `{record['expected_evidence_file']}`")
        if record.get("expected_subagent_result"):
            lines.append(f"- Sub-agent result: `{record['expected_subagent_result']}`")
        if record.get("expected_module_ooc_synthesis_report"):
            lines.append(f"- Module OOC synthesis: `{record['expected_module_ooc_synthesis_report']}`")
        if record.get("verification_report"):
            lines.append(f"- Verification report: `{record['verification_report']}`")
        lines.append("")
    lines.extend(["## Does Not Claim", ""])
    lines.extend(f"- {claim}" for claim in ledger.get("does_not_claim", []))
    lines.append("")
    return "\n".join(lines)


def write_hdl_subagent_spawn_ledger(
    execution_manifest: dict[str, Any],
    out_dir: Path,
    existing_ledger: dict[str, Any] | None = None,
    agent_record_items: list[str] | None = None,
    wave_status: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_hdl_subagent_spawn_ledger(
        execution_manifest,
        existing_ledger=existing_ledger,
        agent_records=_parse_agent_records(agent_record_items),
        wave_status=wave_status,
    )
    json_path = out_dir / "hdl_subagent_spawn_ledger.json"
    markdown_path = out_dir / "hdl_subagent_spawn_ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    markdown_path.write_text(build_hdl_subagent_spawn_ledger_markdown(ledger), encoding="utf-8")
    return json_path, markdown_path


def build_hdl_subagent_execution_manifest(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
) -> dict[str, Any]:
    """Describe the next Codex sub-agent spawns without spawning them in package code."""
    statuses = _wave_status_by_id(wave_status)
    spawn_entries: list[dict[str, Any]] = []
    blocked_waves = []
    for wave in dispatch_plan["waves"]:
        status = statuses.get(wave["wave_id"], {})
        state = status.get("status")
        if state == "ready_to_dispatch":
            spawn_entries.extend(
                _execution_entry_for_task(wave, task)
                for task in wave["implementation_tasks"]
                if any(
                    result.get("task_id") == task["task_id"]
                    and result.get("status")
                    in {
                        "missing",
                        "module_ooc_synthesis_missing",
                        "module_ooc_synthesis_hardware_mismatch",
                        "module_ooc_synthesis_needs_tuning",
                    }
                    for result in status.get("task_results", [])
                )
            )
        elif state == "ready_for_verification":
            spawn_entries.append(_execution_entry_for_verification(wave))
        elif state in {
            "blocked_by_dependency",
            "failed_missing_skill_candidate",
            "failed_waiting_for_skill_update",
            "failed_verification_missing_skill_candidate",
            "failed_verification_waiting_for_skill_update",
            "incomplete_subagent_result",
        }:
            next_action = "wait_for_dependency"
            if state in {"failed_waiting_for_skill_update", "failed_verification_waiting_for_skill_update"}:
                next_action = "run_subagents_skill_draft_and_update_skill_before_retry"
            elif state in {"failed_missing_skill_candidate", "failed_verification_missing_skill_candidate"}:
                next_action = "collect_complete_skill_update_candidate_before_retry"
            elif state == "incomplete_subagent_result":
                next_action = "collect_complete_subagent_result_before_verification"
            blocked_waves.append(
                {
                    "wave_id": wave["wave_id"],
                    "status": state,
                    "reason": status.get("reason"),
                    "depends_on_waves": wave.get("depends_on_waves", []),
                    "next_action": next_action,
                    "verification": status.get("verification"),
                    "task_status_counts": status.get("task_status_counts", {}),
                }
            )
    implementation_entries = [
        entry for entry in spawn_entries if entry["spawn_kind"] == "implementation_agent"
    ]
    verification_entries = [
        entry for entry in spawn_entries if entry["spawn_kind"] == "verification_agent"
    ]
    spawn_batches: list[dict[str, Any]] = []
    for wave in dispatch_plan["waves"]:
        wave_entries = [entry for entry in spawn_entries if entry["wave_id"] == wave["wave_id"]]
        if not wave_entries:
            continue
        spawn_kind = wave_entries[0]["spawn_kind"]
        homogeneous = all(entry["spawn_kind"] == spawn_kind for entry in wave_entries)
        spawn_batches.append(
            {
                "batch_id": f"{_slug(wave['wave_id'])}__{spawn_kind}",
                "wave_id": wave["wave_id"],
                "spawn_kind": spawn_kind if homogeneous else "mixed",
                "parallel_spawn_allowed": homogeneous and spawn_kind == "implementation_agent" and len(wave_entries) > 1,
                "entry_count": len(wave_entries),
                "task_ids": [
                    entry["task_id"]
                    for entry in wave_entries
                    if entry.get("spawn_kind") == "implementation_agent"
                ],
                "prompt_files": [entry["prompt_file"] for entry in wave_entries],
                "spawn_entries": wave_entries,
            }
        )
    parallel_spawn_allowed = any(batch["parallel_spawn_allowed"] for batch in spawn_batches)
    skill_update_required = any(
        wave["status"] in {"failed_waiting_for_skill_update", "failed_verification_waiting_for_skill_update"}
        for wave in blocked_waves
    )
    missing_skill_candidate = any(
        wave["status"] in {"failed_missing_skill_candidate", "failed_verification_missing_skill_candidate"}
        for wave in blocked_waves
    )
    return {
        "artifact": "hdl_subagent_execution_manifest",
        "coverage_level": "dispatch_wave_status_to_codex_spawn_instructions",
        "agent_hierarchy": _parent_subagent_hierarchy(),
        "model": dispatch_plan.get("model", {}),
        "hardware": dispatch_plan.get("hardware", {}),
        "optimization": dispatch_plan.get("optimization", {}),
        "dispatch_policy": dispatch_plan.get("dispatch_policy", {}),
        "parent_agent_runtime_boundary": (
            "Package code emits spawn instructions only; the interactive Codex parent "
            "or an external runner must call sub-agents, deliver feedback packets, "
            "collect evidence, update Skills on reusable failures, and decide retries."
        ),
        "spawn_entry_count": len(spawn_entries),
        "implementation_spawn_count": len(implementation_entries),
        "verification_spawn_count": len(verification_entries),
        "spawn_batch_count": len(spawn_batches),
        "parallel_spawn_allowed": parallel_spawn_allowed,
        "max_parallel_batch_size": max(
            (batch["entry_count"] for batch in spawn_batches if batch["parallel_spawn_allowed"]),
            default=0,
        ),
        "skill_update_required": skill_update_required,
        "missing_skill_update_candidate": missing_skill_candidate,
        "spawn_batches": spawn_batches,
        "spawn_entries": spawn_entries,
        "blocked_waves": blocked_waves,
        "does_not_claim": [
            "sub-agent execution occurred",
            "automatic sub-agent spawning inside package runtime",
            "sub-agents spawned other sub-agents",
            "generated RTL completeness",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ],
    }


def _parent_loop_status(execution_manifest: dict[str, Any]) -> str:
    if execution_manifest.get("skill_update_required"):
        return "waiting_for_parent_skill_update_before_retry"
    if execution_manifest.get("missing_skill_update_candidate"):
        return "waiting_for_subagent_failure_detail"
    if execution_manifest.get("spawn_entry_count", 0) > 0:
        return "ready_to_spawn_subagents"
    if execution_manifest.get("blocked_waves"):
        return "blocked_waiting_for_parent_action"
    return "idle_or_waiting_for_target_evidence"


def _feedback_entry_for_spawn(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "feedback_kind": "spawn_or_retry_instruction",
        "spawn_key": entry.get("spawn_key"),
        "spawn_kind": entry.get("spawn_kind"),
        "subagent_type": entry.get("subagent_type"),
        "wave_id": entry.get("wave_id"),
        "task_id": entry.get("task_id"),
        "prompt_file": entry.get("prompt_file"),
        "expected_evidence_dir": entry.get("expected_evidence_dir"),
        "expected_evidence_file": entry.get("expected_evidence_file"),
        "expected_subagent_result": entry.get("expected_subagent_result"),
        "parent_feedback": {
            "subagent_may_spawn_subagents": False,
            "return_result_to_parent": True,
            "write_required_evidence_before_claiming_success": True,
            "on_failure_return_skill_update_candidate": entry.get("failure_to_skill_required", True),
        },
    }


def _feedback_entry_for_blocked_wave(wave: dict[str, Any]) -> dict[str, Any]:
    next_action = wave.get("next_action")
    return {
        "feedback_kind": "blocked_wave_feedback",
        "wave_id": wave.get("wave_id"),
        "status": wave.get("status"),
        "reason": wave.get("reason"),
        "next_action": next_action,
        "depends_on_waves": wave.get("depends_on_waves", []),
        "verification": wave.get("verification"),
        "task_status_counts": wave.get("task_status_counts", {}),
        "parent_feedback": {
            "parent_decides_retry": True,
            "subagent_may_spawn_subagents": False,
            "skill_update_required_before_retry": next_action
            == "run_subagents_skill_draft_and_update_skill_before_retry",
            "collect_complete_failure_detail": next_action
            == "collect_complete_skill_update_candidate_before_retry",
        },
    }


def build_parent_feedback_loop_state(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build parent-owned feedback and retry artifacts for the sub-agent loop."""
    feedback_entries = [
        _feedback_entry_for_spawn(entry)
        for entry in execution_manifest.get("spawn_entries", [])
        if isinstance(entry, dict)
    ]
    feedback_entries.extend(
        _feedback_entry_for_blocked_wave(wave)
        for wave in execution_manifest.get("blocked_waves", [])
        if isinstance(wave, dict)
    )
    retry_entries = [
        {
            "wave_id": wave.get("wave_id"),
            "status": wave.get("status"),
            "retry_gate": wave.get("next_action"),
            "retry_allowed_now": wave.get("next_action")
            not in {
                "run_subagents_skill_draft_and_update_skill_before_retry",
                "collect_complete_skill_update_candidate_before_retry",
                "collect_complete_subagent_result_before_verification",
                "wait_for_dependency",
            },
            "parent_action_required": wave.get("next_action"),
            "reason": wave.get("reason"),
        }
        for wave in execution_manifest.get("blocked_waves", [])
        if isinstance(wave, dict)
    ]
    retry_entries.extend(
        {
            "batch_id": batch.get("batch_id"),
            "wave_id": batch.get("wave_id"),
            "status": "ready_to_spawn",
            "retry_gate": "spawn_subagents_from_parent",
            "retry_allowed_now": True,
            "parent_action_required": "spawn_subagents_and_record_ledger",
            "task_ids": batch.get("task_ids", []),
        }
        for batch in execution_manifest.get("spawn_batches", [])
        if isinstance(batch, dict)
    )
    loop_status = _parent_loop_status(execution_manifest)
    hierarchy = _parent_subagent_hierarchy()
    feedback_packet = {
        "artifact": "feedback_packet",
        "coverage_level": "parent_to_subagent_feedback",
        "agent_hierarchy": hierarchy,
        "status": loop_status,
        "entry_count": len(feedback_entries),
        "entries": feedback_entries,
        "does_not_claim": [
            "sub-agent execution occurred",
            "sub-agents spawned other sub-agents",
            "retry completed",
            "generated RTL completeness",
        ],
    }
    retry_plan = {
        "artifact": "retry_plan",
        "coverage_level": "parent_owned_retry_decision_plan",
        "agent_hierarchy": hierarchy,
        "status": loop_status,
        "retry_entry_count": len(retry_entries),
        "entries": retry_entries,
        "does_not_claim": [
            "retry was executed",
            "Skill files were updated automatically",
            "failed HDL gate is fixed",
        ],
    }
    parent_loop_state = {
        "artifact": "parent_loop_state",
        "coverage_level": "single_parent_feedback_retry_loop",
        "status": loop_status,
        "agent_hierarchy": hierarchy,
        "model": dispatch_plan.get("model", {}),
        "hardware": dispatch_plan.get("hardware", {}),
        "optimization": dispatch_plan.get("optimization", {}),
        "wave_status_summary": {
            "wave_count": wave_status.get("wave_count"),
            "next_dispatchable_waves": wave_status.get("next_dispatchable_waves", []),
            "passed_wave_count": sum(
                1 for wave in wave_status.get("waves", []) if wave.get("status") == "passed"
            ),
        },
        "execution_summary": {
            "spawn_entry_count": execution_manifest.get("spawn_entry_count", 0),
            "spawn_batch_count": execution_manifest.get("spawn_batch_count", 0),
            "implementation_spawn_count": execution_manifest.get("implementation_spawn_count", 0),
            "verification_spawn_count": execution_manifest.get("verification_spawn_count", 0),
            "skill_update_required": execution_manifest.get("skill_update_required", False),
            "missing_skill_update_candidate": execution_manifest.get("missing_skill_update_candidate", False),
        },
        "feedback_packet": "feedback_packet.json",
        "retry_plan": "retry_plan.json",
        "next_parent_action": (
            "update_skill_then_retry"
            if execution_manifest.get("skill_update_required")
            else "collect_complete_skill_update_candidate"
            if execution_manifest.get("missing_skill_update_candidate")
            else "spawn_ready_subagents"
            if execution_manifest.get("spawn_entry_count", 0) > 0
            else "refresh_status_after_new_evidence_or_target_gate"
        ),
        "does_not_claim": [
            "package code spawned Codex agents",
            "sub-agent execution completed",
            "full LLaMA execution",
            "board-level ZCU104 signoff",
        ],
    }
    return {
        "artifact": "parent_feedback_loop_artifacts",
        "parent_loop_state": parent_loop_state,
        "feedback_packet": feedback_packet,
        "retry_plan": retry_plan,
    }


def write_parent_feedback_loop_state(
    dispatch_plan: dict[str, Any],
    wave_status: dict[str, Any],
    execution_manifest: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    artifacts = build_parent_feedback_loop_state(dispatch_plan, wave_status, execution_manifest)
    paths = {
        "parent_loop_state": out_dir / "parent_loop_state.json",
        "feedback_packet": out_dir / "feedback_packet.json",
        "retry_plan": out_dir / "retry_plan.json",
    }
    paths["parent_loop_state"].write_text(
        json.dumps(artifacts["parent_loop_state"], indent=2),
        encoding="utf-8",
    )
    paths["feedback_packet"].write_text(
        json.dumps(artifacts["feedback_packet"], indent=2),
        encoding="utf-8",
    )
    paths["retry_plan"].write_text(
        json.dumps(artifacts["retry_plan"], indent=2),
        encoding="utf-8",
    )
    return paths


def build_codex_spawn_instructions(execution_manifest: dict[str, Any]) -> str:
    lines = [
        "# Codex Sub-Agent Spawn Instructions",
        "",
        "This file is a runner-facing view of `hdl_subagent_execution_manifest.json`.",
        "Package code does not spawn agents; the interactive Codex parent or an external runner uses these entries.",
        "The Parent Agent is the only orchestrator: every non-parent worker is a Sub-agent, and Sub-agents return evidence to the Parent instead of spawning other agents.",
        "",
        f"- Spawn entries: `{execution_manifest.get('spawn_entry_count', 0)}`",
        f"- Implementation agents: `{execution_manifest.get('implementation_spawn_count', 0)}`",
        f"- Verification agents: `{execution_manifest.get('verification_spawn_count', 0)}`",
        f"- Parallel spawn allowed: `{execution_manifest.get('parallel_spawn_allowed', False)}`",
        f"- Max parallel batch size: `{execution_manifest.get('max_parallel_batch_size', 0)}`",
        "- Parent feedback packet: `feedback_packet.json`",
        "- Parent retry plan: `retry_plan.json`",
        "",
    ]
    batches = execution_manifest.get("spawn_batches", [])
    if not isinstance(batches, list) or not batches:
        lines.extend(
            [
                "## No Ready Spawns",
                "",
                "No implementation or verification sub-agent is ready to spawn from the current evidence state.",
                "",
            ]
        )
    else:
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            lines.extend(
                [
                    f"## Batch `{batch.get('batch_id', 'unknown')}`",
                    "",
                    f"- Wave: `{batch.get('wave_id', 'unknown')}`",
                    f"- Kind: `{batch.get('spawn_kind', 'unknown')}`",
                    f"- Parallel allowed: `{batch.get('parallel_spawn_allowed', False)}`",
                    f"- Entry count: `{batch.get('entry_count', 0)}`",
                    "",
                ]
            )
            for entry in batch.get("spawn_entries", []):
                if not isinstance(entry, dict):
                    continue
                label = entry.get("task_id") or entry.get("wave_id") or "unknown"
                lines.extend(
                    [
                        f"### `{label}`",
                        "",
                        f"- Agent: `{entry.get('agent', 'Codex')}`",
                        f"- Hierarchy role: `{entry.get('agent_hierarchy_role', 'subagent')}`",
                        f"- Sub-agent type: `{entry.get('subagent_type', 'unknown')}`",
                        f"- May spawn sub-agents: `{entry.get('subagent_may_spawn_subagents', False)}`",
                        f"- Mode: `{entry.get('mode', 'unknown')}`",
                        f"- Prompt file: `{entry.get('prompt_file', 'unknown')}`",
                        f"- Fork context: `{entry.get('fork_context', False)}`",
                    ]
                )
                if entry.get("expected_evidence_dir"):
                    lines.append(f"- Evidence dir: `{entry['expected_evidence_dir']}`")
                if entry.get("expected_evidence_file"):
                    lines.append(f"- Evidence file: `{entry['expected_evidence_file']}`")
                if entry.get("expected_module_ooc_synthesis_report"):
                    lines.append(f"- Module OOC synthesis: `{entry['expected_module_ooc_synthesis_report']}`")
                if entry.get("expected_subagent_result"):
                    lines.append(f"- Sub-agent result: `{entry['expected_subagent_result']}`")
                if entry.get("verification_report"):
                    lines.append(f"- Verification report: `{entry['verification_report']}`")
                source_replay = entry.get("source_replay", {})
                if source_replay:
                    lines.append(f"- Replay model name: `{source_replay.get('model_name') or 'not_configured'}`")
                    lines.append(f"- Replay GPTQ checkpoint: `{source_replay.get('gptq_checkpoint') or 'not_configured'}`")
                    lines.append(f"- Replay MLIR graph: `{source_replay.get('mlir_graph') or 'not_configured'}`")
                required_commands = entry.get("required_commands", [])
                if required_commands:
                    lines.extend(["", "Required commands:"])
                    lines.extend(f"- `{command}`" for command in required_commands)
                blocked = entry.get("wave_blocked_target_dependencies", [])
                if blocked:
                    lines.append(f"- Wave blocked target dependencies: `{', '.join(blocked)}`")
                global_blocked = entry.get("wave_global_blocked_target_dependencies", [])
                if global_blocked:
                    lines.append(f"- Wave global blocked target dependencies: `{', '.join(global_blocked)}`")
                lines.extend(
                    [
                        "",
                        "Spawn message:",
                        "",
                        "```text",
                        str(entry.get("codex_spawn_message", "")),
                        "```",
                        "",
                    ]
                )
    blocked_waves = execution_manifest.get("blocked_waves", [])
    if isinstance(blocked_waves, list) and blocked_waves:
        lines.extend(
            [
                "## Blocked Waves",
                "",
            ]
        )
        for wave in blocked_waves:
            if not isinstance(wave, dict):
                continue
            lines.extend(
                [
                    f"### `{wave.get('wave_id', 'unknown')}`",
                    "",
                    f"- Status: `{wave.get('status')}`",
                    f"- Reason: `{wave.get('reason')}`",
                    f"- Next action: `{wave.get('next_action')}`",
                    f"- Depends on waves: `{', '.join(wave.get('depends_on_waves', [])) or 'none'}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Does Not Claim",
            "",
        ]
    )
    for claim in execution_manifest.get("does_not_claim", []):
        lines.append(f"- {claim}")
    lines.append("")
    return "\n".join(lines)


def write_codex_spawn_instructions(execution_manifest: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "codex_spawn_instructions.md"
    path.write_text(build_codex_spawn_instructions(execution_manifest), encoding="utf-8")
    return path


def write_hdl_subagent_execution_manifest(
    dispatch_plan_path: Path,
    wave_status_path: Path,
    out_dir: Path,
) -> Path:
    dispatch_plan = json.loads(dispatch_plan_path.read_text(encoding="utf-8"))
    wave_status = json.loads(wave_status_path.read_text(encoding="utf-8"))
    execution_manifest = build_hdl_subagent_execution_manifest(dispatch_plan, wave_status)
    path = out_dir / "hdl_subagent_execution_manifest.json"
    path.write_text(
        json.dumps(execution_manifest, indent=2),
        encoding="utf-8",
    )
    write_codex_spawn_instructions(execution_manifest, out_dir)
    write_parent_feedback_loop_state(dispatch_plan, wave_status, execution_manifest, out_dir)
    return path


def write_hdl_subagent_wave_status(dispatch_plan_path: Path, out_dir: Path) -> Path:
    dispatch_plan = json.loads(dispatch_plan_path.read_text(encoding="utf-8"))
    status_path = out_dir / "hdl_subagent_wave_status.json"
    status = build_hdl_subagent_wave_status(dispatch_plan, out_dir)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status_path


def write_hdl_subagent_packets(manifest: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = out_dir / "subagent_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    verification_prompt_dir = out_dir / "verification_prompts"
    verification_prompt_dir.mkdir(parents=True, exist_ok=True)
    packets = build_hdl_subagent_packets(manifest)
    for packet in packets["packets"]:
        (out_dir / packet["prompt_file"]).write_text(packet["prompt"], encoding="utf-8")
    packet_json_path = out_dir / "hdl_subagent_tasks.json"
    packet_json_path.write_text(json.dumps(packets, indent=2), encoding="utf-8")
    skill_template_path = out_dir / "skill_update_candidate_template.json"
    skill_template_path.write_text(
        json.dumps(_skill_update_candidate_template(), indent=2),
        encoding="utf-8",
    )
    dispatch_plan_path = out_dir / "hdl_subagent_dispatch_plan.json"
    dispatch_plan = build_hdl_subagent_dispatch_plan(packets)
    for wave in dispatch_plan["waves"]:
        prompt_file = wave["verification_agent"]["prompt_file"]
        (out_dir / prompt_file).write_text(
            _verification_prompt_for_wave(wave, dispatch_plan),
            encoding="utf-8",
        )
    dispatch_plan_path.write_text(json.dumps(dispatch_plan, indent=2), encoding="utf-8")
    return packet_json_path, prompt_dir, dispatch_plan_path
