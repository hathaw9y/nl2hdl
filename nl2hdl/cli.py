from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from dataclasses import replace

from .agent import run_agent
from .config import AgentConfig, load_config, validate_config
from .llm_agent import run_llm_agent
from .llm_kernels import run_zcu104_board_wrapper_axi_bridge_agent
from .llm_plan import write_llm_accelerator_plan
from .parent_loop import ParentLoopOptions, run_parent_loop
from .subagent_tasks import (
    build_board_zcu104_signoff_evidence_agent_task,
    build_board_zcu104_signoff_evidence_template,
    build_board_zcu104_signoff_readiness_report,
    build_full_llama_execution_evidence_agent_task,
    build_full_llama_execution_evidence_template,
    build_full_llama_execution_readiness_report,
    build_model_level_execution_harness_agent_task,
    build_zcu104_board_wrapper_axi_bridge_agent_task,
    build_hdl_subagent_execution_manifest,
    build_hdl_subagent_wave_status,
    build_target_evidence_execution_manifest,
    build_codex_spawn_instructions,
    write_codex_spawn_instructions,
    write_hdl_subagent_skill_update_draft,
    write_hdl_subagent_spawn_ledger,
    write_parent_feedback_loop_state,
)


def _apply_cli_overrides(config: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    gptq_checkpoint = getattr(args, "gptq_checkpoint", None)
    mlir_graph = getattr(args, "mlir_graph", None)
    if gptq_checkpoint is None and mlir_graph is None:
        return config
    model_updates = {}
    if gptq_checkpoint is not None:
        model_updates["gptq_checkpoint"] = str(gptq_checkpoint).strip()
    if mlir_graph is not None:
        model_updates["mlir_graph"] = str(mlir_graph).strip()
    updated = replace(config, model=replace(config.model, **model_updates))
    validate_config(updated)
    return updated


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Hugging Face model name, local model id, or builtin:tiny_mlp")
    parser.add_argument("--spec", help="Optional YAML/JSON hardware and verification spec")
    parser.add_argument(
        "--gptq-checkpoint",
        help="Optional GPTQ metadata checkpoint source path or Hugging Face repo id; overrides model.gptq_checkpoint in --spec.",
    )
    parser.add_argument(
        "--mlir-graph",
        help="Optional provided/exported MLIR graph path; overrides model.mlir_graph in --spec for inspect gating.",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--planner",
        choices=("heuristic", "llm", "auto"),
        default="heuristic",
        help="Accelerator design planner. auto uses LLM only when dependencies and credentials are available.",
    )
    parser.add_argument(
        "--planner-model",
        default="gpt-4.1-mini",
        help="LLM model name used when --planner llm/auto can call an OpenAI-compatible planner.",
    )
    parser.add_argument(
        "--mode",
        choices=("inspect", "kernel", "block", "full"),
        default="full",
        help="Generation scope for LLM accelerator flows.",
    )
    parser.add_argument(
        "--kernel",
        choices=(
            "int4_unpack",
            "gptq_dequant",
            "projection",
            "projection_tile",
            "projection_streaming",
            "projection_parallel_streaming",
            "projection_adapter_stream_integration",
            "projection_target_stream_plan",
            "projection_memory_stream_boundary",
            "projection_internal_stream_shell",
            "projection_axi_read_command_adapter",
            "projection_axi_read_data_channel_adapter",
            "projection_axi_read_transaction_adapter",
            "projection_axi_stream_integration",
            "rmsnorm_rope_source_path",
            "attention_kv_cache_fixture",
            "packed_stream_adapter",
            "packed_stream_adapter_multiword",
            "rmsnorm",
            "rmsnorm_target",
            "rope",
            "rope_target",
            "decoder_block",
            "decoder_block_scaffold",
            "decoder_child_datapath",
            "decoder_child_attention_datapath",
            "decoder_child_axi_attention_datapath",
            "layer_fsm_axi_attention_fixture",
            "top_fsm_axi_attention_fixture",
            "token_loop_axi_attention_fixture",
            "decoder_block_axi_attention_mlp_fixture",
            "layer_fsm_axi_decoder_block_fixture",
            "top_fsm_axi_decoder_block_fixture",
            "token_loop_axi_decoder_block_fixture",
            "model_fsm_axi_decoder_block_fixture",
            "ddr_axi_board_shell_fixture",
            "layer_fsm_fixture",
            "layer_fsm_attention_fixture",
            "top_fsm_fixture",
            "top_fsm_attention_fixture",
            "token_loop_attention_fixture",
            "residual_mlp_fixture",
            "decoder_block_attention_mlp_fixture",
            "layer_fsm_decoder_block_fixture",
            "top_fsm_decoder_block_fixture",
            "token_loop_decoder_block_fixture",
        ),
        help="Kernel to generate when --mode kernel or --mode block is selected.",
    )
    parser.add_argument(
        "--partition",
        choices=("gemm_non_gemm",),
        default="gemm_non_gemm",
        help="Operation partition strategy used by MLIR/semantic model inspection.",
    )
    parser.add_argument("--skip-synth", action="store_true", help="Generate Vivado TCL but do not run Vivado")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep temporary artifacts")
    parser.add_argument("--verbose", action="store_true", help="Print the full agent report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nl2hdl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    agent = subparsers.add_parser("agent", help="Coding-agent model inspection and accelerator generation")
    _add_generate_args(agent)
    generate = subparsers.add_parser("generate", help="Alias for agent")
    _add_generate_args(generate)
    plan = subparsers.add_parser("plan", help="Write a model-to-accelerator framework plan")
    plan.add_argument("--model", required=True, help="Hugging Face model name or local model id")
    plan.add_argument("--spec", help="Optional YAML/JSON hardware and optimization spec")
    plan.add_argument(
        "--gptq-checkpoint",
        help="Optional GPTQ metadata checkpoint source path or Hugging Face repo id; overrides model.gptq_checkpoint in --spec.",
    )
    plan.add_argument(
        "--mlir-graph",
        help="Optional provided/exported MLIR graph path; overrides model.mlir_graph in --spec for planning metadata.",
    )
    plan.add_argument("--out", required=True, help="Output directory")
    parent_loop = subparsers.add_parser(
        "parent-loop",
        help="Run the Parent feedback loop over HDL sub-agent waves and evidence",
    )
    parent_loop.add_argument("--model", required=True, help="Hugging Face model name or local model id")
    parent_loop.add_argument("--spec", help="Optional YAML/JSON hardware and optimization spec")
    parent_loop.add_argument(
        "--gptq-checkpoint",
        help="Optional GPTQ metadata checkpoint source path or Hugging Face repo id; overrides model.gptq_checkpoint in --spec.",
    )
    parent_loop.add_argument(
        "--mlir-graph",
        help="Optional provided/exported MLIR graph path; overrides model.mlir_graph in --spec for inspect gating.",
    )
    parent_loop.add_argument("--out", required=True, help="Output directory")
    parent_loop.add_argument(
        "--partition",
        choices=("gemm_non_gemm",),
        default="gemm_non_gemm",
        help="Operation partition strategy used by parent inspect artifacts.",
    )
    parent_loop.add_argument(
        "--backend",
        choices=("local", "queue"),
        default="local",
        help="local executes deterministic sub-agent backends where possible; queue only writes external Codex work.",
    )
    parent_loop.add_argument("--max-iterations", type=int, default=8, help="Maximum parent feedback iterations")
    parent_loop.add_argument(
        "--max-subagents-per-iteration",
        type=int,
        help="Limit selected ready sub-agents per iteration for smoke tests or staged runs",
    )
    parent_loop.add_argument(
        "--skip-synth",
        action="store_true",
        help="Do not run Vivado synthesis in local HDL sub-agent backend",
    )
    parent_loop.add_argument(
        "--skip-vivado-route",
        action="store_true",
        help="Do not run Vivado route/check in local board-wrapper backend",
    )
    parent_loop.add_argument(
        "--vivado-executable",
        default="vivado",
        help="Vivado executable name/path used by local board-wrapper backend",
    )
    parent_loop.add_argument(
        "--local-verification",
        action="store_true",
        help="Use deterministic verification smoke reports instead of queueing Codex verification sub-agents",
    )
    parent_loop.add_argument("--verbose", action="store_true", help="Print the full parent loop report")
    subagents = subparsers.add_parser("subagents", help="Inspect and refresh HDL sub-agent orchestration state")
    subagent_subparsers = subagents.add_subparsers(dest="subagent_command", required=True)
    status = subagent_subparsers.add_parser(
        "status",
        help="Refresh wave status and next Codex spawn instructions from sub-agent evidence",
    )
    status.add_argument("--dispatch-plan", required=True, help="Path to hdl_subagent_dispatch_plan.json")
    status.add_argument(
        "--evidence-root",
        required=True,
        help="Directory containing <task>_gate/kernel_report.json and verification_results/*.json",
    )
    status.add_argument("--out", required=True, help="Directory where refreshed status artifacts are written")
    skill_draft = subagent_subparsers.add_parser(
        "skill-draft",
        help="Collect failed HDL sub-agent candidates into a SKILL update draft before retry",
    )
    skill_draft.add_argument("--dispatch-plan", required=True, help="Path to hdl_subagent_dispatch_plan.json")
    skill_draft.add_argument(
        "--evidence-root",
        required=True,
        help="Directory containing failed <task>_gate/kernel_report.json or subagent_result.json candidates",
    )
    skill_draft.add_argument("--out", required=True, help="Directory where skill_update_candidates.json is written")
    skill_draft.add_argument(
        "--target-skill",
        default="hdl-kernel-contract-gates",
        help="Skill name to update after reviewing the generated draft",
    )
    ledger = subagent_subparsers.add_parser(
        "ledger",
        help="Create or refresh a parent-owned ledger for externally spawned Codex sub-agents",
    )
    ledger.add_argument("--execution-manifest", required=True, help="Path to hdl_subagent_execution_manifest.json")
    ledger.add_argument("--out", required=True, help="Directory where hdl_subagent_spawn_ledger.json is written")
    ledger.add_argument(
        "--existing-ledger",
        help="Optional previous hdl_subagent_spawn_ledger.json whose agent ids/statuses should be preserved",
    )
    ledger.add_argument(
        "--wave-status",
        help="Optional hdl_subagent_wave_status.json used to reconcile records with collected evidence",
    )
    ledger.add_argument(
        "--agent-record",
        action="append",
        default=[],
        help="Record a spawned agent id as spawn_key=agent_id. Repeat for multiple sub-agents.",
    )
    board_wrapper = subagent_subparsers.add_parser(
        "board-wrapper",
        help="Run the bounded ZCU104 Board Wrapper Agent implementation attempt",
    )
    board_wrapper.add_argument("--spec", required=True, help="YAML/JSON hardware and verification spec")
    board_wrapper.add_argument("--out", required=True, help="board_zcu104_signoff_gate output directory")
    board_wrapper.add_argument(
        "--fixture-report",
        help="Optional strongest bounded fixture kernel_report.json used to reject fixture-only signoff claims",
    )
    board_wrapper.add_argument(
        "--skip-vivado-route",
        action="store_true",
        help="Generate package and reports but do not run Vivado route/check",
    )
    board_wrapper.add_argument(
        "--vivado-executable",
        default="vivado",
        help="Vivado executable name/path to use for version and route/check commands",
    )
    return parser


def _run_subagent_status(args: argparse.Namespace) -> int:
    dispatch_plan_path = Path(args.dispatch_plan)
    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dispatch_plan = json.loads(dispatch_plan_path.read_text(encoding="utf-8"))
    wave_status = build_hdl_subagent_wave_status(dispatch_plan, evidence_root)
    execution_manifest = build_hdl_subagent_execution_manifest(dispatch_plan, wave_status)
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
    model_harness_agent_task = build_model_level_execution_harness_agent_task(
        dispatch_plan,
        wave_status,
        full_execution_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    board_signoff_template = build_board_zcu104_signoff_evidence_template(
        dispatch_plan,
        full_execution_readiness,
        evidence_root,
    )
    board_signoff_agent_task = build_board_zcu104_signoff_evidence_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        board_signoff_template,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    full_execution_agent_task = build_full_llama_execution_evidence_agent_task(
        dispatch_plan,
        wave_status,
        full_execution_readiness,
        full_execution_template,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    board_wrapper_agent_task = build_zcu104_board_wrapper_axi_bridge_agent_task(
        dispatch_plan,
        full_execution_readiness,
        board_signoff_readiness,
        evidence_root,
        dispatch_plan_path=str(dispatch_plan_path),
    )
    model_harness_execution_manifest = build_target_evidence_execution_manifest(model_harness_agent_task)
    target_evidence_execution_manifest = build_target_evidence_execution_manifest(full_execution_agent_task)
    board_signoff_execution_manifest = build_target_evidence_execution_manifest(board_signoff_agent_task)
    board_wrapper_execution_manifest = build_target_evidence_execution_manifest(board_wrapper_agent_task)
    wave_status_path = out_dir / "hdl_subagent_wave_status.json"
    execution_manifest_path = out_dir / "hdl_subagent_execution_manifest.json"
    parent_loop_paths = write_parent_feedback_loop_state(
        dispatch_plan,
        wave_status,
        execution_manifest,
        out_dir,
    )
    model_harness_execution_manifest_path = out_dir / "model_level_execution_harness_manifest.json"
    model_harness_spawn_instructions_path = out_dir / "model_level_execution_harness_spawn_instructions.md"
    target_evidence_execution_manifest_path = out_dir / "target_evidence_execution_manifest.json"
    target_evidence_spawn_instructions_path = out_dir / "target_evidence_spawn_instructions.md"
    board_signoff_execution_manifest_path = out_dir / "board_zcu104_signoff_execution_manifest.json"
    board_signoff_spawn_instructions_path = out_dir / "board_zcu104_signoff_spawn_instructions.md"
    board_wrapper_execution_manifest_path = out_dir / "zcu104_board_wrapper_axi_bridge_execution_manifest.json"
    board_wrapper_spawn_instructions_path = out_dir / "zcu104_board_wrapper_axi_bridge_spawn_instructions.md"
    full_execution_readiness_path = out_dir / "full_llama_execution_readiness.json"
    board_signoff_readiness_path = out_dir / "board_zcu104_signoff_readiness.json"
    full_execution_template_path = out_dir / "full_llama_execution_evidence_template.json"
    board_signoff_template_path = out_dir / "board_zcu104_signoff_evidence_template.json"
    model_harness_prompt_path = out_dir / model_harness_agent_task["prompt_file"]
    model_harness_agent_task_path = out_dir / "model_level_execution_harness_agent_task.json"
    target_evidence_prompt_path = out_dir / full_execution_agent_task["prompt_file"]
    full_execution_agent_task_path = out_dir / "full_llama_execution_evidence_agent_task.json"
    board_signoff_prompt_path = out_dir / board_signoff_agent_task["prompt_file"]
    board_signoff_agent_task_path = out_dir / "board_zcu104_signoff_evidence_agent_task.json"
    board_wrapper_prompt_path = out_dir / board_wrapper_agent_task["prompt_file"]
    board_wrapper_agent_task_path = out_dir / "zcu104_board_wrapper_axi_bridge_agent_task.json"
    wave_status_path.write_text(json.dumps(wave_status, indent=2), encoding="utf-8")
    execution_manifest_path.write_text(json.dumps(execution_manifest, indent=2), encoding="utf-8")
    model_harness_execution_manifest_path.write_text(
        json.dumps(model_harness_execution_manifest, indent=2),
        encoding="utf-8",
    )
    model_harness_spawn_instructions_path.write_text(
        build_codex_spawn_instructions(model_harness_execution_manifest),
        encoding="utf-8",
    )
    target_evidence_execution_manifest_path.write_text(
        json.dumps(target_evidence_execution_manifest, indent=2),
        encoding="utf-8",
    )
    target_evidence_spawn_instructions_path.write_text(
        build_codex_spawn_instructions(target_evidence_execution_manifest),
        encoding="utf-8",
    )
    board_signoff_execution_manifest_path.write_text(
        json.dumps(board_signoff_execution_manifest, indent=2),
        encoding="utf-8",
    )
    board_signoff_spawn_instructions_path.write_text(
        build_codex_spawn_instructions(board_signoff_execution_manifest),
        encoding="utf-8",
    )
    board_wrapper_execution_manifest_path.write_text(
        json.dumps(board_wrapper_execution_manifest, indent=2),
        encoding="utf-8",
    )
    board_wrapper_spawn_instructions_path.write_text(
        build_codex_spawn_instructions(board_wrapper_execution_manifest),
        encoding="utf-8",
    )
    full_execution_readiness_path.write_text(json.dumps(full_execution_readiness, indent=2), encoding="utf-8")
    board_signoff_readiness_path.write_text(json.dumps(board_signoff_readiness, indent=2), encoding="utf-8")
    full_execution_template_path.write_text(json.dumps(full_execution_template, indent=2), encoding="utf-8")
    board_signoff_template_path.write_text(json.dumps(board_signoff_template, indent=2), encoding="utf-8")
    model_harness_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    model_harness_prompt_path.write_text(model_harness_agent_task["prompt"], encoding="utf-8")
    model_harness_task_without_prompt = {
        key: value for key, value in model_harness_agent_task.items() if key != "prompt"
    }
    model_harness_agent_task_path.write_text(json.dumps(model_harness_task_without_prompt, indent=2), encoding="utf-8")
    target_evidence_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    target_evidence_prompt_path.write_text(full_execution_agent_task["prompt"], encoding="utf-8")
    task_without_prompt = {key: value for key, value in full_execution_agent_task.items() if key != "prompt"}
    full_execution_agent_task_path.write_text(json.dumps(task_without_prompt, indent=2), encoding="utf-8")
    board_signoff_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    board_signoff_prompt_path.write_text(board_signoff_agent_task["prompt"], encoding="utf-8")
    board_signoff_task_without_prompt = {key: value for key, value in board_signoff_agent_task.items() if key != "prompt"}
    board_signoff_agent_task_path.write_text(json.dumps(board_signoff_task_without_prompt, indent=2), encoding="utf-8")
    board_wrapper_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    board_wrapper_prompt_path.write_text(board_wrapper_agent_task["prompt"], encoding="utf-8")
    board_wrapper_task_without_prompt = {
        key: value for key, value in board_wrapper_agent_task.items() if key != "prompt"
    }
    board_wrapper_agent_task_path.write_text(json.dumps(board_wrapper_task_without_prompt, indent=2), encoding="utf-8")
    spawn_instructions_path = write_codex_spawn_instructions(execution_manifest, out_dir)
    print(
        json.dumps(
            {
                "status": "passed",
                "out": str(out_dir),
                "evidence_root": str(evidence_root),
                "next_dispatchable_waves": wave_status["next_dispatchable_waves"],
                "spawn_entry_count": execution_manifest["spawn_entry_count"],
                "spawn_batch_count": execution_manifest["spawn_batch_count"],
                "parallel_spawn_allowed": execution_manifest["parallel_spawn_allowed"],
                "max_parallel_batch_size": execution_manifest["max_parallel_batch_size"],
                "implementation_spawn_count": execution_manifest["implementation_spawn_count"],
                "verification_spawn_count": execution_manifest["verification_spawn_count"],
                "target_evidence_spawn_count": target_evidence_execution_manifest["target_evidence_spawn_count"],
                "model_level_harness_spawn_count": model_harness_execution_manifest["target_evidence_spawn_count"],
                "board_signoff_spawn_count": board_signoff_execution_manifest["target_evidence_spawn_count"],
                "board_wrapper_implementation_spawn_count": board_wrapper_execution_manifest[
                    "implementation_spawn_count"
                ],
                "skill_update_required": execution_manifest["skill_update_required"],
                "missing_skill_update_candidate": execution_manifest["missing_skill_update_candidate"],
                "blocked_waves": execution_manifest["blocked_waves"],
                "parent_loop_state": str(parent_loop_paths["parent_loop_state"]),
                "feedback_packet": str(parent_loop_paths["feedback_packet"]),
                "retry_plan": str(parent_loop_paths["retry_plan"]),
                "full_llama_execution_readiness": str(full_execution_readiness_path),
                "full_llama_execution_readiness_status": full_execution_readiness["status"],
                "safe_to_clear_full_llama_model_execution_blocker": full_execution_readiness[
                    "safe_to_clear_full_llama_model_execution_blocker"
                ],
                "full_llama_execution_evidence_template": str(full_execution_template_path),
                "model_level_execution_harness_agent_task": str(model_harness_agent_task_path),
                "model_level_execution_harness_agent_prompt": str(model_harness_prompt_path),
                "model_level_execution_harness_agent_ready": model_harness_agent_task["ready_to_spawn"],
                "model_level_execution_harness_manifest": str(model_harness_execution_manifest_path),
                "model_level_execution_harness_spawn_instructions": str(model_harness_spawn_instructions_path),
                "full_llama_execution_evidence_agent_task": str(full_execution_agent_task_path),
                "full_llama_execution_evidence_agent_prompt": str(target_evidence_prompt_path),
                "full_llama_execution_evidence_agent_ready": full_execution_agent_task["ready_to_spawn"],
                "target_evidence_execution_manifest": str(target_evidence_execution_manifest_path),
                "target_evidence_spawn_instructions": str(target_evidence_spawn_instructions_path),
                "board_zcu104_signoff_readiness": str(board_signoff_readiness_path),
                "board_zcu104_signoff_readiness_status": board_signoff_readiness["status"],
                "safe_to_clear_board_level_zcu104_signoff_blocker": board_signoff_readiness[
                    "safe_to_clear_board_level_zcu104_signoff_blocker"
                ],
                "board_zcu104_signoff_evidence_template": str(board_signoff_template_path),
                "board_zcu104_signoff_evidence_agent_task": str(board_signoff_agent_task_path),
                "board_zcu104_signoff_evidence_agent_prompt": str(board_signoff_prompt_path),
                "board_zcu104_signoff_evidence_agent_ready": board_signoff_agent_task["ready_to_spawn"],
                "board_zcu104_signoff_execution_manifest": str(board_signoff_execution_manifest_path),
                "board_zcu104_signoff_spawn_instructions": str(board_signoff_spawn_instructions_path),
                "zcu104_board_wrapper_axi_bridge_agent_task": str(board_wrapper_agent_task_path),
                "zcu104_board_wrapper_axi_bridge_agent_prompt": str(board_wrapper_prompt_path),
                "zcu104_board_wrapper_axi_bridge_agent_ready": board_wrapper_agent_task["ready_to_spawn"],
                "zcu104_board_wrapper_axi_bridge_execution_manifest": str(board_wrapper_execution_manifest_path),
                "zcu104_board_wrapper_axi_bridge_spawn_instructions": str(board_wrapper_spawn_instructions_path),
                "codex_spawn_instructions": str(spawn_instructions_path),
            },
            indent=2,
        )
    )
    return 0


def _run_subagent_skill_draft(args: argparse.Namespace) -> int:
    dispatch_plan_path = Path(args.dispatch_plan)
    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out)
    dispatch_plan = json.loads(dispatch_plan_path.read_text(encoding="utf-8"))
    json_path, markdown_path = write_hdl_subagent_skill_update_draft(
        dispatch_plan,
        evidence_root,
        out_dir,
        target_skill=args.target_skill,
    )
    draft = json.loads(json_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": draft["status"],
                "out": str(out_dir),
                "evidence_root": str(evidence_root),
                "target_skill": draft["target_skill"],
                "candidate_count": draft["candidate_count"],
                "skill_update_candidates": str(json_path),
                "skill_update_draft": str(markdown_path),
                "next_action": draft["next_action"],
            },
            indent=2,
        )
    )
    return 0 if draft["candidate_count"] else 1


def _run_subagent_ledger(args: argparse.Namespace) -> int:
    execution_manifest_path = Path(args.execution_manifest)
    out_dir = Path(args.out)
    execution_manifest = json.loads(execution_manifest_path.read_text(encoding="utf-8"))
    existing_ledger = None
    if args.existing_ledger:
        existing_ledger = json.loads(Path(args.existing_ledger).read_text(encoding="utf-8"))
    wave_status = None
    if args.wave_status:
        wave_status = json.loads(Path(args.wave_status).read_text(encoding="utf-8"))
    json_path, markdown_path = write_hdl_subagent_spawn_ledger(
        execution_manifest,
        out_dir,
        existing_ledger=existing_ledger,
        agent_record_items=args.agent_record,
        wave_status=wave_status,
    )
    ledger = json.loads(json_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "passed",
                "out": str(out_dir),
                "spawn_entry_count": ledger["spawn_entry_count"],
                "status_counts": ledger["status_counts"],
                "wave_status_reconciled": ledger["wave_status_reconciled"],
                "hdl_subagent_spawn_ledger": str(json_path),
                "hdl_subagent_spawn_ledger_markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


def _run_subagent_board_wrapper(args: argparse.Namespace) -> int:
    config = load_config(args.spec)
    report = run_zcu104_board_wrapper_axi_bridge_agent(
        config,
        Path(args.out),
        fixture_report_path=Path(args.fixture_report) if args.fixture_report else None,
        run_vivado=not args.skip_vivado_route,
        vivado_executable=args.vivado_executable,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(Path(args.out)),
                "implementation_report": report["evidence_files"]["implementation_report"],
                "subagent_result": report["evidence_files"]["subagent_result"],
                "route_completed": report["route_completed"],
                "vivado_available": report["vivado_available"],
                "final_board_signoff_still_blocked": report["final_board_signoff_still_blocked"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "subagents":
            if args.subagent_command == "status":
                return _run_subagent_status(args)
            if args.subagent_command == "skill-draft":
                return _run_subagent_skill_draft(args)
            if args.subagent_command == "ledger":
                return _run_subagent_ledger(args)
            if args.subagent_command == "board-wrapper":
                return _run_subagent_board_wrapper(args)
            raise ValueError(f"unsupported subagents command: {args.subagent_command}")
        config = load_config(args.spec)
        config = _apply_cli_overrides(config, args)
        if args.command == "plan":
            plan = write_llm_accelerator_plan(args.model, config, Path(args.out))
            input_clarification = plan["input_clarification"]
            print(
                json.dumps(
                    {
                        "status": "needs_clarification"
                        if input_clarification.get("requires_user_response")
                        else "planned",
                        "out": str(Path(args.out)),
                        "model": plan["model"]["name"],
                        "question_count": input_clarification.get("question_count", 0),
                        "questions_file": "input_clarification_questions.json",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "parent-loop":
            report = run_parent_loop(
                model_name=args.model,
                config=config,
                out_dir=Path(args.out),
                partition=args.partition,
                options=ParentLoopOptions(
                    max_iterations=args.max_iterations,
                    max_subagents_per_iteration=args.max_subagents_per_iteration,
                    skip_synth=args.skip_synth,
                    skip_vivado_route=args.skip_vivado_route,
                    vivado_executable=args.vivado_executable,
                    backend=args.backend,
                    local_verification=args.local_verification,
                    verbose=args.verbose,
                ),
            )
            if args.verbose:
                print(json.dumps(report, indent=2))
            else:
                print(
                    json.dumps(
                        {
                            "status": report["status"],
                            "out": str(Path(args.out)),
                            "parent_loop_run_report": str(Path(args.out) / "parent_loop_run_report.json"),
                            "parent_loop_state": str(Path(args.out) / "status" / "parent_loop_state.json"),
                            "parent_loop_queue": str(Path(args.out) / "status" / "parent_loop_queue.json"),
                        },
                        indent=2,
                    )
                )
            return 1 if str(report["status"]).startswith("failed") else 0
        report = run_agent(
            model_name=args.model,
            config=config,
            out_dir=Path(args.out),
            planner=args.planner,
            planner_model=args.planner_model,
            mode=args.mode,
            kernel=args.kernel,
            partition=args.partition,
            skip_synth=args.skip_synth,
            keep_intermediates=args.keep_intermediates,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"nl2hdl: {exc}", file=sys.stderr)
        return 2
    if not args.verbose:
        print(json.dumps({"status": report["status"], "out": str(Path(args.out))}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
