# Target Evidence Sub-Agent Task: board_zcu104_signoff

You are the Codex target-evidence sub-agent for board-level ZCU104 signoff.
The parent agent must not hand-write HDL or fabricate board evidence. Your job is to inspect or run the required Vivado/board-constraint flow and write board evidence only if the gate is actually proven.

## Target Context

- Model: `meta-llama/Llama-3.2-1B`
- Board: `ZCU104`
- FPGA part: `xczu7ev-ffvc1156-2-e`
- Target clock: `200 MHz`
- Target clock period: `5.000 ns`
- Resource budgets: LUT `207360`, FF `414720`, DSP `1536`, BRAM `280`, URAM `80`
- Quantization: `quarot_w4a4kv4_gptq_weights`
- Design style alias: `systolic_weight_stationary_llm_streaming`
- Compute style: `systolic_array`
- Execution style: `llm_decoder_streaming`
- Memory style: `external_ddr_streaming`
- Control style: `hierarchical_fsm`
- Collection root: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence`
- Dispatch plan: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/00_parent_inspect/hdl_subagent_dispatch_plan.json`

## Preconditions

- Ready to spawn: `False`
- Full execution readiness status: `blocked_by_target_preflight`
- Board readiness status: `blocked_by_full_llama_execution`
- Current board evidence failures: `['board_zcu104_signoff_evidence.json not found', 'full_llama_execution_readiness must be passed before board signoff']`

If `Ready to spawn` is false, do not force the gate. Return a precise missing-evidence report or a complete `skill_update_candidate`.

## Required Output

- Write `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence/board_zcu104_signoff_evidence.json` only after every required field below is backed by real artifacts.
- Also write `build/board_zcu104_signoff_gate/subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.
- Do not write `fixture_only: true`; fixture-only board evidence cannot satisfy this gate.

## Evidence Schema

- Required fields: `artifact, status, board, fpga_part, full_llama_execution_status, constraints, timing, resource_utilization, reports, fixture_only`
- Template artifact: `board_zcu104_signoff_evidence`
- Template JSON: `{'artifact': 'board_zcu104_signoff_evidence', 'status': '<passed>', 'board': 'ZCU104', 'fpga_part': 'xczu7ev-ffvc1156-2-e', 'full_llama_execution_status': '<passed>', 'constraints': {'clock': '<true>', 'reset': '<true>', 'board_io': '<true>', 'ps_pl_interface': '<true>', 'ddr_interface': '<true>'}, 'timing': {'constraints_met': '<true>', 'setup_worst_slack_ns': '<non-negative number>', 'hold_worst_slack_ns': '<non-negative number>', 'pulse_width_worst_slack_ns': '<non-negative number>'}, 'resource_utilization': {'lut': '<number <= 207360>', 'dsp': '<number <= 1536>', 'bram': '<number <= 280>', 'ff': '<number <= 414720>', 'uram': '<number <= 80>', 'io': '<number <= 420>'}, 'reports': {'timing_summary': '<path to timing_summary.rpt>', 'utilization': '<path to utilization.rpt>', 'constraints': '<path to ZCU104 XDC/TCL constraints>', 'vivado_log': '<path to Vivado log>'}, 'fixture_only': False}`

## Evidence That Must Be Proven

- `full_llama_execution_status == passed` from parent readiness.
- Board is `ZCU104` and FPGA part matches `xczu7ev-ffvc1156-2-e`.
- Constraints explicitly cover clock, reset, board I/O, PS/PL interface, and DDR interface.
- Vivado timing constraints are met and setup, hold, and pulse-width worst slack are all non-negative.
- Implemented `report_clocks`, timing summary, and XDC prove the accelerator/PS PL clock period is at or below the configured target period.
- Resource utilization reports numeric LUT/DSP/BRAM usage under configured budgets.
- Reports include timing summary, utilization, constraints, and Vivado log paths.
- Existing bounded fixture synthesis reports are insufficient unless they include the required board I/O, PS/PL, and DDR constraints.

## Allowed Write Scope

- You may write `build/board_zcu104_signoff_evidence.json` and files under `build/board_zcu104_signoff_gate/`.
- You may add or update focused non-HDL scripts/tests that parse or validate board signoff evidence.
- If RTL or SystemVerilog changes are necessary, stop and return a `skill_update_candidate`; board signoff should be retried through a scoped HDL/board-shell sub-agent.
- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.
- Do not weaken existing tests, evidence gates, blocked-target language, or full-execution/board-signoff separation.

## Failure-To-SKILL Candidate

- If you cannot prove board signoff, preserve evidence and return a `skill_update_candidate` with:
  - `failing_command`
  - `symptom`
  - `root_cause_hypothesis`
  - `prevention_rule`
  - `minimal_regression_check`

## Required Commands

- `python3 -m pytest -q tests/test_llm_kernels.py`
- `python3 -m nl2hdl subagents status --dispatch-plan examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/00_parent_inspect/hdl_subagent_dispatch_plan.json --evidence-root examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence --out build/board_zcu104_signoff_gate/status_after`

## Do Not Claim

- Real-time LLaMA inference performance.
- Hardware lab runtime validation unless actual hardware-run evidence is present.
- Board-level signoff from bounded fixture-only synthesis reports.
- New HDL correctness beyond already verified child evidence.
