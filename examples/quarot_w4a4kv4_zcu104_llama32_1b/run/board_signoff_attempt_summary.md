# Board Signoff Attempt Summary

Status: blocked, not signed off.

Date: 2026-06-16

## Commands Run

- `vivado -version`
- `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/quarot_w4a4kv4_zcu104_llama32_1b/input.yaml --mode full --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/full_gate_attempt --verbose`
- `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/quarot_w4a4kv4_zcu104_llama32_1b/input.yaml --mode kernel --kernel ddr_axi_board_shell_fixture --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/ddr_axi_board_shell_fixture_gate --verbose`
- `python3 -m nl2hdl subagents status --dispatch-plan examples/quarot_w4a4kv4_zcu104_llama32_1b/run/ddr_axi_board_shell_fixture_gate/hdl_subagent_dispatch_plan.json --evidence-root examples/quarot_w4a4kv4_zcu104_llama32_1b/run --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/board_signoff_status_from_runroot`
- `python3 -m pytest -q`

## What Passed

- Vivado is available: Vivado 2024.1.
- Full gate correctly refused to claim full LLaMA generation.
- `ddr_axi_board_shell_fixture` generated SystemVerilog, testbench, and Vivado reports.
- Fixture simulation passed.
- Verilator lint passed.
- Vivado post-route fixture timing passed at 200 MHz:
  - WNS: 1.005 ns
  - WHS: 0.017 ns
  - WPWS: 2.225 ns
  - failing setup/hold/pulse-width endpoints: 0
- Fixture utilization:
  - LUT: 1614 / 230400
  - FF: 1058 / 460800
  - DSP: 8 / 1728
  - BRAM: 0 / 312
  - URAM: 0 / 96
  - Bonded IOB: 132 / 360 in the Vivado report
- Full pytest suite passed: 232 passed, 2 warnings.

## Why Board Signoff Is Blocked

This run did not produce true ZCU104 board-level signoff.

Blocking evidence:

- `board_zcu104_signoff_readiness.json` status is `blocked_by_full_llama_execution`.
- `full_llama_execution_readiness.json` status is `blocked_by_target_preflight`.
- `board_zcu104_signoff_evidence.json` was not created.
- Full LLaMA execution evidence is absent.
- Current MLIR analysis is still a synthetic fixture, not a provided/exported target model graph.
- Current board shell is a bounded fixture and explicitly has `board_io_constraints=0`.
- Current fixture does not include PS/PL block design integration, real DDR controller integration, board I/O constraints, DRC report, methodology report, or clock report bundle required for board signoff.

## Key Evidence Files

- `run/full_gate_attempt/llm_agent_report.json`
- `run/ddr_axi_board_shell_fixture_gate/kernel_report.json`
- `run/ddr_axi_board_shell_fixture_gate/timing_summary.rpt`
- `run/ddr_axi_board_shell_fixture_gate/utilization.rpt`
- `run/ddr_axi_board_shell_fixture_gate/vivado.log`
- `run/ddr_axi_board_shell_fixture_gate/post_route.dcp`
- `run/board_signoff_status_from_runroot/board_zcu104_signoff_readiness.json`
- `run/board_signoff_status_from_runroot/full_llama_execution_readiness.json`
- `run/board_signoff_status_from_runroot/board_zcu104_signoff_evidence_template.json`
- `run/board_signoff_status_from_runroot/zcu104_board_wrapper_axi_bridge_agent_task.json`
- `run/board_signoff_status_from_runroot/board_zcu104_signoff_evidence_agent_task.json`

## Next Required Gate

Before board signoff can run as a real pass/fail signoff agent:

1. Clear target preflight with a real exported/provided LLaMA semantic graph.
2. Complete the missing module and integration waves with standard `subagent_result.json` evidence.
3. Produce full model execution evidence for the claimed decoder-layer count.
4. Spawn the ZCU104 board-wrapper agent only after model-level execution is ready.
5. Run board-level Vivado flow with PS/PL, DDR/address map, board I/O/clock/reset constraints, DRC, methodology, clock, timing, utilization, and checkpoint evidence.

No reusable failure-to-SKILL update was required in this attempt; the board gate blocked for expected missing upstream evidence rather than a repeated sub-agent implementation failure.
