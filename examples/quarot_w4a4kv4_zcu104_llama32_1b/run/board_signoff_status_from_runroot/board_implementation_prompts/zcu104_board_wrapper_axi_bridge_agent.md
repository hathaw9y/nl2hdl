# Implementation Sub-Agent Task: zcu104_board_wrapper_axi_bridge

You are the Codex implementation sub-agent for the next ZCU104 board-signoff step.
The parent agent must not hand-write HDL. Your job is to implement the missing board wrapper, accelerator bridge, Vivado block-design wiring, and routed-report flow needed before any evidence-only signoff agent can run.

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
- Collection root: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run`
- Dispatch plan: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/ddr_axi_board_shell_fixture_gate/hdl_subagent_dispatch_plan.json`
- Pre-signoff report: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/board_zcu104_signoff_gate/zcu104_board_shell_signoff_readiness_report.json`
- Prior board-signoff gap report: `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/board_zcu104_signoff_gate/evidence_gap_report.json`
- Prior board-signoff gap status: `None`

## Preconditions

- Ready to spawn: `False`
- Full execution readiness status: `blocked_by_target_preflight`
- Board readiness status: `blocked_by_full_llama_execution`
- Spawn precondition failures: `['full_llama_execution_readiness must be passed before board-wrapper implementation', 'board signoff readiness must still be blocked on board evidence', 'zcu104 pre-signoff constraints package must exist before board-wrapper implementation']`

If `Ready to spawn` is false, do not force implementation. Write the sub-agent result with the missing preconditions and a `skill_update_candidate` if the failure pattern is reusable.

## Required Implementation

- Generate or update a compact PL subsystem such as `zcu104_board_shell_top` with board-visible status and a child accelerator start/done path; do not require this direct PL shell to be the final routed board-signoff top.
- Add an accelerator AXI-lite or AXI-stream bridge stub that exposes a realistic PS-controlled register/address-map path without widening board-level ports.
- Update the ZCU104 BD TCL so the Zynq UltraScale+ PS, reset block, wrapper, and AXI interconnect/address map are generated and validated.
- Update the Vivado route-check TCL so it reads all generated HDL/BD/XDC inputs, runs synth/place/route, and writes timing, utilization, methodology/constraints, checkpoint, and Vivado log artifacts.
- If a prior board-signoff gap report exists, treat it as the retry target. In particular, do not route only the direct PL shell with package-level `aclk`/`aresetn` while the PS/PL/DDR BD is generated as a side artifact.
- If the prior board-signoff gap report mentions a target clock mismatch, treat it as a retry target too. Positive timing/resource reports are not enough when Vivado implements `clk_pl_0` at 5.625 ns / 177.778 MHz for a 200 MHz target.
- The corrected routed design must use the generated PS/PL/DDR BD wrapper, or an equivalent top where PS FCLK/reset drive the accelerator internally and PS AXI reaches the accelerator control path.
- The corrected DRC evidence must have no NSTD-1 or UCIO-1 critical warnings. Positive timing/resource reports are not enough if `aclk` or `aresetn` remain unconstrained/default-standard top-level ports.
- The corrected clock evidence must prove the configured target clock from raw `report_clocks`, `report_timing_summary`, and the implemented XDC. For a 200 MHz target, the accelerator/PS PL clock period must be <= 5.000 ns.
- Run Vivado when feasible. If Vivado cannot complete in this environment, preserve the exact command/log failure and do not write board signoff evidence.

## Required Output

- Write `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/board_zcu104_signoff_gate/zcu104_board_wrapper_axi_bridge_implementation_report.json` with status `passed` only if the board-wrapper flow and routed report bundle are actually produced.
- Always write `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/board_zcu104_signoff_gate/zcu104_board_wrapper_axi_bridge_subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.
- Do not write `build/board_zcu104_signoff_evidence.json`; that file belongs to a later evidence-only agent after this implementation report passes.

## Allowed Write Scope

- You may edit focused generator/source/test files needed for this board wrapper flow, including `nl2hdl/llm_kernels.py` and `tests/test_llm_kernels.py`.
- You may write generated HDL/TCL/XDC/log/report artifacts under `build/board_zcu104_signoff_gate/`.
- You may add focused docs/contracts for the board wrapper if the interface contract would otherwise be ambiguous.
- Do not edit SKILL files, parent orchestration policy, unrelated kernels, or existing passed evidence files.

## Pass Criteria

- The implementation report distinguishes scaffold-only, synthesis-only, and routed board-wrapper evidence.
- The implementation report records whether the routed top is the generated PS/PL/DDR wrapper or equivalent, not just the direct PL shell.
- The implementation report records DRC status and explicitly fails if NSTD-1 or UCIO-1 critical warnings remain.
- The implementation report records target clock MHz, observed accelerator clock name, observed clock period/frequency, and fails if the observed clock period is greater than the target period.
- The implementation report records that PS FCLK/reset, PS AXI, and DDR/address-map evidence are present in the implemented hierarchy.
- Reported timing has non-negative setup, hold, and pulse-width slack if route completes.
- Reported LUT/DSP/BRAM usage is numeric and checked against the configured budgets.
- The flow still does not claim hardware lab runtime validation or real-time LLaMA inference.

## Failure-To-SKILL Candidate

- If you cannot produce a routed board-wrapper report bundle, preserve logs and return a `skill_update_candidate` with:
  - `failing_command`
  - `symptom`
  - `root_cause_hypothesis`
  - `prevention_rule`
  - `minimal_regression_check`

## Required Commands

- `python3 -m compileall nl2hdl/llm_kernels.py nl2hdl/subagent_tasks.py nl2hdl/cli.py`
- `python3 -m pytest -q tests/test_llm_kernels.py`
- `vivado -version`
- `python3 -m nl2hdl subagents status --dispatch-plan examples/quarot_w4a4kv4_zcu104_llama32_1b/run/ddr_axi_board_shell_fixture_gate/hdl_subagent_dispatch_plan.json --evidence-root examples/quarot_w4a4kv4_zcu104_llama32_1b/run --out build/board_zcu104_signoff_gate/status_after_board_wrapper`

## Do Not Claim

- Board-level ZCU104 signoff unless the later evidence-only gate writes `board_zcu104_signoff_evidence.json`.
- Hardware lab runtime validation.
- Real-time LLaMA inference performance.
