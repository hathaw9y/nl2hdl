# Target Evidence Sub-Agent Task: model_level_execution_harness

You are the Codex implementation sub-agent for the model-level execution harness required before full LLaMA execution evidence.
Do not only inspect the existing bounded fixture reports. Implement the smallest non-HDL model-level harness/report path that can prove, or explicitly fail to prove, the model-level loop.
Do not write final `full_llama_execution_evidence.json`. Your job is to create the missing harness report only if it proves the model-level loop is no longer a bounded 2-layer fixture.

## Target Context

- Model: `meta-llama/Llama-3.2-1B`
- Decoder layers required: `16`
- FPGA part: `xczu7ev-ffvc1156-2-e`
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
- Target preflight status: `blocked`
- Non-passed waves: `['wave_12_top_fsm_axi_attention_fixture', 'wave_13_token_loop_axi_attention_fixture', 'wave_14_decoder_block_axi_attention_mlp_fixture', 'wave_15_layer_fsm_axi_decoder_block_fixture', 'wave_16_top_fsm_axi_decoder_block_fixture', 'wave_17_token_loop_axi_decoder_block_fixture', 'wave_18_model_fsm_axi_decoder_block_fixture', 'wave_19_ddr_axi_board_shell_fixture']`
- Existing harness failures: `['model_level_execution_harness_report.json not found']`

## Required Output

- Write `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence/full_llama_execution_gate/model_level_execution_harness_report.json` only after it proves 16-layer model-level execution and Python reference comparison.
- Write `build/full_llama_execution_gate/model_level_harness_subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.
- Do not write `build/full_llama_execution_evidence.json`; that is a downstream target-evidence gate.

## Harness Report Requirements

- `artifact == model_level_execution_harness_report`.
- `status == passed`.
- `executed_layer_count >= dispatch_plan.model.decoder_layers`.
- `full_model_layers_executed == true` or `target_16_layer_iteration == true`.
- `python_reference_comparison.passed == true`.
- Bounded fixture fields such as `fixture_layer_count: 2`, `target_16_layer_iteration: false`, or `full_llama_model: false` must not be used as passing evidence.
- `board_level_signoff` must remain false or absent.

## Allowed Write Scope

- You may write files under `build/full_llama_execution_gate/`.
- You may add or update focused non-HDL harness code and tests needed to produce this report.
- Preferred source scope for harness/report generation is `nl2hdl/llm_kernels.py` or a small focused helper imported by it.
- Preferred test scope is focused assertions in `tests/test_llm_kernels.py`.
- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.
- Do not modify HDL/SystemVerilog unless a separate HDL packet failure is identified; if so, return a `skill_update_candidate` instead of broad edits.

## Failure-To-SKILL Candidate

- If you cannot prove the harness, preserve evidence and return a `skill_update_candidate` with:
  - `failing_command`
  - `symptom`
  - `root_cause_hypothesis`
  - `prevention_rule`
  - `minimal_regression_check`

## Required Commands

- `python3 -m pytest -q tests/test_llm_kernels.py`
- `python3 -m nl2hdl subagents status --dispatch-plan examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/00_parent_inspect/hdl_subagent_dispatch_plan.json --evidence-root examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence --out build/full_llama_execution_gate/model_harness_status_after`

## Do Not Claim

- Full LLaMA execution readiness until the downstream `full_llama_execution_evidence.json` gate passes.
- Board-level ZCU104 signoff.
- Real-time LLaMA inference performance.
- Hardware lab runtime validation.
