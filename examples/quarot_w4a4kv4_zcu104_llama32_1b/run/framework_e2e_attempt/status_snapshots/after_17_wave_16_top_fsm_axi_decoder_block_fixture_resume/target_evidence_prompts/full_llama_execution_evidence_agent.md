# Target Evidence Sub-Agent Task: full_llama_execution

You are the Codex target-evidence sub-agent for full LLaMA execution readiness.
The parent agent must not hand-write HDL or fabricate evidence. Your job is to inspect the passed child evidence, run or add the smallest necessary model-level execution harness, and write the required evidence only if it is proven.

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
- Passed wave count: `17`
- Non-passed waves: `['wave_17_token_loop_axi_decoder_block_fixture', 'wave_18_model_fsm_axi_decoder_block_fixture', 'wave_19_ddr_axi_board_shell_fixture']`
- Current evidence failures: `['full_llama_execution_evidence.json not found']`
- Model-level harness failures: `['model_level_execution_harness_report.json not found']`

If `Ready to spawn` is false, do not force the gate. Return a `skill_update_candidate` or a precise missing-evidence report instead.

## Required Output

- Write `examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence/full_llama_execution_evidence.json` only after every required field below is backed by concrete artifacts.
- Also write `build/full_llama_execution_gate/subagent_result.json` with changed files, commands run, evidence paths, and remaining risks.
- Do not set `board_level_signoff` to true. Board signoff is a separate downstream gate.

## Evidence Schema

- Required fields: `artifact, status, model, target_preflight_status, full_model_layers_executed, executed_layer_count, token_loop_evidence, model_fsm_evidence, checkpoint_payload_evidence, python_reference_comparison, board_level_signoff`
- Template artifact: `full_llama_execution_evidence`
- Template JSON: `{'artifact': 'full_llama_execution_evidence', 'status': '<passed>', 'model': 'meta-llama/Llama-3.2-1B', 'target_preflight_status': '<passed>', 'full_model_layers_executed': '<true>', 'executed_layer_count': 16, 'token_loop_evidence': {'passed': '<true>', 'source_wave_id': 'wave_17_token_loop_axi_decoder_block_fixture', 'report': '<path to token-loop/model execution report>'}, 'model_fsm_evidence': {'passed': '<true>', 'source_wave_id': 'wave_18_model_fsm_axi_decoder_block_fixture', 'report': '<path to model FSM execution report>'}, 'checkpoint_payload_evidence': {'passed': '<true>', 'source': 'gptq_payload_probe.json', 'projection_payloads_verified': '<all required projections>'}, 'python_reference_comparison': {'passed': '<true>', 'tolerance_lsb': '<configured tolerance>', 'reference_artifact': '<path to Python/NumPy reference output>', 'rtl_artifact': '<path to RTL/model output>'}, 'board_level_signoff': False}`

## Evidence That Must Be Proven

- `target_preflight_status == passed` from parent readiness.
- Every decoder layer required by the dispatch plan is executed or explicitly covered by a model-level loop fixture.
- `wave_17_token_loop_axi_decoder_block_fixture` evidence is consumed for token-loop scheduling.
- `wave_18_model_fsm_axi_decoder_block_fixture` evidence is consumed for model-level FSM scheduling.
- GPTQ checkpoint payload evidence covers every required projection, not only q_proj.
- Python/NumPy or PyTorch reference comparison passes within configured tolerance.
- The produced evidence must remain scoped to full model execution and must not claim ZCU104 board signoff.

## Allowed Write Scope

- You may write `build/full_llama_execution_evidence.json` and files under `build/full_llama_execution_gate/`.
- You may add or update focused non-HDL harness/tests only if they are necessary to prove model-level execution evidence.
- Do not edit parent orchestration files: `nl2hdl/subagent_tasks.py`, `nl2hdl/cli.py`, or dispatch/status builders.
- Do not weaken existing tests, evidence gates, blocked-target language, or board-signoff separation.
- Do not rewrite passed child HDL packets unless a new failure is found; if that happens, return a `skill_update_candidate` before retry.

## Failure-To-SKILL Candidate

- If you cannot prove full execution, preserve evidence and return a `skill_update_candidate` with:
  - `failing_command`
  - `symptom`
  - `root_cause_hypothesis`
  - `prevention_rule`
  - `minimal_regression_check`

## Required Commands

- `python3 -m pytest -q tests/test_llm_kernels.py`
- `python3 -m nl2hdl subagents status --dispatch-plan examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/00_parent_inspect/hdl_subagent_dispatch_plan.json --evidence-root examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/evidence --out build/full_llama_execution_gate/status_after`

## Do Not Claim

- Board-level ZCU104 signoff.
- Real-time LLaMA inference performance.
- Hardware lab runtime validation.
- New HDL correctness beyond the already verified child evidence unless you create and verify a separate HDL sub-agent packet.
