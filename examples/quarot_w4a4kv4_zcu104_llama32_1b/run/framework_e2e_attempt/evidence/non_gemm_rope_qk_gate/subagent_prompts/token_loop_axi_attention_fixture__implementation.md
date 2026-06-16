# HDL Sub-Agent Task: token_loop_axi_attention_fixture

Role: `token_loop_axi_agent`.

You are one HDL implementation sub-agent for exactly this packet. The parent agent coordinates only; you own the assigned RTL/generator work and must verify it before reporting success.
Do not wait for the parent to write RTL. If this packet needs RTL or generator changes, make those changes inside the allowed scope and run the required checks yourself.

## Target Context

- Model: `meta-llama/Llama-3.2-1B`
- Replay model name: `meta-llama/Llama-3.2-1B`
- FPGA part: `xczu7ev-ffvc1156-2-e`
- Target clock: `200 MHz`
- Quantization: `quarot_w4a4kv4_gptq_weights`
- Optimization brief: `Use QuaRot-style rotations so the LLaMA decode path can target W4A4KV4: 4-bit GPTQ weights, 4-bit rotated activations, and 4-bit KV-cache storage. The example assumes GPTQ weight tensors already exist and uses a local sparse metadata fixture only to exercise the parent planning flow.
`
- Design style alias: `systolic_weight_stationary_llm_streaming`
- Compute style: `systolic_array`
- Execution style: `llm_decoder_streaming`
- Memory style: `external_ddr_streaming`
- Control style: `hierarchical_fsm`
- Architecture brief: `Use a weight-stationary systolic array for projection GEMM/GEMV tiles. GPTQ-packed W4 weights are streamed from DDR into on-chip tile buffers, unpacked/dequantized near the array edge, and reused in-place while rotated A4 activations stream through the array. Size the array from the active DSP budget rather than hardcoding a final dimension.
`
- Optimization candidates: `[{'name': 'quarot_w4a4kv4', 'scope': 'primary example path', 'notes': 'rotate residual/attention/MLP streams, quantize activations to 4-bit, and store KV-cache in 4-bit form'}, {'name': 'w4a8kv4_fallback', 'scope': 'fallback', 'notes': 'keep weights and KV-cache at 4-bit but allow 8-bit activations if W4A4 timing or accuracy is unsafe'}, {'name': 'mixed_precision_softmax_control', 'scope': 'non_gemm_control', 'notes': 'keep softmax/control accumulations wider while preserving W4A4KV4 storage boundaries'}]`
- Design candidates: `[{'name': 'ws_systolic_32x32', 'focus': 'first candidate under a 1536-DSP active compute budget', 'risk': 'routing pressure and activation/KV4 quantization overhead may reduce timing margin'}, {'name': 'ws_systolic_16x64', 'focus': 'preserve projection throughput while easing vertical routing', 'risk': 'less square data reuse and more edge buffering'}, {'name': 'ws_systolic_24x48', 'focus': 'middle-ground DSP use with easier BRAM banking', 'risk': 'tile scheduler is less regular than 32x32'}]`
- Contract: `docs/token_loop_axi_attention_fixture_contract.md`
- Replay GPTQ checkpoint override: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph override: `not_configured`

## Assigned Operation

- Task id: `token_loop_axi_attention_fixture`
- Partition: `token_loop_memory_to_attention_integration`
- Semantic op: `bounded_two_token_axi_attention_loop`
- Regression kernel: `token_loop_axi_attention_fixture`
- Status before assignment: `ready_for_hdl_subagent_after_top_fsm_axi_attention_fixture`
- Target qweight shard file: `model.safetensors`
- Target qweight tensor key: `model.layers.0.self_attn.q_proj.qweight`
- Target qweight byte offset: `0`
- Target qweight byte count: `2097152`
- Target request byte address: `0`
- Target request beat count: `131072`
- Target first-beat byte offset: `0`
- Target last-beat valid bytes: `16`
- Target request covers unaligned qweight range: `True`
- Target stream plan valid: `True`
- GPTQ payload probe status: `sampled`
- GPTQ payload probe source: `gptq_payload_probe.json`
- GPTQ payload selected projection: `True`
- GPTQ qweight payload order: `safetensors_payload_prefix_32bit_little_endian_words`
- GPTQ qweight sample bytes: `64`
- GPTQ qweight payload word count: `16`
- GPTQ qweight payload words32 LE: `['0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000', '0x00000000']`
- GPTQ qweight first memory beats 128b LE: `['0x00000000000000000000000000000000', '0x00000000000000000000000000000000', '0x00000000000000000000000000000000', '0x00000000000000000000000000000000']`
- GPTQ qweight memory beat word chunks32 LE: `[['0x00000000', '0x00000000', '0x00000000', '0x00000000'], ['0x00000000', '0x00000000', '0x00000000', '0x00000000'], ['0x00000000', '0x00000000', '0x00000000', '0x00000000'], ['0x00000000', '0x00000000', '0x00000000', '0x00000000']]`
- GPTQ qweight covers first memory beat: `True`
- Target checkpoint payload dependency: `satisfied_by_payload_probe`
- Child tasks: `top_fsm_axi_attention_fixture`

## Current Target Gate Blocks

- `real_mlir_model_analysis`: current inspect MLIR does not prove every target LLaMA GEMM/non-GEMM operation from a provided or exported model graph (analysis source `synthetic_llama_block_mlir`, coverage `synthetic_decoder_block_mlir_not_exported_target_model`)
- `full_llama_model_execution`: current gates are bounded fixtures and target planning; full multi-layer/token/DDR/AXI execution is not implemented
- `board_level_zcu104_signoff`: current Vivado evidence is fixture post-route timing without board I/O, DDR/AXI, or PS/PL constraints

These blocks apply to target-scale claims even when this packet can still improve bounded fixture RTL.

## Required Interface

- Follow `docs/hdl_module_interface_contract.md`.
- Include common handshake ports: `aclk`, `aresetn`, `start_i`, and `done_o`.
- Outputs must be stable when `done_o` is asserted.
- Packed vectors use little element order: element `idx` is `[idx*WIDTH +: WIDTH]`.

## Machine-Readable Module Contract

- Contract bundle artifact: `hdl_module_contract_bundle`
- Clock/reset: `{'clock': 'aclk', 'reset': 'aresetn', 'reset_style': 'synchronous_active_low'}`
- Handshake ports: `{'start': 'start_i', 'done': 'done_o'}`
- Parent boundary: `{'parent_must_not_write_hdl': True, 'subagent_owns_rtl_or_generator_changes': True, 'integration_boundary': 'token_loop_schedules_verified_top_fsm'}`
- Final response required fields: `changed_files, commands_run, simulation_evidence, verilator_evidence, vivado_timing_resource_evidence, module_ooc_synthesis_evidence, remaining_risks`

## Allowed Write Scope

- Do not edit unrelated files.
- Assigned generator/source scope: nl2hdl/llm_kernels.py changes only for kernel `token_loop_axi_attention_fixture`.
- Assigned test scope: add or update only task-specific assertions in tests/test_llm_kernels.py.
- Assigned contract scope: read `docs/token_loop_axi_attention_fixture_contract.md`; edit it only if the task explicitly requires contract clarification.
- Generated evidence should go under build/token_loop_axi_attention_fixture_gate/.
- Do not edit parent orchestration files such as nl2hdl/llm_agent.py, nl2hdl/cli.py, nl2hdl/subagent_tasks.py, or manifest/report generators.
- Do not weaken existing tests, contracts, timing gates, or forbidden-claim language.
- Verification agents are read-only unless explicitly reassigned as implementation agents.

## Required Evidence

- Token loop instantiates the AXI-aware Top FSM child RTL
- Token-loop trace shows two deterministic token calls to top_fsm_axi_attention_fixture
- Top child start is held while busy, deasserted after done, and released before each next token call
- AXI projection child metadata-good bits propagated through decoder child, Layer FSM, and Top FSM remain visible in token-loop compact status
- Nested AXI projection payload match, backpressure, projection output, and round-trip evidence are dynamically observed through token-loop hierarchy
- Nested attention/KV evidence and final token output remain dynamically observed
- compact top-level I/O avoids exposing 128-bit AXI data, full KV arrays, child vectors, or debug buses
- post-route setup/hold/pulse-width timing when synthesized
- `kernel_report.json` must state coverage level and implementation stage.
- Real datapath modules must also write `module_ooc_synthesis_report.json` before any integration wave can consume them.
- `module_ooc_synthesis_report.json` must include Vivado part/clock, setup/hold/pulse-width status, LUT/DSP/BRAM/URAM/FF/I/O utilization, selected tuning knobs, and resource assessment.
- If this packet is only a fixture/control scaffold, state the fixture-only OOC waiver explicitly in `kernel_report.json`.
- If Vivado runs, setup, hold, and pulse-width timing must all have non-negative slack and zero failing endpoints.
- Final response must list changed files, commands run, simulation evidence, timing/resource evidence, module OOC synthesis evidence, selected knobs, and remaining risks.
- Also write `subagent_result.json` in your evidence directory with changed files, commands run, simulation evidence, Verilator evidence, Vivado timing/resource evidence, module OOC synthesis evidence, remaining risks, and any `skill_update_candidate`.
- If you cannot pass the gate, preserve the failing evidence and return a reusable `skill_update_candidate` with failing command, symptom, root-cause hypothesis, prevention rule, and minimal regression check.

## Failure-To-SKILL Candidate

- If this gate fails, do not hide the failure and do not retry the same pattern blindly.
- Return a `skill_update_candidate` containing these required fields:
  - `failing_command`
  - `symptom`
  - `root_cause_hypothesis`
  - `prevention_rule`
  - `minimal_regression_check`
- Evidence directory for this candidate: `build/token_loop_axi_attention_fixture_gate/`
- The parent will convert reusable prevention rules into a SKILL before retrying.

## Do Not Claim

- DDR controller integration
- multi-burst or outstanding AXI master
- full qweight payload streaming
- mathematically complete Q/K/V/O attention wiring
- full target projection execution
- full sequence KV-cache
- multi-layer model execution
- full model execution
- board-level signoff
- Full LLaMA execution unless this exact task proves it.
- Board-level ZCU104 signoff unless board I/O, DDR/AXI, and PS/PL constraints are included.

## Required Commands

- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel token_loop_axi_attention_fixture --out build/token_loop_axi_attention_fixture_gate --verbose`
