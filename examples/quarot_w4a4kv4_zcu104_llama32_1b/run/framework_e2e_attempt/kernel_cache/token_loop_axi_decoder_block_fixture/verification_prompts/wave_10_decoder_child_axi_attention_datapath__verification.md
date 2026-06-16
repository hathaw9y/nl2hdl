# Codex Integration Verification: wave_10_decoder_child_axi_attention_datapath

You are the Codex verification agent for this dispatch wave. Audit only; do not edit source, RTL, tests, or contracts.

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
- Replay GPTQ checkpoint override: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph override: `not_configured`

## Wave Scope

- Wave id: `wave_10_decoder_child_axi_attention_datapath`
- Description: Spawn a decoder-child AXI datapath agent after AXI projection stream evidence passes.
- Target scope: `bounded_fixture_only`
- Verification mode: `integration_verification_with_synthesis`
- Direct blocked target dependencies: `none`
- Inherited blocked target dependencies: `none`
- Global blocked target dependencies: `real_mlir_model_analysis`

## Implementation Tasks To Audit

- `decoder_child_axi_attention_datapath`: role `decoder_axi_child_agent`, prompt `subagent_prompts/decoder_child_axi_attention_datapath__implementation.md`, kernel `decoder_child_axi_attention_datapath`, semantic op `decoder_child_attention_path`

## Audit Requirements

- requirement coverage
- generated RTL correctness evidence
- simulation and Verilator evidence
- Vivado setup/hold/pulse-width timing evidence when synthesis ran
- missing tests or unsafe claims
- run or inspect integration-level Vivado synthesis for the composed integration top
- confirm integration-level utilization includes child modules, FSM, adapters, and interconnect buffers
- confirm integration-level timing/resource evidence matches the active hardware spec and selected child knobs

## Integration-Level Synthesis Requirement

- This is an integration wave, so verification must run or inspect Vivado synthesis for the composed integration top after the implementation agent passes simulation.
- Write integration synthesis evidence under `build/wave_10_decoder_child_axi_attention_datapath_integration_verification`.
- Write `build/wave_10_decoder_child_axi_attention_datapath_integration_verification/integration_synthesis_report.json` with hardware spec identity, selected child knobs, command/log paths, timing, utilization, DRC, methodology, and pass/fail status.
- Integration synthesis must include the generated parent integration module plus selected child modules; child module OOC reports alone are not sufficient.
- If Vivado cannot run in this environment, record the exact command, log path, and blocker as a P1/P2 finding instead of claiming synthesis passed.
- Confirm implementation evidence matches this wave's target scope.
- Confirm no target-scale claim bypasses blocked target dependencies.
- Confirm HDL agents did not edit parent orchestration files unless explicitly allowed.
- If an implementation gate failed, confirm a `skill_update_candidate` with all required fields was returned before retry.
- Report findings first as P0/P1/P2/P3 with file and line references where possible.
- If no P0/P1/P2 issues are found, say so clearly.

## Do Not Claim

- automatic sub-agent spawning inside package runtime
- completed target RTL for every packet
- full LLaMA execution
- board-level ZCU104 signoff

## Current Blocked Target Tasks

- `real_mlir_model_analysis`: current inspect MLIR does not prove every target LLaMA GEMM/non-GEMM operation from a provided or exported model graph (analysis source `synthetic_llama_block_mlir`, coverage `synthetic_decoder_block_mlir_not_exported_target_model`)
- `full_llama_model_execution`: current gates are bounded fixtures and target planning; full multi-layer/token/DDR/AXI execution is not implemented
- `board_level_zcu104_signoff`: current Vivado evidence is fixture post-route timing without board I/O, DDR/AXI, or PS/PL constraints

Verification means no source edits, no RTL rewrites, and no test weakening. Integration verification may write generated Vivado evidence and the verification JSON only.
