# Codex Read-Only Verification: wave_1_non_gemm_kernels

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

- Wave id: `wave_1_non_gemm_kernels`
- Description: Spawn non-GEMM implementation agents for RMSNorm/RoPE/attention/control/residual/MLP fixtures.
- Target scope: `bounded_fixture_only`
- Verification mode: `read_only`
- Direct blocked target dependencies: `none`
- Inherited blocked target dependencies: `none`
- Global blocked target dependencies: `real_mlir_model_analysis`

## Implementation Tasks To Audit

- `non_gemm_input_layernorm`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_input_layernorm__implementation.md`, kernel `rmsnorm_rope_source_path`, semantic op `input_layernorm`
- `non_gemm_rope_qk`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_rope_qk__implementation.md`, kernel `rmsnorm_rope_source_path`, semantic op `rope_qk`
- `non_gemm_attention_scores_softmax_kv`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_attention_scores_softmax_kv__implementation.md`, kernel `attention_kv_cache_fixture`, semantic op `attention_scores_softmax_kv`
- `non_gemm_attention_residual`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_attention_residual__implementation.md`, kernel `residual_mlp_fixture`, semantic op `attention_residual`
- `non_gemm_post_attention_layernorm`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_post_attention_layernorm__implementation.md`, kernel `rmsnorm_rope_source_path`, semantic op `post_attention_layernorm`
- `non_gemm_silu_gate`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_silu_gate__implementation.md`, kernel `residual_mlp_fixture`, semantic op `silu_gate`
- `non_gemm_swiglu_multiply`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_swiglu_multiply__implementation.md`, kernel `residual_mlp_fixture`, semantic op `swiglu_multiply`
- `non_gemm_mlp_residual`: role `non_gemm_kernel_agent`, prompt `subagent_prompts/non_gemm_mlp_residual__implementation.md`, kernel `residual_mlp_fixture`, semantic op `mlp_residual`

## Audit Requirements

- requirement coverage
- generated RTL correctness evidence
- simulation and Verilator evidence
- Vivado setup/hold/pulse-width timing evidence when synthesis ran
- missing tests or unsafe claims
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
