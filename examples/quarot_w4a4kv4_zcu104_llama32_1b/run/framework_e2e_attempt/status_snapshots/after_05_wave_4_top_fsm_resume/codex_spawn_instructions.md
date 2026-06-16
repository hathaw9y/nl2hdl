# Codex Sub-Agent Spawn Instructions

This file is a runner-facing view of `hdl_subagent_execution_manifest.json`.
Package code does not spawn agents; the interactive Codex parent or an external runner uses these entries.

- Spawn entries: `1`
- Implementation agents: `1`
- Verification agents: `0`
- Parallel spawn allowed: `False`
- Max parallel batch size: `0`

## Batch `wave_5_token_loop__implementation_agent`

- Wave: `wave_5_token_loop`
- Kind: `implementation_agent`
- Parallel allowed: `False`
- Entry count: `1`

### `token_loop_decoder_block_fixture`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/token_loop_decoder_block_fixture__implementation.md`
- Fork context: `True`
- Evidence dir: `build/token_loop_decoder_block_fixture_gate`
- Module OOC synthesis: `build/token_loop_decoder_block_fixture_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/token_loop_decoder_block_fixture_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel token_loop_decoder_block_fixture --out build/token_loop_decoder_block_fixture_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/token_loop_decoder_block_fixture__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/token_loop_decoder_block_fixture_gate/kernel_report.json` plus `build/token_loop_decoder_block_fixture_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/token_loop_decoder_block_fixture_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

## Blocked Waves

### `wave_6_projection_axi_read_command_adapter`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_5_token_loop`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_5_token_loop`

### `wave_7_projection_axi_read_data_channel_adapter`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_6_projection_axi_read_command_adapter`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_6_projection_axi_read_command_adapter`

### `wave_8_projection_axi_read_transaction_adapter`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_7_projection_axi_read_data_channel_adapter`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_7_projection_axi_read_data_channel_adapter`

### `wave_9_projection_axi_stream_integration`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_8_projection_axi_read_transaction_adapter`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_8_projection_axi_read_transaction_adapter`

### `wave_10_decoder_child_axi_attention_datapath`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_9_projection_axi_stream_integration`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_9_projection_axi_stream_integration`

### `wave_11_layer_fsm_axi_attention_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_10_decoder_child_axi_attention_datapath`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_10_decoder_child_axi_attention_datapath`

### `wave_12_top_fsm_axi_attention_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_11_layer_fsm_axi_attention_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_11_layer_fsm_axi_attention_fixture`

### `wave_13_token_loop_axi_attention_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_12_top_fsm_axi_attention_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_12_top_fsm_axi_attention_fixture`

### `wave_14_decoder_block_axi_attention_mlp_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_13_token_loop_axi_attention_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_13_token_loop_axi_attention_fixture`

### `wave_15_layer_fsm_axi_decoder_block_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_14_decoder_block_axi_attention_mlp_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_14_decoder_block_axi_attention_mlp_fixture`

### `wave_16_top_fsm_axi_decoder_block_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_15_layer_fsm_axi_decoder_block_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_15_layer_fsm_axi_decoder_block_fixture`

### `wave_17_token_loop_axi_decoder_block_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_16_top_fsm_axi_decoder_block_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_16_top_fsm_axi_decoder_block_fixture`

### `wave_18_model_fsm_axi_decoder_block_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_17_token_loop_axi_decoder_block_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_17_token_loop_axi_decoder_block_fixture`

### `wave_19_ddr_axi_board_shell_fixture`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_18_model_fsm_axi_decoder_block_fixture`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_18_model_fsm_axi_decoder_block_fixture`

## Does Not Claim

- sub-agent execution occurred
- automatic sub-agent spawning inside package runtime
- generated RTL completeness
- full LLaMA execution
- board-level ZCU104 signoff
