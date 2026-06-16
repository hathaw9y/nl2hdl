# Codex Sub-Agent Spawn Instructions

This file is a runner-facing view of `hdl_subagent_execution_manifest.json`.
Package code does not spawn agents; the interactive Codex parent or an external runner uses these entries.

- Spawn entries: `7`
- Implementation agents: `7`
- Verification agents: `0`
- Parallel spawn allowed: `True`
- Max parallel batch size: `7`

## Batch `wave_9_projection_axi_stream_integration__implementation_agent`

- Wave: `wave_9_projection_axi_stream_integration`
- Kind: `implementation_agent`
- Parallel allowed: `True`
- Entry count: `7`

### `projection_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=q_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_axi_stream_integration_gate/kernel_report.json` plus `build/projection_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_k_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_k_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_k_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_k_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_k_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=k_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_k_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_k_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_k_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_k_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_k_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_v_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_v_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_v_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_v_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_v_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=v_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_v_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_v_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_v_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_v_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_v_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_o_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_o_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_o_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_o_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_o_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=o_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_o_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_o_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_o_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_o_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_o_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_gate_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_gate_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_gate_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_gate_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_gate_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=gate_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_gate_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_gate_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_gate_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_gate_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_gate_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_up_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_up_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_up_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_up_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_up_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=up_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_up_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_up_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_up_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_up_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_up_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_down_proj_axi_stream_integration`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_down_proj_axi_stream_integration__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_down_proj_axi_stream_integration_gate`
- Module OOC synthesis: `build/projection_down_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_down_proj_axi_stream_integration_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=down_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_axi_stream_integration --out build/projection_down_proj_axi_stream_integration_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_down_proj_axi_stream_integration__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_down_proj_axi_stream_integration_gate/kernel_report.json` plus `build/projection_down_proj_axi_stream_integration_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_down_proj_axi_stream_integration_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

## Blocked Waves

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
