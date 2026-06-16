# Codex Sub-Agent Spawn Instructions

This file is a runner-facing view of `hdl_subagent_execution_manifest.json`.
Package code does not spawn agents; the interactive Codex parent or an external runner uses these entries.

- Spawn entries: `15`
- Implementation agents: `15`
- Verification agents: `0`
- Parallel spawn allowed: `True`
- Max parallel batch size: `8`

## Batch `wave_1_projection_kernels__implementation_agent`

- Wave: `wave_1_projection_kernels`
- Kind: `implementation_agent`
- Parallel allowed: `True`
- Entry count: `7`

### `projection_q_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_q_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_q_proj_gate`
- Module OOC synthesis: `build/projection_q_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_q_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=q_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_q_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_q_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_q_proj_gate/kernel_report.json` plus `build/projection_q_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_q_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_k_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_k_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_k_proj_gate`
- Module OOC synthesis: `build/projection_k_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_k_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=k_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_k_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_k_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_k_proj_gate/kernel_report.json` plus `build/projection_k_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_k_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_v_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_v_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_v_proj_gate`
- Module OOC synthesis: `build/projection_v_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_v_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=v_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_v_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_v_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_v_proj_gate/kernel_report.json` plus `build/projection_v_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_v_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_o_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_o_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_o_proj_gate`
- Module OOC synthesis: `build/projection_o_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_o_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=o_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_o_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_o_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_o_proj_gate/kernel_report.json` plus `build/projection_o_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_o_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_gate_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_gate_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_gate_proj_gate`
- Module OOC synthesis: `build/projection_gate_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_gate_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=gate_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_gate_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_gate_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_gate_proj_gate/kernel_report.json` plus `build/projection_gate_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_gate_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_up_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_up_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_up_proj_gate`
- Module OOC synthesis: `build/projection_up_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_up_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=up_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_up_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_up_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_up_proj_gate/kernel_report.json` plus `build/projection_up_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_up_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `projection_down_proj`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/projection_down_proj__implementation.md`
- Fork context: `True`
- Evidence dir: `build/projection_down_proj_gate`
- Module OOC synthesis: `build/projection_down_proj_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/projection_down_proj_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_PROJECTION=down_proj python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel projection_target_stream_plan --out build/projection_down_proj_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/projection_down_proj__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/projection_down_proj_gate/kernel_report.json` plus `build/projection_down_proj_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/projection_down_proj_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

## Batch `wave_1_non_gemm_kernels__implementation_agent`

- Wave: `wave_1_non_gemm_kernels`
- Kind: `implementation_agent`
- Parallel allowed: `True`
- Entry count: `8`

### `non_gemm_input_layernorm`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_input_layernorm__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_input_layernorm_gate`
- Module OOC synthesis: `build/non_gemm_input_layernorm_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_input_layernorm_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=input_layernorm python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel rmsnorm_rope_source_path --out build/non_gemm_input_layernorm_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_input_layernorm__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_input_layernorm_gate/kernel_report.json` plus `build/non_gemm_input_layernorm_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_input_layernorm_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_rope_qk`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_rope_qk__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_rope_qk_gate`
- Module OOC synthesis: `build/non_gemm_rope_qk_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_rope_qk_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=rope_qk python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel rmsnorm_rope_source_path --out build/non_gemm_rope_qk_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_rope_qk__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_rope_qk_gate/kernel_report.json` plus `build/non_gemm_rope_qk_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_rope_qk_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_attention_scores_softmax_kv`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_attention_scores_softmax_kv__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_attention_scores_softmax_kv_gate`
- Module OOC synthesis: `build/non_gemm_attention_scores_softmax_kv_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_attention_scores_softmax_kv_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=attention_scores_softmax_kv python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel attention_kv_cache_fixture --out build/non_gemm_attention_scores_softmax_kv_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_attention_scores_softmax_kv__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_attention_scores_softmax_kv_gate/kernel_report.json` plus `build/non_gemm_attention_scores_softmax_kv_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_attention_scores_softmax_kv_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_attention_residual`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_attention_residual__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_attention_residual_gate`
- Module OOC synthesis: `build/non_gemm_attention_residual_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_attention_residual_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=attention_residual python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel residual_mlp_fixture --out build/non_gemm_attention_residual_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_attention_residual__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_attention_residual_gate/kernel_report.json` plus `build/non_gemm_attention_residual_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_attention_residual_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_post_attention_layernorm`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_post_attention_layernorm__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_post_attention_layernorm_gate`
- Module OOC synthesis: `build/non_gemm_post_attention_layernorm_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_post_attention_layernorm_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=post_attention_layernorm python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel rmsnorm_rope_source_path --out build/non_gemm_post_attention_layernorm_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_post_attention_layernorm__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_post_attention_layernorm_gate/kernel_report.json` plus `build/non_gemm_post_attention_layernorm_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_post_attention_layernorm_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_silu_gate`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_silu_gate__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_silu_gate_gate`
- Module OOC synthesis: `build/non_gemm_silu_gate_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_silu_gate_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=silu_gate python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel residual_mlp_fixture --out build/non_gemm_silu_gate_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_silu_gate__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_silu_gate_gate/kernel_report.json` plus `build/non_gemm_silu_gate_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_silu_gate_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_swiglu_multiply`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_swiglu_multiply__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_swiglu_multiply_gate`
- Module OOC synthesis: `build/non_gemm_swiglu_multiply_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_swiglu_multiply_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=swiglu_multiply python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel residual_mlp_fixture --out build/non_gemm_swiglu_multiply_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_swiglu_multiply__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_swiglu_multiply_gate/kernel_report.json` plus `build/non_gemm_swiglu_multiply_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_swiglu_multiply_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

### `non_gemm_mlp_residual`

- Agent: `Codex`
- Mode: `read_write_hdl_packet`
- Prompt file: `subagent_prompts/non_gemm_mlp_residual__implementation.md`
- Fork context: `True`
- Evidence dir: `build/non_gemm_mlp_residual_gate`
- Module OOC synthesis: `build/non_gemm_mlp_residual_gate/module_ooc_synthesis_report.json`
- Sub-agent result: `build/non_gemm_mlp_residual_gate/subagent_result.json`
- Replay model name: `meta-llama/Llama-3.2-1B`
- Replay GPTQ checkpoint: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
- Replay MLIR graph: `not_configured`

Required commands:
- `python3 -m pytest -q tests/test_llm_kernels.py -q`
- `NL2HDL_SELECTED_NONGEMM=mlp_residual python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --gptq-checkpoint examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint --mode kernel --kernel residual_mlp_fixture --out build/non_gemm_mlp_residual_gate --verbose`
- Wave blocked target dependencies: `real_mlir_model_analysis`
- Wave global blocked target dependencies: `real_mlir_model_analysis`

Spawn message:

```text
You are the HDL implementation sub-agent for this single packet. Read this execution manifest's sibling prompt file `subagent_prompts/non_gemm_mlp_residual__implementation.md`, edit only the allowed write scope, run the required checks, and write `build/non_gemm_mlp_residual_gate/kernel_report.json` plus `build/non_gemm_mlp_residual_gate/module_ooc_synthesis_report.json` when this is a real datapath module, and `build/non_gemm_mlp_residual_gate/subagent_result.json`. If the gate fails, preserve evidence and return a complete skill_update_candidate.
```

## Blocked Waves

### `wave_2_decoder_block`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_1_projection_kernels, wave_1_non_gemm_kernels`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_1_projection_kernels, wave_1_non_gemm_kernels`

### `wave_3_layer_fsm`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_2_decoder_block`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_2_decoder_block`

### `wave_4_top_fsm`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_3_layer_fsm`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_3_layer_fsm`

### `wave_5_token_loop`

- Status: `blocked_by_dependency`
- Reason: `waiting for waves: wave_4_top_fsm`
- Next action: `wait_for_dependency`
- Depends on waves: `wave_4_top_fsm`

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
