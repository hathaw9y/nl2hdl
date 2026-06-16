# Execution Summary

Command:

```bash
python3 -m nl2hdl agent \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/quarot_w4a4kv4_zcu104_llama32_1b/input.yaml \
  --mode inspect \
  --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/inspect \
  --verbose
```

Result:

- `llm_agent_report.status`: `passed`
- `input_clarification.status`: `clear`
- `input_clarification.question_count`: `0`
- projection tasks: `7`
- non-GEMM tasks: `8`
- integration tasks: `42`
- blocked target tasks: `3`
- generated implementation prompts: `57`
- generated verification prompts: `20`

GPTQ/weight-stream result:

- checkpoint source preflight: `resolved_local_path`
- GPTQ checkpoint metadata: `parsed`
- GPTQ bits: `4`
- GPTQ group size: `128`
- complete GPTQ projection metadata count: `7`
- GPTQ weight layout preflight: `passed`
- projection payload-prefix probes: `7 / 7`
- projection stream plan: ready for all projection names

Target readiness:

- status: `target_blocked`
- safe to spawn bounded sub-agents: `true`
- safe to claim target accelerator: `false`

Blocked target tasks:

- `real_mlir_model_analysis`: the run used the synthetic LLaMA-block MLIR
  fixture, not an exported/provided full model graph.
- `full_llama_model_execution`: full multi-layer/token/DDR/AXI execution is
  not implemented by this inspect example.
- `board_level_zcu104_signoff`: board-level routed evidence is not produced by
  this inspect example.

Generated agent prompts:

- Hand-authored high-level prompts are in `agent_prompts/`.
- Machine-generated implementation prompts are in `run/inspect/subagent_prompts/`.
- Machine-generated verification prompts are in `run/inspect/verification_prompts/`.

