# Target Blocker Remediation Plan

- Status: `blocked`
- Model: `meta-llama/Llama-3.2-1B`
- Blocked target task count: `3`
- Safe to claim target accelerator: `False`

## Canonical Full Preflight

```bash
python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode inspect --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/kernel_cache/top_fsm_axi_attention_fixture --skip-synth --mlir-graph '<exported_or_provided_model_graph.mlir>' --gptq-checkpoint '<local_gptq_int4_checkpoint_dir_or_hf_repo>'
```

## Steps

### real_mlir_model_analysis

- Goal: prove the inspected graph came from the target/exported model and covers every required LLaMA op
- Blocked reason: current inspect MLIR does not prove every target LLaMA GEMM/non-GEMM operation from a provided or exported model graph

```bash
python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode inspect --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/framework_e2e_attempt/kernel_cache/top_fsm_axi_attention_fixture --skip-synth --mlir-graph '<exported_or_provided_model_graph.mlir>'
```

Required evidence:
- `model.mlir_graph or --mlir-graph points to an existing MLIR file`
- `exported node identities may be exact semantic names or HF-style loc()/path aliases recorded in mlir_semantic_alias_map`
- `mlir_model_analysis_readiness.status == passed`
- `missing_semantic_gemm_ops == []`
- `missing_semantic_non_gemm_ops == []`
- `mlir_unsupported_ops == []`

### full_llama_model_execution

- Goal: run an end-to-end multi-layer/token LLaMA numerical fixture after kernel and FSM evidence exists
- Blocked reason: current gates are bounded fixtures and target planning; full multi-layer/token/DDR/AXI execution is not implemented

```bash
blocked until target preflight and decoder/token/model FSM evidence are all passed
```

Required evidence:
- `target_preflight.status == passed`
- `decoder/token/model FSM verification reports are passed`
- `Python reference comparison is recorded for the selected decode path`

### board_level_zcu104_signoff

- Goal: produce board-level Vivado evidence for xczu7ev-ffvc1156-2-e with real clock, reset, I/O, PS/PL, and DDR constraints
- Blocked reason: current Vivado evidence is fixture post-route timing without board I/O, DDR/AXI, or PS/PL constraints

```bash
blocked until target preflight and board shell integration are ready for Vivado
```

Required evidence:
- `Vivado synthesis and timing reports are present`
- `timing summary has no setup/hold/pulse-width violations`
- `resource utilization is under configured ZCU104 budgets`

## Required Artifacts
- `mlir_model_analysis_readiness.json`
- `gptq_checkpoint_source_preflight.json`
- `gptq_checkpoint_metadata.json`
- `gptq_weight_layout_preflight.json`
- `gptq_payload_probe.json`
- `projection_weight_stream_plan.json`
- `target_readiness_report.json`

## Sub-Agent Policy

- Parent agent does not write HDL.
- Implementation sub-agents write RTL or generators.
- Verification sub-agents are read-only.
- Failed HDL attempts must produce a SKILL update before retry.
