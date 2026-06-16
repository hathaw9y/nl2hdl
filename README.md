# nl2hdl

`nl2hdl` is a v1 coding-agent pipeline that looks at a model, decides whether
the graph is small enough for the supported accelerator template, and emits a
direct SystemVerilog implementation.

The first target is intentionally narrow: fixed-shape dense DNNs with `MatMul` /
`Gemm`, optional bias `Add`, `Relu`, `Flatten`, and `Reshape`. Unsupported models
fail early with an actionable report instead of producing unsafe RTL.

## Quick Start

```bash
python3 -m nl2hdl agent --model builtin:tiny_mlp --out build/tiny --skip-synth
```

With an explicit hardware/optimization/design spec:

```bash
python3 -m nl2hdl agent --model builtin:tiny_mlp --spec examples/tiny_mlp.yaml --out build/tiny --skip-synth
```

Optional LLM planning hook:

```bash
python3 -m nl2hdl agent --model builtin:tiny_mlp --planner auto --out build/tiny --skip-synth
```

`--planner auto` uses the local deterministic planner unless the `openai` Python
package and `OPENAI_API_KEY` are available. The emitted RTL path remains
deterministic and verified.

Outputs include:

- `model_top.sv`
- `dense_layer_*.sv`
- `tb_model_top.sv`
- `model_graph.mlir`
- `mlir_analysis.json`
- `graph_summary.json`
- `design_decision_report.json`
- `pruning_report.json`
- `quantization_report.json`
- `agent_report.json`
- `vivado_synth.tcl`

The CLI also accepts the planned spelling:

```bash
python3 -m nl2hdl generate --model builtin:tiny_mlp --out build/tiny --skip-synth
```

For a real Hugging Face model name, the agent attempts a best-effort ONNX export,
then rejects unsupported graph operations with a report.

## LLaMA/ZCU104 Planning

The current dense RTL backend is not a full LLaMA accelerator. For the updated
target, generate the framework plan first:

```bash
python3 -m nl2hdl plan \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --out build/llama_zcu104_plan
```

This emits `llm_accelerator_plan.json` and `llm_accelerator_plan.md`. Direct RTL
generation for `int4_gptq` + `llm_decoder_streaming` intentionally fails until
the LLM kernel backend is implemented.

LLM kernel work follows the HDL sub-agent workflow in
`docs/subagent_hdl_workflow.md`: the parent agent plans and verifies, while HDL
sub-agents write Verilog/SystemVerilog and failed reusable patterns are captured
as Skills.

Small kernel examples:

```bash
python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --mode kernel --kernel projection \
  --out build/projection_kernel
```

Inspect mode also emits HDL sub-agent assignment packets:

- `hdl_task_manifest.json`: semantic GEMM/non-GEMM work mapped to kernel,
  Layer FSM, Top FSM, and token-loop agent roles.
- `hdl_subagent_tasks.json`: machine-readable implementation packets derived
  from the manifest.
- `hdl_subagent_dispatch_plan.json`: dependency-aware waves for parallel
  projection/non-GEMM agents followed by decoder block, Layer FSM, Top FSM, and
  token-loop agents.
- `subagent_prompts/*.md`: prompt files the parent can send to HDL
  implementation agents. These prompts include the assigned contract, allowed
  write scope, required commands, common handshake expectations, timing
  evidence, and claims the sub-agent must not make.

When an exported/provided LLaMA MLIR graph is available, pass it directly
instead of using the synthetic inspect fixture:

```bash
python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --mlir-graph build/model_graph.mlir \
  --mode inspect \
  --out build/llama_inspect
```

`--mlir-graph` clears only the MLIR model-analysis blocker when the graph covers
all required LLaMA GEMM/non-GEMM semantic ops. GPTQ checkpoint, full execution,
and ZCU104 board-signoff gates remain independent.

For a richer multi-agent example using ZCU104, LLaMA-3.2-1B, QuaRot W4A4KV4,
GPTQ weights, and a weight-stationary systolic-array design, see
`examples/quarot_w4a4kv4_zcu104_llama32_1b/`. That folder includes the input
YAML, agent input prompts, generated inspect artifacts, and notes on how this
methodology differs from the earlier SIMD/GPTQ baseline.

## V1 Contract

- Input: Hugging Face model name or `builtin:tiny_mlp`.
- Model analysis: ONNX export followed by `onnx-mlir --EmitONNXIR`; the agent
  records MLIR ops, tensor shapes, and unsupported ops in `mlir_analysis.json`.
- Quantization: `int8_static`.
- Pruning: `none` or `magnitude_unstructured`; pruning is applied statically to
  weights before dense RTL generation.
- Hardware design: `layer_fsm` direct SystemVerilog with `start_i` / `done_o`.
- Verification: Verilator lint, integer RTL simulation, optional Vivado synthesis.
