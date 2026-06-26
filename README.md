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

LLM kernel work follows the Parent feedback loop in
`docs/subagent_hdl_workflow.md`: the Parent Agent is the only orchestrator, and
every non-parent worker is a Sub-agent. HDL, verification, integration,
board-wrapper, model-signoff, and board-signoff Sub-agents return evidence to
the Parent instead of spawning other agents. Failed reusable patterns are
captured as Skills before the Parent retries the responsible Sub-agent.

The executable parent loop is:

```bash
python3 -m nl2hdl parent-loop \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --out build/llama_parent_loop \
  --max-iterations 8
```

`parent-loop` creates inspect artifacts, refreshes `parent_loop_state.json`,
dispatches local deterministic sub-agent backends where possible, collects
`kernel_report.json` and `subagent_result.json`, and writes
`parent_loop_run_report.json`. Local target-evidence backends are available for
the model-level harness, full-execution evidence, ZCU104 board wrapper, and
board-signoff evidence. Codex-only read-only verification work is preserved in
`status/parent_loop_queue.json` unless `--local-verification` is explicitly used
for deterministic smoke coverage. With `--skip-synth`, real datapath modules
still require later module-level OOC synthesis before integration can advance.

Target preflight can use either provided/exported MLIR or resolved Hugging Face
model config. The conservative default is `model_structure_source: mlir`; use
`--model-structure-source hf_config` when the parent should accept resolved
`AutoConfig` semantic structure instead of requiring an MLIR graph. GPTQ
checkpoint preflight can also inspect a Hugging Face GPTQ repo without
downloading the full checkpoint by setting `NL2HDL_ALLOW_HF_REMOTE_PREFLIGHT=1`;
it range-reads safetensors headers and bounded qweight/qzeros/scales payload
prefixes only.

Example target-preflight run:

```bash
NL2HDL_ALLOW_HF_REMOTE_PREFLIGHT=1 python3 -m nl2hdl agent \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --model-structure-source hf_config \
  --gptq-checkpoint Crusadersk/llama3.2-1b-gptq-4bit \
  --mode inspect \
  --out build/llama_target_preflight
```

This clears the target-preflight gates only when model-structure coverage,
GPTQ INT4 metadata, GPTQ tensor layout, and projection payload probes all pass.
It still does not claim full LLaMA execution or board-level signoff.

The ZCU104 board-wrapper backend can be run directly when Vivado 2024.1 is
available:

```bash
python3 -m nl2hdl subagents board-wrapper \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --out build/zcu104_board_wrapper \
  --vivado-executable /tools/Xilinx/Vivado/2024.1/bin/vivado
```

It emits routed Vivado reports, `zcu104_post_route.dcp`, and
`zcu104_board_wrapper.bit`. The 200 MHz gate is checked from routed
`report_clocks` and implemented XDC, not from requested BD clock properties.
Board signoff remains a separate evidence-only step and requires passed full
execution readiness plus passed board-wrapper route/bitstream evidence.
The board-wrapper `.bit` is not enough by itself: board signoff also requires
the wrapper implementation report to prove
`target_scale_accelerator_bitstream: true` and
`accelerator_scope: full_target_llama_accelerator`. A routed control scaffold
or fixture bitstream is preserved as useful Vivado evidence, but the Parent
keeps `board_signoff_readiness` blocked and queues the ZCU104 board-wrapper
implementation sub-agent until a target-scale accelerator bitstream report is
available.

To attach an existing routed board-wrapper/bitstream bundle to a new Parent
run, pass the previous evidence directory:

```bash
python3 -m nl2hdl parent-loop \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --out build/parent_loop_with_bitstream \
  --board-wrapper-evidence-dir build/zcu104_board_wrapper
```

The Parent copies the wrapper reports, checkpoint, and `.bit` into
`evidence/board_zcu104_signoff_gate/` and records the bitstream evidence in
`parent_loop_run_report.json`. This does not clear full LLaMA execution or
board-level signoff by itself; imported reports that identify the bitstream as
a board-wrapper/control scaffold or fixture remain blocked for target-scale
board signoff. Importing a new board-wrapper bundle invalidates any older
`board_zcu104_signoff_evidence.json` by moving it under
`status/invalidated_evidence/`, because signoff evidence is only valid for the
current routed wrapper.

The board-wrapper sub-agent can also wrap a generated accelerator artifact
instead of the internal control scaffold:

```bash
python3 -m nl2hdl subagents board-wrapper \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --out build/zcu104_generated_artifact_board_wrapper \
  --accelerator-artifact-dir build/parent_loop/evidence/ddr_axi_board_shell_fixture_gate \
  --accelerator-top-module ddr_axi_board_shell_fixture \
  --accelerator-kernel-report build/parent_loop/evidence/ddr_axi_board_shell_fixture_gate/kernel_report.json
```

The report marks `target_scale_accelerator_bitstream: true` only when the
wrapped artifact's kernel report proves target-scale, non-fixture full-model
coverage. A routed fixture bitstream is useful route evidence, but it still
keeps board signoff blocked.

When a fixture bitstream already exists, the Parent no longer repeats the same
board-wrapper route. It first queues the ready `target_scale_child_rtl_wave`
packets: GPTQ projection datapaths, non-GEMM datapaths, and the DDR
packed-weight stream scheduler. Decoder-block integration waits for those
three packets, and the token-loop/16-layer model FSM waits for decoder-block
integration. Only after those child reports are target-scale eligible does the
Parent queue `full_model_target_rtl_generator`, which must create the
non-fixture accelerator artifact under
`evidence/full_target_llama_accelerator_gate/`.

When Vivado OOC synthesis is enabled, real datapath module reports also carry
resource ratios, true datapath-lane evidence, and a tuning recommendation. The
projection target-stream fixture now connects `pe_count: 64` to `64` true MAC
lanes and records post-route Vivado evidence such as DSP `128`, LUT `6085`,
FF `3199`, setup WNS `0.060 ns`, hold WHS `0.020 ns`, and pulse-width WPWS
`2.225 ns` in `module_ooc_synthesis_report.json`. If a module is still
resource-light but has exhausted the bounded fixture's true-lane headroom, the
Parent stops doubling `pe_count` and routes the next improvement to tile or
generator expansion.

Small kernel examples:

```bash
python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B \
  --spec examples/zcu104_llama32_1b_gptq.yaml \
  --mode kernel --kernel projection \
  --out build/projection_kernel
```

Inspect mode also emits HDL sub-agent assignment packets:

- `hdl_task_manifest.json`: semantic GEMM/non-GEMM work mapped to kernel,
  Layer FSM, Top FSM, and token-loop sub-agent roles.
- `hdl_subagent_tasks.json`: machine-readable implementation packets derived
  from the manifest.
- `hdl_subagent_dispatch_plan.json`: dependency-aware waves for parallel
  projection/non-GEMM Sub-agents followed by decoder block, Layer FSM, Top FSM,
  and token-loop Sub-agents.
- `hdl_subagent_execution_manifest.json`: next Sub-agents the Parent or an
  external runner should spawn.
- `parent_loop_state.json`: Parent-owned loop state and next parent action.
- `feedback_packet.json`: feedback/assignment packet sent from Parent to
  ready or failed Sub-agents.
- `retry_plan.json`: retry gates, blocked waves, and required Parent action.
- `subagent_prompts/*.md`: prompt files the parent can send to HDL
  implementation Sub-agents. These prompts include the assigned contract, allowed
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
