# QuaRot W4A4KV4 ZCU104 LLaMA-3.2-1B Example

This example records a parent-agent input and sub-agent prompt flow for a
ZCU104 target running a LLaMA-3.2-1B-style decoder accelerator with QuaRot
W4A4KV4 optimization.

The target assumes:

- board: AMD ZCU104;
- FPGA part: `xczu7ev-ffvc1156-2-e`;
- model: `meta-llama/Llama-3.2-1B`;
- optimization: QuaRot W4A4KV4;
- weights: GPTQ INT4, assumed already implemented;
- compute architecture: systolic array;
- dataflow: weight stationary;
- memory: stream packed GPTQ weights from DDR into on-chip tile buffers;
- systolic sizing: selected from the DSP budget after module OOC synthesis.

## Files

- `input.yaml`: structured example input for the parent agent.
- `input/assumed_gptq_checkpoint/`: sparse GPTQ metadata fixture used only to
  exercise planning and payload-prefix probes.
- `agent_prompts/*.md`: hand-authored input prompts for each major agent stage.
- `module_packets/*.packet.json`: parent-owned module packet contracts for the
  generated RTL modules.
- `rtl/projection_systolic/`: generated bounded W4A4 weight-stationary
  systolic projection tile SystemVerilog packet.
- `rtl/quarot_support/`: generated bounded QuaRot W4A4KV4 support
  SystemVerilog packet.
- `run/inspect/`: generated artifacts from the example inspect run.
- `execution_summary.md`: result of the executed example.
- `methodology_differences.md`: differences from the earlier baseline
  methodology.

## Run

From the repository root:

```bash
python3 -m nl2hdl agent \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/quarot_w4a4kv4_zcu104_llama32_1b/input.yaml \
  --mode inspect \
  --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/inspect \
  --verbose
```

The run should emit:

- `llm_agent_report.json`
- `llm_accelerator_plan.json`
- `input_clarification_questions.json`
- `hdl_task_manifest.json`
- `hdl_subagent_tasks.json`
- `hdl_subagent_dispatch_plan.json`
- `subagent_prompts/*.md`
- `verification_prompts/*.md`
- `target_readiness_report.json`

## Current Executed Result

The checked-in run completed parent inspect/planning:

- status: `passed`
- clarification status: `clear`
- projection tasks: `7`
- non-GEMM tasks: `8`
- integration tasks: `42`
- generated implementation prompts: `57`
- generated verification prompts: `20`
- GPTQ metadata/layout preflight: `passed` against the sparse fixture
- projection weight stream plan: target-scale stream plan shape is ready for
  all projection names
- target readiness: `target_blocked`

Remaining blockers:

- `real_mlir_model_analysis`
- `full_llama_model_execution`
- `board_level_zcu104_signoff`

This is expected. The example proves the parent planning and prompt-generation
flow, not real target LLaMA execution.

## Generated RTL

HDL sub-agents generated two bounded SystemVerilog module packets:

- `rtl/projection_systolic/w4a4_ws_systolic_tile.sv`
  - bounded W4A4 projection tile;
  - weight-stationary tile behavior;
  - signed INT4 activations and GPTQ-style signed INT4 weights;
  - INT32 accumulation;
  - common `aclk` / `aresetn` / `start_i` / `done_o` contract.
- `rtl/quarot_support/quarot_w4a4kv4_support.sv`
  - bounded H4-style QuaRot support fixture;
  - signed INT8 input rotation;
  - saturating A4 quantization;
  - KV4 packed round-trip fixture;
  - common `aclk` / `aresetn` / `start_i` / `done_o` contract.

Each packet includes a self-checking testbench, golden vectors, a
`kernel_report.json`, a `subagent_result.json`, and a local README.

Parent-owned module packet contracts are in `module_packets/`:

- `module_packets/w4a4_ws_systolic_tile.packet.json`
- `module_packets/quarot_w4a4kv4_support.packet.json`
- `module_packets/module_packet_manifest.json`

Parent-side verification reran:

```bash
iverilog -g2012 ...
vvp ...
verilator --lint-only ...
xvlog -sv ...
python3 -m json.tool ...
```

Both module packets passed Icarus simulation, DUT Verilator lint, Vivado
frontend parsing, and JSON validation. Vivado OOC timing/resource signoff is
not claimed for these bounded fixtures.

## Important Fixture Boundary

`input/assumed_gptq_checkpoint/model.safetensors` is a sparse fixture, not real
weights. It contains target-shaped safetensors metadata and sparse zero-filled
payload regions so the GPTQ path can be exercised locally without storing a
large checkpoint.

Do not use this example to claim:

- real LLaMA logits;
- numeric QuaRot/GPTQ correctness;
- full checkpoint tensor materialization;
- throughput;
- board-level ZCU104 signoff.
