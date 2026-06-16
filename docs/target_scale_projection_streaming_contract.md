# Target-Scale Projection Streaming Contract

This contract defines the next projection milestone after
`projection_adapter_stream_integration`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a target-scale projection streaming planner plus a bounded RTL fixture
that maps LLaMA-3.2-1B projection dimensions onto ZCU104-oriented streaming
tiles.

This milestone does not require a full target-size projection datapath. It must
prove that the framework can:

- read model-scale projection dimensions from a model/config source or fixture
  metadata;
- derive target projection shapes for q/k/v/o/gate/up/down projections;
- choose tile and stream parameters from the hardware spec;
- generate a small bounded RTL fixture using those selected parameters;
- report the distinction between target-scale planning and fixture-scale RTL
  verification.

This is not AXI, not a DDR controller, not full LLaMA projection execution, not
full decoder execution, and not board-level signoff.

## Kernel Name

Use:

- CLI kernel: `projection_target_stream_plan`
- HDL module: `projection_target_stream_plan`
- report artifact: `projection_target_stream_plan_golden.json` plus
  `kernel_report.json`

## Target Model Metadata

For the LLaMA-3.2-1B planning fixture, record these model parameters in the
report:

- hidden size: 2048
- intermediate size: 8192
- attention heads: 32
- key/value heads: 8
- head dimension: 64
- decoder layers: 16
- sequence length: from the user config, currently 2048 in
  `examples/zcu104_llama32_1b_gptq.yaml`

Derived projection shapes:

- `q_proj`: rows 2048, cols 2048
- `k_proj`: rows 512, cols 2048
- `v_proj`: rows 512, cols 2048
- `o_proj`: rows 2048, cols 2048
- `gate_proj`: rows 8192, cols 2048
- `up_proj`: rows 8192, cols 2048
- `down_proj`: rows 2048, cols 8192

The report may use fixture metadata instead of downloading a gated model, but
it must state the metadata source.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Streaming packed-weight input:

- `input logic [MEM_WORD_WIDTH-1:0] mem_word_i`
- `input logic mem_valid_i`
- `output logic mem_ready_o`
- `input logic mem_last_i`

Optional tile descriptor input ports are allowed only if they are narrow. Do
not expose full target-size activation, scale, zero-point, or weight metadata as
top-level ports.

Fixture outputs:

- at least one small accumulator/output vector;
- compact debug traces proving tile index, payload order, backpressure, and
  lane activity.

## Planner Requirements

The planner must emit `projection_target_stream_plan_golden.json` with:

- target board part and target clock;
- memory data width;
- requested PE lanes from config;
- selected effective fixture lanes;
- selected target planning lanes;
- selected output tile rows and input tile columns;
- selected payload width;
- memory beats per target tile;
- approximate packed INT4 bytes per projection;
- approximate packed INT4 bytes per decoder layer for the listed projections;
- external-memory streaming caveat;
- resource-budget estimate or explicit note that the estimate is a planner
  estimate, not a Vivado measurement.

For INT4 packed weights:

```text
packed_weight_bytes = rows * cols * 4 / 8
mem_beats = ceil(packed_weight_bytes / (memory_data_width / 8))
```

If groupwise GPTQ scales/zero-points are modeled, report the group size and
metadata byte estimate separately from packed weights.

## Fixture Requirements

The bounded RTL fixture must:

- consume at least four configured memory words when `MEM_WORD_WIDTH=128`;
- derive at least sixteen 32-bit payload chunks from those words;
- preserve input beat order and little chunk order within each beat;
- dynamically observe and compare every emitted and consumed payload word in
  the testbench or simulation harness;
- include ready-low backpressure at least once within a beat and at or across a
  beat boundary;
- use the selected fixture tile parameters from the planner report;
- include at least two true same-stage arithmetic lanes;
- compute a small projection-style output from consumed payloads with a
  Python/NumPy golden reference;
- keep `done_o` asserted until `start_i` deasserts;
- keep outputs and trace/debug ports stable while `done_o` is high.

## Numeric Policy

Use the same small deterministic GPTQ policy as the prior projection fixtures:

```text
payload_word[k] = accepted_mem_word[word_idx][chunk_idx*32 +: 32]
unpacked_int4 = signed_nibbles(payload_word stream)
dequant_weight = (unpacked_int4 - zero_point) * scale
projection_out[row] = sum_col(dequant_weight[row, col] * activation[col])
```

Report:

- activation format;
- unpacked weight format;
- payload order;
- zero-point format;
- scale format;
- accumulator/output format;
- lane product format;
- rounding mode;
- saturation policy.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_target_stream_plan`;
- `coverage_level: projection_target_stream_plan_fixture`;
- model metadata and projection shapes;
- target planning tile parameters;
- fixture tile parameters;
- configured memory width, payload width, memory word count, and payload count;
- consumed memory words in hex;
- adapter-emitted payload words in hex from observed transactions;
- projection-consumed payload words in hex from observed transactions;
- proof that emitted and consumed payloads match exactly;
- input handshake trace;
- adapter-to-projection backpressure trace;
- projection output vector and Python/NumPy golden output vector;
- lane policy with requested, target-plan, effective fixture, and true parallel
  datapath lanes;
- packed INT4 byte and memory beat estimates for each target projection;
- `round_trip_passed: true`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- explicit caveats that the fixture is not AXI, not DDR controller
  integration, not full target projection execution, not full model execution,
  and not board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Every observed payload word is compared on the link.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
