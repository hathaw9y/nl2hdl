# Projection Adapter Stream Integration Contract

This contract defines the next fixture after `packed_stream_adapter_multiword`
and `projection_parallel_streaming`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a fixture that connects a multi-word packed memory stream adapter to a
projection streaming consumer through a valid/ready payload link.

This milestone proves that 128-bit packed memory beats can be split into
32-bit packed INT4 payload words and consumed by a projection-style kernel
without dropping, duplicating, or reordering payloads under backpressure.

This is not AXI, not a DDR controller, not target-scale LLaMA projection, and
not board-level signoff.

## Kernel Name

Use:

- CLI kernel: `projection_adapter_stream_integration`
- HDL module: `projection_adapter_stream_integration`
- report artifact: `projection_adapter_stream_integration_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Memory-side input stream:

- `input logic [MEM_WORD_WIDTH-1:0] mem_word_i`
- `input logic mem_valid_i`
- `output logic mem_ready_o`
- `input logic mem_last_i`

Projection result outputs:

- at least a small output tile vector or accumulator vector;
- trace/debug outputs may be exposed for fixture evidence.

Internal link requirements:

- The adapter-to-projection payload link must use 32-bit payload words.
- The link must use `valid` and `ready`.
- The projection side must be able to deassert ready in the fixture so
  backpressure crosses the adapter/consumer boundary.

## Fixture Requirements

The fixture must:

- accept at least two configured 128-bit memory words;
- derive exactly eight 32-bit payload chunks from those words;
- preserve payload order: input beat order, then little chunk order within each
  beat;
- consume all eight payload chunks through the projection-side valid/ready
  interface;
- dynamically observe and compare all eight emitted and consumed payload words
  in the testbench or simulation harness, rather than reporting both lists from
  one generated constant source;
- include ready-low backpressure from the projection side at least once in the
  first input beat and at or across a beat boundary;
- compute a small projection-style output from the consumed payloads using the
  same INT4/GPTQ fixture policy as `projection_parallel_streaming`;
- include at least two true parallel arithmetic lanes or instantiate/reuse a
  fixture that reports true same-stage lane arithmetic;
- prove packed nibble round-trip explicitly;
- keep `done_o` asserted until `start_i` deasserts;
- keep output and trace/debug ports stable while `done_o` is high.

## Numeric Policy

Use a small deterministic projection fixture:

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
- lane product format when parallel lanes are used;
- rounding mode;
- saturation policy.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_adapter_stream_integration`;
- `coverage_level: projection_adapter_stream_integration_fixture`;
- configured memory width, payload width, and payload count;
- consumed memory words in hex;
- adapter-emitted payload words in hex;
- projection-consumed payload words in hex;
- proof that emitted and consumed payloads match exactly, with evidence sourced
  from observed link transactions;
- input handshake trace;
- adapter-to-projection backpressure trace;
- projection output vector and Python/NumPy golden output vector;
- lane policy with requested, effective fixture, and true parallel datapath
  lanes;
- `round_trip_passed: true`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- explicit caveats that the fixture is not AXI, not DDR controller
  integration, not target-scale LLaMA projection, not full model execution, and
  not board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
