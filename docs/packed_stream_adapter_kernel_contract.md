# Packed Stream Adapter Kernel Contract

This contract defines the next memory-path milestone after
`projection_parallel_streaming`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a fixture kernel that accepts a configured-width packed weight stream
word, such as the ZCU104 config `memory_data_width: 128`, and emits narrower
internal packed INT4 payload chunks for downstream projection fixtures.

This is not a DDR controller, not AXI, and not board-level I/O signoff. It is an
adapter fixture that proves the framework can use the configured memory width
instead of silently shrinking the port.

## Kernel Name

Use:

- CLI kernel: `packed_stream_adapter`
- HDL module: `packed_stream_adapter`
- report artifact: `packed_stream_adapter_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md` for top control.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Input stream:

- `mem_word_i`: width must be the configured memory width, expected `128` for
  `examples/zcu104_llama32_1b_gptq.yaml`;
- `mem_valid_i`;
- `mem_ready_o`;
- `mem_last_i` or a fixed/reportable word count.

Output stream:

- `payload_word_o`: narrower fixture payload, such as `32` bits;
- `payload_valid_o`;
- `payload_ready_i`;
- `payload_last_o`.

## Fixture Requirements

The first fixture must:

- use `configured_memory_data_width_bits: 128`;
- emit at least four `32`-bit payload chunks from one 128-bit input word;
- include valid/ready behavior on both input and output sides;
- include at least one output backpressure cycle in the testbench;
- prove payload chunk order explicitly;
- prove packed INT4 nibble round-trip from the emitted payload chunks;
- report that this is not AXI, DDR, or board-level signoff.

## Evidence Requirements

Generated reports must include:

- `coverage_level: packed_stream_adapter_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- input and output stream widths;
- input word count and output payload count;
- consumed input word trace;
- emitted payload trace;
- output backpressure trace;
- explicit `round_trip_passed`;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled;
- caveat that this is not DDR/AXI or board-level I/O signoff.

## Completion Gate

The milestone passes only when:

- common top handshake is present;
- input stream valid/ready behavior is covered by simulation;
- output stream valid/ready and backpressure behavior are covered by simulation;
- 128-bit input word data is consumed rather than ignored;
- emitted payload chunks match the Python/NumPy golden reference;
- packed INT4 round-trip passes from emitted payload chunks;
- outputs are stable while `done_o` is high;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full DDR controller, AXI, full LLaMA projection,
  full model execution, or board-level signoff.

## Integration Notes

Passing this gate can feed a later projection streaming integration that uses a
real 128-bit memory-width stream. It does not replace AXI/DDR shell work.
