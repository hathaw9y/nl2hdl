# Projection Streaming Kernel Contract

This contract defines the next target-like GEMM milestone after the fixture
Top FSM.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a projection streaming fixture that consumes packed GPTQ INT4 weight
words through an explicit stream-style interface and computes a small
projection tile.

This is not a full DDR controller, not AXI, and not full LLaMA projection
coverage. It is the first kernel gate that moves weight storage from static
compile-time constants toward streamed packed-weight data.

## Kernel Name

Use:

- CLI kernel: `projection_streaming`
- HDL module: `projection_streaming`
- report artifact: `projection_streaming_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md` for top control.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Streaming packed-weight input:

- `weight_word_i`: packed INT4 stream word, width from hardware memory width or
  a reported fixture width;
- `weight_valid_i`;
- `weight_ready_o`;
- optional `weight_last_i` or reportable fixed word count.

Data inputs:

- activation input tile, signed INT8 or explicitly reported format;
- groupwise zero-point metadata;
- groupwise scale metadata in explicitly reported fixed-point format.

Outputs:

- output accumulator tile in signed INT32 or explicitly reported accumulator
  format;
- optional stream consumption trace.

## Fixture Requirements

The first fixture must:

- use at least two output rows;
- use at least eight columns;
- use at least two groups per row;
- consume at least one packed stream word rather than hard-coded weights only;
- report requested memory data width and effective fixture stream width;
- prove packed nibble round-trip explicitly;
- compute expected outputs with a Python/NumPy golden reference.

If the configured memory data width is larger than the fixture payload, the
report must distinguish configured width from used fixture bits.

## Numeric Policy

Use the same formula as `projection_tile`:

```text
dequant_weight = (unpacked_int4 - zero_point) * scale
projection_out[row] = sum_col(dequant_weight[row, col] * activation[col])
```

Report:

- activation format;
- unpacked weight format;
- zero-point format;
- scale format;
- accumulator/output format;
- rounding mode;
- saturation policy.

## Evidence Requirements

Generated reports must include:

- `coverage_level: projection_streaming_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- stream interface fields and widths;
- stream word count and consumed word trace;
- tile parameters: rows, columns, group size, groups per row, PE lanes if used;
- numeric policy;
- explicit `round_trip_passed`;
- expected output vector from the Python/NumPy golden reference;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled;
- caveat that this is not DDR/AXI or board-level I/O signoff.

## Completion Gate

The milestone passes only when:

- common top handshake is present;
- stream valid/ready behavior is covered by simulation;
- input stream data is consumed rather than ignored;
- outputs are stable while `done_o` is high;
- Python/NumPy golden comparison passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full DDR controller, AXI, full LLaMA projection,
  full model execution, or board-level signoff.

## Integration Notes

Passing this gate can feed a later target-like projection tile and Top FSM memory
contract. It does not replace the need for a board shell, DDR controller, AXI
interface, or scaled projection timing/resource work.
