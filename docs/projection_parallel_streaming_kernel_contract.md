# Projection Parallel Streaming Kernel Contract

This contract defines the next projection milestone after
`projection_streaming`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a projection streaming fixture that consumes packed GPTQ INT4 weight
words and uses more than one true parallel datapath lane during the MAC phase.

This is not full LLaMA projection coverage, not DDR/AXI, and not board-level
signoff. It is the first gate that must prove parallel lane arithmetic rather
than only scheduling metadata.

## Kernel Name

Use:

- CLI kernel: `projection_parallel_streaming`
- HDL module: `projection_parallel_streaming`
- report artifact: `projection_parallel_streaming_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md` for top control.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Streaming packed-weight input:

- `weight_word_i`;
- `weight_valid_i`;
- `weight_ready_o`;
- `weight_last_i` or a fixed/reportable word count.

Data inputs:

- activation input tile;
- groupwise zero-point metadata;
- groupwise scale metadata.

Outputs:

- output accumulator tile;
- stream and lane trace fields in the report.

## Fixture Requirements

The first fixture must:

- use at least two output rows;
- use at least eight columns;
- use at least two groups per row;
- consume packed stream data through valid/ready;
- include at least one valid-low gap or ready-low backpressure event in the
  testbench;
- use `true_parallel_datapath_lanes >= 2`;
- prove that at least two lane products are formed in the same cycle or same
  pipeline stage and contribute to the same output tile;
- report requested PE lanes, effective fixture PE lanes, and true parallel
  datapath lanes separately;
- prove packed nibble round-trip explicitly;
- compute expected outputs with a Python/NumPy golden reference.

## Numeric Policy

Use the same formula as previous projection fixtures:

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
- lane product format;
- rounding mode;
- saturation policy.

## Evidence Requirements

Generated reports must include:

- `coverage_level: projection_parallel_streaming_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- stream interface fields and widths;
- stream word count and consumed word trace;
- backpressure or valid-gap trace;
- tile parameters;
- numeric policy;
- lane policy with requested/effective/true parallel lanes;
- explicit `round_trip_passed`;
- expected output vector from the Python/NumPy golden reference;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled;
- caveat that this is not DDR/AXI, target LLaMA projection, or board-level
  signoff.

## Completion Gate

The milestone passes only when:

- common top handshake is present;
- stream valid/ready behavior is covered by simulation;
- input stream data is consumed rather than ignored;
- at least two arithmetic lanes are truly active in the same cycle or pipeline
  stage;
- backpressure or a valid-low stream gap is tested;
- outputs are stable while `done_o` is high;
- Python/NumPy golden comparison passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full DDR controller, AXI, full LLaMA projection,
  full model execution, or board-level signoff.

## Integration Notes

Passing this gate can feed later target-scale projection planning. It does not
replace board-shell memory integration, true LLaMA hidden-size tiling, or
full-model scheduling.
