# Projection Tile Kernel Contract

This contract defines the next HDL sub-agent task after the synthetic
`int4_unpack`, `gptq_dequant`, and `projection` gates.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a projection tile fixture that exercises the path from GPTQ INT4 packed
weights through groupwise dequant into a tiled MAC/projection datapath.

This is not full LLaMA-3.2-1B coverage yet. It is a target-like tile milestone
that must keep synthetic and target claims separate.

## Kernel Name

Use:

- CLI kernel: `projection_tile`
- HDL module: `projection_tile`
- report artifact: `projection_tile_report.json` or `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum data contract:

- activation input tile, signed INT8 or explicitly reported fixed-point format;
- packed INT4 weight tile in little nibble order;
- groupwise zero-point metadata;
- groupwise scale metadata in an explicitly reported fixed-point format;
- output accumulator tile in signed INT32 or explicitly reported fixed-point
  accumulator format;
- parameters or report fields for `TILE_ROWS`, `TILE_COLS`, `GROUP_SIZE`, and
  `PE_LANES`.

## Behavioral Requirements

The fixture must exercise more than a single hard-coded dot product:

- at least two output rows;
- at least two groups;
- `GROUP_SIZE` must divide `TILE_COLS`;
- `PE_LANES` must affect scheduling, latency, or datapath grouping;
- expected output must be computed from packed weights, zero-points, scales,
  and activations using a Python/NumPy golden reference.

Dequant formula:

```text
dequant_weight = (unpacked_int4 - zero_point) * scale
projection_out[row] = sum_col(dequant_weight[row, col] * activation[col])
```

The report must state scale Q format, rounding mode, saturation policy, and
accumulator format.

## Evidence Requirements

Generated reports must include:

- `coverage_level: projection_tile_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- numeric policy, including scale format, zero-point signedness, output format,
  rounding mode, and saturation policy;
- tile parameters: rows, columns, group size, PE lanes, memory width if used;
- explicit packed nibble `round_trip_passed` evidence;
- expected output vector from the Python/NumPy golden reference;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled.

## Completion Gate

The milestone passes only when:

- common handshake is present;
- outputs are stable while `done_o` is high;
- Python/NumPy golden comparison passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full LLaMA projection coverage.

## Integration Notes

Do not spawn Layer FSM or Top FSM agents from this gate alone. The next
integration step still needs target RMSNorm and RoPE interfaces, plus a decoder
block contract that composes only kernels that have passed their gates.
