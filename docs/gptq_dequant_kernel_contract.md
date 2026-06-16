# GPTQ Dequant Kernel Contract

This contract defines the next HDL sub-agent task after `int4_unpack`.
The parent agent defines this contract; HDL sub-agents implement RTL.

## Scope

The kernel converts packed GPTQ INT4 weights plus groupwise metadata into
fixed-point dequantized weights that can feed a projection tile.

This is still a milestone kernel, not full LLaMA inference.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum data contract for the first fixture:

- packed INT4 input vector, little nibble order;
- per-group scale input in signed Q format;
- per-group zero-point input;
- dequantized output vector in signed fixed-point format;
- group size parameter.

## Numeric Policy

The first fixture should use deterministic fixed-point arithmetic:

- unpacked value: signed INT4 in range `[-8, 7]`;
- zero-point: signed INT4 or signed INT8 metadata, explicitly reported;
- scale: signed fixed-point Q format, explicitly reported;
- dequant formula: `(unpacked - zero_point) * scale`;
- output: signed fixed-point with enough width to avoid fixture overflow.

The report must state all widths, scale fractional bits, rounding mode, and
saturation policy. If saturation is not implemented in the fixture, the report
must say that the fixture vectors are chosen to avoid overflow.

## Golden Reference

The HDL sub-agent must provide a Python/NumPy golden vector covering:

- positive and negative nibbles;
- at least two groups;
- non-zero zero-points;
- positive and negative outputs;
- a packed nibble round-trip check before dequant.

## Required Evidence

The generated `kernel_report.json` must include:

- `coverage_level: synthetic_fixture` until a target tile is implemented;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- explicit packed nibble round-trip evidence, such as `round_trip_passed` and
  the unpacked values checked before dequant;
- `numeric_policy` with scale format, zero-point signedness, output format,
  rounding mode, and saturation policy;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled.

## Completion Gate

The kernel can feed projection-tile integration only after:

- common handshake is present;
- output is stable while `done_o` is high;
- Python golden comparison passes;
- packed nibble round-trip evidence is explicit, not only implied by dequant
  output checks;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` if synthesis is enabled;
- the report explicitly says it is still synthetic fixture coverage.
