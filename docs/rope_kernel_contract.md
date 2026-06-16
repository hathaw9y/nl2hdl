# RoPE Kernel Contract

This contract defines the next non-GEMM HDL sub-agent task after
`rmsnorm_target`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a RoPE target-like fixture kernel that applies rotary position embedding
pair rotations to a small query/key vector using explicit cos/sin metadata.

This is not full LLaMA RoPE coverage yet. It is a composable fixture that must
not claim full sequence-length or lookup-table coverage unless those paths are
actually generated and verified.

## Kernel Name

Use:

- CLI kernel: `rope_target`
- HDL module: `rope_target`
- report artifact: `rope_target_golden.json` plus `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum data contract:

- input vector in signed fixed-point format;
- cos vector in signed fixed-point format;
- sin vector in signed fixed-point format;
- position metadata or report field;
- output vector in signed fixed-point format;
- parameters or report fields for vector size, pair count, fractional bits, and
  rotation schedule.

## Numeric Policy

Required pair formula for each pair `(even, odd)`:

```text
out_even = even * cos - odd * sin
out_odd  = even * sin + odd * cos
```

The fixture must report:

- input format;
- cos/sin format;
- output format;
- position value;
- rounding mode;
- saturation policy;
- whether cos/sin are supplied by Python metadata, generated lookup, or RTL
  lookup.

## Behavioral Requirements

The fixture must cover:

- at least four vector elements, meaning at least two rotation pairs;
- positive and negative inputs;
- at least two distinct cos/sin pairs;
- non-zero sine values;
- positive and negative outputs;
- output stability while `done_o` is high.

If cos/sin are supplied by Python golden metadata, the report must state that
the fixture does not implement RoPE frequency generation or table lookup in RTL.

## Evidence Requirements

Generated reports must include:

- `coverage_level: rope_target_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- numeric policy fields listed above;
- `position`;
- `cos_sin_source`, such as `python_golden_metadata`, `lookup`, or
  `rtl_generated`;
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
- report text does not claim full LLaMA RoPE if cos/sin are supplied rather than
  generated or looked up in RTL.
