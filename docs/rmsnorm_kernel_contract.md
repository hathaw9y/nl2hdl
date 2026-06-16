# RMSNorm Kernel Contract

This contract defines the next non-GEMM HDL sub-agent task.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create an RMSNorm fixture kernel that is composable by a decoder-block Layer FSM
and verifies fixed-point normalization math against a Python/NumPy reference.

This is not full LLaMA RMSNorm coverage yet. It is a target-like fixture that
must not claim a complete reciprocal-square-root implementation unless that
datapath is actually generated and verified.

## Kernel Name

Use:

- CLI kernel: `rmsnorm_target`
- HDL module: `rmsnorm_target`
- report artifact: `rmsnorm_target_golden.json` plus `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum data contract:

- input vector in signed fixed-point format;
- gamma/weight vector in signed fixed-point format;
- inv-rms input or lookup result in signed/unsigned fixed-point format;
- output vector in signed fixed-point format;
- parameters or report fields for `HIDDEN_SIZE`, vector width, fractional bits,
  and reduction schedule.

## Numeric Policy

The first fixture may accept `inv_rms_i` as metadata computed by the Python
golden reference. If so, the report must state that reciprocal square root is
external or lookup-provided for this milestone.

Required formula:

```text
sumsq = sum_i(x_i * x_i)
mean_square = sumsq / hidden_size
inv_rms = 1 / sqrt(mean_square + epsilon)
y_i = x_i * inv_rms * gamma_i
```

For the fixture, fixed-point math must be deterministic and reported:

- input format;
- gamma format;
- inv-rms format;
- output format;
- epsilon;
- rounding mode;
- saturation policy;
- whether inv-rms is computed in RTL or supplied as metadata.

## Behavioral Requirements

The fixture must cover:

- at least four vector elements;
- positive and negative inputs;
- non-uniform gamma values;
- non-trivial inv-rms value;
- positive and negative outputs;
- output stability while `done_o` is high.

If RTL does not compute reciprocal square root, it must still compute or verify
the sum-of-squares path, or explicitly report that this milestone is
`rmsnorm_apply_fixture` rather than full RMSNorm.

## Evidence Requirements

Generated reports must include:

- `coverage_level: rmsnorm_target_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- numeric policy fields listed above;
- `sumsq_expected`;
- `inv_rms_source`, such as `python_golden_metadata`, `lookup`, or
  `rtl_computed`;
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
- report text does not claim full LLaMA RMSNorm if inv-rms is supplied rather
  than computed in RTL.
