# RMSNorm/RoPE Source Path Contract

This contract defines the next non-GEMM milestone after the existing
`rmsnorm_target` and `rope_target` apply fixtures.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a bounded non-GEMM source-path fixture that replaces externally supplied
Python metadata ports with RTL-internal lookup or source paths for:

- RMSNorm `inv_rms`;
- RoPE `cos` and `sin`.

This milestone proves that the framework can feed normalization and rotary
metadata from hardware-visible source paths rather than treating them as opaque
testbench inputs.

This is not full LLaMA non-GEMM coverage. It does not require an RTL reciprocal
square-root datapath, RoPE frequency generation, full sequence-length tables,
softmax, KV-cache movement, residual scheduling, or a complete decoder block.

## Kernel Name

Use:

- CLI kernel: `rmsnorm_rope_source_path`
- HDL module: `rmsnorm_rope_source_path`
- report artifact: `rmsnorm_rope_source_path_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data may include:

- compact input vector for the RMSNorm fixture;
- compact input vector for the RoPE fixture;
- narrow source selectors such as `norm_token_i`, `rope_position_i`, or
  `table_select_i`;
- small registered output vectors;
- compact status/debug summary.

Top-level data must not include:

- `inv_rms_i` as a direct metadata input;
- `cos_i` or `sin_i` as direct metadata inputs;
- full target hidden-size vectors;
- full target sequence-length tables;
- wide deterministic metadata arrays.

The source lookup tables may be internal constants, inferred ROM-style
functions, or narrow validated streams. For this milestone, internal lookup
constants are acceptable if the report states that the source is a fixture.

## RMSNorm Source Requirements

The fixture must:

- cover at least four vector elements;
- include positive and negative input values;
- use non-uniform gamma values;
- fetch `inv_rms` from an RTL-internal source path using a selector;
- expose or record a lookup trace with selector, valid, and fetched value;
- compute the apply formula using the fetched value:

```text
y_i = (x_i * gamma_i * inv_rms_lookup_value) >>> output_shift
```

Report:

- `inv_rms_source: rtl_lookup_fixture` or a similarly explicit source;
- `reciprocal_sqrt_in_rtl: false` unless a true reciprocal sqrt datapath is
  implemented and verified;
- `sumsq_expected`;
- whether the sum-of-squares path is computed, checked, or omitted;
- fixed-point formats, rounding, and saturation policy.

## RoPE Source Requirements

The fixture must:

- cover at least four vector elements, meaning at least two rotation pairs;
- include positive and negative input values;
- fetch `cos` and `sin` pair values from an RTL-internal source path using a
  position selector;
- include at least two distinct cos/sin pairs;
- include non-zero sine values;
- expose or record a lookup trace with position, pair index, valid, cos, and
  sin values;
- compute the standard pair formula:

```text
out_even = even * cos - odd * sin
out_odd  = even * sin + odd * cos
```

Report:

- `cos_sin_source: rtl_lookup_fixture` or a similarly explicit source;
- `lookup_table_in_rtl: true`;
- `frequency_generation_in_rtl: false` unless real frequency generation is
  implemented and verified;
- position value, pair count, fixed-point formats, rounding, and saturation
  policy.

## Behavioral Requirements

The bounded RTL fixture must:

- start both source paths from `start_i`;
- latch source values before applying RMSNorm/RoPE math;
- record a deterministic source trace for both RMSNorm and RoPE;
- compare RMSNorm output against a Python/NumPy golden vector;
- compare RoPE output against a Python/NumPy golden vector;
- keep `done_o` asserted until `start_i` deasserts;
- keep output and compact debug/status stable while `done_o` is high.

The fixture may run RMSNorm then RoPE sequentially, or run both as independent
small paths under one top-level control FSM. The report must state the schedule.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: rmsnorm_rope_source_path`;
- `coverage_level: rmsnorm_rope_source_path_fixture`;
- `rmsnorm_source_path.coverage`;
- `rope_source_path.coverage`;
- top-level interface summary proving no direct `inv_rms_i`, `cos_i`, or
  `sin_i` metadata input ports;
- RMSNorm lookup trace;
- RoPE lookup trace;
- RMSNorm expected output vector and observed output vector;
- RoPE expected output vector and observed output vector;
- fixed-point numeric policy for both source paths;
- `reciprocal_sqrt_in_rtl`;
- `frequency_generation_in_rtl`;
- `lookup_table_in_rtl`;
- implementation stage, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- `does_not_claim` entries for full RMSNorm reciprocal sqrt, RoPE frequency
  generation, full sequence-length table coverage, softmax, KV-cache movement,
  full decoder block, full model execution, and board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- No top-level direct `inv_rms_i`, `cos_i`, or `sin_i` metadata ports exist.
- RMSNorm and RoPE source lookup traces are dynamically observed and checked.
- RMSNorm and RoPE outputs match Python/NumPy golden vectors.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
