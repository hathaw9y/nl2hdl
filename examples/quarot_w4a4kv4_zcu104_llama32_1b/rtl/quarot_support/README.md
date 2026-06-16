# QuaRot W4A4KV4 Support Fixture

This directory contains a bounded SystemVerilog fixture for the
`quarot_w4a4kv4_zcu104_llama32_1b` example. It is intentionally small and does
not claim to implement full QuaRot, full LLaMA, checkpoint layout handling,
DDR/AXI access, or board integration.

## Module

`quarot_w4a4kv4_support.sv` exposes the common HDL module contract:

```systemverilog
input  logic aclk;
input  logic aresetn;
input  logic start_i;
output logic done_o;
```

On `start_i` while idle, it latches:

- one signed int8 vector with `N=4` default;
- two KV4 nibble vectors.

It then registers:

- an unnormalized H4 Hadamard-like rotation;
- signed A4 saturating quantization clamped to `[-8, 7]`;
- A4 packed nibbles in little-element order;
- two KV4 unpack/repack round-trip outputs;
- compact status bits.

Packed vector order is little-element order:
`element[idx] == vector[idx*WIDTH +: WIDTH]`.

`done_o` remains high until `start_i` is deasserted. Outputs are registered and
remain stable while `done_o` is high.

## Checks

The self-checking testbench is `tb_quarot_w4a4kv4_support.sv`. It covers:

- deterministic H4 rotation outputs;
- signed A4 saturation and packing;
- KV4 round-trip packing;
- `done_o` hold and release;
- output stability while `done_o` is high;
- reuse after `done_o` release.

Expected vectors are recorded in `golden_vectors.json`.

Typical local run:

```bash
iverilog -g2012 -Wall -o /tmp/tb_quarot_w4a4kv4_support.vvp \
  examples/quarot_w4a4kv4_zcu104_llama32_1b/rtl/quarot_support/quarot_w4a4kv4_support.sv \
  examples/quarot_w4a4kv4_zcu104_llama32_1b/rtl/quarot_support/tb_quarot_w4a4kv4_support.sv
vvp /tmp/tb_quarot_w4a4kv4_support.vvp
```
