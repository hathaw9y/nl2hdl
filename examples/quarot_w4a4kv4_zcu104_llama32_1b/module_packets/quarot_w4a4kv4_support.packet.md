# Module Packet: quarot_w4a4kv4_support

Parent-owned packet for the bounded QuaRot W4A4KV4 support fixture.

## Scope

- Family: non-GEMM rotation and quantization support.
- Coverage: bounded H4-style fixture, not full QuaRot target implementation.
- RTL: `../rtl/quarot_support/quarot_w4a4kv4_support.sv`.
- Testbench: `../rtl/quarot_support/tb_quarot_w4a4kv4_support.sv`.

## Interface

- `aclk`
- `aresetn`
- `start_i`
- `done_o`
- signed INT8 input vector
- signed rotated output vector
- signed A4 packed output
- KV4 packed round-trip fixture inputs/outputs
- compact `status_o`

Packed vectors use little-element order.

## Verification

Observed parent verification:

- Icarus simulation: passed.
- DUT Verilator lint: passed.
- Vivado `xvlog -sv`: passed.
- JSON validation: passed.
- Vivado OOC timing/resource: not run.

## Integration Gate

This packet needs pre-integration boundary review before integration consumes
it. It cannot claim full QuaRot numeric correctness, activation calibration,
full KV-cache behavior, DDR/AXI integration, or ZCU104 board signoff.

