# Module Packet: w4a4_ws_systolic_tile

Parent-owned packet for the bounded W4A4 weight-stationary systolic projection
tile.

## Scope

- Family: projection GEMM/GEMV tile.
- Coverage: bounded fixture, not full LLaMA projection.
- RTL: `../rtl/projection_systolic/w4a4_ws_systolic_tile.sv`.
- Testbench: `../rtl/projection_systolic/tb_w4a4_ws_systolic_tile.sv`.

## Interface

- `aclk`
- `aresetn`
- `start_i`
- `done_o`
- packed A4 activation tile input
- packed W4 GPTQ-style weight tile input
- packed INT32 accumulator output
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

This packet needs pre-integration boundary review before any parent FSM or
decoder-block integration consumes it. It cannot claim DDR/AXI streaming,
target-scale GPTQ correctness, final systolic shape selection, or ZCU104 board
signoff.

