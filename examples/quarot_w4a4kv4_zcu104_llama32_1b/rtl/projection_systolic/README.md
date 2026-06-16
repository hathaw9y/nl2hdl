# W4A4 Weight-Stationary Systolic Projection Tile

This packet implements a bounded SystemVerilog fixture for a QuaRot W4A4 projection tile. It is intentionally small and does not claim full LLaMA projection, DDR/AXI integration, checkpoint layout compatibility, or ZCU104 board signoff.

## Files

- `w4a4_ws_systolic_tile.sv`: contract-compliant RTL with `aclk`, `aresetn`, `start_i`, and `done_o`.
- `tb_w4a4_ws_systolic_tile.sv`: self-checking testbench for signed int4 decode, accumulation, done hold/release, and output stability.
- `golden_vectors.json`: deterministic vector and expected INT32 results.
- `kernel_report.json`: coverage, numeric policy, datapath, and evidence report.
- `subagent_result.json`: changed files, commands, verification evidence, and residual risks.

## Interface

Packed inputs use little element order: element `idx` is stored in bits `idx*WIDTH +: WIDTH`.

- `activation_tile_i`: signed INT4 activations in row-major `row*K + k` order.
- `weight_tile_i`: signed INT4 weights in `k*COLS + col` order.
- `acc_tile_o`: signed INT32 accumulators in row-major `row*COLS + col` order.
- `status_o`: compact debug word with run/done flags, current `k`, configured `K`, lane count, and state.

## Datapath

On `start_i` in idle, the module samples activations and weights into internal hold registers. The weight hold register is the stationary tile store and is not modified during compute or done hold. Each `STATE_RUN` cycle processes one `k` slice and performs `ROWS*COLS` true signed INT4 products in parallel, accumulating into INT32 output registers.

`done_o` remains asserted until `start_i` is deasserted. Outputs are registered and stable while `done_o` is high.

## Fixture Limits

This packet is a bounded projection-sized tile fixture. It does not include real GPTQ metadata preflight, dequantization scales/zero-points, requantization, DDR streaming, AXI boundaries, double-buffered BRAM, full systolic array shape selection, or board-level timing/resource signoff.
