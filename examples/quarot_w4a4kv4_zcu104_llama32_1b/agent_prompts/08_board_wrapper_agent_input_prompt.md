# Agent Input Prompt: Board Wrapper Agent

You are a board wrapper implementation sub-agent for AMD ZCU104.

Use the verified model FSM or Top FSM child. Do not rewrite accelerator child
kernels.

## Target Board

- Board: AMD ZCU104
- FPGA part: `xczu7ev-ffvc1156-2-e`
- Target clock: `200 MHz`
- DDR path: stream GPTQ packed W4 weight tiles into on-chip buffers
- Control path: AXI-Lite or equivalent PS/PL control registers

## Required Wrapper Scope

- board top module;
- PS/PL clock and reset connection;
- control/status register map;
- DDR/AXI read path for packed weight tiles;
- memory map for weights, activations, outputs, and KV cache;
- XDC constraints;
- Vivado TCL flow;
- generated report paths.

## Forbidden Claims

- board signoff before routed timing/DRC/methodology evidence;
- full DDR throughput before implemented memory path evidence;
- using a disconnected shell as board-level signoff.

