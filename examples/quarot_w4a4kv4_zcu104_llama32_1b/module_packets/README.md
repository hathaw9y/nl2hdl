# Module Packets

This directory contains parent-owned module packet contracts for the bounded
SystemVerilog modules generated under `../rtl/`.

These packets are the handoff artifacts between the Parent Agent and HDL
implementation, verification, integration, and signoff agents. They define what
the generated RTL is allowed to claim, what evidence exists, and what remains
blocked before target-scale integration.

Current packets:

- `w4a4_ws_systolic_tile.packet.json`
  - bounded W4A4 weight-stationary systolic projection tile.
- `quarot_w4a4kv4_support.packet.json`
  - bounded QuaRot W4A4KV4 support fixture.

The generated RTL is fixture-level. These packets do not authorize full LLaMA
execution, full GPTQ numeric correctness, DDR/AXI integration, or ZCU104
board-level signoff.

