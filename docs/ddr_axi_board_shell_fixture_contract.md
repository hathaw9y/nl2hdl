# DDR AXI Board Shell Fixture Contract

This packet is the first board-shell HDL milestone after
`model_fsm_axi_decoder_block_fixture`.

## Scope

Build a bounded ZCU104-oriented shell fixture that wraps the verified model FSM
path with an external-memory-facing AXI planning boundary.

The shell must:

- instantiate or call the verified `model_fsm_axi_decoder_block_fixture` child;
- expose a compact AXI master planning interface for packed GPTQ qweight reads;
- preserve `aclk`, `aresetn`, `start_i`, and `done_o`;
- keep child output and compact status stable while `done_o` is high;
- record qweight request metadata for all q/k/v/o and MLP projections when the
  parent manifest provides target stream plans;
- preserve the parent rule that bounded shell evidence is not board-level
  ZCU104 signoff.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Minimum top-level ports:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

AXI planning boundary:

- expose only compact request/status metadata in the fixture;
- do not expose full 128-bit data buses, debug arrays, or full KV-cache arrays at
  top level unless a later contract explicitly requires them;
- use little-endian packed 32-bit word order for sampled qweight payload words.

## Child Boundary

The shell agent owns shell RTL only. It must not rewrite:

- projection kernels;
- AXI projection stream integrations;
- decoder-block internals;
- Layer FSM, Top FSM, token-loop, or model FSM child internals.

## Required Evidence

The HDL sub-agent must produce:

- generated SystemVerilog and testbench for `ddr_axi_board_shell_fixture`;
- `kernel_report.json` and `subagent_result.json`;
- simulation evidence that the shell starts the model child and observes done;
- compact AXI request/status evidence for q/k/v/o plus MLP projection streams;
- evidence that output/status stability holds while shell `done_o` is high;
- Verilator lint evidence when enabled;
- Vivado setup, hold, pulse-width, and utilization evidence when synthesis is
  enabled.

## Does Not Claim

- real DDR controller IP integration;
- PS/PL block design integration;
- board-level timing with package I/O delays;
- full checkpoint tensor streaming;
- full target 16-layer LLaMA execution;
- token-dependent KV-cache semantics;
- logits or sampling;
- board-level ZCU104 signoff.
