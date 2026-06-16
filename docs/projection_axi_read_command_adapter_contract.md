# Projection AXI Read Command Adapter Contract

This contract defines the next projection memory milestone after
`projection_internal_stream_shell`.

The parent agent owns this contract and the dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a bounded adapter that converts a checkpoint-aware qweight stream plan
into an AXI-style read-address command fixture for packed INT4 projection
weights.

This milestone proves that the framework can:

- consume the `target_weight_stream_plan` emitted in `hdl_task_manifest.json`;
- derive an aligned read byte address and beat count for a qweight tensor range;
- emit a bounded AXI read-address command with valid/ready backpressure;
- preserve command fields while `arvalid` is high and `arready` is low;
- report burst length, byte alignment, first-beat offset, and last valid bytes;
- keep the command fixture separate from AXI read-data execution, DDR
  controller integration, a complete board shell, full target projection
  execution, full model execution, and board-level signoff.

This milestone is an AXI read-command adapter fixture only. It is not a DDR
controller, not a PS/PL board shell, and not a full qweight payload streaming
implementation.

## Kernel Name

Use:

- CLI kernel: `projection_axi_read_command_adapter`
- HDL module: `projection_axi_read_command_adapter`
- report artifact: `projection_axi_read_command_adapter_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

AXI read-address command boundary:

- `output logic axi_arvalid_o`
- `input logic axi_arready_i`
- `output logic [ADDR_WIDTH-1:0] axi_araddr_o`
- `output logic [7:0] axi_arlen_o`
- `output logic [2:0] axi_arsize_o`
- `output logic [1:0] axi_arburst_o`
- `output logic [7:0] axi_arid_o`

Fixture output and compact debug:

- expose only narrow command/status evidence at the top level;
- keep qweight tensor payloads, full memory contents, and wide traces internal
  to the testbench or report;
- outputs and command fields must be stable while `done_o` is asserted.

## Planner Requirements

The adapter must use the checkpoint-aware qweight stream plan when available.
The sub-agent prompt provides the selected projection, qweight shard file,
tensor key, qweight byte offset, qweight byte count, aligned request byte
address, request beat count, first-beat offset, last valid bytes, and
`stream_plan_valid`.

The report must record:

- selected projection name, initially `q_proj`;
- qweight shard file and tensor key;
- qweight byte offset and byte count;
- aligned request byte address;
- memory data width and bytes per memory beat;
- request beat count;
- first-beat byte offset;
- last valid bytes;
- whether the request covers an unaligned safetensors tensor range;
- derived AXI `araddr`, `arlen`, `arsize`, `arburst`, and `arid`;
- whether the command must be split because the request exceeds the bounded
  fixture's maximum burst length.

For AXI4, `arlen` is beats minus one for a single burst. If the planned request
requires more than the configured fixture burst limit, the bounded fixture may
emit a small deterministic subset, but the report must clearly distinguish:

- `target_checkpoint_request_planning_only: true`;
- `fixture_axi_command_execution: true`;
- target planned request beats;
- fixture executed command beats.

## Fixture Requirements

The bounded RTL fixture must:

- issue at least one AXI read-address command after `start_i`;
- hold all command fields stable while `axi_arvalid_o` is high and
  `axi_arready_i` is low;
- exercise deterministic `arready` backpressure in the testbench;
- compute `axi_arsize_o` from the configured memory data width;
- use incrementing bursts with `axi_arburst_o = 2'b01`;
- keep command ID stable for the command;
- assert `done_o` after the command handshake or bounded command sequence
  completes;
- keep `done_o` asserted until `start_i` deasserts;
- keep outputs and compact status stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_axi_read_command_adapter`;
- `coverage_level: projection_axi_read_command_adapter_fixture`;
- selected projection metadata;
- `checkpoint_target_weight_stream_plan`;
- `checkpoint_target_request_summary`;
- `target_checkpoint_request_planning_only`;
- `fixture_axi_command_execution`;
- AXI command fields and observed command trace;
- command-ready backpressure trace;
- proof that command fields were stable under backpressure;
- configured memory width and derived AXI size;
- target planned beat count and fixture executed beat count;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- `does_not_claim` entries for DDR controller integration, AXI read-data
  channel execution, full qweight payload streaming, complete board shell, full
  target projection execution, full model execution, and board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- AXI command fields are dynamically observed and compared.
- Backpressure stability is dynamically checked.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only Codex verification agent finds no P0/P1 issues.
