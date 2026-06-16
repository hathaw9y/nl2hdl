# Projection AXI Read Data Channel Adapter Contract

This contract defines the next projection memory milestone after
`projection_axi_read_command_adapter`.

The parent agent owns this contract and the dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a bounded adapter that consumes AXI-style read-data channel beats for
packed INT4 projection weights and bridges those beats toward the existing
packed projection payload stream shape.

This milestone proves that the framework can:

- accept deterministic AXI `R` channel beats with valid/ready backpressure;
- check read response ID, response code, beat order, and `last`;
- split each 128-bit read-data beat into four 32-bit packed payload words in
  little chunk order;
- preserve payload order into a bounded payload stream;
- report target qweight request metadata separately from the bounded fixture
  read-data execution;
- keep the fixture separate from DDR controller integration, a complete AXI
  master, a complete board shell, full target projection execution, full model
  execution, and board-level signoff.

This milestone is an AXI read-data channel fixture only. It is not a DDR
controller and does not prove that target checkpoint qweight tensor bytes were
loaded from external memory.

## Kernel Name

Use:

- CLI kernel: `projection_axi_read_data_channel_adapter`
- HDL module: `projection_axi_read_data_channel_adapter`
- report artifact: `projection_axi_read_data_channel_adapter_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

AXI read-data channel boundary:

- `input logic axi_rvalid_i`
- `output logic axi_rready_o`
- `input logic [MEM_DATA_WIDTH-1:0] axi_rdata_i`
- `input logic [7:0] axi_rid_i`
- `input logic [1:0] axi_rresp_i`
- `input logic axi_rlast_i`

Payload output boundary:

- `output logic payload_valid_o`
- `input logic payload_ready_i`
- `output logic [31:0] payload_word_o`
- `output logic payload_last_o`

Fixture output and compact debug:

- expose only narrow payload/status evidence at the top level;
- keep full qweight tensors, DDR contents, and wide traces internal to the
  testbench or report;
- outputs and compact status must be stable while `done_o` is asserted.

## Planner Requirements

The adapter must record the checkpoint-aware qweight stream plan when available.
The sub-agent prompt provides qweight shard file, tensor key, qweight byte
offset, qweight byte count, aligned request byte address, request beat count,
first-beat offset, last valid bytes, and `stream_plan_valid`.

The report must record:

- selected projection name, initially `q_proj`;
- qweight shard file and tensor key when available;
- qweight byte offset and byte count;
- aligned request byte address;
- target planned read beat count;
- bounded fixture read beat count;
- first-beat byte offset and last valid bytes;
- configured memory data width;
- payload width and emitted payload count;
- whether the target request is split or truncated by the bounded fixture.

The bounded fixture may consume a small deterministic response subset, but the
report must clearly distinguish:

- `target_checkpoint_request_planning_only`;
- `fixture_axi_read_data_execution`;
- target planned request beats;
- fixture consumed read-data beats.

## Fixture Requirements

The bounded RTL fixture must:

- accept `start_i` and become ready for AXI `R` channel data;
- consume exactly two deterministic 128-bit read-data beats unless the report
  explains a different bounded fixture size;
- apply deterministic `payload_ready_i` backpressure, including at least one
  ready-low cycle while a payload word is valid;
- require matching `axi_rid_i` for every accepted beat;
- require `axi_rresp_i == 2'b00` for every accepted beat;
- require `axi_rlast_i` only on the final accepted beat;
- split each accepted read-data beat into four 32-bit payload words in little
  chunk order;
- dynamically observe and compare every emitted payload word;
- keep `done_o` asserted until `start_i` deasserts;
- keep outputs and compact status stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_axi_read_data_channel_adapter`;
- `coverage_level: projection_axi_read_data_channel_adapter_fixture`;
- selected projection metadata;
- `checkpoint_target_weight_stream_plan`;
- `checkpoint_target_request_summary`;
- `target_checkpoint_request_planning_only`;
- `fixture_axi_read_data_execution`;
- AXI R-channel accepted beat trace;
- R-channel backpressure trace;
- response ID, response code, and last-beat validation results;
- emitted payload words in hex from observed transactions;
- proof that payload order matches the Python/fixture golden order;
- configured memory width, payload width, and payload count;
- target planned beat count and fixture consumed beat count;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- `does_not_claim` entries for DDR controller integration, full AXI master,
  full qweight payload streaming, complete board shell, full target projection
  execution, full model execution, and board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- AXI R-channel beats and payload words are dynamically observed and compared.
- Backpressure stability is dynamically checked.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only Codex verification agent finds no P0/P1 issues.
