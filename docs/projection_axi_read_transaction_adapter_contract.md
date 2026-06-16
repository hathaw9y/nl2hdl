# Projection AXI Read Transaction Adapter Contract

This contract defines the next projection memory milestone after
`projection_axi_read_data_channel_adapter`.

The parent agent owns this contract and the dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a bounded AXI read transaction fixture for packed INT4 projection
weights. The fixture issues one read-address command, consumes the matching
read-data beats, and emits packed 32-bit payload words in projection stream
order.

This milestone proves that the framework can:

- derive or accept a bounded read request from checkpoint-aware qweight stream
  planning metadata;
- issue one AXI-style read-address command with valid/ready backpressure;
- consume matching AXI `R` channel beats with valid/ready backpressure;
- check read ID, response code, beat count, command length, and final `last`;
- split each 128-bit read-data beat into four 32-bit payload words in little
  chunk order;
- report target qweight request metadata separately from bounded fixture
  execution;
- keep the fixture separate from DDR controller integration, a full AXI master,
  complete board shell, full target projection execution, full model execution,
  and board-level signoff.

This milestone is a bounded transaction fixture. It may model one read burst,
but it is not an external DDR controller, not a PS/PL board shell, and not full
target checkpoint qweight streaming.

## Kernel Name

Use:

- CLI kernel: `projection_axi_read_transaction_adapter`
- HDL module: `projection_axi_read_transaction_adapter`
- report artifact: `projection_axi_read_transaction_adapter_golden.json` plus
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

The top-level fixture must keep metadata and trace outputs compact. Full
qweight tensors, full DDR contents, and wide traces belong in the testbench or
report, not package-level ports.

## Planner Requirements

The adapter must record the checkpoint-aware qweight stream plan when available.
The sub-agent prompt provides qweight shard file, tensor key, qweight byte
offset, qweight byte count, aligned request byte address, request beat count,
first-beat offset, last valid bytes, and `stream_plan_valid`.

The report must record:

- selected projection name, initially `q_proj`;
- qweight shard file and tensor key when available;
- aligned request byte address and target planned beat count;
- bounded fixture command beat count and consumed read-data beat count;
- AXI command fields: `araddr`, `arlen`, `arsize`, `arburst`, `arid`;
- first-beat byte offset and last valid bytes;
- configured memory data width;
- payload width and emitted payload count;
- whether the target request is split or truncated by the bounded fixture.

The bounded fixture may execute only a small deterministic read transaction, but
the report must clearly distinguish:

- `target_checkpoint_request_planning_only`;
- `fixture_axi_read_transaction_execution`;
- target planned request beats;
- fixture command/read-data beats.

## Fixture Requirements

The bounded RTL fixture must:

- issue exactly one AXI read-address command after `start_i`;
- hold AR fields stable while `axi_arvalid_o` is high and `axi_arready_i` is
  low;
- after the AR command is accepted, consume exactly two deterministic 128-bit
  read-data beats unless the report explains a different bounded fixture size;
- require matching `axi_rid_i` for every accepted R beat;
- require `axi_rresp_i == 2'b00` for every accepted R beat;
- require `axi_rlast_i` only on the final accepted R beat;
- prove `axi_arlen_o + 1` matches the fixture consumed R beat count;
- split every accepted R beat into 32-bit payload words in little chunk order;
- dynamically observe and compare every emitted payload word;
- exercise deterministic AR, R, and payload backpressure;
- keep `done_o` asserted until `start_i` deasserts;
- keep outputs and compact status stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_axi_read_transaction_adapter`;
- `coverage_level: projection_axi_read_transaction_adapter_fixture`;
- selected projection metadata;
- `checkpoint_target_weight_stream_plan`;
- `checkpoint_target_request_summary`;
- `target_checkpoint_request_planning_only`;
- `fixture_axi_read_transaction_execution`;
- observed AR command trace;
- observed R-channel accepted beat trace;
- AR/R transaction consistency result;
- response ID, response code, and last-beat validation results;
- emitted payload words in hex from observed transactions;
- proof that payload order matches the fixture golden order;
- AR, R, and payload backpressure traces;
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
- AR command, R-channel beats, and payload words are dynamically observed and
  compared.
- Backpressure stability is dynamically checked for AR, R, and payload paths.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only Codex verification agent finds no P0/P1 issues.
