# Projection AXI Stream Integration Contract

This contract defines the fixture after `projection_axi_read_transaction_adapter`.

The parent agent owns this contract and dispatch packet. HDL sub-agents implement
RTL or RTL generator changes.

## Scope

Create a bounded fixture that connects one verified AXI read transaction payload
stream to a projection-style packed INT4 consumer through a valid/ready link.

This milestone proves that a bounded AXI AR/R transaction can produce 32-bit
packed payload words, preserve R-channel metadata validation status, and feed a
small projection-style consumer without dropping, duplicating, or reordering
payloads under backpressure.

This is not a DDR controller, not a full AXI master, not full qweight streaming,
not a target-scale LLaMA projection, and not board-level signoff.

## Kernel Name

Use:

- CLI kernel: `projection_axi_stream_integration`
- HDL module: `projection_axi_stream_integration`
- report artifact: `projection_axi_stream_integration_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

AXI read boundary:

- One bounded read-address command equivalent to
  `projection_axi_read_transaction_adapter`.
- One bounded R-channel data fixture with matching RID, OKAY RRESP, and RLAST on
  the final beat only.
- DUT-side compact metadata status for RID/RRESP/RLAST validation.

Projection payload boundary:

- 32-bit packed INT4 payload words.
- `valid`/`ready` handshake from the AXI transaction stream into a
  projection-style consumer.
- Projection-side ready-low backpressure must be able to stall payload transfer.

## Fixture Requirements

The fixture must:

- execute at least one bounded two-beat 128-bit AXI read transaction;
- derive exactly eight 32-bit payload chunks from those beats;
- preserve payload order: read beat order, then little chunk order within each
  beat;
- consume all eight payload chunks through the projection-side valid/ready
  interface;
- dynamically observe emitted payloads and consumed payloads from the
  integration run, rather than copying both from one constant list;
- record good-path DUT-observed RID/RRESP/RLAST validation status;
- include a negative metadata regression or preserve a child-proven negative
  regression reference in the report without claiming the integration run itself
  is a bad-path test;
- apply projection-side backpressure at least once inside a beat and at or
  across a beat boundary;
- compute a small deterministic projection-style output from consumed payloads;
- prove packed nibble round-trip explicitly;
- keep `done_o` asserted until `start_i` deasserts;
- keep output and compact status stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_axi_stream_integration`;
- `coverage_level: projection_axi_stream_integration_fixture`;
- configured memory width, AXI beat count, payload width, and payload count;
- observed AXI command trace;
- observed R-channel metadata validation trace;
- emitted payload words in hex;
- consumed payload words in hex;
- proof that emitted and consumed payloads match exactly, with evidence sourced
  from observed integration transactions;
- projection-side backpressure trace;
- projection output vector and Python/NumPy golden output vector;
- lane policy with requested, effective fixture, and true parallel datapath
  lanes when arithmetic lanes are used;
- `round_trip_passed: true`;
- child negative R metadata regression reference or integration bad-path
  regression evidence;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- explicit caveats that the fixture is not a DDR controller, not a full AXI
  master, not full qweight payload streaming, not full target projection
  execution, not full model execution, and not board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1/P2 issues.
