# Top FSM AXI Attention Fixture Contract

## Purpose

`top_fsm_axi_attention_fixture` is a bounded Top FSM fixture that calls the
verified `layer_fsm_axi_attention_fixture` child.

This milestone proves top-level scheduling around one AXI-aware layer call. It
does not prove token-loop scheduling, DDR integration, a full AXI master, or
full LLaMA execution.

## Public Identity

- CLI kernel: `top_fsm_axi_attention_fixture`
- HDL module: `top_fsm_axi_attention_fixture`
- report artifact: `top_fsm_axi_attention_fixture_golden.json` plus
  `kernel_report.json`
- child module: `layer_fsm_axi_attention_fixture`

## Required Interface

The generated top module must use the common HDL handshake:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

The fixture may expose only compact deterministic result/status outputs, such
as a small output vector and a compact status word. Do not expose any of these
as top-level ports:

- 128-bit AXI read data;
- AXI address/id/response debug buses;
- full hidden vectors;
- full KV arrays;
- full nested child debug arrays.

## Required Behavior

The Top FSM must:

1. Accept `start_i`.
2. Start `layer_fsm_axi_attention_fixture`.
3. Hold the layer child `start_i` while the child is busy.
4. Observe child `done_o`.
5. Deassert child `start_i`.
6. Wait until child `done_o` is observed low after child `start_i` is
   deasserted.
7. Latch the child output and compact status.
8. Assert top `done_o`.
9. Hold top outputs stable while `done_o` is asserted.
10. Clear `done_o` after top `start_i` is deasserted.

The FSM must include an explicit release-wait state or equivalent guard before
top completion. A one-child Top FSM still needs release evidence because later
token-loop schedulers may call it repeatedly.

## Required Dynamic Evidence

The testbench must dynamically check child hierarchy rather than print only
hard-coded traces. Required evidence includes:

- Top start/done trace with ordered events:
  `layer_fsm_axi_attention_fixture_start`,
  `layer_fsm_axi_attention_fixture_done`;
- layer child start-hold/deassert/release evidence, including
  `child_done_release_seen_after_start_deassert: true`;
- nested layer trace from `layer_fsm_axi_attention_fixture.status_o`;
- nested decoder child trace from the layer hierarchy;
- nested AXI projection command trace;
- nested AXI R metadata validation trace;
- nested AXI payload emitted/consumed match;
- nested AXI backpressure and payload-hold evidence;
- nested INT4 packed-weight round-trip evidence;
- nested attention/KV write/read/score/control evidence;
- final top output trace;
- top output stability while `done_o` is asserted;
- compact I/O trace.

## Required Status Propagation

The Top FSM must propagate the AXI metadata-good bits that are already visible
in the Layer FSM compact status into its own compact status. The implementation
must document and test stable top-level bit positions.

The expected source path is:

- `projection_axi_stream_integration.integration_status_o[45:42]`
- propagated to `decoder_child_axi_attention_datapath.status_o[63:60]`
- propagated to `layer_fsm_axi_attention_fixture.status_o[79:76]`
- propagated to `top_fsm_axi_attention_fixture.status_o[...]`

The testbench and `kernel_report.json` must check both the nested layer status
and the top compact status bit positions. Observing the child hierarchy alone
is not sufficient evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: top_fsm_axi_attention_fixture`
- `coverage_level: top_fsm_axi_attention_fixture`
- `top_fsm_axi_attention_fixture: true`
- `uses_axi_layer_child: true`
- `layer_child_instantiation: true`
- child module list with `layer_fsm_axi_attention_fixture` and
  `instantiated: true`
- nested child coverage summary including
  `decoder_child_axi_attention_datapath`,
  `projection_axi_stream_integration`, and `attention_kv_cache_fixture`
- `fsm_state_order`, including a release-wait state
- top start/done trace
- child start-hold/deassert/release evidence
- top compact AXI metadata propagation plan and observed evidence
- nested AXI projection evidence summary
- nested attention/KV evidence summary
- final fixture output vector and expected golden vector
- compact top-level I/O summary
- `softmax_exp_in_rtl: false`
- `kv_cache_external_memory: false`
- omitted operations and `does_not_claim`
- simulation, Verilator, and Vivado timing/resource evidence when enabled

## Does Not Claim

This fixture must not claim:

- DDR controller integration;
- multi-burst or outstanding AXI master behavior;
- full AXI master implementation;
- full qweight payload streaming;
- full target projection execution;
- mathematically complete Q/K/V/O attention wiring;
- full sequence KV-cache;
- multi-head attention;
- grouped-query attention;
- true exponential softmax;
- DDR/AXI KV-cache movement;
- residual or MLP scheduling;
- token prefill/decode loop;
- multi-layer model execution;
- full LLaMA model execution;
- board-level ZCU104 signoff.

## Pass Criteria

The milestone passes only if:

- the real `layer_fsm_axi_attention_fixture` child RTL is instantiated or
  generated into the same output directory;
- child `start_i` is held while busy, deasserted after `done_o`, and child
  `done_o` release is dynamically observed before top `done_o`;
- nested AXI metadata-good bits propagate into top compact status;
- outputs are stable while top `done_o` is asserted;
- top-level I/O remains compact;
- simulation passes;
- Verilator evidence is recorded when enabled;
- Vivado timing has non-NA setup, hold, and pulse-width slack when synthesis is
  enabled;
- the report clearly states bounded fixture scope and forbidden target claims.
