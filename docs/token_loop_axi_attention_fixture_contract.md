# Token Loop AXI Attention Fixture Contract

## Purpose

`token_loop_axi_attention_fixture` is a bounded token-loop fixture that calls
the verified `top_fsm_axi_attention_fixture` child for two deterministic token
steps.

This milestone proves repeated top-level scheduling around an AXI-aware
attention fixture. It does not prove DDR integration, a real prefill/decode
loop, full sequence KV-cache movement, multi-layer LLaMA execution, or board
signoff.

## Public Identity

- CLI kernel: `token_loop_axi_attention_fixture`
- HDL module: `token_loop_axi_attention_fixture`
- report artifact: `token_loop_axi_attention_fixture_golden.json` plus
  `kernel_report.json`
- child module: `top_fsm_axi_attention_fixture`

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
- full nested child debug arrays;
- per-token child vectors.

## Required Behavior

The token-loop FSM must:

1. Accept `start_i`.
2. Start `top_fsm_axi_attention_fixture` for token step 0.
3. Hold child `start_i` while the child is busy.
4. Observe child `done_o`.
5. Capture token 0 output and compact child status.
6. Deassert child `start_i`.
7. Wait until child `done_o` is observed low after child `start_i` is
   deasserted.
8. Start `top_fsm_axi_attention_fixture` for token step 1.
9. Repeat the same hold/deassert/release sequence.
10. Latch final token output and compact status.
11. Assert token-loop `done_o`.
12. Hold outputs stable while `done_o` is asserted.
13. Clear `done_o` after top `start_i` is deasserted.

The FSM must include release-wait states or equivalent guards between repeated
child calls. Starting token 1 while token 0 child `done_o` is still high is a
contract failure.

## Required Dynamic Evidence

The testbench must dynamically check child hierarchy rather than print only
hard-coded traces. Required evidence includes:

- token-loop trace with ordered events:
  `token0_start`, `token0_done`, `token1_start`, `token1_done`;
- per-token child call trace for token 0 and token 1;
- per-token child start-hold/deassert/release evidence, including release seen
  after start deassertion for both token calls;
- nested Top FSM trace from `top_fsm_axi_attention_fixture.status_o`;
- nested Layer FSM trace;
- nested decoder child trace;
- nested AXI projection command trace;
- nested AXI R metadata validation trace;
- nested AXI payload emitted/consumed match;
- nested AXI backpressure and payload-hold evidence;
- nested INT4 packed-weight round-trip evidence;
- nested attention/KV write/read/score/control evidence;
- final token-loop output trace;
- output stability while `done_o` is asserted;
- compact I/O trace.

## Required Status Propagation

The token-loop FSM must propagate the AXI metadata-good bits that are already
visible in the Top FSM compact status into its own compact status. The
implementation must document and test stable token-loop bit positions.

The expected source path is:

- `projection_axi_stream_integration.integration_status_o[45:42]`
- propagated to `decoder_child_axi_attention_datapath.status_o[63:60]`
- propagated to `layer_fsm_axi_attention_fixture.status_o[79:76]`
- propagated to `top_fsm_axi_attention_fixture.status_o[95:92]`
- propagated to `token_loop_axi_attention_fixture.status_o[...]`

The testbench and `kernel_report.json` must check both the nested Top FSM
status and the token-loop compact status bit positions. Observing the child
hierarchy alone is not sufficient evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: token_loop_axi_attention_fixture`
- `coverage_level: token_loop_axi_attention_fixture`
- `token_loop_axi_attention_fixture: true`
- `uses_axi_top_fsm_child: true`
- `top_fsm_child_instantiation: true`
- child module list with `top_fsm_axi_attention_fixture` and
  `instantiated: true`
- nested child coverage summary including
  `layer_fsm_axi_attention_fixture`,
  `decoder_child_axi_attention_datapath`,
  `projection_axi_stream_integration`, and `attention_kv_cache_fixture`
- `fsm_state_order`, including release-wait states between token calls
- token-loop start/done trace
- per-token child start-hold/deassert/release evidence
- token-loop compact AXI metadata propagation plan and observed evidence
- nested AXI projection evidence summary
- nested attention/KV evidence summary
- final fixture output vector and expected golden vector
- repeated-output or token-output policy for the bounded deterministic fixture
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
- real prefill/decode token loop;
- multi-layer model execution;
- full LLaMA model execution;
- board-level ZCU104 signoff.

## Pass Criteria

The milestone passes only if:

- the real `top_fsm_axi_attention_fixture` child RTL is instantiated or
  generated into the same output directory;
- child `start_i` is held while busy, deasserted after `done_o`, and child
  `done_o` release is dynamically observed before token 1 starts and before
  token-loop `done_o`;
- nested AXI metadata-good bits propagate into token-loop compact status;
- outputs are stable while token-loop `done_o` is asserted;
- top-level I/O remains compact;
- simulation passes;
- Verilator evidence is recorded when enabled;
- Vivado timing has non-NA setup, hold, and pulse-width slack when synthesis is
  enabled;
- the report clearly states bounded fixture scope and forbidden target claims.
