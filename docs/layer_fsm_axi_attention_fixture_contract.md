# Layer FSM AXI Attention Fixture Contract

This contract defines the next fixture-level Layer FSM milestone after
`decoder_child_axi_attention_datapath`.

The parent agent owns this contract and dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a Layer FSM fixture that instantiates and schedules exactly one
`decoder_child_axi_attention_datapath` child.

This milestone proves that the Layer FSM can call the AXI-aware decoder child,
hold the child `start_i` while it is busy, deassert it after `done_o`, expose a
compact layer status, and preserve the child AXI projection metadata evidence
without widening the Layer FSM top-level interface.

This is not a Top FSM, not token scheduling, not multi-layer LLaMA execution,
not a DDR controller, not a full AXI master, not full qweight streaming, and
not board-level signoff.

## Kernel Name

Use:

- CLI kernel: `layer_fsm_axi_attention_fixture`
- HDL module: `layer_fsm_axi_attention_fixture`
- report artifact: `layer_fsm_axi_attention_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Module

Instantiate or generate child RTL into the same output directory:

- `decoder_child_axi_attention_datapath`

Do not replace the child with counters only. The Layer FSM testbench must read
child hierarchy signals where needed to prove nested AXI projection evidence.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- small registered layer output vector;
- compact layer status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- 128-bit AXI read data;
- AXI address, ID, response, or payload debug buses except compact status;
- full hidden-size vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- full activation, scale, zero-point, or debug trace arrays.

## Sequencing Requirements

The Layer FSM must:

1. Accept top `start_i` while idle.
2. Start `decoder_child_axi_attention_datapath`.
3. Hold child `start_i` while the child is busy.
4. Observe child `done_o`.
5. Deassert child `start_i` after observing child `done_o`.
6. Wait for child `done_o` release if the child contract requires release.
7. Latch final layer output/status and assert top `done_o`.
8. Keep top `done_o` asserted until top `start_i` is deasserted.

The Layer FSM must not own Top FSM scheduling, token loops, global DDR policy,
or multi-layer iteration.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- Layer FSM trace with ordered events
  `decoder_child_axi_attention_datapath_start`,
  `decoder_child_axi_attention_datapath_done`;
- nested decoder child trace with ordered events
  `source_path_start`, `source_path_done`, `projection_axi_start`,
  `projection_axi_done`, `attention_kv_start`, `attention_kv_done`;
- child start-hold/deassert/release evidence;
- AXI projection child command, R metadata, payload match, backpressure,
  projection output, parent compact metadata propagation, and round-trip
  evidence dynamically observed through the Layer FSM hierarchy;
- attention/KV cache write/read/score/control/output evidence dynamically
  observed through the Layer FSM hierarchy;
- final layer output vector and compact status;
- proof that top output/status remain stable while top `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: layer_fsm_axi_attention_fixture`;
- `coverage_level: layer_fsm_axi_attention_fixture`;
- `uses_axi_decoder_child: true`;
- `layer_fsm_fixture: true`;
- `datapath_child_instantiation: true`;
- child module list with `decoder_child_axi_attention_datapath` and
  `instantiated: true`;
- Layer FSM trace and nested child start/done trace;
- child start-hold/deassert/release evidence;
- top-level interface summary and exposed port width summary;
- final layer output vector and expected/golden vector;
- nested AXI projection command, metadata, parent compact propagation, payload,
  backpressure, projection output, and round-trip evidence summary;
- nested attention/KV write, read, score, control, and output evidence;
- `softmax_exp_in_rtl: false` unless true exponential softmax is implemented;
- `kv_cache_external_memory: false` unless an external KV interface is
  implemented;
- `implementation_stage`, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- compact I/O evidence, with no 128-bit AXI data exposed as top-level ports;
- `omitted_operations`;
- `does_not_claim`.

## Required Omitted Operations

Unless independently implemented and verified, the report must list these as
omitted or not claimed:

- DDR controller integration;
- multi-burst or outstanding AXI master;
- full qweight payload streaming;
- mathematically complete Q/K/V/O projection-to-attention wiring;
- full target projection execution;
- full sequence-length KV-cache;
- multi-head attention;
- grouped-query attention;
- true exponential softmax;
- DDR/AXI KV-cache movement;
- residual add scheduling;
- MLP/SwiGLU gate/up/down path;
- Top FSM scheduling;
- token prefill/decode loop;
- multi-layer model execution;
- full LLaMA model execution;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- `decoder_child_axi_attention_datapath` child RTL is instantiated or
  equivalently generated as real child RTL, not replaced by counters only.
- The Layer FSM start/done sequence is dynamically observed and checked.
- The child start-hold/deassert/release protocol is dynamically checked.
- Nested AXI projection evidence is dynamically observed from this Layer FSM
  integration run, not copied only from stale child reports.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim DDR, full AXI, full target projection, Top FSM,
  token scheduling, multi-layer execution, full model execution, or board-level
  signoff.
- A read-only verification agent finds no P0/P1/P2 issues.
