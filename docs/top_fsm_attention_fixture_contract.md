# Top FSM Attention Fixture Contract

This contract defines the refreshed fixture-level Top FSM task after
`layer_fsm_attention_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a fixture-level Top FSM that instantiates and schedules
`layer_fsm_attention_fixture` as one fixture layer.

This milestone proves that model-level orchestration can call the refreshed
Layer FSM path that already includes:

- `decoder_child_attention_datapath`;
- RMSNorm/RoPE source-path fixture behavior;
- projection internal stream-shell fixture behavior;
- attention/KV-cache fixture behavior.

This is not full LLaMA-3.2-1B execution, not a real token prefill/decode loop,
not multi-layer target scheduling, not a DDR/AXI board shell, and not
board-level ZCU104 signoff.

## Kernel Name

Use:

- CLI kernel: `top_fsm_attention_fixture`
- HDL module: `top_fsm_attention_fixture`
- report artifact: `top_fsm_attention_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Module

Instantiate or generate the refreshed layer RTL into the same output directory:

- `layer_fsm_attention_fixture`

The layer's required nested SV files may also be emitted into the same output
directory for simulation and synthesis:

- `decoder_child_attention_datapath`
- `rmsnorm_rope_source_path`
- `projection_internal_stream_shell`
- `attention_kv_cache_fixture`

Do not replace the refreshed Layer FSM with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- a small registered final fixture output vector;
- compact top/layer/child trace or status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- full hidden-size vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- 128-bit memory response words as package-level ports;
- full activation, scale, zero-point, or debug trace arrays.

## Sequencing Requirements

The Top FSM must:

1. Accept top `start_i` while idle.
2. Start `layer_fsm_attention_fixture`.
3. Hold the layer `start_i` while the layer is busy.
4. Observe layer `done_o`.
5. Deassert layer `start_i` before leaving the layer-done phase.
6. Latch final fixture output/status and assert top `done_o`.
7. Keep top `done_o` asserted until top `start_i` is deasserted.

The first gate may schedule exactly one fixture layer. If so, report
`fixture_layer_count: 1`.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- Top FSM layer start/done trace with ordered events
  `layer_fsm_attention_fixture_start` and
  `layer_fsm_attention_fixture_done`;
- the Layer FSM trace inherited from `layer_fsm_attention_fixture`:
  `decoder_child_attention_datapath_start` and
  `decoder_child_attention_datapath_done`;
- the child datapath trace inherited from `decoder_child_attention_datapath`:
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`;
- final fixture output vector `[4, -2, 1, 2]` unless a new deterministic golden
  vector is documented;
- layer compact status or trace vector;
- source-path, projection shell, and attention/KV evidence from the layer/child,
  either dynamically observed through hierarchical child signals in this Top
  FSM integration testbench or explicitly labeled as reused child fixture
  metadata;
- proof that top output/status remain stable while top `done_o` is high.

If evidence is not dynamically observed in the Top FSM integration run, the
report must not label it as observed Top FSM evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: top_fsm_attention_fixture`;
- `coverage_level: top_fsm_attention_fixture`;
- `top_fsm_fixture: true`;
- `uses_refreshed_layer_fsm_attention_fixture: true`;
- `fixture_layer_count: 1` for the first gate;
- child module list with `layer_fsm_attention_fixture` and `instantiated: true`;
- nested child coverage summary for `decoder_child_attention_datapath`,
  RMSNorm/RoPE source path, projection shell, and attention/KV fixture;
- FSM state order;
- top/layer start/done trace from simulation;
- layer-child and decoder-child traces from simulation or clearly labeled reused
  metadata;
- top-level interface summary and exposed port width summary;
- fixture output vector and expected/golden vector;
- source-path evidence summary and source label;
- attention/KV evidence summary and source label;
- projection shell evidence summary and source label;
- `implementation_stage`, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- a caveat that routed timing is fixture timing without board I/O delay
  constraints;
- `omitted_operations`;
- `does_not_claim`.

## Required Omitted Operations

Unless independently implemented and verified, the report must list these as
omitted or not claimed:

- real token prefill/decode loop;
- multi-layer LLaMA target iteration beyond the reported fixture count;
- target sequence scheduling;
- DDR/AXI packed-weight streaming;
- DDR/AXI KV-cache movement;
- board shell and PS/PL integration;
- mathematically complete Q/K/V/O projection-to-attention wiring;
- full sequence-length KV-cache;
- multi-head attention;
- grouped-query attention;
- true exponential softmax;
- residual add scheduling;
- MLP/SwiGLU gate/up/down path;
- full LLaMA layer execution;
- full LLaMA model execution;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- The refreshed Layer FSM module is instantiated or equivalently generated as
  real child RTL, not replaced by counters only.
- The Top FSM layer start/done sequence is dynamically observed and checked.
- The layer `start_i` is held while the layer is busy and deasserted after
  layer `done_o`.
- Outputs are stable while top `done_o` is high.
- Report evidence labels distinguish Top-FSM-observed data from reused layer or
  child fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim full token scheduling, multi-layer target
  scheduling, DDR/AXI movement, full decoder block, full attention, full
  KV-cache, MLP, full model execution, or board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
