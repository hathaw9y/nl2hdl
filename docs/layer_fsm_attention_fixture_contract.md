# Layer FSM Attention Fixture Contract

This contract defines the refreshed fixture-level Layer FSM task after
`decoder_child_attention_datapath`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a fixture-level Layer FSM that instantiates and schedules
`decoder_child_attention_datapath` as the decoder-layer child unit.

This milestone proves that a higher-level layer scheduler can call the refreshed
decoder child datapath that already includes:

- RMSNorm/RoPE source-path fixture behavior;
- projection internal stream-shell fixture behavior;
- attention/KV-cache fixture behavior.

This is not a full LLaMA-3.2-1B layer, not a multi-layer model, and not a Top
FSM. It must not claim mathematically complete Q/K/V/O projection-to-attention
wiring, residual scheduling, MLP/SwiGLU execution, token prefill/decode
scheduling, DDR/AXI movement, full model execution, or board-level signoff.

## Kernel Name

Use:

- CLI kernel: `layer_fsm_attention_fixture`
- HDL module: `layer_fsm_attention_fixture`
- report artifact: `layer_fsm_attention_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Module

Instantiate or generate the refreshed child RTL into the same output directory:

- `decoder_child_attention_datapath`

The child module's required child SV files may also be emitted into the same
output directory for simulation and synthesis:

- `rmsnorm_rope_source_path`
- `projection_internal_stream_shell`
- `attention_kv_cache_fixture`

Do not replace the refreshed child with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- a small registered final fixture output vector;
- compact layer/child trace or status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- full hidden-size vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- 128-bit memory response words as package-level ports;
- full activation, scale, zero-point, or debug trace arrays.

## Sequencing Requirements

The Layer FSM must:

1. Accept top `start_i` while idle.
2. Start `decoder_child_attention_datapath`.
3. Hold the child `start_i` while the child is busy.
4. Observe child `done_o`.
5. Deassert child `start_i` before leaving the child-done phase.
6. Latch final fixture output/status and assert top `done_o`.
7. Keep top `done_o` asserted until top `start_i` is deasserted.

Child inputs and any child selectors must remain stable while the child is
busy.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- Layer FSM start/done trace with ordered events
  `decoder_child_attention_datapath_start` and
  `decoder_child_attention_datapath_done`;
- the child datapath trace inherited from `decoder_child_attention_datapath`:
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`;
- child output vector `[4, -2, 1, 2]` unless a new deterministic golden vector
  is documented;
- child compact status or trace vector;
- attention/KV evidence from the child, either dynamically observed through
  child signals in this Layer FSM integration testbench or explicitly labeled
  as reused `decoder_child_attention_datapath` child-report metadata;
- projection shell stream evidence from the child, either dynamically observed
  in this Layer FSM integration testbench or explicitly labeled as reused
  `decoder_child_attention_datapath` child-report metadata;
- proof that top output/status remain stable while top `done_o` is high.

If evidence is not dynamically observed in the Layer FSM integration run, the
report must not label it as observed Layer FSM evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: layer_fsm_attention_fixture`;
- `coverage_level: layer_fsm_attention_fixture`;
- `layer_fsm_fixture: true`;
- `uses_refreshed_decoder_child_attention_datapath: true`;
- child module list with `decoder_child_attention_datapath` and
  `instantiated: true`;
- nested child module summary for RMSNorm/RoPE source path, projection shell,
  and attention/KV fixture coverage;
- FSM state order;
- layer start/done trace from simulation;
- child start/done trace from simulation or clearly labeled reused child report
  metadata;
- top-level interface summary and exposed port width summary;
- fixture output vector and expected/golden vector;
- attention/KV evidence summary and source label;
- projection shell evidence summary and source label;
- `implementation_stage`, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- `omitted_operations`;
- `does_not_claim`.

## Required Omitted Operations

Unless independently implemented and verified, the report must list these as
omitted or not claimed:

- Top FSM scheduling;
- multi-layer iteration;
- token prefill/decode loop;
- DDR/AXI packed-weight streaming;
- DDR/AXI KV-cache movement;
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
- The refreshed child module is instantiated or equivalently generated as real
  child RTL, not replaced by counters only.
- The Layer FSM child start/done sequence is dynamically observed and checked.
- The child `start_i` is held while the child is busy and deasserted after child
  `done_o`.
- Outputs are stable while top `done_o` is high.
- Report evidence labels distinguish Layer-FSM-observed data from reused child
  fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim Top FSM, full token scheduling, full decoder
  block, full attention, full KV-cache, MLP, full model execution, or
  board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
