# Decoder Child Attention-Datapath Contract

This contract defines the next fixture-level decoder-child milestone after
`attention_kv_cache_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a refreshed decoder child datapath fixture that actually instantiates
and sequences the currently proven child fixtures:

- `rmsnorm_rope_source_path`
- `projection_internal_stream_shell`
- `attention_kv_cache_fixture`

This milestone closes the previous fixture-level gap where a decoder child
datapath could instantiate RMSNorm/projection/RoPE but still omit attention and
KV-cache movement.

This is still not a full LLaMA-3.2-1B decoder block. The projection fixture and
attention fixture may remain deterministic bounded fixtures with independent
small dimensions. The report must not claim that the projection output is a
complete mathematically wired Q/K/V/O attention path unless that is separately
implemented and verified.

## Kernel Name

Use:

- CLI kernel: `decoder_child_attention_datapath`
- HDL module: `decoder_child_attention_datapath`
- report artifact: `decoder_child_attention_datapath_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate child modules into the same output directory for
simulation and synthesis. Do not replace child calls with counters only.

Required child evidence:

- `rmsnorm_rope_source_path`: prove internal RMSNorm/RoPE metadata lookup
  source path, not direct top-level Python metadata ports.
- `projection_internal_stream_shell`: prove compact top-level shell behavior
  around internal packed-weight memory stream evidence.
- `attention_kv_cache_fixture`: prove bounded attention score, softmax/control,
  KV write/read movement, and compact output.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- a small registered final fixture output vector;
- compact child trace or status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- full hidden-size vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- 128-bit memory response words as package-level ports;
- full activation, scale, zero-point, or debug trace arrays.

Detailed child traces may be internal signals or testbench-observed
hierarchical signals, but `kernel_report.json` must record the observed
evidence.

## Sequencing Requirements

The child scheduler must:

1. Accept top `start_i` while idle.
2. Start `rmsnorm_rope_source_path`.
3. Wait for `rmsnorm_rope_source_path.done_o`.
4. Start `projection_internal_stream_shell`.
5. Wait for `projection_internal_stream_shell.done_o`.
6. Start `attention_kv_cache_fixture`.
7. Wait for `attention_kv_cache_fixture.done_o`.
8. Latch final fixture output/status and assert top `done_o`.
9. Keep `done_o` asserted until top `start_i` is deasserted.

Child `start_i` pulses or levels must follow each child module contract. Child
inputs and any child selectors must remain stable while that child is busy.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- child start/done trace with the ordered events
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`;
- RMSNorm lookup trace including selector, valid flag, `inv_rms`, and `sumsq`;
- RoPE lookup trace including position, pair index, valid flag, cos, and sin;
- projection shell internal request/response/payload evidence or a compact
  reference to the child report fields that were dynamically observed in this
  integration run;
- projection shell output vector;
- attention/KV cache write trace;
- attention key/value read trace;
- attention score trace;
- attention softmax/control trace;
- attention output vector;
- final top fixture output vector and compact status;
- proof that top output/status remain stable while top `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: decoder_child_attention_datapath`;
- `coverage_level: decoder_child_attention_datapath_fixture`;
- `previous_gap_closed: attention_kv_fixture_in_child_datapath`;
- `datapath_child_instantiation: true`;
- `attention_kv_child_instantiation: true`;
- child module list with child coverage levels and `instantiated: true`;
- child sequence list;
- child start/done trace from simulation;
- top-level interface summary and exposed port width summary;
- fixture output vector and expected/golden vector;
- source-path RMSNorm/RoPE lookup and output evidence;
- projection shell output and internal stream evidence summary;
- attention/KV write, read, score, control, and output evidence;
- `softmax_exp_in_rtl: false` unless true exponential softmax is implemented;
- `kv_cache_external_memory: false` unless an external KV interface is
  implemented;
- `implementation_stage`, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- compact I/O evidence, with bonded IOB count preferably less than or equal to
  180;
- `omitted_operations`;
- `does_not_claim`.

## Required Omitted Operations

Unless independently implemented and verified, the report must list these as
omitted or not claimed:

- mathematically complete Q/K/V/O projection-to-attention wiring;
- full target projection execution;
- full sequence-length KV-cache;
- multi-head attention;
- grouped-query attention;
- true exponential softmax;
- DDR/AXI KV-cache movement;
- residual add scheduling;
- MLP/SwiGLU gate/up/down path;
- full token prefill/decode loop;
- full decoder block;
- full LLaMA model execution;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Child modules are instantiated or equivalently generated as real child RTL,
  not replaced by counters only.
- The child start/done sequence is dynamically observed and checked.
- Child output/status signals used by the top fixture are stable while each
  child is busy and while the top is done.
- RMSNorm/RoPE source-path evidence is dynamically observed.
- Projection shell evidence is dynamically observed.
- Attention/KV evidence is dynamically observed.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim full decoder block, full attention, full KV-cache,
  MLP, full model execution, or board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
