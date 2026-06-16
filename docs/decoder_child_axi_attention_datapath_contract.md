# Decoder Child AXI Attention-Datapath Contract

This contract defines the next fixture-level decoder-child milestone after the
q/k/v/o `projection_axi_stream_integration` packets.

The parent agent owns this contract and dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a refreshed decoder child datapath fixture that instantiates and
sequences:

- `rmsnorm_rope_source_path`
- q/k/v/o AXI projection stream integrations:
  `projection_axi_stream_integration`,
  `projection_k_proj_axi_stream_integration`,
  `projection_v_proj_axi_stream_integration`, and
  `projection_o_proj_axi_stream_integration`
- `attention_kv_cache_fixture`

This milestone replaces the previous internal-memory projection child with a
bounded AXI projection stream bundle inside the decoder-child fixture. It proves
that the decoder-child scheduler can carry compact q/k/v/o AXI projection
evidence into the attention path without exposing wide AXI data or debug buses
at the child top level.

This is still not a full LLaMA-3.2-1B decoder block, not a full AXI master, not
a DDR controller, not full qweight streaming, and not board-level signoff.

## Kernel Name

Use:

- CLI kernel: `decoder_child_axi_attention_datapath`
- HDL module: `decoder_child_axi_attention_datapath`
- report artifact: `decoder_child_axi_attention_datapath_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate child modules into the same output directory for
simulation and synthesis. Do not replace child calls with counters only.

Required child evidence:

- `rmsnorm_rope_source_path`: internal RMSNorm/RoPE metadata lookup source path.
- q/k/v/o `projection_axi_stream_integration` packets: bounded two-beat AXI
  read transaction, DUT-side RID/RRESP/RLAST good-path validation, payload
  match, projection output, and packed INT4 round-trip evidence for each
  attention projection.
- `attention_kv_cache_fixture`: bounded attention score, softmax/control,
  KV write/read movement, and compact output.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- small registered final fixture output vector;
- compact child trace or status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- 128-bit AXI read data;
- AXI address, ID, response, or payload debug buses except compact status;
- full hidden-size vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- full activation, scale, zero-point, or debug trace arrays.

Detailed child traces may be internal signals or testbench-observed
hierarchical signals, but `kernel_report.json` must record observed evidence.

## Sequencing Requirements

The child scheduler must:

1. Accept top `start_i` while idle.
2. Start `rmsnorm_rope_source_path`.
3. Wait for `rmsnorm_rope_source_path.done_o`.
4. Start q/k/v/o AXI projection stream children, either sequentially or through
   a documented bounded arbitration schedule.
5. Wait until every q/k/v/o projection stream child has asserted `done_o`.
6. Start `attention_kv_cache_fixture`.
7. Wait for `attention_kv_cache_fixture.done_o`.
8. Latch final fixture output/status and assert top `done_o`.
9. Keep `done_o` asserted until top `start_i` is deasserted.

Child `start_i` levels must follow each child module contract. Child inputs and
selectors must remain stable while each child is busy.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- child start/done trace with ordered events
  `source_path_start`, `source_path_done`, `projection_axi_start`,
  `projection_axi_done`, `attention_kv_start`, `attention_kv_done`;
- RMSNorm lookup trace including selector, valid flag, `inv_rms`, and `sumsq`;
- RoPE lookup trace including position, pair index, valid flag, cos, and sin;
- q/k/v/o AXI projection child command, R metadata, payload match, projection
  output, backpressure, and round-trip evidence dynamically observed in this
  integration run;
- compact child status showing RID/RRESP/RLAST good-path validation propagated
  from each q/k/v/o AXI projection child;
- attention/KV cache write trace;
- attention key/value read trace;
- attention score trace;
- attention softmax/control trace;
- attention output vector;
- final top fixture output vector and compact status;
- proof that top output/status remain stable while top `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: decoder_child_axi_attention_datapath`;
- `coverage_level: decoder_child_axi_attention_datapath_fixture`;
- `uses_axi_projection_stream_child: true`;
- `datapath_child_instantiation: true`;
- `attention_kv_child_instantiation: true`;
- child module list with child coverage levels and `instantiated: true`;
- child sequence list;
- child start/done trace from simulation;
- top-level interface summary and exposed port width summary;
- fixture output vector and expected/golden vector;
- source-path RMSNorm/RoPE lookup and output evidence;
- q/k/v/o AXI projection child command, metadata, payload, projection output,
  backpressure, and round-trip evidence summary;
- attention/KV write, read, score, control, and output evidence;
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
- AXI projection child evidence is dynamically observed from this integration
  run, not copied only from a stale report.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim DDR, full AXI, full target projection, full
  decoder block, full model execution, or board-level signoff.
- A read-only verification agent finds no P0/P1/P2 issues.
