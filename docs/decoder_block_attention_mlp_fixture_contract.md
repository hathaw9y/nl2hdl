# Decoder Block Attention MLP Fixture Contract

This contract defines the next bounded integration fixture after
`token_loop_attention_fixture` and `residual_mlp_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a fixture-level decoder block scheduler that composes the refreshed
attention datapath fixture with the residual/MLP fixture:

1. Call `decoder_child_attention_datapath`.
2. Feed or adapt its deterministic attention output into
   `residual_mlp_fixture`.
3. Call `residual_mlp_fixture`.
4. Latch the final decoder-block fixture output and compact status.

This milestone proves that a single decoder block can schedule an
attention-like child path and an MLP/residual child path with common
handshakes. It is not target-scale LLaMA decoder execution.

## Kernel Name

Use:

- CLI kernel: `decoder_block_attention_mlp_fixture`
- HDL module: `decoder_block_attention_mlp_fixture`
- report artifact: `decoder_block_attention_mlp_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate real child RTL into the same output directory:

- `decoder_child_attention_datapath`
- `residual_mlp_fixture`

The attention child may also require nested RTL files:

- `rmsnorm_rope_source_path`
- `projection_internal_stream_shell`
- `attention_kv_cache_fixture`

Do not replace either child with counters only. The scheduler may use fixture
constant hidden inputs for the MLP child, but it must consume the attention
child's observed output for the MLP child's `attention_output_i`.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data must stay compact:

- small registered final decoder-block fixture output vector;
- compact block/attention/MLP trace or status summary.

Top-level data must not expose:

- attention child full source/projection/KV debug buses;
- residual/MLP hidden or attention input vectors as package-level ports;
- residual/MLP intermediate vectors as package-level ports;
- full hidden-size vectors;
- full sequence KV-cache arrays;
- full gate/up/down matrices or intermediate tensors;
- 128-bit memory response words as package-level ports.

Because `residual_mlp_fixture` standalone uses 292/360 bonded IOB, this
integration should internalize child inputs/status where possible and target a
lower package I/O footprint than exposing all child vector ports.

## Sequencing Requirements

The block scheduler must:

1. Accept block `start_i` while idle.
2. Start `decoder_child_attention_datapath`.
3. Hold attention child `start_i` while busy.
4. Observe attention child `done_o`.
5. Capture the attention output vector.
6. Deassert attention child `start_i` and wait for attention child `done_o` to
   clear.
7. Start `residual_mlp_fixture`, feeding deterministic hidden fixture input
   and the captured attention output.
8. Hold MLP child `start_i` while busy.
9. Observe MLP child `done_o`.
10. Capture final MLP output vector.
11. Deassert MLP child `start_i` and wait for MLP child `done_o` to clear.
12. Assert block `done_o` and hold it until block `start_i` deasserts.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- ordered block trace:
  `attention_start`, `attention_done`, `mlp_start`, `mlp_done`;
- attention child trace:
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`;
- MLP child trace:
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`;
- child start-hold/deassert/release evidence for both children;
- captured attention output vector;
- MLP hidden fixture input vector;
- MLP final output vector;
- final decoder-block fixture output vector;
- source/projection/attention/KV evidence from the attention child, either
  dynamically observed through hierarchy in this block testbench or explicitly
  labeled as reused child fixture metadata;
- residual/MLP intermediate evidence from the MLP child, either dynamically
  observed through hierarchy in this block testbench or explicitly labeled as
  reused child fixture metadata;
- proof that block output/status remain stable while block `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the decoder-block integration run,
the report must not label it as observed block-run evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: decoder_block_attention_mlp_fixture`;
- `coverage_level: decoder_block_attention_mlp_fixture`;
- `decoder_block_attention_mlp_fixture: true`;
- child module list with both required children and `instantiated: true`;
- nested attention child coverage summary;
- FSM state order;
- ordered block trace from simulation;
- attention child trace and MLP child trace from simulation or clearly labeled
  reused metadata;
- child start-hold/deassert/release evidence for both children;
- captured attention output vector, MLP hidden input vector, MLP final vector,
  and final decoder-block output vector;
- arithmetic/source policy carried from child fixtures, including that
  `residual_mlp_fixture` uses fixture constant matrices and no true
  SiLU/exponential RTL;
- top-level interface summary and exposed port width summary;
- package I/O mitigation summary relative to the standalone residual/MLP
  fixture;
- `implementation_stage`, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- timing caveat that routed timing is fixture timing without board-level I/O
  delay, DDR/AXI shell, or PS/PL integration constraints;
- `omitted_operations`;
- `does_not_claim`.

## Required Omitted Operations

Unless independently implemented and verified, the report must list these as
omitted or not claimed:

- target-scale LLaMA decoder block dimensions;
- mathematically complete Q/K/V/O projection-to-attention wiring;
- full sequence-length KV-cache;
- multi-head attention and grouped-query attention;
- true exponential softmax;
- target-scale MLP dimensions;
- GPTQ INT4 packed gate/up/down weight streaming;
- DDR/AXI packed-weight movement;
- DDR/AXI KV-cache movement;
- true SiLU or exponential activation math in RTL;
- full hidden-size RMSNorm before MLP;
- token-dependent full-sequence KV-cache accumulation;
- target multi-layer LLaMA iteration;
- logits generation and sampling;
- full LLaMA layer execution;
- full LLaMA model execution;
- board shell and PS/PL integration;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Both required child modules are instantiated as real RTL.
- Common handshake contract is met for the block and both children.
- Child `start_i` is held while busy, deasserted after `done_o`, and followed
  by a release wait for both children.
- Ordered block/attention/MLP traces are checked.
- Numeric final output matches Python/NumPy golden vectors.
- Outputs are stable while block `done_o` is high.
- Top-level I/O is narrower than exposing both children directly and avoids
  wide debug/status arrays.
- Report evidence labels distinguish block-observed data from reused child
  fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim target-scale decoder execution, DDR/AXI movement,
  full LLaMA layer/model execution, or board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
