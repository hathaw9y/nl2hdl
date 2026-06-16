# Decoder Block AXI Attention MLP Fixture Contract

This contract defines the bounded decoder-block integration step after
`decoder_child_axi_attention_datapath` and `residual_mlp_fixture`.

The parent agent owns this contract and dispatch packet. HDL sub-agents
implement RTL or RTL generator changes. The parent agent must not hand-write
generated HDL.

## Scope

Create a fixture-level decoder block scheduler that composes:

- `decoder_child_axi_attention_datapath`
- `residual_mlp_fixture`

This is the AXI-aware successor to `decoder_block_attention_mlp_fixture`.
It must preserve q/k/v/o AXI projection stream evidence from the attention
child and then feed the captured attention output into the residual/MLP child.

This milestone proves bounded single-block scheduling with an AXI-aware
attention path plus MLP/residual fixture. It is not target-scale LLaMA decoder
execution.

## Kernel Name

Use:

- CLI kernel: `decoder_block_axi_attention_mlp_fixture`
- HDL module: `decoder_block_axi_attention_mlp_fixture`
- report artifact: `decoder_block_axi_attention_mlp_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate real child RTL into the same output directory:

- `decoder_child_axi_attention_datapath`
- `residual_mlp_fixture`

The AXI attention child also requires nested RTL files:

- `rmsnorm_rope_source_path`
- q/k/v/o `projection_axi_stream_integration` children
- `attention_kv_cache_fixture`

The MLP child may use fixture constant matrices. It must consume the captured
attention child output through `attention_output_i`; do not replace either
child with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data must stay compact:

- small registered final decoder-block fixture output vector;
- compact block/attention/MLP/AXI metadata status.

Top-level data must not expose:

- 128-bit AXI read data or full AXI debug buses;
- attention child full source/projection/KV debug buses;
- q/k/v/o full projection vectors or packed payload arrays as package ports;
- residual/MLP child hidden or attention package input vectors;
- residual/MLP intermediate vectors;
- full hidden-size vectors;
- full sequence KV-cache arrays;
- full gate/up/down matrices or intermediate tensors.

## Sequencing Requirements

The block scheduler must:

1. Accept block `start_i` while idle.
2. Start `decoder_child_axi_attention_datapath`.
3. Hold attention child `start_i` while busy.
4. Observe attention child `done_o`.
5. Capture the attention output vector and compact AXI metadata bits.
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
  `axi_attention_start`, `axi_attention_done`, `mlp_start`, `mlp_done`;
- AXI attention child trace:
  `source_path_start`, `source_path_done`, `projection_axi_start`,
  `projection_axi_done`, `attention_kv_start`, `attention_kv_done`;
- q/k/v/o AXI projection child command, R metadata, payload, output, and
  round-trip evidence, dynamically observed or clearly labeled as reused child
  fixture metadata;
- aggregate q/k/v/o AXI metadata propagation into block compact status;
- MLP child trace:
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`;
- child start-hold/deassert/release evidence for both children;
- captured attention output vector;
- MLP hidden fixture input vector;
- MLP final output vector;
- final decoder-block fixture output vector;
- proof that the MLP child consumed the captured AXI attention output;
- proof that block output/status remain stable while block `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the decoder-block integration run,
the report must not label it as observed block-run evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: decoder_block_axi_attention_mlp_fixture`;
- `coverage_level: decoder_block_axi_attention_mlp_fixture`;
- `decoder_block_axi_attention_mlp_fixture: true`;
- child module list with both required children and `instantiated: true`;
- nested AXI attention child coverage summary including q/k/v/o projection
  stream children;
- FSM state order;
- ordered block trace from simulation;
- AXI attention child trace and MLP child trace;
- child start-hold/deassert/release evidence for both children;
- aggregate q/k/v/o AXI metadata propagation evidence;
- captured attention output vector, MLP hidden input vector, MLP final vector,
  and final decoder-block output vector;
- arithmetic/source policy carried from child fixtures, including that
  `residual_mlp_fixture` uses fixture constant matrices and no true
  SiLU/exponential RTL;
- top-level interface summary and exposed port width summary;
- package I/O mitigation summary relative to exposing both children directly;
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

- DDR controller integration;
- multi-burst or outstanding AXI master;
- full qweight payload streaming;
- full target projection execution;
- mathematically complete Q/K/V/O projection-to-attention wiring;
- target-scale LLaMA decoder block dimensions;
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
- Ordered block/AXI-attention/MLP traces are checked.
- q/k/v/o AXI projection evidence is preserved or explicitly labeled as reused
  child fixture metadata.
- Numeric final output matches Python/NumPy golden vectors.
- Outputs are stable while block `done_o` is high.
- Top-level I/O avoids wide AXI, child vector, and debug package ports.
- Report evidence labels distinguish block-observed data from reused child
  fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim target-scale decoder execution, DDR/AXI movement,
  full LLaMA layer/model execution, or board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
