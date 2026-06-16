# Layer FSM AXI Decoder Block Fixture Contract

This contract defines the bounded Layer FSM integration fixture after
`decoder_block_axi_attention_mlp_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a fixture-level Layer FSM that instantiates and schedules the verified
AXI-aware decoder-block fixture:

- `decoder_block_axi_attention_mlp_fixture`

This milestone proves that the Layer FSM integration point can call a
decoder-block child containing AXI-backed q/k/v/o projection fixtures,
attention/KV fixture logic, and residual/MLP fixture logic. It does not prove
target multi-layer LLaMA iteration, token scheduling, DDR/AXI movement, real
GPTQ checkpoint streaming, or board-level signoff.

## Kernel Name

Use:

- CLI kernel: `layer_fsm_axi_decoder_block_fixture`
- HDL module: `layer_fsm_axi_decoder_block_fixture`
- report artifact: `layer_fsm_axi_decoder_block_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate real child RTL into the same output directory:

- `decoder_block_axi_attention_mlp_fixture`

The decoder block also requires nested RTL files:

- `decoder_child_axi_attention_datapath`
- `residual_mlp_fixture`
- `rmsnorm_rope_source_path`
- `projection_axi_stream_integration`
- `attention_kv_cache_fixture`

Do not replace the decoder-block child with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data must stay compact:

- small registered final layer fixture output vector;
- compact layer/block/child trace or status summary;
- compact q/k/v/o AXI metadata-good summary propagated from the decoder block.

Top-level data must not expose:

- 128-bit AXI read data words;
- AXI address/response/debug buses;
- decoder-block child full hidden/intermediate vectors;
- attention child KV/debug/status arrays;
- residual/MLP child hidden or attention package input vectors;
- full sequence KV-cache arrays;
- child-wide status arrays wider than needed for the required trace.

## Sequencing Requirements

The Layer FSM must:

1. Accept layer `start_i` while idle.
2. Start `decoder_block_axi_attention_mlp_fixture`.
3. Hold child `start_i` while busy.
4. Observe child `done_o`.
5. Capture the decoder-block final output and compact child status/trace.
6. Deassert child `start_i`.
7. Wait for child `done_o` to clear.
8. Assert layer `done_o` and hold it until layer `start_i` deasserts.

This contract requires exactly one decoder-block child call for the first AXI
Layer FSM decoder-block fixture.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- ordered layer trace:
  `decoder_block_axi_attention_mlp_fixture_start`,
  `decoder_block_axi_attention_mlp_fixture_done`;
- decoder-block trace:
  `axi_attention_start`, `axi_attention_done`, `mlp_start`, `mlp_done`;
- nested AXI attention child trace:
  `source_path_start`, `source_path_done`, `projection_axi_start`,
  `projection_axi_done`, `attention_kv_start`, `attention_kv_done`;
- nested MLP child trace:
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`;
- layer-to-block child start-hold/deassert/release evidence;
- q/k/v/o AXI projection metadata-good bits visible through the layer compact
  status;
- q/k/v/o AXI projection child evidence preserved separately, not collapsed to
  only q;
- final layer output vector and decoder-block output vector;
- evidence that the decoder block consumed attention output into the MLP child,
  either dynamically observed through hierarchy or clearly labeled as reused
  child fixture metadata;
- proof that layer output/status remain stable while layer `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the Layer FSM integration run, the
report must not label it as observed layer-run evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: layer_fsm_axi_decoder_block_fixture`;
- `coverage_level: layer_fsm_axi_decoder_block_fixture`;
- `layer_fsm_axi_decoder_block_fixture: true`;
- child module list with `decoder_block_axi_attention_mlp_fixture` and
  `instantiated: true`;
- nested decoder-block child coverage summary;
- FSM state order;
- ordered layer trace from simulation;
- decoder-block, AXI attention, and MLP traces from simulation or clearly
  labeled reused metadata;
- layer-to-block child start-hold/deassert/release evidence;
- compact q/k/v/o AXI metadata propagation evidence;
- per-projection q/k/v/o AXI child evidence summary;
- final layer output vector and decoder-block child output vector;
- compact I/O and package I/O mitigation summary;
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

- real GPTQ checkpoint payload streaming;
- DDR controller integration;
- multi-burst or outstanding AXI master;
- target multi-layer LLaMA iteration;
- real LLaMA token prefill/decode semantics;
- target sequence scheduling;
- tokenizer or embedding lookup;
- logits generation and sampling;
- target-scale decoder block dimensions;
- complete Q/K/V/O attention math;
- full sequence-length KV-cache;
- multi-head attention and grouped-query attention;
- true exponential softmax;
- target-scale MLP dimensions;
- GPTQ INT4 packed gate/up/down streaming;
- DDR/AXI packed-weight movement;
- DDR/AXI KV-cache movement;
- true SiLU or exponential activation math in RTL;
- full LLaMA layer execution;
- full LLaMA model execution;
- board shell and PS/PL integration;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- The AXI decoder-block child module is instantiated as real RTL.
- Common handshake contract is met for the layer and child.
- Child `start_i` is held while busy, deasserted after `done_o`, and followed
  by a release wait.
- Ordered layer/block/nested traces are checked.
- q/k/v/o AXI metadata bits remain visible at the Layer FSM level.
- Numeric final output matches the verified decoder-block fixture vector.
- Outputs are stable while layer `done_o` is high.
- Top-level I/O does not expose additional child package vectors or wide debug
  arrays.
- Report evidence labels distinguish layer-observed data from reused child
  fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim target multi-layer LLaMA iteration, token
  scheduling, DDR/AXI movement, full LLaMA layer/model execution, or
  board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
