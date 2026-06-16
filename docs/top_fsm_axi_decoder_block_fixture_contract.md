# Top FSM AXI Decoder Block Fixture Contract

This contract defines the bounded Top FSM integration fixture after
`layer_fsm_axi_decoder_block_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a fixture-level Top FSM that instantiates and schedules the verified
AXI decoder-block Layer FSM fixture:

- `layer_fsm_axi_decoder_block_fixture`

This milestone proves that the Top FSM integration point can call a Layer FSM
child containing an AXI-aware decoder block with q/k/v/o projection fixtures,
attention/KV fixture logic, and residual/MLP fixture logic. It does not prove
real token prefill/decode, target multi-layer scheduling, DDR/AXI movement,
real GPTQ checkpoint streaming, full LLaMA execution, or board-level signoff.

## Kernel Name

Use:

- CLI kernel: `top_fsm_axi_decoder_block_fixture`
- HDL module: `top_fsm_axi_decoder_block_fixture`
- report artifact: `top_fsm_axi_decoder_block_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate or generate real child RTL into the same output directory:

- `layer_fsm_axi_decoder_block_fixture`

The layer child also requires nested RTL files:

- `decoder_block_axi_attention_mlp_fixture`
- `decoder_child_axi_attention_datapath`
- `residual_mlp_fixture`
- `rmsnorm_rope_source_path`
- `projection_axi_stream_integration`
- `attention_kv_cache_fixture`

Do not replace the Layer FSM child with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data must stay compact:

- small registered final top fixture output vector;
- compact top/layer/block trace or status summary;
- compact q/k/v/o AXI metadata-good summary propagated from the Layer FSM.

Top-level data must not expose:

- 128-bit AXI read data words;
- AXI address/response/debug buses;
- layer or decoder-block child full hidden/intermediate vectors;
- attention child KV/debug/status arrays;
- residual/MLP child hidden or attention package input vectors;
- full sequence KV-cache arrays;
- child-wide status arrays wider than needed for the required trace.

The previous AXI Layer FSM gate used a 64-bit status and 132 bonded IOB. This
Top FSM should keep a similarly compact top-level interface and avoid
regressing to wide child status exposure.

## Sequencing Requirements

The Top FSM must:

1. Accept top `start_i` while idle.
2. Start `layer_fsm_axi_decoder_block_fixture`.
3. Hold layer child `start_i` while busy.
4. Observe layer child `done_o`.
5. Capture the layer final output and compact child status/trace.
6. Deassert layer child `start_i`.
7. Wait for layer child `done_o` to clear.
8. Assert top `done_o` and hold it until top `start_i` deasserts.

This contract requires exactly one AXI Layer FSM child call for the first Top
FSM decoder-block fixture. Token-loop scheduling and target multi-layer
iteration remain separate future milestones.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- ordered top trace:
  `layer_fsm_axi_decoder_block_fixture_start`,
  `layer_fsm_axi_decoder_block_fixture_done`;
- layer trace:
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
- top-to-layer child start-hold/deassert/release evidence;
- q/k/v/o AXI projection metadata-good bits visible through the top compact
  status;
- q/k/v/o AXI projection child evidence preserved separately, not collapsed to
  only q;
- final top output vector and layer child output vector;
- evidence that the decoder block consumed attention output into the MLP child,
  either dynamically observed through hierarchy or clearly labeled as reused
  child fixture metadata;
- proof that top output/status remain stable while top `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the Top FSM integration run, the
report must not label it as observed top-run evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: top_fsm_axi_decoder_block_fixture`;
- `coverage_level: top_fsm_axi_decoder_block_fixture`;
- `top_fsm_axi_decoder_block_fixture: true`;
- child module list with `layer_fsm_axi_decoder_block_fixture` and
  `instantiated: true`;
- nested layer/decoder-block child coverage summary;
- FSM state order;
- ordered top trace from simulation;
- layer, decoder-block, AXI attention, and MLP traces from simulation or
  clearly labeled reused metadata;
- top-to-layer child start-hold/deassert/release evidence;
- compact q/k/v/o AXI metadata propagation evidence;
- per-projection q/k/v/o AXI child evidence summary;
- final top output vector and layer child output vector;
- compact I/O and package I/O mitigation summary;
- timing margin note that prior AXI Layer FSM WHS was 0.012 ns and status was
  64 bits;
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
- real LLaMA token prefill/decode semantics;
- target sequence scheduling;
- target multi-layer LLaMA iteration;
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
- The AXI Layer FSM child module is instantiated as real RTL.
- Common handshake contract is met for the top and child.
- Child `start_i` is held while busy, deasserted after `done_o`, and followed
  by a release wait.
- Ordered top/layer/block/nested traces are checked.
- q/k/v/o AXI metadata bits remain visible at the Top FSM level.
- Numeric final output matches the verified AXI Layer FSM fixture vector.
- Outputs are stable while top `done_o` is high.
- Top-level I/O does not expose additional child package vectors or wide debug
  arrays.
- Report evidence labels distinguish top-observed data from reused child
  fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim real token scheduling, target multi-layer LLaMA
  iteration, DDR/AXI movement, full LLaMA layer/model execution, or board-level
  signoff.
- A read-only verification agent finds no P0/P1 issues.
