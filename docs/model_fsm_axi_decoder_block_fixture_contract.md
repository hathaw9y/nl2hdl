# Model FSM AXI Decoder Block Fixture Contract

This contract defines the bounded model-level scheduling task after
`token_loop_axi_decoder_block_fixture`.

The parent agent owns this contract and orchestration. HDL sub-agents implement
RTL or RTL generator changes. The parent agent must not hand-write generated
HDL/SystemVerilog.

## Scope

Create a bounded model-level FSM fixture that instantiates and repeatedly
schedules `token_loop_axi_decoder_block_fixture` for a small deterministic
layer count.

This milestone proves that the framework can add a model-level scheduler above
the AXI decoder-block token loop that already includes:

- `token_loop_axi_decoder_block_fixture`;
- `top_fsm_axi_decoder_block_fixture`;
- `layer_fsm_axi_decoder_block_fixture`;
- `decoder_block_axi_attention_mlp_fixture`;
- `decoder_child_axi_attention_datapath`;
- `residual_mlp_fixture`;
- RMSNorm/RoPE source path;
- AXI projection stream integration for q/k/v/o projections;
- attention/KV-cache fixture.

This is not full LLaMA-3.2-1B execution. It does not implement 16 target
decoder layers, target sequence scheduling, real token-dependent KV-cache
accumulation, logits/sampling, DDR/AXI shell integration, or board-level ZCU104
signoff.

## Kernel Name

Use:

- CLI kernel: `model_fsm_axi_decoder_block_fixture`
- HDL module: `model_fsm_axi_decoder_block_fixture`
- report artifact: `model_fsm_axi_decoder_block_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Module

Instantiate or generate the AXI decoder-block token-loop RTL into the same
output directory:

- `token_loop_axi_decoder_block_fixture`

The token-loop child also requires nested RTL files:

- `top_fsm_axi_decoder_block_fixture`
- `layer_fsm_axi_decoder_block_fixture`
- `decoder_block_axi_attention_mlp_fixture`
- `decoder_child_axi_attention_datapath`
- `residual_mlp_fixture`
- `rmsnorm_rope_source_path`
- `projection_axi_stream_integration`
- `attention_kv_cache_fixture`

Do not replace the token-loop child with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- small registered final fixture output vector;
- compact model/layer/token-loop trace or status summary;
- compact q/k/v/o AXI projection metadata summary propagated from the child;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- full hidden-size vectors;
- full layer activation buffers;
- full sequence KV-cache arrays;
- full multi-head tensors;
- 128-bit AXI memory response words as package-level ports;
- full activation, scale, zero-point, or debug trace arrays;
- child-wide status arrays wider than needed for the required trace.

The previous AXI decoder-block token-loop gate used a 64-bit status, 132 bonded
IOB, WNS 1.348 ns, WHS 0.009 ns, and WPWS 2.225 ns. This model FSM fixture
must avoid exposing extra child package vectors or wide combinational
debug/status paths that could erase this thin positive hold margin.

## Sequencing Requirements

The first model-level gate must schedule exactly two deterministic layer steps.
Report `fixture_layer_count: 2` and `fixture_token_count_per_layer: 2`.

The model FSM must:

1. Accept model `start_i` while idle.
2. Start `token_loop_axi_decoder_block_fixture` for layer step 0.
3. Hold child `start_i` while that token-loop child is busy.
4. Observe child `done_o`.
5. Capture layer 0 output and compact child status.
6. Deassert child `start_i`.
7. Wait for child `done_o` to clear.
8. Start `token_loop_axi_decoder_block_fixture` for layer step 1.
9. Repeat the same hold/done/capture/deassert/release protocol.
10. Latch final fixture output/status and assert model `done_o`.
11. Keep model `done_o` asserted until model `start_i` is deasserted.

The child output may be deterministic and identical for both layer steps unless
a new deterministic golden vector is documented. The expected reused
deterministic output vector is `[12, -6, 18, 6]` per layer until a new golden
vector is documented. The report must clearly state that this is a repeated
bounded fixture output, not real layer-dependent LLaMA output.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- model-level trace with ordered events:
  `layer0_start`, `layer0_done`, `layer1_start`, `layer1_done`;
- per-layer child start-hold/deassert/release evidence;
- token-loop trace from each child call:
  `token0_start`, `token0_done`, `token1_start`, `token1_done`;
- top/layer/decoder-block/nested child traces from the token-loop child or
  clearly labeled reused child metadata;
- q/k/v/o AXI projection metadata propagation through model, token-loop, top,
  layer, and decoder-block compact status;
- q/k/v/o AXI projection child evidence preserved separately, not collapsed to
  only q;
- per-layer output vectors and final output vector;
- evidence that the decoder block consumed attention output into the MLP child,
  either dynamically observed through hierarchy in this model FSM testbench or
  explicitly labeled as reused child fixture metadata;
- proof that model output/status remain stable while model `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the model-level integration run, the
report must not label it as observed model-level evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: model_fsm_axi_decoder_block_fixture`;
- `coverage_level: model_fsm_axi_decoder_block_fixture`;
- `model_fsm_axi_decoder_block_fixture: true`;
- `uses_token_loop_axi_decoder_block_fixture: true`;
- `fixture_layer_count: 2`;
- `fixture_token_count_per_layer: 2`;
- child module list with `token_loop_axi_decoder_block_fixture` and
  `instantiated: true`;
- nested child coverage summary for token loop, Top FSM, Layer FSM, decoder
  block, attention datapath, residual/MLP, RMSNorm/RoPE source path, AXI
  projection stream integration, and attention/KV fixture;
- FSM state order;
- model layer start/done trace from simulation;
- per-layer child start-hold/deassert/release evidence;
- token-loop/top/layer/block/nested child traces from simulation or clearly
  labeled reused metadata;
- q/k/v/o AXI metadata propagation evidence;
- top-level interface summary and exposed port width summary;
- per-layer output vectors and final output vector;
- decoder-block attention-to-MLP consumption evidence and source label;
- whether layer outputs are layer-dependent or repeated deterministic fixture
  outputs;
- timing margin note that prior AXI decoder-block token-loop WHS was 0.009 ns
  and status was 64 bits;
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
- real LLaMA token prefill/decode semantics;
- tokenizer or embedding lookup;
- logits generation and sampling;
- target sequence scheduling;
- target 16-layer LLaMA iteration;
- target multi-layer LLaMA numerical execution;
- layer-dependent KV-cache accumulation across full sequence length;
- DDR/AXI packed-weight streaming and full DDR controller integration;
- DDR/AXI KV-cache movement;
- board shell and PS/PL integration;
- mathematically complete Q/K/V/O projection-to-attention wiring;
- full sequence-length KV-cache;
- multi-head attention;
- grouped-query attention;
- true exponential softmax;
- target-scale MLP dimensions;
- GPTQ INT4 packed gate/up/down streaming;
- true SiLU or exponential activation math in RTL;
- full LLaMA layer execution;
- full LLaMA model execution;
- board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- The AXI decoder-block token-loop module is instantiated or equivalently
  generated as real child RTL, not replaced by counters only.
- The model-level layer start/done sequence is dynamically observed and checked
  for both layer steps.
- The child `start_i` is held while the child is busy, deasserted after child
  `done_o`, and followed by an explicit release wait for both layer steps.
- Outputs are stable while model `done_o` is high.
- Top-level I/O does not expose additional child package vectors, 128-bit AXI
  data buses, or wide debug arrays.
- Report evidence labels distinguish model-level observed data from reused
  child fixture metadata.
- Verilator simulation or lint passes when enabled.
- Vivado post-route timing reports nonnegative setup, hold, and pulse-width
  slack with zero failing endpoints when synthesis is enabled.
- A read-only verification agent reports no P0/P1 findings.
