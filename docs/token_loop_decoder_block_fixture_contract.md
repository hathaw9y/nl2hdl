# Token Loop Decoder Block Fixture Contract

This contract defines the next bounded scheduling task after
`top_fsm_decoder_block_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a bounded token-loop fixture that instantiates and repeatedly schedules
`top_fsm_decoder_block_fixture` for a small deterministic token count.

This milestone proves that the framework can add a token-step scheduler above
the refreshed Top FSM path that already includes:

- `top_fsm_decoder_block_fixture`;
- `layer_fsm_decoder_block_fixture`;
- `decoder_block_attention_mlp_fixture`;
- `decoder_child_attention_datapath`;
- `residual_mlp_fixture`;
- RMSNorm/RoPE source path;
- projection internal stream shell;
- attention/KV-cache fixture.

This is not a real LLaMA prefill/decode loop. It does not implement target
sequence scheduling, multi-token KV-cache accumulation, tokenizer interaction,
logits sampling, DDR/AXI memory movement, or board-level ZCU104 signoff.

## Kernel Name

Use:

- CLI kernel: `token_loop_decoder_block_fixture`
- HDL module: `token_loop_decoder_block_fixture`
- report artifact: `token_loop_decoder_block_fixture_golden.json` plus
  `kernel_report.json`

## Required Child Module

Instantiate or generate the refreshed Top FSM RTL into the same output
directory:

- `top_fsm_decoder_block_fixture`

The Top FSM's required nested SV files must also be emitted into the same
output directory for simulation and synthesis:

- `layer_fsm_decoder_block_fixture`
- `decoder_block_attention_mlp_fixture`
- `decoder_child_attention_datapath`
- `residual_mlp_fixture`
- `rmsnorm_rope_source_path`
- `projection_internal_stream_shell`
- `attention_kv_cache_fixture`

Do not replace the refreshed Top FSM with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data should stay compact:

- small registered final fixture output vector;
- compact token/top/layer/block trace or status summary;
- optional narrow selectors for deterministic fixture variants.

Top-level data must not expose:

- full hidden-size vectors;
- residual/MLP child vectors;
- full sequence KV-cache arrays;
- full multi-head tensors;
- 128-bit memory response words as package-level ports;
- full activation, scale, zero-point, or debug trace arrays;
- child-wide status arrays wider than needed for the required trace.

The previous Top FSM decoder-block gate used a 64-bit status and 132 bonded
IOB. The token-loop fixture should avoid exposing extra child package vectors
or wide debug/status paths that could erase the current positive hold margin.

## Sequencing Requirements

The first token-loop decoder-block gate must schedule exactly two deterministic
fixture token steps. Report `fixture_token_count: 2`.

The token-loop FSM must:

1. Accept loop `start_i` while idle.
2. Start `top_fsm_decoder_block_fixture` for token step 0.
3. Hold child `start_i` while that top child is busy.
4. Observe child `done_o`.
5. Capture token 0 output and compact child status.
6. Deassert child `start_i`.
7. Wait for child `done_o` to clear.
8. Start `top_fsm_decoder_block_fixture` for token step 1.
9. Repeat the same hold/done/capture/deassert/release protocol.
10. Latch final fixture output/status and assert loop `done_o`.
11. Keep loop `done_o` asserted until loop `start_i` is deasserted.

The child output may be deterministic and identical for both token steps unless
a new deterministic golden vector is documented. The report must make clear
whether token step outputs are identical fixture outputs or true
token-dependent outputs.

## Dynamic Evidence Requirements

Simulation and report evidence must include:

- token-loop trace with ordered events:
  `token0_start`, `token0_done`, `token1_start`, `token1_done`;
- per-token child start-hold/deassert/release evidence;
- top trace from each child call:
  `layer_fsm_decoder_block_fixture_start`,
  `layer_fsm_decoder_block_fixture_done`;
- layer trace from each child call:
  `decoder_block_attention_mlp_fixture_start`,
  `decoder_block_attention_mlp_fixture_done`;
- decoder-block trace from each child call:
  `attention_start`, `attention_done`, `mlp_start`, `mlp_done`;
- nested attention child trace from at least one child call:
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`;
- nested MLP child trace from at least one child call:
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`;
- per-token output vectors and final output vector;
- evidence that the decoder block consumed attention output into the MLP child,
  either dynamically observed through hierarchy in this token-loop testbench or
  explicitly labeled as reused child fixture metadata;
- proof that loop output/status remain stable while loop `done_o` is high;
- compact I/O evidence and bonded IOB count.

If evidence is not dynamically observed in the token-loop integration run, the
report must not label it as observed token-loop evidence.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: token_loop_decoder_block_fixture`;
- `coverage_level: token_loop_decoder_block_fixture`;
- `token_loop_decoder_block_fixture: true`;
- `uses_top_fsm_decoder_block_fixture: true`;
- `fixture_token_count: 2`;
- child module list with `top_fsm_decoder_block_fixture` and
  `instantiated: true`;
- nested child coverage summary for Top FSM, Layer FSM, decoder block,
  attention datapath, residual/MLP, RMSNorm/RoPE source path, projection shell,
  and attention/KV fixture;
- FSM state order;
- token start/done trace from simulation;
- per-token child start-hold/deassert/release evidence;
- top/layer/block/nested child traces from simulation or clearly labeled reused
  metadata;
- top-level interface summary and exposed port width summary;
- per-token output vectors and final output vector;
- decoder-block attention-to-MLP consumption evidence and source label;
- whether token outputs are token-dependent or repeated deterministic fixture
  outputs;
- timing margin note that prior Top FSM decoder-block WHS was 0.018 ns and
  status was 64 bits;
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

- real LLaMA token prefill/decode semantics;
- tokenizer or embedding lookup;
- logits generation and sampling;
- target sequence scheduling;
- target multi-layer LLaMA iteration;
- token-dependent KV-cache accumulation across full sequence length;
- DDR/AXI packed-weight streaming;
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
- The refreshed Top FSM module is instantiated or equivalently generated as
  real child RTL, not replaced by counters only.
- The token-loop start/done sequence is dynamically observed and checked for
  both token steps.
- The child `start_i` is held while the child is busy, deasserted after child
  `done_o`, and followed by an explicit release wait for both token steps.
- Outputs are stable while loop `done_o` is high.
- Top-level I/O does not expose additional child package vectors or wide debug
  arrays.
- Report evidence labels distinguish token-loop-observed data from reused
  child fixture metadata.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- Report text does not claim real prefill/decode, token-dependent KV-cache
  accumulation, target sequence scheduling, DDR/AXI movement, full LLaMA
  execution, or board-level signoff.
- A read-only verification agent finds no P0/P1 issues.
