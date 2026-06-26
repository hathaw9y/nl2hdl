---
name: hdl-kernel-contract-gates
description: Use when generating or auditing FPGA Verilog/SystemVerilog kernels that must be composable by Layer FSM or Top FSM agents. Applies when a kernel simulates but lacks the common handshake, lacks clock constraints, records only partial verification evidence, or is a placeholder that is too toy-like for the target LLM milestone.
---

# HDL Kernel Contract Gates

Use this skill before declaring an HDL kernel milestone complete.

## Completion Gates

A kernel is not integration-ready unless all are true:

- Ports follow `docs/hdl_module_interface_contract.md`, including `aclk`,
  `aresetn`, `start_i`, and `done_o`.
- Outputs are registered or otherwise stable when `done_o` is asserted.
- Packed vectors use little element order: element `idx` is `[idx*WIDTH +: WIDTH]`.
- The generated Vivado TCL creates a real clock constraint on the kernel clock.
- Vivado timing has valid non-`NA` setup, hold, and pulse-width checks.
- `kernel_report.json` records simulation and, when enabled, Verilator evidence.
- Verilator failures are not hidden by an iverilog-only pass. For lint-only
  testbench checks, pass an explicit timing policy such as `--timing` when the
  testbench uses delays or event controls.
- DUT-only Verilator lint should either be warning-clean or report explicitly
  which warning classes are allowed and why.
- Placeholder math is clearly marked as a scaffold and cannot be reported as a
  LLaMA kernel milestone pass.
- `kernel_report.json` states whether the kernel is a synthetic fixture,
  projection-sized tile, or target implementation.
- `kernel_report.json` states the Vivado implementation stage used for timing,
  such as post-synth, post-place, or post-route.
- For packed INT4/GPTQ kernels, round-trip evidence should be explicit in the
  report. Dequant output checks alone can hide a packing/unpacking bug.
- For target-scope GPTQ projection fixtures, `kernel_report.json` must also
  explicitly record that real checkpoint layout preflight is blocked when no
  real GPTQ metadata was proven. Include fields such as
  `gptq_layout_preflight`, `real_gptq_checkpoint_layout_compatible: false`,
  and `real_checkpoint_layout_compatibility` in `does_not_claim`; absence of
  these fields is a verification-blocking evidence gap even when simulation,
  Verilator, and Vivado pass.
- If a report includes configured parallelism such as `PE_LANES`, distinguish
  requested lanes, effective fixture lanes, and true parallel datapath lanes.
  Scheduling metadata is not the same as full throughput scaling.
- For RMSNorm kernels, distinguish an apply fixture with externally supplied
  `inv_rms` from a full RMSNorm datapath with RTL reduction and reciprocal
  square root or lookup.
- For RoPE kernels, distinguish an apply fixture with externally supplied
  cos/sin metadata from full RoPE frequency generation or table lookup in RTL.
- For decoder block scaffolds, distinguish start/done sequencing traces from
  actual child datapath instantiation and full attention/KV-cache behavior.
- For fixture-level decoder child datapaths, distinguish real child module
  instantiation from full decoder-block coverage. Instantiating RMSNorm,
  projection, and RoPE fixtures is still not attention, KV-cache, residual, MLP,
  or full token scheduling.
- For fixture-level Layer FSMs, distinguish scheduling a decoder fixture from
  Top FSM work, multi-layer iteration, token prefill/decode loops, DDR
  streaming, and full LLaMA scheduling.
- For packed-weight streaming fixtures, distinguish stream-style valid/ready
  payload consumption from real DDR, AXI, board shell, or full memory subsystem
  integration. Report configured memory width separately from effective fixture
  stream width.
- For parallel projection fixtures, distinguish true same-stage arithmetic lanes
  from sequential lane-index scheduling. A report should name the parallel stage
  and products per cycle.
- For OOC resource tuning, prove the tuned knob changes true RTL structure.
  If `pe_count`, tile lanes, or buffering depth changes but Vivado LUT/FF/DSP/
  BRAM/URAM/IO utilization is unchanged, report the tuning as ineffective and
  stop repeating the same knob until the generator connects it to real datapath
  parallelism or memory allocation.
- For target-scale child packets, `target_scale_child_eligible: true` requires
  downstream-consumable payload contracts, not only target-dimension checksums,
  summary words, counters, or scheduling traces. For
  `target_non_gemm_datapath_packets`, require
  `interface_contract.tensor_payload_streams_present: true` plus per-kernel
  payload stream fields for RMSNorm, RoPE, softmax/control, KV-cache, residual,
  and SwiGLU before reporting a pass.

## Known Failure Patterns

- `int4_unpack`, `rmsnorm`, and `rope` generated combinational modules without
  `aclk`, `aresetn`, `start_i`, or `done_o`. They simulated, but cannot be
  called by a Layer FSM.
- Vivado timing reports with `There are no user specified timing constraints`
  and WNS/WHS/WPWS as `NA` are invalid evidence, not a pass.
- Direct Verilator lint may pass while `kernel_report.json` omits Verilator
  evidence. If config enables Verilator, record it.
- A projection kernel can fix post-place hold timing and still fail the
  integration-ready gate if Verilator evidence is missing or the report does
  not distinguish synthetic fixture coverage from target LLaMA tile coverage.
- A GPTQ dequant fixture can pass numeric output checks while only implicitly
  proving packed nibble round-trip. Before projection tile expansion, add an
  explicit `round_trip_passed` or equivalent field.
- A projection fixture can be technically correct and still fail wave
  verification if older `kernel_report.json` artifacts omit explicit blocked
  GPTQ checkpoint-layout evidence. Regenerate stale per-projection evidence
  after adding target-scope report fields; do not advance Layer FSM or Top FSM
  integration from mixed old/new reports.
- A projection tile fixture may exercise `PE_LANES` through scheduling while
  remaining mostly sequential. Do not report this as full PE-lane scaling unless
  the datapath actually performs that many parallel lane operations.
- An RMSNorm apply fixture can be composable and useful, but it must report
  `inv_rms_source` and `reciprocal_sqrt_in_rtl: false` when reciprocal square
  root is not implemented in RTL.
- A RoPE apply fixture can be composable and useful, but it must report
  `cos_sin_source`, `frequency_generation_in_rtl`, and `lookup_table_in_rtl`.
- A decoder block sequencing scaffold can prove FSM ordering, but it must report
  `datapath_child_instantiation: false` when child kernels are not instantiated
  and list omitted operations such as attention, KV-cache, residuals, and MLP.
- A decoder child-datapath fixture can prove child `start_i`/`done_o` wiring, but
  it must report `coverage_level: decoder_child_datapath_fixture` and continue
  listing omitted full-decoder operations.
- A Layer FSM fixture can prove one layer/block call sequence, but it must report
  `coverage_level: layer_fsm_fixture` and explicitly avoid claims about Top FSM,
  token scheduling, or full model execution.
- A projection streaming fixture can prove packed INT4 weight stream
  consumption, but it must report `coverage_level: projection_streaming_fixture`
  and continue listing DDR/AXI/board signoff as omitted or not claimed.
- A projection parallel-streaming fixture can prove more than one arithmetic
  lane, but it must report `true_parallel_datapath_lanes`,
  `parallel_products_per_cycle`, and the stage where the lanes are active.
- A target-dimension non-GEMM RTL prototype can pass simulation and OOC
  synthesis while still being integration-blocked if it emits only checksum or
  summary streams. Do not mark it as a target-scale child pass until the report
  proves decoder-consumable tensor payload streams and per-kernel payload
  contracts, including RMSNorm apply output, RoPE payload output, softmax
  weight/payload output, KV payload/address contract, residual payload output,
  and SwiGLU payload output.
- Parent OOC auto-tuning can double a requested `pe_count` while resource
  utilization remains unchanged if the projection generator only records the
  requested value as metadata. Treat this as a generator/contract gap: compare
  requested lanes, true parallel datapath lanes, and Vivado resource deltas
  before recommending another `pe_count` retry.
- Toy RMSNorm or RoPE kernels should not be treated as target LLaMA kernels
  unless the report explicitly labels them as fixtures.
- Integration fixtures that expose wide deterministic metadata, such as full
  activation vectors, scale tables, or zero-point tables, as top-level ports can
  fail Vivado placement because the fixture creates too much unconstrained
  package I/O. Keep deterministic fixture metadata inside the RTL as constants
  or feed it through a small validated stream, and reserve top-level ports for
  the interface being proven.
- Wide debug traces exposed as top-level ports can cause the same Vivado I/O
  placement failure even when the datapath is small. Keep detailed traces
  internal for the simulation harness, expose only the narrow link being proven
  plus compact summary/debug words, and record the full observed evidence in
  `kernel_report.json`.
- Some SystemVerilog tools reject numeric literals with parameterized widths,
  such as `MEM_WORD_WIDTH'h...`. Generators should emit concrete sized
  literals such as `128'h...` when the width is known, or use an explicit cast.
- Request-backpressure tests should verify field stability while valid is high
  and ready is low. Avoid requiring ready-low to persist for an exact number of
  cycles after a handshake may already have completed.
- AXI read-command adapter fixtures must lock the public AXI boundary exactly
  to the contract, including an 8-bit `axi_arid_o` when specified. Tests should
  reject narrower parameterized IDs if the contract names a fixed width.
- AXI read-command adapter reports must always include explicit target-vs-fixture
  planning fields, even when the real checkpoint stream plan is unavailable:
  `checkpoint_target_weight_stream_plan`, `checkpoint_target_request_summary`,
  `fixture_axi_command_execution`, target planned beat count, fixture executed
  beat count, and split-required status. Do not silently omit unavailable plan
  metadata; record it as invalid/unavailable and keep the bounded fixture
  execution separate.
- AXI read-data or read-transaction fixtures must validate R-channel metadata in
  the DUT, not only in the testbench. If a contract says the fixture requires a
  matching `rid`, OKAY `rresp`, or final-only `rlast`, the generated RTL should
  record or gate errors for those fields, expose compact validation status in
  the report, and include a negative regression that injects bad R metadata.
  A passing good-path testbench that merely drives correct values is incomplete
  evidence for this contract.
- AXI read-data adapter negative metadata regression is not complete unless it
  separately proves bad `rid`, bad `rresp`, early `rlast`, and missing final
  `rlast`. The `kernel_report.json` must record per-case DUT-observed error
  evidence; a single good-path `id_resp_last_validation` field is not enough to
  advance the wave.
- Some SystemVerilog tools reject indexing a function call result inside a
  replication, sign-extension, or part-select expression, such as
  `foo()[WIDTH-1]`. Store the function result in a temporary variable first,
  then index that variable.
- Fixture status and debug top-level ports should expose only the trace bits
  needed by the contract. Oversized status ports can quietly increase bonded
  IOB use even when the datapath is small.
- Even logically compact fixture vector ports can consume most package I/O
  once multiple 64-bit inputs, 64-bit outputs, and status summaries are exposed
  together. Before composing two passed fixtures, prefer internal fixture
  constants, internal child wiring, or a narrow stream/control interface over
  lifting every child vector port to the new top level. Treat bonded IOB above
  roughly 80% as an integration risk even if the standalone fixture routes.
- Positive hold slack of only a few picoseconds, such as WHS around 0.005 ns,
  is a pass for a bounded fixture if the timing report has zero failing
  endpoints, but it is a scaling risk. The next integration contract should
  record it as a residual risk and avoid adding wide combinational status or
  debug paths that could erase the margin.
- Timing-margin recovery that improves a very small WHS, for example from
  0.002 ns to around 0.008 ns, can pass a bounded fixture gate when setup,
  hold, and pulse-width all have zero failing endpoints. Still track it as P3
  scaling risk before adding tokens, layers, DDR/AXI, or board-shell logic, and
  keep status/debug width from growing again.
- When reducing top-level I/O for parent FSM fixtures, it is acceptable to keep
  detailed child traces visible only through simulation hierarchy while exposing
  a narrow parent status word, provided the testbench dynamically checks the
  child hierarchy and the report clearly distinguishes top-level compact status
  from observed hierarchical evidence.
- When a parent FSM sequences child fixtures whose `done_o` remains asserted
  until local `start_i` deasserts, hold each child `start_i` while that child
  is busy, deassert it immediately after observing `done_o`, and only then
  start the next child. Counter-only child sequencing does not prove the common
  handshake contract.
- When a parent FSM fixture claims child start-hold/deassert/release evidence,
  it must dynamically prove the release, not only that the parent deasserted
  child `start_i` after seeing `done_o`. Add a release-wait state or equivalent
  guard that observes child `done_o` low after `start_i` drops, and record a
  report/test field such as `child_done_release_seen_after_start_deassert`.
  This applies even for a single child call, because the next parent FSM level
  may rely on the child being reusable before the parent asserts its own
  completion.
- For repeated calls to the same child fixture, add an explicit release state
  that waits for the child's `done_o` to clear after the parent deasserts the
  child's `start_i`. Starting the next call while the previous `done_o` is
  still high can falsely count a stale done as the next operation's completion.
- Integration testbenches must not print hard-coded child evidence while the
  report labels it as dynamically observed integration evidence. If a parent
  report records child KV/cache, stream, payload, or output traces, the
  integration testbench must read those values from child ports or hierarchical
  child signals and compare them, or the report must explicitly label the data
  as reused static child fixture metadata rather than observed integration-run
  evidence.
- Apply the same parent-FSM evidence rule recursively through Layer FSM and Top
  FSM fixtures. A Top FSM report may call layer and nested-child traces
  observed only when the Top FSM integration testbench reads the layer or child
  hierarchy and checks the values; otherwise label them as reused fixture
  metadata.
- When an integration contract requires child validation status to propagate
  into a parent compact status word, do not only observe the child hierarchy in
  the testbench. The parent RTL must latch or encode the required child status
  bits, such as AXI RID/RRESP/RLAST good-path metadata, into its own compact
  status. The testbench and report should check both the child hierarchy and the
  parent compact status bit positions so a stale low byte or truncated status
  field cannot pass as propagated evidence.

## Sub-Agent Prompt Rule

When assigning a module to an HDL sub-agent, include:

- required common handshake ports;
- exact clock constraint requirement;
- simulation, Verilator, and Vivado report requirements;
- whether the kernel is a scaffold fixture or a target implementation;
- final evidence fields to inspect in `kernel_report.json`.

## Retry Policy

If an HDL sub-agent returns a sim-only pass:

1. Treat the attempt as incomplete.
2. Preserve generated artifacts under `build/`.
3. Ask for a contract-compliant retry before spawning Layer FSM or Top FSM work.
