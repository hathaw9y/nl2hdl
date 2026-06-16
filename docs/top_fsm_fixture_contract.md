# Top FSM Fixture Contract

This contract defines the first fixture-level Top FSM HDL sub-agent task after
`layer_fsm_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a fixture-level Top FSM that instantiates and schedules one or more
`layer_fsm_fixture` calls. This proves model-level start/done orchestration for
fixture layers only.

This is not full LLaMA-3.2-1B execution, not real token prefill/decode
scheduling, and not board-level ZCU104 I/O signoff.

## Kernel Name

Use:

- CLI kernel: `top_fsm_fixture`
- HDL module: `top_fsm_fixture`
- report artifact: `top_fsm_fixture_golden.json` plus `kernel_report.json`

## Required Child Module

Instantiate:

- `layer_fsm_fixture`

The child module and its required child SV files may be emitted into the same
output directory for simulation and synthesis.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum top-level outputs:

- final fixture output vector;
- top/layer completion trace;
- stable outputs while `done_o` is high.

## Sequencing Requirements

The first fixture should schedule a configurable or reported number of fixture
layers. A single layer is acceptable for the first gate if the report states
`fixture_layer_count: 1`.

The FSM must:

1. Accept top `start_i`.
2. Start `layer_fsm_fixture`.
3. Wait for child `done_o`.
4. Latch final fixture output/trace.
5. Assert top `done_o` until `start_i` is deasserted.

If multiple fixture layers are scheduled, each layer start/done event must be
recorded in order.

## Evidence Requirements

Generated reports must include:

- `coverage_level: top_fsm_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- child module list with `layer_fsm_fixture` and `instantiated: true`;
- `fixture_layer_count`;
- FSM state order;
- layer start/done trace from simulation;
- explicit `omitted_operations`;
- expected output or final observed fixture vector from the child fixture;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled;
- a caveat if routed timing is internal fixture timing without board I/O delay
  constraints.

## Completion Gate

The milestone passes only when:

- common top handshake is present;
- child module is instantiated;
- child `start_i`/`done_o` sequencing is covered by simulation;
- child inputs are stable while child busy;
- outputs are stable while top `done_o` is high;
- deterministic fixture trace/output passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full token scheduling, DDR streaming, KV-cache,
  full decoder block, full attention, MLP, full LLaMA execution, or board-level
  signoff.

## Omitted Operations

The report must list unsupported full-model operations:

- real token prefill/decode loop;
- multi-layer LLaMA model iteration beyond the reported fixture count;
- DDR packed-weight streaming;
- KV-cache movement;
- attention and softmax;
- residual paths;
- MLP;
- final output layer;
- board-level I/O and shell integration.

## Integration Notes

Passing this gate proves the coding-agent framework can delegate module kernels,
compose a fixture decoder layer, and schedule it from a fixture Top FSM. It does
not complete the final LLaMA-3.2-1B ZCU104 accelerator objective.
