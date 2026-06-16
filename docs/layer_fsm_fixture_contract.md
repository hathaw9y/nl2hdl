# Layer FSM Fixture Contract

This contract defines the first fixture-level Layer FSM HDL sub-agent task after
`decoder_child_datapath`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a fixture-level Layer FSM that instantiates and schedules the routed
`decoder_child_datapath` fixture as one decoder-layer/block unit.

This is not a full LLaMA layer, not a multi-layer model, and not a Top FSM. It
proves that a higher-level FSM can call a gated decoder-block fixture with the
project's common start/done protocol.

## Kernel Name

Use:

- CLI kernel: `layer_fsm_fixture`
- HDL module: `layer_fsm_fixture`
- report artifact: `layer_fsm_fixture_golden.json` plus `kernel_report.json`

## Required Child Module

Instantiate:

- `decoder_child_datapath`

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
- layer/block completion trace;
- stable outputs while `done_o` is high.

## Sequencing Requirements

The FSM must:

1. Accept top `start_i`.
2. Start `decoder_child_datapath`.
3. Wait for child `done_o`.
4. Latch final fixture output/trace.
5. Assert top `done_o` until `start_i` is deasserted.

Child inputs must remain stable while the child is busy.

## Evidence Requirements

Generated reports must include:

- `coverage_level: layer_fsm_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- child module list with `decoder_child_datapath` and `instantiated: true`;
- FSM state order;
- child start/done trace from simulation;
- explicit `omitted_operations`;
- expected output or final observed fixture vector from the child fixture;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled.

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
- report text does not claim Top FSM, full token scheduling, full decoder block,
  full attention, KV-cache, MLP, or full LLaMA coverage.

## Omitted Operations

The report must list unsupported full-model operations:

- multi-layer iteration;
- token prefill/decode loop;
- DDR packed-weight streaming;
- KV-cache movement;
- attention and softmax;
- residual paths;
- MLP;
- final output layer;
- full Top FSM scheduling.

## Integration Notes

Passing this gate can unlock a fixture-level Top FSM contract only if the next
task clearly reports that it schedules fixture layers, not the full LLaMA model.
