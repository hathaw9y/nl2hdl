# Decoder Child-Datapath Scaffold Contract

This contract defines the next integration HDL sub-agent task after the
`decoder_block_scaffold` sequencing trace.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a fixture-level decoder-block scaffold that actually instantiates and
wires gated child modules with `start_i`/`done_o` handshakes.

This is still not full LLaMA-3.2-1B decoder-block coverage. It proves child
datapath composition for a small fixture and must keep omitted full-model
behavior explicit.

## Kernel Name

Use:

- CLI kernel: `decoder_child_datapath`
- HDL module: `decoder_child_datapath`
- report artifact: `decoder_child_datapath_golden.json` plus
  `kernel_report.json`

## Required Child Modules

Instantiate and sequence these generated fixture kernels:

- `rmsnorm_target`
- `projection_tile`
- `rope_target`

The child modules may be emitted into the same output directory for simulation
and synthesis. Do not model them with counters only.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum top-level outputs:

- final fixture output vector or trace vector;
- child completion trace;
- stable outputs while `done_o` is high.

## Sequencing Requirements

The FSM must:

1. Start `rmsnorm_target`.
2. Wait for `rmsnorm_target.done_o`.
3. Start `projection_tile`.
4. Wait for `projection_tile.done_o`.
5. Start `rope_target`.
6. Wait for `rope_target.done_o`.
7. Latch final fixture output/trace and assert top `done_o`.

Child `start_i` pulses or levels must follow each child module contract. Child
inputs must remain stable while the child is busy.

## Evidence Requirements

Generated reports must include:

- `coverage_level: decoder_child_datapath_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- child module list, child coverage levels, and `instantiated: true`;
- FSM state order;
- child start/done trace from simulation;
- explicit `omitted_operations`;
- expected output or final observed fixture vector from a Python/NumPy golden
  reference or deterministic child fixture trace;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled.

## Completion Gate

The milestone passes only when:

- common top handshake is present;
- child modules are instantiated;
- child `start_i`/`done_o` sequencing is covered by simulation;
- child inputs are stable while child busy;
- outputs are stable while top `done_o` is high;
- Python/NumPy golden comparison or deterministic fixture trace passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full decoder-block, full attention, KV-cache, MLP,
  or full LLaMA coverage.

## Omitted Operations

The report must still list unsupported full decoder operations:

- K/V/O projections;
- attention score and softmax;
- KV-cache movement;
- residual paths;
- MLP up/gate/down projections;
- final residual and output layer;
- full token scheduling.

## Integration Notes

Passing this gate can unlock a fixture-level Layer FSM agent. It does not unlock
Top FSM or full model scheduling.
