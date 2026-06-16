# Decoder Block Scaffold Contract

This contract defines the first integration HDL sub-agent task after the
individual fixture-gated kernels.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes.

## Scope

Create a single decoder-block scaffold that composes already gated fixture
kernels with explicit sequencing and reporting. This is an integration scaffold,
not full LLaMA-3.2-1B decoder-block coverage.

The scaffold should prove that the agent framework can wire child kernels using
the common start/done contract and can report missing full-model behavior
honestly.

## Kernel Name

Use:

- CLI kernel: `decoder_block_scaffold`
- HDL module: `decoder_block_scaffold`
- report artifact: `decoder_block_scaffold_golden.json` plus
  `kernel_report.json`

## Allowed Child Kernels

Only compose child modules that have passed their individual gates:

- `rmsnorm_target`
- `rope_target`
- `projection_tile`

Optional fixture children may be included if they have report evidence:

- `int4_unpack`
- `gptq_dequant`

Do not use the legacy `rmsnorm`, `rope`, or old `decoder_block` scaffold as
target coverage.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `aclk`
- `aresetn`
- `start_i`
- `done_o`

Minimum data/report contract:

- expose or internally generate deterministic fixture inputs for child kernels;
- sequence child kernels using explicit FSM states;
- keep child inputs stable while a child is busy;
- consume child outputs only after child `done_o`;
- report child module order, child coverage levels, and any omitted decoder
  operations.

## Required Scaffold Sequence

The first fixture sequence should be:

1. RMSNorm apply fixture.
2. Projection tile fixture for a Q-like projection.
3. RoPE apply fixture for the Q-like projection output or a compatible fixture
   vector.

The sequence may stop there. If it stops there, the report must list omitted
operations:

- K/V/O projections;
- attention score and softmax;
- KV-cache movement;
- residual paths;
- MLP up/gate/down projections;
- final residual and output layer.

## Evidence Requirements

Generated reports must include:

- `coverage_level: decoder_block_scaffold_fixture`;
- `implementation_stage`, such as `not_run`, `post-place`, or `post-route`;
- child module list and child coverage levels;
- FSM state order;
- explicit `omitted_operations`;
- expected output or final observed fixture vector from a Python/NumPy golden
  reference;
- simulation result;
- Verilator result when enabled;
- Vivado timing result when synthesis is enabled.

## Completion Gate

The milestone passes only when:

- common handshake is present;
- child start/done sequencing is covered by simulation;
- outputs are stable while `done_o` is high;
- Python/NumPy golden comparison or deterministic fixture trace passes;
- Verilator evidence is recorded;
- Vivado timing is valid and non-`NA` when synthesis is enabled;
- setup, hold, and pulse-width pass for the reported implementation stage;
- report text does not claim full decoder-block, full attention, or full LLaMA
  coverage.

## Integration Notes

Passing this scaffold can unlock a Layer FSM sub-agent only for a fixture-level
decoder block. It does not unlock Top FSM or full model scheduling. Top FSM work
still requires DDR streaming, KV-cache movement, token loop scheduling, and
target-size kernel coverage.
