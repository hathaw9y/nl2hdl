# Projection Internal Stream Shell Contract

This contract defines the next projection milestone after
`projection_memory_stream_boundary`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create an internal stream-shell wrapper around the proven
`projection_memory_stream_boundary` fixture so the request, response, and
payload streams are verified internally instead of exposed as wide package-level
ports.

This milestone proves that the framework can:

- instantiate or compose the memory-stream boundary behind a shell-like wrapper;
- drive the boundary from an internal deterministic response source fixture;
- keep memory request, response, and payload observability in reports or
  internal testbench traces;
- reduce standalone top-level I/O versus `projection_memory_stream_boundary`;
- preserve the previously proven request, response, payload, and projection
  output evidence;
- keep the scope separate from AXI, DDR controller integration, a complete board
  shell, full target projection execution, full model execution, and
  board-level signoff.

This is a shell fixture, not a board shell. It should model the shape of an
internal shell-to-kernel connection without claiming a real ZCU104 memory
subsystem.

When `NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON` is supplied, the generator may
copy checkpoint-aware qweight request planning metadata into
`projection_internal_stream_shell_golden.json` and `kernel_report.json`. This is
planning evidence only. The bounded shell fixture still executes its internal
four-beat deterministic response source and must not claim DDR/AXI execution or
full checkpoint qweight payload streaming.

## Kernel Name

Use:

- CLI kernel: `projection_internal_stream_shell`
- HDL module: `projection_internal_stream_shell`
- report artifact: `projection_internal_stream_shell_golden.json` plus
  `kernel_report.json`

## Required Top-Level Interface

Follow `docs/hdl_module_interface_contract.md`.

The top-level module must expose:

- common control: `aclk`, `aresetn`, `start_i`, `done_o`;
- narrow command metadata only, such as projection selector, base address, beat
  count, or tag overrides;
- a small registered output vector;
- compact shell status/debug summary.

The top-level module must not expose these as package-level ports:

- 128-bit memory response words;
- memory request address/beat/tag fields as the primary proven boundary;
- 32-bit payload-link data arrays;
- full activation vectors;
- full scale or zero-point tables;
- wide deterministic metadata or trace arrays.

The previous request/response/payload signals may exist inside the wrapper, but
they must be internal nets or testbench-observed hierarchical signals.

## I/O Budget Requirement

The previous standalone boundary used 340 of 360 bonded IOBs. This milestone
must reduce the standalone top-level bonded IOB count.

Required evidence:

- report the previous reference IOB count as 340;
- report this wrapper's Vivado bonded IOB count;
- require this wrapper's bonded IOB count to be less than or equal to 160;
- if Vivado reports more than 160 bonded IOBs, the milestone fails even if
  timing passes.

## Shell Fixture Requirements

The bounded shell fixture must:

- accept `start_i` and launch exactly one internal boundary run;
- issue or observe exactly one internal memory request from the boundary;
- prove internal request fields are stable during ready backpressure;
- provide exactly four deterministic 128-bit response beats internally;
- apply response valid stalls before response beat indices 1 and 3, or an
  equivalent deterministic stall pattern recorded in the report;
- provide `last` only on the final accepted response beat;
- provide the expected response tag on every accepted response beat;
- derive and observe sixteen 32-bit payload chunks in little chunk order;
- apply or preserve payload backpressure at indices 0, 3, and 4;
- compare every emitted payload with every projection-consumed payload;
- preserve the output vector `[976, 2360]` unless the report explains a new
  deterministic golden vector;
- preserve true same-stage two-lane MAC evidence;
- keep `done_o` asserted until `start_i` deasserts;
- keep output and compact status/debug stable while `done_o` is high.

The shell may reuse the deterministic numeric fixture from
`projection_memory_stream_boundary`.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_internal_stream_shell`;
- `coverage_level: projection_internal_stream_shell_fixture`;
- `wraps_kernel: projection_memory_stream_boundary` or an equivalent explicit
  composition statement;
- whether the child boundary is instantiated or equivalently generated inside
  the wrapper;
- top-level interface summary and exposed port width summary;
- previous reference bonded IOB count: 340;
- current bonded IOB count and whether the I/O reduction gate passed;
- selected projection metadata and target projection shape;
- target tile parameters and fixture tile parameters;
- target tile memory beats and fixture memory beats;
- optional `checkpoint_target_weight_stream_plan`,
  `checkpoint_target_request_summary`, and
  `target_checkpoint_request_planning_only: true` when checkpoint metadata is
  supplied;
- `checkpoint_request_execution_scope` identifying that checkpoint requests are
  planning-only shell metadata, not DDR/AXI execution;
- bounded fixture request execution evidence preserving request address
  `0x120000` and four fixture beats;
- internal request trace;
- internal response handshake trace;
- internal response words in hex;
- emitted payload words in hex from observed transactions;
- projection-consumed payload words in hex from observed transactions;
- proof that emitted and consumed payloads match exactly;
- response stall and payload backpressure trace;
- projection output vector and Python/NumPy golden output vector;
- lane policy with requested, target-plan, effective fixture, and true parallel
  datapath lanes;
- `round_trip_passed: true`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- `does_not_claim` entries for AXI, DDR controller integration, complete board
  shell, full target projection execution, full model execution, and
  board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- The shell top-level exposes no wide memory response or payload data bus.
- The internal memory request is dynamically observed and checked.
- Every internal response beat and payload word is compared.
- The I/O reduction gate passes with bonded IOB count less than or equal to 160.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
