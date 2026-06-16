---
name: hdl-integration-agent
description: Use when an integration implementation sub-agent composes verified HDL child modules into decoder-block, Layer FSM, Top FSM, token-loop, model FSM, or board-shell integration RTL without rewriting child kernels.
---

# HDL Integration Agent

Use this for integration implementation waves. Also apply
`fpga-vivado-systemverilog` and `hdl-kernel-contract-gates`.

## Ownership

Integration agents compose verified children. They may add parent FSM/control,
adapters, compact status, local buffers, and top-level integration testbenches.
They must not rewrite child kernels or silently replace tuned child modules.

Typical integration roles:

- decoder-block agent: composes projection and non-GEMM children;
- Layer FSM agent: calls a verified decoder-block child inside one layer/block;
- Top FSM agent: schedules verified Layer FSM children and model-level control;
- token-loop/model FSM agent: schedules bounded prefill/decode or model-level
  iteration fixtures;
- board-shell integration agent: wraps the verified model FSM child with
  bounded AXI/DDR request/status metadata without claiming board signoff.

## Inputs

Consume:

- passed child `kernel_report.json` files;
- passed child `module_ooc_synthesis_report.json` files for real datapath
  modules;
- selected child tuning knobs;
- child interface and memory contracts;
- blocked target dependencies and forbidden claims.

## Integration Rules

- Preserve child start/done semantics and stable output rules.
- Internalize wide debug/status vectors when they are not required at the
  integration boundary.
- Avoid widening top-level ports unless the contract explicitly requires it.
- Preserve compact traces that prove child ordering, handshakes, and dataflow.
- Keep fixture-only claims explicit. Bounded integration is not full LLaMA
  execution, DDR controller integration, or board signoff.
- Record consumed child resource summaries in the integration report.

## Required Evidence

Produce:

- integration RTL/generator output;
- integration testbench or generated testbench;
- golden/reference report for the bounded integration path;
- simulation evidence showing child call order, start/done behavior, and final
  output/status;
- `kernel_report.json` with child list, consumed knobs, coverage level, and
  forbidden claims;
- `subagent_result.json` with changed files, commands, evidence, remaining
  risks, and any `skill_update_candidate`.

After implementation passes, an `hdl-integration-verification` agent handles
integration-level synthesis.
