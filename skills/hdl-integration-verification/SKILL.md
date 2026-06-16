---
name: hdl-integration-verification
description: Use when an integration verification agent audits an integration wave and runs or inspects Vivado synthesis for the composed integration top, while not editing source, RTL, tests, contracts, or child implementations.
---

# HDL Integration Verification

Integration verification is source-read-only but evidence-producing. It may run
Vivado and write generated synthesis evidence plus the verification JSON.
It must not edit source, RTL, tests, contracts, or child implementations.

Also apply `fpga-vivado-systemverilog`, `hdl-kernel-contract-gates`, and
`hdl-vivado-timing-closure`.

## Prerequisites

Before synthesis, confirm:

- the integration implementation simulation passed;
- child modules passed required verification;
- real datapath children have valid module OOC synthesis reports;
- selected child tuning knobs match the integration top;
- hardware spec identity matches the active config;
- target blockers and forbidden claims are preserved.

## Integration-Level Synthesis

Run or inspect Vivado synthesis for the composed integration top. Child OOC
reports are prerequisites, not substitutes, because they do not show timing and
resource effects from integration FSMs, child wiring, interconnect buffers,
status fanout, adapters, or boundary ports.

Write `integration_synthesis_report.json` with:

- integration top module and selected child module list;
- Vivado command, part, target clock, log path, and generated report paths;
- active hardware spec identity and selected child tuning knobs;
- setup, hold, pulse-width, DRC, and methodology status;
- aggregate LUT, FF, DSP, BRAM, URAM, I/O utilization, and resource assessment;
- evidence scope: fixture-only, bounded integration, or target-scale;
- pass/fail status and retry recommendation.

If Vivado cannot run, preserve the exact command, log path, and environment
blocker. Do not claim synthesis passed.

## Failure Routing

- Integration wiring/control failure: route retry to the integration agent.
- Child internal timing/resource failure: route retry to child module tuning.
- Hardware spec mismatch: mark evidence stale and request fresh module OOC plus
  integration synthesis.
- Reusable failure: return a complete `skill_update_candidate`.

Accepted Skill updates are applied to the project source copy under
`skills/<skill>/SKILL.md` first, then synced into `~/.codex/skills/` with
`scripts/sync_project_skills.py`.

## Verification Report

Write findings first as P0/P1/P2/P3. If no blocking findings exist, state that
the integration wave passed with its evidence scope and residual risks.
