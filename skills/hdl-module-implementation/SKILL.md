---
name: hdl-module-implementation
description: Use when an HDL implementation sub-agent writes or revises one bounded Verilog/SystemVerilog module packet, self-verifies it, emits kernel evidence, and runs module-level OOC synthesis for real datapath modules.
---

# HDL Module Implementation

Use this for one HDL packet at a time. Also apply
`fpga-vivado-systemverilog` and `hdl-kernel-contract-gates`.

## Scope Rules

- Write only the assigned module/generator, task-specific tests, and evidence
  artifacts.
- Do not edit parent orchestration, dispatch, status, CLI, or unrelated tests.
- Do not weaken contracts, timing gates, forbidden-claim language, or existing
  regressions.
- Reuse the provided module packet contract rather than inventing ports or
  semantics.

## Required Interface

Follow the project module contract unless the packet explicitly extends it:

```systemverilog
input  logic aclk;
input  logic aresetn;
input  logic start_i;
output logic done_o;
```

`start_i` is sampled only when idle, `done_o` remains high until `start_i` is
deasserted, inputs stay stable while busy, and outputs stay stable while
`done_o` is high. Packed vectors use little element order:
`element[idx] == vector[idx*WIDTH +: WIDTH]`.

## Implementation Evidence

Every packet must produce:

- generated `.sv` or generator change;
- testbench or generated testbench;
- Python/NumPy golden vector source or golden report;
- simulator or Verilator evidence;
- `kernel_report.json`;
- `subagent_result.json` listing changed files, commands, evidence, remaining
  risks, and any `skill_update_candidate`.

Real datapath modules must also produce `module_ooc_synthesis_report.json`.
Fixture/control scaffolds may waive OOC only when the report explicitly marks
the packet as fixture-only.

## Module OOC Synthesis

For real datapath modules, run Vivado OOC or equivalent module-level synthesis
for the configured FPGA part and clock. Report:

- command, Vivado version when available, part, and target clock;
- setup, hold, pulse-width, DRC, and methodology status;
- LUT, FF, DSP, BRAM, URAM, and I/O utilization;
- latency, initiation interval or start-to-done cycles, and throughput estimate
  for compute kernels;
- selected tuning knobs;
- resource assessment: `underutilized`, `near_budget`, `bandwidth_limited`,
  `timing_limited`, or `fixture_control_scaffold`.

If timing fails, preserve evidence and use `hdl-vivado-timing-closure` before
retrying. Adjust only contract-approved knobs.

## Failure-To-Skill

If the gate fails, do not hide the failure. Return a `skill_update_candidate`
with:

- failing command;
- symptom and log path;
- root-cause hypothesis;
- prevention rule;
- minimal regression check.

The parent applies accepted Skill updates to the repository copy under
`skills/<skill>/SKILL.md` and then syncs that copy into the runtime skill
directory with `scripts/sync_project_skills.py`.
