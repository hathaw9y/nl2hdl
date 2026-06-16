---
name: hdl-module-verification
description: Use when a verification agent audits one or more HDL module packets after implementation, checking requirement coverage, interface contracts, simulation evidence, module OOC synthesis evidence, resource claims, and unsafe target-scale claims without editing source files.
---

# HDL Module Verification

Module verification agents are source-read-only auditors. Also apply
`hdl-kernel-contract-gates` and `fpga-vivado-systemverilog`.

## Audit Scope

Check each module packet for:

- requirement coverage against the module packet and contract document;
- common handshake and packed-vector contract compliance;
- simulation, Verilator, or XSIM evidence against golden vectors;
- `kernel_report.json` and `subagent_result.json` completeness;
- module OOC synthesis evidence for real datapath modules;
- hardware spec identity match: part, clock, budgets, memory width, and selected
  knobs;
- resource assessment and whether low utilization is justified;
- blocked target dependencies and forbidden claims;
- no parent/orchestration edits unless explicitly allowed.

## Findings

Report findings first, ordered by severity:

- P0: correctness, stale/forged evidence, wrong target claim, missing required
  proof.
- P1: timing/resource/methodology failure, incomplete OOC evidence, contract
  mismatch.
- P2: missing important regression, unclear resource/tuning claim, risky
  integration assumption.
- P3: cleanup or documentation issue.

If no P0/P1/P2 issues exist, say so clearly and list residual risk.

## Verification Report

Write the expected verification JSON with:

- status: `passed` or `failed`;
- findings with severity, file/report path, and evidence;
- commands inspected or run;
- required evidence presence;
- skill update candidate when the failure pattern is reusable.

Do not edit RTL, source, tests, contracts, or evidence from implementation
agents. If a command cannot run, report the exact blocker instead of claiming
the gate passed.
