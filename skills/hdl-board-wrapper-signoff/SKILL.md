---
name: hdl-board-wrapper-signoff
description: Use when implementing or auditing a target-board wrapper, board interconnect, external-memory/control integration, Vivado TCL/XDC flows, routed timing/resource evidence, and model or board-level signoff claims for the nl2hdl accelerator. Pair with the matching hardware profile skill for board-specific rules.
---

# HDL Board Wrapper Signoff

Use this for board-wrapper implementation and board/model signoff evidence.
Also apply `fpga-vivado-systemverilog`, `hdl-vivado-timing-closure`, and the
matching hardware profile skill.

## Board Wrapper Implementation

The board wrapper agent owns scoped board artifacts, not child accelerator
kernels:

- target-board wrapper or equivalent integrated top;
- AXI-lite, host, or equivalent control path;
- external-memory/interconnect address map and packed-weight/KV-cache movement
  evidence when those are claimed;
- clock/reset connection from the selected board clocking/reset fabric;
- XDC constraints for clock, reset, board I/O, and any exposed interfaces;
- Vivado TCL that reads generated HDL/BD/XDC inputs and writes reports.

Do not claim board signoff from a bounded fixture or from a disconnected shell
when the required board-specific interconnect/memory/control integration is
only generated beside the implemented top.

## Model-Level Signoff

Model-level signoff requires evidence that the model-level harness covers the
claimed layer count and compares against a Python/reference result. Bounded
fixture reports are not full LLaMA execution unless they prove:

- `executed_layer_count` covers `dispatch_plan.model.decoder_layers`;
- target layer iteration or full model execution is true;
- Python/reference comparison passed;
- child waves and integration verification are current for the active hardware
  spec.

## Board-Level Signoff

Board-level signoff requires routed or otherwise explicitly required board
evidence for the active target:

- top is the generated target-board wrapper or equivalent integrated top;
- the target board clock/reset fabric drives the accelerator internally;
- the board control path reaches the accelerator control registers;
- external-memory/address-map evidence is present when external memory is
  claimed;
- Vivado reports include timing summary, utilization, DRC, methodology,
  constraints, checkpoint, and log paths;
- setup, hold, and pulse-width slack are non-negative with zero failing
  endpoints;
- DRC has no hardware-profile-blocking critical warnings from unconstrained or
  default ports;
- implemented clock period satisfies the configured target clock and the
  hardware profile's clocking requirements.

## Evidence Report

Board evidence JSON must record:

- FPGA part, board, target clock, and hardware resource budgets;
- top module/wrapper name and hierarchy evidence;
- command/log/report/checkpoint paths;
- timing, utilization, DRC, methodology, and clock report summaries;
- board interconnect, control, and external-memory integration status;
- remaining risks and forbidden claims.

If proof is missing, write a gap report and `skill_update_candidate` instead of
clearing signoff.
