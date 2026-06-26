---
name: hdl-vivado-timing-closure
description: Use when generating, reviewing, or fixing Verilog/SystemVerilog that must pass Vivado timing, especially after a sub-agent reports success but Vivado setup, hold, pulse-width, or timing-summary status still fails. Applies to FPGA RTL kernels, board-wrapper synthesis, and timing-closure retry loops; pair with the target hardware profile skill for board-specific clock/signoff rules.
---

# HDL Vivado Timing Closure

Use this skill before claiming an HDL milestone passed on Vivado.

## Required Timing Evidence

Do not call Vivado timing passed from setup WNS alone.

Check all of:

- setup worst slack >= 0
- hold worst slack >= 0
- pulse-width worst slack >= 0
- timing summary does not say constraints are not met
- failing endpoint counts are zero for setup, hold, and pulse-width
- implementation state is named explicitly, such as post-synth, post-place, or
  post-route

If any of these fail, report the milestone as failed or at-risk.

Post-place timing can pass a kernel milestone only if the report names it as
post-place evidence. Do not describe post-place timing as routed timing or final
implementation timing.

Routed timing for an internal fixture can pass a fixture gate even if external
I/O delay constraints are absent, but the report must not describe that as
board-level I/O signoff. Board-level signoff requires explicit I/O, clocking,
and interface constraints appropriate to the chosen board shell and target
hardware profile.

For target-clock board signoff, the raw implemented clock must match the
configured target and any hardware-profile-specific clocking requirements.
Inspect `report_clocks`, `report_timing_summary`, and implemented constraints;
do not emit board signoff when the implemented clock is slower than the target
even if timing slack is positive.

## Sub-Agent Failure Pattern

When an HDL sub-agent says a kernel passed but timing shows hold/setup/PW failure:

1. Preserve the failing command and report path.
2. Record setup, hold, and pulse-width slack separately.
3. Do not overwrite the failure with a successful setup-only result.
4. Assign the HDL rewrite to a sub-agent; the parent agent should not hand-write RTL.
5. If the sub-agent cannot fix it, add the root cause and prevention rule here or in a narrower skill.

## RTL Prevention Rules

- Avoid wide one-cycle reductions such as many PE products plus a large adder tree in one cycle.
- Prefer bounded-depth stages such as `MUL -> ACC -> WRITE`.
- Register state transitions and datapath outputs cleanly.
- Keep generated control FSMs simple enough for Vivado to optimize.
- For target-clock kernels, synthesize early and inspect both setup and hold
  before scaling PE lanes.
- For board-wrapper retries, verify the generated board clock in Vivado after
  board automation. A TCL/IP configuration request is not sufficient unless the
  implemented constraints and `report_clocks` prove the target period.
- When adding a new upper integration kernel above an already routed child
  such as Top FSM, token loop, or model FSM, register the kernel in every
  post-route/timing-enforced kernel set used by the generator before claiming
  the milestone. A new kernel that falls through to post-place-only Vivado can
  show a hold failure or weaker timing evidence even when sibling upper
  integration fixtures require routed timing.
- When possible, run placement before timing reports; unplaced post-synthesis
  timing may hide hold issues.
- For routed fixture gates, preserve unconstrained-I/O warnings as caveats
  instead of silently treating them as board-level closure.
- Treat very small positive hold slack as integration risk even when the fixture
  gate technically passes. If post-route hold slack is only a few picoseconds,
  downstream Decoder/Layer/Top FSM agents should avoid adding wide debug/status
  ports, keep child vectors internal where the contract allows, and rerun
  post-route timing after each integration layer before claiming progress.
- Before marking a target-scale streaming child eligible, run or inspect timing
  in a registered-source/registered-sink wrapper that constrains upstream
  valid/data into downstream payload outputs at the target clock. A standalone
  child top with unconstrained or directly driven inputs can hide single-cycle
  payload math paths that fail when composed by a decoder-block parent.
- If a streaming child uses DSP-heavy payload math, such as SwiGLU or softmax
  approximation, require internal pipeline stages or a documented multicycle
  protocol before integration. Do not rely on a later decoder integration agent
  to absorb an unpipelined child path.

## Verification Commands

For a kernel milestone, run:

```bash
python3 -m pytest -q
python3 -m nl2hdl agent --model <model-name> \
  --spec <hardware-and-verification-spec.yaml> \
  --mode kernel --kernel projection \
  --out build/projection_kernel --verbose
```

Then inspect:

- `kernel_report.json`
- `timing_summary.rpt`
- `utilization.rpt`
- generated `*.sv`

Only claim pass when the report and raw timing file agree.
