---
name: multi-agent-hdl-generation
description: Use when coordinating a parent Codex agent that decomposes model-to-HDL accelerator work into sub-agents, routes each agent to the right HDL/verification/signoff skill, and enforces that the parent does not hand-write Verilog/SystemVerilog.
---

# Multi-Agent HDL Generation

This is the router skill for the nl2hdl agent framework. Keep detailed
role behavior in the role-specific skills listed below.

## Parent Invariants

- The parent agent coordinates; it must not hand-write HDL kernels,
  integration FSM RTL, or board-wrapper RTL.
- The parent is the only orchestrator. Every non-parent worker is a Sub-agent,
  including HDL implementation, verification, integration, board-wrapper,
  model-signoff, and board-signoff workers.
- Sub-agents must not spawn or directly route work to other sub-agents. They
  return evidence, failures, and retry suggestions to the parent only.
- The parent owns input interpretation, module packet definitions, prompt
  packets, dispatch order, feedback packets, evidence collection, retry
  routing, and Skill updates after reusable failures.
- HDL, integration, board-wrapper, and verification work is delegated to
  sub-agents with narrow scopes and explicit evidence requirements.
- A dependent wave cannot start until its implementation evidence and required
  verification gate pass.

## Skill Routing

Load the smallest role-specific skill that matches the current agent:

- Parent decomposition and module packet definition:
  `parent-module-decomposition`.
- HDL module implementation:
  `hdl-module-implementation`, plus `fpga-vivado-systemverilog` and
  `hdl-kernel-contract-gates`.
- Module verification:
  `hdl-module-verification`, plus `hdl-kernel-contract-gates` and
  `fpga-vivado-systemverilog`.
- Module timing failure or retry tuning:
  `hdl-vivado-timing-closure`.
- Decoder-block, Layer FSM, Top FSM, token-loop, model FSM, or board-shell
  integration implementation:
  `hdl-integration-agent`, plus `fpga-vivado-systemverilog` and
  `hdl-kernel-contract-gates`.
- Integration verification after an integration wave:
  `hdl-integration-verification`, plus `fpga-vivado-systemverilog` and
  `hdl-vivado-timing-closure`.
- Hardware-specific board wrapper, external-memory flow, or board-level
  signoff:
  the matching hardware profile skill, such as `zcu104-xczu7ev-hardware`,
  plus `hdl-board-wrapper-signoff`, `fpga-vivado-systemverilog`, and
  `hdl-vivado-timing-closure`.
- Reusable failure lesson capture:
  `skill-creator`.

## High-Level Workflow

1. Parent interprets model, optimization/pruning, hardware spec, and design
   methodology.
2. Parent asks clarification questions before dispatch when methodology is too
   ambiguous to define module boundaries, interfaces, budgets, or verification.
3. Parent decomposes the target into minimal reusable module packets and assigns
   interface contracts, verification contracts, resource budgets, and allowed
   tuning knobs.
4. HDL module sub-agents implement and self-verify independent packets.
5. Real datapath modules pass module-level OOC synthesis and tuning before
   integration.
6. Module verification sub-agents audit each module wave.
7. Integration sub-agents compose only verified/tuned children.
8. Integration verification sub-agents audit the integration wave and run or
   inspect integration-level Vivado synthesis.
9. Board wrapper and signoff sub-agents add the target hardware wrapper,
   interconnect, external-memory, clock/reset, constraint, and routed evidence
   only after model-level integration evidence is sufficient.
10. If a sub-agent fails and the lesson is reusable, preserve evidence and
    update the relevant skill before retrying.

## Parent Feedback Loop

The framework is parent-centered:

1. User input enters the parent once.
2. The parent creates module packets and dispatch waves.
3. The parent emits `feedback_packet.json` for ready or failed sub-agents.
4. Sub-agents implement, verify, synthesize, or audit within their assigned
   scope and return `subagent_result.json` or the required target evidence.
5. The parent refreshes `parent_loop_state.json` and `retry_plan.json`.
6. On reusable failure, the parent updates the relevant Skill, syncs runtime
   skills, and then retries the responsible sub-agent.
7. On pass, the parent advances to the next sub-agent wave.

Required parent-owned loop artifacts:

- `parent_loop_state.json`: global status and next parent action.
- `feedback_packet.json`: scoped feedback sent to ready or failed sub-agents.
- `retry_plan.json`: retry gates, blocked waves, and parent action required
  before another sub-agent attempt.
- `hdl_subagent_spawn_ledger.json`: external sub-agent id and evidence
  bookkeeping.

## Evidence Identity

Hardware spec changes invalidate evidence. If FPGA part, target clock, device
resource inventory, budgets, memory data width, or allowed tuning knobs change,
regenerate planning artifacts and require fresh module OOC synthesis,
integration synthesis, and board evidence.

## Project Skill Location

Project-owned skill baselines live under `skills/`. The active Codex runtime
discovers installed skills from `~/.codex/skills/`; sync validated repository
skills there when future agent runs should use the new behavior.

When a reusable failure requires a Skill update, edit the repository copy under
`skills/<skill>/SKILL.md` first, then run:

```bash
python3 scripts/sync_project_skills.py sync --skill <skill>
python3 scripts/sync_project_skills.py check --skill <skill>
```

Do not make `~/.codex/skills/<skill>/SKILL.md` the only edited copy; it is the
runtime install location, not the project source of truth.
