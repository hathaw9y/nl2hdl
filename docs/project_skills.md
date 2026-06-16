# Project Skills

This repository keeps project-owned Codex skills under `skills/`.

The active Codex runtime still discovers installed skills from
`~/.codex/skills/`. Treat the repository copies as the source-controlled
project baseline, then copy or sync them into `~/.codex/skills/` when the
runtime should use an updated version.

## Included Skills

- `skills/multi-agent-hdl-generation/SKILL.md`
  - Thin router skill for parent/sub-agent workflow and role-specific skill
    selection.
- `skills/parent-module-decomposition/SKILL.md`
  - Parent-only input interpretation, clarification gate, module packet
    definition, resource budget allocation, and pre-integration boundary review.
- `skills/hdl-module-implementation/SKILL.md`
  - HDL sub-agent scope, module implementation evidence, and module-level OOC
    synthesis requirements.
- `skills/hdl-module-verification/SKILL.md`
  - Module verification agent audit rules for contracts, simulation evidence,
    OOC synthesis, and unsafe claims.
- `skills/hdl-integration-agent/SKILL.md`
  - Integration agent rules for composing verified children into decoder-block,
    Layer FSM, Top FSM, token-loop, model FSM, or board-shell integration RTL.
- `skills/hdl-integration-verification/SKILL.md`
  - Integration verification agent rules for simulation audit and
    integration-level Vivado synthesis.
- `skills/hdl-board-wrapper-signoff/SKILL.md`
  - Generic board wrapper, external-memory/control integration, model signoff,
    and board-level signoff evidence rules.
- `skills/zcu104-xczu7ev-hardware/SKILL.md`
  - ZCU104 / XCZU7EV hardware profile: resource inventory, 200 MHz clock
    checks, PS/PL/DDR expectations, and ZCU104-specific Vivado evidence rules.
- `skills/fpga-vivado-systemverilog/SKILL.md`
  - SystemVerilog coding, Vivado simulation/synthesis, timing, and resource
    guidance.
- `skills/hdl-kernel-contract-gates/SKILL.md`
  - HDL module contract, handshake, evidence, and composability gates.
- `skills/hdl-vivado-timing-closure/SKILL.md`
  - Vivado timing closure retry rules and failure handling.

## Sync Rule

When a reusable failure changes the framework behavior, update the repository
copy first. Do not edit `~/.codex/skills/<skill>/SKILL.md` as the source of
truth. After validation, sync the changed skill folder into `~/.codex/skills/`
so future Codex runs pick it up.

Use the project sync helper:

```bash
python3 scripts/sync_project_skills.py validate
python3 scripts/sync_project_skills.py sync
python3 scripts/sync_project_skills.py check
```

For one skill:

```bash
python3 scripts/sync_project_skills.py sync --skill hdl-vivado-timing-closure
```

Manual validation remains available:

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/multi-agent-hdl-generation
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/parent-module-decomposition
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-module-implementation
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-module-verification
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-integration-agent
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-integration-verification
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-board-wrapper-signoff
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/zcu104-xczu7ev-hardware
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/fpga-vivado-systemverilog
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-kernel-contract-gates
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hdl-vivado-timing-closure
```

## Agent-to-Skill Map

| Agent / stage | Primary skills |
|---|---|
| Parent Agent | `multi-agent-hdl-generation`, `parent-module-decomposition` |
| HDL Module Implementation Agent | `hdl-module-implementation`, `fpga-vivado-systemverilog`, `hdl-kernel-contract-gates` |
| Module Verification Agent | `hdl-module-verification`, `hdl-kernel-contract-gates`, `fpga-vivado-systemverilog` |
| Module OOC Synthesis/Tuning Agent | `hdl-module-implementation`, `fpga-vivado-systemverilog`, `hdl-vivado-timing-closure` |
| Pre-Integration Boundary Review | `parent-module-decomposition`, `hdl-kernel-contract-gates` |
| Integration Agent | `hdl-integration-agent`, `fpga-vivado-systemverilog`, `hdl-kernel-contract-gates` |
| Integration Verification Agent | `hdl-integration-verification`, `fpga-vivado-systemverilog`, `hdl-vivado-timing-closure` |
| Board Wrapper / Signoff Agent | `hdl-board-wrapper-signoff`, `fpga-vivado-systemverilog`, `hdl-vivado-timing-closure` |
| ZCU104 / XCZU7EV hardware-specific stages | `zcu104-xczu7ev-hardware` plus the relevant parent, Vivado, integration, or board signoff skill |
| Failure-to-Skill update | `skill-creator` |
