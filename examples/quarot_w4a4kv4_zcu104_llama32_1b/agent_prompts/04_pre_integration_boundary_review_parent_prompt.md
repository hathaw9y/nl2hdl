# Agent Input Prompt: Pre-Integration Boundary Review

You are the parent agent performing pre-integration boundary review.

Do not write HDL. Decide whether each verified child is a minimal reusable unit
that integration may consume.

## Review Questions

1. Is the module still minimal, or did it become a subsystem?
2. Are interface and memory contracts explicit and stable?
3. Does OOC synthesis match the active ZCU104 hardware spec?
4. Is low resource use explained by fixture scope or throughput target?
5. Are QuaRot/W4A4KV4 assumptions visible in the contract?
6. Did any sub-agent claim real checkpoint correctness from the sparse fixture?
7. Can integration instantiate the selected child configuration without
   rewriting the child?

## Output

- `pre_integration_boundary_review.json`
- pass/fail per module packet
- allowed child configurations for integration
- required retry or Skill update candidates

