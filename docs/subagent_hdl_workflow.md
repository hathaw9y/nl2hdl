# HDL Sub-Agent Workflow

nl2hdl treats Verilog/SystemVerilog kernel writing as delegated HDL work.
The parent agent owns planning, requirements, orchestration, and verification.
HDL sub-agents own generated RTL implementation changes.

## Roles

- Parent agent:
  - decomposes the LLM accelerator target into inspect/kernel/block/full milestones;
  - assigns bounded HDL tasks to sub-agents;
  - runs integration tests and Vivado/RTL verification;
  - records failed reusable lessons as Skills when a sub-agent cannot resolve them.
- HDL sub-agent:
  - writes or revises Verilog/SystemVerilog generator code and generated RTL;
  - runs local simulation and synthesis for the assigned kernel;
  - reports changed files, commands, timing/resource evidence, and unresolved risks.
- Verification sub-agent:
  - audits requirement coverage and evidence;
  - does not edit source, RTL, tests, or contracts unless explicitly assigned
    as an implementation sub-agent;
  - for integration waves, runs or inspects Vivado synthesis for the composed
    integration top and writes generated synthesis evidence.

## Agent Topology

Use one implementation sub-agent per HDL packet. Projection and non-GEMM module
agents may run in parallel because each packet has a narrow write scope, its own
regression command, and its own evidence directory.

The parent agent owns contracts, dispatch, and verification. It must not
hand-write Verilog/SystemVerilog kernels or integration FSM RTL. The generated
`agent_topology` field in `hdl_subagent_tasks.json` and
`hdl_subagent_dispatch_plan.json` is the machine-readable source for this
division of labor.

The integration chain is intentionally separate:

1. Module sub-agents implement and self-verify individual GEMM and non-GEMM
   kernels.
2. Module sub-agents or tuning sub-agents run module-level out-of-context synthesis,
   record per-module resources, and tune only allowed knobs before integration.
3. A decoder-block sub-agent composes verified and tuned module fixtures.
4. An integration verification sub-agent runs simulation audit and integration-level
   Vivado synthesis for the composed decoder-block top.
5. A Layer FSM sub-agent calls the verified decoder-block child inside one layer.
6. An integration verification sub-agent repeats simulation audit and synthesis for
   the Layer FSM top.
7. A Top FSM sub-agent schedules verified Layer FSM calls and model-level control.
8. A token-loop sub-agent extends the Top FSM fixture to bounded prefill/decode
   sequencing.
9. Integration verification repeats after each higher-level integration wave.
10. A DDR/AXI board-shell sub-agent wraps the verified model FSM child with bounded
   external-memory request/status metadata without claiming PS/PL integration or
   board-level signoff.

Every wave is followed by a verification sub-agent. Module waves use a read-only
audit. Integration waves use a no-source-edit verification sub-agent that also runs
or inspects Vivado synthesis and writes generated integration evidence.
Dependent waves do not start until the audit and required synthesis gate pass.

## Module-Level OOC Synthesis and Tuning Gate

Before any decoder-block, Layer FSM, Top FSM, token-loop, or board-wrapper
integration wave can consume a real datapath module, the module must have
module-level out-of-context synthesis evidence. This gate is intentionally
earlier than top-level integration so the parent can see which module consumes
LUT, DSP, BRAM, URAM, FF, I/O, timing margin, and memory bandwidth.

Each real datapath module report must include:

- simulation or Verilator pass/fail evidence against golden vectors;
- Vivado out-of-context synthesis command, part, and target clock;
- parsed setup, hold, pulse-width, DRC, and methodology status;
- LUT, DSP, BRAM, URAM, FF, and I/O utilization;
- latency or start-to-done cycles and throughput estimate for compute kernels;
- selected tuning knobs such as PE lanes, tile sizes, buffer depth, memory word
  width, accumulator width, and pipeline stages;
- resource assessment: `underutilized`, `near_budget`, `bandwidth_limited`,
  `timing_limited`, or `fixture_control_scaffold`.

If a compute module is underutilized and still has timing/resource headroom, the
parent should dispatch a tuning retry before integration. The retry may adjust
only contract-approved knobs. If timing fails, the retry should reduce
parallelism, add pipeline stages, or split reductions before trying again. Low
resource use is acceptable only when the throughput target is met or the report
explicitly marks the module as a fixture/control scaffold.

Integration agents must instantiate or generate the selected tuned child
configuration and preserve each child module's resource report in their own
evidence summary. They must not replace a tuned child with a new untuned
implementation.

## Integration-Level Synthesis Gate

After each decoder-block, Layer FSM, Top FSM, token-loop, model FSM, or
board-shell integration wave passes implementation simulation, the integration
verification agent must run or inspect Vivado synthesis for the composed
integration top. This gate is later than module OOC synthesis and catches
timing/resource effects that appear only after child modules are connected.

The integration synthesis report must include:

- integration top module and selected child module list;
- Vivado command, part, target clock, log path, and report paths;
- active hardware spec identity and selected child tuning knobs;
- setup, hold, pulse-width, DRC, and methodology status;
- aggregate LUT, FF, DSP, BRAM, URAM, and I/O utilization;
- whether the evidence is fixture-only, bounded integration, or target-scale;
- pass/fail status and retry recommendation.

Child module OOC reports remain prerequisites, but they are not enough to pass
an integration wave. If synthesis cannot run, the verification agent records the
exact blocker and command/log path instead of claiming timing passed.

## Hardware Spec Change Invalidation

Hardware specs are part of the evidence identity. If `fpga_part`, target clock,
device resource inventory, resource budgets, memory data width, or allowed
tuning knobs change, the parent must regenerate inspect/dispatch artifacts and
require fresh module-level OOC synthesis, integration timing/resource evidence,
and board-level evidence.

Do not reuse older reports unless they explicitly record the same active
hardware spec and selected knobs. A mismatch means the report is stale, even if
the old simulation or synthesis previously passed.

## Assignment Packet Artifacts

`--mode inspect` emits the parent-owned files used to assign HDL work:

- `hdl_task_manifest.json`: canonical task list split into GEMM, non-GEMM, and
  integration tasks.
- `hdl_subagent_tasks.json`: machine-readable sub-agent packets derived from
  the manifest. Each packet carries a `module_contract` bundle with the common
  clock/reset, `start_i`/`done_o` handshake, packed-vector order, final-response
  fields, parent/child ownership boundary, and failure-to-SKILL requirement.
- `hdl_subagent_dispatch_plan.json`: dependency-aware dispatch waves. Wave 1
  can spawn projection and non-GEMM implementation agents in parallel. Module
  synthesis/tuning waves then collect per-module OOC resource evidence before
  later waves advance through decoder block, Layer FSM, Top FSM, and token loop.
Those integration waves start only after verification passes. For integration
waves, verification includes integration-level synthesis evidence in addition
to the simulation/evidence audit.
  Projection waves also carry a
  `target_scope` and any `blocked_target_dependencies`, so a wave can explicitly
  allow bounded fixture work while blocking real checkpoint layout claims until
  `real_gptq_weight_layout_preflight` passes. Downstream decoder-block, Layer
  FSM, Top FSM, and token-loop waves inherit upstream blocked dependencies so
  integration agents do not accidentally turn bounded projection fixture
  evidence into a target-scale claim.
- `hdl_subagent_wave_status.json`: parent-owned result collection gate. It
  reads sub-agent `kernel_report.json` and read-only verification JSON results
  when they exist, then marks each wave as `ready_to_dispatch`,
  `ready_for_verification`, `passed`, `blocked_by_dependency`,
  `failed_waiting_for_skill_update`, or `failed_missing_skill_candidate`.
  This artifact does not generate HDL and does not prove sub-agents ran; it
  records what evidence is still missing before the parent can advance.
- `hdl_subagent_execution_manifest.json`: next-spawn instruction list derived
  from the dispatch plan and wave status. It lists the implementation
  Sub-agents or read-only verification Sub-agent that the interactive Codex parent or an external
  runner should spawn next. It also groups ready entries into `spawn_batches`;
  implementation batches with `parallel_spawn_allowed: true` may be launched as
  separate Codex sub-agents in parallel, while verification batches remain
  read-only and wave-scoped. Implementation spawn entries also include the
  `module_contract` bundle and a short `codex_spawn_message` so external
  runners can enforce the HDL interface contract and launch Codex agents
  without scraping prompt prose. Package code still does not spawn agents
  itself.
- `parent_loop_state.json`: parent-owned loop status. It records that the
  Parent Agent is the only orchestrator, all non-parent workers are Sub-agents,
  and the next parent action is to spawn ready Sub-agents, collect failure
  detail, update a Skill, or wait for dependencies.
- `feedback_packet.json`: parent-to-Sub-agent feedback bundle. Ready
  Sub-agents receive their prompt/evidence paths from here; failed Sub-agents
  receive the gate that must be fixed before retry.
- `retry_plan.json`: parent-owned retry plan. It records which retries are
  allowed immediately and which are blocked until a Skill update or complete
  `skill_update_candidate` exists.
- `subagent_prompts/*.md`: implementation prompts for individual HDL
  sub-agents. Each prompt includes the current target-level blocked gates, so a
  standalone prompt still tells the sub-agent which target-scale claims remain
  forbidden.
- `verification_prompts/*.md`: read-only Codex verification prompts, one per
  dispatch wave, used after implementation agents report their evidence.
  Projection-wave prompts list the manifest-derived expected shape for each
  projection packet, so verification agents do not assume `q_proj`, `k_proj`,
  `v_proj`, and `o_proj` have identical dimensions under grouped-query
  attention.
- `skill_update_candidate_template.json`: machine-readable template for failed
  HDL attempts that must be turned into a SKILL update before retrying the same
  reusable failure pattern.
- `target_blocker_remediation_plan.json` and
  `target_blocker_remediation_plan.md`: parent-owned target-readiness follow-up
  plan. These files list the remaining target blockers, the exact inspect
  command shape for `--mlir-graph` and `--gptq-checkpoint` evidence, and the
  required artifact fields that must pass before the project can claim a real
  LLaMA/GPTQ/ZCU104 accelerator rather than bounded HDL fixtures.
  GPTQ metadata inspection canonicalizes common checkpoint aliases such as
  `q_weight`, `zeros`, `zero_points`, and `scale` into the internal
  `qweight`/`qzeros`/`scales` contract before layout and payload preflight.

The prompt files are not proof that HDL exists. They are the contract handoff:
each HDL implementation sub-agent must still edit the assigned RTL/generator
scope, run the required commands, and report simulation plus timing/resource
evidence before the parent can mark that module gate passed.

In each task evidence directory, an HDL sub-agent should write:

- `kernel_report.json`: machine-readable simulation, Verilator, Vivado, timing,
  and contract-gate evidence for the assigned kernel.
- `module_ooc_synthesis_report.json`: module-level resource/timing/tuning
  evidence for real datapath modules, or an explicit fixture/control waiver.
- `subagent_result.json`: the sub-agent final-response record with
  `changed_files`, `commands_run`, `simulation_evidence`, `verilator_evidence`,
  `vivado_timing_resource_evidence`, `remaining_risks`, and, on failure, any
  `skill_update_candidate`.

Each prompt requires the HDL sub-agent to return a `skill_update_candidate`
when it cannot pass the gate. The parent converts reusable failures into a
Skill update before retrying. Verification prompts also check that failed gates
returned this candidate before another attempt proceeds.

Read-only Codex verification results are collected from
`verification_results/<wave_id>__verification.json` when available. A wave is
only marked `passed` after all implementation reports pass and the verification
JSON has `status: "passed"` with no P0/P1/P2 findings.

After sub-agents write evidence into a collection directory, refresh the parent
view with:

```bash
python3 -m nl2hdl subagents status \
  --dispatch-plan build/inspect/hdl_subagent_dispatch_plan.json \
  --evidence-root build/subagent_evidence \
  --out build/subagent_status
```

This rewrites `hdl_subagent_wave_status.json` and
`hdl_subagent_execution_manifest.json` under `--out`. It also emits
`full_llama_execution_readiness.json`, which is stricter than wave completion:
even when target preflight has passed and every HDL dispatch wave is verified,
the parent does not clear the `full_llama_model_execution` blocker until
`full_llama_execution_evidence.json` records a full-layer decode path,
checkpoint payload evidence, token-loop/model-FSM evidence, and a passing
Python reference comparison. Board-level ZCU104 signoff remains a separate gate
and must not be claimed in that full-execution evidence file. The same refresh
also emits `board_zcu104_signoff_readiness.json`, which requires
`board_zcu104_signoff_evidence.json` with ZCU104 part, board I/O, PS/PL, DDR,
Vivado timing, resource, and report evidence after full execution passes.
Package code still does not spawn agents; it tells the interactive Codex parent
or external runner which implementation or read-only verification Sub-agents
should run next. The same directory also receives `codex_spawn_instructions.md`,
`parent_loop_state.json`, `feedback_packet.json`, and `retry_plan.json`.

After actually spawning external Codex sub-agents, create or refresh the
parent-owned spawn ledger:

```bash
python3 -m nl2hdl subagents ledger \
  --execution-manifest build/subagent_status/hdl_subagent_execution_manifest.json \
  --wave-status build/subagent_status/hdl_subagent_wave_status.json \
  --out build/subagent_status \
  --agent-record wave_1_projection_kernels::implementation::projection_q_proj=<agent-id>
```

This writes `hdl_subagent_spawn_ledger.json` and
`hdl_subagent_spawn_ledger.md`. The ledger maps `spawn_key` values to external
agent ids and expected evidence paths. When `--wave-status` is supplied, it
also reconciles records to `evidence_passed`,
`evidence_incomplete_subagent_result`,
`evidence_failed_waiting_for_skill_update`, or
`evidence_failed_missing_skill_candidate`. It is bookkeeping only; package code
does not spawn Codex agents automatically.

The ledger is parent-owned. A Sub-agent never marks another Sub-agent complete
or spawns a retry; it only writes evidence that the parent later reconciles.

If a `kernel_report.json` passes but `subagent_result.json` is missing or lacks
the required final-response fields, wave status becomes
`incomplete_subagent_result`. The parent must collect the complete
`subagent_result.json` before starting read-only verification or dependent
Layer FSM/Top FSM waves.

## Failure-to-Skill Rule

If an HDL sub-agent fails on a concrete RTL issue, create or update a Skill that
captures the failure pattern and the next attempt policy. A useful Skill entry
must include:

- failing command and target;
- symptom from simulation, lint, or Vivado;
- root cause hypothesis;
- coding rule that prevents recurrence;
- minimal regression test or synthesis check.

When a failed task or failed read-only verification report already returned a
complete `skill_update_candidate`, collect the parent-owned SKILL update draft
before retrying:

```bash
python3 -m nl2hdl subagents skill-draft \
  --dispatch-plan build/inspect/hdl_subagent_dispatch_plan.json \
  --evidence-root build/subagent_evidence \
  --out build/subagent_skill_update \
  --target-skill hdl-kernel-contract-gates
```

This writes `skill_update_candidates.json` and `skill_update_draft.md`. The
collector accepts candidates from failed implementation `kernel_report.json`,
failed implementation `subagent_result.json`, and failed verification reports
or blocking P0/P1/P2 findings. The draft is evidence for the parent
coordination loop; it does not claim the Skill file was edited automatically,
and it does not make the failed HDL gate pass. After reviewing the draft, update
the appropriate local Skill before spawning a retry agent for the same reusable
failure pattern.

Example failure pattern:

- Symptom: projection kernel simulates but misses ZCU104 200 MHz setup timing.
- Prevention: avoid wide one-cycle PE reductions; use bounded-depth sequential or
  pipelined `MUL -> ACC` stages before increasing PE parallelism.

## Current Milestone Policy

For GPTQ INT4 LLaMA kernels, implement and verify in this order:

1. MLIR GEMM/non-GEMM partition.
2. GPTQ packed INT4 metadata and round-trip tests.
3. `int4_unpack`.
4. `int4_projection`.
5. RMSNorm/RoPE/non-GEMM kernels.
6. Module-level OOC synthesis and resource tuning for each real datapath
   module.
7. Decoder-block skeleton.
8. Integration verification synthesis after every decoder, Layer FSM, Top FSM,
   token-loop, model FSM, and board-shell integration wave.
9. DDR/AXI board-shell fixture.
10. ZCU104 synthesis/retry loop.
