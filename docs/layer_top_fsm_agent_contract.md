# Layer FSM and Top FSM Agent Contract

This contract is for future HDL sub-agents. The parent agent defines and verifies
this contract, but does not write Verilog/SystemVerilog for these FSMs.

## Scope

- Target: ZCU104 using `xczu7ev-ffvc1156-2-e` at 200 MHz.
- Model style: LLaMA decoder-only token path with GPTQ INT4 weights streamed
  from external DDR.
- Design style: `llm_decoder_streaming`.
- Base module contract: follow `docs/hdl_module_interface_contract.md`.

## Agent Ownership Boundary

The parent agent defines this contract, emits sub-agent packets, runs
independent verification, and records reusable failure lessons as Skills. It
does not write Layer FSM or Top FSM RTL.

Layer FSM and Top FSM work must be assigned to different implementation
sub-agents. A Layer FSM sub-agent may compose verified child modules inside one
decoder layer/block, but must not own token loops, multi-layer model
scheduling, DDR command policy, or Top FSM state. A Top FSM sub-agent may
schedule verified Layer FSM children and model-level control, but must not
rewrite individual kernels or alter Layer FSM internals.

If either FSM sub-agent fails, the parent must preserve the failing evidence,
write or update a reusable Skill rule, and only then retry with a narrower
prompt.

## Required Proof Before Spawning

The parent agent must not spawn Layer FSM or Top FSM implementation agents until
the following are proven and recorded:

- MLIR or semantic graph inspection emits tensor shapes, operation order, and
  GEMM/non-GEMM partition.
- GPTQ metadata parsing proves packed INT4 round-trip, group scales, zero-points,
  and projection naming.
- Individual kernels required by the target milestone pass Python/unit tests and
  RTL simulation.
- Any synthesized kernel required for the milestone passes setup, hold, and
  pulse-width timing checks, not only setup WNS.
- Kernel reports identify ports, packed vector element order, latency assumptions,
  resource evidence, and unresolved risks.

## Layer FSM Agent

The Layer FSM agent composes validated kernels for one decoder block and one
token path. It owns intra-block scheduling only.

Responsibilities:

- sequence RMSNorm, Q/K/V projection, RoPE, attention or softmax/control,
  output projection, residual, MLP projections, activation, and final residual
  according to the inspected graph partition;
- connect only validated kernel interfaces and preserve their packed vector
  layout assumptions;
- enforce stable inputs while each child kernel is busy and consume outputs only
  after the child reports completion;
- define activation buffer ownership within the block, including ping-pong or
  scratch-buffer lifetimes;
- expose block-level latency, buffer depth, tile-size, and PE-lane assumptions
  in a generated report;
- fail early if a required kernel, tensor shape, scale format, or memory access
  contract is missing.

The Layer FSM agent must not own global token scheduling, DDR command policy, or
multi-layer iteration.

## Top FSM Agent

The Top FSM agent schedules global execution around validated Layer FSM blocks.
It owns model-level orchestration.

Responsibilities:

- handle global start/done behavior and reset sequencing;
- schedule prefill/decode token loops according to fixed-shape configuration;
- issue block/layer calls in model order and track the current layer, token, and
  sequence position;
- coordinate external DDR packed-weight streaming and KV-cache movement;
- select or expose buffering strategy for activations, weights, and KV-cache
  traffic without changing child kernel contracts;
- collect model-level latency/resource estimates and Vivado synthesis evidence;
- fail early if sequence length, batch size, memory width, or graph order is
  dynamic or unsupported.

The Top FSM agent must not rewrite individual kernels or alter Layer FSM internal
scheduling unless explicitly reassigned as a Layer FSM agent.

## Shared Control Assumptions

- All FSMs and kernels use the common clock/reset and command/done convention in
  `docs/hdl_module_interface_contract.md`.
- Parent inputs to a child module remain stable from command acceptance until the
  child reports completion.
- Child outputs are considered valid only when completion is reported.
- Completion remains observable until the parent has returned the command to the
  inactive state.
- Packed vectors use little element order, matching the base module contract.
- Backpressure, DDR stalls, and KV-cache stalls must be represented as explicit
  wait states or reportable scheduling assumptions.

## Required Generated Artifacts

Each Layer FSM or Top FSM sub-agent must produce:

- generated RTL or RTL generator changes within its assigned scope;
- testbench or generated testbench for the assigned integration level;
- Python/NumPy or recorded kernel-level golden reference used for comparison;
- JSON report with graph assumptions, child modules used, latency estimates,
  simulation result, synthesis result when enabled, and pass/fail status;
- Vivado timing and utilization reports when synthesis is requested;
- a short final evidence summary listing changed files and commands run.

## Verification Evidence

Layer FSM evidence must show:

- every child kernel call is covered by a simulation path;
- one small hidden-size decoder-block fixture passes against the reference;
- invalid missing-kernel or unsupported-shape cases fail before RTL integration.

Top FSM evidence must show:

- token-loop sequencing is covered for at least one prefill/decode fixture;
- DDR/KV-cache scheduling assumptions are reflected in the report;
- Vivado reports are parsed for setup, hold, pulse-width, and utilization;
- resource or timing failures trigger bounded retry only for allowed knobs such
  as PE lanes, tile sizes, buffering depth, and memory width.

## Failure-to-Skill Rule

When a Layer FSM or Top FSM sub-agent fails and the lesson is reusable, the
parent agent must create or update a Skill before retrying. The Skill update
must include:

- failing command and generated target;
- observed simulation, lint, timing, resource, or scheduling symptom;
- root-cause hypothesis;
- prevention rule for the next HDL sub-agent prompt;
- minimal regression test, report check, or Vivado parser check that prevents
  the same false pass.
