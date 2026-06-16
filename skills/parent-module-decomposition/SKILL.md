---
name: parent-module-decomposition
description: Use when a parent agent interprets model, optimization, hardware spec, and design methodology inputs, asks clarification questions, decomposes an accelerator into minimal HDL module packets, assigns interfaces/resource budgets/tuning knobs, and performs pre-integration boundary review without writing HDL.
---

# Parent Module Decomposition

The parent turns a user target into minimal, independently verifiable HDL
packets. It does not write HDL.

## Input Interpretation

Treat hardware design input as four axes, not one overloaded style:

- `compute_style`: PE/MAC architecture pattern, such as `simd_vector_mac`,
  `systolic_array`, `tiled_pe_array`, `time_multiplexed_pe`, or `scalar_fsm`.
- `execution_style`: scheduling/dataflow, such as `layer_by_layer`,
  `operator_by_operator`, `token_streaming`, `llm_decoder_streaming`,
  `prefill_decode_split`, or `batch_pipeline`.
- `memory_style`: weight, activation, and KV-cache movement/storage, such as
  `external_ddr_gptq_packed`, `external_ddr_streaming`, `uram_bram_tiled`, or
  `onchip_weight_storage`.
- `control_style`: controller structure, such as `hierarchical_fsm`,
  `layer_fsm`, `top_fsm`, or `microcoded_controller`.

These are structured slots, not closed enums. Preserve free-form
`architecture_brief`, `design_candidates`, `optimization_brief`,
`optimization_candidates`, and `extra_options` in parent artifacts and prompts.

## Clarification Gate

Before dispatch, ask focused questions if the parent cannot infer details that
affect module boundaries, interfaces, evidence, or budgets:

- quantization bit width, scale/zero-point/group layout, calibration data;
- pruning/sparsity format, storage layout, or sparse kernel requirement;
- compute fabric, memory movement, execution schedule, or control style;
- verification tolerance, golden source, target clock, or resource budget.

Emit `input_clarification_questions.json` and stop with
`needs_clarification` instead of inventing missing methodology details.

## Hardware Resource Inventory

Keep device capacity and project budget separate:

- `device_*` fields describe board/device inventory.
- `max_*` fields describe the current design budget or tuning limit.

Load the matching hardware profile skill when the board or FPGA part has
profile-specific inventory or signoff rules. For example, use
`zcu104-xczu7ev-hardware` for ZCU104 / XCZU7EV targets. Do not hardcode
board-specific capacity numbers in this generic decomposition skill.

## Module Boundary Rules

Create a separate HDL packet when one or more criteria apply:

- mathematical class differs: GEMM/GEMV, RMSNorm, RoPE, softmax/control,
  residual add, KV-cache movement;
- data movement differs: packed weight streaming, activation buffering,
  KV-cache read/write, AXI/DDR movement;
- bottleneck differs: PE lanes, unpack/dequant bandwidth, softmax control,
  address generation, BRAM/URAM buffering;
- the unit can be independently verified with golden vectors, simulation, and
  timing/resource evidence when required;
- the function is reused across model locations.

Similar projections such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj` should usually share a generic projection packet
with shape, address, and tiling parameters.

## Module Packet Contents

Each module packet should define:

- task id, agent role, semantic op, and target coverage level;
- exact source/generator scope and files that must not be edited;
- interface contract, including clock/reset, `start_i`/`done_o`, packed vector
  layout, valid/ready streams when used, and memory/AXI ports;
- memory contract, including activation buffer storage, word width, depth,
  bank count, ping-pong policy, producer, and consumer;
- verification contract, including golden source, simulator, tolerances, and
  required evidence files;
- resource budget and resource objective;
- allowed tuning knobs such as PE lanes, tile size, buffer depth, memory word
  width, accumulator width, and pipeline stages;
- blocked target dependencies and forbidden claims.

## Pre-Integration Boundary Review

Before integration, the parent verifies that each child packet is still a
minimal reusable unit, not an accidental large subsystem or toy-only fixture.
Check that:

- interface and memory contracts are explicit and stable;
- module OOC synthesis evidence matches the active hardware spec and selected
  knobs;
- low utilization is explained by a throughput target or fixture-only waiver;
- no target-scale claim bypasses blocked checkpoint/model dependencies;
- integration can consume the selected child configuration without rewriting
  the child.

## Resource Objective

After correctness and timing are proven, optimize for maximum useful hardware
utilization under configured limits. Do not treat low LUT/DSP/BRAM as success
for compute kernels unless throughput is already met or the packet is clearly a
fixture/control scaffold.
