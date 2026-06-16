# Token Loop Decoder Block Timing Margin Recovery Contract

This contract defines the next recommended small milestone after
`token_loop_decoder_block_fixture`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a timing-margin recovery revision for the bounded
`token_loop_decoder_block_fixture` without adding new LLaMA functionality.

The current token-loop decoder-block fixture passes post-route timing, but its
hold slack is very thin:

- setup WNS: 1.361 ns;
- hold WHS: 0.002 ns;
- pulse-width WPWS: 2.225 ns;
- bonded IOB: 164.

Before adding more token steps, layers, DDR/AXI behavior, or board-shell logic,
the next HDL sub-agent should preserve the same functional evidence while
reducing timing and I/O risk.

## Kernel Name

Use one of these names, depending on whether the implementation replaces the
existing fixture or creates a side-by-side comparison:

- preferred replacement: `token_loop_decoder_block_fixture`
- optional comparison kernel: `token_loop_decoder_block_timing_fixture`

If a comparison kernel is added, keep the original kernel available as a
regression scaffold.

## Required Functional Equivalence

The recovered fixture must preserve:

- exactly two deterministic token steps;
- real `top_fsm_decoder_block_fixture` instantiation;
- token trace `0x64636261`;
- top trace `0x5453`;
- layer trace `0x4241`;
- block trace `0xb2b1a2a1`;
- nested attention trace `0x323122211211`;
- nested MLP trace `0x52514241323122211211`;
- per-token and final output `[12, -6, 18, 6]`;
- repeated deterministic output labeling, not token-dependent LLaMA behavior;
- per-token start-hold/deassert/release checks.

## Allowed Changes

The HDL sub-agent may:

- compress top-level status below the current 96 bits if detailed child traces
  remain dynamically checked through simulation hierarchy;
- register or pipeline status packing if it reduces hold risk;
- move detailed debug evidence from top-level status to testbench hierarchy
  checks and `kernel_report.json`;
- keep only the minimum top-level status needed to prove the token-loop
  contract;
- add a report field comparing old and recovered timing/resource numbers.

The HDL sub-agent must not:

- replace real child instantiation with counters only;
- drop dynamic hierarchy checks while still claiming observed evidence;
- add new top-level child vectors, 128-bit memory responses, KV arrays, or wide
  debug/status arrays;
- claim board-level signoff or real LLaMA token semantics.

## Timing Goals

Pass criteria remain non-NA positive post-route setup, hold, and pulse-width
slack with zero failing endpoints.

Target improvement goals:

- hold WHS should improve above the prior 0.002 ns when practical;
- bonded IOB should not exceed the prior 164 count;
- status width should not grow beyond the prior 96 bits.

If Vivado still reports very small positive WHS, the report must label it as a
P3 scaling risk and explain what was attempted.

## Verification

Required commands:

- `python3 -m pytest -q`
- `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel <kernel> --out build/<kernel>_gate --verbose`

A read-only verification agent must audit the result before any later
multi-token, multi-layer, DDR/AXI, or board-shell milestone is accepted.
