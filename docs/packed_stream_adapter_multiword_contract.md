# Multi-Word Packed Stream Adapter Contract

This contract extends the passed single-word `packed_stream_adapter` fixture.
It remains an internal fixture and is not an AXI, DDR controller, or board
shell milestone.

## Scope

- Accept multiple configured memory stream beats.
- For the ZCU104 GPTQ config, each memory beat is 128 bits.
- Emit 32-bit packed payload chunks in deterministic little chunk order.
- Preserve output valid/ready backpressure across beat boundaries.
- Report fixture coverage separately from board-level or full LLaMA streaming.

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Required common ports:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Required input stream ports:

- `input logic [MEM_WORD_WIDTH-1:0] mem_word_i`
- `input logic mem_valid_i`
- `output logic mem_ready_o`
- `input logic mem_last_i`

Required output stream ports:

- `output logic [PAYLOAD_WIDTH-1:0] payload_word_o`
- `output logic payload_valid_o`
- `input logic payload_ready_i`
- `output logic payload_last_o`

Debug/evidence ports may be added if useful, but they must be clearly marked as
fixture evidence rather than production memory-system ports.

## Parameters

- `MEM_WORD_WIDTH`: use config `hardware.memory_data_width`, expected `128`.
- `PAYLOAD_WIDTH`: `32`.
- `MAX_MEM_WORDS`: at least `2` for the fixture.
- `PAYLOADS_PER_WORD`: `MEM_WORD_WIDTH / PAYLOAD_WIDTH`.
- `MAX_PAYLOADS`: `MAX_MEM_WORDS * PAYLOADS_PER_WORD`.

Reject or fail early if `MEM_WORD_WIDTH` is not a positive multiple of
`PAYLOAD_WIDTH`.

## Functional Semantics

- On `start_i`, the adapter becomes active and accepts memory words through the
  input valid/ready stream.
- `mem_ready_o` may deassert while the adapter is draining payload chunks and
  must never acknowledge a word that cannot be preserved.
- A memory word is accepted only when `mem_valid_i && mem_ready_o`.
- `mem_last_i` marks the final accepted input beat.
- For each accepted memory word, output payload chunk `idx` is:
  `accepted_mem_word[idx*PAYLOAD_WIDTH +: PAYLOAD_WIDTH]`.
- Payload chunks must appear in input beat order, and within each beat in little
  chunk order.
- `payload_valid_o` must remain asserted and `payload_word_o` stable while
  `payload_ready_i` is low.
- `payload_last_o` is asserted only on the final payload chunk derived from the
  input beat that was accepted with `mem_last_i`.
- `done_o` remains high until `start_i` is deasserted.
- Output payload and trace/debug outputs must be stable while `done_o` is high.

## Fixture Test Requirements

The testbench must use at least two 128-bit memory words, generating at least
eight 32-bit output payload chunks.

It must check:

- input valid/ready handshakes for both memory words;
- `mem_last_i` is consumed only on the final input beat;
- all payload chunks match Python golden data in deterministic order;
- output backpressure holds at least one payload from the first beat;
- output backpressure also occurs at or across a beat boundary;
- `payload_last_o` is asserted exactly on the final payload;
- `done_o` asserts after the final payload and stays high while `start_i` is
  high;
- `done_o` clears after `start_i` is deasserted;
- output payload remains stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: packed_stream_adapter_multiword` or a distinct mode name chosen by
  the sub-agent;
- `coverage_level: packed_stream_adapter_multiword_fixture`;
- configured memory width and payload width;
- number of input words and emitted payload chunks;
- consumed input words in hex;
- emitted payload words in hex;
- input handshake trace;
- output backpressure trace with at least two ready-low events;
- `round_trip_passed: true` for the packed INT4 fixture data;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- explicit caveats that the fixture is not AXI, not DDR controller
  integration, not full LLaMA projection streaming, and not board-level
  signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.

