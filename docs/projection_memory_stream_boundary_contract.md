# Projection Memory Stream Boundary Contract

This contract defines the next projection milestone after
`projection_target_stream_plan`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write the generated HDL.

## Scope

Create a narrow request/response memory-stream boundary fixture for planned
packed INT4 target projection tiles.

This milestone proves that the framework can:

- derive a packed-weight tile request from target-scale projection planning
  metadata;
- issue a compact memory request with address, beat length, and tag;
- consume a bounded response stream with valid/ready and last;
- preserve beat order and little chunk order into 32-bit payload words;
- connect the response payload stream to a bounded projection-style consumer;
- report all observed request, response, payload, stall, and output evidence.

This is not AXI, not a DDR controller, not a complete board shell, not full
target projection execution, not full model execution, and not board-level
signoff.

## Kernel Name

Use:

- CLI kernel: `projection_memory_stream_boundary`
- HDL module: `projection_memory_stream_boundary`
- report artifact: `projection_memory_stream_boundary_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Memory request boundary:

- `output logic mem_req_valid_o`
- `input logic mem_req_ready_i`
- `output logic [ADDR_WIDTH-1:0] mem_req_addr_o`
- `output logic [15:0] mem_req_beats_o`
- `output logic [7:0] mem_req_tag_o`

Memory response boundary:

- `input logic [MEM_WORD_WIDTH-1:0] mem_rsp_word_i`
- `input logic mem_rsp_valid_i`
- `output logic mem_rsp_ready_o`
- `input logic mem_rsp_last_i`
- `input logic [7:0] mem_rsp_tag_i`

Fixture output and compact debug:

- at least one small registered output vector;
- one narrow payload-observability link may be exposed, such as
  `payload_link_valid_o`, `payload_link_ready_i`, `payload_link_word_o`, and
  `payload_link_last_o`;
- compact debug summary words are allowed;
- wide traces, target metadata vectors, activation vectors, scales,
  zero-points, and full payload arrays must stay internal to the testbench or
  report.

## Planner Requirements

Use the same LLaMA-3.2-1B planning metadata as
`docs/target_scale_projection_streaming_contract.md`.

The report must record:

- selected projection name, initially `q_proj` unless the config requests
  another supported projection;
- checkpoint-aware qweight source plan from `hdl_task_manifest.json` when
  available, including shard file, tensor key, byte offset, byte count, aligned
  request byte address, request beat count, first-beat byte offset, and whether
  the request covers an unaligned safetensors tensor range;
- direct kernel runs may pass the same plan as
  `NL2HDL_TARGET_WEIGHT_STREAM_PLAN_JSON`; this is target checkpoint
  request planning evidence only and must stay separate from the bounded
  four-beat fixture memory request that the RTL executes;
- target projection rows and columns;
- target tile rows and columns;
- target memory beat count for the planned tile;
- fixture memory beat count;
- base address;
- byte offset;
- memory word width;
- payload width and payload count;
- group size, scale policy, and zero-point policy when GPTQ metadata is
  modeled.

For packed INT4 weights:

```text
packed_weight_bytes = rows * cols * 4 / 8
mem_beats = ceil(packed_weight_bytes / (memory_data_width / 8))
tile_packed_bytes = tile_rows * tile_cols * 4 / 8
tile_mem_beats = ceil(tile_packed_bytes / (memory_data_width / 8))
```

For the target planning tile, prefer `tile_rows=64`, `tile_cols=128`, and
`MEM_WORD_WIDTH=128`, which yields 4096 packed INT4 bytes and 256 memory beats.
The bounded RTL fixture may request and consume only a four-beat slice, but the
report must distinguish target tile beats from fixture beats.

## Fixture Requirements

The bounded RTL fixture must:

- issue exactly one memory request after `start_i`;
- hold request fields stable while `mem_req_valid_o` is high and
  `mem_req_ready_i` is low;
- record request-ready backpressure in the testbench or report;
- consume exactly four configured response beats when `MEM_WORD_WIDTH=128`;
- require the final accepted response beat to carry `mem_rsp_last_i`;
- reject or fail simulation on an unexpected response tag;
- derive sixteen 32-bit payload chunks from four 128-bit response beats;
- preserve response beat order and little chunk order within each beat;
- dynamically observe and compare every payload word;
- apply payload-output backpressure at payload indices 0, 3, and 4, or an
  equivalent deterministic pattern recorded in the report;
- compute a small projection-style output from consumed payloads with a
  Python/NumPy golden reference;
- include at least two true same-stage arithmetic lanes, or explicitly reuse a
  bounded projection consumer that reports this evidence;
- keep `done_o` asserted until `start_i` deasserts;
- keep outputs and compact debug stable while `done_o` is high.

## Numeric Policy

Use the same deterministic GPTQ fixture policy as
`projection_target_stream_plan` unless the report explains a narrower policy:

```text
payload_word[k] = accepted_response_word[beat_idx][chunk_idx*32 +: 32]
unpacked_int4 = signed_nibbles(payload_word stream)
dequant_weight = (unpacked_int4 - zero_point) * scale
projection_out[row] = sum_col(dequant_weight[row, col] * activation[col])
```

Report:

- activation format;
- packed payload order;
- unpacked weight format;
- zero-point format;
- scale format;
- accumulator/output format;
- lane product format;
- rounding mode;
- saturation policy.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: projection_memory_stream_boundary`;
- `coverage_level: projection_memory_stream_boundary_fixture`;
- selected projection metadata and target projection shape;
- target tile parameters and fixture tile parameters;
- target tile memory beats and fixture memory beats;
- request address, request beat count, and request tag;
- observed memory request trace;
- observed response handshake trace;
- consumed response words in hex;
- emitted payload words in hex from observed transactions;
- projection-consumed payload words in hex from observed transactions;
- proof that emitted and consumed payloads match exactly;
- response stall and payload backpressure trace;
- projection output vector and Python/NumPy golden output vector;
- lane policy with requested, target-plan, effective fixture, and true parallel
  datapath lanes;
- configured memory width and effective fixture stream width;
- `round_trip_passed: true`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- implementation stage, preferably `post-route`;
- `does_not_claim` entries for AXI, DDR controller integration, complete board
  shell, full target projection execution, full model execution, and
  board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- The single memory request is dynamically observed and checked.
- Every observed response beat and payload word is compared.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
