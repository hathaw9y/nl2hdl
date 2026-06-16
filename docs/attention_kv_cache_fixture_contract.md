# Attention/KV-Cache Fixture Contract

This contract defines the next non-GEMM decoder milestone after
`rmsnorm_rope_source_path`.

The parent agent owns this contract. HDL sub-agents implement RTL or RTL
generator changes. The parent agent must not hand-write generated HDL.

## Scope

Create a bounded attention and KV-cache movement fixture that proves the
framework can model the missing attention-side decoder behavior at a small,
verifiable scale.

This milestone proves:

- attention score datapath over a compact query and two cached key vectors;
- softmax/control approximation over the two scores;
- weighted value read over two cached value vectors;
- KV-cache write of a new key/value entry and subsequent read trace;
- explicit report separation between fixture coverage and full LLaMA attention.

This is not full target LLaMA attention. It does not require full sequence
length, multi-head attention, grouped-query attention, real exponential softmax,
KV-cache DDR movement, residual scheduling, MLP, a full decoder block, full
model execution, or board-level signoff.

## Kernel Name

Use:

- CLI kernel: `attention_kv_cache_fixture`
- HDL module: `attention_kv_cache_fixture`
- report artifact: `attention_kv_cache_fixture_golden.json` plus
  `kernel_report.json`

## Required Interface

Follow `docs/hdl_module_interface_contract.md`.

Common control:

- `input logic aclk`
- `input logic aresetn`
- `input logic start_i`
- `output logic done_o`

Top-level data may include:

- a compact query vector;
- one compact key/value update entry and update index;
- small registered attention output vector;
- compact status/debug summary.

Top-level data must not include:

- full target sequence-length KV-cache arrays;
- full target hidden-size vectors;
- full target multi-head tensors;
- wide debug traces.

Detailed cache contents and traces may be internal constants, internal state, or
testbench-observed hierarchical signals, but the report must record all
observed movements.

## Fixture Requirements

Use a deterministic bounded fixture:

- head dimension: at least 4;
- cache slots: at least 2;
- score count: 2;
- output vector width: at least 4 elements;
- signed fixed-point or signed integer query, key, and value vectors;
- positive and negative values in query, key, and value data;
- one cache write for a new key/value entry;
- at least two cache reads for attention score/value consumption.

The fixture must compute or verify:

```text
score_i = dot(query, key_cache[i])
softmax_control = deterministic two-score approximation or lookup
out_j = sum_i(weight_i * value_cache[i][j]) >>> output_shift
```

The softmax/control approximation may be a small RTL lookup, max-selector, or
fixed two-weight policy. It must be explicitly labeled. If true exponential
softmax is not implemented, the report must state `softmax_exp_in_rtl: false`.

## KV-Cache Movement Requirements

The bounded RTL fixture must:

- write one key/value pair into an internal cache slot after `start_i`;
- keep write address, key data, and value data stable for the accepted write;
- read at least two key entries to compute scores;
- read at least two value entries to compute the attention output;
- record write trace and read trace in simulation and `kernel_report.json`;
- distinguish internal cache movement from DDR, AXI, or full memory subsystem
  movement.

## Attention/Control Requirements

The bounded RTL fixture must:

- compute two dot-product scores from the query and two key vectors;
- record score trace in simulation and report;
- produce a deterministic control/weight trace;
- compute output vector against a Python/NumPy golden reference;
- keep `done_o` asserted until `start_i` deasserts;
- keep output and compact status/debug stable while `done_o` is high.

## Required Report Fields

`kernel_report.json` must include:

- `kernel: attention_kv_cache_fixture`;
- `coverage_level: attention_kv_cache_fixture`;
- top-level interface summary;
- fixture dimensions: head dimension, cache slots, score count, output elements;
- numeric policy for query/key/value/score/weights/output;
- cache write trace;
- key read trace;
- value read trace;
- attention score trace;
- softmax/control trace;
- expected output vector and observed output vector;
- `softmax_policy`;
- `softmax_exp_in_rtl`;
- `kv_cache_storage`;
- `kv_cache_external_memory: false`;
- implementation stage, preferably `post-route`;
- Verilator evidence when enabled;
- Vivado timing/resource evidence when synthesis is enabled;
- `does_not_claim` entries for full attention, true exponential softmax when
  omitted, full sequence-length KV-cache, multi-head attention,
  grouped-query attention, DDR/AXI KV-cache movement, full decoder block, full
  model execution, and board-level signoff.

## Pass Criteria

- Unit tests pass.
- RTL simulation passes.
- Cache write and read traces are dynamically observed and checked.
- Two attention scores are dynamically observed and checked.
- Softmax/control trace is dynamically observed and checked.
- Output vector matches Python/NumPy golden reference.
- Verilator evidence is recorded and passes when enabled.
- Vivado timing has non-`NA` setup, hold, and pulse-width checks and zero
  failing endpoints when synthesis is enabled.
- A read-only verification agent finds no P0/P1 issues.
