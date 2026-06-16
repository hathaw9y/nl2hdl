# Methodology Differences From The Earlier Baseline

The earlier baseline centered on GPTQ INT4 weight-only projection acceleration
with a SIMD/vector-MAC-style streaming path. This example changes the target
methodology in several important ways.

## Optimization Difference

Baseline:

- `int4_gptq` weight-only path;
- activations and KV-cache were not explicitly W4/KV4;
- non-GEMM numeric work was mostly RMSNorm/RoPE/softmax/residual fixtures.

This example:

- uses QuaRot-style rotations;
- targets W4A4KV4, not only W4 weights;
- adds A4 activation quantization boundaries;
- adds KV4 cache pack/unpack and dequant boundaries;
- requires rotation kernels or approved rotation fusion before integration.

Impact:

- Parent decomposition must add or preserve QuaRot rotation and A4/KV4 module
  packets.
- Verification must include activation/KV quantization and rotation golden
  vectors, not just GPTQ weight unpack/dequant.

## Hardware Architecture Difference

Baseline:

- first concrete path was `simd_vector_mac`;
- PE count was a flat knob;
- projection kernels were treated as tiled/vector streaming kernels.

This example:

- sets `compute_style: systolic_array`;
- uses weight-stationary dataflow;
- sizes the array from DSP budget;
- candidate shapes are `16x64`, `24x48`, and `32x32`;
- weight tiles are streamed from DDR into on-chip buffers and reused in the
  array.

Impact:

- Module packets need array-shape, tile-scheduler, edge-unpack, and backpressure
  contracts.
- Module OOC synthesis must choose the final systolic shape before integration.

## Memory Movement Difference

Baseline:

- external DDR GPTQ packed weights were already expected, but the compute path
  was not specifically weight stationary.

This example:

- makes DDR-to-on-chip tile streaming central to the projection module;
- requires double-buffered BRAM/URAM tile storage;
- unpacks/dequantizes near the systolic array edge;
- requires explicit valid/ready backpressure at DDR, tile-buffer, and array
  boundaries.

Impact:

- Memory packets become first-class: DDR read command, read data, transaction,
  stream integration, tile buffer, and weight-stationary scheduler contracts
  must remain visible.

## Current Framework Gap Exposed By This Example

The current generated task manifest preserves the free-form QuaRot/W4A4KV4 and
systolic-array configuration, and it generates bounded sub-agent prompts.
However, the built-in semantic task generator still emits the baseline LLaMA
non-GEMM set:

- `input_layernorm`
- `rope_qk`
- `attention_scores_softmax_kv`
- `attention_residual`
- `post_attention_layernorm`
- `silu_gate`
- `swiglu_multiply`
- `mlp_residual`

It does not yet automatically add dedicated QuaRot rotation, A4 activation
quantizer, or KV4 cache quantizer module packets. The hand-authored prompts in
`agent_prompts/` record the intended additional boundaries. A future framework
change should make those boundaries appear directly in `hdl_task_manifest.json`
when `optimization.extra_options.rotation` or W4A4KV4 is present.

## Signoff Difference

Baseline:

- board wrapper/signoff could be discussed after bounded model/control
  fixtures.

This example:

- cannot claim model signoff until W4A4KV4 numeric evidence exists;
- cannot claim board signoff until the systolic array, DDR streaming path,
  AXI/control path, timing, DRC, and clock evidence are all implemented and
  current.

The executed example remains a parent planning and prompt-generation example.
It is safe for bounded sub-agent dispatch, but not safe for target accelerator
claims.

