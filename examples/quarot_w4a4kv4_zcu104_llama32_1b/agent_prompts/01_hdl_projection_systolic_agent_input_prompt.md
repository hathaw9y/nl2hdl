# Agent Input Prompt: HDL Projection Systolic Agent

You are an HDL implementation sub-agent. The parent owns orchestration; you own
only the assigned projection module packet.

Do not edit unrelated files or parent orchestration code.

## Assigned Module Family

- Module family: projection GEMM/GEMV
- Target projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- Compute style: systolic array
- Dataflow: weight stationary
- Numeric format: W4A4 with INT32 accumulation and explicit requantization
- Weight source: GPTQ packed INT4 streamed from DDR
- Activation source: QuaRot-rotated A4 activation stream
- Candidate array shapes: `16x64`, `24x48`, `32x32`
- Selected shape: choose only after module OOC synthesis evidence

## Required Interface

- Common ports: `aclk`, `aresetn`, `start_i`, `done_o`
- Streaming ports: valid/ready for activation tiles, weight tiles, and output
  tiles
- Weight stream width: `128` bits
- Packed weight order: GPTQ checkpoint order must be stated in the report
- Tile buffer: double-buffered BRAM/URAM weight tile storage
- Backpressure: every DDR/array boundary must be lossless under stall

## Verification Contract

- Python/NumPy golden for a small W4A4 tile
- Packed INT4 GPTQ unpack/dequant test
- Systolic accumulation order test
- Requantization/saturation test
- Verilator or XSIM simulation
- Module OOC synthesis on `xczu7ev-ffvc1156-2-e`
- Resource report must include LUT, FF, DSP, BRAM, URAM, timing, and selected
  array shape

## Forbidden Claims

- full LLaMA throughput;
- real GPTQ numeric correctness from the sparse example fixture;
- board-level ZCU104 signoff;
- final systolic size without OOC synthesis evidence.

