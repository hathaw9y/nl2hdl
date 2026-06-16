# LLM Accelerator Coding-Agent Plan

Model: `meta-llama/Llama-3.2-1B`
GPTQ metadata source: `examples/quarot_w4a4kv4_zcu104_llama32_1b/input/assumed_gptq_checkpoint`
GPTQ metadata source kind: `configured_override`
MLIR graph source: `not_provided`
MLIR graph source kind: `synthetic_or_export_required`
Target board/device: `AMD ZCU104` / `xczu7ev-ffvc1156-2-e`
Device resources: LUT `230400`, FF `460800`, DSP `1728`, BRAM36 `312`, URAM `96`, I/O `464`
Active budgets: LUT `207360`, FF `414720`, DSP `1536`, BRAM `280`, URAM `80`, I/O `420`
Quantization: `quarot_w4a4kv4_gptq_weights`
Optimization brief: `Use QuaRot-style rotations so the LLaMA decode path can target W4A4KV4: 4-bit GPTQ weights, 4-bit rotated activations, and 4-bit KV-cache storage. The example assumes GPTQ weight tensors already exist and uses a local sparse metadata fixture only to exercise the parent planning flow.
`
Design style alias: `systolic_weight_stationary_llm_streaming`
Compute style: `systolic_array`
Execution style: `llm_decoder_streaming`
Memory style: `external_ddr_streaming`
Control style: `hierarchical_fsm`
Architecture brief: `Use a weight-stationary systolic array for projection GEMM/GEMV tiles. GPTQ-packed W4 weights are streamed from DDR into on-chip tile buffers, unpacked/dequantized near the array edge, and reused in-place while rotated A4 activations stream through the array. Size the array from the active DSP budget rather than hardcoding a final dimension.
`

## Key Constraint

1.23B INT4 weights are roughly 615 MB before GPTQ metadata, so weights cannot reside wholly on-chip

## Input Clarification

Status: `clear`
Requires user response: `False`
Question count: `0`

Questions:
- none

## Candidate Directions

Optimization candidates:
- `{'name': 'quarot_w4a4kv4', 'scope': 'primary example path', 'notes': 'rotate residual/attention/MLP streams, quantize activations to 4-bit, and store KV-cache in 4-bit form'}`
- `{'name': 'w4a8kv4_fallback', 'scope': 'fallback', 'notes': 'keep weights and KV-cache at 4-bit but allow 8-bit activations if W4A4 timing or accuracy is unsafe'}`
- `{'name': 'mixed_precision_softmax_control', 'scope': 'non_gemm_control', 'notes': 'keep softmax/control accumulations wider while preserving W4A4KV4 storage boundaries'}`

Design candidates:
- `{'name': 'ws_systolic_32x32', 'focus': 'first candidate under a 1536-DSP active compute budget', 'risk': 'routing pressure and activation/KV4 quantization overhead may reduce timing margin'}`
- `{'name': 'ws_systolic_16x64', 'focus': 'preserve projection throughput while easing vertical routing', 'risk': 'less square data reuse and more edge buffering'}`
- `{'name': 'ws_systolic_24x48', 'focus': 'middle-ground DSP use with easier BRAM banking', 'risk': 'tile scheduler is less regular than 32x32'}`

## Agent Pipeline

- Load model config and quantization metadata from Hugging Face/local checkpoint
- Export or lower supported model graph fragments to ONNX/MLIR
- Parse MLIR into a semantic LLM graph: embedding, decoder blocks, attention, MLP, norms, lm_head
- Partition MLIR operations into GEMM and non-GEMM groups before hardware planning
- Build a hardware design plan from the semantic graph and ZCU104 resource constraints
- Generate module-level SystemVerilog for selected kernels first: INT4 GEMV/GEMM, RMSNorm, RoPE, attention score/value, SwiGLU
- Generate top-level token loop controller and AXI interfaces
- Run golden-model tests against PyTorch for prefill=small and decode=single-token paths
- Run Verilator/XSIM simulation and Vivado synthesis; retry design knobs such as tile sizes, PE lanes, and buffering

## First RTL Milestones

- MLIR GEMM/non-GEMM partition report for the selected LLaMA checkpoint
- INT4 GPTQ unpack/dequant tile reader
- INT4xINT8 or INT4xINT16 projection GEMV kernel
- RMSNorm kernel
- single decoder block skeleton with AXI-stream-like tile handshakes
- single-token decode loop for one block, then all blocks

## Acceptance Gates

- MLIR/semantic graph report names every LLaMA submodule and tensor shape
- GEMM ops are mapped to tiled INT4 projection kernels and non-GEMM ops are mapped to dedicated RMSNorm/RoPE/softmax/control kernels
- GPTQ metadata parser round-trips packed INT4 weights for at least one projection
- kernel-level RTL matches PyTorch/NumPy references for deterministic vectors
- Vivado synthesis reports timing and resource usage on the ZCU104 part
- agent report records attempted design knobs and final effective design

## Source Notes

- AMD ZCU104 product page: https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu104.html
- AMD UG1267 ZCU104 board guide: https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd
- Meta Llama 3.2 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
