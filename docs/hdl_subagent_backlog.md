# HDL Sub-Agent Backlog

This backlog tracks which HDL tasks are ready for implementation agents and
which tasks must wait. The parent agent owns this file; HDL sub-agents own RTL
or RTL generator changes.

## Current Gates

- `llama_semantic_inspect`: inspect gate passed for
  `meta-llama/Llama-3.2-1B` semantic model metadata. `--mode inspect` now emits
  `model_semantic_graph.json` alongside `synthetic_llama_block.mlir` and
  `mlir_analysis.json`, and `llm_agent_report.json` records the semantic graph
  artifact. The graph first attempts Hugging Face `AutoConfig` from local cache
  by default, or with download only when `NL2HDL_ALLOW_HF_CONFIG_DOWNLOAD=1`;
  if config is unavailable or not a LLaMA config, it falls back to explicit
  fixture metadata and records `metadata_resolution.status: fallback`. In the
  current environment the target resolved from local cache with hidden size
  2048, intermediate size 8192, 16 decoder layers, 32 attention heads, 8 KV
  heads, and head dim 64. Projection shapes are derived from metadata:
  q/o hidden x hidden, k/v KV-width x hidden, gate/up intermediate x hidden,
  and down hidden x intermediate. The graph records GEMM vs non-GEMM partition,
  INT4 GPTQ metadata requirements, memory-beat estimates, and next HDL contract
  inputs. It explicitly labels coverage as
  `target_model_semantic_inspect_no_checkpoint_weights`, so it does not claim
  checkpoint weight loading, full GPTQ tensor value loading, or full LLaMA
  execution. Parent tests, including resolved-config and fallback-path tests,
  pass and read-only audit found no P0/P1/P2 issues.
- `gptq_checkpoint_metadata_inspect`: inspect gate passed for local GPTQ
  metadata discovery. `--mode inspect` now emits
  `gptq_checkpoint_metadata.json` and records it in both
  `llm_agent_report.json` and `hdl_task_manifest.json`. The parser checks local
  paths or Hugging Face local-cache snapshots only, looks for common GPTQ
  metadata files such as `quantize_config.json`, extracts bits, group size,
  desc-act/symmetry fields, and inspects safetensors/bin index maps or
  single-file `.safetensors` headers for projection `qweight`, `qzeros`,
  `scales`, and `g_idx` keys without materializing tensor values. For direct
  safetensors headers it also records header-only tensor summaries with dtype,
  shape, data offsets, byte count, and `header_only_no_tensor_payload` status;
  index JSON-only checkpoints keep `tensor_summary_count: 0` and
  `tensor_summary_source: not_available` rather than inventing unavailable
  shape metadata. The report separates raw projection keys, qweight-bearing
  quantized projection keys, and complete GPTQ projection metadata where
  qweight, qzeros, and scales are all observed. It also emits
  `gptq_weight_layout_preflight.json`, which now checks header-only qweight
  byte counts plus AutoGPTQ-style INT32 packed qweight shape/dtype and
  groupwise qzeros/scales shape/dtype against the semantic per-projection
  INT4 estimates without reading tensor payloads or asserting
  checkpoint-specific qweight ordering. If quant fields are unavailable, or if
  parsed quant fields lack complete
  qweight/qzeros/scales projection tensor metadata, the manifest keeps
  `real_gptq_checkpoint_metadata` blocked with the observed metadata status and
  counts. Only parsed metadata with at least one complete GPTQ projection
  metadata entry removes that block. A separate
  `real_gptq_weight_layout_preflight` block remains until all seven target
  projection qweight summaries match the semantic target packed-byte counts,
  packed header shapes, and qzeros/scales groupwise summaries, while full LLaMA
  execution and board-level ZCU104 signoff remain blocked.
  `llm_agent_report.json` also
  surfaces a top-level
  `target_gate_summary`, so an inspect run can be artifact-generation `passed`
  while still showing blocked target gates without requiring the operator to
  open the manifest first. In the current environment the target has a cached
  base `config.json` and base safetensors projection headers, so the report
  status is `metadata_json_without_quant_fields`, with raw projection key count
  `7`, quantized projection count `0`, complete GPTQ projection count `0`, and
  `real_gptq_checkpoint_metadata` still blocked. This gate does not claim
  checkpoint weight loading, full tensor materialization, numeric GPTQ
  correctness, or full LLaMA execution. Parent tests cover local parsed
  metadata, single-file safetensors header parsing, plain non-GPTQ safetensors,
  unavailable metadata, inspect artifact emission, top-level blocked gate
  reporting, parsed-and-complete metadata unblocking, and parsed but incomplete
  metadata remaining blocked. The primary config now accepts optional
  `model.gptq_checkpoint` so the target model name can remain
  `meta-llama/Llama-3.2-1B` while GPTQ metadata is read from a separate local
  checkpoint path or Hugging Face cache entry. Inspect reports record both
  `target_model_name` and `metadata_source_model_name`; configured-source
  evidence under `build/parent_configured_gptq_checkpoint_source_verify/`
  shows a complete header-only fake GPTQ metadata source unblocks
  `real_gptq_checkpoint_metadata` but keeps
  `real_gptq_weight_layout_preflight` blocked because the fake qweight
  `byte_count` does not match target LLaMA projection bytes. The CLI also
  accepts `--gptq-checkpoint` on
  `agent`, `generate`, and `plan` as a direct override for
  `model.gptq_checkpoint`; CLI override evidence under
  `build/parent_cli_gptq_checkpoint_override_verify/` shows inspect keeps the
  semantic target model as the CLI `--model` while reading GPTQ metadata from
  the override source. Plan JSON and Markdown record both the GPTQ metadata
  source and whether it came from a configured override or the model name.
  Header-summary evidence under
  `build/parent_gptq_tensor_header_summary_verify/` shows a configured fake GPTQ
  source with `tensor_summary_count: 3`, `tensor_summary_source:
  safetensors_header`, and complete GPTQ projection count `1`; read-only audits
  found no P0/P1/P2 issues and reiterated that this remains metadata-only, not
  tensor payload loading or numeric GPTQ correctness. Layout-preflight evidence
  under `build/parent_gptq_weight_layout_preflight_verify/` records
  `weight_layout_preflight_status: blocked`, target-compatible projection count
  `0/7`, q_proj expected qweight bytes `2097152`, observed fake header
  byte_count `4`, and the explicit `real_gptq_weight_layout_preflight` blocked
  target gate.
- `gptq_payload_probe`: inspect gate now also emits
  `gptq_payload_probe.json` for the selected `q_proj`. This probe reads only a
  bounded safetensors payload prefix for `qweight`, `qzeros`, and `scales`,
  records byte hex, SHA-256, and little-endian 32-bit qweight payload words for
  downstream HDL golden-vector contracts. It intentionally does not claim full
  checkpoint tensor materialization, numeric GPTQ correctness,
  checkpoint-specific qweight ordering, full qweight streaming, or full LLaMA
  execution. Header-only layout fixtures can pass
  `real_gptq_weight_layout_preflight` while still leaving payload sampling
  partial/unavailable if no tensor bytes exist; real safetensors payload bytes
  are required for `gptq_payload_probe.status: sampled`.
  The manifest now propagates the selected projection payload probe into
  projection and AXI/memory integration task packets as `gptq_payload_probe`,
  `target_checkpoint_payload_dependency`, and prompt text naming the
  `gptq_payload_probe.json` golden source plus sampled little-endian qweight
  words. Dispatch waves include `gptq_payload_probe` as a blocked target
  dependency when the selected projection payload is unavailable, so HDL
  implementation agents cannot confuse synthetic fixture payloads with real
  checkpoint payload evidence.
- `hdl_task_manifest`: inspect gate passed for mapping
  `model_semantic_graph.json` into HDL sub-agent work items. `--mode inspect`
  emits `hdl_task_manifest.json` and records it in `llm_agent_report.json`,
  alongside the GPTQ checkpoint metadata status. The inspect gate also emits
  `hdl_subagent_tasks.json`, `hdl_subagent_dispatch_plan.json`, and
  `subagent_prompts/*.md`, which turn manifest entries into HDL
  implementation-agent assignment packets with the contract, narrow allowed
  write scope, required commands, handshake requirements, timing evidence, and
  forbidden claims. The dispatch plan groups the seven projection agents and
  eight non-GEMM agents into parallel first-wave work, then orders decoder
  block, Layer FSM, Top FSM, and token-loop waves behind read-only verification
  gates. The manifest maps all seven GEMM projection
  ops to projection streaming INT4
  GPTQ tasks with shape, packed-byte, memory-beat, contract, regression-kernel,
  evidence requirements, and per-projection GPTQ layout preflight status so
  sub-agent prompts distinguish bounded fixture work from real checkpoint
  layout compatibility. With the fake configured checkpoint evidence, q_proj
  prompts now show expected qweight bytes `2097152`, observed fake header
  byte_count `4`, and dependency
  `blocked_by_real_gptq_weight_layout_preflight`. It maps all eight non-GEMM
  semantic ops to non-GEMM kernel tasks/contracts and records integration tasks
  for decoder block, Layer FSM, Top FSM, and recovered token-loop fixture gates.
  It conditionally blocks real GPTQ checkpoint metadata when the local artifact
  is unavailable or incomplete, separately blocks real GPTQ weight layout when
  header byte counts do not match semantic target shapes, and always blocks
  full LLaMA model execution and board-level ZCU104 signoff rather than
  overclaiming them. Dispatch-plan evidence under
  `build/parent_dispatch_layout_scope_verify/` now marks
  `wave_1_projection_kernels` as `target_scope: bounded_fixture_only` with
  blocked dependency `real_gptq_weight_layout_preflight`, while non-GEMM wave 1
  remains `target_preflight_satisfied_or_not_applicable`. Follow-up evidence
  under `build/parent_dispatch_inherited_layout_scope_verify/` propagates that
  projection dependency into decoder-block, Layer FSM, Top FSM, and token-loop
  waves as an inherited blocked dependency, so integration dispatch remains
  bounded-fixture-only until real GPTQ layout preflight passes. Prompt evidence
  under `build/parent_prompt_target_blocks_verify/` also embeds the current
  target-level blocked gates directly in standalone integration prompt files,
  including `real_gptq_weight_layout_preflight`, full LLaMA execution, and
  board-level signoff. Inspect now also emits `verification_prompts/*.md`, one
  read-only Codex verification prompt per dispatch wave; evidence under
  `build/parent_codex_verification_prompts_verify/` contains six verification
  prompts and records their prompt files in `hdl_subagent_dispatch_plan.json`.
  Failure-to-SKILL evidence under `build/parent_failure_skill_template_verify/`
  emits `skill_update_candidate_template.json`, adds a
  `Failure-To-SKILL Candidate` section to implementation prompts, and asks
  Codex verification prompts to confirm failed gates returned a candidate before
  retry.
  It also records the
  multi-agent policy that the parent must not write HDL and verification agents
  are read-only. The sub-agent prompt artifacts are handoff contracts, not proof
  that target HDL exists. Parent tests, including blocked-task,
  verification-agent policy, Layer FSM/Top FSM packet, all-packet prompt
  content, dispatch-wave ordering, and prompt-emission assertions, pass. A
  read-only audit initially flagged broad write scope and shallow prompt tests;
  the prompt scope now forbids parent orchestration edits and test/contract
  weakening, the tests cover every packet, and re-audit found no P0/P1/P2/P3
  issues.
- `projection_q_proj`: wave-1 GEMM sub-agent implementation gate passed for
  q_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. The HDL implementation sub-agent
  added explicit q_proj report fields: `selected_projection: q_proj`, target
  projection shape `2048 x 2048`, packed INT4 bytes `2097152`, memory beats
  `131072`, and target-vs-fixture distinction metadata. Parent verification
  reran `tests/test_llm_kernels.py`, the full test suite, and
  `python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  projection_target_stream_plan --out build/parent_projection_q_proj_gate_verify
  --verbose`. Evidence records simulation pass, Verilator lint pass, and
  post-route Vivado timing/resource reports with setup WNS `0.508 ns`, hold WHS
  `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints,
  1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282 bonded IOBs. This remains a
  bounded q_proj planning/streaming fixture, not full q_proj execution, AXI/DDR
  integration, full LLaMA execution, real checkpoint GPTQ metadata, or
  board-level ZCU104 signoff. Read-only audit found no P0/P1/P2 issues; a P3
  golden-report alias assertion gap was closed in tests.
- `projection_k_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected k_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. Projection task packets now prefix
  required commands with `NL2HDL_SELECTED_PROJECTION=<semantic_op>`, and
  the kernel defaults to q_proj while rejecting unknown projection names before
  artifact generation. Parent verification reran `tests/test_llm_kernels.py`,
  the full test suite, and `NL2HDL_SELECTED_PROJECTION=k_proj python3 -m
  nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  projection_target_stream_plan --out build/parent_projection_k_proj_gate_verify
  --verbose`. Evidence records `selected_projection: k_proj`, target projection
  shape `512 x 2048`, packed INT4 bytes `524288`, memory beats `32768`,
  simulation pass, Verilator lint pass, and post-route Vivado timing/resource
  reports with setup WNS `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS
  `2.225 ns`, 0 failing setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6
  DSPs, 0 BRAM, and 282 bonded IOBs. This remains a bounded k_proj
  planning/streaming fixture, not full k_proj execution, AXI/DDR integration,
  full LLaMA execution, real checkpoint GPTQ metadata, board/interface timing
  signoff, or board-level ZCU104 signoff. Read-only audit found no P0/P1/P2
  blocking issues; residual P3s note missing input/output delay constraints for
  board signoff and that k_proj selection changes target metadata rather than
  the bounded 2x64 fixture tile.
- `projection_v_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected v_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. The existing env-selected projection
  mechanism already supported v_proj; the HDL sub-agent added v_proj-specific
  report/golden assertions and generated evidence under
  `build/projection_v_proj_gate/`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected the generated
  v_proj sub-agent prompt, and independently ran
  `NL2HDL_SELECTED_PROJECTION=v_proj python3 -m nl2hdl agent --model
  meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode
  kernel --kernel projection_target_stream_plan --out
  build/parent_projection_v_proj_gate_verify --verbose`. Evidence records
  `selected_projection: v_proj`, target projection shape `512 x 2048`, packed
  INT4 bytes `524288`, memory beats `32768`, simulation pass, Verilator lint
  pass, and post-route Vivado timing/resource reports with setup WNS
  `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282
  bonded IOBs. This remains a bounded v_proj planning/streaming fixture, not
  real V-weight execution, full v_proj execution, AXI/DDR integration, full
  LLaMA execution, real checkpoint GPTQ metadata, board/interface timing
  signoff, or board-level ZCU104 signoff. Read-only audit found no P0/P1/P2
  blocking issues; residual P3s note missing input/output delay constraints for
  board signoff and that v_proj/k_proj share shape, so this gate distinguishes
  them through selected metadata rather than distinct real tensor payloads.
- `projection_o_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected o_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. The existing env-selected projection
  mechanism already supported o_proj; the HDL sub-agent added o_proj-specific
  report/golden assertions, added an explicit top-level
  `full_target_projection_execution: false` field to the projection target
  stream report/golden metadata, and generated evidence under
  `build/projection_o_proj_gate/`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected the generated
  o_proj sub-agent prompt, and independently ran
  `NL2HDL_SELECTED_PROJECTION=o_proj python3 -m nl2hdl agent --model
  meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode
  kernel --kernel projection_target_stream_plan --out
  build/parent_projection_o_proj_gate_verify --verbose`. Evidence records
  `selected_projection: o_proj`, target projection shape `2048 x 2048`, packed
  INT4 bytes `2097152`, memory beats `131072`, simulation pass, Verilator lint
  pass, and post-route Vivado timing/resource reports with setup WNS
  `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282
  bonded IOBs. This remains a bounded o_proj planning/streaming fixture, not
  real O-weight execution, full o_proj execution, AXI/DDR integration, full
  LLaMA execution, real checkpoint GPTQ metadata, board/interface timing
  signoff, or board-level ZCU104 signoff. Read-only audit found no P0/P1/P2
  blocking issues; residual P3s note high fixture I/O usage and that q_proj and
  o_proj share shape/bytes/beats, so this gate distinguishes them through
  selected metadata and distinction text rather than distinct real tensor
  payloads.
- `projection_gate_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected gate_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. The existing env-selected projection
  mechanism already supported gate_proj; the HDL sub-agent added
  gate_proj-specific report/golden assertions and generated evidence under
  `build/projection_gate_proj_gate/`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected the generated
  gate_proj sub-agent prompt, and independently ran
  `NL2HDL_SELECTED_PROJECTION=gate_proj python3 -m nl2hdl agent
  --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  projection_target_stream_plan --out
  build/parent_projection_gate_proj_gate_verify --verbose`. Evidence records
  `selected_projection: gate_proj`, target projection shape `8192 x 2048`,
  packed INT4 bytes `8388608`, memory beats `524288`, simulation pass,
  Verilator lint pass, and post-route Vivado timing/resource reports with setup
  WNS `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282
  bonded IOBs. This remains a bounded gate_proj planning/streaming fixture, not
  real gate-projection weight execution, full gate_proj execution, full
  SwiGLU/MLP execution, AXI/DDR integration, full LLaMA execution, real
  checkpoint GPTQ metadata, board/interface timing signoff, or board-level
  ZCU104 signoff. Read-only audit found no P0/P1/P2 blocking issues; residual
  P3s note high fixture I/O usage and that gate_proj/up_proj share
  shape/bytes/beats, so this gate distinguishes gate_proj through selected
  metadata and distinction text rather than distinct real tensor payloads or
  full MLP semantics.
- `projection_up_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected up_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture. The existing env-selected projection
  mechanism already supported up_proj; the HDL sub-agent added up_proj-specific
  report/golden assertions, explicitly checked that the distinction text does
  not accidentally report gate_proj, and generated evidence under
  `build/projection_up_proj_gate/`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected the generated
  up_proj sub-agent prompt, and independently ran
  `NL2HDL_SELECTED_PROJECTION=up_proj python3 -m nl2hdl agent --model
  meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode
  kernel --kernel projection_target_stream_plan --out
  build/parent_projection_up_proj_gate_verify --verbose`. Evidence records
  `selected_projection: up_proj`, target projection shape `8192 x 2048`, packed
  INT4 bytes `8388608`, memory beats `524288`, simulation pass, Verilator lint
  pass, and post-route Vivado timing/resource reports with setup WNS
  `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282
  bonded IOBs. This remains a bounded up_proj planning/streaming fixture, not
  real up-projection weight execution, full up_proj execution, full SwiGLU/MLP
  execution, AXI/DDR integration, full LLaMA execution, real checkpoint GPTQ
  metadata, board/interface timing signoff, or board-level ZCU104 signoff.
  Read-only audit found no P0/P1/P2 blocking issues; residual P3s note high
  fixture I/O usage and that gate_proj/up_proj share shape/bytes/beats, so this
  gate distinguishes up_proj through selected metadata and distinction text
  rather than distinct real tensor payloads or full MLP semantics.
- `projection_down_proj`: wave-1 GEMM sub-agent implementation gate passed for
  env-selected down_proj target planning metadata on top of the bounded
  `projection_target_stream_plan` fixture, completing the seven projection
  tasks q/k/v/o/gate/up/down at bounded planning-fixture level. The existing
  env-selected projection mechanism already supported down_proj; the HDL
  sub-agent added down_proj-specific report/golden assertions, explicitly
  checked that the distinction text does not accidentally report gate_proj or
  up_proj, and generated evidence under `build/projection_down_proj_gate/`.
  Parent verification reran `tests/test_llm_kernels.py`, the full test suite,
  inspected the generated down_proj sub-agent prompt, and independently ran
  `NL2HDL_SELECTED_PROJECTION=down_proj python3 -m nl2hdl agent
  --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  projection_target_stream_plan --out
  build/parent_projection_down_proj_gate_verify --verbose`. Evidence records
  `selected_projection: down_proj`, target projection shape `2048 x 8192`,
  packed INT4 bytes `8388608`, memory beats `524288`, simulation pass,
  Verilator lint pass, and post-route Vivado timing/resource reports with setup
  WNS `0.508 ns`, hold WHS `0.042 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 1183 LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282
  bonded IOBs. This remains a bounded down_proj planning/streaming fixture, not
  real down-projection weight execution, full down_proj execution, full MLP
  execution, AXI/DDR integration, full LLaMA execution, real checkpoint GPTQ
  metadata, board/interface timing signoff, or board-level ZCU104 signoff.
  Read-only audit found no P0/P1/P2 blocking issues; residual P3s note high
  fixture I/O usage and bounded 2x64 fixture scope, so this gate distinguishes
  down_proj through selected metadata, transposed target shape, and distinction
  text rather than full target datapath execution.
- `non_gemm_input_layernorm`: wave-1 non-GEMM sub-agent implementation gate
  passed for env-selected input_layernorm target semantic metadata on top of
  the bounded `rmsnorm_rope_source_path` fixture. Parent orchestration now
  prefixes non-GEMM task commands with
  `NL2HDL_SELECTED_NONGEMM=<semantic_op>`, and the HDL sub-agent added
  selector validation/defaulting for `rmsnorm_rope_source_path`, report/golden
  metadata, invalid-selection rejection, and input_layernorm-specific tests.
  Parent verification reran `tests/test_llm_kernels.py`, the full test suite,
  inspected the generated input_layernorm sub-agent prompt, and independently
  ran `NL2HDL_SELECTED_NONGEMM=input_layernorm python3 -m nl2hdl
  agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  rmsnorm_rope_source_path --out
  build/parent_non_gemm_input_layernorm_gate_verify --verbose`. Evidence
  records `selected_non_gemm: input_layernorm`,
  `selected_non_gemm_op_type: RMSNorm`, target shape `{hidden_size: 2048}`,
  `full_target_non_gemm_execution: false`, simulation pass, Verilator lint
  pass, RMS output `[658, -1372, 1152, -659]`, RoPE output `[37, -13, 3,
  -42]`, and post-route Vivado timing/resource reports with setup WNS
  `1.660 ns`, hold WHS `0.057 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 106 LUTs, 205 registers, 4 DSPs, 0 BRAM, and 234
  bonded IOBs. This remains a bounded 4-element RMSNorm/RoPE source-path
  fixture, not full 2048-element RMSNorm execution, reciprocal-sqrt RTL, RoPE
  frequency generation, AXI/DDR integration, full LLaMA execution, or
  board-level ZCU104 signoff. Read-only audit found no P0/P1/P2 blocking
  issues; residual P3 notes that explicit input_layernorm and default
  input_layernorm select the same metadata, so non-default selector-change
  evidence remains for later non-GEMM semantic gates.
- `non_gemm_rope_qk`: wave-1 non-GEMM sub-agent implementation gate passed for
  env-selected RoPE target semantic metadata on top of the bounded
  `rmsnorm_rope_source_path` fixture. The implementation sub-agent added a
  `NL2HDL_SELECTED_NONGEMM=rope_qk` regression in
  `tests/test_llm_kernels.py`; no generator change was needed because
  `llm_kernels.py` already mapped `rope_qk` to RoPE target metadata and
  `rope_source_path`. Parent verification reran `tests/test_llm_kernels.py`,
  the full test suite, inspected report/golden JSON, and independently ran
  `NL2HDL_SELECTED_NONGEMM=rope_qk python3 -m nl2hdl agent --model
  meta-llama/Llama-3.2-1B --spec examples/zcu104_llama32_1b_gptq.yaml --mode
  kernel --kernel rmsnorm_rope_source_path --out
  build/parent_non_gemm_rope_qk_gate_verify --verbose`. Evidence records
  `selected_non_gemm: rope_qk`, `selected_non_gemm_op_type: RoPE`, target shape
  `{head_dim: 64, attention_heads: 32, key_value_heads: 8}`,
  `target_source_path: rope_source_path`, `full_target_non_gemm_execution:
  false`, simulation pass, Verilator lint pass, RMS output `[658, -1372, 1152,
  -659]`, RoPE output `[37, -13, 3, -42]`, and post-route Vivado
  timing/resource reports with setup WNS `1.660 ns`, hold WHS `0.057 ns`,
  pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints, 106 LUTs,
  205 registers, 4 DSPs, 0 BRAM, and 234 bonded IOBs. This remains a bounded
  4-element RMSNorm/RoPE lookup fixture, not full RoPE frequency generation,
  full sequence-length table coverage, Q/K projection wiring, AXI/DDR
  integration, full LLaMA execution, or board-level ZCU104 signoff. Read-only
  audit found no P0/P1/P2 blocking issues; residual P3 notes that generic plan
  artifacts still mention future AXI/interface milestones as planning text, not
  current kernel evidence.
- `non_gemm_post_attention_layernorm`: wave-1 non-GEMM sub-agent implementation
  gate passed for env-selected post-attention RMSNorm target semantic metadata
  on top of the bounded `rmsnorm_rope_source_path` fixture. The implementation
  sub-agent added a
  `NL2HDL_SELECTED_NONGEMM=post_attention_layernorm` regression in
  `tests/test_llm_kernels.py`; no generator change was needed because
  `llm_kernels.py` already mapped `post_attention_layernorm` to RMSNorm target
  metadata and `rmsnorm_source_path`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected report/golden
  JSON, and independently ran
  `NL2HDL_SELECTED_NONGEMM=post_attention_layernorm python3 -m nl2hdl
  agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  rmsnorm_rope_source_path --out
  build/parent_non_gemm_post_attention_layernorm_gate_verify --verbose`.
  Evidence records `selected_non_gemm: post_attention_layernorm`,
  `selected_non_gemm_op_type: RMSNorm`, target shape `{hidden_size: 2048}`,
  `target_source_path: rmsnorm_source_path`, `full_target_non_gemm_execution:
  false`, simulation pass, Verilator lint pass, RMS output `[658, -1372, 1152,
  -659]`, RoPE output `[37, -13, 3, -42]`, and post-route Vivado
  timing/resource reports with setup WNS `1.660 ns`, hold WHS `0.057 ns`,
  pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints, 106 LUTs,
  205 registers, 4 DSPs, 0 BRAM, and 234 bonded IOBs. This remains a bounded
  4-element RMSNorm/RoPE lookup fixture, not full 2048-element RMSNorm
  execution, reciprocal-sqrt RTL, full MLP/residual integration, AXI/DDR
  integration, full LLaMA execution, or board-level ZCU104 signoff. Read-only
  audit found no P0/P1/P2 blocking issues; residual P3 notes the workspace is
  not a Git repo and bonded IOB remains a scaling caveat for this fixture.
- `non_gemm_attention_scores_softmax_kv`: wave-1 non-GEMM sub-agent
  implementation gate passed for env-selected LLaMA attention-control target
  semantic metadata on top of the bounded `attention_kv_cache_fixture`. The
  implementation sub-agent added attention-only selector metadata in
  `llm_kernels.py`, defaulted no-env behavior to
  `attention_scores_softmax_kv`, rejected invalid
  `NL2HDL_SELECTED_NONGEMM` values before artifact generation, and added
  report/golden plus invalid-selector tests in `tests/test_llm_kernels.py`.
  Parent verification reran `tests/test_llm_kernels.py`, the full test suite,
  inspected report/golden JSON, and independently ran
  `NL2HDL_SELECTED_NONGEMM=attention_scores_softmax_kv python3 -m
  nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  attention_kv_cache_fixture --out
  build/parent_non_gemm_attention_scores_softmax_kv_gate_verify --verbose`.
  Evidence records `selected_non_gemm: attention_scores_softmax_kv`,
  `selected_non_gemm_op_type: AttentionControl`, target shape `{head_dim: 64,
  attention_heads: 32, key_value_heads: 8, sequence_length: 2048}`,
  `target_source_path: attention_kv_cache_path`,
  `full_target_non_gemm_execution: false`, fixture head dim `4`, fixture cache
  slots `2`, simulation pass, Verilator lint pass, score trace `[25, -14]`,
  softmax/control policy `two_score_winner_loser_q0_4` with
  `softmax_exp_in_rtl: false`, output `[4, -2, 1, 2]`, internal register
  two-slot KV cache, and post-route Vivado timing/resource reports with setup
  WNS `2.032 ns`, hold WHS `0.061 ns`, pulse-width WPWS `2.225 ns`, 0 failing
  setup/hold/PW endpoints, 170 LUTs, 109 registers, 0 DSPs, 0 BRAM, and 132
  bonded IOBs. This remains a bounded 4-lane, 2-slot attention KV-cache
  fixture, not full attention, true exponential softmax, full sequence-length
  KV cache, grouped-query/multi-head attention, DDR/AXI KV-cache movement, full
  LLaMA execution, or board-level ZCU104 signoff. Read-only audit found no
  P0/P1/P2 blocking issues; residual P3 notes that raw Vivado output includes
  transient early-routing failed-net counts but final timing and route reports
  are clean.
- `non_gemm_attention_residual`: wave-1 non-GEMM sub-agent implementation gate
  passed for env-selected LLaMA attention residual Add target semantic metadata
  on top of the bounded `residual_mlp_fixture`. The implementation sub-agent
  added residual-fixture selector metadata in `llm_kernels.py`, defaulted
  no-env behavior to `attention_residual`, rejected invalid
  `NL2HDL_SELECTED_NONGEMM` values before artifact generation, and added
  report/golden plus invalid-selector tests in `tests/test_llm_kernels.py`.
  Parent verification reran `tests/test_llm_kernels.py`, the full test suite,
  inspected report/golden JSON, and independently ran
  `NL2HDL_SELECTED_NONGEMM=attention_residual python3 -m nl2hdl
  agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  residual_mlp_fixture --out build/parent_non_gemm_attention_residual_gate_verify
  --verbose`. Evidence records `selected_non_gemm: attention_residual`,
  `selected_non_gemm_op_type: Add`, target shape `{hidden_size: 2048}`,
  `target_source_path: attention_residual_path`,
  `full_target_non_gemm_execution: false`, fixture hidden width `4`, fixture
  intermediate width `4`, simulation pass, Verilator lint pass, hidden input
  `[3, -2, 5, 1]`, attention output `[1, 4, -3, 2]`, residual0 `[4, 2, 2,
  3]`, gate `[8, 5, -3, 5]`, up `[9, 2, 1, 0]`, SwiGLU `[9, 1, -1, 0]`, final
  output `[12, -6, 18, 6]`, compact I/O evidence, and post-route Vivado
  timing/resource reports with setup WNS `0.897 ns`, hold WHS `0.057 ns`,
  pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints, 1220 LUTs,
  477 registers, 8 DSPs, 0 BRAM, and 292 bonded IOBs. This remains a bounded
  4-wide residual/MLP fixture, not full hidden-size residual execution,
  target-scale MLP dimensions, GPTQ INT4 packed gate/up/down streaming, true
  SiLU/exponential RTL, DDR/AXI movement, full LLaMA layer/model execution, or
  board-level ZCU104 signoff. Read-only audit found no P0/P1/P2 blocking
  issues; residual P3 notes the ancillary accelerator plan contains
  aspirational future AXI/token-loop wording and that 292/360 bonded IOBs is a
  scaling risk for this fixture.
- `non_gemm_silu_gate`: wave-1 non-GEMM sub-agent implementation gate passed
  for env-selected LLaMA SiLU gate target semantic metadata on top of the
  bounded `residual_mlp_fixture`. The implementation sub-agent added
  `silu_gate` metadata to the residual-fixture selector in `llm_kernels.py`
  while preserving the existing `attention_residual` default and invalid
  selector rejection, then added report/golden assertions in
  `tests/test_llm_kernels.py`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected report/golden
  JSON, and independently ran `NL2HDL_SELECTED_NONGEMM=silu_gate python3
  -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  residual_mlp_fixture --out build/parent_non_gemm_silu_gate_gate_verify
  --verbose`. Evidence records `selected_non_gemm: silu_gate`,
  `selected_non_gemm_op_type: SiLU`, target shape `{intermediate_size: 8192}`,
  `target_source_path: silu_gate_path`, `full_target_non_gemm_execution:
  false`, fixture hidden width `4`, fixture intermediate width `4`, simulation
  pass, Verilator lint pass, gate `[8, 5, -3, 5]`, sigmoid approximation `[16,
  13, 5, 13]`, SiLU approximation `[8, 4, -1, 4]`, SwiGLU `[9, 1, -1, 0]`,
  final output `[12, -6, 18, 6]`, compact I/O evidence, and post-route Vivado
  timing/resource reports with setup WNS `0.897 ns`, hold WHS `0.057 ns`,
  pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints, 1220 LUTs,
  477 registers, 8 DSPs, 0 BRAM, and 292 bonded IOBs. This remains a bounded
  4-wide residual/MLP fixture, not true SiLU/exponential RTL, full
  intermediate-size SiLU execution, target-scale MLP dimensions, GPTQ INT4
  packed gate/up/down streaming, DDR/AXI movement, full LLaMA layer/model
  execution, or board-level ZCU104 signoff. Read-only audit found no P0/P1/P2
  blocking issues; residual P3 notes that sigmoid/SiLU intermediate vectors are
  in JSON rather than standalone simulation trace lines and that 292/360 bonded
  IOBs remains a scaling risk for this fixture.
- `non_gemm_swiglu_multiply`: wave-1 non-GEMM sub-agent implementation gate
  passed for env-selected LLaMA SwiGLU multiply target semantic metadata on top
  of the bounded `residual_mlp_fixture`. The implementation sub-agent added
  `swiglu_multiply` metadata to the residual-fixture selector in
  `llm_kernels.py` while preserving existing `attention_residual`, `silu_gate`,
  and invalid selector behavior, then added report/golden assertions in
  `tests/test_llm_kernels.py`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected report/golden
  JSON, and independently ran `NL2HDL_SELECTED_NONGEMM=swiglu_multiply
  python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  residual_mlp_fixture --out build/parent_non_gemm_swiglu_multiply_gate_verify
  --verbose`. Evidence records `selected_non_gemm: swiglu_multiply`,
  `selected_non_gemm_op_type: Mul`, target shape `{intermediate_size: 8192}`,
  `target_source_path: swiglu_multiply_path`,
  `full_target_non_gemm_execution: false`, fixture hidden width `4`, fixture
  intermediate width `4`, simulation pass, Verilator lint pass, up `[9, 2, 1,
  0]`, SiLU approximation `[8, 4, -1, 4]`, SwiGLU `[9, 1, -1, 0]`, final
  output `[12, -6, 18, 6]`, compact I/O evidence, and post-route Vivado
  timing/resource reports with setup WNS `0.897 ns`, hold WHS `0.057 ns`,
  pulse-width WPWS `2.225 ns`, 0 failing setup/hold/PW endpoints, 1220 LUTs,
  477 registers, 8 DSPs, 0 BRAM, and 292 bonded IOBs. This remains a bounded
  4-wide residual/MLP fixture, not full intermediate-size SwiGLU execution,
  true SiLU/exponential RTL, target-scale MLP dimensions, GPTQ INT4 packed
  gate/up/down streaming, DDR/AXI movement, full LLaMA layer/model execution,
  or board-level ZCU104 signoff. Read-only audit found no P0/P1/P2 blocking
  issues; residual P3 notes that 292/360 bonded IOBs remains a scaling risk for
  this fixture.
- `non_gemm_mlp_residual`: wave-1 non-GEMM sub-agent implementation gate passed
  for env-selected LLaMA MLP final residual target semantic metadata on top of
  the bounded `residual_mlp_fixture`. The implementation sub-agent added
  `mlp_residual` metadata to the residual-fixture selector in
  `llm_kernels.py` while preserving existing residual/SiLU/SwiGLU selector and
  invalid-selector behavior, then added report/golden assertions in
  `tests/test_llm_kernels.py`. Parent verification reran
  `tests/test_llm_kernels.py`, the full test suite, inspected report/golden
  JSON, and independently ran `NL2HDL_SELECTED_NONGEMM=mlp_residual
  python3 -m nl2hdl agent --model meta-llama/Llama-3.2-1B --spec
  examples/zcu104_llama32_1b_gptq.yaml --mode kernel --kernel
  residual_mlp_fixture --out build/parent_non_gemm_mlp_residual_gate_verify
  --verbose`. Evidence records `selected_non_gemm: mlp_residual`,
  `selected_non_gemm_op_type: Add`, target shape `{hidden_size: 2048}`,
  `target_source_path: mlp_residual_path`, `full_target_non_gemm_execution:
  false`, fixture hidden width `4`, fixture intermediate width `4`, simulation
  pass, Verilator lint pass, hidden `[3, -2, 5, 1]`, attention `[1, 4, -3,
  2]`, residual0 `[4, 2, 2, 3]`, gate `[8, 5, -3, 5]`, up `[9, 2, 1, 0]`,
  sigmoid approximation `[16, 13, 5, 13]`, SiLU approximation `[8, 4, -1,
  4]`, SwiGLU `[9, 1, -1, 0]`, down `[8, -8, 16, 3]`, final output `[12, -6,
  18, 6]`, compact I/O evidence, and post-route Vivado timing/resource reports
  with setup WNS `0.897 ns`, hold WHS `0.057 ns`, pulse-width WPWS `2.225 ns`,
  0 failing setup/hold/PW endpoints, 1220 LUTs, 477 registers, 8 DSPs, 0 BRAM,
  and 292 bonded IOBs. This remains a bounded 4-wide residual/MLP fixture, not
  full hidden-size residual execution, target-scale MLP dimensions, GPTQ INT4
  packed gate/up/down streaming, true SiLU/exponential RTL, DDR/AXI movement,
  full LLaMA layer/model execution, or board-level ZCU104 signoff. Read-only
  audit found no P0/P1/P2 blocking issues; residual P3 notes Verilator evidence
  is lint-only, sigmoid/SiLU intermediate vectors are in JSON rather than
  standalone simulation trace lines, and 292/360 bonded IOBs remains a scaling
  risk for this fixture.
- `projection`: kernel gate passed for a synthetic fixture with recorded
  Verilator evidence and valid post-place setup/hold/pulse-width timing.
- `int4_unpack`: kernel gate passed for a synthetic fixture with common
  handshake, recorded Verilator evidence, and valid post-place timing.
- `gptq_dequant`: kernel gate passed for a synthetic fixture with groupwise
  zero-points, signed Q4.4 scales, recorded Verilator evidence, and valid
  post-place timing.
- `projection_tile`: kernel gate passed for a projection tile fixture with
  packed INT4 round-trip, groupwise dequant metadata, PE-lane scheduling
  evidence, recorded Verilator evidence, and valid post-place timing.
- `rmsnorm_target`: kernel gate passed for an RMSNorm apply fixture with
  external Python-golden `inv_rms`, non-uniform gamma, recorded Verilator
  evidence, and valid post-place timing. Reciprocal square root is not in RTL.
- `rmsnorm`: legacy scaffold remains non-target and should not be used for
  decoder integration.
- `rope_target`: kernel gate passed for a RoPE apply fixture with Python-golden
  cos/sin metadata, pair rotation, recorded Verilator evidence, and valid
  post-place timing. Frequency generation and lookup are not in RTL.
- `rope`: legacy scaffold remains non-target and should not be used for decoder
  integration.
- `decoder_block_scaffold`: kernel gate passed for a start/done sequencing
  scaffold over `rmsnorm_target -> projection_tile -> rope_target`, with
  omitted full-decoder operations explicitly reported. Child datapaths are not
  instantiated.
- `decoder_child_datapath`: kernel gate passed for a fixture-level child
  datapath scaffold that instantiates `rmsnorm_target`, `projection_tile`, and
  `rope_target`, records child start/done trace, and passes routed timing.
- `layer_fsm_fixture`: kernel gate passed for a fixture-level Layer FSM that
  instantiates `decoder_child_datapath`, records layer/child traces, and passes
  routed timing. It is not Top FSM or full token scheduling.
- `top_fsm_fixture`: kernel gate passed for a fixture-level Top FSM that
  instantiates `layer_fsm_fixture`, records top/layer/child traces, and passes
  routed timing. It is not full token scheduling, DDR streaming, KV-cache, full
  LLaMA execution, or board-level signoff.
- `projection_streaming`: kernel gate passed for a projection streaming fixture
  that consumes packed INT4 weight words with valid/ready, records stream trace,
  and passes routed timing. It is not DDR, AXI, full LLaMA projection, or
  board-level signoff.
- `projection_parallel_streaming`: kernel gate passed for a projection streaming
  fixture with true same-stage two-lane MAC arithmetic, valid/ready packed
  weight stream consumption, and routed timing. It is not DDR, AXI, target-scale
  LLaMA projection, or board-level signoff.
- `packed_stream_adapter`: kernel gate passed for a packed memory-stream
  adapter fixture that accepts one configured 128-bit memory word, emits four
  32-bit payload chunks with valid/ready output backpressure, records INT4
  round-trip evidence, and passes routed timing. It is not multi-word streaming,
  AXI, DDR controller integration, full LLaMA projection streaming, or
  board-level signoff.
- `packed_stream_adapter_multiword`: kernel gate passed for a packed
  memory-stream adapter fixture that accepts two configured 128-bit memory
  words, emits eight 32-bit payload chunks in little chunk order, preserves
  output backpressure across beat boundaries, records input handshake and INT4
  round-trip evidence, and passes routed timing. It is not arbitrary-depth
  streaming, AXI, DDR controller integration, full LLaMA projection streaming,
  or board-level signoff.
- `projection_adapter_stream_integration`: kernel gate passed for an
  adapter-to-projection fixture that accepts two 128-bit memory words, emits and
  consumes eight 32-bit payload chunks through a valid/ready link, preserves
  deterministic backpressure at payload indices 0, 3, and 4, computes output
  vector `[3816, 1532]` with true same-stage two-lane MAC evidence, records
  Verilator evidence, and passes routed timing with 717 LUTs, 690 registers, 6
  DSPs, and 0 BRAM. It is not arbitrary-depth streaming, AXI, DDR controller
  integration, target-scale LLaMA projection, full model execution, or
  board-level signoff. Verification audit found no P0/P1 issues; next related
  agents should close the P2 observability gap by comparing all observed
  adapter/projection payload words dynamically.
- `projection_target_stream_plan`: kernel gate passed for target-scale
  projection planning plus bounded fixture RTL. It records LLaMA-3.2-1B
  projection metadata, derives q/k/v/o/gate/up/down packed INT4 byte and memory
  beat estimates, selects a 64-lane target planning tile, verifies a 2x64
  fixture with four 128-bit memory words and sixteen dynamically observed
  32-bit payload chunks, preserves backpressure at payload indices 0, 3, and 4,
  computes output vector `[976, 2360]`, records true same-stage two-lane MAC
  evidence, records Verilator evidence, and passes routed timing with 1183
  LUTs, 992 registers, 6 DSPs, 0 BRAM, and 282 bonded IOBs. It is not AXI, DDR
  controller integration, full target projection execution, full model
  execution, or board-level signoff. Verification audit found no P0/P1 issues.
- `projection_memory_stream_boundary`: kernel gate passed for a narrow
  request/response memory-stream boundary fixture after
  `projection_target_stream_plan`. It issues exactly one checked request
  (`addr=0x120000`, `beats=4`, `tag=0x2a`), proves request-field stability
  during ready backpressure, consumes four 128-bit response beats with final
  `last`, checks response tags, derives and dynamically compares sixteen
  little-order 32-bit payload chunks, preserves payload backpressure at indices
  0, 3, and 4, computes output vector `[976, 2360]`, records true same-stage
  two-lane MAC evidence, distinguishes the target 64x128 tile with 256 beats
  from the four-beat fixture slice, records Verilator evidence, and passes
  routed timing with setup WNS 1.410 ns, hold WHS 0.036 ns, pulse-width WPWS
  2.225 ns, 1198 LUTs, 1001 registers, 6 DSPs, 0 BRAM, and 340 bonded IOBs.
  It is not AXI, DDR controller integration, a complete board shell, full
  target projection execution, full model execution, or board-level signoff.
  Verification audit found no P0/P1 issues. P2: because standalone IOB usage is
  340/360, the next shell step should internalize this boundary instead of
  exposing additional package-level pins.
- `projection_internal_stream_shell`: kernel gate passed for an internal shell
  fixture around `projection_memory_stream_boundary`. It keeps the previous
  request, response, and payload links internal instead of exposing them as
  package-level ports, exposes only `aclk`, `aresetn`, `start_i`, `done_o`,
  `output_o`, and `shell_status_o`, observes exactly one internal request,
  verifies request-field stability under backpressure, supplies four internal
  128-bit response beats with final `last` and matching tags, records response
  stalls before indices 1 and 3, dynamically compares sixteen little-order
  payload chunks, preserves payload backpressure at indices 0, 3, and 4,
  computes output vector `[976, 2360]`, records true same-stage two-lane MAC
  evidence, records Verilator evidence, and passes routed timing with setup WNS
  1.134 ns, hold WHS 0.042 ns, pulse-width WPWS 2.225 ns, 477 LUTs, 340
  registers, 6 DSPs, 0 BRAM, and 132 bonded IOBs. The I/O reduction gate passes
  against the previous 340 bonded IOB reference and the `<=160` limit. It is not
  AXI, DDR controller integration, a complete board shell, full target
  projection execution, full model execution, or board-level signoff.
  Verification audit found no P0/P1 issues.
- `rmsnorm_rope_source_path`: kernel gate passed for a bounded non-GEMM source
  path fixture that replaces direct Python metadata ports with RTL-internal
  lookup fixtures for RMSNorm `inv_rms` and RoPE `cos/sin`. The top-level
  exposes common handshake, narrow selectors `norm_token_i` and
  `rope_position_i`, bounded output vectors, and compact status, with no direct
  `inv_rms_i`, `cos_i`, or `sin_i` metadata input ports and no full hidden
  vector or full sequence table exposure. It dynamically observes RMSNorm lookup
  selector 0, valid 1, `inv_rms=7024`, `sumsq=5568`, RoPE position 7 cos/sin
  lookups for two pairs `(13,9)` and `(7,-14)`, computes RMSNorm output
  `[658, -1372, 1152, -659]` and RoPE output `[37, -13, 3, -42]`, records
  Verilator evidence, and passes routed timing with setup WNS 1.660 ns, hold
  WHS 0.057 ns, pulse-width WPWS 2.225 ns, 106 LUTs, 205 registers, 4 DSPs,
  0 BRAM, and 234 bonded IOBs. It does not claim full RMSNorm reciprocal sqrt,
  RoPE frequency generation, full sequence-length table coverage, softmax,
  KV-cache movement, full decoder block, full model execution, or board-level
  signoff. Verification audit found no P0/P1 issues. P2: future integration
  should avoid growing package-level status/output width.
- `attention_kv_cache_fixture`: kernel gate passed for a bounded attention and
  KV-cache movement fixture. It exposes common handshake plus compact
  `output_o` and `status_o`, does not expose full sequence KV arrays, full
  hidden-size vectors, full multi-head tensors, or wide debug traces, performs
  one internal KV write with field stability checked, reads two key entries and
  two value entries from a two-slot internal register cache, computes attention
  scores `[25, -14]`, applies deterministic softmax/control policy
  `two_score_winner_loser_q0_4` with weights `[12, 4]`, computes output vector
  `[4, -2, 1, 2]`, records Verilator evidence, and passes routed timing with
  setup WNS 2.032 ns, hold WHS 0.061 ns, pulse-width WPWS 2.225 ns, 170 LUTs,
  109 registers, 0 DSPs, 0 BRAM, and 132 bonded IOBs. It uses
  `kv_cache_storage: internal_register_fixture_two_slots`,
  `kv_cache_external_memory: false`, and `softmax_exp_in_rtl: false`. It does
  not claim full attention, true exponential softmax, full sequence-length
  KV-cache, multi-head attention, grouped-query attention, DDR/AXI KV-cache
  movement, full decoder block, full model execution, or board-level signoff.
  Verification audit found no P0/P1 issues. P2: future integration should avoid
  widening top-level output/status/debug ports.
- `decoder_child_attention_datapath`: kernel gate passed for a refreshed
  fixture-level decoder child datapath that instantiates
  `rmsnorm_rope_source_path`, `projection_internal_stream_shell`, and
  `attention_kv_cache_fixture` under a compact scheduler. It records the
  ordered child trace `source_path_start`, `source_path_done`,
  `projection_shell_start`, `projection_shell_done`, `attention_kv_start`,
  `attention_kv_done`, dynamically observes RMSNorm/RoPE source lookup
  evidence, projection shell stream status, and attention/KV write/read,
  score/control, and output evidence, computes final fixture output
  `[4, -2, 1, 2]`, records Verilator evidence, and passes routed timing with
  setup WNS 2.073 ns, hold WHS 0.036 ns, pulse-width WPWS 2.225 ns, 292 LUTs,
  253 registers, 0 DSPs, 0 BRAM, and 164 bonded IOBs. It closes the previous
  fixture-level attention/KV omission but does not claim mathematically
  complete Q/K/V/O projection-to-attention wiring, full sequence KV-cache,
  multi-head attention, DDR/AXI KV movement, residual scheduling, MLP, full
  decoder block, full model execution, or board-level signoff. Verification
  audit found no P0/P1 issues after the static-evidence P1 was fixed.
- `layer_fsm_attention_fixture`: kernel gate passed for a refreshed
  fixture-level Layer FSM that instantiates `decoder_child_attention_datapath`
  instead of the older `decoder_child_datapath`. It emits the refreshed child
  RTL and nested child RTL in the same gate directory, records the layer trace
  `decoder_child_attention_datapath_start`,
  `decoder_child_attention_datapath_done`, records the nested child trace
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`,
  dynamically observes child start-hold/deassert protocol evidence, source-path
  lookup evidence, projection stream status, and attention/KV write/read,
  score/control, and output evidence, computes final fixture output
  `[4, -2, 1, 2]`, records Verilator evidence, and passes routed timing with
  setup WNS 2.127 ns, hold WHS 0.030 ns, pulse-width WPWS 2.225 ns, 332 LUTs,
  329 registers, 0 DSPs, 0 BRAM, and 180 bonded IOBs. It is not Top FSM,
  multi-layer iteration, token scheduling, DDR/AXI movement, residual
  scheduling, MLP, full LLaMA layer/model execution, or board-level signoff.
  Verification audit found no P0/P1/P2/P3 issues.
- `top_fsm_attention_fixture`: kernel gate passed for a refreshed fixture-level
  Top FSM that instantiates `layer_fsm_attention_fixture` rather than the older
  `layer_fsm_fixture`, schedules one fixture layer, emits refreshed layer,
  decoder child, source-path, projection shell, and attention/KV RTL in the
  same gate directory, records top trace `layer_fsm_attention_fixture_start`,
  `layer_fsm_attention_fixture_done`, records layer trace
  `decoder_child_attention_datapath_start`,
  `decoder_child_attention_datapath_done`, records nested child trace
  `source_path_start`, `source_path_done`, `projection_shell_start`,
  `projection_shell_done`, `attention_kv_start`, `attention_kv_done`,
  dynamically observes layer start-hold/deassert protocol evidence,
  source-path lookup evidence, projection stream status, and attention/KV
  write/read, score/control, and output evidence, computes final fixture output
  `[4, -2, 1, 2]`, records Verilator evidence, and passes routed timing with
  setup WNS 2.171 ns, hold WHS 0.022 ns, pulse-width WPWS 2.225 ns, 374 LUTs,
  410 registers, 0 DSPs, 0 BRAM, and 196 bonded IOBs. The report includes a
  timing scope caveat that this is fixture timing without board-level I/O,
  DDR/AXI shell, or PS/PL integration constraints. It is not real token
  prefill/decode, multi-layer target scheduling, DDR/AXI movement, residual
  scheduling, MLP, full LLaMA layer/model execution, or board-level signoff.
  Verification audit found no P0/P1 issues; P3 timing-parser hardening was
  addressed in `nl2hdl.verify`.
- `token_loop_attention_fixture`: kernel gate passed for a bounded
  fixture-level token-loop scheduler that instantiates
  `top_fsm_attention_fixture` and calls it for exactly two deterministic token
  steps. It records the ordered token trace `token0_start`, `token0_done`,
  `token1_start`, `token1_done`; dynamically observes per-token child
  start-hold/deassert protocol evidence; captures top/layer/decoder child
  traces for both token calls; and observes source-path, projection-shell, and
  attention/KV evidence through hierarchy for token 1. The fixture output is
  `[4, -2, 1, 2]` for both token steps and is explicitly reported as repeated
  deterministic fixture output rather than token-dependent output. It records
  Verilator evidence and passes routed timing with setup WNS 1.982 ns, hold WHS
  0.036 ns, pulse-width WPWS 2.225 ns, 426 LUTs, 502 registers, 0 DSPs, 0
  BRAM, and 228 bonded IOBs. The report includes a timing scope caveat that
  this is fixture timing without board-level I/O, DDR/AXI shell, or PS/PL
  integration constraints. It is not real LLaMA token prefill/decode, target
  sequence scheduling, target multi-layer iteration, token-dependent
  full-sequence KV accumulation, DDR/AXI movement, logits/sampling, full
  LLaMA layer/model execution, or board-level signoff. Verification audit found
  no P0/P1 issues. Residual risk: detailed hierarchical child evidence is
  deeply observed for token 1 only, which the contract permits but is not
  symmetric across both token steps.
- `residual_mlp_fixture`: kernel gate passed for a bounded fixture-level
  residual add plus MLP/SwiGLU path. It records the ordered trace
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`; computes hidden `[3, -2, 5, 1]`,
  attention `[1, 4, -3, 2]`, residual0 `[4, 2, 2, 3]`, gate
  `[8, 5, -3, 5]`, up `[9, 2, 1, 0]`, SwiGLU `[9, 1, -1, 0]`, down
  `[8, -8, 16, 3]`, and final output `[12, -6, 18, 6]`; and labels gate/up/down
  projection data as fixture constant matrices rather than streamed projection
  evidence. It records Verilator evidence and passes routed timing with setup
  WNS 0.897 ns, hold WHS 0.057 ns, pulse-width WPWS 2.225 ns, 1220 LUTs, 477
  registers, 8 DSPs, 0 BRAM, and 292 bonded IOBs. It is not target-scale LLaMA
  MLP, GPTQ INT4 gate/up/down streaming, DDR/AXI movement, true SiLU or
  exponential RTL, full LLaMA layer/model execution, or board-level signoff.
  Verification audit found no P0/P1 issues. P2: 292/360 bonded IOB is high
  enough that the next integration should internalize fixture inputs/status
  rather than exposing all child vector ports at the package boundary.
- `decoder_block_attention_mlp_fixture`: kernel gate passed for a refreshed
  fixture-level decoder block scheduler that instantiates real
  `decoder_child_attention_datapath` and `residual_mlp_fixture` children. It
  records the ordered block trace `attention_start`, `attention_done`,
  `mlp_start`, `mlp_done`; observes the attention trace `source_path_start`,
  `source_path_done`, `projection_shell_start`, `projection_shell_done`,
  `attention_kv_start`, `attention_kv_done`; observes the MLP trace
  `residual0_start`, `residual0_done`, `gate_up_start`, `gate_up_done`,
  `swiglu_start`, `swiglu_done`, `down_start`, `down_done`,
  `residual1_start`, `residual1_done`; and dynamically verifies child
  start-hold/deassert/release protocol for both children. It captures attention
  output `[4, -2, 1, 2]`, feeds it to the real residual/MLP child, uses an
  internal hidden fixture `[0, 4, 1, 1]`, and computes final decoder-block
  fixture output `[12, -6, 18, 6]`. It internalizes MLP hidden/attention inputs
  and avoids exposing child-wide status, reducing bonded IOB from the
  standalone residual/MLP reference 292 to 228. It records Verilator evidence
  and passes routed timing with setup WNS 1.057 ns, hold WHS 0.005 ns,
  pulse-width WPWS 2.225 ns, 1446 LUTs, 743 registers, 8 DSPs, 0 BRAM, and 228
  bonded IOBs. It is not target-scale decoder execution, complete Q/K/V/O
  attention math, full KV-cache, target-scale MLP, GPTQ/DDR/AXI streaming, true
  softmax or SiLU/exponential RTL, full LLaMA layer/model execution, or
  board-level signoff. Verification audit found no P0/P1 issues. P3: hold
  slack is positive but very narrow, and 160-bit status/228 IOB remains a
  composition risk.
- `layer_fsm_decoder_block_fixture`: kernel gate passed for a refreshed
  fixture-level Layer FSM that instantiates real
  `decoder_block_attention_mlp_fixture` RTL and nested decoder-block children.
  It records the ordered layer trace
  `decoder_block_attention_mlp_fixture_start`,
  `decoder_block_attention_mlp_fixture_done`; observes block trace
  `attention_start`, `attention_done`, `mlp_start`, `mlp_done`; observes nested
  attention and MLP traces through hierarchy; verifies layer-to-block
  start-hold/deassert/release protocol with `busy_cycles=324`; and computes
  final layer output `[12, -6, 18, 6]`. It compresses top-level status from the
  previous 160-bit block status to 64 bits and reduces bonded IOB from the
  decoder-block reference 228 to 132 while retaining detailed child evidence in
  the simulation hierarchy. It records Verilator evidence and passes routed
  timing with setup WNS 1.048 ns, hold WHS 0.021 ns, pulse-width WPWS 2.225 ns,
  1464 LUTs, 764 registers, 8 DSPs, 0 BRAM, and 132 bonded IOBs. It is not
  target multi-layer LLaMA iteration, token scheduling, DDR/AXI movement, full
  LLaMA layer/model execution, or board-level signoff. Verification audit found
  no P0/P1/P2 issues. P3: fixture timing lacks board I/O delay constraints,
  hold margin is still small for future composition, and evidence is
  RTL-simulation hierarchical evidence rather than gate-level functional sim.
- `top_fsm_decoder_block_fixture`: kernel gate passed for a refreshed
  fixture-level Top FSM that instantiates real `layer_fsm_decoder_block_fixture`
  RTL and nested decoder-block children. It records the ordered top trace
  `layer_fsm_decoder_block_fixture_start`,
  `layer_fsm_decoder_block_fixture_done`; observes layer trace
  `decoder_block_attention_mlp_fixture_start`,
  `decoder_block_attention_mlp_fixture_done`; observes block trace
  `attention_start`, `attention_done`, `mlp_start`, `mlp_done`; observes nested
  attention and MLP traces through hierarchy; verifies top-to-layer
  start-hold/deassert/release protocol with `busy_cycles=329`; and computes
  final top output `[12, -6, 18, 6]`. It keeps the top-level interface compact
  at a 64-bit final output and 64-bit status, avoids exposing child vectors,
  child-wide status arrays, 128-bit memory responses, or KV/debug arrays, and
  preserves bonded IOB at 132. It records Verilator evidence and passes routed
  timing with setup WNS 1.265 ns, hold WHS 0.018 ns, pulse-width WPWS
  2.225 ns, 1504 LUTs, 843 registers, 8 DSPs, 0 BRAM, and 132 bonded IOBs. It
  is not real token prefill/decode, target sequence scheduling, target
  multi-layer LLaMA iteration, DDR/AXI movement, full LLaMA layer/model
  execution, or board-level signoff. Parent verification and read-only audit
  found no P0/P1/P2 issues. P3: fixture timing lacks board I/O delay
  constraints and hold margin is positive but modest for future composition.
- `token_loop_decoder_block_fixture`: kernel gate passed for a recovered
  fixture-level token-loop scheduler that instantiates real
  `top_fsm_decoder_block_fixture` RTL and calls it for exactly two
  deterministic token steps. It records the ordered token trace `token0_start`,
  `token0_done`, `token1_start`, `token1_done`; captures per-token top traces
  `0x5453`, layer traces `0x4241`, and block traces `0xb2b1a2a1`; observes
  nested attention trace `0x323122211211` and MLP trace
  `0x52514241323122211211` through captured child status and hierarchy for
  token 1; verifies per-token top-child start-hold/deassert/release protocol
  with busy cycles 334 and 333; and computes repeated deterministic token
  outputs `[12, -6, 18, 6]` plus final output `[12, -6, 18, 6]`. It keeps child
  vectors, residual/MLP vectors, 128-bit memory responses, KV arrays, and debug
  arrays internal, exposing a 64-bit output and recovered 64-bit compact status.
  Timing-margin recovery reduced status width from 96 to 64 bits and bonded
  IOB from 164 to 132. It records Verilator evidence and passes routed timing
  with setup WNS 1.216 ns, hold WHS 0.008 ns, pulse-width WPWS 2.225 ns, 1544
  LUTs, 908 registers, 8 DSPs, 0 BRAM, and 132 bonded IOBs. It is not real
  token prefill/decode, token-dependent KV-cache accumulation, target sequence
  scheduling, target multi-layer LLaMA iteration, DDR/AXI movement, full LLaMA
  layer/model execution, or board-level signoff. Parent verification and
  read-only audit found no P0/P1/P2 issues. P3: WHS improved from 0.002 ns but
  remains thin at 0.008 ns, so future token/layer/DDR/AXI/board-shell expansion
  should keep status narrow and track hold margin as a scaling risk.
- `decoder_block`: legacy scaffold remains non-target and should not be used
  for decoder integration.

## Delegation Order

1. `token_loop_decoder_block_fixture`:
   - gate passed; use it as the current highest refreshed scheduler fixture for
     decoder-block attention plus residual/MLP coverage.
2. Future target-scale scheduling:
   - only after timing margin is recovered, extend toward target sequence
     scheduling, target multi-layer scheduling, DDR/AXI movement, full model
     execution, and board signoff as separate independently verified claims;
   - keep top-level status at or below the recovered 64-bit width unless a new
     contract proves the extra evidence is worth the timing and I/O cost.

## Blocked Integrations

Full target Layer FSM, Top FSM, and token-loop agents are blocked until at
least:

- reports distinguish fixture coverage from target LLaMA coverage;
- refreshed decoder-block attention plus residual/MLP fixture coverage is
  available for token-loop integration;
- token-loop fixtures prove repeated child-call release waits without relying
  on stale child `done_o` signals;
- the current token-loop decoder-block WHS of 0.008 ns is treated as a scaling
  risk before additional composition, even though it is improved and passing.
