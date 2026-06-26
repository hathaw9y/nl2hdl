from nl2hdl.config import load_config
import pytest


def test_default_config_is_model_only_agent_ready():
    cfg = load_config(None)
    assert cfg.optimization.quantization == "int8_static"
    assert cfg.optimization.pruning == "none"
    assert cfg.optimization.optimization_brief is None
    assert cfg.optimization.optimization_candidates == []
    assert cfg.optimization.extra_options == {}
    assert cfg.design.style == "layer_fsm"
    assert cfg.design.compute_style == "scalar_fsm"
    assert cfg.design.execution_style == "layer_by_layer"
    assert cfg.design.memory_style == "onchip_weight_storage"
    assert cfg.design.control_style == "layer_fsm"
    assert cfg.design.architecture_brief is None
    assert cfg.design.design_candidates == []
    assert cfg.design.extra_options == {}
    assert cfg.hardware.max_ff is None
    assert cfg.hardware.device_ff is None
    assert cfg.model.input_shape == (1, 4)
    assert cfg.model.gptq_checkpoint is None
    assert cfg.model.mlir_graph is None
    assert cfg.model.model_structure_source == "mlir"


def test_accepts_free_form_optimization_method(tmp_path):
    config = tmp_path / "free_optimization.yaml"
    config.write_text(
        """
optimization:
  quantization: adaptive mixed INT4/INT8 with activation-aware calibration
  pruning: block sparse MLP pruning if accuracy audit passes
  optimization_brief: Compare GPTQ INT4, AWQ-style INT4, and mixed precision per projection.
  optimization_candidates:
    - name: gptq_int4_weight_only
      priority: first
    - name: mixed_precision_attention
      priority: fallback
  calibration_dataset: small prompt set
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.optimization.quantization == "adaptive mixed INT4/INT8 with activation-aware calibration"
    assert cfg.optimization.pruning == "block sparse MLP pruning if accuracy audit passes"
    assert cfg.optimization.optimization_candidates[0]["name"] == "gptq_int4_weight_only"
    assert cfg.optimization.extra_options["calibration_dataset"] == "small prompt set"


def test_accepts_llama_zcu104_planning_config():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    assert cfg.hardware.fpga_part == "xczu7ev-ffvc1156-2-e"
    assert cfg.hardware.max_lut == 230400
    assert cfg.hardware.max_ff == 460800
    assert cfg.hardware.max_dsp == 1728
    assert cfg.hardware.max_bram == 312
    assert cfg.hardware.max_uram == 96
    assert cfg.hardware.max_io == 464
    assert cfg.hardware.device_logic_cells == 504000
    assert cfg.hardware.device_lut == 230400
    assert cfg.hardware.device_ff == 460800
    assert cfg.hardware.device_dsp == 1728
    assert cfg.hardware.device_bram_36k == 312
    assert cfg.hardware.device_uram == 96
    assert cfg.hardware.device_io == 464
    assert cfg.hardware.device_distributed_ram_mb == 6.2
    assert cfg.hardware.device_bram_mb == 11.0
    assert cfg.hardware.device_uram_mb == 27.0
    assert cfg.optimization.quantization == "int4_gptq"
    assert cfg.optimization.optimization_brief is not None
    assert cfg.optimization.optimization_candidates
    assert cfg.design.style == "llm_decoder_streaming"
    assert cfg.design.compute_style == "simd_vector_mac"
    assert cfg.design.execution_style == "llm_decoder_streaming"
    assert cfg.design.memory_style == "external_ddr_gptq_packed"
    assert cfg.design.control_style == "hierarchical_fsm"
    assert cfg.design.architecture_brief is not None
    assert cfg.design.design_candidates
    assert cfg.model.gptq_checkpoint is None
    assert cfg.model.mlir_graph is None
    assert cfg.model.model_structure_source == "mlir"


def test_accepts_hf_config_model_structure_source(tmp_path):
    config = tmp_path / "hf_config_structure.yaml"
    config.write_text(
        """
model:
  model_structure_source: hf_config
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.model.model_structure_source == "hf_config"


def test_rejects_unknown_model_structure_source(tmp_path):
    config = tmp_path / "bad_structure_source.yaml"
    config.write_text(
        """
model:
  model_structure_source: handwritten_guess
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.model_structure_source"):
        load_config(config)


def test_legacy_design_style_populates_execution_alias(tmp_path):
    config = tmp_path / "legacy.yaml"
    config.write_text(
        """
design:
  style: llm_decoder_streaming
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.design.style == "llm_decoder_streaming"
    assert cfg.design.execution_style == "llm_decoder_streaming"


def test_accepts_split_design_taxonomy(tmp_path):
    config = tmp_path / "split_design.yaml"
    config.write_text(
        """
design:
  compute_style: systolic_array
  execution_style: prefill_decode_split
  memory_style: external_ddr_streaming
  control_style: microcoded_controller
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.design.compute_style == "systolic_array"
    assert cfg.design.execution_style == "prefill_decode_split"
    assert cfg.design.memory_style == "external_ddr_streaming"
    assert cfg.design.control_style == "microcoded_controller"


def test_accepts_free_form_design_content_and_extra_options(tmp_path):
    config = tmp_path / "free_design.yaml"
    config.write_text(
        """
design:
  style: exploratory multi-kernel accelerator
  compute_style: hybrid tensor-cloud inspired systolic/SIMD fabric
  execution_style: speculative token streaming with prefill/decode split
  memory_style: compressed external-memory stream with KV-cache locality hints
  control_style: event-driven microcoded scheduler with hierarchical FSM fallback
  architecture_brief: Let the parent agent compare several designs before choosing module packets.
  design_candidates:
    - name: systolic_projection_array
      risk: routing pressure
    - name: simd_vector_mac_stream
      risk: memory bandwidth
  novel_axis: allow parent to preserve user-defined design metadata
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.design.compute_style == "hybrid tensor-cloud inspired systolic/SIMD fabric"
    assert cfg.design.execution_style == "speculative token streaming with prefill/decode split"
    assert cfg.design.design_candidates[1]["name"] == "simd_vector_mac_stream"
    assert cfg.design.extra_options["novel_axis"] == "allow parent to preserve user-defined design metadata"


def test_rejects_empty_free_form_design_axis(tmp_path):
    config = tmp_path / "bad_design.yaml"
    config.write_text(
        """
design:
  compute_style: ""
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="design.compute_style"):
        load_config(config)


def test_accepts_optional_gptq_checkpoint_source(tmp_path):
    config = tmp_path / "gptq_source.yaml"
    config.write_text(
        """
model:
  input_shape: [1, 1]
  gptq_checkpoint: /tmp/local-gptq
optimization:
  quantization: int4_gptq
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.model.gptq_checkpoint == "/tmp/local-gptq"


def test_rejects_empty_gptq_checkpoint_source(tmp_path):
    config = tmp_path / "bad_gptq_source.yaml"
    config.write_text(
        """
model:
  gptq_checkpoint: ""
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.gptq_checkpoint"):
        load_config(config)


def test_accepts_optional_mlir_graph_source(tmp_path):
    mlir_path = tmp_path / "model_graph.mlir"
    mlir_path.write_text("module {}\n", encoding="utf-8")
    config = tmp_path / "mlir_source.yaml"
    config.write_text(
        f"""
model:
  input_shape: [1, 1]
  mlir_graph: {mlir_path}
optimization:
  quantization: int4_gptq
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg.model.mlir_graph == str(mlir_path)


def test_rejects_empty_mlir_graph_source(tmp_path):
    config = tmp_path / "bad_mlir_source.yaml"
    config.write_text(
        """
model:
  mlir_graph: ""
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.mlir_graph"):
        load_config(config)
