from pathlib import Path
import json

from nl2hdl.config import load_config
from nl2hdl.llm_agent import run_llm_agent
from nl2hdl.llm_plan import write_llm_accelerator_plan


def test_writes_llama_zcu104_plan(tmp_path: Path):
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    plan = write_llm_accelerator_plan("meta-llama/Llama-3.2-1B", cfg, tmp_path)
    assert plan["optimization"]["quantization"] == "int4_gptq"
    assert plan["input_clarification"]["status"] == "clear"
    assert plan["input_clarification"]["requires_user_response"] is False
    assert plan["model"]["gptq_checkpoint_source"] == "meta-llama/Llama-3.2-1B"
    assert plan["model"]["gptq_checkpoint_source_kind"] == "same_as_model_name"
    assert plan["model"]["mlir_graph_source"] is None
    assert plan["model"]["mlir_graph_source_kind"] == "synthetic_or_export_required"
    assert any("GEMM and non-GEMM" in step for step in plan["agent_pipeline"])
    assert "llm_accelerator_plan.json" in {path.name for path in tmp_path.iterdir()}
    md = (tmp_path / "llm_accelerator_plan.md").read_text(encoding="utf-8")
    assert "INT4 GPTQ unpack" in md
    assert "GPTQ metadata source" in md
    assert "GPTQ metadata source kind" in md
    assert "MLIR graph source" in md
    assert "MLIR graph source kind" in md
    assert "Input Clarification" in md
    clarification = json.loads((tmp_path / "input_clarification_questions.json").read_text(encoding="utf-8"))
    assert clarification["status"] == "clear"


def test_plan_records_configured_gptq_checkpoint_source(tmp_path: Path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """
model:
  input_shape: [1, 1]
  gptq_checkpoint: /tmp/local-gptq
optimization:
  quantization: int4_gptq
design:
  style: llm_decoder_streaming
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    plan = write_llm_accelerator_plan("meta-llama/Llama-3.2-1B", cfg, tmp_path)
    assert plan["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert plan["model"]["gptq_checkpoint_source"] == "/tmp/local-gptq"
    assert plan["model"]["gptq_checkpoint_source_kind"] == "configured_override"


def test_plan_records_configured_mlir_graph_source(tmp_path: Path):
    mlir_path = tmp_path / "model_graph.mlir"
    mlir_path.write_text("module {}\n", encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"""
model:
  input_shape: [1, 1]
  mlir_graph: {mlir_path}
optimization:
  quantization: int4_gptq
design:
  style: llm_decoder_streaming
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    plan = write_llm_accelerator_plan("meta-llama/Llama-3.2-1B", cfg, tmp_path)
    assert plan["model"]["mlir_graph_source"] == str(mlir_path)
    assert plan["model"]["mlir_graph_source_kind"] == "configured_override"
    md = (tmp_path / "llm_accelerator_plan.md").read_text(encoding="utf-8")
    assert str(mlir_path) in md


def test_plan_asks_for_clarification_when_free_form_methods_are_ambiguous(tmp_path: Path):
    spec = tmp_path / "ambiguous.yaml"
    spec.write_text(
        """
model:
  input_shape: [1, 1]
optimization:
  quantization: custom best quantization
  pruning: custom sparse pruning
design:
  style: custom accelerator
  compute_style: custom compute
  execution_style: auto
  memory_style: custom memory
  control_style: custom control
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    plan = write_llm_accelerator_plan("meta-llama/Llama-3.2-1B", cfg, tmp_path)
    clarification = plan["input_clarification"]

    assert clarification["status"] == "needs_clarification"
    assert clarification["requires_user_response"] is True
    question_ids = {question["id"] for question in clarification["questions"]}
    assert "clarify_quantization_method" in question_ids
    assert "clarify_pruning_or_sparsity_method" in question_ids
    assert "clarify_hardware_design_methodology" in question_ids
    assert (tmp_path / "input_clarification_questions.json").exists()
    md = (tmp_path / "llm_accelerator_plan.md").read_text(encoding="utf-8")
    assert "clarify_quantization_method" in md


def test_llm_agent_stops_before_dispatch_when_clarification_is_required(tmp_path: Path):
    spec = tmp_path / "ambiguous_agent.yaml"
    spec.write_text(
        """
model:
  input_shape: [1, 1]
optimization:
  quantization: custom best quantization
  pruning: none
design:
  style: custom accelerator
  compute_style: custom compute
  execution_style: auto
  memory_style: custom memory
  control_style: custom control
""",
        encoding="utf-8",
    )
    cfg = load_config(spec)
    report = run_llm_agent(
        model_name="meta-llama/Llama-3.2-1B",
        config=cfg,
        out_dir=tmp_path / "agent",
        mode="inspect",
        kernel=None,
        partition="gemm_non_gemm",
        skip_synth=True,
    )

    assert report["status"] == "needs_clarification"
    assert report["input_clarification"]["requires_user_response"] is True
    assert (tmp_path / "agent" / "input_clarification_questions.json").exists()
    assert not (tmp_path / "agent" / "hdl_subagent_dispatch_plan.json").exists()
