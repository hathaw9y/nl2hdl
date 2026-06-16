from pathlib import Path

from nl2hdl.agent import run_agent
from nl2hdl.config import load_config


def test_builtin_tiny_mlp_generates_rtl(tmp_path: Path):
    cfg = load_config(None)
    report = run_agent("builtin:tiny_mlp", cfg, tmp_path, skip_synth=True)
    assert report["status"] == "passed"
    assert (tmp_path / "model_top.sv").exists()
    assert (tmp_path / "tb_model_top.sv").exists()
    assert "PASS" in (tmp_path / "simulation.log").read_text(encoding="utf-8")


def test_generated_rtl_uses_configured_pe_count_and_layer_fsm(tmp_path: Path):
    cfg = load_config(None)
    report = run_agent("builtin:tiny_mlp", cfg, tmp_path, skip_synth=True)
    assert report["status"] == "passed"
    dense = (tmp_path / "dense_layer_0.sv").read_text(encoding="utf-8")
    top = (tmp_path / "model_top.sv").read_text(encoding="utf-8")
    assert f"parameter int PE_COUNT = {cfg.design.pe_count}" in dense
    assert "START_LAYER_0" in top
    assert "WAIT_LAYER_0" in top
    assert ".start_i(layer_0_start_w)" in top
    assert ".done_o(layer_0_done_w)" in top
