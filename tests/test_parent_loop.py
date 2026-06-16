from __future__ import annotations

from pathlib import Path
import json

from nl2hdl.cli import main


def _run_parent_loop(tmp_path: Path, *, max_iterations: int) -> Path:
    out_dir = tmp_path / f"parent_loop_{max_iterations}"
    rc = main(
        [
            "parent-loop",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--spec",
            "examples/zcu104_llama32_1b_gptq.yaml",
            "--out",
            str(out_dir),
            "--max-iterations",
            str(max_iterations),
            "--max-subagents-per-iteration",
            "1",
            "--skip-synth",
            "--skip-vivado-route",
        ]
    )
    assert rc == 0
    return out_dir


def test_parent_loop_cli_executes_one_local_subagent(tmp_path: Path):
    out_dir = _run_parent_loop(tmp_path, max_iterations=1)

    report = json.loads((out_dir / "parent_loop_run_report.json").read_text(encoding="utf-8"))
    execution = json.loads((out_dir / "status" / "hdl_subagent_execution_manifest.json").read_text(encoding="utf-8"))
    subagent_result = json.loads(
        (out_dir / "evidence" / "projection_q_proj_gate" / "subagent_result.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "ready_to_continue"
    assert report["parent_must_not_write_hdl"] is True
    assert report["iterations"][0]["executed"][0]["backend"] == "local_subagent"
    assert execution["spawn_entries"][0]["requires_module_ooc_synthesis"] is True
    assert subagent_result["status"] == "passed"
    assert subagent_result["subagent_may_spawn_subagents"] is False
    assert subagent_result["module_ooc_synthesis_evidence"] == {
        "passed": False,
        "source": "not_run",
    }


def test_parent_loop_queues_ooc_when_skip_synth_after_kernel_evidence(tmp_path: Path):
    out_dir = _run_parent_loop(tmp_path, max_iterations=2)

    report = json.loads((out_dir / "parent_loop_run_report.json").read_text(encoding="utf-8"))
    queue = json.loads((out_dir / "status" / "parent_loop_queue.json").read_text(encoding="utf-8"))

    assert report["status"] == "queued_external_subagents"
    assert report["iterations"][1]["executed"] == []
    assert queue["status"] == "queued_external_subagents"
    assert queue["entries"][0]["reason"] == "module_ooc_synthesis_required_but_skip_synth_enabled"
    assert queue["entries"][0]["task_id"] == "projection_q_proj"
