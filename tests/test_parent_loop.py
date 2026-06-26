from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import nl2hdl.parent_loop as parent_loop_module
from nl2hdl.cli import main
from nl2hdl.config import load_config
from nl2hdl.parent_loop import (
    ParentLoopOptions,
    _archive_ooc_tuning_attempt,
    _board_wrapper_bitstream_summary,
    _config_with_ooc_tuning,
    _existing_module_ooc_tuning_blocker,
    _inspect_artifacts_match_inputs,
    _import_board_wrapper_evidence,
    _load_ooc_tuning_history,
    _module_ooc_report_from_kernel_result,
    _module_ooc_tuning_effectiveness,
    _run_local_implementation_entry,
    _run_local_or_queue_entry,
    _run_local_verification_entry,
    _target_preflight_queue_entries,
)
from nl2hdl.subagent_tasks import build_hdl_subagent_wave_status


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


def test_parent_loop_bitstream_summary_reports_board_wrapper_scope(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    gate_dir = evidence_root / "board_zcu104_signoff_gate"
    gate_dir.mkdir(parents=True)
    bitstream = gate_dir / "zcu104_board_wrapper.bit"
    bitstream.write_bytes(b"fake-bitstream")
    report = {
        "status": "passed",
        "bitstream_generated": True,
        "bitstream_file": str(bitstream),
        "bitstream_size_bytes": bitstream.stat().st_size,
        "route_completed": True,
        "route_check_command_passed": True,
        "route_report_analysis": {
            "clock": {"implemented_period_ns": 5.0, "implemented_frequency_mhz": 200.0},
            "timing": {"setup_worst_slack_ns": 1.773, "hold_worst_slack_ns": 0.010},
            "utilization": {"lut": 464, "ff": 706, "dsp": 0, "bram": 0},
            "gate_failures": [],
        },
    }
    (gate_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = _board_wrapper_bitstream_summary(evidence_root)

    assert summary["generated"] is True
    assert summary["bitstream_file"] == str(bitstream)
    assert summary["bitstream_size_bytes"] == len(b"fake-bitstream")
    assert summary["clock"]["implemented_period_ns"] == 5.0
    assert summary["scope"] == "zcu104_board_wrapper_control_scaffold"
    assert summary["target_scale_accelerator_bitstream"] is False
    assert summary["not_full_llama_accelerator"] is True


def test_parent_loop_bitstream_summary_accepts_target_scale_scope(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    gate_dir = evidence_root / "board_zcu104_signoff_gate"
    gate_dir.mkdir(parents=True)
    bitstream = gate_dir / "zcu104_board_wrapper.bit"
    bitstream.write_bytes(b"fake-target-scale-bitstream")
    report = {
        "status": "passed",
        "bitstream_generated": True,
        "bitstream_file": str(bitstream),
        "bitstream_size_bytes": bitstream.stat().st_size,
        "target_scale_accelerator_bitstream": True,
        "accelerator_scope": "full_target_llama_accelerator",
        "route_completed": True,
        "route_check_command_passed": True,
        "route_report_analysis": {
            "clock": {"implemented_period_ns": 5.0, "implemented_frequency_mhz": 200.0},
            "timing": {"setup_worst_slack_ns": 1.773, "hold_worst_slack_ns": 0.010},
            "utilization": {"lut": 120000, "ff": 180000, "dsp": 900, "bram": 180},
            "gate_failures": [],
        },
    }
    (gate_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = _board_wrapper_bitstream_summary(evidence_root)

    assert summary["generated"] is True
    assert summary["scope"] == "full_target_llama_accelerator"
    assert summary["target_scale_accelerator_bitstream"] is True
    assert summary["not_full_llama_accelerator"] is False
    assert "full target-scale LLaMA accelerator bitstream" not in summary["does_not_claim"]


def test_parent_loop_selects_strongest_board_wrapper_accelerator_artifact(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    gate = evidence_root / "ddr_axi_board_shell_fixture_gate"
    gate.mkdir(parents=True)
    (gate / "ddr_axi_board_shell_fixture.sv").write_text("module ddr_axi_board_shell_fixture; endmodule\n")
    (gate / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "kernel": "ddr_axi_board_shell_fixture",
                "coverage_level": "ddr_axi_board_shell_fixture",
                "numeric_policy": {
                    "full_llama_model": False,
                    "board_level_signoff": False,
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = parent_loop_module._select_board_wrapper_accelerator_artifact(evidence_root)

    assert artifact is not None
    assert artifact["artifact_dir"] == str(gate)
    assert artifact["kernel_report"] == str(gate / "kernel_report.json")
    assert artifact["top_module"] == "ddr_axi_board_shell_fixture"
    assert artifact["target_scale_eligible"] is False


def test_parent_loop_imports_existing_board_wrapper_evidence(tmp_path: Path):
    source_dir = tmp_path / "source_board_wrapper"
    source_dir.mkdir()
    for name in (
        "vivado.log",
        "zcu104_timing_summary.rpt",
        "zcu104_utilization.rpt",
        "zcu104_drc.rpt",
        "zcu104_methodology.rpt",
        "zcu104_clocks.rpt",
        "zcu104_implemented_constraints.xdc",
        "zcu104_post_route.dcp",
        "zcu104_board_route_check.tcl",
        "zcu104_board_wrapper_axi_bridge_subagent_result.json",
    ):
        (source_dir / name).write_text(f"{name}\n", encoding="utf-8")
    (source_dir / "zcu104_board_wrapper.bit").write_bytes(b"fake-bitstream")
    report = {
        "artifact": "zcu104_board_wrapper_axi_bridge_implementation_report",
        "status": "passed",
        "bitstream_generated": True,
        "bitstream_file": str(source_dir / "zcu104_board_wrapper.bit"),
        "bitstream_size_bytes": (source_dir / "zcu104_board_wrapper.bit").stat().st_size,
        "route_completed": True,
        "route_check_command_passed": True,
        "route_report_analysis": {
            "clock": {"observed_period_ns": 5.0, "target_period_ns": 5.0},
            "timing": {"constraints_met": True},
            "utilization": {"lut": 464, "ff": 706, "dsp": 0, "bram": 0},
            "gate_failures": [],
        },
        "evidence_files": {
            "implementation_report": str(source_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"),
            "subagent_result": str(source_dir / "zcu104_board_wrapper_axi_bridge_subagent_result.json"),
            "route_tcl": str(source_dir / "zcu104_board_route_check.tcl"),
            "vivado_log": str(source_dir / "vivado.log"),
            "timing_summary": str(source_dir / "zcu104_timing_summary.rpt"),
            "utilization": str(source_dir / "zcu104_utilization.rpt"),
            "drc": str(source_dir / "zcu104_drc.rpt"),
            "methodology": str(source_dir / "zcu104_methodology.rpt"),
            "clocks": str(source_dir / "zcu104_clocks.rpt"),
            "implemented_constraints": str(source_dir / "zcu104_implemented_constraints.xdc"),
            "checkpoint": str(source_dir / "zcu104_post_route.dcp"),
            "bitstream": str(source_dir / "zcu104_board_wrapper.bit"),
        },
    }
    (source_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    evidence_root = tmp_path / "parent_evidence"
    evidence_root.mkdir()
    (evidence_root / "board_zcu104_signoff_evidence.json").write_text(
        json.dumps({"artifact": "board_zcu104_signoff_evidence", "status": "passed"}),
        encoding="utf-8",
    )
    status_dir = tmp_path / "status"
    result = _import_board_wrapper_evidence(source_dir, evidence_root, status_dir)
    summary = _board_wrapper_bitstream_summary(evidence_root)
    imported_report = json.loads(
        (
            evidence_root
            / "board_zcu104_signoff_gate"
            / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "passed"
    assert result["bitstream_imported"] is True
    assert summary["generated"] is True
    assert summary["bitstream_file"].endswith("parent_evidence/board_zcu104_signoff_gate/zcu104_board_wrapper.bit")
    assert imported_report["bitstream_file"] == summary["bitstream_file"]
    assert Path(imported_report["evidence_files"]["checkpoint"]).exists()
    assert (status_dir / "board_wrapper_evidence_import.json").exists()
    assert not (evidence_root / "board_zcu104_signoff_evidence.json").exists()
    assert result["invalidated_board_signoff_evidence"] == str(
        status_dir / "invalidated_evidence" / "board_zcu104_signoff_evidence_after_board_wrapper_import.json"
    )
    assert Path(result["invalidated_board_signoff_evidence"]).exists()


def test_parent_loop_module_ooc_report_marks_underutilized_for_tuning():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    report = _module_ooc_report_from_kernel_result(
        {"task_id": "projection_q_proj"},
        cfg,
        {
            "implementation_stage": "post-route",
            "synthesis": {
                "passed": True,
                "timing_report": "timing_summary.rpt",
                "utilization_report": "utilization.rpt",
                "timing": {
                    "constraints_met": True,
                    "setup_worst_slack_ns": 0.5,
                    "hold_worst_slack_ns": 0.04,
                    "pulse_width_worst_slack_ns": 2.2,
                },
                "resource_utilization": {
                    "lut_as_logic": {"used": 1183},
                    "clb_registers": {"used": 992},
                    "block_ram_tile": {"used": 0},
                    "dsps": {"used": 6},
                    "uram": {"used": 0},
                    "bonded_iob": {"used": 282},
                },
            },
        },
    )

    assert report["status"] == "passed"
    assert report["resource_assessment"] == "underutilized"
    assert report["throughput_target_met"] is False
    assert report["tuning_recommendation"]["required"] is True
    assert report["tuning_recommendation"]["suggested_next_knobs"]["pe_count"] > cfg.design.pe_count


def test_parent_loop_module_ooc_report_stops_pe_tuning_when_fixture_lane_headroom_is_exhausted():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    report = _module_ooc_report_from_kernel_result(
        {"task_id": "projection_q_proj"},
        cfg,
        {
            "implementation_stage": "post-route",
            "lane_policy": {
                "requested_pe_lanes": 64,
                "selected_target_planning_lanes": 64,
                "true_parallel_datapath_lanes": 64,
                "max_fixture_true_parallel_lanes": 64,
                "parallel_products_per_cycle": 64,
                "pe_count_controls_true_parallel_datapath": True,
                "pe_count_headroom_in_fixture": False,
            },
            "synthesis": {
                "passed": True,
                "timing": {"constraints_met": True},
                "resource_utilization": {
                    "lut_as_logic": {"used": 4000},
                    "clb_registers": {"used": 3000},
                    "block_ram_tile": {"used": 0},
                    "dsps": {"used": 128},
                    "uram": {"used": 0},
                    "bonded_iob": {"used": 32},
                },
            },
        },
    )

    assert report["resource_assessment"] == "underutilized"
    assert report["datapath_parallelism"]["available"] is True
    assert report["datapath_parallelism"]["true_parallel_datapath_lanes"] == 64
    assert report["throughput_target_met"] is True
    assert "bounded module packet" in report["throughput_target_basis"]
    assert report["tuning_recommendation"]["required"] is False
    assert report["tuning_recommendation"]["suggested_next_knobs"]["pe_count"] == cfg.design.pe_count
    assert "fixture's true parallel lane limit" in report["tuning_recommendation"]["blocked_knob_reason"]


def test_parent_loop_module_ooc_report_accepts_non_gemm_fixture_control_scaffold():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    report = _module_ooc_report_from_kernel_result(
        {"task_id": "non_gemm_input_layernorm"},
        cfg,
        {
            "implementation_stage": "post-route",
            "coverage_level": "rmsnorm_rope_source_path_fixture",
            "does_not_claim": [
                "full_RMSNorm_reciprocal_sqrt_datapath",
                "full_decoder_block",
                "full_model_execution",
            ],
            "synthesis": {
                "passed": True,
                "timing": {
                    "constraints_met": True,
                    "setup_worst_slack_ns": 1.6,
                    "hold_worst_slack_ns": 0.05,
                    "pulse_width_worst_slack_ns": 2.2,
                },
                "resource_utilization": {
                    "lut_as_logic": {"used": 106},
                    "clb_registers": {"used": 205},
                    "block_ram_tile": {"used": 0},
                    "dsps": {"used": 4},
                    "uram": {"used": 0},
                    "bonded_iob": {"used": 234},
                },
            },
        },
    )

    assert report["resource_assessment"] == "fixture_control_scaffold"
    assert report["throughput_target_met"] is True
    assert "bounded fixture/control scaffold" in report["throughput_target_basis"]
    assert report["tuning_recommendation"]["required"] is False
    assert "without resource-saturation tuning" in report["tuning_recommendation"]["reason"]


def test_parent_loop_applies_ooc_tuning_recommendation_to_allowed_knobs():
    cfg = load_config("examples/zcu104_llama32_1b_gptq.yaml")
    tuned, change = _config_with_ooc_tuning(
        cfg,
        {
            "selected_tuning_knobs": {
                "pe_count": cfg.design.pe_count * 2,
                "memory_data_width": cfg.hardware.memory_data_width,
                "activation_buffer": cfg.design.activation_buffer,
                "weight_storage": cfg.design.weight_storage,
            },
            "tuning_recommendation": {
                "suggested_next_knobs": {
                    "pe_count": cfg.design.pe_count * 4,
                    "memory_data_width": cfg.hardware.memory_data_width,
                    "activation_buffer": cfg.design.activation_buffer,
                    "weight_storage": cfg.design.weight_storage,
                }
            }
        },
    )

    assert change["changed"] is True
    assert change["current_knobs"]["pe_count"] == cfg.design.pe_count * 2
    assert change["updated_knobs"]["pe_count"] == cfg.design.pe_count * 4
    assert tuned.design.pe_count == cfg.design.pe_count * 4
    assert tuned.hardware.memory_data_width == cfg.hardware.memory_data_width


def test_parent_loop_archives_ooc_evidence_before_tuning(tmp_path: Path):
    evidence_dir = tmp_path / "projection_q_proj_gate"
    evidence_dir.mkdir()
    (evidence_dir / "kernel_report.json").write_text('{"status":"passed"}', encoding="utf-8")
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        '{"status":"passed","resource_assessment":"underutilized","throughput_target_met":false}',
        encoding="utf-8",
    )
    (evidence_dir / "subagent_result.json").write_text('{"status":"passed"}', encoding="utf-8")

    archive_dir = _archive_ooc_tuning_attempt(
        evidence_dir,
        1,
        {"status": "passed", "resource_assessment": "underutilized"},
    )
    history = _load_ooc_tuning_history(evidence_dir)

    assert archive_dir.name == "attempt_01_before_tuning"
    assert (archive_dir / "kernel_report.json").exists()
    assert (archive_dir / "module_ooc_synthesis_report.json").exists()
    assert (archive_dir / "source_tuning_blocker.json").exists()
    assert history["artifact"] == "module_ooc_tuning_history"
    assert history["attempts"] == []


def test_parent_loop_ignores_stale_underutilized_fixture_control_scaffold_blocker(tmp_path: Path):
    evidence_dir = tmp_path / "non_gemm_input_layernorm_gate"
    evidence_dir.mkdir()
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "resource_assessment": "underutilized",
                "throughput_target_met": False,
            }
        ),
        encoding="utf-8",
    )
    (evidence_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "coverage_level": "rmsnorm_rope_source_path_fixture",
                "does_not_claim": ["full_decoder_block", "full_model_execution"],
            }
        ),
        encoding="utf-8",
    )

    assert _existing_module_ooc_tuning_blocker(evidence_dir) is None


def test_parent_loop_marks_ooc_tuning_ineffective_when_resources_do_not_change():
    previous = {
        "utilization": {
            "lut": 1183,
            "ff": 992,
            "dsp": 6,
            "bram": 0,
            "uram": 0,
            "io": 282,
        }
    }
    current = {
        "utilization": {
            "lut": 1183,
            "ff": 992,
            "dsp": 6,
            "bram": 0,
            "uram": 0,
            "io": 282,
        }
    }

    result = _module_ooc_tuning_effectiveness(
        previous,
        current,
        {
            "changed": True,
            "current_knobs": {"pe_count": 64},
            "updated_knobs": {"pe_count": 128},
        },
    )

    assert result["effective"] is False
    assert result["changed_resources"] == {}
    assert result["resource_deltas"]["dsp"] == 0
    assert result["skill_update_candidate"]["prevention_rule"].startswith(
        "When a sub-agent tunes pe_count"
    )


def test_parent_loop_queues_generator_fix_after_ineffective_ooc_tuning(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    evidence_dir = evidence_root / "projection_q_proj_gate"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "module_ooc_synthesis_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "resource_assessment": "underutilized",
                "throughput_target_met": False,
                "tuning_effectiveness": {
                    "effective": False,
                    "reason": "applied OOC tuning knobs did not change Vivado resource utilization",
                    "skill_update_candidate": {
                        "prevention_rule": "connect tuned pe_count to true RTL lanes"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    entry = {
        "spawn_kind": "implementation_agent",
        "spawn_key": "projection_q_proj_gate",
        "task_id": "projection_q_proj",
        "wave_id": "kernel_projection",
        "current_regression_kernel": "projection",
        "expected_evidence_dir": "build/projection_q_proj_gate",
    }

    result = _run_local_implementation_entry(
        entry,
        config,
        evidence_root,
        ParentLoopOptions(auto_tune_ooc=True, max_ooc_tuning_attempts=3),
    )

    assert result["parent_loop_action"] == "queued"
    assert result["reason"] == "module_ooc_tuning_ineffective_requires_generator_fix"
    assert result["completed_ooc_tuning_attempts"] == 0
    assert result["skill_update_candidate"]["prevention_rule"].startswith("connect tuned")


def test_parent_loop_local_integration_verification_writes_synthesis_report(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    gate_dir = evidence_root / "decoder_block_attention_mlp_fixture_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "timing_summary.rpt").write_text(
        """
        Setup : 0 Failing Endpoints, Worst Slack 1.057ns
        Hold : 0 Failing Endpoints, Worst Slack 0.005ns
        PW : 0 Failing Endpoints, Worst Slack 2.225ns
        All user specified timing constraints are met.
        """,
        encoding="utf-8",
    )
    (gate_dir / "utilization.rpt").write_text("utilization\n", encoding="utf-8")
    (gate_dir / "vivado.log").write_text("route_design completed successfully\n", encoding="utf-8")
    (gate_dir / "post_route.dcp").write_text("checkpoint\n", encoding="utf-8")
    (gate_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "coverage_level": "decoder_block_attention_mlp_fixture",
                "implementation_stage": "post-route",
                "files": [
                    "rmsnorm_rope_source_path.sv",
                    "projection_internal_stream_shell.sv",
                    "decoder_block_attention_mlp_fixture.sv",
                    "tb_decoder_block_attention_mlp_fixture.sv",
                ],
                "simulation": {"passed": True},
                "verilator": {"passed": True},
                "synthesis": {
                    "passed": True,
                    "cmd": ["vivado", "-mode", "batch", "-source", "vivado_synth.tcl"],
                    "timing_report": "timing_summary.rpt",
                    "utilization_report": "utilization.rpt",
                    "timing": {
                        "constraints_met": True,
                        "setup_worst_slack_ns": 1.057,
                        "hold_worst_slack_ns": 0.005,
                        "pulse_width_worst_slack_ns": 2.225,
                    },
                    "resource_utilization": {
                        "lut_as_logic": {"used": 1446},
                        "clb_registers": {"used": 743},
                        "block_ram_tile": {"used": 0},
                        "dsps": {"used": 8},
                        "uram": {"used": 0},
                        "bonded_iob": {"used": 228},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (gate_dir / "subagent_result.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    entry = {
        "spawn_key": "wave_2_decoder_block::verification::wave_2_decoder_block",
        "spawn_kind": "verification_agent",
        "wave_id": "wave_2_decoder_block",
        "verification_report": "verification_results/wave_2_decoder_block__verification.json",
        "runs_integration_synthesis": True,
        "expected_integration_synthesis_dir": "build/wave_2_decoder_block_integration_verification",
        "expected_integration_synthesis_report": (
            "build/wave_2_decoder_block_integration_verification/integration_synthesis_report.json"
        ),
        "implementation_tasks": [
            {
                "task_id": "decoder_block_attention_mlp_fixture",
                "expected_evidence_dir": "build/decoder_block_attention_mlp_fixture_gate",
            }
        ],
    }

    result = _run_local_verification_entry(entry, cfg, evidence_root)
    verification = json.loads(
        (evidence_root / "verification_results" / "wave_2_decoder_block__verification.json").read_text(
            encoding="utf-8"
        )
    )
    integration = json.loads(
        (
            evidence_root
            / "wave_2_decoder_block_integration_verification"
            / "integration_synthesis_report.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "passed"
    assert result["backend"] == "local_integration_verification"
    assert verification["status"] == "passed"
    assert verification["integration_synthesis_status"] == "passed"
    assert integration["status"] == "passed"
    assert integration["timing"]["setup_failing_endpoints"] == 0
    assert integration["timing"]["hold_failing_endpoints"] == 0
    assert integration["timing"]["pulse_width_failing_endpoints"] == 0
    assert integration["utilization"]["dsp"] == 8
    assert "full LLaMA execution" in integration["does_not_claim"]


def test_parent_loop_queues_integration_verification_when_skip_synth_missing_child_synthesis(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    gate_dir = evidence_root / "projection_down_proj_axi_read_data_channel_adapter_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "coverage_level": "projection_axi_read_data_channel_adapter_fixture",
                "implementation_stage": "rtl_sim_only",
                "files": [
                    "projection_axi_read_data_channel_adapter.sv",
                    "tb_projection_axi_read_data_channel_adapter.sv",
                ],
                "simulation": {"passed": True},
                "verilator": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    (gate_dir / "subagent_result.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    entry = {
        "spawn_key": "wave_7_projection_axi_read_data_channel_adapter::verification::wave_7",
        "spawn_kind": "verification_agent",
        "wave_id": "wave_7_projection_axi_read_data_channel_adapter",
        "verification_report": "verification_results/wave_7_projection_axi_read_data_channel_adapter__verification.json",
        "runs_integration_synthesis": True,
        "implementation_tasks": [
            {
                "task_id": "projection_down_proj_axi_read_data_channel_adapter",
                "expected_evidence_dir": "build/projection_down_proj_axi_read_data_channel_adapter_gate",
            }
        ],
    }

    result = _run_local_or_queue_entry(
        entry,
        cfg,
        evidence_root,
        ParentLoopOptions(local_verification=True, skip_synth=True),
    )

    assert result["parent_loop_action"] == "queued"
    assert result["reason"] == "integration_synthesis_required_but_skip_synth_enabled"
    assert result["missing_synthesis_task_ids"] == ["projection_down_proj_axi_read_data_channel_adapter"]
    assert not (evidence_root / "verification_results" / "wave_7_projection_axi_read_data_channel_adapter__verification.json").exists()


def test_parent_loop_reruns_child_synthesis_before_local_integration_verification(
    tmp_path: Path,
    monkeypatch,
):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    gate_dir = evidence_root / "projection_down_proj_axi_read_data_channel_adapter_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "coverage_level": "projection_axi_read_data_channel_adapter_fixture",
                "implementation_stage": "rtl_sim_only",
                "files": [
                    "projection_axi_read_data_channel_adapter.sv",
                    "tb_projection_axi_read_data_channel_adapter.sv",
                ],
                "simulation": {"passed": True},
                "verilator": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    (gate_dir / "subagent_result.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    entry = {
        "spawn_key": "wave_7_projection_axi_read_data_channel_adapter::verification::wave_7",
        "spawn_kind": "verification_agent",
        "wave_id": "wave_7_projection_axi_read_data_channel_adapter",
        "verification_report": "verification_results/wave_7_projection_axi_read_data_channel_adapter__verification.json",
        "runs_integration_synthesis": True,
        "implementation_tasks": [
            {
                "task_id": "projection_down_proj_axi_read_data_channel_adapter",
                "current_regression_kernel": "projection_axi_read_data_channel_adapter",
                "semantic_op": "down_proj",
                "expected_evidence_dir": "build/projection_down_proj_axi_read_data_channel_adapter_gate",
            }
        ],
    }
    child_calls = []

    def fake_child_synthesis(child_entry, child_cfg, child_root, child_options):
        child_calls.append(
            {
                "task_id": child_entry["task_id"],
                "semantic_op": child_entry["semantic_op"],
                "skip_synth": child_options.skip_synth,
            }
        )
        gate = child_root / Path(child_entry["expected_evidence_dir"]).name
        (gate / "kernel_report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "coverage_level": "projection_axi_read_data_channel_adapter_fixture",
                    "implementation_stage": "post_route_ooc",
                    "files": [
                        "projection_axi_read_data_channel_adapter.sv",
                        "tb_projection_axi_read_data_channel_adapter.sv",
                    ],
                    "simulation": {"passed": True},
                    "verilator": {"passed": True},
                    "synthesis": {
                        "passed": True,
                        "cmd": ["vivado", "-mode", "batch", "-source", "vivado_synth.tcl"],
                        "timing_report": "timing_summary.rpt",
                        "utilization_report": "utilization.rpt",
                        "timing": {
                            "constraints_met": True,
                            "setup_worst_slack_ns": 1.057,
                            "hold_worst_slack_ns": 0.005,
                            "pulse_width_worst_slack_ns": 2.225,
                        },
                        "resource_utilization": {
                            "lut_as_logic": {"used": 120},
                            "clb_registers": {"used": 80},
                            "block_ram_tile": {"used": 0},
                            "dsps": {"used": 0},
                            "uram": {"used": 0},
                            "bonded_iob": {"used": 96},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "parent_loop_action": "executed",
            "backend": "fake_child_synthesis",
            "task_id": child_entry["task_id"],
            "status": "passed",
            "evidence_dir": str(gate),
        }

    monkeypatch.setattr(parent_loop_module, "_run_local_implementation_entry", fake_child_synthesis)

    result = _run_local_or_queue_entry(
        entry,
        cfg,
        evidence_root,
        ParentLoopOptions(local_verification=True, skip_synth=False),
    )
    verification = json.loads(
        (
            evidence_root
            / "verification_results"
            / "wave_7_projection_axi_read_data_channel_adapter__verification.json"
        ).read_text(encoding="utf-8")
    )
    integration = json.loads(
        (
            evidence_root
            / "wave_7_projection_axi_read_data_channel_adapter_integration_verification"
            / "integration_synthesis_report.json"
        ).read_text(encoding="utf-8")
    )

    assert child_calls == [
        {
            "task_id": "projection_down_proj_axi_read_data_channel_adapter",
            "semantic_op": "down_proj",
            "skip_synth": False,
        }
    ]
    assert result["parent_loop_action"] == "executed"
    assert result["backend"] == "local_feedback_child_synthesis_then_integration_verification"
    assert result["child_synthesis_task_ids"] == ["projection_down_proj_axi_read_data_channel_adapter"]
    assert verification["status"] == "passed"
    assert integration["status"] == "passed"
    assert integration["utilization"]["lut"] == 120


def test_parent_loop_queue_entries_are_deduped(tmp_path: Path):
    status_dir = tmp_path / "status"
    entries = [
        {
            "parent_loop_action": "queued",
            "reason": "integration_synthesis_required_but_skip_synth_enabled",
            "spawn_key": "wave_7::verification",
            "spawn_kind": "verification_agent",
            "wave_id": "wave_7",
            "missing_synthesis_task_ids": ["b", "a"],
        },
        {
            "parent_loop_action": "queued",
            "reason": "integration_synthesis_required_but_skip_synth_enabled",
            "spawn_key": "wave_7::verification",
            "spawn_kind": "verification_agent",
            "wave_id": "wave_7",
            "missing_synthesis_task_ids": ["a", "b"],
        },
    ]

    result = parent_loop_module._write_parent_queue(status_dir, entries)
    queue = json.loads((status_dir / "parent_loop_queue.json").read_text(encoding="utf-8"))

    assert result["entry_count"] == 1
    assert result["deduped_entry_count"] == 1
    assert queue["entry_count"] == 1
    assert queue["deduped_entry_count"] == 1


def test_parent_loop_status_passes_when_board_signoff_readiness_passed(tmp_path: Path):
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "passed"},
        target_tasks={},
        status_paths={},
    )

    status = parent_loop_module._parent_loop_status(
        state,
        final_reason="max_iterations_reached",
        executed_total=1,
        queued_total=0,
    )

    assert status == "passed"


def test_parent_loop_status_blocks_when_full_execution_passed_but_board_signoff_missing(
    tmp_path: Path,
):
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={},
        status_paths={},
    )

    status = parent_loop_module._parent_loop_status(
        state,
        final_reason="max_iterations_reached",
        executed_total=1,
        queued_total=0,
    )

    assert status == "blocked_board_signoff_evidence"


def test_parent_loop_queues_target_scale_board_wrapper_when_vivado_route_is_skipped(
    tmp_path: Path,
):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    status_dir = tmp_path / "status"
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={
            "zcu104_board_wrapper_axi_bridge": {
                "ready_to_spawn": True,
                "prompt_file": "board_implementation_prompts/zcu104_board_wrapper_axi_bridge_agent.md",
                "expected_evidence_file": str(
                    evidence_root
                    / "board_zcu104_signoff_gate"
                    / "zcu104_board_wrapper_axi_bridge_implementation_report.json"
                ),
                "expected_subagent_result": str(
                    evidence_root
                    / "board_zcu104_signoff_gate"
                    / "zcu104_board_wrapper_axi_bridge_subagent_result.json"
                ),
            }
        },
        status_paths={},
    )

    actions = parent_loop_module._run_or_queue_target_tasks(
        state,
        cfg,
        evidence_root,
        status_dir,
        ParentLoopOptions(skip_vivado_route=True),
    )

    assert actions["executed"] == []
    assert actions["queued"][0]["target_task"] == "zcu104_board_wrapper_axi_bridge"
    assert (
        actions["queued"][0]["reason"]
        == "target_scale_board_wrapper_route_required_but_skip_vivado_route_enabled"
    )
    assert "without --skip-vivado-route" in actions["queued"][0]["next_action"]


def test_parent_loop_queues_target_artifact_instead_of_rerouting_fixture_wrapper(
    tmp_path: Path,
):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    status_dir = tmp_path / "status"
    wrapper_dir = evidence_root / "board_zcu104_signoff_gate"
    wrapper_dir.mkdir(parents=True)
    bitstream = wrapper_dir / "zcu104_board_wrapper.bit"
    bitstream.write_bytes(b"fixture bitstream")
    (wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "bitstream_generated": True,
                "bitstream_file": str(bitstream),
                "target_scale_accelerator_bitstream": False,
                "accelerator_scope": "ddr_axi_board_shell_fixture",
            }
        ),
        encoding="utf-8",
    )
    gate = evidence_root / "ddr_axi_board_shell_fixture_gate"
    gate.mkdir()
    (gate / "ddr_axi_board_shell_fixture.sv").write_text("module ddr_axi_board_shell_fixture; endmodule\n")
    (gate / "kernel_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "kernel": "ddr_axi_board_shell_fixture",
                "coverage_level": "ddr_axi_board_shell_fixture",
                "numeric_policy": {
                    "full_llama_model": False,
                    "board_level_signoff": False,
                },
            }
        ),
        encoding="utf-8",
    )
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={
            "zcu104_board_wrapper_axi_bridge": {
                "ready_to_spawn": True,
                "prompt_file": "board_implementation_prompts/zcu104_board_wrapper_axi_bridge_agent.md",
                "expected_evidence_file": str(wrapper_dir / "zcu104_board_wrapper_axi_bridge_implementation_report.json"),
                "expected_subagent_result": str(wrapper_dir / "zcu104_board_wrapper_axi_bridge_subagent_result.json"),
            }
        },
        status_paths={},
    )

    actions = parent_loop_module._run_or_queue_target_tasks(
        state,
        cfg,
        evidence_root,
        status_dir,
        ParentLoopOptions(skip_vivado_route=False),
    )

    assert actions["executed"] == []
    assert actions["queued"][0]["reason"] == "target_scale_accelerator_artifact_required"
    assert actions["queued"][0]["current_accelerator_artifact"]["target_scale_eligible"] is False


def test_parent_loop_queues_full_model_target_rtl_generator(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    status_dir = tmp_path / "status"
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={
            "full_model_target_rtl_generator": {
                "ready_to_spawn": True,
                "prompt_file": "target_implementation_prompts/full_model_target_rtl_generator_agent.md",
                "expected_evidence_file": str(
                    evidence_root / "full_target_llama_accelerator_gate" / "kernel_report.json"
                ),
                "expected_subagent_result": str(
                    evidence_root / "full_target_llama_accelerator_gate" / "subagent_result.json"
                ),
            }
        },
        status_paths={},
    )

    actions = parent_loop_module._run_or_queue_target_tasks(
        state,
        cfg,
        evidence_root,
        status_dir,
        ParentLoopOptions(skip_vivado_route=False),
    )

    assert actions["executed"] == []
    assert actions["queued"][0]["target_task"] == "full_model_target_rtl_generator"
    assert actions["queued"][0]["reason"] == "full_model_target_rtl_generator_requires_codex_implementation_subagent"
    assert "before any ZCU104 board-wrapper reroute" in actions["queued"][0]["next_action"]


def test_parent_loop_queues_target_scale_child_rtl_packet(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    status_dir = tmp_path / "status"
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={
            "target_gptq_projection_datapath_packets": {
                "ready_to_spawn": True,
                "target_wave": "target_scale_child_rtl_wave",
                "prompt_file": "target_implementation_prompts/target_gptq_projection_datapath_packets_agent.md",
                "expected_evidence_file": str(
                    evidence_root / "target_gptq_projection_datapath_packets_gate" / "kernel_report.json"
                ),
                "expected_subagent_result": str(
                    evidence_root / "target_gptq_projection_datapath_packets_gate" / "subagent_result.json"
                ),
                "depends_on": [],
            }
        },
        status_paths={},
    )

    actions = parent_loop_module._run_or_queue_target_tasks(
        state,
        cfg,
        evidence_root,
        status_dir,
        ParentLoopOptions(skip_vivado_route=False),
    )

    assert actions["executed"] == []
    assert actions["queued"][0]["target_task"] == "target_gptq_projection_datapath_packets"
    assert actions["queued"][0]["target_wave"] == "target_scale_child_rtl_wave"
    assert actions["queued"][0]["reason"] == "target_scale_child_rtl_packet_requires_codex_implementation_subagent"


def test_parent_loop_queues_target_child_skill_update_blocker(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    evidence_root = tmp_path / "evidence"
    status_dir = tmp_path / "status"
    state = parent_loop_module.ParentState(
        dispatch_plan={},
        wave_status={},
        execution_manifest={"spawn_entry_count": 0},
        full_execution_readiness={"status": "passed"},
        board_signoff_readiness={"status": "blocked_by_missing_or_incomplete_board_signoff_evidence"},
        target_tasks={
            "target_non_gemm_datapath_packets": {
                "ready_to_spawn": False,
                "target_wave": "target_scale_child_rtl_wave",
                "expected_evidence_file": str(
                    evidence_root / "target_non_gemm_datapath_packets_gate" / "kernel_report.json"
                ),
                "expected_subagent_result": str(
                    evidence_root / "target_non_gemm_datapath_packets_gate" / "subagent_result.json"
                ),
                "existing_child_blocker": {
                    "status": "blocked",
                    "coverage_level": "target_non_gemm_scheduled_datapath_prototype",
                    "target_scale_child_eligible": False,
                    "skill_update_candidate_complete": True,
                    "next_action": "run_subagents_skill_draft_and_update_skill_before_retry",
                },
            }
        },
        status_paths={},
    )

    actions = parent_loop_module._run_or_queue_target_tasks(
        state,
        cfg,
        evidence_root,
        status_dir,
        ParentLoopOptions(skip_vivado_route=False),
    )

    assert actions["executed"] == []
    assert actions["queued"][0]["target_task"] == "target_non_gemm_datapath_packets"
    assert actions["queued"][0]["reason"] == "target_scale_child_rtl_packet_skill_update_required"
    assert actions["queued"][0]["next_action"] == "run_subagents_skill_draft_and_update_skill_before_retry"


def test_parent_loop_retries_stale_local_integration_placeholder(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    verification_dir = evidence_root / "verification_results"
    verification_dir.mkdir(parents=True)
    (verification_dir / "wave_2_decoder_block__verification.json").write_text(
        json.dumps(
            {
                "status": "blocked_by_integration_synthesis_requirement",
                "verification_backend": "local_deterministic_smoke",
                "findings": [],
            }
        ),
        encoding="utf-8",
    )
    dispatch_plan = {
        "waves": [
            {
                "wave_id": "wave_2_decoder_block",
                "depends_on_waves": [],
                "target_scope": "bounded_fixture_only",
                "blocked_target_dependencies": [],
                "implementation_tasks": [],
                "verification_agent": {
                    "prompt_file": "verification_prompts/wave_2_decoder_block__verification.md",
                    "runs_integration_synthesis": True,
                },
            }
        ]
    }

    status = build_hdl_subagent_wave_status(dispatch_plan, evidence_root)

    assert status["waves"][0]["status"] == "ready_for_verification"
    assert status["waves"][0]["verification"]["status"] == "missing"
    assert "stale local verification placeholder" in status["waves"][0]["verification"]["reason"]


def test_parent_loop_inspect_cache_invalidates_when_preflight_sources_change(tmp_path: Path):
    cfg = load_config(Path("examples/zcu104_llama32_1b_gptq.yaml"))
    inspect_dir = tmp_path / "inspect"
    inspect_dir.mkdir()
    (inspect_dir / "llm_agent_report.json").write_text(
        json.dumps(
            {
                "model": "meta-llama/Llama-3.2-1B",
                "partition": "gemm_non_gemm",
            }
        ),
        encoding="utf-8",
    )
    (inspect_dir / "hdl_subagent_dispatch_plan.json").write_text(
        json.dumps(
            {
                "source_replay": {
                    "model_name": "meta-llama/Llama-3.2-1B",
                    "gptq_checkpoint": None,
                    "mlir_graph": None,
                },
                "hardware": {
                    "fpga_part": cfg.hardware.fpga_part,
                    "target_clock_mhz": cfg.hardware.target_clock_mhz,
                    "memory_data_width": cfg.hardware.memory_data_width,
                },
                "optimization": {
                    "quantization": cfg.optimization.quantization,
                    "design_style": cfg.design.style,
                    "compute_style": cfg.design.compute_style,
                    "execution_style": cfg.design.execution_style,
                    "memory_style": cfg.design.memory_style,
                    "control_style": cfg.design.control_style,
                    "pe_count": cfg.design.pe_count,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _inspect_artifacts_match_inputs(
        inspect_dir,
        "meta-llama/Llama-3.2-1B",
        cfg,
        "gemm_non_gemm",
    )

    cfg_with_sources = replace(
        cfg,
        model=replace(
            cfg.model,
            gptq_checkpoint="/tmp/gptq",
            mlir_graph="/tmp/model.mlir",
        ),
    )

    assert not _inspect_artifacts_match_inputs(
        inspect_dir,
        "meta-llama/Llama-3.2-1B",
        cfg_with_sources,
        "gemm_non_gemm",
    )


def test_parent_loop_queues_target_preflight_blockers_after_waves_pass():
    entries = _target_preflight_queue_entries(
        {
            "source_replay": {
                "model_name": "meta-llama/Llama-3.2-1B",
                "gptq_checkpoint": None,
                "mlir_graph": None,
            },
            "blocked_target_tasks": [
                {
                    "task_id": "real_mlir_model_analysis",
                    "reason": "provided MLIR graph is missing",
                },
                {
                    "task_id": "real_gptq_checkpoint_metadata",
                    "reason": "GPTQ metadata is missing",
                },
            ],
        },
        {
            "target_preflight": {
                "status": "blocked",
                "preflight_blockers": [
                    "real_gptq_checkpoint_metadata",
                    "real_mlir_model_analysis",
                ],
            }
        },
    )

    assert [entry["task_id"] for entry in entries] == [
        "real_gptq_checkpoint_metadata",
        "real_mlir_model_analysis",
    ]
    assert all(entry["spawn_kind"] == "target_preflight_agent" for entry in entries)
    assert all(entry["reason"] == "target_preflight_blocker" for entry in entries)
    assert "inspect/gptq_checkpoint_metadata.json" in entries[0]["expected_artifacts"]
    assert "inspect/mlir_model_analysis_readiness.json" in entries[1]["expected_artifacts"]
