from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any
import re

from .quant import QuantizedModel


def estimate_resources(qmodel: QuantizedModel, pe_count: int) -> dict[str, int]:
    macs = sum(layer.input_size * layer.output_size for layer in qmodel.layers)
    weights = macs
    activations = qmodel.input_size + sum(layer.output_size for layer in qmodel.layers)
    return {
        "macs": macs,
        "weight_bytes": weights,
        "activation_bytes": activations,
        "estimated_dsp": min(pe_count, macs),
        "estimated_bram": max(1, (weights + activations + 4095) // 4096),
    }


def run_iverilog(out_dir: Path, qmodel: QuantizedModel) -> dict[str, Any]:
    sv_files = [f"dense_layer_{idx}.sv" for idx, _ in enumerate(qmodel.layers)]
    sv_files.extend(["model_top.sv", "tb_model_top.sv"])
    sim_path = "sim_model"
    compile_cmd = ["iverilog", "-g2012", "-o", sim_path, *sv_files]
    compile_result = subprocess.run(
        compile_cmd,
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if compile_result.returncode != 0:
        return {"passed": False, "stage": "compile", "cmd": compile_cmd, "output": compile_result.stdout}
    run_result = subprocess.run(
        ["vvp", sim_path],
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "passed": run_result.returncode == 0,
        "stage": "run",
        "cmd": compile_cmd,
        "output": compile_result.stdout + run_result.stdout,
    }


def run_verilator_lint(out_dir: Path, qmodel: QuantizedModel) -> dict[str, Any]:
    sv_files = [f"dense_layer_{idx}.sv" for idx, _ in enumerate(qmodel.layers)]
    sv_files.append("model_top.sv")
    cmd = ["verilator", "--lint-only", "-sv", *sv_files]
    result = subprocess.run(
        cmd,
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"passed": result.returncode == 0, "cmd": cmd, "output": result.stdout}


def run_vivado_synth(out_dir: Path, timeout_sec: int) -> dict[str, Any]:
    cmd = ["vivado", "-mode", "batch", "-source", "vivado_synth.tcl"]
    try:
        result = subprocess.run(
            cmd,
            cwd=out_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
        timing = _parse_timing_summary(out_dir / "timing_summary.rpt")
        passed = result.returncode == 0 and _timing_passed(timing)
        return {"passed": passed, "cmd": cmd, "output": result.stdout[-8000:], "timing": timing}
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "cmd": cmd, "output": f"Vivado timed out after {timeout_sec}s\n{exc.stdout or ''}"}


def _parse_timing_summary(path: Path) -> dict[str, float | bool | str | None]:
    if not path.exists():
        return {
            "setup_worst_slack_ns": None,
            "hold_worst_slack_ns": None,
            "pulse_width_worst_slack_ns": None,
            "constraints_met": False,
            "parse_status": "missing_timing_summary",
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    setup = _extract_check_slack(text, "Setup")
    hold = _extract_check_slack(text, "Hold")
    pulse_width = _extract_check_slack(text, "PW")
    all_slacks_parsed = setup is not None and hold is not None and pulse_width is not None
    return {
        "setup_worst_slack_ns": setup,
        "hold_worst_slack_ns": hold,
        "pulse_width_worst_slack_ns": pulse_width,
        "constraints_met": "timing constraints are met" in text.lower()
        and "timing constraints are not met" not in text.lower(),
        "worst_slack_ns": setup,
        "parse_status": "parsed" if all_slacks_parsed else "missing_required_slack",
    }


def _extract_check_slack(text: str, label: str) -> float | None:
    match = re.search(
        rf"{label}\s+:\s+\d+\s+Failing Endpoints,\s+Worst Slack\s+(-?\d+(?:\.\d+)?)ns",
        text,
    )
    if match:
        return float(match.group(1))
    return None


def _timing_passed(timing: dict[str, float | bool | None]) -> bool:
    if timing.get("constraints_met") is not True:
        return False
    for key in ("setup_worst_slack_ns", "hold_worst_slack_ns", "pulse_width_worst_slack_ns"):
        value = timing.get(key)
        if value is None or value < 0.0:
            return False
    return True
