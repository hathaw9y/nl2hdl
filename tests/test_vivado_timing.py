from pathlib import Path

from nl2hdl.verify import _parse_timing_summary, _timing_passed


def test_vivado_timing_parser_rejects_hold_violation(tmp_path: Path):
    report = tmp_path / "timing_summary.rpt"
    report.write_text(
        """
Timing constraints are not met.
Setup :            0  Failing Endpoints,  Worst Slack        2.132ns,  Total Violation        0.000ns
Hold  :          209  Failing Endpoints,  Worst Slack       -0.043ns,  Total Violation       -2.826ns
PW    :            0  Failing Endpoints,  Worst Slack        2.225ns,  Total Violation        0.000ns
""",
        encoding="utf-8",
    )
    timing = _parse_timing_summary(report)
    assert timing["setup_worst_slack_ns"] == 2.132
    assert timing["hold_worst_slack_ns"] == -0.043
    assert not _timing_passed(timing)


def test_vivado_timing_parser_accepts_all_non_negative_slacks(tmp_path: Path):
    report = tmp_path / "timing_summary.rpt"
    report.write_text(
        """
Timing constraints are met.
Setup :            0  Failing Endpoints,  Worst Slack        2.132ns,  Total Violation        0.000ns
Hold  :            0  Failing Endpoints,  Worst Slack        0.043ns,  Total Violation        0.000ns
PW    :            0  Failing Endpoints,  Worst Slack        2.225ns,  Total Violation        0.000ns
""",
        encoding="utf-8",
    )
    timing = _parse_timing_summary(report)
    assert timing["parse_status"] == "parsed"
    assert _timing_passed(timing)


def test_vivado_timing_parser_rejects_missing_report(tmp_path: Path):
    timing = _parse_timing_summary(tmp_path / "timing_summary.rpt")
    assert timing["parse_status"] == "missing_timing_summary"
    assert not _timing_passed(timing)


def test_vivado_timing_parser_rejects_unparsed_required_slack(tmp_path: Path):
    report = tmp_path / "timing_summary.rpt"
    report.write_text(
        """
Timing constraints are met.
Setup :            0  Failing Endpoints,  Worst Slack        2.132ns,  Total Violation        0.000ns
Hold  :            0  Failing Endpoints,  Worst Slack        0.043ns,  Total Violation        0.000ns
""",
        encoding="utf-8",
    )
    timing = _parse_timing_summary(report)
    assert timing["pulse_width_worst_slack_ns"] is None
    assert timing["parse_status"] == "missing_required_slack"
    assert not _timing_passed(timing)
