from pathlib import Path
import sys

import pytest

from samplepath import cli


def test_parse_args_defaults_to_analyze():
    _, args = cli.parse_args(["flow", "events.csv", "--output-dir", "charts"])

    assert args.cmd == "analyze"
    assert args.csv == "events.csv"


def test_parse_args_explicit_analyze():
    _, args = cli.parse_args(["flow", "analyze", "events.csv"])

    assert args.cmd == "analyze"
    assert args.csv == "events.csv"


def test_flow_help_top_level(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["flow", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "analyze" in out
    assert "configure" not in out


def test_flow_help_analyze(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["flow", "analyze", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "CSV Parsing" in out
    assert "Output Configuration" in out


def test_parse_args_mutual_exclusive_flags(monkeypatch):
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(["flow", "events.csv", "--completed", "--incomplete"])

    assert excinfo.value.code == 1


def test_parse_args_defaults_and_types():
    _, args = cli.parse_args(["flow", "events.csv"])

    assert args.output_dir == Path("charts").resolve()
    assert args.scenario == "latest"
    assert args.epsilon == 0.05
    assert args.horizon_days == 28.0
    assert args.lambda_warmup == 0.0
    assert args.save_input is True
    assert args.with_event_marks is False
    assert args.show_derivations is False
    assert args.chart_format == "png"
    assert args.chart_dpi == 150


def test_with_event_marks_flag_can_be_enabled():
    _, args = cli.parse_args(["flow", "events.csv", "--with-event-marks"])

    assert args.with_event_marks is True


def test_show_derivations_flag_can_be_enabled():
    _, args = cli.parse_args(["flow", "events.csv", "--show-derivations"])

    assert args.show_derivations is True


def test_chart_format_flag_can_be_enabled():
    _, args = cli.parse_args(["flow", "events.csv", "--chart-format", "svg"])

    assert args.chart_format == "svg"


def test_chart_dpi_flag_can_be_enabled():
    _, args = cli.parse_args(["flow", "events.csv", "--chart-dpi", "150"])

    assert args.chart_dpi == 150


def test_sampling_frequency_defaults_to_none():
    _, args = cli.parse_args(["flow", "events.csv"])

    assert args.sampling_frequency is None


def test_sampling_frequency_accepts_value():
    _, args = cli.parse_args(["flow", "events.csv", "--sampling-frequency", "week"])

    assert args.sampling_frequency == "week"


def test_anchor_defaults_to_none():
    _, args = cli.parse_args(["flow", "events.csv"])

    assert args.anchor is None


def test_anchor_accepts_value():
    _, args = cli.parse_args(["flow", "events.csv", "--anchor", "WED"])

    assert args.anchor == "WED"


def test_output_dir_expands_user():
    home = Path("~").expanduser().resolve()
    _, args = cli.parse_args(["flow", "events.csv", "--output-dir", "~"])

    assert args.output_dir == home


def test_invalid_subcommand_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["flow", "bogus"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_main_calls_pipeline(monkeypatch, capsys):
    calls = {}

    def fake_ensure(csv, output_dir, scenario_dir, clean):
        calls["ensure"] = (csv, output_dir, scenario_dir, clean)
        return Path("/tmp/out")

    def fake_copy(csv, out_dir):
        calls["copy"] = (csv, out_dir)

    def fake_write(parser, args, out_dir):
        calls["write"] = (parser is not None, args is not None, out_dir)

    def fake_run(csv, args, out_dir):
        calls["run"] = (csv, out_dir)
        return ["chart-a", "chart-b"]

    monkeypatch.setattr(sys, "argv", ["flow", "events.csv"])
    monkeypatch.setattr(cli, "ensure_output_dirs", fake_ensure)
    monkeypatch.setattr(cli, "copy_input_csv_to_output", fake_copy)
    monkeypatch.setattr(cli, "write_cli_args_to_file", fake_write)
    monkeypatch.setattr(cli, "run_analysis", fake_run)

    cli.main()

    assert calls["ensure"] == ("events.csv", Path("charts").resolve(), "latest", False)
    assert calls["copy"][0] == "events.csv"
    assert calls["run"] == ("events.csv", Path("/tmp/out"))
    out = capsys.readouterr().out
    assert "Wrote charts" in out


def test_main_reports_error(monkeypatch, capsys):
    def fake_ensure(csv, output_dir, scenario_dir, clean):
        return Path("/tmp/out")

    def fake_copy(csv, out_dir):
        return None

    def fake_write(parser, args, out_dir):
        return None

    def fake_run(csv, args, out_dir):
        raise ValueError("boom")

    monkeypatch.setattr(sys, "argv", ["flow", "events.csv"])
    monkeypatch.setattr(cli, "ensure_output_dirs", fake_ensure)
    monkeypatch.setattr(cli, "copy_input_csv_to_output", fake_copy)
    monkeypatch.setattr(cli, "write_cli_args_to_file", fake_write)
    monkeypatch.setattr(cli, "run_analysis", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    assert "Error: boom" in capsys.readouterr().err


def test_export_data_defaults_to_true():
    _, args = cli.parse_args(["flow", "events.csv"])

    assert args.export_data is True


def test_export_only_defaults_to_false():
    _, args = cli.parse_args(["flow", "events.csv"])

    assert args.export_only is False


def test_export_data_flag_can_be_disabled():
    _, args = cli.parse_args(["flow", "events.csv", "--no-export-data"])

    assert args.export_data is False


def test_export_only_flag_can_be_enabled():
    _, args = cli.parse_args(["flow", "events.csv", "--export-only"])

    assert args.export_only is True


def test_no_export_data_and_export_only_are_mutually_exclusive():
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(["flow", "events.csv", "--no-export-data", "--export-only"])

    assert excinfo.value.code == 1
