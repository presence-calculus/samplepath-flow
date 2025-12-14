# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .file_utils import (
    copy_input_csv_to_output,
    ensure_output_dirs,
    write_cli_args_to_file,
)
from .sample_path_analysis import run_analysis


def _looks_like_path(token: str) -> bool:
    path = Path(token)
    return path.suffix != "" or path.exists() or ("/" in token or "\\" in token)


def normalize_argv(
    argv: list[str], subcommands: set[str], parser: argparse.ArgumentParser
) -> list[str]:
    # argv[0] is program name
    if len(argv) <= 1:
        return argv
    first = argv[1]
    if first.startswith("-"):
        return argv
    if first in subcommands:
        return argv
    if _looks_like_path(first):
        return [argv[0], "analyze", *argv[1:]]

    parser.print_help()
    print(f"Error: command '{first}' not recognized", file=sys.stderr)
    sys.exit(1)


def validate_args(args):
    error = False

    if getattr(args, "completed", False) and getattr(args, "incomplete", False):
        print(
            "Error: --completed and --incomplete cannot be used together",
            file=sys.stderr,
        )
        error = True

    if error:
        sys.exit(1)


def build_parser() -> tuple[argparse.ArgumentParser, set[str]]:
    parser = argparse.ArgumentParser(
        description="Finite-window Little’s Law charts from intervals CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Generate finite-window flow-metrics charts from an intervals CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- CSV Parsing --- #
    csv_group = analyze.add_argument_group("CSV Parsing")
    csv_group.add_argument(
        "csv", type=str, help="Path to CSV (id,start_ts,end_ts[,class])"
    )

    csv_group.add_argument(
        "--delimiter", type=str, default=None, help="Optional delimiter for csv"
    )

    csv_group.add_argument(
        "--start_column",
        type=str,
        help="Name of csv column representing start time",
        default="start_ts",
    )
    csv_group.add_argument(
        "--end_column",
        type=str,
        help="Name of csv column representing end time",
        default="end_ts",
    )

    csv_group.add_argument(
        "--date-format",
        type=str,
        default=None,
        help="Optional explicit datetime format string for parsing CSV timestamps (e.g. '%%d/%%m/%%Y %%H:%%M').",
    )

    csv_group.add_argument(
        "--dayfirst",
        action="store_true",
        default=False,
        help="Interpret ambiguous dates as day-first (e.g., 03/04/2024 → 3 April 2024).",
    )

    # Input Data Filters ---#
    data_filters = analyze.add_argument_group("Data filters")
    data_filters.add_argument(
        "--completed",
        action="store_true",
        help="Only include items with an end_ts (completed work only)",
    )
    data_filters.add_argument(
        "--incomplete",
        action="store_true",
        help="Only include items without an end_ts (aging view)",
    )

    data_filters.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of class tags to include (requires a 'class' column)",
    )

    # - Outlier trimming --#
    outliers_group = analyze.add_argument_group("Outlier Trimming")
    outliers_group.add_argument(
        "--outlier-hours",
        type=float,
        default=None,
        help="Drop completed items whose (end_ts - start_ts) exceeds this many hours",
    )
    outliers_group.add_argument(
        "--outlier-pctl",
        type=float,
        default=None,
        help="Drop completed items above the Pth percentile of duration (0<P<100) after other filters",
    )
    outliers_group.add_argument(
        "--outlier-iqr",
        type=float,
        default=None,
        help="Drop completed items above Q3+K·IQR (Tukey high fence); pass K (e.g., 1.5)",
    )
    outliers_group.add_argument(
        "--outlier-iqr-two-sided",
        action="store_true",
        help="Also drop items below Q1−K·IQR when used with --outlier-iqr",
    )

    # - Fine tuning lambda display --#
    lambda_fine_tuning = analyze.add_argument_group("Lambda Fine Tuning")
    lambda_fine_tuning.add_argument(
        "--lambda-pctl",
        type=float,
        default=None,
        help="Clip Λ(T) y-axis to the upper Pth percentile (0<P<100)",
    )
    lambda_fine_tuning.add_argument(
        "--lambda-lower-pctl",
        type=float,
        default=None,
        help="Optionally clip the lower end to the Pth percentile as well",
    )
    lambda_fine_tuning.add_argument(
        "--lambda-warmup",
        type=float,
        default=0.0,
        help="Ignore the first H hours when computing Λ(T) percentiles",
    )

    # -- Parameters for tuning sample path convergence charts ---#
    convergence_thresholds = analyze.add_argument_group("Convergence Thresholds")
    convergence_thresholds.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Relative error threshold for convergence (default 0.05)",
    )
    convergence_thresholds.add_argument(
        "--horizon-days",
        type=float,
        default=28.0,
        help="Ignore this many initial days when assessing convergence - suppress the mixing period (default 28)",
    )

    # output directory handling
    output_dirs = analyze.add_argument_group("Output Configuration")
    output_dirs.add_argument(
        "--output-dir",
        type=lambda p: Path(p).expanduser().resolve(),
        default="charts",
        help="Root directory where charts will be written. Output will be written under a subdirectory of this directory named with the csv file name",
    )

    output_dirs.add_argument(
        "--scenario",
        type=str,
        default="latest",
        help="create the output under a named folder. The default is 'latest' under the output folder created under output-dir",
    )

    output_dirs.add_argument(
        "--save-input",
        action="store_true",
        default=True,
        help="Copy the input csv to the output path (saved under input subdirectory)",
    )
    output_dirs.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="removing existing charts in output directory",
    )

    subcommand_names = set(subparsers.choices.keys())
    return parser, subcommand_names


def parse_args(argv: list[str] | None = None):
    parser, subcommand_names = build_parser()
    if argv is None:
        argv = sys.argv
    argv = normalize_argv(argv, subcommand_names, parser)

    if len(argv) <= 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv[1:])
    validate_args(args)
    return parser, args


def get_class_filters(classes):
    class_filters = None
    if classes:
        class_filters = [c for c in classes.split(",") if c.strip() != ""]
    return class_filters


def main():
    parser, args = parse_args()

    try:
        out_dir = ensure_output_dirs(
            args.csv,
            output_dir=args.output_dir,
            scenario_dir=args.scenario,
            clean=args.clean,
        )
        if args.save_input:
            copy_input_csv_to_output(args.csv, out_dir)

        write_cli_args_to_file(parser, args, out_dir)
        paths = run_analysis(args.csv, args, out_dir)
        print("Wrote charts:\n" + "\n".join(paths))
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
