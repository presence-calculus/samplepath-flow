# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
import shutil
import argparse
import textwrap
from pathlib import Path
from typing import LiteralString


def make_fresh_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_root_dir(csv_path, output_dir, clean):
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    out_dir = os.path.join(output_dir, stem)
    if clean:
        make_fresh_dir(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def ensure_output_dirs(csv_path: str, output_dir=None, clean=False) -> str:
    out_dir = make_root_dir(csv_path, output_dir, clean)
    for chart_dir in ['input', 'core',  'convergence', 'convergence/panels', 'stability/panels', 'advanced', 'misc']:
        sub_dir = os.path.join(out_dir, chart_dir)
        os.makedirs(sub_dir, exist_ok=True)

    return out_dir

def write_cli_args_to_file(parser: argparse.ArgumentParser,
                           args: argparse.Namespace,
                           output_path: str | Path) -> None:
    """
    Write all CLI arguments, their help text, defaults, and actual values
    to a neatly formatted text file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser instance defining CLI arguments.
    args : argparse.Namespace
        The parsed arguments (e.g., from parser.parse_args()).
    output_path : str or Path
        Path to the file where the formatted output should be written.

    Example
    -------
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--window", help="Observation window length", default=14)
    args = parser.parse_args()
    write_cli_args_to_file(parser, args, "cli_args.txt")
    """

    output_path = Path(os.path.join(output_path, "parameters.txt"))
    lines = []
    lines.append("Scenario Parameters")
    lines.append("=" * 80)
    lines.append("")

    for action in parser._actions:
        # Skip the built-in help flag
        if action.dest == "help":
            continue

        name = ", ".join(action.option_strings) if action.option_strings else action.dest
        help_text = (action.help or "").strip()
        default = action.default if action.default != argparse.SUPPRESS else None
        value = getattr(args, action.dest, None)

        lines.append(f"{name}")
        lines.append("-" * len(name))

        if help_text:
            wrapped = textwrap.fill(help_text, width=76, subsequent_indent="  ")
            lines.append(f"Help: {wrapped}")

        if default is not None:
            lines.append(f"Default: {default}")

        lines.append(f"Value: {value}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"[INFO] Wrote CLI argument summary to {output_path.resolve()}")

def copy_input_csv_to_output(input_path: str | Path, output_dir: str | Path) -> Path:
    """
    Copy the input CSV file to the output directory, preserving its filename.

    Parameters
    ----------
    input_path : str or Path
        Path to the input CSV file.
    output_dir : str or Path
        Directory where the file should be copied. Created if it doesn't exist.

    Returns
    -------
    Path
        Path to the copied file in the output directory.
    """
    input_path = Path(input_path)
    output_dir = Path(os.path.join(output_dir, "input"))


    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dest_path = output_dir / input_path.name
    shutil.copy2(input_path, dest_path)

    print(f"[INFO] Copied input CSV to {dest_path.resolve()}")
    return dest_path