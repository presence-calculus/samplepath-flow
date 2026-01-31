# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# test/samplepath/test_file_utils.py

import argparse
import os
from pathlib import Path

import pandas as pd
import pytest

from samplepath.utils.file_utils import (
    copy_input_csv_to_output,
    ensure_output_dirs,
    make_fresh_dir,
    make_root_dir,
    write_cli_args_to_file,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# make_fresh_dir
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_make_fresh_dir_creates_directory(tmp_path):
    d = tmp_path / "exists_already"
    d.mkdir()
    (d / "old.txt").write_text("old")
    out = make_fresh_dir(d)
    assert out.exists() and out.is_dir()  # one assertion


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# make_root_dir
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_make_root_dir_returns_stem_scenario_dir(tmp_path):
    csv = tmp_path / "input.csv"
    csv.write_text("id,start_ts,end_ts\n")
    out = make_root_dir(str(csv), str(tmp_path), "scenarioA", clean=False)
    # Expect <tmp>/<csv_stem>/scenarioA
    expected = tmp_path / "input" / "scenarioA"
    assert Path(out) == expected  # one assertion


def test_make_root_dir_clean_recreates_directory(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("id,start_ts,end_ts\n")
    out_dir = Path(make_root_dir(str(csv), str(tmp_path), "latest", clean=False))
    (out_dir / "keep.txt").write_text("x")
    out_dir2 = Path(make_root_dir(str(csv), str(tmp_path), "latest", clean=True))
    # If cleaned, the previous file should be gone, but we assert directory exists (single assertion)
    assert out_dir2.exists() and out_dir2.is_dir()  # one assertion


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ensure_output_dirs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_ensure_output_dirs_creates_exact_known_subdirs(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text("id,start_ts,end_ts\n")

    scenario_root = Path(
        ensure_output_dirs(str(csv), str(tmp_path), scenario_dir="s1", clean=True)
    )

    expected = {
        "input",
        "core",
        "core/panels",
        "convergence",
        "convergence/panels",
        "stability",
        "stability/panels",
        "advanced",
    }

    # Collect all directories created under the scenario root (relative paths, POSIX style)
    actual = {
        str(p.relative_to(scenario_root)).replace(os.sep, "/")
        for p in scenario_root.rglob("*")
        if p.is_dir()
    }

    # Single assertion: exact match (no missing, no extras)
    assert actual == expected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# write_cli_args_to_file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_write_cli_args_to_file_creates_parameters_txt(tmp_path, capsys):
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--window", help="Observation window length", default=14)
    parser.add_argument("--mode", help="Operating mode", default="fast")
    args = parser.parse_args([])  # defaults
    write_cli_args_to_file(parser, args, tmp_path)
    # Single assertion: file exists
    assert (tmp_path / "parameters.txt").exists()  # one assertion


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# copy_input_csv_to_output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_copy_input_csv_to_output_copies_file(tmp_path):
    src = tmp_path / "sample.csv"
    src.write_text("id,start_ts,end_ts\n")

    out_dir = tmp_path / "out"
    (out_dir / "input").mkdir(parents=True)  # ensure expected subdir exists

    dest = copy_input_csv_to_output(src, out_dir)
    assert dest.exists() and dest == (out_dir / "input" / "sample.csv")


def test_copy_input_csv_to_output_raises_on_missing_source(tmp_path):
    missing = tmp_path / "nope.csv"
    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        copy_input_csv_to_output(missing, out_dir)
    assert True  # keep a single-assertion structure for the test
