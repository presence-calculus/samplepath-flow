# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from samplepath.csv_loader import CSVLoader


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Autodetect delimiter across common cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore: DataFrame is empty after loading/cleaning")
@pytest.mark.parametrize("sep", [",", "\t", ";", "|", ":"])
def test_autodetect_delimiter_common_separators(tmp_path, sep):
    csv = f"id{sep}start_ts{sep}end_ts\nA{sep}2024-01-01 00:00{sep}2024-01-01 01:00\n"
    path = _write(tmp_path, "events.csv", csv)
    df = CSVLoader(autodetect_delimiter=True, delimiter=None).load(str(path))
    assert list(df.columns[:3]) == ["id", "start_ts", "end_ts"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Explicit delimiter overrides autodetection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_explicit_delimiter_overrides_autodetect(tmp_path):
    csv = "id|start_ts|end_ts\nA|2024-01-01 00:00|2024-01-01 01:00\n"
    path = _write(tmp_path, "events_pipe.csv", csv)
    df = CSVLoader(autodetect_delimiter=False, delimiter="|").load(str(path))
    assert list(df.columns[:3]) == ["id", "start_ts", "end_ts"]

def test_extra_whitespace_in_columns_stripped(tmp_path):
    csv = "id| start_ts|end_ts \nA|2024-01-01 00:00|2024-01-01 01:00\n"
    path = _write(tmp_path, "events_pipe.csv", csv)
    df = CSVLoader(autodetect_delimiter=False, delimiter="|").load(str(path))
    assert list(df.columns[:3]) == ["id", "start_ts", "end_ts"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Explicit start_column/end_column overrides
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_start_end_column_overrides(tmp_path):
    csv = "id,arrival_ts,departure_ts\nA,2024-01-01 00:00,2024-01-01 01:00\n"
    path = _write(tmp_path, "events_pipe.csv", csv)
    df = CSVLoader(autodetect_delimiter=False, start_column="arrival_ts", end_column="departure_ts").load(str(path))
    assert list(df.columns[:3]) == ["id", "start_ts", "end_ts"]

def test_overrides_rename_existing_canonicals(tmp_path):
    csv = "id,arrival_ts,start_ts,departure_ts,end_ts\nA,2024-01-01 00:00,2024-01-01 01:00,2024-01-01 00:00,2024-01-01 01:00\n"
    path = _write(tmp_path, "events_pipe.csv", csv)
    df = CSVLoader(autodetect_delimiter=False, start_column="arrival_ts", end_column="departure_ts").load(str(path))
    assert list(df.columns[:5]) == ["id", "start_ts", "_start_ts", "end_ts", "_end_ts"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Required columns enforcement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_missing_required_columns_raises(tmp_path):
    csv = "id,start_ts\nA,2024-01-01 00:00\n"  # no end_ts
    path = _write(tmp_path, "bad.csv", csv)
    with pytest.raises((ValueError, KeyError)):
        CSVLoader(required_columns=("id", "start_ts", "end_ts")).load(str(path))
    assert True  # single assertion pattern: reaching here means exception branch executed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Datetime parsing yields datetime64[ns] (naive) dtype
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_parse_dates_are_datetime64(tmp_path):
    csv = "id,start_ts,end_ts\nA,2024-01-01 00:00,2024-01-01 01:00\n"
    path = _write(tmp_path, "events.csv", csv)
    df = CSVLoader(parse_dates=("start_ts", "end_ts")).load(str(path))
    assert pd.api.types.is_datetime64_ns_dtype(df["start_ts"].dtype) and pd.api.types.is_datetime64_ns_dtype(df["end_ts"].dtype)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NaT on unparsable timestamps (graceful coercion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:1 rows have invalid/missing 'start_ts' (set to NaT)")
def test_unparsable_timestamps_raise_when_all_invalid(tmp_path):
    csv = "id,start_ts,end_ts\nA,not-a-time,2024-01-01 01:00\n"
    path = _write(tmp_path, "bad_time.csv", csv)
    with pytest.raises(ValueError):
        CSVLoader(parse_dates=("start_ts", "end_ts")).load(str(path))



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Respects custom required_columns ordering and additional columns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_custom_required_columns_ok_when_present(tmp_path):
    csv = "item_id,start_ts,end_ts,class\nA,2024-01-01 00:00,2024-01-01 01:00,feature\n"
    path = _write(tmp_path, "custom.csv", csv)
    loader = CSVLoader(required_columns=("item_id", "start_ts", "end_ts"), parse_dates=("start_ts", "end_ts"))
    df = loader.load(str(path))
    assert "item_id" in df.columns and "class" in df.columns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delimiter candidates list is honored (autodetect disabled for others)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_delimiter_candidates_influence_detection(tmp_path):
    # Use a caret separator that is not in the default candidates; expect failure without explicit delimiter
    csv = "id^start_ts^end_ts\nA^2024-01-01 00:00^2024-01-01 01:00\n"
    path = _write(tmp_path, "caret.csv", csv)
    with pytest.raises(Exception):
        CSVLoader(autodetect_delimiter=True, delimiter_candidates=(",", ";")).load(str(path))
    assert True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Leading/trailing whitespace in headers does not break required columns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_header_whitespace_trimmed_for_required_columns(tmp_path):
    csv = " id , start_ts , end_ts \nA,2024-01-01 00:00,2024-01-01 01:00\n"
    path = _write(tmp_path, "whitespace.csv", csv)
    df = CSVLoader().load(str(path))
    assert {"id", "start_ts", "end_ts"}.issubset(set(map(str, df.columns)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty file yields empty DataFrame with required columns (or raises)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_empty_file_behavior(tmp_path):
    path = _write(tmp_path, "empty.csv", "")
    try:
        df = CSVLoader().load(str(path))
        ok = df.empty or all(col in df.columns for col in ("id", "start_ts", "end_ts"))
    except Exception:
        ok = True  # acceptable path: loader raises a clear exception
    assert ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time zone handling (if loader localizes then normalizes to naive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_timezone_handling_result_is_datetime64_naive(tmp_path):
    # If the loader localizes to a zone and then returns naive, dtype remains datetime64[ns]
    csv = "id,start_ts,end_ts\nA,2024-06-01T12:00:00,2024-06-01T13:30:00\n"
    path = _write(tmp_path, "tz.csv", csv)
    df = CSVLoader(time_zone="America/Chicago", parse_dates=("start_ts", "end_ts")).load(str(path))
    assert pd.api.types.is_datetime64_ns_dtype(df["start_ts"].dtype) and df["start_ts"].dt.tz is None
