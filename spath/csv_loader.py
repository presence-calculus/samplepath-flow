# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import warnings
import numpy as np
import pandas as pd


@dataclass
class CSVLoader:
    """
    Robust CSV -> DataFrame loader for event/sample-path data.

    CSV Contract:
      - 'id' - string representing the id of an item - need not be unique across all rows (ie - same id can appear multiple times)
      - 'start_ts' and 'end_ts' - strings in a format that can be parsed with <a href=https://dateutil.readthedocs.io/en/stable/parser.html>dateutil.parser.parse</a>
      - Optional 'class' attribute - string

    DataFrame Contract:
      - 'start_ts' and 'end_ts' are pandas datetimes (tz-normalized per config)
      - optional 'class' cast to 'category' if present

      Computed columns
      - 'duration_td' is Timedelta (NaT for incomplete)
      - 'duration_hr' is float hours (NaN for incomplete)


    Safety:
      - Validates required columns
      - Coerces bad timestamps to NaT
        * start_ts: warn on missing or parse-fail
        * end_ts:   warn only on parse-fail (missing is allowed)
      - Normalizes timezones (naive/aware/mixed)
      - Negative duration policy: 'drop' | 'nan' | 'raise'
      - Optional empty-frame and all-NaT checks
    """

    # --- configuration ---
    required_columns: Tuple[str, str, str] = ("id", "start_ts", "end_ts")
    class_column: str = "class"
    parse_dates: Tuple[str, str] = ("start_ts", "end_ts")

    # date parsing options.
    date_format: Optional[str] = None
    dayfirst: bool = False

    # Timezone handling
    normalize_tz: bool = True
    target_tz: str = "UTC"

    # Data-quality policies
    negative_duration_policy: str = "drop"  # 'drop' | 'nan' | 'raise'
    cast_class_to_category: bool = True
    warn_on_empty: bool = True
    error_on_all_invalid_times: bool = True

    def load(self, path: str) -> pd.DataFrame:
        # --- read CSV ---
        df = pd.read_csv(path)

        # --- column presence checks ---
        self._require_columns(df, self.required_columns)

        # --- parse datetimes with targeted warnings ---
        id, start_ts, end_ts = self.required_columns  # start_ts, end_ts

        df[start_ts], n_start_missing, n_start_fail = self._parse_dt_with_stats(df[start_ts])
        df[end_ts], n_end_missing,   n_end_fail   = self._parse_dt_with_stats(df[end_ts])

        # start_ts: any NaT (missing or failed) is noteworthy
        if n_start_missing or n_start_fail:
            warnings.warn(
                f"{n_start_missing + n_start_fail} rows have invalid/missing '{start_ts}' (set to NaT).",
                RuntimeWarning,
            )

        # end_ts: ONLY warn on parse failures; legit missing is allowed
        if n_end_fail:
            warnings.warn(
                f"{n_end_fail} rows had non-empty '{end_ts}' values that failed to parse (set to NaT).",
                RuntimeWarning,
            )

        # optional hard stop if all start_ts are NaT
        if self.error_on_all_invalid_times and df[start_ts].isna().all():
            raise ValueError(f"All '{start_ts}' values are NaT after parsing.")

        # We cannot compute duration without start_ts. Drop those rows with a warning.
        miss_start = df[start_ts].isna()
        n_miss_start = int(miss_start.sum())
        if n_miss_start:
            warnings.warn(
                f"{n_miss_start} rows have NaT in '{start_ts}' and will be dropped.",
                RuntimeWarning,
            )
            df = df.loc[~miss_start]

        # --- timezone normalization (handles naive/aware/mixed) ---
        if self.normalize_tz and not df.empty:
            for col in (start_ts, end_ts):
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    raise TypeError(f"Column '{col}' must be datetime64[ns][tz] after parsing.")
            df = self._normalize_timezones(df, (start_ts,end_ts), self.target_tz)

        # --- compute durations ---
        df["duration_td"] = df[end_ts] - df[start_ts]                                # NaT when end_ts is NaT
        df["duration_hr"] = df["duration_td"].dt.total_seconds() / 3600.0  # NaN when duration_td is NaT

        # --- negative duration handling ---
        neg_mask = df["duration_hr"].notna() & (df["duration_hr"] < 0)
        n_neg = int(neg_mask.sum())
        if n_neg:
            msg = f"{n_neg} rows have negative durations (end < start). Policy: {self.negative_duration_policy!r}."
            if self.negative_duration_policy == "drop":
                warnings.warn(msg + " Dropping those rows.", RuntimeWarning)
                df = df.loc[~neg_mask]
            elif self.negative_duration_policy == "nan":
                warnings.warn(msg + " Setting durations to NaN/NaT.", RuntimeWarning)
                df.loc[neg_mask, "duration_hr"] = np.nan
                df.loc[neg_mask, "duration_td"] = pd.NaT
                # Optionally also invalidate end_ts:
                # df.loc[neg_mask, "end_ts"] = pd.NaT
            elif self.negative_duration_policy == "raise":
                raise ValueError(msg)
            else:
                raise ValueError("negative_duration_policy must be one of: 'drop' | 'nan' | 'raise'")

        # --- 'class' column to category (if present) ---
        if self.cast_class_to_category and self.class_column in df.columns:
            if df[self.class_column].dtype.name != "category":
                try:
                    df[self.class_column] = df[self.class_column].astype("category")
                except Exception as e:
                    warnings.warn(
                        f"Failed to cast '{self.class_column}' to category: {e}. Leaving as-is.",
                        RuntimeWarning,
                    )

        # --- empty frame warning ---
        if self.warn_on_empty and df.empty:
            warnings.warn("DataFrame is empty after loading/cleaning.", RuntimeWarning)

        return df

    # ------------------------ helpers ------------------------

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")


    def _parse_dt_with_stats(self, s: pd.Series):
        """
        Returns (parsed_series, n_missing_raw, n_parse_failures).

        - n_missing_raw: rows that were empty/NaN BEFORE parsing.
        - n_parse_failures: rows that had non-empty text but still became NaT after parsing.
        """
        raw = s.copy()
        parsed = pd.to_datetime(
            raw,
            errors="coerce",
            utc=False,
            format=self.date_format,
            dayfirst=self.dayfirst,
        )

        is_missing_raw = raw.isna() | (raw.astype(str).str.strip() == "")
        n_missing_raw = int(is_missing_raw.sum())

        n_parse_fail = int((~is_missing_raw & parsed.isna()).sum())
        return parsed, n_missing_raw, n_parse_fail

    @staticmethod
    def _normalize_timezones(df: pd.DataFrame, cols: Tuple[str, str], target_tz: str) -> pd.DataFrame:
        """
        Normalize tz-naive and tz-aware columns to a single timezone.

        - both naive  -> localize both to target_tz
        - both aware  -> convert both to target_tz
        - mixed       -> localize the naive, convert the aware
        """
        c0, c1 = cols
        tz0 = df[c0].dt.tz
        tz1 = df[c1].dt.tz

        def localize(series: pd.Series) -> pd.Series:
            # Only localize truly naive entries; keep NaT as-is
            if series.dt.tz is not None:
                return series
            return series.dt.tz_localize(target_tz)

        def convert(series: pd.Series) -> pd.Series:
            if series.dt.tz is None:
                return series
            return series.dt.tz_convert(target_tz)

        if tz0 is None and tz1 is None:
            df[c0] = localize(df[c0])
            df[c1] = localize(df[c1])
        elif tz0 is not None and tz1 is not None:
            df[c0] = convert(df[c0])
            df[c1] = convert(df[c1])
        else:
            if tz0 is None:
                df[c0] = localize(df[c0])
                df[c1] = convert(df[c1])
            else:
                df[c0] = convert(df[c0])
                df[c1] = localize(df[c1])

        return df


def csv_to_dataframe(
    csv_path: str,
    normalize_tz: bool = True,
    target_tz: str = "UTC",
    negative_duration_policy: str = "drop",
    args: Namespace = None,
) -> pd.DataFrame:
    """Read CSV into normalized schema: start_ts, end_ts, (optional) class, plus duration columns."""
    loader = CSVLoader(
        normalize_tz=normalize_tz,
        target_tz=target_tz,
        negative_duration_policy=negative_duration_policy,
        date_format=args.date_format,
        dayfirst=args.dayfirst,
    )
    df = loader.load(csv_path)
    df = df.sort_values("start_ts").reset_index(drop=True)
    return df
