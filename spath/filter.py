# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# filters.py
# -*- coding: utf-8 -*-
# filters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from argparse import Namespace

import numpy as np
import pandas as pd


# ---------- Spec / Result ----------

@dataclass
class FilterSpec:
    completed_only: bool = False
    incomplete_only: bool = False
    classes: Optional[str] = None        # comma-separated string
    outlier_hours: Optional[float] = None
    outlier_pctl: Optional[float] = None
    outlier_iqr: Optional[float] = None
    outlier_iqr_two_sided: bool = False
    raise_on_empty_classes: bool = True
    copy_result: bool = False

@dataclass
class FilterResult:
    df: pd.DataFrame
    label: str
    applied: List[str]
    dropped_per_filter: Dict[str, int]
    thresholds: Dict[str, float]


# ---------- Helpers ----------

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

def _parse_classes(s: Optional[str]) -> list[str] | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    seen, out = set(), []
    for p in parts:
        v = p.lower()
        if v not in seen:
            seen.add(v); out.append(v)
    return out or None

def _completed_base_label(spec: FilterSpec) -> str:
    if spec.completed_only:
        return "completed work"
    if spec.incomplete_only:
        return "incomplete (aging view)"
    return "all work"

def _comp_mask(df: pd.DataFrame, cur_mask: pd.Series) -> pd.Series:
    return cur_mask & df["end_ts"].notna()


# ---------- Individual filter functions (mask-first) ----------

def _f_completed_only(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int]
) -> pd.Series:
    if not spec.completed_only:
        return mask
    _require_cols(df, ["end_ts"])
    new_mask = mask & df["end_ts"].notna()
    dropped["completed_only"] = int((mask & ~new_mask).sum())
    applied.append("completed_only")
    return new_mask

def _f_incomplete_only(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int]
) -> pd.Series:
    if not spec.incomplete_only:
        return mask
    _require_cols(df, ["end_ts"])
    new_mask = mask & df["end_ts"].isna()
    dropped["incomplete_only"] = int((mask & ~new_mask).sum())
    applied.append("incomplete_only")
    return new_mask

def _f_classes(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int]
) -> Tuple[pd.Series, Optional[str]]:
    norm = _parse_classes(spec.classes)
    if not norm:
        return mask, None
    _require_cols(df, ["class"])
    wanted = set(norm)
    new_mask = mask & df["class"].astype(str).str.lower().isin(wanted)
    if spec.raise_on_empty_classes and int(new_mask.sum()) == 0:
        raise ValueError(f"No rows match the requested classes: {norm}")
    dropped["classes"] = int((mask & ~new_mask).sum())
    applied.append(f"classes={','.join(norm)}")
    label_add = f", classes: {','.join(norm)}"
    return new_mask, label_add

def _f_outlier_hours(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int], outlier_tags: List[str]
) -> pd.Series:
    if spec.outlier_hours is None:
        return mask
    _require_cols(df, ["end_ts", "duration_hr"])
    hrs = float(spec.outlier_hours)
    new_mask = mask & (df["end_ts"].isna() | (df["duration_hr"] <= hrs))
    dropped_count = int((mask & ~new_mask).sum())
    dropped["outlier_hours"] = dropped_count
    applied.append(f"outlier_hours<={hrs:g}h")
    if dropped_count > 0:
        outlier_tags.append(f">{hrs:g}h")
    return new_mask

def _f_outlier_pctl(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int], thresholds: Dict[str, float],
    outlier_tags: List[str]
) -> pd.Series:
    if spec.outlier_pctl is None:
        return mask
    _require_cols(df, ["end_ts", "duration_hr"])
    p = float(spec.outlier_pctl)
    if not (0.0 < p < 100.0):
        raise ValueError(f"--outlier-pctl must be between 0 and 100 (got {spec.outlier_pctl})")
    comp = _comp_mask(df, mask)
    if not comp.any():
        return mask
    thresh = float(np.nanpercentile(df.loc[comp, "duration_hr"].to_numpy(), p))
    thresholds[f"pctl{p:g}_hr"] = thresh
    new_mask = mask & (df["end_ts"].isna() | (df["duration_hr"] <= thresh))
    dropped_count = int((mask & ~new_mask).sum())
    dropped["outlier_pctl"] = dropped_count
    applied.append(f"outlier_pctl<={p:g} (th={thresh:.2f}h)")
    if dropped_count > 0:
        outlier_tags.append(f">p{p:g} (>{thresh:.2f}h)")
    return new_mask

def _f_outlier_iqr(
    df: pd.DataFrame, mask: pd.Series, spec: FilterSpec,
    applied: List[str], dropped: Dict[str, int], thresholds: Dict[str, float],
    outlier_tags: List[str]
) -> pd.Series:
    if spec.outlier_iqr is None:
        return mask
    _require_cols(df, ["end_ts", "duration_hr"])
    k = float(spec.outlier_iqr)
    comp_vals = df.loc[_comp_mask(df, mask), "duration_hr"].dropna().to_numpy()
    if comp_vals.size < 4:
        return mask
    q1, q3 = np.nanpercentile(comp_vals, [25, 75])
    iqr = q3 - q1
    high_fence = q3 + k * iqr
    low_fence = q1 - k * iqr
    thresholds["iqr_q1_hr"] = float(q1)
    thresholds["iqr_q3_hr"] = float(q3)
    thresholds["iqr_high_hr"] = float(high_fence)
    if spec.outlier_iqr_two_sided:
        thresholds["iqr_low_hr"] = float(low_fence)

    keep = df["end_ts"].isna() | (df["duration_hr"] <= high_fence)
    if spec.outlier_iqr_two_sided:
        keep &= df["end_ts"].isna() | (df["duration_hr"] >= low_fence)

    new_mask = mask & keep
    dropped_count = int((mask & ~new_mask).sum())
    dropped["outlier_iqr"] = dropped_count
    if spec.outlier_iqr_two_sided:
        applied.append(f"outlier_iqr k={k:g} two-sided")
    else:
        applied.append(f"outlier_iqr k={k:g}")
    if dropped_count > 0:
        outlier_tags.append(f">Q3+{k:g}·IQR (>{high_fence:.2f}h)")
        if spec.outlier_iqr_two_sided:
            outlier_tags.append(f"<Q1−{k:g}·IQR (<{low_fence:.2f}h)")
    return new_mask


# ---------- Main driver ----------

def run_filters(df: pd.DataFrame, spec: FilterSpec) -> FilterResult:
    _require_cols(df, ["start_ts", "end_ts", "duration_hr"])

    if spec.completed_only and spec.incomplete_only:
        raise ValueError("--completed and --incomplete are mutually exclusive")

    mask = pd.Series(True, index=df.index)
    applied: List[str] = []
    dropped: Dict[str, int] = {}
    thresholds: Dict[str, float] = {}
    outlier_tags: List[str] = []

    label = _completed_base_label(spec)

    # Apply each filter via its function
    mask = _f_completed_only(df, mask, spec, applied, dropped)
    mask = _f_incomplete_only(df, mask, spec, applied, dropped)
    new_mask, label_add = _f_classes(df, mask, spec, applied, dropped)
    mask = new_mask
    if label_add:
        label += label_add

    mask = _f_outlier_hours(df, mask, spec, applied, dropped, outlier_tags)
    mask = _f_outlier_pctl(df, mask, spec, applied, dropped, thresholds, outlier_tags)
    mask = _f_outlier_iqr(df, mask, spec, applied, dropped, thresholds, outlier_tags)

    # Finalize DF (single slice)
    df_out = df.loc[mask]
    if spec.copy_result:
        df_out = df_out.copy()

    # Finalize label for outliers (only if anything actually dropped)
    if outlier_tags:
        # Confirm at least one outlier filter dropped rows
        if any(dropped.get(k, 0) > 0 for k in ("outlier_hours", "outlier_pctl", "outlier_iqr")):
            label += f", outliers {' & '.join(outlier_tags)} removed"

    return FilterResult(
        df=df_out,
        label=label,
        applied=applied,
        dropped_per_filter=dropped,
        thresholds=thresholds,
    )


def apply_filters(df: pd.DataFrame, args: Namespace) -> FilterResult:
    spec = FilterSpec(
        completed_only=args.completed,
        incomplete_only=args.incomplete,
        classes=args.classes,
        outlier_hours=args.outlier_hours,
        outlier_pctl=args.outlier_pctl,
        outlier_iqr=args.outlier_iqr,
        outlier_iqr_two_sided=args.outlier_iqr_two_sided,
        copy_result=False,
    )
    return run_filters(df, spec)


