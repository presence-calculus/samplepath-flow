# -*- coding: utf-8 -*-
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Finite-window Little's Law charts from intervals CSV")
    parser.add_argument("--completed", action="store_true",
                        help="Only include items with an end_ts (completed work only)")
    parser.add_argument("--incomplete", action="store_true", help="Only include items without an end_ts (aging view)")
    parser.add_argument("--with-A", action="store_true",
                        help="Also render cumulative area A(T)=∫N(t)dt charts and 5-panel stacks")
    parser.add_argument("--with-daily-breakdown", action="store_true",
                        help="Add daily ΔA (WIP-hours/day) and daily average WIP charts")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated list of class tags to include (requires a 'class' column)")
    parser.add_argument("--outlier-hours", type=float, default=None,
                        help="Drop completed items whose (end_ts - start_ts) exceeds this many hours")
    parser.add_argument("--outlier-pctl", type=float, default=None,
                        help="Drop completed items above the Pth percentile of duration (0<P<100) after other filters")
    parser.add_argument("--outlier-iqr", type=float, default=None,
                        help="Drop completed items above Q3+K·IQR (Tukey high fence); pass K (e.g., 1.5)")
    parser.add_argument("--outlier-iqr-two-sided", action="store_true",
                        help="Also drop items below Q1−K·IQR when used with --outlier-iqr")
    parser.add_argument("--lambda-pctl", type=float, default=None,
                        help="Clip Λ(T) y-axis to the upper Pth percentile (0<P<100)")
    parser.add_argument("--lambda-lower-pctl", type=float, default=None,
                        help="Optionally clip the lower end to the Pth percentile as well")
    parser.add_argument("--lambda-warmup", type=float, default=0.0,
                        help="Ignore the first H hours when computing Λ(T) percentiles")
    parser.add_argument("--scatter", action="store_true",
                        help="Overlay scatter of per-item time-in-system at completion on w(T) charts")
    parser.add_argument("--epsilon", type=float, default=0.10,
                        help="Relative error threshold for coherence (default 0.10 = 10%)")
    parser.add_argument("--horizon-days", type=float, default=28.0,
                        help="Minimum horizon in days when assessing coherence (default 28)")
    parser.add_argument("csv", type=str, help="Path to CSV (id,start_ts,end_ts[,class])")
    return parser


def get_class_filters(classes):
    class_filters = None
    if classes:
        class_filters = [c for c in classes.split(',') if c.strip() != '']
    return class_filters
