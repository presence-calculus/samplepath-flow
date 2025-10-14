# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
import shutil
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
    for chart_dir in ['core', 'core/panels',  'convergence', 'convergence/panels', 'stability', 'advanced', 'misc']:
        sub_dir = os.path.join(out_dir, chart_dir)
        os.makedirs(sub_dir, exist_ok=True)

    return out_dir
