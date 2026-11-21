"""Render CSV tables using shared defaults and an optional adjacent YAML layout."""

import argparse
from html import escape
from pathlib import Path
import sys
from typing import Any

from great_tables import GT, html
from latex2mathml.converter import convert
import pandas as pd
import yaml

DEFAULT_OPTIONS = {
    "table_width": "640px",
    "table_font_names": ["Arial", "sans-serif"],
    "table_font_size": "18px",
    "heading_align": "left",
    "column_labels_background_color": "#edf2f5",
    "column_labels_font_weight": "bold",
    "data_row_padding": "12px",
}


def render_label(label: str) -> Any:
    """Typeset $...$ spans locally; escape surrounding text as ordinary text."""
    if "$" not in label:
        return label
    parts = []
    text = []
    expression = []
    in_math = False
    index = 0
    while index < len(label):
        char = label[index]
        if char == "\\" and index + 1 < len(label) and label[index + 1] == "$":
            (expression if in_math else text).append("$")
            index += 2
            continue
        if char == "$":
            if in_math:
                formula = "".join(expression).strip()
                if not formula:
                    raise ValueError("Use nonempty inline math: $expression$.")
                try:
                    parts.append(convert(formula, display="inline"))
                except Exception as exc:
                    raise ValueError(f"Cannot render header math: {formula}") from exc
                expression = []
            else:
                parts.append(escape("".join(text)))
                text = []
            in_math = not in_math
        else:
            (expression if in_math else text).append(char)
        index += 1
    if in_math:
        raise ValueError(
            "Unclosed $ in column label; use \\$ for a literal dollar sign."
        )
    parts.append(escape("".join(text)))
    return html("".join(parts))


def read_layout(path: Path) -> dict[str, Any]:
    """Read a layout, rejecting typos rather than silently ignoring settings."""
    layout = yaml.safe_load(path.read_text(encoding="utf-8"))
    if layout is None:
        return {}
    if not isinstance(layout, dict):
        raise ValueError("Layout must be a YAML mapping.")
    unknown = set(layout) - {"title", "subtitle", "note", "columns", "options"}
    if unknown:
        raise ValueError(f"Unknown layout keys: {', '.join(map(str, unknown))}")
    for key in ("title", "subtitle", "note"):
        if key in layout and not isinstance(layout[key], str):
            raise ValueError(f"{key} must be text.")
    for key in ("columns", "options"):
        if key in layout and not isinstance(layout[key], dict):
            raise ValueError(f"{key} must be a mapping.")
    return layout


def build_table(data: pd.DataFrame, layout: dict[str, Any]) -> GT:
    """Apply reusable defaults followed by table-specific overrides."""
    table = GT(data)
    if "title" in layout or "subtitle" in layout:
        table = table.tab_header(
            title=layout.get("title", ""), subtitle=layout.get("subtitle")
        )
    if "note" in layout:
        table = table.tab_source_note(source_note=layout["note"])
    columns = layout.get("columns", {})
    unknown = set(columns) - set(data.columns)
    if unknown:
        raise ValueError(f"Columns absent from CSV: {', '.join(map(str, unknown))}")
    widths = {}
    for column in data.columns:
        settings = columns.get(column, {})
        if not isinstance(settings, dict):
            raise ValueError(f"Settings for {column} must be a mapping.")
        unknown = set(settings) - {"label", "align", "width", "decimals"}
        if unknown:
            raise ValueError(
                f"Unknown settings for {column}: {', '.join(map(str, unknown))}"
            )
        numeric = pd.api.types.is_numeric_dtype(data[column])
        align = settings.get("align", "right" if numeric else "left")
        if align not in {"left", "center", "right"}:
            raise ValueError(f"Invalid alignment for {column}: {align}")
        table = table.cols_align(align=align, columns=column)
        if "label" in settings:
            if not isinstance(settings["label"], str):
                raise ValueError(f"Label for {column} must be text.")
        table = table.cols_label(
            cases={column: render_label(settings.get("label", column))}
        )
        if "width" in settings:
            if not isinstance(settings["width"], str):
                raise ValueError(f"Width for {column} must be text, e.g. '30%'.")
            widths[column] = settings["width"]
        if "decimals" in settings:
            decimals = settings["decimals"]
            if type(decimals) is not int or decimals < 0 or not numeric:
                raise ValueError(
                    f"Decimals for {column} requires a numeric column and a nonnegative integer."
                )
            table = table.fmt_number(columns=column, decimals=decimals)
    if widths:
        table = table.cols_width(widths)
    return table.tab_options(**{**DEFAULT_OPTIONS, **layout.get("options", {})})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dispatch table", description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV source; PNG is saved beside it")
    parser.add_argument(
        "--layout", type=Path, help="Override the adjacent .yaml layout"
    )
    args = parser.parse_args(argv)
    try:
        source = args.csv.resolve()
        output = source.with_suffix(".png")
        if source == output:
            raise ValueError("Input must not be a PNG file.")
        layout_path = (
            args.layout if args.layout is not None else source.with_suffix(".yaml")
        )
        layout = (
            read_layout(layout_path)
            if args.layout is not None or layout_path.exists()
            else {}
        )
        table = build_table(pd.read_csv(source), layout)
        table.gtsave(output, zoom=2, expand=12)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {output}")
    return 0
