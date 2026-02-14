---
title: <strong>Command Line Reference</strong>
author: |
  <a href="https://github.com/presence-calculus/samplepath"><em>The Samplepath Analysis Toolkit</em></a>

# Configure TOC
toc: true
toc-title: Contents
toc-depth: 3
# Configure section numbers
numberSections: true
sectionsDepth: 2
# Configure figures
figPrefix: Figure
# Configure citations
citations: false

document-root: ../..

---

# Scope

This document is the reference for the `flow`/`samplepath` command-line interface.
It defines:

- command invocation,
- option semantics,
- output layout, and
- CSV export schemas.

For mathematical definitions and conceptual background, see
[Sample Path Theory]($document-root/articles/theory).

For file-by-file chart interpretation, see
[Chart Reference]($document-root/articles/chart-reference).

# Invocation

Use either script name:

```bash
flow analyze <csv-file> [options]
```

```bash
samplepath analyze <csv-file> [options]
```

If you pass a CSV path directly, the CLI treats it as `analyze`.

# Analyze Command

`analyze` runs the full pipeline:

1. Parse and validate CLI arguments.
2. Load CSV and apply filters.
3. Compute finite-window metrics.
4. Write exports (unless disabled).
5. Render charts (unless `--export-only`).

## Example

```bash
flow analyze events.csv \
  --completed \
  --outlier-iqr 1.5 \
  --lambda-pctl 99 \
  --output-dir charts \
  --scenario weekly_report \
  --clean
```

# Input Contract

The input CSV must contain:

- `id`
- `start_ts`
- `end_ts`

Optional columns are preserved in `elements.csv`. If present, `class` can be used with
`--classes`.

Column names can be remapped with:

- `--start-column`
- `--end-column`

# Output Layout

Given `events.csv`, outputs are written under:

```bash
<output-dir>/
└── events/
    └── <scenario>/
        ├── input/
        ├── exports/
        ├── core/
        │   └── panels/
        ├── convergence/
        │   └── panels/
        ├── stability/
        │   └── panels/
        └── advanced/
```

Top-level summary charts are also written at `<scenario>/`.

# Analysis Modes

## Event-Indexed Mode (default)

When `--sampling-frequency` is not set:

- metrics are evaluated at event timestamps,
- output export filename is `event_indexed_metrics.csv`.

## Calendar-Indexed Mode

When `--sampling-frequency` is set:

- the same cumulative metrics are evaluated at calendar boundaries,
- output export filename is `<sampling_frequency>_indexed_metrics.csv`.

`--anchor` configures boundary anchors:

- week: day name (for example `MON`),
- quarter/year: month name (for example `JAN`).

For metric semantics and why this remains a sub-sampled cumulative view, see
[Sample Path Theory]($document-root/articles/theory#event-indexed-vs-calendar-indexed-views).

# Options

## CSV Parsing

- `csv` (positional): path to input CSV
- `--delimiter`: optional delimiter override
- `--start-column` (default `start_ts`)
- `--end-column` (default `end_ts`)
- `--date-format`: explicit datetime format string
- `--day-first`: parse ambiguous dates as day-first

## Data Filters

- `--completed`: include only rows with `end_ts`
- `--incomplete`: include only rows without `end_ts`
- `--classes`: comma-separated class filter list

`--completed` and `--incomplete` are mutually exclusive.

## Outlier Trimming

- `--outlier-hours`: drop items above fixed duration (hours)
- `--outlier-pctl`: drop items above duration percentile
- `--outlier-iqr`: drop items above `Q3 + K*IQR`
- `--outlier-iqr-two-sided`: also drop below `Q1 - K*IQR` when using IQR rule

## Lambda Fine Tuning

- `--lambda-pctl`: upper percentile clip for `Lambda(T)` chart axis
- `--lambda-lower-pctl`: optional lower percentile clip
- `--lambda-warmup`: warmup hours excluded from percentile estimation

## Convergence Thresholds

- `--epsilon`: relative error threshold used in convergence diagnostics
- `--horizon-days`: initial days excluded from convergence assessment

## Output Configuration

- `--output-dir` (default `charts`)
- `--scenario` (default `latest`)
- `--save-input` (enabled by default)
- `--clean`: clear existing scenario outputs before writing

## Chart Configuration

- `--with-event-marks`
- `--show-derivations`
- `--chart-format {png,svg}`
- `--chart-dpi` (PNG only)
- `--grid-lines` / `--no-grid-lines`
- `--sampling-frequency`
- `--anchor`

## Export Configuration

- `--export-data` / `--no-export-data`
- `--export-only`

`--no-export-data` and `--export-only` cannot be combined.

# CSV Exports

When exports are enabled, two files are written under `<scenario>/exports/`.

## Flow Metrics Export

### Event-indexed filename

`event_indexed_metrics.csv`

### Event-indexed columns

| Column | Meaning |
| --- | --- |
| `timestamp` | event time |
| `element_id` | resolved element ID(s) at timestamp |
| `event_type` | `A`, `D`, `A/D`, or `-` |
| `A(T)` | cumulative arrivals |
| `D(T)` | cumulative departures |
| `N(t)` | instantaneous WIP |
| `H(T)` | cumulative presence mass |
| `L(T)` | time-average presence |
| `Lambda(T)` | cumulative arrival rate |
| `Theta(T)` | cumulative departure rate |
| `w(T)` | residence time per arrival |
| `w'(T)` | residence time per departure |
| `W*(T)` | empirical mean sojourn of completed items |

### Calendar-indexed filename

`<sampling_frequency>_indexed_metrics.csv` (for example `week_indexed_metrics.csv`)

### Calendar-indexed columns

| Column | Meaning |
| --- | --- |
| `<sampling_frequency>` | calendar boundary timestamp |
| `A(T)` | cumulative arrivals |
| `D(T)` | cumulative departures |
| `N(T)` | WIP at boundary |
| `H(T)` | cumulative presence mass |
| `L(T)` | time-average presence |
| `Lambda(T)` | cumulative arrival rate |
| `Theta(T)` | cumulative departure rate |
| `w(T)` | residence time per arrival |
| `w'(T)` | residence time per departure |
| `W*(T)` | empirical mean sojourn of completed items |

## Element Export

Filename: `elements.csv`

Core columns:

- `element_id` (renamed from input `id`)
- `start_ts`
- `end_ts`
- original passthrough input columns
- `sojourn_time`
- `residence_time`

# Charts

Generated chart files are documented in
[Chart Reference]($document-root/articles/chart-reference).

# Additional Examples

## Calendar-indexed weekly reporting

```bash
flow analyze events.csv \
  --sampling-frequency week \
  --anchor MON \
  --scenario weekly_monday
```

## Export-only run

```bash
flow analyze events.csv --export-only --scenario exports_only
```
