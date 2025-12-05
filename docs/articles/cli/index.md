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
sectionsDepth: 1
# Configure figures
figPrefix: Figure
# Configure citations
citations: false
---

# Invocation

This tool generates finite-window Little’s Law charts from an CSV file containing `id`,
`start_ts`, `end_ts`, and optionally a `class` column. It produces a full set of long
run samplepath flow-metrics charts and writes them under an output directory.

Invoke it on the command line with

```bash
samplepath <csv-file> [options]
```

# Options

## CSV Parsing

- **csv** *(positional)*\
  Path to the CSV. Should contain at least the columns (`id,start_ts,end_ts[, class]`)

- **--delimiter** *(default: `","`)*\
  Optional delimiter override

- **--start_column** *(default: `start_ts`)*\
  Name of start timestamp column

- **--end_column** *(default: `end_ts`)*\
  Name of end timestamp column

- **--date-format** *(default: `None`)*\
  Explicit datetime format string for parsing

______________________________________________________________________

## Data Filters

Drop rows from the CSV before running the analysis. Useful for isolating subprocesses in
the main file. Use with `--scenario` to save subprocess results.

- **--completed** *(default: `False`)*\
  Include only items with `end_ts`

- **--incomplete** *(default: `False`)*\
  Include only items without `end_ts`

- **--classes** *(default: `None`)*\
  Comma-separated list of class tags to include

______________________________________________________________________

## Outlier Trimming

Remove outliers to see whether the remaining process converges.

- **--outlier-hours** *(default: `None`)*\
  Drop items exceeding this many hours in sojourn time

- **--outlier-pctl** *(default: `None`)*\
  Drop items above the given percentile of sojourn times

- **--outlier-iqr** *(default: `None`)*\
  Drop items above Q3 + K·IQR
  ([Tukey fence](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences))

- **--outlier-iqr-two-sided** *(default: `False`)*\
  Also drop items below Q1 − K·IQR when combined with `--outlier-iqr`

______________________________________________________________________

## Lambda Fine Tuning

Sometimes it helps to drop early points in the λ(T) chart so the remainder displays on a
more meaningful scale.

- **--lambda-pctl** *(default: `None`)*\
  Clip Λ(T) to the upper percentile (e.g., use `99` to clarify charts)

- **--lambda-lower-pctl** *(default: `None`)*\
  Clip the lower bound as well

- **--lambda-warmup** *(default: `0.0`)*\
  Ignore the first H hours when computing Λ(T) percentiles

______________________________________________________________________

## Convergence Thresholds

- **--epsilon** *(default: `0.05`)*\
  Relative error threshold for convergence

- **--horizon-days** *(default: `28.0`)*\
  Ignore this many initial days when assessing convergence

______________________________________________________________________

## Output Configuration

- **--output-dir** *(default: `charts`)*\
  Root directory where charts will be written

- **--scenario** *(default: `latest`)*\
  Subdirectory inside the output root

- **--save-input** *(default: `True`)*\
  Copy the input CSV into the output directory

- **--clean** *(default: `False`)*\
  Remove existing charts before writing new results

______________________________________________________________________

## What the command does

1. Parse CLI arguments
2. Create the output directory structure
3. Copy input CSV under scenario
4. Write CLI parameters into the scenario folder
5. Run the sample-path analysis
6. Generate the charts and write to the output directory.
7. Print paths to generated charts

______________________________________________________________________

## Example

```bash
samplepath events.csv \
  --completed \
  --outlier-iqr 1.5 \
  --lambda-pctl 99 \
  --output-dir charts \
  --scenario weekly_report \
  --clean
```

______________________________________________________________________

# Inputs and Outputs

## Input Format

The input format is simple.

The csv requires three columns

- _id_: any string identifier to denote an element/item
- _start_ts_: the start time of an event
- _end_ts_: the end time of an event

Additionally you may pass any other columns. They are all ignored for now, except for a
column called _class_ which you can use to filter results by event/item type.

- If your csv has different column names, you can map them with `--start_column` and
  `--end_column` options.
- You might need to explicitly pass a date format for the time stamps if you see date
  parsing errors. The `--date-format` argument does this.

Results and charts are saved to the output directory as follows:

- The default output directory is "charts" in your current directory.
- You can override this with the --output-dir argument.

See the [CLI Documentation](docs/src/cli.md) for the full list of command line options.

## Output Layout

For input `events.csv`, output is organized as:

```bash
<output-dir>/
└── events/
    └── <scenario>/                 # e.g., latest
        ├── input/                  # input snapshots
        ├── core/                   # core metrics & tables
        ├── convergence/            # limit estimates & diagnostics
        ├── convergence/panels/     # multi-panel figures
        ├── stability/panels/       # stability/variance panels
        ├── advanced/               # optional deep-dive charts
        └── misc/                   # ancillary artifacts
```

\--

A complete reference to the charts produced can be found in
[The Chart Reference](chart-reference/index).
