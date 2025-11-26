---
title: "<strong>Command Line Reference</strong>"
author: |
  <a href="https://github.com/presence-calculus/samplepath"><em>The Samplepath Analysis Toolkit</em></a>

figures-numbered: true
link-citations: true
toc-title: "Contents"
toc-depth: 2

figPrefix: "Figure"
numberSections: true
sectionsDepth: 2
---
# Samplepath CLI Documentation

This tool generates finite-window Little’s Law charts from an CSV file containing `id`, `start_ts`, `end_ts`, and optionally a `class` column. 
It produces a full set of long run samplepath flow-metrics charts and writes them under an output directory.

---

## CSV Parsing

- **csv** *(positional)*  
  Path to the CSV (`id,start_ts,end_ts[,class]`)

- **--delimiter** *(default: `None`)*  
  Optional delimiter override

- **--start_column** *(default: `start_ts`)*  
  Name of start timestamp column

- **--end_column** *(default: `end_ts`)*  
  Name of end timestamp column

- **--date-format** *(default: `None`)*  
  Explicit datetime format string for parsing

- **--dayfirst** *(default: `False`)*  
  Interpret ambiguous dates as day-first

---

## Data Filters

Drop rows from the CSV before running the analysis. Useful for isolating subprocesses in the main file. Use with `--scenario` to save subprocess results.

- **--completed** *(default: `False`)*  
  Include only items with `end_ts`

- **--incomplete** *(default: `False`)*  
  Include only items without `end_ts`

- **--classes** *(default: `None`)*  
  Comma-separated list of class tags to include

---

## Outlier Trimming

Remove outliers to see whether the remaining process converges.

- **--outlier-hours** *(default: `None`)*  
  Drop items exceeding this many hours in sojourn time

- **--outlier-pctl** *(default: `None`)*  
  Drop items above the given percentile of sojourn times

- **--outlier-iqr** *(default: `None`)*  
  Drop items above Q3 + K·IQR ([Tukey fence](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences))

- **--outlier-iqr-two-sided** *(default: `False`)*  
  Also drop items below Q1 − K·IQR when combined with `--outlier-iqr`

---

## Lambda Fine Tuning

Sometimes it helps to drop early points in the λ(T) chart so the remainder displays on a more meaningful scale.

- **--lambda-pctl** *(default: `None`)*  
  Clip Λ(T) to the upper percentile (e.g., use `99` to clarify charts)

- **--lambda-lower-pctl** *(default: `None`)*  
  Clip the lower bound as well

- **--lambda-warmup** *(default: `0.0`)*  
  Ignore the first H hours when computing Λ(T) percentiles

---

## Convergence Thresholds

- **--epsilon** *(default: `0.05`)*  
  Relative error threshold for convergence

- **--horizon-days** *(default: `28.0`)*  
  Ignore this many initial days when assessing convergence

---

## Output Configuration

- **--output-dir** *(default: `charts`)*  
  Root directory where charts will be written

- **--scenario** *(default: `latest`)*  
  Subdirectory inside the output root

- **--save-input** *(default: `True`)*  
  Copy the input CSV into the output directory

- **--clean** *(default: `False`)*  
  Remove existing charts before writing new results

---

## Process Overview

1. Parse CLI arguments  
2. Create the output directory structure  
3. Optionally copy input CSV  
4. Write CLI parameters into the scenario folder  
5. Run the sample-path analysis  
6. Print paths to generated charts  

---
## Example

samplepath events.csv \
  --completed \
  --outlier-iqr 1.5 \
  --lambda-pctl 99 \
  --output-dir charts \
  --scenario weekly_report \
  --clean

