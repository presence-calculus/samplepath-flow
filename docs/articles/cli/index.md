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

# Invocation

This tool provides command line utilities for sample-path analysis of flow-process datasets in CSV
form and writes outputs to the local filesystem.

Invoke it on the command line with

```bash
flow <command> <csv-file> [options]
```

# Commands

- `Analyze`: Generates finite-window Little’s Law charts from a CSV file containing `id`,
`start_ts`, `end_ts`, and optionally a `class` column. It produces a full set of long
run samplepath flow-metrics charts and writes them under an output directory.


# Analyze Command

What it does:

1. Parse CLI arguments
2. Create the output directory structure
3. Copy input CSV under scenario
4. Write CLI parameters into the scenario folder
5. Run the sample-path analysis
6. Generate charts and/or export files and write them to the output directory.
7. Print the paths to generated charts and/or export files.

### Example

```bash
flow analyze events.csv \
  --completed \
  --outlier-iqr 1.5 \
  --lambda-pctl 99 \
  --output-dir charts \
  --scenario weekly_report \
  --clean
```

______________________________________________________________________

# CLI Options

## Inputs and Outputs

### Input Format

The input format is simple. The CSV requires three columns:

- _id_: any string identifier to denote an element/item
- _start_ts_: the start time of an event
- _end_ts_: the end time of an event

Additionally, you may pass other columns. They are ignored for computation (except a
_class_ column used for filtering), but preserved in the elements export.

- If your CSV has different column names, you can map them with `--start-column` and
  `--end-column` options.
- You might need to explicitly pass a date format for the timestamps if you see date
  parsing errors. The `--date-format` argument does this.

Results and charts are saved to the output directory as follows:

- The default output directory is "charts" in your current directory.
- You can override this with the `--output-dir` argument.


### Output Layout

For input `events.csv`, output is organized as:

```bash
<output-dir>/
└── events/
    └── <scenario>/                 # e.g., latest
        ├── input/                  # input snapshots
        ├── exports/                # CSV exports (if enabled)
        ├── core/                   # core metrics & tables
        ├── convergence/            # limit estimates & diagnostics
        ├── convergence/panels/     # multi-panel figures
        ├── stability/panels/       # stability/variance panels
        ├── advanced/               # optional deep-dive charts
```



### Output Configuration

- **--output-dir** *(default: `charts`)*\
  Root directory where outputs are written

- **--scenario** *(default: `latest`)*\
  Subdirectory inside the output root

- **--save-input** *(default: `True`)*\
  Copy the input CSV into the output directory

- **--clean** *(default: `False`)*\
  Remove existing charts before writing new results



## CSV Parsing

- **csv** *(positional)*\
  Path to the CSV. Should contain at least the columns (`id,start_ts,end_ts[, class]`)

- **--delimiter** *(default: `None`)*\
  Optional delimiter override (otherwise auto-detected)

- **--start-column** *(default: `start_ts`)*\
  Name of the start timestamp column

- **--end-column** *(default: `end_ts`)*\
  Name of the end timestamp column

- **--date-format** *(default: `None`)*\
  Explicit datetime format string for parsing

- **--day-first** *(default: `False`)*\
  Treat ambiguous dates as day-first (e.g., 03/04/2025 → April 3, 2025)


## Data Filters

### Row filters

Drop rows from the CSV before running the analysis. Useful for isolating subprocesses in
the main file; use with `--scenario` to save subprocess results.

- **--completed** *(default: `False`)*\
  Include only items with `end_ts`

- **--incomplete** *(default: `False`)*\
  Include only items without `end_ts`

- **--classes** *(default: `None`)*\
  Comma-separated list of class tags to include



### Outlier Trimming

Remove outliers to assess convergence on the remaining process.

- **--outlier-hours** *(default: `None`)*\
  Drop items exceeding this many hours in sojourn time

- **--outlier-pctl** *(default: `None`)*\
  Drop items above the given percentile of sojourn times

- **--outlier-iqr** *(default: `None`)*\
  Drop items above Q3 + K·IQR
  ([Tukey fence](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences))

- **--outlier-iqr-two-sided** *(default: `False`)*\
  Also drop items below Q1 − K·IQR when combined with `--outlier-iqr`

# Analysis Mode

This option selects between **event mode** and **calendar mode** when producing charts and exports.

In **event mode** (the default), metrics are indexed by _events_ and evaluated at exact event timestamps. In **calendar mode**, metrics are indexed by calendar boundaries.

Calendar mode metrics are **always derived by sub-sampling**, not by aggregation. In principle, this behaves as if all metrics are first computed in event mode at the full timestamp resolution of the input data, and calendar mode then selects exact pre-computed values at calendar boundaries for reporting. In practice, values at calendar boundaries are computed by taking definite integrals of the finest-granularity data. In either case, _no information is lost in this process_.

Both modes present identical metric values at shared timestamps; calendar mode simply reports fewer points. This is fundamentally different from the way flow metrics are computed by metrics tools today.

See the metrics definition section under *Exports* for details.



**`--sampling-frequency`** *(default: `None`)*  
Enables calendar mode analysis when set. Charts and exported data are indexed to calendar boundaries rather than raw event timestamps.  
Accepted values: `day`, `week`, `month`, `quarter`, `year`, or pandas aliases like `D`, `W-MON`, `MS`, `QS-JAN`, `YS-JAN`.

In calendar mode:

- Charts display markers at sampled data points.
- X-axis ticks align to calendar boundaries.
- Event overlays are rendered as rug plots at `y = 0`.



**`--anchor`** *(default: `None`)*  
Specifies the anchor for calendar frequency boundaries.

- For `week`: a day name (e.g. `MON`, `WED`, `SUN`)
- For `quarter` or `year`: a month name (e.g. `JAN`, `APR`)
- Ignored for `day` and `month`

Only meaningful when `--sampling-frequency` is set.



## Export Configuration

- **--export-data** *(default: `True`)*\
  Export flow metrics and element data to CSV files alongside charts. Creates two files
  under `<scenario>/exports/`: a flow metrics CSV and an element-level CSV.

- **--export-only** *(default: `False`)*\
  Export flow metrics and element data to CSV files without generating charts. Mutually
  exclusive with `--no-export-data`.



## Chart Configuration

- **--with-event-marks** *(default: `False`)*\
  Show point process event markers on sample path charts. In event mode, arrivals are shown
  as purple dots and departures as green dots with drop lines to the event timestamps. In
  calendar mode, events are shown as rug plots at y = 0.

- **--show-derivations** *(default: `False`)*\
  Show formulas for key metrics in titles/legends of charts (e.g., N(t) = A(T) - D(T)).

- **--chart-format** *(default: `png`)*\
  Chart output format. Use `svg` for scalable vector output.

- **--chart-dpi** *(default: `150`)*\
  DPI for PNG output. Ignored for SVG.


# Charts and Exports

This package produces two outputs: `CSV` files containing flow analysis data and charts
visualizing key metrics, exported as `png` or `svg` images.

The CSV dataset is the normative output of the package; all charts are derived views and
introduce no additional semantics. Treat this dataset as the primary analytical artifact
and data contract, and explore it using other tools.

## Output files
When exports are enabled, two CSV files are written under `<scenario>/exports/`.

- Flow Metrics CSV: A time series CSV with core flow metrics ordered by events or calendar indices depending on analysis mode.
- Elements CSV: An element-wise CSV ordered by start timestamp.

The two output files and the metrics they export are described below.

## Flow Metrics CSV

The output filename depends on the analysis mode (see above):

- **Event mode** (no `--sampling-frequency` is provided): `event_indexed_metrics.csv`
- **Calendar mode** (`--sampling-frequency` is provided): `<frequency>_indexed_metrics.csv`  
  (e.g. `week_indexed_metrics.csv` if `--sampling-frequency week` is provided)



### Event mode flow metrics

Each metric is indexed to one or more events occurring at a single timestamp (mathematically, a *point process*) and represents the cumulative effect of those events on the evolution of flow over the observation window. A key measure of flow accumulation is *cumulative presence mass* (defined below).

Flow metrics are meaningful for reasoning about flow dynamics only when computed in exact event order and accumulated via integration of presence mass.

> **Note:** This is the **only** correct definition of flow metrics for an arrival–departure process when the objective is to reason rigorously about flow dynamics.


### Conventions

- Rows are indexed by event timestamps within a finite observation interval $(t_0, t_N]$.
- By convention, set $t_0 = 0$, and for any event timestamp $t$, define
  $$
  T = t - t_0
  $$
Thus the interval $[0, T]$ is shorthand for $(t_0, t]$, representing the observation window up to the current event; the _duration_ of the interval is $T$.
- Rows are ordered by ascending timestamp.
- Each row corresponds to one or more events (arrival, departure, or both) occurring at the same timestamp.
- Multiple events at the same timestamp are aggregated into a single row, reflecting their cumulative effect on all metrics.
- Metrics indexed by lowercase $t$ are *instantaneous*. $N(t)$ is the only instantaneous metric.
- Metrics indexed by uppercase $T$ are *cumulative*, computed over the interval $[0, T]$.

> All metrics except $N(t)$ are cumulative. Whether a definition depends on the instantaneous time $t$ or the interval length $T$ is intentional and significant.

| Column       | Description                                                                                                                                                  |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `timestamp`  | Event timestamp $t$                                                                                                                                          |
| `element_id` | Element ID: single ID if unique,<br/>A text string with format A:id;id&#124;D:id;id when multiple elements share the timestamp,<br/>or empty if unresolvable |
| `event_type` | `A` (arrival), `D` (departure), `A/D` (both), or `-` (none)                                                                                                  |
| `A(T)`       | Cumulative arrivals up to and including the current timestamp t                                                                                              |
| `D(T)`       | Cumulative departures up to and including the current timestamp t                                                                                            |
| `N(t)`       | Net number in process at this timestamp: $$N(t) = A(T) - D(T)$$                                                                                              |
| `H(T)`       | Cumulative presence mass, the area under N(t): $$H(T) = \int_0^T N(t)\,dt$$                                                                                  |
| `L(T)`       | Time-average of presence mass: $$L(T) = \frac{H(T)}{T}$$                                                                                                     |
| `Lambda(T)`  | Arrival rate: $$\Lambda(T) = \frac{A(T)}{T}$$                                                                                                                |
| `Theta(T)`   | Departure rate: $$\Theta(T) = \frac{D(T)}{T}$$                                                                                                               |
| `w(T)`       | Residence time per arrival: $$w(T) = \frac{H(T)}{A(T)}$$                                                                                                     |
| `w'(T)`      | Residence time per departure: $$w'(T) = \frac{H(T)}{D(T)}$$                                                                                                  |
| `W*(T)`      | Element-wise empirical mean sojourn time for completed items: $$W^*(T) = \text{Avg}(d_i - a_i), \quad d_i \in (0, T]$$                                       |

### Calendar mode flow metrics

Calendar mode reports the *same metrics* as event mode, evaluated at fixed calendar boundaries rather than at individual events.

### Conventions

- Rows correspond to calendar boundary timestamps determined by `--sampling-frequency` (and `--anchor`, if specified).
- For each boundary timestamp $t$, define $T = t - t_0$ exactly as in event mode.
- All metrics have the same mathematical definitions as in event mode and are evaluated over the interval $[0, T]$.
- Calendar mode omits `element_id` and `event_type`.

| Column        | Description                                                                                        |
|---------------|----------------------------------------------------------------------------------------------------|
| `<frequency>` | Calendar boundary timestamp $t$                                                                    |
| `A(T)`        | Cumulative arrivals: $A(T)$                                                                        |
| `D(T)`        | Cumulative departures: $D(T)$                                                                      |
| `N(T)`        | Work in process at the boundary: $$N(T) = N(t)$$                                                   |
| `H(T)`        | Cumulative presence mass: $$H(T) = \int_0^T N(t)\,dt$$                                             |
| `L(T)`        | Time-average WIP: $$L(T) = \frac{H(T)}{T}$$                                                        |
| `Lambda(T)`   | Arrival rate: $$\Lambda(T) = \frac{A(T)}{T}$$                                                      |
| `Theta(T)`    | Departure rate: $$\Theta(T) = \frac{D(T)}{T}$$                                                     |
| `w(T)`        | Residence time per arrival: $$w(T) = \frac{H(T)}{A(T)}$$                                           |
| `w'(T)`       | Residence time per departure (completed items only): $$w'(T) = \frac{H(T)}{D(T)}$$                 |
| `W*(T)`       | Element-wise empirical mean sojourn time: $$W^*(T) = \text{Avg}(d_i - a_i), \quad d_i \in (0, T]$$ |


### Lean/Kanban Flow Metrics Mappings

The following table defines the normative mapping between sample path flow metrics and common industry terminology used in Lean/Kanban practice and commercial tooling.

> **Normative statement**
>
> Sample path flow metrics, as defined above, are the canonical and mathematically correct definitions of flow metrics. They are the only definitions that allow rigorous reasoning about flow using Little’s Law in both stable and unstable processes, consistently.
>
> Industry-standard definitions and Lean/Kanban tools commonly diverge from these definitions in the following ways:
>
> - Metrics are defined over calendar buckets (days, weeks, etc.) rather than event timestamps.
> - Instantaneous WIP and time-average WIP are conflated when averaging instantaneous WIP over calendar buckets _before_ computing the initial cumulative flow diagrams. At this point any causal reasoning is impossible because the connection between events and their impact on flow has already been discarded.
> - While CFDs are computed over the long run (as they should be), flow metrics like cycle time and throughput are computed over short operational windows rather than cumulatively over the same long observation horizon that CFDs are constructed over.  Consistency of observation windows is required to reason rigorously about flow dynamics such as stability and convergence, and why the industry standard flow metrics are not helpful in this regard.
>
> Any flow metric that does not preserve event ordering, cumulative integration of presence mass, and long-run convergence properties should be treated as a point-in-time reporting snapshot of unrelated flow statistics over a process, not as a metric suitable for rigorous reasoning about flow dynamics.

The mappings in the table below should be interpreted as *terminological correspondences only* and do not imply equivalence of computation or mathematical meaning.



| Metric        | Vernacular name                      | Rough Lean / Kanban mapping                                                                                                                                    |
|---------------|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `A(T)`        | Cumulative arrivals                  | Top line in cumulative flow diagram  (lossy)                                                                                                                   |
| `D(T)`        | Cumulative departures                | Bottom line in cumulative flow diagram (lossy)                                                                                                                 |
| `N(t)`        | Instantaneous Work in process        | WIP: the length of the line between two lines at the same calendar bucket in the CFD                                                                           |
| `H(T)`        | Cumulative presence mass             | Not measured - informally recognized, visually, as the area in between the two lines in the CFD                                                                |
| `L(T)`        | Time Average of WIP                  | Average WIP at fixed calendar buckets (lossy)                                                                                                                  |
| `Lambda(T)`   | Cumulative arrival rate              | Arrival rate (within a short operational window)                                                                                                               |
| `Theta(T)`    | Cumulative departure rate            | Throughput (within a short operational window)                                                                                                                 |
| `w(T)`        | Average residence time per arrival   | No equivalent                                                                                                                                                  |
| `w'(T)`       | Average residence time per departure | No equivalent                                                                                                                                                  |
| `W*(T)`       | Empirical average sojourn time       | Lead Time/Cycle Time etc. depending on the arrival/departure event semantics. Usually measured over a different window compared to Throughput/Arrival rate/WIP |






## Element CSV — `elements.csv`

One row per element, sorted by `start_ts` ascending.

| Column           | Description                                             |
|------------------|---------------------------------------------------------|
| `element_id`     | Element identifier (renamed from input `id`)            |
| `start_ts`       | Start timestamp $a_i$                                   |
| `end_ts`         | End timestamp $d_i$ (empty for incomplete items)        |
| *input columns*  | All remaining columns from the input CSV                |
| `sojourn_time`   | Element sojourn time: $$d_i - a_i$$ (NaN if incomplete) |
| `residence_time` | Element residence time within the observation window    |

### Residence time definition

Let the observation window be $(t_0, t_n]$.

- For completed items:
  $$
  \text{residence\_time} = d_i - a_i
  $$
- For incomplete items:
  $$
  \text{residence\_time} = t_n - \max(a_i, t_0)
  $$

## Charts
A complete reference to the charts produced can be found in
[The Chart Reference]($document-root/articles/chart-reference).

# Esoteric settings

These settings are useful for configuring the visualizations and parameters of specific charts and analyses.

## Lambda Fine Tuning

Sometimes it helps to drop early points in the λ(T) chart so the remainder displays on a
more meaningful scale.

- **--lambda-pctl** *(default: `None`)*\
  Clip Λ(T) to the upper percentile (e.g., use `99` to clarify charts)

- **--lambda-lower-pctl** *(default: `None`)*\
  Clip the lower bound as well

- **--lambda-warmup** *(default: `0.0`)*\
  Ignore the first H hours when computing Λ(T) percentiles



## Convergence Thresholds

- **--epsilon** *(default: `0.05`)*\
  Relative error threshold for convergence

- **--horizon-days** *(default: `28.0`)*\
  Ignore this many initial days when assessing convergence
