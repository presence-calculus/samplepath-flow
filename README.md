# The Sample Path Analysis Toolkit

> Deterministic flow analytics for non-deterministic software processes.

[![PyPI](https://img.shields.io/pypi/v/samplepath.svg)](https://pypi.org/project/samplepath/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://samplepath.pcalc.org)

![Sample Path Flow Metrics](docs/assets/sample_path_N.png)

______________________________________________________________________

## The Problem with Flow Metrics Today

You measure throughput, cycle time, WIP. Measure again a bit later and everything has
changed, but you can't explain *why*.

You might think this is simply the nature of software development. That the tools are
faithfully reflecting the highly variable, uncertain nature of the work. But you would
be wrong. This is not a problem with the domain. It's because the metrics are defined
and measured the wrong way, _given_ the nature of the domain.

Current flow metrics tools treat throughput, WIP, and cycle time as independent
statistics measured over a business-facing operational window; say the last 30 days.
The metrics themselves rely on imprecise definitions: they conflate instantaneous WIP with
time-averaged WIP, measure throughput and cycle time over windows different from WIP,
and discard the causal connection between arrival/departure events and their impact on flow.

These are category errors, not implementation details. They explain why you end up chasing some
mythical state called stability where supposedly, these metrics finally behave the
way theory says they should. In practice, we measure flow metrics, periodically review operational dashboards and
"start conversations."

It doesn't have to be this way.

The theory behind Little's Law tells us that flow metrics provide precise, actionable operational insights when all the key
metrics, arrival/departure rates, time-average WIP, and process time, are measured correctly over consistent
observation windows. Applied to software processes that theory implies we
must measure different quantities and in different ways than the manufacturing-inspired techniques we use today.

In short, we need a fundamentally different approach to measuring flow in software processes: sample path analysis.

______________________________________________________________________

## What This Toolkit Does

**samplepath** implements *sample path analysis*: a technique from stochastic process
theory for analyzing flow in processes that may or may not be stable.
The core ideas are nearly 50 years old -- Dr. Shaler Stidham discovered the technique
when he provided the first deterministic proof of Little's Law in 1974 -- but they have
not been turned into practical measurement techniques until now.

The technique works with a single observed trajectory of a flow process as it evolves over
time. It uses the finite-window formulation of Little's Law and establishes _deterministic
cause and effect relationships_ between input and output metrics.

Three properties make it especially suited to processes in the volatile, uncertain, complex and
ambiguous operating environments of real world software development.

- **Distribution-free.** No assumptions about underlying statistical distributions.
  Works even when distributions are non-stationary, poorly defined, or don't exist.
- **Finite-window.** Applies at all times over any finite observation window, without
  requiring steady-state conditions.
- **Deterministic.** Yields deterministic measurements over processes that are inherently
  non-deterministic.

Given a sample path, the measurements are exact quantities not statistical measures.
  Further, _changes_ in the measurements are explainable deterministically in terms of changes in the _inputs_ to the measurements.

This means you can reason rigorously about flow dynamics in processes that operate far
from equilibrium -- exactly the conditions where traditional flow metrics break
down.

[Sample path analysis is not statistics](https://samplepath.pcalc.org/articles/not-statistics)
| [Package overview and history](https://samplepath.pcalc.org/articles/package-overview)

______________________________________________________________________

## What You Get

### Sample Path Flow Metrics

The table below shows these metrics and maps them to the
Lean/Kanban terminology you might be familiar with.

| Metric      | What it measures                     | Lean / Kanban mapping                        |
|-------------|--------------------------------------|----------------------------------------------|
| `A(T)`      | Cumulative arrivals                  | Top line in cumulative flow diagram          |
| `D(T)`      | Cumulative departures                | Bottom line in cumulative flow diagram       |
| `N(t)`      | Instantaneous work in process        | WIP (snapshot)                               |
| `H(T)`      | Cumulative presence mass             | No equivalent                                |
| `L(T)`      | Time-average WIP                     | Average WIP (lossy in current tools)         |
| `Lambda(T)` | Cumulative arrival rate              | Arrival rate (short window in current tools) |
| `Theta(T)`  | Cumulative departure rate            | Throughput (short window in current tools)   |
| `w(T)`      | Average residence time per arrival   | No equivalent                                |
| `w'(T)`     | Average residence time per departure | No equivalent                                |
| `W*(T)`     | Empirical average sojourn time       | Lead Time / Cycle Time (different window)    |


These metrics enable rigorous study of **equilibrium** (arrival/departure rate
convergence), **coherence** (residence time/sojourn time convergence), and
**stability** (convergence of process measures to limits) -- even when the process
has never reached steady state.

For a detailed description of these metrics and their derivations see the [definitions](https://samplepath.pcalc.org/articles/cli/#flow-metrics-csv) in our command
line interface document.

### Charts and Exports

The toolkit generates publication-ready charts and CSV data exports organized into:

- **Core metrics** -- sample path, time-average WIP, arrival rates, residence times,
  Little's Law invariant
- **Convergence diagnostics** -- arrival/departure equilibrium, residence/sojourn
  coherence, multi-panel convergence figures
- **Stability panels** -- WIP growth rate, total age growth rate
- **Advanced** -- invariant manifold, convergence error analysis

[Chart Reference](https://samplepath.pcalc.org/articles/chart-reference)

______________________________________________________________________

## Quick Start

### Install

```bash
uv tool install samplepath
```

This installs the `flow` command globally. (See
[Installation Alternatives](#installation-alternatives) if you use pip or pipx.)

### Run
Assuming `events.csv` in your current directory (see input format below), run

```bash
flow analyze events.csv --completed
```

### Output

Results are saved to the output directory (default: `charts/`):

```
charts/
└── events/
    └── latest/
        ├── input/              # input snapshots
        ├── exports/            # CSV exports of flow metrics and element data
        ├── core/               # core metric charts
        │   └── panels/         # multi-panel core figures
        ├── convergence/        # limit estimates and diagnostics
        │   └── panels/         # multi-panel convergence figures
        ├── stability/
        │   └── panels/         # stability and variance panels
        └── advanced/           # deep-dive charts
```

______________________________________________________________________

## Input Format

A CSV with three columns:

| Column     | Description                              |
|------------|------------------------------------------|
| `id`       | Any string identifier for an element     |
| `start_ts` | Start timestamp of the event (required)  |
| `end_ts`   | End timestamp (empty if still in process)|

An optional `class` column lets you filter by element type (e.g., story, defect).

If your columns have different names, map them with `--start-column` and
`--end-column`. For non-US date formats, use `--day-first` or `--date-format`.

See the [CLI Documentation](https://samplepath.pcalc.org/articles/cli) for the
complete set of options.

______________________________________________________________________

## Examples

The repository includes example datasets and pre-generated analysis output from
the [Polaris](https://www.polaris-flow-dispatch.com) case study:

```bash
# Run the included example
flow analyze examples/polaris/csv/work_tracking.csv --completed --output-dir polaris-analysis

# Filter by item class
flow analyze examples/polaris/csv/work_tracking.csv --classes story --completed

# Remove sojourn time outliers before analysis
flow analyze examples/polaris/csv/work_tracking.csv --outlier-iqr 1.5 --completed
```

Pre-generated output is in `examples/polaris/flow-of-work/`.

______________________________________________________________________

## Learn More

**Articles**

- [Little's Law](https://docs.pcalc.org/articles/littles-law) -- comprehensive
  background on the mathematical foundations
- [Sample Path Analysis is Not Statistics](https://samplepath.pcalc.org/articles/not-statistics) --
  why this technique works where statistical methods struggle
- [Package Overview](https://samplepath.pcalc.org/articles/package-overview) --
  history, significance, and key concepts
- [Chart Reference](https://samplepath.pcalc.org/articles/chart-reference) --
  detailed reference for all computations and charts

**Worked examples**

- [Sample Path Construction](https://www.polaris-flow-dispatch.com/i/172332418/sample-path-construction-for-l%CE%BBw)
- [Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/p/littles-law-in-a-complex-adaptive)
- [The Many Faces of Little's Law](https://www.polaris-flow-dispatch.com/p/the-many-faces-of-littles-law)

**Newsletter**

Subscribe to [The Polaris Flow Dispatch](https://www.polaris-flow-dispatch.com) for
ongoing developments and applications.

______________________________________________________________________

## Installation Alternatives

The recommended install uses [uv](https://docs.astral.sh/uv/):

```bash
uv tool install samplepath
```

**Run without installing** (useful for CI/automation):

```bash
uvx samplepath events.csv --completed
```

**pip** (requires Python 3.11+):

```bash
pip install samplepath
samplepath --help
```

**pipx** (global CLI install):

```bash
pipx install samplepath
flow --help
```

______________________________________________________________________

## Development Setup

For contributors working on the library:

```bash
git clone https://github.com/presence-calculus/samplepath-flow.git
cd samplepath-flow
uv sync --all-extras          # install all dependencies
uv run pytest                 # run tests
uv run flow analyze examples/polaris/csv/work_tracking.csv --help  # run CLI from source
```

______________________________________________________________________

## Package Layout

```
samplepath/
├── cli.py                    # Command-line interface
├── csv_loader.py             # CSV import and parsing
├── data_export.py            # CSV export of flow metrics and element data
├── filter.py                 # Row filtering and outlier trimming
├── metrics.py                # Flow metric calculations
├── limits.py                 # Convergence and limit estimators
├── sample_path_analysis.py   # Top-level analysis orchestration
├── point_process.py          # Point process construction
├── plots/                    # Chart and panel generation
│   ├── core.py               #   Core metric charts
│   ├── convergence.py        #   Convergence diagnostics
│   ├── stability.py          #   Stability panels
│   └── advanced.py           #   Deep-dive charts
└── utils/                    # Shared utilities
test/                         # Pytest suite
```

______________________________________________________________________

## License

Licensed under the **MIT License**.\
See `LICENSE` for details.

Copyright (c) 2025 Dr. Krishna Kumar

Part of [The Presence Calculus Project](https://docs.pcalc.org).
