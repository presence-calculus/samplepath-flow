# The Sample Path Analysis Library and Toolkit

A reference implementation of sample-path flow metrics, convergence analysis, and
stability diagnostics for flow processes using the finite
window formulation of **[Little's Law](https://docs.pcalc.org/articles/littles-law)**.

See documentation [here](https://samplepath.pcalc.org).

[![PyPI](https://img.shields.io/pypi/v/samplepath.svg)](https://pypi.org/project/samplepath/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://py.pcalc.org)

![Sample Path Flow Metrics](docs/assets/sample_path_N.png)

______________________________________________________________________

# 1. Overview

**samplepath** is a Python library for analyzing _macro dynamics_ of flow processes in
complex adaptive systems. It provides deterministic tools to precisely describe the
_long-run_ behavior of stochastic flow processes:

- Arrival/departure equilibrium
- Process time coherence, and
- Process stability

using the finite-window formulation of
[**Little’s Law**](https://docs.pcalc.org/articles/littles-law).

The focus of the analysis is a single _sample path_ of a flow process: a _continuous_
real-valued function that describes a particular process behavior when observed over a
long, but finite period of time.

A key aspect of this technique is that it is _distribution-free_. It does not require
well-defined statistical or probability distributions to reason rigorously about a flow
process. Please see
[sample path analysis is not a statistical method](https://samplepath.pcalc.org/articles/not-statistics)
for more details.

As a result, this technique allows us to extend many results from stochastic process
theory to processes operating in complex adaptive systems, where stable statistical
distributions often don't exist.

If you are new to Little's Law or are only familiar with the traditional idea of
Little's Law from manufacuring applications, please see our overview article on
[**Little’s Law**](https://docs.pcalc.org/articles/littles-law).

Our focus is operations management in software development, but the techniques here are
much more general.

[More background and history is here ...](https://samplepath.pcalc.org/articles/package-overview)

______________________________________________________________________

# 2. Data Requirements and Key Metrics

The data requirements for the sample path analysis of a flow process are minimal: a CSV
file that represents the observed timeline of a binary flow process with element ID,
start, and end date columns.

- The start and end dates may be empty, but for a meaningful analysis, we require at
  least some of these dates be non-empty. Empty end dates denote elements that have
  started but not ended. Empty start dates denote items whose start date is unknown.
  Both are considered elements currently present in the boundary.
- The system boundary is optional (the name of the CSV file becomes the default name of
  the boundary). Boundaries become useful when we start to model the dynamics of
  interconnected flow processes.

Given this input, this library implements:

A. Core Python modules that implement the computations for sample path construction and
analysis:

- Time-averaged flow metrics governed by the finite version of Little's Law `N(t)`,
  `L(T)`, `Λ(T)`, `w(T)`, `λ*(T)`, `W*(T)`
- Performing *equilibrium* and **coherence** calculations (e.g., verifying
  `L(T) ≈ λ*(T)·W*(T)`)
- Estimating empirical **limits** with uncertainty and **tail** checks to verify
  stability (alpha)

B. Command line tools provide utilities that wrap these calculations

- Simple workflows that take CSV files as input to run sample path analysis with a rich
  set of parameters and options.
- Generate publication-ready **charts and panel visualizations** as static png files.
- The ability to save different parametrized analyses from a single CSV file as named
  scenarios.

## Sample Path Flow Metrics

Deterministic, finite-window analogues of Little’s Law:

| Quantity | Meaning                                               |
| -------- | ----------------------------------------------------- |
| `L(T)`   | Average work-in-process over window `T`               |
| `Λ(T)`   | Cumulative arrivals per unit time up to `T`           |
| `w(T)`   | Average residence time over window `T`                |
| `λ*(T)`  | Empirical arrival rate up to `T`                      |
| `W*(T)`  | Empirical mean sojourn time of items completed by `T` |

These quantities enable rigorous study of **equilibrium** (arrival/departure rate
convergence), **coherence** (residence time/sojourn time convergence), and **stability**
(convergence of process measures to limits) even when processes operate far from steady
state.

Please see
[Sample Path Construction](https://www.polaris-flow-dispatch.com/i/172332418/sample-path-construction-for-l%CE%BBw)
for background on what these metrics mean.

Please see
[Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/p/littles-law-in-a-complex-adaptive)
for a worked example on how to apply the concepts.

## Computations and Charts

For a detailed reference of the computations, charts and visualizations produced by
sample path analysis, please see the
[Chart Reference](http://samplepath.pcalc.org/articles/chart-reference).

For complete documentation, see our [documentation site](http://samplepath.pcalc.org).

______________________________________________________________________

# 3. Package Scope

This package is a part of [The Presence Calculus Project](https://docs.pcalc.org): an
open source computational toolkit that is intended to make sample path methods and
concepts more accessible to practitioners working on operations management problems in
the software industry including engineering/product/sales/marketing operations and
related disciplines: value stream management, developer experience and platforms, and
lean continuous process improvement.

This library and toolkit is intended to be used by practitioners to understand the
theory and _develop their intuition about the dynamics of flow processes_, using their
own environments. Understanding the context behind the data greatly helps makes the
abstract ideas here concrete and there in no substitute for getting your hands dirty and
trying things out directly. This toolkit is designed for that.

It is not ready nor intended to support production quality operations management
tooling.

[See more..](https://samplepath.pcalc.org/package-overview/goals.html)

______________________________________________________________________

# 4. Installation (End Users)

## Quick Start with uv (Recommended)

**uv** is a fast, modern Python package manager that handles your setup.

### 1. Install uv

- **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows:**

  ```bash
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Install the samplepath CLI globally

```bash
uv tool install samplepath
```

This will install Python automatically if needed and make `samplepath` available
globally.

### 3. Verify installation

```bash
samplepath --help
```

If this prints the help message, you're ready to go.

**Note:** On some machines the very first time you run this command it might take 8 to
10 seconds to complete due to the plotting library downloading fonts. Subsequent calls
should be fine.

### Alternative: Run without installation

You can also run samplepath directly without installing it globally using `uvx`:

```bash
uvx samplepath events.csv --help
```

### Alternative: Use pip and pipx

If you already have a Python 3.11+ environment and don't want to switch package
managers, the standard installs via pip and pipx will also work.

Using pip

```bash
pip install samplepath
samplepath --help
```

Using pipx (for end users/global CLI usage)

```bash
pipx install samplepath
samplepath --help
```

To upgrade later

```bash
pipx upgrade samplepath
```

______________________________________________________________________

# 5. Usage

The complete CLI documentation is [here](https://samplepath.pcalc.org/articles/cli).
Here are a few examples.

```bash
# Analyze completed items, save analysis to the output-dir under the scenario name shipped. Clean existing output directories
samplepath events.csv --output-dir spath-analysis --scenario shipped --completed --clean

# Pass an explicit date format (example below shows the typical case for non-US date formats).
# We use standard Python date formats: https://docs.python.org/3/library/datetime.html#format-codes

samplepath events.csv --date-format "%d/%m/%Y" --output-dir spath-analysis --scenario shipped --completed --clean

# Limit analysis to elements with class story
samplepath events.csv --class story

# Apply Tukey filter to remove items with outlier sojourn times before analysis of completed items
samplepath events.csv  --outlier-iqr 1.5 --completed
```

## 📂 Input Format

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

See the [CLI Documentation](https://samplepath.pcalc.org/articles/cli) for the full list
of command line options.

## 📂 Output Layout

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

A complete reference to the charts produced can be found
[here](https://samplepath.pcalc.org/articles/chart-reference).

______________________________________________________________________

# 6. Development Setup (for Contributors)

Developers working on **samplepath** use [uv](https://docs.astral.sh/uv/) for dependency
and build management.

### Prerequisites

Install uv following the [Quick Start](#quick-start-with-uv-recommended) section above.

### 1. Clone and enter the repository

```bash
git clone https://github.com/krishnaku/samplepath-flow.git
cd samplepath-flow
```

### 2. Sync development dependencies

```bash
uv sync --all-extras
```

This creates a virtual environment and installs all dependencies (including dev
dependencies) based on `uv.lock`.

### 3. Run tests

```bash
uv run pytest
```

### 4. Code quality checks

```bash
uv run black samplepath/      # Format Python code
uv run isort samplepath/      # Sort imports
uv run mypy samplepath/       # Type checking
uv run mdformat .             # Format markdown files
```

### 5. Run the CLI from source

During development, run samplepath directly from the source code:

```bash
uv run samplepath examples/polaris/csv/work_tracking.csv --help
```

### 6. Build and publish (maintainers)

To build the distributable wheel and sdist:

```bash
uv build
```

To upload to PyPI (maintainers only):

```bash
uv publish
```

## 📦 Package Layout

```bash
samplepath/
├── cli.py               # Command-line interface
├── csv_loader.py        # CSV import utilities
├── metrics.py           # Empirical flow metric calculations
├── limits.py            # Convergence and limit estimators
├── plots.py             # Chart and panel generation
└── tests/               # Pytest suite
```

______________________________________________________________________

# 7. Documentation

Please see our [documentation site](https://samplepath.pcalc.org)

______________________________________________________________________

# 8. License

Licensed under the **MIT License**.\
See `LICENSE` for details.

Copyright (c) 2025 Krishna Kumar
