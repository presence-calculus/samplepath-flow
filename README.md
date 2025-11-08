# The Sample Path Analysis Library and Toolkit

A reference implementation of sample-path–based flow metrics, convergence analysis, and stability diagnostics for flow processes in 
complex adaptive systems.

[![PyPI](https://img.shields.io/pypi/v/samplepath.svg)](https://pypi.org/project/samplepath/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://py.pcalc.org)

---

# 🔍 Overview

**samplepath** is a Python library for the analysis of stability of flow processes
using the finite-window formulation of **Little’s Law**.  

It provides deterministic, pathwise measurement tools for analyzing long run
flow process dynamics: arrival/departure equilibrium, process time coherence, and
process stability along an observed sample path.

The package implements components of [The Presence Calculus Project](https://docs.pcalc.org): a computational toolkit for modeling
and measuring flow processes.


## Background

For an overview of the key concepts behind this library and how they can be applied in practice, please see 
our posts continuing series on Little's Law and sample path analysis at 

[The Polaris Flow Dispatch](https://www.polaris-flow-dispatch.com)
- [The Many Faces of Little's Law](https://www.polaris-flow-dispatch.com/p/the-many-faces-of-littles-law).
- [Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/p/littles-law-in-a-complex-adaptive)

The analyses in these posts  were produced using this toolkit 
and can be found in the [examples](./examples/polaris) directory together with their original source data.

Please subscribe to The Polaris Flow Dispatch, if you are interested in this topic. 

## Core capabilities

A [flow process](https://www.polaris-flow-dispatch.com/i/172332418/flow-processes) is simply a timeline of events from some underlying domain, where
events have *effects* that persist beyond the time of the event. These effects are encoded using
metadata (called marks) to describe those effects. 

The current version only supports the analysis of binary flow processes. These are
flow processes where the marks denote the start or end of an observed presence of a domain element within some system boundary. 
They are governed by the L=λW form of Little's Law.

These are simplest kind of flow processes we analyze in the presence calculus, but they cover the vast 
majority of operational use cases we currently model in software delivery, so we will start there.

## Data Requirements

The data requirements for this analysis are  minimal: a csv file that represents 
the observed timeline of a binary flow process: with element id, start and end date columns. 

- The start and end dates may be empty, but for a meaningful analysis, we
  require at least some of these dates be non-empty. Empty end dates denote
  elements that have started but not ended. Empty start dates denote items whose
  start date is unknown. Both are considered elements currently present in the
  boundary.
- The system boundary is optional (the name of csv file becomes the default name of the boundary)

Given this input, the toolkit provides

A. Core python modules that implement the computations for sample path construction and analysis:

- Time-averaged flow metrics governed by the finite version of Little's Law
   `N(t), L(T)`,`Λ(T)`, `w(T)`, `λ*(T)`, `W*(T)` 
- Performing *equilibrium* and **coherence** calculations (e.g., verifying `L(T) ≈ λ*(T)·W*(T)`)
- Estimating empirical **limits** with uncertainty and **tail** checks to verify stability (alpha)

Please see [Sample Path Construction](https://www.polaris-flow-dispatch.com/i/172332418/sample-path-construction-for-lλw)
for background. 

B. Command line tools provide utilities that that wrap these calculations

- Simple workflows that take csv files as input to run sample path analysis with a rich set of parameters and options.
- Generate publication-ready **charts and panel visualizations** as static png files. 
- The ability to save different parametrized analyses from a single csv file as named scenarios.

This toolkit provides the computational foundation for analyzing flow dynamics in 
software delivery, operations, and other knowledge-work systems.

## 🧠 Key Metrics

Deterministic, sample-path analogues of Little’s Law:

| Quantity | Meaning |
|-----------|----------|
| `L(T)` | Average work-in-process over window `T` |
| `Λ(T)` | Cumulative arrivals per unit time up to `T` |
| `w(T)` | Average residence time over window `T` |
| `λ*(T)` | Empirical arrival rate up to `T` |
| `W*(T)` | Empirical mean sojourn time of items completed by `T` |

These quantities enable rigorous study of **equilibrium** (arrival/departure rate convergence), **coherence** (residence time/sojourn time convergence), and **stability** (convergence of process measures to limits) even when processes operate far from steady state.

---

# 🚀 Installation (End Users)

Pre-requisites Python 3 (3.11 or higher at present).

## Quick Start

**1. Install Python (≥ 3.11)**  
- **macOS:**  
  ```bash
  brew install python@3.11
  ```
- **Windows / Linux:**  
  Download from [python.org/downloads](https://www.python.org/downloads) and follow the installer.



**2. Install pipx** *(recommended for CLI tools)*  
```bash
pip install --user pipx
pipx ensurepath
```
Then restart your terminal so the new PATH takes effect.



**3. Install the CLI**
```bash
pipx install samplepath
```



**4. Verify installation**
```bash
samplepath --help
```

If this prints the help message, you’re ready to go.


## 🧩 Example Usage

```bash
# Analyze completed items, save analysis to the output-dir under the scenario name shipped. Clean existing output directories
samplepath events.csv --output-dir spath-analysis --scenario shipped --completed --clean

# Limit analysis to elements with class story
samplepath events.csv --class story

# Apply Tukey filter to remove items with outlier soujourn times before analysis of completed items
samplepath events.csv  --outlier-iqr 1.5 --completed
```

### 📂 Output Layout

Results and charts are saved to the output directory as following

For input `events.csv`, output is organized as:

```
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


# 🛠 Development Setup (for Contributors)
Developers working on **samplepath** use [Poetry](https://python-poetry.org/) for dependency and build management.

### 1. Clone and enter the repository
```bash
git clone https://github.com/krishnaku/samplepath.git
cd samplepath
```

### 2. Install development dependencies
```bash
poetry install
```

### 3. Activate the virtual environment
```bash
poetry shell
```

### 4. Run tests
```bash
pytest
```

### 5. Code quality checks
```bash
black samplepath/
isort samplepath/
mypy samplepath/
```

### 6. Build and publish (maintainers)
To build the distributable wheel and sdist:

```bash
poetry build
```

To upload to PyPI (maintainers only):

```bash
poetry publish --build
```

## 📦 Package Layout

```
samplepath/
├── cli.py               # Command-line interface
├── csv_loader.py        # CSV import utilities
├── metrics.py           # Empirical flow metric calculations
├── limits.py            # Convergence and limit estimators
├── plots.py             # Chart and panel generation
└── tests/               # Pytest suite
```

---

## 📚 Documentation

Further documentation, will be added to this repo. In the meantime, use the
documentation links provided at the top of this README.

---

## 📝 License

Licensed under the **MIT License**.  
See `LICENSE` for details.
