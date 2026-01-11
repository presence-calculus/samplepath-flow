# Samplepath Codebase

Sample path analysis library for flow processes using the finite-window formulation of Little's Law: `L(T) = Λ(T) · w(T)`.

> See `AGENTS.md` for task workflow, commit format, and review protocols.

## Key Files

- `samplepath/metrics.py` - Core algorithms: `FlowMetricsResult`, `compute_sample_path_metrics()`, `compute_finite_window_flow_metrics()`
- `samplepath/cli.py` - CLI interface and argument parsing
- `samplepath/csv_loader.py` - CSV loading with `CSVLoader` class, timezone normalization, validation
- `samplepath/filter.py` - Data filtering and outlier removal
- `samplepath/plots/` - Visualization (matplotlib charts)
- `samplepath/limits.py` - Convergence analysis (work in progress)

## Commands

```bash
uv run pytest                    # Run tests with coverage
uv run flow analyze <csv>        # Run analysis on a CSV file
uv run flow analyze --help       # Show all CLI options
uv run black samplepath/         # Format code
uv run mypy samplepath/          # Type check
```

## Input Format

CSV with columns: `id`, `start_ts`, `end_ts`, optional `class`

## Patterns

- Use `@dataclass` for result types (see `FlowMetricsResult`, `CSVLoader`)
- Tests verify mathematical identities (e.g., `L = A / elapsed`), not just execution
- Time units are hours internally
- Timestamps are normalized to UTC then made timezone-naive

## Architecture

```
CSV file → CSVLoader → DataFrame → to_arrival_departure_process() → events
    → compute_finite_window_flow_metrics() → FlowMetricsResult → plots/
```
