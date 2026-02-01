---
ID: 29
Task: data export
Branch: data-export
---

Spec: We want to export the data from each analysis run if the user asks for it.

1. A new cli argument --export-data enables exports with charts.
2. Another cli argument --export-only only exports the data and does not generate any charts.

Both cli arguments go into a new export-configuration argument group

Exports:

1. Go under a new subdirectory exports under the <scenario> directory where all charts are.
2. Ensure the directory exists or create it in file_utils if exports are enabled.

Data formats: Two csv files are produced.
1. If invoked with no sampling frequency, a file named event_indexed_metrics.csv is produced. Data is in ascending timestamp order, with the columns
timestamp, event_type (A/D), N(T), H(T), L(T), \Lambda(T), \Theta(T), w(T), w'(T), W*(T),

2. If invoked with a sampling frequency the file is called <sampling_frequency>_indexed_metrics csv, where the
sampling frequency is what is provided in the --sampling frequency cli arg. In this case the column names are

<sampling_frequency>, N(T), H(T), L(T), \Lambda(T), \Theta(T), w(T), w'(T), W*(T).


2. An element oriented output in ascending order of arrival time stamp showing
element_id, start_ts, end_ts, <all input attributes in the csv file>, sojourn_time (this is the same regardless of sampling frequency)

## Implementation Plan

### New CLI Arguments

Both go in a new **Export Configuration** argument group. They are mutually exclusive (enforced in `validate_args()`).

| Flag | Action | Effect |
|------|--------|--------|
| `--export-data` | `store_true` | Export CSVs **alongside** charts |
| `--export-only` | `store_true` | Export CSVs, **skip** chart generation |

### Output Files

Both files go under `<out_dir>/exports/`.

#### File 1: Flow metrics CSV

- **Event mode** (no `--sampling-frequency`): `event_indexed_metrics.csv`
  - Columns: `timestamp`, `event_type`, `N(T)`, `H(T)`, `L(T)`, `Lambda(T)`, `Theta(T)`, `w(T)`, `w'(T)`, `W*(T)`
  - `event_type` derived from diff of cumulative `Arrivals`/`Departures` arrays: `A`, `D`, or `A/D`
- **Calendar mode**: `<sampling_frequency>_indexed_metrics.csv`
  - Columns: `<sampling_frequency>`, `N(T)`, `H(T)`, `L(T)`, `Lambda(T)`, `Theta(T)`, `w(T)`, `w'(T)`, `W*(T)`
  - First column name and filename use the raw `args.sampling_frequency` string

Data sources: `FlowMetricsResult` fields (`times`, `N`, `H`, `L`, `Lambda`, `Theta`, `w`, `w_prime`) + `ElementWiseEmpiricalMetrics.W_star`.

#### File 2: Element CSV — `elements.csv`

- Sorted by `start_ts` ascending
- Columns: `element_id`, `start_ts`, `end_ts`, `<all other original CSV columns>`, `sojourn_time`
- `element_id` = renamed `id` column
- `sojourn_time` = `duration_td.dt.total_seconds()` (NaN for incomplete items)
- Drop computed columns (`duration_td`, `duration_hr`) — not original CSV attributes

### Files to Change

#### 1. NEW: `samplepath/data_export.py`

```python
def _derive_event_types(metrics: FlowMetricsResult) -> List[str]
    # Diff Arrivals/Departures arrays -> "A", "D", "A/D", or "-"

def export_flow_metrics_csv(
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    export_dir: str,
    sampling_frequency: str | None = None,
) -> str
    # Build DataFrame from metrics fields, add W*(T), rename columns,
    # add event_type for event mode. Write CSV. Return path.

def export_element_csv(df: pd.DataFrame, export_dir: str) -> str
    # Rename id->element_id, add sojourn_time, drop duration_td/duration_hr,
    # sort by start_ts. Write CSV. Return path.

def export_data(
    df: pd.DataFrame,
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    export_dir: str,
    sampling_frequency: str | None = None,
) -> List[str]
    # Call both export functions, return list of paths.
```

#### 2. `samplepath/cli.py`

- **Line ~256**: Add "Export Configuration" argument group with `--export-data` and `--export-only` (both `store_true`, default `False`)
- **`validate_args()`** (line 40): Add mutual exclusivity check for `--export-data` / `--export-only`
- **`main()`** (line 291): Change output message -- use `"Wrote exports"` when `--export-only`, keep `"Wrote charts"` otherwise

#### 3. `samplepath/sample_path_analysis.py`

Modify `run_analysis()` (line 60):
- After `write_limits()` (line 86), add export block:
  - If `args.export_data` or `args.export_only`: call `ensure_export_dir()`, then `export_data()`
- Conditionally skip `produce_all_charts()` when `args.export_only`
- Return combined list (export paths + chart paths)
- Use `getattr(args, ..., False)` for backward compatibility

#### 4. `samplepath/utils/file_utils.py`

Add `ensure_export_dir(out_dir: str) -> str` -- creates `<out_dir>/exports/` and returns the path.

#### 5. NEW: `test/test_data_export.py`

Tests for all export functions:
- `_derive_event_types`: arrival-only -> "A", departure-only -> "D", mixed -> "A/D", baseline -> "-"
- `export_flow_metrics_csv` event mode: file name, column names, column count (10), row count
- `export_flow_metrics_csv` calendar mode: file name with freq, first column name = freq, no event_type (9 cols)
- `export_element_csv`: file created, sorted by start_ts, has element_id/sojourn_time, excludes duration_td/duration_hr
- `export_data`: returns 2 paths, both files exist

#### 6. `test/test_cli.py`

Add tests: defaults are `False`, flags enable correctly, mutual exclusivity raises `SystemExit`.

#### 7. `test/test_sample_path_analysis.py`

Add tests: export called when flags set, not called by default, charts skipped with `--export-only`.

#### 8. Documentation

- `docs/articles/cli/index.md`: Add Export Configuration section with `--export-data` and `--export-only`
- `README.md`: Mention export capability
- `docs/site/articles/cli/index.html`: Regenerated by pre-commit

### Execution Order

1. ✅ Create branch `data-export` from `main`
2. ✅ Write failing tests first (`test/test_data_export.py`, additions to `test/test_cli.py` and `test/test_sample_path_analysis.py`) -- present for review
3. ✅ Implement `samplepath/data_export.py`
4. ✅ Add CLI args to `samplepath/cli.py`
5. ✅ Add `ensure_export_dir` to `samplepath/utils/file_utils.py`
6. ✅ Integrate into `samplepath/sample_path_analysis.py`
7. ✅ Update documentation
8. ✅ Run `uv run pytest` and `pre-commit run`

### Verification

- ✅ `uv run pytest` -- all tests pass (510 tests)
- ✅ `pre-commit run` -- all checks pass
- ✅ `uv run samplepath examples/polaris/csv/work_tracking.csv --export-only --output-dir /tmp/test-export` -- verified CSVs under `exports/`
- ✅ `uv run samplepath examples/polaris/csv/work_tracking.csv --export-data --output-dir /tmp/test-export2` -- verified both charts and CSVs
- ✅ `uv run samplepath examples/polaris/csv/work_tracking.csv --export-data --export-only` -- verified error
- ✅ `uv run samplepath examples/polaris/csv/work_tracking.csv --sampling-frequency week --export-only --output-dir /tmp/test-export3` -- verified `week_indexed_metrics.csv` filename

---

## Implementation Complete

All steps completed successfully. The data export feature is fully implemented and tested.

# Implementation Review Feedback

Findings (ordered by severity)

  - ✅ FIXED: elements.csv column order does not follow the spec's required ordering (element_id, start_ts, end_ts, then remaining input attributes, then sojourn_time). The implementation keeps the existing DataFrame order and
    appends sojourn_time, so if the input CSV order differs, the export won't match the spec. samplepath/data_export.py:156
    - **Resolution**: Added explicit column reordering in `export_element_csv()` to ensure: `element_id, start_ts, end_ts, <other columns>, sojourn_time`

  - ✅ FIXED: Event-indexed export always sets the first event_type to "-" even when the first observation time equals an actual event timestamp. The spec doesn't mention a baseline row, so this can yield a non-event row in an
    "event-indexed" file. samplepath/data_export.py:38
    - **Resolution**: Modified `_derive_event_types()` to check first observation for arrivals/departures and set event type accordingly (A/D/A+D/-)

Findings (ordered by severity)

  - ✅ FIXED: The new "first observation event" logic in _derive_event_types() is untested, so the fix isn't covered. Add a case where Arrivals[0] > 0 or Departures[0] > 0 to assert the first row becomes A/D/A/D.
    - **Resolution**: Added three new test cases:
      - `test_derive_event_types_first_observation_arrival()` - Tests Arrivals[0] > 0 → "A"
      - `test_derive_event_types_first_observation_departure()` - Tests Departures[0] > 0 → "D"
      - `test_derive_event_types_first_observation_both()` - Tests both > 0 → "A/D"

  - ✅ FIXED: The column ordering fix for elements.csv is not validated by tests. The spec requires element_id, start_ts, end_ts, <other columns>, sojourn_time — add an assertion for exact column order so regressions are caught.
    - **Resolution**: Added explicit column order assertion in `test_export_element_csv()`:
      ```python
      expected_columns = ["element_id", "start_ts", "end_ts", "class", "sojourn_time"]
      assert df.columns.tolist() == expected_columns
      ```

**Test Summary:**
- Total tests: 513 (added 3 new tests)
- data_export.py coverage: 95.70% (improved from 91.40%)
- All tests pass ✅
- Pre-commit checks pass ✅
