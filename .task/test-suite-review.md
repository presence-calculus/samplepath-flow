# Improving the `flow` CLI Test Suite

The current test suite provides basic smoke coverage, but it is light relative to the importance of the CLI surface and recent refactor. Below are concrete recommendations to make the test suite more robust, scalable, and regression-resistant.

---

## 1. Separate Parsing Tests from Execution Tests

Introduce two explicit layers:

### A. Parser / Dispatch Unit Tests
Fast tests with minimal mocking that validate:

- `normalize_argv()` behavior
- Subcommand selection (`args.cmd`)
- Positional and option binding (`csv`, `output_dir`, flags)

These tests should *not* call `main()`.

### B. Integration Tests for `main()`
Use mocks for side-effecting functions:

- `ensure_output_dirs`
- `copy_input_csv_to_output`
- `write_cli_args_to_file`
- `run_analysis`

Assert correct call order, arguments, stdout/stderr, and exit codes.

---

## 2. Table-Driven Dispatch Test Matrix

Use `pytest.mark.parametrize` to cover all important invocation shapes:

### Default Dispatch
- `["flow", "events.csv"]` → `cmd="analyze"`
- `["flow", "events.csv", "--help"]` → analyze help (exit 0)

### Explicit Subcommands
- `["flow", "analyze", "events.csv"]`
- `["flow", "configure"]`

### Help Semantics
- `["flow", "--help"]` → top-level help (lists subcommands)
- `["flow", "analyze", "--help"]`
- `["flow", "configure", "--help"]`

### Option-First (No Default Dispatch)
- `["flow", "-h"]`
- `["flow", "--help"]`

### Invalid Input
- `["flow", "bogus"]` → argparse error, nonzero exit

---

## 3. Validation and Error-Path Tests

Lock down error semantics explicitly:

- `--completed` + `--incomplete` must exit nonzero
- Assert exit code (`1` vs `2`) consistently
- If certain flags are intended to be paired (e.g. `--outlier-iqr-two-sided`), decide and test the behavior

---

## 4. Defaults and Type Conversion Coverage

Add assertions for:

- `output_dir` is a resolved `Path`
- Default values:
  - `scenario == "latest"`
  - `epsilon == 0.05`
  - `horizon_days == 28.0`
  - `lambda_warmup == 0.0`

Note: `--save-input` currently uses `store_true` with `default=True`, making it impossible to disable. If this is intentional, test it explicitly.

---

## 5. Integration Wiring Tests (With Mocks)

Verify that `main()`:

- Calls `ensure_output_dirs` with correct parameters
- Conditionally calls `copy_input_csv_to_output`
- Always calls `write_cli_args_to_file`
- Calls `run_analysis(args.csv, args, out_dir)`
- Prints chart paths on success
- Exits with code `1` and writes to stderr on exceptions

---

## 6. Help Text Assertions (Golden Fragments)

Avoid full snapshot tests. Instead assert key substrings:

- Top-level help contains `analyze` and `configure`
- Analyze help includes section headers like:
  - `CSV Parsing`
  - `Output Configuration`

This guards against accidental CLI regressions without brittleness.

---

## 7. Path and Platform Edge Cases

Add low-cost tests for:

- Input paths with spaces (`"data/my events.csv"`)
- `--output-dir ~` expands correctly (use `monkeypatch` if needed)

---

## 8. CLI Import / Entry-Point Test

Add a simple test that imports the CLI module and calls `main()` with mocked `sys.argv` to ensure packaging and entrypoints remain valid.

---

## 9. Property-Based Tests for Dispatch Logic (Optional)

For high confidence in `normalize_argv()`:

- Any argv where `argv[1]` starts with `-` must not be rewritten
- Any argv where `argv[1]` is a known subcommand must not be rewritten
- All other argv must inject `analyze`

This can be implemented with Hypothesis or a small randomized test.

---

## Summary

The goal is to evolve the test suite from smoke tests to **behavioral guarantees** around dispatch, validation, defaults, and wiring. This level of coverage is appropriate for a CLI that is intended to be stable, extensible, and user-facing.
