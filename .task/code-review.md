# Code Review: `flow` CLI Subcommand Implementation

This review covers the recent refactor introducing subcommands (`analyze`, `configure`) with default dispatch behavior.

Overall assessment: **the structure is sound and the approach is correct**, but there are two issues that should be fixed immediately, plus a few minor hardening suggestions.

---

## Must‑Fix Issues

### 1. `validate_args()` Never Exits on Error

The function prints an error when mutually exclusive flags are used, but it never sets the error flag, so execution continues.

**Current behavior:**  
`--completed` and `--incomplete` can be passed together without terminating the program.

**Fix:**

```python
def validate_args(args):
    error = False

    if getattr(args, "completed", False) and getattr(args, "incomplete", False):
        print(
            "Error: --completed and --incomplete cannot be used together",
            file=sys.stderr,
        )
        error = True

    if error:
        sys.exit(1)
```

(Alternatively, use `parser.error(...)` to let argparse handle formatting and exit codes, but the above is the minimal correction.)

---

### 2. `flow --help` Is Hijacked by Default Dispatch

Because `normalize_argv()` injects `"analyze"` whenever the first token is not a known subcommand, this occurs:

```bash
flow --help  →  flow analyze --help
```

This prevents users from discovering other subcommands (`configure`) via top‑level help.

**Fix:**  
Do not inject the default subcommand when the first token looks like an option.

```python
def normalize_argv(argv: list[str]) -> list[str]:
    if len(argv) <= 1:
        return argv

    first = argv[1]

    # Allow top-level flags like --help
    if first.startswith("-"):
        return argv

    if first in SUBCOMMANDS:
        return argv

    return [argv[0], "analyze", *argv[1:]]
```

**Resulting behavior:**

- `flow --help` → top‑level help with subcommands
- `flow events.csv --help` → analyze help (default dispatch)

---

## Test Coverage Improvements (Recommended)

Add explicit tests for the following cases:

1. **Top‑level help**
   - Invocation: `flow --help`
   - Expected: argparse exits with code `0`, showing top‑level help (no default dispatch)

2. **Mutually exclusive flags**
   - Invocation: `flow events.csv --completed --incomplete`
   - Expected: exit with non‑zero status (`1` or `2`, depending on implementation)

These tests lock in the intended semantics and prevent regressions.

---

## Improvements

- Derive `SUBCOMMANDS` dynamically from the subparsers to avoid drift if new subcommands are added.
- Current behavior when invoking `flow` with no arguments results in an error due to `required=True` subparsers. Special‑case this to show help by default.

---

## Summary

- Architecture: **correct and scalable**
- Default dispatch approach: **appropriate**
- Two must‑fix issues: **error handling and help dispatch**
- Once fixed, the CLI will be robust and future‑proof for `samplepath flow analyze ...`

This implementation is close to final‑quality with only small corrective edits needed.
