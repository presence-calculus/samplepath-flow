# Refactor `flow` CLI to Use Subcommands With Default Dispatch

## Objective

Refactor the `flow` CLI (implemented with `argparse`) so that:

- `flow <csv> ...` defaults to `flow analyze <csv> ...`
- Explicit subcommands are supported: `flow analyze ...`, `flow configure ...`
- Future usage supports `samplepath flow analyze ...`
- Existing flags and behavior remain unchanged

This is implemented via **default subcommand dispatch** (not optional positionals).

---

## Step 1: Add Subcommand Registry and argv Normalizer

Add near the top of `cli.py` (after imports):

```python
SUBCOMMANDS = {"analyze", "configure"}

def normalize_argv(argv: list[str]) -> list[str]:
    # argv[0] is program name
    if len(argv) <= 1:
        return argv
    first = argv[1]
    if first in SUBCOMMANDS:
        return argv
    # Default: treat as `analyze ...`
    return [argv[0], "analyze", *argv[1:]]
```

---

## Step 2: Update `parse_args()` Signature and Normalize argv

Change the function signature:

```python
def parse_args(argv: list[str] | None = None):
```

At the top of `parse_args`, add:

```python
if argv is None:
    argv = sys.argv
argv = normalize_argv(argv)
```

---

## Step 3: Add Subparsers

Replace the root parser construction with:

```python
parser = argparse.ArgumentParser(
    description="Finite-window Little’s Law charts from intervals CSV",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

subparsers = parser.add_subparsers(dest="cmd", required=True)
```

Create the `analyze` subcommand:

```python
analyze = subparsers.add_parser(
    "analyze",
    help="Generate finite-window flow-metrics charts from an intervals CSV",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
```

Create a placeholder `configure` subcommand:

```python
configure = subparsers.add_parser(
    "configure",
    help="Manage samplepath configuration (placeholder)",
)
configure.add_argument("config", nargs="?", help="Path to config file")
```

---

## Step 4: Move All Existing Arguments Under `analyze`

Every existing call of the form:

```python
parser.add_argument(...)
parser.add_argument_group(...)
```

must be changed to:

```python
analyze.add_argument(...)
analyze.add_argument_group(...)
```

This applies to all argument groups:

- CSV Parsing
- Data Filters
- Outlier Trimming
- Lambda Fine Tuning
- Convergence Thresholds
- Output Configuration

The positional argument:

```python
csv_group.add_argument("csv", ...)
```

remains positional but is now scoped to `analyze`.

No flag semantics or defaults should be changed.

---

## Step 5: Parse Using Normalized argv

Replace:

```python
args = parser.parse_args()
```

with:

```python
args = parser.parse_args(argv[1:])
```

Keep validation and return behavior unchanged:

```python
validate_args(args)
return parser, args
```

---

## Step 6: Route Behavior in `main()`

In `main()`, after parsing:

```python
parser, args = parse_args()
```

Add subcommand routing:

```python
if args.cmd == "configure":
    print("configure: not implemented yet", file=sys.stderr)
    sys.exit(2)
```

Leave the existing analysis path unchanged. It should execute for both:

- `flow analyze ...`
- `flow <csv> ...` (default-dispatched)

---

## Expected Behavior After Refactor

These invocations must work:

```bash
flow events.csv --help
flow analyze events.csv --help
flow events.csv --output-dir charts
flow configure
```

The first and third forms must behave identically to `flow analyze ...`.

---

## Constraints

- Do not change existing flags, defaults, or validation logic
- Default dispatch must occur *before* argparse parsing
- This refactor prepares the CLI for future `samplepath flow analyze ...` usage
- No packaging or entrypoint changes are required
