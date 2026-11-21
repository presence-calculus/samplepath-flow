"""Compatibility entry point; layout now lives in flow.yaml."""

from pathlib import Path

from table_renderer import main

if __name__ == "__main__":
    raise SystemExit(main([str(Path(__file__).with_name("flow.csv"))]))
