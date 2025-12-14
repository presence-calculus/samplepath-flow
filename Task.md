---
ID: 2
Task: cli subcommand refactoring
Branch: cli-subcommand-refactoring
---

1. Read the detailed specification in .context/features/flow-git-subcommand-refactor.md
2. Summarize the plan and add it as steps in this document.
3. Add CLI tests covering default dispatch to analyze and the configure placeholder.
4. Refactor samplepath/cli.py to introduce argv normalization, subcommands, and routing without changing existing flags.
5. Run uv run pytest (and pre-commit if time allows) to verify changes.
