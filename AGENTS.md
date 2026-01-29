# AGENTS.md

## Repository Guidelines

### Project Structure

Core logic lives under `samplepath/` (`cli.py`, `metrics.py`, `limits.py`,
`csv_loader.py`, `plots/`). Treat this package as the source of truth for all features.
Documentation lives in `docs/`; examples in `examples/`. Generated artifacts (`charts/`,
`dist/`, `htmlcov/`, `coverage.xml`) should not be modified. Tests mirror the source
tree under `test/`.

Agent must not modify any file outside the scope of the task being executed.
----------
## Coding and architectural guidance
See decisions/codiing_style_guidelines.md for additional rules.

______________________________________________________________________

## Agent Task Workflow

Tasks are defined in `tasks/Task.md`. Each task has:

- an **ID**
- a **Name**
- a **Branch**
- a **Specification block**

Agent must:

1. Locate the task block matching the given **Task ID**.
2. Use only that block as the authoritative task description.
3. Modify **only** the files referenced or required by that task.
4. If the task-specified branch does not exist, create it from the latest `main`.
5. Make only the minimal changes needed to complete the task and nothing else.

### Execution Protocol

- Before making any code changes, Agent must present a step-by-step plan and wait for
  explicit approval.
- After approval, apply patches only to the task branch.
- Agent must not open pull requests unless explicitly instructed.

### Commit Requirements

- “Every commit must target the current task. Allowed files: only those required by the task specification.”
  -Before staging files or proposing a commit:
  - "Always run the full test suite and ensure all tests pass"
  - “Always run pre-commit and make sure all checks pass  ”
  - If Task.md has changed always include it in the commit.
  - When changing a file under docs/* if the corresponding generated doc under docs/site changes include both in the commit.
  - When proposing a commit provide a summary of the commit message and the files that will be committed.
  - Ask for review *only* after all these checks have cleared.

    - When merging to `main`, prefer a squash merge. Merge commit message format:
    ```
    [Task ID]: (Task Name): Merge <branch name> to main
    ```
    Craft a concise yet complete summary of the changes that were made in the branch being merged and get the message reviewed before performing the merge.


------------
## Code Review
When instructed to 'see code feedback':
- check the latest content of task.md which will have feedback on the latest task
- summarize the changes you plan to make including test changes and get them approved.
- make the changes and wait for approval
- commit the changes on approval

When instructed to 'see testing feedback'
- check the latest content of task.md
- summarize the changes you plan to make and get them approved.
- make the changes and report test stats
- commit the changes on approval


______________________________________________________________________

## Documentation Workflow

This applies only to files in the doc/* sub-directory
Pandoc converts Markdown to HTML using the tooling in `docs/build/`. Before committing
Markdown changes:

1. Run `pre-commit`.
2. If `mdformat` reports changes, present them for review before committing.
3. Do not rewrite any YAML front matter in markdown files unless explicitly instructed.
4. If a generated file in docs/site/* has changed because the corresponding .md file has changed, stage both for commit.

______________________________________________________________________

## Build, Test, and Development Commands

Provision environment:

```
uv sync --all-extras
```

Run CLI checks:

```
uv run samplepath examples/polaris/csv/work_tracking.csv --help
```

Run tests:

```
uv run pytest
```

### Formatting and Linting (canonical order)

Agent must always run pre-commit before committing any code.

```
pre-commit run
```

______________________________________________________________________

## Coding Style & Naming Conventions

Follow PEP 8 with explicit type hints. Snake_case for modules; CamelCase for public
classes. CLI flags use kebab-case. Avoid side effects at module scope. Keep helpers near
their call sites.

______________________________________________________________________

## Testing Guidelines

Add tests under `test/` mirroring the source tree. Before implementing code changes
propose one or more failing tests that will verify the task specification and wait for
review. Use one assertion per test. Use parametrized tests for scenario coverage.
Fixtures should be deterministic. Agent may modify existing tests only when required by
the task’s acceptance criteria. Always present modifications of existing tests for
review before making any changes.

Policy for automated test/formatter execution (no per-command prompts):

  - You have blanket approval to run tests and formatters without asking each time, as long as they write only inside the repo workspace.
  - Use local caches to avoid sandbox/network prompts: UV_CACHE_DIR=.uv-cache uv run pytest and PRE_COMMIT_HOME=.pre-commit-cache pre-commit run --all-files.
  - Do not request escalation unless a executing tests must write outside the workspace or needs network; pause and ask only in those cases.
  - IF git actions fail, run with escalation and try again.


# End of file
