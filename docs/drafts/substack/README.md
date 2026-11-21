# Substack publishing workflow

Local authoring tools for The Polaris Flow Dispatch. The `dispatch` command wraps
`python-substack` 0.10.0 to create, edit, and publish Markdown drafts. It also renders
CSV tables as PNG images for use in articles.

## Setup

Install Python 3.11 or later and `uv`, then run from the repository root:

```bash
cd docs/drafts/substack
uv sync
```

Run the commands below from this directory. Table rendering also requires Chrome
or Chromium to be installed.

Create a local `.env` file beside `substack_cli.py`:

```dotenv
PUBLICATION_URL=https://your-publication.substack.com
COOKIES_STRING="your authenticated browser cookie string"
```

Use cookies from an authenticated session on the publication's domain. Keep these
credentials private; `.env` and `cookies*.json` are ignored by Git. When a session
expires, refresh the cookies. Alternatively, configure `COOKIES_PATH` with a cookies
JSON file, or supply `--cookies cookies.json` before the subcommand.

Check authentication and the selected publication:

```bash
uv run dispatch status
uv run dispatch publications list
```

The wrapper loads this directory's `.env` explicitly. Existing environment
variables take precedence. `PUBLICATION_URL` must be an HTTPS publication origin
without an additional path.

## Prepare an article

Write an article in Markdown with optional leading YAML front matter:

```markdown
---
title: "Understanding flow"
subtitle: "A sample-path perspective"
---

Article text with inline math such as $L = \lambda W$.

![Flow table](flow.png)
```

The wrapper removes the front matter before uploading the body. It uses `title`
and `subtitle` as defaults when creating a draft, and `subtitle` when updating one.
Explicit `--title` and `--subtitle` options take precedence. Other front-matter
fields are not forwarded as CLI options. Without a title default or option, the
library uses the first `# Heading`, falling back to the Markdown filename stem.

Local images are uploaded by the library. Relative image paths are resolved from
the command's working directory, so use paths relative to this directory or run
from the article directory with `uv run --project /absolute/path/to/docs/drafts/substack
dispatch ...`. The wrapper translates inline math between the library's schema and
Substack's editor schema during uploads and exports.

## Create and inspect a draft

```bash
uv run dispatch drafts create article.md
uv run dispatch drafts list
uv run dispatch drafts get 12345
```

Replace `12345` throughout this guide with the ID returned by `create`. Creating a
draft does not publish it or send email. Open the draft in Substack to review its
formatting, images, and math.

Optional creation flags include `--title`, `--subtitle`, `--audience`, `--slug`,
`--tag` (repeatable), and search-engine metadata. See all options with:

```bash
uv run dispatch drafts create --help
```

## Update an existing draft

Export the remote draft to a separate file before editing:

```bash
uv run dispatch drafts export 12345 --output article-export.md
```

The export includes a revision marker used to detect remote changes. Preserve
that marker and any image markers while editing the exported body. Keep your
original article if you want to retain its front matter; export does not recreate
that local metadata.

Preview the update, then apply it:

```bash
uv run dispatch drafts update 12345 article-export.md --dry-run
uv run dispatch drafts update 12345 article-export.md
```

The CLI requests confirmation for both commands; use `--yes` in non-interactive
execution. A dry run previews the update without saving the draft, but processing
new local images can still upload them.

Updates refuse missing or stale revision metadata, unsupported content changes,
and ambiguous image replacements by default. If the remote draft changed,
re-export and reconcile your edits. A successful update also makes the previous
export's revision marker stale; export again before the next editing cycle.

To intentionally replace the body from your original local Markdown without an
exported revision marker:

```bash
uv run dispatch drafts update 12345 article.md --allow-conflict --yes
```

This bypasses the revision check. Separate overrides exist for unsupported content
and image replacement; inspect `drafts update --help` before using them. Updates
do not take a new title from front matter, and the update command has no `--title`
option.

## Publish or schedule

After reviewing the draft in Substack, publish it:

```bash
uv run dispatch drafts publish 12345
```

**Publishing sends email by default.** To publish without emailing subscribers:

```bash
uv run dispatch drafts publish 12345 --no-send
```

Publishing requests confirmation; `--yes` skips that prompt. Add
`--share-automatically` if desired.

To schedule a draft, supply an ISO 8601 timestamp with a timezone offset or `Z`.
Replace the example with your intended future time:

```bash
uv run dispatch drafts schedule 12345 --at "2027-01-15T09:00:00-06:00"
uv run dispatch drafts unschedule 12345
```

## Render a table image

Create a CSV such as `flow.csv`:

```csv
metric,value
Average inventory,12.5
Throughput,3.25
```

Optionally add an adjacent `flow.yaml` to customize the table:

```yaml
title: "Flow metrics"
subtitle: "Example observation window"
note: "Source: sample-path measurements"
columns:
  metric:
    label: "Metric"
    align: left
    width: "65%"
  value:
    label: "Value"
    align: right
    decimals: 2
options:
  table_width: "640px"
```

Render the image:

```bash
uv run dispatch table flow.csv
# Or choose a different layout file:
uv run dispatch table flow.csv --layout custom-layout.yaml
```

This writes `flow.png` beside the CSV, replacing an existing PNG. Without a layout,
the renderer uses shared defaults: 640px table width, 18px Arial/sans-serif text,
right-aligned numeric columns, and left-aligned text columns. Images use 2x zoom
and 12px expansion.

Column settings support `label`, `align`, `width`, and `decimals`. Labels support
inline `$LaTeX$`; use `\$` for a literal dollar sign. The `options` mapping overrides
Great Tables `tab_options` settings. Embed the PNG in your article with Markdown
image syntax.

## Tools and help

- `substack_cli.py`: loads local credentials, uses the publication domain for
  authentication, handles front-matter defaults, and adapts inline math.
- `table_renderer.py`: reads CSV and optional YAML, formats with Great Tables,
  converts header math with `latex2mathml`, and saves through `nokap` and Chrome.
- `pyproject.toml` and `uv.lock`: define the local package, dependencies, and
  `dispatch` entry point.

Use `dispatch` to retain the local wrapper behavior rather than invoking the
upstream `substack` or `substack-publish-markdown` commands directly.

```bash
uv run dispatch --help
uv run dispatch drafts --help
uv run dispatch table --help
uv run dispatch --json drafts list
```

Global options such as `--json`, `--cookies`, and `--publication-url` go before the
subcommand. Draft deletion is available through `dispatch drafts delete ID` and
requests confirmation.
