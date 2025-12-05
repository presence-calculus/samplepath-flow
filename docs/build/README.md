# Documentation Build Pipeline

Pandoc turns the Markdown articles in `docs/articles` into the published HTML that lives
in `docs/site`. The `docs/build` folder houses a small wrapper that keeps the process
reproducible and aligned with the GitHub Pages deployment.

## What gets built

- `docs/build/pandocs.conf` maps sources to targets:
  `docs/articles/** → docs/site/articles`, `docs/drafts/** → docs/site/drafts`,
  `docs/assets/** → docs/site/assets`, and `examples/** → docs/site/examples`.
- HTML generated under `docs/site` is committed and later published by GitHub Pages; the
  workflow intentionally skips `docs/site/drafts`.
- Asset files matching `png/jpg/jpeg/gif/svg/css/js/pdf` are copied verbatim alongside
  the rendered pages so relative links continue to work.

## Prerequisites

- `pandoc` available on your PATH; `pandoc-crossref` is picked up when an article sets
  `numberSections: true`.
- Default math support uses MathJax (`MATH_ENGINE="--mathjax"` in `pandocs.conf`), but
  you can override via environment variable.

## Running the build

From the repo root:

```sh
./docs/build/pandocs.sh
```

- The script is Git-aware by default and rebuilds only changed or untracked files; set
  `GIT_AWARE=0` to force a full rebuild.
- To point at a different config file, set `PANDOCS_CONFIG=/path/to/custom.conf`.

## How `pandocs.sh` works

- Iterates through each `SOURCE_MAP` entry (recursive when the pattern ends with `/**`),
  creating the matching directory under `docs/site`.
- For every Markdown file, builds a Pandoc CLI with:
  - Lua filters `interpolate-vars.lua` (expands `$var` placeholders from YAML metadata)
    and `md2html-links.lua` (rewrites `.md` links to `.html`).
  - Optional features triggered by front matter: `numberSections: true` adds
    `pandoc-crossref`; `citations: true` or `link-citations: true` enables `--citeproc`
    plus per-directory `references.bib`/`ieee.csl`; `toc: true` includes a table of
    contents and respects `toc-depth`.
  - The shared HTML template at `docs/build/pandoc_template.html`, plus the configured
    math engine.
- Copies static assets that match `ASSET_EXTENSIONS` into the corresponding location
  under `docs/site`, preserving directory structure.

## Authoring tips

- Use YAML metadata to define reusable paths (for example `document-root: ../..`) and
  interpolate them in links or images via `$document-root/...` thanks to
  `interpolate-vars.lua`.
- `header-image`, `title`, `subtitle`, `author`, and `date` are passed into
  `pandoc_template.html` to render a consistent header block.
- Cross-document links should point to the Markdown filename; the Lua filter converts
  them to `.html` in the rendered site.

## Deployment

- `.github/workflows/documentation-site.yml` runs on pushes to `main` that touch
  `docs/site/**`. It rsyncs `docs/site` to a staging folder (omitting `drafts/`),
  uploads the artifact, and deploys it to GitHub Pages.
