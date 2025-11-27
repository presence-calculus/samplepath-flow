#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# -------- Configuration --------
# Map each SOURCE_DIR to a TARGET_DIR (non-recursive, mirrors only one level)
# Example: "pcalc/docs:docs/pandoc" means:
#   pcalc/docs/*.md              → docs/pandoc/*.html
#   pcalc/docs/<sub>/*.md        → docs/pandoc/<sub>/*.html
PAIRS=(
  "docs/src:docs/html"
  ".:.local/html"
)

MATH_ENGINE="--mathjax"     # e.g., "--mathjax" or leave empty ""

# -------- Git helpers --------
GIT_AWARE=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_AWARE=1
fi

git_file_changed() {
  local f="$1"
  [[ "$GIT_AWARE" -eq 1 ]] || return 0
  # Untracked?
  if ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    return 0
  fi
  # Staged?
  if ! git diff --quiet --cached -- "$f"; then
    return 0
  fi
  # Unstaged?
  if ! git diff --quiet -- "$f"; then
    return 0
  fi
  return 1
}

# Return first existing file from a list of candidates
first_existing() {
  local cand
  for cand in "$@"; do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

# -------- Pandoc helper --------
pandoc_for_file() {
  local file="$1"
  local out_html="$2"
  local src_root="$3"

  local dir repo_root=""
  dir="$(dirname "$file")"
  if [[ "$GIT_AWARE" -eq 1 ]]; then
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  fi

  # Find resources with fallback order: file dir -> src_root -> git root
  local bib csl tpl
  bib="$(first_existing \
        "$dir/references.bib" \
        "$src_root/references.bib" \
        "${repo_root:+$repo_root/references.bib}" || true)"
  csl="$(first_existing \
        "$dir/ieee.csl" \
        "$src_root/ieee.csl" \
        "${repo_root:+$repo_root/ieee.csl}" || true)"
  tpl="$(first_existing \
        "$dir/pandoc_template.html" \
        "$src_root/pandoc_template.html" \
        "${repo_root:+$repo_root/pandoc_template.html}" || true)"

  # Build args safely under `set -u`
  local -a args=(
    --filter pandoc-crossref
    --lua-filter docs/src/md2html-links.lua
    --toc
    --citeproc
    --wrap=auto
    --quiet
  )

  [[ -n "${bib:-}" ]] && args+=( --bibliography="$bib" )
  [[ -n "${csl:-}" ]] && args+=( --csl="$csl" )
  [[ -n "${tpl:-}" ]] && args+=( --template="$tpl" )

  args+=( -s "$file" -o "$out_html" )

  if [[ -n "${MATH_ENGINE:-}" ]]; then
    # Allow multi-token engines like "--mathjax --katex"
    # shellcheck disable=SC2206
    local -a math_tokens=( $MATH_ENGINE )
    args+=( "${math_tokens[@]}" )
  fi

  pandoc "${args[@]}"
}

# -------- Processing --------
for pair in "${PAIRS[@]}"; do
  IFS=":" read -r SRC_ROOT TGT_ROOT <<<"$pair"

  if [[ ! -d "$SRC_ROOT" ]]; then
    echo "⚠️  Skipping missing source dir: $SRC_ROOT"
    continue
  fi

  mkdir -p "$TGT_ROOT"

  # Top-level files in SRC_ROOT → TGT_ROOT
  for file in "$SRC_ROOT"/*.md; do
    if git_file_changed "$file"; then
      filename="$(basename "$file" .md)"
      out_html="$TGT_ROOT/${filename}.html"
      pandoc_for_file "$file" "$out_html" "$SRC_ROOT"
      echo "✓ Converted $file → $out_html"
    fi
  done

  # Immediate subdirectories ONLY
  for subdir in "$SRC_ROOT"/*/; do
    [[ -d "$subdir" ]] || continue
    subname="$(basename "$subdir")"
    out_dir="$TGT_ROOT/$subname"
    mkdir -p "$out_dir"

    for file in "$subdir"/*.md; do
      if git_file_changed "$file"; then
        filename="$(basename "$file" .md)"
        out_html="$out_dir/${filename}.html"
        pandoc_for_file "$file" "$out_html" "$SRC_ROOT"
        echo "✓ Converted $file → $out_html"
      fi
    done
  done
done

echo "✅ Done. Outputs written under the specified target directories."
