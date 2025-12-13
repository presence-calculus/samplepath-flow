#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob  # allow empty globs without errors

# ---------------------------------------------------------
# Locate script directory (portable, works via symlinks too)
# ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------
# Config file: default is pandocs.conf next to this script,
# but can be overridden via PANDOCS_CONFIG env var
# ---------------------------------------------------------
CONFIG_FILE="${PANDOCS_CONFIG:-$SCRIPT_DIR/pandocs.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

is_asset() {
  local f="$1"
  local ext="${f##*.}"
  ext="$(printf '%s' "$ext" | tr 'A-Z' 'a-z')"
  local a
  for a in "${ASSET_EXTENSIONS[@]}"; do
    if [[ "$ext" == "$a" ]]; then
      return 0
    fi
  done
  return 1
}

# -------- Git helpers --------
# Allow overriding from the command line, e.g.:
#   GIT_AWARE=0 ./pandocs.sh   # rebuild everything
if [[ -z "${GIT_AWARE+x}" ]]; then
  GIT_AWARE=0
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_AWARE=1
  fi
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

  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"

  # Find resources with fallback order: file dir -> src_root -> git root
  local bib csl tpl
  bib="$(first_existing \
        "$dir/references.bib" \
        "$src_root/references.bib" \
        ${repo_root:+$repo_root/docs/build/references.bib} 2>/dev/null || true)"
  csl="$(first_existing \
        "$dir/ieee.csl" \
        "$src_root/ieee.csl" \
        ${repo_root:+$repo_root/docs/build/ieee.csl} 2>/dev/null || true)"
  tpl="$(first_existing \
        "$dir/pandoc_template.html" \
        "$src_root/pandoc_template.html" \
        ${repo_root:+$repo_root/docs/build/pandoc_template.html} 2>/dev/null || true)"

  # ---------------------------------------------------------
  # 1. Base Arguments (Minimal)
  # ---------------------------------------------------------
  #
  local -a args=(
    --lua-filter="$repo_root/docs/build/interpolate-vars.lua"
    --lua-filter="$repo_root/docs/build/md2html-links.lua"
    --wrap=auto
    --quiet
  )

  # ---------------------------------------------------------
  # 2. Conditional Logic (The "Checks")
  # ---------------------------------------------------------

  # FEATURE A: Cross-references (Section & Figure Numbers)
  # Triggered by: 'numberSections: true' in YAML
  # We use PREPEND to ensure crossref runs before other filters.
  if grep -qE "^[[:space:]]*numberSections:[[:space:]]*true" "$file"; then
     args=( --filter pandoc-crossref "${args[@]}" )
     # Note: We do NOT add --number-sections here; pandoc-crossref handles it.
  fi

  # FEATURE B: Citations
  # Triggered by: 'citations: true' OR 'link-citations: true'
  if grep -qE "^[[:space:]]*(citations|link-citations):[[:space:]]*true" "$file"; then
    args+=( --citeproc )

    # Only load bibliography/csl if citation mode is active AND files exist
    [[ -n "${bib:-}" ]] && args+=( --bibliography="$bib" )
    [[ -n "${csl:-}" ]] && args+=( --csl="$csl" )
  fi

  # FEATURE C: Table of Contents
  # Triggered by: 'toc: true'
  if grep -qE "^[[:space:]]*toc:[[:space:]]*true" "$file"; then
    args+=( --toc )

    # Check for optional Depth limit (toc-depth: X)
    local toc_depth_val
    toc_depth_val=$(grep "^[[:space:]]*toc-depth:" "$file" | head -n 1 | awk -F: '{print $2}' | tr -d '[:space:]')
    if [[ -n "$toc_depth_val" ]]; then
      args+=( --toc-depth="$toc_depth_val" )
    fi
  fi

  # ---------------------------------------------------------
  # 3. Final Assembly
  # ---------------------------------------------------------

  # Template is applied to all HTML outputs (if found)
  [[ -n "${tpl:-}" ]] && args+=( --template="$tpl" )

  args+=( -s "$file" -o "$out_html" )

  if [[ -n "${MATH_ENGINE:-}" ]]; then
    # Allow multi-token engines like "--mathjax --katex"
    # shellcheck disable=SC2206
    local -a math_tokens=( $MATH_ENGINE )
    args+=( "${math_tokens[@]}" )
  fi

  echo "${args[@]}"
  pandoc "${args[@]}"
}

# -------- Processing --------
for pair in "${SOURCE_MAP[@]}"; do
  IFS=":" read -r SRC_SPEC TGT_ROOT <<<"$pair"

  # Per-pair recursion: "src/**:dst" means recursive under src
  RECURSIVE=0
  SRC_ROOT="$SRC_SPEC"
  if [[ "$SRC_SPEC" == *"/**" ]]; then
    RECURSIVE=1
    SRC_ROOT="${SRC_SPEC%"/**"}"
  fi

  if [[ ! -d "$SRC_ROOT" ]]; then
    echo "⚠️  Skipping missing source dir: $SRC_ROOT"
    continue
  fi

  mkdir -p "$TGT_ROOT"

  if [[ "$RECURSIVE" -eq 1 ]]; then
    # Recursive: walk entire tree under SRC_ROOT for markdown
    while IFS= read -r -d '' file; do
      [[ -f "$file" ]] || continue
      if git_file_changed "$file"; then
        rel="${file#$SRC_ROOT/}"
        out_html="$TGT_ROOT/${rel%.md}.html"
        mkdir -p "$(dirname "$out_html")"
        pandoc_for_file "$file" "$out_html" "$SRC_ROOT"
        echo "✓ Converted $file → $out_html"
      fi
    done < <(find "$SRC_ROOT" -type f -name '*.md' -print0)
  else
    # Flat mode: top-level files in SRC_ROOT → TGT_ROOT
    for file in "$SRC_ROOT"/*.md; do
      [[ -f "$file" ]] || continue
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
        [[ -f "$file" ]] || continue
        if git_file_changed "$file"; then
          filename="$(basename "$file" .md)"
          out_html="$out_dir/${filename}.html"
          pandoc_for_file "$file" "$out_html" "$SRC_ROOT"
          echo "✓ Converted $file → $out_html"
        fi
      done
    done
  fi

  # -------- Asset copying (per pair) --------
  if [[ "$RECURSIVE" -eq 1 ]]; then
    # Recursive assets: all files excluding .md, filtered by ASSET_EXTENSIONS
    while IFS= read -r -d '' src; do
      [[ -f "$src" ]] || continue
      [[ "$src" == *.md ]] && continue
      if ! is_asset "$src"; then
        continue
      fi
      if ! git_file_changed "$src"; then
        continue
      fi
      rel="${src#$SRC_ROOT/}"
      dst="$TGT_ROOT/$rel"
      mkdir -p "$(dirname "$dst")"
      cp "$src" "$dst"
      echo "✓ Copied asset $src → $dst"
    done < <(find "$SRC_ROOT" -type f -print0)
  else
    # Flat assets: top + one level
    for src in "$SRC_ROOT"/* "$SRC_ROOT"/*/*; do
      [[ -f "$src" ]] || continue
      [[ "$src" == *.md ]] && continue
      if ! is_asset "$src"; then
        continue
      fi
      if ! git_file_changed "$src"; then
        continue
      fi
      rel="${src#$SRC_ROOT/}"
      dst="$TGT_ROOT/$rel"
      mkdir -p "$(dirname "$dst")"
      cp "$src" "$dst"
      echo "✓ Copied asset $src → $dst"
    done
  fi
done

echo "✅ Done. Outputs written under the specified target directories."
