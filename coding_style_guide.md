# Coding Style Guide

## General function signature rules
- Group related keyword args together in signatures: `show_title`, `title`, `show_derivations` (feature toggle first, then feature controls).

## Title and Derivation Handling (Core Panels)
- Place default titles as `title: str = "..."` parameters on the canonical renderer for each metric (one default per metric).
- Use a shared `construct_title(base_title, show_derivations, derivation_key)` helper for derivation-aware titles.
- Use a single `derivation_key` per title (no varargs) for these renderers.
