# Reorg Edit Checklist (Lean)

Reference plan: `docs/drafts/reorg/edit-plan.md`

Use this file to track execution only. Keep details in the plan.

## Status Legend

- `[ ]` Not started
- `[-]` In progress
- `[x]` Done
- `[!]` Blocked

## Phase 1: Foundation (Do First)

- [ ] Finalize canonical `Start Here` content draft
  - File: `docs/drafts/reorg/00-start-here/index.md`
  - Owner: Krishna (draft), Codex+Claude (review)

- [ ] Decide one-page vs two-page orientation model
  - Decision: keep `why-different` separate or fold into `Start Here`:
    - Decision is to keep them separate. I will do a much more informal start here, and we can probably do a slightly more technical why-different page that sits somewhere between the methodology and theory docs. Can always consolidate if redundant.
  - Owner: Krishna

- [ ] Lock canonical Polaris URL format for docs
  - Decision: `/p/...` vs mixed patterns
    - Decision is to use /p/ paths. with no url parameters. If there is a way to avoid having substack show a signup page before each view that is ideal and we should use that url.
  - Owner: Krishna

- [ ] Confirm target publish path for first orientation page
  - Decision: `docs/articles/start-here/index.md` (or alternative)
    - ok.
  - Owner: Krishna

- [ ] Choose primary invariant term
  - Decision: "Presence Invariant" (current chart-reference usage) vs "finite-window invariant"
    - Decision: Presence Invariant. I am going to fully pivot to Presence Calculus and Presence Mass framing and say that this is the finite window version of Little's LAw at the introduction and then leave it there.
  - Affects: Phase 3 terminology pass
  - Owner: Krishna

## Phase 2a: Surgical Edits — No Phase 1 Dependency

These can start immediately, before Phase 1 decisions are finalized.

- [ ] Apply Theory role statement + fix broken anchor link
  - File: `docs/articles/theory/index.md`
  - Follow: plan section 2C

- [ ] Consolidate methodology entry points
  - Files:
    - `docs/articles/not-statistics/index.md`
    - `docs/articles/a-methodological-contrast/index.md`
  - Follow: plan section 2D

- [ ] Add CLI common recipes block
  - File: `docs/articles/cli/index.md`
  - Follow: plan section 2B (recipes only; start-here pointer added in 2b)

## Phase 2b: Surgical Edits — Blocked on Start Here Path

These require the Start Here page path to be finalized (Phase 1).

- [ ] Apply Chart Reference orientation callouts + start-here pointer
  - File: `docs/articles/chart-reference/index.md`
  - Follow: plan section 2A

- [ ] Add CLI start-here pointer
  - File: `docs/articles/cli/index.md`
  - Follow: plan section 2B (pointer only)

- [ ] Trim/refocus README with Start Here pointer
  - File: `README.md`
  - Follow: plan section 2E

- [ ] Reassign Package Overview role + fix link + framing pass
  - File: `docs/articles/package-overview/index.md`
  - Follow: plan section 2F

## Phase 3: Mechanical Cleanup

- [ ] Normalize Polaris links across in-scope docs
  - Follow: plan section 3A

- [ ] Run cross-link audit and fix broken internal links
  - Follow: plan section 3B

- [ ] VUCA-forward framing pass (editorial; package-overview, README, not-statistics)
  - Follow: plan section 2F framing + section 3C scoping
  - Note: judgment call on specific paragraphs, not mechanical find-replace

- [ ] Normalize core terminology (presence invariant, event-indexed, residence/sojourn)
  - Follow: plan section 3C
  - Depends on: Phase 1 invariant term decision

## Phase 4: Build, Verify, Promote

- [ ] Build docs and sync generated site outputs
  - Command: `./docs/build/pandocs.sh`

- [ ] Run pre-commit
  - Command: `PRE_COMMIT_HOME=.pre-commit-cache pre-commit run --all-files`

- [ ] Verify no code changes require pytest
  - If code changed: run `UV_CACHE_DIR=.uv-cache uv run pytest`

- [ ] Final coherence review pass (voice + technical consistency)
  - References:
    - `docs/drafts/author-voice-profile.md`
    - `docs/drafts/reorg/edit-plan.md`

## Execution Notes

- Keep commits scoped to one coherent slice.
- Keep this checklist short; do not mirror full rationale here.
- Update statuses in this file as the single execution tracker for the surgical plan.
- Package overview long-term fate (trim vs retire and absorb history into Start Here)
  is deferred. Current plan trims and refocuses only. Revisit after this pass ships.
