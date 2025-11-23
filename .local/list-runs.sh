#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./list-runs.sh <owner> <repo> [workflow_file]
#
# Examples:
#   ./list-runs.sh presence-calculus samplepath
#   ./list-runs.sh presence-calculus samplepath ci.yml

OWNER="$1"
REPO="$2"
WORKFLOW_FILE="${3:-}"

# Base gh api path
if [[ -n "$WORKFLOW_FILE" ]]; then
  # Filter runs for a specific workflow file
  API_PATH="/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/runs"
else
  # All workflow runs for the repo
  API_PATH="/repos/${OWNER}/${REPO}/actions/runs"
fi


# Paginate through all runs
PAGE=1

echo "run_id,run_number,status,conclusion,run_started_at,completed_at"

gh api "$API_PATH" \
  --paginate \
  --jq '
    .workflow_runs
    | map({
        id,
        run_number,
        status,
        conclusion,
        run_started_at,
        completed_at: (if .status == "completed" then .updated_at else null end)
      })
    | .[]
    | [
        .id,
        .run_number,
        .status,
        (.conclusion // ""),
        (.run_started_at // ""),
        (.completed_at // "")
      ]
    | @csv
  '

  # gh --paginate already walks pages; we can just break after first call
  # if you’d rather not rely on --paginate, comment out the break and manage PAGE++.

