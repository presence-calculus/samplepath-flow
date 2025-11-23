#!/usr/bin/env bash
uv run ../pypcalc/sim/kanban_sim.py --weeks 24 --wip-limit 1 --sla-pct 99 --soft-sla 1 --hard-sla 1.5
uv run samplepath --scenario 24_weeks_a  kanban_tasks.csv --start_column arrival_ts
uv run samplepath --scenario 24_weeks_s  kanban_tasks.csv
open charts/kanban_tasks/24_weeks_a/sample_path_flow_metrics.png
open charts/kanban_tasks/24_weeks_s/sample_path_flow_metrics.png