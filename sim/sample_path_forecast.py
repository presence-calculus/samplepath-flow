#!/usr/bin/env python3
"""
Minimal arrival-departure sample-path simulator.

Input formats supported:

1) Sample-path CSV:
   timestamp,event
   2025-04-07T09:03:56,A
   2025-04-09T03:20:56,D

2) Work-item CSV with start/end columns, e.g.:
   id,start_ts,end_ts
   1,2025-04-07T09:03:56,2025-04-13T10:19:05

The script converts the input to an event sample path, learns empirical transition
laws for:

    P(E_{k+1}, Delta_{k+1} | E_k)

and writes inspection CSVs. Optionally, it simulates feasible continuations of the
sample path, enforcing N(t) >= 0, and estimates:

    P(D(T+h) - D(T) >= target_completions)

where h is the forecast horizon.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EVENT_A = "A"
EVENT_D = "D"
VALID_EVENTS = {EVENT_A, EVENT_D}


@dataclass(frozen=True)
class Event:
    timestamp: datetime
    event: str
    source_id: str = ""


@dataclass(frozen=True)
class Transition:
    prev_timestamp: datetime
    prev_event: str
    next_timestamp: datetime
    next_event: str
    delta_seconds: float


def parse_timestamp(value: str) -> Optional[datetime]:
    """Parse an ISO-ish timestamp. Blank values return None."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    # Allow common UTC suffix.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    # datetime.fromisoformat handles strings like 2025-04-07T09:03:56.
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse timestamp: {value!r}") from exc

    # Normalize timezone-aware timestamps to UTC-naive so sorting/comparison works
    # even if all inputs carry offsets. Naive timestamps are left as-is.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def fmt_ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        rows = list(reader)
    return reader.fieldnames, rows


def detect_or_build_events(
    headers: Sequence[str],
    rows: Sequence[Dict[str, str]],
    timestamp_col: str,
    event_col: str,
    arrival_col: str,
    departure_col: str,
    id_col: str,
    tie_break: str,
) -> List[Event]:
    header_set = set(headers)
    events: List[Event] = []

    # Format 1: already an event sample path.
    if timestamp_col in header_set and event_col in header_set:
        for i, row in enumerate(rows, start=1):
            ts = parse_timestamp(row.get(timestamp_col, ""))
            if ts is None:
                continue
            event = row.get(event_col, "").strip().upper()
            if event not in VALID_EVENTS:
                raise ValueError(f"Row {i}: event must be A or D, got {event!r}.")
            source_id = row.get(id_col, "") if id_col in header_set else ""
            events.append(Event(ts, event, source_id))
    else:
        # Format 2: work-item rows converted to A/D events.
        if arrival_col not in header_set:
            # Convenience fallback for files that call the entry time arrival_ts.
            if arrival_col == "start_ts" and "arrival_ts" in header_set:
                arrival_col = "arrival_ts"
            else:
                raise ValueError(
                    f"Could not find sample-path columns ({timestamp_col},{event_col}) "
                    f"or work-item arrival column {arrival_col!r}."
                )
        if departure_col not in header_set:
            raise ValueError(
                f"Could not find sample-path columns ({timestamp_col},{event_col}) "
                f"or work-item departure column {departure_col!r}."
            )

        for i, row in enumerate(rows, start=1):
            source_id = row.get(id_col, str(i)) if id_col in header_set else str(i)
            a_ts = parse_timestamp(row.get(arrival_col, ""))
            d_ts = parse_timestamp(row.get(departure_col, ""))
            if a_ts is not None:
                events.append(Event(a_ts, EVENT_A, source_id))
            if d_ts is not None:
                events.append(Event(d_ts, EVENT_D, source_id))

    if not events:
        raise ValueError("No events found in input.")

    # Tie break is only relevant for equal timestamps.
    if tie_break == "A_before_D":
        event_rank = {EVENT_A: 0, EVENT_D: 1}
    else:
        event_rank = {EVENT_D: 0, EVENT_A: 1}

    return sorted(events, key=lambda e: (e.timestamp, event_rank[e.event], e.source_id))


def compute_n_path(events: Sequence[Event], initial_n: int) -> List[int]:
    n = initial_n
    path = []
    for event in events:
        if event.event == EVENT_A:
            n += 1
        elif event.event == EVENT_D:
            n -= 1
        else:
            raise ValueError(f"Unknown event type: {event.event}")
        path.append(n)
    return path


def build_transitions(events: Sequence[Event]) -> List[Transition]:
    transitions: List[Transition] = []
    for prev, nxt in zip(events, events[1:]):
        delta = (nxt.timestamp - prev.timestamp).total_seconds()
        if delta < 0:
            raise ValueError("Events must be sorted by nondecreasing timestamp.")
        transitions.append(
            Transition(
                prev_timestamp=prev.timestamp,
                prev_event=prev.event,
                next_timestamp=nxt.timestamp,
                next_event=nxt.event,
                delta_seconds=delta,
            )
        )
    return transitions


def write_csv(
    path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sample_path(
    events: Sequence[Event], n_after: Sequence[int], out_dir: Path
) -> None:
    rows = []
    for idx, (event, n) in enumerate(zip(events, n_after), start=1):
        rows.append(
            {
                "event_index": idx,
                "timestamp": fmt_ts(event.timestamp),
                "event": event.event,
                "source_id": event.source_id,
                "n_after_event": n,
            }
        )
    write_csv(
        out_dir / "sample_path_events.csv",
        ["event_index", "timestamp", "event", "source_id", "n_after_event"],
        rows,
    )


def write_transition_inspection(
    transitions: Sequence[Transition], out_dir: Path
) -> None:
    transition_rows = []
    for i, tr in enumerate(transitions, start=1):
        transition_rows.append(
            {
                "transition_index": i,
                "prev_timestamp": fmt_ts(tr.prev_timestamp),
                "prev_event": tr.prev_event,
                "next_timestamp": fmt_ts(tr.next_timestamp),
                "next_event": tr.next_event,
                "delta_seconds": f"{tr.delta_seconds:.6f}",
                "delta_days": f"{tr.delta_seconds / 86400:.9f}",
            }
        )
    write_csv(
        out_dir / "empirical_transitions.csv",
        [
            "transition_index",
            "prev_timestamp",
            "prev_event",
            "next_timestamp",
            "next_event",
            "delta_seconds",
            "delta_days",
        ],
        transition_rows,
    )

    # P(E_{k+1} | E_k)
    counts_by_prev: Dict[str, int] = Counter(tr.prev_event for tr in transitions)
    pair_counts: Dict[Tuple[str, str], int] = Counter(
        (tr.prev_event, tr.next_event) for tr in transitions
    )
    next_event_rows = []
    for prev_event in sorted(VALID_EVENTS):
        denom = counts_by_prev.get(prev_event, 0)
        for next_event in sorted(VALID_EVENTS):
            count = pair_counts.get((prev_event, next_event), 0)
            prob = count / denom if denom else 0.0
            next_event_rows.append(
                {
                    "given_prev_event": prev_event,
                    "next_event": next_event,
                    "count": count,
                    "probability": f"{prob:.9f}",
                }
            )
    write_csv(
        out_dir / "next_event_distribution.csv",
        ["given_prev_event", "next_event", "count", "probability"],
        next_event_rows,
    )

    # Empirical marginal distribution of Delta_{k+1} | E_k.
    delta_counts: Dict[Tuple[str, float], int] = Counter(
        (tr.prev_event, tr.delta_seconds) for tr in transitions
    )
    delta_rows = []
    for (prev_event, delta_seconds), count in sorted(delta_counts.items()):
        denom = counts_by_prev.get(prev_event, 0)
        prob = count / denom if denom else 0.0
        delta_rows.append(
            {
                "given_prev_event": prev_event,
                "delta_seconds": f"{delta_seconds:.6f}",
                "delta_days": f"{delta_seconds / 86400:.9f}",
                "count": count,
                "probability": f"{prob:.9f}",
            }
        )
    write_csv(
        out_dir / "delta_distribution_by_prev_event.csv",
        ["given_prev_event", "delta_seconds", "delta_days", "count", "probability"],
        delta_rows,
    )

    # Empirical joint law P(E_{k+1}, Delta_{k+1} | E_k).
    joint_counts: Dict[Tuple[str, str, float], int] = Counter(
        (tr.prev_event, tr.next_event, tr.delta_seconds) for tr in transitions
    )
    joint_rows = []
    for (prev_event, next_event, delta_seconds), count in sorted(joint_counts.items()):
        denom = counts_by_prev.get(prev_event, 0)
        prob = count / denom if denom else 0.0
        joint_rows.append(
            {
                "given_prev_event": prev_event,
                "next_event": next_event,
                "delta_seconds": f"{delta_seconds:.6f}",
                "delta_days": f"{delta_seconds / 86400:.9f}",
                "count": count,
                "probability": f"{prob:.9f}",
            }
        )
    write_csv(
        out_dir / "joint_event_delta_distribution.csv",
        [
            "given_prev_event",
            "next_event",
            "delta_seconds",
            "delta_days",
            "count",
            "probability",
        ],
        joint_rows,
    )


def summarize_observed_path(
    events: Sequence[Event], n_after: Sequence[int], out_dir: Path, initial_n: int
) -> None:
    start = events[0].timestamp
    end = events[-1].timestamp
    elapsed_seconds = (end - start).total_seconds()
    arrivals = sum(1 for e in events if e.event == EVENT_A)
    departures = sum(1 for e in events if e.event == EVENT_D)
    current_n = n_after[-1]
    min_n = min([initial_n] + list(n_after))

    rows = [
        {"metric": "initial_timestamp", "value": fmt_ts(start)},
        {"metric": "final_timestamp", "value": fmt_ts(end)},
        {"metric": "elapsed_days", "value": f"{elapsed_seconds / 86400:.9f}"},
        {"metric": "initial_n", "value": initial_n},
        {"metric": "current_n", "value": current_n},
        {"metric": "min_n", "value": min_n},
        {"metric": "event_count", "value": len(events)},
        {"metric": "arrival_count", "value": arrivals},
        {"metric": "departure_count", "value": departures},
        {
            "metric": "observed_departure_rate_per_day",
            "value": (
                f"{departures / (elapsed_seconds / 86400):.9f}"
                if elapsed_seconds > 0
                else ""
            ),
        },
    ]
    write_csv(out_dir / "observed_path_summary.csv", ["metric", "value"], rows)


def group_transitions_for_sampling(
    transitions: Sequence[Transition],
) -> Dict[str, List[Transition]]:
    grouped: Dict[str, List[Transition]] = defaultdict(list)
    for tr in transitions:
        grouped[tr.prev_event].append(tr)
    return grouped


def simulate_one_path(
    grouped: Dict[str, List[Transition]],
    rng: random.Random,
    start_prev_event: str,
    start_n: int,
    horizon_seconds: float,
    max_resample_attempts: int,
) -> Dict[str, object]:
    elapsed = 0.0
    prev_event = start_prev_event
    n = start_n
    arrivals = 0
    departures = 0
    event_count = 0
    rejected_draws = 0

    while True:
        candidates = grouped.get(prev_event, [])
        if not candidates:
            break

        chosen: Optional[Transition] = None
        for _ in range(max_resample_attempts):
            tr = rng.choice(candidates)
            if tr.next_event == EVENT_D and n <= 0:
                rejected_draws += 1
                continue
            chosen = tr
            break

        # If rejection cannot find a feasible event at N=0, force a feasible arrival
        # if the empirical conditional law contains one. Otherwise stop the path.
        if chosen is None:
            feasible = [
                tr for tr in candidates if not (tr.next_event == EVENT_D and n <= 0)
            ]
            if not feasible:
                break
            chosen = rng.choice(feasible)

        if elapsed + chosen.delta_seconds > horizon_seconds:
            break

        elapsed += chosen.delta_seconds
        event_count += 1
        prev_event = chosen.next_event

        if chosen.next_event == EVENT_A:
            n += 1
            arrivals += 1
        else:
            n -= 1
            departures += 1

    return {
        "simulated_event_count": event_count,
        "simulated_arrivals": arrivals,
        "simulated_departures": departures,
        "final_n": n,
        "final_prev_event": prev_event,
        "elapsed_days_last_event": elapsed / 86400,
        "rejected_draws": rejected_draws,
    }


def run_simulation(
    transitions: Sequence[Transition],
    last_event: str,
    current_n: int,
    horizon_days: float,
    n_sims: int,
    target_completions: Optional[int],
    seed: int,
    max_resample_attempts: int,
    out_dir: Path,
) -> None:
    grouped = group_transitions_for_sampling(transitions)
    rng = random.Random(seed)
    horizon_seconds = horizon_days * 86400

    rows = []
    success_count = 0
    for sim_id in range(1, n_sims + 1):
        result = simulate_one_path(
            grouped=grouped,
            rng=rng,
            start_prev_event=last_event,
            start_n=current_n,
            horizon_seconds=horizon_seconds,
            max_resample_attempts=max_resample_attempts,
        )
        completions = int(result["simulated_departures"])
        success = target_completions is not None and completions >= target_completions
        if success:
            success_count += 1
        rows.append(
            {
                "simulation_id": sim_id,
                "horizon_days": f"{horizon_days:.6f}",
                "target_completions": (
                    "" if target_completions is None else target_completions
                ),
                "success": "" if target_completions is None else int(success),
                **{
                    k: (f"{v:.9f}" if isinstance(v, float) else v)
                    for k, v in result.items()
                },
                "future_departure_rate_per_day": (
                    f"{completions / horizon_days:.9f}" if horizon_days > 0 else ""
                ),
            }
        )

    write_csv(
        out_dir / "simulation_runs.csv",
        [
            "simulation_id",
            "horizon_days",
            "target_completions",
            "success",
            "simulated_event_count",
            "simulated_arrivals",
            "simulated_departures",
            "final_n",
            "final_prev_event",
            "elapsed_days_last_event",
            "rejected_draws",
            "future_departure_rate_per_day",
        ],
        rows,
    )

    departures = [int(row["simulated_departures"]) for row in rows]
    mean_departures = sum(departures) / len(departures) if departures else math.nan
    departures_sorted = sorted(departures)

    def quantile(p: float) -> float:
        if not departures_sorted:
            return math.nan
        # Nearest-rank style for simple inspection.
        idx = max(
            0,
            min(len(departures_sorted) - 1, math.ceil(p * len(departures_sorted)) - 1),
        )
        return float(departures_sorted[idx])

    summary_rows = [
        {"metric": "n_sims", "value": n_sims},
        {"metric": "seed", "value": seed},
        {"metric": "horizon_days", "value": f"{horizon_days:.6f}"},
        {"metric": "start_prev_event", "value": last_event},
        {"metric": "start_n", "value": current_n},
        {"metric": "mean_future_departures", "value": f"{mean_departures:.9f}"},
        {"metric": "p50_future_departures", "value": f"{quantile(0.50):.0f}"},
        {"metric": "p85_future_departures", "value": f"{quantile(0.85):.0f}"},
        {"metric": "p95_future_departures", "value": f"{quantile(0.95):.0f}"},
    ]
    if target_completions is not None:
        summary_rows.extend(
            [
                {"metric": "target_completions", "value": target_completions},
                {
                    "metric": "estimated_probability_target_met",
                    "value": f"{success_count / n_sims:.9f}",
                },
            ]
        )

    write_csv(out_dir / "simulation_summary.csv", ["metric", "value"], summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn empirical arrival-departure transition distributions and simulate feasible sample-path continuations."
    )
    parser.add_argument("input_csv", help="Input CSV path.")
    parser.add_argument(
        "--out-dir",
        default="sample_path_output",
        help="Directory for output CSV files.",
    )

    # Already-converted event sample path columns.
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--event-col", default="event")

    # Work-item columns to convert to A/D sample path.
    parser.add_argument(
        "--arrival-col",
        default="start_ts",
        help="Arrival/start timestamp column for work-item input.",
    )
    parser.add_argument(
        "--departure-col",
        default="end_ts",
        help="Departure/completion timestamp column for work-item input.",
    )
    parser.add_argument("--id-col", default="id")

    parser.add_argument(
        "--initial-n",
        type=int,
        default=0,
        help="Initial N before the first event in the input window.",
    )
    parser.add_argument(
        "--tie-break",
        choices=["A_before_D", "D_before_A"],
        default="A_before_D",
        help="Ordering for events with identical timestamps.",
    )
    parser.add_argument(
        "--strict-nonnegative-observed",
        action="store_true",
        help="Fail if the observed input path has N(t) < 0.",
    )

    # Simulation options. If horizon_days is omitted, only inspection outputs are written.
    parser.add_argument("--horizon-days", type=float, default=None)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--target-completions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-resample-attempts", type=int, default=1000)

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers, rows = read_csv_rows(input_path)
    events = detect_or_build_events(
        headers=headers,
        rows=rows,
        timestamp_col=args.timestamp_col,
        event_col=args.event_col,
        arrival_col=args.arrival_col,
        departure_col=args.departure_col,
        id_col=args.id_col,
        tie_break=args.tie_break,
    )

    n_after = compute_n_path(events, initial_n=args.initial_n)
    min_n = min([args.initial_n] + n_after)
    if args.strict_nonnegative_observed and min_n < 0:
        raise ValueError(
            f"Observed path violates N(t) >= 0 with initial_n={args.initial_n}; min_n={min_n}. "
            "Increase --initial-n or inspect the input window."
        )

    transitions = build_transitions(events)
    if not transitions:
        raise ValueError(
            "Need at least two events to estimate transition distributions."
        )

    write_sample_path(events, n_after, out_dir)
    write_transition_inspection(transitions, out_dir)
    summarize_observed_path(events, n_after, out_dir, initial_n=args.initial_n)

    if args.horizon_days is not None:
        if args.horizon_days <= 0:
            raise ValueError("--horizon-days must be positive.")
        if args.n_sims <= 0:
            raise ValueError("--n-sims must be positive.")
        run_simulation(
            transitions=transitions,
            last_event=events[-1].event,
            current_n=n_after[-1],
            horizon_days=args.horizon_days,
            n_sims=args.n_sims,
            target_completions=args.target_completions,
            seed=args.seed,
            max_resample_attempts=args.max_resample_attempts,
            out_dir=out_dir,
        )

    print(f"Wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
