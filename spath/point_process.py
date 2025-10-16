# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from typing import List, Tuple
import pandas as pd
"""
point_process
-------------

Marked point processes model events in time (or space) with extra data attached
to each event, cleanly separating *when* things happen from *what* they are.
They are a powerful abstraction for queues, monitoring systems, concurrent
processes, spatio-temporal data, and more.

From these primitives, we can define stochastic processes such as N(t) — the
number of items in a queue at time t — and study their sample paths. This module
provides the core tools for building, sorting, and analyzing marked point processes.

Currently we only model arrival departure processes. 

In more concrete terms:
    - You have a list of event times: T1, T2, T3, ...
    - Each event carries a *mark* — data you care about, such as:
        * how long it lasted
        * what type of event it was
        * a location or severity score
        * a vector of such attributes
    - Together, each event is represented as: (time, mark)

Examples:
    - In a queue: arrivals and departures happen at certain times, with marks
      indicating whether it is an arrival or departure, how many units arrived,
      IDs, classes of events, etc.
    - In monitoring: errors occur at certain times, with marks for error codes
      or subsystems.
    - In concurrent systems: timestamps mark process start or end events, with
      marks indicating launch or termination.
    - In spatio-temporal data: earthquakes occur at certain times, with marks
      for magnitude and location.

Why this abstraction is useful:
    - Applies to many systems: queueing, engineering processes, sensor logs, and more.
    - Easy to simulate or analyze statistically.
    - Cleanly separates *when* events occur from *what* they are.


## Point Processes and Sample Paths

Given a marked point process, we can define stochastic processes as random
variables derived from it. For example, for an arrival/departure point process
in a queue, we can define N(t) — the number of items in the queue at time t —
as a stochastic process. A sample path is a single realization of this process,
showing how N(t) evolves over time in one specific scenario.

Point processes are the primitives from which we build sample paths. This
module provides the basics: transforming input data into marked point processes,
sorting them, and performing simple analyses.

The combination of event times and marks allows us to define and study a wide
range of stochastic processes and their sample paths.
"""




def to_arrival_departure_process(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, int, int]]:
    """
    Construct a sorted list of arrival and departure events from a time interval DataFrame.

    Each row in the input DataFrame is expected to represent the active interval of an
    entity in the system, with a start timestamp (`start_ts`) and an optional end timestamp (`end_ts`).
    This function transforms those intervals into a set of discrete events, where each event
    encodes:

        - The event time (pd.Timestamp)
        - The change in active entity count at that instant (`deltaN`)
            * +1 for an arrival
            * -1 for a departure
        - The number of arrivals that occurred at exactly that time (`arrivals_at_time`)
            * 1 for arrival events
            * 0 for departure events

    Event ordering:
        - Events are sorted by timestamp in ascending order.
        - For events with the same timestamp, arrivals (+1) are ordered before departures (-1).
          This ensures that simultaneous arrivals and departures at the same time step
          increment the count before decrementing it.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
            - 'start_ts': pd.Timestamp of arrival
            - 'end_ts':   pd.Timestamp of departure (NaT if ongoing)

    Returns
    -------
    List[Tuple[pd.Timestamp, int, int]]
        Sorted list of (time, deltaN, arrivals_at_time) tuples.

    Notes
    -----
    - Rows with missing `end_ts` will contribute only an arrival event and no departure.
    - Multiple intervals starting or ending at the same timestamp will produce multiple events
      at that time, each captured separately in the output list.
    - This representation is suitable for constructing point processes, cumulative counts,
      or feeding into downstream event-based simulations.

    Examples
    --------
    >>> import pandas as pd
    >>> data = [
    ...     {"start_ts": pd.Timestamp("2024-01-01 09:00"), "end_ts": pd.Timestamp("2024-01-01 10:00")},
    ...     {"start_ts": pd.Timestamp("2024-01-01 09:30"), "end_ts": pd.Timestamp("2024-01-01 11:00")},
    ... ]
    >>> df = pd.DataFrame(data)
    >>> to_arrival_departure_process(df)
    [
        (Timestamp('2024-01-01 09:00:00'),  1, 1),
        (Timestamp('2024-01-01 09:30:00'),  1, 1),
        (Timestamp('2024-01-01 10:00:00'), -1, 0),
        (Timestamp('2024-01-01 11:00:00'), -1, 0)
    ]
    """
    events: List[Tuple[pd.Timestamp, int, int]] = []
    for _, row in df.iterrows():
        st = row["start_ts"]
        events.append((st, +1, 1))
        et = row["end_ts"]
        if pd.notna(et):
            events.append((et, -1, 0))
    events.sort(key=lambda x: (x[0], -x[1]))
    return events
