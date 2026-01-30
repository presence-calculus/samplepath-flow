from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

# ---------- Core sample path flow metrics construction ----------


@dataclass
class FlowMetricsResult:
    """
    Structured finite-window flow metrics evaluated at observation times.

    Fields
    ------
    events : list[(Timestamp, int, int)]
        The (prepped) source events used for computation. If a driver zeroed-out
        arrivals prior to t0, those prepped events are stored here.
    times : list[pd.Timestamp]
        Observation times in ascending order (report points).
    L : np.ndarray                # processes
    Lambda : np.ndarray           # processes/hour
    w : np.ndarray                # hours (finite-window average residence contribution per arrival)
    w_prime : np.ndarray          # hours (finite-window average residence contribution per departure)
    N : np.ndarray                # processes
    H : np.ndarray                # process·hours
    Arrivals : np.ndarray         # cumulative arrivals Arr(T) in (t0, T]
    Departures : np.ndarray       # cumulative departures Dep(T) in (t0, T]
    Theta : np.ndarray            # processes/hour
    mode : Literal["event","calendar"]
        Observation schedule flavor used by the driver.
    freq : str | None
        Resolved pandas frequency alias when mode == "calendar", else None.
    t0 : pd.Timestamp
        Start of the finite reporting window (first observation time).
    tn : pd.Timestamp
        End of the finite reporting window (last observation time).
    arrival_times : List[pd.Timestamp]
        Timestamps of arrival events (for overlay plotting).
    departure_times : List[pd.Timestamp]
        Timestamps of departure events (for overlay plotting).

    Methods
    -------
    to_dataframe() -> pd.DataFrame
        Tabular view with columns: time, L, Lambda, Theta, w, w_prime, N, H,
        Arrivals, Departures.
    """

    events: List[Tuple[pd.Timestamp, int, int]]
    times: List[pd.Timestamp]
    L: np.ndarray
    Lambda: np.ndarray
    w: np.ndarray
    w_prime: np.ndarray
    N: np.ndarray
    H: np.ndarray
    Arrivals: np.ndarray
    Departures: np.ndarray
    Theta: np.ndarray
    mode: Literal["event", "calendar"]
    freq: Optional[str]
    t0: pd.Timestamp | NaTType = pd.NaT
    tn: pd.Timestamp | NaTType = pd.NaT
    arrival_times: List[pd.Timestamp] = field(default_factory=list)
    departure_times: List[pd.Timestamp] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.times,
                "L": self.L,
                "Lambda": self.Lambda,
                "w": self.w,
                "w_prime": self.w_prime,
                "N": self.N,
                "H": self.H,
                "Arrivals": self.Arrivals,
                "Departures": self.Departures,
                "Theta": self.Theta,
            }
        )


class MetricDerivations:
    """Centralized derivation strings for core flow metrics."""

    DERIVATIONS: dict[str, str] = {
        "A": "A(T) = ∑ arrivals in [0, T]",
        "D": "D(T) = ∑ departures in [0, T]",
        "N": "N(t) = A(t) − D(t)",
        "H": "H(T) = ∫₀ᵀ N(t) dt",
        "L": "L(T) = H(T) / T",
        "Lambda": "Λ(T) = A(T) / T",
        "Theta": "Θ(T) = D(T) / T",
        "w": "w(T) = H(T) / A(T)",
        "w_prime": "w'(T) = H(T) / D(T)",
        "W_star": "W*(T) = AVG(departure time − arrival time) for departures in [0, T]",
    }

    @classmethod
    def get(cls, key: str) -> Optional[str]:
        return cls.DERIVATIONS.get(key)


# --- Core Metrics Calculations
def compute_sample_path_flow_metrics(
    events: List[Tuple[pd.Timestamp, int, int]],
    sample_times: List[pd.Timestamp],
) -> Tuple[
    List[pd.Timestamp],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute Little’s Law–related metrics for a piecewise-constant sample path N(t)
    defined by an event log. Observation times need not be regular.

    Inputs
    ------
    events : list of (time, dN, a)
        Event time, jump in N, and arrival mark (arrivals at that instant).
        Supports batched/mixed events: if r arrivals and d departures occur at the
        same timestamp, then dN = r - d and a = r. Cumulative departures at that
        instant are recovered as dep = a - dN = d (>= 0).
    sample_times : list of pd.Timestamp
        Observation times (arbitrary, unsorted allowed). The reporting window starts
        at t0 = min(sample_times).

    Outputs (aligned to sorted sample_times)
    -------
    T_sorted : list[pd.Timestamp]
    L(T)     : np.ndarray  (processes)            Time-Average WIP since t0
    Lambda(T): np.ndarray  (processes/hour)       average arrival rate since t0
    w(T)     : np.ndarray  (hours)                finite-window average residence contribution per arrival
    N(T)     : np.ndarray  (processes)            number in system right after events ≤ T
    H(T)     : np.ndarray  (process·hours)        cumulative presence mass from t0 to T
    Arr(T)   : np.ndarray  (count)                cumulative arrivals in (t0, T]
    Dep(T)   : np.ndarray  (count)                cumulative departures in (t0, T]
    w'(T)    : np.ndarray  (hours)                finite-window average residence contribution per departure
    Θ(T)     : np.ndarray  (processes/hour)       average departure rate since t0

    Notes
    -----
    • The sample path N(t) is determined by events and does not depend on sampling times.
      We integrate exactly between events (rectangles); sampling merely selects report points.
    • Departures per event are computed as dep = a - dN, consistent with r arrivals, d departures.
    """
    if not events:
        T = sorted(sample_times)
        return (
            T,
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Sort inputs
    events = sorted(events, key=lambda e: e[0])
    T = sorted(sample_times)
    t0 = T[0]

    # Running state
    N = 0
    H = 0.0
    cum_arr = 0
    cum_dep = 0
    prev = t0

    out_L, out_Lam, out_w, out_N, out_H, out_Arr, out_Dep = ([] for _ in range(7))
    out_w_prime, out_Theta = ([] for _ in range(2))
    i = 0  # event index

    for t in T:
        # Process all events up to and including t
        while i < len(events) and events[i][0] <= t:
            t_ev, dN, a = events[i]
            # Area up to event
            dt_h = (t_ev - prev).total_seconds() / 3600.0
            if dt_h > 0:
                H += N * dt_h
                prev = t_ev
            # Jump and counts at event
            N += dN
            cum_arr += a
            dep = a - dN  # r - (r - d) = d
            if dep < 0:
                # Defensive clamp; if your data respects the r,d semantics this won't trigger.
                dep = 0
            cum_dep += dep
            i += 1

        # Tail from last event to t
        dt_h = (t - prev).total_seconds() / 3600.0
        if dt_h > 0:
            H += N * dt_h
            prev = t

        # Report metrics at t
        elapsed_h = (t - t0).total_seconds() / 3600.0
        L = (H / elapsed_h) if elapsed_h > 0 else np.nan
        Lam = (cum_arr / elapsed_h) if elapsed_h > 0 else np.nan
        w = (H / cum_arr) if cum_arr > 0 else np.nan
        w_prime = (H / cum_dep) if cum_dep > 0 else np.nan
        Theta = (cum_dep / elapsed_h) if elapsed_h > 0 else np.nan

        out_L.append(L)
        out_Lam.append(Lam)
        out_w.append(w)
        out_N.append(N)
        out_H.append(H)
        out_Arr.append(cum_arr)
        out_Dep.append(cum_dep)
        out_w_prime.append(w_prime)
        out_Theta.append(Theta)

    return (
        T,
        np.array(out_L, dtype=float),
        np.array(out_Lam, dtype=float),
        np.array(out_w, dtype=float),
        np.array(out_N, dtype=float),
        np.array(out_H, dtype=float),
        np.array(out_Arr, dtype=float),
        np.array(out_Dep, dtype=float),
        np.array(out_w_prime, dtype=float),
        np.array(out_Theta, dtype=float),
    )


def compute_finite_window_flow_metrics(
    events: List[Tuple[pd.Timestamp, int, int]],
    *,
    freq: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    include_next_boundary: bool = False,
    week_anchor: str = "SUN",
    quarter_anchor: str = "JAN",
    year_anchor: str = "JAN",
) -> FlowMetricsResult:
    """
    Consolidated driver for finite-window flow metrics using either event boundaries
    (default) or calendar boundaries for observation times.

    • freq is None  → event mode. Observations at t0, each event time in (t0, tn], and tn.
    • freq provided → calendar mode. Observations at calendar boundaries derived from `freq`
      (e.g., "D", "W-MON", "MS", "QS-JAN", "YS-JAN" or human aliases with anchors).
      Per-bucket values can be obtained by differencing cumulative arrays at consecutive boundaries.

    Window endpoints:
      t0 := first observation time; tn := last observation time.
      Arrivals prior to t0 are **zeroed** in the event marks so that Arrivals(T) counts
      only within (t0, T], while dN still establishes the correct initial N(t0).

    Returns
    -------
    FlowMetricsResult with:
      • reference to prepped `events`
      • observation `times`
      • L, Lambda, Theta, w, w_prime, N, H
      • cumulative Arrivals(T) and Departures(T)
      • mode, freq, t0, tn
    """
    if not events:
        # Empty result with minimal structure
        return FlowMetricsResult(
            events=[],
            times=[],
            L=np.array([]),
            Lambda=np.array([]),
            w=np.array([]),
            w_prime=np.array([]),
            N=np.array([]),
            H=np.array([]),
            Arrivals=np.array([]),
            Departures=np.array([]),
            Theta=np.array([]),
            mode="event" if freq is None else "calendar",
            freq=freq if freq is not None else None,
            t0=pd.NaT,
            tn=pd.NaT,
        )

    # Sort events and extract times
    events_sorted = sorted(events, key=lambda e: e[0])
    ev_times = [t for (t, _, _) in events_sorted]

    # Build observation schedule
    if freq is None:
        mode: Literal["event"] = "event"
        window_start = start if start is not None else ev_times[0]
        window_end = end if end is not None else ev_times[-1]

        obs: List[pd.Timestamp] = [pd.Timestamp(window_start)]
        for t in ev_times:
            if t > window_start and t <= window_end:
                obs.append(pd.Timestamp(t))
        if pd.Timestamp(window_end) != obs[-1]:
            obs.append(pd.Timestamp(window_end))
        obs = sorted(dict.fromkeys(obs))
        resolved_freq = None
    else:
        mode: Literal["calendar"] = "calendar"
        resolved_freq = _resolve_freq(
            freq,
            week_anchor=week_anchor,
            quarter_anchor=quarter_anchor,
            year_anchor=year_anchor,
        )
        first_ev = ev_times[0]
        last_ev = ev_times[-1]
        start_aligned = (start if start is not None else first_ev).floor(resolved_freq)
        end_aligned = (end if end is not None else last_ev).floor(resolved_freq)
        boundaries = pd.date_range(
            start=start_aligned, end=end_aligned, freq=resolved_freq
        )
        if include_next_boundary:
            off = pd.tseries.frequencies.to_offset(resolved_freq)
            if len(boundaries) == 0:
                boundaries = pd.DatetimeIndex([start_aligned])
            boundaries = boundaries.append(pd.DatetimeIndex([boundaries[-1] + off]))
        obs = list(boundaries)

    if len(obs) == 0:
        return FlowMetricsResult(
            events=[],
            times=[],
            L=np.array([]),
            Lambda=np.array([]),
            w=np.array([]),
            w_prime=np.array([]),
            N=np.array([]),
            H=np.array([]),
            Arrivals=np.array([]),
            Departures=np.array([]),
            Theta=np.array([]),
            mode=mode,
            freq=resolved_freq,
            t0=pd.NaT,
            tn=pd.NaT,
        )

    t0 = obs[0]
    tn = obs[-1]

    # Zero-out arrival marks prior to t0; retain dN to set N(t0)
    events_prepped: List[Tuple[pd.Timestamp, int, int]] = []
    for t, dN, a in events_sorted:
        events_prepped.append((t, dN, 0 if t < t0 else a))

    # Compute metrics
    T, L, Lam, w, N, H, Arr, Dep, w_prime, Theta = compute_sample_path_flow_metrics(
        events_prepped, obs
    )

    # Extract arrival and departure timestamps for overlay plotting
    arrival_times: List[pd.Timestamp] = []
    departure_times: List[pd.Timestamp] = []
    for t, delta_n, arrivals in events_prepped:
        departures = arrivals - delta_n
        if arrivals > 0:
            arrival_times.append(t)
        if departures > 0:
            departure_times.append(t)

    return FlowMetricsResult(
        events=events_prepped,
        times=T,
        L=L,
        Lambda=Lam,
        w=w,
        w_prime=w_prime,
        N=N,
        H=H,
        Arrivals=Arr,
        Departures=Dep,
        Theta=Theta,
        mode=mode,
        freq=resolved_freq,
        t0=t0,
        tn=tn,
        arrival_times=arrival_times,
        departure_times=departure_times,
    )


# --- helper to map human bucket names to pandas freq strings ---
def _resolve_freq(
    bucket: str,
    *,
    week_anchor: str = "SUN",
    quarter_anchor: str = "JAN",
    year_anchor: str = "JAN",
) -> str:
    """
    Map human-friendly names to pandas offset aliases.
    Accepts raw pandas aliases and returns them unchanged.
    """
    b = bucket.strip()
    try:
        pd.tseries.frequencies.to_offset(b)
        return b
    except Exception:
        pass

    bl = b.lower()
    if bl in ("day", "d"):
        return "D"
    if bl in ("week", "w"):
        return f"W-{week_anchor.upper()}"
    if bl in ("month", "m"):
        return "MS"
    if bl in ("quarter", "q"):
        return f"QS-{quarter_anchor.upper()}"
    if bl in ("year", "y"):
        return f"YS-{year_anchor.upper()}"

    raise ValueError(
        f"Unknown frequency '{bucket}'. Use pandas alias (e.g., 'D','W-MON','MS','QS-JAN','YS-JAN') "
        f"or one of {{day, week, month, quarter, year}}."
    )


# -------- Element-wise empirical metrics ------


@dataclass
class ElementWiseEmpiricalMetrics:
    """
    Element-wise empirical flow metrics evaluated along a sample-path timeline.

    Attributes
    ----------
    W_star : np.ndarray
        Array of empirical mean sojourn (or residence) times `W*(t)`
        aligned to the corresponding sample times.
        Each element represents the average active duration of completed
        items up to that time on the sample path.

    lam_star : np.ndarray
        Array of empirical arrival rates `λ*(t)`
        aligned to the same sample times.
        Each element represents the cumulative number of arrivals
        per unit time observed up to that point.

    sojourn_vals : np.ndarray
        Sojourn times in hours aligned to departure timestamps.

    residence_time_vals : np.ndarray
        Residence times in hours aligned to arrival timestamps.
    residence_completed : np.ndarray
        Boolean flags aligned to arrival timestamps indicating completion.


    Notes
    -----
    These quantities together form the empirical counterpart to Little’s Law:
        L*(t) = λ*(t) · W*(t)
    where `L*(t)` is the computed average work-in-process within the window
    ending at time `t`.

    The process is fully convergent when `L*(T) = L(T)`, i.e., when the empirical
    value matches the observed time-average of work-in-process.

    These metrics are used in convergence checks that compare the values
    returned by the `FlowMetricsResult` for an observation window
    against these empirical series.

    The arrays are element-wise aligned and can be safely combined with
    corresponding `times` vectors in downstream analysis or visualization.
    """

    times: List[pd.Timestamp]
    W_star: np.ndarray
    lam_star: np.ndarray
    sojourn_vals: np.ndarray
    residence_time_vals: np.ndarray
    residence_completed: np.ndarray

    def as_tuple(self) -> [np.ndarray, np.ndarray]:
        return self.W_star, self.lam_star


def compute_elementwise_empirical_metrics(
    df: pd.DataFrame, times: List[pd.Timestamp]
) -> ElementWiseEmpiricalMetrics:
    def _compute_elementwise_empirical_metrics(
        df: pd.DataFrame, times: List[pd.Timestamp]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return W*(t), λ*(t), sojourn values, and residence values."""
        n = len(times)
        W_star = np.full(n, np.nan, dtype=float)
        lam_star = np.full(n, np.nan, dtype=float)
        sojourn_vals = np.array([])
        residence_vals = np.array([])
        residence_completed = np.array([], dtype=bool)
        if n == 0:
            return W_star, lam_star, sojourn_vals, residence_vals, residence_completed

        comp = df[df["end_ts"].notna()].copy().sort_values("end_ts")
        if not comp.empty:
            sojourn_vals = (
                (comp["end_ts"] - comp["start_ts"]).dt.total_seconds() / 3600.0
            ).to_numpy()
        comp_end_times = comp["end_ts"].to_list()

        starts = df["start_ts"].sort_values(kind="mergesort").to_list()

        j = 0
        count_c = 0
        sum_c = 0.0
        k = 0
        count_starts = 0
        t0 = times[0]

        for i, t in enumerate(times):
            while j < len(comp_end_times) and comp_end_times[j] <= t:
                sum_c += sojourn_vals[j]
                count_c += 1
                j += 1
            if count_c > 0:
                W_star[i] = sum_c / count_c

            while k < len(starts) and starts[k] <= t:
                count_starts += 1
                k += 1
            elapsed_h = (t - t0).total_seconds() / 3600.0
            if elapsed_h > 0:
                lam_star[i] = count_starts / elapsed_h

        arrivals = df.sort_values("start_ts", kind="mergesort").copy()
        if not arrivals.empty:
            residence_completed = arrivals["end_ts"].notna().to_numpy(dtype=bool)
            residence_vals = np.full(len(arrivals), np.nan, dtype=float)
            completed = arrivals[residence_completed]
            if not completed.empty:
                residence_vals[residence_completed] = (
                    completed["end_ts"] - completed["start_ts"]
                ).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            open_items = arrivals[~residence_completed]
            if not open_items.empty and times:
                tn = times[-1]
                if pd.notna(tn):
                    residence_vals[~residence_completed] = (
                        tn - open_items["start_ts"]
                    ).dt.total_seconds().to_numpy(dtype=float) / 3600.0

        return W_star, lam_star, sojourn_vals, residence_vals, residence_completed

    (
        W_star,
        lam_star,
        sojourn_vals,
        residence_vals,
        residence_completed,
    ) = _compute_elementwise_empirical_metrics(df, times)
    return ElementWiseEmpiricalMetrics(
        times=times,
        W_star=W_star,
        lam_star=lam_star,
        sojourn_vals=sojourn_vals,
        residence_time_vals=residence_vals,
        residence_completed=residence_completed,
    )
