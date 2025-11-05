from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
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
    N : np.ndarray                # processes
    A : np.ndarray                # process·hours
    Arrivals : np.ndarray         # cumulative arrivals Arr(T) in (t0, T]
    Departures : np.ndarray       # cumulative departures Dep(T) in (t0, T]
    mode : Literal["event","calendar"]
        Observation schedule flavor used by the driver.
    freq : str | None
        Resolved pandas frequency alias when mode == "calendar", else None.
    t0 : pd.Timestamp
        Start of the finite reporting window (first observation time).
    tn : pd.Timestamp
        End of the finite reporting window (last observation time).

    Methods
    -------
    to_dataframe() -> pd.DataFrame
        Tabular view with columns: time, L, Lambda, w, N, A, Arrivals, Departures.
    """
    events: List[Tuple[pd.Timestamp, int, int]]
    times: List[pd.Timestamp]
    L: np.ndarray
    Lambda: np.ndarray
    w: np.ndarray
    N: np.ndarray
    A: np.ndarray
    Arrivals: np.ndarray
    Departures: np.ndarray
    mode: Literal["event", "calendar"]
    freq: Optional[str]
    t0: pd.Timestamp | NaTType = pd.NaT
    tn: pd.Timestamp | NaTType = pd.NaT

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.times,
                "L": self.L,
                "Lambda": self.Lambda,
                "w": self.w,
                "N": self.N,
                "A": self.A,
                "Arrivals": self.Arrivals,
                "Departures": self.Departures,
            }
        )

#--- Core Metrics Calculations
def compute_sample_path_metrics(
    events: List[Tuple[pd.Timestamp, int, int]],
    sample_times: List[pd.Timestamp],
) -> Tuple[List[pd.Timestamp], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    A(T)     : np.ndarray  (process·hours)        area under N(t) from t0 to T
    Arr(T)   : np.ndarray  (count)                cumulative arrivals in (t0, T]
    Dep(T)   : np.ndarray  (count)                cumulative departures in (t0, T]

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
        )

    # Sort inputs
    events = sorted(events, key=lambda e: e[0])
    T = sorted(sample_times)
    t0 = T[0]

    # Running state
    N = 0
    A = 0.0
    cum_arr = 0
    cum_dep = 0
    prev = t0

    out_L, out_Lam, out_w, out_N, out_A, out_Arr, out_Dep = ([] for _ in range(7))
    i = 0  # event index

    for t in T:
        # Process all events up to and including t
        while i < len(events) and events[i][0] <= t:
            t_ev, dN, a = events[i]
            # Area up to event
            dt_h = (t_ev - prev).total_seconds() / 3600.0
            if dt_h > 0:
                A += N * dt_h
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
            A += N * dt_h
            prev = t

        # Report metrics at t
        elapsed_h = (t - t0).total_seconds() / 3600.0
        L = (A / elapsed_h) if elapsed_h > 0 else np.nan
        Lam = (cum_arr / elapsed_h) if elapsed_h > 0 else np.nan
        w = (A / cum_arr) if cum_arr > 0 else np.nan

        out_L.append(L)
        out_Lam.append(Lam)
        out_w.append(w)
        out_N.append(N)
        out_A.append(A)
        out_Arr.append(cum_arr)
        out_Dep.append(cum_dep)

    return (
        T,
        np.array(out_L, dtype=float),
        np.array(out_Lam, dtype=float),
        np.array(out_w, dtype=float),
        np.array(out_N, dtype=float),
        np.array(out_A, dtype=float),
        np.array(out_Arr, dtype=float),
        np.array(out_Dep, dtype=float),
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
      • L, Lambda, w, N, A
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
            N=np.array([]),
            A=np.array([]),
            Arrivals=np.array([]),
            Departures=np.array([]),
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
        boundaries = pd.date_range(start=start_aligned, end=end_aligned, freq=resolved_freq)
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
            N=np.array([]),
            A=np.array([]),
            Arrivals=np.array([]),
            Departures=np.array([]),
            mode=mode,
            freq=resolved_freq,
            t0=pd.NaT,
            tn=pd.NaT,
        )

    t0 = obs[0]
    tn = obs[-1]

    # Zero-out arrival marks prior to t0; retain dN to set N(t0)
    events_prepped: List[Tuple[pd.Timestamp, int, int]] = []
    for (t, dN, a) in events_sorted:
        events_prepped.append((t, dN, 0 if t < t0 else a))

    # Compute metrics
    T, L, Lam, w, N, A, Arr, Dep = compute_sample_path_metrics(events_prepped, obs)

    return FlowMetricsResult(
        events=events_prepped,
        times=T,
        L=L,
        Lambda=Lam,
        w=w,
        N=N,
        A=A,
        Arrivals=Arr,
        Departures=Dep,
        mode=mode,
        freq=resolved_freq,
        t0=t0,
        tn=tn,
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

#-------- Element-wise empirical metrics ------

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

    times:  List[pd.Timestamp]
    W_star: np.ndarray
    lam_star: np.ndarray

    def as_tuple(self) -> [np.ndarray, np.ndarray]:
        return self.W_star, self.lam_star


def compute_elementwise_empirical_metrics(df: pd.DataFrame, times: List[pd.Timestamp]) -> ElementWiseEmpiricalMetrics:
    def _compute_elementwise_empirical_metrics(df: pd.DataFrame,
                                                  times: List[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
        """Return W*(t) and λ*(t) aligned to `times`."""
        n = len(times)
        W_star = np.full(n, np.nan, dtype=float)
        lam_star = np.full(n, np.nan, dtype=float)
        if n == 0:
            return W_star, lam_star

        comp = df[df["end_ts"].notna()].copy().sort_values("end_ts")
        comp_durations = ((comp["end_ts"] - comp["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
        comp_end_times = comp["end_ts"].to_list()

        starts = df["start_ts"].sort_values().to_list()

        j = 0
        count_c = 0
        sum_c = 0.0
        k = 0
        count_starts = 0
        t0 = times[0]

        for i, t in enumerate(times):
            while j < len(comp_end_times) and comp_end_times[j] <= t:
                sum_c += comp_durations[j]
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

        return W_star, lam_star

    W_star, lam_star = _compute_elementwise_empirical_metrics(df, times)
    return ElementWiseEmpiricalMetrics(
        times=times,
        W_star=W_star,
        lam_star=lam_star
    )

#--------- Calculating end effects and tracking errors ------ #

def compute_end_effect_series(df: pd.DataFrame,
                              times: List[pd.Timestamp],
                              A_vals: np.ndarray,
                              W_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute end-effect diagnostics over [t0, t]:

    Returns arrays aligned to `times`:
      - rA(T) = E(T) / A(T), where E(T) = A(T) - sum(full durations of items fully contained)
      - rB(T) = B(T) / total_items_started_by_t, boundary share
      - rho(T) = T / W*(t), window/typical-duration ratio
    """
    n = len(times)
    rA = np.full(n, np.nan, dtype=float)
    rB = np.full(n, np.nan, dtype=float)
    rho = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rA, rB, rho

    df = df.copy()
    df["duration_h"] = (df["end_ts"] - df["start_ts"]).dt.total_seconds() / 3600.0
    df_sorted_by_end = df.sort_values("end_ts")
    df_sorted_by_start = df.sort_values("start_ts")

    t0 = times[0]

    for i, t in enumerate(times):
        elapsed_h = (t - t0).total_seconds() / 3600.0
        if elapsed_h <= 0:
            continue

        A_T = float(A_vals[i]) if i < len(A_vals) and np.isfinite(A_vals[i]) else np.nan
        if not np.isfinite(A_T) or A_T <= 0:
            continue

        mask_full = df_sorted_by_end["end_ts"].notna() & (df_sorted_by_end["end_ts"] <= t)
        A_full = float(df_sorted_by_end.loc[mask_full, "duration_h"].sum()) if mask_full.any() else 0.0

        E_T = max(A_T - A_full, 0.0)
        rA[i] = E_T / A_T if A_T > 0 else np.nan

        mask_started = (df_sorted_by_start["start_ts"] <= t)
        total_started = int(mask_started.sum())
        mask_incomplete_by_t = mask_started & ((df_sorted_by_start["end_ts"].isna()) | (df_sorted_by_start["end_ts"] > t))
        B_T = int(mask_incomplete_by_t.sum())
        rB[i] = (B_T / total_started) if total_started > 0 else np.nan

        Wstar_t = float(W_star[i]) if i < len(W_star) else float('nan')
        rho[i] = (elapsed_h / Wstar_t) if np.isfinite(Wstar_t) and Wstar_t > 0 else np.nan

    return rA, rB, rho

def compute_tracking_errors(times: List[pd.Timestamp],
                            w_vals: np.ndarray,
                            lam_vals: np.ndarray,
                            W_star: np.ndarray,
                            lam_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    t0 = times[0]
    elapsed_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)

    eW = np.full(n, np.nan, dtype=float)
    eLam = np.full(n, np.nan, dtype=float)

    valid_W = np.isfinite(w_vals) & np.isfinite(W_star) & (W_star > 0)
    valid_L = np.isfinite(lam_vals) & np.isfinite(lam_star) & (lam_star > 0)

    eW[valid_W] = np.abs(w_vals[valid_W] - W_star[valid_W]) / W_star[valid_W]
    eLam[valid_L] = np.abs(lam_vals[valid_L] - lam_star[valid_L]) / lam_star[valid_L]

    return eW, eLam, elapsed_hours


def compute_coherence_score(eW: np.ndarray,
                            eLam: np.ndarray,
                            elapsed_hours: np.ndarray,
                            epsilon: float,
                            horizon_hours: float) -> Tuple[float, int, int]:
    ok_idx = np.isfinite(eW) & np.isfinite(eLam) & (elapsed_hours >= horizon_hours)
    total = int(np.sum(ok_idx))
    if total == 0:
        return float('nan'), 0, 0
    coherent = int(np.sum(np.maximum(eW[ok_idx], eLam[ok_idx]) <= epsilon))
    return coherent / total, coherent, total

