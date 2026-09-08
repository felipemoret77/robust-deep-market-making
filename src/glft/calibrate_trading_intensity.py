#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_trading_intensity.py
==============================

Calibration toolkit for the GLFT (Gueant-Lehalle-Fernandez-Tapia) market-making
model.  Everything revolves around a single parametric assumption:

        lambda(delta) = A * exp(-kappa * delta)

where:
    - lambda(delta) is the *execution intensity* — the rate at which an order
      posted at distance delta from the mid-price gets filled.
    - A > 0 is the baseline intensity (intercept).
    - kappa > 0 is the decay parameter (how fast fills drop off with distance).

This file provides:
    (1) Functions to measure execution depth from the simulator's event log,
    (2) Non-queue-aware AND queue-aware tail-count / intensity curves,
    (3) A unified fitting routine for the A/kappa parameters via WLS regression,
    (4) Volatility estimation (event-clock, bucket-clock, calendar-time),
    (5) Volatility signature plots (with Monte Carlo averaging),
    (6) Per-level execution intensity decomposition (PMF + rate),
    (7) Pr(MO | delta) estimation for conditional fill probabilities.

===========================================================================
FILE LAYOUT (section order)
===========================================================================
  SECTION 0 — Imports
  SECTION 1 — Basic utilities (seeding, depth mapping, MO detection, pre-event
               book/mid extraction, side-array extraction, split-sweep collapse,
               event-level-from-price)
  SECTION 2 — Aggregation / Bucketing (event, tob, time, fixed_events clocks)
  SECTION 3 — Non-queue-aware depth sweep + tail counts
  SECTION 4 — Queue-aware depth sweep + tail counts (rank semantics)
  SECTION 5 — Sanity checks (rank monotonicity)
  SECTION 6 — Calibration wrappers (one-call simulator + tail count)
  SECTION 7 — Fit A and kappa via log-linear WLS regression
  SECTION 8 — Mid-price series builder (_build_mid_series_ticks)
  SECTION 9 — Fit volatility (event/bucket/calendar clocks)
  SECTION 10 — Volatility signature plot (multi-path Monte Carlo)
  SECTION 11 — Event-time signature plot (standalone)
  SECTION 12 — Per-level execution intensity from PMF (lambda_exec)
  SECTION 13 — Pr(MO | delta) estimation (touch vs exact modes)
  SECTION 14 — Example usage (__main__)

===========================================================================
KEY CONCEPTS — read this before modifying the code
===========================================================================

DEPTH INDEX CONVENTION
----------------------
We discretize "distance from the pre-event mid-price" into an integer index k:

    k = round(depth / half_tick) - 1

with half_tick = 0.5, which is the standard when:
    mid = (best_bid + best_ask) / 2

This means:
    k = 0  <-->  delta = 0.5 ticks from mid (best price level)
    k = 1  <-->  delta = 1.0 ticks from mid
    k = 2  <-->  delta = 1.5 ticks from mid
    ...

QUEUE RANK SEMANTICS
--------------------
"Rank r" means position r in the FIFO queue *at a single price level*.
A market order sweeping a level with volume V executes ranks 0, 1, ..., V-1
at that level.  For each MO event, we define:

    D_r = deepest depth index k at which rank r is executed
          (across all price levels touched by the sweep)

This guarantees: D_0 >= D_1 >= D_2 >= ...
and therefore:   tail[:,0] >= tail[:,1] >= tail[:,2] >= ...

TAIL COUNTS
-----------
Non-queue-aware (1D):
    tail[k]    = #{ events with D >= k }    (strict=False)
    tail[k]    = #{ events with D >  k }    (strict=True, hftbacktest-style)

Queue-aware (2D, by rank r):
    tail[k, r] = #{ events with D_r >= k }  (strict=False)
    tail[k, r] = #{ events with D_r >  k }  (strict=True)

AGGREGATION MODES ("decision clocks")
--------------------------------------
All tail-count and intensity functions accept an ``aggregation_mode`` parameter
that controls how events are grouped before counting:

    "event"        — One observation per simulator step (default, finest grain).
    "tob"          — One observation per N top-of-book (TOB) changes.
    "time"         — One observation per T seconds of wall-clock time.
    "fixed_events" — One observation per N simulator steps (fixed blocks).

In bucket modes, each bucket collapses to a single observation (e.g. the
deepest MO depth within that bucket), and the denominator for intensity
computation is the TOTAL number of buckets (including empty ones), not just
those containing a trade.

OB SNAPSHOT ALIGNMENT (critical)
---------------------------------
In this simulator, ``update_ob_dict(i)`` runs AFTER event i.  So:
    - ob_df.iloc[i]   = post-event snapshot for event i
    - ob_df.iloc[i-1]  = PRE-event snapshot for event i  (for i >= 1)

The pre-event mid is derived from the message log:
    mid_pre = MidPrice[i] - Return[i]
"""

# =========================================================================
# SECTION 0 — IMPORTS
# =========================================================================

from typing import Tuple, Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MM_LOB_SIM import simulate_LOB_with_MM
import random
import inspect
import math

# =========================================================================
# SECTION 1 — BASIC UTILITIES
# =========================================================================
# Small, stateless helpers used throughout the file.  They handle:
#   - Reproducible seeding (numpy + stdlib random)
#   - Depth-to-tick-index conversion (the k = round(d/h) - 1 formula)
#   - Market-order row detection (robust to float/string Type column)
#   - Pre-event mid-price and pre-event book snapshot extraction
#   - Side-array extraction (bid or ask prices/sizes from an OB snapshot)
#   - Dynamic plot labels for the different aggregation modes
# =========================================================================

def _format_aggregation_label(mode: str, time_interval: float, n_tob: int, n_fixed: int) -> str:
    """Build a human-readable string describing the current aggregation clock.

    This is used in plot titles and legends so the reader immediately knows
    which "decision clock" was active.  For example:
        mode="time", time_interval=1.0 -> "Time (1.0s)"
        mode="tob",  n_tob=10          -> "TOB (10 moves)"
    """
    if mode == "time":
        return f"Time ({time_interval}s)"
    elif mode == "tob":
        return f"TOB ({n_tob} moves)"
    elif mode == "fixed_events":
        return f"Fixed Events ({n_fixed})"
    elif mode == "event":
        return "Event (Tick-by-Tick)"
    else:
        return f"{mode}"

def _seed_everything(seed: Optional[int]) -> None:
    """Seed both numpy and Python's stdlib ``random`` for full reproducibility.

    The simulator and some MM policies mix both RNG sources.  Seeding only
    numpy leaves ``random.random()`` calls non-deterministic, producing
    different LOB paths on each run even with the same numpy seed.

    If *seed* is None (or not convertible to int), this function is a no-op.
    """
    if seed is None:
        return
    try:
        s = int(seed)
    except Exception:
        return
    random.seed(s)
    np.random.seed(s)


def depth_to_tick_index(depth: float, half_tick: float) -> int:
    """Convert a continuous depth (distance from mid, in tick units) to an integer
    depth index k using the convention:

        k = round(depth / half_tick) - 1

    Examples (with half_tick = 0.5):
        depth = 0.5  ->  k = 0   (best level)
        depth = 1.0  ->  k = 1   (second level)
        depth = 1.5  ->  k = 2
    """
    return int(round(depth / half_tick) - 1)


def _is_mo_row(message_df: pd.DataFrame, idx: int) -> bool:
    """Return True if row *idx* of *message_df* represents a Market Order.

    The simulator may store the Type column as a float (1.0 for MO) or as a
    string ("MO").  This helper handles both representations robustly.
    """
    t = message_df.iloc[idx]["Type"]
    try:
        return int(t) == 1
    except Exception:
        return str(t) == "MO"


def _get_pre_event_mid_from_message(message_df: pd.DataFrame, idx: int) -> float:
    """Compute the PRE-event mid-price from the message log.

    The simulator stores:
        Return[i] = MidPrice_after_event_i - MidPrice_before_event_i

    Therefore the pre-event mid is:
        mid_pre = MidPrice[i] - Return[i]

    Returns NaN if either column is missing or non-finite.
    """
    row = message_df.iloc[idx]
    mid = float(row.get("MidPrice", np.nan))
    ret = float(row.get("Return", np.nan))
    if np.isfinite(mid) and np.isfinite(ret):
        return mid - ret
    return np.nan


def _get_pre_event_book(ob_df: pd.DataFrame, idx: int) -> Optional[pd.Series]:
    """Return the PRE-event order-book snapshot for event *idx*.

    Because the simulator stores the **post**-event snapshot at ``ob_df.iloc[i]``,
    the pre-event state for event i is ``ob_df.iloc[i - 1]`` (valid for i >= 1).

    Returns None if ob_df is empty or the index is out of range.
    """
    if ob_df is None or ob_df.empty:
        return None
    if idx - 1 < 0 or idx - 1 >= len(ob_df):
        return None
    return ob_df.iloc[idx - 1]


def _extract_side_arrays(book_pre: pd.Series, direction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sorted (prices, sizes) arrays for the side of the book hit by an MO.

    A BUY market order (direction > 0) consumes ask liquidity, so we extract
    AskPrice_* and AskSize_* columns, sorted by level number.
    A SELL market order (direction < 0) consumes bid liquidity analogously.

    Returns
    -------
    (prices, sizes) : (np.ndarray, np.ndarray)
        Both 1D float arrays, sorted by price level from best outward.
    """
    if direction > 0:
        price_cols = [c for c in book_pre.index if c.startswith("AskPrice_")]
        size_cols  = [c for c in book_pre.index if c.startswith("AskSize_")]
    else:
        price_cols = [c for c in book_pre.index if c.startswith("BidPrice_")]
        size_cols  = [c for c in book_pre.index if c.startswith("BidSize_")]

    def _lvl(col: str) -> int:
        return int(col.split("_")[1])

    price_cols = sorted(price_cols, key=_lvl)
    size_cols  = sorted(size_cols, key=_lvl)

    prices = book_pre[price_cols].to_numpy(dtype=float)
    sizes  = book_pre[size_cols].to_numpy(dtype=float)

    return prices, sizes


def _maybe_collapse_split_sweeps(
    msg: pd.DataFrame,
    book: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    """Collapse split-sweep log rows back to one-row-per-event alignment.

    When the simulator runs with ``split_sweeps=True``, market-order sweep
    events produce *multiple* rows in ``message_df`` — one "summary" row with a
    finite event index ``i`` and additional "detail" rows where ``i`` is NaN.
    These detail rows carry per-level fill information but break the fundamental
    row-alignment assumption used everywhere in this file:

        book_pre(event_i) = ob_df.iloc[i - 1]

    This helper restores 1-row-per-event alignment by keeping only the rows
    whose ``i`` column is finite.

    Alignment of ``book`` (ob_df)
    -----------------------------
    * If ``book`` has the **same length** as ``msg``, both DataFrames are
      filtered with the same boolean mask (they share the split-sweep layout).
    * If ``book`` has a **different length**, it is assumed to already be
      event-aligned (e.g. it was recorded without sweep detail) and is left
      untouched apart from a ``reset_index``.

    Returns
    -------
    (msg_filtered, book_filtered, did_collapse : bool)
        ``did_collapse`` is True if any rows were actually removed.
    """
    if msg is None or getattr(msg, "empty", True):
        return msg, book, False
    if "i" not in msg.columns:
        return msg, book, False

    i_col = pd.to_numeric(msg["i"], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isfinite(i_col)):
        return msg, book, False

    mask = np.isfinite(i_col)
    msg_ev = msg.loc[mask].reset_index(drop=True)

    if book is None or getattr(book, "empty", True):
        return msg_ev, book, True

    if len(book) == len(msg):
        book_ev = book.loc[mask].reset_index(drop=True)
    else:
        book_ev = book.reset_index(drop=True)

    return msg_ev, book_ev, True


def _event_level_from_price(
    msg: pd.DataFrame,
    idx: int,
    half_tick: float,
    max_levels: int,
) -> float:
    """Map a non-MO event's Price to a depth-level index k relative to the pre-event mid.

    For limit orders and cancellations we want to know *which price level* was
    affected so we can attribute this event to a depth bucket.  The formula is:

        depth = |Price - mid_pre|
        k     = round(depth / half_tick) - 1

    Returns ``NaN`` if the price, mid, or resulting depth is invalid (e.g. the
    event has no Price column, the pre-event mid is not computable, or the
    depth maps to a negative tick index).

    Parameters
    ----------
    msg : pd.DataFrame
        The message log (must contain ``Price`` and be compatible with
        ``_get_pre_event_mid_from_message``).
    idx : int
        Row index inside *msg*.
    half_tick : float
        Tick resolution (typically 0.5 in this codebase).
    max_levels : int
        Maximum number of depth levels tracked.  Indices >= max_levels are
        clamped to ``max_levels - 1``.
    """
    row = msg.iloc[idx]
    px = float(row.get("Price", np.nan))
    if not np.isfinite(px):
        return np.nan

    mid_pre = _get_pre_event_mid_from_message(msg, idx)
    if not np.isfinite(mid_pre):
        return np.nan

    depth = abs(px - mid_pre)
    if not (np.isfinite(depth) and depth > 0.0):
        return np.nan

    k = depth_to_tick_index(depth, half_tick)
    if k < 0:
        return np.nan
    if k >= max_levels:
        k = max_levels - 1
    return float(k)


# =========================================================================
# SECTION 2 — AGGREGATION / BUCKETING ("DECISION CLOCKS")
# =========================================================================
# In the GLFT framework the market maker reprices on a "decision clock" that
# may be faster or slower than the raw event stream.  These utilities assign
# each simulator event to a bucket ID so that downstream functions can
# aggregate (e.g. take the max MO depth per bucket) and normalize correctly.
#
# Supported clocks:
#   "event"        — finest grain: bucket_id = event index (no grouping)
#   "tob"          — group by N top-of-book changes
#   "time"         — group by T seconds of wall-clock time
#   "fixed_events" — group by N consecutive simulator steps
# =========================================================================

def _extract_tob_key_from_book_row(book_row: pd.Series) -> Tuple[float, float, float, float]:
    """Return a 4-tuple (best_bid, best_ask, bid_size_1, ask_size_1) that
    uniquely identifies the top-of-book state.  A change in this key signals
    a TOB move for the ``tob`` aggregation clock."""
    best_bid = float(book_row["BidPrice_1"])
    best_ask = float(book_row["AskPrice_1"])
    bidsize = float(book_row["BidSize_1"])
    asksize = float(book_row["AskSize_1"])
    return (best_bid, best_ask, bidsize, asksize)

def build_aggregation_buckets(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0
) -> Tuple[Optional[np.ndarray], int]:
    """Assign each event in *message_df* to a bucket ID based on the chosen clock.

    This is the central bucketing function used by ALL downstream intensity,
    volatility, and probability estimators.  It converts the raw event stream
    into a coarser "decision clock" that matches how the market maker reprices.

    Modes
    -----
    "event"        No grouping.  Returns ``(None, n_events)``.
                   Each simulator step is its own bucket.
    "tob"          Groups events into buckets of *n_tob_moves* top-of-book
                   changes.  A "TOB change" is detected when the 4-tuple
                   (best_bid, best_ask, bid_size_1, ask_size_1) differs from
                   the previous step.
    "time"         Groups events into buckets of *time_interval* seconds
                   using the ``Time`` column in *message_df*.
    "fixed_events" Groups events into blocks of *n_events_interval*
                   consecutive steps.  Simple and fast, but ignores market
                   microstructure.

    Returns
    -------
    (bucket_ids, n_buckets) : (Optional[np.ndarray], int)
        ``bucket_ids`` is None only in ``"event"`` mode.  Otherwise it is an
        int64 array of length ``len(message_df)`` mapping each row to its
        bucket.  ``n_buckets`` is the total number of buckets (including those
        that may contain zero MO events).

    Raises
    ------
    ValueError
        If *mode* is not one of the four supported strings.
    """
    n = len(message_df)
    
    if mode == "event":
        return None, n
    
    bucket_ids = np.zeros(n, dtype=np.int64)
    
    # --- 1. Fixed Events Aggregation ---
    if mode == "fixed_events":
        step = max(1, int(n_events_interval))
        bucket_ids = np.arange(n) // step
        return bucket_ids, bucket_ids[-1] + 1
    
    # --- 2. Time Aggregation ---
    if mode == "time":
        if "Time" not in message_df.columns:
            # Fallback if no Time column
            return None, n
        
        t = message_df["Time"].to_numpy(dtype=np.float64)
        if len(t) == 0: return np.zeros(0, dtype=np.int64), 0
        
        t_start = t[0]
        dt = float(max(1e-9, time_interval))
        bucket_ids = np.floor((t - t_start) / dt).astype(np.int64)
        return bucket_ids, bucket_ids[-1] + 1

    # --- 3. TOB Moves Aggregation (Legacy Logic) ---
    if mode == "tob":
        threshold = max(1, int(n_tob_moves))
        tob_moves = 0
        last_key = None

        for i in range(n):
            book_pre = _get_pre_event_book(ob_df, i)
            if book_pre is None:
                bucket_ids[i] = 0
                continue

            key = _extract_tob_key_from_book_row(book_pre)

            if last_key is None:
                last_key = key
            else:
                if key != last_key:
                    tob_moves += 1
                    last_key = key

            bucket_ids[i] = tob_moves // threshold

        return bucket_ids, bucket_ids[-1] + 1
    
    # No valid mode matched — raise instead of silently returning defaults
    raise ValueError(
        f"Unknown aggregation_mode: '{mode}'. "
        "Supported modes: 'event', 'tob', 'time', 'fixed_events'."
    )


# =========================================================================
# SECTION 3 — NON-QUEUE-AWARE DEPTH SWEEP AND TAIL COUNTS
# =========================================================================
# For each market order (MO) event, we reconstruct the sweep on the
# PRE-event book to find the *deepest* price level the MO reaches.
# This produces a single number D per MO (the maximum tick index touched).
#
# Why reconstruct?  The simulator's message_df["Price"] often records the
# *first* fill price, not the deepest.  Reconstructing from the book
# gives the true sweep depth, making this metric directly comparable
# with queue-aware rank 0.
# =========================================================================

def compute_nonqueue_deepest_tick_per_mo(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    max_depth_levels: int,
    half_tick: float,
) -> Tuple[np.ndarray, int]:
    """For each MO event, reconstruct the sweep on the PRE-event book and
    return the deepest depth index D reached (as an integer tick index).

    Algorithm (per MO event i):
        1. Look up the pre-event book: book_pre = ob_df.iloc[i-1]
        2. Determine the pre-event mid from message_df
        3. Extract the relevant side (asks for BUY MO, bids for SELL MO)
        4. Walk through price levels from best outward, consuming volume:
           - For each level j, compute depth_j = |price_j - mid_pre|
           - Convert to tick index: tick_j = round(depth_j / half_tick) - 1
           - Track the maximum tick_j across all levels touched
        5. Record D[i] = max tick reached (NaN if the MO is invalid)

    Returns
    -------
    (D, n_used) : (np.ndarray, int)
        D has length len(message_df); non-MO rows and invalid MOs are NaN.
        n_used is the count of MOs with a valid (finite) D.
    """
    n = len(message_df)
    D = np.full(n, np.nan, dtype=float)
    n_used = 0

    for i in range(n):
        if not _is_mo_row(message_df, i):
            continue

        row = message_df.iloc[i]
        direction = float(row.get("Direction", np.nan))
        size_exec = float(row.get("Size", np.nan))

        if not np.isfinite(direction) or direction == 0.0:
            continue
        if not np.isfinite(size_exec) or size_exec <= 0.0:
            continue

        book_pre = _get_pre_event_book(ob_df, i)
        if book_pre is None:
            continue

        mid_pre = _get_pre_event_mid_from_message(message_df, i)
        if not np.isfinite(mid_pre):
            continue

        prices, sizes = _extract_side_arrays(book_pre, direction)

        remaining = int(round(size_exec))
        if remaining <= 0:
            continue

        deepest_tick = -np.inf

        # Sweep the book level-by-level from best price outward,
        # consuming volume until the MO size is exhausted.
        for j in range(len(prices)):
            if remaining <= 0:
                break

            level_vol = float(sizes[j])
            price_j = float(prices[j])

            if level_vol <= 0.0:
                break  # stop at padding / empty levels

            take = min(remaining, int(round(level_vol)))
            if take <= 0:
                continue

            depth_j = (price_j - mid_pre) if direction > 0 else (mid_pre - price_j)
            if depth_j <= 0.0 or not np.isfinite(depth_j):
                # If this happens often, your mid/book alignment is off.
                # We skip this MO for consistency (same rule for queue-aware).
                deepest_tick = -np.inf
                break

            tick_j = depth_to_tick_index(depth_j, half_tick)
            if tick_j < 0:
                # Same exclusion logic as hftbacktest ("trade printed through previous mid").
                deepest_tick = -np.inf
                break

            if tick_j >= max_depth_levels:
                tick_j = max_depth_levels - 1

            deepest_tick = max(deepest_tick, float(tick_j))
            remaining -= take

        if np.isfinite(deepest_tick):
            D[i] = deepest_tick
            n_used += 1

    return D, n_used


def measure_tailcounts_from_tick_pmf(pmf_counts: np.ndarray, strict: bool) -> np.ndarray:
    """Convert a point-mass histogram pmf[k] into a tail-count array.

    Two conventions are supported (matching hftbacktest):
        strict=False :  tail[k] = sum_{j >= k} pmf[j]   ("at or beyond k")
        strict=True  :  tail[k] = sum_{j >  k} pmf[j]   ("strictly beyond k")

    The strict=True convention is used in the hftbacktest reference and in
    most GLFT calibration papers.
    """
    tail_ge = np.cumsum(pmf_counts[::-1])[::-1]  # >=
    if not strict:
        return tail_ge

    tail = np.zeros_like(tail_ge)
    tail[:-1] = tail_ge[1:]  # >k  == >=(k+1)
    tail[-1] = 0
    return tail


def _normalize_tailcounts_to_intensity(
    tail_counts: np.ndarray,
    n_buckets_valid: int,
) -> np.ndarray:
    """Convert raw tail COUNTS into a *trading intensity* (rate per decision step).

    intensity[k] = tail_counts[k] / denominator

    The denominator (total number of decision steps — events or buckets) is
    passed explicitly because different aggregation modes require different
    normalization.  A minimum of 1 is enforced to prevent division by zero.
    """
    denom = float(max(1, int(n_buckets_valid)))
    return tail_counts.astype(np.float64) / denom


def compute_trading_intensity_tailcounts_from_message_df(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    max_levels: int = 200,
    half_tick: float = 0.5,
    strict: bool = True,
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,
    output: str = "counts",
) -> Tuple[np.ndarray, int]:
    """Compute the NON-queue-aware tail-count (or intensity) curve.

    This is the main entry point for obtaining lambda(delta) data suitable
    for fitting the GLFT exponential model.  Steps:
        1. Compute deepest tick D per MO via ``compute_nonqueue_deepest_tick_per_mo``.
        2. Optionally aggregate into buckets (max depth per bucket).
        3. Build a point-mass histogram (PMF) over tick indices.
        4. Convert to tail counts (cumulative sum from the right).
        5. Optionally normalize to intensity (counts / denominator).

    Parameters
    ----------
    output : str
        "counts" (default) returns raw tail counts.
        "intensity" divides by the total number of decision steps.
    use_tob_buckets : bool
        Legacy flag: if True and aggregation_mode="event", forces "tob" mode.

    Returns
    -------
    (tail, n_events_used) : (np.ndarray, int)
        tail is trimmed to the last non-zero entry.
    """
    # Legacy compat
    if use_tob_buckets and aggregation_mode == "event":
        aggregation_mode = "tob"

    D_by_event, _n = compute_nonqueue_deepest_tick_per_mo(
        message_df=message_df,
        ob_df=ob_df,
        max_depth_levels=max_levels,
        half_tick=half_tick,
    )

    # Use the new builder
    bucket_ids, n_buckets_total = build_aggregation_buckets(
        message_df, ob_df, mode=aggregation_mode,
        n_tob_moves=n_tob_moves, n_events_interval=n_events_interval, time_interval=time_interval
    )

    # Build tick list (per-MO or per-bucket max)
    if bucket_ids is None:
        # Event mode
        ticks = D_by_event[np.isfinite(D_by_event)].astype(int)
        n_events_used = int(ticks.size)
        n_denom = int(len(message_df)) # Denom is total events for normalization
    else:
        # Bucket mode (max depth per bucket)
        if bucket_ids.size != len(D_by_event):
            raise ValueError("Bucket mode requires bucket_ids aligned with message_df.")
        
        max_tick_by_bucket: Dict[int, int] = {}
        for i, dt in enumerate(D_by_event):
            if not np.isfinite(dt):
                continue
            b = int(bucket_ids[i])
            t = int(dt)
            prev = max_tick_by_bucket.get(b, -10**18)
            if t > prev:
                max_tick_by_bucket[b] = t
        
        ticks = np.asarray(list(max_tick_by_bucket.values()), dtype=int)
        n_events_used = int(ticks.size) # Valid buckets that had a trade
        n_denom = int(n_buckets_total)  # Total buckets for normalization

    pmf = np.zeros(max_levels, dtype=np.int64)
    for t in ticks:
        if 0 <= t < max_levels:
            pmf[t] += 1

    tail = measure_tailcounts_from_tick_pmf(pmf, strict=strict)

    nz = np.where(tail > 0)[0]
    if nz.size == 0:
        return np.array([], dtype=np.int64), 0

    tail = tail[: int(nz.max()) + 1]

    # Optional normalization: convert COUNTS into an intensity.
    if str(output).lower() == "intensity":
        tail = _normalize_tailcounts_to_intensity(tail, n_denom)

    return tail, n_events_used


def plot_trading_intensity_tailcounts(
    tail_counts: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    ylabel: str = "Count (tail in k)",
    title: str = "Non-queue-aware tail COUNTS",
    *,
    half_tick: float = 0.5,
    x_mode: str = "k",          # "k" or "delta"
    style: str = "step",        # "step" or "line"
) -> plt.Axes:
    """Plot non-queue-aware tail counts (or intensity) vs depth.

    Supports two x-axis modes:
        "k"     — raw depth index (0, 1, 2, ...)
        "delta" — delta in ticks from mid: delta_k = (k + 1) * half_tick

    And two drawing styles:
        "step"  — staircase plot (traditional for histograms)
        "line"  — smooth line (better for fitted curves)
    """
    if ax is None:
        _, ax = plt.subplots()

    k = np.arange(len(tail_counts), dtype=float)

    # ------------------------------------------------------------
    # X-axis: either k (legacy) or delta in ticks from mid (fit-like)
    # ------------------------------------------------------------
    if x_mode == "k":
        x = k
        xlabel = "Depth index from mid (k)"
    elif x_mode == "delta":
        # IMPORTANT: your convention in hftbacktest-style mapping:
        #   tick = round(depth / half_tick) - 1
        # so k=0 corresponds to delta=half_tick, k=1 -> 2*half_tick, etc.
        x = (k + 1.0) * float(half_tick)
        xlabel = r"$\delta$ (ticks from the mid-price)"
    else:
        raise ValueError("x_mode must be 'k' or 'delta'")

    # ------------------------------------------------------------
    # Style: step (legacy) or line (fit-like)
    # ------------------------------------------------------------
    if style == "step":
        ax.step(x, tail_counts, where="post")
    elif style == "line":
        ax.plot(x, tail_counts)
    else:
        raise ValueError("style must be 'step' or 'line'")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show:
        plt.show()
    return ax



# =========================================================================
# SECTION 4 — QUEUE-AWARE DEPTH SWEEP AND TAIL COUNTS
# =========================================================================
# Extends Section 3 by tracking not just *how deep* an MO went, but also
# *which queue ranks* were executed at each depth.  This produces a 2D
# array: executability[k, r] = count of events where rank r was executed
# at depth k.  From this we derive queue-aware tail-count curves per rank.
#
# Key insight: at each price level the MO consumes ranks 0, 1, ..., take-1.
# So D_r (deepest depth where rank r was executed) is monotonically
# decreasing in r.  This guarantees the tail curves are properly nested:
# tail[:, 0] >= tail[:, 1] >= tail[:, 2] >= ...
# =========================================================================

def compute_executability_pointmass_middepth(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    max_depth_levels: int = 200,
    max_ranks: int = 10,
    half_tick: float = 0.5,
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,
) -> Tuple[np.ndarray, int]:
    """Compute the queue-aware point-mass histogram executability[k, r].

    For each MO event, the sweep on the pre-event book determines, for every
    rank r in [0, max_ranks), the deepest tick index at which rank r was
    executed.  This value is recorded in ``executability[tick, r]``.

    In bucket mode, multiple MOs inside a bucket are summarized as a single
    observation: the deepest tick per rank across all MOs in that bucket.

    Returns
    -------
    (executability, n_events_used) : (np.ndarray[K, R], int)
        executability has shape (max_depth_levels, max_ranks).
        n_events_used is the number of valid observations (MO events in
        event mode, or total buckets in bucket mode).
    """
    executability = np.zeros((max_depth_levels, max_ranks), dtype=np.int64)

    if message_df is None or ob_df is None or message_df.empty or ob_df.empty:
        return executability, 0

    # Legacy compat
    if use_tob_buckets and aggregation_mode == "event":
        aggregation_mode = "tob"

    bucket_ids, n_buckets_total = build_aggregation_buckets(
        message_df, ob_df, mode=aggregation_mode,
        n_tob_moves=n_tob_moves, n_events_interval=n_events_interval, time_interval=time_interval
    )

    # bucket accumulator: bucket_id -> per-rank deepest tick in that bucket
    bucket_best: Dict[int, np.ndarray] = {}

    n_events_used = 0

    for i in range(len(message_df)):
        if not _is_mo_row(message_df, i):
            continue

        row = message_df.iloc[i]
        direction = float(row.get("Direction", np.nan))
        size_exec = float(row.get("Size", np.nan))

        if not np.isfinite(direction) or direction == 0.0:
            continue
        if not np.isfinite(size_exec) or size_exec <= 0.0:
            continue

        book_pre = _get_pre_event_book(ob_df, i)
        if book_pre is None:
            continue

        mid_pre = _get_pre_event_mid_from_message(message_df, i)
        if not np.isfinite(mid_pre):
            continue

        prices, sizes = _extract_side_arrays(book_pre, direction)

        remaining = int(round(size_exec))
        if remaining <= 0:
            continue

        # D_r for this event: deepest tick index where rank r was executed.
        # Initialize to -inf (meaning "rank r was never executed").
        max_tick_for_rank = np.full(max_ranks, fill_value=-np.inf, dtype=float)

        bad = False

        for j in range(len(prices)):
            if remaining <= 0:
                break

            level_vol = float(sizes[j])
            price_j = float(prices[j])

            if level_vol <= 0.0:
                break  # stop at empty/padding levels

            take = min(remaining, int(round(level_vol)))
            if take <= 0:
                continue

            depth_j = (price_j - mid_pre) if direction > 0 else (mid_pre - price_j)
            if depth_j <= 0.0 or not np.isfinite(depth_j):
                bad = True
                break

            tick_j = depth_to_tick_index(depth_j, half_tick)
            if tick_j < 0:
                bad = True
                break

            if tick_j >= max_depth_levels:
                tick_j = max_depth_levels - 1

            # KEY INSIGHT: rank is the FIFO position within this price level.
            # The MO consumes 'take' units, executing ranks 0, 1, ..., take-1.
            # For each executed rank r, update D_r = max(D_r, tick_j).
            n_rank_exec_here = min(int(take), max_ranks)
            for r in range(n_rank_exec_here):
                if float(tick_j) > max_tick_for_rank[r]:
                    max_tick_for_rank[r] = float(tick_j)

            remaining -= take

        if bad:
            continue

        # Event contributes if rank 0 executed at least somewhere.
        if not np.isfinite(max_tick_for_rank[0]):
            continue

        if bucket_ids is None:
            # Event Mode: accumulate directly
            for r in range(max_ranks):
                t = max_tick_for_rank[r]
                if np.isfinite(t):
                    executability[int(t), r] += 1
            n_events_used += 1
        else:
            # Bucket Mode: store max depth per rank for this bucket
            b = int(bucket_ids[i])
            if b not in bucket_best:
                bucket_best[b] = np.full(max_ranks, fill_value=-np.inf, dtype=float)
            for r in range(max_ranks):
                t = max_tick_for_rank[r]
                if np.isfinite(t) and t > bucket_best[b][r]:
                    bucket_best[b][r] = t

    if bucket_ids is not None:
        n_events_used = n_buckets_total
        for _, arr in bucket_best.items():
            any_rank = False
            for r in range(max_ranks):
                t = arr[r]
                if np.isfinite(t):
                    executability[int(t), r] += 1
                    any_rank = True
    return executability, n_events_used


def compute_executability_tailcounts_middepth(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    max_depth_levels: int = 200,
    max_ranks: int = 10,
    half_tick: float = 0.5,
    strict: bool = True,
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,
    output: str = "counts",
) -> Tuple[np.ndarray, int]:
    """Convert queue-aware point-mass counts into tail curves per rank.

    Wraps ``compute_executability_pointmass_middepth`` and then applies the
    same strict/non-strict tail logic as the non-queue-aware version, but
    independently for each rank column.

    Returns
    -------
    (tail, n_denom) : (np.ndarray[K, R], int)
        2D tail counts (or intensities if output="intensity").
    """
    pmf, n_denom = compute_executability_pointmass_middepth(
        message_df=message_df,
        ob_df=ob_df,
        max_depth_levels=max_depth_levels,
        max_ranks=max_ranks,
        half_tick=half_tick,
        aggregation_mode=aggregation_mode,
        n_tob_moves=n_tob_moves,
        n_events_interval=n_events_interval,
        time_interval=time_interval,
        use_tob_buckets=use_tob_buckets
    )

    if n_denom == 0:
        return np.zeros_like(pmf, dtype=np.int64), 0

    tail_ge = np.cumsum(pmf[::-1, :], axis=0)[::-1, :].astype(np.int64)

    # Build the requested tail (>= or >) in COUNT space first.
    if not strict:
        tail = tail_ge
    else:
        tail = np.zeros_like(tail_ge)
        tail[:-1, :] = tail_ge[1:, :]
        tail[-1, :] = 0

    # Optional normalization: convert COUNTS into an intensity.
    if str(output).lower() == "intensity":
        tail = _normalize_tailcounts_to_intensity(tail, n_denom)

    return tail, n_denom

def plot_executability_tailcounts_by_rank_middepth(
    tail_counts: np.ndarray,
    ranks_to_plot=(0, 1, 2, 3, 4),
    ax=None,
    show: bool = True,
    ylabel: str = "Count (tail in k)",
    title: str = "Queue-aware tail COUNTS by rank",
    *,
    # --- Plotting options (match non-queue-aware API) ---
    half_tick: float = 0.5,
    x_mode: str = "k",        # "k" or "delta"
    style: str = "step",      # "step" or "line"
    truncate_zeros: bool = True,  # cut trailing all-zero tail to match non-queue behavior
):
    """Plot queue-aware tail-count curves, one line per rank.

    Each rank r has its own tail curve (tail[:, r]), and by construction
    rank 0 dominates all higher ranks.  The plot helps visually confirm
    this monotonicity and assess how quickly execution probability decays
    for deeper queue positions.
    """
    if ax is None:
        _, ax = plt.subplots()

    if tail_counts.ndim != 2:
        raise ValueError("tail_counts must be a 2D array of shape (K, R)")

    K, R = tail_counts.shape

    # Optionally trim trailing all-zero rows (common reason your x-axis looks longer)
    if truncate_zeros and K > 0:
        nz = np.where(np.any(tail_counts > 0, axis=1))[0]
        if nz.size == 0:
            K_eff = 0
        else:
            K_eff = int(nz.max()) + 1
        tail_counts = tail_counts[:K_eff, :]
        K = K_eff

    k = np.arange(K, dtype=float)

    # X-axis mapping: k -> delta
    if x_mode == "k":
        x = k
        xlabel = "Depth index from mid (k)"
    elif x_mode == "delta":
        x = (k + 1.0) * float(half_tick)
        xlabel = r"$\delta$ (ticks from the mid-price)"
    else:
        raise ValueError("x_mode must be 'k' or 'delta'")

    # Plot each rank
    for r in ranks_to_plot:
        if 0 <= int(r) < R:
            y = tail_counts[:, int(r)]
            if style == "step":
                ax.step(x, y, where="post", label=f"queue rank {int(r)}")
            elif style == "line":
                ax.plot(x, y, label=f"queue rank {int(r)}")
            else:
                raise ValueError("style must be 'step' or 'line'")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()

    return ax


# =========================================================================
# SECTION 5 — SANITY CHECKS
# =========================================================================

def sanity_check_rank_monotonicity(tail_counts: np.ndarray) -> Tuple[bool, Tuple[int, int, int]]:
    """Verify the rank-monotonicity invariant: tail[k, 0] >= tail[k, 1] >= ...

    This invariant must hold by construction (rank 0 is always executed if any
    deeper rank is).  A violation signals a bug in the sweep logic.

    Returns
    -------
    (ok, (k, r, diff)) : (bool, tuple)
        If ok is True, no violation was found.  Otherwise (k, r, diff) gives
        the first depth index k where tail[k, r] < tail[k, r+1], and the
        magnitude of the violation.
    """
    K, R = tail_counts.shape
    for k in range(K):
        for r in range(R - 1):
            if tail_counts[k, r] < tail_counts[k, r + 1]:
                return False, (k, r, int(tail_counts[k, r + 1] - tail_counts[k, r]))
    return True, (-1, -1, 0)


# =========================================================================
# SECTION 6 — CALIBRATION WRAPPERS (one-call simulator + analysis)
# =========================================================================
# These "convenience" functions run the Santa-Fe-style simulator and
# immediately compute tail counts / intensity curves.  They are useful for
# interactive exploration (notebooks, scripts) but should NOT be called
# from tight loops — run the simulator once and pass the DataFrames to the
# lower-level functions instead.
# =========================================================================

def _run_simulator_santa_fe(
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    n_steps: int,
    n_steps_to_equilibrium: int,
    mm_policy,
    exclude_self_from_state: bool,
    mo_size: int = 1,
    split_sweeps: bool = False,
    random_seed: Optional[int] = None,
):
    """Run the Santa-Fe-compatible simulator with robust kwargs + deterministic seeding.

    This wrapper inspects the signature of ``simulate_LOB_with_MM`` and only
    passes keyword arguments that the function actually accepts.  This makes
    the caller resilient to API changes in the simulator (e.g. if a parameter
    is added or removed, this wrapper simply omits unsupported kwargs).
    """
    _seed_everything(random_seed)

    all_kwargs = dict(
        lam=lam,
        mu=mu,
        delta=delta,
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,

        # Different naming conventions used over time:
        iterations=n_steps,
        iterations_to_equilibrium=n_steps_to_equilibrium,
        n_steps=n_steps,
        n_steps_to_equilibrium=n_steps_to_equilibrium,

        mm_policy=mm_policy,
        exclude_self_from_state=exclude_self_from_state,

        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        mean_size_MO=mo_size,
        split_sweeps=split_sweeps,

        # Newer versions may accept this:
        random_seed=random_seed,
    )

    sig = inspect.signature(simulate_LOB_with_MM)
    supported = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    return simulate_LOB_with_MM(**supported)


def calibrate_trading_intensity_santa_fe_tailcounts(
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    n_steps: int,
    n_steps_to_equilibrium: int,
    random_seed: Optional[int] = None,
    mm_policy=None,
    exclude_self_from_state: bool = True,
    max_levels: int = 200,
    half_tick: float = 0.5,
    strict: bool = True,
    # --- Aggregation / Bucketing ---
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,
    plot: bool = True,
    output: str = "counts",
    mo_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    """
    Run the simulator + compute NON-queue-aware tail curve.
    """
    msg_df, ob_df, _mm_df = _run_simulator_santa_fe(
        lam=lam,
        mu=mu,
        delta=delta,
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,
        n_steps=n_steps,
        n_steps_to_equilibrium=n_steps_to_equilibrium,
        mm_policy=mm_policy,
        exclude_self_from_state=exclude_self_from_state,
        mo_size=mo_size,
        random_seed=random_seed,
    )

    depths_by_event, _n_used = compute_nonqueue_deepest_tick_per_mo(
        message_df=msg_df,
        ob_df=ob_df,
        max_depth_levels=max_levels,
        half_tick=half_tick,
    )

    tail_counts, n_events_used = compute_trading_intensity_tailcounts_from_message_df(
        message_df=msg_df,
        ob_df=ob_df,
        max_levels=max_levels,
        half_tick=half_tick,
        strict=strict,
        aggregation_mode=aggregation_mode,
        n_tob_moves=n_tob_moves,
        n_events_interval=n_events_interval,
        time_interval=time_interval,
        use_tob_buckets=use_tob_buckets,
        output=output,
    )

    if plot and tail_counts.size > 0:
        mode_label = aggregation_mode if not use_tob_buckets else "tob"
        ylabel = f"Trading intensity ({'per bucket/time' if mode_label!='event' else 'per step'})"

        plot_trading_intensity_tailcounts(
            tail_counts,
            ylabel=ylabel,
            title="Non-queue-aware trading intensity",
            half_tick=half_tick,
            x_mode="delta",
            style="line",
        )

    return msg_df, ob_df, depths_by_event, tail_counts, n_events_used


def calibrate_executability_santa_fe_middepth_tailcounts(
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    n_steps: int,
    n_steps_to_equilibrium: int,
    random_seed: Optional[int] = None,
    mm_policy=None,
    exclude_self_from_state: bool = True,
    max_depth_levels: int = 40,
    max_ranks: int = 10,
    half_tick: float = 0.5,
    strict: bool = True,
    # --- Aggregation / Bucketing ---
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,
    plot: bool = True,
    output: str = "counts",
    ranks_to_plot: Sequence[int] = (0, 1, 2),
    mo_size: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    """
    Run the simulator + compute QUEUE-aware tail curve by rank.
    """
    msg_df, ob_df, _mm_df = _run_simulator_santa_fe(
        lam=lam,
        mu=mu,
        delta=delta,
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,
        n_steps=n_steps,
        n_steps_to_equilibrium=n_steps_to_equilibrium,
        mm_policy=mm_policy,
        exclude_self_from_state=exclude_self_from_state,
        mo_size=mo_size,
        random_seed=random_seed,
    )

    tail_counts, n_events_used = compute_executability_tailcounts_middepth(
        message_df=msg_df,
        ob_df=ob_df,
        max_depth_levels=max_depth_levels,
        max_ranks=max_ranks,
        half_tick=half_tick,
        strict=strict,
        aggregation_mode=aggregation_mode,
        n_tob_moves=n_tob_moves,
        n_events_interval=n_events_interval,
        time_interval=time_interval,
        use_tob_buckets=use_tob_buckets,
        output=output,
    )

    if plot and tail_counts.size > 0:
        mode_label = aggregation_mode if not use_tob_buckets else "tob"
        if str(output).lower() == "intensity":
            ylabel = f"Executability intensity (per {mode_label})"
            title = f"Queue-aware executability intensity ({mode_label}, strict={strict})"
        else:
            ylabel = "Count (tail in k)"
            title = f"Queue-aware tail COUNTS ({mode_label}, strict={strict})"

        plot_executability_tailcounts_by_rank_middepth(
            tail_counts=tail_counts,
            ranks_to_plot=ranks_to_plot,
            ylabel=ylabel,
            title=title,
            half_tick=half_tick,
            x_mode="delta",
            style="line",
            truncate_zeros=True,
        )

    return msg_df, ob_df, tail_counts, n_events_used

# =========================================================================
# SECTION 7 — FIT A AND KAPPA: log(lambda) = -kappa * delta + log(A)
# =========================================================================
# The GLFT model assumes execution intensity decays exponentially with
# distance from the mid:
#
#     lambda(delta) = A * exp(-kappa * delta)
#
# Taking logs:
#     log(lambda) = -kappa * delta + log(A)
#
# This is a simple linear regression on (x = delta, y = log(lambda)).
# We offer both OLS (unweighted) and WLS (weighted) regression.  The
# default weight for WLS is the raw tail count at each depth: deeper
# levels have fewer observations and therefore noisier estimates, so
# tail-count weighting down-weights unreliable far-out points.
# =========================================================================

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares (OLS) linear regression: y = slope * x + intercept.

    Closed-form solution with no regularization or weighting.
    Identical in spirit to the hftbacktest calibration snippet.

    Raises ValueError if the regression is degenerate (all x identical).
    """
    sx = float(np.sum(x))
    sy = float(np.sum(y))
    sx2 = float(np.sum(x * x))
    sxy = float(np.sum(x * y))
    w = float(len(x))

    denom = (w * sx2 - sx * sx)
    if abs(denom) < 1e-30:
        raise ValueError("Degenerate regression: all x may be identical.")

    slope = (w * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / w
    return slope, intercept

def weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> Tuple[float, float]:
    """Weighted least squares (WLS) linear regression: y = slope * x + intercept.

    Minimizes:  sum_i  w_i * (y_i - slope * x_i - intercept)^2

    In the context of GLFT calibration, typical usage is:
        x = delta_grid          (distances from mid in ticks)
        y = log(lambda_curve)   (log of the empirical intensity)
        w = tail_counts         (raw counts — reliability proxy)

    Only points with finite x, y and positive w are used.  Returns (slope,
    intercept).  Raises ValueError on degenerate inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("x, y, w must have the same shape.")
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x, y, w must be 1D with length >= 2.")

    # Keep only valid points (finite x/y, positive weight)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    x = x[mask]
    y = y[mask]
    w = w[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid points after filtering.")
    sw = float(np.sum(w))
    if sw <= 0.0:
        raise ValueError("Sum of weights must be > 0.")

    # Weighted means
    xbar = float(np.sum(w * x) / sw)
    ybar = float(np.sum(w * y) / sw)

    # Weighted covariance/variance
    sxx = float(np.sum(w * (x - xbar) * (x - xbar)))
    if abs(sxx) < 1e-30:
        raise ValueError("Degenerate regression: all x may be identical (under weights).")
    sxy = float(np.sum(w * (x - xbar) * (y - ybar)))

    slope = sxy / sxx
    intercept = ybar - slope * xbar
    return slope, intercept

def fit_trading_intensity_parameters(
    *,
    # --- DATA SOURCE CONTROL ---
    data_mode: str = "simulation",  # "simulation" or "backtest"
    message_df: Optional[pd.DataFrame] = None,
    ob_df: Optional[pd.DataFrame] = None,

    # --- Aggregation / Bucketing ---
    queue_aware: bool,
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,

    # --- Intensity / Depth Params ---
    half_tick: float = 0.5,
    max_depth_levels: int = 200,
    strict: bool = True,
    
    # --- Simulation Params (Ignored if backtest) ---
    lam: float = 0.05, 
    mu: float = 0.05, 
    delta: float = 0.05,
    number_tick_levels: int = 100,
    n_priority_ranks: int = 20,
    n_steps: int = 100_000,
    n_steps_to_equilibrium: int = 2000,
    random_seed: Optional[int] = None,
    mm_policy = None,
    exclude_self_from_state: bool = True,
    mo_size: int = 1,

    # --- Regression / Fit Params ---
    weights_full: Optional[np.ndarray] = None,
    max_ranks: int = 5,
    rank_to_fit: int = 0,
    fit_start_k: int = 0,
    fit_end_k: Optional[int] = None,

    # --- Plotting ---
    plot: bool = True,
    plot_title_prefix: str = "",
) -> Dict[str, object]:
    """Fit the GLFT exponential intensity model: lambda(delta) = A * exp(-kappa * delta).

    This is the main calibration entry point.  It supports two data sources:

    data_mode="simulation"
        Runs the Santa-Fe simulator with the provided LOB parameters, then
        computes tail counts from the resulting event log.

    data_mode="backtest"
        Uses externally provided ``message_df`` and ``ob_df`` (e.g. from a
        real data replay or a separate simulation run).

    Pipeline
    --------
    1. Acquire data (simulate or accept external DataFrames).
    2. Compute raw tail counts (non-queue-aware or queue-aware at a chosen rank).
    3. Normalize counts into an intensity curve: lambda = counts / denominator.
    4. Fit log(lambda) = -kappa * delta + log(A) via WLS regression.
    5. Optionally plot the empirical curve and the fit.

    Returns
    -------
    dict with keys: "A", "kappa", "lambda_curve", "delta_grid",
    "n_events_used", "unit", "msg_df", "ob_df".
    """

    # -------------------------------------------------------------------------
    # 1. Acquire Data (Simulation or Direct Input)
    # -------------------------------------------------------------------------
    if data_mode == "simulation":
        # Run simulator
        msg_df, ob_df, _ = _run_simulator_santa_fe(
            lam=lam, mu=mu, delta=delta,
            number_tick_levels=number_tick_levels,
            n_priority_ranks=n_priority_ranks,
            n_steps=n_steps,
            n_steps_to_equilibrium=n_steps_to_equilibrium,
            mm_policy=mm_policy,
            exclude_self_from_state=exclude_self_from_state,
            mo_size=mo_size,
            random_seed=random_seed,
        )
        sim_desc = "Simulation"
    elif data_mode == "backtest":
        if message_df is None or ob_df is None:
            raise ValueError("For backtest mode, you must provide message_df and ob_df.")
        msg_df = message_df
        ob_df = ob_df
        sim_desc = "Backtest"
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    # -------------------------------------------------------------------------
    # 2. Compute Raw Counts (Shared Logic)
    # -------------------------------------------------------------------------
    if not queue_aware:
        # Non-queue aware counts
        D_by_event, _ = compute_nonqueue_deepest_tick_per_mo(
            message_df=msg_df, ob_df=ob_df,
            max_depth_levels=max_depth_levels, half_tick=half_tick
        )
        
        tail_counts, n_events_used = compute_trading_intensity_tailcounts_from_message_df(
            message_df=msg_df, ob_df=ob_df,
            max_levels=max_depth_levels, half_tick=half_tick, strict=strict,
            aggregation_mode=aggregation_mode, n_tob_moves=n_tob_moves,
            n_events_interval=n_events_interval, time_interval=time_interval,
            use_tob_buckets=use_tob_buckets, output="counts"
        )
        tail_counts = np.asarray(tail_counts, dtype=np.float64)
        rank_label = None
    else:
        # Queue-aware counts
        tail_counts_rank, n_events_used = compute_executability_tailcounts_middepth(
            message_df=msg_df, ob_df=ob_df,
            max_depth_levels=max_depth_levels, max_ranks=max_ranks,
            half_tick=half_tick, strict=strict,
            aggregation_mode=aggregation_mode, n_tob_moves=n_tob_moves,
            n_events_interval=n_events_interval, time_interval=time_interval,
            use_tob_buckets=use_tob_buckets, output="counts"
        )
        
        if rank_to_fit < 0 or rank_to_fit >= tail_counts_rank.shape[1]:
            raise ValueError(f"rank_to_fit={rank_to_fit} out of bounds.")
            
        tail_counts = np.asarray(tail_counts_rank[:, rank_to_fit], dtype=np.float64)
        rank_label = rank_to_fit

    if tail_counts.size < 3:
        raise ValueError("Tail-count curve is too short/empty to fit.")

    # -------------------------------------------------------------------------
    # 3. Normalize to Intensity (Lambda)
    # -------------------------------------------------------------------------
    # Determine denominator based on aggregation
    if aggregation_mode == "event" and not use_tob_buckets:
        denom = float(max(1, int(len(msg_df))))
        unit = "per step"
    else:
        denom = float(max(1, int(n_events_used)))
        unit = f"per {aggregation_mode}"

    lambda_curve = tail_counts / denom

    # -------------------------------------------------------------------------
    # 4. Fit Log-Linear Model: log(lambda) = -kappa * delta + log(A)
    # -------------------------------------------------------------------------
    K = int(lambda_curve.size)
    delta_grid = (np.arange(K, dtype=np.float64) + 1.0) * float(half_tick)

    if fit_end_k is None: fit_end_k = K
    fit_start_k = max(0, int(fit_start_k))
    fit_end_k = min(K, int(fit_end_k))
    fit_slice = slice(fit_start_k, fit_end_k)

    x = delta_grid[fit_slice]
    lam_hat = lambda_curve[fit_slice]

    # Weights: default to raw tail counts (variance proxy)
    if weights_full is None:
        w_full = tail_counts 
    else:
        w_full = np.asarray(weights_full, dtype=np.float64)
        if w_full.shape[0] > tail_counts.shape[0]:
            w_full = w_full[:tail_counts.shape[0]]

    w_hat = w_full[fit_slice]

    # Filter for valid log(y)
    mask = (np.isfinite(x) & (lam_hat > 0.0) & (w_hat > 0.0))
    x_fit, y_fit, w_fit = x[mask], np.log(lam_hat[mask]), w_hat[mask]

    if x_fit.size < 2:
        raise ValueError("Not enough valid points for regression.")

    slope, intercept = weighted_linear_regression(x_fit, y_fit, w_fit)
    kappa = -slope
    A = float(np.exp(intercept))
    fitted = A * np.exp(-kappa * delta_grid)

    # -------------------------------------------------------------------------
    # 5. Plotting
    # -------------------------------------------------------------------------
    if plot:
        algo_desc = _format_aggregation_label(aggregation_mode, time_interval, n_tob_moves, n_events_interval)
        prefix = plot_title_prefix if plot_title_prefix else f"{sim_desc}: "
        
        title = f"{prefix}{algo_desc} "
        if not queue_aware:
            title += "Fit λ(δ) | non-queue-aware"
        else:
            title += f"Fit λ(δ) | queue-aware rank {rank_label}"

        plt.figure()
        plt.plot(delta_grid, lambda_curve, label="Actual Data", alpha=0.8)
        plt.plot(delta_grid, fitted, 'r--', label=f"Fit (k={kappa:.3f})")
        plt.xlabel(r"$\delta$ (ticks)")
        plt.ylabel(f"Intensity ({unit})")
        plt.title(title + f"\nA={A:.4g}, k={kappa:.4g} | n_used={n_events_used}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        "A": A, "kappa": kappa, "lambda_curve": lambda_curve, 
        "delta_grid": delta_grid, "n_events_used": n_events_used, 
        "unit": unit, "msg_df": msg_df, "ob_df": ob_df
    }

# =========================================================================
# SECTION 8 — MID-PRICE SERIES BUILDER
# =========================================================================
# Volatility and signature functions need a continuous mid-price series in
# tick units.  This helper handles the two storage formats (MidPrice column
# vs cumulative Return) and applies the grid-recentering correction (Shift).
# =========================================================================

def _build_mid_series_ticks(msg_df: pd.DataFrame) -> np.ndarray:
    """Return a mid-price series in **tick units**.

    In this codebase, MidPrice is stored as a price-grid index (ticks), not dollars.
    That means any volatility computed from MidPrice/Return is automatically in ticks.

    Minimal fix for re-centering (Shift)
    -----------------------------------
    If your simulator re-centers the price grid, raw `MidPrice` can look artificially
    bounded (e.g. it stays near the grid center). When present, the column `Shift`
    stores the per-event grid shift applied by the simulator.

    For volatility / signature calculations we want an "absolute" mid coordinate in ticks
    (up to an additive constant):

        mid_abs[t] = mid_rel[t] + cumsum(Shift)[t]

    If `Shift` is absent we keep the legacy behavior (relative mid).
    If `MidPrice` is missing but `Return` exists, we reconstruct a mid series up to an
    additive constant (which cancels in differences anyway).
    """
    if msg_df is None or getattr(msg_df, "empty", True):
        return np.asarray([], dtype=np.float64)

    # --- 1) Build the *relative* mid series (legacy behavior) ---
    if "MidPrice" in msg_df.columns:
        mid_rel = msg_df["MidPrice"].to_numpy(dtype=np.float64)
    elif "Return" in msg_df.columns:
        ret = msg_df["Return"].to_numpy(dtype=np.float64)
        mid_rel = np.zeros(len(msg_df), dtype=np.float64)
        # In your simulator, Return[0] is typically a 0.0 placeholder.
        mid_rel[1:] = np.cumsum(np.nan_to_num(ret[1:], nan=0.0))
    else:
        raise ValueError('message_df must contain "MidPrice" or "Return" to build a mid series.')

    # --- 2) Minimal fix: if Shift exists, reconstruct an "absolute" tick coordinate ---
    if "Shift" in msg_df.columns:
        sh = msg_df["Shift"].to_numpy(dtype=np.float64)
        sh = np.nan_to_num(sh, nan=0.0)
        return mid_rel + np.cumsum(sh)

    return mid_rel


# =========================================================================
# SECTION 9 — FIT VOLATILITY (event / bucket / calendar clocks)
# =========================================================================
# The GLFT optimal spread formula includes sigma (mid-price volatility).
# This section estimates sigma under three different "clocks":
#
#   (A) Event clock    -> sigma_event   = std(delta_mid_per_event)
#                         Units: ticks / sqrt(event)
#
#   (B) Bucket clock   -> sigma_bucket  = std(delta_mid_per_bucket)
#                         Units: ticks / sqrt(bucket)
#                         Use this when the MM reprices every N TOB moves.
#
#   (C) Calendar time  -> sigma_second  = std(delta_mid / sqrt(delta_t))
#                         Units: ticks / sqrt(second)
#                         Useful for comparing with real-time targets.
#
# The function automatically selects the PRIMARY sigma based on the
# aggregation_mode (event clock if no bucketing, bucket clock otherwise).
# All three are always returned in the output dict for comparison.
# =========================================================================

def fit_volatility(
    *,
    # --- DATA SOURCE CONTROL ---------------------------------------------------
    # Identical dual-mode API as fit_trading_intensity_parameters.
    # "simulation" -> generate synthetic data via _run_simulator_santa_fe.
    # "backtest"   -> use externally provided message_df / ob_df (e.g. from
    #                 a real data replay, an L3 reconstruction, or another sim).
    data_mode: str = "simulation",       # "simulation" | "backtest"
    message_df: Optional[pd.DataFrame] = None,   # required when data_mode="backtest"
    ob_df_input: Optional[pd.DataFrame] = None,   # required when data_mode="backtest"
    collapse_split_sweeps: bool = True,  # collapse NaN detail rows when split_sweeps=True

    # --- Aggregation / Bucketing -----------------------------------------------
    aggregation_mode: str = "event",
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False, # legacy

    # ---------------------
    queue_aware: bool,

    # --- Simulator params (Santa-Fe) — ignored when data_mode="backtest" -------
    lam: float = 0.05,
    mu: float = 0.05,
    delta: float = 0.05,
    number_tick_levels: int = 100,
    n_priority_ranks: int = 20,
    n_steps: int = 100_000,
    n_steps_to_equilibrium: int = 2000,
    random_seed: Optional[int] = None,

    # --- MM policy (passed through to the simulator) ---------------------------
    mm_policy=None,
    exclude_self_from_state: bool = True,

    # --- Kept for API symmetry with intensity fit (not used for vol itself) -----
    half_tick: float = 0.5,
    max_depth_levels: int = 200,
    max_ranks: int = 5,
    rank_to_fit: int = 0,
    strict: bool = True,

    # --- MO size used in the simulator -----------------------------------------
    mo_size: int = 1,

    # --- Plot controls ---------------------------------------------------------
    plot: bool = True,
    plot_title_prefix: str = "",
) -> Dict[str, object]:
    """Estimate mid-price volatility under multiple clock conventions.

    This function supports two data sources, exactly like
    ``fit_trading_intensity_parameters``:

    data_mode="simulation"  (default)
        Runs the Santa-Fe LOB simulator with the provided parameters,
        producing a synthetic event log from which the mid-price path is
        extracted.  All simulation-specific keyword arguments (lam, mu,
        delta, …) are forwarded to ``_run_simulator_santa_fe``.

    data_mode="backtest"
        Accepts externally provided ``message_df`` and ``ob_df_input``
        DataFrames.  These may come from a real data replay, an L3
        reconstruction, or a separate simulation run.  The function
        validates that the required columns exist (at minimum "MidPrice"
        or "Return" for building the mid-price series).

        When the backtest DataFrames originate from a simulator run with
        ``split_sweeps=True``, each market-order sweep generates multiple
        rows — one per filled level — with NaN detail columns for the
        "continuation" rows.  Setting ``collapse_split_sweeps=True``
        (default) invokes ``_maybe_collapse_split_sweeps`` to restore the
        1-row-per-event alignment that the volatility estimators expect.

    Clock Conventions
    -----------------
    Volatility is always reported under three "clocks":

    (A) Event clock  (one tick per LOB event)
        sigma_event = std(Δmid per event)
        Unit: ticks / sqrt(event).
        Natural when the MM takes an action on every event.

    (B) Bucket clock  (one tick per aggregation bucket)
        sigma_bucket = std(Δmid per bucket)
        Unit: ticks / sqrt(bucket).
        Natural when the MM reprices every N TOB moves (or N events,
        or every T seconds) — matching the aggregation_mode.

    (C) Calendar time  (seconds)
        sigma_second = std(Δmid_i / sqrt(Δt_i))
        Unit: ticks / sqrt(second).
        Useful for comparing across instruments or with real-time targets.

    Selection Logic (PRIMARY output "volatility")
    ----------------------------------------------
    - If aggregation_mode == "event" and use_tob_buckets is False:
          PRIMARY = sigma_event   (ticks / sqrt(event))
    - Otherwise:
          PRIMARY = sigma_bucket  (ticks / sqrt(bucket))
          Falls back to sigma_event if bucketing produces too few points.

    All three estimates are always included in the returned dict so the
    caller can compare scales and cross-check.

    Parameters
    ----------
    data_mode : str
        "simulation" (default) or "backtest".
    message_df : pd.DataFrame or None
        Required when data_mode="backtest".  Must contain "MidPrice" or
        "Return" column for building the mid-price path.
    ob_df_input : pd.DataFrame or None
        Required when data_mode="backtest".  Orderbook snapshot DataFrame
        aligned row-by-row with message_df.
    collapse_split_sweeps : bool
        If True (default) and backtest DataFrames contain NaN detail rows
        from split_sweeps=True, collapse them to 1 row per event.
    aggregation_mode : str
        Bucketing mode: "event", "tob", "time", "fixed_events".
    queue_aware : bool
        Not used for volatility estimation itself, but kept for API
        symmetry with fit_trading_intensity_parameters.
    plot : bool
        If True, show a diagnostic plot of mid-price increments.

    Returns
    -------
    dict with keys:
      - "volatility", "volatility_unit", "clock_mode"  (PRIMARY choice)
      - "volatility_per_sqrt_event"                    (always)
      - "volatility_per_sqrt_second"                   (if Time column exists)
      - "volatility_per_sqrt_bucket"                   (if bucketing is active)
      - "volatility_per_sqrt_second_bucketed"          (if bucketing + Time)
      - "bucket_stats"                                 (diagnostics)
      - "data_mode"                                    (echo of input)
      - "msg_df", "ob_df"                              (raw DataFrames)
      - plus echo of config args
    """

    # =========================================================================
    # 1) DATA ACQUISITION — simulation or backtest
    # =========================================================================
    # This mirrors the dual-mode pattern of fit_trading_intensity_parameters.
    # In simulation mode we generate data from scratch; in backtest mode we
    # accept externally provided DataFrames and optionally collapse split
    # sweep rows.
    # =========================================================================

    if data_mode == "simulation":
        # -----------------------------------------------------------------
        # SIMULATION MODE: run the Santa-Fe LOB simulator.
        # All LOB parameters (lam, mu, delta, …) are forwarded.
        # split_sweeps is set to False because volatility estimation needs
        # exactly one row per event for consistent Δmid computation.
        # -----------------------------------------------------------------
        msg_df, ob_df, mm_df = _run_simulator_santa_fe(
            lam=lam,
            mu=mu,
            delta=delta,
            number_tick_levels=number_tick_levels,
            n_priority_ranks=n_priority_ranks,
            n_steps=n_steps,
            n_steps_to_equilibrium=n_steps_to_equilibrium,
            mm_policy=mm_policy,
            exclude_self_from_state=exclude_self_from_state,
            mo_size=mo_size,
            split_sweeps=False,
            random_seed=random_seed,
        )

        if msg_df is None or msg_df.empty:
            raise ValueError("Simulator returned empty message_df; cannot estimate volatility.")

    elif data_mode == "backtest":
        # -----------------------------------------------------------------
        # BACKTEST MODE: use externally provided DataFrames.
        #
        # Validation:
        #   1. Both message_df and ob_df_input must be provided.
        #   2. message_df must contain "MidPrice" or "Return" for building
        #      the mid-price series (_build_mid_series_ticks uses these).
        #   3. If collapse_split_sweeps is True, we invoke
        #      _maybe_collapse_split_sweeps to restore 1-row-per-event
        #      alignment (important when the data was logged with
        #      split_sweeps=True).
        # -----------------------------------------------------------------
        if message_df is None or ob_df_input is None:
            raise ValueError(
                "For data_mode='backtest', you must provide both "
                "message_df and ob_df_input DataFrames."
            )

        msg_df = message_df.copy()
        ob_df = ob_df_input.copy()

        # Validate that we can build a mid-price series from the data.
        if "MidPrice" not in msg_df.columns and "Return" not in msg_df.columns:
            raise ValueError(
                "Backtest message_df must contain a 'MidPrice' or 'Return' "
                "column to build the mid-price series for volatility estimation."
            )

        # Collapse split-sweep detail rows if requested.
        # When split_sweeps=True was used during data generation, each
        # market order that sweeps multiple levels produces N rows (one per
        # filled level).  The "continuation" rows have NaN in most columns.
        # _maybe_collapse_split_sweeps keeps only the summary row for each
        # event, restoring the 1-row-per-event alignment that our Δmid
        # computation requires.
        if collapse_split_sweeps:
            msg_df, ob_df, _ = _maybe_collapse_split_sweeps(msg_df, ob_df)

    else:
        raise ValueError(
            f"Unknown data_mode: '{data_mode}'.  "
            f"Expected 'simulation' or 'backtest'."
        )

    # -------------------------------------------------------------------------
    # 2) Build Δmid per event (event clock; dt ≡ 1)
    # -------------------------------------------------------------------------
    # Minimal fix: use the same mid-series builder as the signature plot.
    # This automatically switches to an "absolute" mid in ticks when Shift exists:
    #   mid_abs[t] = mid_rel[t] + cumsum(Shift)[t]
    # Then Δmid_event = diff(mid_abs) is consistent with the true price motion.
    mid_series = _build_mid_series_ticks(msg_df)
    dmid_event = np.diff(mid_series)

    mask_event = np.isfinite(dmid_event)
    if int(np.sum(mask_event)) < 2:
        raise ValueError("Not enough finite Δmid points to estimate event volatility.")
    dmid_event_valid = dmid_event[mask_event]
    sigma_per_sqrt_event = float(np.nanstd(dmid_event_valid))  # ticks / sqrt(event)

    # -------------------------------------------------------------------------
    # 3) Calendar-time (ticks / sqrt(second)) — OPTIONAL diagnostic
    # -------------------------------------------------------------------------
    sigma_per_sqrt_second = None
    dmid_valid = None
    dt_valid = None

    if "Time" in msg_df.columns:
        # Time is defined at the message/event level (same length as msg_df).
        # For Δt we always use diff(Time) aligned with Δmid.
        t = msg_df["Time"].to_numpy(dtype=np.float64)
        dt = np.diff(t)

        # Align dmid_event (length = len(msg_df)-1) with dt (same length).
        dmid = dmid_event  # already length len(msg_df)-1
        mask = np.isfinite(dmid) & np.isfinite(dt) & (dt > 0.0)

        if int(np.sum(mask)) >= 2:
            dmid_valid = dmid[mask]
            dt_valid = dt[mask]
            sigma_per_sqrt_second = float(np.nanstd(dmid_valid / np.sqrt(dt_valid)))
        else:
            # We keep sigma_per_sqrt_second=None instead of raising: the primary estimator is event/bucket clock.
            sigma_per_sqrt_second = None

    # -------------------------------------------------------------------------
    # 4) TOB-bucket volatility (ticks / sqrt(bucket)) — OPTIONAL
    # -------------------------------------------------------------------------
    sigma_per_sqrt_bucket = None
    sigma_per_sqrt_second_bucketed = None
    bucket_stats: Dict[str, object] = {}
    
    # --- Legacy compat: if use_tob_buckets was set, convert to the new API ---
    if use_tob_buckets and aggregation_mode == "event":
        aggregation_mode = "tob"

    bucket_id, n_buckets = build_aggregation_buckets(
        message_df=msg_df, ob_df=ob_df, mode=aggregation_mode,
        n_tob_moves=n_tob_moves, n_events_interval=n_events_interval, time_interval=time_interval
    )
    
    # If bucket_id is None, we are in event mode
    if bucket_id is None:
        pass # sigma_per_sqrt_bucket remains None
    else:
        # Using first/last makes each bucket a single "decision step" for policies that reprice per bucket.
        unique_b = np.unique(bucket_id)
        dmid_b = []  # type: list[float]
        dt_b = []  # type: list[float]

        have_time = "Time" in msg_df.columns
        t_full = msg_df["Time"].to_numpy(dtype=np.float64) if have_time else None

        for b in unique_b:
            idxs = np.where(bucket_id == b)[0]
            if idxs.size < 2:
                continue
            i0 = int(idxs[0])
            i1 = int(idxs[-1])

            dm = float(mid_series[i1] - mid_series[i0])
            if not np.isfinite(dm):
                continue

            dmid_b.append(dm)

            # Calendar-time bucket duration is OPTIONAL (used only for sigma_per_sqrt_second_bucketed)
            if have_time and t_full is not None:
                dtt = float(t_full[i1] - t_full[i0])
                if np.isfinite(dtt) and dtt > 0.0:
                    dt_b.append(dtt)

        dmid_b_arr = np.asarray(dmid_b, dtype=np.float64)
        if dmid_b_arr.size >= 2:
            sigma_per_sqrt_bucket = float(np.nanstd(dmid_b_arr))  # ticks / sqrt(bucket)

            # If we also managed to collect bucket durations, compute ticks/sqrt(second) on bucketed returns
            if len(dt_b) == len(dmid_b):
                dt_b_arr = np.asarray(dt_b, dtype=np.float64)
                sigma_per_sqrt_second_bucketed = float(np.nanstd(dmid_b_arr / np.sqrt(dt_b_arr)))

                bucket_stats = {
                    "n_buckets_used": int(dmid_b_arr.size),
                    "mean_bucket_dt": float(np.mean(dt_b_arr)),
                    "median_bucket_dt": float(np.median(dt_b_arr)),
                }
            else:
                bucket_stats = {
                    "n_buckets_used": int(dmid_b_arr.size),
                    "mean_bucket_dt": None,
                    "median_bucket_dt": None,
                }

    # -------------------------------------------------------------------------
    # 5) Choose PRIMARY volatility consistent with your decision clock
    # -------------------------------------------------------------------------
    if aggregation_mode == "event" and not use_tob_buckets:
        volatility = float(sigma_per_sqrt_event)
        volatility_unit = "ticks/sqrt(event)"
        clock_mode = "event"
    else:
        # Default to bucket calc
        if sigma_per_sqrt_bucket is not None:
            volatility = float(sigma_per_sqrt_bucket)
            volatility_unit = f"ticks/sqrt({aggregation_mode})"
            clock_mode = aggregation_mode
        else:
            # Fallback if bucketing failed (e.g. not enough buckets)
            volatility = float(sigma_per_sqrt_event)
            volatility_unit = "ticks/sqrt(event)"
            clock_mode = "event (fallback)"

    # -------------------------------------------------------------------------
    # 6) Plot (optional): quick diagnostic
    # -------------------------------------------------------------------------
    # Show the data source (Simulation / Backtest) in the title so the user
    # can immediately tell which mode produced the estimate.
    data_label = "Simulation" if data_mode == "simulation" else "Backtest"

    if plot:
        # Dynamic label construction
        algo_desc = _format_aggregation_label(aggregation_mode, time_interval, n_tob_moves, n_events_interval)

        # Use provided prefix OR dynamic label
        if not plot_title_prefix:
            title = f"[{data_label}] Volatility Estimate: {algo_desc}"
        else:
            title = f"{plot_title_prefix} Volatility Estimate"

        subtitle = f"({volatility_unit})"

        plt.figure()

        # Plot the raw increments that correspond to the PRIMARY clock
        if clock_mode == "event":
            series = dmid_event_valid
            xlabel = "Event Index"
            ylabel = "Δmid_event (ticks)"
        else:
            # If bucket mode is selected but dmid_b_arr is too small, fall back to event increments for plotting.
            series = dmid_b_arr if (sigma_per_sqrt_bucket is not None) else dmid_event_valid
            xlabel = f"Bucket Index ({algo_desc})"
            ylabel = f"Δmid_{aggregation_mode} (ticks)"

        plt.plot(series[: min(2000, int(series.size))])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + "\n" + subtitle + f" | vol={volatility:.6g}")
        plt.show()

    return {
        # PRIMARY choice aligned with your MM decision clock
        "volatility": volatility,
        "volatility_unit": volatility_unit,
        "clock_mode": clock_mode,

        # Always reported diagnostics (so you can compare scales)
        "volatility_per_sqrt_event": sigma_per_sqrt_event,
        "volatility_per_sqrt_second": sigma_per_sqrt_second,
        "volatility_per_sqrt_bucket": sigma_per_sqrt_bucket,
        "volatility_per_sqrt_second_bucketed": sigma_per_sqrt_second_bucketed,
        "bucket_stats": bucket_stats,

        # Data source identification + raw DataFrames for debugging / reuse
        "data_mode": data_mode,
        "msg_df": msg_df,
        "ob_df": ob_df,

        # Echo of config
        "queue_aware": bool(queue_aware),
        "use_tob_buckets": bool(use_tob_buckets),
        "n_tob_moves": int(n_tob_moves),
        "half_tick": float(half_tick),
        "strict": bool(strict),
        "mo_size": int(mo_size),
    }



# =========================================================================
# SECTION 10 — VOLATILITY SIGNATURE PLOT (multi-path Monte Carlo)
# =========================================================================
# The volatility signature sigma(tau) reveals the timescale structure of
# mid-price movements.  For a diffusive process, sigma(tau) is constant
# across all tau.  Deviations indicate microstructure noise (sigma rises
# for small tau) or mean-reversion (sigma drops for small tau).
#
# Definition:
#     sigma(tau) = sqrt( E[(m_{t+tau} - m_t)^2] / tau )
#
# This section provides:
#   - compute_volatility_signature:       core signature computation
#   - _default_tau_grid:                  log/linear tau grids
#   - plot_volatility_signature_santa_fe: multi-path Monte Carlo wrapper
#   - plot_vol_signature_event_time:      standalone event-time signature
# =========================================================================

def compute_volatility_signature(mid_ticks: "np.ndarray", taus: "np.ndarray") -> "np.ndarray":
    """Compute the volatility signature curve in the chosen discrete clock.

    Signature definition (classic):
        sigma(tau) = sqrt( E[(m_{t+tau} - m_t)^2] / tau )

    - Here, m_t is the mid-price in **ticks**.
    - tau is an integer number of steps in the chosen clock (events or buckets).
    - Output units:
          ticks / sqrt(event)   if tau is in events
          ticks / sqrt(bucket)  if tau is in buckets

    Notes
    -----
    - We do NOT drop events where Δmid = 0. Those are real "no-move" events in event time.
      If you want a TOB-only clock, use clock_mode="tob_bucket" in the wrapper below.
    - NaNs are handled safely.
    """
    mid = np.asarray(mid_ticks, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.int64)

    if mid.size < 3:
        return np.full(taus.shape, np.nan, dtype=np.float64)

    out = np.full(taus.shape, np.nan, dtype=np.float64)

    for i, tau in enumerate(taus):
        tau = int(tau)
        if tau <= 0 or tau >= mid.size:
            continue

        dm = mid[tau:] - mid[:-tau]  # length (T - tau,)
        dm = dm[np.isfinite(dm)]
        if dm.size == 0:
            continue

        sig2 = float(np.mean(dm * dm)) / float(tau)
        out[i] = math.sqrt(sig2) if (sig2 >= 0.0 and math.isfinite(sig2)) else np.nan

    return out


def _default_tau_grid(max_tau: int, n_tau: int = 30, mode: str = "log") -> np.ndarray:
    """Generate a default tau grid for volatility signature plots.

    Returns an array of unique, increasing positive integers.

    mode="log"    Log-spaced grid.  This is the standard in microstructure
                  literature because it gives good resolution at short lags
                  (where sigma(tau) varies most) and fewer points at long lags.
    mode="linear" Linearly spaced grid.  Simpler but may waste resolution.

    Both modes guarantee that tau=1 and tau=max_tau are included.
    """
    max_tau = int(max(1, max_tau))
    n_tau = int(max(2, n_tau))
    mode = str(mode).lower().strip()

    if mode in ("log", "logspace", "log-spaced", "log_spaced"):
        taus = np.logspace(0.0, math.log10(float(max_tau)), num=n_tau)
        taus = np.unique(taus.astype(np.int64))
        taus = taus[taus >= 1]
        taus = taus[taus <= max_tau]
        # Guarantee endpoints if possible
        if taus.size == 0 or taus[0] != 1:
            taus = np.unique(np.r_[1, taus])
        if taus[-1] != max_tau:
            taus = np.unique(np.r_[taus, max_tau])
        return taus.astype(np.int64)

    if mode in ("linear", "lin", "linspace"):
        taus = np.linspace(1, max_tau, num=n_tau)
        taus = np.unique(taus.astype(np.int64))
        taus = taus[taus >= 1]
        return taus.astype(np.int64)

    raise ValueError('tau_mode must be "log" or "linear".')

def plot_volatility_signature_santa_fe(
    *,
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    n_steps: int,
    n_steps_to_equilibrium: int,
    half_tick: float = 0.5,
    strict: bool = True,
    mm_policy=None,
    exclude_self_from_state: bool = False,
    mo_size: int = 1,
    split_sweeps: bool = False,

    # Clock control
    clock_mode: str = "event",          # "event", "tob", "time", "fixed_events"
    n_tob_moves: int = 10,              # for clock_mode="tob"
    n_events_interval: int = 100,       # for clock_mode="fixed_events"
    time_interval: float = 1.0,         # for clock_mode="time"

    # Signature controls
    tau_grid: Optional[np.ndarray] = None,
    max_tau: Optional[int] = None,
    n_tau: int = 30,
    tau_mode: str = "log",

    # Monte Carlo control
    n_paths: int = 1,
    random_seed: Optional[int] = None,

    # Plot controls
    plot: bool = True,
    plot_bars: bool = True,
    bar_k_std: float = 2.0,             # show ±k std
    bar_capsize: float = 3.0,
    bar_show_mean_line: bool = True,
):
    """Monte Carlo volatility signature plot using the Santa-Fe simulator.

    Runs *n_paths* independent simulator paths with offset seeds, computes
    the signature sigma(tau) for each path using a shared global tau grid,
    then plots the cross-path mean +/- k*std error bars.

    The tau grid is determined from the SHORTEST path (so that the result
    matrix is rectangular and safely stackable with np.vstack).

    Supports all four aggregation clocks: in event mode the mid series is
    used directly; in bucket modes, the mid is resampled at bucket boundaries
    (last mid-price per bucket).

    Returns
    -------
    dict with keys: "taus", "signature_mean", "signature_std", "per_path",
    "clock_mode", "unit", "msg_df", "ob_df".
    """
    clock_mode = str(clock_mode).lower().strip()
    
    n_paths = int(max(1, n_paths))

    # Optional deterministic control for the whole call
    if random_seed is not None:
        import random as _random
        _random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    series_list: list[np.ndarray] = []
    last_msg_df = None
    last_ob_df = None

    # -------------------------------------------------------------------------
    # PASS 1: simulate and build the per-path series (event clock or tob buckets)
    # -------------------------------------------------------------------------
    unit = f"ticks/sqrt({clock_mode})"

    for p in range(n_paths):
        # If we want multiple *different* paths but still reproducible, offset the seed per path.
        if random_seed is not None:
            import random as _random
            seed_p = int(random_seed) + int(p)
            _random.seed(seed_p)
            np.random.seed(seed_p)

        msg_df, ob_df, _mm_df = simulate_LOB_with_MM(
            lam=lam,
            mu=mu,
            delta=delta,
            random_seed=(None if random_seed is None else int(random_seed) + int(p)),
            number_tick_levels=number_tick_levels,
            n_priority_ranks=n_priority_ranks,
            iterations=n_steps,
            iterations_to_equilibrium=n_steps_to_equilibrium,
            mm_policy=mm_policy,
            exclude_self_from_state=bool(exclude_self_from_state),
            beta_exp_weighted_return=0.0,
            intensity_exp_weighted_return=0.0,
            mean_size_MO=int(mo_size),
            split_sweeps=bool(split_sweeps),
        )

        if msg_df is None or getattr(msg_df, "empty", True):
            raise ValueError("Simulator returned empty message_df; cannot compute signature.")

        mid = _build_mid_series_ticks(msg_df)
        mid = np.asarray(mid, dtype=np.float64)

        # Build buckets for resampling
        bucket_ids, _ = build_aggregation_buckets(
            msg_df, ob_df, mode=clock_mode,
            n_tob_moves=n_tob_moves, n_events_interval=n_events_interval, time_interval=time_interval
        )

        if bucket_ids is None:
            series = mid
        else:
            # Resample at bucket edges
            df = pd.DataFrame({"mid": mid, "b": bucket_ids})
            series = df.groupby("b")["mid"].last().to_numpy()

        if series.size < 2:
            raise ValueError(
                f"Path {p} produced too-short series (len={series.size}). "
                "Cannot compute signature."
            )

        series_list.append(series)

        last_msg_df = msg_df
        last_ob_df = ob_df

    # -------------------------------------------------------------------------
    # PASS 2: build ONE taus grid valid for ALL paths (based on smallest series)
    # -------------------------------------------------------------------------
    min_len = int(min(s.size for s in series_list))

    if tau_grid is None:
        if max_tau is None:
            # same spirit as your default, but based on min_len (GLOBAL), not per-path length
            max_tau_eff = int(max(1, min(min_len // 10, 10_000)))
        else:
            max_tau_eff = int(max(1, min(int(max_tau), min_len - 1)))

        if max_tau_eff < 1:
            raise ValueError("Not enough points to compute signature (max_tau_eff < 1).")

        taus = _default_tau_grid(
            max_tau=max_tau_eff,
            n_tau=int(n_tau),
            mode=str(tau_mode).lower()
        )
        taus = np.asarray(taus, dtype=np.int64)
        taus = np.unique(taus[taus >= 1])
        taus = taus[taus < min_len]
        if taus.size == 0:
            raise ValueError("Auto tau grid produced no valid taus (need 1 <= tau < min_len).")
    else:
        taus = np.asarray(list(tau_grid), dtype=np.int64)
        taus = np.unique(taus[taus >= 1])
        taus = taus[taus < min_len]   # IMPORTANT: make it valid for ALL paths
        if taus.size == 0:
            raise ValueError(
                "tau_grid produced no valid taus shared by all paths "
                "(need 1 <= tau < min_len across paths)."
            )

    # -------------------------------------------------------------------------
    # PASS 3: compute signature for each path with SAME taus and stack safely
    # -------------------------------------------------------------------------
    per_path = [compute_volatility_signature(s, taus) for s in series_list]
    per_path_arr = np.vstack([np.asarray(sig, dtype=np.float64) for sig in per_path])

    sig_mean = np.nanmean(per_path_arr, axis=0)
    sig_std = np.nanstd(per_path_arr, axis=0)

    if plot:
        plt.figure()

        if plot_bars and int(n_paths) >= 2:
            yerr = float(bar_k_std) * sig_std
            plt.errorbar(
                taus,
                sig_mean,
                yerr=yerr,
                fmt="o",
                markerfacecolor="none",
                capsize=float(bar_capsize),
            )
            if bar_show_mean_line:
                plt.plot(taus, sig_mean)
        else:
            if bar_show_mean_line:
                plt.plot(taus, sig_mean, marker="o", markerfacecolor="none")
            else:
                plt.plot(taus, sig_mean, marker="o", linestyle="None", markerfacecolor="none")

        plt.xlabel(f"tau ({clock_mode} units)")
        plt.ylabel(f"signature sigma(tau) [{unit}]")
        plt.grid(True)
        plt.xscale("log" if str(tau_mode).lower().startswith("log") else "linear")
        
        # Dynamic label
        algo_desc = _format_aggregation_label(clock_mode, time_interval, n_tob_moves, n_events_interval)
        plt.title(
            f"Volatility Signature Plot\nMode: {algo_desc}, n_paths={n_paths}\n"
            f"(lam={lam}, mu={mu}, delta={delta}, mo_size={mo_size})"
        )
        plt.show()

    return {
        "taus": taus,
        "signature_mean": sig_mean,
        "signature_std": sig_std,
        "per_path": per_path_arr,
        "clock_mode": clock_mode,
        "unit": unit,
        "msg_df": last_msg_df,
        "ob_df": last_ob_df,
    }


# =========================================================================
# SECTION 11 — EVENT-TIME SIGNATURE PLOT (standalone, no simulator)
# =========================================================================
# A simpler version of the signature plot that works directly on a single
# message_df without running the simulator.  Useful when you already have
# the data and just want a quick diagnostic.
# =========================================================================

def plot_vol_signature_event_time(
    message_df,
    min_tau: int = 1,
    max_tau: Optional[int] = None,
    step: int = 1,
    taus: Optional[np.ndarray] = None,
    mid_col: str = "MidPrice",
    return_col: str = "Return",
    drop_flat: bool = False,
    logx: bool = False,
    ax=None,
    title: Optional[str] = None,
):
    """
    Volatility signature plot in EVENT TIME (ticks/events), i.e. dt ≡ 1.

    We estimate, for each lag τ (in number of events):
        sigma(τ) = sqrt( E[(m_{t+τ} - m_t)^2] / τ )
    Units: ticks / sqrt(event)

    Parameters
    ----------
    message_df : pd.DataFrame
        Must contain either `mid_col` (preferred) or `return_col`.
        - If `mid_col` exists, we use it directly.
        - Else we reconstruct mid as cumulative sum of `return_col` (relative mid).
    min_tau, max_tau : int
        Smallest/largest lag in number of events.
    step : int
        Compute only every `step` lags (useful when max_tau is large).
    taus : np.ndarray or None
        If provided, overrides (min_tau, max_tau, step). Must be positive integers.
    drop_flat : bool
        If True, remove dm == 0 increments before computing mean(dm^2).
        NOTE: this changes the meaning (conditioning on mid changes).
    logx : bool
        If True, plot τ on log scale.
    ax : matplotlib axis or None
    title : str or None

    Returns
    -------
    taus_out : np.ndarray (K,)
    sigmas   : np.ndarray (K,)
    """

    # ----------------------------
    # 1) Get mid series (ticks)
    # ----------------------------
    if mid_col in message_df.columns:
        mid = np.asarray(message_df[mid_col].values, dtype=np.float64)
    elif return_col in message_df.columns:
        # Reconstruct a relative mid: mid[t] = sum_{i<=t} Return[i]
        # (If you want absolute levels, add an initial mid offset.)
        r = np.asarray(message_df[return_col].values, dtype=np.float64)
        mid = np.cumsum(r)
    else:
        raise ValueError(f"message_df must contain '{mid_col}' or '{return_col}'")

    # Need at least 2 points
    T = int(mid.shape[0])
    if T < 2:
        raise ValueError("Need at least 2 events to compute a signature plot.")

    # ----------------------------
    # 2) Choose taus (event lags)
    # ----------------------------
    if taus is None:
        if max_tau is None:
            max_tau = T - 1
        max_tau = int(min(max_tau, T - 1))
        min_tau = int(max(min_tau, 1))
        step = int(max(step, 1))
        taus_out = np.arange(min_tau, max_tau + 1, step, dtype=np.int64)
    else:
        taus_out = np.asarray(taus, dtype=np.int64)
        taus_out = taus_out[(taus_out >= 1) & (taus_out <= T - 1)]
        if taus_out.size == 0:
            raise ValueError("Provided taus are empty or out of valid range [1, T-1].")

    # ----------------------------
    # 3) Compute sigma(τ)
    # ----------------------------
    sigmas = np.empty_like(taus_out, dtype=np.float64)

    for i, tau in enumerate(taus_out):
        dm = mid[tau:] - mid[:-tau]          # (T - tau,)
        if drop_flat:
            dm = dm[dm != 0.0]
            if dm.size == 0:
                sigmas[i] = np.nan
                continue

        # signature definition: sqrt( E[dm^2] / tau )
        sig2 = float(np.mean(dm * dm)) / float(tau)
        sigmas[i] = np.sqrt(sig2) if sig2 >= 0.0 else np.nan

    # ----------------------------
    # 4) Plot
    # ----------------------------
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(taus_out, sigmas)
    ax.set_xlabel("Lag τ (number of events)")
    ax.set_ylabel(r"$\sigma(\tau)$  [ticks / $\sqrt{\mathrm{event}}$]")
    ax.grid(True)

    if logx:
        ax.set_xscale("log")

    if title is None:
        title = "Volatility Signature Plot (event time)"
        if drop_flat:
            title += " — conditioned on Δmid≠0"
    ax.set_title(title)

    return taus_out, sigmas


# =========================================================================
# SECTION 12 — PER-LEVEL EXECUTION INTENSITY FROM PMF (lambda_exec)
# =========================================================================
# Instead of fitting a parametric A*exp(-kappa*delta) curve, this section
# computes the *empirical* execution intensity at each depth level as a
# point-mass function (PMF), decomposed into:
#
#   lambda_exec(delta) = r_mo * P(D = delta | MO-step)
#
# where:
#   r_mo       = (# MO-steps) / (# total decision steps)
#   P(D=delta) = fraction of MO-steps whose deepest fill is at depth delta
#
# This non-parametric view is useful for:
#   - Validating that the A*exp(-kappa*delta) fit is a good approximation
#   - Understanding the shape of the true execution distribution
#   - Feeding exact per-level intensities into a discretized GLFT optimizer
# =========================================================================

def compute_lambda_exec_from_pmf_and_throttle(
    *,
    # ---------------------------------------------------------------------
    # DATA SOURCE CONTROL (same style as fit_trading_intensity_parameters)
    # ---------------------------------------------------------------------
    data_mode: str = "backtest",  # "simulation" or "backtest"
    message_df: Optional[pd.DataFrame] = None,
    ob_df: Optional[pd.DataFrame] = None,

    # ---------------------------------------------------------------------
    # THROTTLE / AGGREGATION (same API as the rest of the file)
    # ---------------------------------------------------------------------
    aggregation_mode: str = "event",   # "event", "tob", "time", "fixed_events"
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,     # legacy: if True and mode="event", force "tob"

    # ---------------------------------------------------------------------
    # DEPTH GRID
    # ---------------------------------------------------------------------
    max_levels: int = 200,
    half_tick: float = 0.5,

    # ---------------------------------------------------------------------
    # SPLIT-SWEEP COLLAPSE (important for backtests with split_sweeps=True logs)
    # ---------------------------------------------------------------------
    collapse_split_sweeps: bool = True,

    # ---------------------------------------------------------------------
    # SIMULATION PARAMS (ignored if data_mode="backtest")
    # ---------------------------------------------------------------------
    lam: float = 0.05,
    mu: float = 0.05,
    delta: float = 0.05,
    number_tick_levels: int = 100,
    n_priority_ranks: int = 20,
    n_steps: int = 100_000,
    n_steps_to_equilibrium: int = 2_000,
    random_seed: Optional[int] = None,
    mm_policy=None,
    exclude_self_from_state: bool = True,
    mo_size: int = 1,
    split_sweeps: bool = False,  # if your simulator supports it

    # ---------------------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------------------
    plot: bool = True,
    plot_pmf: bool = False,  # optional: also plot the conditional PMF
    plot_title_prefix: str = "",
    title: str = r"Execution intensity by level: $\lambda_{\mathrm{exec}}(\delta)$",
) -> Dict[str, Any]:
    """
    Compute an execution intensity per depth level on a chosen "clock"
    (event / tob buckets / time buckets / fixed-events buckets), in the same
    spirit as fit_trading_intensity_parameters.

    Definitions (consistent with your bucket semantics)
    ---------------------------------------------------
    1) For each MO event i, compute D_i:
        D_i = deepest tick index reached by the MO sweep on the PRE-event book.

    2) If aggregation_mode == "event":
        - We build a PMF over D_i across MO events (one observation per MO event).

       If aggregation_mode in {"tob","time","fixed_events"} (bucket mode):
        - We define one observation per bucket:
            D_bucket = max{ D_i : i is an MO event inside this bucket }.
          (This matches your existing bucket logic used elsewhere.)

    3) Let denom_total be:
        - event mode:        denom_total = #events
        - bucket mode:       denom_total = #buckets_total   (includes buckets with no MO)

    4) We build pmf_counts over D observations (event-level or bucket-level).
       Let n_mo_steps_used = sum(pmf_counts) = number of MO-steps used:
        - event mode:  number of MO events with finite D
        - bucket mode: number of buckets that had >=1 MO with finite D

    5) We return:
        - pmf_cond(δ)  = P(D = δ | MO-step used) = pmf_counts / n_mo_steps_used
        - r_mo         = n_mo_steps_used / denom_total     (MO-step rate in the chosen clock)
        - lambda_exec  = pmf_counts / denom_total          (unconditional intensity per clock step)
                         = r_mo * pmf_cond

    Notes
    -----
    - This formulation is *exactly* aligned with your bucketed tail intensity:
        tail_intensity(k) = (#buckets with depth>=k) / (#buckets_total)
    - We do NOT use "#MO events / #buckets_total" in bucket mode, because we have
      collapsed multiple MOs inside a bucket into ONE observation (max depth). Using
      total MO events would mismatch the PMF semantics.
    """

    # ---------------------------------------------------------------------
    # 0) Normalize aggregation mode + legacy compat
    # ---------------------------------------------------------------------
    aggregation_mode = str(aggregation_mode).lower().strip()
    if use_tob_buckets and aggregation_mode == "event":
        aggregation_mode = "tob"

    # ---------------------------------------------------------------------
    # 1) Acquire data (simulation or backtest)
    # ---------------------------------------------------------------------
    if str(data_mode).lower().strip() == "simulation":
        # Run Santa-Fe simulator (same wrapper used elsewhere)
        msg_df, book_df, _mm_df = _run_simulator_santa_fe(
            lam=lam,
            mu=mu,
            delta=delta,
            number_tick_levels=number_tick_levels,
            n_priority_ranks=n_priority_ranks,
            n_steps=n_steps,
            n_steps_to_equilibrium=n_steps_to_equilibrium,
            mm_policy=mm_policy,
            exclude_self_from_state=exclude_self_from_state,
            mo_size=mo_size,
            split_sweeps=split_sweeps,
            random_seed=random_seed,
        )
        sim_desc = "Simulation"

    elif str(data_mode).lower().strip() == "backtest":
        if message_df is None or ob_df is None:
            raise ValueError("Backtest mode requires both message_df and ob_df.")
        msg_df = message_df
        book_df = ob_df
        sim_desc = "Backtest"

    else:
        raise ValueError(f"Unknown data_mode='{data_mode}'. Use 'simulation' or 'backtest'.")

    if msg_df is None or getattr(msg_df, "empty", True):
        raise ValueError("message_df is empty; cannot compute lambda_exec.")
    if book_df is None or getattr(book_df, "empty", True):
        raise ValueError("ob_df is empty; cannot compute lambda_exec (needs pre-event book).")


    # ---------------------------------------------------------------------
    # 2) Optional collapse of split-sweep logs (i=NaN detail rows)
    # ---------------------------------------------------------------------
    collapsed = False
    if collapse_split_sweeps:
        msg_df, book_df, collapsed = _maybe_collapse_split_sweeps(msg_df, book_df)

    # Critical alignment check: your pre-event book uses ob_df.iloc[i-1]
    if len(book_df) != len(msg_df):
        raise ValueError(
            "message_df and ob_df must be event-aligned and have the SAME length after optional collapse.\n"
            f"Got len(message_df)={len(msg_df)} and len(ob_df)={len(book_df)}.\n"
            "If your logs use split_sweeps=True, set collapse_split_sweeps=True and ensure ob_df is either\n"
            "(a) split-sweep aligned (same length as msg) or\n"
            "(b) already event-aligned (same length as msg after filtering i-finite rows)."
        )

    # ---------------------------------------------------------------------
    # 3) Compute per-event deepest tick D_i (MO sweep on PRE-event book)
    # ---------------------------------------------------------------------
    D_by_event, _n_used = compute_nonqueue_deepest_tick_per_mo(
        message_df=msg_df,
        ob_df=book_df,
        max_depth_levels=max_levels,
        half_tick=half_tick,
    )

    n = int(len(msg_df))
    if n <= 0:
        raise ValueError("Empty message_df after preprocessing.")

    # MO mask over events
    mo_mask = np.array([_is_mo_row(msg_df, i) for i in range(n)], dtype=bool)
    if int(np.sum(mo_mask)) == 0:
        raise ValueError("No MO events found; cannot compute lambda_exec.")

    # ---------------------------------------------------------------------
    # 4) Build aggregation buckets (None in event mode)
    # ---------------------------------------------------------------------
    bucket_ids, n_buckets_total = build_aggregation_buckets(
        message_df=msg_df,
        ob_df=book_df,
        mode=aggregation_mode,
        n_tob_moves=n_tob_moves,
        n_events_interval=n_events_interval,
        time_interval=time_interval,
    )

    # denom_total = total number of "decision steps" in the chosen clock
    if bucket_ids is None:
        denom_total = int(n)
        clock_unit = "event"
    else:
        denom_total = int(n_buckets_total)
        clock_unit = f"{aggregation_mode} bucket"

    if denom_total <= 0:
        raise ValueError("Denominator for the chosen clock is zero; check your bucketing configuration.")

    # ---------------------------------------------------------------------
    # 5) Build depth observations: event-level vs bucket-level (max per bucket)
    # ---------------------------------------------------------------------
    if bucket_ids is None:
        # One observation per MO event (with finite D)
        used_mask = mo_mask & np.isfinite(D_by_event)
        ticks = D_by_event[used_mask].astype(int)
    else:
        # One observation per bucket that contains >=1 valid MO:
        # D_bucket = max D_i among MO events inside bucket
        if int(bucket_ids.size) != n:
            raise ValueError("bucket_ids must be aligned with message_df length.")

        bucket_max_tick: Dict[int, int] = {}
        for i in range(n):
            if not mo_mask[i]:
                continue
            d = D_by_event[i]
            if not np.isfinite(d):
                continue
            b = int(bucket_ids[i])
            t = int(d)
            prev = bucket_max_tick.get(b, -10**18)
            if t > prev:
                bucket_max_tick[b] = t

        ticks = np.asarray(list(bucket_max_tick.values()), dtype=int)

    n_mo_steps_used = int(ticks.size)
    if n_mo_steps_used == 0:
        raise ValueError(
            "No valid MO observations to build the PMF.\n"
            "This typically means D_by_event is NaN for all MOs (mid/book alignment issue),\n"
            "or max_levels is too small, or the pre/post snapshot indexing is off."
        )

    # ---------------------------------------------------------------------
    # 6) PMF counts over depth ticks (trim to last non-zero)
    # ---------------------------------------------------------------------
    pmf_counts_full = np.zeros(int(max_levels), dtype=np.int64)
    for t in ticks:
        if 0 <= int(t) < int(max_levels):
            pmf_counts_full[int(t)] += 1

    nz = np.where(pmf_counts_full > 0)[0]
    if nz.size == 0:
        raise ValueError("PMF counts are all zero after filtering; check max_levels and depth mapping.")
    K = int(nz.max()) + 1
    pmf_counts = pmf_counts_full[:K]

    # δ-grid consistent with your convention: δ_k = (k+1)*half_tick
    k = np.arange(K, dtype=np.float64)
    delta_grid = (k + 1.0) * float(half_tick)

    # Conditional PMF: P(D=δ | MO-step used)
    pmf_cond = pmf_counts.astype(np.float64) / float(n_mo_steps_used)

    # ---------------------------------------------------------------------
    # 7) Unconditional intensity per clock step (THIS is what we plot by default)
    # ---------------------------------------------------------------------
    # r_mo: "MO-step rate" on chosen clock (events with MO, or buckets with >=1 MO)
    r_mo = float(n_mo_steps_used) / float(denom_total)

    # lambda_exec: unconditional per-step intensity at depth δ
    # (equivalently: pmf_counts / denom_total)
    lambda_exec = pmf_counts.astype(np.float64) / float(denom_total)

    # ---------------------------------------------------------------------
    # 8) Plot (optional) with same tidy style as fit_trading_intensity
    # ---------------------------------------------------------------------
    if plot:
        algo_desc = _format_aggregation_label(
            aggregation_mode, time_interval, n_tob_moves, n_events_interval
        )
        prefix = plot_title_prefix if plot_title_prefix else f"{sim_desc}: "

        fig, ax = plt.subplots()
        ax.plot(delta_grid, lambda_exec, label=r"$\lambda_{\mathrm{exec}}(\delta)$")
        ax.set_xlabel(r"$\delta$ (ticks from the mid-price)")
        ax.set_ylabel(f"execution intensity (per {clock_unit})")
        ax.set_title(
            f"{prefix}{algo_desc} | {title}\n"
            f"r_mo={r_mo:.6g} (MO-steps/{clock_unit}), "
            f"n_mo_steps_used={n_mo_steps_used}, denom_total={denom_total}, collapsed={collapsed}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

        if plot_pmf:
            fig, ax = plt.subplots()
            ax.plot(delta_grid, pmf_cond, label=r"$\Pr(D=\delta \mid \mathrm{MO\mbox{-}step})$")
            ax.set_xlabel(r"$\delta$ (ticks from the mid-price)")
            ax.set_ylabel("conditional PMF")
            ax.set_title(
                f"{prefix}{algo_desc} | Conditional depth PMF\n"
                f"n_mo_steps_used={n_mo_steps_used}"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()

    # ---------------------------------------------------------------------
    # 9) Return a rich dict (mirrors fit_trading_intensity style)
    # ---------------------------------------------------------------------
    return {
        # Primary outputs
        "lambda_exec": lambda_exec,   # unconditional per-clock-step intensity
        "pmf_cond": pmf_cond,         # conditional PMF given an MO-step
        "pmf_counts": pmf_counts,     # raw counts (trimmed)
        "delta_grid": delta_grid,

        # Rates / denominators
        "r_mo": r_mo,                 # MO-step rate on chosen clock
        "rate_unit": f"MO-steps per {clock_unit}",
        "n_mo_steps_used": n_mo_steps_used,
        "denom_total": denom_total,   # #events or #buckets_total

        # Clock config (echo)
        "aggregation_mode": aggregation_mode,
        "use_tob_buckets": bool(use_tob_buckets),
        "n_tob_moves": int(n_tob_moves),
        "n_events_interval": int(n_events_interval),
        "time_interval": float(time_interval),

        # Diagnostics / bookkeeping
        "collapsed_split_sweeps": bool(collapsed),
        "data_mode": str(data_mode),
        "half_tick": float(half_tick),
        "max_levels": int(max_levels),

        # Keep dfs for reproducibility / debugging
        "msg_df": msg_df,
        "ob_df": book_df,

        # Also expose bucket total explicitly (this is what you asked: buckets_total)
        "n_buckets_total": (None if bucket_ids is None else int(n_buckets_total)),
    }



# =========================================================================
# SECTION 13 — Pr(MO | delta) ESTIMATION (conditional fill probability)
# =========================================================================
# This estimates the probability that an event affecting depth level delta
# is a market order (rather than a limit order or cancellation).  It is
# useful when the GLFT model is extended to condition execution probability
# on the event type.
#
# Two "level sampling" conventions are supported:
#
#   "touch" mode:  An MO that sweeps to depth D "touches" all levels
#                  0, 1, ..., D (or 0..D-1 if strict=True).  A non-MO
#                  touches only its own price level.
#
#   "exact" mode:  Each event maps to exactly one depth.  MOs use their
#                  deepest D; non-MOs use their price level.
#
# The output is:  p_k = M_k / N_k
#   where N_k = # decision steps that touched level k (any event type)
#         M_k = # decision steps that touched level k via an MO
# =========================================================================

def compute_prob_event_is_mo_given_level(
    *,
    # ---------------------------------------------------------------------
    # DATA SOURCE CONTROL
    # ---------------------------------------------------------------------
    data_mode: str = "backtest",  # "simulation" or "backtest"
    message_df: Optional[pd.DataFrame] = None,
    ob_df: Optional[pd.DataFrame] = None,

    # ---------------------------------------------------------------------
    # THROTTLE / AGGREGATION ("decision clock")
    # ---------------------------------------------------------------------
    aggregation_mode: str = "event",   # "event", "tob", "time", "fixed_events"
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,
    use_tob_buckets: bool = False,     # legacy: if True and mode="event", force "tob"

    # ---------------------------------------------------------------------
    # DEPTH GRID
    # ---------------------------------------------------------------------
    max_levels: int = 200,
    half_tick: float = 0.5,

    # ---------------------------------------------------------------------
    # SPLIT-SWEEP COLLAPSE (important for backtests with split_sweeps=True logs)
    # ---------------------------------------------------------------------
    collapse_split_sweeps: bool = True,

    # ---------------------------------------------------------------------
    # SIMULATION PARAMS (ignored if data_mode="backtest")
    # ---------------------------------------------------------------------
    lam: float = 0.05,
    mu: float = 0.05,
    delta: float = 0.05,
    number_tick_levels: int = 100,
    n_priority_ranks: int = 20,
    n_steps: int = 100_000,
    n_steps_to_equilibrium: int = 2_000,
    random_seed: Optional[int] = None,
    mm_policy=None,
    exclude_self_from_state: bool = True,
    mo_size: int = 1,
    split_sweeps: bool = False,

    # ---------------------------------------------------------------------
    # DEFINITION CHOICES
    # ---------------------------------------------------------------------
    level_mode: str = "touch",   # "touch" or "exact"
    strict: bool = False,        # see note in docstring

    # ---------------------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------------------
    plot: bool = True,
    plot_counts: bool = False,
    plot_title_prefix: str = "",
    title: str = r"$\Pr(\mathrm{MO}\mid \delta)$",
) -> Dict[str, Any]:
    """
    Estimate:   p_k = Pr(event is MO | level delta_k from the mid)

    We work on a chosen "decision clock":
      - event mode: one step per simulator event row
      - bucket modes: one step per bucket (TOB/time/fixed-events), including empty buckets

    We define a level index k by:
        k = round(depth / half_tick) - 1
    where depth is measured from the *pre-event* mid.

    Two level sampling conventions (choose via level_mode):
    ------------------------------------------------------
    (A) level_mode="touch"  [matches your "MO at depth 3 also touched 0..3" logic]
        - For an MO event with deepest D: it touches all levels 0..D
          (or 0..D-1 if strict=True, i.e. "strictly beyond k")
        - For a non-MO event: it touches exactly ONE level, the one implied by its Price vs pre-mid.
        - In bucket modes: a bucket "touches" a level if ANY event inside the bucket touches it.

        Counts:
          N_k = #steps (events or buckets) where level k was touched by ANY event
          M_k = #steps (events or buckets) where level k was touched by an MO
        Output:
          p_k = M_k / N_k  (NaN where N_k = 0)

    (B) level_mode="exact"  [each step maps to exactly one depth]
        - MO step contributes to level = deepest D (per event, or per bucket deepest MO)
        - non-MO step contributes to its own price-level
        - In bucket mode, we use the deepest level touched by ANY event as the bucket's level,
          and label MO if the bucket contains >=1 MO (this matches "one observation per bucket").

    About strict:
    -------------
    If strict=True, in "touch" mode an MO with deepest D touches levels 0..D-1 (i.e. D > k).
    This mirrors your strict tail convention.
    With mo_size=1, many MOs have D=0, so strict=True can make MO-touch counts near-zero.
    """

    # ---------------------------------------------------------------------
    # 0) Normalize modes (legacy compat)
    # ---------------------------------------------------------------------
    aggregation_mode = str(aggregation_mode).lower().strip()
    if use_tob_buckets and aggregation_mode == "event":
        aggregation_mode = "tob"

    level_mode = str(level_mode).lower().strip()
    if level_mode not in ("touch", "exact"):
        raise ValueError("level_mode must be 'touch' or 'exact'.")

    max_levels = int(max(1, max_levels))
    half_tick = float(half_tick)

    # ---------------------------------------------------------------------
    # 1) Acquire data (simulation or backtest)
    # ---------------------------------------------------------------------
    if str(data_mode).lower().strip() == "simulation":
        msg_df, book_df, _mm_df = _run_simulator_santa_fe(
            lam=lam,
            mu=mu,
            delta=delta,
            number_tick_levels=number_tick_levels,
            n_priority_ranks=n_priority_ranks,
            n_steps=n_steps,
            n_steps_to_equilibrium=n_steps_to_equilibrium,
            mm_policy=mm_policy,
            exclude_self_from_state=exclude_self_from_state,
            mo_size=mo_size,
            split_sweeps=split_sweeps,
            random_seed=random_seed,
        )
        sim_desc = "Simulation"
    elif str(data_mode).lower().strip() == "backtest":
        if message_df is None or ob_df is None:
            raise ValueError("Backtest mode requires both message_df and ob_df.")
        msg_df = message_df.reset_index(drop=True)
        book_df = ob_df.reset_index(drop=True)
        sim_desc = "Backtest"
    else:
        raise ValueError(f"Unknown data_mode='{data_mode}'. Use 'simulation' or 'backtest'.")

    if msg_df is None or getattr(msg_df, "empty", True):
        raise ValueError("message_df is empty; cannot compute Pr(MO|delta).")
    if book_df is None or getattr(book_df, "empty", True):
        raise ValueError("ob_df is empty; cannot compute Pr(MO|delta) (needs pre-event book).")

    # Optional collapse
    collapsed = False
    if collapse_split_sweeps:
        msg_df, book_df, collapsed = _maybe_collapse_split_sweeps(msg_df, book_df)

    # Critical alignment check: our MO deepest uses pre-event snapshot book_df.iloc[i-1]
    if len(msg_df) != len(book_df):
        raise ValueError(
            f"Alignment error: len(message_df)={len(msg_df)} != len(ob_df)={len(book_df)}.\n"
            "This estimator assumes 1 row per event in BOTH dfs after optional collapse."
        )

    n = int(len(msg_df))
    if n <= 0:
        raise ValueError("Empty message_df after preprocessing.")

    # ---------------------------------------------------------------------
    # 2) Identify MO events, compute their deepest D_i from the PRE-event book
    # ---------------------------------------------------------------------
    mo_mask = np.array([_is_mo_row(msg_df, i) for i in range(n)], dtype=bool)

    # D_by_event is only meaningful for MOs; will be NaN for non-MOs by construction.
    D_by_event, _n_used_mo = compute_nonqueue_deepest_tick_per_mo(
        message_df=msg_df,
        ob_df=book_df,
        max_depth_levels=max_levels,
        half_tick=half_tick,
    )

    # ---------------------------------------------------------------------
    # 3) Build per-event "level" for non-MOs (from Price vs pre-mid)
    #    and a unified per-event level for "exact" mode.
    # ---------------------------------------------------------------------
    K_price_by_event = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if mo_mask[i]:
            continue
        K_price_by_event[i] = _event_level_from_price(msg_df, i, half_tick, max_levels)

    # Unified event level for "exact" mode:
    # - MO uses deepest D_i
    # - non-MO uses its own price level
    K_exact_by_event = np.where(mo_mask, D_by_event, K_price_by_event)

    # ---------------------------------------------------------------------
    # 4) Build aggregation buckets (None if event mode)
    # ---------------------------------------------------------------------
    bucket_ids, n_buckets_total = build_aggregation_buckets(
        message_df=msg_df,
        ob_df=book_df,
        mode=aggregation_mode,
        n_tob_moves=n_tob_moves,
        n_events_interval=n_events_interval,
        time_interval=time_interval,
    )

    if bucket_ids is None:
        denom_total = int(n)
        clock_unit = "event"
    else:
        denom_total = int(n_buckets_total)
        clock_unit = f"{aggregation_mode} bucket"

    if denom_total <= 0:
        raise ValueError("Denominator for the chosen clock is zero; check bucketing configuration.")

    # ---------------------------------------------------------------------
    # 5) Count N_k (any-event touches) and M_k (MO touches) on the chosen clock
    # ---------------------------------------------------------------------
    N_counts = np.zeros(max_levels, dtype=np.int64)  # touched by any event
    M_counts = np.zeros(max_levels, dtype=np.int64)  # touched by an MO

    if bucket_ids is None:
        # ==========================================================
        # EVENT CLOCK: each simulator event is one step
        # ==========================================================
        if level_mode == "exact":
            # One level per event
            for i in range(n):
                k = K_exact_by_event[i]
                if not np.isfinite(k):
                    continue
                kk = int(k)
                if 0 <= kk < max_levels:
                    N_counts[kk] += 1
                    if mo_mask[i]:
                        M_counts[kk] += 1

        else:
            # level_mode == "touch"
            # Non-MO touches exactly its own level.
            # MO touches a whole prefix of levels up to its deepest.
            for i in range(n):
                if mo_mask[i]:
                    d = D_by_event[i]
                    if not np.isfinite(d):
                        continue
                    D = int(d)
                    if D < 0:
                        continue
                    if D >= max_levels:
                        D = max_levels - 1

                    # strict=True => MO "touches" levels k such that D > k  => k = 0..D-1
                    kmax = (D - 1) if strict else D
                    if kmax >= 0:
                        N_counts[: kmax + 1] += 1
                        M_counts[: kmax + 1] += 1
                else:
                    k = K_price_by_event[i]
                    if not np.isfinite(k):
                        continue
                    kk = int(k)
                    if 0 <= kk < max_levels:
                        N_counts[kk] += 1

    else:
        # ==========================================================
        # BUCKET CLOCK: each bucket is one step (including empty ones)
        # ==========================================================
        if int(bucket_ids.size) != n:
            raise ValueError("bucket_ids must be aligned with message_df length.")

        # We build bucket summaries.
        # For touch mode, we need:
        #   - bucket_nonmo_levels[b] = set(levels touched by non-MO events)
        #   - bucket_mo_deepest[b]   = max deepest among MO events
        bucket_nonmo_levels: Dict[int, set] = {}
        bucket_mo_deepest: Dict[int, int] = {}

        # For exact mode, we just need:
        #   - bucket_any_deepest[b]  = deepest among ALL events (MO deepest or non-MO price level)
        #   - bucket_has_mo[b]       = whether bucket contains >=1 MO
        bucket_any_deepest: Dict[int, int] = {}
        bucket_has_mo: Dict[int, bool] = {}

        for i in range(n):
            b = int(bucket_ids[i])

            if level_mode == "exact":
                k = K_exact_by_event[i]
                if np.isfinite(k):
                    kk = int(k)
                    if kk < 0:
                        pass
                    else:
                        if kk >= max_levels:
                            kk = max_levels - 1
                        prev = bucket_any_deepest.get(b, -10**18)
                        if kk > prev:
                            bucket_any_deepest[b] = kk
                if mo_mask[i]:
                    bucket_has_mo[b] = True

            else:
                # touch mode
                if mo_mask[i]:
                    d = D_by_event[i]
                    if np.isfinite(d):
                        D = int(d)
                        if D >= max_levels:
                            D = max_levels - 1
                        if D >= 0:
                            prev = bucket_mo_deepest.get(b, -10**18)
                            if D > prev:
                                bucket_mo_deepest[b] = D
                else:
                    k = K_price_by_event[i]
                    if np.isfinite(k):
                        kk = int(k)
                        if 0 <= kk < max_levels:
                            if b not in bucket_nonmo_levels:
                                bucket_nonmo_levels[b] = set()
                            bucket_nonmo_levels[b].add(kk)

        if level_mode == "exact":
            # One observation per bucket: (deepest level, label MO if bucket has any MO)
            for b in range(denom_total):
                if b not in bucket_any_deepest:
                    continue
                kk = int(bucket_any_deepest[b])
                if 0 <= kk < max_levels:
                    N_counts[kk] += 1
                    if bool(bucket_has_mo.get(b, False)):
                        M_counts[kk] += 1

        else:
            # touch mode: bucket touches a level if any event in bucket touches it.
            # We treat MO touches as an interval 0..kmax (or 0..kmax-1 if strict),
            # and non-MO touches as discrete levels.
            for b in range(denom_total):
                # MO interval contribution
                D = bucket_mo_deepest.get(b, None)
                kmax = -1
                if D is not None:
                    D = int(D)
                    kmax = (D - 1) if strict else D
                    if kmax >= max_levels:
                        kmax = max_levels - 1

                # First, count the MO interval (counts each bucket at most once per level)
                if kmax >= 0:
                    N_counts[: kmax + 1] += 1
                    M_counts[: kmax + 1] += 1

                # Then, count non-MO discrete levels that are OUTSIDE the MO interval
                s = bucket_nonmo_levels.get(b, None)
                if s:
                    for kk in s:
                        if kk > kmax:
                            N_counts[int(kk)] += 1

    # ---------------------------------------------------------------------
    # 6) Convert counts into conditional probabilities p_k = M_k / N_k
    # ---------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mo_given_level = M_counts.astype(np.float64) / N_counts.astype(np.float64)
    p_mo_given_level[N_counts == 0] = np.nan

    # Trim to last level with any support in N_counts
    nz = np.where(N_counts > 0)[0]
    if nz.size == 0:
        return {
            "p_mo_given_level": np.asarray([], dtype=np.float64),
            "n_level_touched": np.asarray([], dtype=np.int64),
            "n_level_touched_by_mo": np.asarray([], dtype=np.int64),
            "delta_grid": np.asarray([], dtype=np.float64),
            "denom_total": denom_total,
            "clock_unit": clock_unit,
            "aggregation_mode": aggregation_mode,
            "level_mode": level_mode,
            "strict": bool(strict),
            "collapsed_split_sweeps": bool(collapsed),
            "msg_df": msg_df,
            "ob_df": book_df,
        }

    K = int(nz.max()) + 1
    p_mo_given_level = p_mo_given_level[:K]
    N_counts = N_counts[:K]
    M_counts = M_counts[:K]
    delta_grid = (np.arange(K, dtype=np.float64) + 1.0) * float(half_tick)

    # ---------------------------------------------------------------------
    # 7) Plot
    # ---------------------------------------------------------------------
    if plot:
        algo_desc = _format_aggregation_label(
            aggregation_mode, time_interval, n_tob_moves, n_events_interval
        )
        prefix = plot_title_prefix if plot_title_prefix else f"{sim_desc}: "

        fig, ax = plt.subplots()
        ax.plot(delta_grid, p_mo_given_level, marker="o", markerfacecolor="none")
        ax.set_xlabel(r"$\delta$ (ticks from the mid-price)")
        ax.set_ylabel(r"$\Pr(\mathrm{MO}\mid \delta)$")
        ax.set_title(
            f"{prefix}{algo_desc} | {title}\n"
            f"level_mode={level_mode}, strict={strict}, clock={clock_unit}, denom_total={denom_total}, collapsed={collapsed}"
        )
        ax.grid(True, alpha=0.3)
        plt.show()

        if plot_counts:
            fig, ax = plt.subplots()
            ax.plot(delta_grid, N_counts, label="N_k (any-event touches)")
            ax.plot(delta_grid, M_counts, label="M_k (MO touches)")
            ax.set_xlabel(r"$\delta$ (ticks from the mid-price)")
            ax.set_ylabel("Counts (per decision step)")
            ax.set_title(
                f"{prefix}{algo_desc} | Support counts by level\n"
                f"(These are counts of steps where level was touched.)"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()

    # ---------------------------------------------------------------------
    # 8) Return
    # ---------------------------------------------------------------------
    return {
        "p_mo_given_level": p_mo_given_level,
        "n_level_touched": N_counts,             # N_k
        "n_level_touched_by_mo": M_counts,       # M_k
        "delta_grid": delta_grid,

        # bookkeeping / echo
        "denom_total": int(denom_total),
        "clock_unit": clock_unit,
        "aggregation_mode": aggregation_mode,
        "use_tob_buckets": bool(use_tob_buckets),
        "n_tob_moves": int(n_tob_moves),
        "n_events_interval": int(n_events_interval),
        "time_interval": float(time_interval),
        "half_tick": float(half_tick),
        "max_levels": int(max_levels),
        "level_mode": level_mode,
        "strict": bool(strict),
        "collapsed_split_sweeps": bool(collapsed),

        # keep dfs for debugging / reuse
        "msg_df": msg_df,
        "ob_df": book_df,
        "n_buckets_total": (None if bucket_ids is None else int(n_buckets_total)),
    }


# =========================================================================
# SECTION 14 — EXAMPLE USAGE
# =========================================================================
if __name__ == "__main__":
    LAM = 0.06
    MU = 0.1
    DELTA = 0.02

    NUMBER_TICK_LEVELS = 50
    N_PRIORITY_RANKS = 100

    N_STEPS = 200_000
    N_STEPS_TO_EQUIL = 10_000

    HALF_TICK = 0.5
    
    SEED = 12345
    
    random.seed(SEED)
    np.random.seed(SEED)

    # print("--- [EXAMPLE] Calibrate Intensity using Time Clock (1.0s) ---")
    
    # # 1. Fit Intensity using Time aggregation
    # res = fit_trading_intensity_parameters(
    #     queue_aware=False,
    #     # NEW ARGS:
    #     aggregation_mode="time", time_interval=10.0,
        
    #     lam=LAM, mu=MU, delta=DELTA,
    #     number_tick_levels=NUMBER_TICK_LEVELS,
    #     n_priority_ranks=N_PRIORITY_RANKS,
    #     n_steps=N_STEPS,
    #     n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
    #     half_tick=HALF_TICK,
    #     max_depth_levels=20,
    #     strict=True,
    #     mo_size=1,
    #     plot=True,
    #     # Removed hardcoded plot_title_prefix to test auto-labeling
    # )
    # print(f"Time-Based A: {res['A']:.4f}, Kappa: {res['kappa']:.4f}")
    
    # -------------------------------------------------------------------------
    # Queue-aware (tail counts / intensity by rank)
    # -------------------------------------------------------------------------
    
#     msg_df, ob_df, tail_counts_rank, n_events_used = calibrate_executability_santa_fe_middepth_tailcounts(
#     # --- Parâmetros do Simulador (Santa Fe) ---
#     lam=LAM, 
#     mu=MU, 
#     delta=DELTA,
#     number_tick_levels=NUMBER_TICK_LEVELS, 
#     n_priority_ranks=N_PRIORITY_RANKS,
#     n_steps=N_STEPS, 
#     n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
#     random_seed=SEED,

#     # --- Configurações do Agente/MM ---
#     mm_policy=None, 
#     exclude_self_from_state=True,  # O agente vê o livro SEM as próprias ordens

#     # --- Configurações da Curva de Execução ---
#     max_depth_levels=20,    # Profundidade máxima analisada (em ticks)
#     max_ranks=5,            # Número de posições na fila a analisar (Rank 0 a 4)
#     half_tick=HALF_TICK,    # Conversão de depth contínuo para discreto (k)
#     strict=True,            # True: Prob(Execute > k); False: Prob(Execute >= k)

#     # --- NOVOS MODOS DE AGREGAÇÃO ("CLOCKS") ---
#     aggregation_mode="time",  # Opções: "event", "tob", "time", "fixed_events"
#     time_interval=1.0,        # Intervalo de tempo em segundos (usado se mode="time")
    
#     # (Parâmetros ignorados neste modo, mas listados para completude)
#     n_tob_moves=10,           # Usado apenas se aggregation_mode="tob"
#     n_events_interval=100,    # Usado apenas se aggregation_mode="fixed_events"
#     use_tob_buckets=False,    # Legacy (manter False para usar aggregation_mode)

#     # --- Output e Visualização ---
#     plot=True,
#     output="intensity",             # Retorna 'intensity' (normalizado) ou 'counts' (bruto)
#     ranks_to_plot=(0, 1, 2, 3, 4),  # Quais ranks mostrar no gráfico
#     mo_size=5                       # Tamanho da MO simulada (afeta o consumo da fila)
# )


    # print("\n--- [EXAMPLE] Calibrate Volatility using Time Clock (1.0s) ---")
    # res_vol = fit_volatility(
    #     queue_aware=False,
    #     aggregation_mode="time", time_interval=2.0,
        
    #     lam=LAM, mu=MU, delta=DELTA,
    #     number_tick_levels=NUMBER_TICK_LEVELS,
    #     n_priority_ranks=N_PRIORITY_RANKS,
    #     n_steps=N_STEPS,
    #     n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
    #     plot=True,
    #     # Removed hardcoded prefix, letting the function handle the title
    # )
    # # CORRECTED LINE BELOW:
    # print(f"Sigma: {res_vol['volatility']:.4f} {res_vol['volatility_unit']}")


    # print("\n--- [EXAMPLE] Calibrate Volatility using Time Clock (1.0s) ---")
    # res_vol = fit_volatility(
    #     queue_aware=False,
    #     aggregation_mode="time", time_interval=10.0,
        
    #     lam=LAM, mu=MU, delta=DELTA,
    #     number_tick_levels=NUMBER_TICK_LEVELS,
    #     n_priority_ranks=N_PRIORITY_RANKS,
    #     n_steps=N_STEPS,
    #     n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
    #     plot=True
    # )
    # print(f"Sigma: {res_vol['volatility']:.4f} {res_vol['unit']}")

    print("\n--- [EXAMPLE] Volatility Signature using Time Clock ---")
    plot_volatility_signature_santa_fe(
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS, n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=N_STEPS, n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
        clock_mode="time", time_interval=1.0,
        n_paths=50, max_tau=1000, plot=True
    )