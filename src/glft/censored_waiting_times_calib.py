import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import random
from typing import Any, Dict, Optional, Tuple, List, Callable
import inspect
from MM_LOB_SIM import simulate_LOB_with_MM
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ---------------------------------------------------------------------------
# Imports from the trading-intensity calibration module.
# Used only in backtest mode for counterfactual fill detection:
#   - compute_nonqueue_deepest_tick_per_mo: computes the deepest tick each MO
#     reached on the pre-event book (the core building block for determining
#     whether a hypothetical LO at distance delta would have been filled).
#   - _maybe_collapse_split_sweeps: restores 1-row-per-event alignment when
#     backtest data was logged with split_sweeps=True.
# ---------------------------------------------------------------------------
from calibrate_trading_intensity import (
    compute_nonqueue_deepest_tick_per_mo,
    _maybe_collapse_split_sweeps,
)


# =============================================================================
# Basic utilities
# =============================================================================

def _seed_everything(seed: Optional[int]) -> None:
    """Seed numpy + Python `random` for reproducible simulator runs.

    Why: parts of the simulator / policies may use either RNG source.
    If you seed only numpy, you can still get non-reproducible paths if
    `random.*` is used anywhere.

    If seed is None, we do nothing.
    """
    if seed is None:
        return
    try:
        s = int(seed)
    except Exception:
        return
    random.seed(s)
    np.random.seed(s)


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
    """Run your Santa-Fe-compatible simulator with robust kwargs + deterministic seeding."""
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


# =============================================================================
# Robust parsing helpers (mm_df often stores lists as strings)
# =============================================================================

def _as_pylist(x: Any) -> List[Any]:
    """
    Robustly parse a cell that is supposed to be a Python list.

    Your mm_df columns like MM_OwnPrices/MM_OwnSides/MM_OwnQtys are often:
      - already Python lists (dtype=object), or
      - strings like "[124.0, 126.0]" when loaded from CSV.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            return []
    return [x]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_bool(x: Any) -> Optional[bool]:
    """
    Convert typical pandas/py objects to bool, returning None if ambiguous/NA.
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    try:
        return bool(int(x))
    except Exception:
        pass

    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("true", "t", "1", "yes", "y"):
            return True
        if t in ("false", "f", "0", "no", "n"):
            return False

    return None


# =============================================================================
# Robust fill detection from mm_df (primary: MM_HadFill; fallbacks: dInv/dCash)
# =============================================================================

def get_mm_fill_event(
    mm_df: pd.DataFrame,
    i: int,
    *,
    eps_inv: float = 1e-12,
    eps_cash: float = 1e-12,
) -> Dict[str, Any]:
    """
    Robust fill detector using mm_df telemetry ONLY.

    Returns:
      - had_fill: bool
      - fill_side: {+1 buy fill (our BID executed), -1 sell fill (our ASK executed), 0 unknown}
      - fill_price: float or nan (ONLY reliable when MM_HadFill==True)
      - fill_qty: abs(d_inventory) if available else nan
      - reason: explains detection source

    Why:
      - MM_LastFillPrice can be stale even when MM_HadFill is False.
      - Inventory / CashPnL jumps are strong signals of a trade fill.
    """
    n = len(mm_df)
    if i < 0 or i >= n:
        raise IndexError(f"i={i} out of bounds for mm_df with len={n}")

    hadfill_flag = None
    if "MM_HadFill" in mm_df.columns:
        hadfill_flag = _safe_bool(mm_df.iloc[i]["MM_HadFill"])

    d_inv = 0.0
    d_cash = 0.0

    inv_i = _safe_float(mm_df.iloc[i]["MM_Inventory"]) if "MM_Inventory" in mm_df.columns else float("nan")
    cash_i = _safe_float(mm_df.iloc[i]["MM_CashPnL"]) if "MM_CashPnL" in mm_df.columns else float("nan")

    if i > 0:
        inv_prev = _safe_float(mm_df.iloc[i - 1]["MM_Inventory"]) if "MM_Inventory" in mm_df.columns else float("nan")
        cash_prev = _safe_float(mm_df.iloc[i - 1]["MM_CashPnL"]) if "MM_CashPnL" in mm_df.columns else float("nan")
        if np.isfinite(inv_i) and np.isfinite(inv_prev):
            d_inv = inv_i - inv_prev
        if np.isfinite(cash_i) and np.isfinite(cash_prev):
            d_cash = cash_i - cash_prev

    inv_jump = (np.isfinite(d_inv) and abs(d_inv) > eps_inv)
    cash_jump = (np.isfinite(d_cash) and abs(d_cash) > eps_cash)

    if hadfill_flag is True:
        had_fill = True
        reason = "MM_HadFill"
    elif hadfill_flag is False:
        if inv_jump:
            had_fill = True
            reason = "inventory_jump (flag false)"
        elif cash_jump:
            had_fill = True
            reason = "cash_jump (flag false)"
        else:
            had_fill = False
            reason = "no_fill"
    else:
        if inv_jump:
            had_fill = True
            reason = "inventory_jump"
        elif cash_jump:
            had_fill = True
            reason = "cash_jump"
        else:
            had_fill = False
            reason = "no_fill"

    fill_side = 0
    fill_price = float("nan")

    # Only trust LastFillSide/LastFillPrice when MM_HadFill is True
    if had_fill and hadfill_flag is True:
        if "MM_LastFillSide" in mm_df.columns:
            s = _safe_float(mm_df.loc[i, "MM_LastFillSide"])
            if s > 0:
                fill_side = +1
            elif s < 0:
                fill_side = -1

        if "MM_LastFillPrice" in mm_df.columns:
            p = _safe_float(mm_df.loc[i, "MM_LastFillPrice"])
            if np.isfinite(p):
                fill_price = p

    # Infer side from inventory jump if still unknown
    if had_fill and fill_side == 0 and inv_jump:
        fill_side = +1 if d_inv > 0 else -1

    fill_qty = abs(d_inv) if (had_fill and inv_jump) else float("nan")

    return {
        "had_fill": bool(had_fill),
        "fill_side": int(fill_side),
        "fill_price": float(fill_price),
        "fill_qty": float(fill_qty),
        "reason": reason,
        "d_inventory": float(d_inv),
        "d_cashpnl": float(d_cash),
    }


def window_first_fill(
    mm_df: pd.DataFrame,
    i0: int,
    i1: int,
    *,
    target_side: Optional[int] = None,   # +1 bid-fill, -1 ask-fill
) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    """
    Scan [i0, i1) and return:
      (found_fill, first_fill_index, fill_event_dict)

    If target_side is given, accept fills only when fill_side is inferable
    and matches the target.
    """
    n = len(mm_df)
    i0 = max(0, int(i0))
    i1 = min(n, int(i1))
    if i0 >= i1:
        return (False, None, None)

    for i in range(i0, i1):
        ev = get_mm_fill_event(mm_df, i)
        if not ev["had_fill"]:
            continue

        if target_side in (+1, -1):
            # Skip fills where side is unknown (0) — conservative filtering
            if ev["fill_side"] not in (+1, -1):
                continue
            if ev["fill_side"] != target_side:
                continue

        return (True, i, ev)

    return (False, None, None)


# =============================================================================
# Clock + interval construction for throttling modes (event/fixed_events/time/tob)
# =============================================================================

def _get_time_series(msg_df: pd.DataFrame, mm_df: pd.DataFrame) -> np.ndarray:
    """
    Prefer msg_df['Time'] if present; else mm_df['Time']; else fallback to event index.
    """
    if msg_df is not None and "Time" in msg_df.columns:
        t = pd.to_numeric(msg_df["Time"], errors="coerce").to_numpy(float)
        return t

    if mm_df is not None and "Time" in mm_df.columns:
        t = pd.to_numeric(mm_df["Time"], errors="coerce").to_numpy(float)
        return t

    return np.arange(len(mm_df), dtype=float)


def _get_best_prices_for_tob(msg_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    For TOB move counting we need a best bid/ask series.
    Accept either:
      - BestBidPrice/BestAskPrice, or
      - IndBestBid/IndBestAsk
    """
    if msg_df is None:
        raise ValueError("TOB mode needs message_df with best bid/ask series.")

    for bid_col, ask_col in [("BestBidPrice", "BestAskPrice"), ("IndBestBid", "IndBestAsk")]:
        if bid_col in msg_df.columns and ask_col in msg_df.columns:
            bb = pd.to_numeric(msg_df[bid_col], errors="coerce").to_numpy(float)
            ba = pd.to_numeric(msg_df[ask_col], errors="coerce").to_numpy(float)
            return bb, ba

    raise ValueError("TOB mode: could not find BestBid/BestAsk columns in message_df.")


def build_clock(
    *,
    aggregation_mode: str,
    msg_df: pd.DataFrame,
    mm_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a monotone 'clock' series clock_t[i] aligned to event index i.

    - event / fixed_events: clock = i
    - time: clock = Time
    - tob:  clock = cumulative TOB moves (count changes in best bid OR best ask)
    """
    mode = str(aggregation_mode).lower().strip()
    n = len(mm_df)

    if mode in ("event", "fixed_events"):
        return np.arange(n, dtype=float)

    if mode == "time":
        return _get_time_series(msg_df, mm_df)

    if mode == "tob":
        bb, ba = _get_best_prices_for_tob(msg_df)
        if len(bb) != n or len(ba) != n:
            raise ValueError(
                f"TOB mode requires msg_df and mm_df aligned lengths. got len(bb)={len(bb)} len(mm_df)={n}"
            )

        moves = np.zeros(n, dtype=float)
        for i in range(1, n):
            moved = False
            if np.isfinite(bb[i]) and np.isfinite(bb[i-1]) and bb[i] != bb[i-1]:
                moved = True
            if np.isfinite(ba[i]) and np.isfinite(ba[i-1]) and ba[i] != ba[i-1]:
                moved = True
            moves[i] = 1.0 if moved else 0.0

        return np.cumsum(moves)

    raise ValueError(f"Unknown aggregation_mode='{aggregation_mode}'")


def build_intervals(
    *,
    aggregation_mode: str,
    clock_t: np.ndarray,
    n_events_interval: int,
    time_interval: float,
    n_tob_moves: int,
) -> Tuple[List[Tuple[int, int, float, float]], float, str]:
    """
    Build non-overlapping intervals [i0, i1) representing throttle windows.

    Returns:
      intervals: list of (i0, i1, c0, c1)
      DeltaT:    window length in the chosen clock unit
      unit:      "event", "second", or "tob-move"
    """
    mode = str(aggregation_mode).lower().strip()
    n = int(len(clock_t))
    intervals: List[Tuple[int, int, float, float]] = []

    if n <= 0:
        return intervals, 0.0, "none"

    if mode == "event":
        for i in range(n):
            c0 = float(clock_t[i])
            c1 = c0 + 1.0
            intervals.append((i, i + 1, c0, c1))
        return intervals, 1.0, "event"

    if mode == "fixed_events":
        if n_events_interval <= 0:
            raise ValueError("n_events_interval must be > 0 for fixed_events mode.")
        step = int(n_events_interval)
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            c0 = float(clock_t[i0])
            c1 = c0 + float(step)
            intervals.append((i0, i1, c0, c1))
        return intervals, float(step), "event"

    if mode == "time":
        if time_interval <= 0:
            raise ValueError("time_interval must be > 0 for time mode.")
        t0 = float(clock_t[0])
        t_end = float(clock_t[-1])

        k = 0
        i = 0
        while True:
            c0 = t0 + k * float(time_interval)
            c1 = c0 + float(time_interval)
            if c0 > t_end:
                break

            while i < n and clock_t[i] < c0:
                i += 1
            i0 = i

            j = i0
            while j < n and clock_t[j] < c1:
                j += 1
            i1 = j

            if i0 < n:
                intervals.append((i0, i1, c0, c1))
            k += 1
            if i0 >= n:
                break

        return intervals, float(time_interval), "second"

    if mode == "tob":
        if n_tob_moves <= 0:
            raise ValueError("n_tob_moves must be > 0 for tob mode.")
        c0 = float(clock_t[0])
        c_end = float(clock_t[-1])

        target = c0
        i = 0
        while target <= c_end:
            c1 = target + float(n_tob_moves)

            while i < n and clock_t[i] < target:
                i += 1
            i0 = i

            j = i0
            while j < n and clock_t[j] < c1:
                j += 1
            i1 = j

            if i0 < n:
                intervals.append((i0, i1, target, c1))
            target = c1
            if i0 >= n:
                break

        return intervals, float(n_tob_moves), "tob-move"

    raise ValueError(f"Unknown aggregation_mode='{aggregation_mode}'")


# =============================================================================
# Simulator adapter (NO NotImplemented): uses YOUR _run_simulator_santa_fe(...)
# =============================================================================

def _run_simulator_for_calibration(
    *,
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    n_steps: int,
    n_steps_to_equilibrium: int,
    random_seed: Optional[int],
    mm_policy,
    exclude_self_from_state: bool,
    mo_size: int,
    split_sweeps: bool,
):
    """
    Thin adapter that calls YOUR existing _run_simulator_santa_fe(...) exactly.

    It must return:
      msg_df, ob_df, mm_df
    """
    return _run_simulator_santa_fe(
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


# =============================================================================
# Probe policy: post ONE LO at distance δ from mid at the START of each window,
# then DO NOTHING (and DO NOT repost) until next window.
# =============================================================================

def censored_wait_probe_policy_factory(
    *,
    delta_value: float,
    half_tick: float,
    aggregation_mode: str,
    n_events_interval: int,
    time_interval: float,
    n_tob_moves: int,

    # Which side to probe:
    #   - "buy":  post BID only (we detect fills as +1 inventory jumps)
    #   - "sell": post ASK only
    side_mode: str = "buy",

    # These MUST match your MarketMaker action vocabulary:
    action_hold: str = "hold",
    action_place_bid: str = "place_bid",
    action_place_ask: str = "place_ask",
) -> Callable:
    """
    Probe policy for censored-waiting-time calibration.

    Critical behavior:
      - At the START of each throttle window, we post exactly ONE order at distance δ.
      - If it fills, we DO NOT repost until the next window boundary.
      - This makes each window a single "attempt", matching the censoring model.

    Implementation details:
      - We compute mid from mm.lob.compute_mid_price() when available.
      - We build a window-id from the chosen throttle mode using mm.lob:
          * event: window changes every step
          * fixed_events: step//n_events_interval
          * time: floor(current_time / time_interval)
          * tob:  floor(tob_moves / n_tob_moves) where tob_moves counts L1 changes
    """
    mode = str(aggregation_mode).lower().strip()
    side_mode = str(side_mode).lower().strip()
    if side_mode not in ("buy", "sell"):
        raise ValueError("side_mode must be 'buy' or 'sell' for this probe policy.")

    # Policy state
    step_idx = -1
    current_window_id = None
    placed_this_window = False
    filled_this_window = False
    placed_step = -10**9  # event index when we placed in current window
    tob_moves = 0
    last_bb = None
    last_ba = None

    def _current_time_from_mm(mm: Any) -> float:
        """
        Use the last written time in the tape if possible; else fallback to step index.
        At pre_step, the tape already includes the previous event time.
        """
        try:
            tlist = mm.lob.message_dict.get("Time", None)
            if isinstance(tlist, list) and len(tlist) > 0:
                t = float(tlist[-1])
                if np.isfinite(t):
                    return t
        except Exception:
            pass
        return float(step_idx)

    def _best_bid_ask_from_book(mm: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute L1 bid/ask from message_dict (same source as post-hoc msg_df),
        falling back to lob_state.

        Using message_dict keeps the TOB-move counting in the policy aligned
        with build_clock's post-hoc counting.  Both see the state logged AFTER
        the previous event, so window boundaries stay consistent.

        Known minor limitation: the policy is called BEFORE the current market
        event is processed (pre_step), so it reads the state of event i-1.
        message_dict[-1] at that moment also reflects event i-1.  This means
        both the policy and post-hoc detect the same TOB moves, but a window
        boundary triggered by event i-1's move is registered by the policy at
        step i (1 event late).  The resulting ~1-event offset on window starts
        is negligible for window sizes of n_tob_moves >= 5.
        """
        # Prefer message_dict: same data source as post-hoc build_clock
        try:
            for bid_col, ask_col in [("BestBidPrice", "BestAskPrice"),
                                      ("IndBestBid", "IndBestAsk")]:
                bd = mm.lob.message_dict.get(bid_col, None)
                ad = mm.lob.message_dict.get(ask_col, None)
                if bd and ad and len(bd) > 0 and len(ad) > 0:
                    bb_raw = bd[-1]
                    ba_raw = ad[-1]
                    if (bb_raw is not None and np.isfinite(float(bb_raw))
                            and float(bb_raw) >= 0
                            and ba_raw is not None and np.isfinite(float(ba_raw))
                            and float(ba_raw) >= 0):
                        return int(float(bb_raw)), int(float(ba_raw))
        except Exception:
            pass
        # Fallback: lob_state
        try:
            st = mm.lob.lob_state
            bid_idxs = np.where(st > 0)[0]
            ask_idxs = np.where(st < 0)[0]
            bb = int(bid_idxs[-1]) if bid_idxs.size > 0 else None
            ba = int(ask_idxs[0])  if ask_idxs.size > 0 else None
            return bb, ba
        except Exception:
            return None, None

    def _mid_from_book(mm: Any) -> float:
        """
        Use engine mid-price (can be x.5 in odd spread).
        """
        try:
            m = float(mm.lob.compute_mid_price())
            return m
        except Exception:
            # fallback: average of best bid/ask
            bb, ba = _best_bid_ask_from_book(mm)
            if bb is None or ba is None:
                return float("nan")
            return 0.5 * (float(bb) + float(ba))

    def _window_id(mm: Any) -> int:
        nonlocal tob_moves, last_bb, last_ba

        if mode == "event":
            return int(step_idx)

        if mode == "fixed_events":
            return int(step_idx) // int(n_events_interval)

        if mode == "time":
            t = _current_time_from_mm(mm)
            return int(np.floor(t / float(time_interval)))

        if mode == "tob":
            bb, ba = _best_bid_ask_from_book(mm)
            moved = False
            if bb is not None:
                if last_bb is None:
                    last_bb = bb
                elif bb != last_bb:
                    moved = True
                    last_bb = bb
            if ba is not None:
                if last_ba is None:
                    last_ba = ba
                elif ba != last_ba:
                    moved = True
                    last_ba = ba
            if moved:
                tob_moves += 1
            return int(tob_moves) // int(n_tob_moves)

        raise ValueError(f"Unknown aggregation_mode='{aggregation_mode}' in probe policy.")

    def _quote_price(mid: float) -> int:
        """
        Convert mid +/- δ to an integer tick index WITHOUT crossing.

        We want:
          bid_price = floor(mid - δ)
          ask_price = ceil(mid + δ)

        This avoids rounding bias when mid is .5 and δ is half-tick grid.
        """
        if not np.isfinite(mid):
            return -1
        if side_mode == "buy":
            return int(np.floor(mid - float(delta_value)))
        else:
            return int(np.ceil(mid + float(delta_value)))

    def policy(state: Any, mm: Any = None):
        nonlocal step_idx, current_window_id, placed_this_window, filled_this_window, placed_step

        step_idx += 1
        if mm is None or not hasattr(mm, "lob"):
            # Without mm we cannot compute mid reliably; safest is to do nothing.
            return (action_hold, 0, -1)

        wid = _window_id(mm)

        # New window -> reset attempt flags
        if current_window_id is None or wid != current_window_id:
            current_window_id = wid
            placed_this_window = False
            filled_this_window = False
            placed_step = -10**9

        # Detect if a fill already happened in this window (from previous step)
        # mm.last_fill is set in your runner when passive fill occurs.
        if (not filled_this_window) and placed_this_window:
            try:
                lf = getattr(mm, "last_fill", None)
                if isinstance(lf, dict):
                    s = int(lf.get("step", -10**9))
                    # fill must occur after we placed in this window
                    if s >= placed_step and s >= 0:
                        filled_this_window = True
            except Exception:
                pass

        # If we already got filled, do nothing until next window.
        if filled_this_window:
            return (action_hold, 0, -1)

        # Place exactly once per window (start-of-window action)
        if not placed_this_window:
            mid = _mid_from_book(mm)
            px = _quote_price(mid)
            placed_this_window = True
            placed_step = int(step_idx)

            if side_mode == "buy":
                return ("cancel_all_then_place", +1, int(px))
            else:
                return ("cancel_all_then_place", -1, int(px))

        # Otherwise hold
        return (action_hold, 0, -1)

    return policy


# =============================================================================
# BACKTEST HELPER — counterfactual (virtual) fill detection for one window
# =============================================================================
# In simulation mode the probe policy physically places an LO in the book, and
# fills are detected from mm_df (the MarketMaker telemetry DataFrame).
#
# In backtest mode there is NO actual order placement.  Instead, we ask a
# counterfactual question:
#
#     "If I had placed a limit order at distance delta from the window-start
#      mid-price, would any incoming market order have swept deep enough to
#      fill me during this window?"
#
# This is the ZERO MARKET-IMPACT ASSUMPTION: the hypothetical LO does not
# change the book state, so MO behavior (depth, timing) is identical to what
# was observed in the historical data.  This is a standard assumption for
# small-order-size analysis and is consistent with the probe-policy design
# in simulation mode (which also posts a single 1-lot order).
#
# The helper below encapsulates this logic for a single window [i0, i1).
# It is called once per window per delta inside the backtest delta loop.
# =============================================================================

def _backtest_window_first_virtual_fill(
    *,
    i0: int,
    i1: int,
    delta_value: float,
    half_tick: float,
    target_side: int,
    D_by_event: "np.ndarray",
    direction_arr: "np.ndarray",
    mid_pre_arr: "np.ndarray",
    mid_0: float,
) -> Tuple[bool, Optional[int]]:
    """Counterfactual fill detection for a single throttle window.

    Scans events [i0, i1) looking for the FIRST market order that would have
    filled a hypothetical limit order sitting at distance ``delta_value`` from
    the window-start mid-price ``mid_0``.

    Parameters
    ----------
    i0, i1 : int
        Half-open event index range defining the window: [i0, i1).
    delta_value : float
        Distance (in price/tick units) from the window-start mid where the
        hypothetical LO is placed.
    half_tick : float
        Tick size / 2.  Used for the depth-to-tick-index conversion:
            tick_index = round(depth / half_tick) - 1
    target_side : int
        +1 if the hypothetical LO is a BID (buy) order.
        -1 if the hypothetical LO is an ASK (sell) order.
        This determines which MO direction can fill us:
            target_side = +1 (buy LO on BID) -> only SELL MOs (direction < 0)
            target_side = -1 (sell LO on ASK) -> only BUY MOs (direction > 0)
    D_by_event : np.ndarray
        Pre-computed array of shape (n_events,).  D_by_event[i] is the deepest
        tick index that MO at event i reached on the pre-event book.  NaN for
        non-MO events.  Computed by ``compute_nonqueue_deepest_tick_per_mo``.
    direction_arr : np.ndarray
        msg_df["Direction"] as a float array.  Positive = BUY MO (sweeps ask),
        negative = SELL MO (sweeps bid).
    mid_pre_arr : np.ndarray
        Pre-event mid-price for each event: mid_pre[i] = MidPrice[i] - Return[i].
    mid_0 : float
        Mid-price at the start of the window (= mid_pre_arr[i0]).  The
        hypothetical LO is placed at mid_0 ± delta_value.

    Returns
    -------
    (found, i_fill) : Tuple[bool, Optional[int]]
        found : True if a qualifying MO was found.
        i_fill : event index of the first qualifying MO, or None.

    Algorithm — Mid-Drift Correction
    ---------------------------------
    The hypothetical LO is placed at a FIXED price at the start of the window:

        P_lo = mid_0 - delta_value    (for a buy LO on the BID)
        P_lo = mid_0 + delta_value    (for a sell LO on the ASK)

    During the window the mid-price may drift.  At event i, the pre-event mid
    is mid_pre[i] ≠ mid_0.  The MO at event i measures depth from its own
    pre-event mid, not from mid_0.  So the "effective delta" — the distance
    from the MO's reference mid to the LO's fixed price — is:

        Buy LO:  delta_eff = delta_value + (mid_pre[i] - mid_0)
                 Intuition: if mid drifted UP, the LO is now further away
                 from the MO's reference point → harder to fill.

        Sell LO: delta_eff = delta_value + (mid_0 - mid_pre[i])
                 Intuition: if mid drifted DOWN, the LO is now further away.

    The MO fills the LO if the depth it actually reached (in price units)
    exceeds delta_eff:

        (D_by_event[i] + 1) * half_tick >= delta_eff

    Special case: if delta_eff <= 0, the mid has drifted past the LO price,
    meaning the LO is now at or through the mid — any MO on the correct side
    would fill it immediately.
    """
    import numpy as _np  # local alias for speed in tight loop

    for i in range(i0, i1):
        # Skip non-MO events (D is NaN) and events with unknown direction.
        d_tick = D_by_event[i]
        if not _np.isfinite(d_tick):
            continue

        d_dir = direction_arr[i]

        # Direction filter:
        #   Buy LO (target_side=+1) is filled by SELL MOs (direction < 0).
        #   Sell LO (target_side=-1) is filled by BUY MOs (direction > 0).
        if target_side == +1 and d_dir >= 0.0:
            continue  # need SELL MO for a buy LO
        if target_side == -1 and d_dir <= 0.0:
            continue  # need BUY MO for a sell LO

        # Mid-drift correction: compute the effective delta from the MO's
        # own pre-event mid to the hypothetical LO's fixed price.
        mid_i = mid_pre_arr[i]
        if not _np.isfinite(mid_i):
            continue

        if target_side == +1:
            # Buy LO at price P = mid_0 - delta_value.
            # Effective delta from mid_i: delta_eff = delta_value + (mid_i - mid_0)
            delta_eff = delta_value + (mid_i - mid_0)
        else:
            # Sell LO at price P = mid_0 + delta_value.
            # Effective delta from mid_i: delta_eff = delta_value + (mid_0 - mid_i)
            delta_eff = delta_value + (mid_0 - mid_i)

        # If delta_eff <= 0, the mid has drifted past the LO → trivial fill.
        if delta_eff <= 0.0:
            return (True, i)

        # Check if the MO swept deep enough to reach the LO.
        # D_by_event[i] is a tick INDEX: tick = round(depth/half_tick) - 1.
        # So the actual depth in price units is (D + 1) * half_tick.
        depth_reached = (d_tick + 1.0) * half_tick
        if depth_reached >= delta_eff:
            return (True, i)

    # No qualifying MO found in this window → censored observation.
    return (False, None)


# =============================================================================
# MAIN: censored waiting times calibration across a grid of deltas
# =============================================================================

def fit_execution_intensity_censored_waiting_times(
    *,
    # ---------------------------------------------------------------------
    # DATA SOURCE CONTROL
    # Same dual-mode API as fit_trading_intensity_parameters and fit_volatility
    # in calibrate_trading_intensity.py.
    # ---------------------------------------------------------------------
    data_mode: str = "simulation",                     # "simulation" or "backtest"
    message_df: Optional[pd.DataFrame] = None,         # required when data_mode="backtest"
    ob_df_input: Optional[pd.DataFrame] = None,        # required when data_mode="backtest"
    collapse_split_sweeps: bool = True,                # collapse NaN detail rows from split_sweeps

    # ---------------------------------------------------------------------
    # THROTTLE / AGGREGATION
    # ---------------------------------------------------------------------
    aggregation_mode: str = "fixed_events",   # "event", "tob", "time", "fixed_events"
    n_tob_moves: int = 10,
    n_events_interval: int = 100,
    time_interval: float = 1.0,

    # ---------------------------------------------------------------------
    # DELTA GRID
    # ---------------------------------------------------------------------
    max_levels: int = 50,
    half_tick: float = 0.5,

    # ---------------------------------------------------------------------
    # EXPERIMENT CONTROL
    # ---------------------------------------------------------------------
    side_mode: str = "buy",          # "buy" (BID only) or "sell" (ASK only)
    burn_in_windows: int = 5,        # ignore first windows (stabilize)

    # ---------------------------------------------------------------------
    # SIMULATION PARAMS — ignored when data_mode="backtest"
    # ---------------------------------------------------------------------
    lam: float = 0.05,
    mu: float = 0.05,
    delta: float = 0.05,
    number_tick_levels: int = 100,
    n_priority_ranks: int = 20,
    n_steps: int = 200_000,
    n_steps_to_equilibrium: int = 2_000,
    random_seed: Optional[int] = None,
    exclude_self_from_state: bool = True,
    mo_size: int = 1,
    split_sweeps: bool = False,  # keep False for alignment in simulation mode

    # ---------------------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------------------
    plot: bool = True,
    plot_title_prefix: str = "",
    title: str = r"Censored waiting times: $\hat\lambda(\delta)$",
    use_tqdm: bool = False,
    tqdm_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Calibrate execution intensity λ(δ) via censored waiting times.

    This function supports two data sources:

    data_mode="simulation"  (default)
        For each delta in the grid, runs ONE dedicated Santa-Fe simulation
        with a "probe policy" that posts exactly ONE limit order per throttle
        window at distance delta from mid.  Fills are detected physically
        from the MarketMaker telemetry DataFrame (mm_df).  This is the
        Fer15-style direct measurement approach.

    data_mode="backtest"
        Uses externally provided ``message_df`` and ``ob_df_input`` from a
        single real data replay (or previous simulation run).  For each delta,
        the SAME data is replayed with **counterfactual fill detection**:

            "If I had placed a hypothetical LO at distance delta from the
             window-start mid, would any incoming MO have swept deep enough
             to fill me during this window?"

        This avoids running N separate simulations (one per delta).  Instead,
        MO depth is pre-computed ONCE via ``compute_nonqueue_deepest_tick_per_mo``
        and reused for every delta — a significant speedup.

        **Zero market-impact assumption:** the hypothetical LO does not alter
        the book state, so MO behavior (depth, timing) is identical to what
        was observed in the data.  This is reasonable for small order sizes.

        **Mid-drift correction:** the LO is placed at a FIXED price at the
        window start.  During the window the mid may drift.  For each MO at
        event i, we compute the "effective delta" accounting for the mid
        shift since the window start:

            Buy LO:  delta_eff = delta + (mid_pre[i] - mid_0)
            Sell LO: delta_eff = delta + (mid_0 - mid_pre[i])

        The MO fills the LO if its sweep depth ≥ delta_eff.

    MLE Formula (identical for both modes)
    ---------------------------------------
    For each delta and each throttle window:
        - If the LO is filled before ΔT:  τ = actual fill time (uncensored)
        - If the LO is NOT filled by ΔT:  τ = ΔT (right-censored)

    Under the exponential censoring model:
        λ̂(δ) = (#uncensored fills) / Σ_windows min(τ, ΔT)

    Parameters
    ----------
    data_mode : str
        "simulation" (default) or "backtest".
    message_df : pd.DataFrame or None
        Required when data_mode="backtest".  Must contain columns:
        MidPrice, Return, Direction, Type.  Optionally Time (for time-clock),
        BestBidPrice/BestAskPrice (for tob-clock).
    ob_df_input : pd.DataFrame or None
        Required when data_mode="backtest".  Must contain BidPrice_*/BidSize_*
        and AskPrice_*/AskSize_* columns for pre-event book reconstruction.
    collapse_split_sweeps : bool
        If True (default) and backtest data contains NaN detail rows from
        split_sweeps=True, collapse them to 1 row per event.
    aggregation_mode : str
        Clock / throttle mode: "event", "fixed_events", "time", "tob".
    side_mode : str
        "buy" (probe the BID side) or "sell" (probe the ASK side).
    burn_in_windows : int
        Number of initial windows to discard (allow the book to stabilize).
    use_tqdm : bool
        If True, show progress bars over the delta loop (when `tqdm` is
        available in the environment).
    tqdm_desc : Optional[str]
        Optional custom tqdm description text (e.g., include throttle/repeat
        context from an outer loop).

    Returns
    -------
    dict with keys:
        "delta_grid", "lambda_hat", "n_exec", "denom_sum_tau",
        "n_windows_used", "aggregation_mode", "data_mode",
        "side_mode", "burn_in_windows", "runs_debug", ...
    """
    # =================================================================
    # VALIDATION — common to both modes
    # =================================================================
    mode = str(aggregation_mode).lower().strip()
    if mode not in ("event", "fixed_events", "time", "tob"):
        raise ValueError("aggregation_mode must be one of: event, fixed_events, time, tob")

    side_mode = str(side_mode).lower().strip()
    if side_mode not in ("buy", "sell"):
        raise ValueError("side_mode must be 'buy' or 'sell'")

    data_mode = str(data_mode).lower().strip()
    if data_mode not in ("simulation", "backtest"):
        raise ValueError(
            f"data_mode must be 'simulation' or 'backtest', got '{data_mode}'."
        )

    # split_sweeps is a simulation-only concern: it controls alignment
    # between msg_df and mm_df.  In backtest mode, we handle split sweeps
    # via collapse_split_sweeps instead.
    if data_mode == "simulation" and split_sweeps:
        raise ValueError(
            "For simulation mode, use split_sweeps=False to keep "
            "msg_df/mm_df aligned 1:1."
        )

    # =================================================================
    # BACKTEST VALIDATION — required DataFrames and columns
    # =================================================================
    if data_mode == "backtest":
        if message_df is None or ob_df_input is None:
            raise ValueError(
                "For data_mode='backtest', you must provide both "
                "message_df and ob_df_input DataFrames."
            )
        # Required columns for counterfactual fill detection:
        #   MidPrice + Return  -> reconstruct pre-event mid (mid_pre = Mid - Ret)
        #   Direction          -> filter MOs by side (buy vs sell)
        #   Type               -> identify market orders (Type == 1 or "MO")
        for col in ("MidPrice", "Return", "Direction", "Type"):
            if col not in message_df.columns:
                raise ValueError(
                    f"Backtest message_df must contain column '{col}'. "
                    f"Available columns: {list(message_df.columns)}"
                )
        # ob_df_input needs orderbook columns for the depth computation
        # (compute_nonqueue_deepest_tick_per_mo reads BidPrice_*/AskPrice_*)
        has_bid = any(c.startswith("BidPrice_") for c in ob_df_input.columns)
        has_ask = any(c.startswith("AskPrice_") for c in ob_df_input.columns)
        if not (has_bid and has_ask):
            raise ValueError(
                "Backtest ob_df_input must contain BidPrice_*/AskPrice_* "
                "columns for pre-event book reconstruction."
            )
        if len(message_df) != len(ob_df_input):
            raise ValueError(
                f"message_df and ob_df_input must have the same length. "
                f"Got {len(message_df)} and {len(ob_df_input)}."
            )

    # =================================================================
    # δ GRID + OUTPUT ARRAYS — shared by both modes
    # =================================================================
    # δ_k = (k+1) * half_tick  for k = 0, 1, ..., max_levels-1
    # Example with half_tick=0.5, max_levels=10:
    #     delta_grid = [0.5, 1.0, 1.5, ..., 5.0]
    k = np.arange(int(max_levels), dtype=float)
    delta_grid = (k + 1.0) * float(half_tick)

    lambda_hat = np.zeros_like(delta_grid, dtype=float)
    n_exec = np.zeros_like(delta_grid, dtype=int)
    denom_sum = np.zeros_like(delta_grid, dtype=float)
    n_windows_used = np.zeros_like(delta_grid, dtype=int)

    runs_debug = []

    # =================================================================
    # BACKTEST PRE-COMPUTATION — done ONCE, reused for all deltas
    # =================================================================
    # In simulation mode, each delta gets its own simulation run (with a
    # dedicated probe policy).  In backtest mode, the data is fixed: we
    # pre-compute the expensive quantities ONCE before the delta loop.
    #
    # Key pre-computed arrays:
    #   D_by_event   : deepest tick index each MO reached (NaN for non-MOs)
    #   direction_arr: msg_df["Direction"] — sign tells us BUY vs SELL MO
    #   mid_pre_arr  : pre-event mid = MidPrice - Return for each event
    #   bt_clock_t   : monotone clock values for the chosen aggregation mode
    #   bt_intervals  : list of (i0, i1, c0, c1) throttle windows
    #
    # The clock and intervals are also built once (not per-delta like in
    # simulation mode) because the same data is replayed for every delta.
    # =================================================================
    if data_mode == "backtest":
        # 1) Local copies + optional split-sweep collapse
        bt_msg_df = message_df.copy()
        bt_ob_df = ob_df_input.copy()

        if collapse_split_sweeps:
            bt_msg_df, bt_ob_df, _did_collapse = _maybe_collapse_split_sweeps(
                bt_msg_df, bt_ob_df
            )

        bt_n = len(bt_msg_df)
        if bt_n < 10:
            raise ValueError("Not enough events in backtest data to calibrate.")

        # 2) Compute deepest tick per MO event — the heavy computation.
        #    D_by_event[i] = NaN for non-MO rows; integer tick index for MOs.
        #    The tick index follows the convention:
        #        tick = round(depth_in_price_units / half_tick) - 1
        #    So tick=0 corresponds to the best opposing level, tick=1 to the
        #    next level out, etc.
        D_by_event, _n_mo_used = compute_nonqueue_deepest_tick_per_mo(
            message_df=bt_msg_df,
            ob_df=bt_ob_df,
            max_depth_levels=int(max_levels),
            half_tick=float(half_tick),
        )

        # 3) Pre-extract arrays for fast inner-loop access.
        #    direction_arr: positive = BUY MO (sweeps ASK), negative = SELL MO (sweeps BID)
        #    mid_pre_arr: mid-price BEFORE event i (reconstructed from message log)
        direction_arr = bt_msg_df["Direction"].to_numpy(dtype=float)
        mid_arr = bt_msg_df["MidPrice"].to_numpy(dtype=float)
        ret_arr = bt_msg_df["Return"].to_numpy(dtype=float)
        mid_pre_arr = mid_arr - ret_arr

        # 4) Build clock + intervals ONCE.
        #    build_clock requires mm_df; in backtest mode we pass bt_msg_df
        #    because:
        #      - event/fixed_events mode: uses len(mm_df) -> len(bt_msg_df) ✓
        #      - time mode: tries msg_df['Time'] first -> correct ✓
        #      - tob mode: reads bid/ask from msg_df only -> correct ✓
        bt_clock_t = build_clock(
            aggregation_mode=mode,
            msg_df=bt_msg_df,
            mm_df=bt_msg_df,
        )
        bt_intervals, bt_DeltaT, bt_unit = build_intervals(
            aggregation_mode=mode,
            clock_t=bt_clock_t,
            n_events_interval=int(n_events_interval),
            time_interval=float(time_interval),
            n_tob_moves=int(n_tob_moves),
        )
        if len(bt_intervals) == 0 or bt_DeltaT <= 0:
            raise ValueError(
                "No intervals built from backtest data; check throttle parameters."
            )

        bt_intervals_used = bt_intervals[int(max(0, burn_in_windows)):]
        if len(bt_intervals_used) == 0:
            raise ValueError("All windows removed by burn_in_windows.")

        # target_side for direction filtering inside the virtual fill helper:
        #   +1 = buy LO (BID) -> only SELL MOs (direction < 0) can fill
        #   -1 = sell LO (ASK) -> only BUY MOs (direction > 0) can fill
        bt_target_side = +1 if side_mode == "buy" else -1

    # =================================================================
    # DELTA LOOP — SIMULATION MODE
    # =================================================================
    # For each delta, create a dedicated probe policy, run a full
    # simulation, and detect fills from the MarketMaker telemetry (mm_df).
    # This path is the original Fer15-style measurement and is UNCHANGED.
    # =================================================================
    if data_mode == "simulation":
      if use_tqdm and (tqdm is not None):
        _desc = str(tqdm_desc) if (tqdm_desc is not None) else f"censored-calib[{mode}] simulation"
        _position = 1 if (tqdm_desc is not None) else 0
        _delta_iter = tqdm(
            enumerate(delta_grid),
            total=len(delta_grid),
            desc=_desc,
            position=_position,
            leave=False,
        )
      else:
        _delta_iter = enumerate(delta_grid)

      for idx, delta_value in _delta_iter:
        if use_tqdm and (tqdm is not None):
            try:
                _dist = float(delta_value)
                _unit = "tick" if np.isclose(_dist, 1.0) else "ticks"
                _delta_iter.set_postfix_str(f"dist mid: {_dist:g} {_unit}")
            except Exception:
                pass
        # -------------------------------------------------------------
        # 1) Build probe policy for THIS δ and simulate
        # -------------------------------------------------------------
        mm_policy = censored_wait_probe_policy_factory(
            delta_value=float(delta_value),
            half_tick=float(half_tick),
            aggregation_mode=mode,
            n_events_interval=int(n_events_interval),
            time_interval=float(time_interval),
            n_tob_moves=int(n_tob_moves),
            side_mode=side_mode,
            # Must match your MarketMaker action strings:
            action_hold="hold",
            action_place_bid="place_bid",
            action_place_ask="place_ask",
        )

        msg_df, ob_df, mm_df = _run_simulator_for_calibration(
            lam=float(lam),
            mu=float(mu),
            delta=float(delta),
            number_tick_levels=int(number_tick_levels),
            n_priority_ranks=int(n_priority_ranks),
            n_steps=int(n_steps),
            n_steps_to_equilibrium=int(n_steps_to_equilibrium),
            random_seed=random_seed if random_seed is not None else None,
            mm_policy=mm_policy,
            exclude_self_from_state=bool(exclude_self_from_state),
            mo_size=int(mo_size),
            split_sweeps=bool(split_sweeps),
        )

        if msg_df is None or mm_df is None or getattr(mm_df, "empty", True):
            raise ValueError("Simulator returned empty dataframes.")

        if len(msg_df) != len(mm_df):
            raise ValueError(
                "Need msg_df and mm_df aligned by event index.\n"
                f"Got len(msg_df)={len(msg_df)} and len(mm_df)={len(mm_df)}.\n"
                "Ensure split_sweeps=False for this calibration."
            )

        n = len(mm_df)
        if n < 10:
            raise ValueError("Not enough events produced to calibrate.")

        # -------------------------------------------------------------
        # 2) Build clock + throttle intervals
        # -------------------------------------------------------------
        clock_t = build_clock(aggregation_mode=mode, msg_df=msg_df, mm_df=mm_df)

        intervals, DeltaT, unit = build_intervals(
            aggregation_mode=mode,
            clock_t=clock_t,
            n_events_interval=int(n_events_interval),
            time_interval=float(time_interval),
            n_tob_moves=int(n_tob_moves),
        )
        if len(intervals) == 0 or DeltaT <= 0:
            raise ValueError("No intervals built; check throttle parameters.")

        # burn-in
        intervals_used = intervals[int(max(0, burn_in_windows)):]
        if len(intervals_used) == 0:
            raise ValueError("All windows removed by burn_in_windows.")

        # -------------------------------------------------------------
        # 3) For each window: detect first fill and compute censored τ
        # -------------------------------------------------------------
        I_sum = 0
        tau_sum = 0.0
        win_count = 0

        # We only probe ONE side here -> target_side is fixed:
        target_side = +1 if side_mode == "buy" else -1

        for (i0, i1, c0, c1) in intervals_used:
            if i0 >= n:
                break

            # τ is measured from the actual placement step (i0), not the
            # theoretical window boundary c0.  In "time" mode the first event
            # in the window can fall slightly after c0, so using c0 would
            # overestimate τ.  Censoring uses the remaining window from i0.
            t_placed = float(clock_t[i0])
            effective_DeltaT = float(c1) - t_placed
            if not (effective_DeltaT > 0.0):
                effective_DeltaT = float(DeltaT)

            found, i_fill, ev = window_first_fill(mm_df, i0, i1, target_side=target_side)

            if found and i_fill is not None:
                tau = float(clock_t[i_fill]) - t_placed
                if not np.isfinite(tau) or tau <= 0.0:
                    # τ ≤ 0: fill at placement step or unresolvable timing.
                    # Treat as censored to avoid contributing 0 to the MLE
                    # denominator (would inflate λ̂ → ∞ for that window).
                    tau_tilde = effective_DeltaT
                    I = 0
                else:
                    tau_tilde = min(float(tau), float(effective_DeltaT))
                    I = 1 if tau < float(effective_DeltaT) else 0
            else:
                tau_tilde = float(effective_DeltaT)
                I = 0

            I_sum += int(I)
            tau_sum += float(tau_tilde)
            win_count += 1

        # MLE
        if win_count > 0 and tau_sum > 0:
            lam_hat = float(I_sum) / float(tau_sum)
        else:
            lam_hat = 0.0

        lambda_hat[idx] = lam_hat
        n_exec[idx] = int(I_sum)
        denom_sum[idx] = float(tau_sum)
        n_windows_used[idx] = int(win_count)

        runs_debug.append({
            "delta": float(delta_value),
            "lambda_hat": float(lam_hat),
            "n_exec": int(I_sum),
            "tau_sum": float(tau_sum),
            "n_windows": int(win_count),
            "DeltaT": float(DeltaT),
            "unit": str(unit),
        })

    # =================================================================
    # DELTA LOOP — BACKTEST MODE (counterfactual fill detection)
    # =================================================================
    # In backtest mode, we do NOT run simulations.  Instead, for each
    # delta we scan the pre-computed D_by_event array to determine
    # whether a hypothetical LO at that distance would have been filled
    # in each window.
    #
    # The MLE formula is IDENTICAL to simulation mode:
    #     λ̂(δ) = (#uncensored fills) / Σ min(τ̃, ΔT)
    #
    # The only difference is HOW fills are detected:
    #   Simulation:  window_first_fill(mm_df, i0, i1, target_side)
    #   Backtest:    _backtest_window_first_virtual_fill(D_by_event, ...)
    #
    # Because D_by_event, clock, and intervals are pre-computed once,
    # the backtest loop is O(max_levels * total_events) — much faster
    # than simulation mode which runs O(max_levels) full simulations.
    # =================================================================
    elif data_mode == "backtest":
      if use_tqdm and (tqdm is not None):
        _desc = str(tqdm_desc) if (tqdm_desc is not None) else f"censored-calib[{mode}] backtest"
        _position = 1 if (tqdm_desc is not None) else 0
        _delta_iter = tqdm(
            enumerate(delta_grid),
            total=len(delta_grid),
            desc=_desc,
            position=_position,
            leave=False,
        )
      else:
        _delta_iter = enumerate(delta_grid)

      for idx, delta_value in _delta_iter:
        if use_tqdm and (tqdm is not None):
            try:
                _dist = float(delta_value)
                _unit = "tick" if np.isclose(_dist, 1.0) else "ticks"
                _delta_iter.set_postfix_str(f"dist mid: {_dist:g} {_unit}")
            except Exception:
                pass
        I_sum = 0
        tau_sum = 0.0
        win_count = 0

        for (i0, i1, c0, c1) in bt_intervals_used:
            if i0 >= bt_n:
                break

            # τ is measured from the actual placement step (i0), just like
            # simulation mode.  The "placement" is virtual but the timing
            # reference is the same.
            t_placed = float(bt_clock_t[i0])
            effective_DeltaT = float(c1) - t_placed
            if not (effective_DeltaT > 0.0):
                effective_DeltaT = float(bt_DeltaT)

            # Mid at window start = pre-event mid of the first event in
            # the window.  This is where the hypothetical LO "would be
            # placed" (at distance delta_value from mid_0).
            mid_0 = float(mid_pre_arr[i0])
            if not np.isfinite(mid_0):
                # Cannot determine where the LO would sit → treat as
                # censored (no fill, full ΔT exposure).
                tau_sum += float(effective_DeltaT)
                win_count += 1
                continue

            # Counterfactual fill detection: did any MO in [i0, i1)
            # sweep deep enough to reach our hypothetical LO?
            found, i_fill = _backtest_window_first_virtual_fill(
                i0=i0,
                i1=min(i1, bt_n),
                delta_value=float(delta_value),
                half_tick=float(half_tick),
                target_side=bt_target_side,
                D_by_event=D_by_event,
                direction_arr=direction_arr,
                mid_pre_arr=mid_pre_arr,
                mid_0=mid_0,
            )

            # Compute censored waiting time τ̃ — IDENTICAL logic to
            # simulation mode (same MLE, different fill source).
            if found and i_fill is not None:
                tau = float(bt_clock_t[i_fill]) - t_placed
                if not np.isfinite(tau) or tau <= 0.0:
                    tau_tilde = effective_DeltaT
                    I = 0
                else:
                    tau_tilde = min(float(tau), float(effective_DeltaT))
                    I = 1 if tau < float(effective_DeltaT) else 0
            else:
                tau_tilde = float(effective_DeltaT)
                I = 0

            I_sum += int(I)
            tau_sum += float(tau_tilde)
            win_count += 1

        # MLE: identical formula to simulation mode.
        if win_count > 0 and tau_sum > 0:
            lam_hat = float(I_sum) / float(tau_sum)
        else:
            lam_hat = 0.0

        lambda_hat[idx] = lam_hat
        n_exec[idx] = int(I_sum)
        denom_sum[idx] = float(tau_sum)
        n_windows_used[idx] = int(win_count)

        runs_debug.append({
            "delta": float(delta_value),
            "lambda_hat": float(lam_hat),
            "n_exec": int(I_sum),
            "tau_sum": float(tau_sum),
            "n_windows": int(win_count),
            "DeltaT": float(bt_DeltaT),
            "unit": str(bt_unit),
        })

    # -----------------------------------------------------------------
    # 4) Plot
    # -----------------------------------------------------------------
    data_label = "Simulation" if data_mode == "simulation" else "Backtest"

    if plot:
        prefix = (plot_title_prefix + " ") if plot_title_prefix else ""
        fig, ax = plt.subplots()
        ax.plot(delta_grid, lambda_hat, marker="o", linestyle="-", label=r"$\hat\lambda(\delta)$")
        ax.set_xlabel(r"$\delta$ (ticks from mid, grid step = half_tick)")
        ax.set_ylabel(f"intensity per {runs_debug[0]['unit'] if runs_debug else 'clock'}")
        ax.set_title(
            f"{prefix}[{data_label}] {title}\n"
            f"clock={mode}, ΔT={runs_debug[0]['DeltaT']:.6g} {runs_debug[0]['unit'] if runs_debug else ''}, "
            f"side_mode={side_mode}, mo_size={mo_size}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

    return {
        "delta_grid": delta_grid,
        "lambda_hat": lambda_hat,
        "n_exec": n_exec,
        "denom_sum_tau": denom_sum,
        "n_windows_used": n_windows_used,
        "data_mode": data_mode,
        "aggregation_mode": mode,
        "n_tob_moves": int(n_tob_moves),
        "n_events_interval": int(n_events_interval),
        "time_interval": float(time_interval),
        "half_tick": float(half_tick),
        "max_levels": int(max_levels),
        "side_mode": str(side_mode),
        "burn_in_windows": int(burn_in_windows),
        "runs_debug": runs_debug,
    }


def fit_A_kappa_loglinear(delta_grid, lambda_hat, weights=None, eps=1e-15):
    """
    Fit: lambda(delta) ≈ A * exp(-kappa * delta)
    via log-linear regression:
        log(lambda) = log(A) - kappa * delta

    weights: opcional (ex: denom_sum_tau) para WLS no log-espaco.
    Retorna: A, kappa, dict com diagnósticos.
    """
    d = np.asarray(delta_grid, dtype=float)
    y = np.asarray(lambda_hat, dtype=float)

    mask = np.isfinite(d) & np.isfinite(y) & (y > eps)
    if mask.sum() < 2:
        return 0.0, 0.0, {"ok": False, "reason": "not enough positive lambda points"}

    d = d[mask]
    y = y[mask]

    logy = np.log(y)

    # Design: logy = b0 + b1 * d  onde b0=logA e b1=-kappa
    X = np.column_stack([np.ones_like(d), d])

    if weights is None:
        beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
        b0, b1 = beta
    else:
        w = np.asarray(weights, dtype=float)[mask]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

        # WLS: resolve (sqrtW*X) beta = (sqrtW*logy)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = logy * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        b0, b1 = beta

    A = float(np.exp(b0))
    kappa = float(-b1)

    # Diagnósticos (R^2 no log-espaço)
    logy_hat = X @ np.array([b0, b1])
    ss_res = float(np.sum((logy - logy_hat) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return A, kappa, {
        "ok": True,
        "n_used": int(mask.sum()),
        "r2_log": r2_log,
        "b0_logA": float(b0),
        "b1_slope": float(b1),
    }
