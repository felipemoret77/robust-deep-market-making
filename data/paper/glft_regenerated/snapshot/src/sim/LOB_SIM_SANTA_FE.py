#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:01:51 2026

@author: felipemoret
"""

from typing import Callable, Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm


"""
Non-Markovian Zero-Intelligence LOB — engine (filter-enabled, MM-safe cancels)
+ Optional external MarketMaker wrapper/runner compatibility hooks.

Original author: Adele Ravagnani (engine)
Edits: cancel-filter API, avoid-mask API, safer queue mirroring, clearer comments/docstrings,
       and housekeeping.

MINIMAL NEW FEATURE:
--------------------
- Add a "Size" column to the message tape.
- Make the engine return the *actual executed size* for each event:
    * LO size = 1
    * Cancel size = 1
    * MO size = number of units actually executed (<= mean_size_MO)
This enables downstream executability / calibration code to use message_df["Size"].
"""


class LOB_simulation:
    """
    Simulates a zero-intelligence limit order book with explicit FIFO queues
    for each price level (matrix priorities_lob_state: rank × price).

    New (important) features in this version
    ----------------------------------------
    1) Cancel filter API
       - `set_cancel_filter(fn)`: register a callable that chooses which rank
         to cancel among the currently valid candidates at a given price.
       - Signature: fn(price:int, candidate_ranks:np.ndarray[int]) -> int

    2) Avoid-mask provider
       - `set_cancel_avoid_mask_provider(fn)`: register a callable that returns
         a boolean mask per rank at a given price (True = must avoid cancelling).
       - Signature: fn(price:int) -> np.ndarray[bool] (length = n_priority_ranks)

    Both are optional. If neither is set, behavior matches the original engine.
    """

    # ------------------------------- lifecycle -------------------------------

    def __init__(
        self,
        number_tick_levels: int,
        n_priority_ranks: int,
        p0: int,
        mean_size_LO: float = 1,
        number_levels_to_store: int = 20,
        beta_exp_weighted_return: float = 1e-3,
        intensity_exp_weighted_return: float = 1e-3,
        mean_size_MO: float = 1,
        rng: Optional[np.random.RandomState] = None,
        buy_mo_prob: float = 0.5,
    ):

        # Validate that number_levels_to_store is even (Bug #17)
        if number_levels_to_store % 2 != 0:
            raise ValueError(f"number_levels_to_store must be even, got {number_levels_to_store}")

        # Main book state: signed depth per price index (bid>0, ask<0)
        self.lob_state: np.ndarray = None
        self.number_tick_levels = number_tick_levels

        # Queue matrix: shape = (ranks, prices); +1 on bid side, -1 on ask side, 0 empty
        self.priorities_lob_state: np.ndarray = None
        self.n_priority_ranks = n_priority_ranks

        # Price normalization / volume unit
        self.p0 = p0
        # Round to integers: each event is discrete (no fractional shares).
        self.mean_size_LO = max(1, int(round(mean_size_LO)))
        self.mean_size_MO = max(1, int(round(mean_size_MO)))
        if self.mean_size_MO < self.mean_size_LO:
            raise ValueError(
                f"mean_size_MO ({self.mean_size_MO}) must be >= mean_size_LO "
                f"({self.mean_size_LO}). Otherwise MO sweep rounds to 0 and "
                f"market orders become no-ops."
            )
        if self.mean_size_MO % self.mean_size_LO != 0:
            raise ValueError(
                f"mean_size_MO ({self.mean_size_MO}) must be an exact multiple "
                f"of mean_size_LO ({self.mean_size_LO}). Otherwise the MO sweep "
                f"rounds down to {(self.mean_size_MO // self.mean_size_LO) * self.mean_size_LO} "
                f"shares, silently discarding "
                f"{self.mean_size_MO - (self.mean_size_MO // self.mean_size_LO) * self.mean_size_LO} "
                f"shares per market order."
            )

        # Event time increment (Δt) produced by the exponential-clocks event sampler.
        # This is set inside draw_next_order_type() every time an event is drawn.
        self.last_event_dt: float = 0.0

        # Storage for “top N levels per side” snapshots
        self.number_levels_to_store = number_levels_to_store
        self.ob_dict: dict = {}
        self.message_dict: dict = {}

        # Final outputs
        self.message_df_simulation: Optional[pd.DataFrame] = None
        self.ob_df_simulation: Optional[pd.DataFrame] = None

        # Non-Markovian feedback via EWMA of returns
        self.beta_exp_weighted_return = beta_exp_weighted_return
        self.gamma_exp_weighted_return = np.exp(-self.beta_exp_weighted_return)
        self.intensity_exp_weighted_return = intensity_exp_weighted_return
        self.mid_price_to_store = {"Current": 0.0, "Previous": 0.0}
        self.exp_weighted_return_to_store = 0.0


        # Probability that a market order is a BUY (hits the ask side).
        # Default 0.5 = symmetric flow. Values > 0.5 bias towards buying pressure.
        # Can be a float OR a callable(step_idx) -> float for regime-switching.
        if callable(buy_mo_prob):
            self._buy_mo_prob_fn = buy_mo_prob
            self.buy_mo_prob = 0.5  # initial default
        else:
            self._buy_mo_prob_fn = None
            self.buy_mo_prob = float(buy_mo_prob)
        self._event_step: int = 0  # counter for callable schedule

        # Per-instance RNG avoids global np.random state collisions across envs/threads.
        self.rng = rng if rng is not None else np.random.RandomState()

        # Cancel-steering plug-ins (both optional)
        self.cancel_filter: Optional[Callable[[int, np.ndarray], int]] = None
        self.cancel_avoid_mask_provider: Optional[Callable[[int], np.ndarray]] = None

        # Simulation time placeholder (Bug #13): set by simulate_LOB before each event.
        self.time: float = np.nan

        # Callback for rank-shift notifications (Bug #19): external code can set this
        # to be notified when queue ranks shift after a removal.
        # Signature: fn(order_price: int, priority_rank: int) -> None
        self.on_rank_shift: Optional[Callable[[int, int], None]] = None

        # Last priority expansion tracker (Bug #8): external code can check this
        # to detect when the priority matrix was dynamically expanded.
        self._last_priority_expansion: int = n_priority_ranks

        # Best bid/ask cache with VERSION tracking.
        # Every mutation of lob_state increments _lob_version.
        # _best_bid_ask_indices() only recomputes when the version changed.
        # This correctly handles multiple mutations within the same event
        # (e.g., MO sweep consuming multiple levels).
        self._lob_version: int = 0
        self._cached_best_bid: int = -1
        self._cached_best_ask: int = -1
        self._cached_version: int = -1  # force recompute on first call

    # ------------------------------- plug-ins --------------------------------

    def set_cancel_filter(self, fn: Callable[[int, np.ndarray], int]):
        """
        Register a filter to choose which rank to cancel at a given price.

        Parameters
        ----------
        fn : callable
            Signature: fn(price:int, candidate_ranks:np.ndarray[int]) -> int
            Must return ONE valid rank in `candidate_ranks`.
        """
        self.cancel_filter = fn

    def set_cancel_avoid_mask_provider(self, fn: Callable[[int], np.ndarray]):
        """
        Register a provider of avoid masks per price.

        Parameters
        ----------
        fn : callable
            Signature: fn(price:int) -> np.ndarray[bool] of length n_priority_ranks.
            True means “avoid cancelling this rank”.
        """
        self.cancel_avoid_mask_provider = fn

    # ------------------------------- init helpers ----------------------------

    def initialize_message_and_ob_dictionary(self):
        """Pre-create keys for order book snapshots and message tape."""
        for i in range(1, self.number_levels_to_store // 2 + 1):
            self.ob_dict[f"AskPrice_{i}"] = []
            self.ob_dict[f"AskSize_{i}"] = []
            self.ob_dict[f"BidPrice_{i}"] = []
            self.ob_dict[f"BidSize_{i}"] = []
    
        keys_message = [
            "Time",
            "Type",
            "Price",
            "Direction",
            "Size",
            "Spread",
            "MidPrice",
            "Shift",
            "Return",
            "TotNumberBidOrders",
            "TotNumberAskOrders",
            "BestBidPrice",
            "BestAskPrice",
            "IndBestBid",
            "IndBestAsk",
    
            # NEW: cancellation queue-position diagnostics (per-row)
            "CancelVolAhead",   # volume ahead of cancelled order at that price (pre-cancel)
            "CancelQueueLen",   # total queue length at that price (pre-cancel)
            "CancelRank",       # which FIFO rank was cancelled
        ]
        for k in keys_message:
            self.message_dict[k] = []


    def initialize_lob_state(self):
        """
        Start with a symmetric toy book: bids on the left half,
        asks on the right half.  Depth per level = mean_size_LO shares.
        """
        self.lob_state = np.zeros(self.number_tick_levels, dtype=float)
        self.lob_state[: self.number_tick_levels // 2] = float(self.mean_size_LO)
        self.lob_state[self.number_tick_levels // 2 :] = -float(self.mean_size_LO)

    def initialize_priorities_lob_state(self):
        """Rank 0 occupied across both sides, others empty."""
        self.priorities_lob_state = np.zeros((self.n_priority_ranks, self.number_tick_levels), dtype=float)
        self.priorities_lob_state[0, : self.number_tick_levels // 2] = 1.0
        self.priorities_lob_state[0, self.number_tick_levels // 2 :] = -1.0

    def initialize(self):
        """Full engine init: states, dictionaries, EWMA anchors."""
        self.initialize_lob_state()
        self.initialize_priorities_lob_state()
        self._invalidate_bba_cache()
        self.initialize_message_and_ob_dictionary()
        self.mid_price_to_store["Current"] = self.compute_mid_price()
        self.mid_price_to_store["Previous"] = self.compute_mid_price()

    # ------------------------------ book metrics -----------------------------

    def _touch_lob_state(self) -> None:
        """Increment the lob_state version counter.  Call this after ANY
        mutation of lob_state (add/remove/clear/shift/reinit).
        The best bid/ask cache auto-recomputes on next access."""
        self._lob_version += 1

    # Keep old name as alias for backward compat (QRM subclass uses it)
    _invalidate_bba_cache = _touch_lob_state

    def _best_bid_ask_indices(self) -> Tuple[int, int]:
        """Return (best_bid_idx, best_ask_idx) on the CURRENT local price grid.

        Uses a version-gated cache: recomputes only when _lob_version has
        changed since the last call.  This correctly handles multiple
        mutations within the same event (each mutation bumps the version).

        Returns
        -------
        best_bid_idx : int
            Highest index with positive depth (>=0), or -1 if the bid side is empty.
        best_ask_idx : int
            Lowest index with negative depth (>=0), or -1 if the ask side is empty.
        """
        if self._cached_version == self._lob_version:
            return self._cached_best_bid, self._cached_best_ask

        bid_idx = np.where(self.lob_state > 0)[0]
        ask_idx = np.where(self.lob_state < 0)[0]
        self._cached_best_bid = int(bid_idx[-1]) if bid_idx.size else -1
        self._cached_best_ask = int(ask_idx[0]) if ask_idx.size else -1
        self._cached_version = self._lob_version
        return self._cached_best_bid, self._cached_best_ask

    def compute_mid_price(self) -> float:
        """Compute mid-price in *local tick indices* (pre p0 / shift adjustments).

        Robustness
        ----------
        If either side is empty, we fall back to the last stored mid if available,
        otherwise to the center of the grid. This prevents IndexError crashes and keeps
        the simulation running in edge cases.
        """
        best_bid, best_ask = self._best_bid_ask_indices()

        if best_bid < 0 or best_ask < 0:
            # Prefer the last known mid (already on local grid), if we have one.
            prev = float(self.mid_price_to_store.get("Current", np.nan))
            if np.isfinite(prev):
                return float(prev)
            prev = float(self.mid_price_to_store.get("Previous", np.nan))
            if np.isfinite(prev):
                return float(prev)

            # Ultimate fallback: center of the grid
            return float(self.number_tick_levels // 2)

        return 0.5 * (float(best_bid) + float(best_ask))

    def compute_spread(self) -> int:
        """Compute spread (best_ask - best_bid) in local tick indices.

        Returns -1 (sentinel for "undefined") if either side is empty
        (best_ask < 0 or best_bid < 0).
        """
        best_bid, best_ask = self._best_bid_ask_indices()
        if best_bid < 0 or best_ask < 0:
            return -1
        return int(best_ask - best_bid)

    # ------------------------------ re-centering ------------------------------

    def center_lob_state(self) -> int:
        """
        Keep the mid near the middle of the price grid by shifting the whole book.
        Returns the integer shift applied (positive = moved left in the array).
        """
        new_mid = self.compute_mid_price()
        shift = int(new_mid + 0.5 - self.number_tick_levels // 2)

        # Bug #9: guard against shift magnitude >= number_tick_levels
        if abs(shift) >= self.number_tick_levels:
            self.lob_state[:] = 0.0
            self.priorities_lob_state[:, :] = 0.0
            self._touch_lob_state()
            self._last_center_shift = shift
            return shift

        if shift > 0:
            k = shift
            self.lob_state[:-k] = self.lob_state[k:]
            self.lob_state[-k:] = 0.0
            self.priorities_lob_state[:, :-k] = self.priorities_lob_state[:, k:]
            self.priorities_lob_state[:, -k:] = 0.0
        elif shift < 0:
            k = -shift
            self.lob_state[k:] = self.lob_state[:-k]
            self.lob_state[:k] = 0.0
            self.priorities_lob_state[:, k:] = self.priorities_lob_state[:, :-k]
            self.priorities_lob_state[:, :k] = 0.0

        self._last_center_shift = shift
        if shift != 0:
            self._touch_lob_state()
        return shift

    # --------------------------- snapshot extraction -------------------------

    def update_ob_dict(self, i: int):
        """
        Append top N levels to ob_dict for the current state.
        Ensures all lists have equal length (pads with zeros when a side is shallow).

        .. deprecated::
            Use :meth:`make_ob_snapshot_row` instead.
        """
        warnings.warn("update_ob_dict is deprecated; use make_ob_snapshot_row instead", DeprecationWarning, stacklevel=2)
        n_quotes_bid = (self.lob_state > 0).sum()
        n_quotes_ask = (self.lob_state < 0).sum()

        # Bid side (closest to mid first)
        for n in range(min(self.number_levels_to_store // 2, n_quotes_bid)):
            self.ob_dict[f"BidPrice_{n+1}"].append(np.where(self.lob_state > 0)[0][-n-1])
            self.ob_dict[f"BidSize_{n+1}"].append(self.lob_state[self.lob_state > 0][-n-1])
        for n in range(self.number_levels_to_store // 2):
            if len(self.ob_dict[f"BidPrice_{n+1}"]) < i + 1:
                self.ob_dict[f"BidPrice_{n+1}"].append(0)
                self.ob_dict[f"BidSize_{n+1}"].append(0)

        # Ask side (closest to mid first)
        for n in range(min(self.number_levels_to_store // 2, n_quotes_ask)):
            self.ob_dict[f"AskPrice_{n+1}"].append(np.where(self.lob_state < 0)[0][n])
            self.ob_dict[f"AskSize_{n+1}"].append(-self.lob_state[self.lob_state < 0][n])
        for n in range(self.number_levels_to_store // 2):
            if len(self.ob_dict[f"AskPrice_{n+1}"]) < i + 1:
                self.ob_dict[f"AskPrice_{n+1}"].append(0)
                self.ob_dict[f"AskSize_{n+1}"].append(0)

    def make_ob_snapshot_row(self) -> Dict[str, float]:
        """
        Build ONE snapshot row (dict) with the same keys as ob_dict columns:
            AskPrice_1..K, AskSize_1..K, BidPrice_1..K, BidSize_1..K
        where K = number_levels_to_store//2.

        IMPORTANT:
        - Values are on the CURRENT "local" index grid (whatever the current lob_state uses).
        - This function ALWAYS fills missing levels with zeros so that the resulting
          ob_df is rectangular and consistent with fix_zero_size().
        """
        out: Dict[str, float] = {}

        half = int(self.number_levels_to_store // 2)

        # Indices (price levels) that currently have liquidity
        bid_idx = np.where(self.lob_state > 0)[0]  # increasing indices
        ask_idx = np.where(self.lob_state < 0)[0]  # increasing indices

        n_bid = int(bid_idx.size)
        n_ask = int(ask_idx.size)

        # -------------------------
        # BID side: closest-to-mid first
        # That means take from the END of bid_idx (highest index among bids).
        # -------------------------
        for n in range(half):
            key_p = f"BidPrice_{n+1}"
            key_s = f"BidSize_{n+1}"

            if n < n_bid:
                px = int(bid_idx[-n - 1])
                sz = float(self.lob_state[px])  # positive size (lob_state already in shares)
                out[key_p] = float(px)
                out[key_s] = float(sz)
            else:
                out[key_p] = 0.0
                out[key_s] = 0.0

        # -------------------------
        # ASK side: closest-to-mid first
        # That means take from the START of ask_idx (lowest index among asks).
        # -------------------------
        for n in range(half):
            key_p = f"AskPrice_{n+1}"
            key_s = f"AskSize_{n+1}"

            if n < n_ask:
                px = int(ask_idx[n])
                sz = float(-self.lob_state[px])  # store as positive size (lob_state already in shares)
                out[key_p] = float(px)
                out[key_s] = float(sz)
            else:
                out[key_p] = 0.0
                out[key_s] = 0.0

        return out

    # --------------------------- event sampling logic ------------------------

    def draw_next_order_type(self, lam: float, mu: float, delta: float) -> int:
        """
        Draw the next event type by independent exponential clocks:
        0 = LO, 1 = MO, 2 = Cancel.
        """
        Lam = lam * self.number_tick_levels
        Mu = 2 * mu

        # Count orders (not shares) from the priority queue for cancel intensity.
        # Each non-zero entry in priorities_lob_state is one order.
        orders_per_price = np.count_nonzero(self.priorities_lob_state, axis=0)
        n_orders = float(orders_per_price.sum())

        # Robust cancellation intensity: `delta` may be a scalar or array-like.
        # - scalar: per-order cancellation intensity (original behavior)
        # - size=2: per-order intensities by side -> [delta_bid, delta_ask]
        # - shape == lob_state.shape: per-order intensity per price level
        delta_arr = np.asarray(delta, dtype=float)

        if delta_arr.size == 1:
            # Original behavior: scalar per-order cancellation intensity.
            Delta = float(max(float(delta_arr) * n_orders, 1e-9))

        elif delta_arr.size == 2:
            # Interpret delta = [delta_bid, delta_ask] as per-order intensities by side.
            n_bid = float(np.sum(self.priorities_lob_state > 0))
            n_ask = float(np.sum(self.priorities_lob_state < 0))
            Delta = float(max(delta_arr[0] * n_bid + delta_arr[1] * n_ask, 1e-9))

        elif delta_arr.shape == self.lob_state.shape:
            # Interpret delta[level] as per-order intensity at each price level.
            Delta = float(max(np.sum(delta_arr * orders_per_price), 1e-9))

        else:
            # Fallback: reduce any other shape to a scalar rate (keeps the simulator running).
            Delta = float(max(float(np.mean(delta_arr)) * n_orders, 1e-9))

        Lam_time = self.rng.exponential(1.0 / Lam)
        Mu_time = self.rng.exponential(1.0 / Mu)
        Delta_time = self.rng.exponential(1.0 / Delta)

        # Store the actual event time increment corresponding to the winning clock.
        # This makes simulated timestamps consistent with the LO/MO/C exponential-clock mechanism.
        self.last_event_dt = float(min(Lam_time, Mu_time, Delta_time))
        return int(np.argmin([Lam_time, Mu_time, Delta_time]))

    def draw_next_order(self, lam: float, mu: float, delta: float) -> Tuple[int, int]:
        """
        Sample (type, sign). Sign convention: +1 = bid side, -1 = ask side.
        """
        # Count orders (not shares) for cancel side selection.
        n_orders_bid = float(np.sum(self.priorities_lob_state > 0))
        n_orders_ask = float(np.sum(self.priorities_lob_state < 0))

        order_type = self.draw_next_order_type(lam, mu, delta)

        if order_type == 0:  # LIMIT ORDER
            if self.intensity_exp_weighted_return <= 0.0:
                prob_sell = 0.5
            else:
                prob_sell = 1.0 / (1.0 + np.exp(-self.intensity_exp_weighted_return * self.exp_weighted_return_to_store))
            sign = self.rng.choice([+1, -1], p=[1 - prob_sell, prob_sell])

        elif order_type == 1:  # MARKET ORDER
            # Resolve dynamic buy_mo_prob if a callable schedule was provided
            if self._buy_mo_prob_fn is not None:
                self.buy_mo_prob = float(self._buy_mo_prob_fn(self._event_step))
            self._event_step += 1
            sign = self.rng.choice([+1, -1], p=[self.buy_mo_prob, 1.0 - self.buy_mo_prob])

        else:  # Cancel
            total = n_orders_bid + n_orders_ask
            if total <= 0:
                sign = self.rng.choice([+1, -1])
            else:
                sign = self.rng.choice([+1, -1], p=[n_orders_bid / total, n_orders_ask / total])

        return order_type, int(sign)

    def sample_limit_order_price(self, order_sign: int) -> int:
            """Sample a passive limit-order price on the chosen side.

            For bids (order_sign=+1) we sample uniformly in [0, best_ask).
            For asks (order_sign=-1) we sample uniformly in (best_bid, number_tick_levels).

            Robustness
            ----------
            If the book is temporarily empty/crossed (rare), fall back to sampling around the
            current mid so we never crash on missing best quotes.
            """
            best_bid, best_ask = self._best_bid_ask_indices()

            # Fallback if the book is invalid / crossed / empty.
            if best_bid < 0 or best_ask < 0 or int(best_ask) <= int(best_bid):
                mid = int(round(float(self.compute_mid_price())))
                mid = int(np.clip(mid, 0, int(self.number_tick_levels) - 1))

                if order_sign == +1:
                    hi = max(mid, 1)
                    return int(self.rng.randint(0, hi))
                else:
                    lo = min(mid + 1, int(self.number_tick_levels) - 1)
                    hi = int(self.number_tick_levels)
                    if lo >= hi:
                        lo = max(0, hi - 1)
                    return int(self.rng.randint(lo, hi))

            if order_sign == +1:
                # place somewhere below the best ask
                hi = int(best_ask)
                if hi <= 0:
                    return -1  # no room for a passive bid
                return int(self.rng.randint(0, hi))
            else:
                # place somewhere above the best bid
                lo = int(best_bid + 1)
                hi = int(self.number_tick_levels)
                if lo >= hi:
                    lo = max(0, hi - 1)
                return int(self.rng.randint(lo, hi))

    def compute_market_order_price(self, order_sign: int) -> int:
            """Return the price level hit by a market order on the chosen side.

            +1 (buy MO) hits the best ask.
            -1 (sell MO) hits the best bid.

            Returns -1 if the corresponding side is empty.
            """
            best_bid, best_ask = self._best_bid_ask_indices()
            if order_sign == +1:
                return int(best_ask) if best_ask >= 0 else -1
            else:
                return int(best_bid) if best_bid >= 0 else -1

    def execute_mm_market_order(self, order_sign: int, order_price: int) -> int:
        """Execute a 1-unit external market order at a GIVEN price index.
    
        This is used by the MarketMaker wrapper ("cross" actions).
        We keep it intentionally simple and deterministic: one unit, at the provided
        price level.
    
        Safety guards (minimal but important)
        ------------------------------------
        - If the provided price has no opposite-side liquidity, do nothing.
        - If executing would *empty the entire opposite side* (total units <= 1),
          do nothing. This avoids pathological states where mid/spread become undefined.
    
        Returns
        -------
        int
            The executed price index, or -1 if nothing was executed.
        """
        px = int(order_price)
        if px < 0 or px >= int(self.number_tick_levels):
            return -1
    
        if int(order_sign) == +1:
            # BUY MO hits ASK liquidity (negative depth)
            if float(self.lob_state[px]) >= 0.0:
                return -1  # no ask liquidity at this px
            tot_ask_orders = int(np.sum(self.priorities_lob_state < 0))
            if tot_ask_orders <= 1:
                return -1  # do not empty the ask side (keep at least 1 order)
        else:
            # SELL MO hits BID liquidity (positive depth)
            if float(self.lob_state[px]) <= 0.0:
                return -1  # no bid liquidity at this px
            tot_bid_orders = int(np.sum(self.priorities_lob_state > 0))
            if tot_bid_orders <= 1:
                return -1  # do not empty the bid side (keep at least 1 order)

        # Apply the execution: consume mean_size_LO shares (= 1 order)
        self.lob_state[px] += float(order_sign) * float(self.mean_size_LO)
        self._touch_lob_state()

        # Remove the matched unit from the priority queue (best rank at this price).
        # In this simplified engine, rank 0 corresponds to the oldest order at that price.
        self.remove_order_from_queue(px, 0)

        # Bug #7: update mid_price, EWMA, and recenter after execution
        self.mid_price_to_store["Previous"] = self.mid_price_to_store["Current"]
        self.mid_price_to_store["Current"] = float(self.compute_mid_price())

        if self.beta_exp_weighted_return is not None and self.beta_exp_weighted_return > 0.0:
            gamma = self.gamma_exp_weighted_return
            self.exp_weighted_return_to_store = (
                gamma * float(self.exp_weighted_return_to_store)
                + (1.0 - gamma)
                * float(self.mid_price_to_store["Current"] - self.mid_price_to_store["Previous"])
            )

        self.center_lob_state()

        return int(px)

    def sample_cancellation_price(self, order_sign: int) -> int:
        # Count orders (not shares) per price level for uniform cancel sampling.
        orders_per_price = np.count_nonzero(self.priorities_lob_state, axis=0)
        bid_mask = self.lob_state > 0
        ask_mask = self.lob_state < 0
        n_bid = int(orders_per_price[bid_mask].sum())
        n_ask = int(orders_per_price[ask_mask].sum())

        # Guard: if the requested side is empty, return the best price on that side
        # (caller is expected to handle the -1 sentinel).
        if order_sign == +1 and n_bid == 0:
            bid_idx = np.where(self.lob_state > 0)[0]
            return int(bid_idx[-1]) if bid_idx.size else -1
        if order_sign != +1 and n_ask == 0:
            ask_idx = np.where(self.lob_state < 0)[0]
            return int(ask_idx[0]) if ask_idx.size else -1

        numeration_orders = orders_per_price.cumsum()

        if order_sign == +1:
            ind_to_cancel = self.rng.randint(n_bid)
        else:
            ind_to_cancel = n_bid + self.rng.randint(n_ask)

        candidates = np.where(numeration_orders > ind_to_cancel)[0]
        if candidates.size == 0:
            return -1
        px = int(candidates[0])

        # Safety: verify the returned price is on the correct side.
        # Bug #12: resample randomly from correct side instead of biased best-price fallback.
        if order_sign == +1 and self.lob_state[px] <= 0:
            correct_side_indices = np.where(self.lob_state > 0)[0]
            if correct_side_indices.size == 0:
                return -1
            return int(self.rng.choice(correct_side_indices))
        if order_sign != +1 and self.lob_state[px] >= 0:
            correct_side_indices = np.where(self.lob_state < 0)[0]
            if correct_side_indices.size == 0:
                return -1
            return int(self.rng.choice(correct_side_indices))

        return px

    # ----------------------- rank selection with filters ---------------------

    def sample_cancellation_priority_rank(
        self,
        order_price: int,
        avoid_mask: Optional[np.ndarray] = None,
    ) -> int:
        # Guard: -1 sentinel means the side is empty — nothing to cancel.
        if order_price < 0 or order_price >= self.priorities_lob_state.shape[1]:
            return -1
        candidates = np.where(self.priorities_lob_state[:, order_price] != 0)[0]
        if candidates.size == 0:
            return -1

        # Robustify avoid-mask handling:
        # - The MM may keep its own owner mask with a fixed number of ranks.
        # - This engine may *dynamically* expand `n_priority_ranks` (see add_order_to_queue()).
        # To avoid either (a) crashing or (b) silently ignoring the avoid mask,
        # we pad / truncate to the current `n_priority_ranks` and then apply it.
        if avoid_mask is not None:
            am = np.asarray(avoid_mask, dtype=bool).ravel()
            if am.size != int(self.n_priority_ranks):
                # Bug #8: pad with True (protect new ranks) instead of False,
                # since new ranks likely contain the MM's just-placed order.
                am_fixed = np.ones(int(self.n_priority_ranks), dtype=bool)
                m = min(am.size, am_fixed.size)
                if m > 0:
                    am_fixed[:m] = am[:m]
                am = am_fixed

            safe = candidates[~am[candidates]]
            if safe.size == 0:
                # No non-MM candidates at this price -> signal "cannot cancel here".
                return -1
            candidates = safe

        if self.cancel_filter is not None:
            rank = int(self.cancel_filter(order_price, candidates))
            if rank in candidates:
                return rank
            # Bug #10: warn when cancel_filter returns an invalid rank
            warnings.warn(
                f"cancel_filter returned rank {rank} not in candidates {candidates.tolist()}, falling back to random"
            )

        return int(self.rng.choice(candidates))

    # -------------------------------- queue ops ------------------------------

    def add_order_to_queue(self, order_price: int, order_signed_size: int) -> Optional[int]:
        empties = np.where(self.priorities_lob_state[:, order_price] == 0)[0]
        if empties.size != 0:
            pr = int(empties[0])
            # BUG FIX (F): use assignment (=) not accumulation (+=).  The slot
            # is known to be zero (we just found it via np.where == 0), so +=
            # is semantically equivalent — but if a residual float ~1e-15
            # lingers from prior cancellation, += accumulates the error.
            self.priorities_lob_state[pr, order_price] = order_signed_size
            return pr

        warnings.warn(
            "Queue is full at this price; expanding priorities_lob_state by 1 rank. "
            "Consider increasing n_priority_ranks to avoid dynamic resizing.",
            RuntimeWarning,
        )

        # Expand FIFO rank dimension (rare; only happens if n_priority_ranks was too small).
        self.priorities_lob_state = np.vstack(
            [
                self.priorities_lob_state,
                np.zeros((1, self.number_tick_levels), dtype=self.priorities_lob_state.dtype),
            ]
        )
        self.n_priority_ranks += 1

        # Bug #8: store expansion event so external code (e.g. MM wrapper) can detect it.
        self._last_priority_expansion = self.n_priority_ranks

        pr = self.n_priority_ranks - 1
        self.priorities_lob_state[pr, order_price] = order_signed_size
        return pr

    def remove_order_from_queue(self, order_price: int, priority_rank: int):
        if 0 <= priority_rank < self.n_priority_ranks:
            # Shift the FIFO queue at this price one step "forward" (older ranks come first).
            self.priorities_lob_state[priority_rank:-1, order_price] = self.priorities_lob_state[
                priority_rank + 1 :, order_price
            ]
            self.priorities_lob_state[-1, order_price] = 0.0

            # Bug #19: notify external code that ranks above priority_rank
            # at order_price have shifted down by 1.
            if self.on_rank_shift is not None:
                self.on_rank_shift(order_price, priority_rank)
        else:
            warnings.warn("Tried to remove a non-existing rank from the queue.")

    # -------------------------------- simulate -------------------------------
    
    def compute_best_bid_ask(self):
        """
        Backward-compatible alias.
    
        Older code (or older forks of simulate_order) may call `computeself.compute_best_bid_ask()`.
        In this engine the canonical helper is `_best_bid_ask_indices()` which returns
        best bid/ask as *local tick indices*.
        """
        return self._best_bid_ask_indices()
    
    def sample_order(self, lam: float, mu: float, delta: float):
        """
        Backward-compat wrapper.
    
        Older versions of `simulate_order()` expected a monolithic `sample_order()`.
        This engine version samples in pieces:
          - draw_next_order() -> (type, sign)
          - then pick price/rank depending on the type
        """
        order_type, order_sign = self.draw_next_order(lam, mu, delta)
    
        if order_type == 0:
            # LIMIT ORDER
            order_price = int(self.sample_limit_order_price(order_sign))
            if order_price < 0:
                # No room for a passive order on this side — signal skip.
                # Return price=-1 so simulate_order() can detect and skip
                # instead of injecting artificial liquidity at price 0.
                return 0, int(order_sign), -1, None
            priority_rank = None  # not used for LO (queue rank comes from add_order_to_queue)
    
        elif order_type == 1:
            # MARKET ORDER
            order_price = int(self.compute_market_order_price(order_sign))
            priority_rank = 0  # best FIFO rank is consumed first
    
        else:
            # CANCEL
            order_price = int(self.sample_cancellation_price(order_sign))
            if order_price < 0:
                # Side is empty — no valid cancel target; return sentinel rank.
                priority_rank = -1
            else:
                avoid = None
                if self.cancel_avoid_mask_provider is not None:
                    avoid = self.cancel_avoid_mask_provider(order_price)
                priority_rank = int(self.sample_cancellation_priority_rank(order_price, avoid_mask=avoid))
    
        return order_type, int(order_sign), int(order_price), priority_rank
    

    def simulate_order(
        self,
        lam: float,
        mu: float,
        delta: float,
        split_sweeps: bool = False,
    ) -> Tuple[
        List[Tuple[int, int, int, int, int]],
        List[Dict[str, float]],
        List[Dict[str, float]],
    ]:
        """
        Simulate ONE parent event (LO / MO / Cancel).

        MO size is controlled by self.mean_size_MO (calibrated mean MO size).
        LO/Cancel size is controlled by self.mean_size_LO.
        """

        # -------------------------
        # Decide which parent event happens (LO / MO / Cancel)
        # Bug #20/#21: removed dead-code helpers _best_bid_ask and _sample_parent_event;
        # using self.computeself.compute_best_bid_ask() and self.sample_order() directly.
        # -------------------------
        order_type, order_sign, order_price, priority_rank = self.sample_order(lam=lam, mu=mu, delta=delta)
    
        # -------------------------
        # Pre-event book state used for "Previous" mid in your conventions
        # -------------------------
        self.mid_price_to_store["Previous"] = float(self.mid_price_to_store["Current"])
    
        # -------------------------
        # Logging containers
        # -------------------------
        rows: List[Tuple[int, int, int, int, int]] = []
        metrics: List[Dict[str, float]] = []
        snaps: List[Dict[str, float]] = []
    
        cancel_rank_export: float = float("nan")
        executed_by_price_pre_shift_parent: Dict[int, int] = {}
    
        # Helper to capture state-derived metrics at the CURRENT moment
        def _capture_metrics(mid_now: float, ret_val: float) -> Dict[str, float]:
            spread = float(self.compute_spread())
            best_bid, best_ask = self.compute_best_bid_ask()
            # NOTE: these count total DEPTH (shares/volume), not the number
            # of individual FIFO orders.  The field names are legacy and
            # potentially misleading — they are used for logging only,
            # not as RL state features.
            tot_bid = float(np.sum(self.lob_state[self.lob_state > 0]))
            tot_ask = float(np.sum(np.abs(self.lob_state[self.lob_state < 0])))

            return {
                "Time": float(getattr(self, "time", np.nan)),
                "MidPrice": float(mid_now),
                "Return": float(ret_val),
                "Spread": float(spread),
                "IndBestBid": float(int(best_bid)) if best_bid >= 0 else float("nan"),
                "IndBestAsk": float(int(best_ask)) if best_ask >= 0 else float("nan"),
                "BestBidPrice": float(int(best_bid)) if best_bid >= 0 else np.nan,
                "BestAskPrice": float(int(best_ask)) if best_ask >= 0 else np.nan,
                "TotNumberBidOrders": float(tot_bid),
                "TotNumberAskOrders": float(tot_ask),
                "TotNumberOrders": float(tot_bid + tot_ask),
                "CancelVolAhead": float("nan"),
                "CancelQueueLen": float("nan"),
                "CancelRank": float("nan"),
            }
    
        # -------------------------
        # Event handling
        # -------------------------
        if order_type == 0 and order_price < 0:
            # No room for a passive LO on this side — skip event entirely.
            # Return empty rows/metrics/snaps so the caller sees a no-op.
            return rows, metrics, snaps

        if order_type == 0:
            # LIMIT ORDER — add mean_size_LO shares
            self.lob_state[order_price] += float(order_sign) * float(self.mean_size_LO)
            self._touch_lob_state()
            self.add_order_to_queue(order_price, order_sign)
    
            # CORRECTION: Always capture metrics
            mid_now = float(self.compute_mid_price())
            ret_val = mid_now - float(self.mid_price_to_store["Previous"])
    
            rows.append((0, int(order_sign), int(order_price), 0, 1))
            metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
            snaps.append(self.make_ob_snapshot_row())
    
        elif order_type == 1:
            # MARKET ORDER
            executed_total = 0
            first_exec_price = None
            remaining = self.mean_size_MO
            mid_prev_fill = float(self.mid_price_to_store["Previous"])
    
            while remaining > 0:
                best_bid, best_ask = self.compute_best_bid_ask()
                if order_sign == +1:
                    px = int(best_ask)
                    if px < 0: break
                else:
                    px = int(best_bid)
                    if px < 0: break
    
                available = int(abs(self.lob_state[px]))
                if available <= 0: break

                # Guard against emptying the entire opposite side.
                # Count orders (not shares) to ensure at least 1 order remains.
                if order_sign == +1:
                    tot_opposite_orders = int(np.sum(self.priorities_lob_state[:, px] != 0))
                    tot_side_orders = int(np.sum(self.priorities_lob_state < 0))
                else:
                    tot_opposite_orders = int(np.sum(self.priorities_lob_state[:, px] != 0))
                    tot_side_orders = int(np.sum(self.priorities_lob_state > 0))
                # Allow taking at most (side_orders - 1) orders worth of shares
                max_take = max((tot_side_orders - 1) * self.mean_size_LO, 0)
                if max_take <= 0:
                    break

                take = int(min(available, remaining, max_take))
                # Round down to nearest multiple of mean_size_LO so that
                # queue entries (which represent mean_size_LO shares each)
                # stay consistent with the share-level lob_state.
                take = (take // self.mean_size_LO) * self.mean_size_LO
                if take <= 0:
                    break
                remaining -= take
                executed_total += take
                executed_by_price_pre_shift_parent[int(px)] = (
                    executed_by_price_pre_shift_parent.get(int(px), 0) + int(take)
                )
    
                if first_exec_price is None:
                    first_exec_price = int(px)
    
                self.lob_state[px] += float(order_sign) * float(take)
                # Bug #16/#22: zero out float residuals near zero
                if abs(self.lob_state[px]) < 1e-10:
                    self.lob_state[px] = 0.0
                self._touch_lob_state()
                # Convert shares consumed to number of queue entries (orders)
                orders_consumed = take // self.mean_size_LO

                # Batch-remove `orders_consumed` front-of-queue entries in one shift
                if orders_consumed >= self.n_priority_ranks:
                    self.priorities_lob_state[:, px] = 0.0
                elif orders_consumed > 0:
                    self.priorities_lob_state[:-orders_consumed, px] = self.priorities_lob_state[orders_consumed:, px]
                    self.priorities_lob_state[-orders_consumed:, px] = 0.0

                # H1: notify external code (e.g. MM) that ranks shifted
                if self.on_rank_shift is not None and orders_consumed > 0:
                    for _k in range(orders_consumed):
                        self.on_rank_shift(int(px), 0)
    
                # Only log individual sweep steps if requested
                if split_sweeps:
                    mid_now = float(self.compute_mid_price())
                    ret_val = mid_now - float(mid_prev_fill)
                    mid_prev_fill = mid_now
    
                    rows.append((1, int(order_sign), int(px), 0, int(take)))
                    metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
                    snaps.append(self.make_ob_snapshot_row())
    
            if executed_total <= 0:
                # No-op MO
                mid_now = float(self.mid_price_to_store["Previous"])
                ret_val = 0.0
                rows.append((1, int(order_sign), int(order_price), 0, 0))
                metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
                snaps.append(self.make_ob_snapshot_row())
            else:
                # If NOT splitting sweeps, log the aggregated event here
                if not split_sweeps:
                    if first_exec_price is None:
                        first_exec_price = int(order_price)
                    
                    mid_now = float(self.compute_mid_price())
                    ret_val = mid_now - float(self.mid_price_to_store["Previous"])

                    rows.append((1, int(order_sign), int(first_exec_price), 0, int(executed_total)))
                    metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
                    snaps.append(self.make_ob_snapshot_row())
    
        else:
            # CANCEL EVENT
            no_op_cancel = False
            if priority_rank is None or int(priority_rank) < 0:
                found = False
                for _ in range(10):
                    cand_price = int(self.sample_cancellation_price(order_sign))
                    if cand_price < 0:
                        continue  # side is empty — retry won't help, but loop will exhaust
                    avoid = None
                    if getattr(self, "cancel_avoid_mask_provider", None) is not None:
                        avoid = self.cancel_avoid_mask_provider(cand_price)
                    if hasattr(self, "sample_cancellation_rank"):
                        cand_rank = self.sample_cancellation_rank(cand_price, avoid_mask=avoid)
                    else:
                        cand_rank = self.sample_cancellation_priority_rank(cand_price, avoid_mask=avoid)
                    if cand_rank is not None and int(cand_rank) >= 0:
                        order_price = int(cand_price)
                        priority_rank = int(cand_rank)
                        found = True
                        break
                if not found:
                    no_op_cancel = True
                    mid_now = float(self.mid_price_to_store["Previous"])
                    ret_val = 0.0
                    rows.append((2, int(order_sign), int(order_price), 0, 0))
                    metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
                    snaps.append(self.make_ob_snapshot_row())
        
            if not no_op_cancel:
                if priority_rank is None:
                    no_op_cancel = True
                else:
                    pr = int(priority_rank)
                    if (pr < 0) or (pr >= int(self.n_priority_ranks)) or (self.priorities_lob_state[pr, order_price] == 0):
                        no_op_cancel = True
                    else:
                        priority_rank = pr
        
                if no_op_cancel:
                    mid_now = float(self.mid_price_to_store["Previous"])
                    ret_val = 0.0
                    rows.append((2, int(order_sign), int(order_price), 0, 0))
                    metrics.append(_capture_metrics(mid_now=mid_now, ret_val=ret_val))
                    snaps.append(self.make_ob_snapshot_row())
        
            if not no_op_cancel:
                queue_len_before_units = float(abs(self.lob_state[order_price]))
                cancel_rank_val = float(priority_rank)
                cancel_rank_export = float(priority_rank)
                vol_ahead_units = float(
                    np.sum(np.abs(self.priorities_lob_state[:priority_rank, order_price])) * float(self.mean_size_LO)
                )
                self.lob_state[order_price] -= float(order_sign) * float(self.mean_size_LO)
                if abs(self.lob_state[order_price]) < 1e-10:
                    self.lob_state[order_price] = 0.0
                self._touch_lob_state()
                self.remove_order_from_queue(order_price, priority_rank)
                
                mid_now = float(self.compute_mid_price())
                ret_val = mid_now - float(self.mid_price_to_store["Previous"])
        
                rows.append((2, int(order_sign), int(order_price), 0, 1))
                
                m = _capture_metrics(mid_now=mid_now, ret_val=ret_val)
                m["CancelVolAhead"] = float(vol_ahead_units)
                m["CancelQueueLen"] = float(queue_len_before_units)
                m["CancelRank"] = float(cancel_rank_val)
                metrics.append(m)
                
                snaps.append(self.make_ob_snapshot_row())

    
        # -------------------------
        # Mid AFTER the event (still pre-shift), used for EWMA update
        # -------------------------
        if order_type == 1 and metrics:
            # Store the execution-by-price map as a dict (not str) so that
            # downstream consumers (MM_LOB_SIM fill detection) can use it
            # directly via isinstance(emap, dict) without parsing.
            metrics[-1]["ExecutedByPricePreShift"] = {
                int(px): int(k) for px, k in executed_by_price_pre_shift_parent.items() if int(k) > 0
            }

        self.mid_price_to_store["Current"] = float(self.compute_mid_price())
    
        # EWMA update using gamma = exp(-beta) as decay factor
        if self.beta_exp_weighted_return is not None and self.beta_exp_weighted_return > 0.0:
            gamma = self.gamma_exp_weighted_return  # = exp(-beta), computed in __init__
            self.exp_weighted_return_to_store = (
                gamma * float(self.exp_weighted_return_to_store)
                + (1.0 - gamma)
                * float(self.mid_price_to_store["Current"] - self.mid_price_to_store["Previous"])
            )

    
        # -------------------------
        # Re-center the LOB ONCE per parent event
        # -------------------------
        shift = int(self.center_lob_state())
        
        if shift != 0:
            self.mid_price_to_store["Current"] = float(self.mid_price_to_store["Current"]) - float(shift)
            self.mid_price_to_store["Previous"] = float(self.mid_price_to_store["Previous"]) - float(shift)
    
        # Attach the true shift to the LAST row only
        if rows:
            ot, os, px_last, _, sz_last = rows[-1]
            rows[-1] = (int(ot), int(os), int(px_last), int(shift), int(sz_last))
    
        # Adjust metrics for the shift (metrics were captured pre-shift)
        if metrics:
            metrics[-1]["MidPrice"] = float(metrics[-1]["MidPrice"]) - float(shift)
            if np.isfinite(metrics[-1].get("BestBidPrice", np.nan)):
                metrics[-1]["BestBidPrice"] = float(metrics[-1]["BestBidPrice"]) - float(shift)
            if np.isfinite(metrics[-1].get("BestAskPrice", np.nan)):
                metrics[-1]["BestAskPrice"] = float(metrics[-1]["BestAskPrice"]) - float(shift)
            if np.isfinite(metrics[-1]["IndBestBid"]):
                metrics[-1]["IndBestBid"] = float(metrics[-1]["IndBestBid"]) - float(shift)
            if np.isfinite(metrics[-1]["IndBestAsk"]):
                metrics[-1]["IndBestAsk"] = float(metrics[-1]["IndBestAsk"]) - float(shift)
    
        if snaps:
            snaps[-1] = self.make_ob_snapshot_row()
    
        return rows, metrics, snaps
    
    # ------------------------------- post-process ----------------------------

    def fix_zero_size(self):
        header_price = [f"AskPrice_{i+1}" for i in range(self.number_levels_to_store // 2)] + [
            f"BidPrice_{i+1}" for i in range(self.number_levels_to_store // 2)
        ]
        header_vol = [f"AskSize_{i+1}" for i in range(self.number_levels_to_store // 2)] + [
            f"BidSize_{i+1}" for i in range(self.number_levels_to_store // 2)
        ]
        for price, size in zip(header_price, header_vol):
            zero_idx = self.ob_df_simulation[self.ob_df_simulation[size] == 0].index
            self.ob_df_simulation.loc[zero_idx, price] = 0

    def save_results(self, path_save_files: Optional[str], label_simulation: Optional[str], i_cut: int):
        self.message_df_simulation = pd.DataFrame(self.message_dict).iloc[: i_cut + 1, :]
        self.ob_df_simulation = pd.DataFrame(self.ob_dict).iloc[: i_cut + 1, :]

        self.message_df_simulation["Type"] = self.message_df_simulation["Type"].replace(
            [0, 1, 2, 3, 4, 5],
            ["LO", "MO", "C", "LOUser", "AggressiveMOUser", "PassiveMOUser"],
        )

        increment_prices = self.message_df_simulation["Shift"].cumsum() + self.p0

        # MidPrice is computed on the *current* (post-shift) index grid, so it uses the
        # current cumulative shift.
        self.message_df_simulation["MidPrice"] += increment_prices.to_numpy()

        # Price is recorded in the *pre-shift* index space (i.e., before applying "Shift"
        # for that same row). Therefore, to convert Price to the absolute index space we
        # must use the cumulative shift up to the *previous* row.
        inc = increment_prices.to_numpy()
        price_offsets_pre_shift = np.empty_like(inc)
        price_offsets_pre_shift[0] = self.p0
        price_offsets_pre_shift[1:] = inc[:-1]
        # Preserve -1 sentinels (no-op cancel on empty side) and NaN
        def _offset_preserving_sentinel(col, offsets):
            vals = np.asarray(col, dtype=float).copy()
            sentinel_mask = (vals < 0) | np.isnan(vals)
            vals += offsets
            vals[sentinel_mask] = np.nan
            return vals

        self.message_df_simulation["Price"] = _offset_preserving_sentinel(
            self.message_df_simulation["Price"].to_numpy(), price_offsets_pre_shift
        )

        # Best bid/ask indices are recorded after recentering for that row -> use the
        # current cumulative shift.  Preserve sentinel values (-1 or NaN) that indicate
        # an empty side, so they are not corrupted by the offset addition.

        for _col_name in ("IndBestBid", "IndBestAsk", "BestBidPrice", "BestAskPrice"):
            if _col_name in self.message_df_simulation.columns:
                self.message_df_simulation[_col_name] = _offset_preserving_sentinel(
                    self.message_df_simulation[_col_name].to_numpy(), inc
                )

        columns_ob_df_prices = [c for c in self.ob_df_simulation.columns if "Price" in c]
        for column in columns_ob_df_prices:
            self.ob_df_simulation[column] += increment_prices.to_numpy()

        self.message_df_simulation.drop("Shift", axis=1, inplace=True)
        self.fix_zero_size()

        if isinstance(path_save_files, str) and isinstance(label_simulation, str):
            self.message_df_simulation.to_csv(path_save_files + "message_file_simulation_" + label_simulation + ".csv")
            self.ob_df_simulation.to_csv(path_save_files + "ob_file_simulation_" + label_simulation + ".csv")


# =============================================================================
# Top-level function simulate_LOB (updated for per-fill sweep logging + per-fill snapshots)
# =============================================================================

def simulate_LOB(
    lam: float,
    mu: float,
    delta: float,
    number_tick_levels: int,
    n_priority_ranks: int,
    number_levels_to_store: int = 20,
    p0: int = 0,
    mean_size_LO: float = 1,
    mean_size_MO: float = 1,
    iterations: int = 50_000,
    iterations_to_equilibrium: int = 10_000,
    path_save_files: Optional[str] = None,
    label_simulation: Optional[str] = None,
    beta_exp_weighted_return: float = 1e-3,
    intensity_exp_weighted_return: float = 1e-3,
    random_seed: Optional[int] = None,
    buy_mo_prob: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """
    Updated simulate_LOB:
    - Uses simulate_order(..., split_sweeps=True) so that sweep MOs become multiple message rows.
    - Logs per-fill snapshots and per-fill metrics (Spread/Mid/Return/etc) returned by simulate_order.
    - Keeps your save_results() conventions correct by ensuring:
        * Shift is nonzero ONLY on the last sub-row of each parent event.
        * Snapshot/MidPrice for intermediate sub-rows are pre-recenter (Shift=0).
        * Snapshot/MidPrice for the final sub-row are post-recenter (Shift=shift).

    NOTE ON TIME:
    - We advance the simulation clock ONCE per parent event (shared time for all sub-fills).
    - The Δt for each parent event is generated internally by draw_next_order_type() and stored
      in LOB_sim.last_event_dt.
    """

    rng = (
        np.random.RandomState(int(random_seed))
        if random_seed is not None
        else np.random.RandomState()
    )

    LOB_sim = LOB_simulation(
        number_tick_levels,
        n_priority_ranks,
        p0,
        mean_size_LO=mean_size_LO,
        number_levels_to_store=number_levels_to_store,
        beta_exp_weighted_return=beta_exp_weighted_return,
        intensity_exp_weighted_return=intensity_exp_weighted_return,
        mean_size_MO=mean_size_MO,
        rng=rng,
        buy_mo_prob=buy_mo_prob,
    )
    LOB_sim.initialize()

    # Warm-up phase (not logged, same behavior as original)
    for _ in tqdm(range(iterations_to_equilibrium)):
        _ = LOB_sim.simulate_order(
            lam=lam,
            mu=mu,
            delta=delta,
            split_sweeps=False,
        )

    # Reset EWMA accumulator after warm-up
    LOB_sim.exp_weighted_return_to_store = 0.0

    t_i: float = 0.0
    exp_weighted_return_list: List[float] = []

    # Bug #15: message_dict and ob_dict grow unboundedly via .append() during the main loop.
    # For very long simulations (iterations > 1M), consider pre-allocating lists with
    # estimated capacity (e.g., `[None] * iterations` and using index assignment) to reduce
    # list resize overhead. The lists are converted to DataFrames in save_results().

    # Count the ACTUAL number of logged rows (can exceed `iterations` due to MO sweeps).
    row_idx: int = 0

    for _ in tqdm(range(iterations)):
        # Bug #13: set simulation time on the engine before each event
        LOB_sim.time = t_i
        # simulate_order returns per-fill rows/metrics/snapshots
        rows, metrics, snaps = LOB_sim.simulate_order(
            lam=lam,
            mu=mu,
            delta=delta,
            split_sweeps=True,
        )

        # Advance time ONCE per parent event (same timestamp for all sub-fills).
        # NOTE on timestamp convention:
        #   LOB_sim.time (set above) = t_i BEFORE the event → used by
        #   _capture_metrics() for state snapshots (pre-event state).
        #   t_i AFTER increment = end-of-event time → recorded in the
        #   message tape.  This means the tape's Time column is the
        #   arrival time of the NEXT event, not the current one.
        #   This is a logging inconsistency (not an RL bug) — the RL
        #   controller uses state_before/state_after, not timestamps.
        t_i += float(getattr(LOB_sim, "last_event_dt", 0.0))

        # Log each sub-row
        for (order_type, order_direction, order_price, shift_lob_state, event_size), m, snap in zip(
            rows, metrics, snaps
        ):
            # Message tape fields
            LOB_sim.message_dict["Time"].append(float(t_i))
            LOB_sim.message_dict["Type"].append(int(order_type))
            LOB_sim.message_dict["Direction"].append(int(order_direction))
            LOB_sim.message_dict["Price"].append(int(order_price))
            LOB_sim.message_dict["Shift"].append(int(shift_lob_state))
            LOB_sim.message_dict["Size"].append(float(event_size))

            # Per-fill state-derived metrics (already computed inside simulate_order)
            LOB_sim.message_dict["Spread"].append(float(m["Spread"]))
            LOB_sim.message_dict["MidPrice"].append(float(m["MidPrice"]))
            LOB_sim.message_dict["Return"].append(float(m["Return"]))
            LOB_sim.message_dict["TotNumberBidOrders"].append(float(m["TotNumberBidOrders"]))
            LOB_sim.message_dict["TotNumberAskOrders"].append(float(m["TotNumberAskOrders"]))
            LOB_sim.message_dict["IndBestBid"].append(float(m["IndBestBid"]))
            LOB_sim.message_dict["IndBestAsk"].append(float(m["IndBestAsk"]))
            LOB_sim.message_dict["BestBidPrice"].append(float(m.get("BestBidPrice", np.nan)))
            LOB_sim.message_dict["BestAskPrice"].append(float(m.get("BestAskPrice", np.nan)))

            
            # Cancellation queue-position fields (NaN for non-cancel rows)
            LOB_sim.message_dict["CancelVolAhead"].append(float(m.get("CancelVolAhead", float("nan"))))
            LOB_sim.message_dict["CancelQueueLen"].append(float(m.get("CancelQueueLen", float("nan"))))
            LOB_sim.message_dict["CancelRank"].append(float(m.get("CancelRank", float("nan"))))


            # Per-fill snapshot row
            for k, v in snap.items():
                # setdefault makes this robust if ob_dict keys weren't pre-created
                LOB_sim.ob_dict.setdefault(k, []).append(v)

            # Track EWMA per logged row (this value is constant across subrows of the same parent event)
            exp_weighted_return_list.append(float(LOB_sim.exp_weighted_return_to_store))

            row_idx += 1

    # Cut index must be based on the last logged ROW, not the number of parent events
    i_cut = max(row_idx - 1, 0)

    LOB_sim.save_results(path_save_files, label_simulation, i_cut)

    return LOB_sim.message_df_simulation, LOB_sim.ob_df_simulation, exp_weighted_return_list
