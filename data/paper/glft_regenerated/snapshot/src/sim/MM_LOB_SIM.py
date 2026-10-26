#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 04:28:36 2025
Updated on Sat Jan 31 22:33:20 2026

@author: felipemoret
"""

"""
HYBRID MarketMaker (Simulation + Backtest with queue-ahead)

- Simulation mode: interacts with LOB_simulation engine.
- Backtest mode: uses only external snapshots (no engine calls).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import inspect
import random
import warnings

from LOB_SIM_SANTA_FE import LOB_simulation
from RLController import RLController
from bayes_online_MM_input import (
    BayesOnlineMMInput,
    BayesOnlineMMInputConfig,
    BayesOnlineMMMixtureInput,
)


# =============================================================================
# Factory: create Santa Fe or QRM engine from a single interface
# =============================================================================

def _create_lob_engine(
    *,
    number_tick_levels: int,
    n_priority_ranks: int,
    p0: int,
    mean_size_LO: float,
    number_levels_to_store: int,
    beta_exp_weighted_return: float,
    intensity_exp_weighted_return: float,
    mean_size_MO: float,
    rng,
    buy_mo_prob: Union[float, Callable] = 0.5,
    # --- QRM selection ---
    qrm_params: Optional[dict] = None,
):
    """
    Instantiate the LOB engine.

    If *qrm_params* is ``None`` (default), creates a Santa Fe engine
    (``LOB_simulation``).  Otherwise creates ``LOB_simulation_QRM``
    with the QRM-specific keys contained in *qrm_params*.

    ``buy_mo_prob`` can be a float (constant) or a callable
    ``(step_idx: int) -> float`` for regime-switching MO flow.

    Expected *qrm_params* keys (all optional with defaults in the
    QRM constructor):
        intens_val, intens_val_bis, statprob,
        theta, theta_reinit, tick_qrm,
        size_q, size_s, aes,
        intensity_model, use_dynamic_pref,
        intens_levels, intens_k1
    """
    # --- Shared kwargs (parent constructor) ---
    base_kw = dict(
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,
        p0=p0,
        mean_size_LO=mean_size_LO,
        number_levels_to_store=number_levels_to_store,
        beta_exp_weighted_return=beta_exp_weighted_return,
        intensity_exp_weighted_return=intensity_exp_weighted_return,
        mean_size_MO=mean_size_MO,
        rng=rng,
        buy_mo_prob=buy_mo_prob,
    )

    if qrm_params is not None:
        from LOB_SIM_QRM import LOB_simulation_QRM
        return LOB_simulation_QRM(**base_kw, **qrm_params)

    return LOB_simulation(**base_kw)

from tqdm import tqdm

# =============================================================================
# Utility: keep dict-of-lists aligned for safe DataFrame construction
# =============================================================================
def _align_dict_of_lists(d: Dict[str, Any], target_len: int, fill_value=np.nan) -> None:
    """Pad or truncate every list-like value in `d` to `target_len`.

    Why this exists:
    - Over time, the simulator/runner may introduce NEW message fields (extra keys)
      that are not appended every step (e.g., CancelVolAhead/CancelQueueLen, sweep metadata).
    - Pandas requires all arrays to be the same length when building DataFrames.

    We keep this minimal and robust:
    - If a value is a list/tuple/np.ndarray, we coerce to a Python list.
    - If it is missing entries, we pad with `fill_value`.
    - If it is longer, we truncate.
    """
    if target_len < 0:
        target_len = 0

    for k in list(d.keys()):
        v = d[k]

        # Coerce common containers to a mutable Python list
        if isinstance(v, np.ndarray):
            v_list = v.tolist()
        elif isinstance(v, tuple):
            v_list = list(v)
        elif isinstance(v, list):
            v_list = v
        else:
            # Non-list scalars are not expected in message_dict/ob_dict;
            # wrap them into a list so we can align.
            v_list = [v]

        n = len(v_list)
        if n < target_len:
            v_list.extend([fill_value] * (target_len - n))
        elif n > target_len:
            del v_list[target_len:]

        d[k] = v_list

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HYBRID MarketMaker (Simulation + Backtest with queue-ahead).

Key idea for backtest PnL
-------------------------
In backtest mode the price grid can be centered at 0 (indices can be
negative). Fills and FPT logic work in index space, but for PnL we
usually want to value inventory at a "real" price:

    real_price = (base_price_idx + px) * tick_size

To get this behaviour we introduce:

    - self.base_price_idx
    - a helper _px_to_dollars(px)

and we use this helper only when converting indices to dollar prices:
    - mark_to_market()
    - cross_mo() (both branches)
    - post_step_detect_fill_by_env_mo() (simulation passive fills)
    - register_backtest_passive_fill()
"""

# ============================================================
# FIFOQueueTracker — exact queue position from LOBSTER order IDs
# ============================================================

class FIFOQueueTracker:
    """Maintains per-price FIFO queues of (order_id, remaining_size).

    Updated incrementally from LOBSTER messages.  Used by the backtest
    to determine the exact number of shares ahead of the MM's order,
    replacing the stochastic approximation.

    Internally keeps two lookup structures:
      - queues[price]: list of [order_id, remaining_size]  (FIFO order)
      - oid_loc[order_id]: price  (for O(1) lookup of which queue an OID lives in)
    """

    __slots__ = ("queues", "oid_loc")

    def __init__(self):
        self.queues: Dict[int, list] = {}          # price -> [[oid, rem_size], ...]
        self.oid_loc: Dict[int, int] = {}           # oid -> price

    # ------------------------------------------------------------------
    def process_event(self, price: int, order_id: int, lobster_type: int,
                      size: float, direction: int) -> None:
        """Process one LOBSTER event to update queues.

        lobster_type: 1=new LO, 2=partial cancel, 3=full cancel,
                      4=visible exec, 5=hidden exec, 6/7=misc.
        """
        if order_id < 0:
            return

        if lobster_type == 1:
            # New limit order → append to back of queue at this price
            q = self.queues.setdefault(price, [])
            q.append([order_id, float(size)])
            self.oid_loc[order_id] = price

        elif lobster_type in (2, 3):
            # Cancellation (partial or full)
            self._reduce_or_remove(order_id, float(size), full=(lobster_type == 3))

        elif lobster_type == 4:
            # Visible execution — reduce size; remove if depleted
            self._reduce_or_remove(order_id, float(size), full=False)

        # Type 5 (hidden exec), 6, 7: ignore — not in visible book

    # ------------------------------------------------------------------
    def _reduce_or_remove(self, order_id: int, size: float, full: bool) -> None:
        px = self.oid_loc.get(order_id)
        if px is None:
            return
        q = self.queues.get(px)
        if q is None:
            self.oid_loc.pop(order_id, None)
            return
        for i, entry in enumerate(q):
            if entry[0] == order_id:
                if full:
                    q.pop(i)
                    self.oid_loc.pop(order_id, None)
                else:
                    entry[1] -= size
                    if entry[1] <= 0:
                        q.pop(i)
                        self.oid_loc.pop(order_id, None)
                return
        # OID not found in queue (already removed) — clean up loc
        self.oid_loc.pop(order_id, None)

    # ------------------------------------------------------------------
    def volume_ahead_of_mm(self, price: int) -> float:
        """Total volume at *price* (all orders ahead of the MM, which is last)."""
        q = self.queues.get(price)
        if not q:
            return 0.0
        return sum(e[1] for e in q)

    def queue_snapshot(self, price: int) -> list:
        """Return a copy of the queue at *price* for storing in MM order info."""
        q = self.queues.get(price)
        if not q:
            return []
        return [[oid, sz] for oid, sz in q]

    def remaining_for_oid(self, order_id: int) -> float:
        """Return remaining size for a specific order, or 0 if gone."""
        px = self.oid_loc.get(order_id)
        if px is None:
            return 0.0
        q = self.queues.get(px)
        if q is None:
            return 0.0
        for oid, sz in q:
            if oid == order_id:
                return sz
        return 0.0

    # ------------------------------------------------------------------
    # MM sentinel support: track the MM's virtual position in the FIFO
    # ------------------------------------------------------------------
    def add_mm_order(self, price: int, mm_oid: int) -> float:
        """Insert MM's virtual sentinel at the back of the queue at *price*.

        The sentinel has size 0 (it is not a real order in the LOBSTER tape).
        Returns the volume ahead of the MM at insertion time.
        """
        q = self.queues.setdefault(price, [])
        vol_ahead = sum(e[1] for e in q)
        q.append([mm_oid, 0.0])          # size=0 sentinel
        self.oid_loc[mm_oid] = price
        return vol_ahead

    def remove_mm_order(self, mm_oid: int) -> None:
        """Remove MM's sentinel from its queue."""
        self._reduce_or_remove(mm_oid, 0.0, full=True)

    def volume_ahead_of(self, price: int, mm_oid: int) -> float:
        """Return total volume in the queue at *price* strictly ahead of *mm_oid*.

        Orders appended after the sentinel are behind the MM and excluded.
        """
        q = self.queues.get(price)
        if not q:
            return 0.0
        total = 0.0
        for oid, sz in q:
            if oid == mm_oid:
                break
            total += sz
        return total

    # ------------------------------------------------------------------
    def apply_shift(self, shift: int) -> None:
        """Re-key all price levels after a grid re-centering."""
        if shift == 0:
            return
        new_queues: Dict[int, list] = {}
        for px, q in self.queues.items():
            new_queues[px - shift] = q
        self.queues = new_queues
        # Update oid_loc prices
        for oid in self.oid_loc:
            self.oid_loc[oid] -= shift


# ============================================================
# MarketMaker — HYBRID (simulation + backtest)
# ============================================================

class MarketMaker:
    """
    Unified MarketMaker wrapper that supports:

    (A) Simulation mode (backtest_mode = False)
    - Interacts with the LOB_simulation engine.
    - Uses real 'owner' and 'priorities_lob_state'.
    - Detects fills via environment market orders.
    - Mirrors engine recentering via apply_shift().

    (B) Backtest mode (backtest_mode = True)
    - Does NOT touch the engine.
    - Does NOT modify lob_state or priorities.
    - Keeps working orders only in a local dictionary.
    - Passive execution via:
        * FPT (First Passage Time)
        * optional queue-ahead estimation (use_queue_ahead).
    - Uses external snapshots for:
        * best bid/ask, spread, OBI, full depth.
    - cross_mo() uses only the snapshot best bid/ask.

    Additional pure market-making mode
    ----------------------------------
    pure_MM : bool
        When True, we assume we are running a "pure" quoting policy
        (e.g. GLFT or Avellaneda-Stoikov) without a Deep RL controller.

        In this mode:
        - The MarketMaker still exposes a state dictionary via build_state().
        - However, bidsize/asksize are interpreted as the *total queue size*
          on bid/ask up to some maximum offset from the best bid/ask.

        The maximum offsets are inferred from:

            pure_mm_offsets = [
                (bid_offset_0, ask_offset_0),
                (bid_offset_1, ask_offset_1),
                ...
            ]

        Example:
            pure_mm_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
            → max_bid_offset = 1
              max_ask_offset = 1

        Meaning:
        - For bids we aggregate depth at levels:
              best_bid (offset 0), best_bid-1 (offset 1), ...
        - For asks we aggregate depth at levels:
              best_ask (offset 0), best_ask+1 (offset 1), ...

        This is only applied in backtest mode, where we have full_depth_bids
        and full_depth_asks in the environment snapshot.
    """

    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def __init__(
        self,
        lob: Optional["LOB_simulation"],
        policy: Optional[Callable[[Dict[str, Any]], tuple]],
        exclude_self_from_state: bool = True,
        use_sticky_price: bool = True,
        tick_size: float = 0.01,
        backtest_mode: bool = False,
        use_queue_ahead: bool = False,
        base_price_idx: float = 0.0,
        pure_MM: bool = False,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        max_offset_override: Optional[int] = None,
        max_inventory: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        # Engine handle (None when backtest)
        self.lob = lob
        self.policy = policy

        # Register forced-cancel callback if the engine supports it (QRM).
        # This notifies the MM before reinit/regen clears the book, allowing
        # immediate bookkeeping cleanup instead of delayed _prune_orphan_orders.
        if lob is not None and hasattr(lob, "on_forced_cancel"):
            lob.on_forced_cancel = self._handle_forced_cancel

        # Maximum inventory for reward normalization.
        # Used by reward functions to normalize inv to [-1, 1].
        # If None, reward functions fall back to 10.0 (legacy default).
        self.max_inventory: Optional[float] = (
            float(max_inventory) if max_inventory is not None else None
        )

        # Modes
        self.backtest_mode: bool = bool(backtest_mode)
        self.use_queue_ahead: bool = bool(use_queue_ahead)

        # State / price conversion
        self.exclude_self_from_state = exclude_self_from_state
        self.use_sticky_price = bool(use_sticky_price)
        
        self.tick_size: float = float(tick_size)
        self.base_price_idx: float = float(base_price_idx)
        self.rng = rng if rng is not None else np.random.RandomState()

        # Pure market-making mode (used mainly in backtest with GLFT/AS policies)
        self.pure_MM: bool = bool(pure_MM)

        # Offsets used by the pure_MM policy, if any.
        self.pure_mm_offsets: List[Tuple[int, int]] = (
            list(pure_mm_offsets) if pure_mm_offsets is not None else []
        )

        # Maximum offsets inferred from pure_mm_offsets.
        self.max_bid_offset: int = 0
        self.max_ask_offset: int = 0
        self.max_offset: int = 0

        # Infer effective offsets for pure_MM mode (no effect if pure_MM=False).
        self._infer_offsets_from_pure_mm_actions(max_offset_override)

        # MO flow EWMA tracker (for regime-switching / adversarial state signal)
        # Updated each time a market order arrives in the simulation loop.
        # Value in [-1, +1]: +1 = all buy MOs, -1 = all sell MOs, 0 = balanced.
        self.mo_flow_ewma: float = 0.0
        self.mo_flow_ewma_alpha: float = 0.03  # smoothing factor (~1/30, matches mean regime duration)
        # Fast MO-flow EWMA tracker.
        #
        # The slow tracker above is useful for detecting persistent order-flow
        # imbalance, but it intentionally reacts sluggishly to local reversals.
        # We therefore maintain a second, faster smoother so downstream
        # controllers can distinguish background pressure from short-horizon
        # tactical pressure.
        #
        # By default, the fast alpha corresponds to an approximately 4x shorter
        # memory horizon than the slow tracker. Callers may override it
        # explicitly via the simulation API when a different calibration is
        # desired.
        self.mo_flow_fast_ewma: float = 0.0
        self.mo_flow_fast_ewma_alpha: float = min(1.0, 4.0 * self.mo_flow_ewma_alpha)
        # Quantity scale used to normalize signed MO size before clipping to [-1, +1].
        _default_mo_q = 1.0
        if self.lob is not None:
            try:
                _default_mo_q = float(getattr(self.lob, "mean_size_MO", 1.0))
            except Exception:
                _default_mo_q = 1.0
        self.mo_flow_q_scale: float = max(float(_default_mo_q), 1.0)

        # Bayesian MO-flow filter.  It is disabled by default, so legacy runs
        # keep the exact same controller input unless explicitly requested.
        self.use_bayes_flow_signal: bool = False
        self.mo_flow_bayes = BayesOnlineMMInput()

        # Fill imbalance tracker (endogenous signal � how our quotes are being hit).
        # We keep separate bid/ask EWMAs and expose their normalized bias so the
        # feature reflects which side is being hit more, not just fill sparsity.
        # Warm-start both sides with a small equal prior so the normalized
        # bias (B-A)/(B+A) starts at 0 instead of spiking to ±1 on the first fill.
        self.fill_imbalance_alpha: float = 0.1
        _fill_prior = 2.0 * float(self.fill_imbalance_alpha)
        self.fill_imbalance_ewma: float = 0.0
        self.fill_bid_ewma: float = _fill_prior
        self.fill_ask_ewma: float = _fill_prior
        # Quantity scale used to normalize signed fill size before clipping to [-1, +1].
        # In simulation mode this defaults to the structural MO size; in backtest it
        # falls back to 1.0 unless the caller overrides it.
        _default_fill_q = 1.0
        if self.lob is not None:
            try:
                _default_fill_q = float(getattr(self.lob, "mean_size_MO", 1.0))
            except Exception:
                _default_fill_q = 1.0
        self.fill_imbalance_q_scale: float = max(float(_default_fill_q), 1.0)
        self.use_quote_exposure_imbalance_signal: bool = False

        # Invariant check frequency: run _check_invariants every N steps
        # instead of every step.  Set to 0 to disable entirely.
        # Default 100 = check every 100th step (50× fewer checks than every step).
        self._invariant_check_interval: int = 100
        self._step_counter: int = 0

        # Core MM variables
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.orders: Dict[int, Dict[str, Any]] = {}
        self.next_oid: int = 1

        # Owner matrix (simulation mode)
        if (self.lob is not None) and hasattr(self.lob, "priorities_lob_state"):
            self.owner = np.zeros_like(self.lob.priorities_lob_state, dtype=bool)
            n_priority_ranks = self.lob.n_priority_ranks
        else:
            # Dummy owner for backtest; shape is irrelevant
            self.owner = np.zeros((1, 1), dtype=bool)
            n_priority_ranks = 1

        # Telemetry log
        self.log: List[Dict[str, Any]] = []

        # Last fill information (sim + backtest)
        self.last_fill: Optional[Dict[str, Any]] = None

        # External snapshot state for backtest
        self._last_env_snapshot: Optional[Dict[str, Any]] = None

        # Last policy state and action (for logging / debugging)
        self.last_policy_state: Optional[Dict[str, Any]] = None
        self.last_policy_action: Optional[tuple] = None
        
        # Cache for policy calling convention (plain vs pass mm=self)
        self._policy_call_cached_obj = None
        self._policy_call_kind = None   # "plain" | "mm_kw" | "mm_pos"

        # Cancel-protection callbacks (simulation only)
        # --------------------------------------------
        # Some runners prefer to (re)attach these providers explicitly via:
        #   lob.set_cancel_avoid_mask_provider(mm.cancel_avoid_mask_provider)
        # Therefore we store them as attributes as well (even though we also
        # attach them to the engine here in __init__).
        self.cancel_avoid_mask_provider = None  # type: Optional[Callable[[int], np.ndarray]]
        self.cancel_filter = None              # type: Optional[Callable[[int, np.ndarray], int]]

        # ---------------------------------------------------------
        # Cancel-protection callbacks (simulation only)
        # ---------------------------------------------------------
        if (self.lob is not None) and not self.backtest_mode:

            def _avoid_mask(price: int) -> np.ndarray:
                """
                Self-trade avoidance mask for cancellations in the engine.

                Marks our own queue positions at the given price as 'avoid'.

                IMPORTANT ROBUSTNESS NOTE
                -------------------------
                The engine may *dynamically* expand `lob.priorities_lob_state` along the
                FIFO-rank dimension when a queue becomes full (see LOB_simulation.add_order_to_queue()).
                Therefore, we must build the avoid-mask using the CURRENT engine shape,
                not the init-time `n_priority_ranks` captured in this closure.
                """
                if self.lob is None:
                    # Should never happen in this closure, but keep it safe.
                    return np.zeros(1, dtype=bool)

                ref = getattr(self.lob, "priorities_lob_state", None)
                if ref is None:
                    return np.zeros(1, dtype=bool)

                # Ensure our owner matrix matches the current engine matrix.
                owner_mat = self._owner_mask_like(ref)

                n_ranks = int(ref.shape[0])
                n_prices = int(ref.shape[1])

                if price < 0 or price >= n_prices:
                    return np.zeros(n_ranks, dtype=bool)

                # Return a COPY to avoid accidental external mutation.
                return np.asarray(owner_mat[:, int(price)], dtype=bool).copy()


            def _cancel_filter(price: int, candidates: np.ndarray) -> int:
                """
                Simple random selection among allowed cancellation candidates.
                """
                return int(self.rng.choice(candidates))

            # Expose providers as MM attributes (useful for external runners).
            self.cancel_avoid_mask_provider = _avoid_mask
            self.cancel_filter = _cancel_filter

            # Attach to engine.
            self.lob.set_cancel_avoid_mask_provider(_avoid_mask)
            self.lob.set_cancel_filter(_cancel_filter)

    # ======================================================================
    # Helper: infer max offsets for pure_MM from (bid_offset, ask_offset) tuples
    # ======================================================================
    def _infer_offsets_from_pure_mm_actions(self, max_offset_override: Optional[int]) -> None:
        if max_offset_override is not None:
            m = int(max_offset_override)
            self.max_bid_offset = max(0, m)
            self.max_ask_offset = max(0, m)
            self.max_offset = max(self.max_bid_offset, self.max_ask_offset)
            return

        if not self.pure_mm_offsets:
            self.max_bid_offset = 0
            self.max_ask_offset = 0
            self.max_offset = 0
            return

        self.max_bid_offset = max(int(b) for (b, a) in self.pure_mm_offsets)
        self.max_ask_offset = max(int(a) for (b, a) in self.pure_mm_offsets)
        self.max_bid_offset = max(0, self.max_bid_offset)
        self.max_ask_offset = max(0, self.max_ask_offset)
        self.max_offset = max(self.max_bid_offset, self.max_ask_offset)

    # ======================================================================
    # Price conversion helper
    # ======================================================================
    def _px_to_dollars(self, px: float) -> float:
        # Use the same mapping in BOTH modes: absolute_idx = base_price_idx + px
        return (self.base_price_idx + float(px)) * self.tick_size

    # ======================================================================
    # MO-flow EWMA helpers
    # ======================================================================
    def _mo_flow_signal(
        self,
        order_direction: int,
        order_size: float = 1.0,
    ) -> float:
        """
        Build the signed MO-flow signal for one market order.

        The signal is normalized by a structural MO size scale and clipped to
        [-1, +1]:

            s_t = clip(signed_mo_size / q_scale, -1, +1)
        """
        side = int(np.sign(order_direction))
        if side == 0:
            return 0.0
        q_scale = float(getattr(self, "mo_flow_q_scale", 1.0))
        if (not np.isfinite(q_scale)) or q_scale <= 0.0:
            q_scale = 1.0
        qty = max(float(order_size), 0.0)
        signed_qty = float(side) * qty
        return float(np.clip(signed_qty / q_scale, -1.0, 1.0))

    def update_mo_flow_ewma(
        self,
        order_direction: int,
        order_size: float = 1.0,
    ) -> float:
        """
        Update the slow and fast MO-flow EWMAs using signed, size-aware
        market-order flow.

        Both smoothers consume the same clipped per-order signal. Their only
        difference is the memory horizon induced by their respective alphas.
        """
        alpha = float(self.mo_flow_ewma_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        alpha_fast = float(getattr(self, "mo_flow_fast_ewma_alpha", alpha))
        alpha_fast = min(max(alpha_fast, 0.0), 1.0)
        signal = self._mo_flow_signal(
            order_direction=order_direction,
            order_size=order_size,
        )
        self.mo_flow_ewma = alpha * signal + (1.0 - alpha) * self.mo_flow_ewma
        self.mo_flow_fast_ewma = (
            alpha_fast * signal + (1.0 - alpha_fast) * self.mo_flow_fast_ewma
        )
        return signal

    def configure_bayes_flow_signal(
        self,
        *,
        enabled: bool = False,
        tau_r: float = 60.0,
        p_low: float = 0.2,
        p_high: float = 0.8,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        prior_mode: str = "truncated_beta_grid",
        hazard: Optional[float] = None,
        max_run_length: Optional[int] = None,
        cp_window: int = 5,
        use_mixture: bool = False,
        mixture_taus: Optional[List[float]] = None,
        mixture_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Configure the optional Bayesian online changepoint MO-flow filter.

        The filter is intentionally separate from the legacy slow/fast EWMAs:
        when enabled, callers may append its four features to the controller
        state; when disabled, legacy behavior is unchanged.
        """
        self.use_bayes_flow_signal = bool(enabled)
        base_config = BayesOnlineMMInputConfig(
            tau_r=float(tau_r),
            prior_a=float(prior_a),
            prior_b=float(prior_b),
            prior_mode=str(prior_mode),
            p_low=float(p_low),
            p_high=float(p_high),
            hazard=hazard,
            max_run_length=max_run_length,
            cp_window=int(cp_window),
        )
        if bool(use_mixture):
            self.mo_flow_bayes = BayesOnlineMMMixtureInput.from_base_config(
                base_config=base_config,
                tau_grid=mixture_taus,
                prior_weights=mixture_weights,
            )
        else:
            self.mo_flow_bayes = BayesOnlineMMInput(base_config)

    def update_mo_flow_bayes(
        self,
        order_direction: int,
        order_size: float = 1.0,
    ) -> Dict[str, float]:
        """Update the optional Bayesian MO-flow filter from one market order."""
        if not bool(getattr(self, "use_bayes_flow_signal", False)):
            return self.get_bayes_flow_features()
        return self.mo_flow_bayes.update_from_order_direction(
            order_direction=order_direction,
            order_size=order_size,
        )

    def get_bayes_flow_features(self) -> Dict[str, float]:
        """Return the four Bayesian features expected by DeepRLController."""
        filt = getattr(self, "mo_flow_bayes", None)
        if filt is None:
            return {
                "bayes_m_hat": 0.0,
                "bayes_cp_prob": 0.0,
                "bayes_expected_run_length": 0.0,
                "bayes_uncertainty": 0.0,
            }
        return filt.feature_dict(prefix="bayes")

    def get_bayes_flow_telemetry(self) -> Dict[str, float]:
        """Return Bayesian features plus raw diagnostics for logging."""
        filt = getattr(self, "mo_flow_bayes", None)
        if filt is None:
            out = self.get_bayes_flow_features()
            out.update(
                {
                    "bayes_p_hat": 0.5,
                    "bayes_expected_run_length_raw": 0.0,
                    "bayes_n_components": 0.0,
                    "bayes_n_mo_updates": 0.0,
                    "bayes_hazard": 0.0,
                    "bayes_mixture_enabled": 0.0,
                    "bayes_mixture_n_models": 0.0,
                    "bayes_mixture_tau_mean": 0.0,
                }
            )
            return out
        return filt.telemetry_dict(prefix="bayes")

    def get_mo_flow_p_hat(self) -> float:
        """
        Return the EWMA-implied local buy-MO probability estimate in [0, 1].
        """
        return float(np.clip(0.5 * (float(self.mo_flow_ewma) + 1.0), 0.0, 1.0))

    def get_mo_flow_fast_p_hat(self) -> float:
        """
        Return the fast-EWMA local buy-MO probability estimate in [0, 1].

        This short-horizon estimate is meant to complement the slower
        ``get_mo_flow_p_hat()`` signal, not replace it.
        """
        return float(np.clip(0.5 * (float(self.mo_flow_fast_ewma) + 1.0), 0.0, 1.0))

    # ======================================================================
    # Fill-imbalance EWMA helpers
    # ======================================================================
    def _fill_activity_signal(
        self,
        fill_qty: float = 0.0,
    ) -> float:
        """
        Build the per-side fill-activity signal for one step.

        The signal is quantity-aware and clipped to [0, 1]:

            x_t = clip(fill_qty / q_scale, 0, 1)
        """
        q_scale = float(getattr(self, "fill_imbalance_q_scale", 1.0))
        if (not np.isfinite(q_scale)) or q_scale <= 0.0:
            q_scale = 1.0
        qty = max(float(fill_qty), 0.0)
        return float(np.clip(qty / q_scale, 0.0, 1.0))

    def get_fill_imbalance_bias(self) -> float:
        """
        Return the normalized fill-side bias in [-1, +1].

        +1 means recent bid-side fills dominate, -1 means ask-side fills dominate,
        and 0 means balanced or no recent fill activity.
        """
        bid = max(float(getattr(self, "fill_bid_ewma", 0.0)), 0.0)
        ask = max(float(getattr(self, "fill_ask_ewma", 0.0)), 0.0)
        denom = bid + ask
        if denom <= 1e-12:
            return 0.0
        return float(np.clip((bid - ask) / denom, -1.0, 1.0))

    def reset_fill_imbalance_prior(self, prior_mult: float = 2.0) -> None:
        """
        Reinitialize the bid/ask fill EWMAs to a small symmetric prior tied to
        the currently configured fill_imbalance_alpha.

        This avoids an artificial +/-1 spike on the first unilateral fill while
        keeping the prior scale principled and automatically consistent with the
        chosen fill-memory horizon.
        """
        alpha = min(max(float(self.fill_imbalance_alpha), 0.0), 1.0)
        prior = max(float(prior_mult) * alpha, 0.0)
        self.fill_bid_ewma = float(prior)
        self.fill_ask_ewma = float(prior)
        self.fill_imbalance_ewma = 0.0

    def configure_quote_exposure_imbalance_signal(
        self,
        *,
        enabled: bool = False,
    ) -> None:
        """Configure the tau-free current quote-exposure imbalance signal."""
        self.use_quote_exposure_imbalance_signal = bool(enabled)

    def get_current_fill_opportunity(self) -> Tuple[float, float]:
        """
        Return queue-adjusted bid/ask opportunity created by the current policy.

        A resting order with more quantity and less queue ahead creates more
        opportunity to be filled.  This is intentionally local and tau-free:
        it measures what the policy actually exposed to the market at this step.
        """
        q_scale = max(float(getattr(self, "fill_imbalance_q_scale", 1.0)), 1.0)
        bid_opp = 0.0
        ask_opp = 0.0
        for info in getattr(self, "orders", {}).values():
            try:
                side = int(info.get("side", 0))
                qty = max(float(info.get("qty", 1.0)), 0.0) / q_scale
                if bool(getattr(self, "backtest_mode", False)):
                    queue_ahead = float(info.get("queue_ahead", 0.0))
                else:
                    queue_ahead = float(info.get("priority", 0.0))
            except Exception:
                continue
            if qty <= 0.0:
                continue
            queue_ahead = max(queue_ahead, 0.0)
            opportunity = qty / (1.0 + queue_ahead)
            if side > 0:
                bid_opp += opportunity
            elif side < 0:
                ask_opp += opportunity
        return float(bid_opp), float(ask_opp)

    def get_quote_exposure_imbalance_features(self) -> Dict[str, float]:
        """Return current queue-adjusted quote exposure imbalance."""
        bid_opp, ask_opp = self.get_current_fill_opportunity()
        denom = 1.0 + float(bid_opp) + float(ask_opp)
        imbalance = (float(bid_opp) - float(ask_opp)) / denom
        raw_ratio = 0.0
        raw_denom = float(bid_opp) + float(ask_opp)
        if raw_denom > 1e-12:
            raw_ratio = (float(bid_opp) - float(ask_opp)) / (raw_denom + 1e-12)
        return {
            "quote_exposure_imbalance": float(np.clip(imbalance, -1.0, 1.0)),
            "quote_exposure_imbalance_raw_ratio": float(np.clip(raw_ratio, -1.0, 1.0)),
            "quote_exposure_bid_opportunity": float(bid_opp),
            "quote_exposure_ask_opportunity": float(ask_opp),
            "quote_exposure_total_opportunity": float(float(bid_opp) + float(ask_opp)),
        }

    def get_effective_fill_imbalance_feature(self) -> float:
        """Return the fill feature consumed by the DQN."""
        if bool(getattr(self, "use_quote_exposure_imbalance_signal", False)):
            feats = self.get_quote_exposure_imbalance_features()
            return float(feats.get("quote_exposure_imbalance", 0.0))
        return float(self.fill_imbalance_ewma)

    def update_fill_imbalance_ewma(
        self,
        bid_fill_qty: float = 0.0,
        ask_fill_qty: float = 0.0,
    ) -> float:
        """
        Update the fill-side EWMAs on a common event clock.

        Callers should invoke this once per environment step. When there is no
        fill, both per-side EWMAs decay smoothly back to 0. The exposed
        fill_imbalance_ewma attribute stores the normalized bias derived from
        those two EWMAs.
        """
        alpha = float(self.fill_imbalance_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        bid_signal = self._fill_activity_signal(fill_qty=bid_fill_qty)
        ask_signal = self._fill_activity_signal(fill_qty=ask_fill_qty)
        self.fill_bid_ewma = (
            alpha * bid_signal + (1.0 - alpha) * float(self.fill_bid_ewma)
        )
        self.fill_ask_ewma = (
            alpha * ask_signal + (1.0 - alpha) * float(self.fill_ask_ewma)
        )
        self.fill_imbalance_ewma = self.get_fill_imbalance_bias()
        return self.fill_imbalance_ewma

    def update_fill_imbalance_ewma_from_fills(
        self,
        fills: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Convenience wrapper: aggregate fills and update fill-side trackers.
        """
        bid_fill_qty = 0.0
        ask_fill_qty = 0.0
        for fill in fills or []:
            try:
                side = int(fill.get("side", 0))
                qty = float(fill.get("qty", 0.0))
            except Exception:
                continue
            if qty <= 0.0:
                continue
            if side > 0:
                bid_fill_qty += qty
            elif side < 0:
                ask_fill_qty += qty
        old_value = self.update_fill_imbalance_ewma(
            bid_fill_qty=bid_fill_qty,
            ask_fill_qty=ask_fill_qty,
        )
        if bool(getattr(self, "use_quote_exposure_imbalance_signal", False)):
            return self.get_effective_fill_imbalance_feature()
        return old_value

    # ======================================================================
    # Hygiene — prune orphan orders (simulation only)
    # ======================================================================
    # def _prune_orphan_orders(self):
    #     """
    #     Hygiene pass to keep (self.orders, self.owner) consistent with engine.
    
    #     Key idea:
    #     - If an order claims (side, price) but the ENGINE has no volume of that side
    #       at that price anymore, that order is a ghost -> delete it.
    #     - Keep the old (price, priority) sanity checks, but add engine-based checks.
    #     """
    #     if self.backtest_mode:
    #         return
    
    #     to_kill = []
    #     n_prices = int(self.owner.shape[1])
    #     n_ranks = int(self.owner.shape[0])
    
    #     lob_state = getattr(self.lob, "lob_state", None)
    
    #     for oid, info in list(self.orders.items()):
    #         p = int(info.get("price", -1))
    #         pr = info.get("priority", None)
    #         side = int(info.get("side", 0))
    
    #         # basic sanity
    #         if side not in (+1, -1):
    #             to_kill.append(oid)
    #             continue
    
    #         if not (0 <= p < n_prices):
    #             to_kill.append(oid)
    #             continue
    
    #         # -----------------------------
    #         # NEW: engine-based ghost check
    #         # -----------------------------
    #         if lob_state is not None and 0 <= p < int(lob_state.shape[0]):
    #             v = float(lob_state[p])
    #             # In this simulator: bids are +, asks are -
    #             if side == +1 and v <= 0.0:
    #                 to_kill.append(oid)
    #                 continue
    #             if side == -1 and v >= 0.0:
    #                 to_kill.append(oid)
    #                 continue
    
    #         # old logic (owner/priority consistency)
    #         col = self.owner[:, p]
    
    #         if pr is None:
    #             if not np.any(col):
    #                 to_kill.append(oid)
    #             continue
    
    #         pr = int(pr)
    #         if not (0 <= pr < n_ranks):
    #             to_kill.append(oid)
    #             continue
    
    #         if not bool(self.owner[pr, p]):
    #             to_kill.append(oid)
    
    #     # apply deletions
    #     for oid in to_kill:
    #         info = self.orders.get(oid)
    #         if info is None:
    #             continue
    #         p = int(info.get("price", -1))
    #         pr = info.get("priority", None)
    
    #         # clear owner mark if possible
    #         if pr is not None:
    #             pr = int(pr)
    #             if 0 <= p < n_prices and 0 <= pr < n_ranks:
    #                 self.owner[pr, p] = False
    
    #         del self.orders[oid]
    
    #     # final cleanup: if we have no orders left at some price, clear that column
    #     # (optional; keeps owner mask tidy)
    #     # NOTE: this is conservative — you can skip if you prefer.
    #     if self.orders:
    #         active_prices = {int(i["price"]) for i in self.orders.values()}
    #         for p in range(n_prices):
    #             if p not in active_prices:
    #                 self.owner[:, p] = False
    #     else:
    #         self.owner[:, :] = False
        
    def _handle_forced_cancel(self, prices=None, reason: str = ""):
        """
        Handle forced cancellation of MM orders by the LOB engine.

        Called by the QRM engine's on_forced_cancel callback BEFORE the
        engine clears the book (reinit) or specific levels (regen/cross-fix).
        This gives the MM a chance to clean up its bookkeeping IMMEDIATELY
        rather than discovering orphaned orders via _prune_orphan_orders()
        one step later.

        Semantics:
          - This is a CANCEL, not a fill: cash and inventory are unchanged.
          - The MM loses its queue position at the affected prices.
          - Orders are removed from self.orders and self.owner.

        Parameters
        ----------
        prices : list of int, or None
            If None, ALL prices are affected (full book reinit).
            If a list, only those specific price indices are cleared.
        reason : str
            Human-readable reason for logging ("qrm_reinit", "qrm_level_clear").
        """
        if not hasattr(self, "orders") or not self.orders:
            return

        # Track which oids we cancel for logging
        cancelled_oids = []

        if prices is None:
            # Full book clear — cancel ALL MM orders
            for oid in list(self.orders.keys()):
                cancelled_oids.append(int(oid))
            self.orders.clear()
            if hasattr(self, "owner") and self.owner is not None:
                self.owner[:, :] = False
        else:
            # Specific price levels — cancel only orders at those prices
            prices_set = set(int(p) for p in prices)
            to_remove = []
            for oid, info in self.orders.items():
                if int(info.get("price", -1)) in prices_set:
                    to_remove.append(int(oid))
            for oid in to_remove:
                info = self.orders.pop(oid, None)
                if info is not None:
                    cancelled_oids.append(oid)
                    px = int(info.get("price", -1))
                    pr = info.get("priority", None)
                    if (hasattr(self, "owner") and self.owner is not None
                            and pr is not None and 0 <= pr < self.owner.shape[0]
                            and 0 <= px < self.owner.shape[1]):
                        self.owner[pr, px] = False

        # Store for downstream logging (reset at start of each step)
        if not hasattr(self, "_last_forced_cancels"):
            self._last_forced_cancels = []
        self._last_forced_cancels.extend(cancelled_oids)

    def _prune_orphan_orders(self):
        """
        Hygiene pass to keep (self.orders, self.owner) consistent.

        Why this function exists
        ------------------------
        The simulator maintains two parallel representations of "our" resting limit orders:

          1) self.orders : dict[oid -> info]
             - stores (side, price, priority, qty, ...)

          2) self.owner : boolean mask [rank, price]
             - indicates which queue-ranks at each price are ours.

        Over time, these two can drift due to:
          - fills / partial fills,
          - environment market orders (queue shifts),
          - environment cancellations (queue shifts),
          - global re-centering shifts (price grid shift),
          - bugs/edge-cases in bookkeeping.

        IMPORTANT (critical for your project)
        ------------------------------------
        Many builds of this simulator expose an "external" lob_state used for the agent state
        (often excluding the MM's own orders when exclude_self_from_state=True).

        In that case, using lob_state as a "ground truth" for whether our orders exist at a
        price is WRONG and will delete valid orders whenever external liquidity at that price
        becomes 0 (even if we are the only remaining liquidity there).

        Therefore:
          - We ONLY use an engine-based check if the engine explicitly exposes a FULL book
            that includes our MM orders (e.g., lob_state_full / lob_state_with_mm).
          - Otherwise, we prune purely based on internal consistency and bounds.

        Behavior (conservative)
        -----------------------
        - Drop orders with impossible metadata (bad side/price/priority).
        - Drop duplicate orders that claim the same (price, priority) cell.
        - Rebuild self.owner from self.orders as the final authoritative state.

        This is intentionally conservative: it will NOT try to infer "which order was filled"
        by comparing aggregated volumes when other agents share the same price level.
        """
        if self.backtest_mode:
            return

        # If we have no local tracking at all, ensure owner is clean and exit.
        if not getattr(self, "orders", None):
            if getattr(self, "owner", None) is not None:
                self.owner[:, :] = False
            return

        owner = getattr(self, "owner", None)
        if owner is None or getattr(owner, "ndim", 0) != 2:
            # If owner is missing/corrupt, safest is to drop local tracking.
            self.orders.clear()
            return

        n_ranks, n_prices = map(int, owner.shape)

        # Optional: use a FULL book (including MM) if the engine provides it.
        # DO NOT use `lob_state` if it excludes self orders.
        lob_state_full = None
        if getattr(self, "lob", None) is not None:
            # Use next() to safely pick the first non-None attribute,
            # avoiding Python `or` on numpy arrays (ambiguous truth value).
            for _attr in ("lob_state_with_mm", "lob_state_full", "lob_state_including_mm"):
                _val = getattr(self.lob, _attr, None)
                if _val is not None:
                    lob_state_full = _val
                    break

        to_kill = set()
        occupied = set()  # tracks claimed (priority, price) cells to prevent duplicates

        for oid, info in list(self.orders.items()):
            # -----------------------
            # Validate basic metadata
            # -----------------------
            try:
                side = int(info.get("side", 0))
            except Exception:
                side = 0
            if side not in (+1, -1):
                to_kill.add(int(oid))
                continue

            try:
                p = int(info.get("price", -1))
            except Exception:
                p = -1
            if not (0 <= p < n_prices):
                to_kill.add(int(oid))
                continue

            pr = info.get("priority", None)
            if pr is None:
                # Without a priority we cannot represent the order in self.owner reliably.
                to_kill.add(int(oid))
                continue

            try:
                pr = int(pr)
            except Exception:
                to_kill.add(int(oid))
                continue
            if not (0 <= pr < n_ranks):
                to_kill.add(int(oid))
                continue

            # -------------------------------
            # Prevent duplicate cell claims
            # -------------------------------
            cell = (pr, p)
            if cell in occupied:
                # Two orders claiming the same queue cell -> at least one is ghost.
                # Drop the later one we encounter (arbitrary but stable).
                to_kill.add(int(oid))
                continue
            occupied.add(cell)

            # -------------------------------------------------------------
            # OPTIONAL: engine-based "impossible side at price" (FULL book)
            # -------------------------------------------------------------
            if lob_state_full is not None and 0 <= p < int(getattr(lob_state_full, "shape", [0])[0]):
                v = float(lob_state_full[p])
                # Convention in this simulator: bids are +, asks are -
                if side == +1 and v <= 1e-8:
                    to_kill.add(int(oid))
                    continue
                if side == -1 and v >= -1e-8:
                    to_kill.add(int(oid))
                    continue

        # Apply deletions from dict
        for oid in to_kill:
            if oid in self.orders:
                del self.orders[oid]

        # -------------------------------
        # Rebuild owner mask from orders
        # -------------------------------
        owner[:, :] = False
        for oid, info in self.orders.items():
            try:
                p = int(info.get("price", -1))
                pr = int(info.get("priority", -1))
            except Exception:
                continue
            if 0 <= p < n_prices and 0 <= pr < n_ranks:
                owner[pr, p] = True

    def _owner_mask_like(self, ref: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask with the SAME shape as `ref`.

        In simulation mode, `self.owner` is expected to have the same shape as
        `lob.priorities_lob_state` (n_priority_ranks x number_tick_levels).

        However, in practice it is easy for shapes to drift when:
          - older checkpoints/configs created `self.owner` with a different shape,
          - a previous code version stored `owner` as a 1D per-price mask,
          - max_offset / window sizes were changed without reinitializing MM.

        Instead of crashing with NumPy's boolean-index mismatch, we pad/truncate
        to a safe all-False mask in the extra area.
        """
        owner = getattr(self, "owner", None)

        if owner is None:
            owner_fixed = np.zeros(ref.shape, dtype=bool)
            self.owner = owner_fixed
            return owner_fixed

        owner = np.asarray(owner, dtype=bool)

        # Fast path: already compatible
        if owner.shape == ref.shape:
            return owner

        # Slow path (rare): coerce shape by pad/truncate.
        owner_fixed = np.zeros(ref.shape, dtype=bool)

        if owner.ndim == 2 and ref.ndim == 2:
            r = min(owner.shape[0], ref.shape[0])
            c = min(owner.shape[1], ref.shape[1])
            if r > 0 and c > 0:
                owner_fixed[:r, :c] = owner[:r, :c]

        elif owner.ndim == 1 and ref.ndim == 1:
            m = min(owner.size, ref.size)
            if m > 0:
                owner_fixed[:m] = owner[:m]

        elif owner.ndim == 1 and ref.ndim == 2:
            # Interpret a 1D owner as a per-price mask (apply to ALL ranks).
            m = min(owner.size, ref.shape[1])
            if m > 0:
                owner_fixed[:, :m] = owner[:m][None, :]

        elif owner.ndim == 2 and ref.ndim == 1:
            # Interpret a 2D owner as "any owned rank at each price".
            m = min(owner.shape[1], ref.shape[0])
            if m > 0:
                owner_fixed[:m] = np.any(owner[:, :m], axis=0)

        # Unknown shapes: keep all-False (safe default).
        self.owner = owner_fixed
        return owner_fixed

# ======================================================================
    # Internal helpers for queue / ranks (simulation only)
    # ======================================================================
    def _best_pos_at_price(self, price: int) -> int:
        """
        Return the best (lowest) queue rank at which we own an order
        at the given price, or -1 if we do not own any.
        """
        if price < 0 or price >= self.owner.shape[1]:
            return -1
        idx = np.where(self.owner[:, price])[0]
        return int(idx[0]) if idx.size else -1

    def _my_best_bid_price(self) -> int:
        """Highest price at which MM has an open bid, or -1."""
        prices = [int(v["price"]) for v in self.orders.values() if v.get("side") == +1]
        return max(prices) if prices else -1

    def _my_best_ask_price(self) -> int:
        """Lowest price at which MM has an open ask, or -1."""
        prices = [int(v["price"]) for v in self.orders.values() if v.get("side") == -1]
        return min(prices) if prices else -1

    def _current_bid_pos(self) -> int:
        return self._best_pos_at_price(self._my_best_bid_price())

    def _current_ask_pos(self) -> int:
        return self._best_pos_at_price(self._my_best_ask_price())

    # ======================================================================
    # Best bid / ask helpers (simulation and backtest)
    # ======================================================================
    def _safe_best_bid_idx(self) -> int:
        if self.lob is None:
            return -1
        idxs = np.where(self.lob.lob_state > 0)[0]
        return int(idxs[-1]) if idxs.size else -1

    def _safe_best_ask_idx(self) -> int:
        if self.lob is None:
            return -1
        idxs = np.where(self.lob.lob_state < 0)[0]
        return int(idxs[0]) if idxs.size else -1

    def best_bid(self) -> int:
        if self.backtest_mode and self._last_env_snapshot is not None:
            return int(self._last_env_snapshot["best_bid"])
        return self._safe_best_bid_idx()

    def best_ask(self) -> int:
        if self.backtest_mode and self._last_env_snapshot is not None:
            return int(self._last_env_snapshot["best_ask"])
        return self._safe_best_ask_idx()

    def mid(self) -> float:
        """
        Mid-price in index space.
        """
        if self.backtest_mode and self._last_env_snapshot is not None:
            return float(self._last_env_snapshot["mid"])
        if self.lob is None:
            return 0.0
        return self.lob.compute_mid_price()

    def spread(self) -> int:
        if self.backtest_mode and self._last_env_snapshot is not None:
            return int(self._last_env_snapshot["spread"])
        if self.lob is None:
            return 0
        return self.lob.compute_spread()

    def mark_to_market(self) -> float:
        mid_idx = float(self.mid())
        mid_dollar = self._px_to_dollars(mid_idx)
        return float(self.inventory) * mid_dollar

    def total_pnl(self) -> float:
        return float(self.cash + self.mark_to_market())

    # ======================================================================
    # Order ID generator
    # ======================================================================
    def _alloc_order_id(self) -> int:
        oid = self.next_oid
        self.next_oid += 1
        return oid

    # ======================================================================
    # Internal helper — register own order (simulation + backtest)
    # ======================================================================
    def _register_own_order(
        self,
        oid: int,
        side: int,
        price: int,
        qty: float,
        priority: int,
        queue_ahead: float = 0.0,
        executed_ahead: float = 0.0,
    ):
        """
        Common registration of a new MM order in both modes.
        """
        self.orders[oid] = {
            "side": int(side),
            "price": int(price),
            "qty": float(qty),
            "priority": int(priority),
            "queue_ahead": float(queue_ahead),
            "executed_ahead": float(executed_ahead),
        }

        if self.backtest_mode or (self.lob is None):
            return

        # IMPORTANT:
        # The LOB engine can dynamically expand `priorities_lob_state` (more priority ranks)
        # when a queue grows. Our `self.owner` mask must be expanded *before* marking ownership,
        # otherwise orders placed at a newly-created priority rank would not be protected from
        # environment cancellations.
        if hasattr(self.lob, "priorities_lob_state"):
            self._owner_mask_like(self.lob.priorities_lob_state)

        try:
            pr_i = int(priority)
            p_i = int(price)
        except (TypeError, ValueError):
            return

        if 0 <= pr_i < self.owner.shape[0] and 0 <= p_i < self.owner.shape[1]:
            self.owner[pr_i, p_i] = True

    # ======================================================================
    # Environment-only view (simulation only)
    # ======================================================================
    def _env_view(self) -> Dict[str, Any]:
        """
        Returns the LOB "as if" our own orders did not exist (simulation only).
        """
        if self.lob is None or self.backtest_mode:
            raise RuntimeError("env_view should only be used in simulation mode.")

        env_lob_state = self.lob.lob_state.copy()
        env_prior = self.lob.priorities_lob_state.copy()

        # Robustness: ensure boolean mask matches the priorities matrix shape
        owner_mask = self._owner_mask_like(env_prior)
        env_prior[owner_mask] = 0.0

        for info in self.orders.values():
            p = info["price"]
            s = info["side"]
            q = float(info.get("qty", 1.0))
            if 0 <= p < env_lob_state.shape[0]:
                env_lob_state[p] -= s * q

        bid_idxs = np.where(env_lob_state > 0)[0]
        ask_idxs = np.where(env_lob_state < 0)[0]
        best_bid = int(bid_idxs[-1]) if bid_idxs.size else -1
        best_ask = int(ask_idxs[0]) if ask_idxs.size else -1

        spread = int(best_ask - best_bid) if (best_bid >= 0 and best_ask >= 0) else 0
        # Use actual volume from lob_state, not rank-count from priorities matrix.
        # count_nonzero(env_prior[:, px]) counts occupied queue slots (always ≤ actual
        # volume when orders > 1 per slot), giving wrong OBI.
        bid_size = int(env_lob_state[best_bid]) if best_bid >= 0 else 0
        ask_size = int(-env_lob_state[best_ask]) if best_ask >= 0 else 0

        denom = max(1, bid_size + ask_size)
        obi = float(bid_size - ask_size) / float(denom)

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "bidsize": bid_size,
            "asksize": ask_size,
            "obi": obi,
        }

    # ======================================================================
    # Backtest-only: queue-ahead estimation
    # ======================================================================
    def _estimate_queue_ahead_backtest(self, side: int, price: int) -> float:
        if not self.backtest_mode:
            return 0.0
        if self._last_env_snapshot is None:
            return 0.0

        if side == +1:
            depth = self._last_env_snapshot.get("full_depth_bids", [])
        else:
            depth = self._last_env_snapshot.get("full_depth_asks", [])

        queue_ahead = 0.0
        for px, sz in depth:
            if int(px) == int(price):
                queue_ahead += float(sz)

        return queue_ahead

    # ======================================================================
    # Helper: compute inside-spread prices (simulation + backtest)
    # ======================================================================
    def _compute_inside_spread_prices(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute candidate inside-spread prices for BID and ASK separately.
        FIXED for Backtest Mode: handles negative prices correctly.
        """
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        # --- FIX START ---
        # Determine valid liquidity not just by value < 0, but by mode.
        has_bid = False
        has_ask = False

        if self.backtest_mode and self._last_env_snapshot is not None:
            # In backtest, utilize explicit flags if available, or rely on snapshot existence
            # Assuming snapshot always has 'has_bids'/'has_asks' from your backtest runner
            has_bid = self._last_env_snapshot.get("has_bids", True)
            has_ask = self._last_env_snapshot.get("has_asks", True)
        else:
            # In simulation, indices must be non-negative
            has_bid = (best_bid >= 0)
            has_ask = (best_ask >= 0)

        if (not has_bid) or (not has_ask):
            return None, None
        # --- FIX END ---

        spread = best_ask - best_bid
        
        # No space strictly inside the spread (need at least 2 ticks gap)
        if spread < 2:
            return None, None

        bid_inside = None
        ask_inside = None

        # Candidate inside BID: one tick above best_bid, still below best_ask
        candidate_bid = best_bid + 1
        if candidate_bid < best_ask:
            bid_inside = int(candidate_bid)

        # Candidate inside ASK: one tick below best_ask, still above best_bid
        candidate_ask = best_ask - 1
        if candidate_ask > best_bid:
            ask_inside = int(candidate_ask)

        return bid_inside, ask_inside

    # ======================================================================
    # Main actions — place_limit
    # ======================================================================
    def place_limit(
        self,
        side: int,
        price_idx: int,
        qty: float = 1.0,
        priority: int = 0,
    ) -> Tuple[int, int]:
        """
        Place a limit order at a given price index.

        Returns (order_id, effective_priority).
        """
        try:
            side = int(side)
        except (TypeError, ValueError):
            return -1, -1

        if side not in (+1, -1):
            return -1, -1

        # BACKTEST BRANCH
        if self.backtest_mode:
            snap = self._last_env_snapshot
            if snap is None:
                return -1, -1

            try:
                px = int(price_idx)
            except (TypeError, ValueError):
                return -1, -1

            # FIX 3.3: Reject crossed orders in backtest mode (same guard
            # as simulation).  A bid at/above the best ask or an ask
            # at/below the best bid would violate the passive-only
            # assumption and generate spurious PnL.
            if snap.get("has_asks", False) and side == +1:
                ba = int(snap.get("best_ask", 0))
                if px >= ba:
                    return -1, -1
            if snap.get("has_bids", False) and side == -1:
                bb = int(snap.get("best_bid", 0))
                if px <= bb:
                    return -1, -1

            oid = self._alloc_order_id()
            eff_prio = int(priority)

            qa = 0.0
            if self.use_queue_ahead:
                # Exact FIFO: insert sentinel and get precise volume ahead
                fifo = getattr(self, "_fifo_tracker", None)
                if fifo is not None:
                    qa = fifo.add_mm_order(px, oid)
                else:
                    qa = self._estimate_queue_ahead_backtest(side, px)

            self._register_own_order(
                oid=oid,
                side=side,
                price=px,
                qty=qty,
                priority=eff_prio,
                queue_ahead=qa,
                executed_ahead=0.0,
            )

            return oid, eff_prio

        # SIMULATION BRANCH
        try:
            px = int(price_idx)
        except (TypeError, ValueError):
            return -1, -1

        if px < 0 or px >= self.lob.lob_state.shape[0]:
            return -1, -1

        # Enforce passive-only limits in simulation mode:
        # - buy limits must be strictly below current best ask
        # - sell limits must be strictly above current best bid
        bb = self._safe_best_bid_idx()
        ba = self._safe_best_ask_idx()
        if side == +1 and ba >= 0 and px >= ba:
            return -1, -1
        if side == -1 and bb >= 0 and px <= bb:
            return -1, -1

        # Each priority rank corresponds to 1 order = mean_size_LO shares.
        # Clamp qty to match the engine's per-order granularity.
        qty = float(self.lob.mean_size_LO)

        signed_unit = +1 if side == +1 else -1
        self.lob.lob_state[px] += signed_unit * float(self.lob.mean_size_LO)
        self.lob._touch_lob_state()  # invalidate BBA cache after MM mutation

        pr = self.lob.add_order_to_queue(px, signed_unit * 1)
        if pr is None:
            self.lob.lob_state[px] -= signed_unit * float(self.lob.mean_size_LO)
            self.lob._touch_lob_state()  # invalidate after revert
            return -1, -1

        oid = self._alloc_order_id()
        self._register_own_order(oid=oid, side=side, price=px, qty=qty, priority=pr)

        return oid, int(pr)

    # ======================================================================
    # Inside-spread helpers (macro-actions)
    # ======================================================================
    def place_bid_inside_spread(
        self,
        qty: float = 1.0,
        priority: int = 0,
    ) -> Tuple[int, int]:
        """
        Place a BID limit order strictly inside the spread, if possible.

        If inside-spread quoting is not possible (e.g. spread < 2 or missing
        one side), this method returns (-1, -1) and does nothing.
        """
        bid_px, _ = self._compute_inside_spread_prices()
        if bid_px is None:
            return -1, -1
        return self.place_limit(+1, bid_px, qty=qty, priority=priority)

    def place_ask_inside_spread(
        self,
        qty: float = 1.0,
        priority: int = 0,
    ) -> Tuple[int, int]:
        """
        Place an ASK limit order strictly inside the spread, if possible.

        If inside-spread quoting is not possible, returns (-1, -1) and
        does nothing.
        """
        _, ask_px = self._compute_inside_spread_prices()
        if ask_px is None:
            return -1, -1
        return self.place_limit(-1, ask_px, qty=qty, priority=priority)

    def place_bid_ask_inside_spread(
        self,
        qty: float = 1.0,
        priority: int = 0,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Place BOTH BID and ASK strictly inside the spread, when possible.
    
        Constraints
        ----------
        - Requires valid inside-spread prices for BOTH sides.
        - Enforces:
              bid_px < ask_px
          so that the captured spread is at least 1 tick.
    
        Returns
        -------
        ((bid_oid, bid_prio), (ask_oid, ask_prio))
    
        If constraints are not satisfied, no orders are placed and
        ((-1, -1), (-1, -1)) is returned.
        """
    
        bid_px, ask_px = self._compute_inside_spread_prices()
    
        # Need both sides for a true two-sided inside-spread quote
        if bid_px is None or ask_px is None:
            return (-1, -1), (-1, -1)
    
        # Enforce "bid < ask" -> captured spread >= 1 tick
        if not (bid_px < ask_px):
            return (-1, -1), (-1, -1)
    
        # If your invariants require at most one LO per side, you may want
        # to cancel existing ones here:
        # self.cancel_all_bid_orders()
        # self.cancel_all_ask_orders()
    
        bid_res = self.place_limit(+1, bid_px, qty=qty, priority=priority)
        ask_res = self.place_limit(-1, ask_px, qty=qty, priority=priority)
    
        return bid_res, ask_res



    # ======================================================================
    # Cancel operations
    # ======================================================================
    def cancel(self, oid: int, _prune: bool = True) -> bool:
        """
        Cancel one of OUR resting limit orders.

        Critical detail (SIMULATION mode):
        ---------------------------------
        The engine uses a FIFO queue per (side, price) and represents it via a
        priority-rank matrix (`priorities_lob_state`).

        When an order is removed at rank r (whether by a cancel or by a trade),
        all orders BEHIND it shift forward by 1 rank (r+1 -> r, etc.) so that
        the queue stays packed.

        Therefore, when WE cancel an order, we must mirror the same rank shift
        in our local `owner` mask and in every stored order's `priority` at that
        price level. If we don't, our local queue positions drift and future
        fill-detection becomes wrong (serious bug).
        """
        if oid not in self.orders:
            return False

        info = self.orders[oid]
        px = int(info.get("price", -1))
        side = int(info.get("side", 0))
        prio = info.get("priority", None)
        prio_rank = int(prio) if prio is not None else -1

        # Remove from our dict early (so we don't accidentally re-process it).
        del self.orders[oid]

        if self.backtest_mode or self.lob is None:
            # In backtest mode we do not touch the engine; we only update our local
            # tracking.  Remove sentinel from FIFO tracker if present.
            fifo = getattr(self, "_fifo_tracker", None)
            if fifo is not None:
                fifo.remove_mm_order(oid)
            return True

        # Basic bounds checks
        if px < 0 or px >= self.owner.shape[1]:
            if _prune:
                self._prune_orphan_orders()
            return True

        # --------------------------
        # Update engine book state
        # --------------------------
        # Our order changes the aggregated LOB state by ±mean_size_LO shares.
        signed_unit = +1.0 if side == +1 else -1.0
        self.lob.lob_state[px] -= signed_unit * float(self.lob.mean_size_LO)
        self.lob._touch_lob_state()  # invalidate BBA cache after MM mutation

        # Remove from the engine's FIFO queue at the recorded rank.
        if hasattr(self.lob, "remove_order_from_queue") and prio_rank >= 0:
            self.lob.remove_order_from_queue(px, prio_rank)

        # --------------------------
        # Mirror FIFO shift locally
        # --------------------------
        # 1) Owner mask: clear the canceled slot, then shift everything behind it.
        if 0 <= prio_rank < self.owner.shape[0]:
            self.owner[prio_rank, px] = False
            # Shift ranks (r+1 -> r) for this price column
            self.owner[prio_rank:-1, px] = self.owner[prio_rank + 1 :, px]
            self.owner[-1, px] = False

        # 2) Stored priorities: any of our remaining orders at the SAME price and
        #    with priority > prio_rank must decrement by 1.
        if prio_rank >= 0:
            for oid2, info2 in self.orders.items():
                if int(info2.get("price", -999)) != px:
                    continue
                pr2 = info2.get("priority", None)
                if pr2 is None:
                    continue
                pr2i = int(pr2)
                if pr2i > prio_rank:
                    info2["priority"] = pr2i - 1

        # Hygiene
        if _prune:
            self._prune_orphan_orders()
        return True

    def cancel_all(self):
        """
        Cancel all of our own working orders.
        Works in both simulation and backtest mode.
        """
        for oid in list(self.orders.keys()):
            self.cancel(oid, _prune=False)

        # Single prune at the end instead of one per order.
        self._prune_orphan_orders()

    def cancel_bid_all(self):
        """Cancel all bid-side orders with a single prune at the end."""
        for oid in [o for o, v in list(self.orders.items()) if v.get("side") == +1]:
            self.cancel(oid, _prune=False)
        self._prune_orphan_orders()

    def cancel_ask_all(self):
        """Cancel all ask-side orders with a single prune at the end."""
        for oid in [o for o, v in list(self.orders.items()) if v.get("side") == -1]:
            self.cancel(oid, _prune=False)
        self._prune_orphan_orders()

    # ------------------------------------------------------------------
    # Side-specific cancel helpers
    # ------------------------------------------------------------------
    def _find_order_to_cancel_on_side(self, side: int) -> Optional[int]:
        """
        Find one order_id to cancel on a given side.

        FIX: Previously used the *market* best bid/ask as target, which could
        miss the MM's own best-priced order when the MM is not at top-of-book.
        Now scans all MM orders on the requested side and returns the one at the
        best price (highest for bids, lowest for asks).
        """
        best_oid: Optional[int] = None
        best_px: Optional[int] = None

        for oid, info in self.orders.items():
            if int(info.get("side", 0)) != side:
                continue
            px = int(info.get("price", -1))
            if best_oid is None:
                best_oid, best_px = oid, px
            elif side == +1 and px > best_px:       # bids: higher is better
                best_oid, best_px = oid, px
            elif side == -1 and px < best_px:       # asks: lower is better
                best_oid, best_px = oid, px

        return best_oid

    def cancel_bid_side(self) -> bool:
        oid = self._find_order_to_cancel_on_side(+1)
        if oid is None:
            return False
        return self.cancel(oid)

    def cancel_ask_side(self) -> bool:
        oid = self._find_order_to_cancel_on_side(-1)
        if oid is None:
            return False
        return self.cancel(oid)

   # ======================================================================
    # Market orders (cross_mo)
    # ======================================================================
    def cross_mo(self, side: int, qty: float = 1.0) -> Tuple[int, float]:
        """
        Send an aggressive MARKET order (buy if side=+1, sell if side=-1).

        SIMULATION mode details
        -----------------------
        The engine may allow 'sweeps' (qty > 1) where a market order consumes
        multiple queue units and may walk across multiple price levels.

        A common bug is to:
          - record only the initial best price, and
          - assume the entire qty executes there.

        Here we do the *minimal* correct thing without refactoring the engine:
          - execute unit-by-unit,
          - re-read the current best price each unit (so we naturally sweep),
          - accumulate the cash/inventory changes using the actual executed prices.

        Returns
        -------
        (last_px, executed_qty) where:
          - last_px is the last executed price index (or -1 if nothing executed),
          - executed_qty is the total quantity actually executed.
        """
        if self.lob is None and not self.backtest_mode:
            return -1, 0.0

        side = int(np.sign(side)) if side != 0 else 0
        if side == 0:
            return -1, 0.0

        qty_i = int(max(0, round(float(qty))))
        if qty_i <= 0:
            return -1, 0.0

        # --------------------------
        # BACKTEST mode (no depth model)
        # --------------------------
        if self.backtest_mode:
            # [FIX] In backtest mode, the grid is centered, so negative price indices are VALID.
            # We must check liquidity availability via flags, not by checking if price < 0.
            snap = self._last_env_snapshot
            if snap is None:
                return -1, 0.0

            # Determine if there is liquidity on the opposing side
            # If we Buy (+1), we hit Asks. If we Sell (-1), we hit Bids.
            has_liquidity = snap.get("has_asks", False) if side == +1 else snap.get("has_bids", False)
            
            if not has_liquidity:
                return -1, 0.0

            # FIX: sweep through depth levels instead of pricing all units
            # at best price.  For qty > 1 the old code understated execution
            # costs because every unit was priced at the single best level.
            # Now we walk the full_depth ladder, consuming available size at
            # each level before moving to the next (matching simulation mode).
            if side == +1:
                depth = list(snap.get("full_depth_asks", []))   # [(px, sz), ...]
                depth.sort(key=lambda x: x[0])                  # ascending price (best first)
            else:
                depth = list(snap.get("full_depth_bids", []))   # [(px, sz), ...]
                depth.sort(key=lambda x: -x[0])                 # descending price (best first)

            remaining = float(qty_i)
            executed_qty = 0.0
            last_px = -1

            for lvl_px, lvl_sz in depth:
                if remaining <= 0:
                    break
                fill_here = min(remaining, float(lvl_sz))
                price_dollar = self._px_to_dollars(int(lvl_px)) * fill_here
                if side == +1:
                    self.inventory += fill_here
                    self.cash -= price_dollar
                else:
                    self.inventory -= fill_here
                    self.cash += price_dollar
                remaining -= fill_here
                executed_qty += fill_here
                last_px = int(lvl_px)

            if executed_qty <= 0:
                return -1, 0.0
            return last_px, executed_qty

        # --------------------------
        # SIMULATION mode (sweep-safe)
        # --------------------------
        # The engine consumes mean_size_LO shares per order.
        # qty is in shares (caller convention), so convert to orders.
        lot_size = float(getattr(self.lob, 'mean_size_LO', 1))
        n_orders = int(max(0, round(float(qty) / lot_size)))
        if n_orders <= 0:
            return -1, 0.0

        executed_qty = 0.0
        last_px = -1

        for _ in range(n_orders):
            # Re-read the current best price at each unit to naturally handle sweeps.
            px = self.best_ask() if side == +1 else self.best_bid()

            # In Simulation mode (array indices), negative index implies empty book/invalid state.
            if px < 0:
                break

            # Self-trade guard: only block if our order is at the FRONT of the FIFO
            # queue (rank 0) at this price. If external liquidity is ahead of us,
            # the MO consumes the external order, not ours.
            own_px = self._my_best_ask_price() if side == +1 else self._my_best_bid_price()
            if own_px >= 0 and px == own_px:
                own_rank = self._best_pos_at_price(px)
                if own_rank == 0:
                    break

            # Execute one unit against the engine.
            exec_px = -1
            if hasattr(self.lob, "execute_mm_market_order"):
                out = self.lob.execute_mm_market_order(side, px)
                try:
                    exec_px = int(out)
                except Exception:
                    exec_px = -1
            elif hasattr(self.lob, "execute_market_order"):
                # Fallback: use generic method name if present
                pre_v = None
                if 0 <= int(px) < int(self.lob.lob_state.shape[0]):
                    pre_v = float(self.lob.lob_state[int(px)])
                out = self.lob.execute_market_order(side, px)
                try:
                    exec_px = int(out)
                except Exception:
                    exec_px = -1
                if exec_px < 0 and pre_v is not None:
                    post_v = float(self.lob.lob_state[int(px)])
                    if post_v != pre_v:
                        exec_px = int(px)
            else:
                break

            # If the engine rejected execution (e.g. thin-book guard), do not account it.
            if exec_px < 0:
                break

            # Bug fix: capture grid recentering shift that happened inside
            # execute_mm_market_order (via center_lob_state).
            shift = getattr(self.lob, '_last_center_shift', 0)

            # Mirror FIFO consumption BEFORE applying shift (owner mask
            # is still in pre-shift coordinates at this point).
            consumed_side = -1 if side == +1 else +1
            self._apply_local_fifo_depletion(
                price=int(exec_px),
                consumed_units=1,
                consumed_side=consumed_side,
            )

            # Accounting at the executed price (use pre-shift base_price_idx
            # so that _px_to_dollars returns the correct dollar price).
            price_dollar = self._px_to_dollars(exec_px) * lot_size
            if side == +1:
                self.inventory += lot_size
                self.cash -= price_dollar
            else:
                self.inventory -= lot_size
                self.cash += price_dollar

            # Propagate grid recentering shift to MM tracking.
            if shift != 0:
                self.base_price_idx += shift
                self.apply_shift(shift)

            executed_qty += lot_size
            last_px = int(exec_px) - shift

        # Hygiene (aggressive MOs can create edge cases if we self-trade etc.)
        self._prune_orphan_orders()
        return int(last_px), float(executed_qty)
    
    # =========================================================================
    # STATE CONTRACT & OBSERVATION BUILDER
    # -------------------------------------------------------------------------
    def build_state(self, external_env: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
        """
        Builds the observable state dictionary passed to the MM policy/controller.

        This method guarantees a STABLE schema and data types, ensuring compatibility
        with both Rule-Based policies (GLFT, Avellaneda) and RL Agents (DQN, PPO).

        ======================================================================
        STATE CONTRACT DEFINITION
        ----------------------------------------------------------------------
        1. Price Conventions:
           - SIMULATION: Uses integer price INDICES [0 .. n_levels-1].
           - BACKTEST: May use centered grid (negative indices allowed).
           - Missing quotes are ALWAYS encoded as -1 (Sentinel), never None.

        2. Time Tracking [NEW]:
           - The state now includes a "time" key (float, in seconds).
           - Critical for policies using Time-Based Throttling.

        3. The Three Views of "Top of Book" (TOB):
           A) EFFECTIVE TOB (best_bid, best_ask):
              - The price the agent receives as "market price" for decision making.
              - Behavior depends on `use_sticky_price` flag (when exclude_self=True):
                * If use_sticky_price=True (Default): It stays at our price if we are the best.
                  Prevents self-cancellation loops in naive policies (e.g. Always Best).
                * If use_sticky_price=False (Transparent): It shows the best EXTERNAL price.
                  Useful for policies that want to see the "real" market depth behind them.

           B) EXTERNAL/ENV TOB (best_bid_env, best_ask_env):
              - The market strictly WITHOUT our orders.
              - Used for calculating "Market Moves" (throttling triggers) and reference prices.

           C) RAW TOB (best_bid_raw, best_ask_raw):
              - The actual full book (Environment + Our Orders).
              - Used for debugging, snapshot consistency, and order matching.
        ======================================================================
        """

        # ------------------------------------------------------------------
        # Helper Functions: Robust Type Casting & Sentinel Handling
        # ------------------------------------------------------------------
        def _safe_int(x, default: int = -1) -> int:
            if x is None: return default
            try: return int(x)
            except Exception: return default

        def _safe_float(x, default: float = 0.0) -> float:
            if x is None: return default
            try: return float(x)
            except Exception: return default

        # ------------------------------------------------------------------
        # 0) Internal State: Summarize our own live orders
        # ------------------------------------------------------------------
        # We aggregate our own volume per price level to overlay it on the
        # external book (Backtest) or separate it (Simulation).
        mm_orders_prices: List[int] = []
        mm_orders_sides: List[int] = []

        # [FIX] Use large sentinels to allow valid comparisons with negative prices
        MIN_INT = -2_000_000_000
        MAX_INT = 2_000_000_000

        my_best_bid = MIN_INT
        my_best_ask = MAX_INT
        
        has_my_bid = False
        has_my_ask = False

        own_bid_qty: Dict[int, float] = {}
        own_ask_qty: Dict[int, float] = {}

        for info in self.orders.values():
            p = info.get("price", None)
            s = info.get("side", None)
            if p is None or s is None:
                continue
            try:
                p_i = int(p)
                s_i = int(s)
            except Exception:
                continue

            mm_orders_prices.append(p_i)
            mm_orders_sides.append(s_i)

            q = _safe_float(info.get("qty", 1.0), default=1.0)

            if s_i == +1: # Bid
                has_my_bid = True
                if p_i > my_best_bid:
                    my_best_bid = p_i
                own_bid_qty[p_i] = own_bid_qty.get(p_i, 0.0) + q
            elif s_i == -1: # Ask
                has_my_ask = True
                if p_i < my_best_ask:
                    my_best_ask = p_i
                own_ask_qty[p_i] = own_ask_qty.get(p_i, 0.0) + q

        n_own_orders_valid = int(len(mm_orders_prices))
        n_own_orders_total = int(len(self.orders))

        # ------------------------------------------------------------------
        # Initialize Default Variables (Fallbacks)
        # ------------------------------------------------------------------
        best_bid_env = -1; best_ask_env = -1; spread_env = 0
        bid_size_env = 0.0; ask_size_env = 0.0; obi_env = 0.0

        best_bid_raw = -1; best_ask_raw = -1; spread_raw = 0
        bidsize_raw = 0.0; asksize_raw = 0.0; obi_raw = 0.0

        best_bid = -1; best_ask = -1; spread = 0
        bid_size = 0.0; ask_size = 0.0; obi = 0.0

        mid_val = 0.0
        current_time = 0.0  # <--- CRITICAL FOR THROTTLING

        pure_mm_bid_sizes = None
        pure_mm_ask_sizes = None

        # ------------------------------------------------------------------
        # BACKTEST MODE Logic
        # ------------------------------------------------------------------
        if self.backtest_mode:
            # Check for new snapshot
            if external_env is not None:
                self._last_env_snapshot = dict(external_env)
            
            # Process snapshot if available
            if self._last_env_snapshot is not None:
                snap = self._last_env_snapshot
                
                # --- [Time Capture] ---
                current_time = float(snap.get("time", 0.0))

                # --- 1. External View (Snapshot Data) ---
                # [FIX] Rely on explicit flags for validity, NOT on price value >= 0
                has_env_bid = snap.get("has_bids", False)
                has_env_ask = snap.get("has_asks", False)

                best_bid_env = _safe_int(snap.get("best_bid", -1))
                best_ask_env = _safe_int(snap.get("best_ask", -1))
                bid_size_env = _safe_float(snap.get("bidsize", 0.0))
                ask_size_env = _safe_float(snap.get("asksize", 0.0))
                obi_env = _safe_float(snap.get("obi", 0.0))
                mid_val = _safe_float(snap.get("mid", 0.0))

                # [FIX] Use flags to calculate spread
                if has_env_bid and has_env_ask:
                    spread_env = int(best_ask_env - best_bid_env)

                # --- 2. Raw View (Snapshot + Own Orders Overlay) ---
                # [FIX] Bid: max of env and mine (checking flags)
                if has_env_bid and has_my_bid:
                    best_bid_raw = int(max(best_bid_env, my_best_bid))
                elif has_env_bid:
                    best_bid_raw = int(best_bid_env)
                elif has_my_bid:
                    best_bid_raw = int(my_best_bid)
                else:
                    best_bid_raw = -1 # Empty

                # [FIX] Ask: min of env and mine (checking flags)
                if has_env_ask and has_my_ask:
                    best_ask_raw = int(min(best_ask_env, my_best_ask))
                elif has_env_ask:
                    best_ask_raw = int(best_ask_env)
                elif has_my_ask:
                    best_ask_raw = int(my_best_ask)
                else:
                    best_ask_raw = -1 # Empty

                # Valid liquidity check for Raw
                has_raw_bid = (has_env_bid or has_my_bid)
                has_raw_ask = (has_env_ask or has_my_ask)

                if has_raw_bid and has_raw_ask:
                    spread_raw = int(best_ask_raw - best_bid_raw)

                # Reconstruct Volumes at the new Raw TOB
                raw_bid_vol = 0.0
                if has_env_bid and (best_bid_raw == best_bid_env):
                    raw_bid_vol += bid_size_env
                if has_raw_bid:
                    raw_bid_vol += float(own_bid_qty.get(best_bid_raw, 0.0))

                raw_ask_vol = 0.0
                if has_env_ask and (best_ask_raw == best_ask_env):
                    raw_ask_vol += ask_size_env
                if has_raw_ask:
                    raw_ask_vol += float(own_ask_qty.get(best_ask_raw, 0.0))

                bidsize_raw = raw_bid_vol
                asksize_raw = raw_ask_vol
                denom = max(1.0, bidsize_raw + asksize_raw)
                obi_raw = (bidsize_raw - asksize_raw) / denom

                # --- 3. Effective View (For Decision Making) ---
                # NEW LOGIC: Support for Transparent Pricing via use_sticky_price
                
                # Check if we should expose the raw Environment Price even if self-excluded
                use_sticky = getattr(self, "use_sticky_price", True)
                
                if getattr(self, "exclude_self_from_state", False) and not use_sticky:
                    # TRANSPARENT MODE:
                    # Agent sees the External Env price, ignoring its own presence.
                    # Warning: Can cause self-cancellation loops if policy is naive.
                    best_bid = best_bid_env
                    best_ask = best_ask_env
                    spread = spread_env
                else:
                    # DEFAULT/STICKY MODE:
                    # If exclude_self=False, Raw == Full Book.
                    # If exclude_self=True AND sticky=True, Raw still includes our price 
                    # if we are the best (Effective TOB).
                    best_bid = best_bid_raw
                    best_ask = best_ask_raw
                    spread = spread_raw

                # Determine which volume view to expose to the agent
                if getattr(self, "exclude_self_from_state", False):
                    # FIX 2.1: When sticky mode causes the effective best
                    # price to be the agent's own (better than env), the
                    # external volume at that price is 0.  Using env volume
                    # would hallucinate protection that doesn't exist.
                    if use_sticky:
                        bid_size = bid_size_env if best_bid == best_bid_env else 0.0
                        ask_size = ask_size_env if best_ask == best_ask_env else 0.0
                        denom = max(1.0, bid_size + ask_size)
                        obi = (bid_size - ask_size) / denom
                    else:
                        bid_size, ask_size, obi = bid_size_env, ask_size_env, obi_env
                else:
                    # Agent sees full liquidity (including self)
                    bid_size, ask_size, obi = bidsize_raw, asksize_raw, obi_raw

                # --- 4. Pure MM Depth Features (Backtest) ---
                if self.pure_MM:
                    depth_bids = snap.get("full_depth_bids", [])
                    depth_asks = snap.get("full_depth_asks", [])
                    # (Fast lookup dicts)
                    db_map = { _safe_int(p, 10**9): _safe_float(v) for p, v in depth_bids }
                    da_map = { _safe_int(p, 10**9): _safe_float(v) for p, v in depth_asks }

                    max_b_off = max(0, getattr(self, "max_bid_offset", 0))
                    max_a_off = max(0, getattr(self, "max_ask_offset", 0))

                    # Anchor depth window based on exclusion setting
                    ref_bb = best_bid_env if getattr(self, "exclude_self_from_state", False) else best_bid
                    ref_ba = best_ask_env if getattr(self, "exclude_self_from_state", False) else best_ask

                    # [FIX] Check validity using flags, not value >= 0
                    valid_bb = has_env_bid if getattr(self, "exclude_self_from_state", False) else has_raw_bid
                    valid_ba = has_env_ask if getattr(self, "exclude_self_from_state", False) else has_raw_ask

                    pure_mm_bid_sizes = []
                    if valid_bb:
                        for k in range(max_b_off + 1):
                            px = int(ref_bb - k)
                            vol = db_map.get(px, 0.0)
                            if not getattr(self, "exclude_self_from_state", False):
                                vol += own_bid_qty.get(px, 0.0)
                            pure_mm_bid_sizes.append(vol)
                    else:
                        pure_mm_bid_sizes = [0.0] * (max_b_off + 1)

                    pure_mm_ask_sizes = []
                    if valid_ba:
                        for k in range(max_a_off + 1):
                            px = int(ref_ba + k)
                            vol = da_map.get(px, 0.0)
                            if not getattr(self, "exclude_self_from_state", False):
                                vol += own_ask_qty.get(px, 0.0)
                            pure_mm_ask_sizes.append(vol)
                    else:
                        pure_mm_ask_sizes = [0.0] * (max_a_off + 1)

        # ------------------------------------------------------------------
        # SIMULATION MODE Logic (Live Engine)
        # ------------------------------------------------------------------
        else:
            # --- [Time Capture] ---
            if self.lob is not None:
                # The engine records history in message_dict["Time"]. 
                # We retrieve the timestamp of the last processed event.
                times = self.lob.message_dict.get("Time", [])
                if times:
                    current_time = float(times[-1])
                else:
                    current_time = 0.0

            # --- 1. Raw/Full View (Engine State includes everything) ---
            best_bid_raw = _safe_int(self.best_bid())
            best_ask_raw = _safe_int(self.best_ask())
            
            # Simulation: Valid means >= 0 (strictly positive indices)
            has_raw_bid = (best_bid_raw >= 0)
            has_raw_ask = (best_ask_raw >= 0)

            if self.lob is not None:
                n_lev = int(self.lob.lob_state.shape[0])
                if 0 <= best_bid_raw < n_lev:
                    bidsize_raw = float(max(self.lob.lob_state[best_bid_raw], 0.0))
                if 0 <= best_ask_raw < n_lev:
                    asksize_raw = float(abs(min(self.lob.lob_state[best_ask_raw], 0.0)))

            if has_raw_bid and has_raw_ask:
                spread_raw = int(best_ask_raw - best_bid_raw)
                denom = max(1.0, bidsize_raw + asksize_raw)
                obi_raw = (bidsize_raw - asksize_raw) / denom
            
            mid_val = float(self.mid())

            # --- 2. Environment View (Calculated if exclusion is needed) ---
            if getattr(self, "exclude_self_from_state", False) and (self.lob is not None):
                env = self._env_view() # Generates book state minus MM orders
                
                best_bid_env = _safe_int(env.get("best_bid", -1))
                best_ask_env = _safe_int(env.get("best_ask", -1))
                bid_size_env = _safe_float(env.get("bidsize", 0.0))
                ask_size_env = _safe_float(env.get("asksize", 0.0))
                obi_env = _safe_float(env.get("obi", 0.0))

                has_env_bid = (best_bid_env >= 0)
                has_env_ask = (best_ask_env >= 0)

                if has_env_bid and has_env_ask:
                    spread_env = int(best_ask_env - best_bid_env)

                # --- Effective TOB Selection ---
                # NEW LOGIC: Check use_sticky_price flag
                
                use_sticky = getattr(self, "use_sticky_price", True)

                if not use_sticky:
                    # TRANSPARENT MODE:
                    # Agent sees exactly where the market would be without him.
                    best_bid = int(best_bid_env)
                    best_ask = int(best_ask_env)
                    # BUG FIX (X2): recompute mid from the env TOB so that
                    # mid, best_bid, best_ask, and spread are all consistent.
                    # Previously mid came from self.mid() (raw book including
                    # MM orders) while best_bid/best_ask came from _env_view()
                    # (book excluding MM), creating an inconsistent state.
                    if has_env_bid and has_env_ask:
                        mid_val = float(best_bid_env + best_ask_env) / 2.0
                else:
                    # STICKY MODE (Default):
                    # Effective TOB becomes sticky (max of Env and Own)
                    # Manual combine for simulation (where indices >= 0)
                    if has_env_bid and has_my_bid:
                        best_bid = int(max(best_bid_env, my_best_bid))
                    elif has_env_bid:
                        best_bid = int(best_bid_env)
                    elif has_my_bid:
                        best_bid = int(my_best_bid)
                    else:
                        best_bid = -1

                    if not has_env_ask and not has_my_ask: best_ask = -1
                    elif not has_env_ask: best_ask = int(my_best_ask)
                    elif not has_my_ask: best_ask = int(best_ask_env)
                    else: best_ask = int(min(best_ask_env, my_best_ask))

                if best_bid >= 0 and best_ask >= 0:
                    spread = int(best_ask - best_bid)

                # FIX 2.1: When the agent's price is strictly better than the
                # environment's best, the external volume at that improved
                # price is 0 (the agent is alone).  Showing the env volume
                # would hallucinate a "liquidity wall" that doesn't exist,
                # preventing the NN from learning adverse selection risk.
                if use_sticky:
                    bid_size = bid_size_env if best_bid == best_bid_env else 0.0
                    ask_size = ask_size_env if best_ask == best_ask_env else 0.0
                    denom = max(1.0, bid_size + ask_size)
                    obi = (bid_size - ask_size) / denom
                else:
                    bid_size, ask_size, obi = bid_size_env, ask_size_env, obi_env

            else:
                # No exclusion: Raw == Env == Effective
                best_bid_env, best_ask_env = best_bid_raw, best_ask_raw
                spread_env = spread_raw
                
                best_bid, best_ask = best_bid_raw, best_ask_raw
                spread = spread_raw
                
                bid_size, ask_size, obi = bidsize_raw, asksize_raw, obi_raw

            # --- 3. Pure MM Depth Features (Simulation) ---
            if self.pure_MM and (self.lob is not None):
                max_b_off = max(0, getattr(self, "max_bid_offset", 0))
                max_a_off = max(0, getattr(self, "max_ask_offset", 0))
                n_lev = int(self.lob.lob_state.shape[0])

                ref_bb = best_bid_env if getattr(self, "exclude_self_from_state", False) else best_bid
                ref_ba = best_ask_env if getattr(self, "exclude_self_from_state", False) else best_ask

                pure_mm_bid_sizes = []
                if ref_bb >= 0:
                    for k in range(max_b_off + 1):
                        px = int(ref_bb - k)
                        if px < 0: break
                        vol = float(max(self.lob.lob_state[px], 0.0))
                        if getattr(self, "exclude_self_from_state", False):
                            vol = max(0.0, vol - float(own_bid_qty.get(px, 0.0)))
                        pure_mm_bid_sizes.append(vol)
                    # Pad if needed
                    while len(pure_mm_bid_sizes) <= max_b_off: pure_mm_bid_sizes.append(0.0)
                else:
                    pure_mm_bid_sizes = [0.0] * (max_b_off + 1)

                pure_mm_ask_sizes = []
                if ref_ba >= 0:
                    for k in range(max_a_off + 1):
                        px = int(ref_ba + k)
                        if px >= n_lev: break
                        vol = float(abs(min(self.lob.lob_state[px], 0.0)))
                        if getattr(self, "exclude_self_from_state", False):
                            vol = max(0.0, vol - float(own_ask_qty.get(px, 0.0)))
                        pure_mm_ask_sizes.append(vol)
                    while len(pure_mm_ask_sizes) <= max_a_off: pure_mm_ask_sizes.append(0.0)
                else:
                    pure_mm_ask_sizes = [0.0] * (max_a_off + 1)

        # ------------------------------------------------------------------
        # Final State Dictionary Assembly
        # ------------------------------------------------------------------
        state: Dict[str, object] = {
            # --- [NEW] Time Key for Throttling ---
            "time": float(current_time),
            
            # Effective TOB (Decision Price)
            "mid": float(mid_val),
            "best_bid": int(best_bid),
            "best_ask": int(best_ask),
            "spread": int(spread),

            # External TOB (Throttling Signals)
            "best_bid_env": int(best_bid_env),
            "best_ask_env": int(best_ask_env),
            "spread_env": int(spread_env),

            # Volume & Imbalance (View dependent)
            "bidsize": float(bid_size),
            "asksize": float(ask_size),
            "obi": float(obi),

            # Raw/Debug Fields
            "best_bid_raw": int(best_bid_raw),
            "best_ask_raw": int(best_ask_raw),
            "spread_raw": int(spread_raw),
            "bidsize_raw": float(bidsize_raw),
            "asksize_raw": float(asksize_raw),
            "obi_raw": float(obi_raw),

            # Agent Internal State
            "my_best_bid": int(my_best_bid) if has_my_bid else -1,
            "my_best_ask": int(my_best_ask) if has_my_ask else -1,
            "inventory": float(self.inventory),  # FIX: was int(), truncated fractional inventory in crypto/forex mode
            "cash": float(self.cash),
            
            # Active Orders Meta
            "mm_n_orders": int(n_own_orders_valid),
            "n_own_orders": int(n_own_orders_valid),
            "mm_n_orders_total": int(n_own_orders_total),
            "mm_orders_prices": mm_orders_prices,
            "mm_orders_sides": mm_orders_sides,

            # Flags & Config
            "last_fill": self.last_fill,
            "has_bid": bool(has_my_bid),
            "has_ask": bool(has_my_ask),
            "exclude_self_from_state": bool(getattr(self, "exclude_self_from_state", False)),
            "backtest_mode": bool(getattr(self, "backtest_mode", False)),
        }

        # Append Pure MM Depth Vectors if generated
        if self.pure_MM and (pure_mm_bid_sizes is not None) and (pure_mm_ask_sizes is not None):
            state["pure_mm_bid_sizes"] = pure_mm_bid_sizes
            state["pure_mm_ask_sizes"] = pure_mm_ask_sizes

        # MO flow EWMA (for regime-switching / adversarial state signal)
        state["mo_flow_ewma"] = float(self.mo_flow_ewma)
        state["mo_flow_p_hat"] = float(self.get_mo_flow_p_hat())
        state["mo_flow_fast_ewma"] = float(self.mo_flow_fast_ewma)
        state["mo_flow_fast_p_hat"] = float(self.get_mo_flow_fast_p_hat())
        if bool(getattr(self, "use_bayes_flow_signal", False)):
            state.update(self.get_bayes_flow_features())

        # Fill imbalance signal (endogenous signal -- how our quotes are being hit).
        # The DQN keeps reading the legacy key for checkpoint compatibility; when
        # use_quote_exposure_imbalance_signal=True it is backed by
        # quote_exposure_imbalance; otherwise it is the original fill EWMA.
        state["fill_imbalance_ewma"] = float(self.get_effective_fill_imbalance_feature())
        state["fill_imbalance_old_ewma"] = float(self.fill_imbalance_ewma)
        state["fill_bid_ewma"] = float(self.fill_bid_ewma)
        state["fill_ask_ewma"] = float(self.fill_ask_ewma)
        if bool(getattr(self, "use_quote_exposure_imbalance_signal", False)):
            state.update(self.get_quote_exposure_imbalance_features())

        return state

    # ======================================================================
    # Pre-step: ask policy, execute, return telemetry
    # ======================================================================
    def pre_step(self) -> Tuple[str, int, int, Optional[int], Optional[int]]:
        """
        Ask the policy/controller what to do, execute it, and return telemetry.

        OPTIMIZATION - QUEUE PRESERVATION (SMART MODIFY):
        -------------------------------------------------
        This method now implements a "Smart Modify" logic for ALL quoting actions,
        including 'Inside Spread' strategies.

        Logic:
        1. Calculate the target price based on the action (L1 or Inside).
        2. Check if an active order already exists on that side.
        3. If existing_order.price == target_price:
           -> DO NOTHING (Hold). This preserves the order's FIFO queue priority.
        4. If prices differ (or no order exists):
           -> Cancel existing orders on that side.
           -> Place a new order at the target price (new timestamp, back of queue).

        This eliminates the "bias" where aggressive actions were previously
        disadvantaged by constantly resetting their queue position.
        """
        # 0. Reset per-step forced cancel log
        self._last_forced_cancels = []

        # 1. Internal bookkeeping hygiene (remove ghosts)
        self._prune_orphan_orders()

        # Default return values (No-Op)
        action, side, price, oid, last_pos = "hold", 0, -1, None, None

        if self.policy is None:
            return action, side, price, oid, last_pos

        # 2. Build State & Query Policy
        state_for_policy = self.build_state()

        # --- Policy Calling Convention (Robustness for different signatures) ---
        if getattr(self, "_policy_call_cached_obj", None) is not self.policy:
            self._policy_call_cached_obj = self.policy
            self._policy_call_kind = "plain"
            try:
                sig = inspect.signature(self.policy)
                params = sig.parameters
                if "mm" in params:
                    if params["mm"].kind == inspect.Parameter.POSITIONAL_ONLY:
                        self._policy_call_kind = "mm_pos"
                    else:
                        self._policy_call_kind = "mm_kw"
                else:
                    for p in params.values():
                        if p.kind == inspect.Parameter.VAR_KEYWORD:
                            self._policy_call_kind = "mm_kw"
                            break
            except (TypeError, ValueError):
                self._policy_call_kind = "plain"
        
        if self._policy_call_kind == "mm_pos":
            act = self.policy(state_for_policy, self)
        elif self._policy_call_kind == "mm_kw":
            act = self.policy(state_for_policy, mm=self)
        else:
            act = self.policy(state_for_policy)

        self.last_policy_action = act
        # Snapshot state AFTER policy call so that RL controllers can inject
        # metadata (e.g. RL_State_Vector) into state_for_policy during act().
        self.last_policy_state = dict(state_for_policy)

        if not isinstance(act, tuple) or len(act) == 0:
            return action, side, price, oid, last_pos

        a0 = act[0]

        # Helper to ensure prices stay within grid bounds
        def clamp_price(px: int) -> int:
            if self.backtest_mode: return int(px)
            return max(0, min(int(px), self.owner.shape[1] - 1))

        # ------------------------------------------------------------------
        # HELPER: Smart Place/Replace Logic (The Core Optimization)
        # ------------------------------------------------------------------
        def _smart_place(target_side: int, target_price: int) -> Tuple[int, int]:
            """
            Intelligently manages order placement to preserve queue priority.
            
            Returns:
                (order_id, priority)
            """
            # Identify any existing order on this side
            curr_oid = self._find_order_to_cancel_on_side(target_side)
            
            # Scenario A: Preservation
            # We are already sitting at the target price.
            # Do NOT cancel. Return current details.
            if curr_oid is not None and self.orders[curr_oid]["price"] == target_price:
                return curr_oid, int(self.orders[curr_oid].get("priority", 0))
            
            # Scenario B: Modification
            # Price changed or no order exists.
            # Must clear the side to prevent stacking, then place new.
            if target_side == +1:
                self.cancel_bid_all()
            else:
                self.cancel_ask_all()

            oid, prio = self.place_limit(target_side, target_price)
            if oid == -1:
                # place_limit rejected the price (e.g., crosses market or
                # out of grid bounds).  Try a safe fallback: best_bid-1 for
                # buys, best_ask+1 for sells.  If that also fails, return
                # the failure so the caller falls back to "hold".
                bb = self._safe_best_bid_idx()
                ba = self._safe_best_ask_idx()
                if target_side == +1 and ba >= 0:
                    oid, prio = self.place_limit(+1, ba - 1)
                elif target_side == -1 and bb >= 0:
                    oid, prio = self.place_limit(-1, bb + 1)
            return oid, prio

        # ==================================================================
        # DECODE ACTIONS
        # ==================================================================
        
        # --- Standard Actions (L1 Pegging) ---
        if a0 == "place_bid":
            raw_px = act[1] if len(act) > 1 else self.best_bid()
            target_px = clamp_price(raw_px)
            
            oid, prio = _smart_place(+1, target_px)
            
            if oid != -1: return "place_bid", +1, target_px, oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "place_ask":
            raw_px = act[1] if len(act) > 1 else self.best_ask()
            target_px = clamp_price(raw_px)
            
            oid, prio = _smart_place(-1, target_px)
            
            if oid != -1: return "place_ask", -1, target_px, oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "place_bid_ask":
            if len(act) < 3: return "hold", 0, -1, None, None
            bid_px = clamp_price(act[1])
            ask_px = clamp_price(act[2])
            if not (bid_px < ask_px):
                return "hold", 0, -1, None, None

            # Process sides independently to maximize queue preservation
            oid_b, prio_b = _smart_place(+1, bid_px)
            oid_a, prio_a = _smart_place(-1, ask_px)

            # Return logic handles partial failures (e.g. if one side is out of bounds)
            if oid_b != -1 and oid_a != -1: return "place_bid_ask", 0, bid_px, None, None
            if oid_b != -1: return "place_bid", +1, bid_px, oid_b, prio_b
            if oid_a != -1: return "place_ask", -1, ask_px, oid_a, prio_a
            return "hold", 0, -1, None, None

        # --- Inside Spread Actions (Now Intelligent) ---
        elif a0 == "place_bid_inside_spread":
            # 1. Determine Target Price explicitly BEFORE cancelling
            bid_px, _ = self._compute_inside_spread_prices()
            
            if bid_px is None:
                # Spread too tight or invalid.
                # If we hold here, we might keep an L1 order if we had one.
                return "hold", 0, -1, None, None

            # 2. Use Smart Place to preserve queue if price matches
            oid, prio = _smart_place(+1, int(bid_px))
            
            if oid != -1: return "place_bid_inside_spread", +1, int(bid_px), oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "place_ask_inside_spread":
            _, ask_px = self._compute_inside_spread_prices()
            
            if ask_px is None:
                return "hold", 0, -1, None, None

            oid, prio = _smart_place(-1, int(ask_px))
            
            if oid != -1: return "place_ask_inside_spread", -1, int(ask_px), oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "place_bid_ask_inside_spread":
            bid_px, ask_px = self._compute_inside_spread_prices()
            
            # Validation: Need both sides valid and uncrossed
            if (bid_px is None) or (ask_px is None) or not (bid_px < ask_px):
                return "hold", 0, -1, None, None

            # Attempt Smart Place on both sides
            oid_b, prio_b = _smart_place(+1, int(bid_px))
            oid_a, prio_a = _smart_place(-1, int(ask_px))

            # Robust Returns for partial success
            if oid_b != -1 and oid_a != -1: return "place_bid_ask_inside_spread", 0, int(bid_px), None, None
            if oid_b != -1: return "place_bid_inside_spread", +1, int(bid_px), oid_b, prio_b
            if oid_a != -1: return "place_ask_inside_spread", -1, int(ask_px), oid_a, prio_a
            return "hold", 0, -1, None, None

        # --- Cancellation & Clearing ---
        elif a0 == "cancel" and len(act) > 1:
            ok = self.cancel(int(act[1]))
            return ("cancel", 0, -1, int(act[1]) if ok else None, None)

        elif a0 == "cancel_bid":
            ok = self.cancel_bid_side()
            if ok: return "cancel_bid", +1, -1, None, None
            return "hold", 0, -1, None, None

        elif a0 == "cancel_ask":
            ok = self.cancel_ask_side()
            if ok: return "cancel_ask", -1, -1, None, None
            return "hold", 0, -1, None, None

        elif a0 == "cancel_all":
            self.cancel_all()
            return "cancel_all", 0, -1, None, None
        
        # --- Forced Reset Actions (Explicit "Cancel then Place") ---
        # These actions INTENTIONALLY sacrifice queue priority.
        elif a0 == "cancel_bid_then_place" and len(act) >= 2:
            bid_px = clamp_price(act[1])
            self.cancel_bid_all()
            oid, prio = self.place_limit(+1, bid_px)
            if oid != -1: return "cancel_bid_then_place", +1, bid_px, oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "cancel_ask_then_place" and len(act) >= 2:
            ask_px = clamp_price(act[1])
            self.cancel_ask_all()
            oid, prio = self.place_limit(-1, ask_px)
            if oid != -1: return "cancel_ask_then_place", -1, ask_px, oid, prio
            return "hold", 0, -1, None, None

        elif a0 == "cancel_all_then_place" and len(act) >= 3:
            side = int(act[1])
            price = clamp_price(act[2])
            self.cancel_all()
            oid, prio = self.place_limit(side, price)
            if oid != -1:
                return ("place_bid" if side == +1 else "place_ask"), side, price, oid, prio
            return "hold", 0, -1, None, None
        
        elif a0 == "cancel_all_then_place_bid_ask" and len(act) >= 3:
            bid_px = clamp_price(act[1])
            ask_px = clamp_price(act[2])
            if not (bid_px < ask_px):
                return "hold", 0, -1, None, None
            self.cancel_all()
            self.place_limit(+1, bid_px)
            self.place_limit(-1, ask_px)
            return "place_bid_ask", 0, -1, None, None

        # --- Aggressive Crossing ---
        elif a0 == "cross_buy":
            px, ex_qty = self.cross_mo(+1)
            return "cross_buy", +1, int(px), None, None
        
        elif a0 == "cross_sell":
            px, ex_qty = self.cross_mo(-1)
            return "cross_sell", -1, int(px), None, None

        # Fallback
        return "hold", 0, -1, None, None
    # ======================================================================
    # Mirror engine transforms (simulation only)
    # ======================================================================
    def apply_shift(self, shift: int):
        """
        Apply a global *re-centering* shift to our own order prices and our owner mask.

        The LOB engine periodically recenters the book by shifting the entire price
        grid by an integer number of ticks. When that happens:
          - Every *price index* we store must be shifted by the same amount.
          - Our boolean owner mask must be shifted in the same way, to remain
            aligned with the engine's `priorities_lob_state`.

        Convention (must match the engine)
        ---------------------------------
        If `shift > 0`, the engine shifts the book *left* by `shift` ticks:
            new_px = old_px - shift
        If `shift < 0`, the engine shifts the book *right* by `abs(shift)` ticks:
            new_px = old_px - shift   (still true, since shift is negative)

        This function does NOT change queue priorities (rank index). It only
        updates price indices.

        Robustness:
        - We also prune any orders that become out-of-bounds after shifting.
          (It can happen if the engine shifts and some of our orders are near
           the edge of the finite price grid.)
        """
        if shift == 0:
            return

        if self.backtest_mode:
            # In backtest mode, we do not maintain a live `owner` matrix, but we
            # still update the stored prices for consistency (e.g., logs).
            for oid, info in list(self.orders.items()):
                if "price" not in info:
                    continue
                info["price"] = int(info["price"]) - int(shift)
            return

        # --------------------------
        # Shift the owner mask
        # --------------------------
        k = int(shift)

        if k > 0:
            # Shift LEFT by k: new[:, j] = old[:, j+k]
            self.owner[:, :-k] = self.owner[:, k:]
            self.owner[:, -k:] = False
        else:
            # Shift RIGHT by |k|: new[:, j] = old[:, j-|k|]
            kk = -k
            self.owner[:, kk:] = self.owner[:, :-kk]
            self.owner[:, :kk] = False

        # --------------------------
        # Shift stored order prices
        # --------------------------
        for oid, info in list(self.orders.items()):
            if "price" not in info:
                continue
            info["price"] = int(info["price"]) - int(shift)

        # After any shift, clean up inconsistent / out-of-bounds entries.
        self._prune_orphan_orders()

    def _apply_local_fifo_depletion(
        self,
        price: int,
        consumed_units: int,
        consumed_side: Optional[int] = None,
    ) -> None:
        """
        Mirror a FIFO dequeue of `consumed_units` at one price level in local MM tracking.

        This is needed when executions happen outside post_step_detect_fill_by_env_mo(),
        e.g. MM aggressive market orders via cross_mo().
        """
        if self.backtest_mode or self.lob is None:
            return

        p = int(price)
        k = int(consumed_units)
        if k <= 0:
            return

        if hasattr(self.lob, "priorities_lob_state"):
            self._owner_mask_like(self.lob.priorities_lob_state)

        if p < 0 or p >= int(self.owner.shape[1]):
            return

        # 1) Remove own orders that were in the consumed front ranks, when side matches.
        if consumed_side in (+1, -1):
            to_del: List[int] = []
            for oid, info in self.orders.items():
                if int(info.get("price", -999)) != p:
                    continue
                if int(info.get("side", 0)) != int(consumed_side):
                    continue
                pr = info.get("priority", None)
                if pr is None:
                    continue
                try:
                    pri = int(pr)
                except Exception:
                    continue
                if pri < k:
                    to_del.append(int(oid))
            for oid in to_del:
                if oid in self.orders:
                    del self.orders[oid]

        # 2) Shift local owner mask by k ranks (front consumed).
        col = self.owner[:, p].copy()
        if k >= col.size:
            self.owner[:, p] = False
        else:
            self.owner[:-k, p] = col[k:]
            self.owner[-k:, p] = False

        # 3) Shift stored priorities by k for remaining orders at this price (same side only).
        for info in self.orders.values():
            if int(info.get("price", -999)) != p:
                continue
            if consumed_side in (+1, -1) and int(info.get("side", 0)) != int(consumed_side):
                continue
            pr = info.get("priority", None)
            if pr is None:
                continue
            try:
                pri = int(pr)
            except Exception:
                continue
            info["priority"] = max(0, pri - k)
        
    def post_step_adjust_for_env_mo(
        self,
        order_type: int,
        order_price_pre_shift: int,
        order_size: int = 1,
    ):
        """
        Mirror environment MO queue effect (simulation only).

        This is a *fallback* utility that shifts our local queue positions at a
        single price level by the amount of volume consumed at that price.

        In the main simulation driver, we prefer to call
        post_step_detect_fill_by_env_mo(..., executed_by_price_pre_shift=...)
        which handles BOTH:
          (i) queue shifts and
          (ii) our fills
        across ALL price levels in a sweep.
        """
        if self.backtest_mode or self.lob is None:
            return

        if order_type != 1:
            return

        p = int(order_price_pre_shift)
        if p < 0 or p >= self.owner.shape[1]:
            return

        k = int(order_size)
        if k <= 0:
            return

        # We only have a finite number of priority ranks in the owner matrix.
        k = min(k, int(self.owner.shape[0]))
        if k <= 0:
            return

        # Shift our owner marks forward by k ranks (k units were consumed ahead).
        col = self.owner[:, p].copy()
        self.owner[:-k, p] = col[k:]
        self.owner[-k:, p] = False

        # Every remaining order at this price moves k steps closer to the front.
        for oid, info in self.orders.items():
            if int(info.get("price", -1)) != p:
                continue
            pr = info.get("priority", None)
            if pr is None:
                continue
            info["priority"] = max(0, int(pr) - k)

        # Extra hygiene after queue adjustments
        self._prune_orphan_orders()

    # ======================================================================
    # Detect fills caused by ENVIRONMENT MOs (simulation only)
    # ======================================================================
    def post_step_detect_fill_by_env_mo(
        self,
        order_type: int,
        order_sign: Optional[int] = None,
        order_price_pre_shift: int = 0,
        step_idx: int = 0,
        order_size: int = 1,
        executed_by_price_pre_shift: Optional[Dict[int, int]] = None,
        pre_priorities_cols: Optional[np.ndarray] = None,
        # Backward/forward compatibility: some call sites use order_direction
        order_direction: Optional[int] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """
        Detect passive fills caused by an ENVIRONMENT market order (MO).

        Why this matters
        ----------------
        When the environment sends a market order, the engine consumes queue
        volume at one or more price levels (a 'sweep'). For each affected price:
          - the first k units in the FIFO queue are removed;
          - the entire queue shifts forward by k ranks.

        If we do NOT mirror this rank shift in our `owner` mask and in our stored
        `priority` ranks, our local queue positions drift over time and fill
        detection becomes wrong (serious bug, affects PnL and inventory paths).

        Inputs
        ------
        executed_by_price_pre_shift:
            A dict {price_index_pre_shift: executed_qty_at_that_price}
            computed by the simulation driver from (pre_state - post_state mapped
            back into pre-shift coordinates). This is the most reliable way to
            handle sweeps.

        Notes
        -----
        - All operations here are done in the PRE-SHIFT price coordinate system.
          The driver applies the global recentering shift after this call.
        - We assume unit orders (qty=1 per rank). If you use per-order qty != 1,
          you would need to adapt the FIFO logic accordingly.
        """
        # Only meaningful in simulation mode with a live engine.
        if self.backtest_mode or self.lob is None:
            return {"filled_oid": -1, "px": int(order_price_pre_shift), "side": 0, "qty": 0.0}

        if int(order_type) != 1:
            return {"filled_oid": -1, "px": int(order_price_pre_shift), "side": 0, "qty": 0.0}
        # Resolve the environment MO direction sign.
        # Some call sites pass `order_sign` (legacy), others pass `order_direction`.
        # We accept both and normalize to `order_direction`.
        if order_sign is None:
            order_sign = order_direction
        if order_sign is None:
            order_sign = 0
        order_direction = int(order_sign)


        # If the driver did not provide sweep info, we can at best shift by 1 at
        # the reported price (legacy fallback). Fill detection across sweeps is
        # impossible without executed_by_price_pre_shift.
        # If the driver did not provide sweep-by-price info, we can only apply a
        # conservative FIFO shift at the reported price (legacy fallback).
        #
        # IMPORTANT:
        # - We cannot reliably detect WHICH of our orders filled across a sweep without
        #   the per-price executed quantities.
        # - We still *must* mirror the rank shift to prevent our local queue positions
        #   from drifting over time (serious bug otherwise).
        if executed_by_price_pre_shift is None:
            # Fallback when the driver does not provide sweep-by-price information.
            #
            # In your setup, every environment market order has unit size (order_size == 1),
            # so treating the event as "k units executed at the reported pre-shift price"
            # is both safe and, in most cases, exact.
            #
            # Crucially, this avoids a serious accounting bug where our resting order
            # can be consumed but we return side=0 (no fill) simply because
            # executed_by_price_pre_shift is missing.
            # Guard: if order_size is 0 (no-op MO), skip — no queue shift needed.
            if int(order_size) <= 0:
                return {"side": 0, "px": -1, "qty": 0.0}
            executed_by_price_pre_shift = {int(order_price_pre_shift): int(order_size)}

        # Env direction: +1 = buy MO (hits asks), -1 = sell MO (hits bids)
        filled_side = -1 if int(order_direction) == +1 else +1

        total_filled_qty = 0.0
        last_filled_oid: Optional[int] = None
        last_px: Optional[int] = None
        # Per-fill list: one entry per individual fill at each price level.
        # Used by reward functions that need per-fill spread capture
        # (avoids the collapsed last_px + total_qty aggregation).
        individual_fills: List[Dict[str, Any]] = []

        # We will process each swept price level independently.
        for p_raw, k_raw in executed_by_price_pre_shift.items():
            p = int(p_raw)
            k = int(k_raw)

            if k <= 0:
                continue
            if p < 0 or p >= self.owner.shape[1]:
                continue

            # ---------------------------------------------------------
            # 1) Determine which of OUR orders were in the first k ranks
            # ---------------------------------------------------------
            # FIFO convention:
            #   - rank 0 is at the FRONT (oldest, executed first)
            #   - ranks increase towards the BACK (newer, executed later)
            #
            # Therefore, any of our orders at this price with priority < k are filled.
            our_at_p: List[Tuple[int, int]] = []
            for oid, info in self.orders.items():
                if int(info.get("price", -999)) != p:
                    continue
                if int(info.get("side", 0)) != filled_side:
                    # Only orders on the side being consumed can be filled.
                    continue
                pr = info.get("priority", None)
                if pr is None:
                    continue
                our_at_p.append((int(oid), int(pr)))

            if our_at_p:
                # Fill in increasing priority order (front to back)
                our_at_p.sort(key=lambda x: x[1])

                for oid, pr in our_at_p:
                    if pr >= k:
                        break  # remaining are behind the executed volume

                    info = self.orders.get(oid, None)
                    if info is None:
                        continue

                    qty_fill = float(info.get("qty", 1.0))

                    # Accounting at the executed price (pre-shift)
                    price_dollar = self._px_to_dollars(p) * qty_fill
                    if filled_side == -1:
                        # Our ASK got lifted -> we SELL
                        self.inventory -= qty_fill
                        self.cash += price_dollar
                    else:
                        # Our BID got hit -> we BUY
                        self.inventory += qty_fill
                        self.cash -= price_dollar

                    total_filled_qty += qty_fill
                    last_filled_oid = int(oid)
                    last_px = int(p)

                    # Record individual fill for per-fill reward computation
                    individual_fills.append({
                        "filled_oid": int(oid),
                        "px": int(p),
                        "side": int(filled_side),
                        "qty": float(qty_fill),
                    })

                    # Remove from local tracking.
                    # IMPORTANT: we do NOT need to manually clear owner[pr, p] here,
                    # because the queue shift below will drop all ranks < k anyway.
                    del self.orders[oid]

            # ---------------------------------------------------------
            # 2) Mirror the FIFO rank shift by k at this price level
            # ---------------------------------------------------------
            # Shift our owner mask column forward by k ranks:
            #   new_rank = old_rank - k
            col = self.owner[:, p].copy()
            if k >= col.size:
                self.owner[:, p] = False
            else:
                self.owner[:-k, p] = col[k:]
                self.owner[-k:, p] = False

            # Update stored priorities for remaining orders at this price (same side only).
            # FIX: was applying priority decrement to BOTH sides at this price.
            # Only orders on the filled_side had their queue consumed.
            for oid2, info2 in self.orders.items():
                if int(info2.get("price", -999)) != p:
                    continue
                if int(info2.get("side", 0)) != filled_side:
                    continue
                pr2 = info2.get("priority", None)
                if pr2 is None:
                    continue
                info2["priority"] = max(0, int(pr2) - k)

        # Final hygiene after multiple deletions and shifts
        self._prune_orphan_orders()

        return {
            "filled_oid": (int(last_filled_oid) if last_filled_oid is not None else -1),
            "px": (int(last_px) if last_px is not None else int(order_price_pre_shift)),
            "side": int(filled_side if total_filled_qty > 0 else 0),
            "qty": float(total_filled_qty),
            "fills_list": individual_fills,
        }

    # ======================================================================
    # Mirror ENVIRONMENT cancellations into MM queue-ahead bookkeeping (sim only)
    # ======================================================================
    def adjust_for_cancel_at_price(
        self,
        price: int,
        pre_col: np.ndarray,
        post_col: np.ndarray,
        cancel_rank: Optional[int] = None,
    ):
        """
        Mirror an *environment* cancellation at a given price into the MM's internal
        queue-ahead bookkeeping (simulation mode only).
    
        Key idea
        --------
        In this simulator, the priority column stores *indistinguishable unit placeholders*.
        Therefore, (pre_col, post_col) alone does NOT uniquely identify which FIFO rank was cancelled.
        We must use the engine-provided `cancel_rank`.
    
        Safety contract
        ---------------
        - Environment cancels are background liquidity changes (they must NOT delete our orders).
        - If cancel_rank is missing/invalid => do nothing (never guess).
        - If cancel_rank lands on one of *our tracked ranks* => do nothing (this indicates an upstream bug;
          we refuse to corrupt our tracking by "fixing" it here).
        - If cancel_rank is beyond our tracked depth => do nothing (it does not affect our tracked prefix).
        """
    
        # ------------------------------------------------------------
        # 0) Only meaningful in simulation mode with a live engine ref.
        # ------------------------------------------------------------
        if self.backtest_mode or self.lob is None:
            return
    
        # Basic bounds check for price column index.
        if price < 0 or price >= self.owner.shape[1]:
            return
    
        # ------------------------------------------------------------
        # 1) Parse / validate cancel_rank robustly
        # ------------------------------------------------------------
        # In practice cancel_rank may come from pandas and be float NaN even if annotated Optional[int].
        # Our contract is: if it's missing/invalid => no-op (never guess).
        if cancel_rank is None:
            return
    
        try:
            # Accept int-like values, but reject NaN / inf.
            cr = float(cancel_rank)
            if not np.isfinite(cr):
                return
            r = int(cr)
        except Exception:
            return
    
        if r < 0:
            return
    
        # ------------------------------------------------------------
        # 2) Defensive: normalize input columns
        # ------------------------------------------------------------
        # We only use these to sanity-check that a single unit cancel happened at this price.
        try:
            pre = np.asarray(pre_col).ravel()
            post = np.asarray(post_col).ravel()
        except Exception:
            return
    
        if pre.size == 0 or post.size == 0:
            return
    
        # Queue lengths in "unit placeholders" (how many non-zeros exist in that price column).
        q_pre = int(np.sum(np.abs(pre) > 0))
        q_post = int(np.sum(np.abs(post) > 0))
    
        # ------------------------------------------------------------
        # 3) Sanity checks about the cancel event
        # ------------------------------------------------------------
        # If the cancel did not actually reduce the queue by 1, ignore (no-op cancel or inconsistency).
        if q_pre <= 0:
            return
        if q_post != max(q_pre - 1, 0):
            return
    
        # If the engine rank is outside the pre-event queue length, ignore.
        if r >= q_pre:
            return
    
        # Our tracking depth (how many FIFO ranks we track locally for ownership/queue-ahead).
        depth = int(self.owner.shape[0])
    
        # IMPORTANT FIX:
        # If the cancelled rank is beyond our tracked depth, it does NOT affect the tracked prefix,
        # so we do nothing (otherwise we'd incorrectly "shift" inside our tracked ranks).
        if r >= depth:
            return
    
        # ------------------------------------------------------------
        # 4) CRITICAL SAFETY RULE: env cancels must not delete our orders
        # ------------------------------------------------------------
        # If cancel_rank points to a rank that we believe is ours, something upstream is wrong:
        # - engine misclassified our order as "environment"
        # - or our tracking drifted
        # In both cases, we refuse to apply destructive corrections here.
        if bool(self.owner[r, price]):
            return
    
        # ------------------------------------------------------------
        # 5) Mirror FIFO shift in our local owner-mask at this price
        # ------------------------------------------------------------
        # Removing rank r shifts all ranks > r one step forward.
        # Example: ranks [0,1,2,3] and cancel at r=1 => old rank2 becomes new rank1, etc.
        if r < depth - 1:
            self.owner[r : depth - 1, price] = self.owner[r + 1 : depth, price]
            self.owner[depth - 1, price] = False
        else:
            # r == last tracked rank
            self.owner[depth - 1, price] = False
    
        # Clear any owner slots beyond the NEW queue length (q_post).
        # This prevents stale True's living at impossible ranks after the queue shrinks.
        new_len = max(q_pre - 1, 0)  # equals q_post by the check above
        if new_len < depth:
            self.owner[new_len:, price] = False
    
        # ------------------------------------------------------------
        # 6) Decrement stored priorities for our orders behind the cancelled unit
        # ------------------------------------------------------------
        # Any of our orders at this price with priority > r move one step closer to front.
        for oid, info in self.orders.items():
            if int(info.get("price", -999)) != int(price):
                continue
    
            pr = info.get("priority", None)
            if pr is None:
                continue
    
            # Be defensive: priority could be float (from pandas) or malformed.
            try:
                pri = int(pr)
            except Exception:
                continue
    
            if pri > r:
                info["priority"] = pri - 1
    
        # ------------------------------------------------------------
        # 7) Final hygiene: drop any now-invalid (price, priority) pairs
        # ------------------------------------------------------------
        self._prune_orphan_orders()
    

    # ======================================================================
    # Backtest-only: register passive fill from FPT + queue-ahead
    # ======================================================================
    def register_backtest_passive_fill(
        self,
        oid: int,
        step_idx: int,
        time_val: float,
        qty: float = 1.0,
    ):
        """
        Apply the PnL effects of a passive fill in backtest mode.

        IMPORTANT:
        ----------
        Backtest mode supports non-unit quantities. Therefore this method
        supports partial fills. We only delete the order when its remaining
        qty reaches zero.
        """
        if not self.backtest_mode:
            return

        info = self.orders.get(oid)
        if info is None:
            return

        qty_req = float(qty)
        if qty_req <= 0.0:
            return

        px = int(info["price"])
        side = int(info["side"])

        remaining = float(info.get("qty", 1.0))
        if remaining <= 0.0:
            # Defensive: stale order with non-positive qty.
            del self.orders[oid]
            return

        qty_fill = min(qty_req, remaining)
        if qty_fill <= 0.0:
            return

        price_dollar = self._px_to_dollars(px) * qty_fill

        if side == -1:
            self.inventory -= qty_fill
            self.cash += price_dollar
        else:
            self.inventory += qty_fill
            self.cash -= price_dollar

        # Update last fill telemetry
        self.last_fill = {
            "side": side,
            "price": px,
            "time": float(time_val),
            "step": int(step_idx),
            "qty": float(qty_fill),
        }

        # Decrease remaining qty; only delete if fully filled.
        remaining_after = remaining - qty_fill
        if remaining_after <= 1e-12:
            del self.orders[oid]
        else:
            info["qty"] = float(remaining_after)
            # If queue-ahead is tracked, we stay at the front once queue_ahead==0.
            info["queue_ahead"] = float(info.get("queue_ahead", 0.0))

    # ======================================================================
    # Telemetry → DataFrame + logging
    # ======================================================================
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def log_step(
        self,
        step_idx: int,
        time_val: float,
        action: str,
        side: int,
        price: Any,
        oid: Optional[int],
        last_pos: Optional[int],
        action_idx: int = -1,
    ):
        """
        Append current MM state to the log for later analysis / animation.
        """
        # ---------------------------------------------------------------------
        # Robustness: some call sites may pass a *payload* instead of a scalar
        # price (e.g. (price, qty) or (price, fill_list) for sweep MOs).
        # For MM logging we only need a scalar tick index, so we coerce here.
        # ---------------------------------------------------------------------
        def _coerce_price_to_int(p) -> int:
            """Best-effort conversion of `p` to an integer tick index."""
            if p is None:
                return -1
            # Unwrap containers: (price, ...) -> price
            if isinstance(p, (tuple, list)):
                if len(p) == 0:
                    return -1
                return _coerce_price_to_int(p[0])
            try:
                import numpy as _np
                if isinstance(p, _np.ndarray):
                    if p.size == 0:
                        return -1
                    return _coerce_price_to_int(p.ravel()[0])
                if isinstance(p, _np.generic):
                    return int(p)
            except Exception:
                pass
            # Plain scalars / last resort
            try:
                return int(p)
            except Exception:
                return -1
        
        price_int = _coerce_price_to_int(price)
        
        had_fill = False
        fill_side = 0
        fill_price = -1
        if self.last_fill is not None:
            lf_step = int(self.last_fill.get("step", -1))
            if lf_step == int(step_idx):
                had_fill = True
                fill_side = int(self.last_fill.get("side", 0))
                fill_price = int(self.last_fill.get("price", -1))

        logged_action = action
        if had_fill and action == "hold":
            logged_action = "passive_fill"

        rl_spread = float("nan")
        rl_bidsize = float("nan")
        rl_asksize = float("nan")
        rl_inventory = float("nan")
        rl_has_bid = False
        rl_has_ask = False

        rl_state_vec = None
        rl_state_mode = None
        rl_state_dim = None
        rl_state_max_offset = None

        if self.last_policy_state is not None:
            rl_spread = float(self.last_policy_state.get("spread", float("nan")))
            rl_bidsize = float(self.last_policy_state.get("bidsize", float("nan")))
            rl_asksize = float(self.last_policy_state.get("asksize", float("nan")))
            rl_inventory = float(self.last_policy_state.get("inventory", float("nan")))
            rl_has_bid = bool(self.last_policy_state.get("has_bid", False))
            rl_has_ask = bool(self.last_policy_state.get("has_ask", False))

            rl_state_vec = self.last_policy_state.get("RL_State_Vector", None)
            rl_state_mode = self.last_policy_state.get("RL_State_Mode", None)
            rl_state_dim = self.last_policy_state.get("RL_State_Dim", None)
            rl_state_max_offset = self.last_policy_state.get("RL_State_MaxOffset", None)

        bid_pos = self._current_bid_pos()
        ask_pos = self._current_ask_pos()

        logged_price = price_int
        logged_last_pos = last_pos

        if action in ("place_bid", "place_ask") and oid is not None and oid in self.orders:
            info = self.orders[oid]
            logged_price = int(info["price"])
            logged_last_pos = int(info.get("priority", 0))

        if action in ("cross_buy", "cross_sell") and price_int >= 0:
            logged_price = price_int

        bid_dist: Optional[float] = None
        ask_dist: Optional[float] = None

        has_bid_any = any(info["side"] == +1 for info in self.orders.values())
        has_ask_any = any(info["side"] == -1 for info in self.orders.values())

        mid_idx = float(self.mid())

        if has_bid_any:
            best_mm_bid = max(
                (info["price"] for info in self.orders.values() if info["side"] == +1),
                default=None,
            )
            if best_mm_bid is not None:
                bid_dist = float(best_mm_bid - mid_idx)

        if has_ask_any:
            best_mm_ask = min(
                (info["price"] for info in self.orders.values() if info["side"] == -1),
                default=None,
            )
            if best_mm_ask is not None:
                ask_dist = float(best_mm_ask - mid_idx)

        # own_prices: List[int] = []
        # own_sides: List[int] = []
        # own_prios: List[float] = []
        # own_qlens: List[float] = []

        # for oid_i, info in self.orders.items():
        #     px = int(info["price"])
        #     side_i = int(info["side"])

        #     if (self.lob is not None) and (not self.backtest_mode):
        #         total_vol = 0.0
        #         if 0 <= px < self.lob.lob_state.shape[0]:
        #             total_vol = float(abs(self.lob.lob_state[px]))

        #         qlen_orders = 0
        #         if 0 <= px < self.lob.priorities_lob_state.shape[1]:
        #             col = self.lob.priorities_lob_state[:, px]
        #             qlen_orders = int(np.count_nonzero(col))

        #         if total_vol <= 0.0 and qlen_orders > 0:
        #             total_vol = float(qlen_orders)

        #         prio_rank = int(info.get("priority", 0))

        #         if qlen_orders > 0 and total_vol > 0.0:
        #             vol_per_order = total_vol / float(qlen_orders)
        #             vol_ahead = vol_per_order * max(0, prio_rank)
        #         else:
        #             vol_ahead = float(max(0, prio_rank))

        #         qlen_val = total_vol

        #     else:
        #         queue_rem = float(info.get("queue_ahead", 0.0))
        #         executed_ahead = float(info.get("executed_ahead", 0.0))

        #         total_vol = 0.0
        #         snap = self._last_env_snapshot
        #         if snap is not None:
        #             if side_i == +1:
        #                 depth = snap.get("full_depth_bids", [])
        #             else:
        #                 depth = snap.get("full_depth_asks", [])
        #             for dp, sz in depth:
        #                 if int(dp) == px:
        #                     total_vol = float(sz)
        #                     break

        #         if total_vol < queue_rem:
        #             total_vol = queue_rem

        #         if total_vol <= 0.0:
        #             total_vol = 1.0

        #         vol_ahead = min(queue_rem, total_vol)
        #         qlen_val = total_vol

        #     own_prices.append(px)
        #     own_sides.append(side_i)
        #     own_prios.append(float(vol_ahead))
        #     own_qlens.append(float(qlen_val))
        
        
        # BEFORE the loop over self.orders
        own_prices, own_sides, own_prios, own_qlens = [], [], [], []
        own_qtys = []  # NEW: store remaining qty per own order (for plotting size segments)
        
        for oid_i, info in list(self.orders.items()):
            px = int(info.get("price", -999))
            side_i = int(info.get("side", 0))
        
            # Remaining qty (backtest may have qty > 1; simulation currently forces qty=1)
            qty_rem = float(info.get("qty", 1.0))
            if not np.isfinite(qty_rem) or qty_rem <= 0.0:
                qty_rem = 1.0
        
            if not self.backtest_mode:
                # --- SIMULATION MODE ---
                # Here the engine book already includes our order volume at px.
                # priority is rank index (FIFO). Convert it to "volume ahead" in a unit-volume engine.
                vol_ahead = 0.0
                qlen_val = 0.0
        
                if self.lob is not None:
                    # total volume at this price from engine state (includes us)
                    try:
                        total_vol = float(abs(self.lob.lob_state[px]))
                    except Exception:
                        total_vol = 0.0
        
                    pr = info.get("priority", 0)
                    try:
                        pr_rank = int(pr)
                    except Exception:
                        pr_rank = 0
        
                    # Convert rank -> approximate volume ahead (unit-volume engine)
                    vol_ahead = float(max(pr_rank, 0))
                    qlen_val = float(max(total_vol, 1.0))
        
                    if vol_ahead > qlen_val:
                        vol_ahead = qlen_val
                else:
                    qlen_val = 1.0
                    vol_ahead = 0.0
        
            else:
                # --- BACKTEST MODE ---
                # queue_ahead tracks *environment volume ahead of us* at this px
                queue_ahead = float(info.get("queue_ahead", 0.0))
                if not np.isfinite(queue_ahead) or queue_ahead < 0.0:
                    queue_ahead = 0.0
        
                # Snapshot depth at this price (environment-only, since we don't inject our orders into snapshots)
                env_depth_here = 0.0
                snap = self._last_env_snapshot
                if snap is not None:
                    depth = snap.get("full_depth_bids", []) if side_i == +1 else snap.get("full_depth_asks", [])
                    for dp, sz in depth:
                        if int(dp) == px:
                            env_depth_here = float(sz)
                            break
        
                if not np.isfinite(env_depth_here) or env_depth_here < 0.0:
                    env_depth_here = 0.0
        
                # TOTAL queue at px should include the environment depth PLUS our remaining qty
                qlen_val = env_depth_here + qty_rem
        
                # Defensive: if queue_ahead is larger than env depth (possible due to timing),
                # ensure total queue is still consistent.
                if qlen_val < queue_ahead + qty_rem:
                    qlen_val = queue_ahead + qty_rem
        
                if qlen_val <= 0.0:
                    qlen_val = max(qty_rem, 1.0)
        
                vol_ahead = min(queue_ahead, qlen_val)
        
            own_prices.append(px)
            own_sides.append(side_i)
            own_prios.append(float(vol_ahead))   # "volume ahead"
            own_qlens.append(float(qlen_val))    # "total queue volume at px" (now includes us in backtest)
            own_qtys.append(float(qty_rem))      # NEW

        row = {
            "i": step_idx,
            "Time": time_val,
            "MM_Action": logged_action,
            "MM_Side": side,
            "MM_Price": logged_price,
            "MM_OrderID": -1 if oid is None else int(oid),
            "MM_Inventory": self.inventory,
            "MM_CashPnL": float(self.cash),
            "MM_UPnL": float(self.mark_to_market()),
            "MM_TotalPnL": self.total_pnl(),
            "MM_BidPos": int(bid_pos) if bid_pos is not None else -1,
            "MM_AskPos": int(ask_pos) if ask_pos is not None else -1,
            "MM_LastOrderPos": -1 if logged_last_pos is None else int(logged_last_pos),
            "MM_OwnPrices": own_prices,
            "MM_OwnSides": own_sides,
            "MM_OwnQtys": own_qtys,
            "MM_OwnPriorities": own_prios,
            "MM_OwnQueueLens": own_qlens,
            "MM_HadFill": bool(had_fill),
            "MM_LastFillSide": int(fill_side),
            "MM_LastFillPrice": int(fill_price),
            "MM_BidDist": float(bid_dist) if bid_dist is not None else float("nan"),
            "MM_AskDist": float(ask_dist) if ask_dist is not None else float("nan"),
            "MM_State_Spread": rl_spread,
            "MM_State_BidSize": rl_bidsize,
            "MM_State_AskSize": rl_asksize,
            "MM_State_Inventory": rl_inventory,
            "MM_State_HasBid": rl_has_bid,
            "MM_State_HasAsk": rl_has_ask,
            "MM_RL_State_Vector": rl_state_vec,
            "MM_RL_State_Mode": rl_state_mode,
            "MM_RL_State_Dim": rl_state_dim,
            "MM_RL_State_MaxOffset": rl_state_max_offset,
            "MM_ActionIdx": int(action_idx),
            "MM_MO_Flow_EWMA": float(self.mo_flow_ewma),
            "MM_MO_Flow_PHat": float(self.get_mo_flow_p_hat()),
            "MM_MO_Flow_Fast_EWMA": float(self.mo_flow_fast_ewma),
            "MM_MO_Flow_Fast_PHat": float(self.get_mo_flow_fast_p_hat()),
            "MM_Fill_Imbalance_EWMA": float(self.fill_imbalance_ewma),
            "MM_Fill_Imbalance_Feature": float(self.get_effective_fill_imbalance_feature()),
            "MM_Quote_Exposure_Imbalance_Enabled": bool(getattr(self, "use_quote_exposure_imbalance_signal", False)),
            "MM_Fill_Bid_EWMA": float(self.fill_bid_ewma),
            "MM_Fill_Ask_EWMA": float(self.fill_ask_ewma),
            "MM_Forced_Cancels": len(getattr(self, "_last_forced_cancels", [])),
            "MM_Reinit_Regen": str(getattr(self.lob, "_last_reinit_regen", "") or ""),
        }

        if bool(getattr(self, "use_quote_exposure_imbalance_signal", False)):
            quote_features = self.get_quote_exposure_imbalance_features()
            row.update({
                "MM_Quote_Exposure_Imbalance": float(quote_features.get("quote_exposure_imbalance", 0.0)),
                "MM_Quote_Exposure_Imbalance_RawRatio": float(quote_features.get("quote_exposure_imbalance_raw_ratio", 0.0)),
                "MM_Quote_Exposure_BidOpportunity": float(quote_features.get("quote_exposure_bid_opportunity", 0.0)),
                "MM_Quote_Exposure_AskOpportunity": float(quote_features.get("quote_exposure_ask_opportunity", 0.0)),
                "MM_Quote_Exposure_TotalOpportunity": float(quote_features.get("quote_exposure_total_opportunity", 0.0)),
            })

        if bool(getattr(self, "use_bayes_flow_signal", False)):
            bayes_tel = self.get_bayes_flow_telemetry()
            row.update({
                "MM_Bayes_Flow_Enabled": True,
                "MM_Bayes_M_Hat": float(bayes_tel.get("bayes_m_hat", 0.0)),
                "MM_Bayes_PHat": float(bayes_tel.get("bayes_p_hat", 0.5)),
                "MM_Bayes_CP_Prob": float(bayes_tel.get("bayes_cp_prob", 0.0)),
                "MM_Bayes_Run_Length": float(bayes_tel.get("bayes_expected_run_length_raw", 0.0)),
                "MM_Bayes_Run_Length_Feature": float(bayes_tel.get("bayes_expected_run_length", 0.0)),
                "MM_Bayes_Uncertainty": float(bayes_tel.get("bayes_uncertainty", 0.0)),
                "MM_Bayes_N_Components": float(bayes_tel.get("bayes_n_components", 0.0)),
                "MM_Bayes_N_MO_Updates": float(bayes_tel.get("bayes_n_mo_updates", 0.0)),
                "MM_Bayes_Hazard": float(bayes_tel.get("bayes_hazard", 0.0)),
                "MM_Bayes_Mixture_Enabled": float(bayes_tel.get("bayes_mixture_enabled", 0.0)),
                "MM_Bayes_Mixture_N_Models": float(bayes_tel.get("bayes_mixture_n_models", 0.0)),
                "MM_Bayes_Mixture_Tau_Mean": float(bayes_tel.get("bayes_mixture_tau_mean", 0.0)),
            })
            for key, value in bayes_tel.items():
                if key.startswith("bayes_mix_w_tau_"):
                    suffix = key[len("bayes_mix_w_tau_"):]
                    row[f"MM_Bayes_MixW_Tau_{suffix}"] = float(value)

        self.log.append(row)

        # Optional internal consistency check (simulation only) — auto-heal, don't raise.
        # Run every _invariant_check_interval steps instead of every step for performance.
        self._step_counter += 1
        if (self._invariant_check_interval > 0
                and not self.backtest_mode
                and self.lob is not None
                and self._step_counter % self._invariant_check_interval == 0):
            self._check_invariants(strict=False)

    # ======================================================================
    # Optional debug helper — internal consistency checks
    # ======================================================================
    def _check_invariants(self, strict: bool = True):
        """
        Consistency check between:
        - self.orders
        - self.owner
        - engine dimensions (lob_state / priorities_lob_state)
        """
        if self.backtest_mode or self.lob is None:
            return

        self._prune_orphan_orders()

        n_ranks, n_prices = self.owner.shape
        errors: List[str] = []

        for oid, info in list(self.orders.items()):
            p = info["price"]
            pr = info.get("priority", None)

            if pr is None:
                continue

            if not (0 <= p < n_prices):
                msg = f"[MM invariant] Order {oid} price out of range: {p}"
                errors.append(msg)
                if not strict:
                    del self.orders[oid]
                continue

            if not (0 <= pr < n_ranks):
                msg = f"[MM invariant] Order {oid} priority out of range: {pr}"
                errors.append(msg)
                if not strict:
                    del self.orders[oid]
                continue

            if not self.owner[pr, p]:
                msg = (
                    f"[MM invariant] Order {oid} has (price={p}, priority={pr}) "
                    f"but owner[{pr}, {p}] is False"
                )
                errors.append(msg)
                if not strict:
                    del self.orders[oid]

        for p in range(n_prices):
            rows = np.where(self.owner[:, p])[0]
            for r in rows:
                has_order = any(
                    (info["price"] == p and info.get("priority", None) == int(r))
                    for info in self.orders.values()
                )
                if not has_order:
                    msg = (
                        f"[MM invariant] Owner mark at (rank={r}, price={p}) "
                        f"but no matching order in self.orders"
                    )
                    errors.append(msg)
                    if not strict:
                        self.owner[r, p] = False

        if errors:
            if strict:
                print("=== INVARIANT FAILURE DEBUG ===")
                print(f"  n_orders = {len(self.orders)}")
                print(f"  owner.shape = {self.owner.shape}")
                print(f"  last_fill = {self.last_fill}")
                print(f"  last_policy_action = {self.last_policy_action}")
                for e in errors:
                    print("  ", e)
                raise RuntimeError(" | ".join(errors))
            else:
                print("[MM invariant] Auto-healed inconsistencies detected:")
                for e in errors:
                    print("  ", e)
                self._prune_orphan_orders()


# ============================================================
# Runner — LOB_simulation + external MM agent + RL controller
# ============================================================

def _add_offset_cellwise(series, offsets):
    # >>> CRITICAL: pandas 3.x pode retornar array read-only (Copy-on-Write)
    vals = series.to_numpy(dtype=object, copy=True)   # <- garante writeable
    offsets = np.asarray(offsets)

    for i, v in enumerate(vals):
        off = float(offsets[i])
        if isinstance(v, (list, tuple, np.ndarray)):
            vals[i] = [float(x) + off for x in v]
        else:
            vals[i] = float(v) + off

    return pd.Series(vals, index=series.index, name=series.name)


def _add_offset_fast(series: pd.Series, offsets: np.ndarray) -> pd.Series:
    # Fast path for numeric columns (float/int)
    return pd.to_numeric(series, errors="coerce") + offsets

def _add_offset_preserving_sentinel(series: pd.Series, offsets: np.ndarray,
                                    sentinel: float = -1.0) -> pd.Series:
    """Add offsets but leave sentinel values (default -1) unchanged."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).copy()
    # FIX: was using exact float ==, which can miss values like -0.9999999999
    # due to floating-point arithmetic.  np.isclose is robust.
    mask = np.isclose(vals, sentinel, atol=1e-9)
    vals += offsets
    vals[mask] = sentinel
    return pd.Series(vals, index=series.index, name=series.name)


def make_all_prices_absolute(msg_df: pd.DataFrame,
                             ob_df: Optional[pd.DataFrame],
                             mm_df: Optional[pd.DataFrame],
                             p0: float = 0.0):
    """
    Converts everything to absolute prices using the save_results convention:
      - 'state' columns (post-shift) add offset_now[i]
      - execution prices (pre-shift) add offset_pre[i] = offset_now[i-1]

    Fixes applied:
    - Truncation: msg_df length is canonical; ob_df/mm_df are clipped only if longer.
    - Sentinel: price index columns with -1 sentinel are preserved (not offset).
    """
    if msg_df is None or len(msg_df) == 0 or "Shift" not in msg_df.columns:
        return msg_df, ob_df, mm_df

    inc = msg_df["Shift"].cumsum().to_numpy(dtype=float)
    offset_now = inc + float(p0)               # post-shift for row i
    offset_pre = np.empty_like(offset_now)     # pre-shift for row i
    offset_pre[0] = float(p0)
    offset_pre[1:] = offset_now[:-1]

    # ---- align lengths: msg_df is canonical; clip others if they are longer ----
    L = len(msg_df)
    msg_df = msg_df.copy()

    if ob_df is not None and len(ob_df) > L:
        ob_df = ob_df.iloc[:L].copy()
    elif ob_df is not None:
        ob_df = ob_df.copy()

    if mm_df is not None and len(mm_df) > L:
        mm_df = mm_df.iloc[:L].copy()
    elif mm_df is not None:
        mm_df = mm_df.copy()

    # ---------------- msg_df ----------------
    # Continuous prices (no sentinel): MidPrice
    for col in ("MidPrice",):
        if col in msg_df.columns:
            msg_df[col] = _add_offset_fast(msg_df[col], offset_now)

    # Index-based prices that use -1 sentinel: preserve sentinel
    for col in ("IndBestBid", "IndBestAsk", "BestBidPrice", "BestAskPrice"):
        if col in msg_df.columns:
            msg_df[col] = _add_offset_preserving_sentinel(msg_df[col], offset_now)

    # Execution in PRE-shift coordinates (if column exists)
    if "Price" in msg_df.columns:
        msg_df["Price"] = _add_offset_preserving_sentinel(msg_df["Price"], offset_pre)

    # ---------------- ob_df (post-shift) ----------------
    if ob_df is not None and len(ob_df) > 0:
        ob_off = offset_now[:len(ob_df)]
        for c in [c for c in ob_df.columns if "Price" in c]:
            # Add offset to ALL prices (including index 0, which is a valid price).
            # Then re-zero where size==0 to mark empty book levels.
            ob_df[c] = _add_offset_fast(ob_df[c], ob_off)
            size_col = c.replace("Price", "Size")
            if size_col in ob_df.columns:
                zero_mask = pd.to_numeric(ob_df[size_col], errors="coerce").fillna(0.0) == 0.0
                ob_df.loc[zero_mask, c] = 0.0

    # ---------------- mm_df (mixed: numeric + list columns) ----------------
    if mm_df is not None and len(mm_df) > 0:
        # With split_sweeps, msg_df may have multiple rows per simulation step
        # while mm_df has exactly 1 row per step. We must align offsets to the
        # LAST msg_df row of each step (where the shift is recorded).
        if len(mm_df) < L and "Time" in msg_df.columns:
            time_vals = msg_df["Time"].values
            # Boolean mask: True at the last msg_df row of each step-group
            is_step_end = np.append(time_vals[:-1] != time_vals[1:], True)
            mm_off_now = offset_now[is_step_end][:len(mm_df)]
            mm_off_pre = offset_pre[is_step_end][:len(mm_df)]
        else:
            # No split_sweeps or 1:1 alignment — original logic is correct
            mm_off_now = offset_now[:len(mm_df)]
            mm_off_pre = offset_pre[:len(mm_df)]

        # 1) Prices/positions on the current grid (post-shift)
        post_shift_mm = [
            "MM_BestBidPrice","MM_BestAskPrice","MM_MidPrice",
            "MM_Price",       # actual column name logged by log_step
            "MM_OwnPrices",   # lista
        ]

        # 2) Execution/fill in pre-shift (prices recorded BEFORE the shift is applied)
        pre_shift_mm = [
            "MM_TradePrice","MM_ExecPrice",
        ]

        # MM_LastFillPrice is stored as px_post (post-shift coordinate) → treat as post-shift
        post_shift_mm.append("MM_LastFillPrice")

        # DO NOT shift: spreads/dists/pnl/inv/state_vector etc.
        dont_touch = {
            "MM_BidDist","MM_AskDist","MM_State_Spread",
            "MM_CashPnL","MM_UPnL","MM_TotalPnL","MM_Inventory",
            "MM_RL_State_Vector",
        }

        for col in post_shift_mm:
            if col in mm_df.columns and col not in dont_touch:
                if mm_df[col].dtype == "O":
                    mm_df[col] = _add_offset_cellwise(mm_df[col].astype(object), mm_off_now)
                else:
                    mm_df[col] = _add_offset_preserving_sentinel(mm_df[col], mm_off_now)

        for col in pre_shift_mm:
            if col in mm_df.columns and col not in dont_touch:
                if mm_df[col].dtype == "O":
                    mm_df[col] = _add_offset_cellwise(mm_df[col].astype(object), mm_off_pre)
                else:
                    mm_df[col] = _add_offset_preserving_sentinel(mm_df[col], mm_off_pre)

    return msg_df, ob_df, mm_df


# ==============================================================================
# RUNNER: LOB_simulation + External MarketMaker + RL Controller
# ==============================================================================

def simulate_LOB_with_MM(
    # --- Simulation Parameters ---
    lam: float,
    mu: float,
    delta: float,
    random_seed: Optional[int],
    number_tick_levels: int,
    n_priority_ranks: int,
    number_levels_to_store: int = 20,
    p0: int = 100,
    mean_size_LO: float = 1,
    mean_size_MO: float = 1,
    make_absolute_prices: bool = True,
    iterations: int = 2_000,
    iterations_to_equilibrium: int = 10_000,
    path_save_files: Optional[str] = None,
    label_simulation: Optional[str] = None,

    # --- RL / Strategy Parameters ---
    beta_exp_weighted_return: float = 0.0,
    intensity_exp_weighted_return: float = 0.0,
    controller: "Optional[RLController]" = None,
    mm_policy=None,  # Optional[Callable[[Dict[str, Any]], tuple]]

    # --- Market Maker Configuration ---
    exclude_self_from_state: bool = False,
    # [NEW] Controls if the agent sees "Sticky" prices (its own price) or "Transparent" prices (env only)
    use_sticky_price: bool = True,

    reward_fn=None,
    tick_size: float = 0.01,
    base_price_idx: Optional[float] = 0.0,

    # --- Pure Market Making (Analytical) Flags ---
    pure_MM: bool = False,
    pure_mm_offsets=None,  # Optional[List[Tuple[int, int]]]
    max_offset_override: Optional[int] = None,

    # --- Execution / Engine Config ---
    split_sweeps: bool = False,

    # --- MO flow asymmetry ---
    buy_mo_prob: Union[float, Callable] = 0.5,

    # --- Debug ---
    debug_integrity: bool = False,

    # --- LOB Engine Selection ---
    # Pass a dict of QRM parameters to use the QRM engine instead of Santa Fe.
    # Example: qrm_params=dict(intens_k1=intens_k1, size_q=5, aes=25.0, ...)
    # If None (default), uses the Santa Fe constant-rate engine.
    qrm_params: Optional[dict] = None,

    # --- EWMA smoothing factor overrides ---
    # ewma_alpha: sets BOTH mo_flow and fill_imbalance (backward compatible).
    # mo_flow_ewma_alpha: overrides ONLY slow mo_flow (applied after ewma_alpha).
    # mo_flow_fast_ewma_alpha: overrides ONLY fast mo_flow (applied after ewma_alpha).
    # fill_imbalance_alpha: overrides ONLY fill_imbalance (applied after ewma_alpha).
    ewma_alpha: Optional[float] = None,
    mo_flow_ewma_alpha: Optional[float] = None,
    mo_flow_fast_ewma_alpha: Optional[float] = None,
    fill_imbalance_alpha: Optional[float] = None,
    quote_exposure_imbalance_signal: bool = False,
    bayes_flow_enabled: bool = False,
    bayes_flow_tau_r: float = 60.0,
    bayes_flow_p_low: float = 0.2,
    bayes_flow_p_high: float = 0.8,
    bayes_flow_prior_a: float = 1.0,
    bayes_flow_prior_b: float = 1.0,
    bayes_flow_prior_mode: str = "truncated_beta_grid",
    bayes_flow_hazard: Optional[float] = None,
    bayes_flow_max_run_length: Optional[int] = None,
    bayes_flow_cp_window: int = 5,
    bayes_flow_use_mixture: bool = False,
    bayes_flow_mixture_taus: Optional[List[float]] = None,
    bayes_flow_mixture_weights: Optional[List[float]] = None,

    # --- Lightweight evaluation output ---
    # When True, run the exact same simulator/controller dynamics but avoid
    # materialising the heavy message/orderbook/MM telemetry DataFrames.
    metrics_only: bool = False,
    metrics_trace_max_points: Optional[int] = None,
):
    """
    High-level simulation runner instrumented for DEBUGGING CROSSED MARKETS.
    """

    rng = (
        np.random.RandomState(int(random_seed))
        if random_seed is not None
        else np.random.RandomState()
    )

    # 1. Initialize LOB (Santa Fe or QRM depending on qrm_params)
    lob = _create_lob_engine(
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,
        p0=p0,
        mean_size_LO=mean_size_LO,
        number_levels_to_store=number_levels_to_store,
        beta_exp_weighted_return=beta_exp_weighted_return,
        intensity_exp_weighted_return=intensity_exp_weighted_return,
        mean_size_MO=mean_size_MO,
        rng=rng,
        buy_mo_prob=buy_mo_prob,
        qrm_params=qrm_params,
    )
    lob.initialize()

    # Initialize Tape Columns
    if "Size" not in lob.message_dict: lob.message_dict["Size"] = []
    for _k in ("CancelVolAhead", "CancelQueueLen", "CancelRank", "BestBidPrice", "BestAskPrice", "TotNumberOrders"):
        if _k not in lob.message_dict: lob.message_dict[_k] = []

    if base_price_idx is None: base_price_idx = 0.0

    # 2. Warmup
    for _ in range(iterations_to_equilibrium):
        lob.simulate_order(lam, mu, delta)
    lob.exp_weighted_return_to_store = 0.0

    # Reset the MO event counter so that regime/adversarial schedules
    # start from step 0 at the beginning of the actual episode, not
    # offset by however many MOs occurred during warmup.
    if hasattr(lob, '_event_step'):
        lob._event_step = 0
    # Also reset the schedule's internal state (adversarial transitions,
    # reward accumulator, etc.) that may have been polluted by warmup MOs.
    if callable(buy_mo_prob) and hasattr(buy_mo_prob, 'reset'):
        buy_mo_prob.reset()

    # 3. Setup Agent
    effective_policy = None
    if controller is not None:
        effective_policy = controller.act
    elif mm_policy is not None:
        effective_policy = mm_policy

    if controller is not None and getattr(controller, "pure_mm", False):
        pure_MM = True
        if pure_mm_offsets is None and hasattr(controller, "pure_mm_offsets"):
            pure_mm_offsets = controller.pure_mm_offsets
        if max_offset_override is None and hasattr(controller, "max_offset"):
            max_offset_override = controller.max_offset

    mm = MarketMaker(
        lob=lob,
        policy=effective_policy,
        exclude_self_from_state=exclude_self_from_state,
        use_sticky_price=use_sticky_price, 
        tick_size=tick_size,
        backtest_mode=False,
        use_queue_ahead=False,
        base_price_idx=base_price_idx,
        pure_MM=pure_MM,
        pure_mm_offsets=pure_mm_offsets,
        max_offset_override=max_offset_override,
        rng=rng,
    )

    # Override EWMA smoothing factors if provided.
    #
    # Layered override system (applied in order of specificity):
    #   1. ewma_alpha               — sets slow MO, fast MO, and fill imbalance
    #                                 together (backward-compatible umbrella knob)
    #   2. mo_flow_ewma_alpha       — overrides only the slow MO-flow signal
    #   3. mo_flow_fast_ewma_alpha  — overrides only the fast MO-flow signal
    #   4. fill_imbalance_alpha     — overrides only the fill-imbalance signal
    #
    # In adversary-tau mode with alpha_recalc, the runner passes per-episode
    # overrides for both signals. mo_flow_ewma_alpha is calibrated on the
    # MO clock (1 - exp(-1/tau)), while fill_imbalance_alpha is PRE-RESCALED
    # by the runner to the all-event clock before it is passed here.
    if ewma_alpha is not None:
        mm.mo_flow_ewma_alpha = ewma_alpha
        mm.mo_flow_fast_ewma_alpha = ewma_alpha
        mm.fill_imbalance_alpha = ewma_alpha
    if mo_flow_ewma_alpha is not None:
        mm.mo_flow_ewma_alpha = mo_flow_ewma_alpha
    if mo_flow_fast_ewma_alpha is not None:
        mm.mo_flow_fast_ewma_alpha = mo_flow_fast_ewma_alpha
    if fill_imbalance_alpha is not None:
        mm.fill_imbalance_alpha = fill_imbalance_alpha
    mm.mo_flow_q_scale = max(float(mean_size_MO), 1.0)
    mm.fill_imbalance_q_scale = max(float(mean_size_MO), 1.0)
    mm.reset_fill_imbalance_prior(prior_mult=2.0)
    mm.configure_quote_exposure_imbalance_signal(
        enabled=bool(quote_exposure_imbalance_signal),
    )
    mm.configure_bayes_flow_signal(
        enabled=bool(bayes_flow_enabled),
        tau_r=float(bayes_flow_tau_r),
        p_low=float(bayes_flow_p_low),
        p_high=float(bayes_flow_p_high),
        prior_a=float(bayes_flow_prior_a),
        prior_b=float(bayes_flow_prior_b),
        prior_mode=str(bayes_flow_prior_mode),
        hazard=bayes_flow_hazard,
        max_run_length=bayes_flow_max_run_length,
        cp_window=int(bayes_flow_cp_window),
        use_mixture=bool(bayes_flow_use_mixture),
        mixture_taus=bayes_flow_mixture_taus,
        mixture_weights=bayes_flow_mixture_weights,
    )

    # Wire max_inventory from the controller's inv_limit so that reward
    # functions normalize inventory correctly (fixes fallback=10.0 bug).
    if controller is not None and hasattr(controller, "inv_limit") and controller.inv_limit is not None:
        mm.max_inventory = float(controller.inv_limit)

    # NOTE: cancel_avoid_mask_provider + cancel_filter are already attached to the
    # engine inside MarketMaker.__init__; no need to re-attach here.

    t_i = 0.0
    row_idx = 0

    # Lightweight metrics collected online when metrics_only=True.  This avoids
    # building per-step DataFrames for long Monte Carlo evaluations.
    _metrics_n = 0
    _metrics_abs_inv_sum = 0.0
    _metrics_limit_count = 0
    _metrics_steps: List[int] = []
    _metrics_pnl_trace: List[float] = []
    _metrics_inv_trace: List[float] = []
    if metrics_trace_max_points is None:
        _metrics_trace_stride = 0
    else:
        _trace_max = max(1, int(metrics_trace_max_points))
        _metrics_trace_stride = max(1, int(np.ceil(float(iterations) / float(_trace_max))))

    # ==============================================================================
    # DEBUG HELPER: Check Market Integrity
    # ==============================================================================
    def verify_lob_integrity(step, stage, extra_info=""):
        bid_idx = np.where(lob.lob_state > 0)[0]
        ask_idx = np.where(lob.lob_state < 0)[0]
        
        bb = int(bid_idx[-1]) if bid_idx.size else -1
        ba = int(ask_idx[0]) if ask_idx.size else -1
        
        # Se existem Bid e Ask, e Bid >= Ask, o mercado cruzou!
        if (bb >= 0 and ba >= 0 and bb >= ba):
            print("\n" + "#"*80)
            print(f"[CRITICAL DEBUG TRAP] MARKET CROSSED at Step {step}")
            print(f"Stage: {stage}")
            print(f"LOB State: BestBid={bb} >= BestAsk={ba} (Spread={ba-bb})")
            print(f"Extra Info: {extra_info}")
            print("-" * 40)
            print("MM Internal View (before crash):")
            _mm_bids = [info["price"] for info in mm.orders.values() if info.get("side") == +1]
            _mm_asks = [info["price"] for info in mm.orders.values() if info.get("side") == -1]
            print(f"  My Best Bid: {max(_mm_bids) if _mm_bids else 'none'}")
            print(f"  My Best Ask: {min(_mm_asks) if _mm_asks else 'none'}")
            print(f"  Inventory: {mm.inventory}")
            print("#"*80 + "\n")

            # Emit warning instead of crashing. For strict debug mode,
            # use: warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.warn(
                f"Market Crossed ({stage}): Bid {bb} >= Ask {ba} at step {step}",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # ==============================================================================

    # 4. Main Loop
    for i in range(iterations):
        
        if debug_integrity: verify_lob_integrity(i, "START_OF_LOOP")

        # A+B) Agent Decision (build_state is called inside pre_step; reuse last_policy_state)
        action, a_side, a_price, a_oid, a_pos = mm.pre_step()
        # When policy is None, pre_step returns early without calling build_state,
        # so last_policy_state would be stale from a previous iteration.
        # Always rebuild in that case so reward_fn/controller see current book state.
        if effective_policy is None or mm.last_policy_state is None:
            mm.last_policy_state = mm.build_state()
        state_before = mm.last_policy_state

        # Snapshot for cancel tracking — AFTER MM action, BEFORE engine.
        # If MM places an order at the same price the engine cancels in the
        # same step, a pre-MM snapshot causes the sanity check (q_pre - 1 ==
        # q_post) to fail, silently skipping cancel bookkeeping.
        pre_cols_for_cancel = lob.priorities_lob_state.copy()

        if debug_integrity: verify_lob_integrity(i, "AFTER_MM_ACTION", f"Action={action}, Side={a_side}, Price={a_price}")

        # D) Environment Step
        sim_out = lob.simulate_order(
            lam=lam, mu=mu, delta=delta,
            split_sweeps=split_sweeps,
        )

        if debug_integrity: verify_lob_integrity(i, "AFTER_SIMULATOR", "Engine generated an event")

        # Advance Clock
        # The LOB engine (LOB_SIM_SANTA_FE) always sets last_event_dt
        # from its internal exponential clocks (Lam, Mu, Delta).
        dt_event = float(lob.last_event_dt)
        t_i += dt_event

        # E) Normalize Engine Output
        # simulate_order always returns (rows, metrics, snaps) since the
        # return_cancel_rank parameter was removed (it was dead code).
        if isinstance(sim_out, tuple) and len(sim_out) == 3:
            rows, metrics, snaps = sim_out
        else:
            rows, metrics, snaps = [sim_out], [None], [None]

        # Extract Parent Event Summary
        order_type = int(rows[0][0])
        order_direction = int(rows[0][1])
        order_price = int(rows[0][2])
        shift = int(rows[-1][3]) 
        order_size = int(sum(int(r[4]) for r in rows))

        # F) Write to Tape. In metrics_only mode the event dynamics are
        # unchanged, but we skip the heavy output logs.
        if not metrics_only:
            for j, row in enumerate(rows):
                ot, od, op, sh, sz = row
                lob.message_dict["Time"].append(t_i)
                lob.message_dict["Type"].append(int(ot))
                lob.message_dict["Direction"].append(int(od))
                lob.message_dict["Price"].append(int(op))
                lob.message_dict["Shift"].append(int(sh))
                lob.message_dict["Size"].append(float(sz))

                m = metrics[j]
                if isinstance(m, dict):
                    # ... (mantendo logica original de logging) ...
                    lob.message_dict["Spread"].append(float(m.get("Spread", np.nan)))
                    lob.message_dict["MidPrice"].append(float(m.get("MidPrice", np.nan)))
                    lob.message_dict["Return"].append(float(m.get("Return", np.nan)))
                    lob.message_dict["TotNumberOrders"].append(float(m.get("TotNumberOrders", np.abs(lob.lob_state).sum())))
                    # Best Bid/Ask handling
                    bb = m.get("IndBestBid", -1)
                    ba = m.get("IndBestAsk", -1)
                    lob.message_dict["IndBestBid"].append(int(bb) if (bb == bb) else -1)
                    lob.message_dict["IndBestAsk"].append(int(ba) if (ba == ba) else -1)
                    bbp = m.get("BestBidPrice", bb)
                    bap = m.get("BestAskPrice", ba)
                    lob.message_dict["BestBidPrice"].append(float(bbp) if (bbp == bbp and bbp >= 0) else np.nan)
                    lob.message_dict["BestAskPrice"].append(float(bap) if (bap == bap and bap >= 0) else np.nan)
                    # Cancels
                    lob.message_dict["CancelVolAhead"].append(float(m.get("CancelVolAhead", np.nan)))
                    lob.message_dict["CancelQueueLen"].append(float(m.get("CancelQueueLen", np.nan)))
                    lob.message_dict["CancelRank"].append(float(m.get("CancelRank", np.nan)))
                    
                    # Others
                    lob.message_dict["TotNumberBidOrders"].append(float(m.get("TotNumberBidOrders", np.nan)))
                    lob.message_dict["TotNumberAskOrders"].append(float(m.get("TotNumberAskOrders", np.nan)))
                else:
                    # Fallback logging
                    lob.message_dict["Spread"].append(lob.compute_spread())
                    lob.message_dict["MidPrice"].append(lob.compute_mid_price())
                    lob.message_dict["Return"].append(lob.mid_price_to_store["Current"] - lob.mid_price_to_store["Previous"])
                    lob.message_dict["TotNumberOrders"].append(float(np.abs(lob.lob_state).sum()))
                    
                    bid_idxs = np.where(lob.lob_state > 0)[0]
                    ask_idxs = np.where(lob.lob_state < 0)[0]
                    lob.message_dict["IndBestBid"].append(int(bid_idxs[-1]) if bid_idxs.size > 0 else -1)
                    lob.message_dict["IndBestAsk"].append(int(ask_idxs[0]) if ask_idxs.size > 0 else -1)
                    lob.message_dict["BestBidPrice"].append(float(bid_idxs[-1]) if bid_idxs.size > 0 else np.nan)
                    lob.message_dict["BestAskPrice"].append(float(ask_idxs[0]) if ask_idxs.size > 0 else np.nan)
                    
                    lob.message_dict["CancelVolAhead"].append(np.nan)
                    lob.message_dict["CancelQueueLen"].append(np.nan)
                    lob.message_dict["CancelRank"].append(np.nan)
                    lob.message_dict["TotNumberBidOrders"].append(np.nan)
                    lob.message_dict["TotNumberAskOrders"].append(np.nan)

                snap = snaps[j]
                if isinstance(snap, dict):
                    for _k in lob.ob_dict.keys():
                        lob.ob_dict[_k].append(snap.get(_k, 0))
                else:
                    lob.update_ob_dict(row_idx)
                row_idx += 1
        else:
            # build_state() and time-gated controllers read the latest event
            # time from lob.message_dict["Time"].  Keep this tiny tape so the
            # controller clock is identical without storing full telemetry.
            for _ in rows:
                lob.message_dict["Time"].append(t_i)

        # G) MM Bookkeeping: Handle Environment Cancellations
        # Use pre_cols_for_cancel (post-MM, pre-engine) so that orders placed
        # by MM in this same step are visible when computing the queue shift.
        pre_col = None
        if 0 <= order_price < pre_cols_for_cancel.shape[1]:
            pre_col = pre_cols_for_cancel[:, order_price].copy()
        
        post_idx = order_price - shift
        post_col = None
        if 0 <= post_idx < lob.priorities_lob_state.shape[1]:
            post_col = lob.priorities_lob_state[:, post_idx].copy()

        if order_type == 2 and (pre_col is not None) and (post_col is not None):
            cancel_rank = None
            try:
                m0 = metrics[0] if (isinstance(metrics, list) and len(metrics) > 0) else None
                if isinstance(m0, dict):
                    cr = m0.get("CancelRank", np.nan)
                    if cr == cr: cancel_rank = int(cr)
            except Exception:
                cancel_rank = None
            
            mm.adjust_for_cancel_at_price(order_price, pre_col, post_col, cancel_rank=cancel_rank)

        # H) MM Bookkeeping: Handle Environment Market Orders (Fills)
        executed_by_price_pre_shift = None
        if order_type == 1:
            # Update MO flow EWMA using signed, size-aware MO flow.
            mm.update_mo_flow_ewma(
                order_direction=order_direction,
                order_size=float(order_size),
            )
            mm.update_mo_flow_bayes(
                order_direction=order_direction,
                order_size=float(order_size),
            )

            executed_by_price_pre_shift = {}

            # Preferred source: explicit execution-by-price map produced by engine.
            for m in metrics:
                if not isinstance(m, dict):
                    continue
                emap = m.get("ExecutedByPricePreShift", None)
                if not isinstance(emap, dict):
                    continue
                for px_raw, k_raw in emap.items():
                    try:
                        px_i = int(px_raw)
                        k_i = int(k_raw)
                    except Exception:
                        continue
                    if k_i <= 0:
                        continue
                    executed_by_price_pre_shift[px_i] = (
                        executed_by_price_pre_shift.get(px_i, 0) + k_i
                    )

            # Backward compatibility: if engine did not expose the map, derive from split rows.
            if (not executed_by_price_pre_shift) and split_sweeps:
                for _ot, _od, _op, _sh, _sz in rows:
                    if int(_ot) != 1:
                        continue
                    k = int(_sz)
                    if k > 0:
                        px = int(_op)
                        executed_by_price_pre_shift[px] = executed_by_price_pre_shift.get(px, 0) + k
            
            if not executed_by_price_pre_shift and int(order_size) > 0:
                # Best-effort fallback: for non-split MOs order_price == first_exec_price.
                # Multi-level sweeps without split_sweeps cannot be reconstructed per-price.
                executed_by_price_pre_shift = {int(order_price): int(order_size)}

        filled = mm.post_step_detect_fill_by_env_mo(
            order_type=order_type,
            order_sign=order_direction,
            order_price_pre_shift=order_price,
            step_idx=i,
            order_size=int(order_size),
            executed_by_price_pre_shift=executed_by_price_pre_shift,
        )

        # I) Apply Grid Shift to MM
        base_price_idx += shift
        mm.base_price_idx = base_price_idx
        mm.apply_shift(shift)

        env_fills = filled.get("fills_list", []) if isinstance(filled, dict) else []
        if (not env_fills) and isinstance(filled, dict) and float(filled.get("qty", 0.0)) > 0.0:
            env_fills = [filled]

        # J) Record MM Fills
        if isinstance(filled, dict) and float(filled.get("qty", 0.0)) > 0.0:
            px_pre = int(filled.get("px", int(order_price)))
            px_post = int(px_pre) - int(shift)
            try:
                n_levels = int(mm.owner.shape[1])
                px_post = max(0, min(px_post, n_levels - 1))
            except Exception: pass
            
            mm.last_fill = {
                "side": int(filled.get("side", 0)),
                "price": int(px_post),
                "time": float(t_i),
                "step": int(i),
                "qty": float(filled.get("qty", 0.0))
            }

        # Update fill imbalance on every environment step. When there is no
        # fill, the zero signal makes the EWMA decay back toward 0 instead of
        # freezing at the last signed value between MOs.
        mm.update_fill_imbalance_ewma_from_fills(env_fills)

        # K) Finalize Step — build state_after only when reward_fn / controller need it
        state_after = mm.build_state() if (reward_fn is not None or controller is not None) else {}
        _aidx_raw = getattr(controller, "last_action_idx", None) if controller is not None else None
        _aidx = int(_aidx_raw) if _aidx_raw is not None else -1
        # Always call log_step: besides telemetry, it performs small MM-side
        # bookkeeping used by controllers. In metrics_only mode we clear the
        # accumulated row after learn() so memory stays bounded.
        mm.log_step(i, t_i, action, a_side, a_price, a_oid, a_pos, action_idx=_aidx)

        done = (i == iterations - 1)
        info: dict = {
            "mm_action": action, "mm_action_side": a_side, "mm_action_price": a_price,
            "mm_action_oid": a_oid, "mm_action_pos": a_pos,
            "env_order_type": order_type, "env_order_sign": order_direction,
            "env_order_price": order_price, "env_order_size": float(order_size),
            "env_shift": shift, "env_filled": filled, "env_fills": env_fills,
            "env_forced_cancels": list(getattr(mm, "_last_forced_cancels", [])),
            "env_reinit_regen": getattr(lob, "_last_reinit_regen", None),
            "done": done,
        }

        if reward_fn is not None:
            reward = float(reward_fn(i, mm, lob, state_before, state_after, info))
        else:
            reward = 0.0
            
        if mm.log:
            mm.log[-1]["MM_Reward"] = reward

        # Hook for adversarial training: notify callable of step reward.
        # We pass is_mo so the adversary can count in MO-event time
        # (matching the regime boundary clock).
        if callable(buy_mo_prob) and hasattr(buy_mo_prob, 'on_step_reward'):
            buy_mo_prob.on_step_reward(reward, mm, is_mo=(order_type == 1))

        if controller is not None:
            controller.learn(i, mm, lob, state_before, state_after, reward, info)

        if metrics_only:
            _inv_now = float(mm.inventory)
            _pnl_now = float(mm.total_pnl())
            _limit_raw = getattr(mm, "max_inventory", None)
            _limit = float(_limit_raw) if _limit_raw else float("inf")
            _metrics_n += 1
            _metrics_abs_inv_sum += abs(_inv_now)
            if np.isfinite(_limit) and abs(_inv_now) >= _limit:
                _metrics_limit_count += 1
            if _metrics_trace_stride and (i % _metrics_trace_stride == 0 or done):
                _metrics_steps.append(int(i))
                _metrics_pnl_trace.append(_pnl_now)
                _metrics_inv_trace.append(_inv_now)
            mm.log.clear()

    # 5. Final Output Generation
    if metrics_only:
        _terminal_pnl = float(mm.total_pnl())
        _terminal_inv = float(mm.inventory)
        if _metrics_trace_stride and (
            not _metrics_steps or _metrics_steps[-1] != int(iterations - 1)
        ):
            _metrics_steps.append(int(iterations - 1))
            _metrics_pnl_trace.append(_terminal_pnl)
            _metrics_inv_trace.append(_terminal_inv)

        metrics = {
            "MM_TotalPnL": _terminal_pnl,
            "MM_Inventory": _terminal_inv,
            "mean_abs_inventory": (
                float(_metrics_abs_inv_sum / _metrics_n) if _metrics_n > 0 else float("nan")
            ),
            "limit_frac": (
                float(_metrics_limit_count / _metrics_n) if _metrics_n > 0 else float("nan")
            ),
            "n_steps": int(_metrics_n),
            "trace_step": np.asarray(_metrics_steps, dtype=np.int64),
            "trace_pnl": np.asarray(_metrics_pnl_trace, dtype=np.float32),
            "trace_inventory": np.asarray(_metrics_inv_trace, dtype=np.float32),
        }
        return None, None, metrics

    if path_save_files is not None and label_simulation is not None:
        # row_idx is the actual number of rows in the tape (can be > iterations
        # when split_sweeps=True generates multiple rows per event).
        lob.save_results(path_save_files, label_simulation, max(row_idx - 1, 0))

    _align_dict_of_lists(lob.message_dict, row_idx, fill_value=np.nan)
    _align_dict_of_lists(lob.ob_dict, row_idx, fill_value=0)

    msg_df = pd.DataFrame(lob.message_dict)
    ob_df = pd.DataFrame(lob.ob_dict)
    mm_df = mm.to_dataframe()

    if "MM_Reward" in mm_df.columns:
        mm_df["MM_CumReward"] = mm_df["MM_Reward"].cumsum()

    if make_absolute_prices and ("Shift" in msg_df.columns):
        msg_df, ob_df, mm_df = make_all_prices_absolute(msg_df, ob_df, mm_df, p0=p0)

    return msg_df, ob_df, mm_df


# ==============================================================================
# GENERATOR VERSION: simulate_LOB_with_MM_generator
# ==============================================================================
#
# This is the generator counterpart of ``simulate_LOB_with_MM``.  It runs the
# exact same simulation logic (LOB initialization, warmup, agent action
# selection, environment stepping, MM bookkeeping, tape recording) but instead
# of calling ``controller.learn()`` internally, it **yields** the step data to
# an external training loop.
#
# This decoupling is essential for synchronous A2C training, where N parallel
# environments must be stepped in lockstep and their gradients aggregated
# before any network update.
#
# See the function docstring below for full protocol specification and usage
# examples.
# ==============================================================================

def simulate_LOB_with_MM_generator(
    # --- Simulation Parameters ---
    lam: float,
    mu: float,
    delta: float,
    random_seed: Optional[int],
    number_tick_levels: int,
    n_priority_ranks: int,
    number_levels_to_store: int = 20,
    p0: int = 100,
    mean_size_LO: float = 1,
    mean_size_MO: float = 1,
    make_absolute_prices: bool = True,
    iterations: int = 2_000,
    iterations_to_equilibrium: int = 10_000,
    path_save_files: Optional[str] = None,
    label_simulation: Optional[str] = None,

    # --- RL / Strategy Parameters ---
    beta_exp_weighted_return: float = 0.0,
    intensity_exp_weighted_return: float = 0.0,
    controller: "Optional[RLController]" = None,
    mm_policy=None,  # Optional[Callable[[Dict[str, Any]], tuple]]

    # --- Market Maker Configuration ---
    exclude_self_from_state: bool = False,
    use_sticky_price: bool = True,

    reward_fn=None,
    tick_size: float = 0.01,
    base_price_idx: Optional[float] = 0.0,

    # --- Pure Market Making (Analytical) Flags ---
    pure_MM: bool = False,
    pure_mm_offsets=None,  # Optional[List[Tuple[int, int]]]
    max_offset_override: Optional[int] = None,

    # --- Execution / Engine Config ---
    split_sweeps: bool = False,

    # --- MO flow asymmetry ---
    buy_mo_prob: Union[float, Callable] = 0.5,

    # --- Debug ---
    debug_integrity: bool = False,

    # --- LOB Engine Selection ---
    qrm_params: Optional[dict] = None,

    # --- EWMA smoothing factor overrides ---
    ewma_alpha: Optional[float] = None,
    mo_flow_ewma_alpha: Optional[float] = None,
    mo_flow_fast_ewma_alpha: Optional[float] = None,
    fill_imbalance_alpha: Optional[float] = None,
    quote_exposure_imbalance_signal: bool = False,
    bayes_flow_enabled: bool = False,
    bayes_flow_tau_r: float = 60.0,
    bayes_flow_p_low: float = 0.2,
    bayes_flow_p_high: float = 0.8,
    bayes_flow_prior_a: float = 1.0,
    bayes_flow_prior_b: float = 1.0,
    bayes_flow_prior_mode: str = "truncated_beta_grid",
    bayes_flow_hazard: Optional[float] = None,
    bayes_flow_max_run_length: Optional[int] = None,
    bayes_flow_cp_window: int = 5,
    bayes_flow_use_mixture: bool = False,
    bayes_flow_mixture_taus: Optional[List[float]] = None,
    bayes_flow_mixture_weights: Optional[List[float]] = None,
):
    """
    Generator version of ``simulate_LOB_with_MM`` for external training loops.

    Purpose and Motivation
    ======================
    The standard ``simulate_LOB_with_MM`` function runs the full simulation
    in a single call, invoking ``controller.learn()`` internally at every
    micro-step.  This tightly couples the simulation with the learning
    algorithm — the controller's ``learn()`` method handles SMDP reward
    aggregation AND gradient updates in one place.

    For **synchronous Advantage Actor-Critic (A2C)** training, this coupling
    is problematic:

        - A2C requires N parallel environments to be stepped in lockstep.
        - After each step (or each decision point), transitions from ALL N
          environments are collected and used to compute a single, low-variance
          gradient update.
        - This is impossible when ``learn()`` is called inside the simulation,
          because each simulation runs independently without synchronization.

    This generator solves the problem by **yielding** the step data at each
    iteration instead of calling ``controller.learn()``.  The external training
    loop receives the data, performs SMDP aggregation and gradient updates on
    its own schedule, and resumes the generator by calling ``next()``.

    How It Differs from ``simulate_LOB_with_MM``
    =============================================
    The ONLY difference is at the point where the original function calls
    ``controller.learn()``.  In this generator:

        - ``controller.learn()`` is **NOT** called.
        - Instead, a dictionary with the step data is **yielded** to the caller.
        - The caller is responsible for calling ``controller.learn()`` (for SMDP
          aggregation) and/or performing gradient updates externally.

    Everything else is identical: LOB initialization, warmup, action selection
    via ``controller.act()`` (through ``mm.pre_step()``), environment stepping,
    MM bookkeeping (cancels, fills, shifts), tape recording, and DataFrame
    construction.

    Yield Protocol
    ==============
    The generator yields ONLY at **decision points** — when the controller's
    throttle fires and a real action is selected (``controller._last_was_decision
    == True``).  It also yields on episode termination (``done == True``).

    Throttled micro-steps (where the agent holds) are handled internally:
    the generator calls ``controller.learn()`` on those steps for SMDP reward
    aggregation, but does NOT yield them to the consumer.

    This means ``zip(*generators)`` over N environments naturally synchronizes
    at the decision level: all N environments produce their k-th decision
    before any advances to decision k+1.

    **Per-decision yield** (yielded ~D times, where D = number of decisions):

        {
            "step_idx":     int,           # Iteration index (0 .. iterations-1)
            "mm":           MarketMaker,   # Live reference to the MM instance
            "lob":          LOB_simulation,# Live reference to the LOB instance
            "state_before": dict,          # State dict BEFORE the agent's action
            "state_after":  dict,          # State dict AFTER the environment step
            "reward":       float,         # Scalar reward from reward_fn (0.0 if None)
            "info":         dict,          # Step metadata (see below)
        }

        The ``info`` dict contains:
            - "done" (bool): True only on the last iteration (i == iterations-1).
            - "mm_action", "mm_action_side", "mm_action_price", etc.: Agent action details.
            - "env_order_type", "env_order_sign", "env_order_price", etc.: ZI event details.
            - "env_shift", "env_filled": Grid shift and fill information.

        IMPORTANT: ``mm`` and ``lob`` are **live mutable references** to the
        generator's internal objects.  They are valid for inspection or for
        passing to ``controller.learn()``, but their state changes on every
        ``next()`` call.  Do NOT store them for later use without deep-copying.

    **Final sentinel yield** (yielded once, after the main loop):

        {
            "step_idx":   -1,
            "done_final": True,
            "msg_df":     pd.DataFrame,   # Message tape (LOB events)
            "ob_df":      pd.DataFrame,   # Order book snapshots
            "mm_df":      pd.DataFrame,   # Market maker log with PnL
        }

        This sentinel signals that the simulation is complete and provides
        the same DataFrames that ``simulate_LOB_with_MM`` would return.
        Using a final yield (instead of Python's generator ``return``) avoids
        the need for ``StopIteration.value`` handling, which is error-prone.

    Internal learn() Dispatch
    =========================
    The ``learn()`` method on the controller serves two purposes:

        1. **SMDP reward aggregation**: Accumulates micro-step rewards between
           decision points and tracks holding times (K steps).
        2. **Gradient updates**: In "batch" mode, appends committed transitions
           to episode buffers for later gradient computation.

    This generator handles purpose (1) internally on throttled micro-steps by
    calling ``controller.learn()`` itself.  The consumer only receives yields
    at decision points and is responsible for calling ``controller.learn()``
    on those steps to commit the SMDP transition (purpose 2).

    Usage Examples
    ==============

    **Example 1 — Drop-in replacement (equivalent to original function):**

        >>> gen = simulate_LOB_with_MM_generator(
        ...     lam=lam, mu=mu, delta=delta, ..., controller=ctrl
        ... )
        >>> for step in gen:
        ...     if step.get("done_final"):
        ...         msg_df = step["msg_df"]
        ...         ob_df  = step["ob_df"]
        ...         mm_df  = step["mm_df"]
        ...         break
        ...     ctrl.learn(
        ...         step["step_idx"], step["mm"], step["lob"],
        ...         step["state_before"], step["state_after"],
        ...         step["reward"], step["info"],
        ...     )

    **Example 2 — Synchronous A2C with N parallel environments:**

        >>> # Each controller shares the same actor_net and critic_net,
        >>> # but has its own SMDP state (throttle counters, reward accumulators).
        >>> envs = [
        ...     simulate_LOB_with_MM_generator(..., controller=ctrls[i])
        ...     for i in range(N_ENVS)
        ... ]
        >>> for steps in zip(*envs):
        ...     # steps is a tuple of N step dicts — one per environment,
        ...     # all from the SAME decision index (k-th decision).
        ...     if any(s.get("done_final") for s in steps):
        ...         break
        ...     for s, ctrl in zip(steps, ctrls):
        ...         ctrl.learn(s["step_idx"], s["mm"], s["lob"],
        ...                    s["state_before"], s["state_after"],
        ...                    s["reward"], s["info"])
        ...     # All N workers now have 1 committed transition each.
        ...     # Aggregate and perform one gradient update.
        ...     batch = aggregate_transitions(ctrls)
        ...     actor_critic_update_batch(actor_net, critic_net, batch, ...)

    Parameters
    ----------
    Same as ``simulate_LOB_with_MM``.  See that function's signature for
    full parameter documentation.

    Yields
    ------
    step_dict : dict
        Per-step data dictionary (see Yield Protocol above), or the final
        sentinel dictionary containing the output DataFrames.

    See Also
    --------
    simulate_LOB_with_MM : The original non-generator version.
    actor_critic.ActorCriticController.learn : SMDP-aware learning step.
    actor_critic.actor_critic_update_batch : Batch A2C gradient update.
    """

    # =====================================================================
    # 1. INITIALIZATION (identical to simulate_LOB_with_MM)
    # =====================================================================

    rng = (
        np.random.RandomState(int(random_seed))
        if random_seed is not None
        else np.random.RandomState()
    )

    # Initialize LOB engine (Santa Fe or QRM depending on qrm_params)
    lob = _create_lob_engine(
        number_tick_levels=number_tick_levels,
        n_priority_ranks=n_priority_ranks,
        p0=p0,
        mean_size_LO=mean_size_LO,
        number_levels_to_store=number_levels_to_store,
        beta_exp_weighted_return=beta_exp_weighted_return,
        intensity_exp_weighted_return=intensity_exp_weighted_return,
        mean_size_MO=mean_size_MO,
        rng=rng,
        buy_mo_prob=buy_mo_prob,
        qrm_params=qrm_params,
    )
    lob.initialize()

    # Initialize tape columns
    if "Size" not in lob.message_dict: lob.message_dict["Size"] = []
    for _k in ("CancelVolAhead", "CancelQueueLen", "CancelRank", "BestBidPrice", "BestAskPrice", "TotNumberOrders"):
        if _k not in lob.message_dict: lob.message_dict[_k] = []

    if base_price_idx is None: base_price_idx = 0.0

    # =====================================================================
    # 2. WARMUP (identical to simulate_LOB_with_MM)
    # =====================================================================

    for _ in range(iterations_to_equilibrium):
        lob.simulate_order(lam, mu, delta)
    lob.exp_weighted_return_to_store = 0.0

    # Reset MO event counter and adversarial/regime schedule state that
    # may have been polluted by warmup MOs (mirrors simulate_LOB_with_MM).
    if hasattr(lob, '_event_step'):
        lob._event_step = 0
    if callable(buy_mo_prob) and hasattr(buy_mo_prob, 'reset'):
        buy_mo_prob.reset()

    # =====================================================================
    # 3. AGENT SETUP (identical to simulate_LOB_with_MM)
    # =====================================================================

    effective_policy = None
    if controller is not None:
        effective_policy = controller.act
    elif mm_policy is not None:
        effective_policy = mm_policy

    if controller is not None and getattr(controller, "pure_mm", False):
        pure_MM = True
        if pure_mm_offsets is None and hasattr(controller, "pure_mm_offsets"):
            pure_mm_offsets = controller.pure_mm_offsets
        if max_offset_override is None and hasattr(controller, "max_offset"):
            max_offset_override = controller.max_offset

    mm = MarketMaker(
        lob=lob,
        policy=effective_policy,
        exclude_self_from_state=exclude_self_from_state,
        use_sticky_price=use_sticky_price,
        tick_size=tick_size,
        backtest_mode=False,
        use_queue_ahead=False,
        base_price_idx=base_price_idx,
        pure_MM=pure_MM,
        pure_mm_offsets=pure_mm_offsets,
        max_offset_override=max_offset_override,
        rng=rng,
    )

    # Override EWMA smoothing factors if provided (same layered logic
    # as simulate_LOB_with_MM — see comments there for full rationale).
    if ewma_alpha is not None:
        mm.mo_flow_ewma_alpha = ewma_alpha
        mm.mo_flow_fast_ewma_alpha = ewma_alpha
        mm.fill_imbalance_alpha = ewma_alpha
    if mo_flow_ewma_alpha is not None:
        mm.mo_flow_ewma_alpha = mo_flow_ewma_alpha
    if mo_flow_fast_ewma_alpha is not None:
        mm.mo_flow_fast_ewma_alpha = mo_flow_fast_ewma_alpha
    if fill_imbalance_alpha is not None:
        mm.fill_imbalance_alpha = fill_imbalance_alpha
    mm.mo_flow_q_scale = max(float(mean_size_MO), 1.0)
    mm.fill_imbalance_q_scale = max(float(mean_size_MO), 1.0)
    mm.reset_fill_imbalance_prior(prior_mult=2.0)
    mm.configure_quote_exposure_imbalance_signal(
        enabled=bool(quote_exposure_imbalance_signal),
    )
    mm.configure_bayes_flow_signal(
        enabled=bool(bayes_flow_enabled),
        tau_r=float(bayes_flow_tau_r),
        p_low=float(bayes_flow_p_low),
        p_high=float(bayes_flow_p_high),
        prior_a=float(bayes_flow_prior_a),
        prior_b=float(bayes_flow_prior_b),
        prior_mode=str(bayes_flow_prior_mode),
        hazard=bayes_flow_hazard,
        max_run_length=bayes_flow_max_run_length,
        cp_window=int(bayes_flow_cp_window),
        use_mixture=bool(bayes_flow_use_mixture),
        mixture_taus=bayes_flow_mixture_taus,
        mixture_weights=bayes_flow_mixture_weights,
    )

    # Wire max_inventory from the controller's inv_limit so that reward
    # functions normalize inventory correctly (fixes fallback=10.0 bug).
    if controller is not None and hasattr(controller, "inv_limit") and controller.inv_limit is not None:
        mm.max_inventory = float(controller.inv_limit)

    t_i = 0.0
    row_idx = 0
    total_reward_all_steps = 0.0

    # =====================================================================
    # DEBUG HELPER (identical to simulate_LOB_with_MM)
    # =====================================================================

    def verify_lob_integrity(step, stage, extra_info=""):
        bid_idx = np.where(lob.lob_state > 0)[0]
        ask_idx = np.where(lob.lob_state < 0)[0]

        bb = int(bid_idx[-1]) if bid_idx.size else -1
        ba = int(ask_idx[0]) if ask_idx.size else -1

        if (bb >= 0 and ba >= 0 and bb >= ba):
            print("\n" + "#"*80)
            print(f"[CRITICAL DEBUG TRAP] MARKET CROSSED at Step {step}")
            print(f"Stage: {stage}")
            print(f"LOB State: BestBid={bb} >= BestAsk={ba} (Spread={ba-bb})")
            print(f"Extra Info: {extra_info}")
            print("-" * 40)
            print("MM Internal View (before crash):")
            _mm_bids = [info["price"] for info in mm.orders.values() if info.get("side") == +1]
            _mm_asks = [info["price"] for info in mm.orders.values() if info.get("side") == -1]
            print(f"  My Best Bid: {max(_mm_bids) if _mm_bids else 'none'}")
            print(f"  My Best Ask: {min(_mm_asks) if _mm_asks else 'none'}")
            print(f"  Inventory: {mm.inventory}")
            print("#"*80 + "\n")

            warnings.warn(
                f"Market Crossed ({stage}): Bid {bb} >= Ask {ba} at step {step}",
                RuntimeWarning,
                stacklevel=2,
            )

    # =====================================================================
    # 4. MAIN SIMULATION LOOP
    #
    # This is identical to simulate_LOB_with_MM EXCEPT for the final step
    # where controller.learn() would be called.  Instead, we YIELD the step
    # data to the external consumer.
    # =====================================================================

    for i in range(iterations):

        if debug_integrity: verify_lob_integrity(i, "START_OF_LOOP")

        # A+B) Agent Decision
        # controller.act() is called inside mm.pre_step() — this is where
        # throttling, action masking, and action selection happen.
        action, a_side, a_price, a_oid, a_pos = mm.pre_step()
        if effective_policy is None or mm.last_policy_state is None:
            mm.last_policy_state = mm.build_state()
        state_before = mm.last_policy_state

        # Snapshot for cancel tracking — AFTER MM action, BEFORE engine.
        pre_cols_for_cancel = lob.priorities_lob_state.copy()

        if debug_integrity: verify_lob_integrity(i, "AFTER_MM_ACTION", f"Action={action}, Side={a_side}, Price={a_price}")

        # D) Environment Step — ZI agent generates one event
        sim_out = lob.simulate_order(
            lam=lam, mu=mu, delta=delta,
            split_sweeps=split_sweeps,
        )

        if debug_integrity: verify_lob_integrity(i, "AFTER_SIMULATOR", "Engine generated an event")

        # Advance Clock
        dt_event = float(lob.last_event_dt)
        t_i += dt_event

        # E) Normalize Engine Output
        if isinstance(sim_out, tuple) and len(sim_out) == 3:
            rows, metrics, snaps = sim_out
        else:
            rows, metrics, snaps = [sim_out], [None], [None]

        # Extract Parent Event Summary
        order_type = int(rows[0][0])
        order_direction = int(rows[0][1])
        order_price = int(rows[0][2])
        shift = int(rows[-1][3])
        order_size = int(sum(int(r[4]) for r in rows))

        # F) Write to Tape
        for j, row in enumerate(rows):
            ot, od, op, sh, sz = row
            lob.message_dict["Time"].append(t_i)
            lob.message_dict["Type"].append(int(ot))
            lob.message_dict["Direction"].append(int(od))
            lob.message_dict["Price"].append(int(op))
            lob.message_dict["Shift"].append(int(sh))
            lob.message_dict["Size"].append(float(sz))

            m = metrics[j]
            if isinstance(m, dict):
                lob.message_dict["Spread"].append(float(m.get("Spread", np.nan)))
                lob.message_dict["MidPrice"].append(float(m.get("MidPrice", np.nan)))
                lob.message_dict["Return"].append(float(m.get("Return", np.nan)))
                lob.message_dict["TotNumberOrders"].append(float(m.get("TotNumberOrders", np.abs(lob.lob_state).sum())))
                bb = m.get("IndBestBid", -1)
                ba = m.get("IndBestAsk", -1)
                lob.message_dict["IndBestBid"].append(int(bb) if (bb == bb) else -1)
                lob.message_dict["IndBestAsk"].append(int(ba) if (ba == ba) else -1)
                bbp = m.get("BestBidPrice", bb)
                bap = m.get("BestAskPrice", ba)
                lob.message_dict["BestBidPrice"].append(float(bbp) if (bbp == bbp and bbp >= 0) else np.nan)
                lob.message_dict["BestAskPrice"].append(float(bap) if (bap == bap and bap >= 0) else np.nan)
                lob.message_dict["CancelVolAhead"].append(float(m.get("CancelVolAhead", np.nan)))
                lob.message_dict["CancelQueueLen"].append(float(m.get("CancelQueueLen", np.nan)))
                lob.message_dict["CancelRank"].append(float(m.get("CancelRank", np.nan)))
                lob.message_dict["TotNumberBidOrders"].append(float(m.get("TotNumberBidOrders", np.nan)))
                lob.message_dict["TotNumberAskOrders"].append(float(m.get("TotNumberAskOrders", np.nan)))
            else:
                lob.message_dict["Spread"].append(lob.compute_spread())
                lob.message_dict["MidPrice"].append(lob.compute_mid_price())
                lob.message_dict["Return"].append(lob.mid_price_to_store["Current"] - lob.mid_price_to_store["Previous"])
                lob.message_dict["TotNumberOrders"].append(float(np.abs(lob.lob_state).sum()))

                bid_idxs = np.where(lob.lob_state > 0)[0]
                ask_idxs = np.where(lob.lob_state < 0)[0]
                lob.message_dict["IndBestBid"].append(int(bid_idxs[-1]) if bid_idxs.size > 0 else -1)
                lob.message_dict["IndBestAsk"].append(int(ask_idxs[0]) if ask_idxs.size > 0 else -1)
                lob.message_dict["BestBidPrice"].append(float(bid_idxs[-1]) if bid_idxs.size > 0 else np.nan)
                lob.message_dict["BestAskPrice"].append(float(ask_idxs[0]) if ask_idxs.size > 0 else np.nan)

                lob.message_dict["CancelVolAhead"].append(np.nan)
                lob.message_dict["CancelQueueLen"].append(np.nan)
                lob.message_dict["CancelRank"].append(np.nan)
                lob.message_dict["TotNumberBidOrders"].append(np.nan)
                lob.message_dict["TotNumberAskOrders"].append(np.nan)

            snap = snaps[j]
            if isinstance(snap, dict):
                for _k in lob.ob_dict.keys():
                    lob.ob_dict[_k].append(snap.get(_k, 0))
            else:
                lob.update_ob_dict(row_idx)
            row_idx += 1

        # G) MM Bookkeeping: Handle Environment Cancellations
        pre_col = None
        if 0 <= order_price < pre_cols_for_cancel.shape[1]:
            pre_col = pre_cols_for_cancel[:, order_price].copy()

        post_idx = order_price - shift
        post_col = None
        if 0 <= post_idx < lob.priorities_lob_state.shape[1]:
            post_col = lob.priorities_lob_state[:, post_idx].copy()

        if order_type == 2 and (pre_col is not None) and (post_col is not None):
            cancel_rank = None
            try:
                m0 = metrics[0] if (isinstance(metrics, list) and len(metrics) > 0) else None
                if isinstance(m0, dict):
                    cr = m0.get("CancelRank", np.nan)
                    if cr == cr: cancel_rank = int(cr)
            except Exception:
                cancel_rank = None

            mm.adjust_for_cancel_at_price(order_price, pre_col, post_col, cancel_rank=cancel_rank)

        # H) MM Bookkeeping: Handle Environment Market Orders (Fills)
        executed_by_price_pre_shift = None
        if order_type == 1:
            # Update MO flow EWMA using signed, size-aware MO flow.
            mm.update_mo_flow_ewma(
                order_direction=order_direction,
                order_size=float(order_size),
            )
            mm.update_mo_flow_bayes(
                order_direction=order_direction,
                order_size=float(order_size),
            )

            executed_by_price_pre_shift = {}

            for m in metrics:
                if not isinstance(m, dict):
                    continue
                emap = m.get("ExecutedByPricePreShift", None)
                if not isinstance(emap, dict):
                    continue
                for px_raw, k_raw in emap.items():
                    try:
                        px_i = int(px_raw)
                        k_i = int(k_raw)
                    except Exception:
                        continue
                    if k_i <= 0:
                        continue
                    executed_by_price_pre_shift[px_i] = (
                        executed_by_price_pre_shift.get(px_i, 0) + k_i
                    )

            if (not executed_by_price_pre_shift) and split_sweeps:
                for _ot, _od, _op, _sh, _sz in rows:
                    if int(_ot) != 1:
                        continue
                    k = int(_sz)
                    if k > 0:
                        px = int(_op)
                        executed_by_price_pre_shift[px] = executed_by_price_pre_shift.get(px, 0) + k

            if not executed_by_price_pre_shift and int(order_size) > 0:
                executed_by_price_pre_shift = {int(order_price): int(order_size)}

        filled = mm.post_step_detect_fill_by_env_mo(
            order_type=order_type,
            order_sign=order_direction,
            order_price_pre_shift=order_price,
            step_idx=i,
            order_size=int(order_size),
            executed_by_price_pre_shift=executed_by_price_pre_shift,
        )

        # I) Apply Grid Shift to MM
        base_price_idx += shift
        mm.base_price_idx = base_price_idx
        mm.apply_shift(shift)

        env_fills = filled.get("fills_list", []) if isinstance(filled, dict) else []
        if (not env_fills) and isinstance(filled, dict) and float(filled.get("qty", 0.0)) > 0.0:
            env_fills = [filled]

        # J) Record MM Fills
        if isinstance(filled, dict) and float(filled.get("qty", 0.0)) > 0.0:
            px_pre = int(filled.get("px", int(order_price)))
            px_post = int(px_pre) - int(shift)
            try:
                n_levels = int(mm.owner.shape[1])
                px_post = max(0, min(px_post, n_levels - 1))
            except Exception: pass

            mm.last_fill = {
                "side": int(filled.get("side", 0)),
                "price": int(px_post),
                "time": float(t_i),
                "step": int(i),
                "qty": float(filled.get("qty", 0.0))
            }

        # Same common-clock update as the main simulator path above.
        mm.update_fill_imbalance_ewma_from_fills(env_fills)

        # K) Finalize Step — build state_after only when needed
        state_after = mm.build_state() if (reward_fn is not None or controller is not None) else {}
        _aidx_raw = getattr(controller, "last_action_idx", None) if controller is not None else None
        _aidx = int(_aidx_raw) if _aidx_raw is not None else -1
        mm.log_step(i, t_i, action, a_side, a_price, a_oid, a_pos, action_idx=_aidx)

        done = (i == iterations - 1)
        info: dict = {
            "mm_action": action, "mm_action_side": a_side, "mm_action_price": a_price,
            "mm_action_oid": a_oid, "mm_action_pos": a_pos,
            "env_order_type": order_type, "env_order_sign": order_direction,
            "env_order_price": order_price, "env_order_size": float(order_size),
            "env_shift": shift, "env_filled": filled, "env_fills": env_fills,
            "env_forced_cancels": list(getattr(mm, "_last_forced_cancels", [])),
            "env_reinit_regen": getattr(lob, "_last_reinit_regen", None),
            "done": done,
        }

        if reward_fn is not None:
            reward = float(reward_fn(i, mm, lob, state_before, state_after, info))
        else:
            reward = 0.0
        total_reward_all_steps += reward

        if mm.log:
            mm.log[-1]["MM_Reward"] = reward

        # Hook for adversarial training (mirrors simulate_LOB_with_MM).
        if callable(buy_mo_prob) and hasattr(buy_mo_prob, 'on_step_reward'):
            buy_mo_prob.on_step_reward(reward, mm, is_mo=(order_type == 1))

        # =================================================================
        # YIELD / LEARN DISPATCH
        #
        # The generator yields ONLY at decision points (when the controller's
        # throttle fired and a real action was selected).  On throttled
        # micro-steps, controller.learn() is called internally for SMDP
        # reward aggregation — the consumer never sees these steps.
        #
        # This ensures that when multiple generators are zipped together
        # for synchronous A2C training, zip() aligns at the decision level:
        # all N environments produce their k-th decision before any of them
        # advances to decision k+1.
        #
        # At episode termination (done=True), a final yield is always
        # emitted so that the consumer can process the terminal transition.
        # =================================================================
        is_decision = (
            controller is not None
            and getattr(controller, "_last_was_decision", False)
        )
        is_done = info.get("done", False)

        if is_decision or is_done:
            # Decision point (or terminal): yield to the external consumer.
            # The consumer is responsible for calling controller.learn()
            # and performing gradient updates across all N environments.
            yield {
                "step_idx": i,
                "mm": mm,
                "lob": lob,
                "state_before": state_before,
                "state_after": state_after,
                "reward": reward,
                "info": info,
            }
        else:
            # Throttled micro-step: call learn() internally for SMDP
            # reward aggregation (accumulates gamma^k * r into the
            # pending transition).  No yield — the consumer does not
            # see this step.
            if controller is not None:
                controller.learn(
                    i, mm, lob, state_before, state_after, reward, info,
                )

    # =====================================================================
    # 5. FINAL OUTPUT GENERATION (identical to simulate_LOB_with_MM)
    # =====================================================================

    if path_save_files is not None and label_simulation is not None:
        lob.save_results(path_save_files, label_simulation, max(row_idx - 1, 0))

    _align_dict_of_lists(lob.message_dict, row_idx, fill_value=np.nan)
    _align_dict_of_lists(lob.ob_dict, row_idx, fill_value=0)

    msg_df = pd.DataFrame(lob.message_dict)
    ob_df = pd.DataFrame(lob.ob_dict)
    mm_df = mm.to_dataframe()

    if "MM_Reward" in mm_df.columns:
        mm_df["MM_CumReward"] = mm_df["MM_Reward"].cumsum()

    if make_absolute_prices and ("Shift" in msg_df.columns):
        msg_df, ob_df, mm_df = make_all_prices_absolute(msg_df, ob_df, mm_df, p0=p0)

    # Final sentinel yield — signals simulation completion and delivers
    # the output DataFrames.  Using yield (not return) so the consumer
    # does not need to catch StopIteration.value.
    yield {
        "step_idx": -1,
        "done_final": True,
        "total_reward": float(total_reward_all_steps),
        "msg_df": msg_df,
        "ob_df": ob_df,
        "mm_df": mm_df,
    }


# ==============================================================================
# Adapter: make a DataFrame look enough like a LOB_simulation for reward fns
# ==============================================================================

class _BacktestLobAdapter:
    """Thin wrapper around the messages DataFrame passed as ``lob`` to reward
    functions in backtest mode.

    FIX: Custom reward functions that access ``lob.message_dict`` would crash
    because the backtest runners pass a raw DataFrame.  This adapter provides
    a ``message_dict`` property that lazily builds the dict-of-lists view,
    plus transparent delegation of all other DataFrame attributes (``iloc``,
    ``columns``, ``__len__``, etc.) so existing ``hasattr(lob, "iloc")``
    guards continue to work.
    """

    __slots__ = ("_df", "_cached_message_dict")

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._cached_message_dict: Optional[Dict[str, list]] = None

    # --- Core shim: message_dict ----
    @property
    def message_dict(self) -> Dict[str, list]:
        if self._cached_message_dict is None:
            self._cached_message_dict = {
                col: self._df[col].tolist() for col in self._df.columns
            }
        return self._cached_message_dict

    # --- Transparent delegation to the underlying DataFrame ----
    def __getattr__(self, name: str):
        return getattr(self._df, name)

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, item):
        return item in self._df


# ==============================================================================
# RUNNER: Backtest LOB + External Market Maker (Standard Fixed Grid)
# ==============================================================================

def backtest_LOB_with_MM(
    # --- Input Data (Pandas DataFrames) ---
    messages: pd.DataFrame,   # Tape: Time, Type, OrderID, Size, Price, Direction
    orderbook: pd.DataFrame,  # L1 Snapshots: AskPrice_1, AskSize_1, BidPrice_1, ...
    
    # --- Agent Configuration ---
    controller: Optional["RLController"] = None,       # Reinforcement Learning Brain
    mm_policy: Optional[Callable[[Dict[str, Any]], tuple]] = None, # Rule-Based Policy
    exclude_self_from_state: bool = False,             # Hides MM's own orders from state?
    
    # Controls "Sticky" vs "Transparent" price perception when excluding self
    use_sticky_price: bool = True,                     
    
    reward_fn: Optional[Callable] = None,              # Custom reward calculation
    tick_size: float = 0.01,                           # Tick size (e.g. 0.01 for stocks)
    
    # --- Execution Mechanics ---
    use_queue_ahead: bool = False,    # If True, simulates FIFO queue position
    base_price_idx: Optional[float] = None, # Base price for PnL calculations
    
    # --- Pure Market Making (Analytical) ---
    pure_MM: bool = False,            # Mode for classic strategies (Avellaneda/GLFT)
    pure_mm_offsets: Optional[List[Tuple[int, int]]] = None, # Offsets for Pure MM depth calculation
    max_offset_override: Optional[int] = None, # Override max offset

    # --- [NEW] Execution Integrity Configuration ---
    # If True: Forces executions to be integers via stochastic rounding.
    #          Prevents "dust" inventory (e.g. 0.004) while preserving PnL expected value.
    # If False: Accepts exact fractional executions (typical for Crypto).
    force_integer_fills: bool = True  
):
    """
    Backtest driver that replays a LOBSTER-like tape with an external Market Maker.
    Standard Version: The price grid is FIXED (static base_price_idx).

    Key Features:
    -------------
    1. Replays Order Book state step-by-step.
    2. Simulates Execution using "First Passage Time" (FPT).
    3. Simulates Queue Priority using Stochastic Rounding (if use_queue_ahead=True).
    4. Handles Fractional Trades utilizing Stochastic Execution to maintain integer inventory.
    """

    # ------------------------------------------------------------------
    # 0. Data Sanitization
    # ------------------------------------------------------------------
    # Ensure the 'Size' column exists in the message tape (default to 1.0 if missing)
    if "Size" not in messages.columns:
        messages = messages.copy() # Create a copy to avoid modifying original
        messages["Size"] = 1.0     # Default size

    # Calculate total steps (min length of messages and orderbook)
    n_steps = min(len(messages), len(orderbook))
    
    if n_steps == 0:
        raise ValueError("Empty messages or orderbook dataframes for backtest.")

    # ------------------------------------------------------------------
    # 1. Helper Function: Extract L1 Depth from Orderbook Row
    # ------------------------------------------------------------------
    def _extract_depth_from_row(row) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        bids: List[Tuple[int, float]] = []
        asks: List[Tuple[int, float]] = []
        cols = list(row.index)

        # --- Parse Bids ---
        # Filter columns starting with 'BidPrice'
        bid_price_cols = [c for c in cols if c.lower().startswith("bidprice_")]
        
        # Iterate sorted by level (1, 2, 3...)
        for pc in sorted(bid_price_cols, key=lambda x: int(x.split("_")[-1])):
            lvl = pc.split("_")[-1] # Extract level number
            
            # Try to find corresponding Size column
            sc_candidates = [f"BidSize_{lvl}", f"bid_size_{lvl}", f"bidSize_{lvl}"]
            sc = next((s for s in sc_candidates if s in cols), None)
            
            if sc is None: continue # Skip if size is missing
            
            px = row[pc]
            sz = row[sc]
            
            # Validate numeric and positive values
            if np.isfinite(px) and np.isfinite(sz) and sz > 0:
                bids.append((int(px), float(sz)))

        # --- Parse Asks (Identical logic) ---
        ask_price_cols = [c for c in cols if c.lower().startswith("askprice_")]
        for pc in sorted(ask_price_cols, key=lambda x: int(x.split("_")[-1])):
            lvl = pc.split("_")[-1]
            sc_candidates = [f"AskSize_{lvl}", f"ask_size_{lvl}", f"askSize_{lvl}"]
            sc = next((s for s in sc_candidates if s in cols), None)
            
            if sc is None: continue
            
            px = row[pc]
            sz = row[sc]
            if np.isfinite(px) and np.isfinite(sz) and sz > 0:
                asks.append((int(px), float(sz)))

        return bids, asks

    # ------------------------------------------------------------------
    # 2. Initialization & Base Price Setup
    # ------------------------------------------------------------------
    # Determine "Zero Point" for PnL if not provided.
    if base_price_idx is None:
        first_ob = orderbook.iloc[0]
        bids0, asks0 = _extract_depth_from_row(first_ob)

        if bids0 and asks0:
            # Initial Mid Price (average of best bid and ask)
            best_bid0 = max(p for p, _ in bids0)
            best_ask0 = min(p for p, _ in asks0)
            mid0_idx = 0.5 * (best_bid0 + best_ask0)
        elif bids0:
            mid0_idx = float(max(p for p, _ in bids0)) # Fallback bids only
        elif asks0:
            mid0_idx = float(min(p for p, _ in asks0)) # Fallback asks only
        else:
            mid0_idx = 0.0 # Fallback zero

        base_price_idx = float(mid0_idx)

    # ------------------------------------------------------------------
    # 3. Helper: Build Environment Snapshot
    # ------------------------------------------------------------------
    def _build_env_snapshot(i: int) -> Dict[str, Any]:
        ob_row = orderbook.iloc[i]
        msg_row = messages.iloc[i]

        bids, asks = _extract_depth_from_row(ob_row)
        has_bids = bool(bids)
        has_asks = bool(asks)

        # --- Top of Book (TOB) ---
        if has_bids:
            best_bid = max(p for p, _ in bids)
            bid_size = sum(sz for p, sz in bids if p == best_bid)
        else:
            best_bid = -1
            bid_size = 0.0

        if has_asks:
            best_ask = min(p for p, _ in asks)
            ask_size = sum(sz for p, sz in asks if p == best_ask)
        else:
            best_ask = -1
            ask_size = 0.0

        # --- Mid Price & Spread ---
        if has_bids and has_asks:
            spread = best_ask - best_bid
            mid_idx = 0.5 * (best_bid + best_ask)
        elif has_bids:
            spread = 0
            mid_idx = float(best_bid)
        elif has_asks:
            spread = 0
            mid_idx = float(best_ask)
        else:
            spread = 0
            mid_idx = 0.0

        # Convert mid index to dollar price for PnL
        mid_dollar = (base_price_idx + mid_idx) * tick_size

        # OBI (Order Book Imbalance)
        denom = max(1.0, bid_size + ask_size)
        obi = float(bid_size - ask_size) / denom

        # --- Trades (Executions) in this step ---
        trades = []
        t_type = msg_row.get("Type", None)

        # MidPrice in the converted LOBSTER message is the post-event mid.
        # Return is post - pre, so MidPrice - Return is the pre-event mid.
        raw_mid_post = None
        raw_return = 0.0
        try:
            raw_mid_val = msg_row.get("MidPrice", None)
            if raw_mid_val is not None and pd.notna(raw_mid_val):
                raw_mid_post = float(raw_mid_val)
        except Exception:
            raw_mid_post = None
        try:
            raw_return_val = msg_row.get("Return", 0.0)
            if raw_return_val is not None and pd.notna(raw_return_val):
                raw_return = float(raw_return_val)
        except Exception:
            raw_return = 0.0
        raw_mid_pre = (raw_mid_post - raw_return) if raw_mid_post is not None else None
        
        # Type 1 = Execution (Trade) in LOBSTER format
        if t_type == 1:
            px = int(msg_row["Price"])
            direction = int(msg_row.get("Direction", 0))
            if direction not in (-1, 1): direction = 0
            size = float(msg_row.get("Size", 1.0))
            
            trades.append({
                "price": px,
                "direction": direction,
                "size": size,
            })

        # --- Time Handling ---
        raw_time = msg_row.get("Time", ob_row.get("Time", 0.0))
        if isinstance(raw_time, pd.Timestamp):
            time_val = raw_time.timestamp()
        else:
            if hasattr(raw_time, "timestamp"):
                time_val = raw_time.timestamp()
            else:
                time_val = float(raw_time)

        env = {
            "time": time_val,
            "mid": mid_idx,
            "mid_dollar": mid_dollar,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "bidsize": bid_size,
            "asksize": ask_size,
            "obi": obi,
            "full_depth_bids": bids,
            "full_depth_asks": asks,
            "trades": trades,
            "raw_type": t_type,
            "raw_direction": int(msg_row.get("Direction", 0)),
            "raw_price": int(msg_row.get("Price", -1)),
            "raw_mid": raw_mid_post if raw_mid_post is not None else mid_idx,
            "raw_mid_pre": raw_mid_pre if raw_mid_pre is not None else mid_idx,
            "raw_return": raw_return,
            "has_bids": has_bids,
            "has_asks": has_asks,
        }
        return env

    # ------------------------------------------------------------------
    # 4. Instantiate Agent (MarketMaker)
    # ------------------------------------------------------------------
    effective_policy = None
    if controller is not None:
        effective_policy = controller.act # Use RL policy
    elif mm_policy is not None:
        effective_policy = mm_policy      # Use rule-based policy

    mm = MarketMaker(
        lob=None, # No LOB simulation engine in backtest mode
        policy=effective_policy,
        exclude_self_from_state=exclude_self_from_state,
        use_sticky_price=use_sticky_price,
        tick_size=tick_size,
        backtest_mode=True,
        use_queue_ahead=use_queue_ahead,
        base_price_idx=base_price_idx,
        pure_MM=pure_MM,
        pure_mm_offsets=pure_mm_offsets,
        max_offset_override=max_offset_override,
    )

    # Wire max_inventory from controller (same fix as simulation path).
    if controller is not None and hasattr(controller, "inv_limit") and controller.inv_limit is not None:
        mm.max_inventory = float(controller.inv_limit)

    env_handle = _BacktestLobAdapter(messages)

    # ------------------------------------------------------------------
    # 5. Main Replay Loop
    # ------------------------------------------------------------------
    for i in range(n_steps):
        # A) Build Environment Snapshot
        env = _build_env_snapshot(i)

        # B) Agent Decision (Action)
        # BUG FIX (M5): build state_before AFTER pre_step() so it includes
        # the agent's order placement, consistent with the simulation runner
        # which uses mm.last_policy_state (built inside pre_step).
        # Passing external_env first so pre_step's internal build_state sees it.
        mm.build_state(external_env=env)
        action, a_side, a_price, a_oid, a_pos = mm.pre_step()
        state_before = mm.last_policy_state

        trades = env["trades"]
        # BUG FIX (J): accumulate ALL fills in a list instead of
        # overwriting env_filled on each iteration (which discarded
        # earlier fills when multiple orders are hit in the same step).
        env_fills = []

        # Helper maps for queue logic (O(1) access)
        depth_bids = {px: sz for px, sz in env["full_depth_bids"]}
        depth_asks = {px: sz for px, sz in env["full_depth_asks"]}

        # D) Aggregate Traded Volume (for queue depletion)
        traded_map: Dict[Tuple[int, int], float] = {}
        for tr in trades:
            tr_px = int(tr["price"])
            tr_sz = float(tr["size"])
            tr_dir = int(tr["direction"])
            
            # Logic: Sell (-1) hits Bid (+1), Buy (+1) hits Ask (-1)
            if tr_dir == -1: side_hit = +1
            elif tr_dir == +1: side_hit = -1
            else: continue

            key = (side_hit, tr_px)
            traded_map[key] = traded_map.get(key, 0.0) + tr_sz

        # --- E) Queue Position Logic (STOCHASTIC UPDATE) ---------------------
        if mm.use_queue_ahead and mm.orders:
            for oid, info in list(mm.orders.items()):
                side = int(info["side"])
                px = int(info["price"])
                depth_map = depth_bids if side == +1 else depth_asks
                
                # Get current depth from snapshot
                D_curr = float(depth_map.get(px, 0.0))
                D_prev = float(info.get("last_depth", D_curr))
                Q_prev = float(info.get("queue_ahead", 0.0))
                
                info["last_depth"] = D_curr # Update memory
                
                if D_prev <= 0.0 or Q_prev <= 0.0: continue
                
                # Volume traded at this price level
                T_here = float(traded_map.get((side, px), 0.0))
                
                # Calculate raw liquidity drop
                raw_drop = max(D_prev - D_curr, 0.0)
                if raw_drop > D_prev: raw_drop = D_prev
                
                # Volume lost due to cancellations (Total Drop - Trades)
                cancel_drop = max(raw_drop - T_here, 0.0)

                # --- Stochastic Rounding for Cancellations ---
                if cancel_drop > 0.0:
                    # 1. Theoretical fraction ahead
                    frac_ahead = min(max(Q_prev / D_prev, 0.0), 1.0)
                    raw_delta = cancel_drop * frac_ahead # Mathematical reduction (float)
                    
                    delta_int = int(raw_delta)         # Integer part
                    delta_frac = raw_delta - delta_int # Fractional part (probability)
                    
                    # Stochastic update: preserves integer queue
                    # FIX 3.1: Use seeded RNG for reproducibility.
                    if mm.rng.uniform() < delta_frac:
                        delta_final = delta_int + 1
                    else:
                        delta_final = delta_int

                    # Subtract integer delta from previous queue
                    Q_new = max(int(Q_prev) - int(delta_final), 0)
                else:
                    Q_new = Q_prev

                # Ensure queue is not larger than current depth
                if D_curr <= 0.0: Q_new = 0.0
                else: Q_new = min(float(Q_new), D_curr)

                info["queue_ahead"] = float(Q_new)

        # --- F) Execution Logic (STOCHASTIC FILL, TRADE CONSUMPTION) ------
        # FIX 1.1: Track remaining volume per trade to prevent double counting.
        if trades and mm.orders:
            trade_remains = [float(tr["size"]) for tr in trades]

            for oid, info in list(mm.orders.items()):
                if oid not in mm.orders: continue
                side = int(info["side"])
                price = int(info["price"])
                queue_ahead = float(info.get("queue_ahead", 0.0))

                for t_idx, tr in enumerate(trades):
                    if trade_remains[t_idx] <= 0.0: continue
                    tr_price = int(tr["price"])
                    tr_dir = int(tr["direction"])

                    # 1. Check Crossing
                    crosses = False
                    if side == +1 and tr_dir == -1 and tr_price <= price: crosses = True
                    elif side == -1 and tr_dir == +1 and tr_price >= price: crosses = True
                    if not crosses: continue

                    # 2. Queue Consumption
                    if mm.use_queue_ahead and queue_ahead > 0.0:
                        consumed = min(trade_remains[t_idx], queue_ahead)
                        queue_ahead = max(queue_ahead - consumed, 0.0)
                        info["queue_ahead"] = queue_ahead
                        trade_remains[t_idx] -= consumed
                        if queue_ahead > 0.0 or trade_remains[t_idx] <= 0.0: continue
                        rem_trade = trade_remains[t_idx]
                    else:
                        rem_trade = trade_remains[t_idx]

                    # 3. Execution
                    remaining_order_qty = float(info.get("qty", 1.0))
                    raw_fill = min(float(rem_trade), remaining_order_qty)

                    if force_integer_fills:
                        fill_int = int(raw_fill)
                        fill_frac = raw_fill - fill_int

                        # FIX 3.1: Use seeded RNG for reproducibility.
                        if mm.rng.uniform() < fill_frac:
                            extra_fill = 1.0
                        else:
                            extra_fill = 0.0

                        fill_qty = float(fill_int + extra_fill)
                        # FIX 1.3: Cap at remaining order qty to prevent overfill.
                        fill_qty = min(fill_qty, remaining_order_qty)

                        if fill_qty <= 0.0: continue
                    else:
                        fill_qty = raw_fill

                    # Register fill if positive
                    if fill_qty > 0.0:
                        mm.register_backtest_passive_fill(
                            oid,
                            step_idx=i,
                            time_val=env["time"],
                            qty=fill_qty,
                        )
                        # FIX 1.1: Deduct consumed volume from the trade.
                        trade_remains[t_idx] -= fill_qty

                    # BUG FIX (J): append to list instead of overwriting.
                    env_fills.append({
                        "filled_oid": oid,
                        "px": price,
                        "side": side,
                        "qty": fill_qty,
                    })
                    # Stop checking trades for this order if fully filled
                    if fill_qty >= int(remaining_order_qty):
                        break

        # Backtest/playback mode has no explicit MO tape, so use one update per step.
        mm.update_fill_imbalance_ewma_from_fills(env_fills)

        # G) Finalize Step (State Update & Logging)
        state_after = mm.build_state(external_env=env)
        _aidx_raw = getattr(controller, "last_action_idx", None) if controller is not None else None
        _aidx = int(_aidx_raw) if _aidx_raw is not None else -1
        mm.log_step(i, env["time"], action, a_side, a_price, a_oid, a_pos, action_idx=_aidx)

        # Backward-compatible: env_filled = last fill (or None)
        env_filled = env_fills[-1] if env_fills else None

        done = (i == n_steps - 1)
        info: Dict[str, Any] = {
            "mm_action": action,
            "mm_action_side": a_side,
            "mm_action_price": a_price,
            "mm_action_oid": a_oid,
            "mm_action_pos": a_pos,
            "env_order_type": env["raw_type"],
            "env_order_sign": env["raw_direction"],
            "env_order_price": env["raw_price"],
            "env_shift": 0,  # Fixed grid in this runner
            "env_filled": env_filled,  # backward compat: last fill or None
            "env_fills": env_fills,    # BUG FIX (J): all fills this step
            "done": done,
        }

        # H) Calculate Reward & Learn
        if reward_fn is not None:
            reward = float(reward_fn(i, mm, env_handle, state_before, state_after, info))
        else:
            reward = 0.0

        if mm.log: mm.log[-1]["MM_Reward"] = reward

        if controller is not None:
            # Learning step (if applicable)
            controller.learn(i, mm, env_handle, state_before, state_after, reward, info)

    # ------------------------------------------------------------------
    # 6. Final Outputs
    # ------------------------------------------------------------------
    msg_df = messages.copy()
    ob_df = orderbook.copy()
    mm_df = mm.to_dataframe()

    if "MM_Reward" in mm_df.columns:
        mm_df["MM_CumReward"] = mm_df["MM_Reward"].cumsum() # Calculate cumulative reward

    return msg_df, ob_df, mm_df

# ==============================================================================
# RUNNER: Robust Backtest LOB + Dynamic Shifting + Stochastic Queues
# ==============================================================================


def backtest_LOB_with_MM_dynamic_shift(
    # --- Input Data ---
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    
    # --- Agent Config ---
    controller: Optional["RLController"] = None,
    mm_policy: Optional[Callable[[Dict[str, Any]], tuple]] = None,
    exclude_self_from_state: bool = False,
    use_sticky_price: bool = True,
    
    # --- Environment Config ---
    reward_fn: Optional[Callable] = None,
    tick_size: float = 0.01,
    base_price_idx: Optional[float] = None,
    
    # --- Execution Mechanics (The "Physics" of the Sim) ---
    use_queue_ahead: bool = False,
    use_exact_fifo: bool = False,   # Exact FIFO tracking via LOBSTER order IDs

    # [NEW] Dynamic Grid Shifting
    # If the relative mid-price exceeds this threshold (e.g., 100 ticks),
    # the grid re-centers to 0. Critical for RL stability.
    shift_threshold: int = 100, 
    
    # [NEW] Integer Integrity
    # If True, partial fills from fractional market orders are floored.
    # Ensures inventory remains strictly integer (e.g., 0, 1, 2).
    force_integer_fills: bool = True,

    # --- Pure MM (Analytical) ---
    pure_MM: bool = False,
    pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
    max_offset_override: Optional[int] = None,
):
    """
    Advanced Backtest Driver with Dynamic Grid Re-centering and Stochastic Execution.

    Features:
    ---------
    1. Dynamic Shifting: 
       - Unlike standard backtests that fix the grid at t=0, this runner "moves the camera".
       - If the market trends up by 500 ticks, the grid shifts so the agent sees "0".
       - Prevents input explosion for Neural Networks.

    2. Stochastic Queue Simulation:
       - Uses probability to decide if a cancellation happened "ahead" or "behind" us.
       - Prevents fractional queue positions (e.g., being behind 4.5 orders).

    3. Integer Integrity:
       - Enforces strictly integer executions to prevent "dust" inventory (e.g. 0.0001).
    """

    # ------------------------------------------------------------------
    # 0. Setup & Sanitization
    # ------------------------------------------------------------------
    if "Size" not in messages.columns:
        messages = messages.copy()
        messages["Size"] = 1.0

    n_steps = min(len(messages), len(orderbook))
    if n_steps == 0:
        raise ValueError("Empty dataframes provided.")

    # ------------------------------------------------------------------
    # 1. Helper: Extract Raw Depth (Before Shift)
    # ------------------------------------------------------------------
    def _extract_depth_from_row(row) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        bids, asks = [], []
        cols = list(row.index)
        
        # Bids
        bid_cols = [c for c in cols if c.lower().startswith("bidprice_")]
        for pc in sorted(bid_cols, key=lambda x: int(x.split("_")[-1])):
            lvl = pc.split("_")[-1]
            # Try to find matching size column
            sc = next((s for s in [f"BidSize_{lvl}", f"bid_size_{lvl}"] if s in cols), None)
            if sc:
                px, sz = row[pc], row[sc]
                if np.isfinite(px) and np.isfinite(sz) and sz > 0:
                    bids.append((int(px), float(sz)))
        
        # Asks
        ask_cols = [c for c in cols if c.lower().startswith("askprice_")]
        for pc in sorted(ask_cols, key=lambda x: int(x.split("_")[-1])):
            lvl = pc.split("_")[-1]
            sc = next((s for s in [f"AskSize_{lvl}", f"ask_size_{lvl}"] if s in cols), None)
            if sc:
                px, sz = row[pc], row[sc]
                if np.isfinite(px) and np.isfinite(sz) and sz > 0:
                    asks.append((int(px), float(sz)))
                    
        return bids, asks

    # ------------------------------------------------------------------
    # 2. Initialize Base Price (The "Anchor")
    # ------------------------------------------------------------------
    if base_price_idx is None:
        # Determine initial mid price from the first row
        bids0, asks0 = _extract_depth_from_row(orderbook.iloc[0])
        if bids0 and asks0:
            mid0 = 0.5 * (max(p for p, _ in bids0) + min(p for p, _ in asks0))
        elif bids0:
            mid0 = float(max(p for p, _ in bids0))
        else:
            mid0 = 0.0
        base_price_idx = float(mid0)

    # Global Accumulator for Shifts (Tracks how far the grid has moved from start)
    cumulative_shift = 0

    # ------------------------------------------------------------------
    # 3. Helper: Build Snapshot (With Dynamic Shift Applied)
    # ------------------------------------------------------------------
    def _build_env_snapshot(i: int, current_shift: int) -> Dict[str, Any]:
        ob_row = orderbook.iloc[i]
        msg_row = messages.iloc[i]
        
        # A) Extract Raw Prices from History
        raw_bids, raw_asks = _extract_depth_from_row(ob_row)

        # B) Apply Shift: (Raw_Price - Shift) -> Relative_Price
        #    If market is at 1000 and shift is 1000, Agent sees 0.
        bids = [(px - current_shift, sz) for px, sz in raw_bids]
        asks = [(px - current_shift, sz) for px, sz in raw_asks]

        has_bids = bool(bids)
        has_asks = bool(asks)

        # C) Calculate TOB (Relative)
        if has_bids:
            best_bid = max(p for p, _ in bids)
            bid_size = sum(sz for p, sz in bids if p == best_bid)
        else:
            best_bid = -1
            bid_size = 0.0

        if has_asks:
            best_ask = min(p for p, _ in asks)
            ask_size = sum(sz for p, sz in asks if p == best_ask)
        else:
            best_ask = -1
            ask_size = 0.0

        # D) Mid & Spread (Relative)
        if has_bids and has_asks:
            spread = best_ask - best_bid
            mid_idx = 0.5 * (best_bid + best_ask)
        elif has_bids:
            spread = 0
            mid_idx = float(best_bid)
        elif has_asks:
            spread = 0
            mid_idx = float(best_ask)
        else:
            spread = 0
            mid_idx = 0.0

        # E) OBI
        denom = max(1.0, bid_size + ask_size)
        obi = float(bid_size - ask_size) / denom

        # F) Trades (Must also be shifted!)
        trades = []
        t_type = msg_row.get("Type", None)
        raw_px_msg = int(msg_row.get("Price", -1))

        # MidPrice in the converted LOBSTER message is post-event.  Return
        # is post - pre, so MidPrice - Return recovers the pre-event mid.
        # Apply the same dynamic shift as raw_price so depths are measured
        # in the agent's current relative price grid.
        raw_mid_post = None
        raw_return = 0.0
        try:
            raw_mid_val = msg_row.get("MidPrice", None)
            if raw_mid_val is not None and pd.notna(raw_mid_val):
                raw_mid_post = float(raw_mid_val) - current_shift
        except Exception:
            raw_mid_post = None
        try:
            raw_return_val = msg_row.get("Return", 0.0)
            if raw_return_val is not None and pd.notna(raw_return_val):
                raw_return = float(raw_return_val)
        except Exception:
            raw_return = 0.0
        raw_mid_pre = (raw_mid_post - raw_return) if raw_mid_post is not None else None
        
        if t_type == 1: # Execution
            rel_px = raw_px_msg - current_shift
            direction = int(msg_row.get("Direction", 0))
            if direction not in (-1, 1): direction = 0
            size = float(msg_row.get("Size", 1.0))
            
            trades.append({
                "price": rel_px,
                "direction": direction,
                "size": size
            })

        # G) Time
        raw_time = msg_row.get("Time", ob_row.get("Time", 0.0))
        if hasattr(raw_time, "timestamp"): time_val = raw_time.timestamp()
        else: time_val = float(raw_time)

        # H) Mid Dollar Calculation
        #    Note: 'base_price_idx' in the MM is updated when shifts happen.
        #    So: (Updated_Base + Relative_Mid) = Real_Price.
        #    We compute it here just for the snapshot if needed.
        #    (The MM calculates PnL using its own internal base_price_idx).
        mid_dollar = 0.0 # Placeholder, MM handles PnL.

        return {
            "time": time_val, "mid": mid_idx, "mid_dollar": mid_dollar,
            "best_bid": best_bid, "best_ask": best_ask, "spread": spread,
            "bidsize": bid_size, "asksize": ask_size, "obi": obi,
            "full_depth_bids": bids, "full_depth_asks": asks, "trades": trades,
            "raw_type": t_type, 
            "raw_direction": int(msg_row.get("Direction", 0)),
            "raw_price": raw_px_msg - current_shift, # Shifted msg price
            "raw_mid": raw_mid_post if raw_mid_post is not None else mid_idx,
            "raw_mid_pre": raw_mid_pre if raw_mid_pre is not None else mid_idx,
            "raw_return": raw_return,
            "has_bids": has_bids, "has_asks": has_asks,
        }

    # ------------------------------------------------------------------
    # 4. Instantiate Agent
    # ------------------------------------------------------------------
    effective_policy = None
    if controller is not None: effective_policy = controller.act
    elif mm_policy is not None: effective_policy = mm_policy

    mm = MarketMaker(
        lob=None,
        policy=effective_policy,
        exclude_self_from_state=exclude_self_from_state,
        use_sticky_price=use_sticky_price,
        tick_size=tick_size,
        backtest_mode=True,
        use_queue_ahead=use_queue_ahead,
        
        # Initial Base Price (Will be updated dynamically)
        base_price_idx=base_price_idx, 
        
        pure_MM=pure_MM,
        pure_mm_offsets=pure_mm_offsets,
        max_offset_override=max_offset_override,
    )

    # Wire max_inventory from controller (same fix as simulation path).
    if controller is not None and hasattr(controller, "inv_limit") and controller.inv_limit is not None:
        mm.max_inventory = float(controller.inv_limit)

    # --- Exact FIFO queue tracking via LOBSTER order IDs ---
    fifo_tracker: Optional[FIFOQueueTracker] = None
    if use_exact_fifo:
        if "ID" not in messages.columns or "LobsterType" not in messages.columns:
            raise ValueError(
                "use_exact_fifo=True requires 'ID' and 'LobsterType' columns in messages. "
                "Re-run convert_lobster_to_sim with a LOBSTER dataset that includes order IDs."
            )
        use_queue_ahead = True
        mm.use_queue_ahead = True
        fifo_tracker = FIFOQueueTracker()
        mm._fifo_tracker = fifo_tracker

    env_handle = _BacktestLobAdapter(messages)

    # Track per-step shifts so we can write them into msg_df["Shift"]
    # for make_all_prices_absolute to work correctly with backtest data.
    recorded_shifts = np.zeros(n_steps, dtype=int)

    # ------------------------------------------------------------------
    # 5. Main Loop
    # ------------------------------------------------------------------
    for i in range(n_steps):
        
        # --- A) Dynamic Shift Logic -----------------------------------
        # 1. Build temp view to check mid-price deviation
        env = _build_env_snapshot(i, cumulative_shift)

        # BUG FIX (X3 + X4): Two requirements in tension:
        #   X3: state_before["mid"] must be pre-shift for the reward function
        #       (wealth_before = cash + inv * (prev_base + mid_pre) * tick).
        #   X4: The policy must see post-shift coordinates to place orders
        #       at correct prices.  Without this, the policy places orders
        #       at pre-shift prices (e.g. 498) while the LOB is centered
        #       near 0, making them completely off-book.
        #
        # Solution: capture state_before from the pre-shift env, then
        # update _last_env_snapshot to post-shift before pre_step.

        # 1) Seed pre-shift env and capture state_before for reward
        mm.build_state(external_env=env)
        state_before = dict(mm.build_state())  # pre-shift coords for reward

        current_mid = env["mid"]
        step_shift = 0

        # Check if Grid needs Re-centering
        if abs(current_mid) > shift_threshold:
            # We shift by exactly the integer part of the mid to bring it near 0
            step_shift = int(current_mid)

            # Update Global Accumulator (for future file reads)
            cumulative_shift += step_shift

            # Update MM's Anchor (Base Price)
            mm.base_price_idx += step_shift

            # Notify MM to shift its internal orders
            mm.apply_shift(step_shift)

            # Shift FIFO tracker queues to match new grid
            if fifo_tracker is not None:
                fifo_tracker.apply_shift(step_shift)

            # Rebuild environment so agent sees the new centered world immediately
            env = _build_env_snapshot(i, cumulative_shift)

            # Update MM's snapshot so policy sees post-shift coordinates
            mm.build_state(external_env=env)

        # --------------------------------------------------------------
        recorded_shifts[i] = step_shift

        # B) Agent Logic — policy now sees post-shift coords
        action, a_side, a_price, a_oid, a_pos = mm.pre_step()
        # state_before was already captured above in pre-shift coords (X3)
        
        trades = env["trades"]
        # BUG FIX (J): accumulate ALL fills in a list.
        env_fills = []

        # Helper Maps for Queue Logic
        depth_bids = {px: sz for px, sz in env["full_depth_bids"]}
        depth_asks = {px: sz for px, sz in env["full_depth_asks"]}

        # C) Aggregate Traded Volume
        traded_map: Dict[Tuple[int, int], float] = {}
        for tr in trades:
            key = ( -1 if tr["direction"] == 1 else 1, int(tr["price"]) ) # 1=Buy hits Ask(-1)
            traded_map[key] = traded_map.get(key, 0.0) + float(tr["size"])

        # C2) Feed LOBSTER event into FIFO tracker (before queue/fill logic)
        if fifo_tracker is not None:
            msg_row_i = messages.iloc[i]
            lob_type_i = int(msg_row_i.get("LobsterType", 0))
            if lob_type_i in (1, 2, 3, 4):
                lob_oid_i = int(msg_row_i.get("ID", -1))
                lob_px_i = int(msg_row_i.get("Price", -1)) - cumulative_shift  # relative
                lob_sz_i = float(msg_row_i.get("Size", 0.0))
                lob_dir_i = int(msg_row_i.get("Direction", 0))
                fifo_tracker.process_event(lob_px_i, lob_oid_i, lob_type_i,
                                           lob_sz_i, lob_dir_i)

        # --- D) Queue-Ahead Update ---------------
        if mm.use_queue_ahead and mm.orders:
            if fifo_tracker is not None:
                # EXACT FIFO: read authoritative queue position from tracker.
                # The tracker already processed the LOBSTER event (C2), so
                # volume_ahead_of reflects cancellations and executions that
                # happened at this step.
                for oid, info in list(mm.orders.items()):
                    px = int(info["price"])
                    info["queue_ahead"] = float(fifo_tracker.volume_ahead_of(px, oid))
            else:
                # STOCHASTIC fallback (original proportional model)
                for oid, info in list(mm.orders.items()):
                    side, px = int(info["side"]), int(info["price"])

                    depth_map = depth_bids if side == +1 else depth_asks
                    D_curr = float(depth_map.get(px, 0.0))
                    D_prev = float(info.get("last_depth", D_curr))
                    Q_prev = float(info.get("queue_ahead", 0.0))

                    info["last_depth"] = D_curr

                    if D_prev <= 0.0 or Q_prev <= 0.0: continue

                    T_here = float(traded_map.get((side, px), 0.0))
                    raw_drop = max(D_prev - D_curr, 0.0)
                    if raw_drop > D_prev: raw_drop = D_prev

                    cancel_drop = max(raw_drop - T_here, 0.0)

                    if cancel_drop > 0.0:
                        frac_ahead = min(max(Q_prev / D_prev, 0.0), 1.0)
                        raw_delta = cancel_drop * frac_ahead

                        delta_int = int(raw_delta)
                        delta_frac = raw_delta - delta_int

                        # FIX 3.1: Use seeded RNG for reproducibility.
                        if mm.rng.uniform() < delta_frac:
                            delta_final = delta_int + 1
                        else:
                            delta_final = delta_int

                        Q_new = max(int(Q_prev) - int(delta_final), 0)
                    else:
                        Q_new = Q_prev

                    if D_curr <= 0.0: Q_new = 0.0
                    else: Q_new = min(float(Q_new), D_curr)

                    info["queue_ahead"] = float(Q_new)

        # --- E) Fill Logic (INTEGER ENFORCED, TRADE CONSUMPTION) --------
        # FIX 1.1: Track remaining volume per trade so a single MO cannot
        # fill multiple MM orders beyond its actual size.
        if trades and mm.orders:
            trade_remains = [float(tr["size"]) for tr in trades]

            for oid, info in list(mm.orders.items()):
                if oid not in mm.orders: continue
                side, price = int(info["side"]), int(info["price"])
                queue_ahead = float(info.get("queue_ahead", 0.0))

                for t_idx, tr in enumerate(trades):
                    if trade_remains[t_idx] <= 0.0: continue
                    tr_price = int(tr["price"])
                    tr_dir = int(tr["direction"])

                    # Check Crossing
                    crosses = False
                    if side == +1 and tr_dir == -1 and tr_price <= price: crosses = True
                    elif side == -1 and tr_dir == +1 and tr_price >= price: crosses = True
                    if not crosses: continue

                    # Queue Consumption
                    if mm.use_queue_ahead and queue_ahead > 0.0:
                        if fifo_tracker is not None:
                            # EXACT FIFO: queue_ahead was set by the tracker
                            # (section D) and already reflects this event.
                            # Do NOT subtract trade volume again — just skip
                            # if still blocked.
                            continue
                        else:
                            # Stochastic: consume queue_ahead by trade volume
                            consumed = min(trade_remains[t_idx], queue_ahead)
                            queue_ahead = max(queue_ahead - consumed, 0.0)
                            info["queue_ahead"] = queue_ahead
                            trade_remains[t_idx] -= consumed
                            if queue_ahead > 0.0 or trade_remains[t_idx] <= 0.0: continue
                            rem_trade = trade_remains[t_idx]
                    else:
                        rem_trade = trade_remains[t_idx]

                    # With exact FIFO, the full remaining trade goes to MM (queue is clear)
                    if fifo_tracker is not None:
                        rem_trade = trade_remains[t_idx]

                    # Execution
                    rem_order = float(info.get("qty", 1.0))
                    raw_fill = min(float(rem_trade), rem_order)

                    # [FIX] Force Integer Fills
                    if force_integer_fills:
                        fill_int = int(raw_fill)
                        fill_frac = raw_fill - fill_int

                        # FIX 3.1: Use seeded RNG for reproducibility.
                        if mm.rng.uniform() < fill_frac:
                            extra_fill = 1.0
                        else:
                            extra_fill = 0.0

                        fill_qty = float(fill_int + extra_fill)
                        # FIX 1.3: Cap at remaining order qty to prevent overfill.
                        fill_qty = min(fill_qty, rem_order)

                        if fill_qty <= 0.0:
                            continue
                    else:
                        fill_qty = raw_fill

                    if fill_qty > 0.0:
                        mm.register_backtest_passive_fill(
                            oid, step_idx=i, time_val=env["time"], qty=fill_qty
                        )
                        # FIX 1.1: Deduct consumed volume from the trade.
                        trade_remains[t_idx] -= fill_qty

                    # BUG FIX (J): append to list instead of overwriting.
                    # BUG-FIX #4: Convert execution price back to PRE-SHIFT
                    # coordinates for consistency with reward functions.
                    env_fills.append({
                        "filled_oid": oid,
                        "px": price + step_shift,
                        "side": side,
                        "qty": fill_qty,
                    })
                    # If we filled the integer amount possible, break trade loop for this order
                    if fill_qty >= int(rem_order):
                        break

        # Backtest/playback mode has no explicit MO tape, so use one update per step.
        mm.update_fill_imbalance_ewma_from_fills(env_fills)

        # F) Finalize Step
        state_after = mm.build_state(external_env=env)
        _aidx_raw = getattr(controller, "last_action_idx", None) if controller is not None else None
        _aidx = int(_aidx_raw) if _aidx_raw is not None else -1
        mm.log_step(i, env["time"], action, a_side, a_price, a_oid, a_pos, action_idx=_aidx)

        # Backward-compatible: env_filled = last fill (or None)
        env_filled = env_fills[-1] if env_fills else None

        # -----------------------------------------------------------------
        # [BUG-FIX #5] Include "done" flag so that RL controllers (e.g.
        # DQN with n-step returns) can detect the terminal step and flush
        # their trajectory buffers.  The simulation runner already provides
        # this flag; the dynamic backtest was missing it.
        # -----------------------------------------------------------------
        done = (i == n_steps - 1)
        info: Dict[str, Any] = {
            "mm_action": action,
            "mm_action_side": a_side,
            "mm_action_price": a_price,
            "mm_action_oid": a_oid,
            "mm_action_pos": a_pos,
            "env_order_type": env["raw_type"],
            "env_order_sign": env["raw_direction"],
            "env_order_price": env["raw_price"],
            "env_shift": step_shift,  # Report the dynamic shift!
            "env_filled": env_filled,  # backward compat: last fill or None
            "env_fills": env_fills,    # BUG FIX (J): all fills this step
            "done": done,
        }

        # G) Reward
        if reward_fn is not None:
            reward = float(reward_fn(i, mm, env_handle, state_before, state_after, info))
        else:
            reward = 0.0

        if mm.log: mm.log[-1]["MM_Reward"] = reward
        
        if controller is not None:
            controller.learn(i, mm, env_handle, state_before, state_after, reward, info)

    # ------------------------------------------------------------------
    # 6. Outputs
    # ------------------------------------------------------------------
    msg_df = messages.copy()
    # Record dynamic shifts so make_all_prices_absolute can reconstruct offsets
    msg_df["Shift"] = recorded_shifts
    ob_df = orderbook.copy()
    mm_df = mm.to_dataframe()

    if "MM_Reward" in mm_df.columns:
        mm_df["MM_CumReward"] = mm_df["MM_Reward"].cumsum()
        
    return msg_df, ob_df, mm_df


# --------------------------------------------------
# Reward functions
# --------------------------------------------------


def reward_delta_pnl_with_inventory_penalty(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty: float = 0.01,
    inv_power: float = 1.0,
) -> float:
    """
    Calculates the change in wealth (Delta PnL) taking into account 
    LOB grid shifts to avoid artificial PnL jumps.
    """

    # 1. Get the current (post-step) base price from MM
    current_base = mm.base_price_idx
    tick_size = mm.tick_size

    # 2. Retrieve the shift that occurred during this step
    #    (This is critical: if shift != 0, the grid moved)
    shift = info.get("env_shift", 0)

    # 3. Reconstruct the base price that was valid for 'state_before'
    prev_base = current_base - shift

    # 4. Calculate Wealth BEFORE (using PREVIOUS base)
    #    Wealth = Cash + (Inventory * Mid_Price)
    mid_before_idx = float(state_before["mid"])
    mid_before_dollar = (prev_base + mid_before_idx) * tick_size
    
    cash_before = float(state_before["cash"])
    inv_before = float(state_before["inventory"])
    
    wealth_before = cash_before + (inv_before * mid_before_dollar)

    # 5. Calculate Wealth AFTER (using CURRENT base)
    #    Note: mm.total_pnl() uses the current base, so it is safe for 'after'.
    wealth_after = float(mm.total_pnl())

    # 6. Compute Delta
    delta_pnl = wealth_after - wealth_before

    # 7. Apply Inventory Penalty
    #    DQN agents tend to hoard inventory if not penalized.
    inv_after = float(state_after["inventory"])
    if inv_power == 2.0:
        inv_cost = inv_penalty * (inv_after ** 2)
    else:
        inv_cost = inv_penalty * abs(inv_after)

    return float(delta_pnl - inv_cost)

def reward_spread_capture_with_inventory_penalty(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.0001,
    symm_bonus: float =  0.002,
    inv_neutral_band: int = 9,
    inv_reduction_coeff: float = 0.005,
) -> float:
    """
    Reward function focusing on 'Realized Spread Capture' adjusted for Adverse Selection.
    
    Formula:
      Reward = (Realized_Spread_PnL) 
             - (Inventory_Penalty) 
             + (Inventory_Reduction_Bonus) 
             + (Symmetric_Quoting_Bonus)

    CRITICAL FIXES APPLIED:
    -----------------------
    1. Base Price Alignment (Shift Handling):
       - The LOB simulator shifts the grid to keep the mid-price centered.
       - Execution prices ('px') are reported in the PRE-SHIFT grid coordinates.
       - The final state ('mid_after') is in the POST-SHIFT grid coordinates.
       - We explicitly reconstruct 'prev_base' to convert execution prices to 
         absolute dollars, ensuring valid comparison with 'mid_after'.

    2. Mark-to-Market (Adverse Selection):
       - Instead of comparing execution price vs. mid_before (which ignores price drift),
         we compare execution price vs. mid_after.
       - If we sell at $100 and the price immediately jumps to $105, 
         using mid_after captures this -$5 loss (adverse selection).
    """
    
    tick_size = float(mm.tick_size)
    
    # ------------------------------------------------------------------
    # 1) Reconstruct Base Prices to handle LOB Shifts
    # ------------------------------------------------------------------
    # 'mm.base_price_idx' is already updated to the NEW grid (post-shift).
    current_base = mm.base_price_idx
    
    # Retrieve the shift that happened during this step
    shift = info.get("env_shift", 0)
    
    # Reconstruct the base valid for 'state_before' and the execution 'px'
    prev_base = current_base - shift

    # Convert the NEW Mid-Price to absolute dollars for comparison
    mid_after_idx = float(state_after["mid"])
    mid_after_abs = (current_base + mid_after_idx) * tick_size
    
    # ------------------------------------------------------------------
    # 2) Spread Capture PnL (Passive Fills Only)
    # ------------------------------------------------------------------
    spread_reward = 0.0
    # BUG FIX (J): handle both single fill (dict) and multi-fill (list).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0))
        if qty <= 0:
            continue
        px_idx = float(f["px"])
        px_abs = (prev_base + px_idx) * tick_size
        side = int(f["side"])  # -1 = Ask fill (sold), +1 = Bid fill (bought)

        if side == -1:
            spread_reward += px_abs - mid_after_abs
        elif side == +1:
            spread_reward += mid_after_abs - px_abs

    # ------------------------------------------------------------------
    # 3) Inventory Penalty (Quadratic)
    # ------------------------------------------------------------------
    # Penalizes holding large inventory positions.
    # Using quadratic penalty (inventory^2) discourages extreme positions stronger than linear.
    # BUG FIX (M7): use float() instead of int() to handle fractional inventory.
    inv_after = float(state_after["inventory"])
    inv_penalty = -inv_penalty_coeff * (abs(inv_after) ** 2)

    # ------------------------------------------------------------------
    # 4) Inventory Reduction Bonus (Round-Trip Incentive)
    # ------------------------------------------------------------------
    # Rewards the agent for reducing its exposure (moving closer to zero).
    inv_before = float(state_before["inventory"])
    inv_reduction = max(0, abs(inv_before) - abs(inv_after))
    inv_reduction_bonus = inv_reduction_coeff * inv_reduction

    # ------------------------------------------------------------------
    # 5) Symmetric Quoting Bonus
    # ------------------------------------------------------------------
    # Small incentive to keep both sides active when inventory is low/safe.
    symm_bonus_val = 0.0
    has_bid = bool(state_after.get("has_bid", False))
    has_ask = bool(state_after.get("has_ask", False))

    if has_bid and has_ask and abs(inv_after) <= inv_neutral_band:
        symm_bonus_val = symm_bonus

    # ------------------------------------------------------------------
    # Final Summation
    # ------------------------------------------------------------------
    reward = spread_reward + inv_penalty + inv_reduction_bonus + symm_bonus_val
    
    return float(reward)

def reward_santa_fe_generic(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    risk_aversion: float = 0.1,       # Lower start for Generic mode to encourage trading
    participation_bonus: float = 0.001 # Small incentive to keep orders in the book
) -> float:
    """
    Reward function designed for:
      1. Generic Mode (Discrete actions: Post/Cancel/Hold).
      2. Santa Fe Environment (Stochastic/Zero-Trend).

    Logic:
    ------
    1. Realized PnL (Profit): 
       - Calculates strictly the cash flow from trades.
       - Ignores passive inventory revaluation (Mark-to-Market).
       - Prevents 'Trend Following' behavior in a zero-trend market.

    2. Quadratic Inventory Penalty (Risk):
       - Forces Mean Reversion.
       - Penalizes large inventory accumulations exponentially.

    3. Participation Bonus (Activity):
       - Specific to Generic Mode.
       - Prevents the 'Lazy Agent' problem where the agent learns 
         to simply cancel everything and hold 0 inventory to avoid penalties.
    """

    # 1. Basic Configuration & Price Reconstruction
    # -----------------------------------------------------------
    tick_size = float(mm.tick_size)
    
    # Handle LOB grid shifting: retrieve the current absolute base price
    current_base = mm.base_price_idx
    
    # Calculate current Mid Price in absolute dollars
    # Used to value the inventory exchange at this exact moment.
    mid_after_abs = (current_base + float(state_after["mid"])) * tick_size
    
    # 2. Extract State Variables
    # -----------------------------------------------------------
    inv_before = float(state_before["inventory"])
    inv_after  = float(state_after["inventory"])
    
    cash_before = float(state_before["cash"])
    cash_after  = float(state_after["cash"])

    # 3. Per-Trade PnL Calculation (The "Profit Engine")
    # -----------------------------------------------------------
    delta_cash = cash_after - cash_before
    delta_inv = inv_after - inv_before

    # Per-Trade PnL Formula:
    # Profit = ΔCash + (ΔInventory × Current Mid Price)
    #
    # FIX (docstring): This is NOT pure "realized" PnL.  When a trade occurs
    # (delta_inv ≠ 0), the formula values the new position at mid_after, which
    # includes the spread capture (mid − exec_price) but also any mid-price
    # movement that happened this step.  When delta_inv == 0, the result is
    # exactly 0, so the agent is NOT rewarded for passive inventory revaluation
    # (price drift on existing inventory without a trade).
    per_trade_pnl = delta_cash + (delta_inv * mid_after_abs)

    # 4. Inventory Risk Penalty (The "Brake")
    # -----------------------------------------------------------
    # Get max inventory limit (default to 10 if not defined)
    limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 10.0
    
    # Normalize inventory to [-1, 1] range
    norm_inv = inv_after / limit
    
    # Quadratic Penalty:
    # - Low inventory: Negligible penalty.
    # - High inventory: Massive penalty.
    inventory_risk = risk_aversion * (norm_inv ** 2)

    # 5. Participation Bonus (Generic Mode Specific)
    # -----------------------------------------------------------
    # In Generic mode, the agent can choose to have NO orders (Cancel All).
    # If the inventory penalty is too scary, it might stop quoting entirely.
    # This tiny bonus ensures that Q(Post Order) > Q(Hold/Empty) when inventory is safe.
    p_bonus = 0.0
    if state_after.get("has_bid") or state_after.get("has_ask"):
        p_bonus = participation_bonus

    # 6. Final Reward
    # -----------------------------------------------------------
    reward = per_trade_pnl - inventory_risk + p_bonus

    return float(reward)

def reward_balanced_mm(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # ------------------------------------------------------------------
    # CRITICAL TUNING:
    # Reduced from 0.5 to 0.01.
    # Why? We need the max penalty to be roughly comparable to the 
    # profit of a single tick (approx 0.01). 
    # If this is too high, the agent stops trading to avoid risk.
    # ------------------------------------------------------------------
    inv_penalty_coeff: float = 0.01, 
    
    # ------------------------------------------------------------------
    # ANTI-LAZINESS BONUS:
    # A tiny incentive to keep orders in the book.
    # Essential for "Generic Mode" to prevent the agent from learning
    # that "doing nothing" (Hold/Cancel All) is the safest strategy.
    # ------------------------------------------------------------------
    participation_bonus: float = 0.001 
) -> float:
    """
    Balanced Reward Function for Generic Market Making.
    
    Components:
    1. Delta Wealth (PnL): The main engine. Rewards profit generation.
    2. Scaled Quadratic Penalty: The brake. Penalizes inventory, 
       but calibrated so it doesn't overpower the profit signal.
    3. Participation Bonus: The starter. Ensures the agent remains active.
    """
    
    tick_size = float(mm.tick_size)
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift
    
    # ------------------------------------------------------------------
    # 1. Delta Wealth (Mark-to-Market PnL)
    # ------------------------------------------------------------------
    # Reconstruct wealth at t and t+1 to measure financial performance.
    
    inv_before = float(state_before["inventory"])
    mid_before_idx = float(state_before["mid"])
    mid_before_dollar = (prev_base + mid_before_idx) * tick_size
    cash_before = float(state_before["cash"])
    W_before = inv_before * mid_before_dollar + cash_before

    inv_after = float(state_after["inventory"])
    mid_after_idx = float(state_after["mid"])
    mid_after_dollar = (current_base + mid_after_idx) * tick_size
    cash_after = float(state_after["cash"])
    W_after = inv_after * mid_after_dollar + cash_after

    delta_W = W_after - W_before

    # ------------------------------------------------------------------
    # 2. Adjusted Quadratic Penalty (Risk Control)
    # ------------------------------------------------------------------
    # Use max_inventory to normalize inputs to [-1.0, 1.0].
    # This keeps the penalty scale consistent regardless of the inventory size.
    limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 10.0
    normalized_inv = inv_after / limit
    
    # Quadratic Curve (x^2):
    # - Inventory 10% -> Factor 0.01 (Negligible penalty)
    # - Inventory 100% -> Factor 1.00 (Full penalty)
    # 
    # With coeff=0.01, the max penalty is 0.01 per step.
    # This is small enough to be treated as a "business cost" rather than a ban.
    running_penalty = inv_penalty_coeff * (normalized_inv ** 2)

    # ------------------------------------------------------------------
    # 3. Participation Bonus (Tie-Breaker)
    # ------------------------------------------------------------------
    # If the agent has at least one active order (Bid or Ask), it gets a crumb.
    # This solves the "Lazy Agent" problem in Generic Mode where Q(Hold)=0.
    # We want Q(Quote) > Q(Hold).
    p_bonus = 0.0
    if state_after.get("has_bid") or state_after.get("has_ask"):
        p_bonus = participation_bonus

    # ------------------------------------------------------------------
    # Final Reward Summation
    # ------------------------------------------------------------------
    reward = delta_W - running_penalty + p_bonus
    
    return float(reward)

def reward_delta_wealth_with_quadratic_inventory_penalty(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 1e-4,

) -> float:
    """
    Calculates the reward based on the Change in Wealth (PnL) 
    minus a Quadratic Penalty for holding inventory.

    OBJECTIVE:
    ----------
    This reward function encourages 'Mean Reversion' behavior. 
    1. It rewards making money (Delta Wealth).
    2. It strictly penalizes large inventory positions using a quadratic curve (x^2).
       - Small inventory (e.g., 1 or 2) -> Tiny penalty.
       - Large inventory (e.g., max limit) -> Huge penalty.
    
    This forces the agent to dump inventory quickly after capturing a move, 
    preventing the "Trend Following" (buy and hold) behavior.
    """
    
    tick_size = float(mm.tick_size)

    # ------------------------------------------------------------------
    # 1) Reconstruct Base Prices (Handling LOB Grid Shifts)
    # ------------------------------------------------------------------
    # The simulation environment re-centers the price grid (shift) to keep
    # the agent in the center. We must account for this shift to calculate
    # the real change in wealth accurately.

    # 'mm.base_price_idx' is the CURRENT base price (post-shift).
    current_base = mm.base_price_idx
    
    # Retrieve the shift that happened in this step.
    shift = info.get("env_shift", 0)
    
    # Reconstruct the base price that was valid for 'state_before'.
    prev_base = current_base - shift

    # ------------------------------------------------------------------
    # 2) Calculate Wealth BEFORE (using the PREVIOUS base)
    # ------------------------------------------------------------------
    inv_before = float(state_before["inventory"])
    mid_before_idx = float(state_before["mid"])
    
    # Convert local index to dollar price using prev_base
    mid_before_dollar = (prev_base + mid_before_idx) * tick_size
    
    cash_before = float(state_before["cash"])
    
    # Wealth_t = (Inventory_t * Mid_Price_t) + Cash_t
    W_before = inv_before * mid_before_dollar + cash_before

    # ------------------------------------------------------------------
    # 3) Calculate Wealth AFTER (using the CURRENT base)
    # ------------------------------------------------------------------
    inv_after = float(state_after["inventory"])
    mid_after_idx = float(state_after["mid"])
    
    # Convert local index to dollar price using current_base
    mid_after_dollar = (current_base + mid_after_idx) * tick_size
    
    cash_after = float(state_after["cash"])
    
    # Wealth_{t+1} = (Inventory_{t+1} * Mid_Price_{t+1}) + Cash_{t+1}
    W_after = inv_after * mid_after_dollar + cash_after

    # ------------------------------------------------------------------
    # 4) Change in Wealth (Mark-to-Market PnL)
    # ------------------------------------------------------------------
    delta_W = W_after - W_before

    # ------------------------------------------------------------------
    # 5) QUADRATIC Inventory Penalty (Risk Control)
    # ------------------------------------------------------------------
    # Instead of a linear penalty (|inv|), we use a quadratic one (inv^2).
    # This acts like a rubber band: weak when inventory is low, 
    # extremely strong when inventory is high.
    
    # A) Get the max inventory limit to normalize the values.
    #    (Defaulting to 10.0 if not found in 'mm' to prevent division by zero).
    limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 10.0
    
    # B) Normalize inventory to range [-1.0, 1.0].
    #    This ensures the penalty scale is consistent regardless of whether
    #    max_inventory is 10 or 1000.
    normalized_inv = inv_after / limit
    
    # C) Calculate the Quadratic Factor.
    #    Example: 
    #    - If inv is 10% of limit -> penalty factor is 0.01 (Tiny)
    #    - If inv is 100% of limit -> penalty factor is 1.00 (Full impact)
    inventory_factor = (normalized_inv) ** 2 
    
    # D) Apply the coefficient.
    #    Note: We removed 'dt' (time delta). This applies a per-step pressure.
    #    This is generally more robust for RL in discrete steps.
    running_penalty = inv_penalty_coeff * inventory_factor

    # ------------------------------------------------------------------
    # 6) Final Reward
    # ------------------------------------------------------------------
    # Reward = (Profit) - (Risk Aversion)
    reward = delta_W - running_penalty
    
    return float(reward)


def reward_delta_wealth_with_running_abs_inventory_penalty(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.001,
) -> float:
    """
    Calculates the reward based on the change in total wealth (PnL) 
    minus a running penalty for holding inventory.

    CRITICAL FIX:
    -------------
    This version correctly handles the LOB's price re-centering (shifts).
    
    The 'state_before' and 'state_after' dictionaries contain mid-prices 
    in *local index units* (e.g., 0 to 50). To calculate real wealth, 
    we must convert these to dollar prices using the correct base price 
    for that specific moment in time.

    Since the engine might have shifted the price grid between 'before' 
    and 'after', we must reconstruct the `prev_base` to value the 
    `state_before` correctly. Without this, a shift of +1 tick would 
    look like a massive PnL loss/gain on the entire inventory.
    """
    
    tick_size = float(mm.tick_size)

    # ------------------------------------------------------------------
    # 1) Reconstruct Base Prices (Handling Grid Shifts)
    # ------------------------------------------------------------------
    # 'mm.base_price_idx' has already been updated by the runner 
    # to reflect the POST-shift state.
    current_base = mm.base_price_idx
    
    # Retrieve the shift that occurred during this step (if any).
    # If shift > 0, the engine moved the grid left (price increased).
    shift = info.get("env_shift", 0)
    
    # Reconstruct the base price effectively used at 'state_before'.
    prev_base = current_base - shift

    # ------------------------------------------------------------------
    # 2) Calculate Wealth BEFORE (using the PREVIOUS base)
    # ------------------------------------------------------------------
    inv_before = float(state_before["inventory"])
    mid_before_idx = float(state_before["mid"])
    
    # CRITICAL: Use prev_base to convert the old local index to real price
    mid_before_dollar = (prev_base + mid_before_idx) * tick_size
    
    cash_before = float(state_before["cash"])
    
    # Wealth_t = (Inventory_t * Mid_Price_t) + Cash_t
    W_before = inv_before * mid_before_dollar + cash_before

    # ------------------------------------------------------------------
    # 3) Calculate Wealth AFTER (using the CURRENT base)
    # ------------------------------------------------------------------
    inv_after = float(state_after["inventory"])
    mid_after_idx = float(state_after["mid"])
    
    # Use current_base since 'state_after' is on the new grid
    mid_after_dollar = (current_base + mid_after_idx) * tick_size
    
    cash_after = float(state_after["cash"])
    
    # Wealth_{t+1} = (Inventory_{t+1} * Mid_Price_{t+1}) + Cash_{t+1}
    W_after = inv_after * mid_after_dollar + cash_after

    # ------------------------------------------------------------------
    # 4) Change in Wealth (Delta W)
    # ------------------------------------------------------------------
    delta_W = W_after - W_before

    # ------------------------------------------------------------------
    # 5) Running Inventory Penalty
    # ------------------------------------------------------------------
    # We penalize holding inventory over time: penalty ~ |Inventory| * dt
    
    # ------------------------------------------------------------------
    # [BUG-FIX #2 — CRITICAL] Extract time stamps to compute dt.
    #
    # In SIMULATION mode, `lob` is a LOB_simulation object which stores
    # event data in `lob.message_dict` (a dict-of-lists populated during
    # the live simulation).
    #
    # In BACKTEST mode, the runner passes `env_handle = messages` (a
    # pd.DataFrame) as the `lob` parameter.  Accessing `.message_dict`
    # on a DataFrame raises AttributeError, crashing any backtest that
    # uses this reward function.
    #
    # Fix: probe for the attribute first; if `lob` is a DataFrame,
    # fall back to its "Time" column.  If neither source is available,
    # gracefully default to dt = 0 (no time-based penalty).
    # ------------------------------------------------------------------
    if hasattr(lob, "message_dict"):
        # Simulation mode: lob is a LOB_simulation object
        times = lob.message_dict.get("Time", [])
    elif hasattr(lob, "iloc"):
        # Backtest mode: lob is a pd.DataFrame (messages)
        # Use the Time column directly.  For the last two events we
        # only need the rows up to step_idx.
        if "Time" in lob.columns and step_idx >= 1 and step_idx < len(lob):
            times = [float(lob["Time"].iloc[step_idx - 1]),
                     float(lob["Time"].iloc[step_idx])]
        else:
            times = []
    else:
        times = []

    # Robustness: if split_sweeps=True, multiple rows might be written.
    # We look at the last two timestamps to find the interval.
    # If it's the very first step or history is empty, dt = 0.
    if step_idx <= 0 or len(times) < 2:
        dt = 0.0
    else:
        # Calculate time delta.
        # Note: In split_sweeps mode, dt might be 0 for sub-fills,
        # which correctly implies no time-based penalty accumulation
        # for instant fills.
        dt = float(times[-1] - times[-2])
        # Clamp to 0 to avoid negative time in case of resets/edge cases
        dt = max(0.0, dt)

    running_penalty = inv_penalty_coeff * abs(inv_before) * dt

    # Final Reward
    reward = delta_W - running_penalty
    
    return float(reward)

def reward_mid_based_spread_with_abs_inventory_penalty(
    step_idx: int,
    mm,
    lob,
    state_before,
    state_after,
    info,
    inv_penalty_coeff: float = 0.01,
) -> float:
    """
    Reward of the form (discrete version of):
        R_{t+1} = (Q_t^ask - M_{t+1}) 1{Q_t^ask executed}
                + (M_{t+1} - Q_t^bid) 1{Q_t^bid executed}
                - λ |I_{t+1}|

    Interpretation in this simulator
    --------------------------------
    - If there is a passive fill (env_filled is not None):

        * side = -1 (our ASK got lifted):
              reward_spread = px_fill - mid_after

        * side = +1 (our BID got hit):
              reward_spread = mid_after - px_fill

      where px_fill is the execution price index and mid_after is the
      mid-price index AFTER the event.

    - If there is no passive fill, spread term = 0.

    - Inventory penalty:
          -λ * |inventory_after|.

    Parameters
    ----------
    step_idx : int
        Environment step index (unused here but kept for consistency).
    mm : MarketMaker
        MM wrapper (unused directly, but passed for symmetry with other rewards).
    lob : LOB_simulation
        LOB engine (unused here).
    state_before : dict
        State at time t (unused here).
    state_after : dict
        State at time t+1, used for mid and inventory.
    info : dict
        Must contain "env_filled" when a passive fill occurs.
    inv_penalty_coeff : float
        λ in the formula; scales the absolute inventory penalty.

    Returns
    -------
    float
        Reward at this step.
    """
    # ------------------------------------------------------------------
    # [BUG-FIX #3 — MEDIUM] Coordinate mismatch between px and mid_after.
    #
    # Previously, this function used raw local-index values for both the
    # execution price (px) and the post-event mid-price (mid_after).
    # However, these two values live in DIFFERENT coordinate systems
    # whenever the LOB engine performs a grid shift during the step:
    #
    #     px          → local index relative to the PRE-SHIFT grid
    #     mid_after   → local index relative to the POST-SHIFT grid
    #
    # For example, if the grid shifts by +2 ticks, an execution at local
    # index 27 (pre-shift) really corresponds to local index 25 (post-shift).
    # Without correction, the spread_term is off by `shift` ticks.
    #
    # Fix: convert BOTH prices to absolute tick coordinates using their
    # respective base prices (prev_base for px, current_base for mid_after).
    # The difference is then in absolute ticks, which is correct.
    # ------------------------------------------------------------------
    tick_size = float(mm.tick_size)
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # Absolute mid-price after the event (in dollar terms)
    mid_after_idx = float(state_after["mid"])
    mid_after_abs = (current_base + mid_after_idx) * tick_size

    # BUG FIX (M7): use float() instead of int() to handle fractional inventory.
    inv_after = float(state_after["inventory"])

    # ----- Spread term: only if we had a passive fill -----
    spread_term = 0.0
    # BUG FIX (J): handle both single fill (dict) and multi-fill (list).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]  # backward compat with old info format

    for f in fills:
        qty = float(f.get("qty", 0))
        if qty <= 0:
            continue
        px_idx = float(f["px"])       # execution price index (pre-shift)
        side = int(f["side"])         # -1 => our ask filled, +1 => our bid filled

        # Convert execution price to absolute dollar terms using the
        # pre-shift base, matching the coordinate system of the fill.
        px_abs = (prev_base + px_idx) * tick_size

        if side == -1:
            spread_term += px_abs - mid_after_abs
        elif side == +1:
            spread_term += mid_after_abs - px_abs

    # ----- Inventory penalty: -λ |I_{t+1}| -----
    inv_penalty = -inv_penalty_coeff * abs(inv_after)

    reward = spread_term + inv_penalty
    return float(reward)

def reward_santa_fe_generic_stable(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    risk_aversion: float = 0.1,
    participation_bonus: float = 0.001,
    # --- Cancellation Cost (penalizes order flickering) ---
    cancel_penalty: float = 0.002
) -> float:
    """
    Stabilized reward for Generic Mode.
    Punishes 'Order Flickering' to force the agent to hold queue position.
    """

    # 1. Basic Configuration
    tick_size = float(mm.tick_size)
    current_base = mm.base_price_idx
    mid_after_abs = (current_base + float(state_after["mid"])) * tick_size

    # 2. Realized PnL
    delta_cash = float(state_after["cash"]) - float(state_before["cash"])
    delta_inv = float(state_after["inventory"]) - float(state_before["inventory"])
    realized_pnl = delta_cash + (delta_inv * mid_after_abs)

    # 3. Inventory Risk
    limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 10.0
    norm_inv = float(state_after["inventory"]) / limit
    inventory_risk = risk_aversion * (norm_inv ** 2)

    # 4. Participation Bonus
    p_bonus = 0.0
    if state_after.get("has_bid") or state_after.get("has_ask"):
        p_bonus = participation_bonus

    # 5. --- CANCEL PENALTY LOGIC ---
    # We detect cancellations by inferring state changes that did NOT
    # produce a trade.  (Ideally we would pass the 'action' to the
    # function, but as a fast hack without changing the signature:
    # if inventory did not change AND cash did not change AND
    # has_bid/has_ask went from 1→0, a cancellation occurred.)

    c_penalty = 0.0

    # Heuristic Cancellation Detection (works for Generic Mode)
    had_bid_before = bool(state_before.get("has_bid"))
    had_ask_before = bool(state_before.get("has_ask"))
    has_bid_now = bool(state_after.get("has_bid"))
    has_ask_now = bool(state_after.get("has_ask"))

    # If we had a Bid, were NOT executed (delta_inv=0), and now have no Bid → cancelled.
    if delta_inv == 0:
        if had_bid_before and not has_bid_now:
            c_penalty += cancel_penalty
        if had_ask_before and not has_ask_now:
            c_penalty += cancel_penalty

    # 6. Final Reward
    reward = realized_pnl - inventory_risk + p_bonus - c_penalty

    return float(reward)


# def reward_santa_fe_pure_mm(
#     step_idx: int,
#     mm,
#     lob,
#     state_before: dict,
#     state_after: dict,
#     info: dict,
#     # --- FINE TUNING ---
#     # Start with 0.1. If the agent accumulates too much inventory, raise to 0.5.
#     risk_aversion: float = 0.1,
# ) -> float:
#     """
#     'Holy Grail' reward for Pure MM mode in Santa Fe.
#
#     Philosophy:
#     1. Realized PnL is the ONLY source of gain.
#        - Ignores passive mark-to-market appreciation.
#        - If price rises and agent is long, Reward = 0.
#        - This forces the agent to SELL to realize the profit.
#
#     2. Quadratic Inventory Penalty.
#        - Ensures the agent does not accumulate unnecessary risk.
#     """
#
#     tick_size = float(mm.tick_size)
#     current_base = mm.base_price_idx
#
#     # Current Mid Price in Dollars (Absolute)
#     mid_after_abs = (current_base + float(state_after["mid"])) * tick_size
#
#     # State Extraction
#     inv_before = float(state_before["inventory"])
#     inv_after  = float(state_after["inventory"])
#     cash_before = float(state_before["cash"])
#     cash_after  = float(state_after["cash"])
#
#     # 1. Realized PnL (Spread Capture)
#     # -----------------------------------------------------------
#     # Formula: Cash Change + (Inventory Change * Current Price)
#     # This mathematically isolates the profit from the closed trade.
#
#     delta_cash = cash_after - cash_before
#     delta_inv = inv_after - inv_before
#
#     realized_pnl = delta_cash + (delta_inv * mid_after_abs)
#
#     # 2. Quadratic Inventory Risk
#     # -----------------------------------------------------------
#     limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 10.0
#     norm_inv = inv_after / limit
#
#     # Penalty grows quadratically.
#     # E.g.: Inv=1 -> 0.01. Inv=10 -> 1.0.
#     inventory_risk = risk_aversion * (norm_inv ** 2)
#
#     # 3. Final Reward
#     # -----------------------------------------------------------
#     # Simple and direct. No artificial bonuses.
#     reward = realized_pnl - inventory_risk
#
#     return float(reward)


def reward_avellaneda_proxy(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # --- Risk Configuration ---
    # The penalty only kicks in above this inventory threshold.
    # For Limit=12, use 8. For Limit=4, use 2.
    inv_neutral_band: int = 8,

    # Strength of the penalty when the agent exits the safe zone.
    inv_penalty_coeff: float = 0.001,

    # Incentive to keep orders alive (anti-laziness bonus)
    symm_bonus: float = 0.002
) -> float:
    """
    Reward function optimized for Market Making in a Random Walk (Santa Fe).

    Logic: "Simplified Avellaneda-Stoikov"
    1. Profit (Alpha): Spread capture adjusted for Adverse Selection.
       - Gains if we sold above the future mid-price.
       - Loses if we got run over (price moved against us).

    2. Risk (Control): Inventory Penalty with a "Dead Zone".
       - If |Inv| <= Band: ZERO penalty (lets the agent trade volume).
       - If |Inv| > Band: Quadratic penalty on the EXCESS.
         This creates a soft barrier that pushes the agent back.
    """

    tick_size = float(mm.tick_size)

    # ------------------------------------------------------------------
    # 1. Grid Shift Correction (mandatory in the simulator)
    # ------------------------------------------------------------------
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # Future Mid Price (post-trade) in Dollars
    mid_after_idx = float(state_after["mid"])
    mid_after_abs = (current_base + mid_after_idx) * tick_size

    # ------------------------------------------------------------------
    # 2. Spread Capture (Execution Quality)
    # ------------------------------------------------------------------
    spread_reward = 0.0
    # BUG FIX (J): handle both single fill (dict) and multi-fill (list).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0))
        if qty <= 0:
            continue
        px_idx = float(f["px"])
        px_abs = (prev_base + px_idx) * tick_size
        side = int(f["side"])  # -1=Ask, +1=Bid

        if side == -1:  # Sold (Ask Fill)
            spread_reward += px_abs - mid_after_abs
        elif side == +1:  # Bought (Bid Fill)
            spread_reward += mid_after_abs - px_abs

    # ------------------------------------------------------------------
    # 3. Deadzone Inventory Penalty
    # ------------------------------------------------------------------
    # BUG FIX (M7): use float() instead of int() to handle fractional inventory.
    inv_after = abs(float(state_after["inventory"]))

    # Only penalize the EXCESS above the neutral band
    excess = max(0, inv_after - inv_neutral_band)

    # Quadratic Penalty on the excess
    # E.g.: Band=8.
    # Inv=5 → Excess=0 → Penalty=0.
    # Inv=10 → Excess=2 → Penalty=2^2 * Coeff = 4 * Coeff.
    inv_penalty = -inv_penalty_coeff * (excess ** 2)

    # ------------------------------------------------------------------
    # 4. Symmetry Bonus (optional, to prevent inactivity)
    # ------------------------------------------------------------------
    symm_bonus_val = 0.0
    if abs(float(state_after["inventory"])) <= inv_neutral_band:
        if state_after.get("has_bid") and state_after.get("has_ask"):
            symm_bonus_val = symm_bonus

    return float(spread_reward + inv_penalty + symm_bonus_val)

def reward_avellaneda_classic(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # ------------------------------------------------------------------
    # GAMMA (Risk Aversion):
    # In the paper, gamma ranges from 0.1 (aggressive) to 1.0 (conservative).
    # Since our tick is 0.01, we need to scale accordingly.
    # Start with 0.5.
    gamma: float = 0.5,
    # ------------------------------------------------------------------
    # VOLATILITY (Sigma):
    # In Santa Fe, volatility is constant (sigma ~ 1 tick per step or less).
    # We assume a constant to simplify the calculation.
    sigma_sq_dt: float = 0.01,
    # ------------------------------------------------------------------
) -> float:
    """
    Reward based on the Mean-Variance approximation of Avellaneda-Stoikov (2008).

    Formula: Reward = (Delta Wealth) - (gamma/2 * sigma^2 * dt * inventory^2)

    Difference from the 'Proxy' version:
    - There is NO 'Neutral Zone' or free band.
    - The agent pays 'rent' (risk) on EVERY unit of inventory from the very first.
    - This forces the agent to flatten the position (mean reversion) much faster.
    """

    tick_size = float(mm.tick_size)

    # 1. Robust Price Reconstruction (Grid Shift Handling)
    # ------------------------------------------------------------------
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # Prices in Dollars
    mid_before_idx = float(state_before["mid"])
    mid_before_abs = (prev_base + mid_before_idx) * tick_size

    mid_after_idx = float(state_after["mid"])
    mid_after_abs = (current_base + mid_after_idx) * tick_size

    # 2. Realized + Unrealized PnL (Total Mark-to-Market)
    # ------------------------------------------------------------------
    inv_before = float(state_before["inventory"])
    cash_before = float(state_before["cash"])
    wealth_before = cash_before + (inv_before * mid_before_abs)

    inv_after = float(state_after["inventory"])
    cash_after = float(state_after["cash"])
    wealth_after = cash_after + (inv_after * mid_after_abs)

    delta_wealth = wealth_after - wealth_before

    # 3. Classic Risk Penalty (No Dead Zone)
    # ------------------------------------------------------------------
    # Penalty = (gamma / 2) * sigma^2 * dt * (inventory^2)
    # Here we group sigma^2 * dt into a single estimated parameter for Santa Fe.

    inventory_term = (inv_after ** 2)
    risk_penalty = (gamma / 2.0) * sigma_sq_dt * inventory_term

    # Final Reward
    reward = delta_wealth - risk_penalty

    return float(reward)

def reward_santa_fe_pure_alpha(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # --- Optimized Configuration for Santa Fe ---
    inv_neutral_band: int = 9,         # Large buffer (for Pure MM)
    inv_penalty_coeff: float = 0.0001, # Soft penalty above the band
    symm_bonus: float = 0.002
) -> float:
    """
    Purist Alpha Reward for Santa Fe.

    Philosophy:
    "In a Random Walk, Carrying a Position is Risk, not Strategy."

    This function EXPLICITLY removes all PnL coming from the passive
    variation of inventory (Beta).
    The Reward reflects only the sum of Captured Spreads (Alpha).
    """

    tick_size = float(mm.tick_size)

    # 1. Grid Alignment (mandatory)
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # Absolute Prices
    mid_before_abs = (prev_base + float(state_before["mid"])) * tick_size
    mid_after_abs  = (current_base + float(state_after["mid"])) * tick_size

    # ------------------------------------------------------------------
    # 2. THE CORE: Alpha (Spread Capture)
    # ------------------------------------------------------------------
    # Here we compute ONLY the gain from the transaction vs. fair price.
    # Since this is Santa Fe, we do NOT explicitly punish 'Adverse
    # Selection', because the post-trade price movement is random.
    # We only measure the entry edge.

    alpha_reward = 0.0
    fill_info = info.get("env_filled", None)

    # Handle single or multiple fills
    # BUG FIX (M1): use env_fills (all fills) instead of env_filled (last only).
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0))
        if qty <= 0: continue

        px_abs = (prev_base + float(f["px"])) * tick_size
        side = int(f["side"])  # -1=Ask, +1=Bid

        # In Santa Fe, the "Edge" is simply the distance from the
        # execution price to the Mid Price at the time of the trade.
        # We use mid_after_abs as a proxy for "immediate fair price".

        if side == -1:  # Sold
            # Gain = Sale Price - Fair Price
            alpha_reward += (px_abs - mid_after_abs)
        else:           # Bought
            # Gain = Fair Price - Purchase Price
            alpha_reward += (mid_after_abs - px_abs)

    # ------------------------------------------------------------------
    # 3. THE FILTER: Beta (Passive Volatility)
    # ------------------------------------------------------------------
    # In Santa Fe, the price moves randomly.
    # If we hold inventory, this generates lucky or unlucky PnL.
    # We IGNORE that.  We neither add nor subtract it (unless we want
    # to punish the volatility of carry PnL).
    #
    # If you want Reward and PnL to be correlated, you should penalize
    # the volatility of carry PnL to incentivize flattening.

    inventory_exposure = abs(float(state_before["inventory"]))
    price_drift = abs(mid_after_abs - mid_before_abs)

    # We penalize 10% of the passive price swing value.
    # This says: "I don't like when inventory value changes randomly. Flatten it."
    passive_volatility_penalty = -0.1 * (inventory_exposure * price_drift)

    # ------------------------------------------------------------------
    # 4. Soft Risk Control
    # ------------------------------------------------------------------
    # BUG FIX (M7): use float() instead of int() to handle fractional inventory.
    inv_abs = abs(float(state_after["inventory"]))
    excess = max(0, inv_abs - inv_neutral_band)
    inv_penalty = -inv_penalty_coeff * (excess ** 2)

    # 5. Symmetry Incentive
    symm_bonus_val = 0.0
    if inv_abs <= inv_neutral_band:
        if state_after.get("has_bid") and state_after.get("has_ask"):
            symm_bonus_val = symm_bonus

    reward = alpha_reward + passive_volatility_penalty + inv_penalty + symm_bonus_val

    return float(reward)


def reward_spread_capture_inv_quadratic(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.01,
    # ── Inventory wall (optional) ──────────────────────────────────
    use_inv_wall: bool = False,
    inv_wall_threshold: float = 0.0,
    # ── Terminal PnL bonus (optional) ──────────────────────────────
    use_terminal_bonus: bool = False,
) -> float:
    """
    Simplified MM reward: spread capture - quadratic inventory penalty.

      r(t) = Σ dq(f)·(mid_before - px(f))  -  φ·inv_after²
             [ - φ·(|inv| - threshold)²             if |inv| > threshold ]
             [ + PnL_in_ticks                       if done & bonus on   ]

    Two terms, one coefficient.  Inventory wall and terminal bonus are
    optional (both default to 0 = off for backward compatibility).
    All values in ticks.
    """

    # Shift handling (ticks)
    current_base = mm.base_price_idx
    shift = int(info.get("env_shift", 0))
    prev_base = current_base - shift
    mid_before = float(prev_base + float(state_before["mid"]))

    # Spread capture term: Σ dq·(mid_before - px)
    spread_capture = 0.0
    # BUG FIX (M2): use env_fills (all fills) instead of env_filled (last only).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0.0))
        if qty <= 0:
            continue
        px = float(prev_base + float(f["px"]))
        side = int(f["side"])   # -1=ask fill (sold), +1=bid fill (bought)
        dq = (+qty if side == +1 else -qty)
        spread_capture += dq * (mid_before - px)

    # Quadratic inventory penalty
    inv_after = float(state_after["inventory"])
    inv_penalty = -inv_penalty_coeff * (inv_after ** 2)

    # Inventory wall: extra quadratic penalty beyond threshold (same φ as base)
    inv_wall = 0.0
    if use_inv_wall and inv_wall_threshold > 0.0:
        abs_inv = abs(inv_after)
        if abs_inv > inv_wall_threshold:
            excess = abs_inv - inv_wall_threshold
            inv_wall = -inv_penalty_coeff * (excess ** 2)

    # Terminal PnL bonus (convert PnL from dollars to ticks for unit consistency)
    terminal_bonus = 0.0
    if use_terminal_bonus and info.get("done", False):
        tick_size = float(getattr(mm, 'tick_size', 0.01))
        pnl_ticks = mm.total_pnl() / tick_size if tick_size > 0 else mm.total_pnl()
        terminal_bonus = pnl_ticks

    reward = spread_capture + inv_penalty + inv_wall + terminal_bonus

    # Store components in info for external logging (no-op if ignored)
    info["_reward_spread_capture"] = float(spread_capture)
    info["_reward_inv_penalty"] = float(inv_penalty)
    info["_reward_inv_wall"] = float(inv_wall)
    info["_reward_terminal_bonus"] = float(terminal_bonus)

    return float(reward)


def reward_pnl_dampened_inv_quadratic(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.01,
    # -- Inventory wall (optional) ---------------------------------
    use_inv_wall: bool = False,
    inv_wall_threshold: float = 0.0,
    # -- Terminal PnL bonus (optional) -----------------------------
    use_terminal_bonus: bool = False,
) -> float:
    """
    MM reward: dampened fill PnL - quadratic inventory penalty.

      r(t) = dampened_fill_term  -  phi*inv_after^2
             [ - phi*(|inv| - threshold)^2           if |inv| > threshold ]
             [ + PnL_in_ticks                        if done & bonus on   ]

    where

      dampened_fill_term = sum_f dq(f)*(mid_after - px(f))

    This keeps the same quadratic inventory penalty, optional inventory
    wall, and optional terminal PnL bonus used by
    reward_spread_capture_inv_quadratic(...), but revalues newly executed
    quantity immediately against ``mid_after``. That matches the same
    adverse-selection / favorable-selection logic of the paper-style
    dampened PnL reward, without adding a separate mark-to-market term for
    previously-carried inventory.

    All values are expressed in ticks.
    """

    # Shift handling (ticks)
    current_base = mm.base_price_idx
    shift = int(info.get("env_shift", 0))
    prev_base = current_base - shift
    mid_after = float(current_base + float(state_after["mid"]))

    # Dampened fill term: sum dq*(mid_after - px)
    spread_capture = 0.0
    # BUG FIX (M3): use env_fills (all fills) instead of env_filled (last only).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0.0))
        if qty <= 0:
            continue
        px = float(prev_base + float(f["px"]))
        side = int(f["side"])   # -1=ask fill (sold), +1=bid fill (bought)
        dq = (+qty if side == +1 else -qty)
        spread_capture += dq * (mid_after - px)

    # Quadratic inventory penalty
    inv_after = float(state_after["inventory"])
    inv_penalty = -inv_penalty_coeff * (inv_after ** 2)

    # Inventory wall: extra quadratic penalty beyond threshold (same phi as base)
    inv_wall = 0.0
    if use_inv_wall and inv_wall_threshold > 0.0:
        abs_inv = abs(inv_after)
        if abs_inv > inv_wall_threshold:
            excess = abs_inv - inv_wall_threshold
            inv_wall = -inv_penalty_coeff * (excess ** 2)

    # Terminal PnL bonus (convert PnL from dollars to ticks for unit consistency)
    terminal_bonus = 0.0
    if use_terminal_bonus and info.get("done", False):
        tick_size = float(getattr(mm, "tick_size", 0.01))
        pnl_ticks = mm.total_pnl() / tick_size if tick_size > 0 else mm.total_pnl()
        terminal_bonus = pnl_ticks

    # Store components in info for external logging
    info["_reward_spread_capture"] = float(spread_capture)
    info["_reward_mtm"] = 0.0
    info["_reward_inv_penalty"] = float(inv_penalty)
    info["_reward_inv_wall"] = float(inv_wall)
    info["_reward_terminal_bonus"] = float(terminal_bonus)

    return float(spread_capture + inv_penalty + inv_wall + terminal_bonus)


def reward_mtm_inv_quadratic(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.01,
    use_inv_wall: bool = False,
    inv_wall_threshold: float = 0.0,
    use_terminal_bonus: bool = False,
) -> float:
    """
    Backward-compatible alias.

    Historically this function included an additional ``inv_before * dmid``
    mark-to-market term. The active dampened-PnL variant now lives in
    ``reward_pnl_dampened_inv_quadratic`` and intentionally excludes that
    carried-inventory MtM component.
    """
    return reward_pnl_dampened_inv_quadratic(
        step_idx=step_idx,
        mm=mm,
        lob=lob,
        state_before=state_before,
        state_after=state_after,
        info=info,
        inv_penalty_coeff=inv_penalty_coeff,
        use_inv_wall=use_inv_wall,
        inv_wall_threshold=inv_wall_threshold,
        use_terminal_bonus=use_terminal_bonus,
    )



def reward_santa_fe_alpha_pure_v2(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # --- Santa Fe knobs (in ticks) ---
    inv_neutral_band: int = 6,
    inv_penalty_coeff: float = 0.001,          # penalizes excess^2 (ticks)
    carry_penalty_coeff: float = 0.50,        # penalizes |inv_before| per step (deterministic)
    symm_bonus: float = 0.01,                 # ticks
    # optional: small bonus for reducing inventory
    inv_reduction_coeff: float = 0.1,        # ticks per unit reduced
) -> float:
    """
    Reward aligned with "PnL from Market Making" in Santa Fe:
      - Main reward = edge (spread capture) measured vs mid_before
      - Penalize holding inventory via |inv_before| (deterministic, no dmid noise)
      - Penalize large inventory (above the neutral band)
      - Incentivize 2-sided quoting when neutral

    All values in ticks (numerically stable).
    """

    # -----------------------
    # 1) Shift handling (ticks)
    # -----------------------
    current_base = mm.base_price_idx
    shift = int(info.get("env_shift", 0))
    prev_base = current_base - shift

    mid_before = float(prev_base + float(state_before["mid"]))   # absolute ticks
    mid_after  = float(current_base + float(state_after["mid"])) # absolute ticks
    dmid = mid_after - mid_before

    # -----------------------
    # 2) Pure Alpha: edge vs mid_before
    #    edge = Σ dq*(mid_before - px)
    # -----------------------
    alpha_edge = 0.0
    # BUG FIX (M4): use env_fills (all fills) instead of env_filled (last only).
    fill_info = info.get("env_filled", None)
    fills = info.get("env_fills", [])
    if not fills and fill_info is not None:
        fills = [fill_info]

    for f in fills:
        qty = float(f.get("qty", 0.0))
        if qty <= 0:
            continue

        px = float(prev_base + float(f["px"]))  # absolute ticks (pre-shift base)
        side = int(f["side"])                   # -1=ask fill (we sold), +1=bid fill (we bought)
        dq = (+qty if side == +1 else -qty)

        # Pure spread capture vs mid_before
        alpha_edge += dq * (mid_before - px)

    # -----------------------
    # 3) Deterministic inventory holding cost
    #    Penalizes |inv_before| per step — no dmid noise.
    #    Old version used |inv * dmid| which mixed controllable (inv)
    #    with uncontrollable (dmid random walk), hurting credit assignment.
    # -----------------------
    inv_before = float(state_before["inventory"])
    carry_penalty = -carry_penalty_coeff * abs(inv_before)

    # -----------------------
    # 4) Large inventory penalty (soft, only above the band)
    # -----------------------
    inv_after = float(state_after["inventory"])
    excess = max(0.0, abs(inv_after) - float(inv_neutral_band))
    inv_penalty = -inv_penalty_coeff * (excess ** 2)

    # Optional: bonus for reducing exposure
    inv_reduction = max(0.0, abs(inv_before) - abs(inv_after))
    inv_reduction_bonus = inv_reduction_coeff * inv_reduction

    # -----------------------
    # 5) Symmetric quoting incentive when neutral
    # -----------------------
    has_bid = bool(state_after.get("has_bid", False))
    has_ask = bool(state_after.get("has_ask", False))
    symm_bonus_val = symm_bonus if (has_bid and has_ask and abs(inv_after) <= inv_neutral_band) else 0.0

    reward = alpha_edge + carry_penalty + inv_penalty + inv_reduction_bonus + symm_bonus_val
    return float(reward)

def reward_liquidation_value(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    # With 100× reward scaling and neutral band = inv_limit//2 = 3:
    #   |inv|=4 → 0.005/step (1.7× Δ_NAV),  |inv|=5 → 0.02/step (6.7×),
    #   |inv|=6 → 0.045/step (15×).
    # Strong enough to deter, but doesn't overwhelm spread capture signal.
    inv_penalty_coeff: float = 0.00005,
    # Neutral band: inventory below this threshold is NOT penalized.
    # Default None → auto = inv_limit // 2 (passed via inv_limit kwarg).
    # If inv_limit is also absent, falls back to 3.
    inv_neutral_band: Optional[int] = None,
    inv_limit: int = 6,
) -> float:
    """
    Reward function based on the Change in Immediate Liquidation Value (NAV Liquidation).
    
    Financial Logic:
    ----------------
    Calculates the Net Asset Value (NAV) assuming the agent executes a Market Order
    to close the entire position IMMEDIATELY.
    - If Long (+): Sell at Best Bid.
    - If Short (-): Buy back at Best Ask.
    
    Why this is optimal for Santa Fe (Random Walk):
    -----------------------------------------------
    1. Penalizes Widening Spreads: If the Bid drops (spread opens), the liquidation 
       value drops instantly. The agent learns: "Inventory with Wide Spread = Danger".
    2. Captures "Luck" and "Skill": If the market moves in favor or the spread tightens,
       the liquidation value rises. The agent receives the profit immediately.
    3. 1:1 Correlation with PnL: It is the most honest metric of "realizable cash".
    """
    
    tick_size = float(mm.tick_size)
    
    # -----------------------------------------------------------
    # 1. Grid Alignment (Shift Handling)
    # -----------------------------------------------------------
    # The simulator shifts the price grid to keep the agent centered.
    # We need absolute prices ($) to compare "before" vs "after" states accurately.
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # -----------------------------------------------------------
    # 2. Calculate Liquidation Value BEFORE (t)
    # -----------------------------------------------------------
    inv_t = float(state_before["inventory"])
    cash_t = float(state_before["cash"])
    
    # --- Reconstruct Absolute Bid/Ask at time t ---
    # FIX: Use direct best_bid/best_ask from state dict for robustness.
    # This is more accurate than reconstructing from mid/spread, especially
    # if the book is one-sided or mid-price is rounded.
    # FIX 2.2: Use has_bid/has_ask boolean flags instead of the -1
    # sentinel, which collides with valid negative indices under
    # dynamic grid shifting (where the mid sits near index 0).
    has_bid_t = state_before.get("has_bid", False)
    has_ask_t = state_before.get("has_ask", False)
    best_bid_t_idx = float(state_before.get("best_bid", 0))
    best_ask_t_idx = float(state_before.get("best_ask", 0))

    best_bid_t = (prev_base + best_bid_t_idx) * tick_size if has_bid_t else np.nan
    best_ask_t = (prev_base + best_ask_t_idx) * tick_size if has_ask_t else np.nan
    
    # Portfolio Value at t
    if inv_t > 0:
        # Long: Close by selling at Bid
        # If no bid exists, liquidation value of inventory is 0.
        liquidation_value_t = inv_t * best_bid_t if np.isfinite(best_bid_t) else 0.0
        nav_t = cash_t + liquidation_value_t
    elif inv_t < 0:
        # Short: Close by buying at Ask
        liquidation_cost_t = inv_t * best_ask_t if np.isfinite(best_ask_t) else 0.0 # inv_t is negative
        nav_t = cash_t + liquidation_cost_t
    else:
        # Flat
        nav_t = cash_t

    # -----------------------------------------------------------
    # 3. Calculate Liquidation Value AFTER (t+1)
    # -----------------------------------------------------------
    inv_t1 = float(state_after["inventory"])
    cash_t1 = float(state_after["cash"])
    
    # FIX 2.2: Use has_bid/has_ask flags (same fix as t).
    has_bid_t1 = state_after.get("has_bid", False)
    has_ask_t1 = state_after.get("has_ask", False)
    best_bid_t1_idx = float(state_after.get("best_bid", 0))
    best_ask_t1_idx = float(state_after.get("best_ask", 0))

    best_bid_t1 = (current_base + best_bid_t1_idx) * tick_size if has_bid_t1 else np.nan
    best_ask_t1 = (current_base + best_ask_t1_idx) * tick_size if has_ask_t1 else np.nan
    
    # Portfolio Value at t+1
    if inv_t1 > 0:
        liquidation_value_t1 = inv_t1 * best_bid_t1 if np.isfinite(best_bid_t1) else 0.0
        nav_t1 = cash_t1 + liquidation_value_t1
    elif inv_t1 < 0:
        liquidation_cost_t1 = inv_t1 * best_ask_t1 if np.isfinite(best_ask_t1) else 0.0
        nav_t1 = cash_t1 + liquidation_cost_t1
    else:
        nav_t1 = cash_t1

    # -----------------------------------------------------------
    # 4. Delta NAV (The Real PnL)
    # -----------------------------------------------------------
    delta_nav = nav_t1 - nav_t

    # -----------------------------------------------------------
    # 5. Residual Risk Penalty (Quadratic)
    # -----------------------------------------------------------
    # Penalizes inventory that exceeds the neutral band.
    # Default: inv_neutral_band = inv_limit // 2 (e.g. 6 // 2 = 3).
    _band = inv_neutral_band if inv_neutral_band is not None else (inv_limit // 2)
    excess_inv = max(0.0, abs(inv_t1) - float(_band))
    risk_penalty = inv_penalty_coeff * (excess_inv ** 2)

    # Final Reward
    return float(delta_nav - risk_penalty)

def reward_microstructure_sensitive(
    step_idx: int,
    mm,
    lob,
    state_before: dict,
    state_after: dict,
    info: dict,
    inv_penalty_coeff: float = 0.0005,
) -> float:
    """
    Reward based on Micro-Price (Mark-to-Micro).

    Similar to Liquidation Value, but instead of using the 'forced exit'
    price (Bid/Ask), it uses the 'future fair price' estimated from the
    book's order-flow imbalance.

    Concept:
    If we are LONG and the Bid is thin (low volume) while the Ask is thick:
    - The Mid-Price is still the average.
    - The Micro-Price drops towards the Bid (predicting a price decline).
    - The agent feels an immediate loss and learns to sell before the drop.
    """

    tick_size = float(mm.tick_size)
    current_base = mm.base_price_idx
    shift = info.get("env_shift", 0)
    prev_base = current_base - shift

    # --- Helper: Compute Absolute Micro-Price ---
    def get_micro_price(state, base_px):
        mid_idx = float(state["mid"])
        spread_ticks = float(state["spread"])

        # Retrieve raw volumes (undo log if needed, or use raw directly).
        # The MM state dict already has 'bidsize_raw' and 'asksize_raw' (float).
        bid_vol = float(state.get("bidsize_raw", 0.0))
        ask_vol = float(state.get("asksize_raw", 0.0))

        total_vol = bid_vol + ask_vol

        # Reconstruct absolute prices
        bid_px = base_px + mid_idx - (0.5 * spread_ticks)
        ask_px = base_px + mid_idx + (0.5 * spread_ticks)

        if total_vol <= 0:
            # No liquidity → Micro = Mid
            micro_idx = mid_idx + base_px
        else:
            # Imbalance = Buying Pressure (0 to 1)
            # If 1.0 (only Bid volume) → price will rise → Micro = Ask
            # If 0.0 (only Ask volume) → price will fall → Micro = Bid
            imbalance = bid_vol / total_vol

            # Formula using Imbalance (more readable)
            # Micro = Bid + (Spread * Imbalance)
            micro_idx_relative = (bid_px - base_px) + (spread_ticks * imbalance)
            micro_idx = base_px + micro_idx_relative

        return micro_idx * tick_size

    # 1. Wealth at t (using the old base)
    micro_t = get_micro_price(state_before, prev_base)
    inv_t = float(state_before["inventory"])
    cash_t = float(state_before["cash"])
    wealth_t = cash_t + (inv_t * micro_t)

    # 2. Wealth at t+1 (using the new base)
    micro_t1 = get_micro_price(state_after, current_base)
    inv_t1 = float(state_after["inventory"])
    cash_t1 = float(state_after["cash"])
    wealth_t1 = cash_t1 + (inv_t1 * micro_t1)

    # 3. Delta Wealth
    delta_wealth = wealth_t1 - wealth_t

    # 4. Soft Inventory Penalty
    # Since Micro-Price already penalizes "Microstructure Directional Risk",
    # the quadratic penalty only serves to prevent hitting the hard limit.
    # FIX: was hardcoded at 6.0 — now derived from mm.max_inventory / 2.
    _inv_limit = float(mm.max_inventory) if hasattr(mm, 'max_inventory') and mm.max_inventory else 12.0
    _band = _inv_limit / 2.0
    excess_inv = max(0.0, abs(inv_t1) - _band)
    risk_penalty = inv_penalty_coeff * (excess_inv ** 2)

    return float(delta_wealth - risk_penalty)
