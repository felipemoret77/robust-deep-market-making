#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOB_SIM_QRM.py — Queue-Reactive Model (QRM) LOB Simulator
==========================================================

Implementation of the Queue-Reactive Model introduced in:

    Huang, W., Lehalle, C.-A., & Rosenbaum, M. (2014).
    "Simulating and analyzing order book data: The queue-reactive model."
    arXiv:1312.0563v2 [q-fin.TR]

This module provides a LOB simulator whose event intensities are **state-dependent**,
calibrated from real market data (LOBSTER format), and whose reference price evolves
stochastically.  It is designed as a drop-in replacement for the zero-intelligence
Santa Fe engine (`LOB_SIM_SANTA_FE.py`) within the `MM_LOB_SIM.py` framework.

Architecture
------------
`LOB_simulation_QRM` **subclasses** `LOB_simulation` from `LOB_SIM_SANTA_FE.py`,
inheriting all queue mechanics:
    - lob_state array (signed depth per price level)
    - priorities_lob_state FIFO rank matrix
    - add_order_to_queue / remove_order_from_queue
    - center_lob_state (re-centering)
    - make_ob_snapshot_row (snapshot generation)
    - save_results (post-processing & CSV export)
    - cancel_filter / cancel_avoid_mask_provider (MM order protection)
    - execute_mm_market_order (aggressive MM crosses)

The only methods **overridden** are:
    - simulate_order()  — replaces the Santa Fe 3-event constant-rate exponential
      clocks (LO / MO / Cancel with uniform price placement) with the QRM 8-event
      state-dependent intensity model that operates at the best bid/ask levels.

Intensity Models
----------------
The paper defines three progressively richer intensity models:

    Model I  — Independent queues.  Intensities at each limit Qi depend ONLY
               on the queue size qi at that limit.  Bid/ask symmetric.  Birth-death
               process with closed-form invariant distribution.

    Model II — Dependent queues.
        (a)  Q±2 intensities additionally depend on whether Q±1 is empty or not
             (regime switch).  Quasi Birth-Death (QBD) structure.
        (b)  Q±1 intensities additionally depend on the opposite-side queue size
             q∓1, discretized into 4 regimes (empty / small / usual / large).

    Model III — The "Queue-Reactive Model" proper.  Any of the above intensity
                models PLUS stochastic reference-price dynamics:
                    - With probability θ, the reference price shifts ±1 tick
                      upon queue depletion or mid-limit insertion (FIRST).
                    - ONLY if pref shifted: with probability θ_reinit, the book
                      is redrawn from its model-based invariant distribution.

This implementation exposes two orthogonal flags:

    base_model : str  (passed as intensity_model for backward compat)
        "I"    — Model I: independent queues, λ depends only on own q_i.
                 Reinit uses analytic birth-death invariant (π⊗π).
        "IIA"  — Model IIa: Q_{±2} regime-switch on q_{±1} emptiness.
                 Reinit k=1: π⊗π; k=2: 2D MC conditional invariant.
        "IIB"  — Model IIb: Q_{±1} depends on opposite-side S_{m,l}(q_{∓1}).
                 Reinit k=1: MC joint invariant; k=2: 3D MC conditional.

    use_dynamic_pref : bool
        True  — Enable θ / θ_reinit reference-price dynamics (Model III layer).
                Per paper Section 3.1.1: pref shifts FIRST, then the book is
                redrawn (reinit/regen) around the NEW pref.  If pref does NOT
                shift (prob 1-θ), no redraw occurs.
        False — Static reference price; depletion is purely mechanical.

The paper uses Model I for its final simulations (Section 3.1: "Model I is used
to describe the LOB dynamics during periods when p_ref is constant").

QRM Event Types
---------------
The QRM defines 8 competing Poisson processes (one exponential clock each):

    Event 1 — Buy  Limit     at best bid      (increment BBSize)
    Event 2 — Buy  Cancel    at best bid      (decrement BBSize; may deplete)
    Event 3 — Sell Market    hitting best bid  (decrement BBSize; may deplete)
    Event 4 — Buy  MidLimit  inside spread     (tighten spread from bid side)
    Event 5 — Sell Limit     at best ask      (increment BASize)
    Event 6 — Sell Cancel    at best ask      (decrement BASize; may deplete)
    Event 7 — Buy  Market    hitting best ask  (decrement BASize; may deplete)
    Event 8 — Sell MidLimit  inside spread     (tighten spread from ask side)

Events 1 and 5 never trigger depletion.  Events 4 and 8 (mid-limit) always trigger
the reinit/regen pathway.  Events 2, 3, 6, 7 trigger it only when the best queue
is about to be fully consumed (depth ≤ 1).

Depletion Handling (θ / θ_reinit) — Paper Section 3.1.1
-------------------------------------------------------
When a depletion-triggering event occurs:

    1) With probability θ: **pref shift** ±1 tick (FIRST — paper order).
           Events 2, 3, 8  →  pref DOWN  (bid depletion or sell mid-limit)
           Events 4, 6, 7  →  pref UP    (ask depletion or buy mid-limit)

    2) ONLY IF pref shifted — with probability θ_reinit:
       **Reinitialization** — redraw k=1 from model-based invariant:
           Model I/IIA: birth-death product π⊗π (analytic).
           Model IIB:   Monte Carlo joint invariant.
           k=2..K: model-based conditional CDFs (IIA/IIB) or per-level invariant.

    3) ONLY IF pref shifted — with probability (1 − θ_reinit):
       **Deterministic regeneration** — neighbor-shift with AES_i/AES_j renormalization.
       New edge level sampled from per-level invariant.

Integration with MM_LOB_SIM.py
-------------------------------
`simulate_order()` accepts the same (lam, mu, delta) signature as the Santa Fe
engine but **ignores** those parameters when the QRM intensity table is available.
This means `simulate_LOB_with_MM()` can use `LOB_simulation_QRM` without any
modification — just swap the import.

The `last_event_dt` attribute is set identically, so clock advancement works.
All MM hooks (cancel_avoid_mask_provider, cancel_filter, on_rank_shift,
execute_mm_market_order) are inherited and work unchanged.

Dependencies
------------
    - LOB_SIM_SANTA_FE.LOB_simulation  (parent class)
    - numpy, pandas, typing, tqdm

Authors
-------
    Felipe Moret  (architecture, integration)
    Claude        (implementation)

References
----------
    [1] Huang, Lehalle, Rosenbaum (2014). "Simulating and analyzing order book
        data: The queue-reactive model." arXiv:1312.0563v2.
    [2] Cont, Stoikov, Talreja (2010). "A stochastic model for order book
        dynamics." Operations Research 58(3), 549–563.
    [3] Smith, Farmer, Gillemot, Krishnamurthy (2003). "Statistical theory of
        the continuous double auction." Quantitative Finance 3(6), 481–514.
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from LOB_SIM_SANTA_FE import LOB_simulation


# Bump when QRM simulator semantics change (useful to verify notebook reloads).
QRM_ENGINE_BUILD_TAG = "2026-04-02-aes-unit-debug3"


# =============================================================================
# ====                    LOB_simulation_QRM (subclass)                    ====
# =============================================================================

class LOB_simulation_QRM(LOB_simulation):
    """
    Queue-Reactive Model LOB simulator.

    Extends the Santa Fe zero-intelligence engine with state-dependent event
    intensities calibrated from market data.  The parent class provides all
    FIFO queue mechanics; this subclass overrides only the event-sampling and
    simulation logic.

    Parameters (beyond the parent class)
    -------------------------------------
    intens_val : pd.DataFrame
        Calibrated intensity table with columns:
            ['BBSize', 'BASize', 'Spread', 'Limit', 'Cancel', 'Market', 'MidLimit']
        Layout: Q² rows for Spread=1, then Q² rows for Spread=2 (where Q = size_q).
        The row ordering follows the MATLAB convention:
            BBSize = tile(1..Q, Q)  (cycles fast)
            BASize = repeat(1..Q, each=Q)  (changes in blocks)
        This is produced by `Intenses_py()` in `QRM_intesity_calib.py`.

    intens_val_bis : pd.DataFrame
        Table with columns ['BBSize', 'BASize'] used during reinitialization to
        sample new queue depths.  Has Q² rows in the same BASize-blocked ordering.

    statprob : np.ndarray
        Probability vector of length Q² for categorical sampling from `intens_val_bis`
        during reinitialization events.  Derived from empirical (BBSize, BASize)
        joint frequencies when Spread == 1.

    theta : float
        Probability of shifting the reference price (pref) by ±1 tick after a
        depletion or mid-limit event.  Typical range: [0.01, 0.20].
        From the paper: θ ∈ {0.05, ..., 1.0}, calibrated to match the empirical
        10-minute volatility and mean-reversion ratio.

    theta_reinit : float
        Probability that a depletion event triggers full reinitialization of the
        book from the invariant distribution (vs. deterministic regeneration).
        Typical range: [0.05, 0.30].
        From the paper: θ_reinit ∈ {0.0, ..., 1.0}, calibrated jointly with θ.

    tick_qrm : float
        Tick size in price units.  Used for reference-price arithmetic.  Must match
        the tick size used during intensity calibration.

    size_q : int
        Maximum queue depth index (Q).  Discretized depths are clamped to [1, Q].
        Inferred from `intens_val['BBSize'].max()` if not provided.

    size_s : int, default 2
        Maximum spread index (in ticks).  Spreads are clamped to [1, size_s].
        For "large tick" assets (spread almost always 1), size_s = 2 is standard.

    aes : float, default 1.0
        Average Event Size — the conversion factor between raw queue depth units
        (in the lob_state array, where each add_order_to_queue adds ±1) and the
        discretized QRM state.  If the calibration used a specific AES from market
        data, pass it here.  For synthetic sims with unit depth, use 1.0.

    intensity_model : str, default "I"
        Which base intensity model to use (stored internally as ``base_model``).
        Currently only ``"I"`` is faithfully implemented:
            "I"  — Independent queues: λ depends only on own q_i.
                   Uses ``intens_k1`` (Q×3 array). 6 clocks at k=1.
        IIA and IIB require pref-relative state and regime-conditioned
        calibration, which are not yet implemented.

    use_dynamic_pref : bool, default True
        Whether to enable stochastic reference-price dynamics (θ / θ_reinit).
        When False, depleted levels simply disappear and the spread widens
        mechanically, with no reinitialization or preference shift.
        When True, per paper Section 3.1.1: pref shifts first (with
        probability θ), then the book is redrawn around the new pref
        (reinit with probability θ_reinit, regen otherwise).
        If pref does NOT shift (prob 1-θ), no redraw occurs.
    """

    # ========================== lifecycle ==========================

    def __init__(
        self,
        # --- Parent (Santa Fe) parameters ---
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

        # --- QRM-specific parameters ---
        intens_val: Optional[pd.DataFrame] = None,
        intens_val_bis: Optional[pd.DataFrame] = None,
        statprob: Optional[np.ndarray] = None,
        theta: float = 0.05,
        theta_reinit: float = 0.10,
        tick_qrm: float = 0.01,
        size_q: Optional[int] = None,
        size_s: int = 2,
        aes: float = 1.0,
        intensity_model: str = "I",
        use_dynamic_pref: bool = True,
        intens_levels: Optional[np.ndarray] = None,
        intens_k1: Optional[np.ndarray] = None,
        # Model IIA specific
        intens_k2_q1pos: Optional[np.ndarray] = None,
        intens_k2_q1zero: Optional[np.ndarray] = None,
        # Model IIB specific
        intens_k1_by_regime: Optional[np.ndarray] = None,
        iib_m: Optional[int] = None,
        iib_l: Optional[int] = None,
        # MidLimit rate (dynamic pref layer, fires at spread >= 2)
        midlimit_rate: float = 0.0,
        # Model-specific reinit data (IIA/IIB k=2 depth distributions)
        reinit_k2_data: Optional[dict] = None,
        # AES per level for regen renormalization
        aes_by_level: Optional[np.ndarray] = None,
        # IIB joint invariant from Monte Carlo
        iib_joint_invariant: Optional[dict] = None,
        # Model-based k=1 invariant for I/IIA reinit
        model_k1_invariant: Optional[dict] = None,
        # Per-level queue depth cap
        size_q_by_level: Optional[np.ndarray] = None,
    ):
        # ------------------------------------------------------------------
        # 1. Initialize the parent (Santa Fe) engine.
        #    This sets up lob_state, priorities_lob_state, message_dict,
        #    ob_dict, cancel filter infrastructure, EWMA accumulators, etc.
        # ------------------------------------------------------------------
        super().__init__(
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

        # ------------------------------------------------------------------
        # 2. Store QRM calibration data.
        # ------------------------------------------------------------------

        # Intensity table: rows indexed by (BBSize, BASize, Spread).
        # Columns of interest: 'Limit', 'Cancel', 'Market', 'MidLimit'.
        self.intens_val: Optional[pd.DataFrame] = intens_val

        # (BBSize, BASize) pairs for reinitialization sampling.
        self.intens_val_bis: Optional[pd.DataFrame] = intens_val_bis

        # Categorical probability vector for reinit.
        self.statprob: Optional[np.ndarray] = statprob

        # ------------------------------------------------------------------
        # 3. QRM dynamics parameters.
        # ------------------------------------------------------------------

        # Probability of shifting the reference price after depletion/midlimit.
        # Guard: if None is passed (uncalibrated), fall back to paper-typical defaults.
        self.theta: float = float(theta) if theta is not None else 0.05

        # Probability of full reinitialization (vs. deterministic regeneration).
        self.theta_reinit: float = float(theta_reinit) if theta_reinit is not None else 0.10

        # Tick size in absolute price units (used for pref arithmetic).
        self.tick_qrm: float = float(tick_qrm)

        # Callback for forced cancellation of external (MM) orders during
        # reinit/regen.  When the QRM redraws the book, any resting orders
        # from the MM are silently destroyed.  If this callback is set, the
        # engine notifies the MM BEFORE clearing, so the MM can cancel
        # its orders explicitly (correct bookkeeping) rather than discovering
        # the loss via _prune_orphan_orders() one step later.
        #
        # Signature: on_forced_cancel(prices: Optional[List[int]], reason: str)
        #   prices=None means the ENTIRE book is being cleared (reinit)
        #   prices=[px, ...] means specific levels are being cleared (regen/cross-fix)
        #   reason: "qrm_reinit", "qrm_level_clear", "qrm_cross_fix"
        self.on_forced_cancel = None

        # Per-step reinit/regen event tracking for logging.
        # Reset at the start of each simulate_order() call.
        self._last_reinit_regen: Optional[str] = None  # "reinit" or "regen" or None

        # Maximum queue depth index (Q).  Inferred from data if not provided.
        if size_q is not None:
            self.size_q: int = int(size_q)
        elif intens_val is not None:
            self.size_q = int(intens_val["BBSize"].max())
        else:
            self.size_q = 10  # safe default

        # Maximum spread index (typically 2 for large-tick assets).
        self.size_s: int = int(size_s)

        # Average Event Size — conversion factor between raw lob_state depth
        # (integer unit counts) and the discretized QRM state (1..Q).
        self.aes: float = float(max(aes, 1e-9))

        # ------------------------------------------------------------------
        # 4. Model selection flags.
        # ------------------------------------------------------------------

        # Base model: "I" (independent queues), "IIA" (Q2 regime-switch
        # on q1 emptiness), or "IIB" (Q1 depends on opposite-side regime).
        # The dynamic pref layer (Model III) is orthogonal and controlled
        # by use_dynamic_pref.  Any base model can be combined with it.
        self.base_model: str = str(intensity_model).upper().strip()
        if self.base_model not in ("I", "IIA", "IIB"):
            raise ValueError(
                f"base_model must be 'I', 'IIA', or 'IIB', got '{self.base_model}'"
            )

        # Whether to enable stochastic reference-price dynamics (θ / θ_reinit).
        self.use_dynamic_pref: bool = bool(use_dynamic_pref)

        # ------------------------------------------------------------------
        # 5. QRM internal state.
        # ------------------------------------------------------------------

        # Reference price (pref) — tracked in **grid-index space** (not
        # absolute price).  Initialized to the grid center (same as p0's
        # pref_half: reference price in HALF-TICK units (integer).
        #
        # The paper defines p_ref at the midpoint of best bid/ask, which
        # falls at a half-tick when spread=1.  Storing pref as a float
        # grid-index leads to ambiguous round(pref ± 0.5) arithmetic.
        #
        # Half-tick representation eliminates this:
        #   pref_half = 2 * grid_index_of_pref
        #   Q_{-1} is at grid index (pref_half - 1) // 2
        #   Q_{+1} is at grid index (pref_half + 1) // 2
        #   Q_{-k} is at grid index (pref_half - (2k-1)) // 2
        #   Q_{+k} is at grid index (pref_half + (2k-1)) // 2
        #
        # A pref shift of ±1 tick = ±2 half-ticks.
        # center_lob_state() shifts by N ticks = 2N half-ticks.
        # Parent initialization creates bids on [0, half-1] and asks on [half, ...].
        # For spread=1 at startup, best bid is (half-1) and best ask is (half),
        # so pref must be exactly between them: (half - 0.5).
        _half = number_tick_levels // 2
        _center_bid = _half - 1
        self.pref_half: int = 2 * _center_bid + 1  # odd half-tick index
        # Keep float pref for backward compat (logging, reinit noise)
        self.pref: float = float(_center_bid) + 0.5

        # ------------------------------------------------------------------
        # 6. Pre-compute intensity lookup arrays for speed.
        #    The DataFrame .loc / .iloc access on every event is slow.
        #    We extract the 4 intensity columns as a contiguous numpy array
        #    for O(1) indexed lookup.
        # ------------------------------------------------------------------
        self._intens_array: Optional[np.ndarray] = None  # shape (N_rows, 4)
        if self.intens_val is not None:
            self._intens_array = self.intens_val[
                ["Limit", "Cancel", "Market", "MidLimit"]
            ].to_numpy(dtype=np.float64).copy()

        # ------------------------------------------------------------------
        # 7. Model I intensity array for k=1 (simple λ(q), no cross-queue).
        # ------------------------------------------------------------------
        self._intens_k1: Optional[np.ndarray] = None
        if intens_k1 is not None:
            self._intens_k1 = np.asarray(intens_k1, dtype=np.float64).copy()

        # ------------------------------------------------------------------
        # 8. Multi-level intensity cube (k=2..K) for deeper price levels.
        # ------------------------------------------------------------------
        self._intens_levels: Optional[np.ndarray] = None
        self._n_deep_levels: int = 0
        self._n_deep_table_levels: int = 0
        if intens_levels is not None:
            self._intens_levels = np.asarray(intens_levels, dtype=np.float64).copy()
            self._n_deep_table_levels = int(self._intens_levels.shape[0])

        # ------------------------------------------------------------------
        # 8b. Model IIA: Q2 regime-switch tables (q1>0 vs q1=0).
        # ------------------------------------------------------------------
        self._intens_k2_q1pos: Optional[np.ndarray] = None
        self._intens_k2_q1zero: Optional[np.ndarray] = None
        if intens_k2_q1pos is not None:
            self._intens_k2_q1pos = np.asarray(intens_k2_q1pos, dtype=np.float64).copy()
        if intens_k2_q1zero is not None:
            self._intens_k2_q1zero = np.asarray(intens_k2_q1zero, dtype=np.float64).copy()

        # For IIA/IIB, k=2 is provided by regime-switch tables and
        # intens_levels starts at k=3. Keep explicit bookkeeping so
        # event block <-> level mapping is correct across all models.
        self._has_regime_k2: bool = (
            self.base_model in ("IIA", "IIB")
            and self._intens_k2_q1pos is not None
            and self._intens_k2_q1zero is not None
        )
        self._n_deep_levels = int(
            (1 if self._has_regime_k2 else 0) + self._n_deep_table_levels
        )

        # ------------------------------------------------------------------
        # 8c. Model IIB: Q1 opposite-regime tables (4 regimes).
        # ------------------------------------------------------------------
        self._intens_k1_by_regime: Optional[np.ndarray] = None
        self._iib_m: int = 0
        self._iib_l: int = 0
        if intens_k1_by_regime is not None:
            self._intens_k1_by_regime = np.asarray(intens_k1_by_regime, dtype=np.float64).copy()
            self._iib_m = int(iib_m) if iib_m is not None else 0
            self._iib_l = int(iib_l) if iib_l is not None else 0

        # ------------------------------------------------------------------
        # 8d. MidLimit rate (scalar, fires at spread >= 2 when use_dynamic_pref).
        # ------------------------------------------------------------------
        # Per paper Section 3.1.1, mid-limit insertions inside the spread are
        # one of three triggers for reference-price changes.  The rate is
        # calibrated as a single scalar (not depth-dependent) from events
        # where a limit order tightened the spread from 2 to 1 tick.
        self._midlimit_rate: float = float(midlimit_rate)

        # ------------------------------------------------------------------
        # 8e. Model-specific reinit data (IIA/IIB: k=2 depth CDFs).
        # ------------------------------------------------------------------
        self._reinit_k2_data: Optional[dict] = reinit_k2_data

        # ------------------------------------------------------------------
        # 8f. AES per level for regen renormalization (paper Section 3.1.1).
        # ------------------------------------------------------------------
        # When queue Q_i becomes Q_j after a pref shift, the depth should
        # be renormalized by AES_i / AES_j.
        if aes_by_level is not None:
            self._aes_by_level = np.asarray(aes_by_level, dtype=np.float64).copy()
        else:
            self._aes_by_level = None

        # ------------------------------------------------------------------
        # 8g. IIB joint invariant from Monte Carlo (paper Section 2.4.5).
        # ------------------------------------------------------------------
        # The joint distribution of (q_{-1}, q_{+1}) under Model IIB is
        # analytically intractable because of the cross-dependence via
        # S_{m,l}(q_opp).  It is estimated by Monte Carlo simulation of
        # the 2D birth-death process during calibration.
        self._iib_joint_invariant: Optional[dict] = iib_joint_invariant
        self._model_k1_invariant: Optional[dict] = model_k1_invariant

        # Per-level queue depth cap (size_q_by_level[k-1] for level k).
        if size_q_by_level is not None:
            self._size_q_by_level = np.asarray(size_q_by_level, dtype=int).copy()
        else:
            self._size_q_by_level = None

        # ------------------------------------------------------------------
        # 9. QRM event unit (shares per +1/-1 jump in discretized q-space)
        # ------------------------------------------------------------------
        # QRM states are calibrated in AES units: q = ceil(raw / AES_k).
        # Using Santa-Fe fixed event sizes here creates a systematic shift when
        # mean_size_LO != AES_k (e.g. q=1 becomes q=2).  Keep transitions in the
        # same unit system as calibration (AES per level).
        self._lo_event_size: float = float(self._aes_for_level(1))
        self._mo_event_size: float = float(self._aes_for_level(1))

        # Debug counters (IIB diagnostics).
        self._debug_sampled_events: int = 0
        self._debug_iib_fallback00_events: int = 0
        self._debug_iib_regime_bid_counts = np.zeros(4, dtype=np.int64)
        self._debug_iib_regime_ask_counts = np.zeros(4, dtype=np.int64)
        self._debug_last_iib_regime_bid: int = -1
        self._debug_last_iib_regime_ask: int = -1
        self._debug_last_iib_used_fallback00: bool = False

        # ------------------------------------------------------------------
        # 10. Precompute invariant distributions for deep levels (k=2..K).
        #     Used by _reinit_book (to sample all K levels) and _regen_book
        #     (to sample the new edge level after a price shift).
        #     Per paper Section 2.3.3, π_i(n) is computed from the birth/death
        #     ratio ρ_i of each level's intensity functions.
        # ------------------------------------------------------------------
        self._deep_inv_cdfs: Optional[list] = None
        self._precompute_deep_invariants()

    # ==================== deep-level invariant distributions ====================

    def _precompute_deep_invariants(self) -> None:
        """Precompute invariant CDFs for each deep level using Model I formula.

        For each deep level ki (0-indexed, corresponding to k=ki+2), we have
        intensities λ^L(q), λ^C(q), λ^M(q) from ``intens_levels[ki, :, :]``.

        The invariant distribution of the birth-death chain is:
            π(0) = 1  (unnormalized)
            π(n) = π(n-1) * b(n-1) / d(n)
        where b(n) = limit arrival rate at depth n, d(n) = cancel + market rate
        at depth n.
        """
        if self._intens_levels is None:
            self._deep_inv_cdfs = None
            return

        n_deep = self._intens_levels.shape[0]
        # intens_levels shape: (n_deep, Q_size, 3)
        # The table has Q_size rows indexed 0..Q_size-1.
        # The invariant distribution covers states 0..Q_size-1 (Q_size states).
        # State n corresponds to row n in the table.
        Q_size = self._intens_levels.shape[1]
        Q_max = Q_size - 1  # maximum valid state index
        self._deep_inv_cdfs = []

        for ki in range(n_deep):
            # States 0..Q_max
            pi = np.zeros(Q_size)
            pi[0] = 1.0

            for n in range(Q_max):
                # Birth rate at state n (n → n+1):
                b_n = self._intens_levels[ki, n, 0]

                # Death rate at state n+1 (n+1 → n):
                d_n1 = (self._intens_levels[ki, n + 1, 1]
                        + self._intens_levels[ki, n + 1, 2])

                if d_n1 < 1e-12:
                    break  # absorbing — truncate distribution here

                pi[n + 1] = pi[n] * b_n / d_n1

            # Normalize
            total = pi.sum()
            if total > 0:
                pi /= total
            else:
                pi[1] = 1.0  # fallback: depth = 1 AES

            cdf = np.cumsum(pi)
            self._deep_inv_cdfs.append(cdf)

    def _sample_deep_level_depth_aes(self, ki: int) -> int:
        """Sample queue depth (in AES units) from invariant distribution of
        deep-table index ``ki`` (0-indexed over self._intens_levels).

        Returns 0 if the level should be empty, otherwise 1..Q.
        """
        if self._deep_inv_cdfs is None or ki >= len(self._deep_inv_cdfs):
            return 1  # fallback
        cdf = self._deep_inv_cdfs[ki]
        u = float(self.rng.random_sample())
        return int(np.searchsorted(cdf, u))

    def _deep_table_idx_for_level(self, k: int) -> Optional[int]:
        """Map queue level k to row-index in self._intens_levels.

        - Model I: intens_levels starts at k=2  -> idx = k-2
        - IIA/IIB: intens_levels starts at k=3 -> idx = k-3 (k=2 uses
          regime-switch tables, not intens_levels)
        """
        if self._intens_levels is None:
            return None
        start_k = 3 if self._has_regime_k2 else 2
        idx = int(k - start_k)
        if idx < 0 or idx >= self._intens_levels.shape[0]:
            return None
        return idx

    def _sample_depth_aes_for_level(self, k: int) -> int:
        """Sample depth (AES units) for queue level k from deep invariants."""
        idx = self._deep_table_idx_for_level(k)
        if idx is None:
            return 1
        return self._sample_deep_level_depth_aes(idx)

    # ==================== pref-relative helpers ====================

    def _queue_grid_idx(self, queue_idx: int) -> int:
        """
        Return the lob_state grid index for pref-relative queue Q_{queue_idx}.

        Uses half-tick arithmetic:
            Q_{-k} at grid index (pref_half - (2k-1)) // 2
            Q_{+k} at grid index (pref_half + (2k-1)) // 2

        Parameters
        ----------
        queue_idx : int
            Signed queue index: -1 = best bid, +1 = best ask, -2, +2, etc.

        Returns
        -------
        int
            Grid index into lob_state.  May be out of bounds.
        """
        if queue_idx == 0:
            return self.pref_half // 2
        sign = 1 if queue_idx > 0 else -1
        k = abs(queue_idx)
        half_offset = sign * (2 * k - 1)
        return (self.pref_half + half_offset) // 2

    def _queue_depth(self, queue_idx: int) -> int:
        """
        Return the discretized depth at pref-relative queue Q_{queue_idx}.

        Uses AES_k for level k = |queue_idx| if aes_by_level is available,
        otherwise falls back to the global AES.

        Returns 0 if empty or out of bounds.
        """
        px = self._queue_grid_idx(queue_idx)
        if px < 0 or px >= self.number_tick_levels:
            return 0
        raw = float(abs(self.lob_state[px]))
        if raw < 1e-10:
            return 0
        k = abs(queue_idx)
        if self._aes_by_level is not None and k >= 1:
            aes_k = self._aes_by_level[min(k - 1, len(self._aes_by_level) - 1)]
        else:
            aes_k = self.aes
        return int(np.clip(int(np.ceil(raw / aes_k)), 1, self._qcap_for_level(k)))

    # ==================== QRM state extraction ====================

    def _current_qrm_state(self) -> Tuple[int, int, int]:
        """
        Extract the current (q_{-1}, q_{+1}, spread) triple using
        pref-relative half-tick arithmetic.

        Q_{-1} and Q_{+1} are at fixed grid positions determined by
        pref_half.  The spread is the distance between the first
        OCCUPIED bid and ask queues (not the fixed Q_{-1}/Q_{+1} distance,
        which is always 1 tick).

        When Q_{-1} is empty, the effective best bid is Q_{-2} or deeper,
        and the spread widens accordingly.  This is critical for the
        MidLimit clock: it only fires when spread >= 2.

        Returns
        -------
        (bb, ba, sp) : Tuple[int, int, int]
            Discretized depths at Q_{-1} (bid) and Q_{+1} (ask), and spread.
            q=0 when the queue is empty.  sp clamped to [1, size_s].
        """
        bb = self._queue_depth(-1)  # Q_{-1}
        ba = self._queue_depth(+1)  # Q_{+1}

        # Spread: distance between first occupied bid and ask queues.
        # k_bid = smallest k such that Q_{-k} > 0
        # k_ask = smallest k such that Q_{+k} > 0
        # spread = k_bid + k_ask - 1  (in ticks)
        K = 1 + self._n_deep_levels
        k_bid = 0
        for k in range(1, K + 1):
            if self._queue_depth(-k) > 0:
                k_bid = k
                break
        k_ask = 0
        for k in range(1, K + 1):
            if self._queue_depth(+k) > 0:
                k_ask = k
                break

        if k_bid > 0 and k_ask > 0:
            sp = k_bid + k_ask - 1
        else:
            sp = 1  # fallback if one side is completely empty

        sp = int(np.clip(sp, 1, self.size_s))
        return bb, ba, sp

    # ==================== multi-level helpers ====================

    def _depth_at_level(self, k: int, side: int) -> Tuple[int, int]:
        """Return (price_idx, discretized_depth) for the k-th pref-relative level.

        Uses pref_half arithmetic:
            side=+1 (bid): Q_{-k} at (pref_half - (2k-1)) // 2
            side=-1 (ask): Q_{+k} at (pref_half + (2k-1)) // 2

        Returns (-1, 0) if the level is out of bounds.
        """
        queue_idx = -k if side == +1 else +k
        px = self._queue_grid_idx(queue_idx)

        if px < 0 or px >= self.number_tick_levels:
            return -1, 0

        raw = float(abs(self.lob_state[px]))
        if raw < 1e-10:
            return px, 0
        # Use per-level AES if available
        if self._aes_by_level is not None and k >= 1:
            aes_k = self._aes_by_level[min(k - 1, len(self._aes_by_level) - 1)]
        else:
            aes_k = self.aes
        q = int(np.clip(int(np.ceil(raw / aes_k)), 1, self._qcap_for_level(k)))
        return px, q

    # ==================== intensity lookup ====================

    def _sample_qrm_event(self) -> Tuple[int, float]:
        """
        Sample the next QRM event via competing exponential clocks.

        For k=1: 8 clocks (limit/cancel/market/midlimit × bid/ask) using the
        full (BBSize, BASize, Spread)-dependent intensity table.

        For k=2..K (if ``intens_levels`` is provided): 6 clocks per level
        (limit/cancel/market × bid/ask, no midlimit) using simpler
        depth-only intensities.

        Event index encoding (1-based):
            1-8   : k=1 events (unchanged QRM convention)
            9-14  : k=2 (bid: limit/cancel/market, ask: limit/cancel/market)
            15-20 : k=3, etc.

        Returns
        -------
        (event_idx, dt) : Tuple[int, float]
        """
        # --- k=1 clocks ---
        n_deep = self._n_deep_levels

        bb, ba, _sp = self._current_qrm_state()

        # When use_dynamic_pref is active: 8 clocks at k=1 (6 base + 2 MidLimit).
        # MidLimit clocks (buy=clock 7, sell=clock 8) only fire when spread >= 2.
        # When use_dynamic_pref is off: 6 clocks (no MidLimit — spread changes
        # are purely mechanical, no p_ref dynamics).
        _use_midlimit = (self.use_dynamic_pref and _sp >= 2
                         and self._midlimit_rate > 0)
        k1_clocks = 8 if _use_midlimit else 6
        total_clocks = k1_clocks + 6 * n_deep
        all_lambdas = np.zeros(total_clocks, dtype=np.float64)
        self._debug_last_iib_regime_bid = -1
        self._debug_last_iib_regime_ask = -1
        self._debug_last_iib_used_fallback00 = False
        _used_fallback00 = False

        def _lookup_k1(q: int, table: np.ndarray) -> np.ndarray:
            """Look up (limit, cancel, market) from a table indexed 0..Q."""
            q_idx = min(q, table.shape[0] - 1)
            if q == 0:
                # Empty queue: only limit fires (bootstrap)
                return np.array([table[q_idx, 0], 0.0, 0.0])
            return table[q_idx].copy()

        if self.base_model == "I":
            # Model I: intensities depend only on own queue depth.
            for side_idx, q in enumerate([bb, ba]):
                offset = side_idx * 3
                rates = _lookup_k1(q, self._intens_k1)
                all_lambdas[offset:offset + 3] = rates

        elif self.base_model == "IIA":
            # Model IIA: Q1 same as Model I, Q2 regime-switches on q1>0.
            # Here we only set k=1 clocks. k=2 is handled below with
            # the regime-switch tables.
            for side_idx, q in enumerate([bb, ba]):
                offset = side_idx * 3
                rates = _lookup_k1(q, self._intens_k1)
                all_lambdas[offset:offset + 3] = rates

        elif self.base_model == "IIB":
            # Model IIB: Q1 rates depend on own queue AND opposite regime.
            # S_{m,l} discretization of opposite-side queue.
            for side_idx, (q_own, q_opp) in enumerate([(bb, ba), (ba, bb)]):
                offset = side_idx * 3
                # Robust bootstrap fallback: when both best queues are empty,
                # use dense Model-I q=0 rates if available.
                if q_own == 0 and q_opp == 0 and self._intens_k1 is not None:
                    _used_fallback00 = True
                    rates = _lookup_k1(q_own, self._intens_k1)
                    all_lambdas[offset:offset + 3] = rates
                    if side_idx == 0:
                        self._debug_last_iib_regime_bid = 0
                        self._debug_iib_regime_bid_counts[0] += 1
                    else:
                        self._debug_last_iib_regime_ask = 0
                        self._debug_iib_regime_ask_counts[0] += 1
                    continue
                # Compute opposite-side regime
                if q_opp == 0:
                    regime = 0  # Q^0
                elif q_opp <= self._iib_m:
                    regime = 1  # Q^-
                elif q_opp <= self._iib_l:
                    regime = 2  # Q_bar
                else:
                    regime = 3  # Q^+
                if side_idx == 0:
                    self._debug_last_iib_regime_bid = int(regime)
                    self._debug_iib_regime_bid_counts[int(regime)] += 1
                else:
                    self._debug_last_iib_regime_ask = int(regime)
                    self._debug_iib_regime_ask_counts[int(regime)] += 1
                # Look up from the 4-regime table
                if self._intens_k1_by_regime is not None:
                    q_idx = min(q_own, self._intens_k1_by_regime.shape[0] - 1)
                    if q_own == 0:
                        all_lambdas[offset + 0] = self._intens_k1_by_regime[q_idx, regime, 0]
                    else:
                        all_lambdas[offset:offset + 3] = self._intens_k1_by_regime[q_idx, regime, :]
                else:
                    # Fallback to Model I if no regime table
                    rates = _lookup_k1(q_own, self._intens_k1)
                    all_lambdas[offset:offset + 3] = rates

        # MidLimit clocks: buy midlimit at slot 6, sell midlimit at slot 7
        # (0-indexed).  These correspond to events 4 and 8 in the 8-clock
        # QRM convention after remapping.
        if _use_midlimit:
            # Per paper: mid-limit rate is symmetric (same for buy and sell).
            # Each side gets half the total mid-limit rate.
            all_lambdas[6] = self._midlimit_rate / 2.0  # buy midlimit
            all_lambdas[7] = self._midlimit_rate / 2.0  # sell midlimit

        k1_offset = k1_clocks

        # --- k=2..K clocks ---
        # For Model IIA/IIB at k=2: regime-switch on q1 emptiness.
        # For k>=3 and Model I: independent, depth-only intensities.
        if n_deep > 0:
            for ki in range(n_deep):
                k = ki + 2
                offset = k1_offset + ki * 6

                # Determine which intensity table to use for this level.
                # Model IIA/IIB at k=2: use regime-switch tables.
                # Otherwise: use the standard intens_levels cube.
                _use_regime_k2 = (
                    k == 2
                    and self.base_model in ("IIA", "IIB")
                    and self._intens_k2_q1pos is not None
                    and self._intens_k2_q1zero is not None
                )

                for side_offset, side in [(0, +1), (3, -1)]:
                    px, q_level = self._depth_at_level(k, side=side)
                    if px < 0:
                        continue  # out of bounds

                    if _use_regime_k2:
                        # IIA/IIB: k=2 rates depend on whether q1 > 0.
                        # Get q1 for the same side.
                        q1_side = bb if side == +1 else ba
                        table = (self._intens_k2_q1pos if q1_side > 0
                                 else self._intens_k2_q1zero)
                        q_idx = min(q_level, table.shape[0] - 1)
                        if q_level == 0:
                            all_lambdas[offset + side_offset] = table[q_idx, 0]
                        else:
                            all_lambdas[offset + side_offset:offset + side_offset + 3] = table[q_idx]
                    else:
                        # Model I or k>=3: standard depth-only lookup.
                        table_idx = self._deep_table_idx_for_level(k)
                        if table_idx is None or self._intens_levels is None:
                            continue
                        q_idx = min(q_level, self._intens_levels.shape[1] - 1)
                        if q_level == 0:
                            all_lambdas[offset + side_offset] = self._intens_levels[table_idx, q_idx, 0]
                        elif q_level > 0:
                            all_lambdas[offset + side_offset:offset + side_offset + 3] = self._intens_levels[table_idx, q_idx]

        # --- Sample competing exponentials ---
        times = np.full(total_clocks, np.inf, dtype=np.float64)
        pos = all_lambdas > 0
        if pos.any():
            times[pos] = self.rng.exponential(scale=1.0 / all_lambdas[pos])

        tmin = float(np.min(times))
        raw_idx = int(np.argmin(times))  # 0-based index into all_lambdas

        # --- Remap to final 8-clock QRM convention ---
        # The executor always uses the paper's 8-event encoding:
        #   1=LO bid, 2=Cancel bid, 3=MO bid, 4=Buy MidLimit,
        #   5=LO ask, 6=Cancel ask, 7=MO ask, 8=Sell MidLimit,
        #   9-14=k=2 events, 15-20=k=3 events, etc.
        #
        # Sampler layout:
        #   If 8 clocks: slots 0-5 = base L/C/M × bid/ask,
        #                slots 6-7 = buy/sell midlimit
        #   If 6 clocks: slots 0-5 = base L/C/M × bid/ask (no midlimit)
        #   Deep slots start at k1_offset.
        if raw_idx < k1_offset:
            # k=1 event
            if k1_clocks == 8:
                # 8-clock layout: 0-2=bid L/C/M, 3-5=ask L/C/M, 6=buy mid, 7=sell mid
                _SLOT_TO_EVENT_8 = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 6:4, 7:8}
                event_idx = _SLOT_TO_EVENT_8[raw_idx]
            else:
                # 6-clock layout: 0-2=bid L/C/M, 3-5=ask L/C/M
                _SLOT_TO_EVENT_6 = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7}
                event_idx = _SLOT_TO_EVENT_6[raw_idx]
        else:
            # Deep level event: map to 9+ encoding
            deep_slot = raw_idx - k1_offset  # 0-based within deep clocks
            event_idx = 9 + deep_slot  # 9, 10, 11, ... (1-based)

        self._debug_sampled_events += 1
        self._debug_last_iib_used_fallback00 = bool(_used_fallback00)
        if _used_fallback00:
            self._debug_iib_fallback00_events += 1

        self.last_event_dt = tmin
        return event_idx, tmin

    # ==================== event execution ====================

    def _execute_qrm_event(
        self, event_idx: int
    ) -> Tuple[int, int, int, int, bool]:
        """
        Execute one QRM event on the LOB.

        Dispatches to the appropriate handler based on the event index.

        For Model III (8 k=1 clocks): indices 1-8 for k=1, 9+ for k>=2.
        For Model I  (6 k=1 clocks): indices 1-6 for k=1, 7+ for k>=2.
            Model I mapping: 1=LO bid, 2=Cancel bid, 3=MO bid,
                             4=LO ask, 5=Cancel ask, 6=MO ask.

        Returns
        -------
        (order_type, order_sign, order_price, executed_size, depleted)
        """
        # No remapping needed — the sampler (_sample_qrm_event) now returns
        # event indices in the final 8-clock QRM convention directly:
        #   1-8   = k=1 events (L/C/M × bid/ask + MidLimit)
        #   9-14  = k=2 events (L/C/M × bid/ask)
        #   15-20 = k=3, etc.

        best_bid, best_ask = self._best_bid_ask_indices()

        # -------------------------------------------------------------------
        # Events 1-3: Bid side (Q_{-1})
        # Events 5-7: Ask side (Q_{+1})
        # Events 4, 8: MidLimit inside spread
        #
        # CRITICAL: events execute at the pref-relative positions Q_{-1}
        # and Q_{+1}, NOT at the current best bid/ask.  During constant-pref
        # periods these coincide, but after depletion (before pref shifts)
        # they may differ.  Using Q_{±1} ensures consistency with the
        # calibration, which estimated intensities at these positions.
        # -------------------------------------------------------------------
        px_q_neg1 = self._queue_grid_idx(-1)  # Q_{-1} grid position
        px_q_pos1 = self._queue_grid_idx(+1)  # Q_{+1} grid position

        # Event 1: Limit order at Q_{-1} (bid)
        if event_idx == 1:
            sz, dep = self._apply_limit_at_queue(px_q_neg1, side=+1)
            return (0, +1, px_q_neg1, sz, dep)

        # Event 2: Cancel at Q_{-1} (bid), may deplete
        elif event_idx == 2:
            sz, dep = self._apply_cancel_at_queue(px_q_neg1, side=+1, level=1)
            return (2, +1, px_q_neg1, sz, dep)

        # Event 3: Sell MO hitting Q_{-1} (consumes bid)
        elif event_idx == 3:
            sz, dep = self._apply_market_at_queue(px_q_neg1, side=+1, level=1)
            return (1, -1, px_q_neg1, sz, dep)

        # Event 4: Buy MidLimit — inside spread from bid side
        elif event_idx == 4:
            sz, dep = self._apply_midlimit(side=+1)
            px = (best_bid + 1) if best_bid >= 0 else px_q_neg1
            return (0, +1, px, sz, dep)

        # Event 5: Limit order at Q_{+1} (ask)
        elif event_idx == 5:
            sz, dep = self._apply_limit_at_queue(px_q_pos1, side=-1)
            return (0, -1, px_q_pos1, sz, dep)

        # Event 6: Cancel at Q_{+1} (ask), may deplete
        elif event_idx == 6:
            sz, dep = self._apply_cancel_at_queue(px_q_pos1, side=-1, level=1)
            return (2, -1, px_q_pos1, sz, dep)

        # Event 7: Buy MO hitting Q_{+1} (consumes ask)
        elif event_idx == 7:
            sz, dep = self._apply_market_at_queue(px_q_pos1, side=-1, level=1)
            return (1, +1, px_q_pos1, sz, dep)

        # Event 8: Sell MidLimit — inside spread from ask side
        elif event_idx == 8:
            sz, dep = self._apply_midlimit(side=-1)
            px = (best_ask - 1) if best_ask >= 0 else px_q_pos1
            return (0, -1, px, sz, dep)

        # -------------------------------------------------------------------
        # Events 9+: deeper levels (k >= 2)
        # -------------------------------------------------------------------
        elif event_idx > 8 and self._n_deep_levels > 0:
            return self._execute_deep_level_event(event_idx)

        else:
            warnings.warn(f"Unknown QRM event index: {event_idx}")
            return (0, +1, 0, 0, False)

    # -------------------- pref-relative queue event handlers --------------------
    # These methods operate on a specific grid index (the pref-relative
    # position of Q_{-1} or Q_{+1}), NOT on the current best bid/ask.

    def _qcap_for_level(self, k: int) -> int:
        """Return the queue depth cap for level k (1-indexed).
        Falls back to global size_q if size_q_by_level not available."""
        if self._size_q_by_level is not None and k >= 1:
            return int(self._size_q_by_level[min(k - 1, len(self._size_q_by_level) - 1)])
        return self.size_q

    def _aes_for_level(self, k: int) -> float:
        """Return AES for queue level k (1-indexed). Falls back to global."""
        if self._aes_by_level is not None and k >= 1:
            return float(self._aes_by_level[min(k - 1, len(self._aes_by_level) - 1)])
        return self.aes

    def _event_unit_for_level(self, k: int) -> float:
        """Shares corresponding to one birth/death jump at level k."""
        return float(max(self._aes_for_level(k), 1e-9))

    def _disc_depth(self, raw_depth: float, level: int) -> int:
        """Discretize raw depth to AES units for the given level.
        Returns 0 for empty queues (not 1 — fixes off-by-one)."""
        if raw_depth <= 1e-12:
            return 0
        aes_k = self._aes_for_level(level)
        return int(np.ceil(raw_depth / aes_k))

    def _apply_limit_at_queue(self, px: int, side: int, level: int = 1) -> Tuple[int, bool]:
        """Place one LO-sized block at grid position px. Returns (size, False)."""
        if px < 0 or px >= self.number_tick_levels:
            return (0, False)
        current_depth = abs(self.lob_state[px])
        if self._disc_depth(current_depth, level) >= self._qcap_for_level(level):
            return (0, False)
        sign_val = +1.0 if side == +1 else -1.0
        unit = self._event_unit_for_level(level)
        self.lob_state[px] += sign_val * unit
        self._touch_lob_state()
        self.add_order_to_queue(px, int(sign_val))
        return (int(round(unit)), False)

    def _apply_cancel_at_queue(self, px: int, side: int, level: int = 1) -> Tuple[int, bool]:
        """Cancel one unit at grid position px. May deplete (returns depleted=True)."""
        if px < 0 or px >= self.number_tick_levels:
            return (0, False)
        current_depth = abs(self.lob_state[px])
        unit = self._event_unit_for_level(level)
        if current_depth <= 0:
            return (0, False)
        if current_depth <= unit:
            self.lob_state[px] = 0.0
            self.priorities_lob_state[:, px] = 0.0
            self._touch_lob_state()
            return (int(current_depth), True)  # depleted
        # Cancel with priority rank awareness.
        # IMPORTANT: second arg of sample_cancellation_priority_rank is avoid_mask,
        # not side/sign.
        rank = self.sample_cancellation_priority_rank(px)
        if rank is None or int(rank) < 0:
            return (0, False)
        rank = int(rank)

        sign_val = +1.0 if side == +1 else -1.0
        self.lob_state[px] -= sign_val * unit
        self._touch_lob_state()
        self.remove_order_from_queue(px, rank)
        return (int(round(unit)), False)

    def _apply_market_at_queue(self, px: int, side: int, level: int = 1) -> Tuple[int, bool]:
        """Consume one MO-sized block at grid position px. May deplete."""
        if px < 0 or px >= self.number_tick_levels:
            return (0, False)
        current_depth = abs(self.lob_state[px])
        unit = self._event_unit_for_level(level)
        if current_depth <= 0:
            return (0, False)
        if current_depth <= unit:
            self.lob_state[px] = 0.0
            self.priorities_lob_state[:, px] = 0.0
            self._touch_lob_state()
            return (int(current_depth), True)  # depleted
        sign_val = +1.0 if side == +1 else -1.0
        self.lob_state[px] -= sign_val * unit
        self._touch_lob_state()
        if self.priorities_lob_state[0, px] != 0.0:
            self.remove_order_from_queue(px, 0)
        return (int(round(unit)), False)

    def _apply_midlimit(self, side: int) -> Tuple[int, bool]:
        """
        Mid-limit event: limit order inside the spread.

        In the reference code (QRM_simu.py), mid-limit events NEVER place
        a mechanical order.  They go directly to reinit/regen, which handles
        setting depths and adjusting spread/prices.  This method only validates
        the preconditions and signals that reinit/regen should fire.

        Parameters
        ----------
        side : int
            +1 = buy midlimit (tighten from bid side),
            -1 = sell midlimit (tighten from ask side).

        Returns
        -------
        (executed_size, depleted) : Tuple[int, bool]
            depleted is always True (mid-limit always triggers reinit/regen).
        """
        best_bid, best_ask = self._best_bid_ask_indices()

        if best_bid < 0 or best_ask < 0:
            return (0, False)
        if (best_ask - best_bid) <= 1:
            return (0, False)

        # No mechanical order placement — reinit/regen will handle everything.
        return (1, True)

    # ==================== deep-level event handlers (k >= 2) ====================

    def _execute_deep_level_event(
        self, event_idx: int
    ) -> Tuple[int, int, int, int, bool]:
        """Execute an event at a deeper price level (k >= 2).

        Decodes the event index into (level, side, event_type) and dispatches
        to the appropriate handler.  Deep-level events never trigger
        depletion/reinit.
        """
        ki = (event_idx - 9) // 6       # 0-based deep level index
        sub = (event_idx - 9) % 6       # 0-5 within the level
        k = ki + 2
        side = +1 if sub < 3 else -1    # bid (0-2) vs ask (3-5)
        etype = sub % 3                 # 0=limit, 1=cancel, 2=market

        # Use arithmetic offset (dense grid) — same as _depth_at_level.
        px, q = self._depth_at_level(k, side)

        if px < 0:
            return (0, side, 0, 0, False)

        if q == 0:
            # Empty level — only limit orders can bootstrap it.
            if etype == 0:
                sz, dep = self._apply_limit_at_level(px, side, level=k)
                return (0, side, px, sz, dep)
            else:
                # Cancel / market on empty level — no-op
                return (0, side, 0, 0, False)

        if etype == 0:  # limit
            sz, dep = self._apply_limit_at_level(px, side, level=k)
            return (0, side, px, sz, dep)
        elif etype == 1:  # cancel
            sz, dep = self._apply_cancel_at_level(px, side, level=k)
            return (2, side, px, sz, dep)
        else:  # market
            sz, dep = self._apply_market_at_level(px, side, level=k)
            return (1, -side, px, sz, dep)

    def _apply_limit_at_level(self, px: int, side: int, level: int = 2) -> Tuple[int, bool]:
        """Add one LO-sized block at a deeper price level. Never triggers depletion."""
        current_depth = abs(self.lob_state[px])
        if self._disc_depth(current_depth, level) >= self._qcap_for_level(level):
            return (0, False)

        sign_val = +1.0 if side == +1 else -1.0
        unit = self._event_unit_for_level(level)
        self.lob_state[px] += sign_val * unit
        self._touch_lob_state()
        self.add_order_to_queue(px, int(sign_val))
        return (int(round(unit)), False)

    def _apply_cancel_at_level(self, px: int, side: int, level: int = 2) -> Tuple[int, bool]:
        """Cancel one LO-sized block at a deeper price level. Level disappears if emptied."""
        current_depth = abs(self.lob_state[px])
        unit = self._event_unit_for_level(level)
        if current_depth <= 0:
            return (0, False)

        # If depth would be fully consumed, just clear the level (no reinit for k>1).
        if current_depth <= unit:
            self.lob_state[px] = 0.0
            self.priorities_lob_state[:, px] = 0.0
            self._touch_lob_state()
            return (int(current_depth), False)

        avoid = None
        if self.cancel_avoid_mask_provider is not None:
            avoid = self.cancel_avoid_mask_provider(px)

        rank = self.sample_cancellation_priority_rank(px, avoid_mask=avoid)
        if rank is None or int(rank) < 0:
            return (0, False)
        rank = int(rank)

        sign_val = +1.0 if side == +1 else -1.0
        self.lob_state[px] -= sign_val * unit
        if abs(self.lob_state[px]) < 1e-10:
            self.lob_state[px] = 0.0
        self._touch_lob_state()
        self.remove_order_from_queue(px, rank)
        return (int(round(unit)), False)

    def _apply_market_at_level(self, px: int, side: int, level: int = 2) -> Tuple[int, bool]:
        """Consume one MO-sized block at a deeper price level.

        ``side`` is the passive side being consumed (+1 = bid queue hit by
        a sell MO, -1 = ask queue hit by a buy MO).  Never triggers
        depletion/reinit for k > 1.
        """
        current_depth = abs(self.lob_state[px])
        unit = self._event_unit_for_level(level)
        if current_depth <= 0:
            return (0, False)

        if current_depth <= unit:
            self.lob_state[px] = 0.0
            self.priorities_lob_state[:, px] = 0.0
            self._touch_lob_state()
            return (int(current_depth), False)

        aggressor_side = -side
        self.lob_state[px] += float(aggressor_side) * unit
        if abs(self.lob_state[px]) < 1e-10:
            self.lob_state[px] = 0.0
        self._touch_lob_state()

        if self.priorities_lob_state[0, px] != 0.0:
            self.remove_order_from_queue(px, 0)

        return (int(round(unit)), False)

    # ==================== depletion handling ====================

    def _handle_depletion(self, event_idx: int) -> None:
        """
        Handle queue depletion or mid-limit event: pref shift + reinit/regen.

        Implements the dynamic reference-price layer (paper Section 3.1.1).
        The ordering is critical: pref shifts FIRST, then the book is
        redrawn around the NEW reference price.  This matches the paper:
        "when p_ref changes, the LOB state is redrawn from its invariant
        distribution around the new reference price."

        Step 1 — Pref shift (with probability θ):
            Events 2, 3  (bid depleted)  → pref DOWN (-tick)
            Event  8     (sell midlimit) → pref DOWN (-tick)
            Events 6, 7  (ask depleted)  → pref UP   (+tick)
            Event  4     (buy midlimit)  → pref UP   (+tick)

        Step 2 — Reinit vs. Regen (anchored to the UPDATED pref):
            With probability θ_reinit:  full redraw from invariant distribution.
            With probability 1-θ_reinit: deterministic regeneration (queue shift).

        Parameters
        ----------
        event_idx : int
            1-based QRM event index (2, 3, 4, 6, 7, or 8).
        """
        if not self.use_dynamic_pref:
            return

        # ------------------------------------------------------------------
        # Step 1: Pref shift (with probability θ) — MUST happen FIRST
        # ------------------------------------------------------------------
        # Per paper Section 3.1.1: "when p_mid changes, p_ref changes by
        # ±δ with probability θ."  The reinit/regen in Step 2 is ONLY
        # triggered when pref actually changes — it redraws the book
        # around the NEW reference price.  If pref does not change
        # (prob 1-θ), no reinit/regen occurs; the depleted/midlimited
        # queue is simply left as-is (mechanical consequence of the event).
        evtpref = float(self.rng.random_sample())
        _pref_shifted = False

        if evtpref <= self.theta:
            if event_idx in (2, 3, 8):
                self.pref -= 1.0
                self.pref_half -= 2  # ±1 tick = ±2 half-ticks
                _pref_shifted = True
            elif event_idx in (4, 6, 7):
                self.pref += 1.0
                self.pref_half += 2
                _pref_shifted = True

        # ------------------------------------------------------------------
        # Step 2: Reinit vs. Regen — ONLY if pref actually shifted
        # ------------------------------------------------------------------
        # Per paper: "when p_ref changes [...] with probability θ_reinit
        # the LOB state is redrawn from its invariant distribution."
        # When pref did NOT change, the depletion/midlimit event has
        # already been applied mechanically — no book redraw needed.
        if _pref_shifted:
            evtreinit = float(self.rng.random_sample())
            if evtreinit <= self.theta_reinit:
                self._reinit_book(event_idx)
                self._last_reinit_regen = "reinit"
            else:
                self._regen_book(event_idx)
                self._last_reinit_regen = "regen"

    def _reinit_book(self, event_idx: int) -> None:
        """
        Reinitialization: redraw the ENTIRE book from its invariant distribution
        around the new reference price.

        Per the paper (Section 3.1.1): "with probability θ^reinit, the LOB state
        is redrawn from its invariant distribution around the new reference price."

        k=1 sampling by model:
            Model I/IIA: model-based invariant (birth-death product π⊗π,
                         from _model_k1_invariant).
            Model IIB:   Monte Carlo joint invariant (from _iib_joint_invariant).
            Fallback:    empirical statprob (if model invariants not available).
        k=2..K:
            Model IIA/IIB: regime-conditioned CDFs from reinit_k2_data.
            Model I:       per-level birth-death invariant CDFs.

        Parameters
        ----------
        event_idx : int
            1-based QRM event index.  Used to determine spread change:
            - Events 2, 3, 6, 7 (depletion) → spread widens (+1)
            - Events 4, 8 (midlimit) → spread tightens (-1)
        """
        if self.statprob is None or self.intens_val_bis is None:
            self._regen_book(event_idx)
            return

        # --- 1. Sample new k=1 queue depths ---
        # For Model IIB: sample from the JOINT invariant distribution
        # of (q_{-1}, q_{+1}) estimated by Monte Carlo simulation during
        # calibration (paper Section 2.4.5).  This captures the full
        # cross-dependence via S_{m,l}(q_opp), not just a sequential
        # approximation.
        # For Model I/IIA: sample from model-based k=1 invariant (π⊗π).
        if (self.base_model == "IIB" and self._iib_joint_invariant is not None
                and "cdf" in self._iib_joint_invariant):
            # IIB: sample (bb, ba) jointly from Monte Carlo invariant
            # (paper Section 2.4.5 — analytically intractable)
            cdf = self._iib_joint_invariant["cdf"]
            states = self._iib_joint_invariant["states"]
            u = float(self.rng.random_sample())
            idx = int(np.searchsorted(cdf, u))
            idx = min(idx, len(states) - 1)
            bb_new, ba_new = states[idx]
            bb_new = max(0, min(bb_new, self.size_q))
            ba_new = max(0, min(ba_new, self.size_q))
        elif (self.base_model in ("I", "IIA")
              and self._model_k1_invariant is not None
              and "cdf" in self._model_k1_invariant):
            # Model I/IIA: sample from model-based k=1 invariant.
            # Per paper Section 2.3.3: the invariant is the product of
            # 1D birth-death invariants (independent bid/ask under symmetry).
            cdf = self._model_k1_invariant["cdf"]
            states = self._model_k1_invariant["states"]
            u = float(self.rng.random_sample())
            idx = int(np.searchsorted(cdf, u))
            idx = min(idx, len(states) - 1)
            bb_new, ba_new = states[idx]
            bb_new = max(0, min(bb_new, self.size_q))
            ba_new = max(0, min(ba_new, self.size_q))
        else:
            # Fallback: empirical joint from statprob
            state_idx = int(self.rng.choice(len(self.statprob), p=self.statprob))
            bb_new = int(self.intens_val_bis.iloc[state_idx, 0])
            ba_new = int(self.intens_val_bis.iloc[state_idx, 1])
            bb_new = max(0, min(bb_new, self.size_q))
            ba_new = max(0, min(ba_new, self.size_q))

        # --- 2. Compute new spread ---
        best_bid, best_ask = self._best_bid_ask_indices()
        if best_bid >= 0 and best_ask >= 0:
            current_sp = best_ask - best_bid
        else:
            current_sp = 1

        if event_idx in (2, 3, 6, 7):
            new_sp = min(current_sp + 1, self.size_s)
        elif event_idx in (4, 8):
            new_sp = max(current_sp - 1, 1)
        else:
            new_sp = max(1, min(current_sp, self.size_s))

        # --- 3. Compute new bid/ask grid positions respecting spread ---
        # When new_sp=1: bid at Q_{-1}, ask at Q_{+1} (standard).
        # When new_sp=2: the gap is on the DEPLETED side.
        #   - Bid depleted (events 2,3): bid was consumed → Q_{-1} empty,
        #     so the gap is on the bid side: bid at Q_{-2}, ask at Q_{+1}.
        #   - Ask depleted (events 6,7): ask was consumed → Q_{+1} empty,
        #     so the gap is on the ask side: bid at Q_{-1}, ask at Q_{+2}.
        #   - MidLimit events tighten the spread, so new_sp < current_sp.
        if new_sp <= 1:
            new_bid_idx = self._queue_grid_idx(-1)
            new_ask_idx = self._queue_grid_idx(+1)
        elif event_idx in (2, 3):
            # Bid depleted → gap on bid side
            new_bid_idx = self._queue_grid_idx(-2)
            new_ask_idx = self._queue_grid_idx(+1)
        else:
            # Ask depleted (6,7) or other → gap on ask side
            new_bid_idx = self._queue_grid_idx(-1)
            new_ask_idx = self._queue_grid_idx(+2)

        # --- 4. Bounds check ---
        K_total = 1 + self._n_deep_levels
        margin = K_total + 1
        new_bid_idx = max(margin, min(new_bid_idx, self.number_tick_levels - margin))
        new_ask_idx = max(new_bid_idx + 1, min(new_ask_idx, self.number_tick_levels - margin))

        # --- 5. Clear the ENTIRE book ---
        # Notify MM (if registered) that all its orders will be destroyed.
        if self.on_forced_cancel is not None:
            self.on_forced_cancel(prices=None, reason="qrm_reinit")
        self.lob_state[:] = 0.0
        self.priorities_lob_state[:, :] = 0.0
        self._invalidate_bba_cache()

        # --- 6. Set k=1 depths (skip if q=0, which means empty) ---
        lo_sz = max(1.0, self._event_unit_for_level(1))
        if bb_new > 0:
            _aes_1 = self._aes_for_level(1)
            target_bb = max(int(round(lo_sz)), int(round(bb_new * _aes_1)))
            self._set_queue_depth(new_bid_idx, target_bb, side=+1, level=1)
        if ba_new > 0:
            _aes_1 = self._aes_for_level(1)
            target_ba = max(int(round(lo_sz)), int(round(ba_new * _aes_1)))
            self._set_queue_depth(new_ask_idx, target_ba, side=-1, level=1)

        # --- 7. Set k=2..K depths from per-level invariant distributions ---
        # When spread=2, one side's k=1 is at Q_{±2} instead of Q_{±1}.
        # The deep levels must be offset by 1 on that side to avoid
        # overlapping with k=1.  Example: bid depleted, new_sp=2 →
        #   k=1 bid at Q_{-2}, so k=2 bid should be at Q_{-3} (not Q_{-2}).
        # On the non-depleted side, deep levels use normal Q_{±k} positions.
        _bid_offset = 1 if (new_sp >= 2 and event_idx in (2, 3)) else 0
        _ask_offset = 1 if (new_sp >= 2 and event_idx in (6, 7)) else 0

        # Effective Q_{-1} / Q_{+1} depths after reinit.
        # In spread-2 depletion, k=1 was placed at Q_{-2} or Q_{+2}, so
        # the actual Q_{-1} / Q_{+1} is EMPTY (depth=0).
        q1_eff_bid = bb_new if _bid_offset == 0 else 0  # 0 when gap is on bid side
        q1_eff_ask = ba_new if _ask_offset == 0 else 0  # 0 when gap is on ask side

        for ki in range(self._n_deep_levels):
            k = ki + 2  # queue level (2, 3, ...)

            for side, base_k in [(+1, k + _bid_offset), (-1, k + _ask_offset)]:
                q_idx_sign = -base_k if side == +1 else +base_k
                px = self._queue_grid_idx(q_idx_sign)
                if not (0 <= px < self.number_tick_levels):
                    continue

                # Model IIA/IIB at k=2: sample from model-based MC invariant.
                if (k == 2 and self.base_model in ("IIA", "IIB")
                        and self._reinit_k2_data is not None):
                    q1 = q1_eff_bid if side == +1 else q1_eff_ask
                    q1_key = "q1pos" if q1 > 0 else "q1zero"

                    if self.base_model == "IIB" and isinstance(
                            self._reinit_k2_data.get(q1_key), dict):
                        # IIB: 3D MC — CDFs conditioned on q1 regime AND
                        # opposite-side S_{m,l} regime.
                        q_opp = q1_eff_ask if side == +1 else q1_eff_bid
                        if q_opp == 0:
                            opp_regime = 0
                        elif q_opp <= self._iib_m:
                            opp_regime = 1
                        elif q_opp <= self._iib_l:
                            opp_regime = 2
                        else:
                            opp_regime = 3
                        cdf = self._reinit_k2_data[q1_key][opp_regime]
                    else:
                        # IIA: 2D MC — CDFs conditioned only on q1 regime.
                        cdf = self._reinit_k2_data.get(
                            f"prob_q2_{q1_key}",
                            self._reinit_k2_data.get(q1_key, np.array([1.0])))

                    u = float(self.rng.random_sample())
                    depth_aes = int(np.searchsorted(cdf, u))
                else:
                    # Standard: sample from per-level invariant CDF
                    depth_aes = self._sample_depth_aes_for_level(k)

                if depth_aes > 0:
                    _aes_k = self._aes_for_level(k)
                    unit_k = self._event_unit_for_level(k)
                    target = max(int(round(unit_k)),
                                 int(round(depth_aes * _aes_k)))
                    self._set_queue_depth(px, target, side=side, level=k)

    def _regen_book(self, event_idx: int) -> None:
        """
        Deterministic regeneration in pref-relative Q_i space.

        Per paper Section 3.1.1: when p_ref changes, "the value of q_i
        switches immediately to the value of one of its neighbors."

        After pref has already shifted (±1 tick = ±2 half-ticks in Step 1),
        the Q_i grid positions have moved by one tick.  The depths at the
        OLD grid positions are now naturally at the NEW Q_i positions.
        The only action needed is:

        For depletion (events 2,3 = bid; 6,7 = ask):
            - The depleted queue was at old Q_{-1} / Q_{+1}, which after
              pref shift is no longer Q_{-1} / Q_{+1}.  It's now empty at
              a position between the new queues.
            - The new edge level Q_{-K} / Q_{+K} (which was untracked
              before) needs to be sampled from the invariant distribution.

        For midlimit (events 4 = buy mid; 8 = sell mid):
            - A new order was placed inside the spread, and pref shifted
              toward it.  The old edge level Q_{-K} / Q_{+K} now falls
              outside the tracked range and should be cleared.

        Parameters
        ----------
        event_idx : int
            1-based QRM event index.
        """
        K_total = 1 + self._n_deep_levels
        _aes_1 = self._aes_for_level(1)
        regen_depth = max(int(round(self._event_unit_for_level(1))), int(round(_aes_1)))

        def _renorm_depth(px: int, old_level: int, new_level: int, side: int):
            """Renormalize depth at px when Q_old_level becomes Q_new_level.

            Per paper Section 3.1.1: when q_i switches to q_j, the
            discretized depth must be renormalized by AES_i / AES_j.

            The operation is:
                q_old_aes = ceil(raw_depth / AES_old)
                q_new_aes = ceil(q_old_aes * AES_old / AES_new)
                new_shares = q_new_aes * AES_new

            Uses _set_queue_depth to maintain consistency between
            lob_state and priorities_lob_state.
            """
            if self._aes_by_level is None:
                return
            if old_level < 1 or new_level < 1:
                return
            old_k = min(old_level - 1, len(self._aes_by_level) - 1)
            new_k = min(new_level - 1, len(self._aes_by_level) - 1)
            aes_old = self._aes_by_level[old_k]
            aes_new = self._aes_by_level[new_k]
            if abs(aes_old - aes_new) < 1e-12:
                return
            if not (0 <= px < self.number_tick_levels):
                return
            raw = abs(self.lob_state[px])
            if raw < 1e-10:
                return
            # Convert to discretized AES units at old level
            q_old = int(np.ceil(raw / aes_old))
            # Renormalize: q_new = ceil(q_old * AES_old / AES_new)
            q_new = max(1, int(np.ceil(q_old * aes_old / aes_new)))
            # Convert back to shares at new level's AES
            unit_new = self._event_unit_for_level(new_level)
            new_shares = max(int(round(unit_new)), int(round(q_new * aes_new)))
            # Use _set_queue_depth to maintain consistency with priorities
            self._set_queue_depth(px, new_shares, side=side, level=new_level)

        if event_idx in (2, 3):
            # Bid depleted, pref shifted DOWN.
            # Per paper: ALL Q_i shift by one position.
            # Bid side: old Q_{-(k+1)} → new Q_{-k} (closer to pref)
            # Ask side: old Q_{+k} → new Q_{+(k+1)} (farther from pref)
            # Process in reverse order to avoid overwriting.
            for k in range(K_total, 0, -1):
                px = self._queue_grid_idx(-k)
                _renorm_depth(px, old_level=k + 1, new_level=k, side=+1)
            for k in range(K_total, 0, -1):
                px = self._queue_grid_idx(+k)
                _renorm_depth(px, old_level=max(k - 1, 1), new_level=k, side=-1)
            # Bootstrap new Q_{-1} if empty
            px_new_q_neg1 = self._queue_grid_idx(-1)
            if 0 <= px_new_q_neg1 < self.number_tick_levels:
                if abs(self.lob_state[px_new_q_neg1]) < 1e-10:
                    self._set_queue_depth(px_new_q_neg1, regen_depth, side=+1, level=1)
            # Sample new edge level Q_{-K}
            if self._n_deep_levels > 0:
                px_edge = self._queue_grid_idx(-K_total)
                if 0 <= px_edge < self.number_tick_levels:
                    depth_aes = self._sample_depth_aes_for_level(K_total)
                    if depth_aes > 0:
                        _aes_edge = self._aes_for_level(K_total)
                        unit_edge = self._event_unit_for_level(K_total)
                        target = max(int(round(unit_edge)), int(round(depth_aes * _aes_edge)))
                        self._set_queue_depth(px_edge, target, side=+1, level=K_total)

        elif event_idx in (6, 7):
            # Ask depleted, pref shifted UP → each Q_{+k} is now the
            # old Q_{+(k+1)}.  Renormalize depths on BOTH sides.
            # Bid side also shifts: old Q_{-k} is now Q_{-(k+1)}.
            for k in range(K_total, 0, -1):  # reverse to avoid overwriting
                px = self._queue_grid_idx(+k)
                _renorm_depth(px, old_level=k + 1, new_level=k, side=-1)
            for k in range(K_total, 0, -1):
                px = self._queue_grid_idx(-k)
                _renorm_depth(px, old_level=max(k - 1, 1), new_level=k, side=+1)
            px_new_q_pos1 = self._queue_grid_idx(+1)
            if 0 <= px_new_q_pos1 < self.number_tick_levels:
                if abs(self.lob_state[px_new_q_pos1]) < 1e-10:
                    self._set_queue_depth(px_new_q_pos1, regen_depth, side=-1, level=1)
            if self._n_deep_levels > 0:
                px_edge = self._queue_grid_idx(+K_total)
                if 0 <= px_edge < self.number_tick_levels:
                    depth_aes = self._sample_depth_aes_for_level(K_total)
                    if depth_aes > 0:
                        _aes_edge = self._aes_for_level(K_total)
                        unit_edge = self._event_unit_for_level(K_total)
                        target = max(int(round(unit_edge)), int(round(depth_aes * _aes_edge)))
                        self._set_queue_depth(px_edge, target, side=-1, level=K_total)

        elif event_idx == 4:
            # Buy midlimit, pref shifted UP → old Q_{-K} falls off.
            # Place regen depth at new Q_{-1} (the midlimit insertion point).
            px_new_q_neg1 = self._queue_grid_idx(-1)
            if 0 <= px_new_q_neg1 < self.number_tick_levels:
                if abs(self.lob_state[px_new_q_neg1]) < 1e-10:
                    self._set_queue_depth(px_new_q_neg1, regen_depth, side=+1, level=1)
            # Clear old edge that fell off
            px_old_edge = self._queue_grid_idx(-K_total)
            if 0 <= px_old_edge < self.number_tick_levels:
                self._clear_level(px_old_edge)

        elif event_idx == 8:
            # Sell midlimit, pref shifted DOWN → old Q_{+K} falls off.
            px_new_q_pos1 = self._queue_grid_idx(+1)
            if 0 <= px_new_q_pos1 < self.number_tick_levels:
                if abs(self.lob_state[px_new_q_pos1]) < 1e-10:
                    self._set_queue_depth(px_new_q_pos1, regen_depth, side=-1, level=1)
            px_old_edge = self._queue_grid_idx(+K_total)
            if 0 <= px_old_edge < self.number_tick_levels:
                self._clear_level(px_old_edge)

    def _set_queue_depth(
        self, price_idx: int, target_shares: int, side: int, level: int = 1
    ) -> None:
        """
        Set the queue at a given price to approximately `target_shares` shares.

        Converts target_shares to a number of orders (each = AES_level shares)
        and adds/removes background orders to match.
        When removing, we remove from the **back** of the FIFO queue (highest
        occupied rank) to preserve the MM's orders, which typically have lower
        ranks (earlier arrival → higher priority).

        Parameters
        ----------
        price_idx : int
            Local tick index in lob_state.
        target_shares : int
            Desired depth in shares at this price level.
        side : int
            +1 for bid, -1 for ask.
        """
        if price_idx < 0 or price_idx >= self.number_tick_levels:
            return

        lo_sz = max(1.0, self._event_unit_for_level(level))
        target_orders = max(0, int(round(float(target_shares) / lo_sz)))
        current_orders = int(np.count_nonzero(self.priorities_lob_state[:, price_idx]))
        sign_val = +1.0 if side == +1 else -1.0

        if current_orders < target_orders:
            # Need to ADD orders.
            for _ in range(target_orders - current_orders):
                self.lob_state[price_idx] += sign_val * lo_sz
                self.add_order_to_queue(price_idx, int(sign_val))

        elif current_orders > target_orders:
            # Need to REMOVE orders (from the back to preserve MM priority).
            for _ in range(current_orders - target_orders):
                # Find the highest occupied rank at this price.
                rank = self._find_last_occupied_rank(price_idx)
                if rank < 0:
                    break  # no more orders to remove
                self.lob_state[price_idx] -= sign_val * lo_sz
                self.remove_order_from_queue(price_idx, rank)

                # Clean up float residuals.
                if abs(self.lob_state[price_idx]) < 1e-10:
                    self.lob_state[price_idx] = 0.0

        if current_orders != target_orders:
            self._touch_lob_state()

    def _find_last_occupied_rank(self, price_idx: int) -> int:
        """
        Find the highest occupied rank at a given price (last in FIFO queue).

        Returns -1 if no orders exist at this price.
        """
        col = self.priorities_lob_state[:, price_idx]
        occupied = np.flatnonzero(col != 0.0)
        if occupied.size == 0:
            return -1
        return int(occupied[-1])

    def _clear_level(self, price_idx: int) -> None:
        """
        Clear all depth and priority entries at a given price level.

        Used during reinit/regen to wipe a depleted or stale level before
        setting new depths.  Zeroes both lob_state and all priority ranks.
        """
        if 0 <= price_idx < self.number_tick_levels:
            # Notify MM before clearing this level
            if self.on_forced_cancel is not None:
                self.on_forced_cancel(prices=[price_idx], reason="qrm_level_clear")
            self.lob_state[price_idx] = 0.0
            self.priorities_lob_state[:, price_idx] = 0.0
            self._invalidate_bba_cache()

    def _fix_crossed_book(self) -> None:
        """Clear any bid volume above best ask or ask volume below best bid.

        After reinit/regen, stale deep levels can end up on the wrong side of
        the new best prices, producing a crossed book (spread < 0).  This
        method removes those orphaned levels.
        """
        best_bid, best_ask = self._best_bid_ask_indices()
        if best_bid < 0 or best_ask < 0:
            return
        if best_bid < best_ask:
            return  # not crossed
        # Crossed: clear all bids >= best_ask and all asks <= best_bid
        for px in range(best_ask, self.number_tick_levels):
            if self.lob_state[px] > 0:
                self._clear_level(px)
        for px in range(0, best_bid + 1):
            if self.lob_state[px] < 0:
                self._clear_level(px)

    # ==================== simulate_order (OVERRIDE) ====================

    def simulate_order(
        self,
        lam: float = None,
        mu: float = None,
        delta: float = None,
        split_sweeps: bool = False,
    ) -> Tuple[
        List[Tuple[int, int, int, int, int]],
        List[Dict[str, float]],
        List[Dict[str, float]],
    ]:
        """
        Simulate ONE QRM event.

        This OVERRIDES the parent's `simulate_order()` to use QRM state-dependent
        intensities instead of the Santa Fe constant-rate (λ, μ, δ) model.

        Parameters `lam`, `mu`, `delta` are accepted for interface compatibility
        with `simulate_LOB_with_MM()` but are **ignored** when the QRM intensity
        table is available.  If `self.intens_val` is None, this method falls back
        to the parent's constant-rate behavior.

        The return format is identical to the parent's:
            rows    : List of (order_type, order_sign, order_price, shift, size)
            metrics : List of metric dicts (same keys as Santa Fe)
            snaps   : List of order book snapshot dicts

        Each QRM event produces exactly ONE row (no multi-fill sweeps, since QRM
        market orders always consume a single AES unit).

        Side Effects
        ------------
        - Sets `self.last_event_dt` to the sampled inter-event time.
        - Updates `self.mid_price_to_store` and `self.exp_weighted_return_to_store`.
        - May modify `self.pref` (via `_handle_depletion`).
        - Calls `self.center_lob_state()` to recenter the book.
        """
        # Reset per-event reinit/regen tracking
        self._last_reinit_regen = None

        # ------------------------------------------------------------------
        # Fallback: if no QRM intensity table is loaded, use the parent's
        # constant-rate Santa Fe engine.
        # ------------------------------------------------------------------
        if self._intens_array is None and self._intens_k1 is None:
            return super().simulate_order(
                lam=lam, mu=mu, delta=delta,
                split_sweeps=split_sweeps,
            )

        # NOTE: No pre-event cache invalidation needed here.
        # _touch_lob_state() is called at each mutation point (LO/cancel/MO
        # handlers, _set_queue_depth, _clear_level, _reinit_book), so the
        # cache auto-recomputes after each mutation.

        # ------------------------------------------------------------------
        # 1. Store "Previous" mid-price for return computation.
        # ------------------------------------------------------------------
        self.mid_price_to_store["Previous"] = float(
            self.mid_price_to_store["Current"]
        )

        # ------------------------------------------------------------------
        # 2. Sample the next QRM event (competing exponentials).
        #    This sets self.last_event_dt.
        # ------------------------------------------------------------------
        event_idx, dt = self._sample_qrm_event()

        # Degenerate case: all intensities are zero → dt is +inf.
        # Return a no-op row to signal the caller.
        if not np.isfinite(dt):
            rows = [(0, +1, 0, 0, 0)]
            mid = float(self.mid_price_to_store["Current"])
            metrics = [self._capture_qrm_metrics(mid_now=mid, ret_val=0.0)]
            snaps = [self.make_ob_snapshot_row()]
            return rows, metrics, snaps

        # ------------------------------------------------------------------
        # 3. Execute the event (modify lob_state + priorities).
        # ------------------------------------------------------------------
        order_type, order_sign, order_price, executed_size, depleted = (
            self._execute_qrm_event(event_idx)
        )

        # ------------------------------------------------------------------
        # 4. Handle depletion if triggered (reinit/regen + pref shift).
        # ------------------------------------------------------------------
        if depleted:
            self._handle_depletion(event_idx)
            self._fix_crossed_book()

        # ------------------------------------------------------------------
        # 5. Compute post-event metrics (pre-shift).
        # ------------------------------------------------------------------
        mid_now = float(self.compute_mid_price())
        ret_val = mid_now - float(self.mid_price_to_store["Previous"])

        metrics_dict = self._capture_qrm_metrics(mid_now=mid_now, ret_val=ret_val)

        # ------------------------------------------------------------------
        # 6. Update EWMA of returns (same logic as parent, lines 1024-1031).
        # ------------------------------------------------------------------
        self.mid_price_to_store["Current"] = mid_now

        if (
            self.beta_exp_weighted_return is not None
            and self.beta_exp_weighted_return > 0.0
        ):
            gamma = self.gamma_exp_weighted_return
            self.exp_weighted_return_to_store = (
                gamma * float(self.exp_weighted_return_to_store)
                + (1.0 - gamma) * ret_val
            )

        # ------------------------------------------------------------------
        # 7. Re-center the LOB ONCE per event (same as parent, line 1037).
        # ------------------------------------------------------------------
        shift = int(self.center_lob_state())

        # Keep pref synchronized with grid re-centering.
        # shift is in ticks (grid indices), so pref_half shifts by 2*shift.
        if shift != 0:
            self.pref -= float(shift)
            self.pref_half -= 2 * shift

        # Adjust mid-price tracking for the shift (same as parent, lines 1039-1041).
        if shift != 0:
            self.mid_price_to_store["Current"] = (
                float(self.mid_price_to_store["Current"]) - float(shift)
            )
            self.mid_price_to_store["Previous"] = (
                float(self.mid_price_to_store["Previous"]) - float(shift)
            )

        # ------------------------------------------------------------------
        # 8. Build return values (matching parent's format exactly).
        # ------------------------------------------------------------------
        rows = [
            (
                int(order_type),
                int(order_sign),
                int(order_price),
                int(shift),
                int(executed_size),
            )
        ]

        # Adjust metrics for the shift (same as parent, lines 1049-1058).
        if shift != 0:
            metrics_dict["MidPrice"] = float(metrics_dict["MidPrice"]) - float(shift)
            for key in ("BestBidPrice", "BestAskPrice", "IndBestBid", "IndBestAsk"):
                if np.isfinite(metrics_dict.get(key, np.nan)):
                    metrics_dict[key] = float(metrics_dict[key]) - float(shift)

        # Rebuild snapshot after shift (same as parent, line 1061).
        snap = self.make_ob_snapshot_row()

        return rows, [metrics_dict], [snap]

    # -------------------- metrics helper --------------------

    def _capture_qrm_metrics(
        self, mid_now: float, ret_val: float
    ) -> Dict[str, float]:
        """
        Capture state-derived metrics at the current moment.

        Produces the same dict schema as the parent's `_capture_metrics()` inner
        function (lines 822-843) for full compatibility with the logging loop
        in `simulate_LOB_with_MM()`.

        Parameters
        ----------
        mid_now : float
            Current mid-price (local tick index).
        ret_val : float
            Return = mid_now - mid_previous.

        Returns
        -------
        Dict[str, float]
        """
        spread = float(self.compute_spread())
        best_bid, best_ask = self._best_bid_ask_indices()
        tot_bid = float(np.sum(self.lob_state[self.lob_state > 0]))
        tot_ask = float(np.sum(np.abs(self.lob_state[self.lob_state < 0])))
        q_bid_1, q_ask_1, _ = self._current_qrm_state()

        return {
            "Time": float(getattr(self, "time", np.nan)),
            "MidPrice": float(mid_now),
            "Return": float(ret_val),
            "Spread": float(spread),
            "IndBestBid": float(int(best_bid)) if best_bid >= 0 else float("nan"),
            "IndBestAsk": float(int(best_ask)) if best_ask >= 0 else float("nan"),
            "BestBidPrice": float(int(best_bid)) if best_bid >= 0 else float("nan"),
            "BestAskPrice": float(int(best_ask)) if best_ask >= 0 else float("nan"),
            "TotNumberBidOrders": tot_bid,
            "TotNumberAskOrders": tot_ask,
            "TotNumberOrders": tot_bid + tot_ask,
            "QBid1": float(q_bid_1),
            "QAsk1": float(q_ask_1),
            "Pref": float(self.pref),
            "PrefHalf": float(self.pref_half),
            "DebugIIBRegimeBid": float(self._debug_last_iib_regime_bid),
            "DebugIIBRegimeAsk": float(self._debug_last_iib_regime_ask),
            "DebugIIBFallback00": float(
                1.0 if self._debug_last_iib_used_fallback00 else 0.0
            ),
            "CancelVolAhead": float("nan"),
            "CancelQueueLen": float("nan"),
            "CancelRank": float("nan"),
        }


# =============================================================================
# ====              Standalone simulation function                         ====
# =============================================================================

def simulate_LOB_QRM(
    # --- QRM Calibrated Parameters ---
    intens_val: Optional[pd.DataFrame] = None,
    intens_k1: Optional[np.ndarray] = None,
    intens_val_bis: Optional[pd.DataFrame] = None,
    statprob: Optional[np.ndarray] = None,
    theta: float = 0.05,
    theta_reinit: float = 0.10,
    tick_qrm: float = 0.01,
    size_q: Optional[int] = None,
    size_s: int = 2,
    aes: float = 1.0,
    intensity_model: str = "I",
    use_dynamic_pref: bool = True,
    intens_levels: Optional[np.ndarray] = None,
    # Model IIA/IIB specific
    intens_k2_q1pos: Optional[np.ndarray] = None,
    intens_k2_q1zero: Optional[np.ndarray] = None,
    intens_k1_by_regime: Optional[np.ndarray] = None,
    iib_m: Optional[int] = None,
    iib_l: Optional[int] = None,
    midlimit_rate: float = 0.0,
    reinit_k2_data: Optional[dict] = None,
    aes_by_level: Optional[np.ndarray] = None,
    iib_joint_invariant: Optional[dict] = None,
    model_k1_invariant: Optional[dict] = None,
    size_q_by_level: Optional[np.ndarray] = None,

    # --- LOB Grid Parameters ---
    number_tick_levels: int = 100,
    n_priority_ranks: int = 50,
    number_levels_to_store: int = 20,
    p0: int = 100,
    mean_size_LO: float = 1,
    mean_size_MO: float = 1,

    # --- Simulation Control ---
    iterations: int = 50_000,
    iterations_to_equilibrium: int = 10_000,
    random_seed: Optional[int] = None,
    buy_mo_prob: float = 0.5,

    # --- EWMA Parameters ---
    beta_exp_weighted_return: float = 1e-3,
    intensity_exp_weighted_return: float = 1e-3,

    # --- Output ---
    path_save_files: Optional[str] = None,
    label_simulation: Optional[str] = None,
    debug_trace: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """
    Standalone QRM LOB simulation (no Market Maker).

    Mirrors the interface of `simulate_LOB()` from `LOB_SIM_SANTA_FE.py`:
    creates the engine, runs warm-up, logs events, and returns DataFrames.

    Parameters
    ----------
    intens_val : pd.DataFrame
        Calibrated intensity table (from `Intenses_py()`).
    intens_val_bis : pd.DataFrame
        (BBSize, BASize) pairs for reinitialization sampling.
    statprob : np.ndarray
        Probability vector for reinit.
    theta, theta_reinit : float
        Reference-price dynamics parameters.
    tick_qrm : float
        Tick size in price units.
    size_q, size_s : int
        Maximum queue depth and spread indices.
    aes : float
        Average event size for discretization.
    intensity_model : str
        "I", "IIA", or "IIB".
    use_dynamic_pref : bool
        Enable/disable reference-price dynamics.
    intens_levels : np.ndarray or None
        Multi-level intensity cube, shape (max_k-1, q_max, 3).
        If None, only best-level (k=1) events are sampled.
    number_tick_levels, n_priority_ranks, number_levels_to_store : int
        LOB grid dimensions.
    p0, mean_size_LO, mean_size_MO : float
        Base price, mean LO size, mean MO size.
    iterations, iterations_to_equilibrium : int
        Number of main events and warm-up events.
    random_seed : int or None
        For reproducibility.
    buy_mo_prob : float
        Probability that a (Santa Fe fallback) MO is a buy.
    beta_exp_weighted_return, intensity_exp_weighted_return : float
        EWMA parameters.
    path_save_files, label_simulation : str or None
        Optional CSV output path and label.

    Returns
    -------
    (message_df, ob_df, exp_weighted_return_list)
        Same format as `simulate_LOB()` from `LOB_SIM_SANTA_FE.py`.
    """
    # ------------------------------------------------------------------
    # 1. Create RNG and engine instance.
    # ------------------------------------------------------------------
    rng = (
        np.random.RandomState(int(random_seed))
        if random_seed is not None
        else np.random.RandomState()
    )

    lob = LOB_simulation_QRM(
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
        intens_val=intens_val,
        intens_val_bis=intens_val_bis,
        statprob=statprob,
        theta=theta,
        theta_reinit=theta_reinit,
        tick_qrm=tick_qrm,
        size_q=size_q,
        size_s=size_s,
        aes=aes,
        intensity_model=intensity_model,
        use_dynamic_pref=use_dynamic_pref,
        intens_levels=intens_levels,
        intens_k1=intens_k1,
        intens_k2_q1pos=intens_k2_q1pos,
        intens_k2_q1zero=intens_k2_q1zero,
        intens_k1_by_regime=intens_k1_by_regime,
        iib_m=iib_m,
        iib_l=iib_l,
        midlimit_rate=midlimit_rate,
        reinit_k2_data=reinit_k2_data,
        aes_by_level=aes_by_level,
        iib_joint_invariant=iib_joint_invariant,
        model_k1_invariant=model_k1_invariant,
        size_q_by_level=size_q_by_level,
    )
    lob.initialize()

    # ------------------------------------------------------------------
    # 2. Warm-up phase (not logged).
    #    Run QRM events to bring the book to a realistic state.
    #    During warm-up, simulate_order() uses the QRM intensity table
    #    (lam/mu/delta are ignored but passed for interface compatibility).
    # ------------------------------------------------------------------
    for _ in tqdm(range(iterations_to_equilibrium), desc="QRM warm-up"):
        lob.simulate_order()

    # Reset EWMA accumulator after warm-up (same as Santa Fe, line 1203).
    lob.exp_weighted_return_to_store = 0.0

    # ------------------------------------------------------------------
    # 3. Main simulation loop (logged).
    # ------------------------------------------------------------------
    t_i: float = 0.0
    exp_weighted_return_list: List[float] = []
    row_idx: int = 0
    for _extra_col in (
        "QBid1",
        "QAsk1",
        "Pref",
        "PrefHalf",
        "DebugIIBRegimeBid",
        "DebugIIBRegimeAsk",
        "DebugIIBFallback00",
    ):
        if _extra_col not in lob.message_dict:
            lob.message_dict[_extra_col] = []

    for _ in tqdm(range(iterations), desc="QRM simulation"):
        # Set simulation time on the engine before each event.
        lob.time = t_i

        # Simulate one QRM event.
        rows, metrics_list, snaps_list = lob.simulate_order()

        # Advance the clock by the inter-event time.
        t_i += lob.last_event_dt

        # Log each returned row (for QRM, this is always exactly 1 row).
        for (ot, od, op, shift_val, sz), m, s in zip(
            rows, metrics_list, snaps_list
        ):
            lob.message_dict["Time"].append(float(t_i))
            lob.message_dict["Type"].append(int(ot))
            lob.message_dict["Direction"].append(int(od))
            lob.message_dict["Price"].append(int(op))
            lob.message_dict["Shift"].append(int(shift_val))
            lob.message_dict["Size"].append(int(sz))

            lob.message_dict["Spread"].append(float(m.get("Spread", np.nan)))
            lob.message_dict["MidPrice"].append(float(m.get("MidPrice", np.nan)))
            lob.message_dict["Return"].append(float(m.get("Return", np.nan)))
            lob.message_dict["TotNumberBidOrders"].append(
                float(m.get("TotNumberBidOrders", np.nan))
            )
            lob.message_dict["TotNumberAskOrders"].append(
                float(m.get("TotNumberAskOrders", np.nan))
            )
            lob.message_dict["BestBidPrice"].append(
                float(m.get("BestBidPrice", np.nan))
            )
            lob.message_dict["BestAskPrice"].append(
                float(m.get("BestAskPrice", np.nan))
            )
            lob.message_dict["IndBestBid"].append(
                float(m.get("IndBestBid", np.nan))
            )
            lob.message_dict["IndBestAsk"].append(
                float(m.get("IndBestAsk", np.nan))
            )
            lob.message_dict["CancelVolAhead"].append(
                float(m.get("CancelVolAhead", np.nan))
            )
            lob.message_dict["CancelQueueLen"].append(
                float(m.get("CancelQueueLen", np.nan))
            )
            lob.message_dict["CancelRank"].append(
                float(m.get("CancelRank", np.nan))
            )
            lob.message_dict["QBid1"].append(
                float(m.get("QBid1", np.nan))
            )
            lob.message_dict["QAsk1"].append(
                float(m.get("QAsk1", np.nan))
            )
            lob.message_dict["Pref"].append(
                float(m.get("Pref", np.nan))
            )
            lob.message_dict["PrefHalf"].append(
                float(m.get("PrefHalf", np.nan))
            )
            lob.message_dict["DebugIIBRegimeBid"].append(
                float(m.get("DebugIIBRegimeBid", np.nan))
            )
            lob.message_dict["DebugIIBRegimeAsk"].append(
                float(m.get("DebugIIBRegimeAsk", np.nan))
            )
            lob.message_dict["DebugIIBFallback00"].append(
                float(m.get("DebugIIBFallback00", 0.0))
            )

            # Snapshot.
            for key, val in s.items():
                lob.ob_dict[key].append(float(val))

            exp_weighted_return_list.append(
                float(lob.exp_weighted_return_to_store)
            )
            row_idx += 1

    # ------------------------------------------------------------------
    # 4. Post-process and return.
    # ------------------------------------------------------------------
    lob.save_results(path_save_files, label_simulation, i_cut=row_idx - 1)

    if lob.message_df_simulation is not None and not lob.message_df_simulation.empty:
        lob.message_df_simulation["EngineBuildTag"] = QRM_ENGINE_BUILD_TAG
        lob.message_df_simulation["EngineBaseModel"] = str(lob.base_model)
        lob.message_df_simulation["EngineUseDynamicPref"] = bool(lob.use_dynamic_pref)

    if debug_trace:
        msg_df = lob.message_df_simulation
        print(
            f"[QRM DEBUG] build={QRM_ENGINE_BUILD_TAG} | model={lob.base_model} | "
            f"use_dynamic_pref={lob.use_dynamic_pref}"
        )
        _aes_l1 = float(lob._aes_for_level(1))
        _unit_l1 = float(lob._event_unit_for_level(1))
        _ratio = (_unit_l1 / _aes_l1) if _aes_l1 > 0 else float("nan")
        print(
            "[QRM DEBUG] units:"
            f" mean_size_LO={float(lob.mean_size_LO):.6g},"
            f" mean_size_MO={float(lob.mean_size_MO):.6g},"
            f" aes_global={float(lob.aes):.6g},"
            f" aes_l1={_aes_l1:.6g},"
            f" event_unit_l1={_unit_l1:.6g},"
            f" unit/aes_l1={_ratio:.6g}"
        )
        if lob._aes_by_level is not None and len(lob._aes_by_level) > 0:
            _arr = np.asarray(lob._aes_by_level, dtype=float)
            _head = ",".join([f"{x:.6g}" for x in _arr[:5]])
            print(f"[QRM DEBUG] aes_by_level_head=[{_head}]")

        def _dt_weights_safe(times: np.ndarray) -> np.ndarray:
            t = np.array(times, dtype=float, copy=True)
            n = t.size
            if n <= 1:
                return np.ones(n, dtype=float)
            finite = np.isfinite(t)
            if not finite.any():
                return np.ones(n, dtype=float) / float(n)
            valid_idx = np.where(finite)[0]
            first = int(valid_idx[0])
            last = int(valid_idx[-1])
            t[:first] = t[first]
            t[last + 1:] = t[last]
            for i in range(first + 1, n):
                if not np.isfinite(t[i]):
                    t[i] = t[i - 1]
            t = np.maximum.accumulate(t)
            dt = np.diff(t, append=t[-1])
            dt[-1] = max(t[-1] - t[-2], 1e-9)
            dt = np.clip(dt, 1e-9, None)
            return dt / max(float(dt.sum()), 1e-12)

        def _print_q_dist(tag: str, q_arr: np.ndarray, w_arr: np.ndarray) -> None:
            q = np.asarray(q_arr, dtype=float)
            w = np.asarray(w_arr, dtype=float)
            mask = np.isfinite(q) & np.isfinite(w)
            if not np.any(mask):
                print(f"[QRM DEBUG] {tag}: no finite samples")
                return
            q = q[mask]
            w = np.clip(w[mask], 0.0, None)
            sw = float(np.sum(w))
            if sw <= 0.0:
                w = np.ones_like(q, dtype=float) / float(q.size)
            else:
                w = w / sw
            p0 = float(np.sum(w[q <= 0.5]))
            p1 = float(np.sum(w[(q > 0.5) & (q <= 1.5)]))
            p2 = float(np.sum(w[(q > 1.5) & (q <= 2.5)]))
            p3 = float(np.sum(w[q > 2.5]))
            q_pos = q[q > 0.0]
            if q_pos.size > 0:
                pct = np.percentile(q_pos, [25, 50, 75, 90, 99])
                pct_txt = (
                    f"q_pos_pct=[{pct[0]:.2f},{pct[1]:.2f},{pct[2]:.2f},"
                    f"{pct[3]:.2f},{pct[4]:.2f}]"
                )
            else:
                pct_txt = "q_pos_pct=[]"
            print(
                f"[QRM DEBUG] {tag}: P0={p0:.2%}, P1={p1:.2%}, P2={p2:.2%}, "
                f"P>=3={p3:.2%} | {pct_txt}"
            )

        if msg_df is None or msg_df.empty:
            print("[QRM DEBUG] message_df is empty")
        else:
            t = msg_df["Time"].to_numpy(dtype=float)
            w = _dt_weights_safe(t)
            q_bid = msg_df["QBid1"].to_numpy(dtype=float)
            q_ask = msg_df["QAsk1"].to_numpy(dtype=float)
            _print_q_dist("QBid1(dt-weighted)", q_bid, w)
            _print_q_dist("QAsk1(dt-weighted)", q_ask, w)
            q1_bid = float(np.sum(w[(q_bid > 0.5) & (q_bid <= 1.5)]))
            q2_bid = float(np.sum(w[(q_bid > 1.5) & (q_bid <= 2.5)]))
            q1_ask = float(np.sum(w[(q_ask > 0.5) & (q_ask <= 1.5)]))
            q2_ask = float(np.sum(w[(q_ask > 1.5) & (q_ask <= 2.5)]))
            print(
                f"[QRM DEBUG] q1/q2(dt): bid q1={q1_bid:.2%}, q2={q2_bid:.2%} | "
                f"ask q1={q1_ask:.2%}, q2={q2_ask:.2%}"
            )

            # Extra probe on raw best-level share sizes from snapshots.
            if lob.ob_df_simulation is not None and not lob.ob_df_simulation.empty:
                for _col in ("BidSize_1", "AskSize_1"):
                    if _col in lob.ob_df_simulation.columns:
                        _x = lob.ob_df_simulation[_col].to_numpy(dtype=float)
                        _xf = _x[np.isfinite(_x) & (_x > 0)]
                        if _xf.size > 0:
                            _p = np.percentile(_xf, [1, 5, 10, 25, 50])
                            _uniq = np.unique(np.round(_xf[:5000], 6))
                            _uq = ",".join([f"{u:.6g}" for u in _uniq[:8]])
                            print(
                                f"[QRM DEBUG] {_col}+: p1/p5/p10/p25/p50="
                                f"[{_p[0]:.6g},{_p[1]:.6g},{_p[2]:.6g},"
                                f"{_p[3]:.6g},{_p[4]:.6g}] | uniq_head=[{_uq}]"
                            )

            if "DebugIIBFallback00" in msg_df.columns:
                fb = msg_df["DebugIIBFallback00"].to_numpy(dtype=float)
                fb_evt = float(np.mean(fb > 0.5))
                fb_dt = float(np.sum(w * (fb > 0.5)) / max(np.sum(w), 1e-12))
                print(
                    f"[QRM DEBUG] fallback00 share: event={fb_evt:.2%}, dt={fb_dt:.2%}"
                )

            if lob.base_model == "IIB":
                total_sampled = max(int(lob._debug_sampled_events), 1)
                fb_global = float(lob._debug_iib_fallback00_events) / float(total_sampled)
                print(
                    f"[QRM DEBUG] IIB sampled_events={lob._debug_sampled_events}, "
                    f"fallback00_events={lob._debug_iib_fallback00_events} "
                    f"({fb_global:.2%})"
                )
                bid_counts = np.asarray(lob._debug_iib_regime_bid_counts, dtype=float)
                ask_counts = np.asarray(lob._debug_iib_regime_ask_counts, dtype=float)
                if bid_counts.sum() > 0:
                    bid_counts = bid_counts / bid_counts.sum()
                if ask_counts.sum() > 0:
                    ask_counts = ask_counts / ask_counts.sum()
                print(
                    "[QRM DEBUG] IIB regime shares bid="
                    f"[Q0={bid_counts[0]:.2%}, Q-={bid_counts[1]:.2%}, "
                    f"Qbar={bid_counts[2]:.2%}, Q+={bid_counts[3]:.2%}]"
                )
                print(
                    "[QRM DEBUG] IIB regime shares ask="
                    f"[Q0={ask_counts[0]:.2%}, Q-={ask_counts[1]:.2%}, "
                    f"Qbar={ask_counts[2]:.2%}, Q+={ask_counts[3]:.2%}]"
                )

    return (
        lob.message_df_simulation,
        lob.ob_df_simulation,
        exp_weighted_return_list,
    )
