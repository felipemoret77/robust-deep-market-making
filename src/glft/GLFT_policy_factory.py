#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLFT_policy_factory.py — Guéant-Lehalle-Fernandez-Tapia Market-Making Policy

This module implements the closed-form asymptotic approximation of the optimal
market-making strategy from:

    Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013).
    "Dealing with the Inventory Risk: a solution to the market making problem."
    Mathematics and Financial Economics, 7(4), 477–507.
    https://arxiv.org/abs/1105.3115

    See also: Guéant (2017), "Optimal Market Making", arXiv:1605.01862,
    equations (4.6)–(4.7) for the generalized version.

Architecture:
    Factory + closure pattern. The factory validates parameters, pre-computes
    the GLFT coefficients (c1, c2), and returns a ``policy(state, mm)``
    function that the simulator calls on every event. Internal mutable state
    (throttle counters, last quoted prices, fill tracking) lives in the
    closure dict ``st``.

GLFT Formula:
    The model assumes order arrivals follow a Poisson process with intensity
    Lambda(delta) = A * exp(-kappa * delta), where delta is the quote depth
    (distance from fair/mid price). The market maker maximizes expected
    utility of terminal wealth with CARA (exponential) utility and risk
    aversion parameter gamma.

    The closed-form asymptotic approximation decomposes into two coefficients:

        c1 = (1 / (xi * Delta)) * ln(1 + (xi * Delta) / kappa)

        c2 = sqrt( gamma / (2 * A * Delta * kappa)
                   * (1 + (xi * Delta) / kappa) ^ (kappa / (xi * Delta) + 1) )

    Where:
        xi     = utility parameter (defaults to gamma)
        Delta  = inventory quantum / lot size (defaults to 1)
        kappa  = order arrival decay rate
        A      = baseline order arrival intensity
        gamma  = risk aversion
        sigma  = price volatility (per unit time)

    The optimal bid and ask prices are then:

        bid = mid - (half_spread + skew_factor * inventory)
        ask = mid + (half_spread - skew_factor * inventory)

    Where:
        half_spread = c1 + (Delta / 2) * c2 * sigma
        skew_factor = c2 * sigma

    The (Delta/2) * c2 * sigma term in half_spread arises from the
    decomposition of (2q + Delta)/2 in the original formula. Even at zero
    inventory, the market maker quotes a spread wider than just 2*c1 because
    providing liquidity in discrete quanta (Delta) carries inherent risk.

    Sign convention:
        inventory > 0 (long)  -> bid widens (less eager to buy),
                                 ask tightens (more eager to sell)
        inventory < 0 (short) -> bid tightens (more eager to buy),
                                 ask widens (less eager to sell)

Rounding Fix (Inventory Bias):
    Python's built-in round() uses "banker's rounding" (ties to even), which
    creates a systematic directional bias when quotes land on *.5 (common
    when the spread is odd). The fix:
        bid -> floor (conservative, never rounds UP into the spread)
        ask -> ceil  (conservative, never rounds DOWN into the spread)
    This preserves symmetry and eliminates the *.5 tie bias that would
    otherwise cause persistent inventory drift.

Throttling:
    Three independent throttle dimensions can be enabled:
    1. TOB (Top-of-Book) moves — react after N market structure changes
    2. Event steps             — react after N simulation events
    3. Time interval           — react after T seconds of simulation time

    When multiple throttles are enabled, the policy uses OR-blocking
    semantics (any gate blocking = throttled). The policy only acts when
    ALL enabled gates are satisfied simultaneously. This matches the
    DQN controller's throttle behavior.

Decision Hierarchy (4 Gates):
    Gate 1 (CRITICAL): Mode changes and forbidden legs — always immediate.
    Gate 2 (URGENT):   Fill detected — smart-fill: preserve unfilled side.
    Gate 3 (THROTTLE): Hard throttle — block refresh unless legs are missing.
    Gate 4 (NORMAL):   Routine repricing — update only changed side(s).
"""

from typing import Dict, Any, Tuple, Optional
import math


def glft_policy_factory(
    gamma: float,
    kappa: float,
    A: float,
    sigma: float,
    inv_limit: int,
    round_to_int: bool = True,

    # --- GLFT Formula Generalization Parameters ---
    # delta (Delta) = inventory quantum / lot size.
    # epsilon (xi) = utility function parameter, defaults to gamma.
    delta: float = 1.0,
    epsilon: Optional[float] = None,

    # --- Throttling: TOB (Market Structure Moves) ---
    use_tob_update: bool = False,
    n_tob_moves: int = 10,

    # --- Throttling: Simulation Steps (Events) ---
    use_event_update: bool = False,
    n_events: int = 100,

    # --- Throttling: Simulation Time (Seconds) ---
    use_time_update: bool = False,
    min_time_interval: float = 1.0,

    # --- MDP Mode (disable bypasses) ---
    use_mdp: bool = False,
):
    """
    Factory that creates a GLFT (Guéant-Lehalle-Fernandez-Tapia) market-making
    policy function.

    Parameters
    ----------
    gamma : float
        Risk aversion parameter (CARA utility). Larger gamma -> wider spreads
        and stronger inventory skew. Must be > 0.

    kappa : float
        Order arrival decay rate. Controls how quickly order flow decreases
        as quotes move away from the fair price. In the intensity model
        Lambda(delta) = A * exp(-kappa * delta), larger kappa means order
        flow is more sensitive to quote depth. Must be > 0.

    A : float
        Baseline order arrival intensity. The expected number of orders per
        unit time when the quote is at the fair price (delta=0). Must be > 0.

    sigma : float
        Price volatility (standard deviation per unit time). Drives both the
        width and the inventory skew of optimal quotes. Must be > 0.

    inv_limit : int
        Maximum absolute inventory before switching to one-sided quoting.
        When inventory >= +inv_limit: ask_only (stop buying).
        When inventory <= -inv_limit: bid_only (stop selling).
        Must be >= 1.

    round_to_int : bool
        If True (default), use asymmetric rounding (bid=floor, ask=ceil) to
        convert continuous GLFT quotes to integer tick indices. If False, use
        simple truncation toward zero (int cast).

    delta : float
        Inventory quantum (Delta in the paper). Controls the granularity of
        inventory steps. Default 1.0. Must be > 0.

    epsilon : float or None
        Utility function parameter (xi in the paper). If None, defaults to
        gamma (the standard choice in the literature). Must be > 0 if provided.

    use_tob_update : bool
        Enable TOB-based throttle.

    n_tob_moves : int
        Number of TOB price changes required to unthrottle. Only used if
        ``use_tob_update`` is True.

    use_event_update : bool
        Enable event-count-based throttle.

    n_events : int
        Number of simulation events required to unthrottle. Only used if
        ``use_event_update`` is True.

    use_time_update : bool
        Enable time-based throttle.

    min_time_interval : float
        Minimum seconds between refreshes. Only used if ``use_time_update``
        is True. Must be > 0.

    use_mdp : bool
        If True, disable bypass priorities (Gate 1: mode change / forbidden
        legs, and Gate 2: fill replenishment).  The policy strictly respects
        the throttle gates — no early decisions for fills, inventory
        emergencies, or mode changes.  This mirrors the ``use_mdp`` flag in
        ``dqn_distributional_with_throttle`` and converts the decision
        process from SMDP (variable decision timing) to MDP (gate-only
        timing).  Default False (SMDP, bypasses active).

    Returns
    -------
    policy : callable
        Function with signature ``policy(state: Dict, mm=None) -> Tuple``
        returning an action tuple for the simulator.
    """

    # =========================================================================
    # 1) Parameter Validation
    # =========================================================================
    if gamma <= 0 or kappa <= 0 or A <= 0 or sigma <= 0:
        raise ValueError(
            "GLFT parameters (gamma, kappa, A, sigma) must be strictly positive."
        )
    if int(inv_limit) < 1:
        raise ValueError("Inventory limit must be >= 1.")
    if use_time_update and min_time_interval <= 0:
        raise ValueError(
            "min_time_interval must be > 0 when time throttling is enabled."
        )
    if delta <= 0:
        raise ValueError(
            "delta (inventory quantum) must be > 0. "
            "A zero or negative delta causes division by zero in the GLFT "
            "coefficient formulas."
        )
    if epsilon is not None and epsilon <= 0:
        raise ValueError(
            "epsilon (utility parameter) must be > 0 when explicitly provided. "
            "A zero or negative epsilon causes division by zero in the GLFT "
            "coefficient formulas."
        )

    # Pre-compute validated thresholds. Enforce minimum of 1 to avoid
    # degenerate "always satisfied" behavior.
    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))

    # =========================================================================
    # 2) GLFT Coefficient Calculation (Closed-Form Asymptotic Approximation)
    #
    # These coefficients are computed once at factory time and reused at every
    # policy call. They capture the static market microstructure parameters.
    #
    # c1 ("width"):
    #   Controls the baseline half-spread. This is the (1/kappa)*ln(1+...)
    #   term from the GLFT paper. It determines how wide the quotes are even
    #   when inventory is zero.
    #
    # c2 ("skew"):
    #   Controls how aggressively the quotes shift with inventory. When
    #   inventory is positive, the bid widens and the ask tightens by
    #   c2 * sigma per unit of inventory. This is the sqrt(gamma/(2Ak)*...)
    #   term from the paper.
    #
    # half_spread_tick:
    #   The full half-spread including the inventory quantum contribution.
    #   = c1 + (Delta/2) * c2 * sigma
    #   The (Delta/2) term comes from the decomposition of (2q + Delta)/2
    #   in the original formula: even at q=0, the discrete nature of the
    #   inventory quantum adds a baseline spread contribution.
    #
    # skew_factor:
    #   The per-unit-inventory shift in quote prices.
    #   = c2 * sigma
    #   Sigma is factored out of c2 (which contains sigma^2 under the sqrt
    #   in the original paper) as an algebraic simplification.
    # =========================================================================
    val_epsilon = float(epsilon) if epsilon is not None else float(gamma)
    val_delta = float(delta)
    volatility = float(sigma)

    # c1: "width" component
    #   = (1 / (xi * Delta)) * ln(1 + (xi * Delta) / kappa)
    term_inside_log = 1.0 + (val_epsilon * val_delta) / float(kappa)
    c1 = (1.0 / (val_epsilon * val_delta)) * math.log(term_inside_log)

    # c2: "skew" component
    #   = sqrt( gamma / (2 * A * Delta * kappa)
    #           * (1 + xi*Delta/kappa) ^ (kappa/(xi*Delta) + 1) )
    exponent = (float(kappa) / (val_epsilon * val_delta)) + 1.0
    base_term = 1.0 + (val_epsilon * val_delta) / float(kappa)
    power_term = math.pow(base_term, exponent)

    pre_factor = float(gamma) / (2.0 * float(A) * val_delta * float(kappa))
    c2 = math.sqrt(pre_factor * power_term)

    # Final static values used at each decision step:
    half_spread_tick = c1 + (val_delta / 2.0) * c2 * volatility
    skew_factor = c2 * volatility

    # Audit log (useful when sweeping gamma or comparing configurations)
    print("-" * 65)
    print("[GLFT Smart Policy] Initialized:")
    print(f"  gamma={gamma}, kappa={kappa}, A={A}, sigma={sigma}")
    print(f"  delta={val_delta}, epsilon={val_epsilon}, inv_limit={inv_limit}")
    print(f"  Throttles: Events={use_event_update}({n_events}), "
          f"Time={use_time_update}({min_time_interval}s), "
          f"TOB={use_tob_update}({n_tob_moves})")
    print("-" * 30)
    print(f"  [Formula] c1 (width coeff):           {c1:.6f}")
    print(f"  [Formula] c2 (skew coeff):            {c2:.6f}")
    print(f"  [Formula] Half Spread (base):         {half_spread_tick:.6f} ticks")
    print(f"  [Formula] Skew Factor (per unit inv):  {skew_factor:.6f} ticks")
    print("-" * 65)

    # =========================================================================
    # 3) Internal Mutable State (Closure)
    #
    # Persists across calls to policy(). Tracks:
    #   - Last quoted prices (for diff detection in Gate 4)
    #   - Last quoting mode (for mode-change detection in Gate 1)
    #   - Throttle counters (TOB moves, event steps, last update time)
    #   - Fill tracking (inventory delta + order presence for simultaneous fills)
    # =========================================================================
    st = {
        # --- Last quoted prices (grid-index space) ---
        # None means "no order on this side" (used in one-sided modes).
        "last_bid_px": None,
        "last_ask_px": None,

        # --- Last quoting mode ---
        # One of: "two_sided", "bid_only", "ask_only", or None (initial).
        "last_mode": None,

        # --- Throttle state ---
        # TOB throttle: tracks the last observed (bid_price, ask_price) key
        # and counts how many times it changed since the last refresh.
        "last_env_tob_key": None,
        "moves_tob": 0,
        # Event throttle: counts simulation steps since last refresh.
        "event_steps": 0,
        # Time throttle: simulation timestamp of last refresh.
        "last_update_time": -1.0,

        # --- Fill tracking ---
        # last_inv: inventory at the previous call (for delta computation).
        "last_inv": None,
        # last_has_bid / last_has_ask: order presence at the previous call.
        # Used as a secondary signal to detect simultaneous opposite-side
        # fills where the inventory delta cancels to zero.
        "last_has_bid": False,
        "last_has_ask": False,
        # Cumulative fill counters (statistics / debugging).
        "n_fills": 0,
        "n_fills_bid": 0,
        "n_fills_ask": 0,
    }

    # Track grid shifts: when the simulator re-centers the price grid,
    # all indices shift. We adjust memorized prices to stay consistent.
    # Stored as a one-element list so the closure can mutate it.
    prev_base_idx = [None]

    # =========================================================================
    # 4) Helper Functions
    # =========================================================================

    def _safe_int(x: Any, d: int = -1) -> int:
        """
        Safely convert ``x`` to int. Returns ``d`` if conversion fails.
        Guards against None, NaN, or string values from the state dict.
        """
        try:
            return int(x)
        except Exception:
            return d

    def _safe_float(x: Any, d: float = 0.0) -> float:
        """
        Safely convert ``x`` to float. Returns ``d`` if conversion fails
        or the result is non-finite (inf, NaN).
        """
        try:
            v = float(x)
            return v if math.isfinite(v) else d
        except Exception:
            return d

    # -------------------------------------------------------------------------
    # Unbiased Tick Conversion
    #
    # We use asymmetric rounding to prevent systematic inventory drift:
    #   BID -> floor (never round UP into the spread; conservative on buys)
    #   ASK -> ceil  (never round DOWN into the spread; conservative on sells)
    #
    # A small epsilon (1e-12) is added/subtracted to handle floating-point
    # edge cases where a value like 100.0 might be stored as 99.9999999999
    # or 100.0000000001 due to IEEE 754 representation.
    # -------------------------------------------------------------------------
    _EPS = 1e-12

    def _bid_to_idx(x: float) -> int:
        """
        Convert a continuous bid quote to integer tick index.
        Always rounds DOWN (floor) for unbiased bid placement.
        """
        if not round_to_int:
            return int(x)
        return int(math.floor(x + _EPS))

    def _ask_to_idx(x: float) -> int:
        """
        Convert a continuous ask quote to integer tick index.
        Always rounds UP (ceil) for unbiased ask placement.
        """
        if not round_to_int:
            return int(x)
        return int(math.ceil(x - _EPS))

    def _get_env_tob(s: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Build a TOB (Top-of-Book) fingerprint from the state dictionary.

        This key is used to detect market structure changes. A "TOB change"
        is defined as any change in the best bid/ask prices OR in the
        queue sizes at those levels. Both price moves and size changes
        (e.g., new orders joining or leaving the queue) count as TOB events.

        We prefer the ``_env`` variants (which exclude the MM's own orders)
        and fall back to standard keys if unavailable.

        Returns
        -------
        tuple of (int, int, int, int)
            ``(best_bid_price, best_ask_price, bid_size, ask_size)``
            as a hashable key for change detection.
        """
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try:
            ibs, ias = int(round(env_bs)), int(round(env_as))
        except Exception:
            ibs, ias = 0, 0
        return (env_bb, env_ba, ibs, ias)

    # -------------------------------------------------------------------------
    # Fill Detection
    # -------------------------------------------------------------------------

    def _consume_fill(inv: int, has_bid: bool, has_ask: bool):
        """
        Detect fills by combining inventory delta with order presence tracking.

        Primary signal — inventory delta:
          delta > 0  ->  net buying occurred (bid side filled)
          delta < 0  ->  net selling occurred (ask side filled)

        Secondary signal — order disappearance:
          When delta == 0 but an order we had is now gone, we infer a fill
          on that side. This catches the edge case where bid and ask fill
          simultaneously (e.g., buy 1 + sell 1 = delta 0).

        Parameters
        ----------
        inv : int
            Current inventory.
        has_bid : bool
            Whether we currently have an active bid order.
        has_ask : bool
            Whether we currently have an active ask order.

        Returns
        -------
        tuple of (bool, int, int)
            ``(new_fill, bid_fills, ask_fills)``
            - new_fill:  True if any fill was detected.
            - bid_fills: Estimated bid-side fills (units bought).
            - ask_fills: Estimated ask-side fills (units sold).
        """
        # First call: initialize state, no fill to report.
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            st["last_has_bid"] = has_bid
            st["last_has_ask"] = has_ask
            return False, 0, 0

        delta_inv = int(inv) - int(st["last_inv"])
        had_bid = st["last_has_bid"]
        had_ask = st["last_has_ask"]

        # Update tracking for next call.
        st["last_inv"] = int(inv)
        st["last_has_bid"] = has_bid
        st["last_has_ask"] = has_ask

        # Primary detection: inventory delta.
        bid_fills = max(0, delta_inv)    # Positive delta = bought (bid filled)
        ask_fills = max(0, -delta_inv)   # Negative delta = sold (ask filled)

        # Secondary detection: order disappearance with zero net delta.
        # Catches simultaneous opposite-side fills that cancel in the delta.
        if delta_inv == 0:
            if had_bid and not has_bid:
                bid_fills = max(bid_fills, 1)
            if had_ask and not has_ask:
                ask_fills = max(ask_fills, 1)

        total = bid_fills + ask_fills
        if total == 0:
            return False, 0, 0

        return True, bid_fills, ask_fills

    # -------------------------------------------------------------------------
    # Clock Management
    # -------------------------------------------------------------------------

    def _update_tob_clock(key) -> None:
        """
        Increment the TOB move counter if the market structure fingerprint
        has changed since the last observation.
        """
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s: Dict[str, Any], current_time: float) -> None:
        """
        Reset all throttle clocks after an action decision.

        Called whenever the policy places, cancels, or refreshes orders.
        Ensures the next action waits for the full throttle interval.
        """
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    # =========================================================================
    # 5) MAIN POLICY FUNCTION
    #
    # Called by the simulator on every event. Returns an action tuple:
    #   ("hold",)                              — do nothing
    #   ("place_bid_ask", bid_px, ask_px)      — place/replace both sides
    #   ("cancel_bid_then_place", bid_px)       — cancel bid, place new bid
    #   ("cancel_ask_then_place", ask_px)       — cancel ask, place new ask
    #   ("cancel_all_then_place", side, price) — cancel all, place one side
    #                                             side: +1 (bid) or -1 (ask)
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:

        # =================================================================
        # A) Time & Grid-Shift Detection
        #
        # The simulator may re-center the price grid (shift base_price_idx)
        # during long simulations. We detect this and adjust memorized
        # prices so diff detection in Gate 4 remains correct.
        # =================================================================
        current_time = _safe_float(state.get("time", 0.0))

        shift_happened = False
        if mm is not None and hasattr(mm, "base_price_idx"):
            try:
                cur_base = int(mm.base_price_idx)
            except Exception:
                cur_base = None

            if cur_base is not None:
                if prev_base_idx[0] is None:
                    prev_base_idx[0] = cur_base
                else:
                    base_shift = cur_base - prev_base_idx[0]
                    if base_shift != 0:
                        shift_happened = True
                        # Adjust memorized prices to new grid coordinates.
                        # Grid shifted UP by N -> same price is now at
                        # index (old_idx - N).
                        if st["last_bid_px"] is not None:
                            st["last_bid_px"] -= base_shift
                        if st["last_ask_px"] is not None:
                            st["last_ask_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # =================================================================
        # B) Update Throttle Counters
        # =================================================================
        if use_event_update:
            st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                # After grid shift, re-anchor without counting as a move.
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # =================================================================
        # C) Read State & Determine Mode
        # =================================================================
        I_t = _safe_int(state.get("inventory", 0))
        M_t = _safe_float(state.get("mid", 0.0), float("nan"))
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))

        # If mid price is not available or invalid, hold.
        if not math.isfinite(M_t):
            return ("hold",)

        # Inventory gating: hard risk limit.
        #   >= +limit -> stop buying (ask_only)
        #   <= -limit -> stop selling (bid_only)
        if I_t >= inv_limit:
            desired_mode = "ask_only"
        elif I_t <= -inv_limit:
            desired_mode = "bid_only"
        else:
            desired_mode = "two_sided"

        mode_changed = (st["last_mode"] != desired_mode)

        # =================================================================
        # D) Fill Detection
        #
        # Uses both inventory delta and order presence tracking to detect
        # fills, including simultaneous opposite-side fills.
        # =================================================================
        new_fill, bid_fills, ask_fills = _consume_fill(I_t, has_bid, has_ask)
        if new_fill:
            st["n_fills"] += (bid_fills + ask_fills)
            st["n_fills_bid"] += bid_fills
            st["n_fills_ask"] += ask_fills

        # =================================================================
        # E) Compute Optimal GLFT Quotes (Continuous -> Integer)
        #
        # The GLFT formula gives continuous-valued optimal quotes:
        #   q_bid = mid - (half_spread + skew * inventory)
        #   q_ask = mid + (half_spread - skew * inventory)
        #
        # When inventory > 0: bid widens (less eager to buy),
        #                      ask tightens (more eager to sell).
        # When inventory < 0: bid tightens (more eager to buy),
        #                      ask widens (less eager to sell).
        #
        # The continuous quotes are then converted to integer tick indices
        # using asymmetric rounding (bid=floor, ask=ceil) to avoid the
        # systematic bias from Python's banker's rounding.
        # =================================================================
        q_bid = M_t - (half_spread_tick + skew_factor * I_t)
        q_ask = M_t + (half_spread_tick - skew_factor * I_t)

        bid_px = _bid_to_idx(q_bid)
        ask_px = _ask_to_idx(q_ask)

        # Enforce minimum 1-tick spread. After asymmetric rounding, it's
        # possible for ask_px <= bid_px when the GLFT spread is very tight
        # (small gamma, large A) or inventory is extreme.
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        # =================================================================
        # F) Missing & Forbidden Legs
        #
        # Missing: we SHOULD have an order but don't (startup, post-fill).
        # Forbidden: we HAVE an order on a side we shouldn't (wrong mode).
        # =================================================================
        missing_bid = (not has_bid) and (desired_mode in ("two_sided", "bid_only"))
        missing_ask = (not has_ask) and (desired_mode in ("two_sided", "ask_only"))
        forbidden_bid = has_bid and (desired_mode == "ask_only")
        forbidden_ask = has_ask and (desired_mode == "bid_only")

        # =============================================================
        #             EXECUTION HIERARCHY (4 Gates)
        # =============================================================

        # ---------------------------------------------------------
        # GATE 1: CRITICAL — Mode Change or Forbidden Legs
        #
        # Immediate action required when the quoting mode changes
        # (e.g., inventory hit the limit) or we have orders on a
        # side we shouldn't be quoting. Ignores all throttles.
        # Disabled in MDP mode — policy waits for throttle gate.
        # ---------------------------------------------------------
        if (mode_changed or forbidden_bid or forbidden_ask) and not use_mdp:
            st["last_mode"] = desired_mode
            _reset_all_clocks(state, current_time)

            if desired_mode == "two_sided":
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)
            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = bid_px, None
                return ("cancel_all_then_place", +1, bid_px)
            else:  # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, ask_px
                return ("cancel_all_then_place", -1, ask_px)

        # ---------------------------------------------------------
        # GATE 2: URGENT — Fill Detected
        #
        # Smart-fill replenishment: after a fill, repost the filled
        # side promptly. Preserve queue priority on the non-filled
        # side if its price is still optimal.
        # Disabled in MDP mode — policy waits for throttle gate.
        # ---------------------------------------------------------
        if new_fill and not use_mdp:
            _reset_all_clocks(state, current_time)

            bid_match = (st["last_bid_px"] == bid_px)
            ask_match = (st["last_ask_px"] == ask_px)

            if desired_mode == "two_sided":
                # Both sides need updating -> refresh both.
                if (not bid_match) and (not ask_match):
                    st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                    return ("place_bid_ask", bid_px, ask_px)

                # Only bid filled, ask at correct price -> replace bid only.
                if bid_fills > 0 and ask_fills == 0 and ask_match:
                    st["last_bid_px"] = bid_px
                    return ("cancel_bid_then_place", bid_px)

                # Only ask filled, bid at correct price -> replace ask only.
                if ask_fills > 0 and bid_fills == 0 and bid_match:
                    st["last_ask_px"] = ask_px
                    return ("cancel_ask_then_place", ask_px)

                # Fallback (both sides filled, or mixed scenario).
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)

            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = bid_px, None
                return ("cancel_all_then_place", +1, bid_px)
            else:  # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, ask_px
                return ("cancel_all_then_place", -1, ask_px)

        # ---------------------------------------------------------
        # GATE 3: HARD THROTTLE
        #
        # When throttles are enabled, suppress routine refreshes
        # until enough activity accumulates.
        #
        # Semantics (OR-blocking / AND-to-unblock):
        #   Any enabled gate whose threshold is NOT yet met will
        #   block the policy. The policy is only unthrottled when
        #   ALL enabled gates are satisfied simultaneously.
        #   This is consistent with the DQN controller's throttle
        #   semantics: "any gate blocking = agent is throttled."
        #
        # Bypass: if a required leg is missing, we skip the throttle
        # to repost immediately (being unquoted is worse than an
        # extra message).
        # In MDP mode, even missing legs don't bypass the throttle.
        # ---------------------------------------------------------
        is_throttled = False

        if use_event_update and (st["event_steps"] < threshold_events):
            is_throttled = True
        if use_time_update:
            if st["last_update_time"] < 0:
                pass  # First call after init: don't throttle.
            elif (current_time - st["last_update_time"]) < min_time_interval:
                is_throttled = True
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_throttled = True

        if is_throttled:
            if use_mdp or not (missing_bid or missing_ask):
                return ("hold",)

        # ---------------------------------------------------------
        # GATE 4: NORMAL REPRICING
        #
        # All throttle conditions satisfied (or none enabled).
        # Check if quotes need updating and act accordingly.
        #
        # Mode-scoped diff: in one-sided modes, only check the
        # ACTIVE side to avoid false diffs from the None-vs-int
        # comparison on the inactive side.
        # ---------------------------------------------------------

        st["last_mode"] = desired_mode

        if desired_mode == "two_sided":
            need_bid = missing_bid or (st["last_bid_px"] != bid_px)
            need_ask = missing_ask or (st["last_ask_px"] != ask_px)

            # Nothing changed -> hold (preserves queue priority).
            if not need_bid and not need_ask:
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)

            if need_bid and need_ask:
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)
            if need_bid:
                st["last_bid_px"] = bid_px
                return ("cancel_bid_then_place", bid_px)
            # need_ask
            st["last_ask_px"] = ask_px
            return ("cancel_ask_then_place", ask_px)

        elif desired_mode == "bid_only":
            # Only check the bid side; ask is not quoted.
            need_bid = missing_bid or (st["last_bid_px"] != bid_px)

            if not need_bid:
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = bid_px, None
            return ("cancel_all_then_place", +1, bid_px)

        else:  # ask_only
            # Only check the ask side; bid is not quoted.
            need_ask = missing_ask or (st["last_ask_px"] != ask_px)

            if not need_ask:
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = None, ask_px
            return ("cancel_all_then_place", -1, ask_px)

    # =========================================================================
    # 6) Debug / Statistics Accessors
    #
    # Attached as attributes on the policy function object for easy access
    # without changing the calling convention:
    #   policy.get_n_fills()           -> total fill count
    #   policy.print_stats("GLFT")     -> one-line status summary
    #   policy._debug_state            -> raw internal state dict
    # =========================================================================
    def get_n_fills() -> int:
        """Return total fills detected since initialization."""
        return int(st["n_fills"])

    def print_stats(prefix: str = "GLFT") -> None:
        """Print a one-line summary of internal state for debugging."""
        print(
            f"{prefix}: fills={st['n_fills']} "
            f"(bid={st['n_fills_bid']}, ask={st['n_fills_ask']}) | "
            f"mode={st['last_mode']} | "
            f"TOB={st['moves_tob']}/{threshold_tob} | "
            f"Evt={st['event_steps']}/{threshold_events} | "
            f"Time={st['last_update_time']:.2f}"
        )

    policy.get_n_fills = get_n_fills
    policy.print_stats = print_stats
    policy._debug_state = st

    return policy
