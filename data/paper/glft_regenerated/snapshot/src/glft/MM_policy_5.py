from typing import Dict, Any, Tuple, Optional, Callable
import math

def always_best_bid_ask_mm_policy_factory(
    inv_limit: int = 10,
    
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
) -> Callable[[Dict[str, Any], Optional[Any]], Tuple]:
    """
    Factory for the "Always Best Bid/Ask" Market Making policy.
    
    STRATEGY OVERVIEW:
    ------------------
    - Pegging: Matches the Market Best Bid and Best Ask (L1) at all times.
    - Risk Management: Switches to one-sided quoting if inventory limits are breached.
    - Execution Intelligence: Uses "Smart Fill" logic to preserve queue priority 
      on the non-filled side.
      
    THROTTLING ARCHITECTURE (STABLE MODE):
    --------------------------------------
    - Implements "Hard Throttling" to prevent over-trading and CPU waste.
    - Includes logic to handle relative price indexing (negative prices allowed).
    """

    # -------------------------------------------------------------------------
    # 1. Parameter Validation
    # -------------------------------------------------------------------------
    if int(inv_limit) < 1:
        raise ValueError("inv_limit must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")

    # Cache thresholds locally for performance to avoid re-casting every step
    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))

    # Initialization Log
    print("-" * 65)
    print(f"[Always Best B/A - Fixed Relative Prices] Initialized:")
    print(f"  Inventory Limit={inv_limit}")
    print(f"  Throttles: Events={use_event_update}({n_events}), "
          f"Time={use_time_update}({min_time_interval}s), "
          f"TOB={use_tob_update}({n_tob_moves}), "
          f"MDP={use_mdp}")
    print("-" * 65)

    # =========================================================================
    # 2. Internal Mutable State (Closure)
    # =========================================================================
    st = {
        # Last quoted prices (to detect market moves and avoid redundant orders)
        "last_bid_px": None,
        "last_ask_px": None,
        "last_mode": None,
        
        # Throttling counters
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # Fill tracking and PnL approximations
        "last_inv": None,
        "n_fills": 0,
        "n_fills_bid": 0,
        "n_fills_ask": 0,
    }

    # Tracking base index shifts from the simulator (grid re-centering)
    # Using a list to allow mutation within the inner function scope
    prev_base_idx = [None]

    # -------------------------------------------------------------------------
    # Helpers: Data Parsing & Clock Management
    # -------------------------------------------------------------------------
    
    # [FIX]: Changed default from -1 to None. 
    # In relative backtests, -1 is a valid price index. We must return None on error only.
    def _safe_int(x, d=None):
        try: return int(x)
        except: return d

    def _safe_float(x, d=0.0):
        try: v = float(x); return v if math.isfinite(v) else d
        except: return d

    def _get_env_tob(s):
        """Extracts market state signature (TOB prices & sizes) for throttling."""
        # Use safe extraction. Note: sizes default to 0.0, prices default to -1 (or None handling)
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        return (env_bb, env_ba, env_bs, env_as)

    def _consume_fill(inv, has_bid, has_ask):
        """
        Detects inventory changes since the last step to identify fills.
        Also detects simultaneous bid+ask fills (inventory unchanged but
        orders disappeared) via has_bid/has_ask flags.
        Returns: (has_fill: bool, side: int (+1/-1/0), quantity: int)
            side=0 means both sides filled simultaneously.
        """
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            return False, 0, 0

        delta = int(inv) - int(st["last_inv"])
        st["last_inv"] = int(inv)

        if delta != 0:
            return True, (+1 if delta > 0 else -1), abs(delta)

        # Simultaneous bid+ask fill: inventory unchanged but both legs gone
        if (st["last_bid_px"] is not None and not has_bid and
                st["last_ask_px"] is not None and not has_ask):
            return True, 0, 2

        return False, 0, 0

    def _update_tob_clock(key):
        """Increments TOB move counter if the Top of Book has changed."""
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time):
        """
        Resets all throttle counters.
        Called when an action is taken or when entering 'sleep' mode.
        """
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    # =========================================================================
    # MAIN POLICY FUNCTION
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:
        
        # --- A. Time & Shift Logic ---
        current_time = _safe_float(state.get("time", 0.0))

        # Handle Simulator Grid Shifts (keep local prices valid)
        # This is critical for simulators that re-center the grid around 0.
        shift_happened = False
        if mm is not None and hasattr(mm, "base_price_idx"):
            try: cur_base = int(mm.base_price_idx)
            except: cur_base = None
            
            if cur_base is not None:
                if prev_base_idx[0] is None:
                    prev_base_idx[0] = cur_base
                else:
                    base_shift = int(cur_base - int(prev_base_idx[0]))
                    if base_shift != 0:
                        shift_happened = True
                        # Adjust our internal memory of placed orders to match new grid
                        if st["last_bid_px"] is not None: st["last_bid_px"] -= base_shift
                        if st["last_ask_px"] is not None: st["last_ask_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # --- B. Update Throttling Counters ---
        if use_event_update: st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # --- C. Read State & Determine Mode ---
        I_t = _safe_int(state.get("inventory", 0))
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))
        
        # Target Prices: Directly Peg to Market Best Bid/Ask
        # [FIX]: Use None as default to detect missing data reliably
        mkt_bb = _safe_int(state.get("best_bid"), None)
        mkt_ba = _safe_int(state.get("best_ask"), None)

        # ---------------------------------------------------------------------
        # [CRITICAL FIX] Data Integrity Check
        # ---------------------------------------------------------------------
        # Old Logic: if mkt_bb < 0 ... (caused bug in relative backtests)
        # New Logic: Only hold if data is explicitly None (missing).
        # Negative integers are valid prices in relative index mode.
        if mkt_bb is None or mkt_ba is None:
            return ("hold",)

        # Inventory Gating (Mode Switching)
        if I_t >= inv_limit: desired_mode = "ask_only"
        elif I_t <= -inv_limit: desired_mode = "bid_only"
        else: desired_mode = "two_sided"
        
        mode_changed = (st["last_mode"] != desired_mode)

        # Fill Detection
        new_fill, filled_side, n_units = _consume_fill(I_t, has_bid, has_ask)
        if new_fill:
            st["n_fills"] += max(1, n_units)
            if filled_side == +1: st["n_fills_bid"] += max(1, n_units)
            elif filled_side == -1: st["n_fills_ask"] += max(1, n_units)

        # Check for missing or forbidden legs relative to the desired mode
        missing_bid = (not has_bid) and (desired_mode in ("two_sided", "bid_only"))
        missing_ask = (not has_ask) and (desired_mode in ("two_sided", "ask_only"))
        forbidden_bid = has_bid and (desired_mode == "ask_only")
        forbidden_ask = has_ask and (desired_mode == "bid_only")

        # =====================================================================
        # EXECUTION HIERARCHY (The 4 Gates)
        # =====================================================================

        # GATE 1: CRITICAL SAFETY (Priority: HIGHEST)
        # Triggered by: Inventory Limit Breach or Mode Change.
        # Action: Immediate Reset & Repricing (ignore throttles).
        # Disabled in MDP mode — policy waits for throttle gate.
        if (mode_changed or forbidden_bid or forbidden_ask) and not use_mdp:
            st["last_mode"] = desired_mode
            _reset_all_clocks(state, current_time)
            
            if desired_mode == "two_sided":
                st["last_bid_px"], st["last_ask_px"] = mkt_bb, mkt_ba
                return ("place_bid_ask", mkt_bb, mkt_ba)
            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = mkt_bb, None
                return ("cancel_all_then_place", +1, mkt_bb)
            else: # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, mkt_ba
                return ("cancel_all_then_place", -1, mkt_ba)

        # GATE 2: URGENT REPLENISHMENT (Priority: HIGH)
        # Triggered by: Execution (Fill).
        # Action: Replenish the filled side immediately.
        # Optimization: Use "Smart Fill" logic to preserve the queue position
        # of the non-filled side if it is still valid.
        # Disabled in MDP mode — policy waits for throttle gate.
        if new_fill and not use_mdp:
            _reset_all_clocks(state, current_time)
            
            # Check peg integrity for both sides
            bid_ok = (st["last_bid_px"] == mkt_bb)
            ask_ok = (st["last_ask_px"] == mkt_ba)

            if desired_mode == "two_sided":
                # If neither quote matches the market, reset both.
                if (not bid_ok) and (not ask_ok):
                    st["last_bid_px"], st["last_ask_px"] = mkt_bb, mkt_ba
                    return ("place_bid_ask", mkt_bb, mkt_ba)
                
                # If Bid was filled (+1), replenish Bid. Preserve Ask if valid.
                if (filled_side == +1) and ask_ok:
                    st["last_bid_px"] = mkt_bb
                    return ("cancel_bid_then_place", mkt_bb)
                
                # If Ask was filled (-1), replenish Ask. Preserve Bid if valid.
                if (filled_side == -1) and bid_ok:
                    st["last_ask_px"] = mkt_ba
                    return ("cancel_ask_then_place", mkt_ba)
                
                # Fallback: Refresh both
                st["last_bid_px"], st["last_ask_px"] = mkt_bb, mkt_ba
                return ("place_bid_ask", mkt_bb, mkt_ba)

            elif desired_mode == "bid_only":
                st["last_bid_px"] = mkt_bb
                return ("cancel_bid_then_place", mkt_bb)
            else: # ask_only
                st["last_ask_px"] = mkt_ba
                return ("cancel_ask_then_place", mkt_ba)

        # GATE 3: HARD THROTTLE (Priority: MEDIUM)
        # Prevents reacting to noise if no critical event happened.
        is_throttled = False
        
        if use_event_update and (st["event_steps"] < threshold_events):
            is_throttled = True
        if use_time_update:
            # Only throttle if time has advanced correctly
            if st["last_update_time"] >= 0 and (current_time - st["last_update_time"]) < min_time_interval:
                is_throttled = True
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_throttled = True

        if is_throttled:
            # Bypass throttle ONLY if we are accidentally missing a required leg
            # (e.g. wiped by market unexpectedly or startup phase).
            # In MDP mode, even missing legs don't bypass the throttle.
            if use_mdp or not (missing_bid or missing_ask):
                return ("hold",)

        # GATE 4: NORMAL REPRICING (Priority: LOW)
        # Check if the market moved away from our last quoted price.
        need_bid = (desired_mode in ("two_sided", "bid_only")) and (
            missing_bid or (st["last_bid_px"] != mkt_bb)
        )
        need_ask = (desired_mode in ("two_sided", "ask_only")) and (
            missing_ask or (st["last_ask_px"] != mkt_ba)
        )

        # ---------------------------------------------------------------------
        # STABLE BEHAVIOR ENFORCEMENT
        # Even if we decide to HOLD (no price change), we MUST reset the clocks.
        # This enforces the "sleep time" interval, preventing the policy from 
        # busy-looping and reacting to noise 1ms later.
        # ---------------------------------------------------------------------
        if not need_bid and not need_ask:
             _reset_all_clocks(state, current_time)
             return ("hold",)

        # Update Phase (Actions)
        _reset_all_clocks(state, current_time)

        if desired_mode == "two_sided":
            # Smart Update: If both moved, update both.
            if need_bid and need_ask:
                st["last_bid_px"], st["last_ask_px"] = mkt_bb, mkt_ba
                return ("place_bid_ask", mkt_bb, mkt_ba)
            
            # If only Bid moved, update Bid, hold Ask (preserve queue priority).
            if need_bid:
                st["last_bid_px"] = mkt_bb
                return ("cancel_bid_then_place", mkt_bb)
            
            # If only Ask moved, update Ask, hold Bid (preserve queue priority).
            if need_ask:
                st["last_ask_px"] = mkt_ba
                return ("cancel_ask_then_place", mkt_ba)

        elif desired_mode == "bid_only":
            st["last_bid_px"] = mkt_bb
            return ("cancel_bid_then_place", mkt_bb)
        
        else: # ask_only
            st["last_ask_px"] = mkt_ba
            return ("cancel_ask_then_place", mkt_ba)

        return ("hold",)

    # -------------------------------------------------------------------------
    # Stats & Debugging Interfaces
    # -------------------------------------------------------------------------
    def get_n_fills() -> int: return int(st["n_fills"])
    def print_stats(prefix: str = "AlwaysBest"):
        print(f"{prefix}: fills={st['n_fills']} | Mode={st['last_mode']} | "
              f"TOB={st['moves_tob']}/{threshold_tob} | "
              f"Evt={st['event_steps']}/{threshold_events} | "
              f"Time={st['last_update_time']:.2f}")

    policy.get_n_fills = get_n_fills
    policy.print_stats = print_stats
    policy._debug_state = st

    return policy