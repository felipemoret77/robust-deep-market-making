#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:00:23 2026

@author: felipemoret

LOB animation with optional MM overlay.

This version fixes the main visualization pitfall when the MM quotes at L1:
- A fixed-width cyan "tile" can look like "there is more queue than just me",
  especially when MM order size > 1 (common in backtest mode).

Key improvements
----------------
1) Base order-book bars:
   - Ask bars: RED
   - Bid bars: BLUE

2) Event markers as SOLID tiles (filled rectangles), with same-color outline:
   - Market Orders (MO): YELLOW solid
   - Cancels (C): GREEN solid

3) MM passive Limit Orders are now drawn as a CYAN **QUEUE SEGMENT**:
   - Position in queue comes from:
       MM_OwnPriorities  ~ "volume ahead"
   - Total queue length at that price comes from:
       MM_OwnQueueLens   ~ "total queue volume at that price"
   - MM order size comes from:
       MM_OwnQtys (recommended; patch provided for MM_LOB_SIM_3.py)
     If MM_OwnQtys is missing, we fall back to a conservative guess.

   This makes the cyan overlay proportional to "how big we are" at that price,
   and placed exactly where we sit in FIFO.

4) Cancels "at rank position" (visual FIFO position):
   - If message_df contains CancelVolAhead and CancelQueueLen:
       frac = CancelVolAhead / CancelQueueLen
     marker is placed inside the bar (mode="queue").

5) show_all_events:
   - If True, draw MO and Cancel markers for *every* event using message_df.
   - Works even if mm_df is None.

Matplotlib backend note
-----------------------
For interactive playback with buttons + moving frames, use a GUI backend:
  * IPython: %matplotlib qt
  * scripts: QtAgg / MacOSX / TkAgg, etc.
Also: keep a reference to `anim` alive until after plt.show().
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from typing import Optional, Tuple, Any, List


# =============================================================================
# Small robust helpers
# =============================================================================
def _parse_listlike(val: Any) -> list:
    """
    Parse list-like fields that may come from a DataFrame/CSV.

    Many pipelines store columns like MM_OwnPrices as:
      - Python lists in-memory
      - Strings like "[101.0, 102.0]" after CSV round-trips

    This helper returns a plain Python list in both cases.
    """
    if val is None:
        return []

    # Treat NaN as empty
    try:
        if isinstance(val, float) and np.isnan(val):
            return []
    except Exception:
        pass

    # If it is a string, attempt to parse safely.
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            val = ast.literal_eval(s)
        except Exception:
            # If parsing fails, return empty (better than crashing animation).
            return []

    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return list(val)

    # Scalar fallback
    return [val]


def _px_key(px: float, price_step: Optional[float]) -> Any:
    """
    Return a stable key for a price level (used for matching MM prices to OB bars).

    - If price_step (tick size) is provided, key by round(px / price_step).
      This is appropriate when px is a REAL price (USD) and price_step is tick size.
    - Otherwise key by rounding px itself. If px is integer-like, use int.
    """
    x = float(px)

    # If it looks like an integer price-index, treat it as int to avoid float-key noise.
    xr = float(np.round(x))
    if abs(x - xr) < 1e-12 and price_step is None:
        return int(xr)

    if price_step is not None and price_step > 0:
        return int(np.round(x / float(price_step)))

    return float(np.round(x, 10))


def _to_numeric_time(s: pd.Series) -> pd.Series:
    """Coerce a time-like Series into float, handling strings gracefully."""
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =============================================================================
# Helpers to read the OB frames
# =============================================================================
def _get_level_cols(ob_df: pd.DataFrame, side: str):
    """
    Build ordered column lists for prices/sizes for a given side ("Bid" or "Ask").

    Assumes columns like:
        'BidPrice_1', 'BidSize_1', ..., 'AskPrice_k', 'AskSize_k'.
    """
    n = len([c for c in ob_df.columns if c.startswith(f"{side}Price_")])
    price_cols = [f"{side}Price_{k}" for k in range(1, n + 1)]
    size_cols = [f"{side}Size_{k}" for k in range(1, n + 1)]
    return price_cols, size_cols, n


def extract_lob_snapshot(
    ob_df: pd.DataFrame,
    t: int,
    max_levels: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a single time slice (row t) from the order book dataframe.

    IMPORTANT:
    - We treat `t` as a *positional* row index and use `.iloc[t]`.
      This avoids KeyError / misalignment issues when df.index is not RangeIndex.

    We also drop missing/padded levels:
      - non-finite prices or sizes
      - non-positive sizes (common padding with 0)

    Returns
    -------
    bid_p : np.ndarray (best bid outward)
    bid_s : np.ndarray (positive sizes)
    ask_p : np.ndarray (best ask outward)
    ask_s : np.ndarray (positive sizes)
    """
    bp_cols, bs_cols, nB = _get_level_cols(ob_df, "Bid")
    ap_cols, as_cols, nA = _get_level_cols(ob_df, "Ask")

    L = min(nB, nA) if max_levels is None else min(max_levels, nB, nA)

    row = ob_df.iloc[int(t)]

    bid_p = row[bp_cols[:L]].to_numpy(dtype=float)
    bid_s = row[bs_cols[:L]].to_numpy(dtype=float)
    ask_p = row[ap_cols[:L]].to_numpy(dtype=float)
    ask_s = row[as_cols[:L]].to_numpy(dtype=float)

    mb = np.isfinite(bid_p) & np.isfinite(bid_s) & (bid_s > 0.0)
    ma = np.isfinite(ask_p) & np.isfinite(ask_s) & (ask_s > 0.0)

    return bid_p[mb], bid_s[mb], ask_p[ma], ask_s[ma]


# =============================================================================
# Main animation entry point
# =============================================================================
def animate_lob(
    message_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    *,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    max_levels: int = 6,
    interval_ms: int = 30,
    show_mid: bool = True,
    mm_df: Optional[pd.DataFrame] = None,

    # Draw MO/C markers for ALL events using message_df, even without mm_df
    show_all_events: bool = False,

    # If mm_df has Time but length differs, align via merge_asof on Time.
    align_mm_by_time: bool = True,

    # Base bar colors
    bid_color: str = "#1f77b4",   # blue
    ask_color: str = "#d62728",   # red

    # MM passive LO overlay (SOLID cyan)
    mm_lo_face: str = "cyan",
    mm_lo_edge: str = "cyan",
    mm_lo_alpha: float = 0.95,
    mm_lo_linewidth: float = 2.0,

    # MM queue segment behavior
    mm_lo_use_segment: bool = True,      # True -> size-proportional segment
    mm_lo_min_px: int = 6,               # ensure visibility for tiny sizes (minimum pixel width)
    mm_lo_fallback_tile_px: int = 14,    # used only when a bar is not visible

    # ALL-EVENTS MO marker (SOLID yellow)
    all_mo_face: str = "yellow",
    all_mo_edge: str = "yellow",
    all_mo_alpha: float = 1.0,
    all_mo_px: int = 26,
    all_mo_linewidth: float = 3.0,

    # ALL-EVENTS Cancel marker (SOLID green)
    all_c_face: str = "green",
    all_c_edge: str = "green",
    all_c_alpha: float = 1.0,
    all_c_px: int = 22,
    all_c_linewidth: float = 3.0,

    # Optional MM aggressive hits (kept as-is, SOLID by default)
    # (Only shown when show_all_events=False)
    mm_hit_face: str = "#ff1493",     # strong pink
    mm_hit_edge: str = "#ff1493",
    mm_hit_alpha: float = 1.0,
    mm_hit_px: int = 26,
    mm_hit_linewidth: float = 3.0,

    # price grid info
    price_step: Optional[float] = None,  # e.g. tick_size in USD, or 1.0 in tick-index space
    bar_height: float = 0.8,             # bar height in price units

    # x-axis (quantity) grid options
    quantity_step: Optional[float] = None,
    quantity_max: Optional[float] = None,
    quantity_label_step: Optional[int] = None,
):
    """
    Build a Matplotlib animation visualizing the LOB and (optionally) the Market Maker’s
    queue position and trades.

    Type conventions supported:
    - If message_df comes from save_results(), Type is typically strings: "LO","MO","C",...
    - If you animate pre-save_results(), Type may be numeric: 0/1/2...
      We accept both conventions.

    Cancel "rank-position" rendering:
    - If message_df has CancelVolAhead and CancelQueueLen, we draw cancel at FIFO fraction
      inside the bar (mode="queue"). Otherwise we draw it at the bar edge (mode="edge").

    MM passive overlay:
    - If mm_lo_use_segment=True, draw a cyan segment from:
        start = vol_ahead / qlen
        end   = (vol_ahead + own_qty) / qlen
      where own_qty is read from MM_OwnQtys when available.
    """

    # -------------------------------------------------------------------------
    # Optional MM alignment by Time (robust to differing lengths)
    # -------------------------------------------------------------------------
    if mm_df is not None and align_mm_by_time:
        if ("Time" in mm_df.columns) and ("Time" in message_df.columns) and (len(mm_df) != len(message_df)):
            left = pd.DataFrame({"Time": _to_numeric_time(message_df["Time"]).to_numpy()})
            left["_pos"] = np.arange(len(left), dtype=int)

            right = mm_df.copy()
            right["Time"] = _to_numeric_time(right["Time"])

            left_s = left.sort_values("Time", kind="mergesort").reset_index(drop=True)
            right_s = right.sort_values("Time", kind="mergesort").reset_index(drop=True)

            merged_s = pd.merge_asof(
                left_s,
                right_s,
                on="Time",
                direction="backward",
                allow_exact_matches=True,
            )

            mm_df = (
                merged_s.sort_values("_pos", kind="mergesort")
                .drop(columns=["_pos"])
                .reset_index(drop=True)
            )

    # -------------------------------------------------------------------------
    # Frame index range
    # -------------------------------------------------------------------------
    if stop is None:
        stop = len(ob_df)

    stop = min(int(stop), len(ob_df), len(message_df))
    if mm_df is not None:
        stop = min(stop, len(mm_df))

    frames_idx = list(range(int(start), int(stop), int(step)))
    n_frames = len(frames_idx)
    if n_frames == 0:
        raise ValueError("No frames to animate: check start/stop/step indexes.")

    # -------------------------------------------------------------------------
    # Fix x-limits (quantity axis) across all frames for stable visualization
    # -------------------------------------------------------------------------
    if quantity_max is not None:
        max_q_plot = float(quantity_max)
    else:
        size_cols = [c for c in ob_df.columns if "Size_" in c]
        if size_cols:
            arr_sizes = ob_df[size_cols].to_numpy()
            max_q_plot = float(np.nanmax(np.abs(arr_sizes)))
        else:
            max_q_plot = 1.0

    if not np.isfinite(max_q_plot) or max_q_plot <= 0:
        max_q_plot = 1.0

    if quantity_step is not None and quantity_step > 0:
        max_q_plot = float(quantity_step) * float(np.ceil(max_q_plot / float(quantity_step)))

    # -------------------------------------------------------------------------
    # Initial frame (for layout)
    # -------------------------------------------------------------------------
    bid_p0, bid_s0, ask_p0, ask_s0 = extract_lob_snapshot(ob_df, frames_idx[0], max_levels=max_levels)

    yb0, xb0 = bid_p0, -bid_s0
    ya0, xa0 = ask_p0, ask_s0

    if show_mid and ("MidPrice" in message_df.columns):
        mid0 = float(message_df.iloc[int(frames_idx[0])]["MidPrice"])
    else:
        mid0 = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))

    # Optional price grid
    if price_step is not None:
        y_locator = mticker.MultipleLocator(price_step)
        ax.yaxis.set_major_locator(y_locator)
        ax.yaxis.set_minor_locator(y_locator)

    # Optional quantity grid
    if quantity_step is not None:
        if quantity_label_step is not None and quantity_label_step > 0:
            major_step = quantity_step * quantity_label_step
        else:
            major_step = quantity_step

        major_locator = mticker.MultipleLocator(major_step)
        ax.xaxis.set_major_locator(major_locator)

        if quantity_label_step is not None and quantity_label_step > 1:
            minor_locator = mticker.MultipleLocator(quantity_step)
            ax.xaxis.set_minor_locator(minor_locator)
        else:
            ax.xaxis.set_minor_locator(major_locator)

    # Base bars (book volumes) — pre-allocate max_levels patches per side
    # to avoid creating new patches dynamically (memory leak).
    _n_alloc = max_levels
    _yb = np.zeros(_n_alloc); _xb = np.zeros(_n_alloc)
    _ya = np.zeros(_n_alloc); _xa = np.zeros(_n_alloc)
    _nb = min(len(yb0), _n_alloc); _na = min(len(ya0), _n_alloc)
    _yb[:_nb] = yb0[:_nb]; _xb[:_nb] = xb0[:_nb]
    _ya[:_na] = ya0[:_na]; _xa[:_na] = xa0[:_na]
    bars_bid = ax.barh(_yb, _xb, height=bar_height, align="center", color=bid_color)
    bars_ask = ax.barh(_ya, _xa, height=bar_height, align="center", color=ask_color)
    for _p in bars_bid.patches[_nb:]: _p.set_visible(False)
    for _p in bars_ask.patches[_na:]: _p.set_visible(False)

    # Mid line
    if not np.isnan(mid0):
        mid_line = ax.axhline(mid0, linestyle="--", linewidth=1)
    else:
        mid_line = ax.axhline(0, linestyle="--", linewidth=0, alpha=0.0)

    ax.set_xlabel("Quantity (Bid < 0 | Ask > 0)")
    ax.set_ylabel("Price")
    ax.set_title("LOB Evolution — frame 0")

    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5)
    ax.set_xlim(-max_q_plot, max_q_plot)

    # y-limits based on initial snapshot
    y_min0 = min(
        bid_p0.min() if len(bid_p0) else np.inf,
        ask_p0.min() if len(ask_p0) else np.inf
    )
    y_max0 = max(
        bid_p0.max() if len(bid_p0) else -np.inf,
        ask_p0.max() if len(ask_p0) else -np.inf
    )
    if not np.isfinite(y_min0) or not np.isfinite(y_max0):
        y_min0, y_max0 = -1.0, 1.0

    pad = price_step if price_step is not None else 1.0
    ax.set_ylim(y_min0 - pad, y_max0 + pad)

    # Overlay pool: pre-allocated Rectangle patches reused across frames
    # (avoids expensive create/destroy of matplotlib artists every frame).
    _overlay_pool: list = []
    _overlay_next: int = 0

    def _acquire_overlay() -> Rectangle:
        nonlocal _overlay_next
        if _overlay_next < len(_overlay_pool):
            p = _overlay_pool[_overlay_next]
        else:
            p = Rectangle((0, 0), 0, 0, visible=False, zorder=6)
            ax.add_patch(p)
            _overlay_pool.append(p)
        _overlay_next += 1
        return p

    current_pos = 0

    # =========================================================================
    # Utilities
    # =========================================================================
    def set_bars(bar_container, new_y, new_x, facecolor):
        """Reuse pre-allocated patches (never creates new ones)."""
        patches = bar_container.patches
        need = min(len(new_y), len(patches))

        for k in range(need, len(patches)):
            patches[k].set_visible(False)

        for k in range(need):
            rect = patches[k]
            rect.set_y(float(new_y[k]) - 0.5 * float(bar_height))
            rect.set_width(float(new_x[k]))
            rect.set_height(float(bar_height))
            rect.set_facecolor(facecolor)
            rect.set_visible(True)

        return bar_container

    def clear_overlays():
        """Hide all overlay patches (returned to pool, not destroyed)."""
        nonlocal _overlay_next
        for k in range(_overlay_next):
            _overlay_pool[k].set_visible(False)
        _overlay_next = 0

    def data_dx_for_pixels(ax_, px: float) -> float:
        """
        Convert a horizontal offset in pixels to x-axis data units.

        This keeps marker tiles the same *screen width* even if xlim changes.
        """
        inv = ax_.transData.inverted()
        x0, _ = inv.transform((0, 0))
        x1, _ = inv.transform((px, 0))
        return float(x1 - x0)

    # -------------------------------------------------------------------------
    # Base building block: draw a solid "tile" (constant pixel width) in a bar
    # -------------------------------------------------------------------------
    def add_tile_aligned_to_rect(
        rect,
        frac: float,
        facecolor: str,
        edgecolor: str,
        alpha: float,
        px_width: int,
        linewidth: float,
        mode: str = "queue",
    ):
        """
        Draw a SOLID tile aligned to a given bar rectangle.

        frac meaning (only in mode="queue"):
          frac=0   -> front of queue (near the mid)
          frac=1   -> back of queue (far edge)

        mode:
          - "queue": place tile at a FIFO fraction inside the bar
          - "edge" : glue tile to the bar edge facing the mid (execution marker)
        """
        f = max(0.0, min(1.0, float(frac)))

        left = rect.get_x()
        width = rect.get_width()   # >0 ask, <0 bid
        y0 = rect.get_y()
        h = rect.get_height()
        y = y0 + 0.5 * h

        L = abs(width)
        if L <= 0:
            return

        w = data_dx_for_pixels(ax, px_width)

        # Compute tile x_left
        if mode == "queue":
            if width > 0:
                # ASK: bar spans [left, left+L], front at left
                x_center = left + f * L
                x_left = x_center - 0.5 * w
                x_left = max(left, min(left + L - w, x_left))
            else:
                # BID: bar spans [left+width, left], front at right edge (near 0)
                edge = left
                x_center = edge - f * L
                x_left = x_center - 0.5 * w
                x_left = max(edge - L, min(edge - w, x_left))

        elif mode == "edge":
            # Glue to best edge (nearest mid)
            if width > 0:
                x_left = left          # ASK edge
            else:
                edge = left            # BID edge (near 0)
                x_left = edge - w
        else:
            return

        r = _acquire_overlay()
        r.set_xy((x_left, y - 0.5 * h))
        r.set_width(w)
        r.set_height(h)
        r.set_facecolor(facecolor)
        r.set_edgecolor(edgecolor)
        r.set_linewidth(linewidth)
        r.set_alpha(alpha)
        r.set_visible(True)

    # -------------------------------------------------------------------------
    # New: draw a queue segment (size-proportional)
    # -------------------------------------------------------------------------
    def add_queue_segment_aligned_to_rect(
        rect,
        start_frac: float,
        end_frac: float,
        facecolor: str,
        edgecolor: str,
        alpha: float,
        linewidth: float,
        min_px_width: int = 6,
    ):
        """
        Draw a SOLID segment inside the bar representing an order slice.

        Fractions are measured from the *front* of the queue (nearest mid):
          start_frac = vol_ahead / total_queue
          end_frac   = (vol_ahead + own_qty) / total_queue

        For asks (width>0): front is at left edge.
        For bids (width<0): front is at right edge (near x=0).
        """
        s = float(start_frac)
        e = float(end_frac)
        if not np.isfinite(s) or not np.isfinite(e):
            return

        s = max(0.0, min(1.0, s))
        e = max(0.0, min(1.0, e))
        if e <= s:
            return

        left = rect.get_x()
        width = rect.get_width()
        y0 = rect.get_y()
        h = rect.get_height()
        y = y0 + 0.5 * h

        L = abs(width)
        if L <= 0:
            return

        # Convert a minimum visible pixel width into data units
        w_min = data_dx_for_pixels(ax, float(min_px_width))

        if width > 0:
            # ASK: bar spans [left, left+L]
            x_left = left + s * L
            x_right = left + e * L
            w = x_right - x_left

            # enforce minimum visibility by extending towards the back (increasing right)
            if w < w_min:
                x_right = min(left + L, x_left + w_min)
                w = x_right - x_left

        else:
            # BID: bar spans [edge-L, edge], where edge = left (usually 0)
            edge = left
            x_right = edge - s * L     # nearer to mid
            x_left = edge - e * L      # further from mid
            w = x_right - x_left

            if w < w_min:
                # extend away from mid (more to the left)
                x_left = max(edge - L, x_right - w_min)
                w = x_right - x_left

        if w <= 0:
            return

        r = _acquire_overlay()
        r.set_xy((x_left, y - 0.5 * h))
        r.set_width(w)
        r.set_height(h)
        r.set_facecolor(facecolor)
        r.set_edgecolor(edgecolor)
        r.set_linewidth(linewidth)
        r.set_alpha(alpha)
        r.set_visible(True)

    def add_standalone_tile(
        px: float,
        side: int,
        facecolor: str,
        edgecolor: str,
        alpha: float,
        px_width: int,
        linewidth: float,
    ):
        """
        Draw a SOLID standalone tile at y=px, even if px is not in the displayed OB levels.

        side:
          +1 -> draw on bid side (left, negative)
          -1 -> draw on ask side (right, positive)
        """
        w = data_dx_for_pixels(ax, px_width)
        y = float(px)
        left = (-w) if side == +1 else 0.0

        r = _acquire_overlay()
        r.set_xy((left, y - 0.5 * bar_height))
        r.set_width(w)
        r.set_height(bar_height)
        r.set_facecolor(facecolor)
        r.set_edgecolor(edgecolor)
        r.set_linewidth(linewidth)
        r.set_alpha(alpha)
        r.set_visible(True)

    def add_marker_at_price(
        px: float,
        side: int,
        bid_idx_map: dict,
        ask_idx_map: dict,
        bars_bid,
        bars_ask,
        facecolor: str,
        edgecolor: str,
        alpha: float,
        px_width: int,
        linewidth: float,
        *,
        mode: str = "edge",
        frac: float = 0.0,
    ):
        """
        Draw a SOLID marker at a given price.

        - If the price is currently visible in OB (within max_levels), glue marker to that bar.
          * mode="edge"  -> marker at the edge (near mid)
          * mode="queue" -> marker inside bar at FIFO fraction `frac`

        - If not visible, draw a standalone tile at y=px on the appropriate side.
        """
        key = _px_key(px, price_step)

        rect = None
        if side == +1 and key in bid_idx_map and len(bars_bid.patches):
            rect = bars_bid.patches[int(bid_idx_map[key])]
        elif side == -1 and key in ask_idx_map and len(bars_ask.patches):
            rect = bars_ask.patches[int(ask_idx_map[key])]

        if rect is not None:
            add_tile_aligned_to_rect(
                rect=rect,
                frac=frac,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                px_width=px_width,
                linewidth=linewidth,
                mode=mode,
            )
        else:
            add_standalone_tile(
                px=px,
                side=side,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                px_width=px_width,
                linewidth=linewidth,
            )

    # =========================================================================
    # Per-frame update
    # =========================================================================
    def update(i: int):
        nonlocal bars_bid, bars_ask, current_pos
        current_pos = i
        frame_idx = frames_idx[i]

        # 1) Extract OB snapshot
        bid_p, bid_s, ask_p, ask_s = extract_lob_snapshot(ob_df, frame_idx, max_levels=max_levels)

        # 2) Convert sizes to bar widths:
        #    - bids are negative (left)
        #    - asks are positive (right)
        yb, xb = bid_p, -bid_s
        ya, xa = ask_p, ask_s

        # 3) Update base bars
        bars_bid = set_bars(bars_bid, yb, xb, bid_color)
        bars_ask = set_bars(bars_ask, ya, xa, ask_color)

        # 4) Update y-limits (optional: dynamic)
        if len(yb) or len(ya):
            y_min = min(yb.min() if len(yb) else np.inf, ya.min() if len(ya) else np.inf)
            y_max = max(yb.max() if len(yb) else -np.inf, ya.max() if len(ya) else -np.inf)
            if np.isfinite(y_min) and np.isfinite(y_max):
                pad_loc = price_step if price_step is not None else 1.0
                ax.set_ylim(y_min - pad_loc, y_max + pad_loc)

        # 5) Mid line
        if show_mid and ("MidPrice" in message_df.columns):
            mid = float(message_df.iloc[int(frame_idx)]["MidPrice"])
            mid_line.set_ydata([mid, mid])

        # 6) Clear overlays for this frame
        clear_overlays()

        # Build price->bar index maps for quick lookup this frame
        bid_idx = {_px_key(p, price_step): j for j, p in enumerate(bid_p)}
        ask_idx = {_px_key(p, price_step): j for j, p in enumerate(ask_p)}

        # ---------------------------------------------------------------------
        # (A) ALL EVENTS markers (MO yellow, Cancel green), driven by message_df
        # ---------------------------------------------------------------------
        if show_all_events:
            if ("Type" in message_df.columns) and ("Price" in message_df.columns) and ("Direction" in message_df.columns):
                tval = message_df.iloc[int(frame_idx)]["Type"]
                px = float(message_df.iloc[int(frame_idx)]["Price"])
                sd = int(message_df.iloc[int(frame_idx)]["Direction"])

                # Robust type handling (string or numeric)
                is_mo = False
                is_c = False

                if isinstance(tval, str):
                    tt = tval.strip().upper()
                    is_c = (tt == "C") or ("CANCEL" in tt)
                    # "MO" but not "LO"
                    is_mo = (tt == "MO") or ("MO" in tt and "LO" not in tt and tt != "C")
                else:
                    # Numeric convention: 1=MO, 2=Cancel
                    try:
                        tnum = int(tval)
                        is_mo = (tnum == 1)
                        is_c = (tnum == 2)
                    except Exception:
                        pass

                if is_mo:
                    # MO direction:
                    #   +1 buy MO consumes ASK -> mark ASK side
                    #   -1 sell MO consumes BID -> mark BID side
                    mo_side = -1 if sd == +1 else +1
                    add_marker_at_price(
                        px=px,
                        side=mo_side,
                        bid_idx_map=bid_idx,
                        ask_idx_map=ask_idx,
                        bars_bid=bars_bid,
                        bars_ask=bars_ask,
                        facecolor=all_mo_face,
                        edgecolor=all_mo_edge,
                        alpha=all_mo_alpha,
                        px_width=all_mo_px,
                        linewidth=all_mo_linewidth,
                        mode="edge",
                        frac=0.0,
                    )

                if is_c:
                    # Cancel direction:
                    #   +1 cancels BID -> mark BID
                    #   -1 cancels ASK -> mark ASK
                    c_side = +1 if sd == +1 else -1

                    use_queue_pos = ("CancelVolAhead" in message_df.columns) and ("CancelQueueLen" in message_df.columns)

                    if use_queue_pos:
                        va = float(message_df.iloc[int(frame_idx)]["CancelVolAhead"])
                        ql = float(message_df.iloc[int(frame_idx)]["CancelQueueLen"])

                        if np.isfinite(va) and np.isfinite(ql) and (ql > 0.0):
                            frac = max(0.0, min(1.0, va / ql))
                            add_marker_at_price(
                                px=px,
                                side=c_side,
                                bid_idx_map=bid_idx,
                                ask_idx_map=ask_idx,
                                bars_bid=bars_bid,
                                bars_ask=bars_ask,
                                facecolor=all_c_face,
                                edgecolor=all_c_edge,
                                alpha=all_c_alpha,
                                px_width=all_c_px,
                                linewidth=all_c_linewidth,
                                mode="queue",
                                frac=frac,
                            )
                        else:
                            add_marker_at_price(
                                px=px,
                                side=c_side,
                                bid_idx_map=bid_idx,
                                ask_idx_map=ask_idx,
                                bars_bid=bars_bid,
                                bars_ask=bars_ask,
                                facecolor=all_c_face,
                                edgecolor=all_c_edge,
                                alpha=all_c_alpha,
                                px_width=all_c_px,
                                linewidth=all_c_linewidth,
                                mode="edge",
                                frac=0.0,
                            )
                    else:
                        add_marker_at_price(
                            px=px,
                            side=c_side,
                            bid_idx_map=bid_idx,
                            ask_idx_map=ask_idx,
                            bars_bid=bars_bid,
                            bars_ask=bars_ask,
                            facecolor=all_c_face,
                            edgecolor=all_c_edge,
                            alpha=all_c_alpha,
                            px_width=all_c_px,
                            linewidth=all_c_linewidth,
                            mode="edge",
                            frac=0.0,
                        )

        # ---------------------------------------------------------------------
        # (B) MM overlays
        # ---------------------------------------------------------------------
        if mm_df is not None and int(frame_idx) < len(mm_df):
            needed_cols = ["MM_OwnPrices", "MM_OwnSides", "MM_OwnPriorities", "MM_OwnQueueLens"]
            if all(c in mm_df.columns for c in needed_cols):
                mm_row = mm_df.iloc[int(frame_idx)]

                mm_prices = np.array(_parse_listlike(mm_row["MM_OwnPrices"]), dtype=float)
                mm_sides = np.array(_parse_listlike(mm_row["MM_OwnSides"]), dtype=int)
                mm_prios = np.array(_parse_listlike(mm_row["MM_OwnPriorities"]), dtype=float)
                mm_qlens = np.array(_parse_listlike(mm_row["MM_OwnQueueLens"]), dtype=float)

                # Optional: own order qty list (strongly recommended; patch provided)
                qty_col = _first_existing_column(mm_df, ["MM_OwnQtys", "MM_OwnSizes", "MM_OwnOrderQtys"])
                if qty_col is not None:
                    mm_qtys = np.array(_parse_listlike(mm_row[qty_col]), dtype=float)
                else:
                    mm_qtys = None

                for idx, (px, sd, pr, ql) in enumerate(zip(mm_prices, mm_sides, mm_prios, mm_qlens)):
                    px_k = _px_key(px, price_step)

                    rect = None
                    if sd == +1 and px_k in bid_idx and len(bars_bid.patches):
                        rect = bars_bid.patches[int(bid_idx[px_k])]
                    elif sd == -1 and px_k in ask_idx and len(bars_ask.patches):
                        rect = bars_ask.patches[int(ask_idx[px_k])]

                    # If the bar is visible, we can place the segment accurately.
                    if rect is not None:
                        # total queue length for fraction computation
                        total_q = float(ql) if (np.isfinite(ql) and ql > 0.0) else abs(float(rect.get_width()))
                        if not np.isfinite(total_q) or total_q <= 0.0:
                            total_q = 1.0

                        vol_ahead = float(pr) if np.isfinite(pr) else 0.0
                        vol_ahead = max(0.0, min(vol_ahead, total_q))

                        # own quantity: prefer logged MM_OwnQtys; else conservative fallback
                        if mm_qtys is not None and idx < len(mm_qtys) and np.isfinite(mm_qtys[idx]):
                            own_qty = float(mm_qtys[idx])
                        else:
                            # Fallback assumption:
                            # If we only know "ahead" and "total", assume the remainder is ours.
                            # This is accurate when the MM is at the tail (usual insertion) and
                            # no new orders appear behind before the next log tick.
                            own_qty = max(total_q - vol_ahead, 0.0)

                        own_qty = max(0.0, own_qty)
                        # Ensure we do not exceed total queue when plotted
                        if vol_ahead + own_qty > total_q:
                            own_qty = max(total_q - vol_ahead, 0.0)

                        if mm_lo_use_segment and own_qty > 0.0:
                            start_frac = vol_ahead / total_q
                            end_frac = (vol_ahead + own_qty) / total_q
                            add_queue_segment_aligned_to_rect(
                                rect=rect,
                                start_frac=start_frac,
                                end_frac=end_frac,
                                facecolor=mm_lo_face,
                                edgecolor=mm_lo_edge,
                                alpha=mm_lo_alpha,
                                linewidth=mm_lo_linewidth,
                                min_px_width=mm_lo_min_px,
                            )
                        else:
                            # fallback: a small tile at the queue position
                            frac = (vol_ahead / total_q) if total_q > 0 else 0.0
                            add_tile_aligned_to_rect(
                                rect=rect,
                                frac=frac,
                                facecolor=mm_lo_face,
                                edgecolor=mm_lo_edge,
                                alpha=mm_lo_alpha,
                                px_width=mm_lo_fallback_tile_px,
                                linewidth=mm_lo_linewidth,
                                mode="queue",
                            )
                    else:
                        # If price is not visible in the current max_levels snapshot,
                        # we draw a small standalone cyan tile so the order remains visible.
                        add_standalone_tile(
                            px=float(px),
                            side=int(sd),
                            facecolor=mm_lo_face,
                            edgecolor=mm_lo_edge,
                            alpha=mm_lo_alpha,
                            px_width=mm_lo_fallback_tile_px,
                            linewidth=mm_lo_linewidth,
                        )

            # ---------- aggressive trades overlays (MM hits) ----------
            if (not show_all_events) and ("MM_Action" in mm_df.columns):
                act = str(mm_df.iloc[int(frame_idx)]["MM_Action"])

                if act == "cross_buy" and len(ask_p) and len(bars_ask.patches):
                    best_ask_j = int(np.argmin(ask_p))
                    rect = bars_ask.patches[best_ask_j]
                    add_tile_aligned_to_rect(
                        rect=rect,
                        frac=0.0,
                        facecolor=mm_hit_face,
                        edgecolor=mm_hit_edge,
                        alpha=mm_hit_alpha,
                        px_width=mm_hit_px,
                        linewidth=mm_hit_linewidth,
                        mode="edge",
                    )

                elif act == "cross_sell" and len(bid_p) and len(bars_bid.patches):
                    best_bid_j = int(np.argmax(bid_p))
                    rect = bars_bid.patches[best_bid_j]
                    add_tile_aligned_to_rect(
                        rect=rect,
                        frac=0.0,
                        facecolor=mm_hit_face,
                        edgecolor=mm_hit_edge,
                        alpha=mm_hit_alpha,
                        px_width=mm_hit_px,
                        linewidth=mm_hit_linewidth,
                        mode="edge",
                    )

        ax.set_title(f"LOB Evolution — frame {frame_idx}")

        return list(bars_bid.patches) + list(bars_ask.patches) + [mid_line] + _overlay_pool[:_overlay_next]

    # Build the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=range(n_frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # =========================================================================
    # Controls: Prev / Next / Pause / Play
    # =========================================================================
    ax_prev = plt.axes([0.55, 0.02, 0.08, 0.05])
    ax_next = plt.axes([0.65, 0.02, 0.08, 0.05])
    ax_pause = plt.axes([0.75, 0.02, 0.08, 0.05])
    ax_play = plt.axes([0.85, 0.02, 0.08, 0.05])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_pause = Button(ax_pause, "Pause")
    b_play = Button(ax_play, "Play")

    def _pause(event):
        try:
            anim.pause()
        except AttributeError:
            anim.event_source.stop()

    def _play(event):
        nonlocal current_pos
        # Set internal frame sequence to resume from current_pos (with fallback)
        try:
            anim.frame_seq = iter(range(current_pos, n_frames))
        except AttributeError:
            pass
        try:
            anim.resume()
        except AttributeError:
            anim.event_source.start()

    def _next(event):
        nonlocal current_pos
        _pause(event)
        if current_pos < n_frames - 1:
            current_pos += 1
            update(current_pos)
            fig.canvas.draw_idle()

    def _prev(event):
        nonlocal current_pos
        _pause(event)
        if current_pos > 0:
            current_pos -= 1
            update(current_pos)
            fig.canvas.draw_idle()

    b_pause.on_clicked(_pause)
    b_play.on_clicked(_play)
    b_next.on_clicked(_next)
    b_prev.on_clicked(_prev)

    # Keep references to buttons to prevent garbage collection
    anim._controls = {"b_prev": b_prev, "b_next": b_next, "b_pause": b_pause, "b_play": b_play}
    return anim
