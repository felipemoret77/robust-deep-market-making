#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adr_lite_config.py â€” shared ADR-lite reference constants
========================================================

Single source of truth for the two numbers that must be measured
OFFLINE from a frozen evaluation of the previous curriculum step (the
reference policy we're fine-tuning from).  Both the main trainer
(`DeepSarsaQRunner_REGIME.py`) and the Optuna tuner
(`tune_dqn_regime_switch.py`) import from here so the ADR-lite
`linear_mean_gap` schedule stays consistent across runs.

How to measure (recipe)
-----------------------

Run the return-risk frontier (or any long, stationary evaluation) at
`p_buy = 0.5` with the reference checkpoint FROZEN, then compute:

    import numpy as np

    per_ep = np.clip(final_pnl_series, -PNL_CLIP, +PNL_CLIP)
    PREV_CURRICULUM_MEAN = float(per_ep.mean())

    # Non-overlapping blocks of size ADR buffer_size (typically 50).
    # NOTE: this is the std of the BLOCK MEAN, NOT the per-episode std.
    # The ADR thermostat decides on buffer-mean PnL, so the Ïƒ it sees
    # under a stationary reference policy is the std of that buffer
    # mean.  Using `np.std(per_ep)` here (by mistake) understates the
    # effective noise by âˆšbuffer_size and collapses the dead zone.
    buffer_size = 50
    n = len(per_ep) // buffer_size
    blocks = per_ep[:n * buffer_size].reshape(n, -1).mean(axis=1)
    PREV_CURRICULUM_BUFFER_STD = float(blocks.std(ddof=1))

Paste the two numbers below.  Both trainer and tuner will refuse to
start until they are not None.
"""

from __future__ import annotations

import math

# Mean of the reference policy's per-episode final_pnl (clipped) under
# the frozen fair-scenario evaluation.  This becomes `t_H(Î´_start)`
# in the linear_mean_gap schedule.
#
# Temporary IID approximation from the current frozen-fair eval:
#   mean_ep â‰ˆ 0.28
#   std_ep  â‰ˆ 0.16
#
# This is good enough to preview / run the linear_mean_gap schedule, but
# the more statistically correct quantity is still the REAL std of the
# non-overlapping 50-episode block means from the frontier.  Once that is
# available, replace the approximation below with the measured block std.
PREV_CURRICULUM_MEAN: float | None = 0.28

# Std of the BLOCK MEAN (non-overlapping blocks of `buffer_size`
# episodes).  This is the noise floor of the ADR decision variable
# and drives the width of the dead zone
# (dead_zone = gap_sigma Â· PREV_CURRICULUM_BUFFER_STD).
#
# Current ADR buffer size in the runner/tuner is 50, so the IID preview
# for the buffer-mean noise is std_ep / sqrt(50).  Keep this in sync if
# ADR_LITE_BUFFER_SIZE changes.
_REFERENCE_STD_EP = 0.16
_REFERENCE_BUFFER_SIZE = 50

PREV_CURRICULUM_BUFFER_STD: float | None = (
    _REFERENCE_STD_EP / math.sqrt(_REFERENCE_BUFFER_SIZE)
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# OPTIONAL ADR-LITE MECHANISMS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Two orthogonal mechanisms that extend the base ADR-lite thermostat to
# spend more training budget at the hard end of the curriculum without
# contaminating the statistical foundations (Ïƒ_block, pnl_clip) or the
# threshold schedule (t_H(Î´), t_L(Î´)).
#
# BOTH ARE OPT-IN AND DEFAULT TO DISABLED.  The out-of-the-box behavior
# of the runner / tuner is identical to the pre-extension semantics: a
# continuous Î´ stepped by Â±delta_step, with a single advance decision
# immediately moving Î´ up.  Enabling either flag below is a deliberate
# opt-in and requires no changes outside this file.
#
# Why two mechanisms?
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   * Mechanism 1 (`DELTA_SCHEDULE_MODE`) multiplies the NUMBER OF GATES.
#     It replaces the continuous Î´ step with a finite ladder of Î´ values
#     whose spacing is non-uniform â€” denser near Î´_max by construction.
#     Each gate still requires only a single advance decision to pass,
#     but there are more gates to pass in the hard region, so the agent
#     naturally spends more episodes there.  Passes through the easy
#     region are fast; passes through the hard region are slow.  The
#     advance/retreat transitions remain exactly reversible (a retreat
#     undoes the last advance by moving one ladder index down).
#
#   * Mechanism 2 (`ADVANCE_CONFIRMATION_MODE`) multiplies the EVIDENCE
#     PER GATE.  The ladder / step layout is untouched; instead, passing
#     a gate at high Î´ requires K â‰¥ 1 CONSECUTIVE advance decisions
#     before δ actually moves up.  Retreat always resets the counter;
#     how holds behave is controlled by ADVANCE_CONFIRMATION_RESET_MODE
#     below.  K is a function of the current δ: K(δ_min)=1
#     K(Î´_max) = K_max (strictest in the hard region).  This is a
#     "lucky-advance filter": it makes the curriculum advance only when
#     the agent's performance is statistically significant across
#     multiple buffers, not just a single favorable sample.
#
# The two mechanisms are INDEPENDENT and can be combined freely.  With
# both enabled: the discrete ladder defines which Î´ values exist, and
# the confirmation function gates each step with a K(Î´)-consecutive
# advance requirement.
#
# Neither mechanism changes `_current_thresholds()` â€” the t_H(Î´) / t_L(Î´)
# schedule continues to be interpolated linearly in the continuous Î´
# coordinate, which preserves the "threshold reflects physical
# difficulty, not gate count" invariant of linear_mean_gap mode.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MECHANISM 1 â€” Î´ schedule mode
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# "continuous"    â†’ Base ADR behavior.  Î´ is a float, stepped by Â±delta_step
#                   per advance/retreat, clamped to [delta_min, delta_max].
#                   This is the current default and what the running
#                   experiment uses.
# "ladder_auto"   â†’ Î´ is constrained to a formula-generated finite ladder.
#                   The ladder is generated by `generate_auto_ladder(...)`
#                   below from three parameters: curvature (`alpha`), max
#                   allowed gap, and min allowed gap.  With alpha > 1 the
#                   ladder is denser near Î´_max, matching the
#                   "more resolution where learning happens" intuition
#                   without any in-sample fitting of ladder points.
# "ladder_manual" â†’ Î´ is constrained to a hand-picked list of values
#                   supplied via `LADDER_MANUAL` below.  Use this when
#                   domain knowledge dictates specific breakpoints.
DELTA_SCHEDULE_MODE: str = "continuous"

# â”€â”€â”€ Parameters for DELTA_SCHEDULE_MODE == "ladder_auto" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Curvature of the ladder density function.  The generator uses
#
#     f(x) = 1 - (1 - x)^alpha,    x âˆˆ [0, 1]
#
# which is strictly increasing on [0, 1] with f(0)=0, f(1)=1, and
# derivative f'(x) = alphaÂ·(1-x)^(alpha-1).  For alpha > 1 the derivative
# is largest near x=0 and smallest near x=1, so the ladder points
# generated from evenly-spaced x values compress near the high end of Î´
# (i.e. the hard region).  Canonical values:
#
#   alpha = 1.0  â†’ uniform ladder (no density bias)
#   alpha = 1.5  â†’ mildly biased toward the top
#   alpha = 2.0  â†’ moderately biased toward the top
#   alpha = 3.0  â†’ strongly biased toward the top
LADDER_AUTO_ALPHA: float = 1.5

# Upper cap on the largest gap between consecutive ladder points.  With
# the current f(x), the largest gap occurs at the sparsest end (x=0, the
# easy region).  The generator automatically chooses the number of
# ladder points `n` so that this largest gap is at most `max_gap`.  The
# natural choice is 1 price tick (0.01) so a single advance/retreat
# never skips more than one tick of Î´ in the easy region.
LADDER_AUTO_MAX_GAP: float = 0.01

# Lower floor on the smallest allowed gap between consecutive ladder
# points.  Without a floor, alpha > 1 generates sub-tick gaps at the
# hard end of the ladder (multiple ladder points that resolve the same
# physical Î´ to within rounding).  The generator post-processes the raw
# formula ladder by removing points that lie within `min_gap` of the
# previously-kept point, so the hard end of the ladder never exceeds
# resolution `min_gap`.  Default: half a tick (0.005), which keeps the
# hard end at sub-tick but not wastefully dense.
LADDER_AUTO_MIN_GAP: float = 0.005

# â”€â”€â”€ Parameters for DELTA_SCHEDULE_MODE == "ladder_manual" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Explicit list of Î´ breakpoints in strictly increasing order.  The
# first element must equal ADR_LITE_DELTA_MIN and the last must equal
# ADR_LITE_DELTA_MAX (validated at init time).  Leave as None unless
# mode == "ladder_manual".
LADDER_MANUAL: list | None = None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MECHANISM 2 â€” advance confirmation mode
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# "off"     â†’ Base ADR behavior.  A single advance decision (pÌ„ â‰¥ t_H and
#             inventory OK) moves Î´ up by one step/ladder-index.  This is
#             the current default and what the running experiment uses.
# "formula" â†’ A parametric K(Î´) function determines how many CONSECUTIVE
#             advance decisions are required before Î´ actually moves up.
#             The formula is:
#
#                 u(Î´) = (Î´ âˆ’ Î´_min) / (Î´_max âˆ’ Î´_min)  âˆˆ [0, 1]
#                 K(Î´) = 1 + floor((K_max âˆ’ 1) Â· u(Î´)^gamma)
#
#             This produces:
#               K(Î´_min) = 1               (easy region: immediate advance)
#               K(Î´_max) = K_max           (hard region: max stringency)
#               gamma controls WHERE the stringency grows:
#                 gamma = 1  â†’ linear growth
#                 gamma > 1  â†’ growth concentrated near Î´_max
#                 gamma < 1  â†’ growth concentrated near Î´_min
#
#             The exact reset semantics for the confirmation counter are
#             controlled separately by ADVANCE_CONFIRMATION_RESET_MODE
#             below.  In the legacy "strict" mode, any hold or retreat
#             resets the counter to zero.  In "retreat_only", holds
#             preserve the current partial progress and only an actual
#             retreat destroys the streak.
ADVANCE_CONFIRMATION_MODE: str = "formula"

# â”€â”€â”€ Parameters for ADVANCE_CONFIRMATION_MODE == "formula" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Maximum number of consecutive advance decisions required to move Î´ up
# at Î´ = Î´_max.  K_max = 1 effectively disables the mechanism.  K_max = 2
# is a mild filter.  K_max = 4 is aggressive (each advance at Î´_max
# consumes â‰¥ 4 full buffer_size episode blocks = â‰¥ 200 episodes).
ADVANCE_CONFIRMATION_K_MAX: int = 4

# Exponent controlling where the stringency grows.  gamma > 1 means K(Î´)
# stays close to 1 over most of the curriculum and ramps up sharply only
# near Î´_max â€” a good default since it preserves the responsiveness of
# the thermostat in the regime where the warmstart is competent, and
# only tightens the gate where the agent is operating at the edge of
# its capability.
ADVANCE_CONFIRMATION_GAMMA: float = 0.60

# Reset semantics for the advance-confirmation counter.
# "strict"       -> legacy behavior: hold OR retreat resets the counter
# "retreat_only" -> requested behavior: only retreat resets; holds preserve
#                   the accumulated advance_pending count
ADVANCE_CONFIRMATION_RESET_MODE: str = "retreat_only"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LADDER GENERATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def generate_auto_ladder(
    d_min: float,
    d_max: float,
    alpha: float = LADDER_AUTO_ALPHA,
    max_gap: float = LADDER_AUTO_MAX_GAP,
    min_gap: float = LADDER_AUTO_MIN_GAP,
) -> list[float]:
    """Generate a Î´ ladder biased toward dense resolution at Î´_max.

    The ladder is constructed in three stages:

    1. **Number of points.** Given the curvature `alpha` and the cap on
       the largest gap `max_gap`, the required number of raw ladder
       points is derived analytically.  The largest gap of the raw
       formula ladder occurs at the sparsest end (x=0 â†’ Î´=d_min), and
       equals `(d_max - d_min) Â· [1 - (1 - 1/n)^alpha]`.  Setting this
       equal to `max_gap` and solving for n gives:

            n = ceil( 1 / (1 - (1 - max_gap/D)^(1/alpha)) )

       where `D = d_max - d_min` is the full curriculum range.

    2. **Raw ladder.** With x âˆˆ [0, 1] equally spaced at `n` points, the
       density function `f(x) = 1 - (1 - x)^alpha` is applied and the
       output is linearly rescaled to `[d_min, d_max]`.  f is strictly
       increasing and concave for alpha > 1, so the raw ladder is
       monotone and progressively denser as x â†’ 1.

    3. **Sub-tick filter.** To avoid wasted ladder points where adjacent
       values of Î´ resolve to within the same fraction of a tick, the
       raw ladder is walked and any point within `min_gap` of the last
       kept point is dropped.  The final entry (d_max) is always
       preserved so the ladder endpoint matches the curriculum ceiling.

    Parameters
    ----------
    d_min : float
        Lower bound of the curriculum range (typically 0.0 for MM
        regime-switching).
    d_max : float
        Upper bound of the curriculum range (typically 0.30).
    alpha : float, default LADDER_AUTO_ALPHA
        Curvature of the density function.  Must be > 0.  alpha = 1 is
        uniform; alpha > 1 biases toward the top; alpha < 1 biases
        toward the bottom (not recommended for this use case).
    max_gap : float, default LADDER_AUTO_MAX_GAP
        Largest allowed gap between consecutive raw ladder points.
        The generator derives n automatically from this constraint so
        that the resulting ladder has at most `max_gap` spacing
        anywhere.
    min_gap : float, default LADDER_AUTO_MIN_GAP
        Smallest allowed gap between consecutive final ladder points.
        Raw ladder points that would produce a smaller gap are dropped
        during the sub-tick filter.  Set to 0 to disable filtering and
        preserve every raw formula point.

    Returns
    -------
    list[float]
        A strictly increasing ladder of Î´ values whose first element is
        `d_min` and whose last element is `d_max`.

    Raises
    ------
    ValueError
        If the input parameters would produce a degenerate ladder
        (e.g. `d_min >= d_max`, `alpha <= 0`, or `max_gap <= 0`), or
        if the solved `n` exceeds a safety limit of 10_000 (usually
        indicates `max_gap` is too small).
    """
    import math as _math

    if d_max <= d_min:
        raise ValueError(f"Require d_max > d_min; got d_min={d_min}, d_max={d_max}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if max_gap <= 0:
        raise ValueError(f"max_gap must be > 0, got {max_gap}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be >= 0, got {min_gap}")

    D = d_max - d_min
    if max_gap >= D:
        # Degenerate: a single step already covers the whole range.
        return [float(d_min), float(d_max)]

    # Stage 1: derive n from the max_gap constraint.
    inner = 1.0 - max_gap / D                    # âˆˆ (0, 1)
    denom = 1.0 - inner ** (1.0 / alpha)         # gap fraction at x=0 for n=âˆž
    if denom <= 0:
        raise ValueError(
            "Failed to derive n: numerical instability in alpha/max_gap. "
            f"alpha={alpha}, max_gap={max_gap}, D={D}"
        )
    n = int(_math.ceil(1.0 / denom))
    # One extra point so the integer-ceil doesn't occasionally leave the
    # largest gap just slightly above `max_gap` at numerical precision.
    n = max(n + 1, 2)
    if n > 10_000:
        raise ValueError(
            f"Derived ladder size n={n} exceeds safety cap 10_000. "
            f"This usually means max_gap ({max_gap}) is too small "
            f"relative to the curriculum range ({D})."
        )

    # Stage 2: raw formula ladder.
    raw: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        f = 1.0 - (1.0 - x) ** alpha
        raw.append(d_min + D * f)

    # Stage 3: sub-tick filter.  Walk raw left-to-right, keep a point
    # only if it is at least `min_gap` beyond the last kept point.  The
    # endpoint d_max is always preserved to guarantee ladder[-1] == d_max.
    if min_gap == 0:
        filtered = raw
    else:
        filtered = [raw[0]]
        for d in raw[1:-1]:
            if d - filtered[-1] >= min_gap:
                filtered.append(d)
        # Preserve the endpoint explicitly (may or may not satisfy min_gap
        # relative to the last kept point; if it doesn't, replace the last
        # kept point with the endpoint instead of inserting a new one to
        # avoid generating a sub-tick final step).
        if raw[-1] - filtered[-1] >= min_gap:
            filtered.append(raw[-1])
        else:
            filtered[-1] = raw[-1]

    # Numerical hygiene: round to 6 decimals to avoid floating-point
    # drift accumulating across many ladder operations, and deduplicate.
    rounded = sorted({round(float(d), 6) for d in filtered})
    # Guarantee endpoints even after rounding edge cases.
    if rounded[0] != round(float(d_min), 6):
        rounded[0] = round(float(d_min), 6)
    if rounded[-1] != round(float(d_max), 6):
        rounded[-1] = round(float(d_max), 6)

    # â”€â”€â”€ Post-filter max_gap enforcement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # The sub-tick filter in stage 3 occasionally leaves residual gaps
    # slightly above `max_gap` because dropping a point can merge two
    # small gaps into one that exceeds the cap, or because 6-decimal
    # rounding introduces sub-`max_gap` drift.  We walk the rounded
    # ladder once more and SUBDIVIDE any gap that still exceeds
    # `max_gap` by inserting the minimum number of equally-spaced
    # intermediate points such that every resulting gap is at most
    # `max_gap`.  This is a strict-guarantee step: after this loop the
    # contract `max(gaps) <= max_gap` holds unconditionally, which is
    # what `max_gap` advertises to the caller.
    _TOL = 1e-9
    enforced: list[float] = [rounded[0]]
    for d in rounded[1:]:
        gap = d - enforced[-1]
        if gap > max_gap + _TOL:
            # How many subdivisions so each piece <= max_gap
            n_sub = int(_math.ceil(gap / max_gap))
            step = gap / n_sub
            for j in range(1, n_sub):
                enforced.append(round(enforced[-1] + step, 6))
        enforced.append(d)

    # Final deduplicate (subdivision + rounding could re-collide with
    # an already-present point in edge cases).
    enforced = sorted(set(enforced))
    return enforced


def make_advance_confirmation_fn(
    d_min: float,
    d_max: float,
    k_max: int = ADVANCE_CONFIRMATION_K_MAX,
    gamma: float = ADVANCE_CONFIRMATION_GAMMA,
):
    """Return a callable K(Î´) for the `formula` advance-confirmation mode.

    The returned function computes, for any Î´ in [d_min, d_max], the
    number of CONSECUTIVE advance decisions the ADR thermostat must
    observe before actually moving Î´ up from that level.  The formula is

        u(Î´) = clip((Î´ âˆ’ d_min) / (d_max âˆ’ d_min), 0, 1)
        K(Î´) = 1 + floor((k_max âˆ’ 1) Â· u(Î´)^gamma)

    with the following properties:

      * K(d_min) = 1          (no stringency in the easy region)
      * K(d_max) = k_max      (maximum stringency at the ceiling)
      * K is non-decreasing, integer-valued, and bounded above by k_max
      * gamma > 1 concentrates the ramp near d_max (preserves
        responsiveness in the middle of the curriculum)
      * gamma = 1 gives a strictly linear ramp
      * gamma < 1 pushes stringency into the easy region (not recommended)

    Parameters
    ----------
    d_min, d_max : float
        Bounds of the curriculum range.  Must satisfy d_max > d_min.
    k_max : int, default ADVANCE_CONFIRMATION_K_MAX
        Stringency cap at Î´ = d_max.  Must be a positive integer.
        k_max = 1 is a no-op (K(Î´) = 1 everywhere).
    gamma : float, default ADVANCE_CONFIRMATION_GAMMA
        Exponent of the stringency ramp.  Must be > 0.

    Returns
    -------
    Callable[[float], int]
        A closure over (d_min, d_max, k_max, gamma) that maps Î´ â†’ K(Î´).

    Raises
    ------
    ValueError
        If any of the parameters is outside its legal range.
    """
    if d_max <= d_min:
        raise ValueError(f"Require d_max > d_min; got d_min={d_min}, d_max={d_max}")
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    D = float(d_max - d_min)
    k_span = int(k_max) - 1

    def k_of_delta(delta: float) -> int:
        u = (float(delta) - float(d_min)) / D
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        return 1 + int((k_span * (u ** float(gamma))))

    return k_of_delta
