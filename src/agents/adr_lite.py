#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adr_lite.py  â  Faithful adaptation of ADR (Akkaya et al. 2019) for MM
========================================================================

This is a direct adaptation of Algorithm 1 from "Solving Rubik's Cube
with a Robot Hand" (OpenAI, 2019), specialised for the symmetric, 1-D
curriculum we use in market making.

Key points of fidelity to the paper
-----------------------------------

1. ABSOLUTE THRESHOLDS, NO BASELINE.

   The paper defines per-episode performance as the number of successes
   in an episode (an integer in [0, 50] for Rubik's cube).  Buffers
   accumulate these scalars.  The advance/retreat decision compares the
   AVERAGE of the buffer against absolute thresholds (`t_H = 20`,
   `t_L = 10` in Table 15).  There is no calibrated baseline, no
   `success_frac Â· baseline(Î´)` formulation, no relative comparison.

   We mirror this structure exactly.  Per-episode metric: clipped
   `final_pnl` of the episode (a scalar in PnL units, naturally bounded
   by clipping).  Buffer of `m` measurements.  Decision:
       pÌ â AVERAGE(buffer)
       if pÌ â¥ t_H_abs: advance Î´
       elif pÌ â¤ t_L_abs (or inventory blow-up): retreat Î´

2. BUFFER SIZE + BATCH-AND-CLEAR.

   ADR uses `m = 240` for the cube.  We use `m = 50` as a compute
   compromise; the structure (CLEAR after every decision) is identical.

3. CURRICULUM SYMMETRY.

   The paper has `d` independent dimensions with `2d` buffers (`D_i^L`,
   `D_i^H` per dimension), and per-side independent advance/retreat
   decisions.  Our problem has a single dimension (curriculum width Î´)
   that is structurally symmetric (regime switching is symmetric around
   p_buy = 0.5), so we collapse to a single shared buffer and a single
   symmetric Î´.  This is a conscious simplification because per-side
   independence is not meaningful for our 1-D symmetric setting.

4. NO BOUNDARY SAMPLING AT TRAINING TIME.

   Every episode is run on the full curriculum interval
   [0.5 - Î´, 0.5 + Î´], matching both the training distribution and the
   intended deployment regime-switching dynamics.  We do not force the
   policy onto constant-boundary points the way ADR does for parameter
   randomization, because for our 1-D MM problem boundary samples test
   a distribution (constant p_buy) that does not exist in deployment.

5. INVENTORY VETO IS RETREAT-ONLY.

   The DQN's training reward already penalizes inventory growth via
   `inv_penalty` and `inv_wall`.  The thermostat does not need a
   separate inventory check on advance.  We keep an inventory veto
   only as a *retreat* emergency brake, in case the policy enters a
   high-inventory regime that the rolling MA50 of final_pnl is too
   slow to detect.

Hyperparameters in the spirit of ADR
------------------------------------

ADR Table 15 (Akkaya et al. 2019, Rubik's cube):
    m   = 240     # samples per buffer
    t_H = 20      # advance if mean(buffer) â¥ 20 successes/episode
    t_L = 10      # retreat if mean(buffer) â¤ 10
    Î   = 0.02    # parameter step size
    p_b = 0.5     # boundary sampling probability

This module's defaults (MM domain, units of PnL):
    m       = 50      # smaller buffer (compute-bound)
    t_H_abs = +0.10   # advance if mean PnL â¥ +0.10 ticks/episode
    t_L_abs =  0.00   # retreat if mean PnL â¤ 0 (no longer profitable)
    Î       = 0.01    # delta_step
    pnl_clip = 1.0    # clip metric to [-1.0, +1.0] (analogue of [0, 50])

Integration
-----------

    from adr_lite import ADRLiteCurriculum

    curriculum = ADRLiteCurriculum(
        delta_start=0.05, delta_max=0.30, delta_step=0.01,
        buffer_size=50,
        t_H_abs=+0.10, t_L_abs=0.00,
        pnl_clip=1.0,
        inv_advance_veto_min=0.05,
        inv_retreat_min=0.10,
    )

    for ep in range(N_EPISODES):
        p_lo, p_hi, mode = curriculum.sample_episode_config()
        ...run training episode...
        events = curriculum.report_episode(
            final_pnl=episode_final_pnl,
            pct_at_inv_limit=episode_pct_at_inv_limit,
        )

References
----------
Akkaya et al. 2019 â "Solving Rubik's Cube with a Robot Hand" â arXiv:1910.07113
    Algorithm 1, Section 5.2 and Appendix C.3, Table 15.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  ADR CURRICULUM (faithful single-buffer thermostat)
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class ADRLiteCurriculum:
    """Single-buffer ADR-style thermostat over symmetric Î´.

    Drop-in replacement for the previous baseline-relative ADR-lite.
    Implements Algorithm 1 of Akkaya et al. 2019 with `final_pnl` as the
    per-episode performance metric, absolute thresholds, and a single
    shared buffer (the per-dimension dual-buffer mechanism of the paper
    is not meaningful for our 1-D symmetric curriculum).

    Parameters
    ----------
    delta_start, delta_max, delta_min
        Initial, maximum, and minimum half-width of the regime-switching
        interval.  The curriculum gates this single scalar.
    delta_step
        Amount Î´ changes per advance/retreat.
    buffer_size
        m â number of episodes accumulated before a decision.
    t_H_abs
        Absolute advance threshold.  Advance Î´ if the buffer average of
        clipped final_pnl is â¥ t_H_abs (in PnL units, ticks).  Default
        +0.10 means "the policy is averaging at least 10 ticks per
        episode of profit at this difficulty level".
    t_L_abs
        Absolute retreat threshold.  Retreat Î´ if the buffer average is
        â¤ t_L_abs.  Default 0.00 means "the policy is no longer
        consistently profitable".  Strictly less than t_H_abs.
    pnl_clip
        Clip the per-episode metric to [-pnl_clip, +pnl_clip] before
        appending to the buffer.  Mirrors the natural [0, 50] bounding
        of the success count in ADR's original Rubik's cube setup, and
        prevents single catastrophic episodes from dominating the
        rolling average.
    inv_advance_veto_min
        Optional advance-only brake: if the rolling inventory-breach
        average exceeds this threshold, advance is blocked for that
        decision, but the curriculum does not retreat on inventory
        grounds alone.
    inv_retreat_min
        Optional emergency brake: if the rolling inventory-breach
        average exceeds this threshold, force a retreat regardless of
        the gate signal.
    inv_history_len
        Rolling window length for the retreat-only inventory veto.
    rng
        Optional numpy Generator for reproducible sampling.
    """

    MODE_INTERVAL = "interval"

    # Threshold modes (passed via `threshold_mode` kwarg).
    THRESHOLD_CONSTANT             = "constant"
    THRESHOLD_LINEAR_PHASE_A       = "linear_phase_a_anchored"
    # `linear_mean_gap`: anchor on Phase A SUSTAINED MEAN (not peak),
    # decouple t_L from t_H via a fixed statistical gap of kÂ·Ï_block50,
    # and floor t_L at `convergence_floor` (default 0.0 = break-even).
    # The gap width is chosen from the NOISE OF THE ADR DECISION VARIABLE
    # (buffer-mean PnL), measured on non-overlapping blocks of 50 from a
    # frozen Phase A eval (e.g. return-risk frontier at p_buy=0.5).
    THRESHOLD_LINEAR_MEAN_GAP      = "linear_mean_gap"

    def __init__(
        self,
        *,
        delta_start:     float = 0.05,
        delta_max:       float = 0.30,
        delta_min:       float = 0.05,
        delta_step:      float = 0.01,
        buffer_size:     int   = 50,
        # ââ Threshold mode âââââââââââââââââââââââââââââââââââââââââââ
        # "constant": classic ADR-style absolute thresholds (t_H_abs,
        #   t_L_abs are constants, the same at every Î´).
        # "linear_phase_a_anchored": the thresholds are linear functions
        #   of the current Î´.  At Î´_start, t_H = phase_a_best_ma50 -
        #   sigma_margin Â· sigma_ma50 (i.e. demand near-Phase-A
        #   competence).  At Î´_max, t_H decays to t_H_abs_end (default 0
        #   = breakeven floor).  t_L follows a parallel linear schedule
        #   from t_L_abs_start (default 0) at Î´_start to t_L_abs_end
        #   (default âsigma_marginÂ·sigma_ma50) at Î´_max.  Linear
        #   interpolation between the two endpoints.
        threshold_mode:  str   = THRESHOLD_CONSTANT,
        # Constant-mode params (used when threshold_mode == "constant"):
        t_H_abs:         float = +0.10,
        t_L_abs:         float =  0.00,
        # Linear-mode params (used when threshold_mode == "linear_phase_a_anchored"):
        phase_a_best_ma50: float = 0.314,
        sigma_ma50:        float = 0.04,
        sigma_margin:      float = 3.0,
        t_H_abs_end:       Optional[float] = None,
        t_L_abs_start:     float = 0.00,
        t_L_abs_end:       Optional[float] = None,   # default: -sigma_margin Â· sigma_ma50
        # Mean-gap linear params (used when threshold_mode == "linear_mean_gap"):
        #   t_H_start   = prev_curriculum_mean
        #   t_H_end     = t_H_abs_end   (derived by default; see below)
        #   gap         = gap_sigma Â· prev_curriculum_buffer_std
        #   t_L(Î´)      = max(convergence_floor, t_H(Î´) - gap)
        #
        # Both constants MUST be measured offline from a frozen
        # evaluation of the previous curriculum step (the "reference"
        # policy â typically whatever warmstart you're fine-tuning from).
        # Recipe, given `per_ep = np.clip(final_pnl_series, -clip, +clip)`:
        #
        #     prev_curriculum_mean = per_ep.mean()
        #
        #     # Non-overlapping blocks of size `buffer_size`.  NOTE: this
        #     # is the std of the BLOCK MEAN, not the std of the raw
        #     # per-episode values â it is the noise of the exact quantity
        #     # the thermostat decides on (mean over a buffer of
        #     # `buffer_size` episodes), so the resulting dead zone has a
        #     # statistically honest interpretation.
        #     n = len(per_ep) // buffer_size
        #     blocks = per_ep[:n*buffer_size].reshape(n, -1).mean(axis=1)
        #     prev_curriculum_buffer_std = blocks.std(ddof=1)
        prev_curriculum_mean:       Optional[float] = None,
        prev_curriculum_buffer_std: Optional[float] = None,
        gap_sigma:                  float = 2.0,
        convergence_floor:          float = 0.0,
        # âââ OPTIONAL EXTENSION 1 â discrete Î´ ladder âââââââââââââââââ
        #
        # If supplied (non-None), `delta` becomes a DISCRETE quantity
        # drawn from this strictly-increasing list.  Advances move one
        # index up, retreats move one index down, and the threshold
        # schedule `_current_thresholds()` continues to interpolate
        # linearly in the continuous Î´ value (never in the index).  The
        # list must satisfy:
        #     ladder[0]  == delta_min
        #     ladder[-1] == delta_max
        #     delta_start â ladder (else snapped to nearest entry)
        # and must be strictly increasing.  When None, the curriculum
        # uses the legacy continuous Î´ stepped by Â±delta_step (the
        # default, backward-compatible behavior).
        delta_ladder: Optional[List[float]] = None,
        # âââ OPTIONAL EXTENSION 2 â advance confirmation gate âââââââââ
        #
        # If supplied (non-None), this callable determines how many
        # CONSECUTIVE advance decisions must be observed before Î´
        # actually moves up.  The function signature is
        #     advance_confirmations_fn(delta: float) -> int
        # and the returned integer must be >= 1.  Typical use is a
        # non-decreasing K(Î´) that returns 1 in the easy region (no
        # stringency added) and k_max in the hard region (strict
        # "sustained dominance" requirement).  How holds interact with
        # the counter is controlled by `advance_confirmation_reset_mode`
        # below; retreats always reset.
        # When None, a single advance decision moves Î´ immediately â
        # the default, backward-compatible behavior.
        advance_confirmations_fn: Optional[Callable[[float], int]] = None,
        # "strict" = hold/retreat reset the counter (legacy behavior)
        # "retreat_only" = hold preserves the counter, only retreat resets
        advance_confirmation_reset_mode: str = "strict",
        # Common
        pnl_clip:             float = 1.0,
        inv_advance_veto_min: Optional[float] = None,
        inv_retreat_min:      float = 0.10,
        inv_history_len: int   = 100,
        rng:             Optional[np.random.Generator] = None,
    ):
        # ââ Mode validation âââââââââââââââââââââââââââââââââââââââââââââ
        _valid_modes = (
            self.THRESHOLD_CONSTANT,
            self.THRESHOLD_LINEAR_PHASE_A,
            self.THRESHOLD_LINEAR_MEAN_GAP,
        )
        if threshold_mode not in _valid_modes:
            raise ValueError(
                f"threshold_mode must be one of {_valid_modes}; got {threshold_mode!r}"
            )

        # ââ Common validation âââââââââââââââââââââââââââââââââââââââââ
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got {buffer_size}")
        if pnl_clip <= 0:
            raise ValueError(f"pnl_clip must be > 0, got {pnl_clip}")
        if not (delta_min <= delta_start <= delta_max):
            raise ValueError(
                f"Require delta_min â¤ delta_start â¤ delta_max; got "
                f"delta_min={delta_min}, delta_start={delta_start}, "
                f"delta_max={delta_max}"
            )
        if delta_step <= 0:
            raise ValueError(f"delta_step must be > 0, got {delta_step}")
        if (
            inv_advance_veto_min is not None
            and not (0.0 <= inv_advance_veto_min <= 1.0)
        ):
            raise ValueError(
                "Require 0 â¤ inv_advance_veto_min â¤ 1 when provided; got "
                f"{inv_advance_veto_min}"
            )
        if not (0.0 <= inv_retreat_min <= 1.0):
            raise ValueError(
                f"Require 0 â¤ inv_retreat_min â¤ 1; got {inv_retreat_min}"
            )
        if (
            inv_advance_veto_min is not None
            and inv_advance_veto_min > inv_retreat_min
        ):
            raise ValueError(
                "Require inv_advance_veto_min â¤ inv_retreat_min so the "
                "advance veto engages no later than the emergency retreat; "
                f"got veto={inv_advance_veto_min}, retreat={inv_retreat_min}"
            )

        # ââ Mode-specific validation ââââââââââââââââââââââââââââââââââââ
        if threshold_mode == self.THRESHOLD_CONSTANT:
            if not (t_L_abs < t_H_abs):
                raise ValueError(
                    f"Require t_L_abs < t_H_abs; got t_L={t_L_abs}, t_H={t_H_abs}"
                )
            if t_H_abs > pnl_clip:
                raise ValueError(
                    f"t_H_abs ({t_H_abs}) > pnl_clip ({pnl_clip}): "
                    f"advance threshold is unreachable after clipping. "
                    f"Either lower t_H_abs to â¤ pnl_clip or raise pnl_clip."
                )
            if t_L_abs < -pnl_clip:
                raise ValueError(
                    f"t_L_abs ({t_L_abs}) < -pnl_clip ({-pnl_clip}): "
                    f"retreat-by-performance threshold is unreachable after "
                    f"clipping. Either raise t_L_abs or raise pnl_clip."
                )
        elif threshold_mode == self.THRESHOLD_LINEAR_PHASE_A:
            # Legacy default for the peak-anchored mode (pre-mean-gap).
            # Use 0.05 (not 0.00) because the runner's historical
            # working config paired this mode with t_L_abs_end = 0.01,
            # and the validator below requires t_L_abs_end < t_H_abs_end.
            # A 0.00 default would spuriously break any attempt to flip
            # back to the legacy mode from the current runner config.
            if t_H_abs_end is None:
                t_H_abs_end = 0.05
            if phase_a_best_ma50 <= 0:
                raise ValueError(
                    f"phase_a_best_ma50 must be > 0, got {phase_a_best_ma50}"
                )
            if sigma_ma50 <= 0:
                raise ValueError(f"sigma_ma50 must be > 0, got {sigma_ma50}")
            if sigma_margin < 0:
                raise ValueError(f"sigma_margin must be â¥ 0, got {sigma_margin}")
            # Default t_L_abs_end = -sigma_margin Â· sigma_ma50 if not set
            if t_L_abs_end is None:
                t_L_abs_end = -float(sigma_margin) * float(sigma_ma50)
            # Compute the implied t_H_start to validate it
            t_H_abs_start_implied = (
                float(phase_a_best_ma50) - float(sigma_margin) * float(sigma_ma50)
            )
            # The two endpoints (start and end) must each be reachable
            # after clipping, AND the t_L line must lie strictly below
            # the t_H line at both endpoints (so the linear interpolation
            # never crosses).
            if t_H_abs_start_implied > pnl_clip:
                raise ValueError(
                    f"Linear t_H_abs_start = phase_a_best_ma50 - sigma_margin*sigma_ma50 "
                    f"= {t_H_abs_start_implied:.4f} > pnl_clip ({pnl_clip}); "
                    f"advance threshold unreachable at Î´=Î´_start. "
                    f"Either lower sigma_margin, lower phase_a_best_ma50, "
                    f"or raise pnl_clip."
                )
            if t_H_abs_end > pnl_clip:
                raise ValueError(
                    f"Linear t_H_abs_end ({t_H_abs_end}) > pnl_clip ({pnl_clip}); "
                    f"advance threshold unreachable at Î´=Î´_max. "
                    f"Lower t_H_abs_end or raise pnl_clip."
                )
            if t_L_abs_start < -pnl_clip:
                raise ValueError(
                    f"Linear t_L_abs_start ({t_L_abs_start}) < -pnl_clip "
                    f"({-pnl_clip}); retreat threshold unreachable at Î´=Î´_start."
                )
            if t_L_abs_end < -pnl_clip:
                raise ValueError(
                    f"Linear t_L_abs_end ({t_L_abs_end}) < -pnl_clip "
                    f"({-pnl_clip}); retreat threshold unreachable at Î´=Î´_max. "
                    f"Either raise t_L_abs_end / lower sigma_margin / raise pnl_clip."
                )
            if not (t_L_abs_start < t_H_abs_start_implied):
                raise ValueError(
                    f"Linear mode requires t_L_abs_start ({t_L_abs_start}) < "
                    f"t_H_abs_start ({t_H_abs_start_implied}) at Î´=Î´_start"
                )
            if not (t_L_abs_end < t_H_abs_end):
                raise ValueError(
                    f"Linear mode requires t_L_abs_end ({t_L_abs_end}) < "
                    f"t_H_abs_end ({t_H_abs_end}) at Î´=Î´_max"
                )

        if threshold_mode == self.THRESHOLD_LINEAR_MEAN_GAP:
            if prev_curriculum_mean is None:
                raise ValueError(
                    "threshold_mode='linear_mean_gap' requires "
                    "prev_curriculum_mean to be set (measured offline from "
                    "a frozen evaluation of the previous curriculum step, "
                    "e.g. the return-risk frontier at p_buy=0.5). Got None."
                )
            if prev_curriculum_buffer_std is None:
                raise ValueError(
                    "threshold_mode='linear_mean_gap' requires "
                    "prev_curriculum_buffer_std to be set (std of the "
                    f"non-overlapping buffer_size={buffer_size}-episode "
                    "block MEANS â NOT the per-episode std â from a frozen "
                    "reference evaluation). Got None."
                )
            # Derive t_H_abs_end from the noise scale if not explicitly
            # set.  The principled default is:
            #   t_H_end = convergence_floor + gap_sigma Â· prev_curriculum_buffer_std
            # This makes the dead zone uniform across Î´ (always exactly
            # gap_sigmaÂ·Ï wide), with the convergence_floor just barely
            # touching t_L at Î´_max.  Overriding this kwarg lets the user
            # pick a different philosophy (e.g. narrower dead zone at
            # Î´_max, or stricter advance requirement) but the default is
            # zero magic numbers.
            if t_H_abs_end is None:
                t_H_abs_end = (
                    float(convergence_floor)
                    + float(gap_sigma) * float(prev_curriculum_buffer_std)
                )
            if prev_curriculum_mean <= 0:
                raise ValueError(
                    f"prev_curriculum_mean must be > 0, got "
                    f"{prev_curriculum_mean}"
                )
            if prev_curriculum_buffer_std <= 0:
                raise ValueError(
                    f"prev_curriculum_buffer_std must be > 0, got "
                    f"{prev_curriculum_buffer_std}"
                )
            if gap_sigma <= 0:
                raise ValueError(
                    f"gap_sigma must be > 0 (gap_sigma=0 collapses the dead "
                    f"zone and causes flip-flop), got {gap_sigma}"
                )
            if convergence_floor < -pnl_clip:
                raise ValueError(
                    f"convergence_floor ({convergence_floor}) < -pnl_clip "
                    f"({-pnl_clip}): floor is unreachable after clipping."
                )
            if prev_curriculum_mean > pnl_clip:
                raise ValueError(
                    f"prev_curriculum_mean ({prev_curriculum_mean}) > "
                    f"pnl_clip ({pnl_clip}): anchor unreachable after "
                    f"clipping. Either raise pnl_clip or verify the "
                    f"measured reference mean."
                )
            if t_H_abs_end > pnl_clip:
                raise ValueError(
                    f"t_H_abs_end ({t_H_abs_end}) > pnl_clip ({pnl_clip}); "
                    f"advance threshold unreachable at Î´=Î´_max."
                )
            if t_H_abs_end <= convergence_floor:
                raise ValueError(
                    f"t_H_abs_end ({t_H_abs_end}) must be > convergence_floor "
                    f"({convergence_floor}); otherwise t_L â¥ t_H at Î´=Î´_max "
                    f"and the dead zone disappears."
                )
            if prev_curriculum_mean <= convergence_floor:
                raise ValueError(
                    f"prev_curriculum_mean ({prev_curriculum_mean}) must "
                    f"be > convergence_floor ({convergence_floor}); otherwise "
                    f"t_L â¥ t_H at Î´=Î´_start."
                )

        # Any mode that didn't explicitly set t_H_abs_end gets a safe
        # default of 0.00 so the common state-persist block below can
        # always do `float(t_H_abs_end)` without a NoneType error.
        # (Constant mode ignores this field entirely; linear modes set
        # their own meaningful value in the branches above.)
        if t_H_abs_end is None:
            t_H_abs_end = 0.00

        # ââ Optional-extension validation (delta_ladder, advance_fn) âââââ
        #
        # These two kwargs gate opt-in mechanisms that are orthogonal to
        # the threshold mode validated above.  Both are None by default
        # (producing the legacy continuous / single-confirmation
        # behavior) and must pass structural checks when supplied.
        if delta_ladder is not None:
            if not isinstance(delta_ladder, (list, tuple)):
                raise TypeError(
                    f"delta_ladder must be a list/tuple of floats, got "
                    f"{type(delta_ladder).__name__}"
                )
            if len(delta_ladder) < 2:
                raise ValueError(
                    f"delta_ladder must have at least 2 entries (got "
                    f"{len(delta_ladder)}); a ladder with fewer than 2 "
                    f"points cannot represent a non-trivial curriculum."
                )
            _ladder_list = [float(d) for d in delta_ladder]
            # Strictly increasing
            for i in range(len(_ladder_list) - 1):
                if not (_ladder_list[i] < _ladder_list[i + 1]):
                    raise ValueError(
                        f"delta_ladder must be strictly increasing; "
                        f"violation at index {i}: "
                        f"{_ladder_list[i]} >= {_ladder_list[i + 1]}"
                    )
            # Endpoint match (with tolerance for numerical rounding from
            # the auto-generator).
            if abs(_ladder_list[0] - float(delta_min)) > 1e-6:
                raise ValueError(
                    f"delta_ladder[0] ({_ladder_list[0]}) must equal "
                    f"delta_min ({delta_min}) within 1e-6."
                )
            if abs(_ladder_list[-1] - float(delta_max)) > 1e-6:
                raise ValueError(
                    f"delta_ladder[-1] ({_ladder_list[-1]}) must equal "
                    f"delta_max ({delta_max}) within 1e-6."
                )
            # Snap endpoints exactly (avoid drift after rounding).
            _ladder_list[0]  = float(delta_min)
            _ladder_list[-1] = float(delta_max)
        else:
            _ladder_list = None

        if advance_confirmations_fn is not None:
            if not callable(advance_confirmations_fn):
                raise TypeError(
                    f"advance_confirmations_fn must be callable, got "
                    f"{type(advance_confirmations_fn).__name__}"
                )
            # Probe the function at the endpoints to fail fast if it
            # returns a non-int or a value < 1 for either boundary.  We
            # intentionally do not probe across the full range â the
            # user's K(Î´) can be arbitrarily complex and a pointwise
            # probe is more permissive than a monotonicity check.
            for _probe_delta in (float(delta_min), float(delta_max)):
                _probe_val = advance_confirmations_fn(_probe_delta)
                if not isinstance(_probe_val, int):
                    raise TypeError(
                        f"advance_confirmations_fn({_probe_delta}) returned "
                        f"{_probe_val!r} (type {type(_probe_val).__name__}); "
                        f"expected int."
                    )
                if _probe_val < 1:
                    raise ValueError(
                        f"advance_confirmations_fn({_probe_delta}) returned "
                        f"{_probe_val}; must be >= 1 (K=1 means 'no "
                        f"extra confirmation required')."
                    )
        _valid_reset_modes = ("strict", "retreat_only")
        if advance_confirmation_reset_mode not in _valid_reset_modes:
            raise ValueError(
                f"advance_confirmation_reset_mode must be one of "
                f"{_valid_reset_modes}; got {advance_confirmation_reset_mode!r}"
            )

        # ââ Persist state âââââââââââââââââââââââââââââââââââââââââââââ
        self.delta = float(delta_start)
        self.delta_start = float(delta_start)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.delta_step = float(delta_step)

        self.m = int(buffer_size)
        self.threshold_mode = str(threshold_mode)

        # Constant-mode state (always set; ignored in linear mode)
        self.t_H_abs = float(t_H_abs)
        self.t_L_abs = float(t_L_abs)

        # Linear-mode state (always set; ignored in constant mode)
        self.phase_a_best_ma50 = float(phase_a_best_ma50)
        self.sigma_ma50 = float(sigma_ma50)
        self.sigma_margin = float(sigma_margin)
        # Cache the derived endpoints so logging/debug can read them.
        # In linear_mean_gap mode these are overwritten below with the
        # mean-anchored start point.
        if threshold_mode == self.THRESHOLD_LINEAR_MEAN_GAP:
            self._t_H_abs_start = float(prev_curriculum_mean)
        else:
            self._t_H_abs_start = (
                self.phase_a_best_ma50 - self.sigma_margin * self.sigma_ma50
            )
        self._t_H_abs_end = float(t_H_abs_end)
        self._t_L_abs_start = float(t_L_abs_start)
        self._t_L_abs_end = float(t_L_abs_end) if t_L_abs_end is not None else (
            -self.sigma_margin * self.sigma_ma50
        )

        # Mean-gap state (used when threshold_mode == "linear_mean_gap";
        # harmless in other modes).  prev_curriculum_mean and
        # prev_curriculum_buffer_std are None by default â any non-mean-gap
        # mode leaves them as None, so downstream consumers can distinguish.
        self.prev_curriculum_mean = (
            float(prev_curriculum_mean)
            if prev_curriculum_mean is not None else None
        )
        self.prev_curriculum_buffer_std = (
            float(prev_curriculum_buffer_std)
            if prev_curriculum_buffer_std is not None else None
        )
        self.gap_sigma = float(gap_sigma)
        self.convergence_floor = float(convergence_floor)
        # Pre-compute the absolute gap in PnL units (constant across Î´
        # because both gap_sigma and prev_curriculum_buffer_std are
        # frozen at init).
        self._mean_gap_abs = (
            self.gap_sigma * self.prev_curriculum_buffer_std
            if self.prev_curriculum_buffer_std is not None else None
        )

        self.pnl_clip = float(pnl_clip)
        self.inv_advance_veto_min = (
            None if inv_advance_veto_min is None
            else float(inv_advance_veto_min)
        )
        self.inv_retreat_min = float(inv_retreat_min)

        self.buffer: list = []
        self.inv_history: deque = deque(maxlen=int(inv_history_len))

        self.rng = rng if rng is not None else np.random.default_rng()

        # ââ Optional-extension state âââââââââââââââââââââââââââââââââââââ
        #
        # `_delta_ladder` is either None (continuous mode) or the
        # validated list of ladder points.  When non-None, `_ladder_idx`
        # tracks the current position and `self.delta` is kept in sync
        # with `_delta_ladder[_ladder_idx]`.  The initial index is
        # chosen by snapping `delta_start` to the nearest ladder entry
        # (exact match whenever `delta_start` is listed; otherwise we
        # pick the closest and log a warning-free snap â the banner
        # will show the actual starting Î´ so any mismatch is visible).
        self._delta_ladder: Optional[List[float]] = _ladder_list
        if self._delta_ladder is not None:
            _diffs = [abs(d - float(delta_start)) for d in self._delta_ladder]
            self._ladder_idx = int(np.argmin(np.asarray(_diffs)))
            self.delta = float(self._delta_ladder[self._ladder_idx])
        else:
            self._ladder_idx = -1   # unused sentinel

        # `_advance_confirmations_fn` is either None (single-decision
        # advance â the base behavior) or a callable K(Î´) returning the
        # number of consecutive advance decisions required to actually
        # move Î´ up.  `_consecutive_advance_signals` is the counter
        # maintained during report_episode; it increments on advance
        # signals; the reset semantics for holds are controlled by
        # `_advance_confirmation_reset_mode`, while retreats always reset.
        self._advance_confirmations_fn: Optional[Callable[[float], int]] = (
            advance_confirmations_fn
        )
        self._advance_confirmation_reset_mode = str(
            advance_confirmation_reset_mode
        )
        self._consecutive_advance_signals = 0

        self._stats = dict(
            advances=0, retreats=0, holds=0, decisions=0,
            # `advance_pending` counts decisions where an advance signal
            # fired but the multi-confirmation counter had not yet
            # reached K(Î´), so Î´ did not move.  These decisions are
            # neither `advance`, `hold`, nor `retreat` in the event
            # dict â they represent an intermediate state.  When
            # `_advance_confirmations_fn is None`, this counter is
            # always 0.
            advance_pending=0,
            inv_advance_vetoes=0,
            episode_count=0,
            # Convergence counters (used by the runner's early-stop logic)
            consecutive_at_max=0,           # # of decisions in a row where Î´ stayed at delta_max
            decisions_since_last_advance=0, # # of decisions since the last ACTUAL Î´ move-up
        )

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  Î´ movement helper (continuous step vs discrete ladder)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def _apply_delta_move(self, direction: int) -> None:
        """Move Î´ one step in the given direction.

        Dispatches between the two Î´-schedule modes:

          * **Ladder mode** (`self._delta_ladder is not None`): advance
            or retreat the ladder index by one, clamped to
            `[0, len(ladder)-1]`, and copy the corresponding ladder
            value into `self.delta`.  Advances and retreats are exactly
            reversible â `advance then retreat` returns Î´ to its
            previous value.
          * **Continuous mode** (`self._delta_ladder is None`, the
            default): step `self.delta` by `Â±delta_step` and clamp to
            `[delta_min, delta_max]`.  This is the legacy behavior
            implemented before the ladder extension was introduced.

        Parameters
        ----------
        direction : int
            +1 for advance, -1 for retreat.  Any other value is a
            programming error.

        Notes
        -----
        The method does NOT touch the advance-confirmation counter
        (`_consecutive_advance_signals`) or any stats counter; it is
        concerned purely with updating `self.delta` (and, in ladder
        mode, `self._ladder_idx`).  Callers in `report_episode` are
        responsible for resetting confirmation state and incrementing
        advance/retreat counters as appropriate.
        """
        if direction not in (+1, -1):
            raise ValueError(
                f"_apply_delta_move: direction must be +1 or -1, got {direction}"
            )

        if self._delta_ladder is not None:
            new_idx = self._ladder_idx + direction
            if new_idx < 0:
                new_idx = 0
            elif new_idx >= len(self._delta_ladder):
                new_idx = len(self._delta_ladder) - 1
            self._ladder_idx = new_idx
            self.delta = float(self._delta_ladder[new_idx])
        else:
            if direction == +1:
                self.delta = min(self.delta + self.delta_step, self.delta_max)
            else:
                self.delta = max(self.delta - self.delta_step, self.delta_min)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  Threshold computation (constant or linear-anchored)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def _current_thresholds(self) -> Tuple[float, float]:
        """Return (t_H, t_L) for the CURRENT Î´.

        In constant mode these are fixed.  In linear-anchored mode they
        are linear interpolations between the start (at Î´_start) and end
        (at Î´_max) values.  In linear_mean_gap mode, t_H is a linear
        interpolation from `prev_curriculum_mean` to `t_H_abs_end`,
        and t_L follows via `max(convergence_floor, t_H â gap)`.
        """
        if self.threshold_mode == self.THRESHOLD_CONSTANT:
            return self.t_H_abs, self.t_L_abs

        # Linear interpolation in Î´-progress (shared by both linear modes)
        denom = (self.delta_max - self.delta_start)
        if denom <= 0:
            progress = 0.0
        else:
            progress = (self.delta - self.delta_start) / denom
            progress = min(1.0, max(0.0, progress))

        t_H = self._t_H_abs_start * (1.0 - progress) + self._t_H_abs_end * progress

        if self.threshold_mode == self.THRESHOLD_LINEAR_MEAN_GAP:
            # t_L coupled to t_H via a fixed statistical gap, floored at
            # the convergence_floor (default 0 = strict break-even).
            t_L = max(self.convergence_floor, t_H - self._mean_gap_abs)
        else:
            # THRESHOLD_LINEAR_PHASE_A â independent linear interp for t_L.
            t_L = (
                self._t_L_abs_start * (1.0 - progress)
                + self._t_L_abs_end * progress
            )
        return float(t_H), float(t_L)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  Episode config sampler (called BEFORE each episode)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def sample_episode_config(self) -> Tuple[float, float, str]:
        """Return the regime-switching interval for the next episode.

        Always returns the full curriculum interval â there is no
        boundary sampling in this faithful adaptation, since constant-
        boundary stress tests are misaligned with our deployment
        distribution.
        """
        delta = self.delta
        p_lo_range = 0.5 - delta
        p_hi_range = 0.5 + delta
        self._stats["episode_count"] += 1
        return (p_lo_range, p_hi_range, self.MODE_INTERVAL)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  Episode result reporter (called AFTER each episode)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def report_episode(
        self,
        final_pnl: float,
        pct_at_inv_limit: float,
    ) -> dict:
        """Update curriculum state with the result of the last episode.

        Parameters
        ----------
        final_pnl
            The per-episode terminal PnL of the market maker.  This is
            the absolute, interpretable measure of episode quality â
            "did the agent make money this episode?" â and is the
            equivalent of ADR's per-episode success count.
        pct_at_inv_limit
            Fraction of steps where |inventory| reached the hard cap.
            Used as inventory telemetry for the curriculum. Depending on
            configuration it can block advance and/or trigger the
            emergency retreat brake.

        Returns
        -------
        events : dict
            {
                "buffer_evaluated": bool,    # decision made this step
                "advance":          bool,
                "retreat":          bool,
                "hold":             bool,
                "p_bar":            Optional[float],  # buffer average at decision
                "inv_ma":           float,
                "metric_clipped":   float,   # clip(final_pnl, -clip, +clip)
                "inv_advance_veto": bool,    # advance blocked by inventory stress
            }
        """
        events = dict(
            buffer_evaluated=False, advance=False, retreat=False, hold=False,
            # Opt-in extension events (only meaningful when the
            # corresponding mechanism is active; always False otherwise
            # so downstream consumers can read them unconditionally).
            advance_pending=False,  # signal fired but K(Î´) not yet reached
            p_bar=None, inv_ma=0.0, metric_clipped=0.0,
            t_H_at_decision=None, t_L_at_decision=None,
            inv_advance_veto=False, inv_retreat_trigger=False,
            # Diagnostic snapshot of the opt-in extension state at the
            # moment of the decision â useful for TB/stdout logging.
            consecutive_advance_signals=0,
            advances_needed=1,
            ladder_idx=self._ladder_idx if self._delta_ladder is not None else -1,
        )

        # Always track inventory telemetry (rolling, used for advance veto
        # and/or emergency retreat depending on the configured thresholds).
        self.inv_history.append(float(pct_at_inv_limit))
        events["inv_ma"] = (
            float(np.mean(self.inv_history)) if self.inv_history else 0.0
        )

        # Per-episode metric: clipped final PnL.  Clipping mirrors ADR's
        # natural bounding of success-count in [0, 50] and makes the
        # buffer mean robust to single catastrophic outliers.
        metric = float(np.clip(final_pnl, -self.pnl_clip, +self.pnl_clip))
        self.buffer.append(metric)
        events["metric_clipped"] = metric

        if len(self.buffer) < self.m:
            return events

        # Buffer is full â decide and clear (faithful to ADR Algorithm 1).
        p_bar = float(np.mean(self.buffer))
        self.buffer.clear()

        # Compute thresholds at the CURRENT Î´.  In linear-anchored mode
        # these will differ across decisions; in constant mode they are
        # fixed.
        t_H_now, t_L_now = self._current_thresholds()

        events["buffer_evaluated"] = True
        events["p_bar"] = p_bar
        events["t_H_at_decision"] = t_H_now
        events["t_L_at_decision"] = t_L_now
        self._stats["decisions"] += 1

        inv_ma = events["inv_ma"]
        inv_advance_veto = (
            self.inv_advance_veto_min is not None
            and inv_ma > self.inv_advance_veto_min
        )
        inv_bad = inv_ma > self.inv_retreat_min

        events["inv_advance_veto"] = bool(inv_advance_veto)
        events["inv_retreat_trigger"] = bool(inv_bad)

        if (p_bar >= t_H_now) and inv_advance_veto and not inv_bad:
            self._stats["inv_advance_vetoes"] += 1

        advance_signal = (
            (p_bar >= t_H_now)
            and not inv_advance_veto
            and not inv_bad
        )
        retreat_signal = (p_bar <= t_L_now) or inv_bad

        # ââ Resolve how many consecutive advance signals we currently
        # need to actually move Î´ up.  When the multi-confirmation
        # extension is disabled (the default), k_needed is always 1 and
        # this branch is a no-op that preserves the legacy behavior
        # (advance_signal â Î´ moves immediately).
        if self._advance_confirmations_fn is not None:
            k_needed = int(self._advance_confirmations_fn(self.delta))
            if k_needed < 1:
                k_needed = 1
        else:
            k_needed = 1
        events["advances_needed"] = k_needed

        # ââ Decision logic âââââââââââââââââââââââââââââââââââââââââââ
        #
        # The flow is:
        #   advance_signal (i.e. strong p_bar with no inventory veto /
        #       retreat trigger) increments the
        #       consecutive-advance counter.  If the counter reaches
        #       k_needed, Î´ actually moves up and the counter resets;
        #       otherwise the decision is flagged `advance_pending`.
        #   retreat_signal (or inventory emergency brake) moves Î´ down and forcibly
        #       resets the consecutive-advance counter to 0 (the
        #       retreat itself destroys the advance evidence streak).
        #   A hold either resets or preserves the counter depending on
        #       `advance_confirmation_reset_mode`:
        #         - "strict"       â reset (legacy strict-consecutive)
        #         - "retreat_only" â preserve partial progress
        #       Inventory advance-vetoes land in this HOLD branch.
        if advance_signal and not retreat_signal:
            self._consecutive_advance_signals += 1
            if self._consecutive_advance_signals >= k_needed:
                # Confirmation reached: perform the actual Î´ move.
                self._apply_delta_move(direction=+1)
                events["advance"] = True
                self._stats["advances"] += 1
                self._stats["decisions_since_last_advance"] = 0
                self._consecutive_advance_signals = 0
            else:
                # Signal received but not enough evidence to move yet.
                # This is a PARTIAL PROGRESS state, not a plateau: the
                # agent is actively demonstrating competence, the ADR
                # is just waiting for K(Î´) consecutive confirmations
                # before committing the Î´ move.  We therefore RESET
                # `decisions_since_last_advance` here so the runner's
                # plateau detector does not prematurely early-stop a
                # run that is legitimately accumulating confirmations
                # (which, at high Î´ with K_max=4, can take up to 4
                # consecutive buffer blocks = 200 eps per actual step).
                # Without this reset, a high-Î´ region with formula
                # confirmation mode would trip the plateau detector
                # while the confirmation mechanism is doing exactly
                # what it was designed to do.
                events["advance_pending"] = True
                self._stats["advance_pending"] += 1
                self._stats["decisions_since_last_advance"] = 0
        elif retreat_signal:
            self._apply_delta_move(direction=-1)
            events["retreat"] = True
            self._stats["retreats"] += 1
            self._stats["decisions_since_last_advance"] += 1
            self._consecutive_advance_signals = 0
        else:
            events["hold"] = True
            self._stats["holds"] += 1
            self._stats["decisions_since_last_advance"] += 1
            if self._advance_confirmation_reset_mode == "strict":
                self._consecutive_advance_signals = 0

        # Expose the post-decision counter state for logging.
        events["consecutive_advance_signals"] = (
            self._consecutive_advance_signals
        )
        events["ladder_idx"] = (
            self._ladder_idx if self._delta_ladder is not None else -1
        )

        # Track time spent at delta_max for the convergence detector.
        # Use a small tolerance because float arithmetic on delta_step
        # can leave residual ULPs.  In linear_mean_gap mode, don't count
        # decisions where the agent is below the convergence floor (we
        # don't want to declare "curriculum converged" while the buffer
        # mean is in the red â that's a retreat-blocked-by-hysteresis
        # state, not mastery).
        at_max = self.delta >= (self.delta_max - 1e-9)
        if self.threshold_mode == self.THRESHOLD_LINEAR_MEAN_GAP:
            at_max = at_max and (p_bar > self.convergence_floor)
        if at_max:
            self._stats["consecutive_at_max"] += 1
        else:
            self._stats["consecutive_at_max"] = 0

        return events

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  Introspection / logging helpers
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def buffer_running_mean(self) -> Optional[float]:
        """Mean of the current (partial) buffer, or None if empty.

        Useful for TensorBoard logging between decisions to visualise
        the gate signal trajectory.
        """
        if not self.buffer:
            return None
        return float(np.mean(self.buffer))

    def get_state(self) -> dict:
        """Current state, suitable for TensorBoard `add_scalar` loops.

        In linear-anchored mode `t_H_abs` / `t_L_abs` reflect the
        threshold AT THE CURRENT Î´ (they change as the curriculum
        moves), so the TensorBoard plot shows the schedule.
        """
        inv_ma = float(np.mean(self.inv_history)) if self.inv_history else 0.0
        running_mean = self.buffer_running_mean()
        t_H_now, t_L_now = self._current_thresholds()
        # Derive the K(Î´) that WOULD apply to the next advance signal at
        # the current Î´, so TensorBoard can plot "how strict is the
        # gate right now?" as the curriculum evolves.  When the
        # multi-confirmation extension is disabled, this is always 1.
        if self._advance_confirmations_fn is not None:
            _advances_needed = int(self._advance_confirmations_fn(self.delta))
        else:
            _advances_needed = 1
        return dict(
            delta=float(self.delta),
            p_lo=float(0.5 - self.delta),
            p_hi=float(0.5 + self.delta),
            buffer_size=int(len(self.buffer)),
            buffer_running_mean=float(running_mean) if running_mean is not None else 0.0,
            inv_ma=inv_ma,
            advances=int(self._stats["advances"]),
            retreats=int(self._stats["retreats"]),
            holds=int(self._stats["holds"]),
            inv_advance_vetoes=int(self._stats["inv_advance_vetoes"]),
            decisions=int(self._stats["decisions"]),
            episode_count=int(self._stats["episode_count"]),
            t_H_abs=float(t_H_now),
            t_L_abs=float(t_L_now),
            # Convergence diagnostics
            consecutive_at_max=int(self._stats["consecutive_at_max"]),
            decisions_since_last_advance=int(self._stats["decisions_since_last_advance"]),
            # Extension diagnostics (0 / -1 when the corresponding
            # extension is disabled, so TB plots are always defined)
            advance_pending=int(self._stats["advance_pending"]),
            consecutive_advance_signals=int(self._consecutive_advance_signals),
            advances_needed=int(_advances_needed),
            ladder_idx=(
                int(self._ladder_idx) if self._delta_ladder is not None else -1
            ),
            ladder_len=(
                int(len(self._delta_ladder))
                if self._delta_ladder is not None else -1
            ),
        )

    def __repr__(self) -> str:
        st = self.get_state()
        mode_short = {
            self.THRESHOLD_CONSTANT:        "const",
            self.THRESHOLD_LINEAR_PHASE_A:  "lin-peak",
            self.THRESHOLD_LINEAR_MEAN_GAP: "lin-mean",
        }.get(self.threshold_mode, self.threshold_mode)
        return (
            f"ADRLiteCurriculum(mode={mode_short}, "
            f"delta={st['delta']:.3f} in [{self.delta_min}, {self.delta_max}], "
            f"thr=[{st['t_L_abs']:+.3f}, {st['t_H_abs']:+.3f}], "
            f"buf={st['buffer_size']}/{self.m} mean={st['buffer_running_mean']:+.3f}, "
            f"adv={st['advances']}, ret={st['retreats']}, hold={st['holds']})"
        )
