#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_switching_stress_test.py  —  Regime-Switching MO Flow Stress Test
=========================================================================

Motivation
----------
A market maker trained under stationary order flow (fixed ``buy_mo_prob=0.5``)
may be fragile when deployed in markets where the proportion of aggressive
buy vs sell flow changes over time — for example, during metaorder execution
by institutional traders.  This script quantifies that fragility by
evaluating a pre-trained DQN checkpoint under **dynamic regime-switching**
MO flow and comparing it against the stationary baseline.

Regime-Switching Model
~~~~~~~~~~~~~~~~~~~~~~
Within each Monte Carlo simulation episode, ``buy_mo_prob`` is no longer
fixed at 0.5 but instead changes at random regime boundaries:

    - Regime duration:  L ~ Exp(1/τ), τ=60             (in MO events)
    - p_buy per regime: U[p_lo, p_hi]                    (default [0.2, 0.8])

The exponential duration model matches the Phase-B regime training setup.
The uniform draw for p_buy within each regime is symmetric around 0.5, so
**globally** the expected buy fraction is still 0.5 — but **locally**, the
agent faces persistent directional flow that can push inventory to extremes
and stress risk management.

Experimental Design
~~~~~~~~~~~~~~~~~~~
The script runs N_SIMS Monte Carlo simulations under two (optionally four)
conditions and compares terminal PnL distributions:

    1. DQN under p=0.5 fixed  (baseline / training distribution)
    2. DQN under regime-switching  (out-of-distribution stress test)
    3. [Optional] GLFT under p=0.5 fixed  (analytical benchmark)
    4. [Optional] GLFT under regime-switching

GLFT (Gueant-Lehalle-Fernandez-Tapia) is the classical continuous-time
optimal market-making policy.  It serves as a model-based benchmark:
since it explicitly models inventory risk, it may degrade less under
regime-switching than a model-free DQN that has only seen stationary flow.

Output
~~~~~~
    - Bar charts: E[PnL], std(PnL), and Sharpe ratio across conditions
    - Overlaid histograms of terminal PnL distributions
    - Saved to: ``regime_switching_stress_test.png``, ``regime_switching_pnl_dist.png``

Usage
-----
    python regime_switching_stress_test.py
"""

import os
import csv
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# === Paper figure style: larger labels/legends, no embedded titles ===
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from MM_LOB_SIM import simulate_LOB_with_MM
from LOB_SIM_SANTA_FE import simulate_LOB
from GLFT_policy_factory import glft_policy_factory
from MM_GLFT_naive_comparison import make_controller_from_checkpoint

from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)
from calibrate_trading_intensity import fit_volatility

from CONFIG_MM import (
    GLOBAL_SEED, lam, mu, delta, qrm_params, mean_size_LO, mean_size_MO,
    USE_QRM,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# All parameters below are grouped by purpose.  LOB parameters are imported
# from CONFIG_MM to ensure consistency with the training configuration.

# ── LOB parameters (from CONFIG_MM) ─────────────────────────────────────────
# These must match the values used during DQN training to ensure the
# stress test evaluates the agent in the same microstructure environment.
LAM = lam           # limit order arrival rate (per tick level per unit time)
MU = mu             # market order arrival rate (total, both sides)
DELTA = delta        # cancellation rate (per order per unit time)

NUMBER_TICK_LEVELS = 50   # price grid depth (ticks above and below mid)
N_PRIORITY_RANKS = 100    # queue priority slots per price level
HALF_TICK = 0.5            # half the tick size (used for spread calculations)
TICK_SIZE = 0.01           # monetary value of one tick in analytical PnL curves

N_STEPS = 5_000            # total simulation events per episode
N_STEPS_TO_EQUIL = 1_000   # warmup events before MM starts trading

# ── MM constraints ──────────────────────────────────────────────────────────
INV_LIMIT = 8   # maximum absolute inventory the MM can hold

# ── Monte Carlo ─────────────────────────────────────────────────────────────
N_SIMS = 100          # number of independent simulation episodes per condition
SEED_BASE = GLOBAL_SEED

# ── Calibration ─────────────────────────────────────────────────────────────
# Parameters for the GLFT calibration step (execution intensity and volatility).
# Only used if RUN_GLFT=True.
SEED_CALIB = 1234
TIME_INTERVAL_CALIB = 0.5   # time bucket width for censored waiting time estimator

# ── DQN checkpoint to stress test ──────────────────────────────────────────
DQN_CHECKPOINT = "checkpoints/deep_mm_mtm_pure_invp0.0010_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt"
DQN_INV_PENALTY = 1e-3     # inventory penalty phi of the checkpoint (invp0.0010)
DQN_TIME_THROTTLE = 1.0    # minimum seconds between MM quote updates

# ── GLFT benchmark ─────────────────────────────────────────────────────────
GLFT_GAMMA = 1e-6          # GLFT risk aversion parameter
GLFT_TIME_THROTTLE = 1.0
USE_MDP_GLFT = True        # use discrete MDP grid for GLFT (vs continuous)
RUN_GLFT = True           # set True to include GLFT in the comparison

# ── GLFT/AS-style misspecified overlay (Expected PnL vs P(buy MO)) ──────────
# When enabled, this script enters an asymmetry-sweep workflow and plots:
#   - DQN empirical curve (blue)
#   - GLFT misspecified analytical curve (red)
# The analytical curve assumes the MM is calibrated under symmetric flow and
# keeps the GLFT inventory skew, while the environment has asymmetric MO flow
# (p != 0.5), which induces inventory-drift losses.
#
# Priority rule:
#   - If PLOT_ONLY_AS_MISS=True  -> plot ONLY the analytical red curve.
#   - Else if PLOT_AS_MISS=True  -> plot DQN (blue) + analytical (red).
#   - Else -> run the original regime-switching stress test workflow.
PLOT_AS_MISS = True
PLOT_ONLY_AS_MISS = False
# When False, skip the analytical misspecified curve entirely (and its
# r_fill_factor / v0 calibrations): the figure shows only the empirical
# DQN and GLFT sweeps.  A/kappa/sigma are still calibrated because the
# GLFT empirical controller needs them to quote.
PLOT_ANALYTICAL = True
# When True, run GLFT empirical sweep over BUY_MO_PROBS_AS and plot the
# Expected PnL vs P(buy MO) curve for the GLFT controller only.
# The GLFT uses symmetric calibration (λ⁺=λ⁻) but skews quotes by
# inventory — so it's misspecified in p but correct in q.
PLOT_ONLY_GLFT = False

# Gamma requested for the GLFT misspecified theoretical curve.
AS_MISS_GAMMA = 1e-6
# Inventory running-penalty coefficient used in the misspecified formula.
# This mirrors the reward coefficient typically used in training runs.
# Use signature-plot approximation for sigma (requested).
AS_MISS_SIGMA_OVERRIDE = 0.3
# When USE_V0 = True, add a MtM loss term based on the endogenous price
# drift caused by asymmetric MO flow in the simulator.  The paper assumes
# dS = σdW (martingale, E[MtM]=0), but in the Santa Fe simulator, excess
# buy/sell MOs push the mid-price, creating a drift proportional to (2p-1).
# This drift is NOT in the paper — it's an extension to match the simulator.
#   USE_V0 = False → paper-faithful (PnL = spread capture only, always ≥ 0)
#   USE_V0 = True  → simulator-faithful (PnL = spread capture - MtM loss)
AS_MISS_USE_V0 = True
# p-grid used for the blue/red expected-PnL curves.
BUY_MO_PROBS_AS = [0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
# Calibration steps for the misspecified analytical curve — use a long simulation (500k) for
# precise A/kappa/sigma estimation.  This is independent of N_STEPS which
# controls the episode length for the DQN sweep.
AS_MISS_CALIB_STEPS = 500_000
AS_MISS_CALIB_EQUIL = 10_000
# Pre-calibrated A and kappa from a 500k-step run.  When both are not None,
# the expensive calibration simulation is skipped and these values are used
# directly.  Set either to None to force re-calibration.
# Canonical pair unified across all paper experiments, reproduced by
# GLFT_studies.run_calibration() (dT=0.5, 500k steps, n_equil=1000, seed=1234).
AS_MISS_A_CACHED     = 0.150707    # execution intensity intercept
AS_MISS_KAPPA_CACHED = 2.33534     # execution intensity decay rate
# Pre-calibrated v0 (mid-price drift rate per unit of flow asymmetry).
# Only used when AS_MISS_USE_V0 = True.
# v0 is defined by:  E[dS/dt] = (2p - 1) * v0
# Set to None to force re-measurement.
# Canonical value measured by generate_as_miss_bundle.py with the canonical
# (A, kappa) — keep in sync with the appendix ("calibrated once offline").
AS_MISS_V0_CACHED    = 0.047667        # set after first calibration run
# Effective fill-rate scaling factor.  The theoretical fill rate
# R = A * exp(-k * delta_bar) assumes the MM captures ALL fills at that
# price level with continuous quoting.  In practice, the MM:
#   (a) shares the queue with other limit orders (queue priority)
#   (b) only re-quotes every min_time_interval=1s (throttle)
# Both reduce the effective fill rate.  R_FILL_FACTOR accounts for this:
#   R_effective = R * R_FILL_FACTOR
# Calibrated by measuring the MM's actual fills per second at p=0.5
# and dividing by the theoretical R.  Set to None to auto-calibrate.
# Canonical value measured by generate_as_miss_bundle.py with the canonical
# (A, kappa) — this is the paper's eta (queue-competition factor).
AS_MISS_R_FILL_FACTOR = 0.819672       # will be measured on first run

# ── Regime-switching parameters ─────────────────────────────────────────────
# These control the non-stationary MO flow environment.
# REGIME_DISTRIBUTION controls the regime-duration law used in evaluations.
# The Phase-B training setup uses exponential durations with mean tau=60 MOs.
# REGIME_ALPHA / REGIME_L_MIN are kept only for the optional Pareto fallback.
# REGIME_P_LO/HI: bounds for the uniform p_buy draw per regime.
#   Symmetric around 0.5 ensures the global expected flow is balanced.
REGIME_DISTRIBUTION = "exponential"  # "exponential" or "pareto"
REGIME_EXP_TAU = 60.0       # mean regime duration in MO events
REGIME_ALPHA = 1.5        # Pareto shape parameter (power law exponent)
REGIME_L_MIN = 10         # minimum regime length (MO events)
REGIME_P_LO = 0.2         # min p_buy per regime
REGIME_P_HI = 0.8         # max p_buy per regime


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════
# This section implements the "environment-driven" regime schedule used for
# stress testing.  Unlike the adversarial schedule (in adversary_agent.py),
# which lets a learned agent choose p_buy, here p_buy is drawn randomly —
# the goal is pure evaluation, not co-training.

def make_regime_schedule(
    seed: int,
    n_mo_events: int,
    L_min: int = REGIME_L_MIN,
    alpha: float = REGIME_ALPHA,
    p_lo: float = REGIME_P_LO,
    p_hi: float = REGIME_P_HI,
    distribution: str = REGIME_DISTRIBUTION,
    exp_rate: float = 1.0 / REGIME_EXP_TAU,
) -> Callable[[int], float]:
    """
    Build a callable ``step_idx -> buy_mo_prob`` with regime-switching.

    This callable is designed to plug directly into the LOB engine's
    ``buy_mo_prob`` parameter (which accepts either a float or a
    ``Callable[[int], float]``).  Within each regime of random duration,
    the buy MO probability is fixed at a value drawn from ``U[p_lo, p_hi]``.

    The schedule is fully pre-generated at construction time (no online
    randomness), which ensures reproducibility given the same seed and
    allows efficient O(log n) lookup per call via ``np.searchsorted``.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility of the regime schedule.
    n_mo_events : int
        Expected number of MO events per episode.  Used to pre-generate
        enough regime boundaries to cover the full episode (with
        a generous +1000 buffer for stochastic variation in MO counts).
    L_min : int
        Minimum regime duration (Pareto scale parameter), in MO events.
        With ``alpha=1.5``, the expected regime length is ``3 * L_min``.
        Ignored when ``distribution="exponential"``.
    alpha : float
        Pareto shape parameter.  Controls the tail heaviness of regime
        durations.  ``alpha=1.5`` produces heavy tails consistent with
        empirical metaorder durations (Lillo, Mike & Farmer 2005).
        Larger alpha -> lighter tails, more uniform regime lengths.
        Ignored when ``distribution="exponential"``.
    p_lo, p_hi : float
        Bounds for the uniform draw of p_buy within each regime.
        Default [0.2, 0.8] is symmetric around 0.5 to preserve the
        global mean flow direction while creating local asymmetry.
    distribution : str
        ``"exponential"`` (default) or ``"pareto"``.
        When ``"exponential"``, regime durations are drawn from
        ``Exp(exp_rate)`` (mean = 1/exp_rate MO events).
    exp_rate : float
        Rate parameter λ for exponential regime durations (λ = 1/τ).
        Required when ``distribution="exponential"``.
        The optimal EWMA alpha (time constant) is ``1 - exp(-λ)``.

    Returns
    -------
    schedule : Callable[[int], float]
        Maps MO event index -> buy_mo_prob for that event.
        Also exposes ``.boundaries`` (list of regime boundary indices)
        and ``.p_values`` (list of p_buy per regime) for diagnostics.
    """

    if distribution not in ("pareto", "exponential"):
        raise ValueError(f"distribution must be 'pareto' or 'exponential', got '{distribution}'")
    if distribution == "exponential" and exp_rate <= 0.0:
        raise ValueError(f"exp_rate must be > 0 for exponential distribution, got {exp_rate}")

    rng = np.random.default_rng(seed)

    # Pre-generate regime boundaries and p_buy values for the expected
    # horizon. If a caller later queries beyond that horizon, we extend the
    # schedule on demand instead of freezing at the last p_buy forever.
    boundaries: List[int] = [0]
    p_values: List[float] = []
    boundaries_arr = np.array(boundaries)
    p_values_arr = np.array(p_values)

    def _draw_length() -> int:
        if distribution == "pareto":
            # Pareto draw via inverse CDF: L = ceil(L_min / U^(1/alpha))
            return int(np.ceil(L_min / rng.uniform() ** (1.0 / alpha)))
        # Exponential draw: L = ceil(Exp(1/rate))
        return max(1, int(np.ceil(rng.exponential(1.0 / exp_rate))))

    def _append_regime() -> None:
        nonlocal boundaries_arr, p_values_arr
        L = _draw_length()
        boundaries.append(boundaries[-1] + L)
        p_values.append(float(rng.uniform(p_lo, p_hi)))
        boundaries_arr = np.array(boundaries)
        p_values_arr = np.array(p_values)

    while boundaries[-1] < n_mo_events + 1000:
        _append_regime()

    boundaries_arr = np.array(boundaries)
    p_values_arr = np.array(p_values)

    def schedule(step_idx: int) -> float:
        """O(log n) lookup: extend if needed, then return the current regime p_buy."""
        while step_idx >= boundaries[-1]:
            _append_regime()
        regime_idx = int(np.searchsorted(boundaries_arr, step_idx, side="right")) - 1
        regime_idx = min(regime_idx, len(p_values_arr) - 1)
        return p_values_arr[regime_idx]

    # Attach metadata for diagnostics and plotting
    schedule.boundaries = boundaries  # type: ignore[attr-defined]
    schedule.p_values = p_values      # type: ignore[attr-defined]

    return schedule


def make_default_regime_schedule(seed: int, n_mo_events: int) -> Callable[[int], float]:
    """Build the regime schedule used by this stress test configuration."""
    exp_rate = 1.0 / REGIME_EXP_TAU if REGIME_DISTRIBUTION == "exponential" else 0.0
    return make_regime_schedule(
        seed=seed,
        n_mo_events=n_mo_events,
        L_min=REGIME_L_MIN,
        alpha=REGIME_ALPHA,
        p_lo=REGIME_P_LO,
        p_hi=REGIME_P_HI,
        distribution=REGIME_DISTRIBUTION,
        exp_rate=exp_rate,
    )


def _regime_description() -> str:
    """Human-readable description of the active regime-duration model."""
    if REGIME_DISTRIBUTION == "exponential":
        return f"exponential, tau={REGIME_EXP_TAU:g} MOs"
    return f"Pareto, alpha={REGIME_ALPHA:g}, L_min={REGIME_L_MIN:g} MOs"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    """Set seeds for all RNGs (Python, NumPy, PyTorch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_seed(run_id: int) -> int:
    """Derive a per-simulation seed from the global base seed and run index."""
    return int(SEED_BASE + run_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION (run once at buy_mo_prob=0.5)
# ═══════════════════════════════════════════════════════════════════════════════

def run_calibration(plot: bool = False,
                    n_steps_override: int = None,
                    n_equil_override: int = None) -> Tuple[float, float, float]:
    """
    Calibrate GLFT model parameters (A, kappa, sigma) from a bare LOB simulation.

    The GLFT (Gueant-Lehalle-Fernandez-Tapia) optimal market-making solution
    requires three microstructure parameters:
        - A : execution intensity intercept (fills per unit time at delta=0)
        - kappa : execution intensity decay rate (how fast fill probability
          drops with quote distance from mid-price)
        - sigma : mid-price volatility

    These are estimated from a simulation run *without* any MM present
    (bare LOB dynamics), using:
        1. Censored waiting time estimation for the execution intensity
           Lambda(delta) = A * exp(-kappa * delta)
        2. Signature plot / realised volatility for sigma

    Parameters
    ----------
    plot : bool
        If True, generate diagnostic calibration plots.
    n_steps_override : int, optional
        Override the number of simulation steps (default: N_STEPS=5000).
        Use a larger value (e.g. 500_000) for more precise calibration
        of A and kappa when the estimates will drive analytical formulas.
    n_equil_override : int, optional
        Override the equilibration steps (default: N_STEPS_TO_EQUIL=1000).
    """
    _n_steps = n_steps_override if n_steps_override is not None else N_STEPS
    _n_equil = n_equil_override if n_equil_override is not None else N_STEPS_TO_EQUIL

    print("=" * 60)
    print(f"  CALIBRATION (n_steps={_n_steps:,}, n_equil={_n_equil:,})")
    print("=" * 60)

    res_wt = fit_execution_intensity_censored_waiting_times(
        aggregation_mode="time",
        time_interval=TIME_INTERVAL_CALIB,
        max_levels=5,
        half_tick=HALF_TICK,
        side_mode="buy",
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=_n_steps,
        n_steps_to_equilibrium=_n_equil,
        split_sweeps=False,
        random_seed=SEED_CALIB,
        plot=plot,
    )

    A, KAPPA, _ = fit_A_kappa_loglinear(
        res_wt["delta_grid"],
        res_wt["lambda_hat"],
        weights=res_wt["denom_sum_tau"],
    )

    res_vol = fit_volatility(
        queue_aware=False,
        aggregation_mode="time",
        time_interval=TIME_INTERVAL_CALIB,
        use_tob_buckets=False,
        n_tob_moves=10,
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=_n_steps,
        n_steps_to_equilibrium=_n_equil,
        half_tick=HALF_TICK,
        max_depth_levels=5,
        max_ranks=5,
        rank_to_fit=0,
        strict=True,
        plot=plot,
        random_seed=SEED_CALIB + 1,
    )

    SIGMA = float(res_vol["volatility"])
    # NOTE: if calibrated sigma is unreliable, override here:
    SIGMA = 0.3  # from signature plot

    print(f"  A     = {A:.6g}")
    print(f"  kappa = {KAPPA:.6g}")
    print(f"  sigma = {SIGMA:.6g}")
    print()

    return float(A), float(KAPPA), SIGMA


# ═══════════════════════════════════════════════════════════════════════════════
# GLFT MISSPECIFIED ANALYTICAL CURVE (Expected PnL vs P(buy MO))
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_episode_time_horizon(seed: int = SEED_CALIB + 2) -> float:
    """
    Estimate the simulation-time horizon ``T`` of one episode.

    We measure T directly from a bare LOB simulation message tape, using the
    same episode length and warmup settings as the stress tests:

        T = Time_last - Time_first

    This avoids mixing "number of events" with "simulator time", which would
    distort analytical formulas that are expressed in continuous-time intensities.
    """
    if USE_QRM:
        print("[GLFT MISS] NOTE: T estimation uses Santa-Fe simulator tape as proxy under USE_QRM.")

    msg_df, _, _ = simulate_LOB(
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        number_levels_to_store=20,
        p0=100,
        mean_size_LO=mean_size_LO,
        mean_size_MO=mean_size_MO,
        iterations=N_STEPS,
        iterations_to_equilibrium=N_STEPS_TO_EQUIL,
        path_save_files=None,
        label_simulation=None,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        random_seed=int(seed),
        buy_mo_prob=0.5,
    )

    if "Time" not in msg_df.columns or len(msg_df) < 2:
        print("[GLFT MISS] WARNING: could not infer Time from tape; falling back to T=1.0")
        return 1.0

    t0 = float(msg_df["Time"].iloc[0])
    t1 = float(msg_df["Time"].iloc[-1])
    T = max(t1 - t0, 1e-9)
    print(f"[GLFT MISS] Episode time horizon estimated: T={T:.6f}")
    return T


def calibrate_v0(
    p_grid: List[float] = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
    n_episodes: int = 30,
    seed_base: int = SEED_CALIB + 100,
) -> float:
    """
    Measure the mid-price drift rate v0 from bare LOB simulations.

    The mid-price drifts when MO flow is asymmetric:

        E[dS/dt] = (2p - 1) * v0

    We estimate v0 by running bare LOB simulations (no MM) at several
    fixed p values, measuring the total mid-price drift per episode,
    and regressing against (2p - 1).

    Parameters
    ----------
    p_grid : list of float
        P(buy MO) values to simulate.  Should be symmetric around 0.5
        and span the range of interest.
    n_episodes : int
        Number of episodes per p value (averaged for noise reduction).
    seed_base : int
        Base seed for reproducibility.

    Returns
    -------
    v0 : float
        Drift rate in price-ticks per unit-time per unit-asymmetry.
        Defined such that E[ΔS] = (2p-1) * v0 * T.
    """
    print(f"[GLFT MISS] Calibrating v0 from bare LOB simulations...")
    print(f"  p_grid    = {p_grid}")
    print(f"  n_episodes = {n_episodes} per p")

    x_vals = []  # (2p - 1)
    y_vals = []  # ΔS / T

    for p in p_grid:
        drift_rates = []
        for ep in range(n_episodes):
            seed = seed_base + int(p * 1000) + ep
            msg_df, _, _ = simulate_LOB(
                lam=LAM, mu=MU, delta=DELTA,
                number_tick_levels=NUMBER_TICK_LEVELS,
                n_priority_ranks=N_PRIORITY_RANKS,
                number_levels_to_store=20,
                p0=100,
                mean_size_LO=mean_size_LO,
                mean_size_MO=mean_size_MO,
                iterations=N_STEPS,
                iterations_to_equilibrium=N_STEPS_TO_EQUIL,
                path_save_files=None,
                label_simulation=None,
                beta_exp_weighted_return=0.0,
                intensity_exp_weighted_return=0.0,
                random_seed=seed,
                buy_mo_prob=float(p),
            )
            if "MidPrice" in msg_df.columns and "Time" in msg_df.columns and len(msg_df) > 1:
                dS = float(msg_df["MidPrice"].iloc[-1] - msg_df["MidPrice"].iloc[0])
                dT = float(msg_df["Time"].iloc[-1] - msg_df["Time"].iloc[0])
                if dT > 0:
                    drift_rates.append(dS / dT)

        if drift_rates:
            mean_drift = float(np.mean(drift_rates))
            asym = 2.0 * p - 1.0
            x_vals.append(asym)
            y_vals.append(mean_drift)
            print(f"  p={p:.2f}  (2p-1)={asym:+.2f}  "
                  f"mean ΔS/T={mean_drift:+.6f}  (n={len(drift_rates)})")

    # Linear regression through origin: y = v0 * x
    # v0 = Σ(x*y) / Σ(x²)
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    v0 = float(np.sum(x_arr * y_arr) / np.sum(x_arr ** 2))

    print(f"\n  [GLFT MISS] Calibrated v0 = {v0:.6f}  "
          f"(ticks/time per unit asymmetry)")
    return v0


def calibrate_r_fill_factor(
    A: float,
    kappa: float,
    sigma: float,
    delta_bar: float,
    n_episodes: int = 20,
    seed_base: int = SEED_CALIB + 200,
) -> float:
    """
    Measure the effective fill-rate scaling factor R_FILL_FACTOR using the
    GLFT controller (not the DQN).

    The theoretical total fill rate at q=0 under the simulator-consistent
    normalization used in the AS-miss derivation is

        R = 2 * A * exp(-k * delta_bar),

    where A is the one-sided intensity calibrated from unilateral
    censored-waiting-times probes. In practice, the MM shares the queue and
    re-quotes at discrete intervals.

    We run the GLFT controller at p=0.5, measure its actual fill rate,
    and divide by the theoretical R.  Using GLFT (not DQN) is correct
    because the analytical curve models a GLFT-style MM, not a DQN.

    Parameters
    ----------
    A, kappa, sigma : float
        Calibrated execution intensity and volatility parameters.
    delta_bar : float
        Theoretical quote distance from mid.
    n_episodes : int
        Number of episodes to average over.
    seed_base : int
        Base seed for reproducibility.

    Returns
    -------
    r_fill_factor : float
        Scaling factor in (0, 1].
    """
    # A is calibrated from ONE side (buy-side probe), while R_measured counts
    # fills on BOTH sides for the GLFT MM at p=0.5. Therefore the simulator-
    # consistent theoretical benchmark is the two-sided rate:
    #   R_theoretical = 2 * A * exp(-k * delta_bar).
    R_theoretical = 2.0 * float(A * np.exp(-kappa * delta_bar))

    print(f"[GLFT MISS] Calibrating R_FILL_FACTOR from {n_episodes} GLFT episodes at p=0.5...")
    print(f"  R_theoretical = {R_theoretical:.6f}")
    print(f"  GLFT_GAMMA    = {AS_MISS_GAMMA:.1e}")

    fill_rates = []
    for ep in range(n_episodes):
        seed = seed_base + ep
        set_seeds(seed)

        # Fresh GLFT policy per episode (same as run_glft_sims)
        policy = glft_policy_factory(
            gamma=float(AS_MISS_GAMMA),
            kappa=kappa,
            A=A,
            sigma=sigma,
            inv_limit=INV_LIMIT,
            round_to_int=True,
            delta=1.0,
            epsilon=None,
            use_tob_update=False,
            n_tob_moves=1,
            use_event_update=False,
            n_events=1000,
            use_time_update=True,
            min_time_interval=GLFT_TIME_THROTTLE,
            use_mdp=USE_MDP_GLFT,
        )

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=mean_size_LO,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            mm_policy=policy,
            random_seed=seed,
            buy_mo_prob=0.5,
            **({"qrm_params": qrm_params} if USE_QRM else {}),
        )

        # Count total fills from the MM trade log.
        if "MM_NumBidFills" in mm_df.columns and "MM_NumAskFills" in mm_df.columns:
            total_fills = int(mm_df["MM_NumBidFills"].iloc[-1] + mm_df["MM_NumAskFills"].iloc[-1])
        else:
            inv = mm_df["MM_Inventory"] if "MM_Inventory" in mm_df.columns else None
            if inv is not None:
                total_fills = int((inv.diff().abs() > 0).sum())
            else:
                total_fills = 0

        # Episode time
        if "Time" in mm_df.columns and len(mm_df) > 1:
            T_ep = float(mm_df["Time"].iloc[-1] - mm_df["Time"].iloc[0])
        else:
            T_ep = 1.0

        if T_ep > 0:
            fill_rates.append(total_fills / T_ep)

    if not fill_rates:
        print("  [WARNING] No fill data — using fallback R_FILL_FACTOR=0.05")
        return 0.05

    R_measured = float(np.mean(fill_rates))
    r_fill_factor = R_measured / R_theoretical if R_theoretical > 0 else 0.05

    print(f"  R_measured     = {R_measured:.6f} fills/s (mean over {len(fill_rates)} eps)")
    print(f"  R_FILL_FACTOR  = {r_fill_factor:.6f}")
    return float(r_fill_factor)


def compute_as_miss_expected_pnl_curve(
    probs: List[float],
    A: float,
    kappa: float,
    sigma: float,
    gamma: float,
    T: float,
    inv_limit: int = 8,
    r_fill_factor: float = 1.0,
    tick_size: float = 0.01,
    use_v0: bool = False,
    v0: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute analytical E[PnL] for a misspecified GLFT market maker,
    faithful to the paper (Guéant-Lehalle-Fernandez-Tapia 2012).

    The MM is misspecified **only in p** (assumes symmetric flow λ⁺ = λ⁻)
    but **correctly skews quotes** based on current inventory q, using
    the asymptotic GLFT solution (Section 4, Gaussian approximation).

    Faithful to the paper means:
        - Price is a martingale: dS = σdW (no drift, Section 2)
        - S is exogenous (independent of fills)
        - Therefore E[MtM] = E[q_T · (S_T - S_0)] = E[q_T] · E[S_T - S_0] = 0
        - PnL = spread capture ONLY (no MtM term)
        - The inverted-U shape comes from loss of fills at the bounds:
          when p ≠ 0.5, inventory drifts toward ±Q where the MM can only
          post one side, reducing total spread capture.

    Model:
        1) GLFT asymptotic quotes with inventory skew (p.11):

               δ_b(q) = (1/γ)·ln(1 + γ/k) + (2q+1)/2 · c
               δ_a(q) = (1/γ)·ln(1 + γ/k) + (1-2q)/2 · c

           where

               c = sqrt[(σ²γ / (2kA)) · (1 + γ/k)^(1 + k/γ)]

        2) Fill rates under true asymmetry + inventory skew:

               r_a(q) = 2p · A · exp(-k · δ_a(q))
               r_b(q) = 2(1-p) · A · exp(-k · δ_b(q))

           At bounds: q = +Q → r_b = 0 (no bid), q = -Q → r_a = 0 (no ask).

        3) Inventory: Birth-Death chain on {-Q, ..., Q} with stationary
           distribution π(q).

        4) Expected PnL (spread capture only, faithful to paper):

               E[PnL] = T · Σ_q π(q) · [r_a(q)·δ_a(q) + r_b(q)·δ_b(q)]

    Parameters
    ----------
    probs : list of float
        P(buy MO) values at which to evaluate the curve.
    A, kappa : float
        Execution intensity fit λ(δ) = A · exp(-k·δ) from a unilateral
        probe calibration. The total MO clock in the simulator is 2A and is
        split between buy and sell flow by p.
    sigma : float
        Mid-price volatility (used for skew intensity c).
    gamma : float
        CARA risk aversion coefficient (controls skew strength).
    T : float
        Episode time horizon (PnL accumulation window).
    inv_limit : int
        Maximum absolute inventory Q (hard bound, as in paper Section 2).
    r_fill_factor : float
        Scales fill rates for queue sharing / throttle effects.
    tick_size : float
        Converts tick-unit PnL to dollar-unit PnL.

    Returns
    -------
    expected : np.ndarray
        E[PnL] for each p in probs (in dollar units).
    meta : dict
        Calibration parameters and diagnostics.
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    if A <= 0.0:
        raise ValueError(f"A must be > 0, got {A}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")

    Q = int(inv_limit)
    N_states = 2 * Q + 1  # q ∈ {-Q, ..., Q}

    # ── GLFT asymptotic base spread (stationary, T→∞, Section 4) ─────
    # δ_base = (1/γ)·ln(1 + γ/k), with stable evaluation for γ → 0.
    if gamma > 1e-12:
        delta_base = np.log1p(gamma / kappa) / gamma
    else:
        delta_base = 1.0 / kappa

    # ── Skew intensity c (Gaussian approximation, p.11) ──────────────
    # c = sqrt[(σ²γ / (2kA)) · (1 + γ/k)^(1 + k/γ)]
    #
    # Numerical stability: evaluate the power term in log-space.
    # Because the whole expression is under the square root, the exponent is
    # 0.5 * (1 + k/γ) in the multiplicative representation.
    if gamma > 1e-12:
        _log_base = 0.5 * (1.0 + kappa / gamma) * np.log1p(gamma / kappa)
        _power_term = np.exp(min(_log_base, 500.0))  # prevent overflow
        c_skew = np.sqrt(sigma**2 * gamma / (2.0 * kappa * A)) * _power_term
    else:
        c_skew = 0.0
    c_skew = float(c_skew)
    delta_bar = float(delta_base + 0.5 * c_skew)

    # ── Quotes for each inventory level q ∈ {-Q, ..., Q} ────────────
    q_values = np.arange(-Q, Q + 1, dtype=float)

    # δ_b(q) = delta_base + (2q + 1)/2 · c   (bid distance from mid)
    # δ_a(q) = delta_base + (1 - 2q)/2 · c   (ask distance from mid)
    delta_b = delta_base + (2.0 * q_values + 1.0) / 2.0 * c_skew
    delta_a = delta_base + (1.0 - 2.0 * q_values) / 2.0 * c_skew

    # Floor at zero (MM never crosses the spread)
    delta_b = np.maximum(delta_b, 0.0)
    delta_a = np.maximum(delta_a, 0.0)

    expected_pnls = []

    for p in probs:
        p = float(p)

        # ── Fill rates under TRUE asymmetry + inventory skew ─────────
        r_a = 2.0 * p * A * np.exp(-kappa * delta_a) * r_fill_factor
        r_b = 2.0 * (1.0 - p) * A * np.exp(-kappa * delta_b) * r_fill_factor

        # Boundary constraints (paper Section 2, p.5):
        # q = +Q → MM never sets a bid quote → r_b = 0
        # q = -Q → MM never sets an ask quote → r_a = 0
        r_a[0] = 0.0    # index 0 = q = -Q
        r_b[-1] = 0.0   # index 2Q = q = +Q

        # ── Stationary distribution π(q) (Birth-Death chain) ─────────
        idx_0 = Q  # index of q=0

        log_pi = np.zeros(N_states)

        # q > 0: π(q) = π(q-1) · r_b(q-1) / r_a(q)
        for i in range(idx_0 + 1, N_states):
            _rb = r_b[i - 1]
            _ra = r_a[i]
            if _ra > 1e-30 and _rb > 1e-30:
                log_pi[i] = log_pi[i - 1] + np.log(_rb / _ra)
            else:
                log_pi[i] = -500.0

        # q < 0: π(q) = π(q+1) · r_a(q+1) / r_b(q)
        for i in range(idx_0 - 1, -1, -1):
            _ra = r_a[i + 1]
            _rb = r_b[i]
            if _rb > 1e-30 and _ra > 1e-30:
                log_pi[i] = log_pi[i + 1] + np.log(_ra / _rb)
            else:
                log_pi[i] = -500.0

        # Normalise (log-sum-exp)
        log_pi -= np.max(log_pi)
        pi = np.exp(log_pi)
        pi /= pi.sum()

        # ── PnL = spread capture only (paper: dS = σdW, no drift) ───
        # In the paper's model, S is an exogenous martingale independent
        # of the fill processes.  Therefore:
        #   E[q_T · (S_T - S_0)] = E[q_T] · E[S_T - S_0] = 0
        # The expected MtM is zero.  All PnL comes from spread capture.
        #
        # The inverted-U shape arises because when p ≠ 0.5, π(q) shifts
        # toward the bounds where the MM can only post one side,
        # reducing the total fill rate and spread revenue.
        spread_rate = np.sum(pi * (r_a * delta_a + r_b * delta_b))
        spread_capture = spread_rate * T

        # ── Optional MtM loss (simulator extension, not in paper) ────
        # In the paper: dS = σdW → E[MtM] = 0 (martingale).
        # In the simulator: asymmetric MOs cause endogenous price drift
        #   E[dS/dt] = (2p-1) · v0
        # The MM holds E[q] ≠ 0 against this drift → MtM loss.
        #   MtM_loss = -v0 · (2p-1) · E[q] · T   (positive = loss)
        if use_v0 and abs(v0) > 1e-15:
            E_q = float(np.sum(pi * q_values))
            mtm_loss = -v0 * (2.0 * p - 1.0) * E_q * T
        else:
            mtm_loss = 0.0

        expected_pnls.append((spread_capture - mtm_loss) * tick_size)

    # Diagnostics
    E_q_at_half = 0.0  # E[q] at p=0.5 (should be ~0 by symmetry)
    meta = {
        "A": float(A),
        "kappa": float(kappa),
        "sigma": float(sigma),
        "gamma": float(gamma),
        "T": float(T),
        "delta_base": float(delta_base),
        "delta_bar": float(delta_bar),
        "c_skew": float(c_skew),
        "R_at_q0": float(2.0 * A * np.exp(-kappa * delta_bar) * r_fill_factor),
        "inv_limit": int(inv_limit),
        "r_fill_factor": float(r_fill_factor),
        "tick_size": float(tick_size),
        "use_v0": bool(use_v0),
        "v0": float(v0) if use_v0 else 0.0,
    }
    return np.array(expected_pnls), meta


def save_expected_pnl_overlay_table(
    probs: List[float],
    analytical_means: np.ndarray,
    analytical_meta: Dict[str, float],
    dqn_means: Optional[List[float]] = None,
    dqn_stds: Optional[List[float]] = None,
    glft_means: Optional[List[float]] = None,
    glft_stds: Optional[List[float]] = None,
    n_sims: int = N_SIMS,
    only_analytical: bool = False,
) -> str:
    """Save the data behind the expected-PnL overlay plot."""
    out = (
        "expected_pnl_vs_buy_MO_misspecification_only.csv"
        if only_analytical
        else "expected_pnl_vs_buy_MO_misspecification_overlay.csv"
    )
    dqn_means = list(dqn_means) if dqn_means is not None else None
    dqn_stds = list(dqn_stds) if dqn_stds is not None else None
    glft_means = list(glft_means) if glft_means is not None else None
    glft_stds = list(glft_stds) if glft_stds is not None else None

    fieldnames = [
        "p_buy_mo",
        "glft_misspecified_analytical_mean",
        "dqn_empirical_mean",
        "dqn_empirical_std",
        "dqn_empirical_se_mean",
        "glft_empirical_mean",
        "glft_empirical_std",
        "glft_empirical_se_mean",
        "analytical_use_v0",
        "analytical_v0",
        "analytical_eta",
        "analytical_tick_size",
        "analytical_T",
        "analytical_A",
        "analytical_kappa",
        "analytical_sigma",
        "analytical_gamma",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(probs):
            dqn_std = dqn_stds[i] if dqn_stds is not None else ""
            glft_std = glft_stds[i] if glft_stds is not None else ""
            writer.writerow({
                "p_buy_mo": float(p),
                "glft_misspecified_analytical_mean": float(analytical_means[i]),
                "dqn_empirical_mean": dqn_means[i] if dqn_means is not None else "",
                "dqn_empirical_std": dqn_std,
                "dqn_empirical_se_mean": (dqn_std / np.sqrt(max(n_sims, 1))) if dqn_stds is not None else "",
                "glft_empirical_mean": glft_means[i] if glft_means is not None else "",
                "glft_empirical_std": glft_std,
                "glft_empirical_se_mean": (glft_std / np.sqrt(max(n_sims, 1))) if glft_stds is not None else "",
                "analytical_use_v0": bool(analytical_meta.get("use_v0", False)),
                "analytical_v0": analytical_meta.get("v0", 0.0),
                "analytical_eta": analytical_meta.get("r_fill_factor", 1.0),
                "analytical_tick_size": analytical_meta.get("tick_size", 0.01),
                "analytical_T": analytical_meta.get("T", ""),
                "analytical_A": analytical_meta.get("A", ""),
                "analytical_kappa": analytical_meta.get("kappa", ""),
                "analytical_sigma": analytical_meta.get("sigma", ""),
                "analytical_gamma": analytical_meta.get("gamma", ""),
            })
    return out


def run_dqn_asymmetry_sweep(ctrl, probs: List[float]) -> Tuple[List[float], List[float]]:
    """
    Run the empirical DQN sweep over fixed buy_mo_prob values.
    Returns pointwise (mean, std) of terminal PnL for each p.
    """
    means: List[float] = []
    stds: List[float] = []

    for i, p in enumerate(probs):
        print(f"\n{'='*60}")
        print(f"  DQN asymmetry sweep: p={p:.2f} [{i+1}/{len(probs)}]")
        print(f"{'='*60}")
        pnls = run_dqn_sims(float(p), ctrl, label=f"[p={p:.2f}]")
        means.append(float(np.mean(pnls)))
        stds.append(float(np.std(pnls, ddof=1)))
        print(f"  => E[PnL]={means[-1]:+.4f}  std={stds[-1]:.4f}")

    return means, stds


def plot_expected_pnl_as_miss_overlay(
    probs: List[float],
    as_means: np.ndarray,
    as_meta: Dict[str, float],
    dqn_means: Optional[List[float]] = None,
    dqn_stds: Optional[List[float]] = None,
    n_sims: int = N_SIMS,
    only_as: bool = False,
    glft_means: Optional[List[float]] = None,
    glft_stds: Optional[List[float]] = None,
) -> None:
    """
    Plot Expected PnL vs P(buy MO):
      - DQN empirical curve in blue (optional)
      - GLFT misspecified analytical curve in red
      - GLFT empirical curve in green (optional)
    """
    x = np.asarray(probs, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))

    if (not only_as) and dqn_means is not None:
        dqn_m = np.asarray(dqn_means, dtype=float)
        if dqn_stds is not None:
            dqn_s = np.asarray(dqn_stds, dtype=float)
            dqn_err = 1.96 * (dqn_s / np.sqrt(max(n_sims, 1)))
        else:
            dqn_err = None
        ax.errorbar(
            x, dqn_m, yerr=dqn_err,
            fmt="o-", color="#E69F00", capsize=4, linewidth=2, markersize=7,
            label=f"DQN (φ={DQN_INV_PENALTY:.1e})",
        )

    # GLFT empirical curve (green) — when available
    if glft_means is not None:
        glft_m = np.asarray(glft_means, dtype=float)
        if glft_stds is not None:
            glft_s = np.asarray(glft_stds, dtype=float)
            glft_err = 1.96 * (glft_s / np.sqrt(max(n_sims, 1)))
        else:
            glft_err = None
        ax.errorbar(
            x, glft_m, yerr=glft_err,
            fmt="s-", color="#0072B2", capsize=4, linewidth=2, markersize=6,
            label=f"GLFT (γ={GLFT_GAMMA:.1e})",
        )

    if as_means is not None:
        ax.plot(
            x, np.asarray(as_means, dtype=float),
            "o--", color="#0072B2", linewidth=2, markersize=6,
            label="GLFT misspecified (analytical)",
        )

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("P(buy MO)", fontsize=15)
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{PnL}]$", fontsize=15)

    # title removed for paper export (caption describes the figure)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    out = "expected_pnl_vs_buy_MO_as_miss_only.png" if only_as else "expected_pnl_vs_buy_MO_as_miss_overlay.png"
    plt.savefig(out, dpi=150)
    _as_means_csv = (
        np.asarray(as_means, dtype=float)
        if as_means is not None
        else np.full(len(probs), np.nan)
    )
    csv_out = save_expected_pnl_overlay_table(
        probs=probs,
        analytical_means=_as_means_csv,
        analytical_meta=as_meta,
        dqn_means=dqn_means,
        dqn_stds=dqn_stds,
        glft_means=glft_means,
        glft_stds=glft_stds,
        n_sims=n_sims,
        only_analytical=only_as,
    )
    plt.show()
    print(f"Saved: {out}")
    print(f"Saved: {csv_out}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════
# Each runner performs N_SIMS independent Monte Carlo episodes with a given
# MM controller (DQN or GLFT) and a given buy_mo_prob specification (either
# a fixed float or the string "regime" to trigger per-episode schedule
# construction).  Terminal PnL is extracted from each episode.

def run_dqn_sims(buy_mo_prob, ctrl, label: str = "") -> np.ndarray:
    """
    Run N_SIMS Monte Carlo episodes with a DQN-based controller.

    Parameters
    ----------
    buy_mo_prob : float or str
        If a float, used as the fixed buy MO probability for all episodes.
        If the string ``"regime"``, a fresh ``make_regime_schedule`` callable
        is built for each episode (with a deterministic seed derived from
        the episode index) to generate non-stationary flow.
    ctrl : object
        The DQN controller (loaded from checkpoint), passed directly to
        ``simulate_LOB_with_MM`` as the ``controller`` argument.
    label : str
        Prefix label for progress print statements.

    Returns
    -------
    np.ndarray, shape (N_SIMS,)
        Array of terminal PnL values, one per simulation episode.
    """

    pnls = np.empty(N_SIMS)

    for i in range(N_SIMS):
        seed = make_seed(i)
        set_seeds(seed)

        # Build a fresh regime schedule per episode (each with its own seed
        # to ensure different regime realisations across episodes while
        # remaining reproducible).
        if buy_mo_prob == "regime":
            bmp = make_default_regime_schedule(seed=seed + 999_999, n_mo_events=N_STEPS)
        else:
            bmp = buy_mo_prob

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=mean_size_LO,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            controller=ctrl,
            random_seed=seed,
            buy_mo_prob=bmp,
            **({"qrm_params": qrm_params} if USE_QRM else {}),
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    {label} sim {i+1}/{N_SIMS}  PnL={pnls[i]:+.4f}")

    return pnls


def run_glft_sims(
    buy_mo_prob,
    A: float,
    kappa: float,
    sigma: float,
    label: str = "",
) -> np.ndarray:
    """
    Run N_SIMS Monte Carlo episodes with the GLFT analytical policy.

    A fresh GLFT policy object is constructed per episode because the
    policy maintains internal state (elapsed time, inventory tracking)
    that must be reset between episodes.

    Parameters
    ----------
    buy_mo_prob : float or str
        Same semantics as ``run_dqn_sims``.
    A, kappa, sigma : float
        Calibrated GLFT model parameters (execution intensity intercept,
        decay rate, and mid-price volatility).
    label : str
        Prefix label for progress print statements.

    Returns
    -------
    np.ndarray, shape (N_SIMS,)
        Array of terminal PnL values.
    """

    pnls = np.empty(N_SIMS)

    for i in range(N_SIMS):
        seed = make_seed(i)
        set_seeds(seed)

        # Fresh GLFT policy per episode (carries internal state that must reset)
        policy = glft_policy_factory(
            gamma=GLFT_GAMMA,
            kappa=kappa,
            A=A,
            sigma=sigma,
            inv_limit=INV_LIMIT,
            round_to_int=True,
            delta=1.0,
            epsilon=None,
            use_tob_update=False,
            n_tob_moves=1,
            use_event_update=False,
            n_events=1000,
            use_time_update=True,
            min_time_interval=GLFT_TIME_THROTTLE,
            use_mdp=USE_MDP_GLFT,
        )

        if buy_mo_prob == "regime":
            bmp = make_default_regime_schedule(seed=seed + 999_999, n_mo_events=N_STEPS)
        else:
            bmp = buy_mo_prob

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=mean_size_LO,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            mm_policy=policy,
            random_seed=seed,
            buy_mo_prob=bmp,
            **({"qrm_params": qrm_params} if USE_QRM else {}),
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    {label} sim {i+1}/{N_SIMS}  PnL={pnls[i]:+.4f}")

    return pnls


def run_glft_only_sweep() -> None:
    """
    Run GLFT empirical sweep over BUY_MO_PROBS_AS and plot E[PnL] vs P(buy MO).

    The GLFT controller is calibrated at p=0.5 (symmetric) and then
    evaluated at each p in the grid.  It skews quotes by inventory
    (correct) but does not know the true p (misspecified in flow).

    This produces a single green curve showing the empirical PnL of the
    GLFT under flow misspecification.
    """
    print("\n=== GLFT EMPIRICAL SWEEP MODE ===")
    print(f"GLFT_GAMMA         = {GLFT_GAMMA:.1e}")
    print(f"USE_MDP_GLFT       = {USE_MDP_GLFT}")
    print(f"GLFT_TIME_THROTTLE = {GLFT_TIME_THROTTLE}")
    print(f"BUY_MO_PROBS_AS    = {BUY_MO_PROBS_AS}")
    print(f"N_SIMS             = {N_SIMS}")

    # Calibrate A, kappa, sigma
    if AS_MISS_A_CACHED is not None and AS_MISS_KAPPA_CACHED is not None:
        A = float(AS_MISS_A_CACHED)
        KAPPA = float(AS_MISS_KAPPA_CACHED)
        SIGMA = float(AS_MISS_SIGMA_OVERRIDE) if AS_MISS_SIGMA_OVERRIDE is not None else 0.3
        print(f"[GLFT] Using cached calibration: A={A:.6f}, kappa={KAPPA:.5f}")
    else:
        A, KAPPA, SIGMA = run_calibration(
            plot=False,
            n_steps_override=AS_MISS_CALIB_STEPS,
            n_equil_override=AS_MISS_CALIB_EQUIL,
        )
        if AS_MISS_SIGMA_OVERRIDE is not None:
            SIGMA = float(AS_MISS_SIGMA_OVERRIDE)

    # Run GLFT at each p
    glft_means = []
    glft_stds = []
    for i, p in enumerate(BUY_MO_PROBS_AS):
        print(f"\n{'='*60}")
        print(f"  GLFT sweep: p={p:.2f} [{i+1}/{len(BUY_MO_PROBS_AS)}]")
        print(f"{'='*60}")
        pnls = run_glft_sims(float(p), A=A, kappa=KAPPA, sigma=SIGMA,
                             label=f"[p={p:.2f}]")
        glft_means.append(float(np.mean(pnls)))
        glft_stds.append(float(np.std(pnls, ddof=1)))
        print(f"  => E[PnL]={glft_means[-1]:+.4f}  std={glft_stds[-1]:.4f}")

    # Plot
    x = np.asarray(BUY_MO_PROBS_AS, dtype=float)
    glft_m = np.asarray(glft_means, dtype=float)
    glft_s = np.asarray(glft_stds, dtype=float)
    glft_err = 1.96 * (glft_s / np.sqrt(max(N_SIMS, 1)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, glft_m, yerr=glft_err,
                fmt="o-", color="forestgreen", capsize=4, linewidth=2, markersize=7,
                label=f"GLFT (γ={GLFT_GAMMA:.1e}, MDP={USE_MDP_GLFT})")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("P(buy MO)", fontsize=13)
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{PnL}]$", fontsize=13)
    ax.set_title("Expected PnL vs MO Flow Asymmetry\n(GLFT empirical)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="upper right")

    txt = (f"A={A:.3g}, k={KAPPA:.3g}, σ={SIGMA:.3g}\n"
           f"γ={GLFT_GAMMA:.1e}, N_SIMS={N_SIMS}")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

    plt.tight_layout()
    out = "expected_pnl_vs_buy_MO_glft_only.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nSaved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS: Visualize a single regime schedule
# ═══════════════════════════════════════════════════════════════════════════════

def plot_regime_example(seed: int = 42, n_mo_events: int = 5000) -> None:
    """
    Plot one example regime schedule to visualize the switching pattern.

    Produces a step-function plot of buy_mo_prob over time, showing how
    the flow direction alternates between buy-heavy and sell-heavy regimes.
    Also prints regime count statistics (mean/median/max length, mean p_buy)
    to help calibrate the regime-duration parameters.
    """

    sched = make_default_regime_schedule(seed=seed, n_mo_events=n_mo_events)
    steps = np.arange(n_mo_events)
    p_vals = np.array([sched(s) for s in steps])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(steps, p_vals, linewidth=0.8, color="steelblue")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="p=0.5")
    ax.set_xlabel("MO Event Index", fontsize=11)
    ax.set_ylabel("P(buy MO)", fontsize=11)
    ax.set_title(
        f"Example Regime Schedule  ({_regime_description()}, seed={seed})",
        fontsize=12,
    )
    ax.set_ylim(0.1, 0.9)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regime_schedule_example.png", dpi=150)
    plt.show()
    print("Saved: regime_schedule_example.png")

    # Print regime statistics
    n_regimes = len(sched.p_values)
    lengths = np.diff(sched.boundaries[:n_regimes + 1])
    print(f"  Regimes: {n_regimes}")
    print(f"  Mean length: {np.mean(lengths):.1f} MO events")
    print(f"  Median length: {np.median(lengths):.1f} MO events")
    print(f"  Max length: {np.max(lengths)} MO events")
    print(f"  Mean p_buy: {np.mean(sched.p_values):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(
    results: Dict[str, Dict[str, float]],
    n_sims: int,
) -> None:
    """
    Three-panel bar chart comparing performance across experimental conditions.

    Panel 1: E[PnL] with 95% confidence intervals (1.96 * SE)
    Panel 2: std(PnL) — measures tail risk / PnL dispersion
    Panel 3: Sharpe ratio (E/std) — risk-adjusted performance metric

    The key diagnostic is the *relative degradation* between the baseline
    (p=0.5 fixed) and regime-switching conditions for each controller type.
    A robust agent should show minimal Sharpe degradation.
    """

    conditions = list(results.keys())
    means = [results[c]["mean"] for c in conditions]
    stds = [results[c]["std"] for c in conditions]
    se_means = [s / np.sqrt(n_sims) for s in stds]

    colors = {
        "DQN: p=0.5 (baseline)": "royalblue",
        "DQN: regime-switching": "crimson",
        "GLFT: p=0.5 (baseline)": "darkorange",
        "GLFT: regime-switching": "orangered",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(conditions))
    bar_colors = [colors.get(c, "gray") for c in conditions]

    # Panel 1: E[PnL]
    ax1.bar(x, means, yerr=[1.96 * se for se in se_means],
            color=bar_colors, capsize=5, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel(r"$\mathbb{E}[\mathrm{PnL}]$", fontsize=12)
    ax1.set_title("Expected PnL", fontsize=13)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: std(PnL)
    ax2.bar(x, stds, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\sqrt{\mathrm{Var}(\mathrm{PnL})}$", fontsize=12)
    ax2.set_title("PnL Volatility", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Sharpe
    sharpes = [m / s if s > 1e-12 else 0.0 for m, s in zip(means, stds)]
    ax3.bar(x, sharpes, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("Sharpe  (E / std)", fontsize=12)
    ax3.set_title("Sharpe Ratio", fontsize=13)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Regime-Switching Stress Test  (N={n_sims}, "
        f"{_regime_description()})",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("regime_switching_stress_test.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: regime_switching_stress_test.png")


def plot_pnl_distributions(
    results_pnls: Dict[str, np.ndarray],
) -> None:
    """
    Overlaid histograms of terminal PnL distributions for each condition.

    This visualisation complements the bar chart by showing the full
    distributional shape — particularly useful for detecting:
      - Left-tail fattening under regime-switching (large drawdowns)
      - Bimodality (agent performs well in some regimes, poorly in others)
      - Distributional shift vs pure mean/variance changes
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "DQN: p=0.5 (baseline)": "royalblue",
        "DQN: regime-switching": "crimson",
        "GLFT: p=0.5 (baseline)": "darkorange",
        "GLFT: regime-switching": "orangered",
    }

    for label, pnls in results_pnls.items():
        ax.hist(pnls, bins=40, alpha=0.4, label=label,
                color=colors.get(label, "gray"), density=True, edgecolor="black",
                linewidth=0.3)

    ax.set_xlabel("Terminal PnL", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"PnL Distribution: Fixed vs Regime-Switching MO Flow  "
        f"(N={N_SIMS})",
        fontsize=13,
    )
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regime_switching_pnl_dist.png", dpi=150)
    plt.show()
    print("Saved: regime_switching_pnl_dist.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GLFT MISSPECIFIED ANALYTICAL WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def run_as_miss_plot_workflow() -> None:
    """
    Run the asymmetry sweep workflow for GLFT misspecified analysis.

    Modes:
      - PLOT_AS_MISS=True,  PLOT_ONLY_AS_MISS=False:
          plot DQN empirical (blue) + GLFT misspecified analytical (red).
      - PLOT_ONLY_AS_MISS=True:
          plot only GLFT misspecified analytical (red).
    """
    only_as = bool(PLOT_ONLY_AS_MISS)

    print("\n=== GLFT MISSPECIFIED ANALYTICAL CURVE MODE ===")
    print(f"PLOT_AS_MISS           = {PLOT_AS_MISS}")
    print(f"PLOT_ONLY_AS_MISS      = {PLOT_ONLY_AS_MISS}")
    print(f"AS_MISS_GAMMA          = {AS_MISS_GAMMA:.1e}")
    print(f"AS_MISS_SIGMA_OVERRIDE = {AS_MISS_SIGMA_OVERRIDE}")
    print(f"BUY_MO_PROBS_AS        = {BUY_MO_PROBS_AS}")

    # Use pre-calibrated A and kappa if available (saves ~5 min of simulation).
    # Set AS_MISS_A_CACHED or AS_MISS_KAPPA_CACHED to None to force re-calibration.
    if AS_MISS_A_CACHED is not None and AS_MISS_KAPPA_CACHED is not None:
        A = float(AS_MISS_A_CACHED)
        KAPPA = float(AS_MISS_KAPPA_CACHED)
        SIGMA_CALIB = float(AS_MISS_SIGMA_OVERRIDE) if AS_MISS_SIGMA_OVERRIDE is not None else 0.3
        print(f"[GLFT MISS] Using cached calibration: A={A:.6f}, kappa={KAPPA:.5f}")
    else:
        # Calibrate A and kappa via censored-waiting-times pipeline.
        # Use a long simulation (AS_MISS_CALIB_STEPS=500k) for precise estimates,
        # since the analytical PnL curve is sensitive to A and kappa values.
        A, KAPPA, SIGMA_CALIB = run_calibration(
            plot=False,
            n_steps_override=AS_MISS_CALIB_STEPS,
            n_equil_override=AS_MISS_CALIB_EQUIL,
        )
    sigma_use = float(AS_MISS_SIGMA_OVERRIDE if AS_MISS_SIGMA_OVERRIDE is not None else SIGMA_CALIB)

    if PLOT_ANALYTICAL:
        # Estimate continuous-time episode horizon from simulation tape.
        T_ep = estimate_episode_time_horizon(seed=SEED_CALIB + 2)

        # Compute the q=0 quote distance delta_bar for R_FILL_FACTOR calibration.
        if float(AS_MISS_GAMMA) > 1e-12:
            _delta_base = np.log1p(float(AS_MISS_GAMMA) / KAPPA) / float(AS_MISS_GAMMA)
            _log_base = 0.5 * (1.0 + KAPPA / float(AS_MISS_GAMMA)) * np.log1p(
                float(AS_MISS_GAMMA) / KAPPA
            )
            _power_term = np.exp(min(_log_base, 500.0))
            _c_skew = np.sqrt(
                sigma_use**2 * float(AS_MISS_GAMMA) / (2.0 * KAPPA * A)
            ) * _power_term
        else:
            _delta_base = 1.0 / KAPPA
            _c_skew = 0.0
        _delta_bar = float(_delta_base + 0.5 * _c_skew)

        # Calibrate R_FILL_FACTOR using GLFT (not DQN) at p=0.5.
        # The analytical curve models a GLFT-style MM, so we measure the fill
        # rate of the GLFT controller in the simulator to calibrate R.
        if AS_MISS_R_FILL_FACTOR is not None:
            r_fill_factor = float(AS_MISS_R_FILL_FACTOR)
            print(f"[GLFT MISS] Using cached R_FILL_FACTOR = {r_fill_factor:.6f}")
        else:
            r_fill_factor = calibrate_r_fill_factor(
                A=A, kappa=KAPPA, sigma=sigma_use, delta_bar=_delta_bar,
            )

        # Calibrate v0 if USE_V0 is enabled
        _v0 = 0.0
        if AS_MISS_USE_V0:
            if AS_MISS_V0_CACHED is not None:
                _v0 = float(AS_MISS_V0_CACHED)
                print(f"[GLFT MISS] Using cached v0 = {_v0:.6f}")
            else:
                _v0 = calibrate_v0()

        as_means, as_meta = compute_as_miss_expected_pnl_curve(
            probs=BUY_MO_PROBS_AS,
            A=A,
            kappa=KAPPA,
            sigma=sigma_use,
            gamma=float(AS_MISS_GAMMA),
            T=float(T_ep),
            inv_limit=INV_LIMIT,
            r_fill_factor=r_fill_factor,
            tick_size=TICK_SIZE,
            use_v0=AS_MISS_USE_V0,
            v0=_v0,
        )
    else:
        print("[GLFT MISS] PLOT_ANALYTICAL=False: skipping analytical curve "
              "(and r_fill_factor / v0 calibrations).")
        as_means = None
        as_meta = {}

    # ── Optional: GLFT empirical sweep ──────────────────────────────────
    glft_means = None
    glft_stds = None
    if RUN_GLFT:
        print(f"\n=== GLFT EMPIRICAL SWEEP (γ={GLFT_GAMMA:.1e}) ===")
        _glft_means_list = []
        _glft_stds_list = []
        for i, p in enumerate(BUY_MO_PROBS_AS):
            print(f"\n{'='*60}")
            print(f"  GLFT sweep: p={p:.2f} [{i+1}/{len(BUY_MO_PROBS_AS)}]")
            print(f"{'='*60}")
            pnls = run_glft_sims(float(p), A=A, kappa=KAPPA, sigma=sigma_use,
                                 label=f"[p={p:.2f}]")
            _glft_means_list.append(float(np.mean(pnls)))
            _glft_stds_list.append(float(np.std(pnls, ddof=1)))
            print(f"  => E[PnL]={_glft_means_list[-1]:+.4f}  std={_glft_stds_list[-1]:.4f}")
        glft_means = _glft_means_list
        glft_stds = _glft_stds_list

    if only_as:
        plot_expected_pnl_as_miss_overlay(
            probs=BUY_MO_PROBS_AS,
            as_means=as_means,
            as_meta=as_meta,
            dqn_means=None,
            dqn_stds=None,
            n_sims=N_SIMS,
            only_as=True,
            glft_means=glft_means,
            glft_stds=glft_stds,
        )
        return

    if not os.path.isfile(DQN_CHECKPOINT):
        print(f"[ERROR] Checkpoint not found for DQN blue curve: {DQN_CHECKPOINT}")
        return

    print(f"\nLoading DQN checkpoint for blue curve: {DQN_CHECKPOINT}")
    ctrl = make_controller_from_checkpoint(
        DQN_CHECKPOINT,
        log_dir="runs_eval/as_miss_dqn",
        use_time_update=True,
        min_time_interval=DQN_TIME_THROTTLE,
        inv_limit_override=INV_LIMIT,
    )

    dqn_means, dqn_stds = run_dqn_asymmetry_sweep(ctrl, BUY_MO_PROBS_AS)

    plot_expected_pnl_as_miss_overlay(
        probs=BUY_MO_PROBS_AS,
        as_means=as_means,
        as_meta=as_meta,
        dqn_means=dqn_means,
        dqn_stds=dqn_stds,
        n_sims=N_SIMS,
        only_as=False,
        glft_means=glft_means,
        glft_stds=glft_stds,
    )

    try:
        ctrl.writer.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point for the regime-switching stress test.

    Execution flow:
      0. Plot an example regime schedule (visual sanity check)
      1. [Optional] Calibrate GLFT parameters from a bare LOB simulation
      2. Load the pre-trained DQN checkpoint
      3. Run Monte Carlo simulations under each condition
      4. Print summary statistics (E[PnL], std, Sharpe, % degradation)
      5. Generate comparison plots and PnL distribution histograms
    """

    os.chdir("/Users/felipemoret/Desktop/MM_LOB_SIM/")

    print("=== REGIME-SWITCHING STRESS TEST ===")
    print(f"Engine             = {'QRM' if USE_QRM else 'Santa Fe'}")
    print(f"lam                = {LAM}")
    print(f"mu                 = {MU}")
    print(f"delta              = {DELTA}")
    print(f"mean_size_LO       = {mean_size_LO}")
    print(f"mean_size_MO       = {mean_size_MO}")
    print(f"N_STEPS            = {N_STEPS}")
    print(f"N_SIMS             = {N_SIMS}")
    print(f"REGIME_DISTRIBUTION = {REGIME_DISTRIBUTION}")
    if REGIME_DISTRIBUTION == "exponential":
        print(f"REGIME_EXP_TAU    = {REGIME_EXP_TAU}")
    else:
        print(f"REGIME_ALPHA      = {REGIME_ALPHA}")
        print(f"REGIME_L_MIN      = {REGIME_L_MIN}")
    print(f"REGIME_P range     = [{REGIME_P_LO}, {REGIME_P_HI}]")
    print(f"PLOT_AS_MISS       = {PLOT_AS_MISS}")
    print(f"PLOT_ONLY_AS_MISS  = {PLOT_ONLY_AS_MISS}")
    if USE_QRM:
        print(f"intensity_model    = {qrm_params.get('intensity_model', 'N/A')}")
        print(f"use_dynamic_pref   = {qrm_params.get('use_dynamic_pref', 'N/A')}")
        print(f"size_q             = {qrm_params.get('size_q', 'N/A')}")
        print(f"aes                = {qrm_params.get('aes', 'N/A')}")
    print("=" * 40)

    # Optional alternative workflows
    if PLOT_AS_MISS or PLOT_ONLY_AS_MISS:
        run_as_miss_plot_workflow()
        return
    if PLOT_ONLY_GLFT:
        run_glft_only_sweep()
        return

    # ── 0. Show an example regime schedule ──────────────────────────────────
    print("\n--- Example regime schedule ---")
    plot_regime_example(seed=42, n_mo_events=N_STEPS)

    # ── 1. Calibrate GLFT (only if needed) ──────────────────────────────────
    if RUN_GLFT:
        A, KAPPA, SIGMA = run_calibration(plot=False)
    else:
        A, KAPPA, SIGMA = 0.0, 0.0, 0.0

    # ── 2. Load DQN controller once ─────────────────────────────────────────
    if not os.path.isfile(DQN_CHECKPOINT):
        print(f"[ERROR] Checkpoint not found: {DQN_CHECKPOINT}")
        return

    print(f"\nLoading DQN checkpoint: {DQN_CHECKPOINT}")
    ctrl = make_controller_from_checkpoint(
        DQN_CHECKPOINT,
        log_dir="runs_eval/regime_switching_dqn",
        use_time_update=True,
        min_time_interval=DQN_TIME_THROTTLE,
        inv_limit_override=INV_LIMIT,
    )

    # ── 3. Run simulations ──────────────────────────────────────────────────
    results = {}
    results_pnls = {}

    # 3a. DQN baseline (p=0.5 fixed)
    print(f"\n{'='*60}")
    print("  DQN: p=0.5 FIXED (baseline)")
    print(f"{'='*60}")
    pnls_baseline = run_dqn_sims(0.5, ctrl, label="[baseline]")
    results["DQN: p=0.5 (baseline)"] = {
        "mean": float(np.mean(pnls_baseline)),
        "std": float(np.std(pnls_baseline, ddof=1)),
    }
    results_pnls["DQN: p=0.5 (baseline)"] = pnls_baseline
    print(f"  => E[PnL]={results['DQN: p=0.5 (baseline)']['mean']:+.4f}  "
          f"std={results['DQN: p=0.5 (baseline)']['std']:.4f}")

    # 3b. DQN regime-switching
    print(f"\n{'='*60}")
    print("  DQN: REGIME-SWITCHING")
    print(f"{'='*60}")
    pnls_regime = run_dqn_sims("regime", ctrl, label="[regime]")
    results["DQN: regime-switching"] = {
        "mean": float(np.mean(pnls_regime)),
        "std": float(np.std(pnls_regime, ddof=1)),
    }
    results_pnls["DQN: regime-switching"] = pnls_regime
    print(f"  => E[PnL]={results['DQN: regime-switching']['mean']:+.4f}  "
          f"std={results['DQN: regime-switching']['std']:.4f}")

    # 3c. GLFT (if enabled)
    if RUN_GLFT:
        print(f"\n{'='*60}")
        print("  GLFT: p=0.5 FIXED (baseline)")
        print(f"{'='*60}")
        pnls_glft_base = run_glft_sims(0.5, A, KAPPA, SIGMA, label="[GLFT base]")
        results["GLFT: p=0.5 (baseline)"] = {
            "mean": float(np.mean(pnls_glft_base)),
            "std": float(np.std(pnls_glft_base, ddof=1)),
        }
        results_pnls["GLFT: p=0.5 (baseline)"] = pnls_glft_base

        print(f"\n{'='*60}")
        print("  GLFT: REGIME-SWITCHING")
        print(f"{'='*60}")
        pnls_glft_regime = run_glft_sims("regime", A, KAPPA, SIGMA, label="[GLFT regime]")
        results["GLFT: regime-switching"] = {
            "mean": float(np.mean(pnls_glft_regime)),
            "std": float(np.std(pnls_glft_regime, ddof=1)),
        }
        results_pnls["GLFT: regime-switching"] = pnls_glft_regime

    # ── 4. Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("REGIME-SWITCHING STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<30s}  {'E[PnL]':>12s}  {'std(PnL)':>10s}  {'Sharpe':>8s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*10}  {'-'*8}")
    for cond, stats in results.items():
        m, s = stats["mean"], stats["std"]
        sharpe = m / s if s > 1e-12 else 0.0
        print(f"  {cond:<30s}  {m:+12.4f}  {s:10.4f}  {sharpe:+8.4f}")
    print(f"{'='*70}")

    # Degradation
    m_base = results["DQN: p=0.5 (baseline)"]["mean"]
    m_regime = results["DQN: regime-switching"]["mean"]
    if abs(m_base) > 1e-12:
        pct_change = 100.0 * (m_regime - m_base) / abs(m_base)
        print(f"\n  DQN PnL change under regime-switching: {pct_change:+.1f}%")
    else:
        print(f"\n  DQN PnL change: {m_regime - m_base:+.4f} (absolute)")

    # ── 5. Plots ────────────────────────────────────────────────────────────
    plot_comparison(results, N_SIMS)
    plot_pnl_distributions(results_pnls)

    # Cleanup
    try:
        ctrl.writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
