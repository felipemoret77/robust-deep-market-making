#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLFT_studies.py  —  GLFT Gamma-Sweep Study Runner
==================================================

Purpose
-------
This script investigates how the **risk aversion parameter gamma** (also
denoted φ in some GLFT papers) affects the performance of the Guéant-Lehalle-
Fernandez-Tapia (2013) optimal market-making policy.  It runs a Monte Carlo
study over a grid of gamma values and produces diagnostic plots that
illustrate the trade-off between profitability and inventory risk.

The GLFT Model (Quick Refresher)
---------------------------------
The GLFT model assumes a market maker operating in a diffusion-driven LOB
with Poisson execution intensities that decay exponentially with the distance
from the best price:

    Λ(δ) = A · exp(-κ · δ)

where:
    A     — execution intensity at zero distance (calibrated from data).
    κ     — exponential decay rate of execution intensity.
    δ     — distance of the quote from the current best price (in ticks).

The optimal half-spread and inventory skew are given by:

    half_spread = (1/κ) · ln(1 + κ/γ)
    skew_factor = (2 · I_t · γ) / κ
    q_bid = mid - half_spread - skew_factor
    q_ask = mid + half_spread + skew_factor

where γ (gamma) is the risk aversion parameter:
    - Small γ (e.g. 1e-6) → risk-neutral → tight spreads, large inventory swings.
    - Large γ (e.g. 1e-1) → risk-averse → wider spreads, tighter inventory control.

Workflow
--------
1) **Calibrate GLFT parameters once** — uses censored-waiting-time calibration
   (fit_execution_intensity_censored_waiting_times + fit_A_kappa_loglinear) to
   obtain the execution intensity parameters (A, kappa), then fits mid-price
   volatility (σ) via fit_volatility.  These three parameters fully
   characterize the GLFT optimal quotes.

2) **Build GLFT policy variants** — one variant per gamma value in
   GAMMA_VALUES.

3) **Run N_SIMS_PER_GAMMA independent simulations per variant** — each uses a
   deterministic seed derived from (gamma_index, run_id), so results are
   fully reproducible.

4) **Produce summary plots**:
   - Mean TotalPnL vs gamma (with error bars).
   - Boxplots of final TotalPnL by gamma.
   - Histograms of final TotalPnL by gamma.
   - Mean TotalPnL trajectory ± 1σ band by gamma.
   - Mean inventory trajectory ± 1σ band by gamma.
   - Cartea-style PnL and lifetime inventory distributions.
   - Optional always-best bid/ask benchmark overlays when
     PLOT_ALWAYS_BEST_BENCHMARK is enabled.

Dependencies
------------
- simulate_LOB_with_MM          : from MM_LOB_SIM.py
- glft_policy_factory           : from GLFT_policy_factory.py
- always_best_bid_ask_mm_policy_factory
                                : from MM_policy_5.py
- fit_volatility                : from calibrate_trading_intensity.py
- fit_execution_intensity_censored_waiting_times, fit_A_kappa_loglinear
                                : from censored_waiting_times_calib.py

Usage
-----
    python GLFT_studies.py

The calibration runs first (~1-2 min), then the Monte Carlo study begins.
Results are saved to ``glft_gamma_study_summary.pkl`` / ``.csv``.
"""

import math
import random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Paper figure style: larger labels/legends, no embedded titles ===
plt.rcParams.update({
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.fancybox": True,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
})
# Paper palette: viridis-sampled colors so that successive lines have
# clearly distinct hues (per supervisor feedback that single-hue shading
# is hard to read in print) while still ordering monotonically along the
# colormap.  Six colors here match the typical gamma-sweep cardinality;
# scripts that need a different N should sample
# ``plt.cm.viridis(np.linspace(0.05, 0.95, N))`` directly.  The 0.05 /
# 0.95 trim avoids the near-black purple and the saturated yellow ends.
from cycler import cycler
import numpy as np
_VIRIDIS_PAPER_PALETTE = [
    tuple(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 6))
]
plt.rcParams["axes.prop_cycle"] = cycler(color=_VIRIDIS_PAPER_PALETTE)

# ---------------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------------
# GLFT policy factory: builds the optimal-quoting callable for the simulator.
from GLFT_policy_factory import glft_policy_factory

# Naive/benchmark policy: always quote at the current best bid and best ask.
from MM_policy_5 import always_best_bid_ask_mm_policy_factory

# Volatility estimation: fits mid-price diffusion σ from simulated data.
from calibrate_trading_intensity import fit_volatility

# LOB simulator with integrated market-maker agent.
from MM_LOB_SIM import simulate_LOB_with_MM

# Censored-waiting-time calibration: fits execution intensity Λ(δ) = A·exp(-κδ).
from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)


# ----------------------------------------------------------------------
# Paper figure helper: mirror every saved figure into the paper folder so
# the LaTeX project picks them up directly.  Stays local-only when the
# paper folder is not present (e.g. when running on a different machine).
# ----------------------------------------------------------------------
from pathlib import Path as _PaperPath

PAPER_FIG_DIR = _PaperPath("/Users/felipemoret/Desktop/extended_first_abstract_RLMM")

def _save_paper_figure(fig_or_plt, name, local_dir=None, **savefig_kwargs):
    savefig_kwargs.setdefault('dpi', 150)
    savefig_kwargs.setdefault('bbox_inches', 'tight')
    local_target = (_PaperPath(local_dir) / name) if local_dir is not None else name
    fig_or_plt.savefig(local_target, **savefig_kwargs)
    if PAPER_FIG_DIR.is_dir():
        fig_or_plt.savefig(PAPER_FIG_DIR / name, **savefig_kwargs)


# ============================================================================
# 1) Global Simulation Parameters (LOB + MM Environment)
#
#    These constants define the Santa Fe LOB model parameters and are shared
#    by both the calibration phase and the Monte Carlo study phase.
# ============================================================================

# --- Santa Fe LOB model rates (AMZN calibration) ---
LAM = 0.06                       # Limit order arrival rate per tick level
MU = 0.1                         # Cancellation rate per order
DELTA = 0.02                     # Market order arrival rate

# --- LOB geometry ---
NUMBER_TICK_LEVELS = 50          # Number of price levels on each side of mid
N_PRIORITY_RANKS = 100           # Max queue depth at each price level
HALF_TICK = 0.5                  # Half-tick size used by calibration functions

# --- Simulation duration ---
N_STEPS = 5_000               # Total simulation events per episode
N_STEPS_TO_EQUIL = 1_000       # Warm-up events before the MM agent activates



# --- Throttling intervals (simulation time units) ---
# time_interval_throttle: minimum time between GLFT policy re-evaluations.
#     Higher values reduce quote flickering but increase staleness risk.
# time_interval_calib: aggregation window for the volatility estimator.
#     Shorter windows give finer resolution but noisier estimates.
time_interval_throttle = 1
time_interval_calib = 0.5

# --- Calibration seed ---
# A fixed seed for calibration ensures that the calibrated parameters
# (A, kappa, sigma) are identical across independent runs of this script.
SEED_CALIB = 1234


# ============================================================================
# 2) Study Configuration (gamma grid, number of sims, etc.)
# ============================================================================

# --- GLFT gamma grid ---
# Small gamma → risk-neutral → tight spreads → higher mean PnL but volatile inventory.
# Large gamma → risk-averse → wider spreads → lower mean PnL but stable inventory.
GAMMA_VALUES = [
    1e-6,    # Near risk-neutral: maximizes expected PnL, accepts large inventory swings
    1e-3,    # Moderate risk aversion: balanced trade-off
    1e-1,    # Strong risk aversion: prioritizes inventory control over PnL
]

N_SIMS_PER_GAMMA = 1000           # Number of independent simulations per gamma value

# --- Seed control ---
# All runs with the same run_id share a deterministic seed, ensuring
# reproducibility.  Changing SEED_BASE produces an entirely different
# but equally reproducible set of market paths.
SEED_BASE = 123456

# --- MM agent constraints ---
INV_LIMIT = 8                 # Maximum absolute inventory before one-sided quoting
ROUND_TO_INT = True              # Round GLFT continuous quotes to integer tick indices

# --- GLFT throttling via TOB moves ---
# If USE_TOB_UPDATE is True, the GLFT policy re-evaluates only when the
# top-of-book moves by at least N_TOB_MOVES ticks.
USE_TOB_UPDATE = False
N_TOB_MOVES = 1
USE_MDP_GLFT = True

# --- Always-best benchmark overlay ---
# When enabled, run the naive always-best bid/ask policy with the same
# time throttle as the GLFT policies and overlay it in the PnL histograms
# and mean PnL trajectory plots.
PLOT_ALWAYS_BEST_BENCHMARK = True
ALWAYS_BEST_LABEL = "At-best baseline"
# None means: inherit the exact same physical-time throttle as GLFT.
ALWAYS_BEST_TIME_THROTTLE: Optional[float] = None
USE_MDP_ALWAYS_BEST = USE_MDP_GLFT

# --- Warmup trimming for plots ---
# If the mm_df includes the warm-up period, set this > 0 to trim initial
# steps from trajectory plots.  Default 0 avoids "double
# trimming" (simulate_LOB_with_MM already skips the warm-up internally).
DROP_WARMUP_PLOT = 0


# ============================================================================
# 3) Calibration Phase
#
#    The calibration functions run a standalone LOB simulation and extract:
#      (a) Execution intensity parameters (A, kappa) via censored waiting times.
#      (b) Mid-price volatility (sigma) via incremental variance estimation.
#
#    These parameters are required by the GLFT policy factory and MUST be
#    computed before the study loop starts.
#
#    IMPORTANT: This code runs at module level because the GLFT study runner
#    needs A, KAPPA, SIGMA defined before make_glft_policy() is called.
#    However, to avoid expensive computation on accidental imports, the
#    actual study loop is guarded by ``if __name__ == "__main__"``.
# ============================================================================

def run_calibration(plot: bool = True) -> Tuple[float, float, float]:
    """
    Run the full GLFT calibration pipeline and return (A, KAPPA, SIGMA).

    This function performs two sequential calibration steps:

    Step 1 — Censored Waiting Times → Execution Intensity
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Runs a simulation, computes the empirical execution intensity Λ(δ) at
    various distances δ from the best price, then fits the GLFT exponential
    model Λ(δ) = A · exp(-κ · δ) via log-linear regression.

    Step 2 — Mid-Price Volatility → σ
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Runs a second simulation (or reuses the first) to estimate the mid-price
    diffusion coefficient σ from the variance of price increments over fixed
    time windows.

    Parameters
    ----------
    plot : bool
        If True, display diagnostic plots for each calibration step.

    Returns
    -------
    A : float
        Execution intensity at zero distance.
    KAPPA : float
        Exponential decay rate of execution intensity.
    SIGMA : float
        Mid-price volatility per unit time.
    """
    # -------------------------------------------------------------------
    # Step 1: Censored Waiting Times → A, kappa
    #
    # The censored waiting time method estimates the execution intensity
    # at each distance δ from the best price by measuring how long limit
    # orders survive before being filled (or censored by cancellation).
    #
    # - aggregation_mode="time" groups events by wall-clock time windows.
    # - time_interval=10 sets each window to 10 simulation time units.
    # - side_mode="buy" uses buy-side fills (symmetric by assumption).
    # - max_levels=5 limits the distance grid to 5 tick levels.
    # -------------------------------------------------------------------
    res_waiting_times = fit_execution_intensity_censored_waiting_times(
        aggregation_mode="time",
        time_interval=time_interval_calib,
        max_levels=5,
        half_tick=HALF_TICK,
        side_mode="buy",

        # LOB simulation parameters (must match the study runs)
        lam=LAM, mu=MU, delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=500_000,
        n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
        split_sweeps=False,
        random_seed=SEED_CALIB,
        plot=plot,
    )

    # -------------------------------------------------------------------
    # Log-linear regression: fit A and kappa from the empirical Λ(δ) curve.
    #
    # The GLFT model assumes Λ(δ) = A · exp(-κ · δ).  Taking logs:
    #     log(Λ(δ)) = log(A) - κ · δ
    # which is a simple linear regression in (δ, log(Λ)) space.
    #
    # The weights from denom_sum_tau give more influence to distances with
    # longer cumulative exposure time (more reliable lambda estimates).
    # -------------------------------------------------------------------
    A, KAPPA, info = fit_A_kappa_loglinear(
        res_waiting_times["delta_grid"],
        res_waiting_times["lambda_hat"],
        weights=res_waiting_times["denom_sum_tau"],   # recommended weighting
    )

    print(f"[CALIB] A = {A:.6g}")
    print(f"[CALIB] kappa = {KAPPA:.6g}")

    # --- Diagnostic plot: empirical vs fitted execution intensity ---
    if plot:
        d = res_waiting_times["delta_grid"]
        lam_hat = res_waiting_times["lambda_hat"]
        lam_fit = A * np.exp(-KAPPA * d)

        plt.figure()
        plt.plot(d, lam_hat, marker="o", linestyle="-", label="lambda_hat (empirical)")
        plt.plot(d, lam_fit, linestyle="--",
                 label=f"fit: A e^(-kappa d)\nA={A:.3g}, kappa={KAPPA:.3g}")
        # plt.yscale("log")  # helps a lot to see the fit on a log scale
        plt.xlabel("delta (distance from best price, ticks)")
        plt.ylabel("lambda (execution intensity)")
        # plt.title("Execution Intensity Calibration: Censored Waiting Times")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.show()

    # -------------------------------------------------------------------
    # Step 2: Fit Volatility (sigma) — Non-Queue-Aware
    #
    # Sigma measures the mid-price diffusion rate: dS_t = σ dW_t.
    # We estimate it from the variance of mid-price increments over
    # time windows of length time_interval_calib.
    # -------------------------------------------------------------------
    # res_vol = fit_volatility(
    #     queue_aware=False,

    #     aggregation_mode="time",
    #     time_interval=time_interval_calib,

    #     use_tob_buckets=False,
    #     n_tob_moves=10,

    #     lam=LAM, mu=MU, delta=DELTA,

    #     number_tick_levels=NUMBER_TICK_LEVELS,
    #     n_priority_ranks=N_PRIORITY_RANKS,
    #     n_steps=N_STEPS,
    #     n_steps_to_equilibrium=N_STEPS_TO_EQUIL,

    #     half_tick=HALF_TICK,
    #     max_depth_levels=5,
    #     max_ranks=5,
    #     rank_to_fit=0,
    #     strict=True,
    #     plot=plot,
    #     random_seed=SEED_CALIB + 1,
    # )

    # SIGMA = float(res_vol["volatility"])
    SIGMA = 0.3 ##from signature plot
    #unit = res_vol.get("volatility_unit", "")

    #print(f"[CALIB] volatility = {SIGMA:.6g} {unit}")
    print(f"[CALIB] ((sigma*kappa)^2 * Delta_T) / 2 = "
          f"{((SIGMA * KAPPA) ** 2) * 1 / 2:.6g}")

    return float(A), float(KAPPA), SIGMA


# ============================================================================
# 4) Helper: Build a GLFT Policy for a Given Gamma
#
#    Creates a GLFT policy callable that the simulator invokes on every event.
#    The factory pattern ensures each call returns a fresh policy with its own
#    internal state (important for stateful throttle logic).
# ============================================================================

def make_glft_policy(
    gamma: float,
    A_param: float,
    kappa: float,
    sigma: float,
):
    """
    Build a GLFT policy with a specific gamma using calibrated (A, kappa, sigma).

    Parameters
    ----------
    gamma : float
        Risk aversion parameter (swept in this study).
    A_param : float
        Execution intensity at zero distance (from calibration).
    kappa : float
        Exponential decay rate (from calibration).
    sigma : float
        Mid-price volatility (from calibration).

    Returns
    -------
    callable
        An mm_policy function: state_dict -> action_tuple.
    """
    policy = glft_policy_factory(
        gamma=gamma,
        kappa=kappa,
        A=A_param,
        sigma=sigma,
        inv_limit=INV_LIMIT,
        round_to_int=ROUND_TO_INT,

        # --- GLFT formula generalization parameters ---
        # delta=1.0: standard GLFT Delta scaling (time step parameter).
        # epsilon=None: defaults to gamma (no separate epsilon override).
        delta=1.0,
        epsilon=None,

        # --- Throttling: TOB (Top-Of-Book moves) ---
        use_tob_update=USE_TOB_UPDATE,
        n_tob_moves=N_TOB_MOVES,

        # --- Throttling: Simulation Steps (Events) ---
        use_event_update=False,
        n_events=1000,

        # --- Throttling: Simulation Time (Seconds) ---
        # Time-based throttling is the primary throttle for this study.
        # min_time_interval = time_interval_throttle ensures the GLFT policy
        # re-evaluates at most once every time_interval_throttle sim-time units.
        use_time_update=True,
        min_time_interval=time_interval_throttle,
        use_mdp=USE_MDP_GLFT,
    )
    return policy


def make_always_best_policy():
    """
    Build the always-best bid/ask benchmark with the same throttle geometry
    used by the GLFT policies in this study.
    """
    min_time_interval = (
        time_interval_throttle
        if ALWAYS_BEST_TIME_THROTTLE is None
        else ALWAYS_BEST_TIME_THROTTLE
    )
    return always_best_bid_ask_mm_policy_factory(
        inv_limit=INV_LIMIT,
        use_tob_update=USE_TOB_UPDATE,
        n_tob_moves=N_TOB_MOVES,
        use_event_update=False,
        n_events=1000,
        use_time_update=True,
        min_time_interval=min_time_interval,
        use_mdp=USE_MDP_ALWAYS_BEST,
    )


# ============================================================================
# 5) Helper: Deterministic Per-Run Seed
#
#    Maps (gamma_index, run_id) → integer seed with a large stride to prevent
#    seed collisions between gamma groups.
# ============================================================================

def make_run_seed(gamma_index: int, run_id: int) -> int:
    """
    Deterministic mapping from (gamma_index, run_id) to an integer seed.

    Uses a large stride (100,000) between gamma groups to ensure no overlaps
    even with hundreds of runs per gamma.

    Parameters
    ----------
    gamma_index : int
        Index of the gamma value in GAMMA_VALUES.
    run_id : int
        Simulation run number within this gamma group.

    Returns
    -------
    int
        Unique deterministic seed.
    """
    return int(SEED_BASE + 100_000 * int(gamma_index) + int(run_id))


def seed_everything(seed: int) -> None:
    """
    Seed BOTH numpy and Python's random for full determinism.

    The simulator also accepts ``random_seed`` to seed its internal RNG,
    but we seed here to make the entire runner deterministic even if other
    code uses random/np.random between calls.

    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))


# ============================================================================
# 6) Helper: Run a Single Simulation and Collect MM DataFrame
#
#    Wraps simulate_LOB_with_MM with the global LOB parameters and a
#    gamma-specific GLFT policy.
# ============================================================================

def run_single_simulation(
    gamma: float,
    A_param: float,
    kappa: float,
    sigma: float,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one LOB simulation with a GLFT policy parameterized by gamma.

    Parameters
    ----------
    gamma : float
        GLFT risk aversion parameter.
    A_param : float
        Calibrated execution intensity at zero distance.
    kappa : float
        Calibrated exponential decay rate.
    sigma : float
        Calibrated mid-price volatility.
    seed : int or None
        If provided, seeds all RNGs before running.

    Returns
    -------
    msg_df : pd.DataFrame
        Message-level LOB event log.
    ob_df : pd.DataFrame
        Order book snapshot time series.
    mm_df : pd.DataFrame
        Market-maker telemetry (PnL, inventory, quotes, etc.).
    """
    policy = make_glft_policy(gamma, A_param, kappa, sigma)

    if seed is not None:
        seed_everything(seed)

    msg_df, ob_df, mm_df = simulate_LOB_with_MM(
        lam=LAM,
        mu=MU,
        delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        iterations=N_STEPS,
        iterations_to_equilibrium=N_STEPS_TO_EQUIL,
        mm_policy=policy,
        exclude_self_from_state=False,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,

        # IMPORTANT: passes the seed to the simulator's internal RNG,
        # ensuring identical LOB event sequences for the same seed.
        random_seed=seed,
    )

    return msg_df, ob_df, mm_df


def run_single_always_best_simulation(
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one LOB simulation with the always-best bid/ask benchmark policy.

    The benchmark uses the same LOB geometry and time throttle as the GLFT
    policies, so it can be overlaid in trajectory and distribution plots.
    """
    policy = make_always_best_policy()

    if seed is not None:
        seed_everything(seed)

    msg_df, ob_df, mm_df = simulate_LOB_with_MM(
        lam=LAM,
        mu=MU,
        delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        iterations=N_STEPS,
        iterations_to_equilibrium=N_STEPS_TO_EQUIL,
        mm_policy=policy,
        exclude_self_from_state=False,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        random_seed=seed,
    )

    return msg_df, ob_df, mm_df


# ============================================================================
# 7) Helper: Summarize a Single MM Run
#
#    Extracts key scalar metrics from a mm_df DataFrame for aggregation
#    across runs.
# ============================================================================

def summarize_mm_run(
    mm_df: pd.DataFrame,
    gamma: float,
    run_id: int,
    policy_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract scalar summary statistics from a single simulation's mm_df.

    Parameters
    ----------
    mm_df : pd.DataFrame
        Market-maker telemetry from one simulation run.
    gamma : float
        The GLFT gamma used for this run.
    run_id : int
        The simulation run index.
    policy_label : str or None
        Human-readable policy label. Defaults to ``GLFT gamma=<gamma>``.

    Returns
    -------
    dict
        Summary with keys: gamma, run_id, final_inventory, final_total_pnl,
        final_cash_pnl, final_upnl, avg_abs_inventory, pnl_volatility.
    """
    if policy_label is None:
        policy_label = f"GLFT gamma={gamma:g}" if np.isfinite(gamma) else "unknown"

    if mm_df is None or mm_df.empty:
        return {
            "gamma": gamma,
            "run_id": run_id,
            "policy": policy_label,
            "final_inventory": np.nan,
            "final_total_pnl": np.nan,
            "final_cash_pnl": np.nan,
            "final_upnl": np.nan,
            "avg_abs_inventory": np.nan,
            "pnl_volatility": np.nan,
        }

    inv_series = mm_df.get("MM_Inventory", pd.Series(dtype=float)).to_numpy()
    total_pnl_series = mm_df.get("MM_TotalPnL", pd.Series(dtype=float)).to_numpy()
    cash_pnl_series = mm_df.get("MM_CashPnL", pd.Series(dtype=float)).to_numpy()
    upnl_series = mm_df.get("MM_UPnL", pd.Series(dtype=float)).to_numpy()

    final_inventory = float(inv_series[-1]) if inv_series.size > 0 else np.nan
    final_total_pnl = float(total_pnl_series[-1]) if total_pnl_series.size > 0 else np.nan
    final_cash_pnl = float(cash_pnl_series[-1]) if cash_pnl_series.size > 0 else np.nan
    final_upnl = float(upnl_series[-1]) if upnl_series.size > 0 else np.nan

    avg_abs_inventory = float(np.mean(np.abs(inv_series))) if inv_series.size > 0 else np.nan

    # PnL volatility: standard deviation of the TotalPnL level over time.
    # For increment-based volatility, use np.std(np.diff(total_pnl_series)).
    pnl_volatility = float(np.std(total_pnl_series)) if total_pnl_series.size > 0 else np.nan

    return {
        "gamma": gamma,
        "run_id": run_id,
        "policy": policy_label,
        "final_inventory": final_inventory,
        "final_total_pnl": final_total_pnl,
        "final_cash_pnl": final_cash_pnl,
        "final_upnl": final_upnl,
        "avg_abs_inventory": avg_abs_inventory,
        "pnl_volatility": pnl_volatility,
    }


def run_always_best_benchmark() -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Run the always-best benchmark for the same number of Monte Carlo episodes
    as each GLFT gamma value.
    """
    all_summaries: List[Dict[str, Any]] = []
    mm_store: List[pd.DataFrame] = []

    print(f"\n=== Running simulations for {ALWAYS_BEST_LABEL} ===")
    for run_id in range(N_SIMS_PER_GAMMA):
        # Use a distinct seed block from the GLFT gamma sweep while preserving
        # deterministic replay via (policy block, run_id).
        seed = make_run_seed(len(GAMMA_VALUES), run_id)

        print(
            f"  -> Simulation {run_id + 1}/{N_SIMS_PER_GAMMA} "
            f"(seed={seed}) ...",
            end="",
            flush=True,
        )

        _, _, mm_df = run_single_always_best_simulation(seed=seed)

        summary = summarize_mm_run(
            mm_df,
            gamma=np.nan,
            run_id=run_id,
            policy_label=ALWAYS_BEST_LABEL,
        )
        summary["seed"] = seed
        all_summaries.append(summary)
        mm_store.append(mm_df)

        print(" done.")

    return pd.DataFrame(all_summaries), mm_store


# ============================================================================
# 8) Plot Helpers
#
#    Each function creates a standalone figure and calls plt.show() at the
#    end to ensure the plot is rendered in all environments (Spyder, Jupyter,
#    standalone scripts).
# ============================================================================

def plot_mean_total_pnl_by_gamma(
    summary_df: pd.DataFrame,
    benchmark_summary_df: Optional[pd.DataFrame] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Plot mean final TotalPnL vs gamma with error bars (±1 std).

    This is the primary "efficiency frontier" view: gamma on the x-axis
    (log scale) and mean terminal PnL on the y-axis.
    """
    glft_df = summary_df[summary_df["gamma"].notna()].copy()
    grouped = glft_df.groupby("gamma")["final_total_pnl"].agg(["mean", "std"])

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=grouped.index,
        y=grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        label="GLFT",
    )

    if benchmark_summary_df is not None and not benchmark_summary_df.empty:
        bench = benchmark_summary_df["final_total_pnl"].dropna()
        if not bench.empty:
            plt.axhline(
                float(bench.mean()),
                color="#000000",
                linestyle="--",
                linewidth=1.5,
                label=f"{benchmark_label} mean",
            )

    plt.xscale("log")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Mean final TotalPnL")
    # plt.title("Mean final TotalPnL vs gamma (GLFT policy)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_boxplot_final_total_pnl(
    summary_df: pd.DataFrame,
    benchmark_summary_df: Optional[pd.DataFrame] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Boxplot of final TotalPnL distributions by gamma.

    Shows the median, quartiles, and outliers for each gamma value,
    revealing the full distribution shape beyond just mean ± std.
    """
    plt.figure(figsize=(8, 5))

    glft_df = summary_df[summary_df["gamma"].notna()].copy()
    gammas = sorted(glft_df["gamma"].unique())
    data = [glft_df.loc[glft_df["gamma"] == g, "final_total_pnl"].dropna() for g in gammas]
    labels = [f"{g:g}" for g in gammas]

    if benchmark_summary_df is not None and not benchmark_summary_df.empty:
        bench = benchmark_summary_df["final_total_pnl"].dropna()
        if not bench.empty:
            data.append(bench)
            labels.append(benchmark_label)

    plt.boxplot(data, labels=labels)
    plt.xlabel("gamma")
    plt.ylabel("Final TotalPnL")
    # plt.title("Final TotalPnL distribution by gamma (boxplots)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Dropped from paper per supervisor #14 ("redundancy: panel (c) is enough").
    # Function kept for in-notebook diagnostics; no longer mirrors to paper.
    # _save_paper_figure(plt, "boxplot_glft.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_final_metric_histograms(
    summary_df: pd.DataFrame,
    metric: str,
    title_prefix: str,
    benchmark_summary_df: Optional[pd.DataFrame] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Plot per-gamma histograms of a final metric (e.g. TotalPnL).

    Creates a subplot grid with one histogram per gamma value, allowing
    visual inspection of the full distribution shape.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary table with "gamma" and the specified metric columns.
    metric : str
        Column name to plot (e.g. "final_total_pnl").
    title_prefix : str
        Prefix for subplot titles (e.g. "TotalPnL").
    """
    glft_df = summary_df[summary_df["gamma"].notna()].copy()
    unique_gammas = sorted(glft_df["gamma"].unique())
    n_gammas = len(unique_gammas)
    benchmark_data = (
        benchmark_summary_df[metric].dropna()
        if benchmark_summary_df is not None and metric in benchmark_summary_df.columns
        else pd.Series(dtype=float)
    )

    n_cols = int(math.ceil(math.sqrt(n_gammas)))
    n_rows = int(math.ceil(n_gammas / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, gamma in enumerate(unique_gammas):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = glft_df.loc[glft_df["gamma"] == gamma, metric].dropna()
        if benchmark_data.empty:
            bins = 15
        else:
            combined = pd.concat([data, benchmark_data], ignore_index=True)
            if combined.nunique() <= 1:
                bins = 15
            else:
                bins = np.linspace(combined.min(), combined.max(), 16)

        ax.hist(data, bins=bins, alpha=0.75, label=f"GLFT gamma={gamma:g}")
        if not benchmark_data.empty:
            ax.hist(
                benchmark_data,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                color="#000000",
                label=benchmark_label,
            )
        ax.set_title(f"{title_prefix} – gamma = {gamma:g}")
        ax.set_xlabel("Terminal PnL")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=14)

    # Hide unused subplots (when n_gammas doesn't fill the grid)
    for idx in range(len(unique_gammas), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # fig.suptitle(f"Final {metric} distributions by gamma (GLFT policy)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Dropped from paper per supervisor #14 ("redundancy: panel (c) is enough").
    # Function kept for in-notebook diagnostics; no longer mirrors to paper.
    # _save_paper_figure(fig, "histograms_isolated_glft_vs_naive.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_mean_inventory_trajectory_by_gamma(
    mm_store: Dict[float, List[pd.DataFrame]],
    gammas: Optional[List[float]] = None,
    inventory_col: str = "MM_Inventory",
    drop_warmup: int = 0,
    title: str = "Average Inventory trajectory by gamma (GLFT policy)",
    benchmark_store: Optional[List[pd.DataFrame]] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Plot mean inventory trajectory ± 1σ band for each gamma.

    All runs within a gamma group are aligned to the shortest common
    length and stacked into a (n_runs, T) matrix.  The mean and std
    are computed column-wise.

    Parameters
    ----------
    mm_store : dict
        Maps gamma → list of mm_df DataFrames.
    gammas : list of float or None
        Gamma values to plot (None = all keys in mm_store).
    inventory_col : str
        Column name for inventory (default "MM_Inventory").
    drop_warmup : int
        Number of initial steps to discard from each run.
    title : str
        Plot title.
    benchmark_store : list of pd.DataFrame or None
        Optional benchmark trajectories to overlay with mean ± 1σ band.
    benchmark_label : str
        Label for the benchmark overlay.
    """
    if gammas is None:
        gammas = sorted(mm_store.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for g in gammas:
        series_list = []
        for df in mm_store.get(g, []):
            if df is None or df.empty or inventory_col not in df.columns:
                continue
            q = df[inventory_col].to_numpy()
            if drop_warmup > 0 and q.size > drop_warmup:
                q = q[drop_warmup:]
            series_list.append(q)

        if not series_list:
            continue

        min_len = min(len(q) for q in series_list)
        if min_len <= 1:
            continue

        # Stack all runs into a (n_runs, T) matrix for vectorized stats
        Q = np.vstack([q[:min_len] for q in series_list])
        mean_q = np.mean(Q, axis=0)
        std_q = np.std(Q, axis=0)

        x = np.arange(min_len)

        (line,) = ax.step(x, mean_q, where="post", label=f"gamma={g:g}")
        c = line.get_color()
        ax.fill_between(x, mean_q - std_q, mean_q + std_q, step="post", alpha=0.15, color=c)

    if benchmark_store:
        series_list = []
        for df in benchmark_store:
            if df is None or df.empty or inventory_col not in df.columns:
                continue
            q = df[inventory_col].to_numpy()
            if drop_warmup > 0 and q.size > drop_warmup:
                q = q[drop_warmup:]
            series_list.append(q)

        if series_list:
            min_len = min(len(q) for q in series_list)
            if min_len > 1:
                Q = np.vstack([q[:min_len] for q in series_list])
                mean_q = np.mean(Q, axis=0)
                std_q = np.std(Q, axis=0)
                x = np.arange(min_len)
                ax.step(
                    x,
                    mean_q,
                    where="post",
                    color="#000000",
                    linestyle="--",
                    linewidth=2.0,
                    label=benchmark_label,
                )
                ax.fill_between(
                    x,
                    mean_q - std_q,
                    mean_q + std_q,
                    step="post",
                    alpha=0.10,
                    color="#000000",
                )

    # ax.set_title(title)  # removed for paper export
    ax.set_xlabel("Simulation step")
    ax.set_ylabel(r"Mean inventory ($q_t$)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mean_total_pnl_trajectory_by_gamma(
    mm_store: Dict[float, List[pd.DataFrame]],
    gammas: Optional[List[float]] = None,
    pnl_col: str = "MM_TotalPnL",
    drop_warmup: int = 0,
    title: str = "Average TotalPnL trajectory by gamma (GLFT policy)",
    benchmark_store: Optional[List[pd.DataFrame]] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Plot mean TotalPnL trajectory ± 1σ band for each gamma.

    All runs within a gamma group are aligned to the shortest common
    length and stacked into a (n_runs, T) matrix.  The mean and std
    are computed column-wise.

    Parameters
    ----------
    mm_store : dict
        Maps gamma → list of mm_df DataFrames.
    gammas : list of float or None
        Gamma values to plot (None = all keys in mm_store).
    pnl_col : str
        Column name for total PnL trajectory (default "MM_TotalPnL").
    drop_warmup : int
        Number of initial steps to discard from each run.
    title : str
        Plot title.
    """
    if gammas is None:
        gammas = sorted(mm_store.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    # Explicit viridis sampling so successive gamma lines have clearly
    # distinct hues rather than shades of blue (supervisor feedback #15).
    viridis_colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(gammas), 1)))

    for idx, g in enumerate(gammas):
        series_list = []
        for df in mm_store.get(g, []):
            if df is None or df.empty or pnl_col not in df.columns:
                continue
            pnl = df[pnl_col].to_numpy()
            if drop_warmup > 0 and pnl.size > drop_warmup:
                pnl = pnl[drop_warmup:]
            series_list.append(pnl)

        if not series_list:
            continue

        min_len = min(len(pnl) for pnl in series_list)
        if min_len <= 1:
            continue

        # Stack all runs into a (n_runs, T) matrix for vectorized stats
        P = np.vstack([pnl[:min_len] for pnl in series_list])
        mean_p = np.mean(P, axis=0)
        std_p = np.std(P, axis=0)

        x = np.arange(min_len)
        c = viridis_colors[idx]
        ax.plot(x, mean_p, color=c, label=rf"$\gamma_{{\mathrm{{GLFT}}}} = {g:.0e}$")
        ax.fill_between(x, mean_p - std_p, mean_p + std_p, alpha=0.15, color=c)

    if benchmark_store:
        series_list = []
        for df in benchmark_store:
            if df is None or df.empty or pnl_col not in df.columns:
                continue
            pnl = df[pnl_col].to_numpy()
            if drop_warmup > 0 and pnl.size > drop_warmup:
                pnl = pnl[drop_warmup:]
            series_list.append(pnl)

        if series_list:
            min_len = min(len(pnl) for pnl in series_list)
            if min_len > 1:
                P = np.vstack([pnl[:min_len] for pnl in series_list])
                mean_p = np.mean(P, axis=0)
                std_p = np.std(P, axis=0)
                x = np.arange(min_len)
                ax.plot(
                    x,
                    mean_p,
                    color="#000000",
                    linestyle="--",
                    linewidth=2.0,
                    label=benchmark_label,
                )
                ax.fill_between(
                    x,
                    mean_p - std_p,
                    mean_p + std_p,
                    alpha=0.10,
                    color="#000000",
                )

    # ax.set_title(title)  # removed for paper export
    ax.set_xlabel("Simulation step", fontsize=15)
    ax.set_ylabel("Mean cumulative PnL", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=13)
    plt.tight_layout()
    _save_paper_figure(fig, "pnl_cumulated_pnl_glft_vs_naive.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_cartea_style_distributions(
    summary_df: pd.DataFrame,
    mm_store: Dict[float, List[pd.DataFrame]],
    benchmark_summary_df: Optional[pd.DataFrame] = None,
    benchmark_store: Optional[List[pd.DataFrame]] = None,
    benchmark_label: str = ALWAYS_BEST_LABEL,
):
    """
    Cartea-style dual-panel figure: PnL distribution + Lifetime Inventory distribution.

    Inspired by Cartea, Jaimungal & Penalva (2015), Chapter 10 figures.
    The left panel shows overlapping PnL histograms by gamma, and the
    right panel shows lifetime inventory frequency distributions.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary table with "gamma" and "final_total_pnl" columns.
    mm_store : dict
        Maps gamma → list of mm_df DataFrames.
    """
    glft_df = summary_df[summary_df["gamma"].notna()].copy()
    gammas = sorted(glft_df["gamma"].unique())

    pnl_arrays = [glft_df["final_total_pnl"].dropna().to_numpy()]
    if benchmark_summary_df is not None and not benchmark_summary_df.empty:
        pnl_arrays.append(benchmark_summary_df["final_total_pnl"].dropna().to_numpy())
    pnl_arrays = [x for x in pnl_arrays if x.size > 0]
    if not pnl_arrays:
        print("No PnL data to plot.")
        return
    pnl_all = np.concatenate(pnl_arrays)

    pnl_bins = np.linspace(np.min(pnl_all), np.max(pnl_all), 40)

    # Collect lifetime inventory from all time steps across all runs
    lifetime_inventory_by_gamma: Dict[float, np.ndarray] = {}
    inv_min = np.inf
    inv_max = -np.inf

    for g in gammas:
        series_list = []
        for df in mm_store.get(g, []):
            if df is not None and (not df.empty) and ("MM_Inventory" in df.columns):
                vals = df["MM_Inventory"].to_numpy()
                series_list.append(vals)
                inv_min = min(inv_min, np.min(vals))
                inv_max = max(inv_max, np.max(vals))

        if series_list:
            lifetime_inventory_by_gamma[g] = np.concatenate(series_list)

    benchmark_inventory = None
    if benchmark_store:
        bench_inv_series = []
        for df in benchmark_store:
            if df is not None and (not df.empty) and ("MM_Inventory" in df.columns):
                vals = df["MM_Inventory"].to_numpy()
                bench_inv_series.append(vals)
                inv_min = min(inv_min, np.min(vals))
                inv_max = max(inv_max, np.max(vals))
        if bench_inv_series:
            benchmark_inventory = np.concatenate(bench_inv_series)

    if not lifetime_inventory_by_gamma and benchmark_inventory is None:
        print("No inventory data to plot.")
        return

    inv_bins = np.arange(inv_min - 0.5, inv_max + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    ax_pnl, ax_inv = axes

    # Explicit viridis sampling per supervisor feedback #15: successive
    # gamma series must use clearly distinct hues, not shades of blue.
    viridis_colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(gammas), 1)))

    for idx, g in enumerate(gammas):
        c = viridis_colors[idx]
        data_pnl = glft_df.loc[glft_df["gamma"] == g, "final_total_pnl"].dropna()

        ax_pnl.hist(
            data_pnl,
            bins=pnl_bins,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=c,
            facecolor=c,
            label=rf"$\gamma_{{\mathrm{{GLFT}}}} = {g:.0e}$",
        )

    if benchmark_summary_df is not None and not benchmark_summary_df.empty:
        bench_pnl = benchmark_summary_df["final_total_pnl"].dropna()
        if not bench_pnl.empty:
            ax_pnl.hist(
                bench_pnl,
                bins=pnl_bins,
                histtype="step",
                linewidth=2.0,
                color="#000000",
                label=benchmark_label,
            )

    ax_pnl.set_xlabel("Profit and Loss", fontsize=15)
    ax_pnl.set_ylabel("Frequency", fontsize=15)
    # ax_pnl.set_title("Profit and Loss")
    ax_pnl.grid(True, linestyle="--", alpha=0.4)
    ax_pnl.tick_params(labelsize=13)
    ax_pnl.legend(loc="upper left", fontsize=13)

    for idx, g in enumerate(gammas):
        c = viridis_colors[idx]
        data_inv = lifetime_inventory_by_gamma[g]

        ax_inv.hist(
            data_inv,
            bins=inv_bins,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=c,
            facecolor=c,
            label=rf"$\gamma_{{\mathrm{{GLFT}}}} = {g:.0e}$",
        )

    if benchmark_inventory is not None:
        ax_inv.hist(
            benchmark_inventory,
            bins=inv_bins,
            histtype="step",
            linewidth=2.0,
            color="#000000",
            label=benchmark_label,
        )

    ax_inv.set_xlabel(r"Lifetime Inventory ($q_t$)", fontsize=15)
    ax_inv.set_ylabel("Frequency", fontsize=15)
    # ax_inv.set_title("Lifetime Inventory")
    ax_inv.grid(True, linestyle="--", alpha=0.4)
    ax_inv.tick_params(labelsize=13)
    # ax_inv.legend()  # dropped: duplicate of ax_pnl legend (same series)

    plt.tight_layout()
    _save_paper_figure(fig, "glft_naive_pnl_lifetime_hist.png", dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================================
# 9) Main Study Runner
#
#    Iterates over the gamma grid, runs N_SIMS_PER_GAMMA simulations per
#    gamma, collects summaries, and produces all diagnostic plots.
# ============================================================================

def _plot_glft_gamma_study_results(
    summary_with_benchmark: pd.DataFrame,
    mm_time_series_store: Dict[float, List[pd.DataFrame]],
    always_best_store: List[pd.DataFrame],
) -> None:
    """
    Render every diagnostic figure for the GLFT gamma-sweep study.

    Factored out of ``run_glft_gamma_study`` so it can be reused by the
    LOAD_FROM_CACHE path in ``__main__`` (regenerate the 4 PNGs without
    re-running the 1000-sim Monte Carlo).
    """
    # Split the combined summary back into GLFT and always-best halves so the
    # plotting helpers receive the same inputs they would in the live-sim path.
    summary_df = summary_with_benchmark[summary_with_benchmark["gamma"].notna()].copy()
    always_best_summary_df = summary_with_benchmark[summary_with_benchmark["gamma"].isna()].copy()

    # --- Aggregated plots ---
    plot_mean_total_pnl_by_gamma(
        summary_df,
        benchmark_summary_df=always_best_summary_df,
    )
    plot_boxplot_final_total_pnl(
        summary_df,
        benchmark_summary_df=always_best_summary_df,
    )
    plot_final_metric_histograms(
        summary_df,
        metric="final_total_pnl",
        title_prefix="TotalPnL",
        benchmark_summary_df=always_best_summary_df,
    )

    # TotalPnL trajectory plot
    plot_mean_total_pnl_trajectory_by_gamma(
        mm_time_series_store,
        gammas=GAMMA_VALUES,
        drop_warmup=DROP_WARMUP_PLOT,
        benchmark_store=always_best_store,
    )

    # Inventory trajectory plot
    plot_mean_inventory_trajectory_by_gamma(
        mm_time_series_store,
        gammas=GAMMA_VALUES,
        drop_warmup=DROP_WARMUP_PLOT,
        benchmark_store=always_best_store,
    )

    # Cartea-style PnL + inventory distributions
    plot_cartea_style_distributions(
        summary_df,
        mm_time_series_store,
        benchmark_summary_df=always_best_summary_df,
        benchmark_store=always_best_store,
    )


def run_glft_gamma_study(
    A_param: float,
    kappa: float,
    sigma: float,
) -> Tuple[pd.DataFrame, Dict[float, List[pd.DataFrame]]]:
    """
    Run the full GLFT gamma-sweep Monte Carlo study.

    Parameters
    ----------
    A_param : float
        Calibrated execution intensity at zero distance.
    kappa : float
        Calibrated exponential decay rate.
    sigma : float
        Calibrated mid-price volatility.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (policy, run_id) with scalar summary statistics.  When
        ``PLOT_ALWAYS_BEST_BENCHMARK`` is false, this is just the GLFT
        gamma sweep.
    mm_time_series_store : dict
        Maps gamma → list of mm_df DataFrames (full time series).
    """
    all_summaries: List[Dict[str, Any]] = []
    mm_time_series_store: Dict[float, List[pd.DataFrame]] = {g: [] for g in GAMMA_VALUES}

    for gamma_idx, gamma in enumerate(GAMMA_VALUES):
        print(f"\n=== Running simulations for gamma = {gamma} ===")
        for run_id in range(N_SIMS_PER_GAMMA):
            seed = make_run_seed(gamma_idx, run_id)

            print(
                f"  -> Simulation {run_id + 1}/{N_SIMS_PER_GAMMA} "
                f"(seed={seed}) ...",
                end="",
                flush=True,
            )

            _, _, mm_df = run_single_simulation(
                gamma, A_param, kappa, sigma, seed=seed,
            )

            summary = summarize_mm_run(mm_df, gamma, run_id)
            summary["seed"] = seed  # store seed for debugging / replay
            all_summaries.append(summary)

            mm_time_series_store[gamma].append(mm_df)

            print(" done.")

    summary_df = pd.DataFrame(all_summaries)

    always_best_summary_df = pd.DataFrame()
    always_best_store: List[pd.DataFrame] = []
    if PLOT_ALWAYS_BEST_BENCHMARK:
        always_best_summary_df, always_best_store = run_always_best_benchmark()

    # Save summary to disk for later analysis
    summary_with_benchmark = (
        pd.concat([summary_df, always_best_summary_df], ignore_index=True)
        if not always_best_summary_df.empty
        else summary_df
    )
    summary_with_benchmark.to_pickle("glft_gamma_study_summary.pkl")
    summary_with_benchmark.to_csv("glft_gamma_study_summary.csv", index=False)

    # Also persist the per-run trajectory stores so that LOAD_FROM_CACHE
    # mode can regenerate the trajectory / Cartea-style plots without
    # re-running the Monte Carlo.  These are intentionally separate files
    # so the small summary pkl/csv stay quick to load on their own.
    pd.to_pickle(mm_time_series_store, "glft_gamma_study_mm_store.pkl")
    pd.to_pickle(always_best_store, "glft_gamma_study_always_best_store.pkl")

    _plot_glft_gamma_study_results(
        summary_with_benchmark,
        mm_time_series_store,
        always_best_store,
    )

    return summary_with_benchmark, mm_time_series_store


# ============================================================================
# 10) Entry Point
#
#    Runs the calibration phase once, then launches the gamma-sweep study.
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Cache controls
    # ------------------------------------------------------------------
    # Set LOAD_FROM_CACHE = True to skip the (very expensive) 1000-sim
    # Monte Carlo and regenerate the 4 paper PNGs from the cached
    # summary + per-run trajectory stores produced by a previous run.
    #
    # We need three companion pickles to fully reproduce every figure:
    #   - _CACHE_PKL              (summary_with_benchmark)
    #   - _CACHE_MM_STORE_PKL     (per-gamma list of mm_df)
    #   - _CACHE_BENCH_STORE_PKL  (always-best benchmark list of mm_df)
    # If ANY of them is missing or fails to load, we fall back to running
    # the simulation so we never produce incorrect plots.
    # ------------------------------------------------------------------
    LOAD_FROM_CACHE = True   # set False to re-run the Monte Carlo
    _CACHE_PKL = "glft_gamma_study_summary.pkl"
    _CACHE_MM_STORE_PKL = "glft_gamma_study_mm_store.pkl"
    _CACHE_BENCH_STORE_PKL = "glft_gamma_study_always_best_store.pkl"

    import os as _os

    _cache_files_present = (
        _os.path.exists(_CACHE_PKL)
        and _os.path.exists(_CACHE_MM_STORE_PKL)
        and _os.path.exists(_CACHE_BENCH_STORE_PKL)
    )

    _loaded_from_cache = False
    if LOAD_FROM_CACHE and _cache_files_present:
        try:
            print(f"[cache] Loading summary from {_CACHE_PKL}")
            summary_with_benchmark = pd.read_pickle(_CACHE_PKL)
            print(f"[cache] Loading mm_store from {_CACHE_MM_STORE_PKL}")
            mm_store = pd.read_pickle(_CACHE_MM_STORE_PKL)
            print(f"[cache] Loading always_best_store from {_CACHE_BENCH_STORE_PKL}")
            always_best_store = pd.read_pickle(_CACHE_BENCH_STORE_PKL)
            _loaded_from_cache = True
            print("[cache] Skipping Monte Carlo - regenerating plots from cache.")
            _plot_glft_gamma_study_results(
                summary_with_benchmark,
                mm_store,
                always_best_store,
            )
        except Exception as _cache_err:
            print(f"[cache] Failed to load cache ({_cache_err!r}); "
                  "falling back to full Monte Carlo simulation.")
            _loaded_from_cache = False

    if not _loaded_from_cache:
        if LOAD_FROM_CACHE and not _cache_files_present:
            print("[cache] LOAD_FROM_CACHE=True but one or more cache files are "
                  "missing; running full Monte Carlo simulation instead.")

        # Step 1: Calibrate GLFT parameters (A, kappa, sigma)
        A, KAPPA, SIGMA = run_calibration(plot=True)

        # Step 2: Run the Monte Carlo gamma study (also writes cache files)
        summary_with_benchmark, mm_store = run_glft_gamma_study(A, KAPPA, SIGMA)
