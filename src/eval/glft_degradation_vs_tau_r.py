"""
GLFT @ gamma=1e-6: Terminal-PnL DEGRADATION vs tau_r under regime-switching.

This standalone script:

  1. Calibrates GLFT (A, kappa, sigma) using the same pipeline as
     glft_fair_vs_regime_hist.py / GLFT_studies.py (loads from pickle cache
     ``glft_calib_A_kappa_sigma.pkl`` when available).
  2. Builds the GLFT policy at gamma = 1e-6 via ``glft_policy_factory`` (same
     factory kwargs as GLFT_studies.make_glft_policy).
  3. Runs N_SIMS simulations PER tau_r in TAU_R_VALUES = [15, 30, 60, 120, 240]
     using a regime-switching schedule with p_buy ~ U[0.2, 0.8] and exponential
     regime durations of mean tau_r MO events.
  4. Also runs N_SIMS no-regime baseline sims with constant p_buy = 0.5.
  5. Plots Terminal PnL mean +/- SEM vs tau_r in the notebook degradation style,
     with a dotted horizontal line for the no-regime baseline.

Output figure is saved to:
  * /Users/felipemoret/Desktop/MM_LOB_SIM/glft_degradation_vs_tau_regime.png
  * /Users/felipemoret/Desktop/extended_first_abstract_RLMM/glft_degradation_vs_tau_regime.png

Run as::

    python glft_degradation_vs_tau_r.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from GLFT_policy_factory import glft_policy_factory
from MM_LOB_SIM import simulate_LOB_with_MM
from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)
from regime_switching_stress_test_2 import make_regime_schedule


# ============================================================================
# 1) Global Configuration (mirrors glft_fair_vs_regime_hist.py)
# ============================================================================

# --- Santa Fe LOB model rates (AMZN-style calibration) ---
LAM = 0.06
MU = 0.10
DELTA = 0.02

# --- LOB geometry ---
NUMBER_TICK_LEVELS = 50
N_PRIORITY_RANKS = 100
HALF_TICK = 0.5

# --- Simulation duration ---
N_STEPS = 100_000
N_STEPS_TO_EQUIL = 1_000

# --- Throttling intervals ---
time_interval_throttle = 1
time_interval_calib = 0.5

# --- Calibration seed ---
SEED_CALIB = 1234

# --- MM agent constraints ---
INV_LIMIT = 8
ROUND_TO_INT = True

# --- GLFT MDP / TOB toggles ---
USE_TOB_UPDATE = False
N_TOB_MOVES = 1
USE_MDP_GLFT = True

# --- This study ---
GAMMA_GLFT = 1e-6
N_SIMS = 1000
SEED_BASE = 42

# --- Sweep over tau_r ---
TAU_R_VALUES: List[int] = [15, 30, 60, 120, 240]
P_LO = 0.2
P_HI = 0.8
REGIME_DIST = "exponential"

# --- Output paths ---
LOCAL_FIG = Path("/Users/felipemoret/Desktop/MM_LOB_SIM/glft_degradation_vs_tau_regime.png")
PAPER_FIG = Path(
    "/Users/felipemoret/Desktop/extended_first_abstract_RLMM/glft_degradation_vs_tau_regime.png"
)

# Cache for the calibrated (A, kappa, sigma).
CALIB_CACHE = Path("/Users/felipemoret/Desktop/MM_LOB_SIM/glft_calib_A_kappa_sigma.pkl")

# Cache for the PnL arrays so repeated runs skip the expensive sim sweep.
CACHE_DIR = Path("/Users/felipemoret/Desktop/MM_LOB_SIM/notebook_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FORCE_REFRESH = False  # set True to bypass the PnL cache and rerun sims


# ============================================================================
# 2) Calibration helper
# ============================================================================

def run_calibration() -> Tuple[float, float, float]:
    """Run the censored-waiting-times calibration and return (A, kappa, sigma)."""
    res = fit_execution_intensity_censored_waiting_times(
        aggregation_mode="time",
        time_interval=time_interval_calib,
        max_levels=5,
        half_tick=HALF_TICK,
        side_mode="buy",
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=500_000,
        n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
        split_sweeps=False,
        random_seed=SEED_CALIB,
        plot=False,
    )

    A, KAPPA, _ = fit_A_kappa_loglinear(
        res["delta_grid"],
        res["lambda_hat"],
        weights=res["denom_sum_tau"],
    )
    SIGMA = 0.3  # matches GLFT_studies (from signature plot)

    print(f"[CALIB] A     = {A:.6g}")
    print(f"[CALIB] kappa = {KAPPA:.6g}")
    print(f"[CALIB] sigma = {SIGMA:.6g}")

    return float(A), float(KAPPA), float(SIGMA)


def load_or_calibrate() -> Tuple[float, float, float]:
    """Load (A, kappa, sigma) from CALIB_CACHE if present, else calibrate."""
    if CALIB_CACHE.exists():
        try:
            with open(CALIB_CACHE, "rb") as fh:
                A, KAPPA, SIGMA = pickle.load(fh)
            print(f"[CALIB] loaded from cache: A={A:.6g}, kappa={KAPPA:.6g}, sigma={SIGMA:.6g}")
            return float(A), float(KAPPA), float(SIGMA)
        except Exception as exc:  # noqa: BLE001
            print(f"[CALIB] cache read failed ({exc!r}); recalibrating.")

    A, KAPPA, SIGMA = run_calibration()
    try:
        with open(CALIB_CACHE, "wb") as fh:
            pickle.dump((A, KAPPA, SIGMA), fh)
        print(f"[CALIB] saved cache -> {CALIB_CACHE}")
    except Exception as exc:  # noqa: BLE001
        print(f"[CALIB] cache write failed: {exc!r}")
    return A, KAPPA, SIGMA


# ============================================================================
# 3) GLFT policy builder
# ============================================================================

def make_glft_policy(gamma: float, A_param: float, kappa: float, sigma: float):
    """Build the GLFT policy callable with the same kwargs as GLFT_studies."""
    return glft_policy_factory(
        gamma=gamma,
        kappa=kappa,
        A=A_param,
        sigma=sigma,
        inv_limit=INV_LIMIT,
        round_to_int=ROUND_TO_INT,
        delta=1.0,
        epsilon=None,
        use_tob_update=USE_TOB_UPDATE,
        n_tob_moves=N_TOB_MOVES,
        use_event_update=False,
        n_events=1000,
        use_time_update=True,
        min_time_interval=time_interval_throttle,
        use_mdp=USE_MDP_GLFT,
    )


# ============================================================================
# 4) Simulation helpers
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _terminal_total_pnl(mm_df) -> float:
    if mm_df is None or len(mm_df) == 0:
        return float("nan")
    series = mm_df.get("MM_TotalPnL")
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.to_numpy()[-1])


def run_one(
    A: float,
    kappa: float,
    sigma: float,
    seed: int,
    buy_mo_prob,
) -> float:
    """Run a single simulation with the given seed and buy_mo_prob spec."""
    seed_everything(seed)
    policy = make_glft_policy(GAMMA_GLFT, A, kappa, sigma)

    _, _, mm_df = simulate_LOB_with_MM(
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
        buy_mo_prob=buy_mo_prob,
    )
    return _terminal_total_pnl(mm_df)


# ============================================================================
# 5) Main study
# ============================================================================

def main() -> None:
    A, KAPPA, SIGMA = load_or_calibrate()

    print(
        f"\nRunning {N_SIMS} sims per tau_r ({len(TAU_R_VALUES)} values) + "
        f"{N_SIMS} baseline sims, gamma={GAMMA_GLFT:g}, N_STEPS={N_STEPS}, "
        f"seed_base={SEED_BASE}.\n"
    )

    # ---- Cache lookup ----------------------------------------------------
    cache_params = {
        "N_SIMS": N_SIMS,
        "N_STEPS": N_STEPS,
        "A": float(A),
        "kappa": float(KAPPA),
        "sigma": float(SIGMA),
        "GAMMA_GLFT": GAMMA_GLFT,
        "TAU_R_VALUES": list(TAU_R_VALUES),
        "P_LO": P_LO,
        "P_HI": P_HI,
        "SEED_BASE": SEED_BASE,
    }
    h = hashlib.md5(
        json.dumps(cache_params, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    cache_file = CACHE_DIR / f"glft_degradation_vs_tau_{h}.pkl"

    if cache_file.exists() and not FORCE_REFRESH:
        print(f"[CACHE HIT] {cache_file.name}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        baseline_pnls = np.asarray(cached["baseline_pnls"], dtype=float)
        regime_pnls: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=float)
            for k, v in cached["regime_pnls"].items()
        }
    else:
        print("[CACHE MISS] running sims...")

        # ---- Baseline (no-regime, p_buy = 0.5) ---------------------------
        baseline_pnls = np.full(N_SIMS, np.nan)
        for i in tqdm(range(N_SIMS), desc="BASELINE (p_buy=0.5)"):
            seed_i = SEED_BASE + i
            baseline_pnls[i] = run_one(A, KAPPA, SIGMA, seed_i, buy_mo_prob=0.5)

        # ---- Regime sweep over tau_r -------------------------------------
        regime_pnls = {int(tau): np.full(N_SIMS, np.nan) for tau in TAU_R_VALUES}
        for tau_idx, tau_r in enumerate(TAU_R_VALUES):
            desc = f"REGIME tau_r={tau_r} (p in [{P_LO},{P_HI}])"
            for i in tqdm(range(N_SIMS), desc=desc):
                seed_i = SEED_BASE + i + tau_idx * 10_000
                sched_i = make_regime_schedule(
                    seed=seed_i + 999_999,
                    n_mo_events=N_STEPS,
                    p_lo=P_LO,
                    p_hi=P_HI,
                    distribution=REGIME_DIST,
                    exp_rate=1.0 / float(tau_r),
                )
                regime_pnls[int(tau_r)][i] = run_one(
                    A, KAPPA, SIGMA, seed_i, buy_mo_prob=sched_i
                )

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "baseline_pnls": baseline_pnls,
                    "regime_pnls": regime_pnls,
                },
                f,
            )
        print(f"[CACHE SAVED] {cache_file.name}")

    # ---- Diagnostics -----------------------------------------------------
    base_finite = baseline_pnls[np.isfinite(baseline_pnls)]
    base_mean = float(np.mean(base_finite)) if base_finite.size else float("nan")
    base_std = float(np.std(base_finite, ddof=1)) if base_finite.size > 1 else float("nan")
    base_sem = (
        float(np.std(base_finite, ddof=1) / np.sqrt(base_finite.size))
        if base_finite.size > 1
        else float("nan")
    )
    print(
        f"\nBASELINE (no regime, p_buy=0.5): "
        f"mean={base_mean:.4g}, std={base_std:.4g}, "
        f"sem={base_sem:.4g}, n={base_finite.size}"
    )

    means: List[float] = []
    sems: List[float] = []
    stds: List[float] = []
    for tau_r in TAU_R_VALUES:
        arr = regime_pnls[int(tau_r)]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            means.append(float("nan"))
            sems.append(float("nan"))
            stds.append(float("nan"))
            print(f"  tau_r={tau_r:>4d}: no finite samples")
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        se = float(s / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        means.append(m)
        sems.append(se)
        stds.append(s)
        print(
            f"  tau_r={tau_r:>4d}: mean={m:.4g}, std={s:.4g}, "
            f"sem={se:.4g}, n={arr.size}"
        )

    # ---- Plot (notebook degradation style) -------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    color = "#E69F00"
    ax.errorbar(
        TAU_R_VALUES,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        color=color,
        linewidth=1.6,
        markersize=7,
        label=r"GLFT $\gamma=10^{-6}$ regime",
    )

    # No-regime baseline horizontal line
    ax.axhline(
        base_mean,
        color=color,
        linestyle=":",
        linewidth=1.6,
        label=r"GLFT $\gamma=10^{-6}$ no regime",
    )

    ax.set_xticks(TAU_R_VALUES)
    ax.set_xticklabels([rf"$\tau_r={t}$" for t in TAU_R_VALUES])
    ax.set_xlabel(r"$\tau_r$ [MO events]")
    ax.set_ylabel("Terminal PnL mean +/- SEM")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=14)
    fig.tight_layout()

    for target in (LOCAL_FIG, PAPER_FIG):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=150, bbox_inches="tight")
            print(f"[FIG] saved -> {target}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FIG] save failed for {target}: {exc!r}")

    plt.close(fig)


if __name__ == "__main__":
    main()
