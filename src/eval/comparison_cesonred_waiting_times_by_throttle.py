#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare censored-waiting-time calibration across throttle intervals.

This script runs the calibration loop multiple times for each chosen throttle
interval and plots one panel: linearized log-intensity.

log(lambda_hat(delta) / A_hat) with mean +/- std + fitted -kappa*delta

This mirrors the linearized fit view from the reference figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# === Paper figure style: matches GLFT_studies.py so this figure renders
# with the same label/legend sizing as the other paper plots when
# included at \linewidth.  No embedded titles -- the LaTeX caption
# describes the figure. ===
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
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)

import hashlib
import pathlib

# ---------------------------------------------------------------------------
# Per-rep resume cache
# ---------------------------------------------------------------------------
# Each (interval, seed) repetition is an expensive LOB simulation (long
# throttle windows run up to N_STEPS_CAP events).  We cache each completed
# rep's raw (delta_grid, lam, den) arrays to disk, keyed by a signature of
# every parameter that affects the result.  On a re-run the cached reps load
# instantly and only missing reps are simulated, so pausing/resuming this
# script is cheap.  Writes are atomic (tmp + rename) so a process killed
# mid-rep never leaves a corrupt cache entry (that rep is simply recomputed).
CENSORED_CACHE_ENABLED = True
CENSORED_CACHE_DIR = pathlib.Path(__file__).resolve().parent / "censored_waiting_cache"


def _rep_cache_path(mode: str, interval_value: float, seed: int):
    """Deterministic cache path for one (interval, seed) rep, or None if
    caching is disabled.  The filename embeds a signature hash of every
    simulation parameter, so changing any of them invalidates the cache."""
    if not CENSORED_CACHE_ENABLED:
        return None
    sig = {
        "mode": str(mode).lower().strip(),
        "interval": float(interval_value),
        "seed": int(seed),
        "n_steps": int(_n_steps_for_interval(mode, interval_value)),
        "n_equil": int(N_STEPS_TO_EQUIL),
        "lam": float(LAM), "mu": float(MU), "delta": float(DELTA),
        "max_levels": int(MAX_LEVELS), "half_tick": float(HALF_TICK),
        "side_mode": str(SIDE_MODE), "burn_in": int(BURN_IN_WINDOWS),
        "n_tick_levels": int(NUMBER_TICK_LEVELS), "n_ranks": int(N_PRIORITY_RANKS),
    }
    h = hashlib.sha1(repr(sorted(sig.items())).encode()).hexdigest()[:10]
    name = f"rep_{sig['mode']}_{interval_value:g}_seed{int(seed)}_{h}.npz"
    return CENSORED_CACHE_DIR / name


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
# User configuration
# ============================================================================

# Throttle clock mode: "time", "fixed_events", "tob", or "event"
THROTTLE_MODE = "time"

# Interval list to compare (interpretation depends on THROTTLE_MODE):
# - time         -> seconds
# - fixed_events -> number of events
# - tob          -> number of TOB moves
# - event        -> ignored (kept for API symmetry)
INTERVAL_THROTTLE_LIST = [0.5, 1, 10, 60, 300]

# Repetitions per interval (for mean/SEM error bars)
N_REPEATS = 20
SEED_BASE = 123456

# If True, repeat r uses the same random seed across all interval values.
# This "common random numbers" setup reduces between-interval path noise and
# makes left-panel comparisons (lambda_hat) closer to an apples-to-apples test.
# If False, each interval gets an independent seed family.
PAIR_SEEDS_ACROSS_INTERVALS = True

# Delta grid control
MAX_LEVELS = 5
HALF_TICK = 0.5
SIDE_MODE = "buy"
BURN_IN_WINDOWS = 0

# Santa-Fe simulator parameters
LAM = 0.06
MU = 0.10
DELTA = 0.02
NUMBER_TICK_LEVELS = 50
N_PRIORITY_RANKS = 100

# Simulated horizon (events) per calibration run.  Longer throttle windows ΔT
# yield FEWER calibration windows for a fixed horizon (#windows ≈ T_total/ΔT),
# so their λ̂(δ) is noisier — this is why the longest-ΔT curve shows the largest
# error bars near the touch.  We therefore scale the horizon with ΔT (more
# events for longer windows), capped to bound runtime.  Runtime grows roughly
# linearly with N_STEPS_CAP and N_REPEATS; lower them if calibration is slow.
N_STEPS_BASE = 200_000       # events at the reference window REF_INTERVAL
REF_INTERVAL = 1.0           # seconds: ΔT <= this keeps the base horizon
N_STEPS_CAP = 2_000_000      # hard cap on events per run (runtime guard)
N_STEPS_TO_EQUIL = 10_000


# Plot/output
PLOT_TITLE = "Censored Waiting Times Comparison by Throttle Interval"
SAVE_FIG_PATH = "censored_waiting_different_throttles.png"
USE_TQDM = True


@dataclass
class IntervalResult:
    interval_value: float
    delta_grid: np.ndarray
    lambda_mean: np.ndarray
    lambda_std: np.ndarray
    lambda_fit: np.ndarray
    ylog_mean: np.ndarray
    ylog_std: np.ndarray
    ylog_fit: np.ndarray
    A_hat: float
    kappa_hat: float
    A_std: float
    kappa_std: float


def _log(message: str) -> None:
    """Write logs without breaking tqdm rendering."""
    if USE_TQDM and (tqdm is not None):
        tqdm.write(str(message))
    else:
        print(message)


def _interval_kwargs(mode: str, interval_value: float) -> Dict[str, float]:
    """Build calibration kwargs for a specific throttle mode/interval."""
    mode = mode.lower().strip()

    # Keep defaults for non-selected clocks.
    kwargs: Dict[str, float] = {
        "n_events_interval": 100,
        "time_interval": 1.0,
        "n_tob_moves": 10,
    }

    if mode == "time":
        kwargs["time_interval"] = float(interval_value)
    elif mode == "fixed_events":
        kwargs["n_events_interval"] = int(interval_value)
    elif mode == "tob":
        kwargs["n_tob_moves"] = int(interval_value)
    elif mode == "event":
        # event mode ignores interval_value by design.
        pass
    else:
        raise ValueError(f"Unknown THROTTLE_MODE='{mode}'")

    return kwargs


def _n_steps_for_interval(mode: str, interval_value: float) -> int:
    """Scale the simulated horizon with the throttle window so longer windows
    still accumulate enough calibration windows (#windows ~ T_total/DeltaT)."""
    if mode.lower().strip() == "time":
        scale = max(1.0, float(interval_value) / REF_INTERVAL)
        return int(min(N_STEPS_BASE * scale, N_STEPS_CAP))
    return N_STEPS_BASE


def _run_one_calibration(
    mode: str,
    interval_value: float,
    random_seed: int,
    progress_desc: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run one censored-waiting-time calibration for a given interval."""
    interval_kwargs = _interval_kwargs(mode, interval_value)

    return fit_execution_intensity_censored_waiting_times(
        data_mode="simulation",
        aggregation_mode=mode,
        max_levels=MAX_LEVELS,
        half_tick=HALF_TICK,
        side_mode=SIDE_MODE,
        burn_in_windows=BURN_IN_WINDOWS,
        lam=LAM,
        mu=MU,
        delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=_n_steps_for_interval(mode, interval_value),
        n_steps_to_equilibrium=N_STEPS_TO_EQUIL,
        random_seed=int(random_seed),
        exclude_self_from_state=True,
        split_sweeps=False,
        plot=False,
        use_tqdm=bool(USE_TQDM),
        tqdm_desc=progress_desc,
        **interval_kwargs,
    )


def _label_for_interval(mode: str, interval_value: float) -> str:
    mode = mode.lower().strip()
    if mode == "time":
        return rf"$\Delta T_{{\mathrm{{calib}}}} = {interval_value:g}$"
    if mode == "fixed_events":
        return rf"$N_{{ev}} = {int(interval_value)}$"
    if mode == "tob":
        return rf"$N_{{TOB}} = {int(interval_value)}$"
    return "event mode"


def _aggregate_interval(
    mode: str,
    interval_value: float,
    n_repeats: int,
    seed_base: int,
    idx_interval: int,
) -> IntervalResult:
    """Run repeated calibrations for one interval and aggregate statistics."""
    lambda_runs: List[np.ndarray] = []
    denom_runs: List[np.ndarray] = []
    delta_grid = None

    A_runs: List[float] = []
    kappa_runs: List[float] = []

    for r in range(n_repeats):
        # Seed policy:
        # - paired mode    : same seed family across all intervals
        # - independent mode: different seed family per interval
        if PAIR_SEEDS_ACROSS_INTERVALS:
            seed = int(seed_base + r)
        else:
            seed = int(seed_base + 10_000 * idx_interval + r)
        progress_desc = f"thr={mode}:{interval_value:g} rep={r + 1}/{n_repeats}"
        _cache_path = _rep_cache_path(mode, interval_value, seed)
        if _cache_path is not None and _cache_path.exists():
            _cz = np.load(_cache_path)
            _dg = np.asarray(_cz["delta_grid"], dtype=float)
            lam = np.asarray(_cz["lam"], dtype=float)
            den = np.asarray(_cz["den"], dtype=float)
            _cz.close()
            _log(f"[CACHE] hit  {progress_desc} -> {_cache_path.name}")
        else:
            res = _run_one_calibration(
                mode=mode,
                interval_value=interval_value,
                random_seed=seed,
                progress_desc=progress_desc,
            )
            _dg = np.asarray(res["delta_grid"], dtype=float)
            lam = np.asarray(res["lambda_hat"], dtype=float)
            den = np.asarray(res["denom_sum_tau"], dtype=float)
            if _cache_path is not None:
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                # tmp name MUST end in .npz, else np.savez appends .npz and the
                # rename target would not exist.
                _tmp = _cache_path.parent / (_cache_path.stem + ".tmp.npz")
                np.savez(_tmp, delta_grid=_dg, lam=lam, den=den)
                _tmp.replace(_cache_path)  # atomic: no corrupt cache on kill
                _log(f"[CACHE] save {progress_desc} -> {_cache_path.name}")

        if delta_grid is None:
            delta_grid = _dg

        lambda_runs.append(lam)
        denom_runs.append(den)

        A_r, kappa_r, _ = fit_A_kappa_loglinear(delta_grid, lam, weights=den)
        A_runs.append(float(A_r))
        kappa_runs.append(float(kappa_r))

    lam_mat = np.vstack(lambda_runs)
    den_mat = np.vstack(denom_runs)

    lam_mean = np.mean(lam_mat, axis=0)
    lam_std = np.std(lam_mat, axis=0, ddof=0)
    den_mean = np.mean(den_mat, axis=0)

    A_hat, kappa_hat, _ = fit_A_kappa_loglinear(delta_grid, lam_mean, weights=None)
    lam_fit = float(A_hat) * np.exp(-float(kappa_hat) * delta_grid)

    # Compute ylog as log(lambda_hat_r / A_hat_r) per repeat, which
    # equals -kappa*delta under the exponential model.  Each repeat
    # normalizes by its own fitted A_r, matching Garaglio (2015) Fig. 5.2(b).
    eps = 1e-15
    ylog_runs = []
    for lam_r, A_r in zip(lambda_runs, A_runs):
        ylog_runs.append(np.log(np.maximum(lam_r, eps) / max(A_r, eps)))
    ylog_mat = np.vstack(ylog_runs)

    ylog_mean = np.mean(ylog_mat, axis=0)
    ylog_std = np.std(ylog_mat, axis=0, ddof=0)
    ylog_fit = -float(kappa_hat) * delta_grid

    return IntervalResult(
        interval_value=float(interval_value),
        delta_grid=delta_grid,
        lambda_mean=lam_mean,
        lambda_std=lam_std,
        lambda_fit=lam_fit,
        ylog_mean=ylog_mean,
        ylog_std=ylog_std,
        ylog_fit=ylog_fit,
        A_hat=float(A_hat),
        kappa_hat=float(kappa_hat),
        A_std=float(np.std(np.asarray(A_runs), ddof=0)),
        kappa_std=float(np.std(np.asarray(kappa_runs), ddof=0)),
    )


def run_comparison_and_plot(
    mode: str = THROTTLE_MODE,
    interval_list: List[float] = INTERVAL_THROTTLE_LIST,
    n_repeats: int = N_REPEATS,
    seed_base: int = SEED_BASE,
) -> List[IntervalResult]:
    """Main entry point: run loop and plot both panels."""
    mode = mode.lower().strip()
    results: List[IntervalResult] = []
    seed_mode = "paired" if PAIR_SEEDS_ACROSS_INTERVALS else "independent"

    if USE_TQDM and (tqdm is not None):
        _interval_iter = tqdm(
            enumerate(interval_list),
            total=len(interval_list),
            desc=f"intervals[{mode}]",
            leave=True,
        )
    else:
        _interval_iter = enumerate(interval_list)

    for idx, interval_value in _interval_iter:
        _log(
            f"[RUN] mode={mode:>12s} | interval={interval_value} | "
            f"repeats={n_repeats} | seeds={seed_mode}"
        )
        r = _aggregate_interval(
            mode=mode,
            interval_value=float(interval_value),
            n_repeats=int(n_repeats),
            seed_base=int(seed_base),
            idx_interval=int(idx),
        )
        _log(
            f"      A={r.A_hat:.6g} (+/- {r.A_std:.3g}), "
            f"kappa={r.kappa_hat:.6g} (+/- {r.kappa_std:.3g})"
        )
        results.append(r)

    fig, ax_r = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    # Viridis palette: distinct hues (purple -> blue -> green -> yellow)
    # so the five lines are clearly separable in print, projection, and
    # colorblind viewing, while the colormap remains perceptually
    # ordered so the throttle-window axis (ΔT short -> long) reads
    # monotonically from dark to light.  Matches the paper style used
    # in GLFT_studies.py.  The 0.05 / 0.95 trim keeps the extremes off
    # the colormap edges where the dark purple is nearly black and the
    # bright yellow loses contrast on white backgrounds.
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(1, len(results))))
    # Error bars show the standard error of the mean (SEM = std / sqrt(repeats)),
    # i.e. the uncertainty of the mean curve rather than the per-run spread.
    sem_scale = 1.0 / np.sqrt(max(1, n_repeats))

    for color, r in zip(colors, results):
        label = _label_for_interval(mode, r.interval_value)

        # (b) linearized normalized log intensity
        ax_r.errorbar(
            r.delta_grid,
            r.ylog_mean,
            yerr=r.ylog_std * sem_scale,
            fmt="o",
            markersize=4,
            capsize=4,
            color=color,
            label=label,
        )
        ax_r.plot(r.delta_grid, r.ylog_fit, linestyle="--", color=color, linewidth=2)

    ax_r.set_xlabel(r"$\delta$ [tick]")
    ax_r.set_ylabel(r"$-\hat{\kappa}\,\delta$")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best")

    # No suptitle or per-axis title: the LaTeX caption describes the
    # figure.  Keep this explicit so future regenerations match the
    # paper style.

    _save_paper_figure(fig, SAVE_FIG_PATH, dpi=180, bbox_inches="tight")
    _log(f"[SAVE] Figure written to: {SAVE_FIG_PATH}")

    plt.show()
    return results


if __name__ == "__main__":
    run_comparison_and_plot()
