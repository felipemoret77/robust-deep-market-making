#!/usr/bin/env python3
"""
Generate a small bundle of theory-facing plots for as_miss.tex.

Outputs are written under:
    ./as_miss_bundle/

The bundle contains:
  - a copy of as_miss.tex
  - a copy of references.bib
  - PNG plots derived from the analytical formulas
  - a README with the parameters used
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_as_miss")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/as_miss_cache")

import matplotlib.pyplot as plt
import numpy as np

# === Paper figure style: matches GLFT_studies.py / risk_return_frontier.py
# / comparison_*_by_throttle.py so the as_miss bundle figures share label
# sizing and the viridis palette with the rest of the paper. ===
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
from cycler import cycler as _cycler
_VIRIDIS_PAPER_PALETTE = [
    tuple(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 6))
]
plt.rcParams["axes.prop_cycle"] = _cycler(color=_VIRIDIS_PAPER_PALETTE)


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


ROOT = Path(__file__).resolve().parent
TEX_SRC = Path("/Users/felipemoret/Desktop/MM_initial_research/as_miss.tex")
BIB_SRC = Path("/Users/felipemoret/Desktop/MM_initial_research/references.bib")
OUT_DIR = ROOT / "as_miss_bundle"


def load_rst2():
    path = ROOT / "regime_switching_stress_test_2.py"
    spec = importlib.util.spec_from_file_location("rst2", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def solve_stationary_distribution(r_a: np.ndarray, r_b: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    q_zero_idx = int(np.where(q_values == 0)[0][0])
    log_pi = np.zeros_like(q_values, dtype=float)

    for i in range(q_zero_idx + 1, len(q_values)):
        rb = r_b[i - 1]
        ra = r_a[i]
        if rb > 1e-30 and ra > 1e-30:
            log_pi[i] = log_pi[i - 1] + np.log(rb / ra)
        else:
            log_pi[i] = -500.0

    for i in range(q_zero_idx - 1, -1, -1):
        ra = r_a[i + 1]
        rb = r_b[i]
        if rb > 1e-30 and ra > 1e-30:
            log_pi[i] = log_pi[i + 1] + np.log(ra / rb)
        else:
            log_pi[i] = -500.0

    log_pi -= np.max(log_pi)
    pi = np.exp(log_pi)
    pi /= pi.sum()
    return pi


def compute_bundle_curves(
    probs: np.ndarray,
    *,
    A: float,
    kappa: float,
    sigma: float,
    gamma: float,
    T: float,
    inv_limit: int,
    r_fill_factor: float,
    tick_size: float,
    v0: float,
):
    q_values = np.arange(-inv_limit, inv_limit + 1, dtype=float)

    if gamma > 1e-12:
        delta_base = np.log1p(gamma / kappa) / gamma
        log_base = 0.5 * (1.0 + kappa / gamma) * np.log1p(gamma / kappa)
        power_term = np.exp(min(log_base, 500.0))
        c_skew = np.sqrt(sigma**2 * gamma / (2.0 * kappa * A)) * power_term
    else:
        delta_base = 1.0 / kappa
        c_skew = 0.0

    delta_bar = float(delta_base + 0.5 * c_skew)
    delta_b = np.maximum(delta_base + (2.0 * q_values + 1.0) * 0.5 * c_skew, 0.0)
    delta_a = np.maximum(delta_base + (1.0 - 2.0 * q_values) * 0.5 * c_skew, 0.0)

    pi_rows = []
    qbar = []
    spread_pnl = []
    full_pnl = []
    mtm_loss = []
    boundary_neg = []
    boundary_pos = []
    boundary_total = []

    for p in probs:
        p = float(p)
        r_a = 2.0 * p * A * np.exp(-kappa * delta_a) * r_fill_factor
        r_b = 2.0 * (1.0 - p) * A * np.exp(-kappa * delta_b) * r_fill_factor
        r_a[0] = 0.0
        r_b[-1] = 0.0

        pi = solve_stationary_distribution(r_a, r_b, q_values)
        eq = float(np.sum(pi * q_values))
        spread_rate = float(np.sum(pi * (r_a * delta_a + r_b * delta_b)))
        spread_val = spread_rate * T * tick_size
        mtm_val = -v0 * (2.0 * p - 1.0) * eq * T * tick_size
        full_val = spread_val - mtm_val

        pi_rows.append(pi)
        qbar.append(eq)
        spread_pnl.append(spread_val)
        full_pnl.append(full_val)
        mtm_loss.append(mtm_val)
        boundary_neg.append(float(pi[0]))
        boundary_pos.append(float(pi[-1]))
        boundary_total.append(float(pi[0] + pi[-1]))

    return {
        "q_values": q_values,
        "delta_base": float(delta_base),
        "delta_bar": float(delta_bar),
        "c_skew": float(c_skew),
        "delta_a": delta_a,
        "delta_b": delta_b,
        "pi": np.vstack(pi_rows),
        "qbar": np.asarray(qbar),
        "spread_pnl": np.asarray(spread_pnl),
        "full_pnl": np.asarray(full_pnl),
        "mtm_loss": np.asarray(mtm_loss),
        "boundary_neg": np.asarray(boundary_neg),
        "boundary_pos": np.asarray(boundary_pos),
        "boundary_total": np.asarray(boundary_total),
    }


def save_plot_pi_lines(probs: list[float], curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(probs)))
    q_values = curves["q_values"]
    probs_dense = np.linspace(0.2, 0.8, curves["pi"].shape[0])
    for color, p in zip(colors, probs):
        idx = int(np.argmin(np.abs(probs_dense - p)))
        ax.plot(q_values, curves["pi"][idx], "o-", lw=2, ms=5, color=color, label=f"p={p:.1f}")
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel(r"Stationary mass $\pi(q)$")
    # ax.set_title(r"Exponentially Tilted Discrete Gaussian: $\pi(q)$ vs $q$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_pi_q_lines.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_pi_heatmap(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    # Match figsize of save_plot_pi_lines so the two Fig. 20 panels render at
    # equal height under width=\linewidth in the paper.
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    q_values = curves["q_values"]
    im = ax.imshow(
        curves["pi"].T,
        aspect="auto",
        origin="lower",
        extent=[float(probs_dense[0]), float(probs_dense[-1]), float(q_values[0]), float(q_values[-1])],
        cmap="viridis",
    )
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel("Inventory q")
    # ax.set_title(r"Heatmap of $\pi(q)$ across asymmetry levels")
    fig.colorbar(im, ax=ax, label=r"$\pi(q)$")
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_pi_heatmap.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_qbar(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(probs_dense, curves["qbar"], color="darkorange", lw=2.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel(r"$\bar q(p)$")
    # ax.set_title(r"Expected Inventory $\bar q$ as a Function of Flow Asymmetry")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_qbar_vs_p.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_boundary_mass(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(probs_dense, curves["boundary_neg"], label=r"$\pi(-Q)$", color="crimson", lw=2)
    ax.plot(probs_dense, curves["boundary_pos"], label=r"$\pi(+Q)$", color="royalblue", lw=2)
    ax.plot(probs_dense, curves["boundary_total"], label=r"$\pi(-Q)+\pi(+Q)$", color="black", lw=2.2)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel("Boundary mass")
    # ax.set_title("Boundary Concentration as p Moves Away from 0.5")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_boundary_mass_vs_p.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_pnl_curves(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(probs_dense, curves["spread_pnl"], color="forestgreen", lw=2.5, label="Spread-only")
    ax.plot(probs_dense, curves["full_pnl"], color="crimson", lw=2.5, label="Full (with drift)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{PnL}]$")
    # ax.set_title(r"Theoretical $\mathbb{E}[\mathrm{PnL}]$ vs Flow Asymmetry")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_expected_pnl_curves.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_pnl_decomposition(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(probs_dense, curves["spread_pnl"], color="forestgreen", lw=2.5, label="Spread capture")
    ax.plot(probs_dense, curves["mtm_loss"], color="crimson", lw=2.5, label="MtM loss")
    ax.plot(probs_dense, curves["full_pnl"], color="black", lw=2.5, label="Net full PnL")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel("Value")
    # ax.set_title("Decomposition: Spread Capture vs Drift-Induced MtM Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_pnl_decomposition.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def save_plot_local_quadratic(probs_dense: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    mask = (probs_dense >= 0.4) & (probs_dense <= 0.6)
    x = probs_dense[mask]
    z = x - 0.5
    y = curves["full_pnl"][mask]

    X = np.column_stack([np.ones_like(z), z**2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_fit = beta[0] + beta[1] * z**2

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(x, y, color="crimson", lw=2.5, label="Exact full curve")
    ax.plot(x, y_fit, color="black", lw=2.0, linestyle="--", label="Quadratic local fit")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Buy MO probability p")
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{PnL}]$")
    # ax.set_title(r"Local Quadratic Approximation Near $p=\frac{1}{2}$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, "as_miss_local_quadratic_fit.png", local_dir=OUT_DIR, dpi=180)
    plt.close(fig)


def write_readme(params: dict[str, float]) -> None:
    text = f"""# as_miss plot bundle

This folder contains theory-facing plots for `as_miss.tex`.

Files
- `as_miss.tex`: copy of the current note
- `references.bib`: bibliography copy
- `as_miss_pi_q_lines.png`: stationary distribution `pi(q)` for selected values of `p`
- `as_miss_pi_heatmap.png`: heatmap of `pi(q)` across the full `p` grid
- `as_miss_qbar_vs_p.png`: expected inventory as a function of `p`
- `as_miss_boundary_mass_vs_p.png`: mass at the inventory boundaries
- `as_miss_expected_pnl_curves.png`: spread-only and full expected PnL
- `as_miss_pnl_decomposition.png`: spread capture, MtM loss, and net full PnL
- `as_miss_local_quadratic_fit.png`: local quadratic fit near `p=0.5`
- `params.json`: numerical parameters used for generation

Parameters used
- A = {params["A"]:.6f}
- kappa = {params["kappa"]:.6f}
- sigma = {params["sigma"]:.6f}
- gamma = {params["gamma"]:.6e}
- T = {params["T"]:.6f}
- inv_limit = {params["inv_limit"]}
- r_fill_factor = {params["r_fill_factor"]:.6f}
- v0 = {params["v0"]:.6f}
- tick_size = {params["tick_size"]:.6f}

Suggested insertion order in the note
1. `as_miss_pi_q_lines.png` after the stationary law `pi(q)`
2. `as_miss_qbar_vs_p.png` near the discussion of `bar q`
3. `as_miss_expected_pnl_curves.png` in the final-results section
4. `as_miss_local_quadratic_fit.png` near the local expansion
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rst2 = load_rst2()

    # Canonical pair unified across all paper experiments, reproduced by
    # GLFT_studies.run_calibration() (dT=0.5, 500k steps, n_equil=1000, seed=1234).
    A = 0.150707
    kappa = 2.33534
    sigma = 0.3
    gamma = 1e-6
    tick_size = 0.01
    inv_limit = 8

    # Keep the generation grounded in the current simulator conventions,
    # but with a lighter calibration budget than the full workflow.
    T = float(rst2.estimate_episode_time_horizon(seed=rst2.SEED_CALIB + 2))
    delta_base = np.log1p(gamma / kappa) / gamma
    log_base = 0.5 * (1.0 + kappa / gamma) * np.log1p(gamma / kappa)
    power_term = np.exp(min(log_base, 500.0))
    c_skew = np.sqrt(sigma**2 * gamma / (2.0 * kappa * A)) * power_term
    delta_bar = float(delta_base + 0.5 * c_skew)

    r_fill_factor = float(
        rst2.calibrate_r_fill_factor(
            A=A,
            kappa=kappa,
            sigma=sigma,
            delta_bar=delta_bar,
            n_episodes=10,
            seed_base=rst2.SEED_CALIB + 200,
        )
    )
    v0 = float(
        rst2.calibrate_v0(
            p_grid=[0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
            n_episodes=10,
            seed_base=rst2.SEED_CALIB + 100,
        )
    )

    probs_dense = np.linspace(0.2, 0.8, 121)
    curves = compute_bundle_curves(
        probs_dense,
        A=A,
        kappa=kappa,
        sigma=sigma,
        gamma=gamma,
        T=T,
        inv_limit=inv_limit,
        r_fill_factor=r_fill_factor,
        tick_size=tick_size,
        v0=v0,
    )

    save_plot_pi_lines([0.3, 0.4, 0.5, 0.6, 0.7], curves)
    save_plot_pi_heatmap(probs_dense, curves)
    save_plot_qbar(probs_dense, curves)
    save_plot_boundary_mass(probs_dense, curves)
    save_plot_pnl_curves(probs_dense, curves)
    save_plot_pnl_decomposition(probs_dense, curves)
    save_plot_local_quadratic(probs_dense, curves)

    shutil.copy2(TEX_SRC, OUT_DIR / "as_miss.tex")
    if BIB_SRC.exists():
        shutil.copy2(BIB_SRC, OUT_DIR / "references.bib")

    params = {
        "A": A,
        "kappa": kappa,
        "sigma": sigma,
        "gamma": gamma,
        "T": T,
        "inv_limit": inv_limit,
        "r_fill_factor": r_fill_factor,
        "v0": v0,
        "tick_size": tick_size,
        "delta_bar_used_for_rfill_calibration": delta_bar,
        "delta_bar_formula": curves["delta_bar"],
        "c_skew": curves["c_skew"],
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    write_readme(params)

    print(f"Bundle written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
