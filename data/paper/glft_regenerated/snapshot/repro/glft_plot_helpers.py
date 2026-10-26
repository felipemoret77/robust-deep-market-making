"""Frozen paper plot functions, extracted without importing campaign side effects.

Original function sources and SHA-256 hashes are listed in the GLFT provenance.
Only the caller supplies the local output destination and save helper.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ALWAYS_BEST_LABEL = 'At-best baseline'
OUT_DIR = None

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
