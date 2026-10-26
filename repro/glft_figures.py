"""Shared plotting functions and entry point for the final GLFT panels."""

import os

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glft_plot_helpers as plots

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper/glft_regenerated/data"
OUT = ROOT / "outputs/glft_figures"


def save(fig, name, **kwargs):
    kwargs.pop("local_dir", None)
    kwargs.pop("dpi", None)
    kwargs.pop("bbox_inches", None)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"{Path(name).stem}.{ext}", dpi=200, bbox_inches="tight", **kwargs
        )


def load(name):
    with np.load(DATA / f"{name}.npz") as source:
        return {key: source[key] for key in source.files}


def fig2():
    store, summaries = {}, []
    for gamma in (1e-6, 1e-3, 1e-1):
        d = load(f"fig2_gamma_{gamma:g}")
        store[gamma] = [
            pd.DataFrame({"MM_TotalPnL": p, "MM_Inventory": q})
            for p, q in zip(d["pnl_trace"], d["inventory_trace"])
        ]
        summaries.extend(
            dict(gamma=gamma, final_total_pnl=float(p)) for p in d["terminal_pnl"]
        )
    base = load("fig2_at_best")
    benchmark = [
        pd.DataFrame({"MM_TotalPnL": p, "MM_Inventory": q})
        for p, q in zip(base["pnl_trace"], base["inventory_trace"])
    ]
    plots.plot_mean_total_pnl_trajectory_by_gamma(store, benchmark_store=benchmark)
    plots.plot_cartea_style_distributions(
        pd.DataFrame(summaries),
        store,
        benchmark_summary_df=pd.DataFrame({"final_total_pnl": base["terminal_pnl"]}),
        benchmark_store=benchmark,
    )
    plt.close("all")


def fig4():
    taus, means, sems = [15, 30, 60, 120, 240], [], []
    for tau in taus:
        y = load(f"fig4_new_tau_{tau}")["terminal_pnl"]
        means.append(y.mean())
        sems.append(y.std(ddof=1) / np.sqrt(len(y)))
    base = load("fig4_new_baseline")["terminal_pnl"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    color = "#E69F00"
    ax.errorbar(
        taus,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        color=color,
        linewidth=1.6,
        markersize=7,
        label=r"GLFT $\gamma=10^{-6}$ regime",
    )
    ax.axhline(
        base,
        color=color,
        linestyle=":",
        linewidth=1.6,
        label=r"GLFT $\gamma=10^{-6}$ no regime",
    )
    ax.set_xticks(taus)
    ax.set_xticklabels([rf"$\tau_r={t}$" for t in taus])
    ax.set_xlabel(r"$\tau_r$ [MO events]")
    ax.set_ylabel("Terminal PnL mean +/- SEM")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=14)
    fig.tight_layout()
    save(fig, "glft_degradation_vs_tau_regime.png")
    plt.close(fig)


if __name__ == "__main__":
    from plot_glft_regenerated import main

    main()
