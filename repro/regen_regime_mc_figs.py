#!/usr/bin/env python3
"""
Regenerate the three Monte-Carlo regime figures from the notebook cache
(CELL_15_mc_quick), faithfully reproducing the notebook plotting cells
17/20/21 but with the legend term "Phase" -> "Algorithm".

Data-only replot: loads the cached MC arrays, no simulation.
"""
from pathlib import Path
import glob
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl_style = {
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.title_fontsize": 12,
}
plt.rcParams.update(mpl_style)

REPO = Path("/Users/felipemoret/Desktop/MM_LOB_SIM")
PAPER = Path("/Users/felipemoret/Desktop/extended_first_abstract_RLMM")
N_STEPS = 100_000
INV_LIMIT = 8
HAS_PHASEB = True

cache_file = glob.glob(str(REPO / "notebook_cache/CELL_15_mc_quick*.pkl"))[0]
(pnls_base_all, pnls_regime_all, pnls_phaseb_all,
 pnl_traj_base, pnl_traj_regime, pnl_traj_phaseb,
 inv_traj_base, inv_traj_regime, inv_traj_phaseb,
 _delta, _impr, _gap) = pickle.load(open(cache_file, "rb"))


def stack_traces(traces):
    min_len = min(len(t) for t in traces)
    return np.stack([np.asarray(t[:min_len], dtype=np.float32) for t in traces], axis=0)


pnl_matrix_base = stack_traces(pnl_traj_base)
pnl_matrix_regime = stack_traces(pnl_traj_regime)
pnl_matrix_phaseb = stack_traces(pnl_traj_phaseb)
inv_matrix_base = stack_traces(inv_traj_base)
inv_matrix_regime = stack_traces(inv_traj_regime)
inv_matrix_phaseb = stack_traces(inv_traj_phaseb)
T = min(pnl_matrix_base.shape[1], pnl_matrix_regime.shape[1], pnl_matrix_phaseb.shape[1])
t = np.linspace(0, N_STEPS - 1, T)


def _save(fig, name):
    fig.savefig(REPO / name, dpi=200, bbox_inches="tight")
    fig.savefig(PAPER / name, dpi=200, bbox_inches="tight")
    print(f"  saved {name} -> repo + paper")


# ── Figure 1: mean cumulative PnL (cell 20) ────────────────────────────────
mean_pnl_base = pnl_matrix_base[:, :T].mean(axis=0); std_pnl_base = pnl_matrix_base[:, :T].std(axis=0)
mean_pnl_regime = pnl_matrix_regime[:, :T].mean(axis=0); std_pnl_regime = pnl_matrix_regime[:, :T].std(axis=0)
mean_pnl_phaseb = pnl_matrix_phaseb[:, :T].mean(axis=0); std_pnl_phaseb = pnl_matrix_phaseb[:, :T].std(axis=0)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, mean_pnl_base, linewidth=1.2, color="#444444",
        label=f"Algorithm A + baseline — final: {mean_pnl_base[-1]:+.3f}")
ax.fill_between(t, mean_pnl_base - std_pnl_base, mean_pnl_base + std_pnl_base, alpha=0.12, color="#444444")
ax.plot(t, mean_pnl_regime, linewidth=1.2, color="#E69F00",
        label=f"Algorithm A + regime-sw — final: {mean_pnl_regime[-1]:+.3f}")
ax.fill_between(t, mean_pnl_regime - std_pnl_regime, mean_pnl_regime + std_pnl_regime, alpha=0.12, color="#E69F00")
ax.plot(t, mean_pnl_phaseb, linewidth=1.2, color="#009E73",
        label=f"Algorithm B + regime-sw — final: {mean_pnl_phaseb[-1]:+.3f}")
ax.fill_between(t, mean_pnl_phaseb - std_pnl_phaseb, mean_pnl_phaseb + std_pnl_phaseb, alpha=0.12, color="#009E73")
ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
ax.set_xlabel("Simulation step", fontsize=14); ax.set_ylabel("Cumulative PnL", fontsize=14)
ax.legend(fontsize=14, loc="upper left"); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); _save(fig, "regime_trained_pnl_mean_across_time.png"); plt.close(fig)

# ── Figure 2: mean inventory (cell 21) ─────────────────────────────────────
mean_inv_base = inv_matrix_base[:, :T].mean(axis=0); std_inv_base = inv_matrix_base[:, :T].std(axis=0)
mean_inv_regime = inv_matrix_regime[:, :T].mean(axis=0); std_inv_regime = inv_matrix_regime[:, :T].std(axis=0)
mean_inv_phaseb = inv_matrix_phaseb[:, :T].mean(axis=0); std_inv_phaseb = inv_matrix_phaseb[:, :T].std(axis=0)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t, mean_inv_base, linewidth=1.2, color="#444444", label="Algorithm A + baseline (p=0.5)")
ax.fill_between(t, mean_inv_base - std_inv_base, mean_inv_base + std_inv_base, alpha=0.12, color="#444444")
ax.plot(t, mean_inv_regime, linewidth=1.2, color="#E69F00", label="Algorithm A + regime-switching")
ax.fill_between(t, mean_inv_regime - std_inv_regime, mean_inv_regime + std_inv_regime, alpha=0.12, color="#E69F00")
ax.plot(t, mean_inv_phaseb, linewidth=1.2, color="#009E73", label="Algorithm B + regime-switching")
ax.fill_between(t, mean_inv_phaseb - std_inv_phaseb, mean_inv_phaseb + std_inv_phaseb, alpha=0.12, color="#009E73")
ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
ax.axhline(y=INV_LIMIT, color="gray", linestyle=":", alpha=0.5, label=f"±INV_LIMIT={INV_LIMIT}")
ax.axhline(y=-INV_LIMIT, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Simulation step", fontsize=14); ax.set_ylabel("Inventory (q)", fontsize=14)
ax.legend(fontsize=14, loc="upper right"); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); _save(fig, "inventory_distribution_phaseA_phaseB.png"); plt.close(fig)

# ── Figure 3: terminal-PnL histogram (cell 17) ─────────────────────────────
all_pnls = np.concatenate([pnls_base_all, pnls_regime_all, pnls_phaseb_all])
bins = np.linspace(all_pnls.min() - 0.05, all_pnls.max() + 0.05, 35)
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(pnls_base_all, bins=bins, alpha=0.35, color="#444444", edgecolor="black", linewidth=0.3,
        label=f"Algorithm A + baseline  mean={np.mean(pnls_base_all):+.3f}, std={np.std(pnls_base_all, ddof=1):.3f}")
ax.hist(pnls_regime_all, bins=bins, alpha=0.35, color="#E69F00", edgecolor="black", linewidth=0.3,
        label=f"Algorithm A + regime-sw  mean={np.mean(pnls_regime_all):+.3f}, std={np.std(pnls_regime_all, ddof=1):.3f}")
ax.hist(pnls_phaseb_all, bins=bins, alpha=0.35, color="#009E73", edgecolor="black", linewidth=0.3,
        label=f"Algorithm B + regime-sw  mean={np.mean(pnls_phaseb_all):+.3f}, std={np.std(pnls_phaseb_all, ddof=1):.3f}")
ax.axvline(np.mean(pnls_base_all), color="#444444", linestyle="--", linewidth=1.5)
ax.axvline(np.mean(pnls_regime_all), color="#E69F00", linestyle="--", linewidth=1.5)
ax.axvline(np.mean(pnls_phaseb_all), color="#009E73", linestyle="--", linewidth=1.5)
ax.set_xlabel("Terminal PnL", fontsize=14); ax.set_ylabel("Count", fontsize=14)
ax.legend(fontsize=14); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(REPO / "pnl_distribution_phaseA_phaseB.png", dpi=150, bbox_inches="tight")
fig.savefig(PAPER / "pnl_distribution_phaseA_phaseB.png", dpi=150, bbox_inches="tight")
print("  saved pnl_distribution_phaseA_phaseB.png -> repo + paper"); plt.close(fig)
print("done")
