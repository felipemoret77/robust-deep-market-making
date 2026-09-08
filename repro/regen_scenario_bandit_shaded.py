"""
Regenerate the two scenario-bandit histograms with SHADED (filled) bars.

The notebook cells used ``fill=False`` alongside ``alpha=0.45`` (the
``fill=False`` overrides the alpha and yields outline-only bars). This
script reloads the CELL_29 and CELL_30 pickle caches and re-plots with
filled bars matching the style we fixed for cells 27/29 earlier
(``alpha=0.45`` with ``fill=True`` implicit).
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Match the paper export style applied at the top of the notebook.
mpl.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})

REPO_DIR = Path("/Users/felipemoret/Desktop/MM_LOB_SIM")
CACHE_DIR = REPO_DIR / "notebook_cache"

CACHE_RANDOM_TAU = CACHE_DIR / "CELL_29_mc_dro_vs_phaseb_random_tau_cb9e32cb.pkl"
CACHE_CLUSTERED = CACHE_DIR / "CELL_30_mc_dro_vs_phaseb_clustered_3dbeaec0.pkl"

OUT_RANDOM_TAU = REPO_DIR / "scenario_bandit_random_tau.png"
OUT_CLUSTERED = REPO_DIR / "scenario_bandit_correlated_regime_stress.png"

COPY_TARGET_DIRS = [
    Path("/Users/felipemoret/Desktop/extended_first_abstract_RLMM"),
    Path("/Users/felipemoret/Desktop/market_making_with_alpha_signals"),
]


def _load(cache_path: Path):
    with cache_path.open("rb") as fh:
        return pickle.load(fh)


def _plot_pair(phaseb_pnls: np.ndarray, dro_pnls: np.ndarray, *,
               phaseb_label_stem: str, out_path: Path) -> None:
    phaseb_pnls = np.asarray(phaseb_pnls, dtype=float)
    dro_pnls = np.asarray(dro_pnls, dtype=float)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    all_pnls = np.concatenate([phaseb_pnls, dro_pnls])
    bins = np.linspace(all_pnls.min() - 0.05, all_pnls.max() + 0.05, 30)

    ax1.hist(
        phaseb_pnls,
        bins=bins,
        alpha=0.45,
        linewidth=2.0,
        color="#009E73",
        edgecolor="#009E73",
        label=f"{phaseb_label_stem}, mu={phaseb_pnls.mean():+.3f}, "
              f"std={phaseb_pnls.std(ddof=1):.3f}",
    )
    ax1.hist(
        dro_pnls,
        bins=bins,
        alpha=0.45,
        linewidth=2.0,
        color="#CC79A7",
        edgecolor="#CC79A7",
        label=f"Algorithm C, mu={dro_pnls.mean():+.3f}, "
              f"std={dro_pnls.std(ddof=1):.3f}",
    )
    ax1.axvline(phaseb_pnls.mean(), color="#009E73", linestyle="--", linewidth=1.5)
    ax1.axvline(dro_pnls.mean(), color="#CC79A7", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Terminal PnL")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=14, loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _copy_to_targets(src: Path) -> None:
    for target_dir in COPY_TARGET_DIRS:
        if not target_dir.exists():
            print(f"  skip {target_dir} (does not exist)")
            continue
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        print(f"  copied -> {dst}")


def main() -> None:
    print(f"[CELL_29] loading {CACHE_RANDOM_TAU.name}")
    (dro_random_tau_pnls,
     _dro_random_tau_absinv,
     _dro_random_tau_limit_frac,
     phaseb_cmp_pnls,
     _phaseb_cmp_absinv,
     _phaseb_cmp_limit_frac,
     _paired_diff_dro,
     _std_ratio) = _load(CACHE_RANDOM_TAU)

    _plot_pair(
        phaseb_cmp_pnls,
        dro_random_tau_pnls,
        phaseb_label_stem="Algorithm B",
        out_path=OUT_RANDOM_TAU,
    )
    _copy_to_targets(OUT_RANDOM_TAU)

    print(f"[CELL_30] loading {CACHE_CLUSTERED.name}")
    (dro_clustered_pnls,
     _dro_clustered_absinv,
     _dro_clustered_limit_frac,
     phaseb_cmp_clustered_pnls,
     _phaseb_cmp_clustered_absinv,
     _phaseb_cmp_clustered_limit_frac,
     _paired_diff_dro_clustered,
     _std_ratio_clustered) = _load(CACHE_CLUSTERED)

    _plot_pair(
        phaseb_cmp_clustered_pnls,
        dro_clustered_pnls,
        phaseb_label_stem="Algorithm B",
        out_path=OUT_CLUSTERED,
    )
    _copy_to_targets(OUT_CLUSTERED)


if __name__ == "__main__":
    main()
