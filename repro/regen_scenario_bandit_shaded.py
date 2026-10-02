"""
Regenerate the two scenario-bandit histograms with SHADED (filled) bars.

The notebook cells used ``fill=False`` alongside ``alpha=0.45`` (the
``fill=False`` overrides the alpha and yields outline-only bars). This
script reloads the CELL_29 and CELL_30 pickle caches and re-plots with
filled bars matching the style we fixed for cells 27/29 earlier
(``alpha=0.45`` with ``fill=True`` implicit).
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]

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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Replot archived B/C stress-test outcomes.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for family, name in [('random', 'scenario_bandit_random_tau.png'),
                         ('correlated', 'scenario_bandit_correlated_regime_stress.png')]:
        with np.load(ROOT / f'data/paper/bandit_{family}.npz', allow_pickle=False) as data:
            _plot_pair(data['B_pnl'], data['C_pnl'], phaseb_label_stem='Algorithm B',
                       out_path=args.output / name)


if __name__ == '__main__':
    main()
