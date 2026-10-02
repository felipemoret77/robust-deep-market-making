"""Replot the corrected persistence sweep from archived per-condition results."""
import argparse
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT / 'data/paper/tau_sweep_summary.csv').open()))
plt.rcParams.update({'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12})
tau_results = {}
for label, key, prefix in [('A','phasea_invp001','A'),('B','phaseb_invp001','B_new')]:
    tau_results[key] = dict(spec={'label':f'Algorithm {label}'},
        means=np.array([float(r[prefix+'_mean_pnl']) for r in rows]),
        sems=np.array([float(r[prefix+'_sem_pnl']) for r in rows]))
plot_labels = ['tau=15','tau=30','tau=60','tau=120','tau=240']
TAU_SWEEP_FIG_PATH = args.output / 'regime_tau_sweep_decay_phasea_vs_phaseb.png'
TAU_SWEEP_PAPER_FIG_PATH = TAU_SWEEP_FIG_PATH
x = np.arange(len(plot_labels))
colors = {
    "phasea_invp001": "#E69F00",
    "phaseb_invp001": "#009E73",
}
markers = {
    "phasea_invp001": "o",
    "phaseb_invp001": "^",
}
text_offsets = {
    "phasea_invp001": -0.06,
    "phaseb_invp001": 0.06,
}
no_regime_styles = {
    "phasea_invp001": {"linestyle": ":", "linewidth": 2.0, "alpha": 0.9},
    "phaseb_invp001": {"linestyle": ":", "linewidth": 2.0, "alpha": 0.9},
}

fig, ax1 = plt.subplots(figsize=(11, 5))

for key, result in tau_results.items():
    spec = result["spec"]
    color = colors.get(key)
    style = no_regime_styles.get(key, {"linestyle": "--", "linewidth": 1.3, "alpha": 0.75})
    ax1.axhline(
        result["means"][0],
        color=color,
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label=f"{spec['label']} — stationary flow",
    )

for key, result in tau_results.items():
    spec = result["spec"]
    color = colors.get(key)
    ax1.errorbar(
        x,
        result["means"][1:],
        yerr=result["sems"][1:],
        fmt=markers.get(key, "o") + "-",
        capsize=4,
        color=color,
        linewidth=1.9,
        markersize=6,
        label=f"{spec['label']} — regime switching",
    )

ax1.set_ylabel("Terminal PnL mean +/- SEM")
regime_p_lo = globals().get("REGIME_P_LO", 0.2)
regime_p_hi = globals().get("REGIME_P_HI", 0.8)
# ax1.set_title("")  # supervisor: no inline titles; caption describes
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend(loc="best", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(plot_labels)
ax1.set_xlabel(r"$\tau_r$ [MO events]")

plt.tight_layout()
plt.savefig(TAU_SWEEP_FIG_PATH, dpi=150)
if TAU_SWEEP_PAPER_FIG_PATH.parent.exists():
    plt.savefig(TAU_SWEEP_PAPER_FIG_PATH, dpi=150)
plt.close(fig)
print(TAU_SWEEP_FIG_PATH)
