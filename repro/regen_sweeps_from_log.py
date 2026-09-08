#!/usr/bin/env python3
"""
Reconstruct the two sweep figures (tau_sweep, degradation) with the legend
term "Algorithm", WITHOUT re-running the heavy sweeps.

The per-tau means/stds are the exact deterministic values already computed by
the sweep re-run (captured in nb_sweeps_rerun.log). We rebuild the results
dicts from those numbers and execute the notebook cells' own plotting slices
verbatim, so the figures are byte-faithful except for the "Phase"->"Algorithm"
legend labels (which come from spec['label']).

sems = std / sqrt(N), N = 100 seeds per tau.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})
plt.show = lambda *a, **k: None

REPO = Path("/Users/felipemoret/Desktop/MM_LOB_SIM")
PAPER = Path("/Users/felipemoret/Desktop/extended_first_abstract_RLMM")
N = 100
sem = lambda stds: (np.asarray(stds) / np.sqrt(N))

nb = json.load(open(REPO / "regime_switching_diagnostic.ipynb"))


def _plot_slice(cell_idx, start_marker, end_marker):
    """Extract the plotting slice of a notebook cell (between markers,
    end inclusive), dedent by 4 spaces, relabel Phase->Algorithm."""
    src = "".join(nb["cells"][cell_idx]["source"]).split("\n")
    i0 = next(i for i, l in enumerate(src) if start_marker in l)
    i1 = next(i for i, l in enumerate(src) if end_marker in l and i >= i0)
    body = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i0:i1 + 1])
    return body.replace("Phase", "Algorithm")


# ---------------- tau_sweep (cell 23) ----------------
tau_results = {
    "phasea_invp001": {
        "spec": {"label": "Algorithm A stationary invp=0.001", "key": "phasea_invp001"},
        "means": np.array([6.6532, 1.9650, -0.1376, -2.4004, -4.4847, -6.0824]),
        "sems": sem([0.7578, 1.3295, 1.8542, 2.5522, 3.3818, 4.5902]),
    },
    "phaseb_invp001": {
        "spec": {"label": "Algorithm B regime-trained invp=0.001", "key": "phaseb_invp001"},
        "means": np.array([3.6788, 3.0441, 2.6617, 2.3304, 2.1378, 1.5264]),
        "sems": sem([0.3728, 0.4409, 0.5347, 0.7668, 0.7863, 1.4552]),
    },
}
plot_labels = ["tau=15", "tau=30", "tau=60", "tau=120", "tau=240"]
tau_labels = ["No regime"] + plot_labels
TAU_SWEEP_FIG_PATH = REPO / "regime_tau_sweep_decay_phasea_vs_phaseb.png"
TAU_SWEEP_PAPER_FIG_PATH = PAPER / "regime_tau_sweep_decay_phasea_vs_phaseb.png"

g23 = dict(np=np, plt=plt, tau_results=tau_results, plot_labels=plot_labels,
           tau_labels=tau_labels, TAU_SWEEP_FIG_PATH=TAU_SWEEP_FIG_PATH,
           TAU_SWEEP_PAPER_FIG_PATH=TAU_SWEEP_PAPER_FIG_PATH)
slice23 = _plot_slice(23, "x = np.arange(len(plot_labels))", "plt.show()")
exec(compile(slice23, "<cell23 plot>", "exec"), g23)
print("  wrote tau_sweep ->", TAU_SWEEP_PAPER_FIG_PATH)

# ---------------- degradation (cell 32) ----------------
degradation_tau_results = {
    "phasea_invp001": {
        "spec": {"label": "Algorithm A invp=0.001", "key": "phasea_invp001"},
        "means": np.array([6.6522, 2.0868, -0.4960, -2.3875, -4.9591, -5.9891]),
        "sems": sem([0.6449, 1.5359, 2.2765, 2.4879, 3.5796, 4.3806]),
    },
    "phasea_invp005": {
        "spec": {"label": "Algorithm A invp=0.005", "key": "phasea_invp005"},
        "means": np.array([5.1500, 1.9033, 0.5441, -0.9327, -4.0426, -5.3422]),
        "sems": sem([0.8588, 1.5699, 2.0949, 2.4140, 4.7916, 5.7796]),
    },
}
degradation_plot_labels = [r"$\tau_r=15$", r"$\tau_r=30$", r"$\tau_r=60$",
                           r"$\tau_r=120$", r"$\tau_r=240$"]
degradation_tau_labels = ["No regime"] + degradation_plot_labels
DEGRADATION_FIG_PATH = REPO / "degradation_vs_tau_regime.png"
DEGRADATION_PAPER_FIG_PATH = PAPER / "degradation_vs_tau_regime.png"

g32 = dict(np=np, plt=plt, degradation_tau_results=degradation_tau_results,
           degradation_plot_labels=degradation_plot_labels,
           degradation_tau_labels=degradation_tau_labels,
           DEGRADATION_FIG_PATH=DEGRADATION_FIG_PATH,
           DEGRADATION_PAPER_FIG_PATH=DEGRADATION_PAPER_FIG_PATH)
slice32 = _plot_slice(32, "x = np.arange(len(degradation_plot_labels))", "plt.show()")
exec(compile(slice32, "<cell32 plot>", "exec"), g32)
print("  wrote degradation ->", DEGRADATION_PAPER_FIG_PATH)
print("done")
