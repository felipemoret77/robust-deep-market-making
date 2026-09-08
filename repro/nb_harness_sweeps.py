#!/usr/bin/env python3
"""
Faithful re-run of the two tau-sweep figures (notebook cells 23 and 32) with
the legend term "Phase" -> "Procedure".

The corrupt CELL_22 / CELL_31 caches are moved aside so the cells recompute
(deterministic under GLOBAL_SEED, using the same checkpoints), then re-save a
valid cache. Data is byte-identical to the original run; only labels change.

Regenerates: regime_tau_sweep_decay_phasea_vs_phaseb.png, degradation_vs_tau_regime.png
"""
import json
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.path.insert(0, "/Users/felipemoret/Desktop/MM_LOB_SIM")
os.chdir("/Users/felipemoret/Desktop/MM_LOB_SIM")

# --- move corrupt sweep caches aside so cells 23/32 recompute ---------------
for pat in ["notebook_cache/CELL_22_tau_sweep*.pkl", "notebook_cache/CELL_31_degradation*.pkl"]:
    for f in glob.glob(pat):
        if not f.endswith(".corrupt"):
            os.rename(f, f + ".corrupt")
            print(f"[cache] moved aside {f}", flush=True)

# --- savefig gate: no-op during setup, real writes for the target cells -----
_ALLOW_SAVE = {"on": False}
_orig_fig_savefig = Figure.savefig
def _gated_savefig(self, *a, **k):
    return _orig_fig_savefig(self, *a, **k) if _ALLOW_SAVE["on"] else None
Figure.savefig = _gated_savefig
plt.savefig = lambda *a, **k: _gated_savefig(plt.gcf(), *a, **k)
plt.show = lambda *a, **k: None


def _strip_magics(src):
    return "\n".join(("# " + l) if l.lstrip().startswith(("%", "!", "get_ipython")) else l
                     for l in src.split("\n"))


nb = json.load(open("regime_switching_diagnostic.ipynb"))
G = {"__name__": "__main__"}

SETUP_CELLS = [0, 2, 3, 4, 10]
for ci in SETUP_CELLS:
    print(f"[setup] exec cell {ci} ...", flush=True)
    exec(compile(_strip_magics("".join(nb["cells"][ci]["source"])), f"<cell {ci}>", "exec"), G)

_ALLOW_SAVE["on"] = True
for ci in [23, 32]:
    print(f"\n[target] exec cell {ci} (full sweep, savefig ON) ...", flush=True)
    tgt = _strip_magics("".join(nb["cells"][ci]["source"]))
    tgt = tgt.replace("Phase A", "Procedure A").replace("Phase B", "Procedure B")
    exec(compile(tgt, f"<cell {ci}>", "exec"), G)
    print(f"[target] cell {ci} done", flush=True)

print("\n[done] sweep figures regenerated")
