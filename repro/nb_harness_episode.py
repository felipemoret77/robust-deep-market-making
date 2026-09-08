#!/usr/bin/env python3
"""
Faithful re-run of the single-episode diagnostic figures (notebook cell 14)
with the legend term "Phase" -> "Algorithm".

Executes the notebook's own setup + episode cells verbatim (so the data is
byte-identical to the original run, deterministic under GLOBAL_SEED); only
the two legend labels are swapped. The corrupt CELL_10 episode cache is
moved aside first so cell 11 recomputes (and re-saves a valid cache).

Regenerates: regime_switching_one_episode_example.png, regime_trained_final_phase_b.png
"""
import json
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.path.insert(0, "/Users/felipemoret/Desktop/MM_LOB_SIM")
os.chdir("/Users/felipemoret/Desktop/MM_LOB_SIM")

# --- move the corrupt episode cache aside so cell 11 recomputes -------------
import glob
for f in glob.glob("notebook_cache/CELL_10_paired_single_episodes*.pkl"):
    os.rename(f, f + ".corrupt")
    print(f"[cache] moved aside {f}")

# (IPython.display is stubbed inline in the target cell below)

# --- savefig gate: no-op during setup, real writes only for the target cell -
_ALLOW_SAVE = {"on": False}
_orig_fig_savefig = Figure.savefig
def _gated_savefig(self, *a, **k):
    if _ALLOW_SAVE["on"]:
        return _orig_fig_savefig(self, *a, **k)
    return None
Figure.savefig = _gated_savefig
plt.savefig = lambda *a, **k: _gated_savefig(plt.gcf(), *a, **k)
plt.show = lambda *a, **k: None

def _strip_magics(src):
    out = []
    for line in src.split("\n"):
        ls = line.lstrip()
        if ls.startswith(("%", "!", "get_ipython")):
            out.append("# " + line)  # comment out Jupyter magics
        else:
            out.append(line)
    return "\n".join(out)


nb = json.load(open("regime_switching_diagnostic.ipynb"))
G = {"__name__": "__main__"}

SETUP_CELLS = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]
TARGET_CELL = 14

for ci in SETUP_CELLS:
    src = _strip_magics("".join(nb["cells"][ci]["source"]))
    print(f"[setup] exec cell {ci} ...", flush=True)
    exec(compile(src, f"<cell {ci}>", "exec"), G)

# --- verify the recomputed episode matches the original figure --------------
print("\n[verify] recomputed episode final PnLs:")
print("  Algorithm A baseline :", round(float(G["mm_base"]["MM_TotalPnL"].iloc[-1]), 4))
print("  Algorithm A regime-sw:", round(float(G["mm_regime"]["MM_TotalPnL"].iloc[-1]), 4))
if G.get("mm_regime_phaseb") is not None:
    print("  Algorithm B regime-sw:", round(float(G["mm_regime_phaseb"]["MM_TotalPnL"].iloc[-1]), 4))
print("  (original figure showed ~ +6.4 / -2.1 / +2.15)\n")

# --- relabel Phase -> Algorithm and run the target plotting cell ------------
G["BASE_LABEL"] = "Algorithm A stationary DQN"
tgt = _strip_magics("".join(nb["cells"][TARGET_CELL]["source"]))
tgt = tgt.replace('"Phase B: regime-switching"', '"Algorithm B: regime-switching"')
tgt = tgt.replace("Phase A baseline:", "Algorithm A baseline:").replace("Phase A RS:", "Algorithm A RS:")
tgt = tgt.replace("Phase B RS:", "Algorithm B RS:")
tgt = tgt.replace(
    "import IPython.display as ipd",
    "import types as _ipyt; ipd = _ipyt.SimpleNamespace("
    "display=lambda *a, **k: None, Image=lambda *a, **k: None)",
)

_ALLOW_SAVE["on"] = True
print(f"[target] exec cell {TARGET_CELL} (savefig ON) ...", flush=True)
exec(compile(tgt, f"<cell {TARGET_CELL}>", "exec"), G)
print("\n[done] episode figures regenerated")
