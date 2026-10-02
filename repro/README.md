# Reproduction entry points

Use these scripts for the current paper configuration:

- `verify_paper.py`: model/data identity and configuration checks.
- `evaluate_paper.py`: frozen A/B/C evaluation; explicit model selection, paired
  seed protocols, no automatic fallback to another checkpoint.
- `paper_runtime.py`, `paper_helpers.py`: original loading/evaluation functions
  with portable path handling. Helper bodies have recorded AST hashes.
- `plot_paper_sweep.py`: corrected sweep from archived summary data.
- `regen_regime_mc_figs.py`: three archived Monte Carlo panels.
- `regen_scenario_bandit_shaded.py`: two archived B/C stress histograms.

`make figures` writes to `outputs/paper_figures`. The manuscript's selected
figures are not overwritten by default.

## Historical material

The following files are retained as research history, not current paper commands:

- `nb_harness_episode.py`, `nb_harness_sweeps.py`: old notebook execution harnesses
  that use external paths and move caches. Do not use them to reproduce the
  corrected paper package.
- `regen_sweeps_from_log.py`: literal aggregates from the old mixed-filter B
  sweep. It does not reproduce the corrected B curve.
- `regime_switching_diagnostic.ipynb`: exploratory notebook with historical
  outputs and cache/path assumptions.
- `viz_checkpoint_mix_off.py`: older plotting defaults; not a verified entry
  point for the published heatmaps.
- `relabel_iota_legend.py`, `relabel_iota_raster.py`: one-time presentation edits.
- `tau_sweep_checkpoint_comparison/` and `bayes_filter_validation/`: original
  local audit campaigns. Some scripts intentionally reference the original
  MM_LOB_SIM workspace and historical manifests. The portable outputs needed
  for current reproduction are curated in `data/paper/`.

These files can be moved to a research archive in a later cleanup. They have
not been deleted, and the supported commands do not execute their top-level code.
