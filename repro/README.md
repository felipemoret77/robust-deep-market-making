# Reproduction entry points

Use these scripts for the current paper configuration:

- `verify_paper.py`: model/data identity and configuration checks.
- `verify_numerical_stability.py`: fixed-state action regressions for A/B/C,
  real near-tie and inventory masks; optional full B/500128 episode.
- `evaluation_environment.py`: environment checks before supported new evaluations.
- `evaluate_corrected.py`: replay exact selected A/B/C cases with both stable C51
  and FIFO preservation; optional full-metric and trajectory comparison.
- `corrected_evaluation.py`: validate 4,405 outcomes and 305 traces, and
  reconstruct all derived plotting inputs from these records.
- `evaluate_paper.py`: stable-C51-only evaluations and historical controls;
  this earlier entry point does not apply the FIFO intervention.
- `paper_runtime.py`, `paper_helpers.py`: original loading/evaluation functions
  with portable path handling. Helper bodies have recorded AST hashes.
- `plot_paper_sweep.py`: corrected sweep from archived summary data.
- `regen_regime_mc_figs.py`: three archived Monte Carlo panels.
- `regen_scenario_bandit_shaded.py`: two B/C stress histograms, using 100 paired outcomes per family (all 100 pairs from the complete corrected evaluation).
- `frontier.py plot` / `plot_frontier_10000.py`: the selected 110,000-outcome frontier
  with the unweighted GLFT calibration and archived bootstrap errors. The
  `frontier.py evaluate` command retains the legacy 1,000-seed protocol.
- `plot_paper_heatmaps.py`: recompute the three slices from canonical B and its
  action mask; `--verify-reference` compares all arrays with the final archive.
- `glft_figures.py`: corrected calibration/benchmark panels and numerical
  recomputation of finite-horizon analytical GLFT PnL from inventory zero.
- `glft_calibration.py`: actual probe-lifetime estimator, compact archived
  sufficient-statistic aggregation, and an explicit full-campaign option.
- `glft_evaluate.py`: rerun the GLFT/at-best benchmark simulations with the
  recorded seeds and compare the resulting arrays with the archive.
- `plot_historical_results.py`: corrected stationary A trajectory, degradation
  summaries and B stress histograms; filename retained for compatibility.
- `plot_paper_episode.py`: corrected A/B episode comparisons and B signals.
  Both scripts use the corrected evaluation protocol for `--verify-rerun`.
- `plot_calibration_windows.py`: recalculate unweighted per-seed fits,
  normalization and SEM from archived probe counts and exposure.
- `plot_paper_diagrams.py`: both explanatory diagrams from vector source;
  exports PNG, PDF and SVG without an external LaTeX installation.
- `check_portability.py`: isolated export, input checks and figure regeneration
  with access to the original research directories blocked.

`make figures` writes to `outputs/paper_figures`. The manuscript's selected
figures are not overwritten by default.

`glft_figures.py` selects the completed `data/paper/glft_regenerated/` archive
and validates its episode seeds, means, calibration and analytical curves.
`--strict-images` additionally checks the selected PNG hashes.
`glft_figures.py --legacy`, `glft_calibration.py`, and `glft_evaluate.py` retain
the earlier calibration as historical controls. Their simulation defaults do
not reproduce the newly selected calibration; the exact new campaign settings
and generating sources are archived in `glft_regenerated/`.

## Final corrected experiments

The exact selected A/B/C cases are in `data/paper/corrected_evaluation/protocol.json`.
They use the archived controller with the two frozen-evaluation interventions in
`paper_evaluation_corrections.py`. Current training code and checkpoints are not
modified. `corrected_evaluation.py` checks the source links, all seed/model/horizon
records, summary statistics and derived plot arrays. Old selections are under
`corrected_evaluation/previous_selection/`; they are not mixed into current curves.

```bash
python repro/corrected_evaluation.py --output outputs/derived_inputs
python repro/evaluate_corrected.py --case illustration_A_stationary_124 --verify-archive --output outputs/A_corrected
python repro/evaluate_corrected.py --case illustration_signals_B_123 --verify-archive --output outputs/B_corrected
python repro/evaluate_corrected.py --case correlated_C --seeds 1 --verify-archive --output outputs/C_corrected
python repro/plot_calibration_windows.py --output outputs/calibration_windows
```

`--seeds` takes the first N seeds in the protocol, without selecting by outcome.
The default is one seed. Horizon overrides are smoke checks and cannot claim
archive reproduction. Figure generation only reads the saved data; it does not
launch a full evaluation campaign.


```bash
# Selected 10,000-seed frontier: render archived outcomes.
python repro/frontier.py plot --output outputs/frontier
# Legacy protocol only: this smoke test does not evaluate the newly selected frontier.
python repro/frontier.py evaluate --seeds 1 --compare-archive --output outputs/frontier_smoke

# Frozen canonical B, including the three-action mask at inventory bounds.
python repro/plot_paper_heatmaps.py --verify-reference --output outputs/heatmaps

# Eleven GLFT panels, including finite-horizon numerical validation.
python repro/glft_figures.py --output outputs/glft_figures

# Historical calibration control: check all five calibration-window fits; optional bootstrap repeats 5,000 resamples.
python repro/glft_calibration.py --mode aggregate --bootstrap --output outputs/glft_aggregate
# Small simulator check, not a new choice of calibrated parameters.
python repro/glft_calibration.py --output outputs/glft_smoke

# Historical GLFT evaluation control: three original episodes; validates against archived arrays.
python repro/glft_evaluate.py --output outputs/glft_evaluation_smoke

# Verify the illustrative stationary A and auxiliary-signal B rollouts.
python repro/plot_historical_results.py --verify-rerun --output outputs/historical_results
python repro/plot_paper_episode.py --verify-rerun --output outputs/episode
```

The selected frontier archive and exact campaign protocol are in
`data/paper/frontier_10000/`; plotting checks every policy/seed pair and the
archived mean and standard deviation. Bootstrap standard errors are reused
from the completed campaign. The legacy evaluation command still defaults to
1,000 seeds and uses `config/frontier_fresh_seeds.json`; it does not implement
the selected campaign's C51/FIFO corrections or new GLFT parameters.

To repeat the historical calibration control, explicitly select
`--mode full --stage primary --workers 2 --output outputs/glft_calibration_full`.
The primary original/corrected comparison simulates 200 million events:
20 seeds × five requested distances × two protocols × one million events.
`--stage all` also repeats the four supplementary windows (600 million events
in total). These campaigns are not launched by `make figures` or the default
calibration command. That historical fit uses effective posting distances and
stationary `p_buy=0.5`; `sigma=0.3` remains the recorded volatility estimate.

`glft_evaluate.py --mode full --workers 2 --output outputs/glft_evaluation_full`
repeats all recorded GLFT benchmark episodes. Use repeated `--experiment NAME`
options to select experiments from `config/glft_experiments.json`. It uses the
legacy calibration; it does not recalibrate or rerun the DQN
asymmetry curve. Default smoke evaluations and full evaluations have separate
output locations.

`config/reproduction_manifest.json` records the supported source and input
hashes. `data/paper/figure_inventory.json` records the generator and provenance
of each selected figure. Numerical and image comparisons were
checked in `requirements-tested.txt`; exact pixels depend on plotting versions
and fonts. Figures and checkpoints are never replaced by a default command.

The selected trajectories and stress histograms now use complete corrected
numerical records. Their previous selections remain available as historical
controls. The regime schematic has an explicit vector reconstruction; the
bandit schematic derives from recovered TikZ, with the anchored C51 update
made explicit. These replacements are identified in the family provenance files.

Replotting archived outcomes does not repeat training or every Monte Carlo
campaign. The older A degradation and comparison-sweep outcomes were separately
regenerated with `recover_A_sweeps.py`: all 18 mean/std pairs agree with the
historical rounded aggregates. Individual terminal outcomes and paired A/B
differences are available in `data/paper/historical_results/A_sweeps/`.

## Historical evidence

Obsolete notebook harnesses, presentation patches and comparison runners have
been removed from the active package. Their retained outputs are numerical
inputs under `data/paper/`; use the supported commands above to reproduce them.

Frozen sources required to verify recorded campaigns remain within the data
archives, with original hashes. They are not alternative training or plotting
entry points. Unused members of the original whole-source evaluation snapshot
are identified by hash in `retired_unused_snapshot_members` in its provenance.
The original calibration estimator remains as a library because the calibration
comparison explicitly uses it as a historical control.

## Numerical stability in new evaluations

`evaluate_paper.py` defaults to `stable_float64_v1` and requires the environment
in `config/evaluation_environment.json`. Install the exact dependencies from
`requirements-tested.txt` with Python 3.13.12. The supported reference is CPU,
macOS arm64, one numerical thread. Other environments can be tested explicitly
with `--allow-environment-mismatch`, which records every mismatch.

```bash
make check-numerics
python repro/verify_numerical_stability.py --full-episode --output outputs/numerics_full
```

The full regression runs only B seed 500128 (500,000 events plus 10,000 warmup),
with frozen weights, and checks the previously established stable outcome 14.90.
It never forces an action and does not apply the separate FIFO intervention.
Output directories must be new. The quick suite can run in another PyTorch
version and records differences without claiming the environment is certified.

Historical `frontier.py evaluate` and `recover_A_sweeps.py` retain their
original float32 protocol. The published illustrative `--verify-rerun` checks
use `evaluate_corrected.py`, which applies both corrections to the archived
controller. `evaluate_paper.py` remains the separate stable-C51-only evaluator.
Replotting and heatmaps do not change stored outcomes or the definition of Q.
