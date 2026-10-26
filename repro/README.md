# Final-paper reproduction

Run commands from the repository root. Every entry point uses repository-local
inputs and writes new results under `outputs/` or an explicit `--output`.

| Task | Command |
| --- | --- |
| Validate final data and checkpoint identities | `make verify-paper` |
| Test frozen C51 ranking and inventory masks | `make check-numerics` |
| Regenerate all 32 figures | `make figures` |
| Compare regenerated figure pixels | `python repro/verify_paper.py --figures-dir outputs/paper_figures` |
| Check an independent export | `make check-portability` |
| Replot final frontier | `python repro/frontier.py plot --output outputs/frontier` |
| Replay one seed for each frontier policy | `python repro/frontier.py evaluate --seeds 1 --compare-archive --output outputs/frontier_replay` |
| Replay all 110,000 frontier outcomes | `python repro/frontier.py evaluate --seeds 10000 --compare-archive --output outputs/frontier_full` |
| Replay selected GLFT cases | `python repro/glft_evaluate.py --output outputs/glft_replay` |
| Replay full GLFT campaign | `python repro/glft_evaluate.py --mode full --output outputs/glft_full` |
| Replay final C stress case | `python repro/evaluate_corrected.py --case correlated_C --seeds 1 --verify-archive --output outputs/C_replay` |
| Inspect available final A/B/C cases | `python repro/evaluate_corrected.py --help` |
| Check calibration observation lifecycle | `python repro/glft_calibration.py --output outputs/calibration_check` |
| Recompute calibration-window sufficient-statistic fits | `python repro/glft_calibration.py --mode aggregate --output outputs/calibration_aggregate` |
| Repeat calibration-window probes | `python repro/glft_calibration.py --mode full --stage all --output outputs/calibration_probes` |

The calibration-window experiments use requested distances 0.5–2.5 ticks and
five observation windows. The selected benchmark calibration is the unweighted
0.5–5 tick fit in `data/paper/glft_regenerated/inputs/calibration_summary.json`;
`glft_figures.py` recomputes it from its pooled empirical observations. The
window aggregation command checks its own recorded fits; it does not replace
the selected benchmark parameters.

## Implementation

- `paper_runtime.py`, `checkpoint_io.py` and `paper_helpers.py` provide frozen
  model loading, signal configuration and simulator calls.
- `evaluate_corrected.py` replays the final A/B/C protocol;
  `paper_evaluation_corrections.py` applies stable C51 ranking and FIFO recovery.
- `corrected_evaluation.py` verifies 4,405 individual outcomes, 305 trajectories
  and their derived plotting arrays.
- `frontier.py` replays the final 10,000-seed frontier;
  `plot_frontier_10000.py` verifies and plots its final summary.
- `glft_evaluate.py` and `glft_evaluation_runtime.py` replay the final benchmark
  cases; `glft_figures.py` invokes `plot_glft_regenerated.py` for their panels.
- The remaining `plot_*` and `regen_*` scripts generate the panels selected by
  `make figures`. `figure_inventory.json` maps the manuscript panels to inputs.

The numerical fixtures test final-policy action ranking and masks. Full
simulation comparisons use recorded final outcomes. No previous model version,
source snapshot or superseded result set is needed by these commands.
