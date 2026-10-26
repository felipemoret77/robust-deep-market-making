# Paper data

`paper/` contains compact derived arrays for the supported plotting commands:

| File | Contents |
|---|---|
| `tau_sweep_B_runs.csv` | 600 new B outcomes: 100 seeds for stationary flow and each of five durations |
| `tau_sweep_summary.csv` | Corrected A/B means and SEM from 100 individual outcomes per condition |
| `regime_mc.npz` | Archived terminal outcomes and sampled PnL/inventory trajectories for 100 paired episodes |
| `bandit_random.npz`, `bandit_correlated.npz` | B/C terminal PnL, 100 paired episodes per stress family; all 100 pairs evaluated with stable C51 and FIFO preservation |
| `bandit_100pairs/` | Historical 100-pair extension; superseded by corrected_evaluation for selected figures |
| `corrected_evaluation/` | Exact protocol, 4,405 individual outcomes, 305 sampled/full trajectories, executed-source hashes and previous selections |
| `calibration_windows_unweighted/` | Corrected per-seed OLS normalization, means and SEM from the archived probe statistics |
| `frontier_10000/` | Selected frontier: 110,000 outcomes, 10,000 per policy, with 5,000 events per episode |
| `data_manifest.json` | Input hashes and provenance of the current portable NPZ exports |
| `helper_provenance.json` | AST hashes of evaluation helper functions copied from the generating notebook |
| `heatmap_reference.json`, `heatmaps/` | Canonical B identity, constructed state, and final arrays with the controller's action mask |
| `frontier/` | 11,000 frozen-policy outcomes with seeds outside A's training range and the exact evaluation configuration |
| `glft/` | Historical calibration and benchmark archive, retained as a control |
| `glft_regenerated/` | Selected calibration, 4,810 new GLFT outcomes, reused naive traces, analytical validation and generating sources |
| `historical_results/` | Current corrected stationary A trajectory, A degradation summaries and B stress outcomes |
| `episode/` | Current corrected A/B episode traces and B auxiliary-signal illustration |
| `diagrams/` | Explicit vector sources and provenance for both explanatory diagrams |
| `figure_inventory.json` | Selected manuscript image hashes and the generator/status of each panel |

`config/reproduction_manifest.json` verifies the current sources, configuration,
and data inputs together. Family-specific provenance records identify the
original campaigns without requiring their local directories at runtime.

NPZ files contain numerical arrays only and load with `allow_pickle=False`.
The original pickle caches and large event tapes remain outside this curated
set. Seed ranges, horizons and model hashes are recorded in
`config/paper_experiments.json`. Regenerated A sweep outcomes and paired A/B differences are archived in
`historical_results/A_sweeps/` as historical controls. Current selected A/B
curves use the complete corrected records in `corrected_evaluation/`.
`python repro/corrected_evaluation.py --output outputs/derived_inputs` reconstructs
the selected NPZ/CSV inputs and verifies them against the shipped exports.
