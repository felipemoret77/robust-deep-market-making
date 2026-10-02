# Paper data

`paper/` contains compact derived arrays for the supported plotting commands:

| File | Contents |
|---|---|
| `tau_sweep_B_runs.csv` | 600 new B outcomes: 100 seeds for stationary flow and each of five durations |
| `tau_sweep_summary.csv` | New B means/SEM plus the original A and historical B aggregates |
| `regime_mc.npz` | Archived terminal outcomes and sampled PnL/inventory trajectories for 100 paired episodes |
| `bandit_random.npz`, `bandit_correlated.npz` | Archived B/C terminal PnL, 50 paired episodes per stress family |
| `data_manifest.json` | Source-cache hashes and hashes of the portable NPZ exports |
| `helper_provenance.json` | AST hashes of evaluation helper functions copied from the generating notebook |
| `heatmap_reference.json` | Original constructed-state record; exact heatmap checkpoint mapping is still pending |

NPZ files contain numerical arrays only and load with `allow_pickle=False`.
The original pickle caches and large event tapes remain outside this curated
set. Seed ranges, horizons and model hashes are recorded in
`config/paper_experiments.json`. Historical A sweep aggregates cannot recover
per-seed differences or paired confidence intervals on their own.
