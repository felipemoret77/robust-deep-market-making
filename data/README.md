# Final paper data

Only the selected paper data and the inputs needed to reproduce or verify it
are included. The protocols retain the exact recorded seeds, horizons and model
identities. Hashes identify the selected files; they are not extra result sets.

| Under `data/paper/` | Purpose |
| --- | --- |
| `frontier_10000/` | Final 110,000 outcomes and 11-point summary with bootstrap errors |
| `corrected_evaluation/` | Final A/B/C protocol, 4,405 outcomes and 305 trajectories |
| `glft_regenerated/` | Final GLFT calibration, benchmark outcomes and analytical inputs |
| `calibration/` | Final probe observations for the five calibration-window panels/checks |
| `calibration_windows_unweighted/` | Selected window points and per-seed normalization |
| `policy_results/` | Selected A trajectory, degradation results and B stress distributions |
| `episode/` | Selected illustrative episode and signal trajectories |
| `heatmaps/` | Selected B value-function inputs and verification records |
| `diagrams/` | Inputs for the selected architecture diagrams |
| `numerical_regression/` | Fixed C51 logits and state probes used by current numerical tests |
| Root `.npz` and sweep `.csv` files | Final derived arrays used by regime/sampling plots |

`make verify-paper` verifies checkpoint and data identities, individual outcomes,
summary statistics and figure coverage. `make figures` reconstructs all selected
panels without rerunning the Monte Carlo campaigns. Full and bounded replay
commands are documented in `repro/README.md`.
