# Robust Deep Market Making

Code, trained policies and numerical data for the final paper in
[`paper/main.tex`](paper/main.tex).

This repository contains the selected paper results and their reproduction
pipeline. Superseded results, exploratory runs and duplicate source snapshots
are excluded from the working tree.

## Layout

| Directory | Contents |
| --- | --- |
| `paper/` | Manuscript and its 32 selected figures |
| `src/sim/` | Limit-order-book simulator, signals and regime schedules |
| `src/agents/` | Algorithms A/B/C, shared C51 controller, checkpoint loader and C's DRO sampler |
| `src/glft/` | GLFT and at-best policies and calibration primitives |
| `repro/` | Final-policy evaluation, figure generation and verification |
| `config/` | Checkpoint identities, experiment settings and reference environment |
| `data/paper/` | Final outcomes, trajectories, calibration observations and plot inputs |
| `checkpoints/` | Seven selected paper checkpoints and two matched-control policies |

## Setup and figures

Use Python 3.13. The tested reference is macOS arm64, Python 3.13.12 and CPU
PyTorch 2.10.0; exact versions are in `requirements-tested.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt
make verify-paper
make check-numerics
make figures
make verify-paper  # validates the shipped inputs
python repro/verify_paper.py --figures-dir outputs/paper_figures
```

`make figures` regenerates all 32 panels from final numerical inputs. It does
not run training or repeat the full Monte Carlo campaigns. Outputs go to
`outputs/paper_figures/`. `make paper` compiles the manuscript with Tectonic.

## Repeat final evaluations

```sh
python repro/frontier.py evaluate --seeds 1 --compare-archive --output outputs/frontier_smoke
python repro/glft_evaluate.py --output outputs/glft_smoke
python repro/evaluate_corrected.py --case correlated_C --seeds 1 --verify-archive --output outputs/C_replay
```

These commands compare fresh simulator runs with the corresponding final
outcomes. The frontier uses 5,000 events plus 1,000 warmup events; the C stress
case uses 500,000 events plus 10,000 warmup events. Increasing `--seeds` repeats
more recorded seeds. The full frontier has 10,000 seeds per policy, 11 policies
and 110,000 outcomes. Its Algorithm A penalties are
`0, 0.001, 0.002, 0.005, 0.008`.

The frozen policies use stable float64 C51 action ranking from float32 logits
and preserve a valid unchanged recovery quote at the inventory boundary.
Training weights remain unchanged during evaluation. The selected GLFT
parameters are `A=0.11321677923549382`, `kappa=2.240448473943383`, `sigma=0.3`.

`make check-portability` exports the required files into a temporary directory,
blocks access to the original workspace, regenerates the figures and runs
representative final-policy replays.
Use `python repro/check_portability.py --skip-long-replays` for the figure,
compilation and bounded control checks without repeating the long historical
evaluation cases.

## Training

```sh
make train-a
make train-b
make train-c
```

Runs write to `outputs/training/`. The checked-in checkpoints are the selected
paper models; training again does not overwrite them or guarantee bit-identical
weights. Algorithm B starts from scratch; C warm-starts from the selected B.
The DRO sampler is part of C's training and is required.

## Matched uniform-sampling control

The additional control starts from the same B checkpoint and fine-tunes two
policies for 2,000 episodes each. The adaptive arm uses C's scenario weighting
and easiest-scenario replacement; the uniform arm samples uniformly and
replaces random pool members. All other settings are matched.

`config/uniform_control.json` records the training settings, final model hashes
and held-out evaluation seeds. The 400 individual outcomes and matched-run
provenance are in `data/paper/uniform_control/`.

```sh
make verify-uniform-control
```

This recomputes the reported PnL gains of 9.4% and 7.0% and checks both final
policies. They are additional controls, not replacements for the C checkpoint
used in the existing figures. The comparison uses one fine-tuning seed;
evaluation intervals do not measure variation across independent trainings.
Portable training and bounded replay commands are in `repro/README.md`.

See [`repro/README.md`](repro/README.md) for commands,
[`data/README.md`](data/README.md) for the final inputs, and
[`checkpoints/README.md`](checkpoints/README.md) for checkpoint identities.
