# Deep Learning of Robust Market Making under Regime-Switching Order Flow

Code and paper source for Felipe Moret and Fabrizio Lillo's study of market
making in a zero-intelligence FIFO limit order book.

## Repository layout

| Directory | Purpose |
|---|---|
| `src/agents/` | A/B/C training, Rainbow-DQN, checkpoint loading and C's scenario sampler |
| `src/sim/` | Santa Fe FIFO simulator, Bayesian signals and regime schedules |
| `src/glft/` | GLFT and at-best policies and calibration libraries |
| `repro/` | Supported figure generators, evaluation commands and verification |
| `config/` | Selected experiments, environment and source hashes |
| `data/paper/` | Numerical inputs, protocols and historical source evidence |
| `checkpoints/` | Paper model weights and their provenance |
| `paper/` | Manuscript and its selected figures |

Start with `make help`; the command inventory is in
[`repro/README.md`](repro/README.md). Research-only PPO, SAC, ADR, EWC,
legacy adversaries, QRM and obsolete plotting runners are excluded from
the active source tree. Algorithm C's `DRO_adversary.py` is retained because
its scenario sampler is part of the paper.

Some original source snapshots under `data/paper/` are retained to audit
recorded experiments. They are immutable historical evidence, not supported
entry points. The hashes of unused members removed from the corrected
evaluation snapshot remain in its provenance record. The active-file cleanup
inventory is [`config/repository_cleanup.json`](config/repository_cleanup.json).

| Algorithm | Training | Paper checkpoint |
|---|---|---|
| A | Stationary Rainbow-DQN; inventory-penalty sweep | `invp0.0010` for the regime comparisons; other A checkpoints for the frontier |
| B | Trained separately under regime-switching flow, using Bayesian flow belief and quote exposure | `mix_off_truncated_off_best_ma100` |
| C | Fine-tunes B with adaptive scenario sampling and L2 anchoring | `dro_tau_..._hfb729bdc32` |

B does not warm-start from A. B and C use a single Beta--Bernoulli filter with
assumed mean duration 60 MOs, history limit 300, and age normalization
`log(1 + age) / log(301)`, bounded at one. The filter scale stays fixed when
the environment's regime duration changes.

The authoritative checkpoint paths, SHA-256 hashes, saved training metadata
and evaluation settings are in [`config/paper_experiments.json`](config/paper_experiments.json).
The checkpoint history is documented in [`checkpoints/README.md`](checkpoints/README.md).

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt
make verify-paper
make check-numerics
```

The reference interpreter is Python 3.13.12 on macOS arm64 (CPU).
The environment used for the reproduction checks is recorded in
[`requirements-tested.txt`](requirements-tested.txt). It is an environment
record, not a claim that every older training run used those versions.
The Makefile supplies the flat-module import paths. The supported evaluation
and plotting scripts below also resolve their inputs from the repository root;
they do not require the author's Desktop directories.

## Reproduce results

Regenerate the supported figures from the shipped numerical data and frozen B:

```bash
make figures
# Output: outputs/paper_figures/; existing paper figures are preserved.
```

This includes the fresh-seed frontier, corrected GLFT calibration and benchmark
panels, finite-horizon analytical GLFT curves starting from zero inventory,
masked value-function heatmaps, the persistence sweep, and the archived
Monte Carlo/stress panels. Heatmaps are recomputed from the canonical B
checkpoint; empirical figures use archived outcomes. This command does not
repeat training or the large simulation campaigns.

The selected GLFT calibration and benchmark protocol are recorded in
[`data/paper/glft_regenerated/`](data/paper/glft_regenerated/). The calibration
gives `A=0.11321677923549382`, `kappa=2.240448473943383`, and `sigma=0.3`.
It uses stationary symmetric flow (`p_buy=0.5`), actual probe lifetimes,
censoring, an unweighted log-linear fit, and the recorded finite-window
conversion. The archive contains 4,810 new GLFT episodes and 1,000 reused
naive episodes. Finite-horizon PnL curves start at zero inventory; stationary
inventory diagnostics are identified separately. The earlier
`config/glft_experiments.json` is retained for historical evaluation controls.
See [`repro/README.md`](repro/README.md) for commands.

The current A/B/C panels use the complete corrected evaluation in
[`data/paper/corrected_evaluation/`](data/paper/corrected_evaluation/):
4,405 individual outcomes and 305 trajectories, with stable C51 action ranking
and FIFO preservation. `python repro/corrected_evaluation.py` reconstructs and
checks every derived plotting input against these records. Both A and B sweep
curves use the corrected individual outcomes.

The calibration-window comparison recomputes each seed's normalization with
unweighted OLS and reports the equal-seed mean with one SEM. Its compact probe
statistics are reused; the selected GLFT parameters are unchanged.


Test that an exported copy works without the original research directories:

```bash
make check-portability
```

This copies only the reproduction package to a temporary directory, blocks
access to the original repository/Desktop, verifies input hashes, regenerates
and checks the supported figures, and runs small frozen-policy/calibration
checks. Pixel comparisons use the
recorded environment; fonts and library versions can affect rendering elsewhere.

Replay the selected corrected evaluations into a new output directory:

```bash
# Full recorded horizon for the first seed; compare all saved metrics/traces.
python repro/evaluate_corrected.py --case illustration_A_stationary_124 --verify-archive --output outputs/A_example
python repro/evaluate_corrected.py --case illustration_signals_B_123 --verify-archive --output outputs/B_example
python repro/evaluate_corrected.py --case correlated_C --seeds 1 --verify-archive --output outputs/C_check
# Full corrected B sweep point: 100 episodes of 100,000 events.
python repro/evaluate_corrected.py --case sweep_B_tau60 --seeds 100 --verify-archive --output outputs/B_tau60
```

`--case` selects an exact checkpoint, seed list, scenario construction, horizon,
and warmup from the recorded protocol. `--events` and `--warmup` are available
for smoke checks; they cannot be combined with `--verify-archive`. Weights stay
frozen, with CPU inference and one numerical thread. This entry point applies
both evaluation corrections used by the selected A/B/C figures.

`evaluate_paper.py` remains available for stable-C51-only comparisons and explicit
historical controls. It does **not** apply the FIFO intervention, so it is not the
replay command for the newly selected panels. Its `stable_float64_v1` mode uses
double precision for softmax and action ranking; heatmaps and training keep
their own value calculation.

Use `--c51-mode legacy_float32` only for an explicit historical control.
The legacy controller is archived with its original hash; archived results
are not relabelled as stable evaluations. `--allow-environment-mismatch` permits
an intentional, recorded compatibility test; it does not certify a new environment.
Run `make check-numerics` before accepting an environment update. See
[`repro/numerical_stability.txt`](repro/numerical_stability.txt) for validation
and which results may require new evaluation. No precision setting guarantees
bitwise equivalence across all PyTorch versions or hardware.

Paired seeds do not imply identical realized order flow: policy actions change
the book and can change subsequent trajectories.

```bash
make paper                 # requires tectonic
make train-b               # separate regime training
make train-c               # fine-tuning from canonical B
```

Training writes under `outputs/training/`, preserving the shipped checkpoints.
The runners exclude unused experimental curricula and post-training animations.
Saved checkpoint
metadata are the record of the published training configuration; the checks
above establish frozen-policy reproduction, not bitwise reproduction of an
entire new training run.

## Scope and remaining checks

- The corrected sweep uses the same single-filter B as the individual recovery
  results. Both curves use the complete corrected per-episode archive.
- Two older B files remain for provenance. They have different weights and a
  truncated prior and are not selected by the canonical evaluator.
- The corrected heatmaps use canonical B, the constructed reference state,
  and the same action mask as the controller. Their archived arrays and
  provenance are under `data/paper/heatmaps/`.
- The selected frontier uses 10,000 fresh paired seeds per policy, outside A's training
  seed range, with 5,000 evaluation events plus 1,000 warmup events. Its 110,000
  outcomes, new unweighted GLFT calibration, and bootstrap errors are archived
  under `data/paper/frontier_10000/`. The other GLFT panels use the same calibration, archived under `data/paper/glft_regenerated/`.
- Analytical GLFT PnL uses finite-horizon occupation starting at inventory
  zero. Numerical validation does not make this reduced model an exact
  representation of the FIFO simulator.
- [`data/paper/figure_inventory.json`](data/paper/figure_inventory.json)
  identifies every selected figure, its hash, and its reproduction status.
  All 32 selected panels have portable generators. `make figures` uses shipped
  numerical records, model inference, analytical calculations, and vector
  diagram sources. It does not repeat every simulation or training campaign.
- Current episode traces, sweep curves and stress outcomes use complete corrected
  evaluations. Previous selections are retained separately for historical checks.
  Both diagrams have vector sources with reconstruction provenance.
- Each B/C stress comparison contains 100 paired episodes. These describe
  empirical mean and dispersion, not reliable rare-loss probabilities.
- Other scripts and the notebook under `repro/`, and general studies under
  `src/eval/`, are historical research entry points. Some require external
  caches or old local paths. They are not substitutes for the supported
  commands above. [`repro/README.md`](repro/README.md) lists their status.

## Layout

- `src/`: simulator, controller, training runners and analytical benchmarks.
- `config/`: paper model identities and experiment settings.
- `checkpoints/`: canonical models plus explicitly identified historical files.
- `data/paper/`: compact numerical inputs and provenance for supported replots.
- `repro/`: supported evaluation/verification commands and historical diagnostics.
- `paper/`: manuscript and selected figures.
- `outputs/`: generated results, excluded from Git.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for citation information and
[`LICENSE`](LICENSE) for the MIT license.
