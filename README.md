# Deep Learning of Robust Market Making under Regime-Switching Order Flow

Code and paper source for Felipe Moret and Fabrizio Lillo's study of market
making in a zero-intelligence FIFO limit order book.

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make verify-paper
```

The environment used for the reproduction checks is recorded in
[`requirements-tested.txt`](requirements-tested.txt). It is an environment
record, not a claim that every older training run used those versions.
The Makefile supplies the flat-module import paths. The supported evaluation
and plotting scripts below also resolve their inputs from the repository root;
they do not require the author's Desktop directories.

## Reproduce results

Redraw six empirical figures from the shipped numerical data:

```bash
make figures
# Output: outputs/paper_figures/; existing paper figures are preserved.
```

This covers the corrected persistence sweep, three regime Monte Carlo panels,
and the two B/C stress histograms. It replots archived data; it does not rerun
training or simulation. See [`data/README.md`](data/README.md) for provenance.

Run new frozen-policy evaluations, each into a new output directory:

```bash
python repro/evaluate_paper.py --algorithm B --scenario sweep --output outputs/B_sweep
python repro/evaluate_paper.py --algorithm B --scenario recovery --output outputs/B_recovery
python repro/evaluate_paper.py --algorithm C --scenario random --output outputs/C_random
python repro/evaluate_paper.py --algorithm C --scenario correlated --output outputs/C_correlated
```

Defaults match the recorded experiment horizons and seed ranges. A full sweep
runs 600 episodes; `--seeds 1 --tau 60` selects a smaller sweep check. Algorithms
A/B/C share the same evaluator. Checkpoints are frozen and inference runs on CPU.
Each evaluation saves per-episode outcomes and the exact configuration.
Paired seeds do not imply identical realized order flow: policy actions change
the book and can change subsequent trajectories.

```bash
make paper                 # requires tectonic
make train-b               # separate regime training
make train-c               # fine-tuning from canonical B
```

Training writes under `outputs/training/`, preserving the shipped checkpoints.
The runner sources contain additional research options. Saved checkpoint
metadata are the record of the published training configuration; the checks
above establish frozen-policy reproduction, not bitwise reproduction of an
entire new training run.

## Scope and remaining checks

- The corrected sweep uses the same single-filter B as the individual recovery
  results. Its A curve is preserved from historical aggregates; A was not rerun
  as part of the checkpoint replacement.
- Two older B files remain for provenance. They have different weights and a
  truncated prior and are not selected by the canonical evaluator.
- The heatmap generation record identifies a `mix_on_final` variant and a
  constructed reference state. Its exact checkpoint-to-figure mapping remains
  to be verified; these panels are not regenerated or silently attributed to
  the canonical B by the commands above.
- The analytical GLFT appendix has a separate validation under
  `repro/appendix_b_validation/` in the local research workspace. Its reduced
  model should not be read as a quantitatively exact FIFO model.
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
