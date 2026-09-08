# Deep Learning of Robust Market Making under Regime-Switching Order Flow

Code and paper source for *Deep Learning of Robust Market Making under
Regime-Switching Order Flow* (Felipe Moret & Fabrizio Lillo, Scuola Normale
Superiore, Pisa).

Classical market-making strategies from stochastic control — Avellaneda–Stoikov
and the Guéant–Lehalle–Fernandez-Tapia (GLFT) extension — give closed-form
quoting rules but assume stationary order flow; under the regimes seen in real
markets (e.g. algorithmic execution of metaorders) they can post a negative PnL.
This work develops a deep reinforcement-learning market maker (**RLMM**), a
Rainbow-style distributional DQN (C51), calibrated and tested in a
zero-intelligence limit order book. In the stationary setting RLMM outperforms
GLFT across the entire observed risk–return frontier; augmenting its state with
a Bayesian online change-point filter over the directional flow bias and a
queue-adjusted quote-exposure imbalance restores profitability under
non-stationary flow, and a final scenario-bandit step further hardens it against
random-persistence and correlated-direction stress. Training proceeds in three
algorithms:

| Algorithm | Description | Entry point |
|-----------|-------------|-------------|
| **Algorithm A** | Stationary Rainbow-DQN controller | [`src/agents/DeepSarsaQRunner.py`](src/agents/DeepSarsaQRunner.py) |
| **Algorithm B** | Regime-aware fine-tuning (Bayesian change-point flow filter + quote-exposure imbalance) | [`src/agents/DeepSarsaQRunner_REGIME.py`](src/agents/DeepSarsaQRunner_REGIME.py) |
| **Algorithm C** | Scenario-bandit robust fine-tuning | [`src/agents/DeepSarsaQRunner_RANDOM_TAU.py`](src/agents/DeepSarsaQRunner_RANDOM_TAU.py) |

## Repository layout

```
robust-deep-market-making/
├── src/
│   ├── sim/      # LOB simulator, MM controller, config, Bayesian flow signal, plot palette
│   ├── agents/   # distributional DQN engine, fine-tuning machinery (ADR-lite, EWC, adversaries), A/B/C runners
│   ├── glft/     # GLFT policy factory + intensity / censored-waiting-time calibration + GLFT studies
│   └── eval/     # risk–return frontier, stress tests, degradation curves, AS-miss bundle
├── repro/        # scripts that regenerate figures from cached simulation outputs
├── paper/        # main.tex, references.bib, and the figures used in the paper
├── checkpoints/  # trained A/B/C checkpoints (see checkpoints/README.md)
├── data/         # small derived CSV/NPZ inputs (see data/README.md)
├── requirements.txt
└── Makefile
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The source modules are imported by top-level name, so the `src/` subdirectories
must be on `PYTHONPATH`. The `Makefile` sets this automatically; to run scripts
directly, export it once:

```bash
export PYTHONPATH="$PWD/src/sim:$PWD/src/agents:$PWD/src/glft:$PWD/src/eval"
```

## Usage

```bash
make paper       # compile paper/main.tex -> paper/main.pdf (requires tectonic)
make train-a     # train Algorithm A   (likewise: make train-b, make train-c)
make figures     # regenerate the data-driven figures
```

Algorithm B fine-tunes from an Algorithm-A checkpoint: set `WARMSTART_CKPT` in
[`src/agents/DeepSarsaQRunner_REGIME.py`](src/agents/DeepSarsaQRunner_REGIME.py)
to the desired checkpoint from `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md) for the A/B/C mapping).

## Reproducibility

- The trained A/B/C checkpoints behind the paper's results are included in
  `checkpoints/`.
- The figures in `paper/figures/` are those used in the manuscript. The
  data-driven figures regenerate from the committed checkpoints and cached
  simulation outputs through the scripts in `src/eval/` and `repro/`; the two
  schematic diagrams (`regime_model_diagram`, `scenario_bandit_diagram`) are
  illustrations.
- GLFT calibration constants used throughout: `A = 0.1507`, `κ = 2.335`,
  `σ = 0.30`.
- Evaluations use paired-seed episodes (identical order-flow realizations across
  policies) under a fixed global seed.

## Roadmap

- Package `src/` as an installable module with namespaced imports.
- Consolidate training, evaluation, and figure generation behind a single set of
  command-line entry points.

## Citation

If you use this code, please cite the paper; see [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
