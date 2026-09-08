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

| Algorithm | What it is | Entry point |
|-----------|------------|-------------|
| **Algorithm A** | Stationary Rainbow-DQN controller | [`src/agents/DeepSarsaQRunner.py`](src/agents/DeepSarsaQRunner.py) |
| **Algorithm B** | Regime-aware fine-tuning (Bayesian change-point flow filter + quote-exposure imbalance) | [`src/agents/DeepSarsaQRunner_REGIME.py`](src/agents/DeepSarsaQRunner_REGIME.py) |
| **Algorithm C** | Scenario-bandit robust fine-tuning | [`src/agents/DeepSarsaQRunner_RANDOM_TAU.py`](src/agents/DeepSarsaQRunner_RANDOM_TAU.py) |

## Repository layout

```
robust-deep-market-making/
├── src/
│   ├── sim/      # LOB simulator, MM controller, config, Bayesian flow signal, plot palette
│   ├── agents/   # distributional DQN engine, fine-tuning machinery (ADR-lite, EWC, adversaries), A/B/C runners
│   ├── glft/     # GLFT policy factory + intensity/censored-waiting-time calibration + GLFT studies
│   └── eval/     # risk–return frontier, stress tests, degradation curves, AS-miss bundle
├── repro/        # figure-reproduction helpers (WIP — see caveats below)
├── paper/        # main.tex, references.bib, figures/ (the 33 figures used by the paper)
├── checkpoints/  # canonical A/B/C checkpoints (see checkpoints/README.md — add via Git LFS/Zenodo)
├── data/         # small derived CSV/NPZ (see data/README.md)
├── requirements.txt
└── Makefile
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Important: flat imports need `PYTHONPATH`

The source modules currently use **flat imports** (`import CONFIG_MM`,
`from MM_LOB_SIM import ...`). They are grouped into `src/{sim,agents,glft,eval}`
for readability, so **every `src/` subfolder must be on `PYTHONPATH`**. The
`Makefile` does this for you; to run scripts by hand:

```bash
export PYTHONPATH="$PWD/src/sim:$PWD/src/agents:$PWD/src/glft:$PWD/src/eval"
python src/agents/DeepSarsaQRunner.py        # Algorithm A
```

(Refactoring these into a proper installable package with clean namespaced
imports is planned — see "Roadmap".)

## Reproduce

```bash
make paper       # compile paper/main.tex -> paper/main.pdf (needs tectonic)
make train-a     # train Algorithm A   (likewise train-b / train-c)
make figures     # regenerate the paper figures that have turnkey generators
```

Building the paper requires [tectonic](https://tectonic-typesetting.github.io/).

## Reproducibility caveats (please read before regenerating figures)

- **`repro/` scripts are not publication-clean.** They use absolute macOS
  paths, address Jupyter cells by index, and read pickle caches. They are kept
  because they are what currently regenerates several figures; they are the
  first thing slated for the clean rewrite (see Roadmap).
- **Seven figures have no turnkey generator yet**: `total_pnl_0001.png`,
  `MM_inv_0001.png`, the three interpretability heatmaps
  (`p_buy_mhat_vs_inv.png`, `bayes_m_vs_spread.png`, `inventory_vs_spread.png`)
  and the two schematic diagrams (`regime_model_diagram.png`,
  `scenario_bandit_diagram.png`). They are versioned as-is. The heatmaps and the
  Bayesian-belief diagnostic come from the trainer at a configuration that is
  not byte-reproducible from a standalone script, so their flow-bias label was
  corrected by raster surgery (`repro/relabel_iota_*.py`) rather than re-render.
- **Freeze the exact Algorithm-B config before relying on it.**
  `src/agents/DeepSarsaQRunner_REGIME.py` currently ships with
  `WARMSTART_CKPT=None`, whereas the paper describes Algorithm B as fine-tuning
  from Algorithm A. Set `WARMSTART_CKPT` to the Algorithm-A checkpoint to match
  the paper's results.
- Canonical GLFT calibration (fixed in code): `A = 0.1507`, `κ = 2.335`,
  `σ = 0.30`.

## Roadmap (clean-up before public release)

1. Turn `src/` into an installable package (`pip install -e .`) with namespaced
   imports, dropping the `PYTHONPATH` requirement.
2. Replace `repro/` with clean, path-free generators:
   `scripts/train_algorithm_{a,b,c}.py`, `scripts/evaluate_{stationary,regimes}.py`,
   `scripts/make_figures.py`, `scripts/build_paper.py`.
3. Add turnkey generators for the seven figures listed above.
4. Publish canonical checkpoints via Git LFS or Zenodo (see `checkpoints/README.md`).

## Citation

See [`CITATION.cff`](CITATION.cff).
