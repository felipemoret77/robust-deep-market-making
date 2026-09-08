# Canonical checkpoints

The three checkpoints used to produce the paper's results go here. They are
large (`*.pt`) and are **git-ignored** by default — publish them via
**Git LFS** or archive them on **Zenodo** and link the DOI here.

| Algorithm | Role | Load into |
|-----------|------|-----------|
| A | stationary DQN | `DeepSarsaQRunner.py` eval / warm-start source for B |
| B | regime-aware fine-tuning | `DeepSarsaQRunner_REGIME.py` (set `WARMSTART_CKPT` to the A checkpoint) |
| C | scenario-bandit fine-tuning | `DeepSarsaQRunner_RANDOM_TAU.py` |

Reference filenames from the development tree (drop the exact `.pt` here):

- **A** — `deep_mm_mtm_pure_invp0.0010_inv_wall_dampened_reward_clr200f0p30_final.pt`
- **B** — `deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_factored_noise_fully_noisy_inv_wall_dampened_reward_cyc_lr_fill_quote_exposure_bayes_on2_mix_off_final.pt` (plus its `_best_ma100.pt`)
- **C** — the scenario-bandit / random-τ checkpoint produced by `DeepSarsaQRunner_RANDOM_TAU.py`

To version with Git LFS:

```bash
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes checkpoints/*.pt
```
