# Canonical checkpoints

The trained checkpoints used to produce the paper's results, kept **flat** (with
their original filenames) so the hard-coded `checkpoints/<name>.pt` paths in the
eval scripts resolve when run from the repo root. Each file is small (~3–6 MB),
so they are committed as plain Git objects — **no Git LFS required**. Use LFS
(or Zenodo) only if you later add many more or larger checkpoints.

## Algorithm A — stationary DQN (risk–return frontier φ-sweep)

Loaded by [`src/eval/risk_return_frontier.py`](../src/eval/risk_return_frontier.py),
one per inventory-penalty φ (= `invp`):

```
deep_mm_mtm_pure_invp{0.0000,0.0010,0.0020,0.0030,0.0040,
                       0.0050,0.0060,0.0080,0.0100,0.1000}_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt
```

The **canonical single Algorithm-A** controller (stress tests, and the
warm-start source for Algorithm B) is **`invp0.0010`**.

## Algorithm B — regime-aware fine-tuning

```
deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_factored_noise_fully_noisy_inv_wall_dampened_reward_cyc_lr_fill_quote_exposure_bayes_on2_mix_off_final.pt
...bayes_on2_mix_off_best_ma100.pt
```

Loaded by `viz_checkpoint_mix_off.py` and the regime diagnostics.

## Algorithm C — scenario-bandit robust fine-tuning

```
deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_random_tau_curriculum_dro_tau_p256_b0p5_eps0p5_cap0p05_ramp0p5_pmix0p5__c_lr_fill_quote_exposure_bayes_on2_mix_off_truncated_off_final_hfb729bdc32.pt
```

(`dro_tau` is the in-code name of the scenario-bandit step; `p256 b0.5 eps0.5
cap0.05` are the bandit pool size / temperature / mixing / weight-cap.) Loaded by
the scenario-bandit stress figures.

## Not included

`risk_return_frontier.py` also references PPO/SAC baselines
(`ppo_mm_pure_final.pt`, `sac_mm_pure_ep0350.pt`); these are **not** part of the
paper's frontier figure and are not shipped here.
