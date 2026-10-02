# Paper checkpoint identities

`config/paper_experiments.json` records exact filenames, SHA-256 hashes, saved
metadata and parentage. Run `make verify-paper` before an evaluation.

## Algorithm A

The `deep_mm_mtm_pure_invp*_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt`
files cover the stationary inventory-penalty sweep. The `invp0.0010` checkpoint
is the A reference in the regime experiments. It is not a warm-start for B.

## Algorithm B

The canonical file is:

```
deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_factored_noise_fully_noisy_inv_wall_dampened_reward_cyc_lr_fill_quote_exposure_bayes_on2_mix_off_truncated_off_best_ma100.pt
```

SHA-256: `3a4649c860f91ffe2ca3747d071880508e953aa45f5a0d14d6b18cdfcff30200`.

This is a byte-preserving copy of the recovered 2026-05-27 checkpoint used for
the individual recovery results and the corrected persistence sweep. It is
restored under the original filename recorded in C's warm-start metadata.
B was trained from scratch. Its filter uses `USE_MIXTURE=False`,
`USE_TRUNCATED_BETA=False`, `BAYES_FLOW_PRIOR_MODE="beta"`, and an assumed
mean duration of 60 MOs.

The two files ending in `bayes_on2_mix_off_best_ma100.pt` and
`bayes_on2_mix_off_final.pt` are historical snapshots from a different run.
Their truncated Beta prior and their weights differ. They are preserved for
traceability and are not fallback choices for a paper evaluation.

## Algorithm C

The canonical file is:

```
deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_random_tau_curriculum_dro_tau_p256_b0p5_eps0p5_cap0p05_ramp0p5_pmix0p5__c_lr_fill_quote_exposure_bayes_on2_mix_off_truncated_off_final_hfb729bdc32.pt
```

C starts from the canonical B filename above and retains the same single-filter
configuration. Its metadata record the scenario pool, sampling schedule,
coefficient `ANCHOR_LAMBDA=0.001`, and other saved training settings.

## Storage

These small curated checkpoints are ordinary Git files, not Git LFS pointers.
New training outputs belong in `outputs/training/`. Do not overwrite a canonical
checkpoint to publish a later run under an old filename.
