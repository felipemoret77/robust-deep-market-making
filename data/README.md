# Derived data

Small, curated, regenerable artifacts that the figure/eval scripts read or
produce — e.g. the GLFT calibration bundle (`glft_calib_A_kappa_sigma.pkl`),
the misspecification-overlay CSV, and paired-frontier PnL arrays.

Keep this directory **small and derived only**. Heavy raw dumps and caches
(`*_sim_df`, `*_store.pkl`, `notebook_cache/`) are git-ignored and must be
regenerated from code, not committed.

Canonical GLFT calibration constants (also hard-coded in `src/glft/`):
`A = 0.1507`, `κ = 2.335`, `σ = 0.30`.
