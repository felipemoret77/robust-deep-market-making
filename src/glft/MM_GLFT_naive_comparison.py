#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MM_GLFT_naive_comparison.py — Multi-Policy Market-Making Comparison Runner

This script performs a Monte Carlo comparison of up to four market-making
policies on the same LOB simulator. It calibrates the GLFT model parameters,
builds each policy, runs N independent simulations per policy, and produces
summary plots (mean +/- 1 std band) for TotalPnL and Inventory over time.

Policies compared:
    1. GLFT (Lehalle/Gueant)   — Optimal MM with calibrated A, kappa, sigma.
    2. Always-Best (naive)     — Optional baseline at L1 best bid/ask.
    3. DeepRL (trained DQN)    — Optional; loaded from checkpoint.
    4. REINFORCE (trained PG)  — Optional; loaded from checkpoint.

Pipeline:
    1. Calibrate GLFT parameters:
       - (A, kappa) via censored waiting times + log-linear fit
       - sigma via fit_volatility
    2. Build each policy with the same LOB parameters
    3. Run N_SIMS simulations per policy with deterministic seeding
    4. Stack results into (n_runs, T) matrices
    5. Plot mean +/- 1 std for TotalPnL(t) and Inventory(t)

Seeding strategy:
    Each policy gets a DIFFERENT random seed per run, so policies face
    independent LOB realizations. This gives unbiased Monte Carlo estimates
    but with higher variance than paired comparisons. The seed scheme is:
        seed = SEED_BASE + run_id
    All policies share the same seed for a given run_id, ensuring they face
    the SAME LOB realization. This paired design reduces variance when
    comparing performance differences between policies.

DeepRL / REINFORCE toggles:
    USE_DEEPRL and USE_REINFORCE control which policies are included.
    Checkpoint paths are auto-selected based on USE_DEEPRL_PURE_MM and
    USE_REINFORCE_PURE_MM flags:
        pure_mm=False  -> *_generic_final.pt
        pure_mm=True   -> *_pure_final.pt

Created on Tue Feb  3 23:53:36 2026
@author: felipemoret
"""

import os
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MM_LOB_SIM import simulate_LOB_with_MM
from GLFT_policy_factory import glft_policy_factory
from calibrate_trading_intensity import fit_volatility
from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)

# Adjust this import to your file name:
from MM_policy_5 import always_best_bid_ask_mm_policy_factory

import torch

from dqn_distributional_with_throttle import DeepRLController
from reinforce_general import make_reinforce_eval_controller
from actor_critic import make_ac_eval_controller
from sac import SACDiscreteController
from ppo import PPOController

# ============================================================
# 0) Policy toggles & checkpoint dictionaries (EVAL)
# ============================================================

USE_ALWAYS_BEST = False

# ── DQN checkpoints ────────────────────────────────────────────
# Map: display label → checkpoint path.
DQN_CHECKPOINTS: Dict[str, str] = {
    "DQN SC (φ=0.002)": "checkpoints/deep_mm_mtm_pure_invp0.0010_final.pt",
    #"DQN SC (φ=0.005)": "checkpoints/deep_mm_mtm_pure_invp0.0050_final.pt",
}

DQN_TIME_THROTTLE = 1.0

# ── SAC checkpoints ────────────────────────────────────────────
SAC_CHECKPOINTS: Dict[str, str] = {
    #"SAC (φ=0.001)": "checkpoints/sac_mm_pure_ep0350.pt",
}

SAC_TIME_THROTTLE = 1.0

# ── PPO checkpoints ────────────────────────────────────────────
PPO_CHECKPOINTS: Dict[str, str] = {
    #"PPO (φ=0.001)": "checkpoints/ppo_mm_pure_final.pt",
}

PPO_TIME_THROTTLE = 1.0

# ── REINFORCE checkpoints ──────────────────────────────────────
REINFORCE_CHECKPOINTS: Dict[str, str] = {
    #"REINFORCE": "checkpoints/reinforce_pure_final.pt",
}

# ── A2C checkpoints ────────────────────────────────────────────
A2C_CHECKPOINTS: Dict[str, str] = {
    #"A2C": "checkpoints/actor_critic_mm_pure_final.pt",
}

# Directory where trained checkpoints are stored:
CKPT_DIR = Path("checkpoints")


def make_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_spyder/deep_mm_eval",
    *,
    use_time_update: Optional[bool] = None,
    min_time_interval: Optional[float] = None,
    inv_limit_override: Optional[int] = None,
):
    """
    Load a DeepRLController from a training checkpoint and configure it
    for pure evaluation (no learning, greedy actions).

    The checkpoint is expected to contain:
        ckpt["q_net"]         — state_dict of the Q-network
        ckpt["target_net"]    — state_dict of the target network (optional)
        ckpt["meta"]          — dict with training config:
            meta["best_params"]      — hyperparameters used during training
            meta["USE_PURE_MM"]      — whether training used pure-MM action space
            meta["pure_mm_offsets"]   — list of (bid_off, ask_off) tuples

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt checkpoint file.
    log_dir : str
        TensorBoard log directory (unused in eval, but required by constructor).

    Returns
    -------
    DeepRLController
        Controller in eval mode with loaded weights, epsilon=0, learning disabled.
    """
    # weights_only=True is safe if checkpoint meta contains only Python builtins.
    # If you ever get missing meta/best_params, set weights_only=False.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    meta = ckpt.get("meta", {})
    bp = meta.get("best_params", {})

    # Whether training was pure-mm or generic is saved in the checkpoint meta.
    USE_PURE_MM = bool(meta.get("USE_PURE_MM", False))
    pure_mm_offsets = meta.get("pure_mm_offsets", None)

    # Auto-detect architecture / feature flags from checkpoint metadata
    # or raw state-dict keys.
    #
    # Important:
    # - Standard trunk checkpoint:
    #       feature.0.weight / feature.0.bias
    # - Fully-noisy trunk checkpoint:
    #       feature.0.w_mu / feature.0.w_sigma / ...
    #
    # The eval factory must rebuild the controller with the SAME trunk
    # architecture as training, otherwise load_state_dict() fails with
    # "missing feature.0.weight / unexpected feature.0.w_mu".
    q_sd = ckpt.get("q_net", {})
    ckpt_has_noisy_params = any((".w_mu" in k) or (".w_sigma" in k) for k in q_sd.keys())
    ckpt_is_fully_noisy = any(k.startswith("feature.") and ".w_mu" in k for k in q_sd.keys())

    # Auto-detect flow/imbalance feature flags from checkpoint metadata or,
    # for older checkpoints, from the first-layer input dimension.
    use_flow_signal = bool(meta.get("USE_FLOW_EWMA", False))
    use_fast_flow_signal = bool(meta.get(
        "USE_FAST_FLOW_EWMA",
        meta.get("USE_MO_FLOW_FAST_EWMA", False),
    ))
    use_bayes_flow_signal = bool(meta.get("USE_BAYES_FLOW_SIGNAL", False))
    bayes_flow_feature_keys = meta.get("BAYES_FLOW_FEATURE_KEYS", None)
    if bayes_flow_feature_keys is not None:
        bayes_flow_feature_keys = list(bayes_flow_feature_keys)
    elif use_bayes_flow_signal:
        # Phase-B regime agents trained with the current runner use these two
        # features. Older checkpoints without explicit metadata should not
        # silently fall back to the controller's four-feature Bayes default.
        bayes_flow_feature_keys = ["bayes_m_hat", "bayes_expected_run_length"]
    use_fill_imbalance = bool(meta.get("USE_FILL_IMBALANCE", False))
    if (
        not use_flow_signal
        or not use_fast_flow_signal
        or not use_bayes_flow_signal
        or not use_fill_imbalance
    ) and q_sd:
        first_weight_key = (
            "feature.0.weight" if "feature.0.weight" in q_sd else "feature.0.w_mu"
        )
        if first_weight_key in q_sd:
            actual_input_dim = q_sd[first_weight_key].shape[1]
            if USE_PURE_MM:
                max_offset = max(max(abs(b), abs(a)) for b, a in pure_mm_offsets) if pure_mm_offsets else 1
                expected_dim = 2 + 2 * (max_offset + 1)
            else:
                expected_dim = 6
            extra = actual_input_dim - expected_dim
            explicit_extra = (
                int(use_flow_signal)
                + int(use_fast_flow_signal)
                + (len(bayes_flow_feature_keys or []) if use_bayes_flow_signal else 0)
                + int(use_fill_imbalance)
            )

            if "USE_BAYES_FLOW_SIGNAL" not in meta and "BAYES_FLOW_FEATURE_KEYS" not in meta:
                # Legacy checkpoints before Bayes metadata used the extra dims
                # in this order: slow flow, fast flow, fill imbalance.
                known_extra = int(use_flow_signal) + int(use_fast_flow_signal) + int(use_fill_imbalance)
                if extra >= 1 and not use_flow_signal:
                    use_flow_signal = True
                if extra >= 2:
                    if "USE_FILL_IMBALANCE" in meta and meta.get("USE_FILL_IMBALANCE") and not use_fill_imbalance:
                        use_fill_imbalance = True
                    elif not use_fast_flow_signal and known_extra < extra:
                        use_fast_flow_signal = True
                    elif not use_fill_imbalance and known_extra < extra:
                        use_fill_imbalance = True
                if extra >= 3 and not use_fill_imbalance:
                    use_fill_imbalance = True
                explicit_extra = int(use_flow_signal) + int(use_fast_flow_signal) + int(use_fill_imbalance)

            if extra > 0:
                print(f"[Auto-detect] use_flow_signal={use_flow_signal}, "
                      f"use_fast_flow_signal={use_fast_flow_signal}, "
                      f"use_bayes_flow_signal={use_bayes_flow_signal}, "
                      f"bayes_flow_feature_keys={bayes_flow_feature_keys}, "
                      f"use_fill_imbalance={use_fill_imbalance} "
                      f"(input_dim={actual_input_dim} vs base={expected_dim})")
            if explicit_extra != extra:
                raise RuntimeError(
                    "Checkpoint feature metadata does not match the Q-network input dimension: "
                    f"actual_input_dim={actual_input_dim}, base_dim={expected_dim}, "
                    f"expected_extra={explicit_extra}, observed_extra={extra}, "
                    f"use_flow_signal={use_flow_signal}, "
                    f"use_fast_flow_signal={use_fast_flow_signal}, "
                    f"use_bayes_flow_signal={use_bayes_flow_signal}, "
                    f"bayes_flow_feature_keys={bayes_flow_feature_keys}, "
                    f"use_fill_imbalance={use_fill_imbalance}."
                )

    # Rebuild controller with the same hyperparameters used during training.
    # (LR, replay_capacity, etc. are irrelevant in eval but required by the constructor.)
    ctrl_use_time_update = (
        bool(bp.get("use_time_update", False))
        if use_time_update is None
        else bool(use_time_update)
    )
    ctrl_min_time_interval = (
        float(bp.get("min_time_interval", 1.0))
        if min_time_interval is None
        else float(min_time_interval)
    )
    ctrl_inv_limit = bp.get("inv_limit", None)
    if inv_limit_override is not None:
        ctrl_inv_limit = int(inv_limit_override)

    ctrl = DeepRLController(
        level_offset=0,
        n_actions=(len(pure_mm_offsets) if USE_PURE_MM else int(bp.get("n_actions", 6))),

        gamma=float(bp.get("gamma", 0.99)),
        lr=float(bp.get("lr", 1e-4)),
        epsilon_start=float(bp.get("epsilon_start", 0.0)),
        epsilon_min=float(bp.get("epsilon_min", 0.0)),
        epsilon_decay=float(bp.get("epsilon_decay", 1.0)),
        batch_size=int(bp.get("batch_size", 128)),
        replay_capacity=int(bp.get("replay_capacity", 1_000)),
        target_update_steps=int(bp.get("target_update_steps", 1000)),

        use_sarsa=bool(bp.get("use_sarsa", False)),
        use_double=bool(bp.get("use_double", True)),
        use_dueling=bool(bp.get("use_dueling", False)),
        use_prioritized_experience=bool(bp.get("use_per", False)),
        use_noisy_net=bool(bp.get("use_noisy_net", False) or ckpt_has_noisy_params),
        use_fully_noisy=bool(ckpt_is_fully_noisy),

        n_neurons=int(bp.get("n_neurons", 128)),
        n_hidden=int(bp.get("n_hidden", 1)),
        n_steps=int(bp.get("n_steps", 1)),

        # PER schedule: use actual training values from checkpoint, not defaults.
        per_alpha_start=float(bp.get("per_alpha_start", 0.6)),
        per_alpha_end=float(bp.get("per_alpha_end", 0.4)),
        per_alpha_last_episode=int(meta.get("per_alpha_last_episode", meta.get("N_EPISODES", 100))),
        per_beta_start=float(bp.get("per_beta_start", 0.4)),
        per_beta_end=float(bp.get("per_beta_end", 1.0)),
        per_beta_last_episode=int(meta.get("per_beta_last_episode", meta.get("N_EPISODES", 100))),

        # Distributional DQN (C51) settings must match training architecture.
        use_distributional=bool(bp.get("use_distributional", False)),
        v_min=float(bp.get("dist_v_min", bp.get("v_min", -10.0))),
        v_max=float(bp.get("dist_v_max", bp.get("v_max", 10.0))),
        atoms=int(bp.get("dist_atoms", bp.get("atoms", 51))),

        log_dir=log_dir,

        pure_mm=USE_PURE_MM,
        inv_limit=ctrl_inv_limit,
        pure_mm_offsets=pure_mm_offsets,

        # Evaluation throttle can be forced independently from the checkpoint.
        use_time_update=ctrl_use_time_update,
        min_time_interval=ctrl_min_time_interval,

        use_mdp=True,
        use_flow_signal=use_flow_signal,
        use_fast_flow_signal=use_fast_flow_signal,
        use_bayes_flow_signal=use_bayes_flow_signal,
        bayes_flow_feature_keys=bayes_flow_feature_keys,
        use_fill_imbalance=use_fill_imbalance,
    )

    # Load trained weights into the controller's networks.
    ctrl.q_net.load_state_dict(ckpt["q_net"])
    if "target_net" in ckpt and hasattr(ctrl, "target_net") and ctrl.target_net is not None:
        ctrl.target_net.load_state_dict(ckpt["target_net"])

    # Force pure evaluation mode:
    #   - Disable learning updates (no replay buffer sampling, no gradient steps)
    #   - Set epsilon to 0 for fully greedy action selection
    #   - Switch networks to eval mode (disables dropout, batchnorm training, etc.)
    ctrl.enable_learning = False
    ctrl.epsilon = 0.0
    ctrl.q_net.eval()
    if hasattr(ctrl, "target_net") and ctrl.target_net is not None:
        ctrl.target_net.eval()

    return ctrl


def make_sac_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_spyder/sac_eval",
    *,
    use_time_update: bool = True,
    min_time_interval: float = 1.0,
    inv_limit_override: Optional[int] = None,
) -> SACDiscreteController:
    """Load a SACDiscreteController from checkpoint for evaluation (greedy)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cc = ckpt.get("controller_config", {})
    rc = ckpt.get("runner_config", {})

    pure_mm = bool(cc.get("pure_mm", True))
    pure_mm_offsets = cc.get("pure_mm_offsets", [(-1,-1),(-1,0),(0,-1),(0,0),(0,1),(1,0)])
    n_actions = int(cc.get("n_actions", len(pure_mm_offsets)))
    inv_limit = inv_limit_override if inv_limit_override is not None else cc.get("inv_limit", 8)
    use_obs_norm = bool(rc.get("USE_OBS_NORMALIZER", True))

    ctrl = SACDiscreteController(
        level_offset=int(cc.get("level_offset", 0)),
        n_actions=n_actions,
        gamma=float(cc.get("gamma", 0.97)),
        lr_actor=1e-4, lr_critic=1e-4,
        weight_decay=float(cc.get("weight_decay", 0.0)),
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
        log_dir=log_dir,
        pure_mm=pure_mm,
        inv_limit=inv_limit,
        pure_mm_offsets=pure_mm_offsets if pure_mm else None,
        n_hidden_actor=int(cc.get("n_hidden_actor", 1)),
        n_neurons_actor=int(cc.get("n_neurons_actor", 128)),
        n_hidden_critic=int(cc.get("n_hidden_critic", 1)),
        n_neurons_critic=int(cc.get("n_neurons_critic", 128)),
        enable_learning=False,
        use_tob_update=False, use_event_update=False,
        use_time_update=use_time_update,
        min_time_interval=min_time_interval,
        use_mdp=bool(cc.get("use_mdp", True)),
        use_obs_normalizer=use_obs_norm,
        lr_alpha=1e-4, tau=0.005, alpha_init=0.2,
        target_entropy_ratio=0.6, replay_capacity=1000, batch_size=128,
        use_dueling=bool(cc.get("use_dueling", True)),
        use_distributional=bool(cc.get("use_distributional", True)),
        n_quantiles=int(cc.get("n_quantiles", 25)),
    )

    ctrl.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
    if use_obs_norm and "obs_normalizer_state" in ckpt:
        ctrl.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])

    ctrl.enable_learning = False
    ctrl.actor_net.eval()
    return ctrl


def make_ppo_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_spyder/ppo_eval",
    *,
    use_time_update: bool = True,
    min_time_interval: float = 1.0,
    inv_limit_override: Optional[int] = None,
) -> PPOController:
    """Load a PPOController from checkpoint for evaluation (greedy)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cc = ckpt.get("controller_config", {})
    rc = ckpt.get("runner_config", {})

    pure_mm = bool(cc.get("pure_mm", True))
    pure_mm_offsets = cc.get("pure_mm_offsets", [(-1,-1),(-1,0),(0,-1),(0,0),(0,1),(1,0)])
    n_actions = int(cc.get("n_actions", len(pure_mm_offsets)))
    inv_limit = inv_limit_override if inv_limit_override is not None else cc.get("inv_limit", 8)
    use_obs_norm = bool(rc.get("USE_OBS_NORMALIZER", True))

    ctrl = PPOController(
        level_offset=int(cc.get("level_offset", 0)),
        n_actions=n_actions,
        gamma=float(cc.get("gamma", 0.97)),
        lr_actor=1e-4, lr_critic=1e-4,
        weight_decay=float(cc.get("weight_decay", 0.0)),
        entropy_coef=0.0,
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
        log_dir=log_dir,
        pure_mm=pure_mm,
        inv_limit=inv_limit,
        pure_mm_offsets=pure_mm_offsets if pure_mm else None,
        n_hidden_actor=int(cc.get("n_hidden_actor", 1)),
        n_neurons_actor=int(cc.get("n_neurons_actor", 128)),
        n_hidden_critic=int(cc.get("n_hidden_critic", 1)),
        n_neurons_critic=int(cc.get("n_neurons_critic", 128)),
        enable_learning=False,
        use_tob_update=False, use_event_update=False,
        use_time_update=use_time_update,
        min_time_interval=min_time_interval,
        use_mdp=bool(cc.get("use_mdp", True)),
        use_obs_normalizer=use_obs_norm,
    )

    ctrl.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
    if use_obs_norm and "obs_normalizer_state" in ckpt:
        ctrl.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])

    ctrl.enable_learning = False
    ctrl.actor_net.eval()
    return ctrl


# ============================================================
# 1) Global simulation parameters (LOB + MM environment)
#
# These parameters define the LOB microstructure and must be
# consistent between calibration and simulation. Any parameter
# used in calibrate_glft_params() must also be passed to
# run_one() to ensure the calibrated GLFT coefficients match
# the actual market dynamics.
# ============================================================

LAM = 0.06                      # Limit order arrival rate
MU = 0.1                        # Cancellation rate
DELTA = 0.02                    # Market order arrival rate

NUMBER_TICK_LEVELS = 50         # Number of tick levels in each half of the book
N_PRIORITY_RANKS = 100           # Number of priority ranks per tick level
HALF_TICK = 0.5                 # Half-tick for price calculations

N_STEPS = 5_000               # Total simulation steps (post-equilibrium)
N_STEPS_TO_EQUIL = 1_000       # Warmup steps to reach LOB equilibrium

INV_LIMIT = 8                   # Max absolute inventory for all policies
ROUND_TO_INT = True             # Use floor/ceil rounding for GLFT quotes

# Policy throttling (kept consistent across policies and with calibration)
USE_TOB_UPDATE = False
N_TOB_MOVES = 1

# How many independent simulations per policy?
N_SIMS = 500

# Seed control for reproducibility
SEED_BASE = 123456

# Plotting: if > 0, drop this many warmup points from the plots
DROP_WARMUP_PLOT = 0


# ============================================================
# 2) GLFT gamma choice
#
# This is the risk aversion parameter that you choose manually.
# It controls the trade-off between spread width and fill rate:
#   - Small gamma (e.g., 1e-6) -> tight spreads, more fills, more risk
#   - Large gamma (e.g., 0.1)  -> wide spreads, fewer fills, less risk
#
# This parameter is NOT calibrated; it reflects your risk preference.
# ============================================================
GAMMA_GLFT = 1e-6


# ============================================================
# 3) Calibration toggles
# ============================================================
DO_CALIBRATION = True      # If False, uses hardcoded fallback values
CALIB_PLOT = True           # Show calibration diagnostic plots

# If calibration has RNG dependency, this seed ensures reproducibility
# across runs of the comparison script.
CALIBRATION_SEED = 777

# Time intervals (seconds in simulation clock).
# GLFT decision throttle and DQN decision throttle are intentionally different.
GLFT_TIME_THROTTLE = 1.0
DEEPRL_TIME_THROTTLE = 1.0

# Censored-waiting-times calibration interval.
time_interval_calib = 0.5


# ============================================================
# 4) Reproducibility helpers
# ============================================================

def seed_everything(seed: int) -> None:
    """Lock Python + NumPy RNG for reproducibility."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def make_seed(run_id: int) -> int:
    """
    Deterministic seed for a given run_id.

    All policies share the same seed for a given run_id, so they
    face the SAME LOB realization. This paired design reduces
    variance when comparing performance differences.
    """
    return int(SEED_BASE + int(run_id))


# ============================================================
# 5) Calibrate GLFT parameters (A, kappa, sigma)
#
# This section runs the LOB simulator once (without an MM) to
# estimate the order arrival intensity parameters (A, kappa) and
# the mid-price volatility (sigma). These are then used to
# compute the GLFT optimal spread and skew coefficients.
# ============================================================

def calibrate_glft_params() -> Tuple[float, float, float]:
    """
    Calibrate GLFT model parameters from the LOB simulator.

    Runs two calibration procedures:
    1. fit_execution_intensity_censored_waiting_times + fit_A_kappa_loglinear:
       estimates (A, kappa) from censored waiting times.
    2. fit_volatility: estimates sigma from mid-price returns.

    Returns
    -------
    tuple of (float, float, float)
        (A_PARAM, KAPPA, SIGMA) — calibrated values ready for
        glft_policy_factory().
    """
    seed_everything(CALIBRATION_SEED)

    # -----------------------------------------------------------------
    # Fit execution intensity by censored waiting times:
    #   lambda(delta) = A * exp(-kappa * delta)
    # and then recover (A, kappa) via log-linear fit.
    # -----------------------------------------------------------------
    res_waiting_times = fit_execution_intensity_censored_waiting_times(
        aggregation_mode="time",
        time_interval=float(time_interval_calib),
        max_levels=10,
        half_tick=float(HALF_TICK),
        side_mode="buy",

        lam=LAM,
        mu=MU,
        delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=100_000,
        n_steps_to_equilibrium=10_000,
        split_sweeps=False,
        plot=bool(CALIB_PLOT),
    )

    A_PARAM, KAPPA, info_fit = fit_A_kappa_loglinear(
        res_waiting_times["delta_grid"],
        res_waiting_times["lambda_hat"],
        weights=res_waiting_times["denom_sum_tau"],
    )
    print(f"[CALIB] A={A_PARAM:.6g} | kappa={KAPPA:.6g} | r2_log={info_fit.get('r2_log', 'N/A')}")

    if CALIB_PLOT:
        d = np.asarray(res_waiting_times["delta_grid"], dtype=float)
        lam_hat = np.asarray(res_waiting_times["lambda_hat"], dtype=float)
        lam_fit = float(A_PARAM) * np.exp(-float(KAPPA) * d)

        plt.figure()
        plt.plot(d, lam_hat, marker="o", linestyle="-", label="lambda_hat (empirical)")
        plt.plot(d, lam_fit, linestyle="--", label="fit: A exp(-kappa d)")
        plt.xlabel("delta (distance from best, ticks)")
        plt.ylabel("lambda (execution intensity)")
        plt.title("Censored waiting times: empirical vs fitted intensity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------
    # Fit volatility: estimate sigma (std dev of mid-price returns
    # per unit time) from a separate simulation run.
    # -----------------------------------------------------------------
    res_vol = fit_volatility(
        queue_aware=False,

        aggregation_mode="time",
        time_interval=time_interval_calib,

        use_tob_buckets=False,
        n_tob_moves=10,

        lam=LAM, mu=MU, delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=N_STEPS,
        n_steps_to_equilibrium=N_STEPS_TO_EQUIL,

        half_tick=HALF_TICK,
        max_depth_levels=5,
        max_ranks=5,
        rank_to_fit=0,
        strict=True,
        plot=bool(CALIB_PLOT),
    )

    print(res_vol["volatility"], res_vol["volatility_unit"])

    SIGMA = float(res_vol["volatility"])
    # NOTE: if calibrated sigma is unreliable, override here:
    # SIGMA = 0.3  # from signature plot

    unit = res_vol.get("volatility_unit", "")
    print(f"[CALIB] sigma={SIGMA:.6g} {unit}")

    return A_PARAM, KAPPA, SIGMA


# ============================================================
# 6) Build policies
#
# Each factory function creates a fresh policy instance with
# internal state initialized to defaults. Policies are recreated
# for each simulation run to ensure independence.
# ============================================================

def make_policy_glft(A_PARAM: float, KAPPA: float, SIGMA: float) -> Any:
    """
    Build one GLFT policy object (callable) using calibrated parameters.

    The GLFT policy computes optimal bid/ask quotes based on:
      - A, kappa: order arrival intensity parameters
      - sigma: mid-price volatility
      - GAMMA_GLFT: risk aversion (set manually, not calibrated)
    """
    policy = glft_policy_factory(
        gamma=GAMMA_GLFT,
        kappa=KAPPA,
        A=A_PARAM,
        sigma=SIGMA,
        inv_limit=INV_LIMIT,
        round_to_int=ROUND_TO_INT,

        # --- GLFT Formula Generalization Parameters ---
        delta=1.0,
        epsilon=None,

        # --- Throttling: TOB (Market Structure Moves) ---
        use_tob_update=USE_TOB_UPDATE,
        n_tob_moves=N_TOB_MOVES,

        # --- Throttling: Simulation Steps (Events) ---
        use_event_update=False,
        n_events=1,

        # --- Throttling: Simulation Time (Seconds) ---
        use_time_update=True,
        min_time_interval=float(GLFT_TIME_THROTTLE),
    )
    return policy


def make_policy_always_best() -> Any:
    """
    Build an Always-Best policy: always quote at the best bid/ask.
    This is the simplest possible MM strategy and serves as a naive baseline.
    """
    return always_best_bid_ask_mm_policy_factory(
        inv_limit=INV_LIMIT,

        # --- Throttling: TOB (Market Structure Moves) ---
        use_tob_update=USE_TOB_UPDATE,
        n_tob_moves=N_TOB_MOVES,

        # --- Throttling: Simulation Steps (Events) ---
        use_event_update=False,
        n_events=1,

        # --- Throttling: Simulation Time (Seconds) ---
        use_time_update=True,
        min_time_interval=float(GLFT_TIME_THROTTLE),
    )


# ============================================================
# 7) Run one simulation
# ============================================================

def run_one(mm_policy: Any = None, controller: Any = None, seed: int = 0) -> pd.DataFrame:
    """
    Run ONE LOB simulation with a single market-making agent.

    Exactly one of ``mm_policy`` or ``controller`` must be provided:
      - mm_policy: a callable policy(state) -> action_tuple
        (for GLFT, Always-Best, fixed-offset, and other analytical policies).
      - controller: an RL controller implementing the act()/learn() interface
        (for DQN, REINFORCE, or any other RLController-compatible agent).

    The routing between these two kwarg paths is handled automatically
    by the caller (run_two_policy_study) using duck-typing:
        hasattr(obj, "act") and hasattr(obj, "learn") -> controller
        otherwise                                      -> mm_policy

    Parameters
    ----------
    mm_policy : callable or None
        Classic callable policy function.
    controller : object or None
        RL controller with act(state) and learn(...) methods.
    seed : int
        Random seed for this simulation run.

    Returns
    -------
    pd.DataFrame
        Market maker time series (inventory, PnL, cash, etc.).
    """
    seed_everything(seed)

    _msg_df, _ob_df, mm_df = simulate_LOB_with_MM(
        lam=LAM,
        mu=MU,
        delta=DELTA,

        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        iterations=N_STEPS,
        iterations_to_equilibrium=N_STEPS_TO_EQUIL,

        # One or another
        mm_policy=mm_policy,
        controller=controller,

        exclude_self_from_state=False,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        random_seed=seed,
    )
    return mm_df


# ============================================================
# 8) Align runs into (n_runs, T) arrays for a given column
# ============================================================

def stack_metric(mm_dfs: List[pd.DataFrame], col: str, drop_warmup: int = 0) -> Optional[np.ndarray]:
    """
    Stack a metric column from multiple simulation DataFrames into a matrix.

    Since different runs may have slightly different lengths (due to the
    stochastic nature of the simulation), we truncate all runs to the
    shortest length to produce a rectangular array.

    Parameters
    ----------
    mm_dfs : list of pd.DataFrame
        List of market-maker DataFrames, one per simulation run.
    col : str
        Column name to extract (e.g., "MM_TotalPnL", "MM_Inventory").
    drop_warmup : int
        Number of initial time steps to discard from each run (useful
        for removing transient effects from plots).

    Returns
    -------
    np.ndarray or None
        Array of shape (n_runs, T_min), or None if no valid data.
    """
    series_list = []
    for df in mm_dfs:
        if df is None or df.empty or col not in df.columns:
            continue
        x = df[col].to_numpy()
        if drop_warmup > 0 and x.size > drop_warmup:
            x = x[drop_warmup:]
        series_list.append(x)

    if not series_list:
        return None

    min_len = min(len(x) for x in series_list)
    if min_len <= 1:
        return None

    return np.vstack([x[:min_len] for x in series_list])  # (n_runs, T)


# ============================================================
# 9) Plot helper (mean +/- std band)
# ============================================================

def plot_mean_std_two_policies(
    store: Dict[str, List[pd.DataFrame]],
    col: str,
    title: str,
    ylabel: str,
    drop_warmup: int = 0,
    step_style: bool = False,
):
    """
    Plot mean +/- 1 std across multiple runs for each policy in ``store``.

    Each policy is plotted as a solid line (mean) with a shaded band
    (+/- 1 std). This visualization shows both the expected performance
    and the run-to-run variability of each policy.

    Parameters
    ----------
    store : dict
        Mapping from policy name to list of simulation DataFrames.
        Example: {"GLFT": [mm_df_run_0, mm_df_run_1, ...], ...}
    col : str
        Column name to plot (e.g., "MM_TotalPnL").
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    drop_warmup : int
        Number of initial steps to discard from each run.
    step_style : bool
        If True, use step plot (good for integer-valued metrics like inventory).
        If False, use smooth line plot (good for continuous metrics like PnL).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, mm_dfs in store.items():
        X = stack_metric(mm_dfs, col=col, drop_warmup=drop_warmup)
        if X is None:
            print(f"[WARN] No data for policy '{name}' / col '{col}'.")
            continue

        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        t = np.arange(mean_x.shape[0])

        if step_style:
            (line,) = ax.step(t, mean_x, where="post", label=name)
            c = line.get_color()
            ax.fill_between(t, mean_x - std_x, mean_x + std_x, step="post", alpha=0.15, color=c)
        else:
            (line,) = ax.plot(t, mean_x, label=name)
            c = line.get_color()
            ax.fill_between(t, mean_x - std_x, mean_x + std_x, alpha=0.15, color=c)

    ax.set_title(title)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)    
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 10) REINFORCE Evaluation Controller
#
# REFACTORING NOTE (2026-02-10):
#     This section previously contained ``ReinforceEvalPolicy``
#     (~420 lines), a standalone class that duplicated the state
#     encoding, action mapping, and price computation logic from
#     ``reinforce_general.PolicyGradientController``.
#
#     That duplication has been eliminated. We now import the factory
#     function ``make_reinforce_eval_controller`` directly from
#     ``reinforce_general.py``, which returns a real
#     ``PolicyGradientController`` in evaluation mode
#     (``enable_learning=False``).
#
#     Benefits:
#       - Single source of truth for state encoding + action mapping.
#       - Bug fixes in the controller (e.g., log1p clamp, crossed-quote
#         protection) are automatically reflected in evaluation.
#       - ~420 fewer lines to maintain in this comparison script.
#       - Backward compatible with both old (``policy_state_dict``) and
#         new (``policy_net_state_dict``) checkpoint formats.
#
#     The factory ``make_reinforce_eval_controller`` is imported at
#     the top of this file from ``reinforce_general``.
# ============================================================


# ============================================================
# 11) Main runner
# ============================================================

def run_two_policy_study(n_sims: int = N_SIMS) -> Dict[str, List[pd.DataFrame]]:
    """
    Run the full multi-policy comparison study.

    Steps:
      1. Calibrate GLFT parameters (A, kappa, sigma) from a fresh LOB run.
      2. Build the policy list (GLFT + Always-Best + optional DeepRL/REINFORCE).
      3. For each policy, run n_sims independent simulations with paired seeds.
      4. Plot mean +/- std for TotalPnL and Inventory.

    Returns
    -------
    dict
        Mapping from policy name to list of simulation DataFrames.
    """
    # --- Calibration (once) ---
    if DO_CALIBRATION:
        A_PARAM, KAPPA, SIGMA = calibrate_glft_params()
    else:
        # Fallback if you want to hardcode values for quick testing:
        A_PARAM, KAPPA, SIGMA = 1.0, 1.0, 1.0

    print(
        "[CFG] "
        f"gamma_glft={GAMMA_GLFT:g} | inv_limit={INV_LIMIT} | "
        f"glft_dt={GLFT_TIME_THROTTLE}s | "
        f"calib_dt={time_interval_calib}s"
    )

    # Build policy makers (GLFT needs calibrated params captured via closure)
    def _make_glft():
        return make_policy_glft(A_PARAM, KAPPA, SIGMA)

    # ============================================================
    # Policy list — built from checkpoint dictionaries
    # ============================================================
    policies = [("GLFT (Lehalle)", _make_glft)]

    if USE_ALWAYS_BEST:
        policies.append(("Always-Best", make_policy_always_best))

    # ── DQN ──
    for label, path in DQN_CHECKPOINTS.items():
        full_path = path if os.path.isabs(path) else str(CKPT_DIR / Path(path).name) if not os.path.isfile(path) else path
        if os.path.isfile(path):
            full_path = path
        elif os.path.isfile(str(CKPT_DIR / Path(path).name)):
            full_path = str(CKPT_DIR / Path(path).name)
        else:
            print(f"[WARN] DQN checkpoint not found: {path} — skipping {label}")
            continue
        _p = full_path
        policies.append((label, lambda _p=_p: make_controller_from_checkpoint(
            _p, use_time_update=True,
            min_time_interval=float(DQN_TIME_THROTTLE),
            inv_limit_override=INV_LIMIT,
        )))

    # ── SAC ──
    for label, path in SAC_CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"[WARN] SAC checkpoint not found: {path} — skipping {label}")
            continue
        _p = path
        policies.append((label, lambda _p=_p: make_sac_controller_from_checkpoint(
            _p, use_time_update=True,
            min_time_interval=float(SAC_TIME_THROTTLE),
            inv_limit_override=INV_LIMIT,
        )))

    # ── PPO ──
    for label, path in PPO_CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"[WARN] PPO checkpoint not found: {path} — skipping {label}")
            continue
        _p = path
        policies.append((label, lambda _p=_p: make_ppo_controller_from_checkpoint(
            _p, use_time_update=True,
            min_time_interval=float(PPO_TIME_THROTTLE),
            inv_limit_override=INV_LIMIT,
        )))

    # ── REINFORCE ──
    for label, path in REINFORCE_CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"[WARN] REINFORCE checkpoint not found: {path} — skipping {label}")
            continue
        _p = path
        policies.append((label, lambda _p=_p: make_reinforce_eval_controller(
            _p, device="cpu",
        )))

    # ── A2C ──
    for label, path in A2C_CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"[WARN] A2C checkpoint not found: {path} — skipping {label}")
            continue
        _p = path
        policies.append((label, lambda _p=_p: make_ac_eval_controller(
            _p, device="cpu",
        )))

    store: Dict[str, List[pd.DataFrame]] = {name: [] for (name, _) in policies}

    # Per-policy action_counts from RL controllers (accumulated across sims).
    # Key: policy name, Value: np.ndarray of shape (n_actions,) or None.
    rl_action_counts: Dict[str, Optional[np.ndarray]] = {}
    rl_action_inv_sum: Dict[str, Optional[np.ndarray]] = {}
    rl_action_inv_abs_sum: Dict[str, Optional[np.ndarray]] = {}
    # Per-policy offset grid (for RL controllers with pure_mm_offsets).
    rl_offset_grids: Dict[str, Optional[list]] = {}

    # --- Run simulations ---
    for name, maker in policies:
        print(f"\n=== Running policy: {name} ===")

        # Pre-create a probe instance to check if this is an RL controller.
        # RL controllers (DQN, SAC, PPO, REINFORCE, A2C) are stateless in
        # eval mode, so we create them ONCE and reuse across all sims.
        # Callable policies (GLFT, Always-Best) have internal state and
        # must be recreated for each simulation.
        probe = maker()
        is_rl = hasattr(probe, "act") and hasattr(probe, "learn")
        rl_ctrl = probe if is_rl else None

        # Reset action_counts if available (DQN has it, SAC/PPO may not)
        if is_rl and hasattr(rl_ctrl, "action_counts"):
            rl_ctrl.action_counts[:] = 0
        if is_rl and hasattr(rl_ctrl, "action_inv_sum"):
            rl_ctrl.action_inv_sum[:] = 0.0
            rl_ctrl.action_inv_abs_sum[:] = 0.0

        for run_id in range(int(n_sims)):
            seed = make_seed(run_id)
            print(f"  -> sim {run_id+1}/{n_sims} (seed={seed}) ...", end="", flush=True)

            if is_rl:
                mm_df = run_one(mm_policy=None, controller=rl_ctrl, seed=seed)
            else:
                obj = maker()
                mm_df = run_one(mm_policy=obj, controller=None, seed=seed)

            store[name].append(mm_df)
            print(" done.")

        # Capture RL action counts and offset grid after all sims for this policy
        if is_rl and hasattr(rl_ctrl, "action_counts"):
            rl_action_counts[name] = rl_ctrl.action_counts.copy()
            rl_action_inv_sum[name] = getattr(rl_ctrl, "action_inv_sum", np.zeros_like(rl_ctrl.action_counts, dtype=np.float64)).copy()
            rl_action_inv_abs_sum[name] = getattr(rl_ctrl, "action_inv_abs_sum", np.zeros_like(rl_ctrl.action_counts, dtype=np.float64)).copy()
            rl_offset_grids[name] = getattr(rl_ctrl, "pure_mm_offsets", None)
        else:
            rl_action_counts[name] = None
            rl_action_inv_sum[name] = None
            rl_action_inv_abs_sum[name] = None
            rl_offset_grids[name] = None

    # --- Build dynamic plot titles from the actual policy names ---
    policy_names = " vs ".join(store.keys())

    plot_mean_std_two_policies(
        store,
        col="MM_TotalPnL",
        title=f"Mean TotalPnL(t) +/- 1 std — {policy_names}",
        ylabel="MM_TotalPnL",
        drop_warmup=DROP_WARMUP_PLOT,
        step_style=False,
    )

    plot_mean_std_two_policies(
        store,
        col="MM_Inventory",
        title=f"Mean Inventory(t) +/- 1 std — {policy_names}",
        ylabel="MM_Inventory",
        drop_warmup=DROP_WARMUP_PLOT,
        step_style=True,
    )

    # --- Action usage report ---
    print_action_usage_report(store, rl_action_counts, rl_offset_grids,
                              rl_action_inv_sum, rl_action_inv_abs_sum)

    # --- Extreme inventory action report ---
    print_extreme_inv_action_report(store, rl_offset_grids)

    return store


def print_action_usage_report(
    store: Dict[str, List[pd.DataFrame]],
    rl_action_counts: Optional[Dict[str, Optional[np.ndarray]]] = None,
    rl_offset_grids: Optional[Dict[str, Optional[list]]] = None,
    rl_action_inv_sum: Optional[Dict[str, Optional[np.ndarray]]] = None,
    rl_action_inv_abs_sum: Optional[Dict[str, Optional[np.ndarray]]] = None,
) -> None:
    """
    For each policy, print action usage % and mean inventory per action.

    For RL controllers with action_counts (DQN): uses the controller's
    internal counters (exact) mapped to the offset grid.

    For other policies (GLFT, SAC, PPO without action_counts): infers
    offsets from MM_BidDist/MM_AskDist on decision steps.
    """
    from collections import Counter

    if rl_action_counts is None:
        rl_action_counts = {}
    if rl_offset_grids is None:
        rl_offset_grids = {}
    if rl_action_inv_sum is None:
        rl_action_inv_sum = {}
    if rl_action_inv_abs_sum is None:
        rl_action_inv_abs_sum = {}

    print("\n" + "=" * 70)
    print("ACTION USAGE REPORT (across all simulations)")
    print("=" * 70)

    for name, mm_dfs in store.items():
        ac = rl_action_counts.get(name)
        offsets = rl_offset_grids.get(name)

        # ----------------------------------------------------------
        # Path A: RL controller with action_counts + offset grid
        #         (exact counts from the controller, no inference)
        # ----------------------------------------------------------
        if ac is not None and offsets is not None:
            total = int(ac.sum())
            if total == 0:
                print(f"\n--- {name}: no decisions recorded ---")
                continue

            inv_sum = rl_action_inv_sum.get(name)
            inv_abs_sum = rl_action_inv_abs_sum.get(name)
            has_inv = inv_sum is not None and inv_abs_sum is not None

            print(f"\n--- {name} ({total:,} decisions across {len(mm_dfs)} sims) ---")
            print(f"  {'Action':<14s} {'Usage %':>8s}  {'Mean Inv':>9s}  {'|Mean Inv|':>10s}  {'Count':>8s}")
            print(f"  {'-'*14} {'-'*8}  {'-'*9}  {'-'*10}  {'-'*8}")

            # Sort by count descending
            order = np.argsort(-ac)
            for idx in order:
                count = int(ac[idx])
                if count == 0:
                    continue
                pct = 100.0 * count / total
                bo, ao = offsets[idx]
                if has_inv and count > 0:
                    mean_inv = inv_sum[idx] / count
                    abs_mean_inv = inv_abs_sum[idx] / count
                else:
                    mean_inv = 0.0
                    abs_mean_inv = 0.0
                print(
                    f"  ({bo:+d}, {ao:+d})      "
                    f"{pct:6.2f}%  {mean_inv:+9.3f}  {abs_mean_inv:10.3f}  {count:8d}"
                )

        # ----------------------------------------------------------
        # Path B: Fallback — infer offsets from distances
        #         (GLFT, SAC, PPO, or any policy without action_counts)
        # ----------------------------------------------------------
        else:
            all_actions = []
            all_invs = []

            for df in mm_dfs:
                if df is None or df.empty:
                    continue
                required = {"MM_BidDist", "MM_AskDist", "MM_Action", "MM_Inventory"}
                if not required.issubset(df.columns):
                    continue

                action_col = df["MM_Action"].values
                decision_mask = np.array([
                    a not in ("hold", "passive_fill") for a in action_col
                ])

                bd = df["MM_BidDist"].values[decision_mask]
                ad = df["MM_AskDist"].values[decision_mask]
                inv = df["MM_Inventory"].values[decision_mask]

                if "MM_State_Spread" in df.columns:
                    spread = df["MM_State_Spread"].values[decision_mask]
                else:
                    spread = np.full_like(bd, 2.0)

                valid = ~(np.isnan(bd) | np.isnan(ad) | np.isnan(spread))
                bid_off = np.round(bd[valid] + spread[valid] / 2.0).astype(int)
                ask_off = np.round(ad[valid] - spread[valid] / 2.0).astype(int)
                inv_v = inv[valid]

                for b, a, q in zip(bid_off, ask_off, inv_v):
                    all_actions.append((int(b), int(a)))
                    all_invs.append(float(q))

            if not all_actions:
                print(f"\n--- {name}: no action data ---")
                continue

            counts = Counter(all_actions)
            total = len(all_actions)
            inv_arr = np.array(all_invs)
            act_arr = all_actions

            print(f"\n--- {name} ({total:,} decisions across {len(mm_dfs)} sims, inferred) ---")
            print(f"  {'Action':<14s} {'Usage %':>8s}  {'Mean Inv':>9s}  {'|Mean Inv|':>10s}  {'Count':>8s}")
            print(f"  {'-'*14} {'-'*8}  {'-'*9}  {'-'*10}  {'-'*8}")

            for action, count in counts.most_common():
                pct = 100.0 * count / total
                mask = [a == action for a in act_arr]
                inv_cond = inv_arr[mask]
                mean_inv = float(np.mean(inv_cond))
                abs_mean_inv = float(np.mean(np.abs(inv_cond)))
                print(
                    f"  ({action[0]:+d}, {action[1]:+d})      "
                    f"{pct:6.2f}%  {mean_inv:+9.3f}  {abs_mean_inv:10.3f}  {count:8d}"
                )

    print("\n" + "=" * 70)


def print_extreme_inv_action_report(
    store: Dict[str, List[pd.DataFrame]],
    rl_offset_grids: Optional[Dict[str, Optional[list]]] = None,
    lo_pct: float = 10.0,
    hi_pct: float = 90.0,
) -> None:
    """
    For each policy, show action usage when inventory is at extreme levels.

    Uses MM_ActionIdx (logged by simulator) for RL controllers with an
    offset grid; falls back to inferred offsets for GLFT/other policies.

    Splits decisions into 3 buckets:
      - VERY SHORT: inventory <= P(lo_pct)  (bottom 10%)
      - NEUTRAL:    P(lo_pct) < inventory < P(hi_pct)
      - VERY LONG:  inventory >= P(hi_pct)  (top 10%)
    """
    from collections import Counter

    if rl_offset_grids is None:
        rl_offset_grids = {}

    print("\n" + "=" * 70)
    print(f"EXTREME INVENTORY ACTION REPORT (bottom/top {100-hi_pct:.0f}% of inventory)")
    print("=" * 70)

    for name, mm_dfs in store.items():
        offsets = rl_offset_grids.get(name)

        # Concatenate all decision steps across sims
        all_inv = []
        all_aidx = []

        for df in mm_dfs:
            if df is None or df.empty:
                continue
            if "MM_Action" not in df.columns or "MM_Inventory" not in df.columns:
                continue

            action_col = df["MM_Action"].values
            decision_mask = np.array([
                a not in ("hold", "passive_fill") for a in action_col
            ])

            inv = df["MM_Inventory"].values[decision_mask]

            # Try MM_ActionIdx first (exact), fall back to inferred offsets
            if "MM_ActionIdx" in df.columns and offsets is not None:
                aidx = df["MM_ActionIdx"].values[decision_mask]
                valid = aidx >= 0
                all_inv.extend(inv[valid].tolist())
                all_aidx.extend(aidx[valid].astype(int).tolist())
            else:
                # Inferred offsets for non-RL policies
                if not {"MM_BidDist", "MM_AskDist"}.issubset(df.columns):
                    continue
                bd = df["MM_BidDist"].values[decision_mask]
                ad = df["MM_AskDist"].values[decision_mask]
                spread = df["MM_State_Spread"].values[decision_mask] if "MM_State_Spread" in df.columns else np.full_like(bd, 2.0)
                valid = ~(np.isnan(bd) | np.isnan(ad) | np.isnan(spread))
                bid_off = np.round(bd[valid] + spread[valid] / 2.0).astype(int)
                ask_off = np.round(ad[valid] - spread[valid] / 2.0).astype(int)
                # Encode as negative index (to distinguish from grid indices)
                for b, a, q in zip(bid_off, ask_off, inv[valid]):
                    all_inv.append(float(q))
                    all_aidx.append((int(b), int(a)))  # tuple for inferred

        if not all_inv:
            print(f"\n--- {name}: no data ---")
            continue

        inv_arr = np.array(all_inv)
        p_lo = np.percentile(inv_arr, lo_pct)
        p_hi = np.percentile(inv_arr, hi_pct)

        buckets = {
            f"VERY SHORT (inv <= {p_lo:+.1f})": inv_arr <= p_lo,
            f"VERY LONG  (inv >= {p_hi:+.1f})": inv_arr >= p_hi,
        }

        print(f"\n--- {name} ---")
        print(f"  Inventory percentiles: P{lo_pct:.0f}={p_lo:+.1f}, P{hi_pct:.0f}={p_hi:+.1f}")

        for bucket_name, mask in buckets.items():
            n_bucket = int(mask.sum())
            if n_bucket == 0:
                print(f"\n  {bucket_name}: no decisions")
                continue

            print(f"\n  {bucket_name} ({n_bucket:,} decisions):")

            # Count actions in this bucket
            counts = Counter()
            for i, m in enumerate(mask):
                if m:
                    counts[all_aidx[i]] += 1

            print(f"    {'Action':<14s} {'Usage %':>8s}  {'Count':>8s}")
            print(f"    {'-'*14} {'-'*8}  {'-'*8}")

            for key, count in counts.most_common():
                pct = 100.0 * count / n_bucket
                if isinstance(key, tuple):
                    label = f"({key[0]:+d}, {key[1]:+d})"
                elif offsets is not None and isinstance(key, (int, np.integer)):
                    bo, ao = offsets[key]
                    label = f"({bo:+d}, {ao:+d})"
                else:
                    label = f"idx={key}"
                print(f"    {label:<14s} {pct:6.2f}%  {count:8d}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    store = run_two_policy_study(n_sims=N_SIMS)
