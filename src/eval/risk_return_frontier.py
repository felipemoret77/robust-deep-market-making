#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_return_frontier.py  —  GLFT vs RL Efficient Frontier
===========================================================

Plots E[PnL_T] vs √Var(PnL_T) for:
  (a) GLFT policies with different gamma (risk aversion) values.
  (b) DQN agents loaded from pre-trained checkpoints (different inv_penalty_coeff).
  (c) SAC-Discrete agents loaded from pre-trained checkpoints.
  (d) PPO agents loaded from pre-trained checkpoints.

Each point on the frontier is computed from N_SIMS independent LOB simulations.
All policies see the same market realizations (paired seeds) for fair comparison.

Usage
-----
    python risk_return_frontier.py
"""

import os
import random
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# === Paper figure style: matches GLFT_studies.py and the throttle/calibration
# comparison plots so all paper figures share label/legend sizing.  Viridis
# cycler gives distinct hues per line while preserving perceptual ordering. ===
plt.rcParams.update({
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.fancybox": True,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
})
from cycler import cycler as _cycler
_VIRIDIS_PAPER_PALETTE = [
    tuple(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 6))
]
plt.rcParams["axes.prop_cycle"] = _cycler(color=_VIRIDIS_PAPER_PALETTE)
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from MM_LOB_SIM import simulate_LOB_with_MM, reward_spread_capture_inv_quadratic
from GLFT_policy_factory import glft_policy_factory
from MM_GLFT_naive_comparison import make_controller_from_checkpoint

from sac import SACDiscreteController
from ppo import PPOController

from censored_waiting_times_calib import (
    fit_execution_intensity_censored_waiting_times,
    fit_A_kappa_loglinear,
)
from calibrate_trading_intensity import fit_volatility

from CONFIG_MM import GLOBAL_SEED
from MM_policy_5 import always_best_bid_ask_mm_policy_factory


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── LOB parameters (Santa Fe / AMZN calibration) ────────────────────────────
LAM = 0.06
MU = 0.1
DELTA = 0.02


NUMBER_TICK_LEVELS = 50
N_PRIORITY_RANKS = 100
HALF_TICK = 0.5

N_STEPS = 5_000
N_STEPS_TO_EQUIL = 1_000

# ── MM constraints ──────────────────────────────────────────────────────────
INV_LIMIT = 8

# ── Monte Carlo ─────────────────────────────────────────────────────────────
N_SIMS = 1000
SEED_BASE = GLOBAL_SEED

# ── Calibration ─────────────────────────────────────────────────────────────
SEED_CALIB = 1234
TIME_INTERVAL_CALIB = 0.5

# Canonical calibration, unified across all paper experiments: the pair
# reproduced by GLFT_studies.run_calibration() (dT=0.5, 500k steps,
# n_equil=1000, seed=1234).  Set FORCE_CANONICAL_CALIB=False to fall back
# to this script's own probe calibration (n_equil=50k, max_levels=10).
FORCE_CANONICAL_CALIB = True
CANONICAL_A = 0.150707
CANONICAL_KAPPA = 2.33534
CANONICAL_SIGMA = 0.3

# ── GLFT sweep ──────────────────────────────────────────────────────────────
#GLFT_GAMMAS = []
GLFT_GAMMAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

GLFT_TIME_THROTTLE = 1.0
USE_MDP_GLFT = True

# ── DQN checkpoints ────────────────────────────────────────────────────────
# Map: display label → checkpoint path.
# Add entries as you train DQN agents with different inv_penalty_coeff values.
DQN_CHECKPOINTS: Dict[str, str] = {
    # New reward (dampened pnl):
    "DQN  (φ=0.000)": "checkpoints/deep_mm_mtm_pure_invp0.0000_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    "DQN  (φ=0.001)": "checkpoints/deep_mm_mtm_pure_invp0.0010_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    "DQN  (φ=0.002)": "checkpoints/deep_mm_mtm_pure_invp0.0020_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    #"DQN  (φ=0.003)": "checkpoints/deep_mm_mtm_pure_invp0.0030_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    #"DQN  (φ=0.004)": "checkpoints/deep_mm_mtm_pure_invp0.0040_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    "DQN  (φ=0.005)": "checkpoints/deep_mm_mtm_pure_invp0.0050_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    #"DQN  (φ=0.006)": "checkpoints/deep_mm_mtm_pure_invp0.0060_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    "DQN  (φ=0.008)": "checkpoints/deep_mm_mtm_pure_invp0.0080_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    #"DQN  (φ=0.010)": "checkpoints/deep_mm_mtm_pure_invp0.0100_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
    #"DQN  (φ=0.100)": "checkpoints/deep_mm_mtm_pure_invp0.1000_inv_wall_dampened_reward_clr200f0p30_best_ma50.pt",
}


DQN_TIME_THROTTLE = 1.0

# ── SAC checkpoints ───────────────────────────────────────────────────────
SAC_CHECKPOINTS: Dict[str, str] = {
    #"SAC (φ=0.001)": "checkpoints/sac_mm_pure_ep0350.pt",
}

SAC_TIME_THROTTLE = 1.0

# ── PPO checkpoints ───────────────────────────────────────────────────────
PPO_CHECKPOINTS: Dict[str, str] = {
    #"PPO (φ=0.001)": "checkpoints/ppo_mm_pure_final.pt",
}

PPO_TIME_THROTTLE = 1.0

# ── Naive MM benchmark ───────────────────────────────────────────────
PLOT_MM_NAIVE = True
NAIVE_MM_TIME_THROTTLE = 1.0
USE_MDP_NAIVE = True


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_seed(run_id: int) -> int:
    """Deterministic seed per run_id — shared across all policies (paired comparison)."""
    return int(SEED_BASE + run_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_calibration(plot: bool = False) -> Tuple[float, float, float]:
    """Calibrate GLFT parameters (A, kappa, sigma) from a bare LOB simulation."""

    if FORCE_CANONICAL_CALIB:
        print("=" * 60)
        print("  CALIBRATION (canonical override)")
        print(f"  A     = {CANONICAL_A}")
        print(f"  kappa = {CANONICAL_KAPPA}")
        print(f"  sigma = {CANONICAL_SIGMA}")
        print("=" * 60)
        return float(CANONICAL_A), float(CANONICAL_KAPPA), float(CANONICAL_SIGMA)

    print("=" * 60)
    print("  CALIBRATION")
    print("=" * 60)

    res_wt = fit_execution_intensity_censored_waiting_times(
        aggregation_mode="time",
        time_interval=TIME_INTERVAL_CALIB,
        max_levels=10,
        half_tick=HALF_TICK,
        side_mode="buy",
        lam=LAM, mu=MU, delta=DELTA,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        n_steps=500_000,
        n_steps_to_equilibrium=50_000,
        split_sweeps=False,
        random_seed=SEED_CALIB,
        plot=plot,
    )

    A, KAPPA, _ = fit_A_kappa_loglinear(
        res_wt["delta_grid"],
        res_wt["lambda_hat"],
        weights=res_wt["denom_sum_tau"],
    )

    res_vol = fit_volatility(
        queue_aware=False,
        aggregation_mode="time",
        time_interval=TIME_INTERVAL_CALIB,
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
        plot=plot,
        random_seed=SEED_CALIB + 1,
    )

    SIGMA = float(res_vol["volatility"])
    SIGMA = 0.3 #from signature plot 

    print(f"  A     = {A:.6g}")
    print(f"  kappa = {KAPPA:.6g}")
    print(f"  sigma = {SIGMA:.6g}")
    print()

    return float(A), float(KAPPA), SIGMA


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_glft_sims(
    gamma: float,
    A: float,
    kappa: float,
    sigma: float,
) -> np.ndarray:
    """Run N_SIMS simulations with a GLFT policy. Returns array of final PnLs."""

    pnls = np.empty(N_SIMS)

    for i in range(N_SIMS):
        print(f"    sim {i+1}/{N_SIMS}", end="\r", flush=True)
        seed = make_seed(i)
        set_seeds(seed)

        policy = glft_policy_factory(
            gamma=gamma,
            kappa=kappa,
            A=A,
            sigma=sigma,
            inv_limit=INV_LIMIT,
            round_to_int=True,
            delta=1.0,
            epsilon=None,
            use_tob_update=False,
            n_tob_moves=1,
            use_event_update=False,
            n_events=1000,
            use_time_update=True,
            min_time_interval=GLFT_TIME_THROTTLE,
            use_mdp=USE_MDP_GLFT,
        )

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=1,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            mm_policy=policy,
            random_seed=seed,
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])

    return pnls


def run_naive_mm_sims() -> np.ndarray:
    """Run N_SIMS simulations with the Always-Best-Bid/Ask naive MM policy.
    Returns array of final PnLs (single point on the frontier)."""

    pnls = np.empty(N_SIMS)

    for i in range(N_SIMS):
        print(f"    sim {i+1}/{N_SIMS}", end="\r", flush=True)
        seed = make_seed(i)
        set_seeds(seed)

        policy = always_best_bid_ask_mm_policy_factory(
            inv_limit=INV_LIMIT,
            use_tob_update=False,
            use_event_update=False,
            use_time_update=True,
            min_time_interval=NAIVE_MM_TIME_THROTTLE,
            use_mdp=USE_MDP_NAIVE,
        )

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=1,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            mm_policy=policy,
            random_seed=seed,
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])

    print()
    return pnls


def run_dqn_sims(ckpt_path: str, inv_penalty_coeff: float = 0.001) -> dict:
    """Run N_SIMS simulations with a DQN agent from checkpoint.

    Returns dict with 'pnls', 'fills', 'action_counts', 'spread_capture', 'inv_penalty'.
    """

    pnls = np.empty(N_SIMS)
    fills = np.empty(N_SIMS)

    ctrl = make_controller_from_checkpoint(
        ckpt_path,
        log_dir="runs_eval/frontier_dqn",
        use_time_update=True,
        min_time_interval=DQN_TIME_THROTTLE,
        inv_limit_override=INV_LIMIT,
    )

    n_actions = ctrl.n_actions
    total_action_counts = np.zeros(n_actions, dtype=np.int64)
    total_spread_capture = 0.0
    total_inv_penalty = 0.0

    for i in range(N_SIMS):
        seed = make_seed(i)
        set_seeds(seed)

        # Track reward decomposition + actions via wrapper
        ep_sc = [0.0]
        ep_ip = [0.0]
        ep_ac = np.zeros(n_actions, dtype=np.int64)

        _sc = ep_sc
        _ip = ep_ip
        _ac = ep_ac
        _ctrl = ctrl

        _phi = inv_penalty_coeff

        def _tracking_reward(step_idx, mm, lob, sb, sa, info, **kw):
            reward_spread_capture_inv_quadratic(
                step_idx, mm, lob, sb, sa, info, inv_penalty_coeff=_phi, **kw)
            _sc[0] += info.get("_reward_spread_capture", 0.0)
            _ip[0] += info.get("_reward_inv_penalty", 0.0)
            a = _ctrl.last_action_idx
            if a is not None and 0 <= a < len(_ac):
                _ac[a] += 1
            return 0.0

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=1,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            controller=ctrl,
            reward_fn=_tracking_reward,
            random_seed=seed,
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])
        if mm_df is not None and "MM_HadFill" in mm_df.columns:
            fills[i] = float((mm_df["MM_HadFill"] == True).sum())
        else:
            fills[i] = 0.0
        total_action_counts += ep_ac
        total_spread_capture += ep_sc[0]
        total_inv_penalty += ep_ip[0]

        # Per-sim log
        ac_total = max(1, ep_ac.sum())
        ac_pcts = ep_ac / ac_total * 100.0
        offsets = getattr(ctrl, "pure_mm_offsets", None)
        if offsets is not None:
            ac_str = " ".join(f"{offsets[j]}={ac_pcts[j]:.0f}%" for j in range(n_actions))
        else:
            ac_str = " ".join(f"a{j}={ac_pcts[j]:.0f}%" for j in range(n_actions))
        print(f"    sim {i+1}/{N_SIMS}  PnL={pnls[i]:+.4f}  fills={fills[i]:.0f}  "
              f"SC={ep_sc[0]:.2f}  IP={ep_ip[0]:.2f}  [{ac_str}]")

    # Close TensorBoard writer
    try:
        ctrl.writer.close()
    except Exception:
        pass

    return {
        "pnls": pnls,
        "fills": fills,
        "action_counts": total_action_counts,
        "mean_spread_capture": total_spread_capture / N_SIMS,
        "mean_inv_penalty": total_inv_penalty / N_SIMS,
        "offsets": getattr(ctrl, "pure_mm_offsets", None),
    }


def _print_rl_summary(label: str, info: dict) -> None:
    """Print summary stats for an RL checkpoint evaluation."""
    pnls = info["pnls"]
    print(f"  E[PnL]={np.mean(pnls):+.4f}  "
          f"√Var={np.std(pnls, ddof=1):.4f}  "
          f"fills={np.mean(info['fills']):.1f}")
    print(f"  spread_capture={info['mean_spread_capture']:.4f}  "
          f"inv_penalty={info['mean_inv_penalty']:.4f}")
    ac = info["action_counts"]
    ac_total = max(1, ac.sum())
    ac_pcts = ac / ac_total * 100.0
    offsets = info["offsets"]
    if offsets is not None:
        ac_parts = [f"{offsets[j]}={ac_pcts[j]:.0f}%" for j in range(len(ac))]
    else:
        ac_parts = [f"a{j}={ac_pcts[j]:.0f}%" for j in range(len(ac))]
    print(f"  actions: {' | '.join(ac_parts)}")


def make_sac_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_eval/frontier_sac",
    *,
    use_time_update: bool = True,
    min_time_interval: float = 1.0,
    inv_limit_override: int = INV_LIMIT,
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
        lr_actor=1e-4,
        lr_critic=1e-4,
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
        use_tob_update=False,
        use_event_update=False,
        use_time_update=use_time_update,
        min_time_interval=min_time_interval,
        use_mdp=bool(cc.get("use_mdp", True)),
        use_obs_normalizer=use_obs_norm,
        # SAC-specific (irrelevant for eval, but required by constructor)
        lr_alpha=1e-4,
        tau=0.005,
        alpha_init=0.2,
        target_entropy_ratio=0.6,
        replay_capacity=1000,
        batch_size=128,
        use_dueling=bool(cc.get("use_dueling", True)),
        use_distributional=bool(cc.get("use_distributional", True)),
        n_quantiles=int(cc.get("n_quantiles", 25)),
    )

    # Load actor weights (only actor needed for greedy eval)
    ctrl.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
    if use_obs_norm and "obs_normalizer_state" in ckpt:
        ctrl.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])

    ctrl.enable_learning = False
    ctrl.actor_net.eval()
    ctrl._eval_inv_penalty_coeff = float(rc.get("INV_PENALTY_COEFF", 0.001))
    return ctrl


def make_ppo_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_eval/frontier_ppo",
    *,
    use_time_update: bool = True,
    min_time_interval: float = 1.0,
    inv_limit_override: int = INV_LIMIT,
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
        lr_actor=1e-4,
        lr_critic=1e-4,
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
        use_tob_update=False,
        use_event_update=False,
        use_time_update=use_time_update,
        min_time_interval=min_time_interval,
        use_mdp=bool(cc.get("use_mdp", True)),
        use_obs_normalizer=use_obs_norm,
    )

    # Load actor weights (only actor needed for greedy eval)
    ctrl.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
    if use_obs_norm and "obs_normalizer_state" in ckpt:
        ctrl.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])

    ctrl.enable_learning = False
    ctrl.actor_net.eval()
    ctrl._eval_inv_penalty_coeff = float(rc.get("INV_PENALTY_COEFF", 0.001))
    return ctrl


def run_rl_sims(ctrl, inv_penalty_coeff: float = 0.001) -> dict:
    """Run N_SIMS simulations with any RL controller (SAC, PPO, etc.).

    Returns dict with 'pnls', 'fills', 'action_counts', 'spread_capture', 'inv_penalty'.
    """

    pnls = np.empty(N_SIMS)
    fills = np.empty(N_SIMS)

    n_actions = ctrl.n_actions
    total_action_counts = np.zeros(n_actions, dtype=np.int64)
    total_spread_capture = 0.0
    total_inv_penalty = 0.0

    for i in range(N_SIMS):
        seed = make_seed(i)
        set_seeds(seed)

        ep_sc = [0.0]
        ep_ip = [0.0]
        ep_ac = np.zeros(n_actions, dtype=np.int64)

        _sc = ep_sc
        _ip = ep_ip
        _ac = ep_ac
        _ctrl = ctrl
        _phi = inv_penalty_coeff

        def _tracking_reward(step_idx, mm, lob, sb, sa, info, **kw):
            reward_spread_capture_inv_quadratic(
                step_idx, mm, lob, sb, sa, info, inv_penalty_coeff=_phi, **kw)
            _sc[0] += info.get("_reward_spread_capture", 0.0)
            _ip[0] += info.get("_reward_inv_penalty", 0.0)
            a = _ctrl.last_action_idx
            if a is not None and 0 <= a < len(_ac):
                _ac[a] += 1
            return 0.0

        _, _, mm_df = simulate_LOB_with_MM(
            lam=LAM, mu=MU, delta=DELTA,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100, mean_size_LO=1,
            iterations=N_STEPS,
            iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None,
            label_simulation=None,
            controller=ctrl,
            reward_fn=_tracking_reward,
            random_seed=seed,
        )

        pnls[i] = float(mm_df["MM_TotalPnL"].iloc[-1])
        if mm_df is not None and "MM_HadFill" in mm_df.columns:
            fills[i] = float((mm_df["MM_HadFill"] == True).sum())
        else:
            fills[i] = 0.0
        total_action_counts += ep_ac
        total_spread_capture += ep_sc[0]
        total_inv_penalty += ep_ip[0]

        ac_total = max(1, ep_ac.sum())
        ac_pcts = ep_ac / ac_total * 100.0
        offsets = getattr(ctrl, "pure_mm_offsets", None)
        if offsets is not None:
            ac_str = " ".join(f"{offsets[j]}={ac_pcts[j]:.0f}%" for j in range(n_actions))
        else:
            ac_str = " ".join(f"a{j}={ac_pcts[j]:.0f}%" for j in range(n_actions))
        print(f"    sim {i+1}/{N_SIMS}  PnL={pnls[i]:+.4f}  fills={fills[i]:.0f}  "
              f"SC={ep_sc[0]:.2f}  IP={ep_ip[0]:.2f}  [{ac_str}]")

    try:
        ctrl.writer.close()
    except Exception:
        pass

    return {
        "pnls": pnls,
        "fills": fills,
        "action_counts": total_action_counts,
        "mean_spread_capture": total_spread_capture / N_SIMS,
        "mean_inv_penalty": total_inv_penalty / N_SIMS,
        "offsets": getattr(ctrl, "pure_mm_offsets", None),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_algo_points(
    ax,
    results: Dict[str, np.ndarray],
    color: str,
    label: str,
    annotation_prefix: str = "",
):
    """Helper: plot a set of RL checkpoint results on the frontier."""
    import re

    parsed = []
    for lbl, pnls in results.items():
        m = re.search(r"[φϕ]=([0-9.]+)", lbl)
        phi = float(m.group(1)) if m else 0.0
        parsed.append((phi, lbl, pnls))
    parsed.sort(key=lambda t: t[0])

    xs, ys, xes, yes, labels = [], [], [], [], []
    for phi, lbl, pnls in parsed:
        n = len(pnls)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        xs.append(std_pnl)
        ys.append(mean_pnl)
        xes.append(std_pnl / np.sqrt(2 * (n - 1)))
        yes.append(std_pnl / np.sqrt(n))
        labels.append(f"φ={phi:.3f}" if annotation_prefix == "" else f"{annotation_prefix} φ={phi:.3f}")

    if xs:
        ax.errorbar(
            xs, ys, xerr=xes, yerr=yes,
            fmt="o-", color=color, capsize=3, linewidth=1.5,
            markersize=7, label=label,
        )
        for lbl_txt, x, y in zip(labels, xs, ys):
            ax.annotate(
                lbl_txt, (x, y),
                textcoords="offset points", xytext=(8, 6),
                fontsize=11, color=color, alpha=0.85,
            )


def plot_frontier(
    glft_results: Dict[float, np.ndarray],
    dqn_results: Dict[str, np.ndarray],
    sac_results: Dict[str, np.ndarray] = None,
    ppo_results: Dict[str, np.ndarray] = None,
    naive_mm_pnls: np.ndarray = None,
) -> None:
    """
    Plot the risk-return efficient frontier.

    X-axis: √Var(PnL_T)  — risk proxy  (std of terminal PnL across sims)
    Y-axis: E[PnL_T]     — return proxy (mean terminal PnL across sims)

    GLFT points form a connected curve; RL points are plotted per algorithm.
    """
    if sac_results is None:
        sac_results = {}
    if ppo_results is None:
        ppo_results = {}

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── GLFT frontier ────────────────────────────────────────────────────────
    gammas_sorted = sorted(glft_results.keys())
    glft_x, glft_y, glft_xe, glft_ye = [], [], [], []

    for g in gammas_sorted:
        pnls = glft_results[g]
        n = len(pnls)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        glft_x.append(std_pnl)
        glft_y.append(mean_pnl)
        glft_xe.append(std_pnl / np.sqrt(2 * (n - 1)))
        glft_ye.append(std_pnl / np.sqrt(n))

    if glft_x:
        ax.errorbar(
            glft_x, glft_y, xerr=glft_xe, yerr=glft_ye,
            fmt="o-", color="#0072B2", capsize=3, linewidth=1.5,
            markersize=7, label="GLFT frontier",
        )
        for g, x, y in zip(gammas_sorted, glft_x, glft_y):
            ax.annotate(
                f"γ={g:.0e}", (x, y),
                textcoords="offset points", xytext=(8, 6),
                fontsize=11, color="#0072B2", alpha=0.85,
            )

    # ── RL algorithms ────────────────────────────────────────────────────────
    _plot_algo_points(ax, dqn_results, "#E69F00", "DQN Rainbow")
    _plot_algo_points(ax, sac_results, "darkorange", "SAC-Discrete")
    _plot_algo_points(ax, ppo_results, "forestgreen", "PPO")

    # ── Naive MM benchmark (single point) ────────────────────────────────
    if naive_mm_pnls is not None and len(naive_mm_pnls) > 0:
        n = len(naive_mm_pnls)
        nm_mean = np.mean(naive_mm_pnls)
        nm_std = np.std(naive_mm_pnls, ddof=1)
        nm_xe = nm_std / np.sqrt(2 * (n - 1))
        nm_ye = nm_std / np.sqrt(n)
        ax.errorbar(
            [nm_std], [nm_mean], xerr=[nm_xe], yerr=[nm_ye],
            fmt="D", color="#999999", capsize=3, markersize=9,
            label="At-best baseline", zorder=5,
        )

    ax.set_xlabel(r"Risk:  $\sqrt{\mathrm{Var}(\mathrm{PnL}_T)}$", fontsize=13)
    ax.set_ylabel(r"Return:  $\mathbb{E}[\mathrm{PnL}_T]$", fontsize=13)
    # title removed for paper export (caption describes the figure)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("risk_return_frontier.png", dpi=150)
    plt.show()
    print("Saved: risk_return_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── 1. Calibrate (only needed for GLFT) ─────────────────────────────────
    # ── 2. GLFT gamma sweep ──────────────────────────────────────────────────
    glft_results: Dict[float, np.ndarray] = {}

    if GLFT_GAMMAS:
        A, KAPPA, SIGMA = run_calibration(plot=False)
        for gi, gamma in enumerate(GLFT_GAMMAS):
            print(f"[GLFT {gi+1}/{len(GLFT_GAMMAS)}] gamma={gamma:.1e}  "
                  f"({N_SIMS} sims)...")
            pnls = run_glft_sims(gamma, A, KAPPA, SIGMA)
            glft_results[gamma] = pnls
            print(f"  E[PnL]={np.mean(pnls):+.4f}  "
                  f"√Var={np.std(pnls, ddof=1):.4f}")

    # ── 3. DQN checkpoints ───────────────────────────────────────────────────
    dqn_results: Dict[str, np.ndarray] = {}

    active_dqn = {
        label: path
        for label, path in DQN_CHECKPOINTS.items()
        if os.path.isfile(path)
    }

    if not active_dqn:
        print("\n[DQN] No checkpoint files found — skipping DQN points.\n")
    else:
        for di, (label, path) in enumerate(active_dqn.items()):
            print(f"[DQN {di+1}/{len(active_dqn)}] {label}  ({N_SIMS} sims)...")
            dqn_ckpt = torch.load(path, map_location="cpu", weights_only=False)
            dqn_meta = dqn_ckpt.get("meta", {})
            dqn_phi = float(dqn_meta.get("best_params", {}).get(
                "inv_penalty_coeff", 0.001))
            del dqn_ckpt
            dqn_info = run_dqn_sims(path, inv_penalty_coeff=dqn_phi)
            pnls = dqn_info["pnls"]
            dqn_results[label] = pnls
            _print_rl_summary(label, dqn_info)

    # ── 4. SAC checkpoints ────────────────────────────────────────────────────
    sac_results: Dict[str, np.ndarray] = {}

    active_sac = {
        label: path
        for label, path in SAC_CHECKPOINTS.items()
        if os.path.isfile(path)
    }

    if not active_sac:
        print("\n[SAC] No checkpoint files found — skipping SAC points.\n")
    else:
        for si, (label, path) in enumerate(active_sac.items()):
            print(f"[SAC {si+1}/{len(active_sac)}] {label}  ({N_SIMS} sims)...")
            ctrl = make_sac_controller_from_checkpoint(
                path, use_time_update=True,
                min_time_interval=SAC_TIME_THROTTLE,
                inv_limit_override=INV_LIMIT,
            )
            sac_info = run_rl_sims(ctrl, inv_penalty_coeff=ctrl._eval_inv_penalty_coeff)
            sac_results[label] = sac_info["pnls"]
            _print_rl_summary(label, sac_info)

    # ── 5. PPO checkpoints ────────────────────────────────────────────────────
    ppo_results: Dict[str, np.ndarray] = {}

    active_ppo = {
        label: path
        for label, path in PPO_CHECKPOINTS.items()
        if os.path.isfile(path)
    }

    if not active_ppo:
        print("\n[PPO] No checkpoint files found — skipping PPO points.\n")
    else:
        for pi_, (label, path) in enumerate(active_ppo.items()):
            print(f"[PPO {pi_+1}/{len(active_ppo)}] {label}  ({N_SIMS} sims)...")
            ctrl = make_ppo_controller_from_checkpoint(
                path, use_time_update=True,
                min_time_interval=PPO_TIME_THROTTLE,
                inv_limit_override=INV_LIMIT,
            )
            ppo_info = run_rl_sims(ctrl, inv_penalty_coeff=ctrl._eval_inv_penalty_coeff)
            ppo_results[label] = ppo_info["pnls"]
            _print_rl_summary(label, ppo_info)

    # ── 6. Naive MM benchmark ─────────────────────────────────────────────
    naive_mm_pnls = None
    if PLOT_MM_NAIVE:
        print(f"[MM Naive] Always-Best-Bid/Ask  ({N_SIMS} sims)...")
        naive_mm_pnls = run_naive_mm_sims()
        print(f"  E[PnL]={np.mean(naive_mm_pnls):+.4f}  "
              f"√Var={np.std(naive_mm_pnls, ddof=1):.4f}")

    # ── 7. Summary table ─────────────────────────────────────────────────────
    all_rl = {**dqn_results, **sac_results, **ppo_results}

    print("\n" + "=" * 70)
    print(f"{'Policy':>25s}  {'E[PnL]':>10s}  {'√Var(PnL)':>10s}  "
          f"{'Sharpe':>8s}  {'N':>4s}")
    print("-" * 70)

    for g in sorted(glft_results.keys()):
        p = glft_results[g]
        m, s = np.mean(p), np.std(p, ddof=1)
        sh = m / s if s > 0 else 0
        print(f"{'GLFT γ=' + f'{g:.0e}':>25s}  {m:>+10.4f}  {s:>10.4f}  "
              f"{sh:>8.3f}  {len(p):>4d}")

    for label, p in all_rl.items():
        m, s = np.mean(p), np.std(p, ddof=1)
        sh = m / s if s > 0 else 0
        print(f"{label:>25s}  {m:>+10.4f}  {s:>10.4f}  "
              f"{sh:>8.3f}  {len(p):>4d}")

    if naive_mm_pnls is not None:
        m, s = np.mean(naive_mm_pnls), np.std(naive_mm_pnls, ddof=1)
        sh = m / s if s > 0 else 0
        print(f"{'MM Naive':>25s}  {m:>+10.4f}  {s:>10.4f}  "
              f"{sh:>8.3f}  {len(naive_mm_pnls):>4d}")

    print("=" * 70)

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    plot_frontier(glft_results, dqn_results, sac_results, ppo_results, naive_mm_pnls)


if __name__ == "__main__":
    main()
