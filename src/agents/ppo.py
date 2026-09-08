#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proximal Policy Optimization (PPO) for LOB Market-Making
==========================================================

Standalone module implementing PPO (Schulman et al., 2017) for the
MarketMaker + LOB simulation environment. This module provides:

    1. ``ActorNetwork``     — MLP policy network π_θ(a|s) → softmax logits
    2. ``CriticNetwork``    — MLP state-value network V_ϕ(s) → scalar
    3. ``PPORolloutBuffer`` — On-policy rollout storage with n-step TD support
    4. ``PPOController``    — Full PPO agent with SMDP throttle support
    5. ``run_ppo_sync_batch()`` — Two-phase PPO training function


Theoretical Foundation
======================

PPO (Schulman et al., 2017, arXiv:1707.06347) is a **policy-gradient** method
that stabilises training by constraining each update to a **trust region**
defined implicitly through a clipped probability-ratio objective. Unlike
TRPO (Schulman et al., 2015), which solves a constrained optimisation with
conjugate gradients, PPO achieves a similar effect via a simple clipped
surrogate loss that is compatible with standard first-order optimisers.

The central idea is to maximise the expected advantage under the new policy
π_θ while preventing the policy from diverging too far from the old policy
π_{θ_old} that was used to collect the data:

    L^{CLIP}(θ) = E_t [ min( r_t(θ) · A_t ,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]

where:
    r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)   (probability ratio)
    A_t    = n-step TD advantage estimate
    ε      = clip hyperparameter (typically 0.1–0.3)

This objective is **pessimistic**: for positive advantages (good actions) it
caps the ratio at (1+ε), preventing exploitation of extrapolated gradients;
for negative advantages (bad actions) it caps at (1-ε), limiting how
aggressively the policy can decrease the probability of bad actions in a
single step.


PPO vs Other RL Algorithms for Market-Making
==============================================

PPO is an **on-policy** algorithm — it collects trajectories with the current
policy, uses them for a few update epochs (typically 3–10), then discards
them. This makes it less sample-efficient than off-policy methods (DQN, SAC)
but more stable:

                    PPO              DQN (C51)        SAC-Discrete
    ───────────────────────────────────────────────────────────────
    Learning        On-policy        Off-policy       Off-policy
    Data reuse      ~4× (epochs)     ~100× (replay)   ~100× (replay)
    Critic type     Scalar V(s)      Q(s,a) distrib.  Twin Q(s,a)
    Policy          Explicit π(a|s)  Implicit ε-greedy Explicit π(a|s)
    Exploration     Entropy bonus    NoisyNets / ε     Auto α·H(π)
    Update rule     Clipped ratio    Distributional TD Soft Bellman
    Env coupling    Parallel envs    Sequential eps    Sequential eps
    ───────────────────────────────────────────────────────────────

PPO's scalar critic V(s) can struggle with bimodal return distributions
common in market-making (spread capture vs adverse selection). For this
reason, SAC-Discrete (see ``sac.py``) is also available as an alternative.


Architecture Overview
=====================

The training loop follows the standard PPO two-phase structure:

    ┌─────────────────────────────────────────────────────┐
    │ Phase 1: COLLECT (frozen π_{θ_old})                 │
    │                                                     │
    │   for each env i = 1..N:                            │
    │     Run LOB simulation to completion                │
    │     ↓ per step: act() → learn() → SMDP aggregation  │
    │     ↓ compute log π_{θ_old}(a|s) for each           │
    │     ↓ append to PPORolloutBuffer                    │
    │                                                     │
    │   Result: T transitions with old log-probs           │
    └─────────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────────┐
    │ Phase 2: UPDATE (multi-epoch mini-batch SGD)         │
    │                                                     │
    │   (a) compute_gae_smdp() → GAE advantages + returns   │
    │                                                     │
    │   (b) for epoch = 1..PPO_EPOCHS:                    │
    │         Shuffle T transitions                       │
    │         for each mini-batch of size B:              │
    │           ppo_update_batch() → gradient step        │
    │         ppo_update_batch() per mini-batch            │
    │                                                     │
    │   Result: θ, ϕ updated via clipped surrogate        │
    └─────────────────────────────────────────────────────┘


N-Step TD Advantage Estimation
================================

Advantages are computed using n-step TD returns with a target value
network for stable bootstrapping. For each transition t in a per-env
trajectory of length L:

    look_ahead = min(n_steps, steps_until_done_or_end)

    G_t = Σ_{i=0}^{look_ahead-1} γ_eff_i · r_{t+i}     (n-step return)
    γ_total = γ^{total_k_steps}                           (cumulative discount)

    If terminal:  target_t = G_t
    Else:         target_t = G_t + γ_total · V_target(s_{t+look_ahead})

    Â_t = target_t - V(s_t)

Advantages are normalised: Â = (Â - μ) / (σ + ε).

With n_steps=1, this reduces to standard 1-step TD advantage.


SMDP-Correct Discounting
==========================

When throttled (time-gate, event-gate, TOB-gate), each SMDP transition
spans K_t micro-steps. The controller correctly aggregates rewards and
adjusts the discount factor:

    R^{cum}_t = Σ_{k=0}^{K_t - 1} γ^k · r_{t,k}      (cumulative reward)
    γ_eff = γ^{K_t}                                     (effective discount)

The n-step TD target becomes:

    G_t = R^{cum}_t + γ^{K_t} · R^{cum}_{t+1} + ... (up to n steps)
    target_t = G_t + γ^{total_K} · V_target(s_{t+n}) · (1-d)

This is mathematically equivalent to standard n-step TD under the
Semi-Markov Decision Process framework (Bradtke & Duff, 1994).


Clipped Surrogate Loss
========================

For each mini-batch of transitions, PPO computes:

    r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)

    L^{CLIP} = -E_t [ min( r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t ) ]

The clip function creates a "pessimistic bound":
    - If Â_t > 0 and r_t > 1+ε: gradient is zero (don't exploit too much)
    - If Â_t < 0 and r_t < 1-ε: gradient is zero (don't penalise too much)

This prevents the policy from making large jumps, even when the advantage
estimate suggests a large improvement would be beneficial.


Value Function Loss
====================

The critic is trained to minimise the Smooth L1 (Huber) loss between
its predictions and the n-step TD return targets:

    L_V = SmoothL1(V_ϕ(s_t), target_t)

Huber loss is more robust to outlier targets than MSE, which is
beneficial in the noisy market-making reward landscape.


State Representation
=====================

GENERIC MODE (pure_mm=False):
    s = [log(1+spread), log(1+asksize), log(1+bidsize),
         inventory/inv_limit, has_bid, has_ask]
    Dimension: 6

PURE MM MODE (pure_mm=True):
    s = [log(1+spread), inventory/inv_limit,
         log(1+bid_size_0), ..., log(1+bid_size_K),
         log(1+ask_size_0), ..., log(1+ask_size_K)]
    Dimension: 2 + 2·(K+1)  where K = max_offset


Action Masking
===============

At inventory limits (|inv| = inv_limit), actions that would increase
exposure are masked out (logits set to -∞ before softmax). In PURE MM
mode, canonical equivalence masks further reduce redundancy:

    At +inv_limit (long): only the ask side matters, so actions that
    differ only in bid_offset are functionally equivalent. Only the
    "canonical" representative of each equivalence class is kept.

    At -inv_limit (short): symmetric treatment for the bid side.


References
==========
    [1] Schulman et al., "Proximal Policy Optimization Algorithms",
        arXiv:1707.06347, 2017.
    [2] Schulman et al., "High-Dimensional Continuous Control Using
        Generalized Advantage Estimation", ICLR 2016.
        (Referenced for context; this module uses n-step TD, not GAE.)
    [3] Bradtke & Duff, "Reinforcement Learning Methods for Continuous-Time
        Markov Decision Processes", NeurIPS 1994.
    [4] Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning"
        (A3C), ICML 2016.
    [5] Andrychowicz et al., "What Matters in On-Policy Reinforcement
        Learning? A Large-Scale Empirical Study", 2020 — insights on
        separate actor/critic networks.
    [6] Engstrom et al., "Implementation Matters in Deep RL: A Case Study
        on PPO and TRPO", ICLR 2020 — importance of implementation details.
"""

# ============================================================
# Imports
# ============================================================

import copy
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from mm_base import ActorNetwork, ObservationNormalizer, BaseMMController


# ============================================================
# Section 1: Neural Networks — Critic (Value Function)
# ============================================================
#
# The actor (ActorNetwork) is defined in mm_base.py and imported above.
# The critic (value function) is PPO-specific and defined here.
#
# Critic V_ϕ : S → ℝ        (scalar state-value estimate)
# ============================================================


class CriticNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) state-value network.

    Maps a state vector s ∈ ℝ^D to a scalar value estimate V_ϕ(s) ∈ ℝ.

    The critic approximates the expected discounted return under the
    current policy π_θ:

        V^π(s) = E_π [ Σ_{t=0}^{∞} γ^t · r_t | s_0 = s ]

    In the SMDP setting with variable holding times K_t, this becomes:

        V^π(s) = E_π [ Σ_{t=0}^{∞} γ^{Σ K_j} · R^{cum}_t | s_0 = s ]

    Architecture
    ------------
        Input(D) → [Linear(D, H) → ReLU]^L → Linear(H, 1)

    The single output neuron (no activation) allows V(s) to take any
    real value, including negative values (e.g., when inventory penalty
    dominates spread capture).
    """

    def __init__(
        self,
        input_dim: int,
        n_hidden: int = 2,
        n_neurons: int = 128,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, n_neurons))
            layers.append(nn.ReLU())
            prev = n_neurons
        # Output layer: scalar V(s) ∈ ℝ (no activation)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: s → V_ϕ(s), scalar state-value estimate."""
        return self.net(x)


# ============================================================
# Section 2: PPO Rollout Buffer
# ============================================================
#
# PPO requires storing an entire rollout before performing any gradient
# updates. This is fundamentally different from online algorithms (A2C,
# A3C) that update after every transition or small batch.
#
# The buffer stores all T transitions from N parallel environments:
#     T = Σ_{i=1}^{N} T_i
# where T_i is the number of SMDP transitions from env i.
#
# Each transition is an SMDP tuple (s_t, a_t, R^{cum}_t, s_{t+1}, d_t, K_t)
# augmented with:
#     - log π_{θ_old}(a_t | s_t) : needed for importance-sampling ratio r_t(θ)
#     - env_id                    : needed for per-env n-step TD computation
#     - mask                      : valid action mask at decision time
#
# After collection, compute_nstep_returns() fills in:
#     - returns     targets       : n-step TD targets (for critic loss)
# Advantages are computed later, per mini-batch, as:
#     Â_mb = returns_mb - V_current(states_mb)
# and then normalised within the mini-batch.
# ============================================================


@dataclass
class PPORolloutBuffer:
    """
    On-policy rollout buffer for Proximal Policy Optimization.

    Stores the full set of T = Σ_i T_i transitions collected from N
    parallel LOB simulation environments during Phase 1 (COLLECT).

    Each entry corresponds to a single SMDP transition:

        (s_t, a_t, R^{cum}_t, s_{t+1}, d_t, K_t, mask_t)

    where:
        s_t        : state at the decision point (ℝ^D tensor)
        a_t        : action index chosen by π_{θ_old}
        R^{cum}_t  : cumulative reward over the K_t micro-steps
        s_{t+1}    : next SMDP state (bootstrap state)
        d_t        : whether the episode ended
        K_t        : number of micro-steps spanned by this SMDP transition
        mask_t     : boolean array — True for actions allowed by inventory limits

    After the collection phase, compute_gae_smdp() populates:
        returns    : G_t ∈ ℝ^T — GAE-based return targets (for critic)
        advantages : Â_t ∈ ℝ^T — GAE advantages (frozen for all epochs)
    """

    # ----- Per-transition data (appended during Phase 1) -----
    states: List[torch.Tensor] = field(default_factory=list)
    """s_t : state tensor at the decision point, shape (D,)."""

    actions: List[int] = field(default_factory=list)
    """a_t : integer action index sampled from π_{θ_old}(·|s_t)."""

    rewards: List[float] = field(default_factory=list)
    """R^{cum}_t : cumulative reward over K_t micro-steps (scalar)."""

    next_states: List[torch.Tensor] = field(default_factory=list)
    """s_{t+1} : next SMDP state, shape (D,)."""

    dones: List[bool] = field(default_factory=list)
    """d_t : True if the episode ended at this transition."""

    k_steps: List[int] = field(default_factory=list)
    """K_t : number of micro-steps spanned by this SMDP transition.
    Used for γ^{K_t} discounting in the n-step TD computation."""

    masks: List[Optional[np.ndarray]] = field(default_factory=list)
    """Boolean action mask at decision time. None ≡ all actions valid."""

    old_log_probs: List[float] = field(default_factory=list)
    """log π_{θ_old}(a_t | s_t) : log-probability under the collection policy.
    Used in the importance-sampling ratio r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)."""

    env_ids: List[int] = field(default_factory=list)
    """Integer environment index ∈ {0, ..., N-1}. Required because n-step TD must
    run backward passes independently per trajectory to avoid mixing
    value bootstraps across unrelated episodes."""

    # ----- Populated by compute_gae_smdp() after Phase 1 -----
    advantages: Optional[torch.Tensor] = None
    """GAE advantages Â_t — frozen for all PPO epochs."""

    returns: Optional[torch.Tensor] = None
    """GAE-based return targets G_t = Â_t + V(s_t)."""

    values: Optional[torch.Tensor] = None
    """Legacy field (unused by current training loop)."""

    def __len__(self) -> int:
        return len(self.states)


# ============================================================
# Section 3: GAE-SMDP Advantage & Return Computation
# ============================================================
#
# Generalized Advantage Estimation (Schulman et al., 2016) adapted
# for Semi-Markov Decision Processes (variable-length transitions).
#
# For each transition t with k_t micro-steps:
#
#   δ_t = r_t + γ^{k_t} · V(s_{t+1}) · (1-d_t) - V(s_t)
#   Â_t = Σ_{l≥0} (γ^k · λ^k)^l · δ_{t+l}   (backward sweep)
#   G_t = Â_t + V(s_t)
#
# Historical note — the original n-step TD return formula:
#   G_t^(n) = Σ_{j=0}^{L-1} d_j · R_{t+j}^{cum}
#             + d_L · V_target(s_{t+L}) · 1[not terminal before L]
#
# where:
#   L = min(n_steps, remaining steps in this env trajectory)
#   d_0 = 1
#   d_{j+1} = d_j · gamma^(k_{t+j})
#   R_{t+j}^{cum} = buffer.rewards[idx]
#   k_{t+j}       = buffer.k_steps[idx]
#
# In code:
#   discount starts at 1.0
#   G += discount * buffer.rewards[idx]
#   discount *= gamma ** buffer.k_steps[idx]
#   if not done_final: G += discount * V_target(next_state_of_last_step)
#
# Larger n_steps reduce bias at the cost of higher variance.
# ============================================================


def compute_gae_smdp(
    buffer: PPORolloutBuffer,
    critic_net: nn.Module,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    target_value_net: nn.Module = None,
) -> None:
    """
    Generalized Advantage Estimation (GAE) adapted for SMDP transitions.

    GAE (Schulman et al., 2016) exponentially blends all n-step TD
    errors via the λ parameter, smoothing out LOB noise while
    preserving the long-horizon signal:

        δ_t = r_t + γ^{k_t} · V(s_{t+1}) · (1-d_t) - V(s_t)

        Â_t = Σ_{l=0}^{∞} (γ^{k} · λ^{k})^l · δ_{t+l}

    The SMDP adaptation scales both γ and λ by the number of
    micro-steps k_t in each decision interval, so that transitions
    spanning more time contribute proportionally less to each
    other's advantage estimates.

    Sets buffer.returns and buffer.advantages in-place.
    """
    T = len(buffer)
    if T == 0:
        buffer.returns = torch.zeros(0)
        buffer.advantages = torch.zeros(0)
        return

    # Batch forward pass: V(s) from critic_net, V(s') from target_value_net
    # Using target_value_net for bootstrap V(s') stabilises the TD error
    # (the target doesn't shift as the critic updates across PPO epochs).
    bootstrap_net = target_value_net if target_value_net is not None else critic_net
    states_t = torch.stack(buffer.states, dim=0).to(device)
    next_states_t = torch.stack(buffer.next_states, dim=0).to(device)

    vals_list, next_vals_list = [], []
    chunk_size = 4096

    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            vals_list.append(
                critic_net(states_t[start:end]).view(-1).cpu()
            )
            next_vals_list.append(
                bootstrap_net(next_states_t[start:end]).view(-1).cpu()
            )

    values = torch.cat(vals_list, dim=0)
    next_values = torch.cat(next_vals_list, dim=0)

    advantages = torch.zeros(T, dtype=torch.float32)
    returns = torch.zeros(T, dtype=torch.float32)

    # Group by env_id to avoid leaking advantages across episodes
    env_indices: Dict[int, List[int]] = {}
    for i, eid in enumerate(buffer.env_ids):
        env_indices.setdefault(eid, []).append(i)

    # Backward GAE sweep per environment
    for eid, indices in env_indices.items():
        last_gae = 0.0
        for pos in reversed(indices):
            r_i = float(buffer.rewards[pos])
            k_i = float(buffer.k_steps[pos])
            d_i = float(buffer.dones[pos])

            # SMDP effective discount
            gamma_eff = gamma ** k_i

            # TD error δ = r + γ^k · V(s') · (1-d) - V(s)
            delta = r_i + gamma_eff * float(next_values[pos]) * (1.0 - d_i) - float(values[pos])

            # GAE accumulation: λ also decays with micro-steps
            last_gae = delta + gamma_eff * (gae_lambda ** k_i) * (1.0 - d_i) * last_gae

            advantages[pos] = last_gae
            returns[pos] = last_gae + float(values[pos])

    buffer.returns = returns
    buffer.advantages = advantages



# ============================================================
# Section 4: PPO Update Function — Clipped Surrogate Loss
# ============================================================


def ppo_update_batch(
    actor_net: nn.Module,
    critic_net: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    masks: list,
    clip_eps: float,
    entropy_coef: float,
    entropy_floor: float,
    grad_clip_norm: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    PPO clipped surrogate loss update on a single mini-batch.

    Returns and advantages are pre-computed GLOBALLY over the entire buffer
    and sliced per mini-batch. This preserves state-conditional signal that
    would be washed out by per-mini-batch normalization.

    Value step:  value_loss = SmoothL1(V(s), returns_norm)
    Policy step: loss = (-min(surr1, surr2) - c_H · entropy).mean()

    Entropy floor: when mean entropy drops below entropy_floor, the
    effective entropy coefficient is boosted by 5× to resist collapse.
    """
    # ---- Value loss (Smooth L1 / Huber) ----
    # Smooth L1 clips gradients for large errors (grad=1 when |e|>1),
    # preventing outlier returns from destabilising critic updates.
    state_values = critic_net(states).squeeze(-1)
    value_loss = F.smooth_l1_loss(state_values, returns)

    critic_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(critic_net.parameters(), grad_clip_norm)
    critic_optimizer.step()

    # ---- Policy forward pass with action masking ----
    logits = actor_net(states)

    has_masks = any(m is not None for m in masks)
    if has_masks:
        n_act = logits.shape[-1]
        mask_np = np.stack([
            m if m is not None else np.ones(n_act, dtype=bool)
            for m in masks
        ], axis=0)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~mask_tensor, float("-inf"))

    log_probs_all = F.log_softmax(logits, dim=-1)
    probs_all = torch.exp(log_probs_all)
    new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # ---- Importance-sampling ratio ----
    rho = torch.exp(new_log_probs - old_log_probs)

    # ---- Clipped surrogate + entropy → combined policy loss ----
    surrogate_1 = rho * advantages
    surrogate_2 = rho.clip(1 - clip_eps, 1 + clip_eps) * advantages

    # Zero out -inf log-probs from masked actions so they contribute
    # exactly 0 to entropy (p=0 ⇒ p·log(p)=0) without killing gradients.
    safe_log = torch.where(torch.isinf(log_probs_all),
                           torch.zeros_like(log_probs_all), log_probs_all)
    entropy = -torch.sum(probs_all * safe_log, dim=-1)

    # Adaptive entropy coefficient: boost 5× when entropy drops below floor
    mean_entropy = entropy.mean().item()
    ent_coef_eff = entropy_coef * 5.0 if mean_entropy < entropy_floor else entropy_coef

    policy_loss = -torch.minimum(surrogate_1, surrogate_2)
    loss = (policy_loss - ent_coef_eff * entropy).mean()

    actor_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor_net.parameters(), grad_clip_norm)
    actor_optimizer.step()

    # ---- Trust-region diagnostics ----
    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        approx_kl = float(((rho - 1) - log_ratio).mean().item())
        clip_fraction = float(
            ((rho < 1 - clip_eps) | (rho > 1 + clip_eps)).float().mean().item()
        )

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.mean().item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
        "mean_value": float(state_values.mean().item()),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "ent_coef_eff": ent_coef_eff,
    }


# ============================================================
# Section 6: PPO Controller
# ============================================================
#
# PPOController extends BaseMMController (mm_base.py) with PPO-specific
# components: critic network, target value network, episode buffers,
# reward normalizer, and the PPO learn() method.
#
# Shared infrastructure (act, state encoding, throttle gates, SMDP,
# action masking) is inherited from BaseMMController.
#
# Gradient updates are NOT performed here — they are handled externally
# by run_ppo_sync_batch() (Section 7).
# ============================================================


class PPOController(BaseMMController):
    """
    PPO controller for the MarketMaker — LOB environment interface.

    Implements the RLController protocol with:
      - Categorical policy π_θ(a|s) via softmax over masked logits
      - State-value critic V_ϕ(s) for n-step TD advantage estimation
      - SMDP throttle gating with cumulative reward aggregation
      - Inventory-limit action masking (canonical equivalence in pure MM)

    Gradient updates are NOT performed here — they are handled externally
    by run_ppo_sync_batch() (Section 7), which implements the two-phase
    PPO algorithm (COLLECT → multi-epoch mini-batch UPDATE).

    Parameters
    ----------
    level_offset : int
        Price offset for generic mode quotes (ticks away from BBO).
    n_actions : int
        Number of discrete actions |A| (used in generic mode only).
    gamma : float
        SMDP discount factor γ ∈ (0, 1). Controls the effective planning
        horizon T_eff ≈ 1/(1-γ) SMDP transitions.
    lr_actor, lr_critic : float
        Learning rates for the actor and critic AdamW optimisers.
    weight_decay : float
        L2 regularisation coefficient for AdamW (decoupled weight decay).
    entropy_coef : float
        Entropy bonus coefficient c_H. Annealed externally by the runner.
    grad_clip_norm : float
        Maximum global gradient norm (gradient clipping).
    device : torch.device
        Computation device (CPU/CUDA).
    pure_mm : bool
        If True, use PURE MM mode with offset-pair action space.
    inv_limit : int or None
        Maximum absolute inventory. Actions that would breach this limit
        are masked out. None = no limit.
    pure_mm_offsets : list of (int, int)
        Offset pairs (bid_off, ask_off) defining the action space in PURE MM mode.
    n_hidden_actor, n_neurons_actor : int
        Actor MLP architecture: number of hidden layers and neurons per layer.
    n_hidden_critic, n_neurons_critic : int
        Critic MLP architecture.
    enable_learning : bool
        If False, networks are set to eval mode and act() uses argmax
        (greedy policy) instead of sampling.
    use_tob_update, n_tob_moves : bool, int
        TOB-update throttle gate configuration.
    use_event_update, n_events : bool, int
        Event-update throttle gate configuration.
    use_time_update, min_time_interval : bool, float
        Time-update throttle gate configuration.
    use_mdp : bool
        If True, disable fill/mode-change bypasses — agent strictly respects
        throttle gates only (no shortcuts on fills or inventory limit crossings).
    """

    def __init__(
        self,
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.97,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        weight_decay: float = 0.01,
        entropy_coef: float = 0.01,
        entropy_floor: float = 0.0,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = "runs/ppo_mm",
        # ----- MM PURE mode -----
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        # ----- Actor architecture -----
        n_hidden_actor: int = 2,
        n_neurons_actor: int = 128,
        # ----- Critic architecture -----
        n_hidden_critic: int = 2,
        n_neurons_critic: int = 128,
        # ----- Learning toggle -----
        enable_learning: bool = True,
        # ----- Throttle gating -----
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_event_update: bool = False,
        n_events: int = 100,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,
        # ----- MDP mode -----
        use_mdp: bool = True,
        # ----- Observation normalisation -----
        use_obs_normalizer: bool = True,
    ):
        # ─────────────────────────────────────────────────────────
        # Initialise BaseMMController (actor, obs_normalizer, SMDP,
        # throttle gates, action masking, device, pure_mm config).
        # ─────────────────────────────────────────────────────────
        super().__init__(
            level_offset=level_offset,
            n_actions=n_actions,
            gamma=gamma,
            lr_actor=lr_actor,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            device=device,
            pure_mm=pure_mm,
            inv_limit=inv_limit,
            pure_mm_offsets=pure_mm_offsets,
            n_hidden_actor=n_hidden_actor,
            n_neurons_actor=n_neurons_actor,
            enable_learning=enable_learning,
            use_tob_update=use_tob_update,
            n_tob_moves=n_tob_moves,
            use_event_update=use_event_update,
            n_events=n_events,
            use_time_update=use_time_update,
            min_time_interval=min_time_interval,
            use_mdp=use_mdp,
            use_obs_normalizer=use_obs_normalizer,
        )

        # =============================================================
        # PPO-SPECIFIC HYPERPARAMETERS
        # =============================================================
        self.entropy_coef = float(entropy_coef)
        self.entropy_floor = float(entropy_floor)
        self.critic_n_hidden = int(n_hidden_critic)
        self.critic_n_neurons = int(n_neurons_critic)

        # =============================================================
        # CRITIC NETWORK — State-Value Function V_ϕ(s)
        # =============================================================
        self.critic_net = CriticNetwork(
            input_dim=self.input_dim,
            n_hidden=self.critic_n_hidden,
            n_neurons=self.critic_n_neurons,
        ).to(self.device)

        # =============================================================
        # TARGET VALUE NETWORK — Lagging Copy for Stable Bootstrap
        # =============================================================
        # Frozen deep copy of critic. Used for V(s') bootstrap when
        # computing n-step TD returns. Synced at the END of each PPO
        # epoch (not before), so it lags behind the current critic —
        # this stabilises the targets during the epoch's mini-batch
        # updates, matching the pattern in the reference Lightning PPO.
        # =============================================================
        self.target_value_net = copy.deepcopy(self.critic_net)
        self.target_value_net.requires_grad_(False)
        self.target_value_net.eval()

        # =============================================================
        # CRITIC OPTIMIZER
        # =============================================================
        self.critic_optimizer = AdamW(
            self.critic_net.parameters(), lr=lr_critic, weight_decay=self.weight_decay,
        )

        # Toggle critic train/eval (actor handled by base)
        if self._enable_learning:
            self.critic_net.train()
        else:
            self.critic_net.eval()

        # =============================================================
        # EPISODE BUFFERS — Per-worker SMDP transition storage
        # =============================================================
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_next_states: List[torch.Tensor] = []
        self.episode_dones: List[bool] = []
        self.episode_k_steps: List[int] = []
        self.episode_masks: List[Optional[np.ndarray]] = []

        self.episode_idx: int = 0

        # Setup printout (master only)
        if log_dir is not None:
            print("\n================ PPO CONTROLLER SETUP ================")
            print(f"pure_mm            : {self.pure_mm}")
            print(f"inv_limit          : {self.inv_limit}")
            if self.pure_mm:
                print(f"pure_mm_offsets    : {self.pure_mm_offsets}")
                print(f"max_offset         : {self.max_offset}")
                print(f"input_dim          : {self.input_dim}")
            print(f"gamma              : {self.gamma}")
            print(f"n_actions          : {self.n_actions}")
            print(f"device             : {self.device}")
            print("======================================================\n")

    # ================================================================
    # WORKER FACTORY — Parallel Environment Clones
    # ================================================================

    def make_worker(self) -> "PPOController":
        """
        Create a lightweight worker clone that shares the master's networks.

        Workers are used in the Phase 1 (COLLECT) of PPO training. Each worker
        runs an independent LOB simulation but shares the same actor_net,
        critic_net, and optimisers with the master controller.

        Shared (by reference):
            actor_net, critic_net, actor_optimizer, critic_optimizer

        Independent (per worker):
            episode_states/actions/rewards/..., throttle counters, SMDP state

        This design avoids copying network parameters N times and ensures
        that all workers use the same (frozen) policy during collection.
        """
        worker = PPOController(
            level_offset=self.level_offset,
            n_actions=self.n_actions,
            gamma=self.gamma,
            lr_actor=self.actor_optimizer.param_groups[0]["lr"],
            lr_critic=self.critic_optimizer.param_groups[0]["lr"],
            weight_decay=self.weight_decay,
            entropy_coef=self.entropy_coef,
            entropy_floor=self.entropy_floor,
            grad_clip_norm=self.grad_clip_norm,
            device=self.device,
            log_dir=None,
            pure_mm=self.pure_mm,
            inv_limit=self.inv_limit,
            pure_mm_offsets=list(self.pure_mm_offsets) if self.pure_mm_offsets else None,
            n_hidden_actor=self.actor_n_hidden,
            n_neurons_actor=self.actor_n_neurons,
            n_hidden_critic=self.critic_n_hidden,
            n_neurons_critic=self.critic_n_neurons,
            enable_learning=True,
            use_tob_update=self.use_tob_update,
            n_tob_moves=self.threshold_tob,
            use_event_update=self.use_event_update,
            n_events=self.threshold_events,
            use_time_update=self.use_time_update,
            min_time_interval=self.min_time_interval,
            use_mdp=self.use_mdp,
        )
        # Share networks, optimizers, and obs normalizer
        worker.actor_net = self.actor_net
        worker.critic_net = self.critic_net
        worker.target_value_net = self.target_value_net
        worker.actor_optimizer = self.actor_optimizer
        worker.critic_optimizer = self.critic_optimizer
        worker.obs_normalizer = self.obs_normalizer  # shared running stats
        return worker

    # ================================================================
    # RLController INTERFACE: learn()
    # ================================================================

    def learn(
        self,
        step_idx: int,
        mm,
        lob,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """
        SMDP-aware learning step — accumulate rewards and commit transitions.

        Called at every simulation step (after act()). Implements the SMDP
        reward aggregation logic:

        CASE A — New decision arrived (_last_was_decision = True):
            If there was a pending SMDP transition from a previous decision,
            commit it to the episode buffer as:
                (s_{t-1}, a_{t-1}, R^{cum}_{t-1}, s_t, False, K_{t-1}, mask_{t-1})

            Then start a new SMDP accumulation:
                s_decision = state_before
                a_decision = last_action_idx
                R^{cum} = reward           (first micro-step reward)
                K = 1                      (one micro-step so far)

        CASE B — Throttled micro-step (_last_was_decision = False):
            Accumulate reward with SMDP discounting:
                R^{cum} += γ^K · reward
                K += 1

        SPECIAL CASE — Episode termination (done = True):
            Commit the pending SMDP transition with done=True:
                (s_t, a_t, R^{cum}_t, s_terminal, True, K_t, mask_t)
            Reset all SMDP state.

        Note: In PPO, learn() does NOT perform gradient updates. It only
        stores transitions. Gradient updates happen in run_ppo_sync_batch()
        after the entire rollout is collected.
        """
        if self.last_action_idx is None:
            return

        done_flag = bool(info.get("done", False))

        if not self._enable_learning:
            if done_flag:
                self.last_action_idx = None
                self._smdp_reset()
                self._reset_throttle_state()
            return

        if self._last_was_decision:
            # CASE A: New decision arrived — commit old, start new.
            # Use cached tensors from act() to guarantee the stored state
            # matches the exact normalizer snapshot the actor saw.
            if self._smdp_pending:
                s_t_cpu = self._smdp_state_tensor_decision
                s_next_t_cpu = self._last_act_state_tensor

                self.episode_states.append(s_t_cpu)
                self.episode_actions.append(int(self._smdp_a_decision))
                self.episode_rewards.append(float(self._smdp_cum_reward))
                self.episode_next_states.append(s_next_t_cpu)
                self.episode_dones.append(False)
                self.episode_k_steps.append(int(self._smdp_k_steps))
                self.episode_masks.append(self._smdp_mask_decision)

            self._smdp_pending = True
            self._smdp_s_decision = state_before
            self._smdp_state_tensor_decision = self._last_act_state_tensor
            self._smdp_a_decision = self.last_action_idx
            self._smdp_mask_decision = self._last_valid_mask
            self._smdp_cum_reward = float(reward)
            self._smdp_k_steps = 1
        else:
            # CASE B: Throttled micro-step
            if self._smdp_pending:
                k = self._smdp_k_steps
                self._smdp_cum_reward += (self.gamma ** k) * float(reward)
                self._smdp_k_steps += 1

        # SPECIAL CASE: Episode termination
        if done_flag:
            if self._smdp_pending:
                s_t_cpu = self._smdp_state_tensor_decision
                # Terminal s_next: re-encode is fine because done=True
                # zeroes the bootstrap V(s') in n-step targets anyway.
                s_next_t = self._state_to_tensor(state_after)
                s_next_t_cpu = s_next_t.squeeze(0).detach().cpu()

                self.episode_states.append(s_t_cpu)
                self.episode_actions.append(int(self._smdp_a_decision))
                self.episode_rewards.append(float(self._smdp_cum_reward))
                self.episode_next_states.append(s_next_t_cpu)
                self.episode_dones.append(True)
                self.episode_k_steps.append(int(self._smdp_k_steps))
                self.episode_masks.append(self._smdp_mask_decision)

                self._smdp_pending = False

            self.last_action_idx = None
            self._smdp_reset()
            self._reset_throttle_state()

    # ================================================================
    # EPISODIC FINISH — Reset for Next Episode
    # ================================================================

    def finish_episode(self, total_reward: float = 0.0) -> None:
        """
        Finalise the current episode and reset all per-episode state.

        Clears episode buffers, resets SMDP aggregation state, and
        resets throttle counters. Called by the training loop after
        each environment has completed its simulation.
        """
        self.episode_idx += 1
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_next_states.clear()
        self.episode_dones.clear()
        self.episode_k_steps.clear()
        self.episode_masks.clear()
        self.last_action_idx = None
        self._smdp_reset()
        self._reset_throttle_state()

    # ================================================================
    # ENABLE LEARNING TOGGLE — Override to include critic networks
    # ================================================================

    @BaseMMController.enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        """Toggle training/eval mode for actor (via base) + critic + target."""
        # Base class handles actor_net and _enable_learning flag
        BaseMMController.enable_learning.fset(self, value)
        if self._enable_learning:
            self.critic_net.train()
        else:
            self.critic_net.eval()


# ============================================================
# Section 7: PPO Training Function — Two-Phase Algorithm
# ============================================================
#
# This function implements the complete PPO training cycle for one
# "rollout" (all N environments run to completion, then multi-epoch update).
#
# The algorithm follows the standard PPO two-phase structure:
#
#     ┌─────────────────────────────────────────────────┐
#     │ Phase 1: COLLECT (frozen π_{θ_old})             │
#     │                                                 │
#     │   for each env i = 1..N:                        │
#     │     Run LOB simulation to completion            │
#     │     ↓ per step: act() → learn() → SMDP         │
#     │     ↓ compute log π_{θ_old}(a|s) for each      │
#     │     ↓ append to PPORolloutBuffer                │
#     │                                                 │
#     │   Result: T transitions with old log-probs      │
#     └─────────────────────────────────────────────────┘
#                          ↓
#     ┌─────────────────────────────────────────────────┐
#     │ Phase 2: UPDATE (multi-epoch mini-batch SGD)    │
#     │                                                 │
#     │   (a) compute_gae_smdp() → GAE advantages + returns│
#     │                                                 │
#     │   (b) for epoch = 1..PPO_EPOCHS:                │
#     │         Shuffle T transitions                   │
#     │         for each mini-batch of size B:          │
#     │           ppo_update_batch() → gradient step    │
#     │                                                 │
#     │   Result: θ, ϕ updated via clipped surrogate    │
#     └─────────────────────────────────────────────────┘
#                          ↓
#     ┌─────────────────────────────────────────────────┐
#     │ Phase 3: METRICS — Aggregate performance stats  │
#     │   PnL, inventory, fills, explained variance,    │
#     │   clip fraction, KL, losses, per-epoch stats    │
#     └─────────────────────────────────────────────────┘
#
# The key invariant is that ALL transitions in the buffer were collected
# under the SAME policy π_{θ_old}. The old log-probabilities are frozen
# and used to compute importance-sampling ratios r_t(θ) during Phase 2.
# ============================================================


def run_ppo_sync_batch(
    master_ctrl: PPOController,
    n_envs: int,
    sim_kwargs: dict,
    reward_fn,
    entropy_coef: float,
    entropy_floor: float = 0.0,
    ppo_epochs: int = 4,
    ppo_mini_batch_size: int = 64,
    clip_eps: float = 0.2,
    gae_lambda: float = 0.95,
) -> dict:
    """
    Run one complete PPO rollout: COLLECT + multi-epoch UPDATE + metrics.

    Returns dict with: loss, entropy, mean_reward, mean_pnl,
    mean_abs_inv, max_abs_inv, mm_dfs, pnls.
    """
    from MM_LOB_SIM import simulate_LOB_with_MM_generator

    # ==================================================================
    # PHASE 1: COLLECT — Run N environments with frozen π_{θ_old}
    # ==================================================================
    # All N environments share the master's actor_net and critic_net via
    # make_worker(). No gradient updates occur during collection — the
    # policy is frozen. Each environment gets a unique random seed for
    # independent LOB dynamics (seed diversity prevents correlated
    # trajectories and reduces variance of the batch gradient).
    # ==================================================================

    workers = [master_ctrl.make_worker() for _ in range(n_envs)]

    generators = []
    base_seed = sim_kwargs.get("random_seed", None)
    for i, worker in enumerate(workers):
        env_kwargs = dict(sim_kwargs)
        env_kwargs["controller"] = worker
        env_kwargs["reward_fn"] = reward_fn
        # Each env gets a unique seed: base + i * 10_000
        if base_seed is not None:
            env_kwargs["random_seed"] = int(base_seed) + i * 10_000
        generators.append(simulate_LOB_with_MM_generator(**env_kwargs))

    gamma = master_ctrl.gamma
    inv_limit = master_ctrl.inv_limit

    pending_steps: List[Optional[Dict[str, Any]]] = [None] * n_envs
    active = [True] * n_envs
    total_rewards = [0.0] * n_envs
    mm_dfs = [None] * n_envs

    ppo_buffer = PPORolloutBuffer()  # Central buffer for all N environments

    while any(active):
        for idx, gen in enumerate(generators):
            if (not active[idx]) or (pending_steps[idx] is not None):
                continue
            try:
                step = next(gen)
            except StopIteration:
                active[idx] = False
                continue
            if step.get("done_final"):
                mm_dfs[idx] = step.get("mm_df")
                total_rewards[idx] = float(step.get("total_reward", total_rewards[idx]))
                active[idx] = False
                continue
            pending_steps[idx] = step

        ready_indices = [idx for idx, step in enumerate(pending_steps) if step is not None]
        if not ready_indices:
            continue

        # (a) learn() for SMDP aggregation:
        #     Each worker's learn() accumulates reward into its SMDP
        #     cumulative reward R^{cum} and commits complete SMDP
        #     transitions to the worker's episode_* buffers.
        for idx in ready_indices:
            step = pending_steps[idx]
            workers[idx].learn(
                step["step_idx"], step["mm"], step["lob"],
                step["state_before"], step["state_after"],
                step["reward"], step["info"],
            )
            total_rewards[idx] += float(step["reward"])

        # (b) Move committed SMDP transitions into the PPO buffer.
        #     Vectorized: one batched forward pass per worker instead of
        #     O(T) individual calls, avoiding GPU launch overhead.
        for idx in ready_indices:
            w = workers[idx]
            ep_len = len(w.episode_states)
            if ep_len > 0:
                with torch.no_grad():
                    states_t = torch.stack(w.episode_states, dim=0).to(master_ctrl.device)
                    actions_t = torch.tensor(
                        w.episode_actions, dtype=torch.long, device=master_ctrl.device,
                    )
                    logits = master_ctrl.actor_net(states_t)

                    has_masks = any(m is not None for m in w.episode_masks)
                    if has_masks:
                        n_act = logits.shape[-1]
                        mask_np = np.stack([
                            m if m is not None else np.ones(n_act, dtype=bool)
                            for m in w.episode_masks
                        ], axis=0)
                        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=master_ctrl.device)
                        logits = logits.masked_fill(~mask_t, float("-inf"))

                    log_probs_all = F.log_softmax(logits, dim=-1)
                    old_lps = log_probs_all.gather(
                        1, actions_t.unsqueeze(1),
                    ).squeeze(1).cpu().tolist()

                ppo_buffer.states.extend(w.episode_states)
                ppo_buffer.actions.extend(w.episode_actions)
                ppo_buffer.rewards.extend(w.episode_rewards)
                ppo_buffer.next_states.extend(w.episode_next_states)
                ppo_buffer.dones.extend(w.episode_dones)
                ppo_buffer.k_steps.extend(w.episode_k_steps)
                ppo_buffer.masks.extend(w.episode_masks)
                ppo_buffer.old_log_probs.extend(old_lps)
                ppo_buffer.env_ids.extend([idx] * ep_len)

            w.episode_states.clear()
            w.episode_actions.clear()
            w.episode_rewards.clear()
            w.episode_next_states.clear()
            w.episode_dones.clear()
            w.episode_k_steps.clear()
            w.episode_masks.clear()
            pending_steps[idx] = None

    # ==================================================================
    # PHASE 2: N-STEP TD RETURNS + MULTI-EPOCH PPO UPDATES
    # ==================================================================

    T = len(ppo_buffer)
    n_updates = 0
    loss_accum = 0.0
    policy_loss_accum = 0.0
    value_loss_accum = 0.0
    entropy_accum = 0.0
    mean_value_accum = 0.0
    kl_accum = 0.0
    clip_frac_accum = 0.0

    adv_mean_raw = 0.0
    adv_std_raw = 0.0

    if T > 0:
        # Tensorize buffer once (immutable across all epochs)
        all_states = torch.stack(ppo_buffer.states, dim=0)
        all_actions = torch.tensor(ppo_buffer.actions, dtype=torch.long)
        all_old_lps = torch.tensor(ppo_buffer.old_log_probs, dtype=torch.float32)
        all_masks = ppo_buffer.masks

        # ── CANONICAL PPO WITH GAE-SMDP ──
        # GAE computes both returns and advantages in a single backward
        # sweep through each per-env trajectory. Both are FROZEN for the
        # entire multi-epoch update, preserving the trust region guarantee.
        compute_gae_smdp(
            ppo_buffer, master_ctrl.critic_net,
            gamma, gae_lambda, master_ctrl.device,
            target_value_net=master_ctrl.target_value_net,
        )
        all_returns_raw = ppo_buffer.returns
        all_adv = ppo_buffer.advantages.clone()

        with torch.no_grad():
            adv_mean_raw = float(all_adv.mean().item())
            adv_std_raw = float(all_adv.std().item())
            # Global normalisation (filters market-noise outliers)
            if all_adv.numel() > 1:
                all_adv = (all_adv - adv_mean_raw) / (adv_std_raw + 1e-8)

        # ── MULTI-EPOCH UPDATE (advantages & returns are immutable) ──
        for epoch in range(ppo_epochs):
            perm = torch.randperm(T)

            for start in range(0, T, ppo_mini_batch_size):
                end = min(start + ppo_mini_batch_size, T)
                mb_idx = perm[start:end]

                mb_states = all_states[mb_idx].to(master_ctrl.device)
                mb_actions = all_actions[mb_idx].to(master_ctrl.device)
                mb_old_lps = all_old_lps[mb_idx].to(master_ctrl.device)
                mb_returns = all_returns_raw[mb_idx].to(master_ctrl.device)
                mb_adv = all_adv[mb_idx].to(master_ctrl.device)
                mb_masks = [all_masks[i] for i in mb_idx.tolist()]

                stats = ppo_update_batch(
                    actor_net=master_ctrl.actor_net,
                    critic_net=master_ctrl.critic_net,
                    actor_optimizer=master_ctrl.actor_optimizer,
                    critic_optimizer=master_ctrl.critic_optimizer,
                    states=mb_states,
                    actions=mb_actions,
                    old_log_probs=mb_old_lps,
                    returns=mb_returns,
                    advantages=mb_adv,
                    masks=mb_masks,
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    entropy_floor=entropy_floor,
                    grad_clip_norm=master_ctrl.grad_clip_norm,
                    device=master_ctrl.device,
                )

                n_updates += 1
                loss_accum += stats["loss"]
                policy_loss_accum += stats["policy_loss"]
                value_loss_accum += stats["value_loss"]
                entropy_accum += stats["entropy"]
                mean_value_accum += stats["mean_value"]
                kl_accum += stats["approx_kl"]
                clip_frac_accum += stats["clip_fraction"]

        # Sync target ← critic after all epochs (for next rollout's returns)
        master_ctrl.target_value_net.load_state_dict(
            master_ctrl.critic_net.state_dict()
        )

    # ==================================================================
    # PHASE 3: Simplified Metrics
    # ==================================================================

    pnls = []
    abs_invs_mean = []
    abs_invs_max = []

    for mm_df in mm_dfs:
        if mm_df is None:
            continue
        if "MM_TotalPnL" in mm_df.columns:
            pnls.append(float(mm_df["MM_TotalPnL"].iloc[-1]))
        if "MM_Inventory" in mm_df.columns:
            inv_series = mm_df["MM_Inventory"].values.astype(float)
            abs_inv = np.abs(inv_series)
            abs_invs_mean.append(float(np.mean(abs_inv)))
            abs_invs_max.append(float(np.max(abs_inv)))

    # Mean target return from the buffer (compare with mean_value for calibration)
    mean_return = float(ppo_buffer.returns.mean().item()) if T > 0 else 0.0

    _nu = max(1, n_updates)
    metrics = {
        "loss": loss_accum / _nu,
        "policy_loss": policy_loss_accum / _nu,
        "value_loss": value_loss_accum / _nu,
        "entropy": entropy_accum / _nu,
        "mean_value": mean_value_accum / _nu,
        "mean_return": mean_return,
        "adv_mean_raw": adv_mean_raw if T > 0 else 0.0,
        "adv_std_raw": adv_std_raw if T > 0 else 0.0,
        "approx_kl": kl_accum / _nu,
        "clip_fraction": clip_frac_accum / _nu,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "std_pnl": float(np.std(pnls)) if len(pnls) > 1 else 0.0,
        "mean_abs_inv": float(np.mean(abs_invs_mean)) if abs_invs_mean else 0.0,
        "max_abs_inv": float(np.mean(abs_invs_max)) if abs_invs_max else 0.0,
        "buffer_size": T,
        "mm_dfs": mm_dfs,
        "pnls": pnls,
    }

    return metrics
