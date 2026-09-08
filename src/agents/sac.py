#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAC-Discrete: Soft Actor-Critic for Discrete Action Spaces
============================================================

Standalone module implementing SAC-Discrete (Christodoulou, 2019) for the
MarketMaker + LOB simulation environment. This module provides two classes:

    1. ``QNetwork``              — Twin Q-network architecture (action-value MLP)
    2. ``SACDiscreteController`` — Full SAC-Discrete agent with SMDP support


Theoretical Foundation
======================

SAC-Discrete adapts the continuous Soft Actor-Critic (Haarnoja et al., 2018)
to discrete action spaces. The key insight is that the maximum-entropy RL
framework can be applied to categorical policies by replacing the continuous
reparameterization trick with exact expectation over the finite action set.

The agent maximises the **entropy-regularised expected return**:

    J(π) = Σ_t E_{π} [ r_t + α · H(π(·|s_t)) ]

where H(π(·|s)) = -Σ_a π(a|s) log π(a|s) is the policy entropy and
α > 0 is the temperature parameter that controls the exploration-exploitation
tradeoff. Higher α encourages more stochastic (exploratory) policies;
lower α approaches the standard (greedy) RL objective.

Unlike PPO, which is on-policy and discards data after each update, SAC is
**off-policy**: it stores all transitions in a replay buffer and reuses
each sample ~100× (vs PPO's ~4× reuse via mini-batch epochs). This makes
SAC significantly more sample-efficient.


Why SAC-Discrete for Market-Making?
====================================

Market-making in LOB environments presents challenges that SAC-Discrete is
well-suited to address:

    1. **Bimodal returns**: Market-making PnL is inherently bimodal — the
       agent either captures spread (small positive) or suffers adverse
       selection (large negative). SAC's twin Q-networks model this
       distribution better than PPO's scalar critic V(s).

    2. **Sample efficiency**: Each LOB simulation episode is expensive
       (5,000+ steps). Off-policy replay allows SAC to extract much more
       learning signal per simulation step than PPO.

    3. **Exploration via entropy**: SAC's automatic temperature tuning
       maintains healthy exploration without the need for ε-greedy (DQN)
       or entropy bonus scheduling (PPO). The agent naturally balances
       exploitation of learned Q-values with exploration of uncertain
       state-action pairs.

    4. **Stability**: The clipped double-Q trick prevents Q-value
       overestimation, a known failure mode in single-critic methods
       applied to stochastic environments like LOBs.


Algorithm Details
==================

**Twin Q-Networks (Clipped Double Q)**

    SAC maintains two independent Q-networks Q_1(s,a) and Q_2(s,a) with
    separate target copies Q̃_1 and Q̃_2. The minimum of the two target
    Q-values is used in the TD target to prevent overestimation:

        y = r + (1-d) · γ^K · V̄(s')

    where the soft value function V̄(s') uses the minimum Q-target:

        V̄(s') = Σ_a π(a|s') · [ min(Q̃_1(s',a), Q̃_2(s',a)) - α·log π(a|s') ]

    Both Q-networks are updated via MSE loss against the same target y:

        L_Q(φ_i) = E_{(s,a,r,s')} [ (Q_i(s,a) - y)² ]    for i ∈ {1, 2}


**Soft Policy Update**

    The actor is updated to minimise the KL divergence between the policy
    and the softmax of Q-values:

        L_π(θ) = E_s [ Σ_a π_θ(a|s) · ( α·log π_θ(a|s) - Q_min(s,a) ) ]

    where Q_min(s,a) = min(Q_1(s,a), Q_2(s,a)) with gradients detached.

    This can be interpreted as: the policy should assign high probability to
    actions with high Q-values (exploitation) while maintaining high entropy
    (exploration). The temperature α controls this balance.


**Automatic Temperature Tuning**

    The temperature α is learned by minimising:

        L_α = -log(α) · [ H(π(·|s)) - H̄ ]

    where H̄ is the target entropy, set to:

        H̄ = ratio × log(|A|)     (typically ratio ≈ 0.98)

    This creates a feedback loop:
        - If H(π) < H̄ (policy too deterministic) → α increases → more entropy
        - If H(π) > H̄ (policy too random) → α decreases → less entropy

    The target entropy is slightly below the maximum entropy log(|A|)
    (uniform distribution), allowing the policy to specialise while
    maintaining meaningful exploration.


**Soft Target Network Update (Polyak Averaging)**

    Target networks are updated smoothly after each gradient step:

        θ̃ ← (1-τ)·θ̃ + τ·θ      (τ = 0.005 by default)

    This is equivalent to an exponential moving average with half-life
    ≈ ln(2)/τ ≈ 139 gradient steps. The slow-moving target provides
    stable TD targets, preventing the oscillation/divergence that occurs
    when the target network changes too rapidly.


SMDP-Correct Discounting
==========================

When throttled (time-gate, event-gate, TOB-gate), each SMDP transition
spans K_t micro-steps. The controller correctly aggregates rewards and
adjusts the discount factor:

    R^{cum}_t = Σ_{k=0}^{K_t - 1} γ^k · r_{t,k}      (cumulative reward)
    γ_eff = γ^{K_t}                                     (effective discount)

The TD target becomes:

    y_t = R^{cum}_t + (1-d_t) · γ^{K_t} · V̄(s_{t+1})

This is mathematically equivalent to the standard Bellman equation under
the Semi-Markov Decision Process framework (Bradtke & Duff, 1994).


Replay Buffer Format
=====================

Each SMDP transition is stored as an 8-field tuple:

    [s, a, R^{cum}, done, s', γ_eff, mask, mask_next]

where:
    s         : state tensor at the decision point         (1, D)
    a         : action index                                (1, 1)
    R^{cum}   : cumulative SMDP reward                      (1, 1)
    done      : episode termination flag                     (1, 1)
    s'        : next state at the following decision point  (1, D)
    γ_eff     : effective discount γ^K                      (1, 1)
    mask      : boolean action mask at decision time        (1, |A|)
    mask_next : boolean action mask at next state s'        (1, |A|)

The action masks capture which actions were valid at each state.
``mask`` is used for correct actor updates; ``mask_next`` ensures the
critic's V(s') bootstrap does not assign probability to impossible
actions (e.g., buying when already at +inv_limit).


Architecture Comparison: SAC vs PPO vs DQN
============================================

                    SAC-Discrete        PPO              DQN (C51)
    ───────────────────────────────────────────────────────────────
    Learning        Off-policy          On-policy        Off-policy
    Data reuse      ~100× (replay)      ~4× (epochs)     ~100× (replay)
    Critic type     Twin Q(s,a)         Scalar V(s)      Q(s,a) distributional
    Policy          Explicit π(a|s)     Explicit π(a|s)  Implicit (ε-greedy)
    Exploration     Entropy (auto α)    Entropy bonus    NoisyNets / ε-greedy
    Update rule     Soft Bellman        Clipped PPO      TD / distributional
    Env coupling    Sequential episodes Parallel envs    Sequential episodes
    ───────────────────────────────────────────────────────────────


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


Inheritance from BaseMMController
==================================

``SACDiscreteController`` inherits from ``BaseMMController`` (defined in
``mm_base.py``) to reuse shared infrastructure:

    - ``_state_to_tensor()``       — Feature engineering (log1p, normalisation)
    - ``_get_valid_action_mask()`` — Inventory-limit action masking
    - ``_sample_action_idx()``     — Categorical sampling from actor logits
    - ``_mm_action_from_idx()``    — Action index → MarketMaker command
    - ``act()``                    — Full SMDP throttle + action selection
    - ``_smdp_reset()``            — Reset SMDP state variables
    - ``_reset_throttle_state()``  — Reset throttle gate counters
    - Throttle gate logic (time/event/TOB)
    - Pure MM configuration (offsets, canonical masks)

Only the following are overridden or added:
    - ``__init__()``               — Twin Q-nets, temperature, replay buffer
    - ``learn()``                  — SMDP → replay insert → gradient step
    - ``_smdp_commit_to_nstep()``  — N-step buffer transition commit
    - ``_train_step()``            — SAC-Discrete gradient step
    - ``enable_learning``          — Property setter for Q-network eval/train
    - ``make_worker()``            — Disabled (sequential training only)
    - ``get_and_reset_train_stats()`` — Training metric accumulator


References
==========
    [1] Christodoulou, "Soft Actor-Critic for Discrete Action Settings",
        arXiv:1910.07207, 2019.
    [2] Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
        Deep Reinforcement Learning with a Stochastic Actor",
        ICML 2018, arXiv:1801.01290.
    [3] Haarnoja et al., "Soft Actor-Critic Algorithms and Applications",
        arXiv:1812.05905, 2018.
    [4] Fujimoto et al., "Addressing Function Approximation Error in
        Actor-Critic Methods" (TD3), ICML 2018 — clipped double-Q trick.
    [5] Bradtke & Duff, "Reinforcement Learning Methods for Continuous-Time
        Markov Decision Processes", NeurIPS 1994 — SMDP framework.
    [6] Puterman, "Markov Decision Processes: Discrete Stochastic Dynamic
        Programming", Wiley 1994 — theoretical foundations.
"""

# ============================================================
# Imports
# ============================================================

from collections import deque
from typing import Dict, Any, Optional, List, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from mm_base import BaseMMController


# ============================================================
# Section 1: Q-Network — Action-Value Function Approximator
# ============================================================
#
# The Q-network maps (state, action) pairs to expected returns.
# In discrete-action SAC, we output Q(s,a) for ALL actions simultaneously
# (analogous to DQN), rather than taking the action as an input.
#
# Two independent Q-networks (Q_1, Q_2) are used to implement the
# "clipped double-Q" trick from TD3 (Fujimoto et al., 2018):
#
#     Q_min(s,a) = min(Q_1(s,a), Q_2(s,a))
#
# This prevents overestimation bias, which occurs because the max
# operator in the Bellman equation systematically overestimates
# Q-values when combined with noisy function approximation.
#
# Each Q-network has a corresponding target network (Q̃_1, Q̃_2)
# that is updated via Polyak averaging:
#
#     θ̃ ← (1-τ)·θ̃ + τ·θ      after each gradient step
#
# The target networks provide stable TD targets, preventing the
# "moving target" problem that destabilises Q-learning.
# ============================================================


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) action-value network for SAC-Discrete.

    Maps a state vector s ∈ ℝ^D to action values Q(s,a) ∈ ℝ^|A| for
    all actions simultaneously. This "all-actions-at-once" design is
    standard for discrete-action Q-learning and avoids the need for
    per-action forward passes.

    Used as twin critics Q_1, Q_2 in SAC-Discrete (Christodoulou, 2019).
    Each network also has a Polyak-averaged target copy (Q̃_1, Q̃_2).

    Architecture
    ------------
        Input(D) → [Linear(D, H) → ReLU]^L → Linear(H, |A|)

    where D = input_dim, H = n_neurons, L = n_hidden, |A| = n_actions.
    Total parameters: D·H + H + (L-1)·(H² + H) + H·|A| + |A|.

    The output layer has NO activation — Q-values can be any real number,
    including negative values (e.g., when inventory penalty dominates
    spread capture in market-making).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the state vector. For pure MM with max_offset=1:
        D = 2 + 2·(1+1) = 6.
    n_actions : int
        Number of discrete actions |A|. For the 6-action pure MM grid:
        [tighten, buy-lean, sell-lean, neutral, widen-ask, widen-bid].
    n_hidden : int
        Number of hidden layers (depth). Default: 2.
    n_neurons : int
        Neurons per hidden layer (width). Default: 128.

    Example
    -------
    >>> q_net = QNetwork(input_dim=6, n_actions=6, n_hidden=1, n_neurons=256)
    >>> s = torch.randn(32, 6)           # batch of 32 states
    >>> q_values = q_net(s)              # → (32, 6) Q-values
    >>> best_action = q_values.argmax(1) # greedy action per state
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        n_hidden: int = 2,
        n_neurons: int = 128,
        use_dueling: bool = False,
        use_distributional: bool = False,
        n_quantiles: int = 25,
    ):
        super().__init__()
        self.use_dueling = bool(use_dueling)
        self.use_distributional = bool(use_distributional)
        self.n_actions = n_actions
        self.n_quantiles = int(n_quantiles) if use_distributional else 0

        # Shared feature trunk: input → [Linear → ReLU]^L
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, n_neurons))
            layers.append(nn.ReLU())
            prev = n_neurons
        self.feature = nn.Sequential(*layers)

        # ── Output heads ──────────────────────────────────────
        #
        # Four combinations of (dueling × distributional):
        #
        #   Scalar standard:   fc_out  → (B, |A|)
        #   Scalar dueling:    fc_value → (B, 1),   fc_adv → (B, |A|)
        #   QR standard:       fc_out  → (B, |A|·N) → view (B, |A|, N)
        #   QR + dueling:      fc_value → (B, N),   fc_adv → (B, |A|·N)
        #                      → view + per-quantile dueling combination
        #
        # QR-DQN (Dabney et al., 2018): outputs N unconstrained quantile
        # values per action (NO softmax, unlike C51). Each quantile θ_i
        # approximates the return at quantile fraction τ_i.
        if self.use_distributional:
            N = self.n_quantiles
            if self.use_dueling:
                self.fc_value = nn.Linear(prev, N)              # V(s) per quantile
                self.fc_adv = nn.Linear(prev, n_actions * N)    # A(s,a) per quantile
            else:
                self.fc_out = nn.Linear(prev, n_actions * N)
        else:
            if self.use_dueling:
                # Dueling heads: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
                self.fc_value = nn.Linear(prev, 1)
                self.fc_adv = nn.Linear(prev, n_actions)
            else:
                # Standard head: Q(s,a) directly
                self.fc_out = nn.Linear(prev, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: s → Q(s, ·).

        Returns
        -------
        torch.Tensor
            Scalar mode:         (B, |A|)         — Q-values.
            Distributional mode: (B, |A|, N_tau)   — quantile values θ_i(s, a).
        """
        x = self.feature(x)

        if self.use_distributional:
            N = self.n_quantiles
            if self.use_dueling:
                # Per-quantile dueling: Q_i(s,a) = V_i(s) + A_i(s,a) - mean_a A_i(s,a)
                value = self.fc_value(x).view(-1, 1, N)                  # (B, 1, N)
                adv = self.fc_adv(x).view(-1, self.n_actions, N)         # (B, A, N)
                return value + adv - adv.mean(dim=1, keepdim=True)       # (B, A, N)
            return self.fc_out(x).view(-1, self.n_actions, N)            # (B, A, N)

        if self.use_dueling:
            value = self.fc_value(x)       # (B, 1)
            adv = self.fc_adv(x)           # (B, |A|)
            return value + adv - adv.mean(dim=1, keepdim=True)
        return self.fc_out(x)


# ============================================================
# Section 2: SACDiscreteController — Full Agent
# ============================================================
#
# This controller implements the complete SAC-Discrete algorithm
# with SMDP-correct discounting, inventory-limit action masking,
# and inline gradient updates (no separate training phase).
#
# The training loop is DQN-like:
#
#     for each episode:
#         for each simulation step:
#             action = ctrl.act(state)        # throttle + sample
#             ...simulator processes action...
#             ctrl.learn(state, reward, ...)   # SMDP → replay → gradient
#
# Gradient updates happen DURING the episode (after each SMDP
# decision commit), not in a separate batch phase like PPO.
#
# Data flow per gradient step:
#
#     ┌─────────────┐    sample batch    ┌──────────────────┐
#     │ ReplayMemory │ ─────────────────→ │  _train_step()   │
#     │  (100k cap)  │                    │                  │
#     │ [s,a,r,d,s', │                    │  1. Critic loss  │
#     │  γ_eff,mask] │                    │  2. Actor loss   │
#     └─────────────┘                    │  3. Alpha loss   │
#           ↑                             │  4. Target update│
#           │ insert                      └──────────────────┘
#     ┌─────────────┐
#     │   learn()   │
#     │  SMDP agg.  │
#     └─────────────┘
# ============================================================


class SACDiscreteController(BaseMMController):
    """
    SAC-Discrete controller for the MarketMaker — LOB environment.

    Off-policy maximum-entropy actor-critic for discrete action spaces,
    following Christodoulou (2019). Combines the data efficiency of
    off-policy learning (replay buffer, like DQN) with the benefits of
    an explicit stochastic policy (like PPO).

    Key Components
    ---------------
    - **Twin Q-networks** Q_1, Q_2: Prevent overestimation via clipped
      double-Q (Fujimoto et al., 2018). Updated via MSE loss against
      the soft Bellman target.

    - **Soft target networks** Q̃_1, Q̃_2: Polyak-averaged copies of the
      online Q-networks. Updated after each gradient step:
      θ̃ ← (1-τ)·θ̃ + τ·θ.

    - **Actor network** π_θ(a|s): Categorical policy producing softmax
      probabilities over discrete actions. Same architecture as
      BaseMMController's actor (inherited).

    - **Learnable temperature** α = exp(log_alpha): Auto-tuned to
      maintain target entropy H̄ ≈ 0.98·log(|A|). No manual entropy
      bonus scheduling needed.

    - **Replay buffer**: Uniform experience replay (imported from
      ``dqn_distributional_with_throttle.py``). Stores SMDP transitions
      with 8 fields: [s, a, R^{cum}, done, s', γ_eff, mask, mask_next].

    Gradient updates occur inside ``learn()`` after each SMDP transition
    commit (DQN-like inline training), NOT in a separate rollout phase.

    Inherits From
    ---------------
    ``BaseMMController`` (from ``mm_base.py``): Provides shared
    infrastructure including state encoding, action masking, throttle
    gates, SMDP state variables, and action mapping. SAC overrides only
    the learning-related methods.

    Parameters
    ----------
    level_offset : int
        Price offset for generic mode quotes.
    n_actions : int
        Number of discrete actions |A|.
    gamma : float
        SMDP discount factor γ ∈ (0, 1).
    lr_actor : float
        Learning rate for the actor (policy) network.
    lr_critic : float
        Learning rate for both Q-networks (shared optimizer).
    weight_decay : float
        L2 regularisation coefficient for AdamW.
    grad_clip_norm : float
        Maximum global gradient norm for all optimisers.
    device : torch.device
        Computation device (CPU/CUDA).
    log_dir : str or None
        TensorBoard log directory. If not None, prints setup summary.
    pure_mm : bool
        If True, use PURE MM mode with offset-pair action space.
    inv_limit : int or None
        Maximum absolute inventory. Actions that would breach this
        limit are masked out.
    pure_mm_offsets : list of (int, int)
        Offset pairs (bid_off, ask_off) defining the action space.
    n_hidden_actor, n_neurons_actor : int
        Actor MLP architecture.
    n_hidden_critic, n_neurons_critic : int
        Q-network MLP architecture.
    enable_learning : bool
        If False, networks are in eval mode and act() uses argmax.
    use_tob_update, n_tob_moves : bool, int
        TOB-update throttle gate configuration.
    use_event_update, n_events : bool, int
        Event-update throttle gate configuration.
    use_time_update, min_time_interval : bool, float
        Time-update throttle gate configuration.
    use_mdp : bool
        If True, disable fill/mode-change bypasses.
    lr_alpha : float
        Learning rate for the temperature parameter α.
    tau : float
        Polyak averaging coefficient for target networks.
        Typical values: 0.005 (slow) to 0.05 (fast).
    alpha_init : float
        Initial value of the temperature parameter α.
    target_entropy_ratio : float
        Target entropy as a fraction of maximum entropy log(|A|).
        H̄ = ratio × log(|A|). Typical: 0.4–0.7 for small action
        spaces (6 actions); higher values keep the policy near-uniform
        and delay specialisation.
    replay_capacity : int
        Maximum number of transitions in the replay buffer.
    batch_size : int
        Mini-batch size for gradient updates.

    Example
    -------
    >>> ctrl = SACDiscreteController(
    ...     pure_mm=True,
    ...     pure_mm_offsets=[(-1,-1),(-1,0),(0,-1),(0,0),(0,1),(1,0)],
    ...     n_hidden_actor=1, n_neurons_actor=256,
    ...     n_hidden_critic=1, n_neurons_critic=256,
    ...     inv_limit=8, gamma=0.97,
    ...     lr_actor=1e-3, lr_critic=1e-3, lr_alpha=3e-4,
    ...     tau=0.005, alpha_init=0.2, target_entropy_ratio=0.60,
    ...     batch_size=256, replay_capacity=100_000,
    ... )
    >>> # Used with simulate_LOB_with_MM:
    >>> # action = ctrl.act(state)     # throttle + categorical sample
    >>> # ctrl.learn(...)              # SMDP → replay → gradient step
    """

    def __init__(
        self,
        # --- BaseMMController args (forwarded to super) ---
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.97,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = "runs/sac_mm",
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        n_hidden_actor: int = 2,
        n_neurons_actor: int = 128,
        n_hidden_critic: int = 2,
        n_neurons_critic: int = 128,
        enable_learning: bool = True,
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_event_update: bool = False,
        n_events: int = 100,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,
        use_mdp: bool = True,
        # --- Observation normalisation ---
        use_obs_normalizer: bool = True,
        # --- SAC-Discrete specific ---
        lr_alpha: float = 3e-4,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        target_entropy_ratio: float = 0.60,
        replay_capacity: int = 100_000,
        batch_size: int = 256,
        # --- Stability ---
        alpha_min: float = 0.01,
        # --- Dueling architecture ---
        use_dueling: bool = False,
        # --- Distributional QR-DQN (Dabney et al., 2018) ---
        use_distributional: bool = False,
        n_quantiles: int = 25,
        quantile_huber_kappa: float = 1.0,
        # --- Prioritized Experience Replay (PER) ---
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-4,
        # --- Update-to-Data ratio (gradient steps per decision) ---
        utd_ratio: int = 1,
        # --- N-step returns (credit propagation across SMDP decisions) ---
        n_steps: int = 1,
    ):
        # ─────────────────────────────────────────────────────────
        # Initialise BaseMMController (actor_net, obs_normalizer,
        # SMDP/throttle infrastructure, action masking).
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

        # ─────────────────────────────────────────────────────────
        # Twin Q-networks: Q_1(s,a) and Q_2(s,a)
        #
        # Two independent networks trained on the SAME TD targets
        # but with different random initialisations. Using
        # min(Q_1, Q_2) in the Bellman target prevents the
        # overestimation bias that occurs with a single Q-network.
        # ─────────────────────────────────────────────────────────
        self.use_dueling = bool(use_dueling)
        self.use_distributional = bool(use_distributional)
        self.n_quantiles = int(n_quantiles) if use_distributional else 0
        self.quantile_huber_kappa = float(quantile_huber_kappa)

        # ─────────────────────────────────────────────────────────
        # QR-DQN quantile midpoints τ (Dabney et al., 2018, Eq. 9)
        #
        # τ_i = (2i + 1) / (2N) for i = 0, ..., N-1.
        # These are the midpoints of N equal-width bins in [0, 1].
        # Each quantile value θ_i approximates F^{-1}(τ_i), the
        # inverse CDF of the return distribution at τ_i.
        # ─────────────────────────────────────────────────────────
        if self.use_distributional:
            _tau = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
            self.quantile_tau = _tau.to(self.device)  # (N,) — quantile midpoints

        _q_kwargs = dict(
            use_dueling=use_dueling,
            use_distributional=use_distributional,
            n_quantiles=n_quantiles,
        )
        self.q1_net = QNetwork(
            self.input_dim, self.n_actions, n_hidden_critic, n_neurons_critic,
            **_q_kwargs,
        ).to(self.device)
        self.q2_net = QNetwork(
            self.input_dim, self.n_actions, n_hidden_critic, n_neurons_critic,
            **_q_kwargs,
        ).to(self.device)

        # ─────────────────────────────────────────────────────────
        # Target networks (Polyak-averaged copies)
        #
        # Initialised as exact copies of the online networks.
        # Gradients are disabled — they are updated only via
        # Polyak averaging in _train_step().
        # ─────────────────────────────────────────────────────────
        self.q1_target = QNetwork(
            self.input_dim, self.n_actions, n_hidden_critic, n_neurons_critic,
            **_q_kwargs,
        ).to(self.device)
        self.q2_target = QNetwork(
            self.input_dim, self.n_actions, n_hidden_critic, n_neurons_critic,
            **_q_kwargs,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        # ─────────────────────────────────────────────────────────
        # Q-network optimiser (shared for both Q-nets)
        #
        # A single optimiser over the concatenated parameter lists
        # of Q_1 and Q_2. This is equivalent to two separate
        # optimisers but more memory-efficient.
        # ─────────────────────────────────────────────────────────
        self.q_optimizer = AdamW(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
            lr=lr_critic,
            weight_decay=weight_decay,
        )

        # ─────────────────────────────────────────────────────────
        # Learnable temperature α = exp(log_alpha)
        #
        # Parameterised in log-space to ensure α > 0 without
        # constrained optimisation. The gradient of L_α flows
        # through exp(log_alpha) via the chain rule.
        # ─────────────────────────────────────────────────────────
        self.log_alpha = torch.tensor(
            [math.log(alpha_init)], dtype=torch.float32,
            device=self.device, requires_grad=True,
        )
        self.alpha_optimizer = AdamW([self.log_alpha], lr=lr_alpha)
        self.alpha_min = alpha_min  # floor to prevent α → 0 collapse
        self.utd_ratio = max(1, int(utd_ratio))

        # ─────────────────────────────────────────────────────────
        # Target entropy: H̄ = ratio × log(|A|)
        #
        # log(|A|) is the maximum entropy (uniform distribution).
        # A ratio of 0.60 means the target is 60% of maximum,
        # allowing meaningful policy specialisation while still
        # maintaining exploration. For 6 actions, this gives
        # H̄ ≈ 1.07 (vs max 1.79).
        # ─────────────────────────────────────────────────────────
        self.target_entropy = float(
            -math.log(1.0 / self.n_actions) * target_entropy_ratio
        )

        # ─────────────────────────────────────────────────────────
        # Replay buffer
        #
        # If use_per=True, use PrioritizedReplayMemory (same
        # implementation as DQN Rainbow). Otherwise uniform.
        # Training starts only after 10 × batch_size transitions
        # are stored (ensures enough diversity).
        # ─────────────────────────────────────────────────────────
        self.use_per = bool(use_per)
        if self.use_per:
            from dqn_distributional_with_throttle import PrioritizedReplayMemory
            self.memory = PrioritizedReplayMemory(
                capacity=replay_capacity,
                alpha=per_alpha,
                beta=per_beta,
                eps=per_eps,
            )
        else:
            from dqn_distributional_with_throttle import ReplayMemory
            self.memory = ReplayMemory(capacity=replay_capacity)

        # ─────────────────────────────────────────────────────────
        # SAC hyperparameters
        # ─────────────────────────────────────────────────────────
        self.tau = float(tau)
        self.sac_batch_size = int(batch_size)

        # N-step returns: chain n_steps SMDP transitions before inserting
        # into replay. Higher n → faster credit propagation (fill reward
        # reaches the placement action sooner) but higher variance.
        self.n_steps = max(1, int(n_steps))
        self.nstep_buffer: deque = deque()

        # Training step counter (total gradient steps across all episodes)
        self.train_steps: int = 0

        # ─────────────────────────────────────────────────────────
        # Accumulator for TensorBoard logging
        #
        # Stats are accumulated during each episode and retrieved
        # via get_and_reset_train_stats() at episode boundaries.
        # ─────────────────────────────────────────────────────────
        self._train_stats: Dict[str, float] = {
            "q1_loss": 0.0, "q2_loss": 0.0,
            "actor_loss": 0.0,
            "alpha": 0.0, "alpha_loss": 0.0,
            "entropy": 0.0, "mean_q": 0.0,
            "n_updates": 0,
        }

        # Print setup summary
        if log_dir is not None:
            print("\n=========== SAC-DISCRETE CONTROLLER SETUP ===========")
            print(f"pure_mm            : {self.pure_mm}")
            print(f"inv_limit          : {self.inv_limit}")
            if self.pure_mm:
                print(f"pure_mm_offsets    : {self.pure_mm_offsets}")
                print(f"max_offset         : {self.max_offset}")
                print(f"input_dim          : {self.input_dim}")
            print(f"gamma              : {self.gamma}")
            print(f"n_actions          : {self.n_actions}")
            print(f"lr_actor           : {lr_actor}")
            print(f"lr_critic (Q)      : {lr_critic}")
            print(f"lr_alpha           : {lr_alpha}")
            print(f"tau                : {self.tau}")
            print(f"alpha_init         : {alpha_init}")
            print(f"target_entropy     : {self.target_entropy:.4f}")
            print(f"replay_capacity    : {replay_capacity}")
            print(f"batch_size         : {self.sac_batch_size}")
            print(f"n_steps (TD)       : {self.n_steps}")
            print(f"use_dueling        : {self.use_dueling}")
            print(f"use_distributional : {self.use_distributional}")
            if self.use_distributional:
                print(f"n_quantiles        : {self.n_quantiles}")
                print(f"huber_kappa        : {self.quantile_huber_kappa}")
            print(f"device             : {self.device}")
            print("======================================================\n")

    # ================================================================
    # Q-VALUE EXTRACTION — Distributional-aware helper
    # ================================================================

    def _q_expected(
        self, net: nn.Module, states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract scalar expected Q-values from a Q-network, transparently
        handling both scalar and distributional (QR-DQN) architectures.

        For scalar networks:
            Returns Q(s, a) directly — shape (B, |A|).

        For distributional (QR-DQN) networks:
            The network outputs N quantile values θ_i(s, a) that approximate
            the return distribution. The expected Q-value is the mean of the
            quantiles (Dabney et al., 2018):

                Q(s, a) = (1/N) Σ_{i=1}^{N} θ_i(s, a)

            This is exact when the quantile values correspond to the midpoints
            of N equal-width bins in [0, 1], which is our parameterisation.

        Parameters
        ----------
        net : nn.Module
            A QNetwork instance (scalar or distributional).
        states : torch.Tensor
            Batch of state vectors, shape (B, D).

        Returns
        -------
        torch.Tensor
            Expected Q-values for all actions, shape (B, |A|).
        """
        out = net(states)
        if self.use_distributional:
            return out.mean(dim=-1)   # (B, A, N) → (B, A)
        return out                     # (B, A)

    # ================================================================
    # ENABLE LEARNING TOGGLE — Override for SAC networks
    # ================================================================

    @BaseMMController.enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        """
        Toggle between training and evaluation mode.

        When enabled (True):  actor + Q-networks in train() mode (dropout,
                               batchnorm active if present).
        When disabled (False): actor + Q-networks in eval() mode, and
                               act() uses argmax (greedy) instead of sampling.
        """
        self._enable_learning = bool(value)
        if self._enable_learning:
            self.actor_net.train()
            self.q1_net.train()
            self.q2_net.train()
        else:
            self.actor_net.eval()
            self.q1_net.eval()
            self.q2_net.eval()

    # ================================================================
    # SMDP COMMIT → N-STEP BUFFER
    # ================================================================

    def _smdp_commit_to_nstep(
        self,
        s_next_dict: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Commit the pending SMDP transition into the n-step buffer.

        Instead of inserting directly into replay, we push an 8-tuple
        into ``self.nstep_buffer``.  The downstream method
        ``_commit_nstep_transition()`` later aggregates up to
        ``n_steps`` SMDP transitions into one n-step return and
        inserts the result into replay.

        Buffer element format (8-tuple):
            (s_t, a_t, r_t, done_t, s_next_t, k_steps, mask_t, mask_next_t)

        ``k_steps`` is a plain int (number of micro-steps this SMDP
        transition spans).  It is used to compute the effective
        discount γ^{Σ k_i} when the n-step return is assembled.
        """
        if not self._smdp_pending:
            return

        # Convert state dicts → tensors, keeping (1, D) batch dim
        s_t = self._state_to_tensor(self._smdp_s_decision).detach().cpu()
        s_next_t = self._state_to_tensor(s_next_dict).detach().cpu()

        a_t = torch.tensor(
            [[self._smdp_a_decision]], dtype=torch.long,
        )
        r_t = torch.tensor(
            [[self._smdp_cum_reward]], dtype=torch.float32,
        )
        done_t = torch.tensor(
            [[done]], dtype=torch.bool,
        )

        k = self._smdp_k_steps  # plain int

        # Action mask at decision time
        if self._smdp_mask_decision is not None:
            mask_t = torch.tensor(
                self._smdp_mask_decision, dtype=torch.bool,
            ).unsqueeze(0)
        else:
            mask_t = torch.ones(1, self.n_actions, dtype=torch.bool)

        # Action mask at next state
        mask_next = self._get_valid_action_mask(s_next_dict)
        mask_next_t = torch.tensor(mask_next, dtype=torch.bool).unsqueeze(0)

        self.nstep_buffer.append(
            (s_t, a_t, r_t, done_t, s_next_t, k, mask_t, mask_next_t)
        )

        # Clear SMDP state for the next transition
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_mask_decision = None
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0

    # ================================================================
    # N-STEP BUFFER → REPLAY MEMORY
    # ================================================================

    def _commit_nstep_transition(self) -> None:
        """
        Build one n-step transition from the oldest elements in
        ``self.nstep_buffer`` and insert it into replay memory.

        For a chunk of L SMDP transitions (L ≤ n_steps):

            G = Σ_{i=0}^{L-1}  (Π_{j<i} γ^{k_j}) · R_i
            γ_eff = γ^{Σ k_i}

        The replay transition keeps the same 8-field format that
        ``_train_step()`` already consumes:

            [s_root, a_root, G, done_final, s_next_final,
             γ_eff, mask_root, mask_next_final]
        """
        chunk_len = min(self.n_steps, len(self.nstep_buffer))
        chunk = list(self.nstep_buffer)[:chunk_len]

        # Root state, action, mask from the FIRST SMDP transition
        s_root, a_root, _, _, _, _, mask_root, _ = chunk[0]

        # Compute n-step return G with variable SMDP discounting
        G = 0.0
        discount = 1.0
        total_micro_steps = 0

        for (_, _, r_i, _, _, k_i, _, _) in chunk:
            G += discount * float(r_i.item())
            discount *= (self.gamma ** k_i)
            total_micro_steps += k_i

        # Effective discount for bootstrapping Q(s_next_final)
        gamma_eff = self.gamma ** total_micro_steps

        # Terminal state, done flag, mask from the LAST element
        _, _, _, done_last, s_next_last, _, _, mask_next_last = chunk[-1]
        done_final = bool(done_last.item())

        # Pack into tensors (same shapes as before)
        G_tensor = torch.tensor([[G]], dtype=torch.float32)
        done_tensor = torch.tensor([[done_final]], dtype=torch.bool)
        gamma_tensor = torch.tensor([[gamma_eff]], dtype=torch.float32)

        self.memory.insert([
            s_root, a_root, G_tensor, done_tensor,
            s_next_last, gamma_tensor, mask_root, mask_next_last,
        ])

        # Slide the window forward
        self.nstep_buffer.popleft()

    # ================================================================
    # learn() — SMDP aggregation + replay insert + gradient step
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
        SMDP-aware learning step for SAC-Discrete.

        This method is called at EVERY simulation step by the LOB engine.
        It implements the same SMDP aggregation logic as BaseMMController but
        commits transitions to the replay buffer and triggers inline
        gradient updates.

        SMDP Aggregation Cases
        -----------------------

        **Case A — New Decision Arrived** (``self._last_was_decision = True``):

            If a previous SMDP transition is pending, commit it to replay
            with ``s_next = state_before`` (the state that triggered this
            new decision). Then start a new SMDP transition:

                s_decision ← state_before
                a_decision ← last_action_idx
                R^{cum} ← reward
                K ← 1

        **Case B — Throttled Micro-Step** (``self._last_was_decision = False``):

            Accumulate the discounted reward into the pending transition:

                R^{cum} += γ^K · reward
                K += 1

        **Special — Episode End** (``info["done"] = True``):

            If a transition is pending, commit it with ``done=True`` and
            ``s_next = state_after``. Reset all SMDP and throttle state.

        **Gradient Trigger**:

            After each decision or episode end, if the replay buffer has
            enough samples (``memory.can_sample(batch_size)``), perform
            one SAC-Discrete gradient step via ``_train_step()``.

        Parameters
        ----------
        step_idx : int
            Current simulation step index.
        mm : MarketMaker
            MarketMaker instance.
        lob : LOB_simulation
            Limit order book engine.
        state_before : dict
            State before the environment event.
        state_after : dict
            State after the environment event.
        reward : float
            Scalar reward from the reward function.
        info : dict
            Metadata including ``done`` flag.
        """
        if self.last_action_idx is None:
            return

        done_flag = bool(info.get("done", False))

        if not self._enable_learning:
            if done_flag:
                self.last_action_idx = None
                self.nstep_buffer.clear()
                self._smdp_reset()
                self._reset_throttle_state()
            return

        # CASE A: New decision arrived — commit old + start new
        if self._last_was_decision:
            if self._smdp_pending:
                self._smdp_commit_to_nstep(state_before, done=False)

            self._smdp_pending = True
            self._smdp_s_decision = state_before
            self._smdp_a_decision = self.last_action_idx
            self._smdp_mask_decision = self._last_valid_mask
            self._smdp_cum_reward = float(reward)
            self._smdp_k_steps = 1
        else:
            # CASE B: Throttled micro-step — accumulate reward
            if self._smdp_pending:
                k = self._smdp_k_steps
                self._smdp_cum_reward += (self.gamma ** k) * float(reward)
                self._smdp_k_steps += 1

        # SPECIAL CASE: Episode termination
        if done_flag:
            if self._smdp_pending:
                self._smdp_commit_to_nstep(state_after, done=True)

            # Flush ALL remaining n-step transitions into replay.
            # Tail transitions have shorter horizons (< n_steps) and
            # always end with done=True, so no bootstrapping occurs.
            while len(self.nstep_buffer) > 0:
                self._commit_nstep_transition()

            self.last_action_idx = None
            self._smdp_reset()
            self._reset_throttle_state()
        else:
            # Normal step: commit one n-step transition when buffer is full.
            if len(self.nstep_buffer) >= self.n_steps:
                self._commit_nstep_transition()

        # Gradient steps: only on decision or done, and if buffer ready.
        # UTD ratio > 1 runs multiple gradient steps per transition,
        # giving the critic more learning per unit of data.
        if (self._last_was_decision or done_flag) and \
                self.memory.can_sample(self.sac_batch_size):
            for _ in range(self.utd_ratio):
                self._train_step()

    # ================================================================
    # _train_step() — One SAC-Discrete gradient step
    # ================================================================

    def _train_step(self) -> None:
        """
        Perform one SAC-Discrete gradient step.

        This is the core learning algorithm, executed after each SMDP
        decision or episode termination (when the replay buffer is ready).

        Steps
        -----

        **1. Sample mini-batch** from replay buffer:

            (s, a, R^{cum}, done, s', γ_eff, mask, mask_next) ~ Uniform(Buffer)


        **2. Critic update** (twin Q-networks):

            Compute the soft state-value of the next state using target
            networks and the CURRENT policy (not the collection policy).
            Invalid actions in s' are masked BEFORE softmax to avoid
            bootstrapping from impossible actions:

                V̄(s') = Σ_a π(a|s') · [min(Q̃_1(s',a), Q̃_2(s',a)) - α·log π(a|s')]

            TD target (SMDP-correct):

                y = R^{cum} + (1-done) · γ_eff · V̄(s')

            Loss for each Q-network:

                L_Q = (1/B) · Σ (Q_i(s,a) - y)²

            Both Q-networks are updated simultaneously via a shared
            optimiser. Gradients are clipped to ``grad_clip_norm``.


        **3. Actor update** (policy network):

            The actor minimises the expected soft-Q deficit:

                L_π = (1/B) · Σ_s [ Σ_a π(a|s) · (α·log π(a|s) - Q_min(s,a)) ]

            where Q_min = min(Q_1, Q_2) with gradients DETACHED (the
            actor should not affect Q-network parameters).

            Invalid actions are masked BEFORE softmax (logits → -∞),
            ensuring zero probability for forbidden actions.


        **4. Temperature update** (auto-tune α):

            The temperature is adjusted to maintain the target entropy:

                L_α = -(1/B) · Σ_s [ log(α) · (H(π(·|s)) - H̄) ]

            where H(π) = -Σ_a π(a|s)·log π(a|s) is the policy entropy.

            Intuition:
                - H(π) < H̄ → loss is positive → α increases → more entropy
                - H(π) > H̄ → loss is negative → α decreases → less entropy


        **5. Soft target update** (Polyak averaging):

            θ̃_i ← (1-τ)·θ̃_i + τ·θ_i    for i ∈ {1, 2}
        """
        # ---- Sample from replay buffer ----
        batch = self.memory.sample(self.sac_batch_size)
        if self.use_per:
            # PER returns extra fields: buffer indices + IS weights
            states_b, actions_b, rewards_b, dones_b, next_states_b, \
                gamma_effs_b, masks_b, next_masks_b, per_idxs_b, per_weights_b = batch
            per_weights_b = per_weights_b.to(self.device)
        else:
            states_b, actions_b, rewards_b, dones_b, next_states_b, \
                gamma_effs_b, masks_b, next_masks_b = batch

        states_b = states_b.to(self.device)
        actions_b = actions_b.to(self.device)
        rewards_b = rewards_b.to(self.device)
        dones_b = dones_b.to(self.device).float()
        next_states_b = next_states_b.to(self.device)
        gamma_effs_b = gamma_effs_b.to(self.device)
        masks_b = masks_b.to(self.device)
        next_masks_b = next_masks_b.to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ──── 1. CRITIC UPDATE ────────────────────────────────
        #
        # Shared policy computation at s' (used by both scalar and
        # distributional branches):
        #   π(a|s'), log π(a|s') with invalid actions masked to -∞.
        #
        with torch.no_grad():
            next_logits = self.actor_net(next_states_b)
            next_logits = next_logits.masked_fill(~next_masks_b, float("-inf"))
            next_probs = F.softmax(next_logits, dim=-1)             # (B, A)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_log_probs = next_log_probs.masked_fill(~next_masks_b, 0.0)

        if self.use_distributional:
            # ── QR-DQN DISTRIBUTIONAL CRITIC UPDATE ────────────
            #
            # Each Q-network outputs N quantile values per action:
            #   Q_i(s,a) = {θ_1(s,a), ..., θ_N(s,a)}
            #
            # Target quantiles (Dabney et al., 2018 + SAC entropy):
            #   For each quantile index n, the soft distributional
            #   Bellman target is:
            #
            #     Z_n^{target} = r + (1-d) · γ_eff · [
            #         Σ_a π(a|s') · θ_n^{min}(s',a)     ← distributional V
            #       + Σ_a π(a|s') · (-α · log π(a|s'))   ← entropy bonus (scalar)
            #     ]
            #
            #   The entropy bonus is a constant shift applied uniformly
            #   to all quantiles (adding a constant to a random variable
            #   shifts all quantiles by that constant).
            #
            N = self.n_quantiles
            kappa = self.quantile_huber_kappa

            with torch.no_grad():
                # Target network quantiles at s': (B, A, N)
                q1_next_q = self.q1_target(next_states_b)
                q2_next_q = self.q2_target(next_states_b)
                # Select the ENTIRE distribution from the more pessimistic
                # network (lower expected Q). Element-wise min would create
                # an invalid quantile function; this preserves distributional
                # shape (TQC, Kuznetsov et al., 2020).
                q1_mean = q1_next_q.mean(dim=-1, keepdim=True)  # (B, A, 1)
                q2_mean = q2_next_q.mean(dim=-1, keepdim=True)  # (B, A, 1)
                q_next_min_q = torch.where(q1_mean <= q2_mean, q1_next_q, q2_next_q)  # (B, A, N)

                # Distributional soft value: V_n(s') = Σ_a π(a|s') · θ_n(s',a)
                #
                # NOTE: This computes the per-quantile expectation over actions.
                # Strictly, quantile(mixture) ≠ mixture(quantiles), but this is
                # the standard approximation for discrete-action distributional
                # RL (IQN-SAC, QR-SAC). The exact mixture projection is O(A·N·log(A·N))
                # and offers negligible improvement for small |A| (6 actions).
                #
                # next_probs: (B, A) → unsqueeze → (B, A, 1), broadcast with (B, A, N)
                v_next_q = (next_probs.unsqueeze(-1) * q_next_min_q).sum(dim=1)  # (B, N)

                # Entropy bonus: Σ_a π(a|s') · (-α · log π(a|s'))  — scalar per sample
                entropy_bonus = (next_probs * (-alpha * next_log_probs)).sum(
                    dim=1, keepdim=True,
                )  # (B, 1) — broadcasts onto N quantiles

                # SMDP-correct distributional Bellman target
                target_q = rewards_b + (1.0 - dones_b) * gamma_effs_b * (
                    v_next_q + entropy_bonus
                )  # (B, N)

            # ── Predicted quantiles for the taken action ──────
            #
            # q_net(s): (B, A, N)  →  gather along action dim  →  (B, 1, N)  →  (B, N)
            #
            _act_expand = actions_b.unsqueeze(-1).expand(-1, -1, N)   # (B, 1, N)
            q1_pred_q = self.q1_net(states_b).gather(1, _act_expand).squeeze(1)  # (B, N)
            q2_pred_q = self.q2_net(states_b).gather(1, _act_expand).squeeze(1)  # (B, N)

            # ── Quantile Huber loss (Dabney et al., 2018, Eq. 10) ──
            #
            # For each pair of predicted quantile i and target quantile j:
            #
            #   δ_{ij} = target_j - pred_i
            #   ρ^κ_τ(δ) = |τ_i - 𝟙(δ < 0)| · L_κ(δ) / κ
            #
            # where L_κ is the Huber loss with threshold κ:
            #   L_κ(δ) = 0.5·δ²       if |δ| ≤ κ
            #          = κ·(|δ| - κ/2)  otherwise
            #
            # The asymmetric weight |τ - 𝟙(δ<0)| penalises
            # under-prediction more for high quantiles (τ close to 1)
            # and over-prediction more for low quantiles (τ close to 0).
            #
            # Total loss: mean over N_tgt, mean over N_pred → per-sample
            #             loss (B,), then IS-weighted mean over batch.
            #
            def _quantile_huber_loss(pred_q, target_q_detached, weights=None):
                # pred_q: (B, N), target_q_detached: (B, N)
                td = target_q_detached.unsqueeze(1) - pred_q.unsqueeze(2)  # (B, N_pred, N_tgt)
                huber = torch.where(
                    td.abs() <= kappa,
                    0.5 * td.pow(2),
                    kappa * (td.abs() - 0.5 * kappa),
                )
                # τ: (N,) → (1, N, 1) to broadcast with (B, N_pred, N_tgt)
                tau_w = (self.quantile_tau.view(1, -1, 1) - (td.detach() < 0).float()).abs()
                # mean over N_tgt (1/N normalisation), then mean over N_pred → (B,)
                per_sample = (tau_w * huber / kappa).mean(dim=-1).mean(dim=-1)  # (B,)
                if weights is not None:
                    # PER returns weights as (B,1); squeeze to (B,) to avoid
                    # (B,1)*(B,) broadcasting explosion → (B,B).
                    loss = (weights.squeeze(-1) * per_sample).mean()
                else:
                    loss = per_sample.mean()
                return loss, td

            w = per_weights_b if self.use_per else None
            q1_loss, td_q1 = _quantile_huber_loss(q1_pred_q, target_q, w)
            q2_loss, td_q2 = _quantile_huber_loss(q2_pred_q, target_q, w)
            q_loss = q1_loss + q2_loss

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
                max_norm=self.grad_clip_norm,
            )
            self.q_optimizer.step()

            # PER priorities: mean absolute pairwise TD error
            if self.use_per:
                with torch.no_grad():
                    prio_q1 = td_q1.abs().mean(dim=(1, 2))           # (B,)
                    prio_q2 = td_q2.abs().mean(dim=(1, 2))
                    new_prios = torch.max(prio_q1, prio_q2).cpu().numpy()
                self.memory.update_priorities(per_idxs_b.numpy(), new_prios)

            # For logging: expected Q of the taken action
            q1_pred_scalar = q1_pred_q.mean(dim=-1, keepdim=True)    # (B, 1)

        else:
            # ── SCALAR CRITIC UPDATE (original SAC-Discrete) ──
            #
            # TD target: y = R^{cum} + (1-done) · γ_eff · V̄(s')
            # V̄(s') = Σ_a π(a|s') [min(Q̃_1, Q̃_2)(s',a) - α·log π(a|s')]
            #
            with torch.no_grad():
                q1_next = self.q1_target(next_states_b)              # (B, A)
                q2_next = self.q2_target(next_states_b)              # (B, A)
                q_next_min = torch.min(q1_next, q2_next)

                v_next = (next_probs * (q_next_min - alpha * next_log_probs)).sum(
                    dim=1, keepdim=True,
                )
                q_target = rewards_b + (1.0 - dones_b) * gamma_effs_b * v_next

            q1_pred = self.q1_net(states_b).gather(1, actions_b)     # (B, 1)
            q2_pred = self.q2_net(states_b).gather(1, actions_b)

            q1_td = q1_pred - q_target
            q2_td = q2_pred - q_target

            if self.use_per:
                q1_loss = (per_weights_b * q1_td.pow(2)).mean()
                q2_loss = (per_weights_b * q2_td.pow(2)).mean()
            else:
                q1_loss = q1_td.pow(2).mean()
                q2_loss = q2_td.pow(2).mean()
            q_loss = q1_loss + q2_loss

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
                max_norm=self.grad_clip_norm,
            )
            self.q_optimizer.step()

            if self.use_per:
                with torch.no_grad():
                    new_prios = torch.max(
                        q1_td.abs(), q2_td.abs(),
                    ).squeeze(1).cpu().numpy()
                self.memory.update_priorities(per_idxs_b.numpy(), new_prios)

            q1_pred_scalar = q1_pred                                 # (B, 1)

        # ──── 2. ACTOR UPDATE ─────────────────────────────────
        #
        # L_π = E_s[ Σ_a π(a|s) · (α·log π(a|s) - E[Q_min](s,a)) ]
        #
        # For distributional critics, the actor sees the EXPECTED
        # Q-value (mean over quantiles). The distributional
        # information benefits the critic's representation quality;
        # the actor only needs the first moment.
        #
        logits = self.actor_net(states_b)
        logits = logits.masked_fill(~masks_b, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.masked_fill(~masks_b, 0.0)

        with torch.no_grad():
            q_min = torch.min(
                self._q_expected(self.q1_net, states_b),
                self._q_expected(self.q2_net, states_b),
            )  # (B, A)

        actor_loss = (probs * (alpha * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_net.parameters(), max_norm=self.grad_clip_norm,
        )
        self.actor_optimizer.step()

        # ──── 3. TEMPERATURE (α) UPDATE ───────────────────────
        #
        # L_α = log(α) · (H(π) - H̄)
        # H(π) = -Σ_a π(a|s) · log π(a|s)
        #
        # Gradient: ∂L/∂log_α = (H - H̄)
        #   H < H̄ → negative gradient → α increases → more exploration ✓
        #   H > H̄ → positive gradient → α decreases → less exploration ✓
        #
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp log_alpha so α ≥ alpha_min (prevents irreversible collapse)
        with torch.no_grad():
            min_log_alpha = math.log(self.alpha_min)
            self.log_alpha.clamp_(min=min_log_alpha)

        # ──── 4. SOFT TARGET UPDATE ───────────────────────────
        #
        # θ̃ ← (1-τ)·θ̃ + τ·θ     (Polyak averaging)
        #
        with torch.no_grad():
            for p_t, p_o in zip(
                self.q1_target.parameters(), self.q1_net.parameters()
            ):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)
            for p_t, p_o in zip(
                self.q2_target.parameters(), self.q2_net.parameters()
            ):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)

        self.train_steps += 1

        # ──── Accumulate stats for TensorBoard ────────────────
        self._train_stats["q1_loss"] += q1_loss.item()
        self._train_stats["q2_loss"] += q2_loss.item()
        self._train_stats["actor_loss"] += actor_loss.item()
        self._train_stats["alpha"] += alpha.item()
        self._train_stats["alpha_loss"] += alpha_loss.item()
        self._train_stats["entropy"] += entropy.mean().item()
        self._train_stats["mean_q"] += q1_pred_scalar.mean().item()
        self._train_stats["n_updates"] += 1

    # ================================================================
    # Training stats accessor
    # ================================================================

    def get_and_reset_train_stats(self) -> Dict[str, float]:
        """
        Return mean training stats since last call and reset accumulators.

        Called at episode boundaries by the training runner to log
        per-episode metrics to TensorBoard.

        During warmup (before the replay buffer has enough samples),
        ``n_updates`` is 0 and all loss/metric fields are ``float('nan')``
        to distinguish "no data" from "zero loss".

        Returns
        -------
        dict
            Dictionary with keys:
                - ``q1_loss``     : Mean Q_1 MSE loss (NaN during warmup)
                - ``q2_loss``     : Mean Q_2 MSE loss (NaN during warmup)
                - ``actor_loss``  : Mean actor (policy) loss (NaN during warmup)
                - ``alpha``       : Mean temperature α (NaN during warmup)
                - ``alpha_loss``  : Mean temperature loss (NaN during warmup)
                - ``entropy``     : Mean policy entropy H(π) (NaN during warmup)
                - ``mean_q``      : Mean Q_1 prediction (NaN during warmup)
                - ``n_updates``   : Total gradient steps this episode
        """
        n = int(self._train_stats["n_updates"])
        if n == 0:
            # Warmup phase: no gradient steps → return NaN for all metrics
            result = {
                "q1_loss": float("nan"),
                "q2_loss": float("nan"),
                "actor_loss": float("nan"),
                "alpha": self.log_alpha.exp().item(),  # still report current α
                "alpha_loss": float("nan"),
                "entropy": float("nan"),
                "mean_q": float("nan"),
                "n_updates": 0,
            }
        else:
            result = {
                "q1_loss": self._train_stats["q1_loss"] / n,
                "q2_loss": self._train_stats["q2_loss"] / n,
                "actor_loss": self._train_stats["actor_loss"] / n,
                "alpha": self._train_stats["alpha"] / n,
                "alpha_loss": self._train_stats["alpha_loss"] / n,
                "entropy": self._train_stats["entropy"] / n,
                "mean_q": self._train_stats["mean_q"] / n,
                "n_updates": n,
            }
        for k in self._train_stats:
            self._train_stats[k] = 0.0
        return result

    # ================================================================
    # PER schedule annealing
    # ================================================================

    def update_per_schedule(
        self,
        episode_idx: int,
        per_alpha_start: float,
        per_alpha_end: float,
        per_beta_start: float,
        per_beta_end: float,
        last_episode: int,
    ) -> Tuple[float, float]:
        """
        Linearly anneal PER alpha and beta over training.

        Alpha: controls prioritization strength (typically 0.6 → 0.4).
        Beta: controls IS-weight correction (typically 0.4 → 1.0).

        Returns the current (alpha, beta) values.
        """
        if not self.use_per:
            return 0.0, 0.0
        denom = max(1, last_episode - 1)
        t = min(1.0, episode_idx / float(denom))
        alpha = per_alpha_start + t * (per_alpha_end - per_alpha_start)
        beta = per_beta_start + t * (per_beta_end - per_beta_start)
        self.memory.alpha = float(alpha)
        self.memory.beta = float(beta)
        return alpha, beta

    # ================================================================
    # make_worker() — Override (SAC trains sequentially)
    # ================================================================

    def make_worker(self) -> "SACDiscreteController":
        """
        Not used in SAC-Discrete.

        SAC uses sequential episodes with a shared replay buffer (like
        DQN), not parallel worker environments (like PPO/A2C). The
        training runner calls ``simulate_LOB_with_MM()`` directly.

        Raises
        ------
        NotImplementedError
            Always. Use ``simulate_LOB_with_MM`` directly.
        """
        raise NotImplementedError(
            "SACDiscreteController uses sequential episodes (like DQN), "
            "not parallel workers. Use simulate_LOB_with_MM directly."
        )
