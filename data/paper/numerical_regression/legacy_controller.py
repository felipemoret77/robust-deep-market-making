#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning controller for Market Making in a LOB simulator.
==============================================================================

This module implements a full-featured Deep RL agent that controls a
MarketMaker inside the LOB_simulation environment. It supports multiple
DQN variants, distributional RL (C51), and SMDP-aware throttle learning.

FILE LAYOUT (in order of appearance):
--------------------------------------
1. ReplayMemory          — Uniform experience replay buffer
2. PrioritizedReplayMemory — Prioritized Experience Replay (PER) buffer
3. NoisyLinear           — NoisyNets layer for parameter-space exploration
4. DeepQNetwork          — MLP Q-network (dueling + distributional + noisy)
5. DeepRLController      — Main controller class (act, learn, train)

INTERFACE:
----------
The controller implements the RLController protocol:
    - act(mm_state)   -> MarketMaker action tuple  (e.g., ("place_bid_ask", 99, 101))
    - learn(...)      -> One RL update step (SMDP aggregation + n-step TD + gradient)

The runner (MM_LOB_SIM.py) calls act() before each environment event and
learn() after each event. The controller handles the mapping between
discrete action indices and MarketMaker command tuples internally.

STATE REPRESENTATIONS:
----------------------
GENERIC mode (pure_mm=False):
    s = [spread, asksize, bidsize, inventory, has_bid, has_ask]
    - 6-dimensional continuous state vector
    - Volumes are log1p-compressed, spread is log1p-compressed,
      inventory is normalized by inv_limit

PURE MM mode (pure_mm=True):
    s = [spread, inventory,
         bid_size_offset_0, ..., bid_size_offset_K,
         ask_size_offset_0, ..., ask_size_offset_K]
    - 2 + 2*(K+1) dimensional vector
    - K = max offset from the pure_mm_offsets grid
    - Includes order-book depth at each offset level

ACTION SPACES:
--------------
GENERIC mode — Discrete actions 0..5 (or 0..8 with inside-spread):
    0: post_bid           — Place a bid at L1 - level_offset
    1: post_ask           — Place an ask at L1 + level_offset
    2: post_bid_ask       — Place both bid and ask
    3: cancel_bid         — Cancel existing bid order
    4: cancel_ask         — Cancel existing ask order
    5: hold               — Do nothing, keep current orders
    6: post_bid_ask_inside_spread  (optional, if n_actions >= 7)
    7: post_bid_inside_spread      (optional)
    8: post_ask_inside_spread      (optional)

PURE MM mode — Discrete actions 0..(n_actions-1):
    Each action maps to an offset pair (Δ_bid, Δ_ask) in ticks.
    The controller ALWAYS tries to be two-sided unless inventory
    limits force one-sided quoting:
        inventory >= +inv_limit → ask only (no new bids)
        inventory <= -inv_limit → bid only (no new asks)
        otherwise              → both bid and ask

RL ALGORITHM FLAGS:
-------------------
    use_sarsa=True     → Deep SARSA (on-policy TD target)
    use_sarsa=False    → DQN (off-policy, max over target Q)
    use_double=True    → Double Q-Learning (online selects, target evaluates)
    use_dueling=True   → Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A)
    use_prioritized_experience=True → PER with alpha/beta annealing
    use_noisy_net=True → NoisyNets (exploration via parameter noise, no ε-greedy)
    use_distributional=True → C51 distributional DQN (categorical atoms)

THROTTLING (Hard Gating):
-------------------------
The controller supports three independent throttle gates that limit how
often the neural network is queried. All gates use OR logic (any gate
blocking = agent is throttled):

    1. Event Gating  (use_event_update) — Wait N environment events
    2. Time Gating   (use_time_update)  — Wait Δt simulated seconds
    3. TOB Gating    (use_tob_update)   — Wait N top-of-book changes

Two bypass priorities override all gates:
    Priority A: Inventory mode change (e.g., crossed limit → must adjust orders)
    Priority B: Fill replenishment (inventory changed → likely got filled)

SMDP LEARNING:
--------------
When throttled, the runner still calls learn() every micro-step. The
controller uses Semi-Markov Decision Process (SMDP) aggregation to
produce correct transitions:
    - Rewards are accumulated with γ^k discounting between decisions
    - Each replay transition stores a per-sample gamma_eff for bootstrapping
    - When throttling is off, this reduces to standard MDP (k=1 always)

See the learn() method for the full SMDP pipeline documentation.
"""

import os
import random
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from collections import deque  # for n-step buffer

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_
from torch.optim import AdamW
from RLController import RLController
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Replay buffer (uniform sampling)
# ============================================================
#
# WHY A REPLAY BUFFER?
# --------------------
# In RL, consecutive transitions are highly correlated (s_t and s_{t+1}
# are almost identical). Training a neural network on correlated batches
# causes unstable learning and poor convergence. A replay buffer breaks
# this correlation by storing past transitions and sampling RANDOM
# mini-batches for training. This is one of the key ingredients that
# makes DQN work (Mnih et al., 2015).
#
# TRANSITION FORMAT (with SMDP):
# ------------------------------
# Each transition stored in the buffer is a list of 6 tensors:
#   [state, action, G_nstep, done, next_state, gamma_eff]
#
# where:
#   state      (1, D)   — state at the root of the n-step window
#   action     (1, 1)   — discrete action index chosen
#   G_nstep    (1, 1)   — n-step return with variable SMDP discounting
#   done       (1, 1)   — True if this transition ends the episode
#   next_state (1, D)   — state at the end of the n-step window
#   gamma_eff  (1, 1)   — effective discount γ^{Σ k_i} for bootstrapping
#
# The buffer is FORMAT-AGNOSTIC: it stores and retrieves arbitrary lists
# of tensors. The 6-field structure is enforced by the controller, not
# the buffer itself.
# ============================================================

class ReplayMemory:
    """
    Uniform experience replay buffer for off-policy RL.

    HOW IT WORKS:
    - Stores transitions as lists of tensors in a circular buffer.
    - When full, new transitions overwrite the oldest ones (FIFO).
    - Sampling is uniform random (every transition has equal probability).
    - Training only starts after batch_size * 10 transitions are stored
      (ensures enough diversity for meaningful gradient updates).

    This is the simplest replay buffer. For prioritized sampling (where
    transitions with higher TD-error are sampled more often), see
    PrioritizedReplayMemory below.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        # Circular buffer: list of transitions, each is a List[Tensor]
        self.memory: List[Optional[List[torch.Tensor]]] = []
        # Optional per-transition metadata for diagnostics.
        self.episode_indices: List[int] = []
        self.delta_values: List[float] = []
        # Write head: index where the next transition will be stored
        self.position: int = 0

    def insert(self, transition: List[torch.Tensor], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert one transition into the circular buffer.

        The transition is a list of tensors (format-agnostic). In the
        current SMDP implementation, it contains 6 fields:
            [state, action, G_nstep, done, next_state, gamma_eff]

        When the buffer is full, the oldest transition is overwritten.
        """
        episode_idx = int((metadata or {}).get("episode_idx", -1))
        delta_value = float((metadata or {}).get("delta", float("nan")))
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.episode_indices.append(episode_idx)
            self.delta_values.append(delta_value)
        self.memory[self.position] = transition
        self.episode_indices[self.position] = episode_idx
        self.delta_values[self.position] = delta_value
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        Sample a random mini-batch and stack each field into a tensor.

        Returns a list of tensors, one per field. Each tensor has shape
        (batch_size, ...) formed by concatenating individual transitions
        along dim=0.

        Example with 6-field SMDP transitions:
            [states_b, actions_b, rewards_b, dones_b, next_states_b, gamma_effs_b]
            shapes: [(B,D), (B,1), (B,1), (B,1), (B,D), (B,1)]
        """
        assert self.can_sample(batch_size)
        indices = random.sample(range(len(self.memory)), batch_size)
        batch = [self.memory[i] for i in indices]
        # Transpose: list of transitions → list of fields
        batch = list(zip(*batch))
        fields = [torch.cat(items, dim=0) for items in batch]
        idxs_b = torch.as_tensor(indices, dtype=torch.long)
        return fields + [idxs_b]

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if we have enough data to start training.

        Heuristic: require at least 10x the batch size. This ensures
        the mini-batches have enough diversity and the buffer isn't
        just replaying the same few transitions over and over.
        """
        return len(self.memory) >= batch_size * 10

    def __len__(self) -> int:
        return len(self.memory)

    def get_episode_indices(self, indices) -> np.ndarray:
        return np.asarray([self.episode_indices[int(i)] for i in indices], dtype=np.int64)

    def get_delta_values(self, indices) -> np.ndarray:
        return np.asarray([self.delta_values[int(i)] for i in indices], dtype=np.float64)


# ============================================================
# Prioritized Replay buffer (PER)
# ============================================================
#
# WHY PRIORITIZED REPLAY?
# -----------------------
# Not all transitions are equally useful for learning. A transition
# where the Q-network prediction is very wrong (high TD-error) contains
# more learning signal than one where the prediction is already accurate.
#
# PER (Schaul et al., 2015) samples transitions proportionally to their
# TD-error magnitude, so "surprising" transitions are replayed more often.
#
# This introduces a BIAS (non-uniform sampling changes the expected
# gradient). To correct this, each sample is weighted by an importance-
# sampling (IS) weight: w_i = (N * p_i)^(-beta). As beta → 1.0, the
# bias is fully corrected. In practice, beta is annealed from ~0.4 to
# 1.0 over training.
# ============================================================

class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) buffer.

    KEY CONCEPTS:
    - Each transition has a priority (typically |TD-error| or cross-entropy).
    - Sampling probability: P(i) = priority_i^alpha / Σ priority_j^alpha
        - alpha=0 → uniform sampling (same as ReplayMemory)
        - alpha=1 → fully prioritized (greedy on TD-error)
    - IS weights: w_i = (N * P(i))^(-beta), normalized by max(w)
        - beta=0 → no correction (biased)
        - beta=1 → full correction (unbiased)
    - New transitions are inserted with max_priority (optimistic: assume
      they are important until proven otherwise).

    The transition format is the same as ReplayMemory (format-agnostic
    list of tensors). sample() returns the fields plus two extras:
        [...fields..., idxs_b, weights_b]
    where idxs_b allows updating priorities after computing TD-errors.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 1.0,
        beta: float = 0.5,
        eps: float = 1e-4,
    ):
        self.capacity = int(capacity)
        self.memory: List[Optional[List[torch.Tensor]]] = []
        self.priorities: List[float] = []
        self.episode_indices: List[int] = []
        self.delta_values: List[float] = []
        self.position: int = 0

        # PER hyperparameters (can be changed externally)
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.eps: float = float(eps)

        # Track maximum priority to assign to new experiences
        self.max_priority: float = 1.0

    def insert(self, transition: List[torch.Tensor], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a new transition with the current max_priority.

        WHY max_priority? New transitions have unknown TD-error (we haven't
        trained on them yet). By assigning max_priority, we guarantee they
        will be sampled at least once, at which point their priority will
        be updated with the actual TD-error.
        """
        episode_idx = int((metadata or {}).get("episode_idx", -1))
        delta_value = float((metadata or {}).get("delta", float("nan")))
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(self.max_priority)
            self.episode_indices.append(episode_idx)
            self.delta_values.append(delta_value)
        else:
            self.memory[self.position] = transition
            self.priorities[self.position] = self.max_priority
            self.episode_indices[self.position] = episode_idx
            self.delta_values[self.position] = delta_value

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample a mini-batch using prioritized probabilities.

        Returns
        -------
        *fields, idxs_b, weights_b

        where fields are the same as ReplayMemory.sample() (one tensor per
        transition field), plus:
            - idxs_b    : (B,) long — indices in the buffer (needed to
                          update priorities after computing TD-errors)
            - weights_b : (B,1) float — importance-sampling weights that
                          correct the bias from non-uniform sampling.
                          Multiply the loss by these weights before
                          backpropagation.
        """
        assert self.can_sample(batch_size)

        # 1) Compute sampling probabilities from priorities
        prios = np.asarray(self.priorities, dtype=np.float64)
        prios = (prios + self.eps) ** self.alpha
        probs = prios / prios.sum()

        # 2) Sample indices according to probs
        indices = np.random.choice(len(self.memory), size=batch_size, p=probs)

        # 3) Compute importance-sampling weights
        #    w_i ∝ (N * p_i)^(-beta), then normalized by max(w_i)
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # 4) Build batch tensors (same format as ReplayMemory)
        transitions = [self.memory[i] for i in indices]
        batch = list(zip(*transitions))
        fields = [torch.cat(items, dim=0) for items in batch]

        idxs_b = torch.as_tensor(indices, dtype=torch.long)
        weights_b = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1)  # (B,1)

        return fields + [idxs_b, weights_b]

    def update_priorities(self, indices, priorities) -> None:
        """
        Update priorities of given indices using TD-error magnitudes.

        Parameters
        ----------
        indices : array-like of ints
        priorities : array-like of floats (typically |TD error| or cross-entropy)
        """
        for idx, pr in zip(indices, priorities):
            idx_int = int(idx)
            pr_float = float(pr)
            # Guard against NaN/Inf which can corrupt PER sampling weights.
            # Clamp to a safe maximum; use current max_priority as fallback
            # for NaN so the sample isn't silently dropped.
            if not math.isfinite(pr_float):
                pr_float = max(self.max_priority, 1.0)
            pr_float = min(pr_float, 1e6)  # v3 config: no effective cap
            self.priorities[idx_int] = pr_float
            if pr_float > self.max_priority:
                self.max_priority = pr_float

    def get_episode_indices(self, indices) -> np.ndarray:
        return np.asarray([self.episode_indices[int(i)] for i in indices], dtype=np.int64)

    def get_delta_values(self, indices) -> np.ndarray:
        return np.asarray([self.delta_values[int(i)] for i in indices], dtype=np.float64)

    def get_priorities(self, indices) -> np.ndarray:
        return np.asarray([self.priorities[int(i)] for i in indices], dtype=np.float64)

    def can_sample(self, batch_size: int) -> bool:
        """
        Same heuristic as uniform buffer.
        """
        return len(self.memory) >= batch_size * 10

    def __len__(self) -> int:
        return len(self.memory)


# ============================================================
# Noisy linear layer (for NoisyNets exploration)
# ============================================================
#
# WHY NOISY NETS?
# ---------------
# Standard DQN uses ε-greedy for exploration: with probability ε, pick
# a random action. This is crude — the randomness is uniform over all
# actions and doesn't adapt to the agent's uncertainty.
#
# NoisyNets (Fortunato et al., 2018) replace ε-greedy with LEARNED
# parameter noise. Each weight and bias has a mean (μ) and a noise
# scale (σ). During training, Gaussian noise is injected:
#     w = μ_w + σ_w * ε     (ε ~ N(0,1))
#
# The network learns WHERE to be noisy (high σ = uncertain regions)
# and WHERE to be precise (low σ = confident predictions). Over time,
# σ tends to shrink as the network becomes more confident, providing
# automatic exploration decay without manual ε schedules.
#
# In this implementation, NoisyLinear replaces nn.Linear ONLY in the
# output heads (value/advantage streams), not in the shared trunk.
# The trunk stays deterministic for training stability.
# ============================================================

class NoisyLinear(nn.Module):
    """
    Noisy linear layer (Fortunato et al., 2018).

    Replaces a standard nn.Linear with a noisy version where each
    parameter has a learned mean (μ) and noise scale (σ):

        Training mode:
            w = w_mu + w_sigma * ε_w
            b = b_mu + b_sigma * ε_b
            → Noise is resampled on EVERY forward pass

        Eval mode:
            w = w_mu,  b = b_mu          (deterministic, no noise)
            → Used for greedy action selection at deployment

    Noise modes (controlled by ``factored`` flag):

        Independent (factored=False, default):
            ε_w ~ N(0, I) of shape (out, in)     — p*q noise samples
            ε_b ~ N(0, I) of shape (out,)

        Factored (factored=True, paper recommendation):
            ε_w = f(ε_out) ⊗ f(ε_in)             — only p+q noise samples
            ε_b = f(ε_out)
            where f(x) = sign(x) * sqrt(|x|)

        Factored noise reduces the number of random samples from p*q to
        p+q.  More importantly, the correlation structure stabilises the
        gradient of sigma, preventing the random-walk growth observed
        with independent noise on high-dimensional outputs (e.g. C51
        with 51 atoms × 7 actions = 357 outputs).

    Parameters
    ----------
    in_features  : int — Number of input neurons (fan_in)
    out_features : int — Number of output neurons
    sigma_init   : float — Initial noise scale (default 0.5)
                   Higher = more exploration initially
    factored     : bool — If True, use factored Gaussian noise (paper
                   default).  If False, use independent noise (legacy).
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5, factored: bool = False):
        super(NoisyLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)
        self.factored = bool(factored)

        # Parameters for mean and noise scales
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))

        # Initialize weight means with Kaiming, bias means with zeros
        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        zeros_(self.b_mu)

        if self.factored:
            # Factored noise init (Fortunato et al. 2018, Section 3.2):
            #   sigma_init_val = sigma_init / sqrt(fan_in)
            # This is the paper's recommended initialisation for factored noise.
            _sigma_val = self.sigma_init / math.sqrt(max(1, self.in_features))
            with torch.no_grad():
                self.w_sigma.fill_(_sigma_val)
                self.b_sigma.fill_(_sigma_val)
        else:
            # Independent noise init (legacy):
            #   w_sigma: Kaiming then rescale by sigma_init
            #   b_sigma: constant sigma_init / sqrt(fan_in)
            kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
            with torch.no_grad():
                self.w_sigma.mul_(self.sigma_init)
                nn.init.constant_(
                    self.b_sigma,
                    self.sigma_init / math.sqrt(max(1, self.in_features)),
                )

    @staticmethod
    def _factored_noise(size: int, device: torch.device) -> torch.Tensor:
        """
        Generate factored noise vector: f(x) = sign(x) * sqrt(|x|).

        This transformation preserves the zero mean of the Gaussian but
        makes the noise heavier-tailed, which helps exploration.  The
        sign preservation ensures the noise can be both positive and
        negative.
        """
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply noisy affine transformation.

        Noise is resampled on every forward pass while training.
        """
        if self.training:
            device = x.device
            if self.factored:
                # Factored noise: ε_w = f(ε_out) ⊗ f(ε_in)
                # Only p + q random samples instead of p * q.
                eps_out = self._factored_noise(self.out_features, device)
                eps_in = self._factored_noise(self.in_features, device)
                w_noise = eps_out.unsqueeze(1) * eps_in.unsqueeze(0)
                b_noise = eps_out
            else:
                # Independent noise: ε_w ~ N(0, I) of shape (out, in)
                w_noise = torch.normal(
                    mean=0.0, std=1.0,
                    size=self.w_mu.size(), device=device,
                )
                b_noise = torch.normal(
                    mean=0.0, std=1.0,
                    size=self.b_mu.size(), device=device,
                )
            w = self.w_mu + self.w_sigma * w_noise
            b = self.b_mu + self.b_sigma * b_noise
            return F.linear(x, w, b)
        else:
            # Deterministic forward at evaluation time
            return F.linear(x, self.w_mu, self.b_mu)


# ============================================================
# Neural network for Q(s, a) with optional Dueling head / NoisyNets
# ============================================================

class DeepQNetwork(nn.Module):
    """
    MLP Q-network for Deep Q-Learning (DQN family).

    What this network learns
    ------------------------
    It approximates the action-value function Q(s, a):
        - input  : state vector s  (shape: [B, input_dim])
        - output : Q-values for each discrete action a (shape: [B, n_actions])

    Dimensions
    ----------
    input_dim:
        Dimension of the state vector.
        - GENERIC mode: typically 6
        - [MM PURE] mode: typically 2 + 2 * (max_offset + 1)

    n_actions:
        Number of discrete actions in your action space.

    Architecture overview
    ---------------------
    The network is split into:
      (1) a shared "trunk" (feature extractor)
      (2) one (or two) output heads

    Shared trunk (deterministic)
    ----------------------------
    Controlled by:
        - n_hidden  : number of hidden layers in the trunk
        - n_neurons : width (neurons) of each hidden layer

    Trunk layout:
        input_dim -> n_neurons -> n_neurons -> ... (n_hidden times)
    with a ReLU after each Linear layer.

    Heads
    -----
    (A) Standard (non-dueling) head: one output head that directly predicts Q(s, a)
        feature_dim (= n_neurons) -> n_actions

    (B) Dueling head (use_dueling=True): two heads
        - Value stream:      V(s) : feature_dim -> 1
        - Advantage stream:  A(s,a): feature_dim -> n_actions

        Combined as:
            Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

        The mean subtraction ensures identifiability (so V and A don't drift
        by adding/subtracting an arbitrary constant).

    NoisyNets option (use_noisy=True)
    --------------------------------
    If enabled, the output head layers (and only the heads) use NoisyLinear
    instead of nn.Linear.
      - This injects learned parameter noise for exploration.
      - In that case, exploration is driven primarily by the network noise
        rather than ε-greedy action selection (though you can still combine them).

    Notes
    -----
    - The shared trunk remains deterministic for stability.
    - Default trunk: n_hidden=2, n_neurons=128  => input -> 128 -> 128
      (this replaces the older hard-coded input -> 128 -> 64 design).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        use_dueling: bool = False,
        use_noisy: bool = False,
        use_fully_noisy: bool = False,
        noisy_sigma_init: float = 0.5,
        use_factored_noise: bool = False,
        n_hidden: int = 2,
        n_neurons: int = 64,
        activation: str = "relu",
        elu_alpha: Optional[float] = None,
        dropout_level: float = 0.0,
        # --- Distributional DQN (C51) option ---
        use_distributional: bool = False,
        atoms: int = 51,
    ):
        super().__init__()
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)
        self.use_fully_noisy = bool(use_fully_noisy) and self.use_noisy
        self.use_factored_noise = bool(use_factored_noise)
        self.n_actions = int(n_actions)

        # Distributional DQN (C51) switch
        self.use_distributional = bool(use_distributional)
        self.atoms = int(atoms)
        if self.use_distributional and self.atoms < 2:
            raise ValueError(
                f"atoms must be >= 2 when use_distributional=True, got {self.atoms}"
            )

        

        # Basic argument checks (helps catch silent config mistakes)
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")
        activation_name = str(activation).strip().lower()
        elu_alpha_eff = None
        if activation_name == "relu":
            activation_factory = lambda: nn.ReLU()
        elif activation_name == "elu":
            elu_alpha_eff = 1.0 if elu_alpha is None else float(elu_alpha)
            if elu_alpha_eff < 0.0:
                raise ValueError(
                    f"elu_alpha must be >= 0, got {elu_alpha_eff}"
                )
            activation_factory = lambda: nn.ELU(alpha=elu_alpha_eff)
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Expected 'relu' or 'elu'."
            )
        self.activation_name = activation_name
        self.elu_alpha = elu_alpha_eff
        self.dropout_level = float(dropout_level)
        if not 0.0 <= self.dropout_level < 1.0:
            raise ValueError(
                f"dropout_level must be in [0, 1), got {self.dropout_level}"
            )

        # ------------------------------------------------------------
        # Shared feature extractor (MLP trunk)
        #   input_dim -> n_neurons -> ... -> n_neurons  (n_hidden times)
        #   use_fully_noisy=True: all trunk layers use NoisyLinear (paper)
        #   use_fully_noisy=False: trunk uses standard nn.Linear (default)
        # ------------------------------------------------------------
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            if self.use_fully_noisy:
                layers.append(NoisyLinear(in_dim, n_neurons,
                                         sigma_init=noisy_sigma_init,
                                         factored=self.use_factored_noise))
            else:
                layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(activation_factory())
            if self.dropout_level > 0.0:
                layers.append(nn.Dropout(p=self.dropout_level))
            in_dim = n_neurons

        self.feature = nn.Sequential(*layers)
        feature_dim = n_neurons  # last hidden size of the trunk

        # ------------------------------------------------------------
        # Output heads
        #   - Standard: one head producing Q(s,a)
        #   - Dueling : value head + advantage head
        # ------------------------------------------------------------
        if self.use_distributional:
            # ------------------------------------------------------------
            # Distributional DQN heads (Categorical / C51)
            #
            # Output logits are reshaped to (B, n_actions, atoms) in forward().
            # Softmax is applied on the atom dimension to obtain probabilities.
            #
            # NOTE:
            # - We keep the same trunk and the same (dueling / non-dueling) structure.
            # - Only the heads change dimensionality.
            # ------------------------------------------------------------
            if self.use_dueling:
                if self.use_noisy:
                    self.fc_value = NoisyLinear(feature_dim, self.atoms, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                    self.fc_adv = NoisyLinear(feature_dim, n_actions * self.atoms, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                else:
                    self.fc_value = nn.Linear(feature_dim, self.atoms)
                    self.fc_adv = nn.Linear(feature_dim, n_actions * self.atoms)
            else:
                if self.use_noisy:
                    self.fc_out = NoisyLinear(feature_dim, n_actions * self.atoms, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                else:
                    self.fc_out = nn.Linear(feature_dim, n_actions * self.atoms)
        else:
            # ------------------------------------------------------------
            # Standard (scalar) Q-value heads
            # ------------------------------------------------------------
            if self.use_dueling:
                if self.use_noisy:
                    # Dueling heads with NoisyLinear
                    self.fc_value = NoisyLinear(feature_dim, 1, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                    self.fc_adv = NoisyLinear(feature_dim, n_actions, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                else:
                    # Dueling heads with standard Linear
                    self.fc_value = nn.Linear(feature_dim, 1)
                    self.fc_adv = nn.Linear(feature_dim, n_actions)
            else:
                if self.use_noisy:
                    # Single head with NoisyLinear
                    self.fc_out = NoisyLinear(feature_dim, n_actions, sigma_init=noisy_sigma_init, factored=self.use_factored_noise)
                else:
                    # Single head with standard Linear
                    self.fc_out = nn.Linear(feature_dim, n_actions)
    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states with shape (B, input_dim).
        return_logits : bool
            If True AND use_distributional, return *both* (probs, logits)
            so the caller can use F.log_softmax on raw logits — avoids
            dead gradients from log(clamp(softmax(...))).

        Returns
        -------
        q : torch.Tensor
            Q-values with shape (B, n_actions).

            If use_distributional=True:
                returns probabilities with shape (B, n_actions, atoms).
                If return_logits=True: returns (probs, logits) tuple.
        """
        # Shared trunk
        x = self.feature(x.float())

        # Distributional DQN (C51): categorical distribution over fixed support
        if self.use_distributional:
            if self.use_dueling:
                adv = self.fc_adv(x).view(-1, self.n_actions, self.atoms)   # (B, A, N)
                value = self.fc_value(x).view(-1, 1, self.atoms)            # (B, 1, N)
                q_logits = value + adv - adv.mean(dim=1, keepdim=True)      # (B, A, N)
            else:
                q_logits = self.fc_out(x).view(-1, self.n_actions, self.atoms)  # (B, A, N)

            q_probs = F.softmax(q_logits, dim=-1)  # (B, A, N)
            if return_logits:
                return q_probs, q_logits
            return q_probs


        # Heads
        if self.use_dueling:
            adv = self.fc_adv(x)             # (B, n_actions)
            value = self.fc_value(x)         # (B, 1)
            # Dueling combination:
            #   Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
            return value + adv - adv.mean(dim=1, keepdim=True)
        else:
            return self.fc_out(x)


# ============================================================
# DeepRLController
# ============================================================

class DeepRLController(RLController):
    """
    Deep RL controller for the MarketMaker in the LOB simulator.

    Responsibilities
    ----------------
    - Maintain:
        * online Q-network (q_net)
        * target Q-network (target_net)
        * replay buffer (uniform or prioritized)
        * epsilon-greedy / greedy policy (depending on use_noisy_net)
        * TensorBoard SummaryWriter (for diagnostics)
    - Provide:
        * act(mm_state)   -> MarketMaker-compatible action tuple
        * learn(...)      -> TD update using replay experiences

    RL Algorithm switches
    ---------------------
    - use_sarsa / use_double / use_dueling / use_prioritized_experience /
      use_noisy_net: control the flavor of SARSA/DQN, Double variants,
      Dueling, PER and NoisyNets.

    n-step TD
    ---------
    - The controller supports n-step returns for TD-learning:
          G_t^{(n)} = Σ_{j=0}^{n-1} γ^j r_{t+j}
      and bootstraps from s_{t+n} with a factor γ^n.
    - When n_steps = 1, this reduces exactly to the original
      one-step TD behavior.

    [MM PURE] structure
    -------------------
    - pure_mm=False (generic mode):
        * Discrete actions 0..5:
              "post_bid", "post_ask", "post_bid_ask",
              "cancel_bid", "cancel_ask", "hold".
        * If n_actions >= 7, indices 6..8 correspond to:
              6: post_bid_ask_inside_spread
              7: post_bid_inside_spread
              8: post_ask_inside_spread
        * The RL policy decides directly which operation to execute.

    - pure_mm=True ("pure" MM mode):
        * Discrete actions 0..(n_actions-1) each correspond to an offset pair
          (Δ_bid, Δ_ask) in ticks relative to the mid-price.
        * The controller ALWAYS tries to quote:
              - BID+ASK if |inventory| < inv_limit (or inv_limit is None),
              - only ASK if inventory >= inv_limit,
              - only BID if inventory <= -inv_limit.
        * The RL policy is only choosing how tight/wide each side is.

      State in [MM PURE] mode:
        s = [spread, inventory,
             bid_size_offset_0, ..., bid_size_offset_K,
             ask_size_offset_0, ..., ask_size_offset_K]

        where K = max offset inferred from pure_mm_offsets.
    """

    def __init__(
        self,
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.99,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epsilon_start: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        batch_size: int = 100,
        replay_capacity: int = 100000,
        target_update_steps: int = 100,
        use_sarsa: bool = False,
        use_double: bool = False,
        use_dueling: bool = False,
        use_prioritized_experience: bool = False,
        use_noisy_net: bool = False,
        use_fully_noisy: bool = False,
        use_factored_noise: bool = False,
        # --- PER annealing hyperparameters ---
        per_alpha_start: float = 1.0,
        per_alpha_end: float = 0.5,
        per_alpha_last_episode: int = 100,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_last_episode: int = 100,
        # --- n-step TD parameter ---
        n_steps: int = 1,
        # --- Intra-decision (SMDP) γ^n reward aggregation toggle ---
        # True  (default, paper convention): r_t = Σ γ^n r_{t,n} (each
        #       throttled micro-step reward inside one decision interval is
        #       discounted by γ^k where k is the micro-step index since the
        #       last decision).
        # False : r_t = Σ r_{t,n}  (undiscounted intra-event sum, γ = 1
        #       intra-event). The outer γ^{N_t} bootstrap on Q(s_{t+1}, a')
        #       is UNCHANGED in both modes.
        use_intra_event_gamma: bool = True,
        device: Optional[torch.device] = None,
        log_dir: str = "runs/deep_mm",
        # [MM PURE] flags / config
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        # --- Q-network architecture hyperparameters ---
        n_hidden: int = 2,
        n_neurons: int = 64,
        activation: str = "relu",
        elu_alpha: Optional[float] = None,
        dropout_level: float = 0.0,
        enable_learning: bool = True,
        # --- Distributional DQN (C51) option ---
        use_distributional: bool = False,
        v_min: float = -10.0,
        v_max: float = 10.0,
        atoms: int = 51,

        # =====================================================================
        # [MODIFICAÇÃO] HARD THROTTLING PARAMETERS
        # =====================================================================
        # These parameters control the "gates" that limit how often the RL agent
        # queries the neural network.
        #
        # - use_tob_update   : if True, only act when Top-Of-Book changes > n_tob_moves
        # - use_event_update : if True, only act every n_events (simulator ticks)
        # - use_time_update  : if True, only act every min_time_interval seconds
        # =====================================================================
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_event_update: bool = False,
        n_events: int = 100,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,

        # --- MDP mode (disable all bypass) ---
        use_mdp: bool = False,

        # --- Flow signal (regime-switching / adversarial) ---
        use_flow_signal: bool = False,
        use_fast_flow_signal: bool = False,
        use_bayes_flow_signal: bool = False,
        bayes_flow_feature_keys: Optional[List[str]] = None,

        # --- Fill imbalance signal (computed in MM_LOB_SIM, read from state dict) ---
        use_fill_imbalance: bool = False,

        # --- Robustness clipping flags ---
        use_g_clip: bool = False,
        use_per_priority_clip: bool = False,

    ):

        # =================================================================
        # CORE RL HYPERPARAMETERS
        # =================================================================
        # These control the fundamental behavior of the DQN agent.
        #
        # level_offset : int
        #     In GENERIC mode, how many ticks away from L1 to place orders.
        #     E.g., level_offset=1 means bid at best_bid - 1, ask at best_ask + 1.
        #     Not used in PURE MM mode (offsets come from pure_mm_offsets).
        #
        # n_actions : int
        #     Size of the discrete action space (= output dimension of Q-network).
        #     GENERIC mode: 6 base actions (+ up to 3 inside-spread variants).
        #     PURE MM mode: must equal len(pure_mm_offsets).
        #
        # gamma : float
        #     Discount factor for future rewards. Higher = more patient agent.
        #     Typical: 0.99 (weighs future ~100 steps) or 0.999 (very patient).
        #
        # batch_size : int
        #     Mini-batch size sampled from replay for each gradient step.
        #
        # target_update_steps : int
        #     How many gradient steps between hard copies of q_net -> target_net.
        #     Hard update (vs soft/polyak) for simplicity.
        #
        # enable_learning : bool
        #     If False, act() still works (for evaluation), but learn() is a
        #     no-op — no replay insertions, no gradient steps, no target updates.
        # =================================================================
        self.level_offset = int(level_offset)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        self.q_n_hidden = int(n_hidden)
        self.q_n_neurons = int(n_neurons)
        self.q_activation = str(activation).strip().lower()
        self.q_elu_alpha = (
            1.0 if self.q_activation == "elu" and elu_alpha is None
            else (float(elu_alpha) if self.q_activation == "elu" else None)
        )
        self.q_dropout_level = float(dropout_level)
        if not 0.0 <= self.q_dropout_level < 1.0:
            raise ValueError(
                f"dropout_level must be in [0, 1), got {self.q_dropout_level}"
            )
        self._enable_learning = bool(enable_learning)

        # -----------------------------------------------------------------
        # n-step TD configuration
        # -----------------------------------------------------------------
        # n_steps controls how many SMDP transitions we chain together to
        # compute the multi-step return G^{(n)} before bootstrapping.
        #   n_steps=1 → standard one-step TD: target = r + γ^k * Q(s', a')
        #   n_steps=3 → 3-step return: G = r_0 + γ^{k_0}*r_1 + γ^{k_0+k_1}*r_2
        #                                + γ^{k_0+k_1+k_2} * Q(s', a')
        # Higher n_steps → faster credit propagation but higher variance.
        self.n_steps = max(1, int(n_steps))

        # -----------------------------------------------------------------
        # Intra-decision (SMDP) reward aggregation toggle
        # -----------------------------------------------------------------
        # use_intra_event_gamma controls how per-micro-step rewards are
        # summed inside ONE SMDP decision interval (between two consecutive
        # controller decisions):
        #
        #   True  (default, paper convention):
        #       r_t = Σ_n γ^n · r_{t,n}
        #       Each throttled micro-step reward is discounted by γ^k.
        #
        #   False (undiscounted intra-decision sum):
        #       r_t = Σ_n r_{t,n}
        #       Equivalent to γ = 1 intra-event.
        #
        # The OUTER bootstrap discount γ^{N_t} on Q(s_{t+1}, a') is
        # UNCHANGED in both modes — only the intra-decision reward
        # aggregation flips.
        # -----------------------------------------------------------------
        self.use_intra_event_gamma = bool(use_intra_event_gamma)

        # -----------------------------------------------------------------
        # Algorithm flavor switches
        # -----------------------------------------------------------------
        # These boolean flags compose orthogonal improvements on top of DQN:
        #
        #   use_sarsa  : True = on-policy SARSA (action from ε-greedy policy)
        #                False = off-policy DQN (action from max Q)
        #
        #   use_double : True = Double DQN/SARSA (decouple selection/evaluation)
        #                Reduces overestimation bias by selecting action with
        #                the online network but evaluating with the target network.
        #
        #   use_dueling : True = Dueling architecture (separate V and A streams)
        #                 Helps the network learn state values independently of
        #                 which action is taken — useful when many actions have
        #                 similar values.
        #
        #   use_prioritized_experience : True = Prioritized Experience Replay (PER)
        #                 Samples transitions proportional to |TD-error|^alpha,
        #                 focusing training on "surprising" experiences.
        #
        #   use_noisy_net : True = NoisyNets exploration (learned noise in heads)
        #                   When True, ε-greedy is DISABLED and the policy is
        #                   purely greedy — exploration comes from parameter noise.
        self.use_sarsa = bool(use_sarsa)
        self.use_double = bool(use_double)
        self.use_dueling = bool(use_dueling)
        self.use_prioritized_experience = bool(use_prioritized_experience)
        self.use_noisy_net = bool(use_noisy_net)
        self.use_fully_noisy = bool(use_fully_noisy)
        self.use_factored_noise = bool(use_factored_noise)

        # Robustness clipping
        self.use_g_clip = bool(use_g_clip)
        self.use_per_priority_clip = bool(use_per_priority_clip)
        self._per_priority_median_ema: float = 1.0
        self._per_priority_ema_alpha: float = 0.01

        # -----------------------------------------------------------------
        # Distributional DQN (C51) configuration
        # -----------------------------------------------------------------
        # Instead of learning E[Z(s,a)] (a scalar), C51 learns the FULL
        # distribution of returns Z(s,a) as a categorical distribution over
        # `atoms` evenly-spaced support points in [v_min, v_max].
        #
        #   v_min, v_max : range of possible return values
        #   atoms        : number of support points (51 in the original paper)
        #
        # The Q-network output changes from (B, A) to (B, A, atoms) where
        # each (A, atoms) slice is a probability distribution (softmax).
        # For action selection, we compute E[Z] = Σ p_i * z_i (dot product
        # with the support vector).
        self.use_distributional = bool(use_distributional)
        self.dist_v_min = float(v_min)
        self.dist_v_max = float(v_max)
        self.dist_atoms = int(atoms)
        if self.use_distributional and self.dist_atoms < 2:
            raise ValueError(
                f"atoms must be >= 2 when use_distributional=True, got {self.dist_atoms}"
            )

        # -----------------------------------------------------------------
        # MM Mode configuration (PURE MM vs GENERIC)
        # -----------------------------------------------------------------
        # pure_mm : bool
        #     False = GENERIC mode: actions are discrete operations
        #             (post_bid, post_ask, cancel_bid, etc.)
        #     True  = PURE MM mode: actions are (Δ_bid, Δ_ask) tick offsets
        #             and the agent always tries to quote both sides.
        #
        # inv_limit : Optional[int]
        #     Maximum allowed inventory in either direction. When breached:
        #     - PURE MM: switches from two-sided to one-sided quoting
        #     - GENERIC: zombie order cleanup + blocks new orders on risky side
        self.pure_mm = bool(pure_mm)
        self.inv_limit = None if inv_limit is None else int(inv_limit)
                
        # [MM PURE] action grid: each action maps to a (bid_offset, ask_offset)
        # pair in ticks relative to L1. Offset 0 = join at L1, positive = deeper
        # in the book (more passive), negative = inside the spread (aggressive).
        if self.pure_mm:
            if pure_mm_offsets is None:
                # Default PURE MM grid (you can override it externally)
                self.pure_mm_offsets: List[Tuple[int, int]] = [
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1),
                ]
            else:
                # Use user-provided offset grid
                self.pure_mm_offsets = list(
                    map(lambda x: (int(x[0]), int(x[1])), pure_mm_offsets)
                )

            # Sanity check: number of actions must match the Q-head
            assert self.n_actions == len(self.pure_mm_offsets), (
                "n_actions must match len(pure_mm_offsets) when pure_mm=True; "
                f"got n_actions={self.n_actions}, len(pure_mm_offsets)={len(self.pure_mm_offsets)}"
            )
            

            # Infer max offsets used for state construction (non-negative only)
            bid_offs = [bo for (bo, _) in self.pure_mm_offsets]
            ask_offs = [ao for (_, ao) in self.pure_mm_offsets]

            self.max_bid_offset = max([bo for bo in bid_offs if bo >= 0], default=0)
            self.max_ask_offset = max([ao for ao in ask_offs if ao >= 0], default=0)
            self.max_offset = max(self.max_bid_offset, self.max_ask_offset)

            # ---------------------------------------------------------
            # CANONICAL EQUIVALENCE MASKS (for inventory-limit states)
            # ---------------------------------------------------------
            # At inv_limit, one side of the (bid_off, ask_off) pair is
            # dropped by the execution layer.  Actions that differ only
            # on the dropped side produce IDENTICAL outcomes (same price,
            # same reward, same next state).
            #
            # For each equivalence group (actions sharing the same
            # recovery-side offset), the smallest-index member is the
            # "canonical" representative.  The boolean masks below are
            # True only for canonical reps and are used in both online
            # action selection (_get_valid_action_mask) and TD-target
            # computation (_get_target_valid_mask_batch) to prevent
            # degenerate actions from being selected.
            # ---------------------------------------------------------
            _canon_ask: Dict[int, int] = {}  # ask_off → first a_idx
            _canon_bid: Dict[int, int] = {}  # bid_off → first a_idx
            for idx, (bo, ao) in enumerate(self.pure_mm_offsets):
                if ao not in _canon_ask:
                    _canon_ask[ao] = idx
                if bo not in _canon_bid:
                    _canon_bid[bo] = idx

            self._canonical_at_long_limit: List[int] = [
                _canon_ask[ao] for (_, ao) in self.pure_mm_offsets
            ]
            self._canonical_at_short_limit: List[int] = [
                _canon_bid[bo] for (bo, _) in self.pure_mm_offsets
            ]

            # Boolean masks: True only for canonical representatives
            self._canonical_mask_long = np.array(
                [self._canonical_at_long_limit[i] == i
                 for i in range(self.n_actions)],
                dtype=bool,
            )
            self._canonical_mask_short = np.array(
                [self._canonical_at_short_limit[i] == i
                 for i in range(self.n_actions)],
                dtype=bool,
            )

        else:
            # Generic mode: base design assumes at least the 6 original actions.
            self.pure_mm_offsets = None
            self.max_bid_offset = 0
            self.max_ask_offset = 0
            self.max_offset = 0
            
            if self.n_actions > 9:
                raise ValueError(
                    "Generic mode supports at most 9 actions (including inside-spread variants). "
                    f"Got n_actions={self.n_actions}."
                )

            if self.n_actions < 6:
                # Hard safety: we need at least the 6 core generic actions.
                print(
                    "[DeepRLController] WARNING: pure_mm=False but n_actions < 6 "
                    f"(n_actions={self.n_actions}). Overriding to 6 generic actions."
                )
                self.n_actions = 6
            elif self.n_actions > 6:
                # Extra actions (indices >= 6) are interpreted as inside-spread
                # variants in _mm_action_from_idx when n_actions >= 7.
                print(
                    "[DeepRLController] INFO: pure_mm=False with extra actions "
                    f"(n_actions={self.n_actions}). Indices >= 6 are interpreted "
                    "as extended (inside-spread) actions by the controller."
                )

        # -----------------------------------------------------------------
        # PER annealing schedules (only used if use_prioritized_experience=True)
        # -----------------------------------------------------------------
        # Alpha controls HOW MUCH prioritization is applied:
        #   alpha=0 → uniform (no prioritization)
        #   alpha=1 → full prioritization (proportional to |TD-error|)
        #   Typical: anneal from 1.0 → 0.5 over training (reduce early bias).
        #
        # Beta controls the importance-sampling (IS) correction:
        #   beta=0 → no correction (biased gradients)
        #   beta=1 → full correction (unbiased, mathematically correct)
        #   Typical: anneal from 0.4 → 1.0 over training (gradual correction).
        self.per_alpha_start = float(per_alpha_start)
        self.per_alpha_end = float(per_alpha_end)
        self.per_alpha_last_episode = int(per_alpha_last_episode)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_last_episode = int(per_beta_last_episode)

        # -----------------------------------------------------------------
        # Epsilon-greedy exploration schedule
        # -----------------------------------------------------------------
        # Epsilon is decayed once per EPISODE (not per step) in the outer
        # training loop: ε_{e+1} = max(ε_min, ε_e * ε_decay).
        #
        # When use_noisy_net=True, epsilon is IGNORED — exploration comes
        # from learned parameter noise in the NoisyLinear layers instead.
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # -----------------------------------------------------------------
        # Device selection (GPU if available, else CPU)
        # -----------------------------------------------------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # -----------------------------------------------------------------
        # Distributional DQN (C51) support vector
        # -----------------------------------------------------------------
        # The support is a fixed set of `atoms` equally-spaced values in
        # [v_min, v_max]. The Q-network predicts a probability for each atom.
        # Expected Q-value = dot(probabilities, support).
        if self.use_distributional:
            self.support = torch.linspace(
                self.dist_v_min,
                self.dist_v_max,
                self.dist_atoms,
                device=self.device,
            )  # (atoms,)
            self.delta = (self.dist_v_max - self.dist_v_min) / (self.dist_atoms - 1)
        else:
            self.support = None
            self.delta = None

        # -----------------------------------------------------------------
        # Q-networks (online + target)
        # -----------------------------------------------------------------
        # We maintain TWO identical networks:
        #   q_net      — the online network (updated every gradient step)
        #   target_net — a frozen copy (updated every target_update_steps)
        #
        # Using a frozen target network prevents the "moving target" problem
        # where the TD target shifts on every gradient step, causing instability.
        #
        # MO-flow signals (extra state dimensions when active).
        # The slow signal is the original EWMA exposed as mo_flow_p_hat in [0, 1],
        # with fallback to the legacy mo_flow_ewma in [-1, +1] for backward
        # compatibility.  The fast signal mirrors the same contract on a shorter
        # horizon, allowing the controller to sense sudden flow imbalances
        # without giving up the smoother long-memory estimate.
        self.use_flow_signal = bool(use_flow_signal)
        self.use_fast_flow_signal = bool(use_fast_flow_signal)
        self.use_bayes_flow_signal = bool(use_bayes_flow_signal)
        self.bayes_flow_feature_keys = (
            list(bayes_flow_feature_keys)
            if bayes_flow_feature_keys is not None
            else [
                "bayes_m_hat",
                "bayes_cp_prob",
                "bayes_expected_run_length",
                "bayes_uncertainty",
            ]
        )
        if self.use_bayes_flow_signal and len(self.bayes_flow_feature_keys) == 0:
            raise ValueError("bayes_flow_feature_keys cannot be empty when use_bayes_flow_signal=True")

        # Fill imbalance EWMA: read from state dict (computed in MM_LOB_SIM)
        self.use_fill_imbalance = bool(use_fill_imbalance)

        # Input dimensionality depends on mode:
        #   GENERIC: 6 features (spread, asksize, bidsize, inventory, has_bid, has_ask)
        #   PURE MM: 2 + 2*(max_offset+1) features (spread, inventory, + depth per offset)
        if self.pure_mm:
            input_dim = 2 + 2 * (self.max_offset + 1)
        else:
            input_dim = 6
        if self.use_flow_signal:
            input_dim += 1
        if self.use_fast_flow_signal:
            input_dim += 1
        if self.use_bayes_flow_signal:
            input_dim += len(self.bayes_flow_feature_keys)
        if self.use_fill_imbalance:
            input_dim += 1

        self.q_net = DeepQNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            use_dueling=self.use_dueling,
            use_noisy=self.use_noisy_net,
            use_fully_noisy=self.use_fully_noisy,
            use_factored_noise=self.use_factored_noise,
            n_hidden=self.q_n_hidden,
            n_neurons=self.q_n_neurons,
            activation=self.q_activation,
            elu_alpha=self.q_elu_alpha,
            dropout_level=self.q_dropout_level,
            use_distributional=self.use_distributional,
            atoms=self.dist_atoms,
        ).to(self.device)

        self.target_net = DeepQNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            use_dueling=self.use_dueling,
            use_noisy=self.use_noisy_net,
            use_fully_noisy=self.use_fully_noisy,
            use_factored_noise=self.use_factored_noise,
            n_hidden=self.q_n_hidden,
            n_neurons=self.q_n_neurons,
            activation=self.q_activation,
            elu_alpha=self.q_elu_alpha,
            dropout_level=self.q_dropout_level,
            use_distributional=self.use_distributional,
            atoms=self.dist_atoms,
        ).to(self.device)


        # Initialize target_net with the same weights as q_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly; always in eval mode

        # If learning is disabled at construction time (evaluation / backtest),
        # put q_net into eval mode so NoisyNet layers become deterministic.
        if not self._enable_learning:
            self.q_net.eval()

        # AdamW optimizer. Keep weight decay configurable; the current
        # default is 0.01 for long-run training, and callers can still
        # override it for legacy-matching experiments.
        # Default: uniform LR.  For fine-tuning with LR differential, use
        # setup_differential_lr() after construction.
        self.optimizer = AdamW(
            self.q_net.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        # L2 anchor regularization for fine-tuning (continual RL).
        # When set (via set_anchor_weights()), an L2 penalty toward the
        # anchor weights is added to the loss: λ_anchor * ||θ - θ_anchor||².
        # This prevents catastrophic forgetting by pulling the network
        # back toward the pre-trained baseline when it drifts too far.
        #
        # When `_ewc_fisher` is additionally set (via set_ewc_fisher), the
        # anchor loss becomes Fisher-weighted (Elastic Weight Consolidation,
        # Kirkpatrick et al. 2017 PNAS):
        #     L_EWC = (λ/2) · Σ_i F_ii · (θ_i - θ*_i)²
        # instead of the plain L2 form.  See ewc.py for the estimator.
        self._anchor_weights: Optional[Dict[str, torch.Tensor]] = None
        self._anchor_lambda: float = 0.0
        self._ewc_fisher: Optional[Dict[str, torch.Tensor]] = None
        # Optional per-parameter binary mask for the anchor term. When set,
        # the anchor penalty becomes λ · Σ ((θ - θ*) ⊙ M)². Useful after a
        # warmstart that pads the first input layer with new columns: the
        # mask zeroes those new columns so the anchor does not fight the
        # learning of brand-new features.
        self._anchor_masks: Optional[Dict[str, torch.Tensor]] = None

        # -----------------------------------------------------------------
        # Replay memory
        # -----------------------------------------------------------------
        # Two options:
        #   Uniform replay    — sample transitions with equal probability
        #   Prioritized (PER) — sample proportional to |TD-error|^alpha
        #
        # Both store 6-field SMDP transitions:
        #   (state, action, G_nstep, done, next_state, gamma_eff)
        if self.use_prioritized_experience:
            # Prioritized replay buffer with initial alpha/beta
            self.memory = PrioritizedReplayMemory(
                capacity=replay_capacity,
                alpha=self.per_alpha_start,
                beta=self.per_beta_start,
            )
        else:
            # Original uniform replay buffer
            self.memory = ReplayMemory(capacity=replay_capacity)

        # -----------------------------------------------------------------
        # Bookkeeping variables
        # -----------------------------------------------------------------

        # Last action index chosen by act(). Used by learn() to know
        # which action to attribute to the current transition.
        # Set to None at start and reset to None at episode end.
        self.last_action_idx: Optional[int] = None

        # Counter for target network hard updates (incremented each gradient step)
        self.train_steps: int = 0

        # TensorBoard writer for logging training curves
        self.writer = SummaryWriter(log_dir=log_dir)

        # Total number of gradient steps performed across all episodes
        self.global_grad_step: int = 0

        # Episode-level accumulators (reset in log_episode_stats at episode end)
        self.episode_loss_sum: float = 0.0    # sum of TD losses this episode
        self.episode_loss_count: int = 0       # number of gradient steps this episode

        # Per-action usage counters for diagnostics (how often each action is chosen)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        # Per-action inventory accumulators (for conditional inventory reports)
        self.action_inv_sum = np.zeros(self.n_actions, dtype=np.float64)
        self.action_inv_abs_sum = np.zeros(self.n_actions, dtype=np.float64)

        # Per-action Q-value accumulators for TensorBoard diagnostics.
        # At each decision point in act(), Q(s, a) for ALL actions is recorded.
        # At episode end, log_episode_stats() computes the mean Q per action
        # and writes it to TensorBoard, then resets.
        self.episode_q_sum = np.zeros(self.n_actions, dtype=np.float64)
        self.episode_q_count: int = 0

        # Q-value range tracking: min/max across all decision states in the episode.
        # Used to verify that C51 support [V_min, V_max] covers the actual Q range.
        self.episode_q_min: float = float('inf')
        self.episode_q_max: float = float('-inf')

        # TD target diagnostics (populated in _train_step, logged in log_episode_stats).
        # scalar_td_target = r + (1-done) * γ_eff * Q_target(s', a*) — the scalar
        # equivalent of the distributional Bellman target.
        self.episode_td_target_min: float = float('inf')
        self.episode_td_target_max: float = float('-inf')
        self.episode_td_target_sum: float = 0.0
        self.episode_td_target_count: int = 0
        # Fraction of C51 Tz atoms that get clipped to [V_min, V_max].
        # High clipping → the support is too narrow and the agent loses information.
        self.episode_tz_clip_frac_sum: float = 0.0

        # Diagnostics: per-episode max KL/TD-error priority and extreme-reward
        # batch fraction.  Used to detect replay buffer poisoning from
        # catastrophic episodes (large KL → high PER priority → over-sampling).
        self.episode_max_priority: float = 0.0
        self.episode_extreme_reward_count: int = 0
        self.episode_batch_total: int = 0
        self.episode_tz_clip_count: int = 0
        self.episode_sample_recent_count: int = 0
        self.episode_sample_total: int = 0
        self.episode_sample_delta_sum: float = 0.0
        self.episode_sample_delta_count: int = 0
        self.episode_sample_rewards: List[float] = []
        self.episode_sample_td_errors: List[float] = []
        self.episode_sample_priorities: List[float] = []

        # Replay-sampling context used for TensorBoard diagnostics.
        self.current_replay_episode_idx: int = -1
        self.current_replay_delta: float = float("nan")
        self.replay_recent_window_episodes: int = 50

        # n-step internal buffer:
        #   It stores SMDP-level transitions for the current episode as:
        #       (s_t, a_t, r_t, done_t, s_{t+1}, k_t)
        #   where k_t is the number of micro-steps this SMDP transition
        #   spans (i.e., how many runner learn() calls were aggregated
        #   into this single decision-level transition).
        #
        #   When no throttling is active, every step is a decision and
        #   k_t = 1, so this reduces to standard n-step behavior.
        #
        #   IMPORTANT: We remove `maxlen` because with SMDP, two commits
        #   can happen in a single learn() call (one for the previous
        #   pending decision + one for episode done). With maxlen, the
        #   deque could silently drop the oldest element before we get
        #   a chance to commit it. Without maxlen, we manage the length
        #   explicitly via popleft() inside commit_nstep_transition().
        self.nstep_buffer: deque = deque()

        # =================================================================
        # SMDP (Semi-Markov Decision Process) STATE VARIABLES
        # =================================================================
        # These variables track the currently-pending SMDP transition.
        #
        # WHY SMDP?
        # ---------
        # When the controller is throttled, the runner still calls learn()
        # every micro-step. Without SMDP, each micro-step would create a
        # replay transition with the SAME stale action index but DIFFERENT
        # states — corrupting the replay buffer with transitions where
        # (s, a) don't correspond to an actual decision.
        #
        # With SMDP, we aggregate all micro-steps between two consecutive
        # decisions into a SINGLE transition:
        #
        #   (s_decision, a_decision, R_cumulative, s_next_decision)
        #
        # where R_cumulative = Σ_{k=0}^{K-1} γ^k r_{t+k} is the
        # discounted sum of micro-step rewards over the K steps between
        # decisions. This is mathematically equivalent to treating the
        # throttled controller as a Semi-Markov Decision Process.
        #
        # BACKWARD COMPATIBILITY:
        # -----------------------
        # When throttling is disabled (all gates off), every step is a
        # decision, K=1 for every transition, and behavior is identical
        # to the original code.
        #
        # VARIABLES:
        #   _smdp_pending       : bool — True if we have an open (uncommitted) transition
        #   _smdp_s_decision    : dict — the state_before at decision time
        #   _smdp_a_decision    : int  — the action index chosen at decision time
        #   _smdp_cum_reward    : float — γ-discounted cumulative reward so far
        #   _smdp_k_steps       : int  — number of micro-steps accumulated
        #   _last_was_decision  : bool — set by act(), read by learn()
        # =================================================================
        self._smdp_pending: bool = False
        self._smdp_s_decision: Optional[Dict[str, Any]] = None
        self._smdp_a_decision: Optional[int] = None
        self._smdp_cum_reward: float = 0.0
        self._smdp_k_steps: int = 0
        self._last_was_decision: bool = False

        # =====================================================================
        # HARD THROTTLING — State Initialization
        # =====================================================================
        #
        # Hard throttling limits HOW OFTEN the RL agent queries the neural
        # network for a new decision. Between decisions, the agent simply
        # holds its current position (returns "hold" from act()).
        #
        # THREE INDEPENDENT GATES (OR logic — any gate blocking = throttled):
        #
        #   Gate 1: TOB Gating (use_tob_update)
        #     - Counts how many times the Top-Of-Book (best bid/ask/sizes)
        #       has changed since the last decision.
        #     - Only acts when moves_tob >= threshold_tob.
        #     - Rationale: Don't react to events that don't change L1.
        #
        #   Gate 2: Event Gating (use_event_update)
        #     - Counts how many environment events (arrivals, cancels) have
        #       occurred since the last decision.
        #     - Only acts when event_steps >= threshold_events.
        #     - Rationale: Batch multiple events before re-evaluating.
        #
        #   Gate 3: Time Gating (use_time_update)
        #     - Tracks simulated wall-clock time since the last decision.
        #     - Only acts when (t_now - t_last) >= min_time_interval.
        #     - Rationale: Model realistic latency constraints (e.g., IBKR ~50ms).
        #
        # TWO BYPASS PRIORITIES (override all gates, force immediate decision):
        #
        #   Priority A: Mode Change
        #     - Inventory crossed a limit (e.g., went from two_sided to ask_only).
        #     - Must adjust orders immediately to avoid risk.
        #
        #   Priority B: Fill Replenishment
        #     - Inventory changed since last decision (a fill happened).
        #     - Must replace the filled order before the next market event.
        #
        # MDP MODE (use_mdp=True):
        #   Disables BOTH bypass priorities (A and B).  The agent strictly
        #   respects the throttle gates — no early decisions for fills,
        #   inventory emergencies, or mode changes.  This removes the
        #   variable decision timing that makes the problem an SMDP:
        #     - Event gate: K is exactly fixed (every N events).
        #     - Time gate:  Δt is fixed, but K varies because inter-arrival
        #       times are stochastic (still MDP — state sampled at constant Δt).
        #     - TOB gate:   K varies (depends on how often L1 moves).
        #   In all cases, bypasses no longer inject extra decisions, so the
        #   decision process is governed solely by the chosen gate.
        #   The first-action bypass is still active (the agent must place
        #   initial quotes at the start of each episode).
        #
        # =====================================================================

        # Gate configuration (which gates are active and their thresholds)
        self.use_tob_update = bool(use_tob_update)
        self.threshold_tob = max(1, int(n_tob_moves))

        self.use_event_update = bool(use_event_update)
        self.threshold_events = max(1, int(n_events))

        self.use_time_update = bool(use_time_update)
        self.min_time_interval = float(min_time_interval)

        # MDP mode: disable all bypass (fills, mode changes).
        # The agent strictly respects the throttle — fixed decision timing.
        self.use_mdp = bool(use_mdp)

        # Gate counters (reset after each decision via _reset_clocks())
        self.moves_tob = 0              # TOB changes since last decision
        self.last_env_tob_key = None    # Last observed TOB signature (bb, ba, bs, as)
        self.event_steps = 0            # Environment events since last decision
        self.last_update_time = -1.0    # Simulated time of last decision (-1 = no baseline)

        # First-action bypass flag.  Set to True after the first real
        # decision (inside _reset_clocks).  While False, ALL throttle
        # gates are bypassed so the agent can place initial quotes.
        # Replaces the old last_update_time<0 sentinel which broke
        # when use_time_update=False (last_update_time stayed -1 forever).
        self._has_acted_once: bool = False

        # Bypass trackers (detect fills and mode changes)
        self.last_inventory = None      # Inventory at last decision (float)
        self.last_mode = None           # Last quoting mode ("two_sided"/"ask_only"/"bid_only")

        # Debug printout
        print("\n================ DEEP CONTROLLER SETUP ================")
        print(f"Algorithm          : {'Deep SARSA' if self.use_sarsa else 'Deep Q-Learning'}")
        print(f"Double             : {self.use_double}")
        print(f"Dueling            : {self.use_dueling}")
        print(f"Prioritized Replay : {self.use_prioritized_experience}")
        _noisy_suffix = ""
        if self.use_noisy_net:
            _scope = "fully-noisy" if self.use_fully_noisy else "heads-only"
            _noise_kind = "factored" if self.use_factored_noise else "independent"
            _noisy_suffix = f" ({_scope}, {_noise_kind})"
        print(f"Noisy Nets         : {self.use_noisy_net}{_noisy_suffix}")
        print(f"[MM PURE]          : {self.pure_mm}")
        print(f"[MM PURE] inv_limit: {self.inv_limit}")
        if self.pure_mm:
            print(f"[MM PURE] offsets          : {self.pure_mm_offsets}")
            print(f"[MM PURE] max_bid_offset   : {self.max_bid_offset}")
            print(f"[MM PURE] max_ask_offset   : {self.max_ask_offset}")
            print(f"[MM PURE] max_offset(state): {self.max_offset}")

        # Throttling Log
        print("\n--- Throttling Configuration ---")
        print(f"Time Gating        : {self.use_time_update} (min {self.min_time_interval}s)")
        print(f"Event Gating       : {self.use_event_update} (min {self.threshold_events} steps)")
        print(f"TOB Gating         : {self.use_tob_update} (min {self.threshold_tob} moves)")
        print(f"MDP Mode (no bypass): {self.use_mdp}")
        print(f"Slow Flow EWMA     : {self.use_flow_signal}")
        print(f"Fast Flow EWMA     : {self.use_fast_flow_signal}")
        print(f"Bayes Flow Signal  : {self.use_bayes_flow_signal}")
        if self.use_bayes_flow_signal:
            print(f"Bayes Flow Features: {self.bayes_flow_feature_keys}")
        print(f"Fill Imbalance     : {self.use_fill_imbalance}")

        if self.use_prioritized_experience:
            print("  PER alpha schedule:")
            print(f"     start={self.per_alpha_start:.4f}  end={self.per_alpha_end:.4f}  last_ep={self.per_alpha_last_episode}")
            print("  PER beta schedule:")
            print(f"     start={self.per_beta_start:.4f}   end={self.per_beta_end:.4f}   last_ep={self.per_beta_last_episode}")

        print("\n--- Core Hyperparameters ---")
        print(f"Gamma           = {self.gamma}")
        print(f"Learning rate   = {self.optimizer.param_groups[0]['lr']}")
        print(f"Batch size      = {self.batch_size}")
        print(f"Replay capacity = {replay_capacity}")
        print(f"Target updates  = {self.target_update_steps} steps")
        print(f"n-steps (TD)    = {self.n_steps}")
        print(f"Q-net n_hidden  = {self.q_n_hidden}")
        print(f"Q-net n_neurons = {self.q_n_neurons}")
        print(f"Q-net activation= {self.q_activation}")
        if self.q_activation == "elu":
            print(f"Q-net elu_alpha = {self.q_elu_alpha}")
        print(f"Q-net dropout   = {self.q_dropout_level}")

        print("\n--- Exploration Schedule (ε-greedy) ---")
        print(f"Epsilon start   = {self.epsilon}")
        print(f"Epsilon min     = {self.epsilon_min}")
        print(f"Epsilon decay   = {self.epsilon_decay} (per episode)")
        if self.use_noisy_net:
            print("NOTE: use_noisy_net=True → policy is greedy; exploration via parameter noise (ε not used).")

        print("\n--- Neural Network & Device ---")
        print(f"Device          = {self.device}")
        print(f"Input dim       = {input_dim}")
        print(f"Num actions     = {self.n_actions}")

        print("=======================================================\n")

    # --------------------------------------------------------
    # Throttling Helpers (private)
    # --------------------------------------------------------

    def _get_env_tob(self, s: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Extract a Top-Of-Book (TOB) signature from the state dictionary.

        Returns a 4-tuple: (best_bid, best_ask, bid_size, ask_size).
        This is used by the TOB gate to detect meaningful L1 changes.
        If the tuple hasn't changed since the last decision, the event
        was "deep in the book" and doesn't warrant a new decision.
        """
        def _safe_int(x, d=-1):
            try: return int(x)
            except: return d
        def _safe_float(x, d=0.0):
            try: return float(x)
            except: return d

        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        # We try to use env specific sizes if available, else 0
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        
        try:
            ibs, ias = int(round(env_bs)), int(round(env_as))
        except:
            ibs, ias = 0, 0
            
        return (env_bb, env_ba, ibs, ias)

    def _reset_clocks(self, current_time: float, tob_key: Tuple[int, int, int, int]):
        """
        Reset all throttle gate counters after making a decision.

        Called from act() whenever should_act=True. This starts a fresh
        counting window for all three gates:
          - TOB gate: reset moves_tob to 0, update baseline signature
          - Event gate: reset event_steps to 0
          - Time gate: record current_time as the new baseline
        """
        # Mark that the agent has acted at least once — disables
        # the first-action bypass for all subsequent steps.
        self._has_acted_once = True

        if self.use_tob_update:
            self.moves_tob = 0
            self.last_env_tob_key = tob_key

        if self.use_event_update:
            self.event_steps = 0

        if self.use_time_update:
            self.last_update_time = current_time

    # --------------------------------------------------------
    # SMDP Helpers
    # --------------------------------------------------------

    def _smdp_reset(self) -> None:
        """
        Clear all SMDP aggregation state.

        Called at the start of each episode and after committing the
        final pending SMDP transition (e.g., on episode termination).
        """
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0

    def _smdp_commit_to_nstep(
        self,
        s_next_dict: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Commit the currently-pending SMDP transition into the n-step buffer.

        This method is called when:
          (a) A new decision arrives (we close the previous pending transition
              with s_next = state_before of the new decision), OR
          (b) The episode ends (we close the pending transition with
              s_next = state_after of the terminal step, done=True).

        The transition pushed into the n-step buffer is:
            (s_decision, a_decision, R_cum, done, s_next, k_steps)

        where:
            s_decision : tensor (1, D) — state at decision time
            a_decision : tensor (1, 1) — action index chosen at decision time
            R_cum      : tensor (1, 1) — γ-discounted cumulative reward
            done       : tensor (1, 1) — whether this transition ends the episode
            s_next     : tensor (1, D) — next state (either next decision or terminal)
            k_steps    : int           — number of micro-steps this transition spans

        IMPORTANT: k_steps is stored as a plain int (not a tensor) because it is
        used later in commit_nstep_transition() to compute the effective discount
        γ^{Σ k_i} for the n-step return. It does NOT go into the replay buffer
        directly; instead, the effective discount is computed when the n-step
        return is assembled and stored as gamma_eff alongside the transition.
        """
        if not self._smdp_pending:
            return  # Nothing to commit (e.g., very first step)

        # Convert to tensors — detach and move to CPU to avoid holding
        # GPU memory in the n-step buffer between SMDP decisions.
        s_t = self._state_to_tensor(self._smdp_s_decision).detach().cpu()
        s_tp1 = self._state_to_tensor(s_next_dict).detach().cpu()
        a_t = torch.tensor(
            [[self._smdp_a_decision]], dtype=torch.long,
        )  # (1, 1)
        r_t = torch.tensor(
            [[self._smdp_cum_reward]], dtype=torch.float32,
        )  # (1, 1)
        done_t = torch.tensor(
            [[done]], dtype=torch.bool,
        )  # (1, 1)

        k = self._smdp_k_steps  # plain int

        # Push the 6-tuple into the n-step buffer
        self.nstep_buffer.append((s_t, a_t, r_t, done_t, s_tp1, k))

        # Clear the pending state
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0

    # ================================================================
    # STATE ENCODING
    # ================================================================
    #
    # The state dictionary produced by MarketMaker.build_state() contains
    # human-readable keys ("spread", "inventory", "bidsize", etc.).
    # The Q-network needs a fixed-length float tensor.
    #
    # This section converts dict → tensor AND applies normalization so
    # that every feature lives in a neural-network-friendly range (~0..10).
    # Without normalization, raw values span wildly different scales
    # (spread: 1-20, volumes: 0-10000, inventory: -50..+50), which causes
    # the network to learn at very different rates per feature.
    # ================================================================

    def _state_to_tensor(self, s: Dict[str, Any]) -> torch.Tensor:
            """
            Convert MM state dictionary to a (1, D) float tensor on self.device.

            WHY NORMALIZATION?
            ------------------
            Neural networks learn fastest when inputs are on similar scales.
            Raw LOB data has wildly different ranges:
              - Spread: 1..20 ticks
              - Volumes: 0..10,000+ shares
              - Inventory: -50..+50 lots

            We apply the following transforms:
              1. Volumes → log1p(x): compresses [0, 10000] → [0, 9.2]
              2. Spread  → log1p(max(0, x)): same compression + guards against
                           negative spread (crossed/empty book)
              3. Inventory → x / inv_limit: maps to approx [-1, +1]
              4. Boolean flags (has_bid, has_ask) → 0.0 or 1.0 (already safe)

            Output shapes:
              GENERIC mode: (1, 6)  — [spread, asksize, bidsize, inventory, has_bid, has_ask]
              PURE MM mode: (1, 2 + 2*(K+1)) — [spread, inventory, bid_depths..., ask_depths...]
            """
            
            # --- Common Normalization Constants ---
            # Normalize inventory by the limit. If no limit is set, default to 10.0.
            inv_denom = float(self.inv_limit) if self.inv_limit is not None else 10.0
            inv_denom = max(inv_denom, 1.0)  # Safety to avoid division by zero
    
            if not self.pure_mm:
                # ======================================================
                # GENERIC MODE (pure_mm=False)
                # ======================================================
                
                # 1. Spread: Normalize to be roughly around 0.1 ~ 0.5
                #    Clip at 20.0 prevents massive outliers from shocking the network.
                raw_spread = float(s["spread"])

                # Guard: raw_spread can be negative if the book is crossed
                # or empty. log1p(x) requires x > -1 to avoid NaN.
                spread = math.log1p(max(0.0, raw_spread))
    
                # 2. Volumes: Use Log1p to handle "long tail" distributions.
                #    Raw volumes can be 1 or 10,000. Neural nets hate this scale difference.
                #    log1p(x) = log(1+x). Example: 5000 -> ~8.5
                # BUG FIX (C): clamp to 0 before log1p — a negative size
                # (e.g. from an upstream LOB bug) would cause ValueError.
                asksize = math.log1p(max(0.0, float(s.get("asksize", 0.0))))
                bidsize = math.log1p(max(0.0, float(s.get("bidsize", 0.0))))
    
                # 3. Inventory: Normalize to [-1.0, 1.0] range
                raw_inv = float(s["inventory"])
                inventory = raw_inv / inv_denom
    
                # 4. Boolean Flags: Already 0.0 or 1.0 (Safe)
                has_bid = 1.0 if bool(s.get("has_bid", False)) else 0.0
                has_ask = 1.0 if bool(s.get("has_ask", False)) else 0.0
    
                features = [spread, asksize, bidsize, inventory, has_bid, has_ask]
                if self.use_flow_signal:
                    features.append(float(s.get("mo_flow_p_hat", s.get("mo_flow_ewma", 0.0))))
                if self.use_fast_flow_signal:
                    _fast_default = 0.5 * (float(s.get("mo_flow_fast_ewma", 0.0)) + 1.0)
                    features.append(float(s.get("mo_flow_fast_p_hat", _fast_default)))
                if self.use_bayes_flow_signal:
                    features.extend(float(s.get(k, 0.0)) for k in self.bayes_flow_feature_keys)
                if self.use_fill_imbalance:
                    features.append(float(s.get("fill_imbalance_ewma", 0.0)))
                arr = np.array(features, dtype=np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
                return tensor
    
            # ======================================================
            # PURE MM MODE (pure_mm=True)
            # ======================================================
            
            # 1. Spread & Inventory (Same normalization logic)
            raw_spread = float(s["spread"])
            # Guard: same as generic mode — clamp to 0 before log1p.
            spread = math.log1p(max(0.0, raw_spread))
      
            raw_inv = float(s["inventory"])
            inventory = raw_inv / inv_denom
    
            # Maximum offset used for state
            K = int(self.max_offset)
    
            # Try to read aggregated sizes from the state dict
            bid_sizes_seq = s.get("pure_mm_bid_sizes", None)
            ask_sizes_seq = s.get("pure_mm_ask_sizes", None)
    
            # Fallback: if not provided, use single L1 sizes and pad with zeros
            if bid_sizes_seq is None:
                base_bid = float(s.get("bidsize", 0.0))
                bid_sizes = [base_bid] + [0.0] * K
            else:
                bid_sizes = list(map(float, bid_sizes_seq))
                
            if ask_sizes_seq is None:
                base_ask = float(s.get("asksize", 0.0))
                ask_sizes = [base_ask] + [0.0] * K
            else:
                ask_sizes = list(map(float, ask_sizes_seq))
    
            # Ensure both lists have length ≥ K+1
            if len(bid_sizes) < K + 1:
                bid_sizes = bid_sizes + [0.0] * (K + 1 - len(bid_sizes))
            if len(ask_sizes) < K + 1:
                ask_sizes = ask_sizes + [0.0] * (K + 1 - len(ask_sizes))
    
            # Truncate to exactly K+1 per side (offset 0..K)
            # AND Apply Log1p normalization to all depth values
            # BUG FIX (C): clamp each depth value to 0 before log1p to
            # prevent ValueError on negative sizes from upstream bugs.
            bid_sizes = [math.log1p(max(0.0, x)) for x in bid_sizes[: K + 1]]
            ask_sizes = [math.log1p(max(0.0, x)) for x in ask_sizes[: K + 1]]
    
            # Final state vector:
            #   [spread, inventory, bid_sizes..., ask_sizes...,
            #    (mo_flow_p_hat), (mo_flow_fast_p_hat),
            #    selected Bayesian flow features,
            #    (fill_imbalance)]
            features = [spread, inventory] + bid_sizes + ask_sizes
            if self.use_flow_signal:
                features.append(float(s.get("mo_flow_p_hat", s.get("mo_flow_ewma", 0.0))))
            if self.use_fast_flow_signal:
                _fast_default = 0.5 * (float(s.get("mo_flow_fast_ewma", 0.0)) + 1.0)
                features.append(float(s.get("mo_flow_fast_p_hat", _fast_default)))
            if self.use_bayes_flow_signal:
                features.extend(float(s.get(k, 0.0)) for k in self.bayes_flow_feature_keys)
            if self.use_fill_imbalance:
                features.append(float(s.get("fill_imbalance_ewma", 0.0)))
            arr = np.array(features, dtype=np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            return tensor

    # ================================================================
    # enable_learning property
    # ================================================================
    #
    # When enable_learning is toggled, we automatically switch q_net
    # between train() and eval() mode. This is critical for NoisyNet:
    # NoisyLinear.forward() checks self.training to decide whether to
    # inject stochastic weight noise. Without this, evaluation with
    # enable_learning=False would remain non-deterministic.

    @property
    def enable_learning(self) -> bool:
        return self._enable_learning

    @enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        self._enable_learning = bool(value)
        if hasattr(self, "q_net"):
            if self._enable_learning:
                self.q_net.train()
            else:
                self.q_net.eval()
                # A frozen controller must be a true eval actor: no pending
                # SMDP/n-step transition should survive into a frozen phase or
                # later be committed into the replay buffer after unfreezing.
                self.last_action_idx = None
                if hasattr(self, "nstep_buffer"):
                    self.nstep_buffer.clear()
                if hasattr(self, "_smdp_reset"):
                    self._smdp_reset()

    # ================================================================
    # ACTION SELECTION (epsilon-greedy / greedy)
    # ================================================================
    #
    # The agent needs to select actions from Q-values. Two modes:
    #
    # (A) Standard ε-greedy:
    #     With probability ε, pick a random action (explore).
    #     With probability (1 - ε), pick argmax Q(s,a) (exploit).
    #     ε is decayed externally once per episode.
    #
    # (B) NoisyNet (use_noisy_net=True):
    #     Always pick argmax Q(s,a) — but Q itself is noisy because
    #     the network has stochastic weights. Exploration is implicit
    #     in the parameter noise and shrinks automatically as σ → 0.
    #
    # For Distributional DQN (C51), the network outputs probability
    # distributions (B, A, N) instead of scalar Q-values. We convert
    # to scalar Q-values via E[Z] = Σ_i p_i * z_i before argmax.
    # ================================================================

    def _q_values_from_net(self, net: torch.nn.Module, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar Q-values from a network, handling both standard and
        distributional (C51) architectures transparently.

        For standard DQN:
            net(state_batch) → (B, n_actions) — used directly.

        For Distributional DQN (C51):
            net(state_batch) → (B, n_actions, atoms) — probability distributions.
            We compute the expected value: Q(s,a) = Σ_i p_i * z_i
            where z_i are the fixed support atoms.

        Parameters
        ----------
        net : nn.Module — either q_net or target_net
        state_batch : (B, D) float tensor

        Returns
        -------
        q_values : (B, n_actions) float tensor — scalar Q per action
        """
        out = net(state_batch)

        if not self.use_distributional:
            return out  # (B, n_actions)

        # C51: expectation over atoms
        # out: (B, A, N) ; support: (N,)
        support = self.support.view(1, 1, -1)
        q_values = (out * support).sum(dim=-1)
        return q_values  # (B, A)
    # ================================================================
    # CONSTRAINED ACTION SELECTION
    # ================================================================
    #
    # WHY CONSTRAINED SELECTION?
    # --------------------------
    # Without masking, the network freely picks argmax Q(s, a) over ALL
    # actions, and a downstream safety layer silently overrides invalid
    # choices (e.g., placing a bid when inventory is at +limit).  The
    # replay buffer then stores the ORIGINAL intention, not the actually
    # executed action.  This creates a mismatch: the network learns from
    # (s, a_intended, r, s') tuples where r was produced by a DIFFERENT
    # action (the safety override).  Over time this adds noise to the
    # Q-function and slows convergence.
    #
    # The fix: BEFORE selecting an action, compute a boolean validity
    # mask based on current inventory constraints.  The mask is applied
    # to both the greedy branch (set Q[invalid] = -∞) and the ε-random
    # branch (sample uniformly from valid actions only).  The selected
    # action index always corresponds to a permitted, actually-executed
    # action, so the replay buffer stores clean transitions.
    #
    # SCOPE:
    # - GENERIC mode: full masking (bid-side actions blocked at long
    #   limit, ask-side at short limit, recovery-order cancel blocked).
    # - PURE MM mode: mask non-canonical duplicates at inv_limit
    #   (keep one representative per recovery-side offset); away from
    #   limits, keep all actions valid.
    # ================================================================

    def _get_valid_action_mask(self, mm_state: dict) -> np.ndarray:
        """
        Compute a boolean mask indicating which actions are permitted
        given the current inventory constraints.

        This is the core of the constrained action selection mechanism.
        It examines the agent's inventory position relative to its limits
        and returns a mask where True = action is allowed, False = action
        is forbidden.

        The mask is consumed by _epsilon_greedy_action() to restrict both
        greedy and exploratory selection to the valid subset.

        GENERIC MODE masking rules:
            inv >= +inv_limit (long):
                BLOCK  0 (post_bid)        — would increase long exposure
                BLOCK  2 (post_bid_ask)     — includes a bid
                BLOCK  3 (cancel_bid)       — no-op (bid consumed by fill)
                BLOCK  4 (cancel_ask)       — would remove recovery order
                BLOCK  6 (bid_ask_inside)   — includes a bid  (if exists)
                BLOCK  7 (bid_inside)       — pure bid        (if exists)
                ALLOW  1 (post_ask), 5 (hold)
                ALLOW  8 (ask_inside)       (if exists)

            inv <= -inv_limit (short):
                BLOCK  1 (post_ask)         — would increase short exposure
                BLOCK  2 (post_bid_ask)     — includes an ask
                BLOCK  3 (cancel_bid)       — would remove recovery order
                BLOCK  4 (cancel_ask)       — no-op (ask consumed by fill)
                BLOCK  6 (bid_ask_inside)   — includes an ask (if exists)
                BLOCK  8 (ask_inside)       — pure ask        (if exists)
                ALLOW  0 (post_bid), 5 (hold)
                ALLOW  7 (bid_inside)       (if exists)

        PURE MM MODE:
            All actions are two-sided offset pairs.  At inventory limits
            the execution layer converts to one-sided, so actions that
            differ only on the dropped side are degenerate (identical
            outcomes).  We mask non-canonical duplicates, keeping only
            the smallest-index representative per recovery-side offset.
            Away from limits, mask is all-True.

        Parameters
        ----------
        mm_state : dict
            State dictionary from MarketMaker.build_state().

        Returns
        -------
        mask : np.ndarray, shape (n_actions,), dtype bool
            True where the action is permitted, False where forbidden.
        """
        mask = np.ones(self.n_actions, dtype=bool)

        if self.inv_limit is None:
            return mask

        inv = float(mm_state.get("inventory", 0.0))

        # ---------------------------------------------------------------
        # PURE MM mode — mask degenerate actions at inventory limits
        # ---------------------------------------------------------------
        # At inv_limit, one side of the (bid_off, ask_off) pair is
        # dropped by the execution layer.  Actions that differ only
        # on the dropped side produce identical outcomes.  We keep
        # only one canonical representative per equivalence group
        # (the smallest-index action with the same recovery-side offset).
        if self.pure_mm:
            if inv >= self.inv_limit:
                mask = self._canonical_mask_long.copy()
            elif inv <= -self.inv_limit:
                mask = self._canonical_mask_short.copy()
            return mask

        # ---------------------------------------------------------------
        # GENERIC mode — mask out inventory-violating actions
        # ---------------------------------------------------------------

        if inv >= self.inv_limit:
            # Long limit reached: only recovery-side actions allowed.
            # The bid was consumed by the fill that pushed us to the limit,
            # so cancel_bid is a no-op.  Block it to avoid wasting
            # exploration budget on an action identical to hold.
            mask[0] = False                              # post_bid (would worsen)
            mask[2] = False                              # post_bid_ask (includes bid)
            mask[3] = False                              # cancel_bid (no-op: no bid exists)
            mask[4] = False                              # cancel_ask (would remove recovery)
            if self.n_actions >= 7:
                mask[6] = False                          # post_bid_ask_inside_spread
            if self.n_actions >= 8:
                mask[7] = False                          # post_bid_inside_spread

        if inv <= -self.inv_limit:
            # Short limit reached: symmetric to long limit.
            # The ask was consumed by the fill, so cancel_ask is a no-op.
            mask[1] = False                              # post_ask (would worsen)
            mask[2] = False                              # post_bid_ask (includes ask)
            mask[3] = False                              # cancel_bid (would remove recovery)
            mask[4] = False                              # cancel_ask (no-op: no ask exists)
            if self.n_actions >= 7:
                mask[6] = False                          # post_bid_ask_inside_spread
            if self.n_actions >= 9:
                mask[8] = False                          # post_ask_inside_spread

        # Safety: if every action got masked (should never happen with
        # the rules above, since hold/cancel/post remain), fall back to
        # allowing everything so the agent doesn't crash.
        if not mask.any():
            mask[:] = True

        return mask

    def _epsilon_greedy_action(
        self,
        state_tensor: torch.Tensor,
        valid_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select a single action for the current state (used by act()).

        This is the ONLINE action selection method called during environment
        interaction. It returns a single integer action index.

        CONSTRAINED SELECTION (valid_mask != None):
            When a validity mask is provided, invalid actions are excluded
            from BOTH the greedy and exploratory branches:
              - Greedy:  Q-values of invalid actions are set to -∞ before
                         argmax, guaranteeing the chosen action is valid.
              - Random:  ε-exploration samples uniformly from the VALID
                         subset only, preventing wasted exploration budget
                         on actions that would be overridden anyway.

            This ensures the stored action index always corresponds to a
            permitted, actually-executed action — eliminating the mismatch
            between intention and execution that previously added noise to
            the replay buffer.

        Behavior depends on the exploration mode:

        (A) NoisyNet (use_noisy_net=True):
            Pure greedy: a = argmax_a Q(s, a)  [over valid actions only]
            Exploration is implicit in the stochastic weights of the network.
            Since q_net is in training mode, NoisyLinear layers add noise.

        (B) Standard ε-greedy (use_noisy_net=False):
            With probability ε → random action (uniform over VALID actions)
            With probability (1-ε) → greedy action argmax_a Q(s, a) [valid]

        Parameters
        ----------
        state_tensor : (1, D) float tensor on self.device
        valid_mask : optional (n_actions,) boolean numpy array
            True = action is permitted, False = forbidden.
            None = all actions are valid (backwards-compatible default).

        Returns
        -------
        a_idx : int — index in [0, n_actions), guaranteed to be valid
        """
        # Pre-compute valid indices for random sampling
        if valid_mask is not None:
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                valid_indices = np.arange(self.n_actions)  # safety fallback
        else:
            valid_indices = np.arange(self.n_actions)

        # NoisyNet case: greedy policy with optional ε-floor.
        # A small ε > 0 prevents self-reinforcing directional bias in the
        # replay buffer by ensuring uniform exploration of all actions.
        if self.use_noisy_net:
            if self.epsilon > 0 and np.random.rand() < self.epsilon:
                return int(np.random.choice(valid_indices))
            with torch.no_grad():
                q_values = self._q_values_from_net(self.q_net, state_tensor)  # (1, n_actions)
                if valid_mask is not None:
                    mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
                    q_values = q_values.clone()
                    q_values[0, ~mask_t] = float('-inf')
                a_idx = int(torch.argmax(q_values, dim=-1).item())
            return a_idx

        # Standard ε-greedy (non-noisy networks)
        if np.random.rand() < self.epsilon:
            # Explore: sample uniformly from VALID actions only
            return int(np.random.choice(valid_indices))

        # Exploit: greedy argmax over VALID Q-values
        with torch.no_grad():
            q_values = self._q_values_from_net(self.q_net, state_tensor)  # (1, n_actions)
            if valid_mask is not None:
                mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.clone()
                q_values[0, ~mask_t] = float('-inf')
            a_idx = int(torch.argmax(q_values, dim=-1).item())
        return a_idx

    def _epsilon_greedy_action_batch_from_net(
        self,
        net: torch.nn.Module,
        state_batch: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Batch action selection (used inside _train_step for SARSA targets).

        Unlike _epsilon_greedy_action() which processes a single state,
        this method processes an entire batch at once — needed when computing
        TD targets for SARSA / Double SARSA, where we need a' for each
        transition in the mini-batch.

        Behavior:
            NoisyNet → pure greedy actions from net (noise is in the weights)
            Standard → vectorized ε-greedy (random mask over the batch)

        Parameters
        ----------
        net : nn.Module — the network to compute Q-values from
        state_batch : (B, D) float tensor
        valid_mask : (B, n_actions) bool tensor, optional
            Per-sample action validity mask from _get_target_valid_mask_batch.
            True = action is allowed, False = action is forbidden.
            When provided, greedy argmax ignores masked actions and random
            exploration only samples from valid actions.

        Returns
        -------
        actions : (B, 1) long tensor — selected action per sample
        """
        B = state_batch.size(0)

        with torch.no_grad():
            q_values = self._q_values_from_net(net, state_batch)  # (B, n_actions)

            # Apply validity mask to greedy selection: set Q of invalid
            # actions to -inf so argmax never picks them.
            if valid_mask is not None:
                q_values = q_values.clone()
                q_values[~valid_mask] = -float("inf")

            greedy_actions = torch.argmax(q_values, dim=1, keepdim=True)  # (B,1)

        if self.use_noisy_net:
            # NoisyNet with optional ε-floor for directional-bias prevention.
            if self.epsilon > 0:
                explore_mask = (torch.rand(B, 1, device=state_batch.device) < self.epsilon)
                if valid_mask is not None:
                    probs = valid_mask.float()
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    random_actions = torch.multinomial(probs, num_samples=1)
                else:
                    random_actions = torch.randint(0, self.n_actions, (B, 1), device=state_batch.device)
                return torch.where(explore_mask, random_actions, greedy_actions)
            return greedy_actions

        # Standard ε-greedy: some random actions with prob ε
        explore_mask = (torch.rand(B, 1, device=state_batch.device) < self.epsilon)

        if valid_mask is not None:
            # Sample random actions only from the valid subset per sample.
            # Use uniform probabilities over valid actions, zero for invalid.
            probs = valid_mask.float()  # (B, n_actions)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            random_actions = torch.multinomial(probs, num_samples=1)  # (B, 1)
        else:
            random_actions = torch.randint(
                low=0,
                high=self.n_actions,
                size=(B, 1),
                device=state_batch.device,
            )

        actions_b = torch.where(explore_mask, random_actions, greedy_actions)
        return actions_b

 
    # =========================================================================
    # ACTION MAPPING & SAFETY LOGIC (REFACTORED)
    # =========================================================================

    def _mm_action_from_idx(self, a_idx: int, state: Dict[str, Any]) -> tuple:
        """
        Main Dispatcher: Map discrete action index to MarketMaker macro-action.
        
        This method routes the decision to the specific logic handler based on 
        the controller mode (PURE MM vs GENERIC).
        
        Parameters
        ----------
        a_idx : int
            The discrete action index selected by the neural network.
        state : Dict[str, Any]
            The current state dictionary (includes L1 prices, inventory, etc.).

        Returns
        -------
        tuple
            The command tuple to be sent to the MarketMaker wrapper.
        """
        if self.pure_mm:
            return self._mm_action_from_idx_pure_mm(a_idx, state)
        else:
            return self._mm_action_from_idx_generic(a_idx, state)

    # -------------------------------------------------------------------------
    # 1. PURE MM MODE (Offsets Logic)
    # -------------------------------------------------------------------------
    def _mm_action_from_idx_pure_mm(self, a_idx: int, state: Dict[str, Any]) -> tuple:
        """
        PURE MM Strategy: Map index -> (Δ_bid, Δ_ask) offsets.
        
        Logic Overview:
        ---------------
        1. Decode the action index into Bid/Ask offsets relative to L1.
        2. Calculate raw target prices based on Best Bid / Best Ask.
        3. Apply Aggressive Logic (negative offsets) ONLY if the spread allows it.
        
        CRITICAL SAFETY NETS IMPLEMENTED:
        ---------------------------------
        1. Internal Consistency (Anti-Cross):
           - The logic guarantees that Ask_Price > Bid_Price (minimum 1 tick spread).
           - If the raw calculation results in a crossed book (e.g., due to tight 
             spreads and fallback logic), we FORCIBLY push the Ask price up.
             
        2. Inventory Limits (Skewing):
           - If inventory is Long (>= limit), we disable Bidding.
           - If inventory is Short (<= -limit), we disable Asking.
        """
        a_idx = int(a_idx)
        inv = float(state.get("inventory", 0.0))

        # --- A. Retrieve Market State ---
        # We need the current Best Bid and Best Ask to calculate relative prices.
        # Use .get() safely because the book might be empty on one side.
        best_bid_raw = state.get("best_bid", None)
        best_ask_raw = state.get("best_ask", None)
        
        # Safely convert to integer (ticks) if they exist
        best_bid = int(best_bid_raw) if best_bid_raw is not None else None
        best_ask = int(best_ask_raw) if best_ask_raw is not None else None

        # Fallback reference: Mid Price (rounded to nearest tick)
        # Used if one side of the book is empty (liquidity hole).
        mid = float(state.get("mid", 0.0))
        mid_px = int(round(mid))

        # Calculate current spread safely
        spread = 0
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        # --- B. Decode Offsets ---
        # Ensure the action index is within the valid range of the offset grid.
        # The grid contains tuples like (0,0), (-1,0), (1,1), etc.
        safe_idx = min(max(0, a_idx), len(self.pure_mm_offsets) - 1)
        bid_off, ask_off = self.pure_mm_offsets[safe_idx]
        bid_off = int(bid_off)
        ask_off = int(ask_off)

        # =================================================================
        # EARLY EXIT: ONE-SIDED QUOTING AT INVENTORY LIMIT
        # =================================================================
        # When the inventory limit is breached, only the RECOVERY side
        # matters (ask for long, bid for short).  The opposite-side offset
        # is irrelevant — the order will be cancelled, not placed.
        #
        # CRITICAL: We compute the recovery-side price INDEPENDENTLY,
        # without calculating the opposite side or running the anti-cross
        # check.  This eliminates a subtle dependency: in the old code,
        # the anti-cross (ask_price = bid_price + 1) could cause the
        # ask_price to vary depending on bid_off, even though bid_off is
        # dropped at inv_limit.  That made actions with the same ask_off
        # but different bid_off produce different outcomes — breaking the
        # equivalence assumption needed for canonical degenerate masking.
        #
        # By computing only the recovery side here, actions with the same
        # recovery-side offset are GUARANTEED to produce identical outcomes.
        # =================================================================
        if self.inv_limit is not None:

            if inv >= self.inv_limit:
                # ---- LONG LIMIT → ask-only recovery ----
                if best_ask is not None:
                    if ask_off == 0:
                        ask_price = best_ask
                    elif ask_off > 0:
                        ask_price = best_ask + ask_off
                    else:
                        # Aggressive: improve inside the spread (if room)
                        if spread >= 2:
                            candidate = best_ask + ask_off
                            floor = best_bid + 1 if best_bid is not None else candidate
                            ask_price = max(candidate, floor)
                        else:
                            ask_price = best_ask
                else:
                    ask_price = mid_px + abs(ask_off)

                ask_price = max(1, int(ask_price))
                return ("cancel_all_then_place", -1, ask_price)

            if inv <= -self.inv_limit:
                # ---- SHORT LIMIT → bid-only recovery ----
                if best_bid is not None:
                    if bid_off == 0:
                        bid_price = best_bid
                    elif bid_off > 0:
                        bid_price = best_bid - bid_off
                    else:
                        # Aggressive: improve inside the spread (if room)
                        if spread >= 2:
                            candidate = best_bid - bid_off
                            ceiling = best_ask - 1 if best_ask is not None else candidate
                            bid_price = min(candidate, ceiling)
                        else:
                            bid_price = best_bid
                else:
                    bid_price = mid_px - abs(bid_off)

                bid_price = max(1, int(bid_price))
                return ("cancel_all_then_place", +1, bid_price)

        # =================================================================
        # NORMAL PATH: TWO-SIDED QUOTING (|inv| < inv_limit)
        # =================================================================

        # --- C. Calculate Target Prices ---

        # 1. Bid Price Logic
        if best_bid is not None:
            if bid_off == 0:
                bid_price = best_bid
            elif bid_off > 0:
                bid_price = best_bid - bid_off
            else:
                if spread >= 2:
                    candidate = best_bid - bid_off
                    limit = best_ask - 1 if best_ask is not None else candidate
                    bid_price = min(candidate, limit)
                else:
                    bid_price = best_bid
        else:
            bid_price = mid_px - abs(bid_off)

        # 2. Ask Price Logic
        if best_ask is not None:
            if ask_off == 0:
                ask_price = best_ask
            elif ask_off > 0:
                ask_price = best_ask + ask_off
            else:
                if spread >= 2:
                    candidate = best_ask + ask_off
                    limit = best_bid + 1 if best_bid is not None else candidate
                    ask_price = max(candidate, limit)
                else:
                    ask_price = best_ask
        else:
            ask_price = mid_px + abs(ask_off)

        bid_price = max(1, int(bid_price))
        ask_price = max(1, int(ask_price))

        # =================================================================
        # >>> ANTI-CROSS FIREWALL (two-sided path only) <<<
        # =================================================================
        # Enforce Ask >= Bid + 1 to prevent locked/crossed market.
        # This only runs in the two-sided path — the one-sided early exit
        # above doesn't need it (there's only one price to compute).
        if ask_price <= bid_price:
            ask_price = bid_price + 1

        # Standard Case: Quote both sides
        return ("place_bid_ask", bid_price, ask_price)

    # -------------------------------------------------------------------------
    # 2. GENERIC MODE (Discrete Actions Logic)
    # -------------------------------------------------------------------------
    def _mm_action_from_idx_generic(self, a_idx: int, state: Dict[str, Any]) -> tuple:
            """
            GENERIC Strategy: Map discrete action index -> MarketMaker macro-action.
            
            This method translates the Neural Network's choice (integer index) into a 
            concrete command for the execution engine (tuple).
            
            CRITICAL FIXES & SAFETY NETS IMPLEMENTED:
            -----------------------------------------
            1. PRIORITY ZOMBIE CLEANUP (Inventory Constraints):
               - Problem: If the inventory limit is reached (e.g., +12), the agent might still 
                 have a resting BID in the book. If the network chooses "Hold", that BID 
                 remains alive. If the market crashes, the BID gets filled, pushing inventory 
                 to +13 (violation).
               - Fix: We check inventory limits FIRST. If we are at the limit and have an 
                 open order on the dangerous side, we FORCE a cancellation immediately, 
                 overriding the network's choice.
            
            2. MARKET CONSISTENCY (Anti-Crossing):
               - We ensure calculated quotes never cross the market (e.g., Buying above Best Ask) 
                 or cross each other (Bid >= Ask), preventing "Math Domain Errors" in the simulator.
            """
    
            # --- 1. Define Action Space ---
            # Map integer indices to human-readable intent strings
            action_list = [
                "post_bid",                     # 0: Place Bid only
                "post_ask",                     # 1: Place Ask only
                "post_bid_ask",                 # 2: Place both Bid and Ask
                "cancel_bid",                   # 3: Cancel Bid
                "cancel_ask",                   # 4: Cancel Ask
                "hold",                         # 5: Do nothing
            ]
            # Append extended actions if configured (for inside-spread quoting)
            if self.n_actions >= 7: action_list.append("post_bid_ask_inside_spread")
            if self.n_actions >= 8: action_list.append("post_bid_inside_spread")
            if self.n_actions >= 9: action_list.append("post_ask_inside_spread")
    
            # Clamp index to ensure safety against out-of-bounds prediction
            safe_idx = min(max(0, int(a_idx)), len(action_list) - 1)
            a_name = action_list[safe_idx]
    
            # --- 2. Retrieve Market & Portfolio State ---
            # Get L1 prices (Best Bid/Ask). Use None if the book side is empty.
            best_bid_raw = state.get("best_bid", None)
            best_ask_raw = state.get("best_ask", None)
            best_bid = int(best_bid_raw) if best_bid_raw is not None else None
            best_ask = int(best_ask_raw) if best_ask_raw is not None else None
    
            # Portfolio state
            has_bid = bool(state.get("has_bid", False))
            has_ask = bool(state.get("has_ask", False))
            inv = float(state.get("inventory", 0.0))
            mid = float(state.get("mid", 0.0))
            mid_px = int(round(mid))
    
            # =================================================================
            # >>> CRITICAL FIX: ZOMBIE ORDER CLEANUP (PRIORITY 1) <<<
            # =================================================================
            # We MUST check limits BEFORE accepting "Hold" or "Cancel".
            # If we are strictly forbidden from holding a position (due to limits)
            # but we still have an order open, we must kill it NOW.
    
            if self.inv_limit is not None:
                # CASE A: Long Inventory Limit Reached (e.g., +12)
                # We are forbidden from Buying. If we have a live Bid, it's a "Zombie".
                if inv >= self.inv_limit and has_bid:
                    # We need to wipe the Bid. Ideally, we replace it with an Ask.
                    # Calculate a SAFE Ask price (L1 or fallback to mid).
                    safe_ask = best_ask if best_ask is not None else (mid_px + max(1, self.level_offset))
                    
                    # Safety: Ensure our Ask doesn't cross the market Bid
                    if best_bid is not None and safe_ask <= best_bid:
                        safe_ask = best_bid + 1
                    
                    # Return force-cancel command: Wipe everything, place Ask at safe price.
                    return ("cancel_all_then_place", -1, int(safe_ask))
    
                # CASE B: Short Inventory Limit Reached (e.g., -12)
                # We are forbidden from Selling. If we have a live Ask, it's a "Zombie".
                if inv <= -self.inv_limit and has_ask:
                    # Calculate a SAFE Bid price.
                    safe_bid = best_bid if best_bid is not None else (mid_px - max(1, self.level_offset))
                    
                    # Safety: Ensure our Bid doesn't cross the market Ask
                    if best_ask is not None and safe_bid >= best_ask:
                        safe_bid = best_ask - 1
                        
                    # Return force-cancel command: Wipe everything, place Bid at safe price.
                    return ("cancel_all_then_place", +1, int(safe_bid))
            # =================================================================
    
            # --- 3. Handle Non-Pricing Actions ---
            # Now that we've handled critical inventory violations, it is safe
            # to process simple actions like "Hold" or "Cancel".
            #
            # INVENTORY-AWARE CANCEL GUARD: At the inventory limit, block
            # cancellation of the recovery-side order (the only order that can
            # reduce inventory).  Without this guard, a random "cancel_ask"
            # at inv=+limit removes the agent's only way to sell, wasting an
            # entire throttle interval before it can re-post.
            if a_name == "cancel_bid":
                if self.inv_limit is not None and inv <= -self.inv_limit:
                    return ("hold",)  # protect the recovery bid
                return ("cancel_bid",)
            if a_name == "cancel_ask":
                if self.inv_limit is not None and inv >= self.inv_limit:
                    return ("hold",)  # protect the recovery ask
                return ("cancel_ask",)
            if a_name == "hold":       return ("hold",)
            # NOTE: "cancel_all" was removed here because it is never present
            # in the action_list (generic actions are 0..5 or 0..8), so this
            # branch was dead code and could never be reached.
    
            # --- 4. Decode Intent ---
            # Determine which side(s) the agent *wants* to quote on based on the action name.
            want_bid = a_name in ("post_bid", "post_bid_ask", "post_bid_ask_inside_spread", "post_bid_inside_spread")
            want_ask = a_name in ("post_ask", "post_bid_ask", "post_bid_ask_inside_spread", "post_ask_inside_spread")
    
            # --- 5. Inventory Constraints (Prevention) ---
            # Prevents placing *NEW* orders if at the limit.
            # (Existing zombie orders were already handled in Section 2).
            if self.inv_limit is not None:
                if inv >= self.inv_limit:
                    want_bid = False  # Block new Bids
                if inv <= -self.inv_limit:
                    want_ask = False  # Block new Asks
    
            # --- 6. Calculate Target Prices ---
            # Default strategy: Quote at L1 +/- level_offset
            if best_bid is not None: 
                bid_price = best_bid - self.level_offset
            else:                      
                # Fallback if book is empty on Bid side
                bid_price = mid_px - max(1, self.level_offset)
    
            if best_ask is not None: 
                ask_price = best_ask + self.level_offset
            else:                      
                # Fallback if book is empty on Ask side
                ask_price = mid_px + max(1, self.level_offset)
    
            # Handle "Inside Spread" actions (Special delegation)
            # If valid, we delegate to the wrapper which handles micro-offsets.
            if "inside_spread" in a_name:
                if want_bid and want_ask: return ("place_bid_ask_inside_spread",)
                if want_bid: return ("place_bid_inside_spread",)
                if want_ask: return ("place_ask_inside_spread",)
                return ("hold",)
    
            # =================================================================
            # >>> SAFETY NET: PRICE VALIDATION <<<
            # =================================================================
            
            # A) External Consistency (Maker-Only Rule)
            # Do not cross the market L1 (taking liquidity).
            if want_bid and (best_ask is not None):
                # If buying, price must be < Best Ask
                if bid_price >= best_ask: 
                    bid_price = best_ask - 1
            
            if want_ask and (best_bid is not None):
                # If selling, price must be > Best Bid
                if ask_price <= best_bid: 
                    ask_price = best_bid + 1
    
            # B) Internal Consistency (Self-Cross Check)
            # If placing both Bid and Ask, ensure Bid < Ask.
            if want_bid and want_ask:
                if ask_price <= bid_price:
                    # Force a minimum spread of 1 tick
                    ask_price = bid_price + 1
                    
                    # Corner Case: If pushing Ask up hits the Market Ask, push Bid down instead.
                    if best_ask is not None and ask_price > best_ask:
                        ask_price = best_ask
                        bid_price = ask_price - 1
            # =================================================================
            
            # Ensure integers for the simulator engine
            bid_price = int(bid_price)
            ask_price = int(ask_price)
    
            # --- 7. Final Dispatch ---
            # Dispatch the calculated and validated orders.
            if want_bid and want_ask: return ("place_bid_ask", bid_price, ask_price)
            if want_bid:              return ("place_bid", bid_price)
            if want_ask:              return ("place_ask", ask_price)
            
            # Default fallback
            return ("hold",)

    
    
    # ================================================================
    # act() — MAIN ACTION SELECTION ENTRY POINT
    # ================================================================
    #
    # This is the method called by the runner every micro-step to get
    # the next MarketMaker command. It implements the full decision
    # pipeline:
    #
    #   1. Update throttle counters (event count, TOB changes, time)
    #   2. Check bypass priorities (mode change, fill replenishment)
    #   3. Check throttle gates (event, time, TOB)
    #   4. If should_act → query network, map action, reset clocks
    #      If throttled → return ("hold",) and set SMDP flag
    #
    # ================================================================
    # Fine-tuning utilities (L2 anchor, LR differential, replay save/load)
    # ================================================================

    def set_anchor_weights(self, lambda_anchor: float = 0.001) -> None:
        """
        Snapshot the current Q-network weights as the anchor for L2
        regularization.  Call this AFTER loading a pre-trained checkpoint
        and BEFORE starting fine-tuning.

        During subsequent gradient steps, an L2 penalty
            λ_anchor * Σ_i ||θ_i - θ_anchor_i||²
        is added to the loss, preventing catastrophic forgetting by
        pulling the weights back toward the pre-trained baseline.

        Parameters
        ----------
        lambda_anchor : float
            Regularization strength.  0.001 is a good starting point —
            strong enough to prevent drift, weak enough to allow adaptation.
        """
        self._anchor_weights = {
            name: p.clone().detach()
            for name, p in self.q_net.named_parameters()
        }
        self._anchor_lambda = lambda_anchor
        # A fresh anchor invalidates any previously set mask.
        self._anchor_masks = None

    def set_anchor_lambda(self, lambda_anchor: float) -> None:
        """
        Update the L2 anchor regularization strength WITHOUT re-snapshotting
        the anchor weights.

        This is used for anchor decay schedules: the anchor weights (the
        "baseline" that we penalise drift from) stay fixed at the pre-trained
        checkpoint, but the STRENGTH of the pull can be adjusted over the
        course of training.

        Typical usage in the training loop:
            progress = ep / (N_EPISODES - 1)
            new_lambda = get_anchor_lambda(progress, start=0.05, end=0.02)
            controller.set_anchor_lambda(new_lambda)

        Parameters
        ----------
        lambda_anchor : float
            New regularization strength.  Must be >= 0.
            Setting to 0 effectively disables anchor regularization
            (equivalent to removing the penalty term from the loss).

        Raises
        ------
        ValueError
            If called before set_anchor_weights() (no anchor to decay).
        """
        if self._anchor_weights is None:
            raise ValueError(
                "set_anchor_lambda() called but no anchor weights have been set. "
                "Call set_anchor_weights() first to snapshot the baseline weights.")
        self._anchor_lambda = max(0.0, lambda_anchor)

    def set_anchor_masks(
        self,
        masks: Dict[str, torch.Tensor],
    ) -> None:
        """
        Register per-parameter binary masks to be applied to the anchor term.

        Must be called AFTER set_anchor_weights(). Keys must be a subset of
        the anchored parameter names; each mask must match the shape of its
        corresponding parameter.  Values should be 0/1 tensors (or floats in
        [0,1] if a softer down-weighting is desired).  Parameters without an
        entry in `masks` are anchored normally (implicit all-ones mask).

        Typical use case: after warmstarting from a checkpoint with a
        smaller input dimension, the first linear layer is zero-padded on
        new input columns.  Passing a mask that is 1 on the original
        columns and 0 on the new ones prevents the anchor from pulling
        those new columns back to zero, letting them learn freely.
        """
        if self._anchor_weights is None:
            raise ValueError(
                "set_anchor_masks() requires anchor weights to be set first. "
                "Call set_anchor_weights() before set_anchor_masks()."
            )
        validated: Dict[str, torch.Tensor] = {}
        for name, mask in masks.items():
            if name not in self._anchor_weights:
                raise ValueError(
                    f"Mask key '{name}' is not in anchor weights. "
                    f"Available: {sorted(self._anchor_weights.keys())[:4]}..."
                )
            ref = self._anchor_weights[name]
            if tuple(mask.shape) != tuple(ref.shape):
                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} does not match anchor "
                    f"shape {tuple(ref.shape)} for parameter '{name}'."
                )
            validated[name] = mask.detach().to(dtype=ref.dtype)
        self._anchor_masks = validated

    def _compute_anchor_loss(self) -> torch.Tensor:
        """
        Compute the anchor regularization term.

        - Plain L2 (default):
              L = λ · Σ_i (θ_i - θ*_i)²
        - Fisher-weighted EWC (when `_ewc_fisher` is set):
              L = (λ/2) · Σ_i F_ii · (θ_i - θ*_i)²

        Returns 0 if no anchor is set.
        """
        if self._anchor_weights is None or self._anchor_lambda <= 0:
            return torch.tensor(0.0, device=next(self.q_net.parameters()).device)

        device = next(self.q_net.parameters()).device
        anchor_loss = torch.tensor(0.0, device=device)
        use_ewc = self._ewc_fisher is not None

        use_mask = self._anchor_masks is not None

        for name, p in self.q_net.named_parameters():
            if name not in self._anchor_weights:
                continue
            theta_star = self._anchor_weights[name]
            diff_sq = (p - theta_star).pow(2)
            if use_mask and name in self._anchor_masks:
                mask = self._anchor_masks[name]
                if mask.device != device:
                    mask = mask.to(device)
                    self._anchor_masks[name] = mask
                diff_sq = diff_sq * mask
            if use_ewc and name in self._ewc_fisher:
                F = self._ewc_fisher[name]
                if F.device != device:
                    F = F.to(device)
                    self._ewc_fisher[name] = F
                # EWC: (λ/2) · Σ F · ((θ - θ*)² ⊙ M)
                anchor_loss = anchor_loss + 0.5 * (F * diff_sq).sum()
            else:
                # Plain L2: λ · Σ ((θ - θ*)² ⊙ M)
                anchor_loss = anchor_loss + diff_sq.sum()

        return self._anchor_lambda * anchor_loss

    def set_ewc_fisher(
        self,
        fisher: Dict[str, torch.Tensor],
    ) -> None:
        """
        Enable Elastic Weight Consolidation mode: weight the anchor loss
        by the provided diagonal Fisher Information Matrix.

        Must be called AFTER `set_anchor_weights()`.  The anchor point θ*
        used by the loss comes from `_anchor_weights`, not from any anchor
        field inside the FisherDiagonal — this lets the caller keep θ*
        consistent with the warmstart checkpoint even if Fisher was
        computed separately.

        The λ multiplier continues to come from `_anchor_lambda`, which is
        set by `set_anchor_weights()` / `set_anchor_lambda()`.  EWC typically
        tolerates a larger λ than plain L2 because the Fisher weighting
        makes the penalty selective.

        Parameters
        ----------
        fisher
            Dict mapping parameter name → tensor with the same shape as
            the corresponding q_net parameter.  Typically produced by
            `ewc.estimate_fisher_diagonal_boltzmann(...)` and then passed
            as `fisher_diagonal.fisher` (not the full FisherDiagonal).

        Raises
        ------
        ValueError
            If anchor weights are not set, or if any Fisher tensor has a
            shape mismatch with the corresponding parameter.
        """
        if self._anchor_weights is None:
            raise ValueError(
                "set_ewc_fisher() requires anchor weights to be set first. "
                "Call set_anchor_weights() before set_ewc_fisher()."
            )

        device = next(self.q_net.parameters()).device
        validated: Dict[str, torch.Tensor] = {}
        missing: list = []

        for name, p in self.q_net.named_parameters():
            if name not in self._anchor_weights:
                continue
            if name not in fisher:
                missing.append(name)
                continue
            F = fisher[name]
            if tuple(F.shape) != tuple(p.shape):
                raise ValueError(
                    f"[EWC] Fisher shape mismatch for '{name}': "
                    f"fisher={tuple(F.shape)}, param={tuple(p.shape)}"
                )
            validated[name] = F.detach().to(device)

        if missing:
            print(
                f"[EWC WARNING] Fisher missing for {len(missing)} anchored "
                f"parameters — they will fall back to plain L2 weighting. "
                f"First 5 missing: {missing[:5]}"
            )

        self._ewc_fisher = validated
        _n_with_f = len(validated)
        _total = sum(F.numel() for F in validated.values())
        print(f"[EWC] Fisher attached: {_n_with_f} parameter groups, "
              f"{_total:,} total weights")

    def setup_differential_lr(self, lr_feature: float, lr_head: float) -> None:
        """
        Replace the uniform-LR optimizer with one that uses different
        learning rates for the feature extractor (trunk) and output heads.

        This is a standard fine-tuning technique: keep the trunk (which
        learned good representations during pre-training) stable with a
        low LR, while allowing the heads to adapt faster.

        IMPORTANT: must be called AFTER optimizer.load_state_dict() if
        restoring from a checkpoint, because load_state_dict() overwrites
        the param_groups and their LRs.

        Parameters
        ----------
        lr_feature : float
            Learning rate for self.q_net.feature (the shared trunk).
        lr_head : float
            Learning rate for output heads (fc_out, fc_value, fc_adv).
        """
        # Separate parameters into feature trunk and output heads.
        feature_params = list(self.q_net.feature.parameters())
        feature_ids = {id(p) for p in feature_params}
        head_params = [p for p in self.q_net.parameters() if id(p) not in feature_ids]

        # Preserve Adam state (momentum estimates) from the loaded checkpoint.
        # We save the state_dict, recreate the optimizer with the new param
        # groups, and then restore the per-parameter state entries.
        old_state = self.optimizer.state_dict()

        self.optimizer = AdamW(
            [
                {
                    "params": feature_params,
                    "lr": lr_feature,
                    # Relative to the head LR; used by external LR schedulers
                    # that decay the base LR while preserving differential LR.
                    "lr_ratio": (lr_feature / lr_head) if lr_head > 0 else 1.0,
                },
                {
                    "params": head_params,
                    "lr": lr_head,
                    "lr_ratio": 1.0,
                },
            ],
            weight_decay=self.weight_decay,
        )

        # Restore per-parameter Adam state (exp_avg, exp_avg_sq, step).
        # The state is keyed by parameter index in the old flat list.
        # Since we're using the same parameters (just regrouped), we can
        # map old state entries to the new parameter list.
        try:
            old_param_states = old_state.get("state", {})
            if old_param_states:
                new_params = []
                for group in self.optimizer.param_groups:
                    new_params.extend(group["params"])
                for i, p in enumerate(new_params):
                    if i in old_param_states:
                        self.optimizer.state[p] = old_param_states[i]
        except Exception:
            pass  # If state restoration fails, start fresh (safe fallback)

    def setup_noisy_weight_decay(self, sigma_weight_decay: float = 0.01) -> None:
        """
        Add L2 weight decay ONLY to NoisyLinear sigma parameters.

        This prevents the random-walk growth of sigma observed with C51
        + NoisyNet.  The decay creates a restoring force:

            σ_new = σ * (1 - lr * wd)

        that balances the random-walk step from the noisy gradient.
        Sigma stabilises at an equilibrium where decay = walk step.

        The implementation splits the optimizer into 3 param groups:
          1. Non-sigma params with weight_decay=0 (original behaviour)
          2. Sigma params with weight_decay=sigma_weight_decay

        If differential LR is already active (multiple param groups),
        this method adds sigma decay to each existing group by splitting
        sigma params into a new group with the same LR but added decay.

        Parameters
        ----------
        sigma_weight_decay : float
            L2 decay coefficient for sigma params.  0.01 is a good
            starting point for C51 + NoisyNet.
        """
        if sigma_weight_decay <= 0:
            return

        # Collect sigma and non-sigma params with their current LRs
        new_groups = []
        for group in self.optimizer.param_groups:
            sigma_params = []
            other_params = []
            for p in group["params"]:
                # Check if this param is a sigma by matching against named_parameters
                is_sigma = False
                for name, param in self.q_net.named_parameters():
                    if param is p and "sigma" in name:
                        is_sigma = True
                        break
                if is_sigma:
                    sigma_params.append(p)
                else:
                    other_params.append(p)

            # Keep non-sigma params with original settings
            if other_params:
                new_group = {k: v for k, v in group.items() if k != "params"}
                new_group["params"] = other_params
                new_group["weight_decay"] = 0.0
                new_groups.append(new_group)

            # Sigma params get the same LR but with weight_decay
            if sigma_params:
                sigma_group = {k: v for k, v in group.items() if k != "params"}
                sigma_group["params"] = sigma_params
                sigma_group["weight_decay"] = sigma_weight_decay
                new_groups.append(sigma_group)

        # Rebuild optimizer preserving Adam state (momentum buffers)
        old_state = self.optimizer.state
        self.optimizer = AdamW(new_groups)
        # Restore Adam state for params that existed before
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in old_state:
                    self.optimizer.state[p] = old_state[p]

        n_sigma = sum(1 for g in self.optimizer.param_groups if g["weight_decay"] > 0
                      for _ in g["params"])
        print(f"[NOISY] Sigma weight decay = {sigma_weight_decay} "
              f"({n_sigma} sigma params, {len(self.optimizer.param_groups)} param groups)")

    def save_replay_buffer(self, path: str, max_transitions: int = None) -> int:
        """
        Save replay buffer transitions to a file.

        Parameters
        ----------
        path : str
            File path (.pt).
        max_transitions : int, optional
            Maximum number of transitions to save.  If None, saves all.

        Returns
        -------
        int : number of transitions saved.
        """
        if self.use_prioritized_experience:
            # PER: save the underlying storage
            data = list(self.memory.memory)
        else:
            data = list(self.memory.memory)

        if max_transitions is not None and len(data) > max_transitions:
            # Sample uniformly to keep a representative subset
            import random
            data = random.sample(data, max_transitions)

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        tmp_path = None
        try:
            dirpath = dirname if dirname else "."
            prefix = f".{os.path.basename(path)}."
            fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=dirpath)
            os.close(fd)
            torch.save({"transitions": data, "count": len(data)}, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return len(data)

    def load_replay_buffer(
        self, path: str, max_load: int = None, reset_priorities: bool = True
    ) -> int:
        """
        Load replay buffer transitions from a file and merge into the
        current buffer.

        For PER: priorities of loaded transitions are reset to 1.0 (stale
        priorities from the old training would bias the sampling).

        Parameters
        ----------
        path : str
            File path (.pt).
        max_load : int, optional
            Maximum number of transitions to load.  If None, loads all.
        reset_priorities : bool
            If True and using PER, reset priorities of loaded transitions.

        Returns
        -------
        int : number of transitions loaded.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        data = ckpt["transitions"]

        if max_load is not None and len(data) > max_load:
            import random
            data = random.sample(data, max_load)

        loaded = 0
        for transition in data:
            # Both buffer types use insert(), not push().
            self.memory.insert(transition)
            loaded += 1

        # For PER: if reset_priorities is True, set all restored transitions
        # to priority 1.0 (stale priorities from old training bias sampling).
        # insert() already sets max_priority, which may be > 1.0 from the
        # old training.  Reset to 1.0 for a clean start.
        if self.use_prioritized_experience and reset_priorities:
            for i in range(len(self.memory.priorities)):
                self.memory.priorities[i] = 1.0
            self.memory.max_priority = 1.0

        return loaded

    def set_replay_context(
        self,
        episode_idx: int,
        replay_delta: Optional[float] = None,
    ) -> None:
        """
        Attach episode-level context to newly inserted replay transitions.

        This is diagnostics-only bookkeeping for TensorBoard replay plots.
        """
        self.current_replay_episode_idx = int(episode_idx)
        self.current_replay_delta = (
            float(replay_delta) if replay_delta is not None else float("nan")
        )

    def _accumulate_replay_sample_stats(
        self,
        idxs_np: np.ndarray,
        rewards_b: torch.Tensor,
        td_error_values: Optional[np.ndarray] = None,
        priority_values: Optional[np.ndarray] = None,
    ) -> None:
        """
        Track what replay actually sampled during the current episode.
        """
        self.episode_sample_total += int(len(idxs_np))

        if hasattr(self.memory, "get_episode_indices"):
            sampled_episode_idxs = self.memory.get_episode_indices(idxs_np)
            if self.current_replay_episode_idx >= 0:
                recent_floor = (
                    self.current_replay_episode_idx
                    - self.replay_recent_window_episodes
                    + 1
                )
                recent_mask = (
                    sampled_episode_idxs >= max(recent_floor, 0)
                ) & (sampled_episode_idxs >= 0)
                self.episode_sample_recent_count += int(recent_mask.sum())

        if hasattr(self.memory, "get_delta_values"):
            sampled_deltas = self.memory.get_delta_values(idxs_np)
            finite_mask = np.isfinite(sampled_deltas)
            if finite_mask.any():
                self.episode_sample_delta_sum += float(sampled_deltas[finite_mask].sum())
                self.episode_sample_delta_count += int(finite_mask.sum())

        self.episode_sample_rewards.extend(
            rewards_b.detach().view(-1).cpu().tolist()
        )

        if td_error_values is not None and len(td_error_values) > 0:
            self.episode_sample_td_errors.extend(
                np.asarray(td_error_values, dtype=np.float64).tolist()
            )

        if priority_values is not None and len(priority_values) > 0:
            self.episode_sample_priorities.extend(
                np.asarray(priority_values, dtype=np.float64).tolist()
            )

    # ================================================================
    # The SMDP integration is minimal here: act() only sets a flag
    # (_last_was_decision) that learn() reads to decide whether to
    # commit or accumulate the current transition.
    # ================================================================

    def act(self, mm_state: dict) -> tuple:
            """
            Select and return a MarketMaker command for the current state.

            This method is called EVERY micro-step by the runner. It decides
            whether to query the neural network for a new action (= decision)
            or to hold the current position (= throttled).

            Decision flow:
                1. Update throttle counters (events, TOB, time)
                2. Check BYPASS priorities:
                   - Mode change (inventory crossed a limit) → force decision
                   - Fill replenishment (inventory changed)  → force decision
                3. Check THROTTLE gates (OR logic: any gate blocking → throttled):
                   - Event gate: not enough events since last decision
                   - Time gate: not enough simulated time elapsed
                   - TOB gate: not enough L1 changes
                4. If should_act:
                   - Convert state → tensor → Q-network → action index
                   - Map action index → MarketMaker command tuple
                   - Reset throttle clocks
                   - Set _last_was_decision = True (for SMDP in learn())
                5. If throttled:
                   - Return ("hold",) without touching the network
                   - Set _last_was_decision = False (for SMDP in learn())

            Parameters
            ----------
            mm_state : dict
                State dictionary from MarketMaker.build_state().

            Returns
            -------
            tuple
                MarketMaker command, e.g. ("place_bid_ask", 100, 102) or ("hold",).
            """

            # -----------------------------------------------------------
            # Step 1: Update throttle counters
            # -----------------------------------------------------------
            current_time = float(mm_state.get("time", 0.0))
            inv = float(mm_state.get("inventory", 0.0))
            
            # Increment event counter if enabled
            if self.use_event_update:
                self.event_steps += 1
                
            # Update TOB signature and check for changes
            tob_key = self._get_env_tob(mm_state)
            if self.use_tob_update:
                if self.last_env_tob_key is None:
                    self.last_env_tob_key = tob_key
                elif self.last_env_tob_key != tob_key:
                    self.moves_tob += 1
                    self.last_env_tob_key = tob_key

            # -----------------------------------------------------------
            # Step 2: Check Bypass Priority A — Mode Change
            # -----------------------------------------------------------
            # Determine desired quoting mode based on inventory limits.
            if self.inv_limit is not None:
                if inv >= self.inv_limit: desired_mode = "ask_only"
                elif inv <= -self.inv_limit: desired_mode = "bid_only"
                else: desired_mode = "two_sided"
            else:
                desired_mode = "two_sided"

            # Detect mode change (requires immediate action to adjust orders)
            mode_changed = (self.last_mode is not None) and (self.last_mode != desired_mode)
            self.last_mode = desired_mode

            # -----------------------------------------------------------
            # Step 3: Check Bypass Priority B — Fill Replenishment
            # -----------------------------------------------------------
            # If inventory changed since the last decision, a fill happened.
            # We need to replace the filled order ASAP (before next market event).
            has_fill = False
            if self.last_inventory is not None:
                if abs(float(inv) - float(self.last_inventory)) > 0.01:
                    has_fill = True
            self.last_inventory = float(inv)

            # -----------------------------------------------------------
            # Step 4: Decision logic — should we query the network?
            # -----------------------------------------------------------
            # In MDP mode (use_mdp=True), bypasses A and B are disabled.
            # The agent strictly respects the throttle gates.  With event
            # gating, K is exactly fixed; with time/TOB gating, K varies
            # but the decision rule is still gate-only (no bypass).
            # -----------------------------------------------------------
            should_act = False

            # Priority A: Safety / Mode Change (Bypass Throttle)
            # Disabled in MDP mode — agent waits for throttle even at limit crossings.
            if mode_changed and not self.use_mdp:
                should_act = True

            # Priority B: Fill Replenishment (Bypass Throttle)
            # Disabled in MDP mode — agent waits for throttle even after fills.
            elif has_fill and not self.use_mdp:
                should_act = True

            # Priority C: Throttle Check (Normal Operation)
            else:
                is_throttled = False
                
                # -------------------------------------------------
                # First-action bypass (applies to ALL gates).
                # Before the agent has acted once this episode,
                # every gate is satisfied so initial quotes are
                # placed immediately.
                #
                # BUG FIX (P0 regression): Previously used
                # last_update_time < 0 as sentinel, which stayed
                # -1 forever when use_time_update=False — making
                # Event/TOB gates permanently bypassed.  Now uses
                # the dedicated _has_acted_once flag.
                # -------------------------------------------------
                if not self._has_acted_once:
                    pass  # Bypass all gates on first action
                else:
                    # Check Event Count
                    if self.use_event_update:
                        if self.event_steps < self.threshold_events:
                            is_throttled = True

                    # Check Time Interval
                    if self.use_time_update:
                        if self.last_update_time < 0:
                            pass  # No baseline yet (should not happen)
                        elif (current_time - self.last_update_time) < self.min_time_interval:
                            is_throttled = True

                    # Check TOB Moves
                    if self.use_tob_update:
                        if self.moves_tob < self.threshold_tob:
                            is_throttled = True
                
                # Only act if NOT throttled
                if not is_throttled:
                    should_act = True

            # -----------------------------------------------------------
            # Step 5: Execute decision or hold
            # -----------------------------------------------------------
            
            if should_act:
                # === NEW DECISION (Query Network) ===

                # 1. State to Tensor
                state_tensor = self._state_to_tensor(mm_state)

                # 2. Compute Valid Action Mask (Constrained Selection)
                # ---------------------------------------------------
                # Build a boolean mask that marks inventory-violating
                # actions as False.  This mask is applied inside
                # _epsilon_greedy_action to ensure the network only
                # picks from PERMITTED actions — both in the greedy
                # branch (Q[invalid] = -inf) and the ε-random branch
                # (sample from valid subset only).
                valid_mask = self._get_valid_action_mask(mm_state)

                # 3. Select Action (Constrained)
                # The returned a_idx is guaranteed to be a valid action
                # given the current inventory state.
                a_idx = self._epsilon_greedy_action(state_tensor, valid_mask=valid_mask)

                # 4. Map Action → MarketMaker Command Tuple
                # Safety nets in _mm_action_from_idx still exist as a
                # defensive fallback (zombie cleanup, price validation)
                # but should rarely override now that selection is
                # constrained.
                act_tuple = self._mm_action_from_idx(a_idx, mm_state)

                # 5. Store Action for Learning
                # The mask in step 2 already ensures that in pure_mm
                # at inv_limit, only canonical representatives (one per
                # recovery-side offset) are selectable.  No post-hoc
                # remap is needed — a_idx is canonical by construction.
                self.last_action_idx = a_idx
                self.action_counts[a_idx] += 1
                self.action_inv_sum[a_idx] += inv
                self.action_inv_abs_sum[a_idx] += abs(inv)

                # 5b. Accumulate Q-values for all actions (TensorBoard diagnostics).
                # Uses the SAME state_tensor already computed above — zero extra cost.
                with torch.no_grad():
                    q_all = self._q_values_from_net(self.q_net, state_tensor)  # (1, n_actions)
                    q_np = q_all.cpu().numpy().ravel()
                    self.episode_q_sum += q_np
                    self.episode_q_count += 1
                    # Track Q-value range (only valid actions, ignoring -inf masked ones)
                    if valid_mask is not None:
                        q_valid = q_np[valid_mask]
                    else:
                        q_valid = q_np
                    if len(q_valid) > 0:
                        self.episode_q_min = min(self.episode_q_min, float(q_valid.min()))
                        self.episode_q_max = max(self.episode_q_max, float(q_valid.max()))
                
                # 6. Inject RL state metadata into mm_state for logging.
                #    MarketMaker.log_step() reads these from last_policy_state
                #    to populate MM_RL_State_* columns in mm_df, which are used
                #    by the interpretability plots at the end of training.
                raw_vec = state_tensor.detach().cpu().numpy().ravel().tolist()
                mm_state["RL_State_Vector"] = raw_vec
                mm_state["RL_State_Mode"] = "pure_mm" if self.pure_mm else "generic"
                mm_state["RL_State_Dim"] = len(raw_vec)
                if self.pure_mm:
                    mm_state["RL_State_MaxOffset"] = int(self.max_offset)

                # 7. Reset Throttling Clocks (Action taken)
                self._reset_clocks(current_time, tob_key)

                # 8. SMDP flag: signal to learn() that this was a true decision.
                #    learn() will commit any pending SMDP transition and start
                #    a new one with the state/action from this decision.
                self._last_was_decision = True

                return act_tuple

            else:
                # === THROTTLED (Hold Position) ===
                # Return "hold" to indicate no changes to orders.
                # SMDP flag: signal to learn() that this was a throttled step.
                # learn() will accumulate the reward into the pending SMDP
                # transition instead of creating a new one.
                self._last_was_decision = False
                return ("hold",)

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
        SMDP-aware learning step.

        OVERVIEW
        --------
        The runner calls this method EVERY micro-step, regardless of whether
        the controller was throttled or made a real decision. This method
        implements Semi-Markov Decision Process (SMDP) reward aggregation
        to produce correct transitions for the replay buffer.

        THE PROBLEM (without SMDP):
            When throttled, act() returns ("hold",) without updating
            last_action_idx. The runner still calls learn() with the
            stale action index, creating replay transitions where the
            (state, action) pair does not correspond to any real decision.
            This corrupts the Q-function approximation.

        THE SOLUTION (SMDP aggregation):
            Instead of creating one replay transition per micro-step, we
            aggregate all micro-steps between two consecutive decisions
            into a SINGLE SMDP transition:

                (s_decision, a_decision, R_cum, done, s_next_decision)

            where R_cum = Σ_{k=0}^{K-1} γ^k r_{t+k} is the discounted
            cumulative reward over the K micro-steps of the holding period.

        BACKWARD COMPATIBILITY:
            When throttling is disabled, every step is a decision (K=1),
            and this method produces identical transitions to the original
            code.

        STAGES
        ------
        Stage 1: SMDP aggregation (decision vs throttled micro-step)
        Stage 2: n-step buffer management (build multi-step returns)
        Stage 3: Gradient step (if enough replay data)
        Stage 4: Episode cleanup

        IMPORTANT:
        - Epsilon is NOT decayed here; it is decayed once per episode
          externally, so that it stays constant within each episode.
        """
        if self.last_action_idx is None:
            # No action chosen yet (e.g., very first step) -> nothing to learn.
            return

        done_flag = bool(info.get("done", False))

        # ===== EVAL MODE: freeze learning (no replay, no grads, no updates) =====
        if not self.enable_learning:
            if done_flag:
                self.last_action_idx = None
                self.nstep_buffer.clear()
                self._smdp_reset()
                # Reset throttle state so the next episode starts clean
                # (mirrors the train-mode done branch and
                # reset_controller_for_episode in tune_dqn_optuna.py).
                self._has_acted_once = False
                self.moves_tob = 0
                self.last_env_tob_key = None
                self.event_steps = 0
                self.last_update_time = -1.0
                self.last_inventory = None
                self.last_mode = None
            return

        # =================================================================
        # STAGE 1: SMDP AGGREGATION
        # =================================================================
        # This stage decides whether to commit the pending SMDP transition
        # or to accumulate the current reward into it.
        #
        # There are two cases:
        #
        # CASE A — This step was a DECISION (act() queried the network):
        #   1. If there is a pending SMDP transition from a previous
        #      decision, commit it now. Its s_next is state_before of
        #      THIS decision (the state the agent sees when re-deciding).
        #   2. Start a NEW pending SMDP transition with:
        #      - s_decision = state_before (state at decision time)
        #      - a_decision = last_action_idx (action chosen)
        #      - R_cum = reward (first micro-step reward)
        #      - k_steps = 1
        #
        # CASE B — This step was THROTTLED (act() returned "hold"):
        #   1. Accumulate the reward into the pending transition:
        #      R_cum += γ^k * reward
        #      k_steps += 1
        #   2. Do NOT commit anything (we are still in the same
        #      SMDP holding period).
        #
        # SPECIAL CASE — Episode termination (done=True):
        #   After handling Case A or B above, if done=True, we
        #   ALSO commit the pending transition with s_next = state_after
        #   and done=True. This ensures the terminal transition is
        #   always captured, even if the episode ends during throttle.
        # =================================================================

        if self._last_was_decision:
            # ----- CASE A: New decision arrived -----

            # Step A.1: Commit any previous pending SMDP transition.
            # The "next state" for the old transition is state_before of
            # THIS step (the state at the moment of the new decision).
            if self._smdp_pending:
                self._smdp_commit_to_nstep(state_before, done=False)

            # Step A.2: Start a new pending SMDP transition.
            self._smdp_pending = True
            self._smdp_s_decision = state_before
            self._smdp_a_decision = self.last_action_idx
            self._smdp_cum_reward = float(reward)
            self._smdp_k_steps = 1

        else:
            # ----- CASE B: Throttled micro-step -----
            # Accumulate the reward inside the pending SMDP decision.
            #
            # If use_intra_event_gamma is True (default, paper convention),
            # apply γ^k discounting: the k-th micro-step reward is
            # discounted by γ^k relative to the decision time, so the
            # committed pending reward equals r_t = Σ_n γ^n r_{t,n}.
            #
            # If use_intra_event_gamma is False, sum rewards undiscounted
            # (γ = 1 intra-event), so r_t = Σ_n r_{t,n}. The OUTER
            # bootstrap discount γ^{N_t} on Q(s_{t+1}, a') is unchanged
            # in both modes.
            if self._smdp_pending:
                k = self._smdp_k_steps
                if self.use_intra_event_gamma:
                    self._smdp_cum_reward += (self.gamma ** k) * float(reward)
                else:
                    self._smdp_cum_reward += float(reward)
                self._smdp_k_steps += 1

        # ----- SPECIAL CASE: Episode termination -----
        # If done=True, we must commit the pending transition NOW with
        # s_next = state_after (the terminal state) and done=True.
        # This can happen in BOTH Case A and Case B:
        #   - Case A + done: The new decision is also the last step.
        #     We just started the pending transition above, so commit it
        #     immediately with the terminal state.
        #   - Case B + done: The episode ends during a throttled period.
        #     Commit the accumulated transition.
        if done_flag and self._smdp_pending:
            self._smdp_commit_to_nstep(state_after, done=True)

        # =================================================================
        # STAGE 2: N-STEP BUFFER MANAGEMENT
        # =================================================================
        # The n-step buffer now contains SMDP-level transitions (one per
        # decision), each with a variable number of micro-steps (k_steps).
        #
        # We build n-step returns by combining `self.n_steps` SMDP
        # transitions from the buffer. The key difference from standard
        # n-step: the discount between SMDP transitions is NOT a fixed γ,
        # but γ^{k_i} where k_i is the number of micro-steps in the i-th
        # SMDP transition.
        #
        # Example with n_steps=3 and SMDP transitions with k=[2, 3, 1]:
        #   G = R_0 + γ^2 * R_1 + γ^{2+3} * R_2
        #   gamma_eff = γ^{2+3+1} = γ^6  (total micro-steps for bootstrap)
        #
        # When throttling is off, k_i = 1 for all transitions, so:
        #   G = R_0 + γ * R_1 + γ^2 * R_2
        #   gamma_eff = γ^3
        # which is identical to the original fixed n-step behavior.
        #
        # The transition stored in replay is now a 6-tuple:
        #   [s_root, a_root, G, done_final, s_next_final, gamma_eff]
        # where gamma_eff is the effective discount for bootstrapping.
        # =================================================================

        def commit_nstep_transition():
            """
            Build and insert one n-step transition from the oldest
            SMDP-level elements in the buffer.

            For a chunk of L SMDP transitions (L <= n_steps):

              G = Σ_{i=0}^{L-1}  (Π_{j=0}^{i-1} γ^{k_j}) * R_i
              gamma_eff = Π_{i=0}^{L-1} γ^{k_i}  = γ^{Σ k_i}
              done_final = done flag of the last element
              s_next_final = s_{t+L} from the last element

            When done_final is True, bootstrapping is skipped in
            _train_step(), so gamma_eff is irrelevant for terminal
            transitions (but we compute it anyway for consistency).
            """
            chunk_len = min(self.n_steps, len(self.nstep_buffer))
            chunk = list(self.nstep_buffer)[:chunk_len]

            # Root state and action (from the FIRST SMDP transition)
            s_root, a_root, _, _, _, _ = chunk[0]

            # Compute the n-step return G with variable discounting.
            # `discount` tracks γ^{Σ_{j=0}^{i-1} k_j}, i.e., the
            # cumulative discount from the root state to the start
            # of the i-th SMDP transition.
            G = 0.0
            discount = 1.0
            total_micro_steps = 0

            for (_, _, r_i, _, _, k_i) in chunk:
                r_scalar = float(r_i.item())
                G += discount * r_scalar
                # Advance the discount by γ^{k_i} micro-steps
                discount *= (self.gamma ** k_i)
                total_micro_steps += k_i

            # gamma_eff = γ^{total_micro_steps} is the effective discount
            # factor for bootstrapping Q(s_next_final, a') in _train_step().
            gamma_eff = self.gamma ** total_micro_steps

            # Terminal state and done flag from the LAST element
            _, _, _, done_last, s_next_last, _ = chunk[-1]
            done_final = bool(done_last.item())
            s_next_final = s_next_last

            if self.use_g_clip:
                G = max(self.dist_v_min, min(self.dist_v_max, G))

            # Pack into tensors
            G_tensor = torch.tensor(
                [[G]], dtype=torch.float32, device=self.device
            )
            done_final_tensor = torch.tensor(
                [[done_final]], dtype=torch.bool, device=self.device
            )
            gamma_eff_tensor = torch.tensor(
                [[gamma_eff]], dtype=torch.float32, device=self.device
            )

            # Build the 6-field transition and move to CPU to avoid GPU OOM
            # in the large replay buffer.
            raw_transition = [
                s_root, a_root, G_tensor,
                done_final_tensor, s_next_final, gamma_eff_tensor,
            ]
            cpu_transition = [t.detach().cpu() for t in raw_transition]

            # Insert into replay memory:
            #   [s_root, a_root, G^{(n)}, done_final, s_next_final, gamma_eff]
            self.memory.insert(
                cpu_transition,
                metadata={
                    "episode_idx": self.current_replay_episode_idx,
                    "delta": self.current_replay_delta,
                },
            )

            # Remove the oldest element (slide the window forward)
            self.nstep_buffer.popleft()

        # --- Commit logic ---
        if not done_flag:
            # Normal step: commit one n-step transition when buffer is full.
            if len(self.nstep_buffer) >= self.n_steps:
                commit_nstep_transition()
        else:
            # Episode ended: flush ALL remaining transitions.
            # Tail transitions have shorter horizons (< n_steps) and
            # always end with done=True, so no bootstrapping occurs.
            while len(self.nstep_buffer) > 0:
                commit_nstep_transition()

        # =================================================================
        # STAGE 3: GRADIENT STEP
        # =================================================================
        # BUG FIX: Previously, _train_step() fired on EVERY micro-step,
        # regardless of whether a new SMDP decision had been made.  With
        # throttling enabled (e.g. 60 s time gate), an episode of 10 000
        # micro-steps produces only ~41 new SMDP transitions but triggered
        # ~10 000 gradient updates — a 244:1 SGD-to-data ratio that causes
        # massive overtraining and Q-value divergence.
        #
        # Fix: only perform a gradient step when this micro-step was an
        # actual decision (i.e. act() queried the network).  This aligns
        # the number of gradient updates with the number of new data
        # points, restoring a healthy ~1:1 ratio per episode.
        #
        # On episode termination (done_flag), we also allow a gradient
        # step so the terminal transition is immediately trained on.
        if (self._last_was_decision or done_flag) and self.memory.can_sample(self.batch_size):
            self._train_step()

        # =================================================================
        # STAGE 4: EPISODE CLEANUP
        # =================================================================
        if done_flag:
            self.last_action_idx = None
            self.nstep_buffer.clear()
            self._smdp_reset()
            # Reset all throttle / bypass state so the next episode starts
            # clean.  Without this, _has_acted_once stays True from episode 1
            # onward and event/TOB gates could block the first action.
            self._has_acted_once = False
            self.moves_tob = 0
            self.last_env_tob_key = None
            self.event_steps = 0
            self.last_update_time = -1.0
            self.last_inventory = None
            self.last_mode = None

        # NOTE: epsilon decay is done externally, once per episode.

    # ================================================================
    # _train_step() — ONE GRADIENT UPDATE FROM REPLAY
    # ================================================================
    #
    # This is where the actual neural network learning happens. Called
    # from learn() whenever the replay buffer has enough samples.
    #
    # The flow is:
    #   1. Sample a mini-batch from replay (uniform or PER)
    #   2. Compute current Q(s, a) from the online network
    #   3. Compute the TD target (algorithm-dependent)
    #   4. Compute loss (MSE / Huber for scalar, cross-entropy for C51)
    #   5. Backpropagate and update weights
    #   6. Periodically hard-update the target network
    #
    # All 4 algorithm combinations are supported:
    #   ┌────────────┬──────────────────────┬────────────────────────┐
    #   │            │ use_double=False      │ use_double=True         │
    #   ├────────────┼──────────────────────┼────────────────────────┤
    #   │ use_sarsa  │ Deep SARSA           │ Double Deep SARSA      │
    #   │ =True      │ a'~π(target_net)     │ a'~π(q_net),           │
    #   │            │ Q_target(s', a')     │ Q_target(s', a')       │
    #   ├────────────┼──────────────────────┼────────────────────────┤
    #   │ use_sarsa  │ Vanilla DQN          │ Double DQN             │
    #   │ =False     │ max_a' Q_target(s',a')│ a*=argmax Q_online,   │
    #   │            │                      │ Q_target(s', a*)       │
    #   └────────────┴──────────────────────┴────────────────────────┘
    # ================================================================

    def _get_target_valid_mask_batch(
        self, next_states_b: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Build a (B, n_actions) boolean mask for target Q-value computation.

        When computing the TD target (max_a Q(s', a) or argmax), we should
        only consider actions that are VALID in the next state s'.  This
        prevents overestimation bias from equivalent/invalid actions:
          - In PURE MM mode at inv_limit, non-canonical duplicate actions
            would artificially inflate the max (the max of N noisy copies
            of the same value > the true value).
          - In GENERIC mode at inv_limit, invalid actions (e.g., post_bid
            when long) should not participate in the max.

        The inventory position is extracted directly from the state tensor
        (index 1 for pure_mm, index 3 for generic; both normalized by
        inv_limit, so ±1.0 indicates the limit).

        Returns None when no masking is needed (no samples at inv_limit),
        allowing the caller to skip the masking logic entirely.

        Parameters
        ----------
        next_states_b : (B, D) float tensor — next-state batch

        Returns
        -------
        mask : (B, n_actions) bool tensor, or None if all actions valid
        """
        if self.inv_limit is None:
            return None

        B = next_states_b.size(0)

        # Extract normalized inventory from the state tensor
        if self.pure_mm:
            inv_norm = next_states_b[:, 1]   # pure_mm: [spread, inv, ...]
        else:
            inv_norm = next_states_b[:, 3]   # generic: [spread, asksize, bidsize, inv, ...]

        # Detect which samples are at an inventory limit
        # (small epsilon for floating-point tolerance)
        at_long  = (inv_norm >=  1.0 - 1e-6)   # (B,)
        at_short = (inv_norm <= -1.0 + 1e-6)   # (B,)

        if not (at_long.any() or at_short.any()):
            return None  # fast path: no samples at limit

        # Start with all True (every action valid)
        mask = torch.ones(B, self.n_actions, dtype=torch.bool,
                          device=next_states_b.device)

        if self.pure_mm:
            # Use canonical masks: only canonical representatives are valid
            long_mask_1d = torch.tensor(
                self._canonical_mask_long, dtype=torch.bool,
                device=next_states_b.device,
            )
            short_mask_1d = torch.tensor(
                self._canonical_mask_short, dtype=torch.bool,
                device=next_states_b.device,
            )
            if at_long.any():
                mask[at_long] = long_mask_1d.unsqueeze(0)
            if at_short.any():
                mask[at_short] = short_mask_1d.unsqueeze(0)
        else:
            # Generic mode: same rules as _get_valid_action_mask
            long_mask_1d = torch.ones(self.n_actions, dtype=torch.bool,
                                      device=next_states_b.device)
            long_mask_1d[0] = False   # post_bid (would worsen)
            long_mask_1d[2] = False   # post_bid_ask (includes bid)
            long_mask_1d[3] = False   # cancel_bid (no-op: bid consumed by fill)
            long_mask_1d[4] = False   # cancel_ask (would remove recovery)
            if self.n_actions >= 7:
                long_mask_1d[6] = False   # post_bid_ask_inside_spread
            if self.n_actions >= 8:
                long_mask_1d[7] = False   # post_bid_inside_spread

            short_mask_1d = torch.ones(self.n_actions, dtype=torch.bool,
                                       device=next_states_b.device)
            short_mask_1d[1] = False  # post_ask (would worsen)
            short_mask_1d[2] = False  # post_bid_ask (includes ask)
            short_mask_1d[3] = False  # cancel_bid (would remove recovery)
            short_mask_1d[4] = False  # cancel_ask (no-op: ask consumed by fill)
            if self.n_actions >= 7:
                short_mask_1d[6] = False  # post_bid_ask_inside_spread
            if self.n_actions >= 9:
                short_mask_1d[8] = False  # post_ask_inside_spread

            if at_long.any():
                mask[at_long] = long_mask_1d.unsqueeze(0)
            if at_short.any():
                mask[at_short] = short_mask_1d.unsqueeze(0)

        return mask

    def _train_step(self) -> None:
        """
        Perform one gradient update using a mini-batch from replay memory.

        SMDP-Aware n-step TD Targets
        ----------------------------
        Each replay transition stores 6 fields:
            [s_root, a_root, G, done_final, s_next_final, gamma_eff]

        The TD target formula is:
            target = G + (1 - done) * gamma_eff * Q(s_next, a')

        where:
          - G is the n-step discounted return (already computed in learn())
          - gamma_eff = γ^{Σ k_i} is the per-sample effective discount
            (accounts for variable holding periods under SMDP/throttling)
          - Q(s_next, a') comes from the target network

        When throttling is off, k_i = 1 for all transitions, so
        gamma_eff = γ^n and this is identical to standard n-step DQN.

        For Distributional DQN (C51):
            Instead of scalar Q-values, we project the Bellman-updated
            distribution onto the fixed support and minimize cross-entropy.
        """

        if not self.enable_learning:
            return

        # 1) Sample from replay buffer (uniform or prioritized).
        #    Transitions now have 6 fields: [s, a, G, done, s_next, gamma_eff].
        #    PER adds 2 extra fields: [idxs, weights].
        if self.use_prioritized_experience:
            (
                states_b,
                actions_b,
                rewards_b,
                dones_b,
                next_states_b,
                gamma_effs_b,
                idxs_b,
                weights_b,
            ) = self.memory.sample(self.batch_size)
        else:
            (
                states_b,
                actions_b,
                rewards_b,
                dones_b,
                next_states_b,
                gamma_effs_b,
                idxs_b,
            ) = self.memory.sample(self.batch_size)
            weights_b = None  # not used in uniform case

        # Move tensors to the correct device
        states_b = states_b.to(self.device)         # (B, D)
        actions_b = actions_b.to(self.device)       # (B, 1)
        rewards_b = rewards_b.to(self.device)       # (B, 1)  (SMDP n-step returns)
        dones_b = dones_b.to(self.device)           # (B, 1)
        next_states_b = next_states_b.to(self.device)  # (B, D)
        gamma_effs_b = gamma_effs_b.to(self.device) # (B, 1)  per-sample discount
        if self.use_prioritized_experience:
            weights_b = weights_b.to(self.device)   # (B, 1)
            sampled_priorities_np = self.memory.get_priorities(
                idxs_b.view(-1).cpu().numpy()
            )
        else:
            sampled_priorities_np = None

        # ---------------------------------------------------------------
        # BRANCH A: Distributional DQN (C51) training step
        # ---------------------------------------------------------------
        # C51 learns the FULL distribution of returns, not just E[Q(s,a)].
        #
        # Key idea: The Q-network outputs a probability distribution over
        # `atoms` support points for each action. Training minimizes the
        # KL-divergence (cross-entropy) between:
        #   - The predicted distribution p(s, a) (online network)
        #   - The projected Bellman target distribution m (from target network)
        #
        # The "projection" step is needed because the Bellman update
        #   Tz = r + γ * z  (for each support atom z)
        # shifts the support atoms to new positions that don't align with
        # the original fixed grid. We redistribute the probability mass
        # back onto the nearest grid atoms using linear interpolation.
        # ---------------------------------------------------------------
        if self.use_distributional:
            batch_size = states_b.size(0)

            # (B, A, N) probs and logits — use F.log_softmax on raw logits
            # instead of log(clamp(softmax(…))). This avoids dead gradients
            # in distribution tails where softmax → 0 and clamp freezes grad.
            q_value_probs, q_logits = self.q_net(states_b, return_logits=True)
            actions_flat = actions_b.view(-1)  # (B,)
            action_value_probs = q_value_probs[
                torch.arange(batch_size, device=self.device), actions_flat, :
            ]  # (B, N)
            action_logits = q_logits[torch.arange(batch_size, device=self.device), actions_flat, :]  # (B, N)
            log_action_value_probs = F.log_softmax(action_logits, dim=-1)  # (B, N)

            # SMDP: use per-sample effective discount for bootstrapping.
            # gamma_effs_b is (B, 1); we need it for the C51 projection
            # which broadcasts over the N atoms dimension.
            gamma_n = gamma_effs_b  # (B, 1) — per-sample γ^{Σ k_i}

            # 2) Build Distributional TD target (projected categorical distribution)
            with torch.no_grad():
                if self.use_sarsa:
                    # ON-POLICY: SARSA / Double SARSA
                    # Build per-sample validity mask so SARSA targets
                    # only bootstrap from actions that the online policy
                    # could actually select at s'.
                    sarsa_mask = self._get_target_valid_mask_batch(next_states_b)
                    if self.use_double:
                        # Action selection from the online network
                        next_actions_b = self._epsilon_greedy_action_batch_from_net(
                            self.q_net, next_states_b, valid_mask=sarsa_mask)
                    else:
                        # Vanilla SARSA: action selection from the target network
                        next_actions_b = self._epsilon_greedy_action_batch_from_net(
                            self.target_net, next_states_b, valid_mask=sarsa_mask)

                    # Evaluate distribution using target_net
                    next_q_value_probs = self.target_net(next_states_b)  # (B, A, N)
                    next_actions_flat = next_actions_b.view(-1)
                    next_action_value_probs = next_q_value_probs[
                        torch.arange(batch_size, device=self.device), next_actions_flat, :
                    ]  # (B, N)

                else:
                    # OFF-POLICY: DQN / Double DQN
                    # Build target mask to exclude invalid/duplicate actions
                    # in s' (inventory-limit aware).
                    _target_mask = self._get_target_valid_mask_batch(next_states_b)

                    if self.use_double:
                        # Double DQN: action selection using online expected Q
                        next_q_online = self._q_values_from_net(self.q_net, next_states_b)  # (B, A)
                        if _target_mask is not None:
                            next_q_online = next_q_online.clone()
                            next_q_online[~_target_mask] = float('-inf')
                        next_actions_flat = torch.argmax(next_q_online, dim=1)  # (B,)
                    else:
                        # Vanilla DQN: action selection using target expected Q
                        next_q_target = self._q_values_from_net(self.target_net, next_states_b)  # (B, A)
                        if _target_mask is not None:
                            next_q_target = next_q_target.clone()
                            next_q_target[~_target_mask] = float('-inf')
                        next_actions_flat = torch.argmax(next_q_target, dim=1)  # (B,)

                    # Evaluate distribution using target_net
                    next_q_value_probs = self.target_net(next_states_b)  # (B, A, N)
                    next_action_value_probs = next_q_value_probs[
                        torch.arange(batch_size, device=self.device), next_actions_flat, :
                    ]  # (B, N)

            # =============================================================
            # C51 CATEGORICAL PROJECTION
            # =============================================================
            # The Bellman update shifts each support atom z_j:
            #     Tz_j = r + γ_eff * z_j      (for non-terminal states)
            #     Tz_j = r                      (for terminal states)
            #
            # These shifted atoms Tz_j typically land BETWEEN the original
            # support grid points. We redistribute the probability mass from
            # the target distribution onto the nearest lower (l) and upper (u)
            # grid atoms using linear interpolation weights.
            # =============================================================
            support = self.support.view(1, -1)  # (1, N) — the fixed atom positions
            Tz = rewards_b + (~dones_b) * gamma_n * support  # (B, N) — shifted atoms

            # --- TD target diagnostics (before clipping) ---
            # Scalar TD target = r + (1-done) * γ_eff * Q_target(s', a*)
            # where Q_target = expectation of target distribution = Σ p_j * z_j
            with torch.no_grad():
                q_next_scalar = (next_action_value_probs * support).sum(dim=-1, keepdim=True)  # (B,1)
                scalar_td_target = rewards_b + (~dones_b) * gamma_n * q_next_scalar  # (B,1)
                q_pred_scalar = (action_value_probs * support).sum(dim=-1, keepdim=True)  # (B,1)
                scalar_td_error_proxy = (q_pred_scalar - scalar_td_target).abs().view(-1).cpu().numpy()
                td_np = scalar_td_target.cpu().numpy().ravel()
                self.episode_td_target_min = min(self.episode_td_target_min, float(td_np.min()))
                self.episode_td_target_max = max(self.episode_td_target_max, float(td_np.max()))
                self.episode_td_target_sum += float(td_np.sum())
                self.episode_td_target_count += len(td_np)
                # Fraction of Tz atoms that fall outside [V_min, V_max]
                clip_frac = float(((Tz < self.dist_v_min) | (Tz > self.dist_v_max)).float().mean().item())
                self.episode_tz_clip_frac_sum += clip_frac
                self.episode_tz_clip_count += 1

            Tz = Tz.clamp(min=self.dist_v_min, max=self.dist_v_max)  # keep within support range

            # Convert Tz to fractional atom indices (0-indexed into the support)
            b = (Tz - self.dist_v_min) / self.delta  # (B, N) — fractional indices
            # Lower and upper neighbor atoms for each shifted Tz value
            l = b.floor().long().clamp(0, self.dist_atoms - 1)  # (B, N) — nearest lower atom index
            u = b.ceil().long().clamp(0, self.dist_atoms - 1)   # (B, N) — nearest upper atom index

            # Linear interpolation weights:
            #   If Tz lands at fractional position b between atoms l and u:
            #     w_l = (u - b) of the mass goes to atom l (closer to l → more mass)
            #     w_u = (b - l) of the mass goes to atom u
            #   Edge case: when b is exactly an integer, l == u, so we put all mass on l.
            eq_mask = (u == l)
            w_l = (u.float() - b)
            w_u = (b - l.float())
            w_l = torch.where(eq_mask, torch.ones_like(w_l), w_l)   # all mass to l when l == u
            w_u = torch.where(eq_mask, torch.zeros_like(w_u), w_u)  # no mass to u when l == u

            # Build the projected target distribution m(s', a') by scattering
            # the target probabilities onto the fixed support grid.
            m = torch.zeros(batch_size, self.dist_atoms, device=self.device)  # (B, N) — starts empty

            # Flatten to 1D for index_add_ (batch-dimension offset trick)
            offset = torch.arange(batch_size, device=self.device).view(-1, 1) * self.dist_atoms  # (B, 1)
            l_idx = (l + offset).view(-1)   # flattened indices into m
            u_idx = (u + offset).view(-1)

            m_flat = m.view(-1)
            # Scatter weighted target probabilities onto lower and upper atoms
            m_flat.index_add_(0, l_idx, (next_action_value_probs * w_l).view(-1))
            m_flat.index_add_(0, u_idx, (next_action_value_probs * w_u).view(-1))
            m = m_flat.view(batch_size, self.dist_atoms)

            # Re-normalize for numerical stability (rounding errors can make sum != 1)
            m = m / m.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # Cross-entropy loss: -Σ_j m_j * log(p_j) for each sample in the batch
            # This is the KL-divergence between target distribution m and predicted p.
            cross_entropies = - (m * log_action_value_probs).sum(dim=1)  # (B,)

            # PER: update priorities using KL divergence (not raw cross-entropy).
            # H(m,p) = H(m) + KL(m||p).  H(m) is constant w.r.t. the network
            # and inflates all priorities equally, reducing discrimination.
            # Subtracting it gives KL(m||p), the true "surprise" signal.
            if self.use_prioritized_experience:
                with torch.no_grad():
                    target_entropy = -(m * torch.log(m.clamp(min=1e-8))).sum(dim=1)  # H(m)
                kl_divs = (cross_entropies - target_entropy).clamp(min=0.0, max=1e6)
                prios_np = kl_divs.detach().cpu().numpy()
                self._accumulate_replay_sample_stats(
                    idxs_np=idxs_b.view(-1).cpu().numpy(),
                    rewards_b=rewards_b,
                    td_error_values=scalar_td_error_proxy,
                    priority_values=sampled_priorities_np,
                )
                if self.use_per_priority_clip:
                    prios_np = self._clip_per_priorities(prios_np)
                idxs_np = idxs_b.detach().cpu().numpy()
                self.memory.update_priorities(idxs_np, prios_np)
                self.episode_max_priority = max(
                    self.episode_max_priority, float(prios_np.max()))

                loss = (weights_b.view(-1) * cross_entropies).mean()
            else:
                self._accumulate_replay_sample_stats(
                    idxs_np=idxs_b.view(-1).cpu().numpy(),
                    rewards_b=rewards_b,
                    td_error_values=scalar_td_error_proxy,
                )
                loss = cross_entropies.mean()

            # Track extreme-reward transitions in this batch
            _extreme_thresh = -80.0
            self.episode_extreme_reward_count += int(
                (rewards_b.view(-1) < _extreme_thresh).sum().item())
            self.episode_batch_total += int(rewards_b.numel())

            # L2 anchor regularization (continual RL — prevents drift from
            # pre-trained baseline during fine-tuning).
            anchor_loss = self._compute_anchor_loss()
            total_loss = loss + anchor_loss

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            # Optional gradient clipping for stability:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Episode-level loss accumulation (use TD loss only, not anchor)
            loss_val = float(loss.item())
            self.episode_loss_sum += loss_val
            self.episode_loss_count += 1

            # Global gradient step counter
            self.global_grad_step += 1

            # Periodic hard update of target network
            self.train_steps += 1
            if self.train_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            return

        # ---------------------------------------------------------------
        # BRANCH B: Standard (scalar) DQN / SARSA training step
        # ---------------------------------------------------------------
        # This branch handles the classic case where Q(s,a) is a scalar.
        # Loss is smooth L1 (Huber) for uniform replay, or IS-weighted MSE
        # for PER.
        # ---------------------------------------------------------------

        # Q(s, a) for the actions actually taken in each transition
        qsa_b = self.q_net(states_b).gather(1, actions_b)  # (B, 1)

        # SMDP: per-sample effective discount for bootstrapping.
        # When throttling is off, every k_i = 1, so gamma_eff = γ^n
        # (identical to standard n-step DQN).
        gamma_n = gamma_effs_b  # (B, 1) — per-sample γ^{Σ k_i}

        # Build TD target
        with torch.no_grad():
            if self.use_sarsa:
                # ON-POLICY METHODS: SARSA + Double Deep SARSA
                # Build per-sample validity mask so SARSA targets only
                # bootstrap from actions that the online policy could
                # actually select at s'.
                sarsa_mask = self._get_target_valid_mask_batch(next_states_b)

                # 1) Select next action a' using epsilon-greedy policy:
                if self.use_double:
                    # Double Deep SARSA: action selection from the online network
                    next_actions_b = self._epsilon_greedy_action_batch_from_net(
                        self.q_net, next_states_b, valid_mask=sarsa_mask
                    )
                else:
                    # Vanilla Deep SARSA: action selection from the target network
                    next_actions_b = self._epsilon_greedy_action_batch_from_net(
                        self.target_net, next_states_b, valid_mask=sarsa_mask
                    )

                # 2) Evaluate Q(s', a') ALWAYS using target_net.
                next_q_eval = self.target_net(next_states_b)          # Q_target(s', ·)
                next_qsa_b = next_q_eval.gather(1, next_actions_b)    # Q_target(s', a')

                # TD target for n-step SARSA:
                #   target = G^{(n)} + (1 - done) * γ^n * Q_target(s', a')
                target_b = rewards_b + (~dones_b) * gamma_n * next_qsa_b

            else:
                # OFF-POLICY METHODS: DQN + Double DQN
                # Build target mask to exclude invalid/duplicate actions
                # in s' (inventory-limit aware).
                _target_mask = self._get_target_valid_mask_batch(next_states_b)

                if self.use_double:
                    # Double DQN:
                    #   1) Select greedy action using the online network
                    #   2) Evaluate that action using target network
                    next_q_online = self.q_net(next_states_b)  # Q_online(s', ·)
                    if _target_mask is not None:
                        next_q_online = next_q_online.clone()
                        next_q_online[~_target_mask] = float('-inf')
                    next_actions_b = torch.argmax(next_q_online, dim=1, keepdim=True)

                    q_next_target = self.target_net(next_states_b)  # Q_target(s', ·)
                    max_next_q = q_next_target.gather(1, next_actions_b)  # Q_target(s', a*_online)
                else:
                    # Vanilla DQN:
                    #   target = G^{(n)} + γ^n * max_a' Q_target(s', a')
                    q_next_target = self.target_net(next_states_b)  # Q_target(s', ·)
                    if _target_mask is not None:
                        q_next_target_masked = q_next_target.clone()
                        q_next_target_masked[~_target_mask] = float('-inf')
                        max_next_q, _ = torch.max(q_next_target_masked, dim=1, keepdim=True)
                    else:
                        max_next_q, _ = torch.max(q_next_target, dim=1, keepdim=True)

                # TD target for n-step DQN
                target_b = rewards_b + (~dones_b) * gamma_n * max_next_q

            # --- Scalar TD target diagnostics (both SARSA and DQN branches) ---
            td_np = target_b.cpu().numpy().ravel()
            self.episode_td_target_min = min(self.episode_td_target_min, float(td_np.min()))
            self.episode_td_target_max = max(self.episode_td_target_max, float(td_np.max()))
            self.episode_td_target_sum += float(td_np.sum())
            self.episode_td_target_count += len(td_np)

        # ---------------------------------------------------------------
        # Loss computation & backpropagation
        # ---------------------------------------------------------------
        if self.use_prioritized_experience:
            # PER path: TD-errors are used to update priorities in the buffer,
            # and importance-sampling (IS) weights correct the non-uniform bias.
            td_errors = (qsa_b - target_b).detach().abs()  # (B, 1)
            td_err_np = td_errors.view(-1).clamp(min=0.0, max=1e6).cpu().numpy()
            self._accumulate_replay_sample_stats(
                idxs_np=idxs_b.view(-1).cpu().numpy(),
                rewards_b=rewards_b,
                td_error_values=td_err_np,
                priority_values=sampled_priorities_np,
            )

            # Update priorities in the buffer (move to CPU for numpy ops).
            # Clamp to guard against NaN/Inf from degenerate targets.
            if self.use_per_priority_clip:
                td_err_np = self._clip_per_priorities(td_err_np)
            idxs_np = idxs_b.view(-1).cpu().numpy()
            self.memory.update_priorities(idxs_np, td_err_np)
            self.episode_max_priority = max(
                self.episode_max_priority, float(td_err_np.max()))

            # IS-weighted Huber (smooth L1): multiply per-sample loss by IS
            # weights.  Huber matches the uniform branch and is less sensitive
            # to outlier TD-errors than MSE — critical under PER where high-
            # priority samples already amplify large errors.
            per_sample_loss = F.smooth_l1_loss(qsa_b, target_b, reduction="none")  # (B, 1)
            loss = (weights_b * per_sample_loss).mean()
        else:
            td_errors = (qsa_b - target_b).detach().abs()
            self._accumulate_replay_sample_stats(
                idxs_np=idxs_b.view(-1).cpu().numpy(),
                rewards_b=rewards_b,
                td_error_values=td_errors.view(-1).cpu().numpy(),
            )
            # Uniform replay path: smooth L1 loss (Huber loss).
            # Huber is less sensitive to outlier TD-errors than MSE,
            # which improves stability when rewards are noisy.
            loss = F.smooth_l1_loss(qsa_b, target_b)

        # Track extreme-reward transitions in this batch (scalar branch)
        if not self.use_distributional:
            _extreme_thresh = -80.0
            self.episode_extreme_reward_count += int(
                (rewards_b.view(-1) < _extreme_thresh).sum().item())
            self.episode_batch_total += int(rewards_b.numel())

        # L2 anchor regularization (same as distributional path).
        anchor_loss = self._compute_anchor_loss()
        total_loss = loss + anchor_loss

        # Standard SGD update cycle
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping prevents exploding gradients from large TD-errors
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Accumulate loss for episode-level averaging (logged in log_episode_stats)
        loss_val = float(loss.item())
        self.episode_loss_sum += loss_val
        self.episode_loss_count += 1

        # Track total gradient steps (used for TensorBoard x-axis)
        self.global_grad_step += 1

        # Hard update: periodically copy q_net weights → target_net.
        # This keeps the TD target stable between updates, preventing
        # the "moving target" instability of online TD learning.
        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ================================================================
    # log_episode_stats() — END-OF-EPISODE BOOKKEEPING
    # ================================================================
    #
    # Called ONCE at the end of each episode from the outer training loop.
    # Responsibilities:
    #   1. Log episode metrics to TensorBoard (reward, PnL, loss, epsilon)
    #   2. Anneal PER alpha/beta schedules (if using prioritized replay)
    #   3. Reset per-episode accumulators for the next episode
    #
    # NOTE: Epsilon decay is NOT done here — it should be done externally
    # by the training loop: ε = max(ε_min, ε * ε_decay)
    # ================================================================

    def _clip_per_priorities(self, prios_np: np.ndarray) -> np.ndarray:
        """Clip priorities to median_ema * 3 and update the running median EMA."""
        batch_median = float(np.median(prios_np))
        alpha = self._per_priority_ema_alpha
        self._per_priority_median_ema = (
            (1.0 - alpha) * self._per_priority_median_ema + alpha * batch_median
        )
        cap = max(self._per_priority_median_ema * 3.0, 1e-6)
        return np.clip(prios_np, 0.0, cap)

    def log_episode_stats(
        self,
        episode_idx: int,
        total_reward: float,
        final_pnl: float,
        discounted_return: float = 0.0,
    ) -> None:
        """
        Log episode-level statistics to TensorBoard and anneal PER schedules.

        Called ONCE at the end of each episode by the outer training loop.
        This method does NOT decay epsilon — that is the caller's responsibility.

        Parameters
        ----------
        episode_idx : int
            Index of the current episode (used as TensorBoard x-axis).
        total_reward : float
            Cumulative (undiscounted) reward over the entire episode.
        final_pnl : float
            Final profit-and-loss at episode end (mark-to-market + realized).
        discounted_return : float
            G_0 = Σ γ^t r_t (micro-step discount).  Diagnostic for checking
            whether C51's V(s_0) predictions converge to the observed return.

        Side Effects
        ------------
        - Writes to self.writer (TensorBoard): total_reward, discounted_return,
          final_pnl, epsilon, mean_td_loss, and optionally per_alpha/per_beta.
        - Updates self.memory.alpha / self.memory.beta (PER annealing).
        - Resets episode_loss_sum, episode_loss_count, action_counts.
        """
        # Compute mean TD-loss for this episode (if we actually trained)
        if self.episode_loss_count > 0:
            mean_loss = self.episode_loss_sum / self.episode_loss_count
        else:
            mean_loss = 0.0

        # Episode-level TensorBoard logging
        self.writer.add_scalar("episode/total_reward", total_reward, episode_idx)
        self.writer.add_scalar("episode/discounted_return", discounted_return, episode_idx)
        self.writer.add_scalar("episode/final_pnl", final_pnl, episode_idx)
        self.writer.add_scalar("episode/epsilon", self.epsilon, episode_idx)
        self.writer.add_scalar("episode/mean_td_loss", mean_loss, episode_idx)

        # -------------------------------------------------------------
        # PER schedule annealing (linear interpolation over episodes)
        # -------------------------------------------------------------
        # Both alpha and beta follow a linear schedule:
        #   val(ep) = start + t * (end - start),   t = ep / last_episode
        #
        # Alpha annealing: typically 1.0 → 0.5
        #   Start with full prioritization (focus on high-error transitions).
        #   Gradually reduce to let more uniform sampling stabilize learning.
        #
        # Beta annealing: typically 0.4 → 1.0
        #   Start with partial IS-correction (biased but high signal).
        #   Gradually increase to full correction (unbiased gradients).
        #   By the end of training, beta=1.0 ensures convergence guarantees.
        if self.use_prioritized_experience:
            # Alpha: controls how much TD-error affects sampling probability
            # BUG FIX: use (last_episode - 1) as denominator so that the final
            # episode (idx = last_episode - 1) reaches t=1.0 exactly.
            # Old formula: episode_idx / last_episode → max t = (N-1)/N < 1.0.
            last_ep_alpha = max(1, self.per_alpha_last_episode - 1)
            t_alpha = min(1.0, episode_idx / float(last_ep_alpha))
            alpha = self.per_alpha_start + t_alpha * (self.per_alpha_end - self.per_alpha_start)

            # Beta: controls importance-sampling weight correction strength
            last_ep_beta = max(1, self.per_beta_last_episode - 1)
            t_beta = min(1.0, episode_idx / float(last_ep_beta))
            beta = self.per_beta_start + t_beta * (self.per_beta_end - self.per_beta_start)

            # Apply the annealed values to the replay buffer
            if isinstance(self.memory, PrioritizedReplayMemory):
                self.memory.alpha = float(alpha)
                self.memory.beta = float(beta)

            # Log the schedule evolution for debugging
            self.writer.add_scalar("episode/per_alpha", alpha, episode_idx)
            self.writer.add_scalar("episode/per_beta", beta, episode_idx)

        # NoisyNet sigma diagnostic — mean |σ| across all NoisyLinear layers.
        # Tracks how much exploration noise remains.  If mean_sigma drops to
        # ~0 early in training, the network has "turned off" exploration
        # prematurely and sigma_init may need to be increased.
        if self.use_noisy_net:
            sigma_abs_sum = 0.0
            sigma_count = 0
            for module in self.q_net.modules():
                if isinstance(module, NoisyLinear):
                    sigma_abs_sum += module.w_sigma.data.abs().sum().item()
                    sigma_count += module.w_sigma.data.numel()
                    sigma_abs_sum += module.b_sigma.data.abs().sum().item()
                    sigma_count += module.b_sigma.data.numel()
            if sigma_count > 0:
                mean_sigma = sigma_abs_sum / sigma_count
                self.writer.add_scalar("episode/mean_noisy_sigma", mean_sigma, episode_idx)

        # Per-action Q-value diagnostics — mean Q(s, a) across all decisions
        # this episode.  Helps diagnose C51 resolution: if Q(a=0) - Q(a=skew)
        # is smaller than the atom spacing, the agent cannot distinguish them.
        if self.episode_q_count > 0:
            mean_q = self.episode_q_sum / self.episode_q_count
            for a_idx in range(self.n_actions):
                self.writer.add_scalar(
                    f"episode/mean_Q_a{a_idx}", mean_q[a_idx], episode_idx
                )
            # Log the spread between best and worst action Q-values
            self.writer.add_scalar(
                "episode/Q_spread", float(mean_q.max() - mean_q.min()), episode_idx
            )
            # Q-value range across all decision states this episode.
            # Compare with [V_min, V_max] to check C51 support adequacy.
            if self.episode_q_min != float('inf'):
                self.writer.add_scalar("episode/Q_min", self.episode_q_min, episode_idx)
                self.writer.add_scalar("episode/Q_max", self.episode_q_max, episode_idx)
                self.writer.add_scalar("episode/Q_mean", float(mean_q.mean()), episode_idx)

        # TD target diagnostics — scalar target range and Tz clipping fraction.
        # If td_target_min << V_min or td_target_max >> V_max, the C51 support
        # is too narrow and should be widened.
        if self.episode_td_target_count > 0:
            td_mean = self.episode_td_target_sum / self.episode_td_target_count
            self.writer.add_scalar("episode/td_target_min", self.episode_td_target_min, episode_idx)
            self.writer.add_scalar("episode/td_target_max", self.episode_td_target_max, episode_idx)
            self.writer.add_scalar("episode/td_target_mean", td_mean, episode_idx)
        if self.episode_tz_clip_count > 0:
            tz_clip_frac = self.episode_tz_clip_frac_sum / self.episode_tz_clip_count
            self.writer.add_scalar("episode/tz_clip_frac", tz_clip_frac, episode_idx)

        # Replay buffer poisoning diagnostics
        if self.use_prioritized_experience and self.episode_max_priority > 0:
            self.writer.add_scalar(
                "debug/max_priority", self.episode_max_priority, episode_idx)
            if self.use_per_priority_clip:
                self.writer.add_scalar(
                    "debug/per_priority_median_ema",
                    self._per_priority_median_ema,
                    episode_idx,
                )
                self.writer.add_scalar(
                    "debug/per_priority_cap",
                    self._per_priority_median_ema * 3.0, episode_idx)
        if self.episode_batch_total > 0:
            extreme_frac = self.episode_extreme_reward_count / self.episode_batch_total
            self.writer.add_scalar(
                "debug/extreme_reward_frac", extreme_frac, episode_idx)

        # Replay-sampling diagnostics: what actually entered the gradient updates.
        if self.episode_sample_total > 0:
            recent_frac = self.episode_sample_recent_count / self.episode_sample_total
            self.writer.add_scalar(
                "debug/replay_recent_sample_frac_50ep",
                recent_frac,
                episode_idx,
            )
        if self.episode_sample_delta_count > 0:
            mean_sampled_delta = (
                self.episode_sample_delta_sum / self.episode_sample_delta_count
            )
            self.writer.add_scalar(
                "debug/replay_sampled_delta_mean",
                mean_sampled_delta,
                episode_idx,
            )

        if self.episode_sample_rewards:
            reward_qs = np.quantile(
                np.asarray(self.episode_sample_rewards, dtype=np.float64),
                [0.10, 0.50, 0.90],
            )
            self.writer.add_scalar("debug/sampled_reward_q10", float(reward_qs[0]), episode_idx)
            self.writer.add_scalar("debug/sampled_reward_q50", float(reward_qs[1]), episode_idx)
            self.writer.add_scalar("debug/sampled_reward_q90", float(reward_qs[2]), episode_idx)

        if self.episode_sample_td_errors:
            td_qs = np.quantile(
                np.asarray(self.episode_sample_td_errors, dtype=np.float64),
                [0.10, 0.50, 0.90],
            )
            self.writer.add_scalar("debug/sampled_td_error_q10", float(td_qs[0]), episode_idx)
            self.writer.add_scalar("debug/sampled_td_error_q50", float(td_qs[1]), episode_idx)
            self.writer.add_scalar("debug/sampled_td_error_q90", float(td_qs[2]), episode_idx)

        if self.episode_sample_priorities:
            priority_median = float(
                np.median(np.asarray(self.episode_sample_priorities, dtype=np.float64))
            )
            self.writer.add_scalar(
                "debug/sampled_priority_median",
                priority_median,
                episode_idx,
            )

        # Reset accumulators for the next episode
        self.reset_episode_accumulators()

    def reset_episode_accumulators(self) -> None:
        """Zero out all per-episode running accumulators.

        Called internally by `log_episode_stats()` at the end of each
        training episode.  Also exposed as a public method so callers
        can clear state after offline eval/calibration phases (e.g.
        ADR-lite baseline calibration, EWC sample collection) that
        run `act()` with `enable_learning=False` but still accumulate
        `action_counts`, `episode_q_sum`, etc.  Without an explicit
        reset between those phases and the start of training, episode 1
        of training would inherit polluted stats.
        """
        self.episode_loss_sum = 0.0
        self.episode_loss_count = 0
        self.action_counts[:] = 0
        self.episode_q_sum[:] = 0.0
        self.episode_q_count = 0
        self.episode_q_min = float('inf')
        self.episode_q_max = float('-inf')
        self.episode_td_target_min = float('inf')
        self.episode_td_target_max = float('-inf')
        self.episode_td_target_sum = 0.0
        self.episode_td_target_count = 0
        self.episode_tz_clip_frac_sum = 0.0
        self.episode_tz_clip_count = 0
        self.episode_max_priority = 0.0
        self.episode_extreme_reward_count = 0
        self.episode_batch_total = 0
        self.episode_sample_recent_count = 0
        self.episode_sample_total = 0
        self.episode_sample_delta_sum = 0.0
        self.episode_sample_delta_count = 0
        self.episode_sample_rewards = []
        self.episode_sample_td_errors = []
        self.episode_sample_priorities = []
