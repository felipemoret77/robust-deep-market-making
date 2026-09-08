#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advantage Actor-Critic (A2C / A3C) for LOB Market-Making
=========================================================

This module implements the Advantage Actor-Critic family of policy-gradient
algorithms (Mnih et al., 2016) adapted for a market-making agent operating
in a Limit Order Book (LOB) simulated via the Santa Fe zero-intelligence
model (Daniels et al., 2003; Smith et al., 2003).

References
----------
[1] Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
    Silver, D. & Kavukcuoglu, K.  "Asynchronous Methods for Deep Reinforcement
    Learning."  ICML, 2016.   (Original A3C paper.)
[2] Sutton, R.S. & Barto, A.G.  "Reinforcement Learning: An Introduction",
    2nd ed., MIT Press, 2018.   (Chapters 13 & 15 on policy gradients.)
[3] Bradtke, S.J. & Duff, M.O.  "Reinforcement Learning Methods for
    Continuous-Time Markov Decision Processes."  NeurIPS, 1994.
    (SMDP discounting framework.)
[4] Williams, R.J.  "Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning."  Machine Learning, 8, 1992.
    (REINFORCE theorem underlying all policy-gradient methods.)
[5] Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P.
    "High-Dimensional Continuous Control Using Generalized Advantage
    Estimation."  ICLR, 2016.  (GAE; relevant to N-step advantage.)
[6] Polyak, B.T. & Juditsky, A.B.  "Acceleration of Stochastic
    Approximation by Averaging."  SIAM J. Control and Optimization, 1992.
    (Polyak averaging, basis for target network soft update.)

Architecture Overview
=====================
::

    ┌──────────────────────────────────────────────────────┐
    │                ActorCriticController                  │
    │  ┌────────────┐   ┌─────────────┐  ┌──────────────┐ │
    │  │ ActorNet   │   │ CriticNet   │  │ TargetCritic │ │
    │  │ π_θ(a | s) │   │ V_ϕ(s)      │  │ V̄_ϕ̄(s)      │ │
    │  └─────┬──────┘   └──────┬──────┘  └──────┬───────┘ │
    │        │ logits          │ V(s)           │ V̄(s')   │
    │  ┌─────▼─────────────────▼────────────────▼───────┐  │
    │  │           actor_critic_update_batch()           │  │
    │  │  TD target:  y = R_cum + γ^K · V̄(s') · (1-d)  │  │
    │  │  Advantage:  Â = y - V(s)                      │  │
    │  │  L_actor  = -E[log π_θ(a|s) · Â] - η·H(π_θ)  │  │
    │  │  L_critic = Huber(V_ϕ(s), y)                   │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  Polyak update:  ϕ̄ ← τ·ϕ + (1-τ)·ϕ̄                 │
    └──────────────────────────────────────────────────────┘

Two training modes, controlled by ``use_synchronous``:

    A2C (synchronous) — ``run_a2c_sync_batch``
        N environments are advanced in lock-step at the decision level.
        At each synchronised decision point the N transitions form a
        single mini-batch.  After every ``n_steps_sync`` such batches
        the accumulated data is fed to one gradient step.
        ∇_θ J ≈ (1/B) Σ_{i=1}^{B} [-log π_θ(a_i|s_i) · Â_i - η · H_i]

    A3C (asynchronous) — ``run_a3c_async_batch``
        Each of the N environments runs in its own thread.  When a worker
        commits an SMDP transition it immediately acquires a gradient lock
        and performs an individual update on the shared networks.  The
        other N-1 workers may observe partially-updated weights between
        their own decisions — this weight staleness provides implicit
        regularisation (Mnih et al., 2016, §3).

Key design features:

    • Separate actor π_θ and critic V_ϕ with independent AdamW optimizers
      and learning-rate schedules (decoupled actor/critic convergence).
    • Target critic network V̄_ϕ̄ with Polyak soft update (Polyak &
      Juditsky, 1992) for stable TD bootstrap targets.
    • Advantage normalisation: "std_only" mode divides by σ_Â while
      preserving the sign of Â.  This prevents mean-subtraction from
      flipping the gradient for reward-sparse regimes (see plan notes).
    • N-step SMDP-correct TD returns via a sliding-window deque per
      worker.  The N-step return is:
          G_t^{(N)} = Σ_{j=0}^{N-1} [∏_{i=0}^{j-1} γ^{K_i}] · R_j^{cum}
                      + [∏_{i=0}^{N-1} γ^{K_i}] · V(s_{t+N}) · (1-d)
      where K_i is the holding time (micro-steps) of SMDP transition i.
    • Three independent throttle gates (TOB change, event count, wall
      time) with OR-blocking semantics, plus two bypass priorities
      (mode change, fill replenishment) disableable via ``use_mdp``.
    • Constrained action selection via ``_get_valid_action_mask()`` to
      prevent inventory-limit violations.

A2C vs REINFORCE — Algorithmic Comparison
==========================================
REINFORCE (Williams, 1992) is an on-policy Monte Carlo method:

    1. Collect a FULL episode τ = {(s_0, a_0, r_0), …, (s_T, a_T, r_T)}.
    2. Compute Monte Carlo returns:  G_t = Σ_{k=t}^{T} γ^{k-t} r_k.
    3. Baseline-subtracted advantage:  Â_t = G_t - b(s_t).
       • REINFORCE with baseline uses b(s_t) = mean(G) or a learned V(s).
    4. Policy gradient theorem (Sutton et al., 2000):
           ∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · Â_t ]
    5. SINGLE update using the entire trajectory (unbiased, high variance).

A2C replaces Monte Carlo returns with bootstrapped TD(0) targets:

    1. At each transition (s_t, a_t, r_t, s_{t+1}):
       • Actor:   π_θ(a | s) — stochastic policy (same as REINFORCE).
       • Critic:  V_ϕ(s) — learned state-value function (neural baseline).
    2. TD target:       y_t = r_t + γ^K · V̄(s_{t+1}) · (1-d_t)
       TD advantage:    Â_t = y_t - V_ϕ(s_t)
    3. Actor loss:   L_actor  = -E[ log π_θ(a_t|s_t) · sg(Â_t) ] - η · H(π_θ)
       Critic loss:  L_critic = Huber( V_ϕ(s_t),  sg(y_t) )
       where sg(·) denotes stop-gradient (``detach()``).
    4. Separate gradient steps for actor and critic, each with its own
       AdamW optimiser (Loshchilov & Hutter, 2019).

Variance–Bias Trade-off
    REINFORCE:  Unbiased (uses true return G_t), but high variance — the
                gradient depends on the entire future trajectory.
    A2C:        Biased (V_ϕ is approximate), but MUCH lower variance —
                bootstrapping reduces the credit assignment horizon from
                the full episode length T to the N-step look-ahead.
    The practical benefit: A2C converges in orders of magnitude fewer
    samples than REINFORCE for the same task.

Why Separate Actor/Critic Learning Rates?
    The critic solves a supervised regression problem (minimise
    Huber(V_ϕ(s), y)) which typically converges faster than the actor's
    noisy policy gradient.  Setting lr_critic > lr_actor (e.g. 5×) lets
    the value function stabilise early, providing accurate advantages
    for the actor.  Konda & Tsitsiklis (2003) show that the two-timescale
    update (slow actor, fast critic) converges under standard conditions.

SMDP-Correct Discounting
=========================
When the controller is throttled, each "macro-transition" spans K > 1
micro-steps.  The SMDP framework (Bradtke & Duff, 1994) requires:

    Cumulative reward:  R^{cum} = Σ_{k=0}^{K-1} γ^k · r_{t+k}
    TD target:          y_t = R^{cum} + γ^K · V̄(s_{t+K}) · (1-d)

Using γ^1 instead of γ^K would under-discount future values when K > 1,
injecting a systematic positive bias into the advantage and over-valuing
distant states.  The ``learn()`` method accumulates R^{cum} on-line;
the training functions pass K to ``actor_critic_update_batch()`` which
computes γ^K for the bootstrap.

State Representation
====================
Both modes apply log1p compression to volumes and spreads (stabilises
gradients and bounds the input range) and normalise inventory by
inv_limit.

GENERIC MODE (pure_mm=False):
    s = [log1p(spread), log1p(ask_size), log1p(bid_size),
         inv / inv_limit, has_bid, has_ask]            ∈ ℝ^6

PURE MM MODE (pure_mm=True):
    s = [log1p(spread), inv / inv_limit,
         log1p(bid_sizes[0..K]), log1p(ask_sizes[0..K])]
    ∈ ℝ^{2+2(K+1)}   where K = max non-negative offset in the grid

Action Space
============
GENERIC ACTIONS (pure_mm=False):
    Index | Action                | Inventory effect
    ------+------------------------+----------------
      0   | post_bid              | +1 on fill
      1   | post_ask              | -1 on fill
      2   | post_bid_ask          | ±1 on fill
      3   | cancel_bid            |  0
      4   | cancel_ask            |  0
      5   | hold                  |  0
    (Optional extensions: 6=inside_spread_both, 7=inside_bid, 8=inside_ask)

PURE MM ACTIONS (pure_mm=True):
    Each action index i maps to an offset pair (δ_bid^i, δ_ask^i) in ticks:
        bid_price = best_bid − δ_bid   (δ > 0 = more passive)
        ask_price = best_ask + δ_ask   (δ > 0 = more passive)
        δ < 0 = inside the spread      (δ = 0 = quote at L1)

Action Masking (Constrained Selection)
======================================
At each decision step, ``_get_valid_action_mask()`` builds a boolean
mask m ∈ {0,1}^|A| that restricts the policy to feasible actions.
Invalid logits are set to −∞ before softmax, so π(a|s) = 0 for masked
actions.  Formally:

    π̃(a|s) = softmax(z_a + (1 − m_a) · (−∞))

GENERIC MODE at inventory limits:
    inv ≥ +limit  →  mask out all buy-side actions
    inv ≤ −limit  →  mask out all sell-side actions

PURE MM MODE at inventory limits (Canonical Degenerate Masking):
    At +limit the execution layer drops the bid.  Any two actions that
    differ only on δ_bid produce identical outcomes (same ask_price,
    no bid).  We define equivalence classes [a] = {a' : δ_ask^{a'} =
    δ_ask^a} and keep only the canonical (lowest-index) representative
    per class.  This avoids wasting policy capacity on indistinguishable
    actions and prevents the gradient from splitting mass across them.

Throttle Gate Hierarchy
=======================
When throttling is enabled, the controller does NOT query the neural
network on every micro-step.  Instead it follows a priority-based
decision hierarchy:

    Priority A (BYPASS): Mode change — inventory crossed ±inv_limit.
    Priority B (BYPASS): Fill detected — inventory changed since last decision.
    Priority C (THROTTLE): Hard throttle check with OR-blocking semantics:
        − TOB gate:   ≥ n_tob_moves changes to (best_bid, best_ask, sizes)
        − Event gate:  ≥ n_events micro-steps elapsed
        − Time gate:   ≥ min_time_interval seconds of simulated time elapsed

    First-Action Bypass: Before the first decision of each episode, ALL
    gates are bypassed so initial quotes are placed immediately.

MDP Mode (use_mdp=True)
========================
Disables BOTH bypass priorities (A and B).  The agent strictly respects
the throttle gates, making K = 1 at every decision (pure MDP, no SMDP
aggregation).  Recommended for initial training experiments to simplify
the credit-assignment problem.

Inventory Band (Pure MM)
========================
::

    inv_limit = None   →  Always two-sided quoting.
    inv_limit = Q      →  inventory ≥ +Q  →  ASK only
                          inventory ≤ −Q  →  BID only
                          otherwise       →  BID + ASK
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import collections
import copy

import os
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import math

from RLController import RLController


# ============================================================
# Actor Network: π_θ(a | s)
# ============================================================

class ActorNetwork(nn.Module):
    """
    MLP policy network that maps states to action logits.

    Formally, the actor is a function approximator for the policy:

        π_θ(a | s) = softmax( f_θ(s) )_a

    where f_θ : ℝ^D → ℝ^|A| is a multi-layer perceptron and the
    softmax converts raw logits z_a = f_θ(s)_a into a valid
    probability distribution over the discrete action space A.

    Architecture::

        s ∈ ℝ^D  ──▶ [Linear(D, H) → ReLU] × L  ──▶ Linear(H, |A|)  ──▶ z ∈ ℝ^|A|

    where D = input_dim, H = n_neurons, L = n_hidden, |A| = n_actions.

    Input Dimensions
    ----------------
    Generic mode (pure_mm=False):
        D = 6: [log1p(spread), log1p(ask_size), log1p(bid_size),
                inv/inv_limit, has_bid, has_ask]

    Pure-MM mode (pure_mm=True):
        D = 2 + 2(K+1): [log1p(spread), inv/inv_limit,
                          log1p(bid_sizes[0..K]), log1p(ask_sizes[0..K])]

    Output
    ------
    z ∈ ℝ^|A| — unnormalised log-probabilities (logits).  The caller
    applies softmax for sampling (training) or argmax for greedy
    selection (evaluation).  Action masking sets z_a = −∞ for
    infeasible actions before the softmax.

    Exploration Mechanism
    ---------------------
    Unlike the DQN controller (which uses NoisyNet linear layers),
    A2C explores naturally through stochastic sampling from π_θ.
    The entropy bonus −η · H(π_θ(·|s)) in the actor loss provides
    additional exploration pressure by penalising peaked distributions.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        n_hidden: int = 2,
        n_neurons: int = 128,
    ):
        super().__init__()

        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")

        # Build the shared MLP trunk: input_dim -> n_neurons -> ... (n_hidden layers)
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_hidden)):
            layers.append(nn.Linear(in_dim, int(n_neurons)))
            layers.append(nn.ReLU())
            in_dim = int(n_neurons)

        self.feature = nn.Sequential(*layers)

        # Final linear readout: n_neurons -> n_actions (no activation)
        self.fc_out = nn.Linear(int(n_neurons), int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state tensor -> action logits.

        Parameters
        ----------
        x : torch.Tensor
            Batch of state vectors, shape (B, input_dim).

        Returns
        -------
        logits : torch.Tensor
            Unnormalized log-probabilities, shape (B, n_actions).
        """
        z = self.feature(x.float())
        logits = self.fc_out(z)
        return logits


# ============================================================
# Critic Network: V_ϕ(s)
# ============================================================

class CriticNetwork(nn.Module):
    """
    MLP value network that outputs a scalar state-value estimate.

    The critic approximates the state-value function under the
    current policy π_θ:

        V_ϕ(s) ≈ V^{π_θ}(s) = E_{π_θ}[ Σ_{k=0}^{∞} γ^k r_{t+k} | s_t = s ]

    Architecture::

        s ∈ ℝ^D  ──▶ [Linear(D, H) → ReLU] × L  ──▶ Linear(H, 1)  ──▶ V̂ ∈ ℝ

    No output activation — V(s) is an unbounded real number (it may be
    negative when inventory penalties dominate spread capture).

    Why a Separate Network?
    -----------------------
    The actor and critic solve qualitatively different problems:

        Actor:   classification-like (map s → probability distribution π)
        Critic:  regression (map s → scalar V)

    Sharing weights (as in some A2C variants) creates gradient
    interference: the critic's Huber/MSE gradient can destabilise the
    actor's policy gradient, especially when the critic loss is much
    larger early in training.  With separate networks, each has its own
    AdamW optimiser and can converge at its own learning rate — the
    "two-timescale" update of Konda & Tsitsiklis (2003).

    The cost is doubled parameters, but for our architectures (1–2
    hidden layers, 128–256 neurons) this adds < 200K parameters total.

    Parameters
    ----------
    input_dim : int
        State vector dimensionality D.
    n_hidden : int
        Number of hidden layers L (each with ReLU activation).
    n_neurons : int
        Width H of each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        n_hidden: int = 2,
        n_neurons: int = 128,
    ):
        super().__init__()

        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")

        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_hidden)):
            layers.append(nn.Linear(in_dim, int(n_neurons)))
            layers.append(nn.ReLU())
            in_dim = int(n_neurons)

        self.feature = nn.Sequential(*layers)

        # Single output: the state-value V(s).
        # No activation — V(s) is an unbounded real number.
        self.fc_out = nn.Linear(int(n_neurons), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state tensor -> scalar value.

        Parameters
        ----------
        x : torch.Tensor
            Batch of state vectors, shape (B, input_dim).

        Returns
        -------
        value : torch.Tensor
            State-value estimates, shape (B, 1).
        """
        z = self.feature(x.float())
        value = self.fc_out(z)
        return value


# ============================================================
# Transition Container for Batch Updates
# ============================================================

@dataclass
class ACTransitionBatch:
    """
    Container for a batch of SMDP-level transitions used by
    ``actor_critic_update_batch()``.

    Each element i in the parallel lists represents one SMDP macro-
    transition:

        τ_i = (s_i, a_i, R_i^{cum}, s'_i, d_i, K_i, m_i)

    Fields
    ------
    states : list[Tensor]
        s_i ∈ ℝ^D — state at decision time (on CPU to save GPU memory).
    actions : list[int]
        a_i ∈ {0, …, |A|−1} — action index sampled from π_θ(·|s_i).
    rewards : list[float]
        R_i^{cum} = Σ_{k=0}^{K_i−1} γ^k r_{t+k} — SMDP cumulative
        discounted reward aggregated by ``learn()``.
    next_states : list[Tensor]
        s'_i ∈ ℝ^D — state at the NEXT decision point (or terminal).
    dones : list[bool]
        d_i ∈ {0,1} — whether the episode terminated during this
        SMDP holding period (if True, V(s') is zeroed in the bootstrap).
    k_steps : list[int]
        K_i ∈ ℕ^+ — number of micro-steps spanned by this SMDP
        transition.  Used for γ^K discounting in the TD target.
        K_i = 1 when throttling is off or use_mdp=True.
    masks : list[ndarray | None]
        m_i ∈ {0,1}^|A| — valid action mask at decision time.
        Reapplied during the update to ensure
        log π_θ^{update}(a_i|s_i) = log π_θ^{collect}(a_i|s_i).
    """
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    k_steps: List[int] = field(default_factory=list)
    masks: List[Optional[np.ndarray]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)


# ============================================================
# N-step TD Return Helpers
# ============================================================
#
# These helpers implement N-step temporal-difference returns for the
# SMDP setting.  The standard N-step return (Sutton & Barto, 2018, §7.1)
# is:
#     G_t^{(N)} = Σ_{j=0}^{N-1} γ^j r_{t+j}  +  γ^N V(s_{t+N})
#
# In an SMDP, each macro-transition j has variable holding time K_j,
# so the discount must compound across holding times:
#
#     G_t^{(N)} = Σ_{j=0}^{N-1} [∏_{i=0}^{j-1} γ^{K_i}] · R_j^{cum}
#                 +  [∏_{i=0}^{N-1} γ^{K_i}] · V(s_{t+N})
#
# where R_j^{cum} is the already-discounted intra-transition reward
# and K_total = Σ K_j is the total micro-steps.  The virtual transition
# (s_0, a_0, G^{(N)}, s_N, d_N, K_total, m_0) can be fed directly to
# ``actor_critic_update_batch()`` which adds the bootstrap γ^{K_total}·V(s_N).

def _compute_nstep_virtual_transition(
    buffer: list,
    gamma: float,
) -> dict:
    """
    Fold N consecutive SMDP transitions into a single virtual transition.

    Given a window [τ_0, τ_1, …, τ_{N-1}], compute:

        G^{(N)} = R_0^{cum}
                 + γ^{K_0} · R_1^{cum}
                 + γ^{K_0+K_1} · R_2^{cum}
                 + …
                 + γ^{Σ_{i<N-1} K_i} · R_{N-1}^{cum}

        K_total = K_0 + K_1 + … + K_{N-1}

    If any transition j is terminal (d_j = True), the sum is truncated
    at j (no bootstrap beyond a terminal state).

    Parameters
    ----------
    buffer : list of dict
        Each dict has keys: ``s``, ``a``, ``r``, ``s_next``, ``done``,
        ``k``, ``mask``.  Must contain at least 1 entry.
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        Virtual transition with keys: ``s``, ``a``, ``r``, ``s_next``,
        ``done``, ``k``, ``mask``.
    """
    cum_discount = 1.0
    g_n = 0.0
    k_total = 0
    done_n = False
    s_n = buffer[0]["s_next"]  # default if only 1 entry

    for tr in buffer:
        g_n += cum_discount * tr["r"]
        k_total += tr["k"]
        s_n = tr["s_next"]
        if tr["done"]:
            done_n = True
            break
        cum_discount *= gamma ** tr["k"]

    return {
        "s": buffer[0]["s"],
        "a": buffer[0]["a"],
        "r": g_n,
        "s_next": s_n,
        "done": done_n,
        "k": k_total,
        "mask": buffer[0]["mask"],
    }


def _drain_nstep_buffer(
    buffer,
    gamma: float,
    n_steps: int,
    output: ACTransitionBatch,
    flush: bool = False,
) -> int:
    """
    Process a per-worker N-step buffer and append virtual transitions to
    ``output``.

    Normal mode (``flush=False``):
        While the buffer has >= ``n_steps`` entries, compute the N-step return
        for the oldest entry, pop it from the front, and append the virtual
        transition to ``output``.

    Flush mode (``flush=True``):
        After the episode ends, process ALL remaining entries with
        progressively shorter windows (N-1, N-2, ..., 1).

    Parameters
    ----------
    buffer : collections.deque
        Per-worker deque of transition dicts.
    gamma : float
        Discount factor.
    n_steps : int
        Number of look-ahead steps.
    output : ACTransitionBatch
        Batch to append virtual transitions to.
    flush : bool
        If True, drain all remaining entries (end of episode).

    Returns
    -------
    int
        Number of virtual transitions appended.
    """
    count = 0

    # Normal: pop while we have enough for a full N-step window
    while len(buffer) >= n_steps:
        window = list(buffer)[:n_steps]
        vt = _compute_nstep_virtual_transition(window, gamma)
        output.states.append(vt["s"])
        output.actions.append(vt["a"])
        output.rewards.append(vt["r"])
        output.next_states.append(vt["s_next"])
        output.dones.append(vt["done"])
        output.k_steps.append(vt["k"])
        output.masks.append(vt["mask"])
        buffer.popleft()
        count += 1

    # Flush: drain remaining entries with shorter windows
    if flush:
        while len(buffer) > 0:
            window = list(buffer)
            vt = _compute_nstep_virtual_transition(window, gamma)
            output.states.append(vt["s"])
            output.actions.append(vt["a"])
            output.rewards.append(vt["r"])
            output.next_states.append(vt["s_next"])
            output.dones.append(vt["done"])
            output.k_steps.append(vt["k"])
            output.masks.append(vt["mask"])
            buffer.popleft()
            count += 1

    return count


# ============================================================
# A2C Batch Update Function
# ============================================================

def actor_critic_update_batch(
    actor_net: nn.Module,
    critic_net: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: ACTransitionBatch,
    gamma: float,
    entropy_coef: float,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    critic_loss_coef: float = 0.5,
    target_critic_net: Optional[nn.Module] = None,
    adv_normalize_mode: str = "std_only",
) -> Dict[str, float]:
    """
    Advantage Actor-Critic batch update over collected SMDP transitions.

    Algorithm
    ---------
    For each transition (s, a, R_cum, s', done, K, mask):

        1. Critic forward pass:
               V(s)  = critic_net(s)
               V(s') = target_critic_net(s') if available, else critic_net(s')
                       (0 if done)

        2. TD target (SMDP-correct discounting):
               TD_target = R_cum + gamma^K · V(s')

        3. Advantage:
               A = TD_target - V(s)

        4. Critic loss:
               L_critic = Huber(V(s), TD_target.detach())

           Why Huber (Smooth L1) instead of MSE?
               Huber loss is linear for |error| > 1 and quadratic for
               |error| <= 1, making it robust to large TD errors from
               outlier transitions (sudden fills, inventory spikes).
               MSE would amplify these outliers quadratically.

           Why detach TD_target?
               This is standard semi-gradient TD learning.  The TD target
               is treated as a FIXED target (like supervised learning),
               not differentiated through.  Full-gradient TD can diverge
               with function approximation (Baird's counterexample).

        5. Actor loss:
               L_actor = -log π(a|s) · A.detach() - entropy_coef · H(π)

           Why detach advantage?
               The advantage A depends on V(s) through the critic.
               If we don't detach, actor gradients would flow into the
               critic (since V(s) is in the computation graph).  With
               separate optimizers, this would mean the actor_optimizer
               is modifying critic weights — a subtle but severe bug.

        6. Entropy regularization:
               H(π) = -Σ_a π(a|s) · log π(a|s)

           Encourages exploration by penalizing peaked distributions.
           The negative sign in the loss means we MAXIMIZE entropy.

    Action Masking (P1 Fix)
    -----------------------
    When computing log π(a|s) in the update, we must reapply the same
    valid action mask that was used during action sampling.  Without
    this, the softmax denominator includes mass on invalid actions,
    making log π_update(a|s) ≠ log π_collect(a|s).

    Parameters
    ----------
    actor_net : nn.Module
        Actor network π(a|s).
    critic_net : nn.Module
        Critic network V(s).
    actor_optimizer, critic_optimizer : torch.optim.Optimizer
        Independent optimizers for actor and critic.
    batch : ACTransitionBatch
        Collected transitions from one or more environments.
    gamma : float
        Discount factor.
    entropy_coef : float
        Weight of the entropy regularization term.
    device : torch.device
        Compute device (CPU or CUDA).
    grad_clip_norm : float
        Maximum L2 norm for gradient clipping (both networks).
    critic_loss_coef : float
        Scaling factor for critic loss (0.5 is standard in A2C).
        This prevents the critic's larger gradients from dominating
        when using a shared optimizer (kept for flexibility, though
        we use separate optimizers by default).
    target_critic_net : nn.Module or None
        Target critic for stable bootstrapping.  If provided, V(s')
        is computed from this slowly-updated copy instead of critic_net.
        This prevents the moving-target problem where critic updates
        destabilize TD targets.  Updated via Polyak averaging externally.
    adv_normalize_mode : str
        How to normalize advantages before the policy gradient:
        - "std_only": A / (std + eps) — preserves sign, prevents NaN.
          Recommended: rare positive advantages (fill events) are not
          neutralized by mean subtraction.
        - "full": (A - mean) / (std + eps) — original normalization.
          Destroys absolute signal; kept for ablation / backward compat.
        - "none": raw advantages, no normalization.

    Returns
    -------
    stats : dict
        Scalar statistics for logging:
            actor_loss, critic_loss, mean_entropy, mean_advantage, mean_value
    """
    n = len(batch)
    if n == 0:
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_advantage": 0.0,
            "mean_value": 0.0,
        }

    # ----------------------------------------------------------------
    # Build batch tensors
    # ----------------------------------------------------------------
    # States were stored on CPU in learn() to save GPU memory during
    # long episodes.  Move them to the compute device for the update.
    states_batch = torch.stack(batch.states, dim=0).to(device)          # (N, D)
    next_states_batch = torch.stack(batch.next_states, dim=0).to(device)  # (N, D)
    actions_batch = torch.tensor(
        batch.actions, dtype=torch.long, device=device
    ).view(-1, 1)                                                         # (N, 1)
    rewards_batch = torch.tensor(
        batch.rewards, dtype=torch.float32, device=device
    ).view(-1, 1)                                                         # (N, 1)
    dones_batch = torch.tensor(
        batch.dones, dtype=torch.float32, device=device
    ).view(-1, 1)                                                         # (N, 1)

    # SMDP holding times for correct discounting: gamma^K per transition
    k_steps_batch = torch.tensor(
        batch.k_steps, dtype=torch.float32, device=device
    ).view(-1, 1)                                                         # (N, 1)

    # ----------------------------------------------------------------
    # Critic forward pass: V(s) and V(s')
    # ----------------------------------------------------------------
    values = critic_net(states_batch)                          # (N, 1)
    with torch.no_grad():
        # Use target critic for V(s') if available (stable bootstrap).
        # The target network is a Polyak-averaged copy of the critic
        # that moves slowly, preventing the moving-target problem
        # where critic updates destabilize TD targets.
        bootstrap_net = target_critic_net if target_critic_net is not None else critic_net
        next_values = bootstrap_net(next_states_batch)         # (N, 1)
        # Zero out V(s') for terminal transitions
        next_values = next_values * (1.0 - dones_batch)

    # ----------------------------------------------------------------
    # TD target with SMDP-correct discounting
    # ----------------------------------------------------------------
    # TD_target = R_cum + gamma^K · V(s')
    #
    # gamma^K accounts for the variable holding time of each SMDP
    # transition.  Without this, the bootstrap would use gamma^1
    # regardless of how many micro-steps elapsed — under-discounting
    # future values when K > 1.
    gamma_k = gamma ** k_steps_batch                           # (N, 1)
    td_targets = rewards_batch + gamma_k * next_values         # (N, 1)

    # ----------------------------------------------------------------
    # Advantage:  Â_i = y_i − V_ϕ(s_i),  then normalize
    # ----------------------------------------------------------------
    # The advantage is DETACHED from the computation graph.  This is
    # critical: Â depends on V_ϕ(s), but the actor gradient must NOT
    # flow into the critic.  With separate optimisers, allowing such
    # cross-flow means actor_optimizer would modify critic weights —
    # a subtle but severe bug.
    #
    # Normalisation modes:
    #
    #   "std_only":  Â_i ← Â_i / (σ_Â + ε)
    #       Preserves the mean and sign of advantages.  In reward-sparse
    #       market-making, ~95% of transitions have negative advantage
    #       (no fill, only −φ·inv² penalty).  std-only keeps them
    #       negative; the rare fill transitions with Â > 0 keep their
    #       sign → the gradient correctly reinforces spread capture.
    #
    #   "full":  Â_i ← (Â_i − μ_Â) / (σ_Â + ε)
    #       Standard (A − mean) / std.  Destroys absolute signal:
    #       mean subtraction flips the "least bad" negative advantages
    #       to positive, causing the gradient to reinforce penalty-
    #       minimising actions instead of spread-capturing ones.
    #       Kept for backward compatibility / ablation only.
    #
    #   "none":  No normalisation (raw advantages).
    #       May cause instability if advantage magnitudes vary widely.
    advantages = (td_targets - values).detach()                # (N, 1)
    if advantages.numel() > 1:
        adv_std = advantages.std() + 1e-8
        if adv_normalize_mode == "std_only":
            advantages = advantages / adv_std
        elif adv_normalize_mode == "full":
            advantages = (advantages - advantages.mean()) / adv_std
        # else: "none" — raw advantages, no normalization

    # ----------------------------------------------------------------
    # Critic loss: Huber / Smooth L1 (V(s), TD_target)
    # ----------------------------------------------------------------
    # Semi-gradient: TD_target is detached (treated as a fixed target).
    # Huber loss is linear for |error| > 1 and quadratic for |error| <= 1,
    # making it robust to large TD errors from outlier transitions
    # (e.g. sudden fills or inventory spikes).  MSE would amplify these
    # outliers quadratically, destabilizing the critic early in training.
    critic_loss = F.smooth_l1_loss(values, td_targets.detach())

    # ----------------------------------------------------------------
    # Actor forward pass: log π(a|s) with action masking (P1 fix)
    # ----------------------------------------------------------------
    logits = actor_net(states_batch)                            # (N, n_actions)

    # P1 FIX: Reapply the valid action mask that was used during
    # action sampling.  Without this, the log_softmax denominator
    # includes probability mass on invalid actions, making
    # log π_update(a|s) ≠ log π_collect(a|s).
    has_masks = any(m is not None for m in batch.masks)
    if has_masks:
        n_act = logits.shape[-1]
        mask_np = np.stack([
            m if m is not None else np.ones(n_act, dtype=bool)
            for m in batch.masks
        ], axis=0)  # (N, n_actions)
        mask_tensor = torch.tensor(
            mask_np, dtype=torch.bool, device=logits.device
        )
        logits = logits.masked_fill(~mask_tensor, float("-inf"))

    log_probs_all = F.log_softmax(logits, dim=-1)              # (N, n_actions)
    probs_all = torch.exp(log_probs_all)                       # (N, n_actions)

    # Log-probability of the chosen action: (N, 1)
    action_log_probs = log_probs_all.gather(1, actions_batch)

    # Entropy: compute safely to avoid NaN gradient from 0 * (-inf).
    # Instead of probs * log_probs (which has NaN gradient at masked
    # positions), use -log_softmax directly with clamped probs.
    probs_clamped = probs_all.clamp(min=1e-8)
    entropy = -torch.sum(probs_clamped * probs_clamped.log(), dim=-1, keepdim=True)

    # ----------------------------------------------------------------
    # Actor loss:  L_actor = −(1/B) Σ_i [ log π_θ(a_i|s_i) · sg(Â_i)
    #                                      + η · H(π_θ(·|s_i)) ]
    # ----------------------------------------------------------------
    # The policy gradient theorem (Williams, 1992; Sutton et al., 2000)
    # states that the direction of steepest ascent for J(θ) = E[Σ γ^t r_t]
    # is:
    #     ∇_θ J = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · Â_t ]
    #
    # The entropy bonus −η · H(π_θ) = −η · (−Σ_a π_θ log π_θ) prevents
    # premature collapse to a deterministic policy by penalising low-entropy
    # distributions.  As a regulariser, η is annealed from a high initial
    # value to a low final value over the course of training.
    #
    # The negative sign converts maximisation of J into minimisation of
    # L_actor (standard PyTorch convention: optimiser.step() descends).
    actor_loss = (-action_log_probs * advantages - entropy_coef * entropy).mean()

    # ----------------------------------------------------------------
    # Backward passes (separate optimisers, sequential)
    # ----------------------------------------------------------------
    # The critic is updated FIRST so that V_ϕ improves before the actor
    # sees the next batch.  This follows the two-timescale principle
    # (Konda & Tsitsiklis, 2003): a fast critic and a slow actor converge
    # to a local optimum of J(θ) under mild conditions.
    #
    # Gradient clipping (max L2 norm) prevents catastrophic parameter
    # jumps from outlier batches — especially important early in training
    # when TD errors can be large.

    # Critic update:  ∇_ϕ [ c_v · Huber(V_ϕ(s), sg(y)) ]
    critic_optimizer.zero_grad()
    (critic_loss_coef * critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=grad_clip_norm)
    critic_optimizer.step()

    # Actor update:  ∇_θ L_actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=grad_clip_norm)
    actor_optimizer.step()

    # ----------------------------------------------------------------
    # Statistics for logging
    # ----------------------------------------------------------------
    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "mean_entropy": float(entropy.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_value": float(values.mean().item()),
    }


# ============================================================
# Actor-Critic Controller with Throttle + SMDP
# ============================================================

class ActorCriticController(RLController):
    """
    Advantage Actor-Critic (A2C) controller for the MarketMaker.

    This controller implements A2C/A3C with:
        - Separate actor π(a|s) and critic V(s) networks
        - Entropy regularization
        - Three independent throttle gates (TOB, Event, Time)
        - SMDP reward aggregation between decisions
        - enable_learning toggle for train/eval mode
        - Checkpoint save/load
        - Separate LR scheduling for actor and critic

    Structural Modes
    ----------------
    1) Generic mode (pure_mm=False):
        - 6 discrete actions (post_bid, post_ask, post_bid_ask,
          cancel_bid, cancel_ask, hold). Optional extensions >= 7.
        - Inventory band constraints.

    2) Pure MM mode (pure_mm=True):
        - Each action selects an offset pair (delta_bid, delta_ask).
        - Inventory band controls single-sided quoting.

    Update Flow
    -----------
    The controller's learn() method handles SMDP aggregation: it
    accumulates rewards during throttled micro-steps and commits
    completed transitions to episode buffers when a new decision
    arrives or the episode ends.

    Gradient updates are handled externally by:
        - run_a2c_sync_batch(): Synchronous A2C — collects transitions
          from N envs in lockstep, then calls actor_critic_update_batch().
        - run_a3c_async_batch(): Asynchronous A3C — each env runs in
          its own thread, doing per-transition updates with a grad lock.

    Usage Pattern
    -------------
    The generator calls the controller at every micro-step:

        action = controller.act(state_before)
        # ... environment processes action ...
        controller.learn(step_idx, mm, lob, state_before, state_after, reward, info)

    At episode end:

        controller.finish_episode(total_reward)

    When throttling is active:
        - act() may return ("hold",) without querying the network.
        - learn() accumulates rewards via SMDP aggregation.
        - Transitions are committed to episode buffers at decision points.
    """

    def __init__(
        self,
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.97,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        weight_decay: float = 0.01,
        entropy_coef: float = 0.01,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: str = "runs/ac_mm",
        # ----- MM PURE mode flags / config -----
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        # ----- Actor network architecture -----
        n_hidden_actor: int = 2,
        n_neurons_actor: int = 128,
        # ----- Critic network architecture -----
        n_hidden_critic: int = 2,
        n_neurons_critic: int = 128,
        # ----- Learning toggle -----
        enable_learning: bool = True,
        # ----- Throttle gating parameters -----
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_event_update: bool = False,
        n_events: int = 100,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,
        # ----- MDP mode (disable all bypass) -----
        use_mdp: bool = True,
        # ----- LR scheduling -----
        lr_scheduler_step_size: int = 100,
        lr_scheduler_gamma: float = 0.95,
        # ----- N-step TD -----
        n_steps: int = 5,
        # ----- A2C sync-step accumulation (A2C only) -----
        n_steps_sync: int = 1,
        # ----- Critic loss coefficient -----
        critic_loss_coef: float = 0.5,
        # ----- Synchronous A2C flag -----
        use_synchronous: bool = True,
        # ----- Target critic (stable bootstrap) -----
        use_target_critic: bool = True,
        target_critic_tau: float = 0.005,
        # ----- Advantage normalization mode -----
        adv_normalize_mode: str = "std_only",
    ):
        # =============================================================
        # CORE HYPERPARAMETERS
        # =============================================================
        self.level_offset = int(level_offset)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.critic_loss_coef = float(critic_loss_coef)
        self.use_synchronous = bool(use_synchronous)

        # N-step TD returns.  With n_steps=1, training uses standard 1-step
        # TD targets.  With n_steps>1, a sliding window buffer in the
        # training functions (run_a2c_sync_batch / run_a3c_async_batch)
        # pre-computes N-step SMDP returns as virtual transitions.
        self.n_steps = max(1, int(n_steps))
        self.n_steps_sync = max(1, int(n_steps_sync))
        _valid_adv_modes = ("std_only", "full", "none")
        if adv_normalize_mode not in _valid_adv_modes:
            raise ValueError(
                f"adv_normalize_mode={adv_normalize_mode!r} invalid. "
                f"Expected one of {_valid_adv_modes}."
            )
        self.adv_normalize_mode = str(adv_normalize_mode)

        # Network architecture parameters
        self.actor_n_hidden = int(n_hidden_actor)
        self.actor_n_neurons = int(n_neurons_actor)
        self.critic_n_hidden = int(n_hidden_critic)
        self.critic_n_neurons = int(n_neurons_critic)

        # =============================================================
        # PURE-MM CONFIGURATION
        # =============================================================
        self.pure_mm = bool(pure_mm)
        self.inv_limit = None if inv_limit is None else int(inv_limit)

        if self.pure_mm:
            if pure_mm_offsets is None:
                self.pure_mm_offsets: List[Tuple[int, int]] = [
                    (0, 0),   # BID at best_bid, ASK at best_ask
                    (0, 1),   # BID at best_bid, ASK at best_ask + 1
                    (1, 0),   # BID at best_bid - 1, ASK at best_ask
                    (1, 1),   # BID at best_bid - 1, ASK at best_ask + 1
                ]
            else:
                self.pure_mm_offsets = [
                    (int(x[0]), int(x[1])) for x in list(pure_mm_offsets)
                ]

            if len(self.pure_mm_offsets) == 0:
                raise ValueError(
                    "pure_mm_offsets cannot be empty: need at least one "
                    "(bid_offset, ask_offset) pair to define the action space."
                )

            self.n_actions = len(self.pure_mm_offsets)

            # Infer max non-negative offset for state vector dimensioning
            bid_offs = [bo for (bo, _) in self.pure_mm_offsets]
            ask_offs = [ao for (_, ao) in self.pure_mm_offsets]
            max_bid_offset = max([bo for bo in bid_offs if bo >= 0], default=0)
            max_ask_offset = max([ao for ao in ask_offs if ao >= 0], default=0)
            self.max_offset = int(max(max_bid_offset, max_ask_offset))

            # ---------------------------------------------------------
            # CANONICAL EQUIVALENCE MASKS (for inventory-limit states)
            # ---------------------------------------------------------
            _canon_ask: Dict[int, int] = {}
            _canon_bid: Dict[int, int] = {}
            for idx, (bo, ao) in enumerate(self.pure_mm_offsets):
                if ao not in _canon_ask:
                    _canon_ask[ao] = idx
                if bo not in _canon_bid:
                    _canon_bid[bo] = idx

            _canonical_at_long = [
                _canon_ask[ao] for (_, ao) in self.pure_mm_offsets
            ]
            _canonical_at_short = [
                _canon_bid[bo] for (bo, _) in self.pure_mm_offsets
            ]

            self._canonical_mask_long = np.array(
                [_canonical_at_long[i] == i for i in range(self.n_actions)],
                dtype=bool,
            )
            self._canonical_mask_short = np.array(
                [_canonical_at_short[i] == i for i in range(self.n_actions)],
                dtype=bool,
            )
        else:
            # Generic mode
            self.pure_mm_offsets = None
            self.max_offset = 0

            self.n_actions = int(n_actions)
            if self.n_actions < 6:
                print(
                    "[ActorCriticController] WARNING: pure_mm=False but "
                    f"n_actions < 6 (n_actions={self.n_actions}). "
                    "Overriding to 6 generic actions."
                )
                self.n_actions = 6
            elif self.n_actions > 9:
                print(
                    f"[ActorCriticController] WARNING: n_actions={self.n_actions} "
                    "exceeds the 9 defined generic actions. Clamping to 9."
                )
                self.n_actions = 9
            elif self.n_actions > 6:
                print(
                    "[ActorCriticController] INFO: pure_mm=False with "
                    f"extended actions (n_actions={self.n_actions})."
                )

        # =============================================================
        # DEVICE SELECTION
        # =============================================================
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # =============================================================
        # ACTOR NETWORK π(a|s)
        # =============================================================
        if self.pure_mm:
            input_dim = 2 + 2 * (self.max_offset + 1)
        else:
            input_dim = 6

        self.actor_net = ActorNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            n_hidden=self.actor_n_hidden,
            n_neurons=self.actor_n_neurons,
        ).to(self.device)

        # =============================================================
        # CRITIC NETWORK V(s)
        # =============================================================
        self.critic_net = CriticNetwork(
            input_dim=input_dim,
            n_hidden=self.critic_n_hidden,
            n_neurons=self.critic_n_neurons,
        ).to(self.device)

        # =============================================================
        # ENABLE LEARNING TOGGLE
        # =============================================================
        self._enable_learning = bool(enable_learning)
        if self._enable_learning:
            self.actor_net.train()
            self.critic_net.train()
        else:
            self.actor_net.eval()
            self.critic_net.eval()

        # =============================================================
        # TARGET CRITIC NETWORK (stable bootstrap)
        # =============================================================
        # A slowly-updated copy of the critic used for V(s') in the TD
        # target.  Prevents the moving-target problem: without it, every
        # critic gradient step shifts V(s'), which shifts the advantage,
        # which shifts the actor gradient — a circular instability.
        # DQN uses the same principle (hard sync every 2000 steps); here
        # we use Polyak soft update (smoother).
        self.use_target_critic = bool(use_target_critic)
        self.target_critic_tau = float(target_critic_tau)

        if self.use_target_critic:
            self.target_critic_net = copy.deepcopy(self.critic_net)
            self.target_critic_net.eval()
            for p in self.target_critic_net.parameters():
                p.requires_grad_(False)
        else:
            self.target_critic_net = None

        # =============================================================
        # SEPARATE OPTIMIZERS + LR SCHEDULERS
        #
        # Why separate optimizers?
        #   The critic solves a supervised regression problem
        #   (minimize MSE(V(s), TD_target)) which typically converges
        #   faster than the actor's policy gradient.  A higher LR for
        #   the critic (e.g., 1e-3 vs 1e-4 for the actor) allows the
        #   value function to stabilize early, providing more accurate
        #   advantages for the actor.
        # =============================================================
        self.actor_optimizer = AdamW(
            self.actor_net.parameters(), lr=lr_actor, weight_decay=self.weight_decay,
        )
        self.critic_optimizer = AdamW(
            self.critic_net.parameters(), lr=lr_critic, weight_decay=self.weight_decay,
        )

        self.lr_scheduler_step_size = int(lr_scheduler_step_size)
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)

        self.actor_lr_scheduler = StepLR(
            self.actor_optimizer,
            step_size=self.lr_scheduler_step_size,
            gamma=self.lr_scheduler_gamma,
        )
        self.critic_lr_scheduler = StepLR(
            self.critic_optimizer,
            step_size=self.lr_scheduler_step_size,
            gamma=self.lr_scheduler_gamma,
        )

        # =============================================================
        # EPISODE BUFFERS
        #
        # These store the SMDP-level transitions for the current episode.
        # Populated by learn() as it commits completed SMDP transitions.
        # Consumed externally by run_a2c_sync_batch / run_a3c_async_batch
        # after each decision point, then cleared.
        # =============================================================
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_next_states: List[torch.Tensor] = []
        self.episode_dones: List[bool] = []
        self.episode_k_steps: List[int] = []
        self.episode_masks: List[Optional[np.ndarray]] = []

        # Last valid mask (set by act(), stored in SMDP pending by learn())
        self._last_valid_mask: Optional[np.ndarray] = None

        # Last action index (set by act(), read by learn())
        self.last_action_idx: Optional[int] = None

        # =============================================================
        # THROTTLE GATE CONFIGURATION
        # =============================================================
        self.use_tob_update = bool(use_tob_update)
        self.threshold_tob = max(1, int(n_tob_moves))

        self.use_event_update = bool(use_event_update)
        self.threshold_events = max(1, int(n_events))

        self.use_time_update = bool(use_time_update)
        self.min_time_interval = float(min_time_interval)

        self.use_mdp = bool(use_mdp)

        # Throttle counters
        self.moves_tob: int = 0
        self.last_env_tob_key: Optional[Tuple[int, int, int, int]] = None
        self.event_steps: int = 0
        self.last_update_time: float = -1.0

        # First-action bypass flag
        self._has_acted_once: bool = False

        # Bypass trackers
        self.last_inventory: Optional[int] = None
        self.last_mode: Optional[str] = None

        # =============================================================
        # SMDP STATE
        # =============================================================
        self._smdp_pending: bool = False
        self._smdp_s_decision: Optional[Dict[str, Any]] = None
        self._smdp_a_decision: Optional[int] = None
        self._smdp_mask_decision: Optional[np.ndarray] = None
        self._smdp_cum_reward: float = 0.0
        self._smdp_k_steps: int = 0
        self._last_was_decision: bool = False

        # =============================================================
        # TENSORBOARD WRITER + EPISODE COUNTER
        # =============================================================
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None  # Workers don't write to TensorBoard
        self.episode_idx: int = 0

        # =============================================================
        # SETUP PRINTOUT (master controller only — workers are silent)
        # =============================================================
        if log_dir is not None:
            print("\n================ A2C CONTROLLER SETUP ================")
            print(f"pure_mm            : {self.pure_mm}")
            print(f"inv_limit          : {self.inv_limit}")
            if self.pure_mm:
                print(f"pure_mm_offsets    : {self.pure_mm_offsets}")
                print(f"max_offset         : {self.max_offset}")
                print(f"input_dim          : {input_dim}")

            print("\n--- Throttle Configuration ---")
            print(f"Time Gating        : {self.use_time_update} "
                  f"(min {self.min_time_interval}s)")
            print(f"Event Gating       : {self.use_event_update} "
                  f"(min {self.threshold_events} steps)")
            print(f"TOB Gating         : {self.use_tob_update} "
                  f"(min {self.threshold_tob} moves)")
            print(f"MDP Mode           : {self.use_mdp}")

            print("\n--- Core Hyperparameters ---")
            print(f"gamma              : {self.gamma}")
            print(f"lr_actor           : {lr_actor}")
            print(f"lr_critic          : {lr_critic}")
            print(f"entropy_coef       : {self.entropy_coef}")
            print(f"n_steps            : {self.n_steps}")
            print(f"n_steps_sync       : {self.n_steps_sync}")
            print(f"enable_learning    : {self._enable_learning}")
            print(f"device             : {self.device}")

            print("\n--- Actor Architecture ---")
            print(f"n_hidden_actor     : {self.actor_n_hidden}")
            print(f"n_neurons_actor    : {self.actor_n_neurons}")

            print("\n--- Critic Architecture ---")
            print(f"n_hidden_critic    : {self.critic_n_hidden}")
            print(f"n_neurons_critic   : {self.critic_n_neurons}")

            print(f"\nn_actions          : {self.n_actions}")
            print(f"weight_decay       : {self.weight_decay}")
            print(f"grad_clip_norm     : {self.grad_clip_norm}")
            print(f"critic_loss_coef   : {self.critic_loss_coef}")
            print("======================================================\n")

    # ================================================================
    # WORKER FACTORY (for synchronous A2C with shared networks)
    # ================================================================

    def make_worker(self) -> "ActorCriticController":
        """
        Create a lightweight worker controller that shares this controller's
        neural networks, optimizers, and LR schedulers.

        Purpose
        -------
        In synchronous A2C training, N environments must each have their own
        controller instance (for independent SMDP state tracking, throttle
        counters, and episode buffers), but all controllers must share the
        **same** actor and critic networks so that gradient updates from any
        worker immediately affect all others.

        This method creates such a worker by:

            1. Constructing a new ``ActorCriticController`` with identical
               hyperparameters (gamma, entropy_coef, pure_mm config, throttle
               settings, etc.).
            2. **Replacing** the worker's networks, optimizers, and LR schedulers
               with references to this (master) controller's objects.

        Shared vs Independent State
        ---------------------------
        **SHARED** (same Python objects as master):
            - ``actor_net``, ``critic_net`` — weight tensors
            - ``target_critic_net`` — Polyak-averaged critic (read-only)
            - ``actor_optimizer``, ``critic_optimizer`` — optimizer state
            - ``actor_lr_scheduler``, ``critic_lr_scheduler`` — LR schedules

        **INDEPENDENT** (fresh per worker):
            - SMDP aggregation state (``_smdp_pending``, ``_smdp_cum_reward``, ...)
            - Throttle counters (``event_steps``, ``moves_tob``, ...)
            - Episode buffers (``episode_states``, ``episode_actions``, ...)
            - TensorBoard writer (``None`` — workers do not log)

        Returns
        -------
        worker : ActorCriticController
            A controller instance ready to be passed to
            ``simulate_LOB_with_MM_generator`` as the ``controller`` argument.

        Example
        -------
        >>> master = ActorCriticController(...)
        >>> workers = [master.make_worker() for _ in range(N_ENVS)]
        >>> assert workers[0].actor_net is master.actor_net  # Same object
        """
        worker = ActorCriticController(
            level_offset=self.level_offset,
            n_actions=self.n_actions,
            gamma=self.gamma,
            lr_actor=self.actor_optimizer.param_groups[0]["lr"],
            lr_critic=self.critic_optimizer.param_groups[0]["lr"],
            weight_decay=self.weight_decay,
            entropy_coef=self.entropy_coef,
            grad_clip_norm=self.grad_clip_norm,
            device=self.device,
            log_dir=None,  # Workers don't write to TensorBoard
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
            lr_scheduler_step_size=self.lr_scheduler_step_size,
            lr_scheduler_gamma=self.lr_scheduler_gamma,
            n_steps=self.n_steps,
            n_steps_sync=self.n_steps_sync,
            critic_loss_coef=self.critic_loss_coef,
            use_synchronous=self.use_synchronous,
            # Workers don't own the target critic — they share the master's.
            # Pass False to avoid creating an unnecessary deepcopy.
            use_target_critic=False,
            adv_normalize_mode=self.adv_normalize_mode,
        )

        # Replace worker's networks with shared references to master's.
        # All workers and the master share the SAME weight tensors, so a
        # gradient update through any optimizer modifies all of them.
        worker.actor_net = self.actor_net
        worker.critic_net = self.critic_net
        worker.actor_optimizer = self.actor_optimizer
        worker.critic_optimizer = self.critic_optimizer
        worker.actor_lr_scheduler = self.actor_lr_scheduler
        worker.critic_lr_scheduler = self.critic_lr_scheduler
        # Share the target critic (read-only; updated only by master).
        worker.target_critic_net = self.target_critic_net

        return worker

    # ================================================================
    # TARGET CRITIC: POLYAK SOFT UPDATE
    # ================================================================

    def _soft_update_target(self) -> None:
        """
        Polyak averaging (exponential moving average) of critic weights.

        After each gradient step on V_ϕ, update the target critic:

            ϕ̄  ←  τ · ϕ  +  (1 − τ) · ϕ̄

        with τ ≪ 1 (e.g. 0.005).  This ensures V̄_ϕ̄ tracks V_ϕ slowly,
        providing a stable bootstrap target for the TD error.  Without
        this, every ∇_ϕ step shifts V(s') in the target, which shifts Â,
        which shifts the actor gradient — creating a destabilising
        circular dependency.

        The Polyak averaging scheme (Polyak & Juditsky, 1992) is smoother
        than DQN's hard target sync (copy every C steps) and avoids the
        discontinuity at sync boundaries.  Lillicrap et al. (2016, DDPG)
        popularised this "soft update" approach for continuous control.
        """
        if self.target_critic_net is None:
            return
        tau = self.target_critic_tau
        for tp, sp in zip(
            self.target_critic_net.parameters(),
            self.critic_net.parameters(),
        ):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    # ================================================================
    # STATE CONVERSION
    # ================================================================

    def _state_to_tensor(self, s: Dict[str, Any]) -> torch.Tensor:
        """
        Convert the MarketMaker state dictionary into a (1, D) float tensor.

        Identical to PolicyGradientController._state_to_tensor() from
        reinforce_general.py.  Uses the same normalization strategy:
            - Spread: log1p(max(0, spread))
            - Volumes: log1p(volume)
            - Inventory: inventory / inv_denom
        """
        inv_denom = float(self.inv_limit) if self.inv_limit is not None else 10.0
        inv_denom = max(inv_denom, 1.0)

        if not self.pure_mm:
            # GENERIC MODE: [spread, asksize, bidsize, inventory, has_bid, has_ask]
            raw_spread = float(s["spread"])
            if raw_spread <= -1.0:
                print(
                    "[BAD SPREAD]", raw_spread,
                    "best_bid", s.get("best_bid"),
                    "best_ask", s.get("best_ask"),
                    "mid", s.get("mid"),
                )
            spread = math.log1p(max(0.0, raw_spread))
            asksize = math.log1p(max(0.0, float(s.get("asksize", 0.0))))
            bidsize = math.log1p(max(0.0, float(s.get("bidsize", 0.0))))
            raw_inv = float(s["inventory"])
            inventory = raw_inv / inv_denom
            has_bid = 1.0 if bool(s.get("has_bid", False)) else 0.0
            has_ask = 1.0 if bool(s.get("has_ask", False)) else 0.0

            arr = np.array(
                [spread, asksize, bidsize, inventory, has_bid, has_ask],
                dtype=np.float32,
            )
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        # PURE MM MODE: [spread, inventory, bid_sizes[0..K], ask_sizes[0..K]]
        raw_spread = float(s["spread"])
        spread = math.log1p(max(0.0, raw_spread))
        raw_inv = float(s["inventory"])
        inventory = raw_inv / inv_denom

        K = int(self.max_offset)

        bid_sizes_seq = s.get("pure_mm_bid_sizes", None)
        ask_sizes_seq = s.get("pure_mm_ask_sizes", None)

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

        if len(bid_sizes) < K + 1:
            bid_sizes += [0.0] * (K + 1 - len(bid_sizes))
        if len(ask_sizes) < K + 1:
            ask_sizes += [0.0] * (K + 1 - len(ask_sizes))

        bid_sizes = [math.log1p(max(0.0, x)) for x in bid_sizes[: K + 1]]
        ask_sizes = [math.log1p(max(0.0, x)) for x in ask_sizes[: K + 1]]

        arr = np.array(
            [spread, inventory] + bid_sizes + ask_sizes,
            dtype=np.float32,
        )
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    # ================================================================
    # THROTTLE HELPERS
    # ================================================================

    def _get_env_tob(self, s: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Build a Top-Of-Book fingerprint from the state dictionary.
        Identical to PolicyGradientController._get_env_tob().
        """
        def _safe_int(x, d: int = -1) -> int:
            try:
                return int(x)
            except Exception:
                return d

        def _safe_float(x, d: float = 0.0) -> float:
            try:
                v = float(x)
                return v if math.isfinite(v) else d
            except Exception:
                return d

        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try:
            ibs, ias = int(round(env_bs)), int(round(env_as))
        except Exception:
            ibs, ias = 0, 0
        return (env_bb, env_ba, ibs, ias)

    def _reset_clocks(
        self,
        current_time: float,
        tob_key: Tuple[int, int, int, int],
    ) -> None:
        """
        Reset all throttle gate counters after making a decision.
        Identical to PolicyGradientController._reset_clocks().
        """
        self._has_acted_once = True

        if self.use_tob_update:
            self.moves_tob = 0
            self.last_env_tob_key = tob_key

        if self.use_event_update:
            self.event_steps = 0

        if self.use_time_update:
            self.last_update_time = current_time

    # ================================================================
    # SMDP HELPERS
    # ================================================================

    def _smdp_reset(self) -> None:
        """
        Clear all SMDP aggregation state.

        Called at the start of each new episode and after episode
        termination cleanup.
        """
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_mask_decision = None
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0
        self._last_was_decision = False

    def _reset_throttle_state(self) -> None:
        """
        P2 FIX: Reset all throttle gate counters and bypass trackers
        to their initial (start-of-episode) values.

        Without this reset, from episode 2 onward:
          - _has_acted_once remains True, so the first-action bypass
            never fires.
          - event_steps / moves_tob / last_update_time carry over.
          - last_inventory / last_mode carry stale values.
        """
        self._has_acted_once = False
        self.event_steps = 0
        self.moves_tob = 0
        self.last_env_tob_key = None
        self.last_update_time = -1.0
        self.last_inventory = None
        self.last_mode = None

    # ================================================================
    # ACTION MASK
    # ================================================================

    def _get_valid_action_mask(self, mm_state: dict) -> np.ndarray:
        """
        Compute a boolean mask indicating which actions are permitted
        given the current inventory constraints.

        Identical to PolicyGradientController._get_valid_action_mask().
        """
        mask = np.ones(self.n_actions, dtype=bool)

        if self.inv_limit is None:
            return mask

        inv = float(mm_state.get("inventory", 0.0))

        # PURE MM mode — canonical degenerate masking
        if self.pure_mm:
            if inv >= self.inv_limit:
                mask = self._canonical_mask_long.copy()
            elif inv <= -self.inv_limit:
                mask = self._canonical_mask_short.copy()
            return mask

        # Generic mode — block actions that would increase the problematic
        # inventory direction.  Also block both cancellations: the filled
        # side has no order left (no-op), and the opposite side holds the
        # hedge that helps reduce inventory back toward zero.
        if inv >= self.inv_limit:
            mask[0] = False   # post_bid
            mask[2] = False   # post_bid_ask
            mask[3] = False   # cancel_bid  (no order — no-op)
            mask[4] = False   # cancel_ask  (protects hedge)
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside
            if self.n_actions >= 8:
                mask[7] = False   # bid_inside

        elif inv <= -self.inv_limit:
            mask[1] = False   # post_ask
            mask[2] = False   # post_bid_ask
            mask[3] = False   # cancel_bid  (protects hedge)
            mask[4] = False   # cancel_ask  (no order — no-op)
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside
            if self.n_actions >= 9:
                mask[8] = False   # ask_inside

        return mask

    # ================================================================
    # ACTION SAMPLING
    # ================================================================

    def _sample_action_idx(
        self,
        state_tensor: torch.Tensor,
        valid_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select an action index from the current actor policy π(.|s),
        constrained by the optional valid_mask.

        Training: sample from softmax(masked_logits).
        Evaluation: argmax over masked_logits.
        """
        with torch.no_grad():
            logits = self.actor_net(state_tensor)

            if valid_mask is not None:
                mask_tensor = torch.tensor(
                    valid_mask, dtype=torch.bool, device=logits.device
                )
                logits = logits.masked_fill(~mask_tensor, float("-inf"))

            if not self._enable_learning:
                a_idx = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                # Safety: if NaN detected (corrupted weights), fall back to
                # uniform random over valid actions to avoid crashing.
                if torch.isnan(probs).any():
                    print("[WARNING] NaN in actor probs — falling back to uniform random.")
                    if valid_mask is not None:
                        valid_idx = np.where(valid_mask)[0]
                    else:
                        valid_idx = np.arange(self.n_actions)
                    a_idx = int(np.random.choice(valid_idx))
                else:
                    dist = torch.distributions.Categorical(probs=probs)
                    a_idx = int(dist.sample().item())

        return a_idx

    # ================================================================
    # PURE-MM ACTION MAPPING
    # ================================================================

    def _mm_action_from_idx_pure_mm(
        self, a_idx: int, state: Dict[str, Any]
    ) -> tuple:
        """
        Pure-MM mapping: discrete action index -> macro-action tuple.
        Identical to PolicyGradientController._mm_action_from_idx_pure_mm().
        """
        a_idx = int(a_idx)
        inv = float(state.get("inventory", 0.0))

        best_bid_raw = state.get("best_bid", None)
        best_ask_raw = state.get("best_ask", None)
        best_bid = int(best_bid_raw) if best_bid_raw is not None else None
        best_ask = int(best_ask_raw) if best_ask_raw is not None else None

        if best_bid is not None and best_bid < 0:
            best_bid = None
        if best_ask is not None and best_ask < 0:
            best_ask = None

        mid = float(state.get("mid", 0.0))
        mid_px = int(round(mid))

        spread = int(state.get("spread", 0))
        if best_bid is not None and best_ask is not None:
            spread = max(spread, best_ask - best_bid)

        bid_off, ask_off = self.pure_mm_offsets[a_idx]
        bid_off = int(bid_off)
        ask_off = int(ask_off)

        # Compute BID price
        if best_bid is not None and best_ask is not None:
            if bid_off == 0:
                bid_price = best_bid
            elif bid_off > 0:
                bid_price = best_bid - bid_off
            else:
                if spread >= 2:
                    candidate = best_bid - bid_off
                    candidate = min(candidate, best_ask - 1)
                    bid_price = best_bid if candidate <= best_bid else candidate
                else:
                    bid_price = best_bid
        elif best_bid is not None:
            bid_price = best_bid - bid_off if bid_off >= 0 else best_bid
        else:
            bid_price = mid_px - bid_off

        # Compute ASK price
        if best_bid is not None and best_ask is not None:
            if ask_off == 0:
                ask_price = best_ask
            elif ask_off > 0:
                ask_price = best_ask + ask_off
            else:
                if spread >= 2:
                    candidate = best_ask + ask_off
                    candidate = max(candidate, best_bid + 1)
                    ask_price = best_ask if candidate >= best_ask else candidate
                else:
                    ask_price = best_ask
        elif best_ask is not None:
            ask_price = best_ask + ask_off if ask_off >= 0 else best_ask
        else:
            ask_price = mid_px + ask_off

        bid_price = int(bid_price)
        ask_price = int(ask_price)

        # Inventory band logic
        if self.inv_limit is not None:
            if inv >= self.inv_limit:
                return ("cancel_all_then_place", -1, ask_price)
            if inv <= -self.inv_limit:
                return ("cancel_all_then_place", +1, bid_price)

        # Safety: enforce bid < ask
        if ask_price <= bid_price:
            if (best_ask is not None) and (best_ask > bid_price):
                ask_price = max(ask_price, best_ask, bid_price + 1)
            else:
                ask_price = bid_price + 1

        return ("cancel_all_then_place_bid_ask", bid_price, ask_price)

    # ================================================================
    # GENERIC ACTION MAPPING
    # ================================================================

    def _mm_action_from_idx_generic(
        self, a_idx: int, state: Dict[str, Any]
    ) -> tuple:
        """
        Generic mapping: discrete action index -> MarketMaker command tuple.
        Identical to PolicyGradientController._mm_action_from_idx_generic().
        """
        action_list = [
            "post_bid",      # 0
            "post_ask",      # 1
            "post_bid_ask",  # 2
            "cancel_bid",    # 3
            "cancel_ask",    # 4
            "hold",          # 5
        ]
        if self.n_actions >= 7:
            action_list += [
                "post_bid_ask_inside_spread",   # 6
                "post_bid_inside_spread",       # 7
                "post_ask_inside_spread",       # 8
            ]

        a_idx = int(a_idx)
        if a_idx >= len(action_list):
            a_idx = len(action_list) - 1
        a_name = action_list[a_idx]

        inv = float(state.get("inventory", 0.0))

        want_bid = False
        want_ask = False
        use_inside_bid = False
        use_inside_ask = False

        if a_name == "post_bid":
            want_bid = True
        elif a_name == "post_ask":
            want_ask = True
        elif a_name == "post_bid_ask":
            want_bid = True
            want_ask = True
        elif a_name == "post_bid_ask_inside_spread":
            want_bid = True
            want_ask = True
            use_inside_bid = True
            use_inside_ask = True
        elif a_name == "post_bid_inside_spread":
            want_bid = True
            use_inside_bid = True
        elif a_name == "post_ask_inside_spread":
            want_ask = True
            use_inside_ask = True

        # Inventory constraints
        if self.inv_limit is not None and (want_bid or want_ask):
            if inv >= self.inv_limit:
                want_bid = False
                use_inside_bid = False
            elif inv <= -self.inv_limit:
                want_ask = False
                use_inside_ask = False

        # NOTE: we intentionally do NOT suppress want_bid/want_ask when
        # has_bid/has_ask is True.  The MM's _smart_place() handles
        # repricing: it preserves queue priority if the price is unchanged,
        # or cancels and replaces if the target price differs.  Collapsing
        # to hold here would prevent the agent from ever updating its quotes.

        if "inside_spread" in a_name:
            if want_bid and want_ask and use_inside_bid and use_inside_ask:
                return ("place_bid_ask_inside_spread",)
            if want_bid and use_inside_bid and not want_ask:
                return ("place_bid_inside_spread",)
            if want_ask and use_inside_ask and not want_bid:
                return ("place_ask_inside_spread",)
            return ("hold",)

        best_bid_raw = state.get("best_bid", None)
        best_ask_raw = state.get("best_ask", None)
        best_bid = int(best_bid_raw) if best_bid_raw is not None else None
        best_ask = int(best_ask_raw) if best_ask_raw is not None else None

        if best_bid is not None and best_bid < 0:
            best_bid = None
        if best_ask is not None and best_ask < 0:
            best_ask = None

        mid = float(state.get("mid", 0.0))
        mid_px = int(round(mid))

        if best_bid is not None:
            bid_price_l1 = best_bid - self.level_offset
        else:
            bid_price_l1 = mid_px - max(1, self.level_offset)

        if best_ask is not None:
            ask_price_l1 = best_ask + self.level_offset
        else:
            ask_price_l1 = mid_px + max(1, self.level_offset)

        bid_price_l1 = int(bid_price_l1)
        ask_price_l1 = int(ask_price_l1)

        if a_name == "hold":
            return ("hold",)
        if a_name == "cancel_bid":
            return ("cancel_bid",)
        if a_name == "cancel_ask":
            return ("cancel_ask",)

        if want_bid and want_ask:
            if ask_price_l1 <= bid_price_l1:
                ask_price_l1 = bid_price_l1 + 1
            return ("place_bid_ask", bid_price_l1, ask_price_l1)
        if want_bid:
            return ("place_bid", bid_price_l1)
        if want_ask:
            return ("place_ask", ask_price_l1)

        return ("hold",)

    def _mm_action_from_idx(self, a_idx: int, state: Dict[str, Any]) -> tuple:
        """Dispatch to pure_mm or generic action mapping."""
        if self.pure_mm:
            return self._mm_action_from_idx_pure_mm(a_idx, state)
        else:
            return self._mm_action_from_idx_generic(a_idx, state)

    # ================================================================
    # RLController INTERFACE: act()
    # ================================================================

    def act(self, mm_state: Dict[str, Any]) -> tuple:
        """
        Select and return a MarketMaker command for the current state.

        This method is called EVERY micro-step by the simulation runner.
        When throttling is active, it decides whether to query the neural
        network for a new action (= decision) or to hold the current
        position (= throttled).

        The decision pipeline is identical to PolicyGradientController.act().
        """
        # STEP 1: UPDATE THROTTLE COUNTERS
        current_time = float(mm_state.get("time", 0.0))
        inv = float(mm_state.get("inventory", 0.0))

        if self.use_event_update:
            self.event_steps += 1

        tob_key = self._get_env_tob(mm_state)
        if self.use_tob_update:
            if self.last_env_tob_key is None:
                self.last_env_tob_key = tob_key
            elif self.last_env_tob_key != tob_key:
                self.moves_tob += 1
                self.last_env_tob_key = tob_key

        # STEP 2: CHECK BYPASS PRIORITY A — MODE CHANGE
        if self.inv_limit is not None:
            if inv >= self.inv_limit:
                desired_mode = "ask_only"
            elif inv <= -self.inv_limit:
                desired_mode = "bid_only"
            else:
                desired_mode = "two_sided"
        else:
            desired_mode = "two_sided"

        mode_changed = (
            (self.last_mode is not None) and (self.last_mode != desired_mode)
        )
        self.last_mode = desired_mode

        # STEP 3: CHECK BYPASS PRIORITY B — FILL REPLENISHMENT
        has_fill = False
        if self.last_inventory is not None:
            if int(inv) != int(self.last_inventory):
                has_fill = True
        self.last_inventory = int(inv)

        # STEP 4: DECISION LOGIC
        should_act = False

        if mode_changed and not self.use_mdp:
            should_act = True
        elif has_fill and not self.use_mdp:
            should_act = True
        else:
            is_throttled = False

            if not self._has_acted_once:
                pass  # Bypass all gates on first action
            else:
                if self.use_event_update:
                    if self.event_steps < self.threshold_events:
                        is_throttled = True
                if self.use_time_update:
                    if self.last_update_time < 0:
                        pass
                    elif (current_time - self.last_update_time) < self.min_time_interval:
                        is_throttled = True
                if self.use_tob_update:
                    if self.moves_tob < self.threshold_tob:
                        is_throttled = True

            if not is_throttled:
                should_act = True

        # STEP 5: EXECUTE DECISION OR HOLD
        if should_act:
            state_tensor = self._state_to_tensor(mm_state)
            valid_mask = self._get_valid_action_mask(mm_state)
            a_idx = self._sample_action_idx(state_tensor, valid_mask=valid_mask)
            act_tuple = self._mm_action_from_idx(a_idx, mm_state)

            self.last_action_idx = a_idx
            self._last_valid_mask = valid_mask
            self._reset_clocks(current_time, tob_key)
            self._last_was_decision = True

            return act_tuple
        else:
            self._last_was_decision = False
            return ("hold",)

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
        SMDP-aware transition aggregation (called every micro-step).

        In the SMDP framework (Bradtke & Duff, 1994), a single
        "macro-transition" spans K ≥ 1 micro-steps between consecutive
        decisions.  The cumulative discounted reward is:

            R^{cum} = r_0 + γ · r_1 + γ^2 · r_2 + … + γ^{K-1} · r_{K-1}

        This method accumulates R^{cum} and K on-line, then commits the
        completed transition tuple (s, a, R^{cum}, s', d, K, mask) to
        the episode buffers when the next decision arrives or the episode
        terminates.

        Three cases:

        CASE A — This micro-step was a DECISION (act() queried the NN):
            1. If a pending transition exists from the previous decision,
               commit it to episode buffers.  The "next state" s' of the
               old transition is ``state_before`` of the current step.
            2. Start a NEW pending transition: record s, a, mask, and
               initialise R^{cum} = r, K = 1.

        CASE B — This micro-step was THROTTLED (act() returned "hold"):
            1. Accumulate into the pending transition:
                   R^{cum} += γ^K · r      (discount by elapsed micro-steps)
                   K += 1

        CASE C — Episode termination (done=True, info["done"]):
            After handling Case A or B, immediately commit the pending
            transition with d = True (so the bootstrap V(s') is zeroed).
        """
        if self.last_action_idx is None:
            return

        done_flag = bool(info.get("done", False))

        # EVAL MODE: no learning
        if not self._enable_learning:
            if done_flag:
                self.last_action_idx = None
                self._smdp_reset()
            return

        # ==============================================================
        # SMDP AGGREGATION
        # ==============================================================

        if self._last_was_decision:
            # ----- CASE A: New decision arrived -----

            # Step A.1: Commit the previous pending SMDP transition
            if self._smdp_pending:
                s_t = self._state_to_tensor(self._smdp_s_decision)
                s_t_cpu = s_t.squeeze(0).detach().cpu()
                # state_before is the "next state" for the old transition
                s_next_t = self._state_to_tensor(state_before)
                s_next_t_cpu = s_next_t.squeeze(0).detach().cpu()

                self.episode_states.append(s_t_cpu)
                self.episode_actions.append(int(self._smdp_a_decision))
                self.episode_rewards.append(float(self._smdp_cum_reward))
                self.episode_next_states.append(s_next_t_cpu)
                self.episode_dones.append(False)
                self.episode_k_steps.append(int(self._smdp_k_steps))
                self.episode_masks.append(self._smdp_mask_decision)

            # Step A.2: Start a new pending SMDP transition
            self._smdp_pending = True
            self._smdp_s_decision = state_before
            self._smdp_a_decision = self.last_action_idx
            self._smdp_mask_decision = self._last_valid_mask
            self._smdp_cum_reward = float(reward)
            self._smdp_k_steps = 1

        else:
            # ----- CASE B: Throttled micro-step -----
            if self._smdp_pending:
                k = self._smdp_k_steps
                self._smdp_cum_reward += (self.gamma ** k) * float(reward)
                self._smdp_k_steps += 1

        # ==============================================================
        # SPECIAL CASE: EPISODE TERMINATION
        # ==============================================================
        if done_flag:
            if self._smdp_pending:
                s_t = self._state_to_tensor(self._smdp_s_decision)
                s_t_cpu = s_t.squeeze(0).detach().cpu()
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

    # ================================================================
    # EPISODIC FINISH
    # ================================================================

    def finish_episode(self, total_reward: float = 0.0) -> None:
        """
        Finalize the current episode.

        Resets throttle state, clears episode buffers, logs stats.
        Gradient updates are handled externally by the training
        functions (run_a2c_sync_batch / run_a3c_async_batch).
        """
        # LR scheduling is handled externally by the runner
        # (get_lr_for_rollout sets param_groups["lr"] directly).
        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]

        # Console logging
        T = len(self.episode_states)
        print(
            f"[A2C] Episode {self.episode_idx} | "
            f"pure_mm={self.pure_mm} | "
            f"T={T}, "
            f"env_return={total_reward:.4f} | "
            f"lr_actor={actor_lr:.6f}, lr_critic={critic_lr:.6f}"
        )

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar(
                "return/env_total", total_reward, self.episode_idx
            )
            self.writer.add_scalar("lr/actor", actor_lr, self.episode_idx)
            self.writer.add_scalar("lr/critic", critic_lr, self.episode_idx)

        # Prepare for next episode
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
    # ENABLE LEARNING TOGGLE
    # ================================================================

    @property
    def enable_learning(self) -> bool:
        """Get current learning state."""
        return self._enable_learning

    @enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        """
        Toggle learning mode.

        When ENABLED (True):
            - Networks are in train() mode.
            - act() samples actions stochastically.
            - learn() performs SMDP aggregation + updates.

        When DISABLED (False):
            - Networks are in eval() mode.
            - act() uses greedy argmax.
            - learn() becomes a no-op.
        """
        self._enable_learning = bool(value)
        if hasattr(self, "actor_net"):
            if self._enable_learning:
                self.actor_net.train()
                self.critic_net.train()
            else:
                self.actor_net.eval()
                self.critic_net.eval()

    # ================================================================
    # CHECKPOINT SAVE / LOAD
    # ================================================================

    def save_checkpoint(self, path: str) -> None:
        """
        Save controller state to disk for training resumption.

        Saves both actor and critic networks, optimizers, schedulers,
        and the controller configuration for eval-mode reconstruction.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        controller_config = {
            "pure_mm": self.pure_mm,
            "inv_limit": self.inv_limit,
            "n_actions": self.n_actions,
            "pure_mm_offsets": list(self.pure_mm_offsets) if self.pure_mm_offsets else None,
            "level_offset": self.level_offset,
            "n_hidden_actor": self.actor_n_hidden,
            "n_neurons_actor": self.actor_n_neurons,
            "n_hidden_critic": self.critic_n_hidden,
            "n_neurons_critic": self.critic_n_neurons,
            "max_offset": self.max_offset,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "n_steps": self.n_steps,
            "n_steps_sync": self.n_steps_sync,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "critic_loss_coef": self.critic_loss_coef,
            "use_target_critic": self.use_target_critic,
            "target_critic_tau": self.target_critic_tau,
            "adv_normalize_mode": self.adv_normalize_mode,
        }

        checkpoint = {
            "actor_net_state_dict": self.actor_net.state_dict(),
            "critic_net_state_dict": self.critic_net.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler_state_dict": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler_state_dict": self.critic_lr_scheduler.state_dict(),
            "episode_idx": self.episode_idx,
            "controller_config": controller_config,
        }

        if self.target_critic_net is not None:
            checkpoint["target_critic_net_state_dict"] = (
                self.target_critic_net.state_dict()
            )

        torch.save(checkpoint, path)
        print(f"[A2C] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load controller state from disk for training resumption.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor_net.load_state_dict(checkpoint["actor_net_state_dict"])
        self.critic_net.load_state_dict(checkpoint["critic_net_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if "actor_lr_scheduler_state_dict" in checkpoint:
            self.actor_lr_scheduler.load_state_dict(checkpoint["actor_lr_scheduler_state_dict"])
        if "critic_lr_scheduler_state_dict" in checkpoint:
            self.critic_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler_state_dict"])

        if "episode_idx" in checkpoint:
            self.episode_idx = int(checkpoint["episode_idx"])

        # Restore target critic weights (backward-compatible: skip if absent)
        if (
            self.target_critic_net is not None
            and "target_critic_net_state_dict" in checkpoint
        ):
            self.target_critic_net.load_state_dict(
                checkpoint["target_critic_net_state_dict"]
            )

        if "controller_config" in checkpoint:
            cfg = checkpoint["controller_config"]
            if "gamma" in cfg:
                self.gamma = float(cfg["gamma"])
            if "entropy_coef" in cfg:
                self.entropy_coef = float(cfg["entropy_coef"])
            if "grad_clip_norm" in cfg:
                self.grad_clip_norm = float(cfg["grad_clip_norm"])
            if "weight_decay" in cfg:
                self.weight_decay = float(cfg["weight_decay"])
            if "n_steps" in cfg:
                self.n_steps = int(cfg["n_steps"])
            if "n_steps_sync" in cfg:
                self.n_steps_sync = max(1, int(cfg["n_steps_sync"]))
            if "critic_loss_coef" in cfg:
                self.critic_loss_coef = float(cfg["critic_loss_coef"])
            if "target_critic_tau" in cfg:
                self.target_critic_tau = float(cfg["target_critic_tau"])
            if "adv_normalize_mode" in cfg:
                self.adv_normalize_mode = str(cfg["adv_normalize_mode"])
            print("[A2C] Restored hyperparams from checkpoint config")

        print(f"[A2C] Checkpoint loaded from {path}")
        print(f"[A2C] Resuming from episode {self.episode_idx}")

    # ================================================================
    # OPTIONAL: External Logging Helper
    # ================================================================

    def log_episode_stats(
        self, episode_idx: int, total_reward: float, final_pnl: float
    ) -> None:
        """Log external episode statistics to TensorBoard."""
        if self.writer is None:
            return
        self.writer.add_scalar(
            "episode/total_reward_env", total_reward, episode_idx
        )
        self.writer.add_scalar("episode/final_pnl", final_pnl, episode_idx)


# ====================================================================
# EVALUATION FACTORY — Load checkpoint and return eval-mode controller
# ====================================================================

def make_ac_eval_controller(
    ckpt_path: str,
    device: str = "cpu",
    log_dir: str = "runs/ac_eval",
) -> ActorCriticController:
    """
    Load an A2C checkpoint and return an ActorCriticController
    configured for pure evaluation (inference only, no training).

    This is the SINGLE ENTRY POINT for loading trained A2C policies
    for evaluation, backtesting, or comparison studies.

    The returned controller:
        - enable_learning = False: no gradient computation
        - act() returns greedy argmax (deterministic policy)
        - learn() is a no-op (safe to call from the runner)
        - All throttle gates are OFF (acts every micro-step)

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt checkpoint file.
    device : str
        PyTorch device string, e.g. "cpu" or "cuda".
    log_dir : str
        TensorBoard log directory (usually unused in eval mode).

    Returns
    -------
    ActorCriticController
        Fully-configured controller in evaluation mode.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"A2C checkpoint not found: {ckpt_path}")

    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    if "controller_config" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'controller_config'. "
            "Cannot reconstruct the A2C controller."
        )

    cfg = ckpt["controller_config"]

    pure_mm = bool(cfg.get("pure_mm", False))
    inv_limit = cfg.get("inv_limit", None)
    n_actions = int(cfg.get("n_actions", 6))
    pure_mm_offsets = cfg.get("pure_mm_offsets", None)
    level_offset = int(cfg.get("level_offset", 0))
    n_hidden_actor = int(cfg.get("n_hidden_actor", 2))
    n_neurons_actor = int(cfg.get("n_neurons_actor", 128))
    n_hidden_critic = int(cfg.get("n_hidden_critic", 2))
    n_neurons_critic = int(cfg.get("n_neurons_critic", 128))
    n_steps_sync = max(1, int(cfg.get("n_steps_sync", 1)))

    ctrl = ActorCriticController(
        pure_mm=pure_mm,
        inv_limit=inv_limit,
        n_actions=n_actions,
        pure_mm_offsets=pure_mm_offsets,
        level_offset=level_offset,
        n_hidden_actor=n_hidden_actor,
        n_neurons_actor=n_neurons_actor,
        n_hidden_critic=n_hidden_critic,
        n_neurons_critic=n_neurons_critic,
        n_steps_sync=n_steps_sync,
        enable_learning=False,
        use_target_critic=False,  # No target critic needed for inference
        device=dev,
        log_dir=log_dir,
    )

    # Load trained weights
    ctrl.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
    ctrl.actor_net.eval()

    if "critic_net_state_dict" in ckpt:
        ctrl.critic_net.load_state_dict(ckpt["critic_net_state_dict"])
        ctrl.critic_net.eval()

    print(f"[A2C] Eval controller loaded from {ckpt_path}")
    print(f"[A2C] pure_mm={pure_mm}, n_actions={n_actions}, "
          f"inv_limit={inv_limit}, enable_learning=False")

    return ctrl


# ====================================================================
# SYNCHRONOUS A2C TRAINING ROLLOUT
# ====================================================================
#
# One rollout of synchronous Advantage Actor-Critic (A2C) with N
# parallel environments advanced in lock-step at the DECISION level.
#
# High-level flow per decision index k:
#
#   1. zip(*generators) → all N envs yield their k-th decision.
#   2. For each env i:  worker_i.learn(…) → commit SMDP transition to
#      episode buffer → push to per-worker N-step deque.
#   3. Drain all N-step deques → virtual transitions (N-step SMDP
#      returns with correct compound discounting).
#   4. Append virtual transitions to pending_update_batch.
#   5. Every n_steps_sync decision steps:
#        • Gradient update via actor_critic_update_batch()
#          (using ~n_steps_sync × N transitions per update).
#        • Polyak soft update of target critic: ϕ̄ ← τ·ϕ + (1-τ)·ϕ̄.
#
# Why decision-level sync (not micro-step)?
#   The generator yields ONLY when the controller's throttle fires.
#   Throttled micro-steps are handled internally (SMDP aggregation in
#   learn()).  This means each gradient update uses N independent
#   observations of the same decision index — maximum variance
#   reduction per update.
#
# Gradient budget per rollout (example):
#   EPISODE_LENGTH=5000, min_time_interval=1.0, ~300 decisions/env
#   n_steps_sync=13 → ~300/13 ≈ 23 updates per rollout, each using
#   ~13×10 = 130 transitions.  Total gradient steps over 2000 rollouts:
#   ~46K.
# ====================================================================

def run_a2c_sync_batch(
    master_ctrl: "ActorCriticController",
    n_envs: int,
    sim_kwargs: dict,
    reward_fn,
    entropy_coef: float,
) -> dict:
    """
    Run one synchronous A2C training batch with N parallel environments.

    This function is the core training driver for synchronous A2C.  It
    creates N worker controllers (sharing the master's networks), launches
    N simulation generators, and steps them in lockstep at the **decision
    level** — not at the micro-step level.

    Decision-Level Synchronization
    ------------------------------
    The generator ``simulate_LOB_with_MM_generator`` yields ONLY when the
    controller's throttle fires (i.e., at decision points).  Throttled
    micro-steps are handled internally by the generator, which calls
    ``controller.learn()`` for SMDP reward aggregation on those steps.

    Because ``zip(*generators)`` advances all N generators one yield at
    a time, all N environments produce their k-th decision before any
    of them advances to decision k+1.  This is true synchronous A2C:
    each gradient update uses exactly N transitions from the same
    decision index across all environments.

    Algorithm (per decision point)
    ------------------------------
    1. ``zip(*generators)`` waits until all N envs reach their next
       decision point (throttle fires).
    2. Call ``worker.learn()`` on each to commit the SMDP transition
       to the episode buffer.
    3. Aggregate all N transitions into one ``ACTransitionBatch``.
    4. Perform one gradient update via ``actor_critic_update_batch()``.
    5. Clear all workers' episode buffers.
    6. Repeat until all generators are exhausted.

    Why This Outperforms Standard Batch Mode
    -----------------------------------------
    Standard batch: 1 gradient update per N_ENVS episodes (~120 total).
    Synchronous A2C: ~1000 updates per batch (one per decision point),
    each using N_ENVS transitions for variance reduction.

    Known Limitation: zip Truncation
    --------------------------------
    ``zip(*generators)`` stops when the SHORTEST generator finishes.
    Because the throttle fires at random times (exponential inter-arrival),
    different environments may have slightly different numbers of decisions.
    When the first env yields ``done_final``, the loop breaks — any
    remaining decisions and terminal transitions from longer-running envs
    are lost.  In practice, the difference is small (<1% of total
    decisions) since all envs run the same number of micro-steps.

    Parameters
    ----------
    master_ctrl : ActorCriticController
        The master controller that owns the shared actor and critic networks.
        Workers are created by calling ``master_ctrl.make_worker()``.
    n_envs : int
        Number of parallel environments to run.
    sim_kwargs : dict
        Keyword arguments for ``simulate_LOB_with_MM_generator``.  Must NOT
        include ``controller`` or ``reward_fn`` — these are supplied internally.
        Example::

            sim_kwargs = {
                "lam": 1.0, "mu": 0.5, "delta": 0.25,
                "number_tick_levels": 100, "n_priority_ranks": 50,
                "number_levels_to_store": 20, "p0": 100, "mean_size_LO": 1,
                "iterations": 5000, "iterations_to_equilibrium": 500,
                "exclude_self_from_state": True,
                "random_seed": 42,
            }

    reward_fn : callable
        Reward function with signature ``reward_fn(step, mm, lob, s, s', info)``.
    entropy_coef : float
        Current entropy coefficient for the actor loss.

    Returns
    -------
    metrics : dict
        Dictionary containing:
            - ``mean_pnl`` (float): Mean final PnL across N environments.
            - ``mean_reward`` (float): Mean total reward across N environments.
            - ``mean_abs_inv`` (float): Mean absolute inventory.
            - ``max_abs_inv`` (float): Maximum absolute inventory.
            - ``pct_at_limit`` (float): Fraction of time at inventory limit.
            - ``n_updates`` (int): Total gradient updates performed.
            - ``mean_batch_size`` (float): Mean transitions per optimizer step.
            - ``mm_dfs`` (list): List of N ``pd.DataFrame`` with MM logs.

    Example
    -------
    >>> metrics = run_a2c_sync_batch(
    ...     master_ctrl=ctrl,
    ...     n_envs=10,
    ...     sim_kwargs={"lam": 1.0, "mu": 0.5, ...},
    ...     reward_fn=my_reward,
    ...     entropy_coef=0.01,
    ... )
    >>> print(f"PnL: {metrics['mean_pnl']:.4f}, Updates: {metrics['n_updates']}")

    See Also
    --------
    ActorCriticController.make_worker : Creates shared-network worker controllers.
    simulate_LOB_with_MM_generator : Generator-based simulation for external training.
    actor_critic_update_batch : The batch gradient update function used internally.
    """
    from MM_LOB_SIM import simulate_LOB_with_MM_generator

    # -----------------------------------------------------------------
    # 1. Create N worker controllers sharing the master's networks
    # -----------------------------------------------------------------
    workers = [master_ctrl.make_worker() for _ in range(n_envs)]

    # -----------------------------------------------------------------
    # 2. Create N simulation generators (one per worker)
    # -----------------------------------------------------------------
    # Each generator gets its own worker controller for act()/learn()
    # and a unique random seed derived from sim_kwargs["random_seed"].
    generators = []
    base_seed = sim_kwargs.get("random_seed", None)
    for i, worker in enumerate(workers):
        env_kwargs = dict(sim_kwargs)
        env_kwargs["controller"] = worker
        env_kwargs["reward_fn"] = reward_fn
        if base_seed is not None:
            env_kwargs["random_seed"] = int(base_seed) + i * 10_000
        generators.append(simulate_LOB_with_MM_generator(**env_kwargs))

    # -----------------------------------------------------------------
    # 3. Step all generators in lockstep (decision-level synchronization)
    # -----------------------------------------------------------------
    #
    # The generator only yields at DECISION POINTS (when the controller's
    # throttle fires).  Throttled micro-steps are handled internally by
    # the generator, which calls controller.learn() for SMDP reward
    # aggregation on those steps.
    #
    # The loop below keeps one pending decision-point yield per ACTIVE
    # environment.  When some environments finish early, the remaining
    # ones continue without being truncated.
    #
    # At each synchronized decision point:
    #   1. Call learn() on each worker to commit the SMDP transition.
    #   2. Aggregate transitions from all ready workers into one batch.
    #   3. Append that batch to the pending optimizer batch.
    #   4. Perform one gradient update once n_steps_sync batches have
    #      been accumulated (or at final flush).
    #
    # With n_steps_sync=1 this matches standard synchronous A2C.  Larger
    # values trade fewer updates for lower-variance gradients.
    # -----------------------------------------------------------------
    n_updates = 0
    entropy_accum = 0.0          # running sum of mean_entropy from each gradient update
    batch_size_accum = 0.0       # running sum of batch sizes used in each gradient update
    total_rewards = [0.0] * n_envs
    total_spread_capture = [0.0] * n_envs
    total_inv_penalty = [0.0] * n_envs
    mm_dfs = [None] * n_envs
    inv_limit = master_ctrl.inv_limit
    n_steps = master_ctrl.n_steps
    n_steps_sync = max(1, int(getattr(master_ctrl, "n_steps_sync", 1)))
    gamma = master_ctrl.gamma

    # Per-worker N-step sliding window buffers.
    # With n_steps=1, the deque fires on every transition (identical to
    # the old 1-step path).  With n_steps>1, it accumulates transitions
    # and produces virtual N-step transitions once the window is full.
    nstep_bufs = [collections.deque() for _ in range(n_envs)]
    pending_steps: List[Optional[Dict[str, Any]]] = [None] * n_envs
    active = [True] * n_envs
    pending_update_batch = ACTransitionBatch()
    sync_steps_since_update = 0

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

        # (a) Call learn() on each worker that reached the next synchronized
        #     decision point.
        for idx in ready_indices:
            step = pending_steps[idx]
            worker = workers[idx]
            worker.learn(
                step["step_idx"], step["mm"], step["lob"],
                step["state_before"], step["state_after"],
                step["reward"], step["info"],
            )
            total_rewards[idx] += float(step["reward"])
            total_spread_capture[idx] += float(step["info"].get("_reward_spread_capture", 0.0))
            total_inv_penalty[idx] += float(step["info"].get("_reward_inv_penalty", 0.0))

        # (b) Extract raw transitions from each worker's episode buffer
        #     and push them into the per-worker N-step deque.
        for idx in ready_indices:
            w = workers[idx]
            for j in range(len(w.episode_states)):
                nstep_bufs[idx].append({
                    "s": w.episode_states[j],
                    "a": w.episode_actions[j],
                    "r": w.episode_rewards[j],
                    "s_next": w.episode_next_states[j],
                    "done": w.episode_dones[j],
                    "k": w.episode_k_steps[j],
                    "mask": w.episode_masks[j],
                })
            w.episode_states.clear()
            w.episode_actions.clear()
            w.episode_rewards.clear()
            w.episode_next_states.clear()
            w.episode_dones.clear()
            w.episode_k_steps.clear()
            w.episode_masks.clear()
            pending_steps[idx] = None

        # (c) Drain N-step buffers → produce virtual transitions.
        #     With n_steps=1, each buffer produces exactly 1 virtual
        #     transition per raw transition (same as old 1-step path).
        batch = ACTransitionBatch()
        for buf in nstep_bufs:
            _drain_nstep_buffer(buf, gamma, n_steps, batch)

        if len(batch) > 0:
            pending_update_batch.states.extend(batch.states)
            pending_update_batch.actions.extend(batch.actions)
            pending_update_batch.rewards.extend(batch.rewards)
            pending_update_batch.next_states.extend(batch.next_states)
            pending_update_batch.dones.extend(batch.dones)
            pending_update_batch.k_steps.extend(batch.k_steps)
            pending_update_batch.masks.extend(batch.masks)
            sync_steps_since_update += 1

            if sync_steps_since_update >= n_steps_sync:
                current_batch_size = len(pending_update_batch)
                update_info = actor_critic_update_batch(
                    actor_net=master_ctrl.actor_net,
                    critic_net=master_ctrl.critic_net,
                    actor_optimizer=master_ctrl.actor_optimizer,
                    critic_optimizer=master_ctrl.critic_optimizer,
                    batch=pending_update_batch,
                    gamma=gamma,
                    entropy_coef=entropy_coef,
                    device=master_ctrl.device,
                    grad_clip_norm=master_ctrl.grad_clip_norm,
                    critic_loss_coef=master_ctrl.critic_loss_coef,
                    target_critic_net=master_ctrl.target_critic_net,
                    adv_normalize_mode=master_ctrl.adv_normalize_mode,
                )
                master_ctrl._soft_update_target()
                n_updates += 1
                entropy_accum += update_info["mean_entropy"]
                batch_size_accum += float(current_batch_size)
                pending_update_batch = ACTransitionBatch()
                sync_steps_since_update = 0

    # -----------------------------------------------------------------
    # 4. Flush remaining N-step buffers after the episode ends.
    #    Remaining entries get progressively shorter returns (N-1, N-2, ...)
    # -----------------------------------------------------------------
    flush_batch = ACTransitionBatch()
    for buf in nstep_bufs:
        _drain_nstep_buffer(buf, gamma, n_steps, flush_batch, flush=True)

    # Also flush any leftover transitions in worker episode buffers
    # (e.g., terminal transitions committed during the last yield)
    for w in workers:
        for j in range(len(w.episode_states)):
            flush_batch.states.append(w.episode_states[j])
            flush_batch.actions.append(w.episode_actions[j])
            flush_batch.rewards.append(w.episode_rewards[j])
            flush_batch.next_states.append(w.episode_next_states[j])
            flush_batch.dones.append(w.episode_dones[j])
            flush_batch.k_steps.append(w.episode_k_steps[j])
            flush_batch.masks.append(w.episode_masks[j])

    if len(flush_batch) > 0:
        pending_update_batch.states.extend(flush_batch.states)
        pending_update_batch.actions.extend(flush_batch.actions)
        pending_update_batch.rewards.extend(flush_batch.rewards)
        pending_update_batch.next_states.extend(flush_batch.next_states)
        pending_update_batch.dones.extend(flush_batch.dones)
        pending_update_batch.k_steps.extend(flush_batch.k_steps)
        pending_update_batch.masks.extend(flush_batch.masks)

    if len(pending_update_batch) > 0:
        current_batch_size = len(pending_update_batch)
        update_info = actor_critic_update_batch(
            actor_net=master_ctrl.actor_net,
            critic_net=master_ctrl.critic_net,
            actor_optimizer=master_ctrl.actor_optimizer,
            critic_optimizer=master_ctrl.critic_optimizer,
            batch=pending_update_batch,
            gamma=gamma,
            entropy_coef=entropy_coef,
            device=master_ctrl.device,
            grad_clip_norm=master_ctrl.grad_clip_norm,
            critic_loss_coef=master_ctrl.critic_loss_coef,
            target_critic_net=master_ctrl.target_critic_net,
            adv_normalize_mode=master_ctrl.adv_normalize_mode,
        )
        master_ctrl._soft_update_target()
        n_updates += 1
        entropy_accum += update_info["mean_entropy"]
        batch_size_accum += float(current_batch_size)

    # -----------------------------------------------------------------
    # 5. Compute batch metrics from the final MM DataFrames
    # -----------------------------------------------------------------
    pnls = []
    abs_invs_mean = []
    abs_invs_max = []
    pcts_at_limit = []

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
            if inv_limit is not None and inv_limit > 0:
                pcts_at_limit.append(float(np.mean(abs_inv >= inv_limit)))
            else:
                pcts_at_limit.append(0.0)

    metrics = {
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_abs_inv": float(np.mean(abs_invs_mean)) if abs_invs_mean else 0.0,
        "max_abs_inv": float(np.max(abs_invs_max)) if abs_invs_max else 0.0,
        "pct_at_limit": float(np.mean(pcts_at_limit)) if pcts_at_limit else 0.0,
        "n_updates": n_updates,
        "mean_entropy": entropy_accum / max(1, n_updates),
        "mean_batch_size": batch_size_accum / max(1, n_updates),
        "mean_spread_capture": float(np.mean(total_spread_capture)),
        "mean_inv_penalty": float(np.mean(total_inv_penalty)),
        "mm_dfs": mm_dfs,
    }

    return metrics


# ====================================================================
# ASYNCHRONOUS A3C TRAINING ROLLOUT  (Mnih et al., 2016)
# ====================================================================
#
# One rollout of Asynchronous Advantage Actor-Critic (A3C).  Unlike
# synchronous A2C (which lock-steps N envs at the decision level),
# A3C runs each environment in its own Python thread and performs
# gradient updates independently with a shared gradient lock.
#
# Thread model:
#   N threads, 1 gradient lock, 1 set of shared parameters (θ, ϕ, ϕ̄).
#   Each worker:
#     1. Steps its own generator to the next decision point.
#     2. Commits the SMDP transition via learn().
#     3. Drains the N-step buffer → virtual transitions.
#     4. Acquires the grad_lock → backward pass → optimiser step
#        → Polyak update → releases lock.
#
# Weight staleness:
#   Between steps (3) and (4), other workers may have updated (θ, ϕ).
#   The worker's forward pass used stale weights, but the gradient is
#   still an unbiased (if noisy) estimate of ∇L.  Mnih et al. (2016)
#   show empirically that this staleness acts as implicit
#   regularisation and can improve exploration.
#
# The key difference from A2C:
#   - A2C: N transitions batched → 1 gradient update (mean of N)
#   - A3C: 1 transition → 1 gradient update, N threads concurrently
#
# Because each thread updates the shared weights independently, other
# threads may do forward passes on slightly stale weights.  This
# staleness acts as implicit regularization and is a feature of A3C,
# not a bug.
#
# Thread safety: a threading.Lock serializes the backward + optimizer
# step to prevent gradient corruption.  Forward passes (in act()) are
# safe to run concurrently since they only read weights.
# ====================================================================

def run_a3c_async_batch(
    master_ctrl: "ActorCriticController",
    n_envs: int,
    sim_kwargs: dict,
    reward_fn,
    entropy_coef: float,
) -> dict:
    """
    Run one asynchronous A3C training batch with N threaded environments.

    Each environment runs in its own thread with its own worker controller
    (sharing the master's actor and critic networks).  When a worker
    commits an SMDP transition, it immediately performs a gradient update
    on the shared networks — no synchronization with other workers.

    A3C vs A2C
    ----------
    A2C (``run_a2c_sync_batch``):
        All N environments advance to the same decision point before any
        gradient update.  Each update uses N transitions (one per env).
        Gradient = mean of N independent TD errors.

    A3C (this function):
        Each environment advances independently in its own thread.
        Each update uses 1 transition from whichever worker finishes first.
        Other workers may see partially-updated weights (stale gradients).

    Thread Safety
    -------------
    A ``threading.Lock`` serializes the gradient computation and optimizer
    step (``actor_critic_update_batch``).  This prevents gradient
    corruption from concurrent backward passes writing to the same
    ``.grad`` buffers.

    Forward passes during ``act()`` are NOT locked — they only read
    weights and are safe to run concurrently.  The slight staleness
    this introduces is the standard A3C behavior.

    Note on Python's GIL
    --------------------
    CPython's Global Interpreter Lock means threads are not truly
    parallel for CPU-bound work.  However, the A3C update pattern is
    still valid: threads interleave execution, and each worker sees
    weights that have been updated by other workers between its own
    decisions.  For true multi-core parallelism, ``multiprocessing``
    with shared memory would be needed (not implemented here).

    Parameters
    ----------
    master_ctrl : ActorCriticController
        The master controller that owns the shared actor and critic networks.
    n_envs : int
        Number of parallel environments (threads) to run.
    sim_kwargs : dict
        Keyword arguments for ``simulate_LOB_with_MM_generator``.  Must NOT
        include ``controller`` or ``reward_fn``.
    reward_fn : callable
        Reward function with signature ``reward_fn(step, mm, lob, s, s', info)``.
    entropy_coef : float
        Current entropy coefficient for the actor loss.

    Returns
    -------
    metrics : dict
        Same structure as ``run_a2c_sync_batch``:
            - ``mean_pnl``, ``mean_reward``, ``mean_abs_inv``,
              ``max_abs_inv``, ``pct_at_limit``, ``n_updates``,
              ``mean_batch_size``, ``mm_dfs``.

    See Also
    --------
    run_a2c_sync_batch : Synchronous A2C alternative.
    ActorCriticController.make_worker : Creates shared-network worker controllers.
    """
    from MM_LOB_SIM import simulate_LOB_with_MM_generator

    # -----------------------------------------------------------------
    # 1. Create N worker controllers sharing the master's networks
    # -----------------------------------------------------------------
    workers = [master_ctrl.make_worker() for _ in range(n_envs)]

    # -----------------------------------------------------------------
    # 2. Shared state for thread synchronization
    # -----------------------------------------------------------------
    grad_lock = threading.Lock()
    # Per-worker results (written by each thread, read after join)
    worker_results = [None] * n_envs

    # -----------------------------------------------------------------
    # 3. Worker thread function
    # -----------------------------------------------------------------
    n_steps = master_ctrl.n_steps
    gamma = master_ctrl.gamma

    def _worker_fn(worker_id: int, worker: "ActorCriticController"):
        """Run one environment to completion, updating shared nets."""
        env_kwargs = dict(sim_kwargs)
        env_kwargs["controller"] = worker
        env_kwargs["reward_fn"] = reward_fn
        base_seed = sim_kwargs.get("random_seed", None)
        if base_seed is not None:
            env_kwargs["random_seed"] = int(base_seed) + worker_id * 10_000

        gen = simulate_LOB_with_MM_generator(**env_kwargs)

        total_reward = 0.0
        n_updates_local = 0
        entropy_accum_local = 0.0
        batch_size_accum_local = 0.0
        spread_capture_local = 0.0
        inv_penalty_local = 0.0
        mm_df = None
        nstep_buf = collections.deque()

        for step in gen:
            # Final sentinel — simulation complete
            if step.get("done_final"):
                mm_df = step.get("mm_df")
                total_reward = float(step.get("total_reward", total_reward))
                break

            # Call learn() to commit the SMDP transition.
            # The generator only yields at decision points (and done),
            # so learn() will either:
            #   - Start a new pending transition (first decision), or
            #   - Commit the previous transition + start a new one.
            worker.learn(
                step["step_idx"], step["mm"], step["lob"],
                step["state_before"], step["state_after"],
                step["reward"], step["info"],
            )
            total_reward += step["reward"]

            # Reward decomposition (mirrors A2C path)
            info = step.get("info", {})
            spread_capture_local += float(info.get("_reward_spread_capture", 0.0))
            inv_penalty_local += float(info.get("_reward_inv_penalty", 0.0))

            # Extract raw transitions → push to N-step deque
            for j in range(len(worker.episode_states)):
                nstep_buf.append({
                    "s": worker.episode_states[j],
                    "a": worker.episode_actions[j],
                    "r": worker.episode_rewards[j],
                    "s_next": worker.episode_next_states[j],
                    "done": worker.episode_dones[j],
                    "k": worker.episode_k_steps[j],
                    "mask": worker.episode_masks[j],
                })
            worker.episode_states.clear()
            worker.episode_actions.clear()
            worker.episode_rewards.clear()
            worker.episode_next_states.clear()
            worker.episode_dones.clear()
            worker.episode_k_steps.clear()
            worker.episode_masks.clear()

            # Drain N-step buffer → produce virtual transitions
            batch = ACTransitionBatch()
            _drain_nstep_buffer(nstep_buf, gamma, n_steps, batch)

            if len(batch) > 0:
                # Asynchronous gradient update — lock prevents
                # concurrent backward passes from corrupting .grad
                current_batch_size = len(batch)
                with grad_lock:
                    update_info = actor_critic_update_batch(
                        actor_net=master_ctrl.actor_net,
                        critic_net=master_ctrl.critic_net,
                        actor_optimizer=master_ctrl.actor_optimizer,
                        critic_optimizer=master_ctrl.critic_optimizer,
                        batch=batch,
                        gamma=gamma,
                        entropy_coef=entropy_coef,
                        device=master_ctrl.device,
                        grad_clip_norm=master_ctrl.grad_clip_norm,
                        critic_loss_coef=master_ctrl.critic_loss_coef,
                        target_critic_net=master_ctrl.target_critic_net,
                        adv_normalize_mode=master_ctrl.adv_normalize_mode,
                    )
                    master_ctrl._soft_update_target()
                n_updates_local += 1
                entropy_accum_local += update_info["mean_entropy"]
                batch_size_accum_local += float(current_batch_size)

        # Flush remaining N-step entries at end of episode
        flush_batch = ACTransitionBatch()
        _drain_nstep_buffer(nstep_buf, gamma, n_steps, flush_batch, flush=True)
        if len(flush_batch) > 0:
            current_batch_size = len(flush_batch)
            with grad_lock:
                update_info = actor_critic_update_batch(
                    actor_net=master_ctrl.actor_net,
                    critic_net=master_ctrl.critic_net,
                    actor_optimizer=master_ctrl.actor_optimizer,
                    critic_optimizer=master_ctrl.critic_optimizer,
                    batch=flush_batch,
                    gamma=gamma,
                    entropy_coef=entropy_coef,
                    device=master_ctrl.device,
                    grad_clip_norm=master_ctrl.grad_clip_norm,
                    critic_loss_coef=master_ctrl.critic_loss_coef,
                    target_critic_net=master_ctrl.target_critic_net,
                    adv_normalize_mode=master_ctrl.adv_normalize_mode,
                )
                master_ctrl._soft_update_target()
            n_updates_local += 1
            entropy_accum_local += update_info["mean_entropy"]
            batch_size_accum_local += float(current_batch_size)

        worker_results[worker_id] = {
            "mm_df": mm_df,
            "total_reward": total_reward,
            "n_updates": n_updates_local,
            "entropy_accum": entropy_accum_local,
            "batch_size_accum": batch_size_accum_local,
            "spread_capture": spread_capture_local,
            "inv_penalty": inv_penalty_local,
        }

    # -----------------------------------------------------------------
    # 4. Launch N threads and wait for completion
    # -----------------------------------------------------------------
    threads = []
    for i, w in enumerate(workers):
        t = threading.Thread(target=_worker_fn, args=(i, w), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # -----------------------------------------------------------------
    # 5. Compute batch metrics from worker results
    # -----------------------------------------------------------------
    inv_limit = master_ctrl.inv_limit
    pnls = []
    abs_invs_mean = []
    abs_invs_max = []
    pcts_at_limit = []
    total_rewards = []
    mm_dfs = []
    total_updates = 0
    total_entropy_accum = 0.0
    total_batch_size_accum = 0.0
    total_spread_capture = []
    total_inv_penalty = []

    for res in worker_results:
        if res is None:
            mm_dfs.append(None)
            continue
        mm_df = res["mm_df"]
        mm_dfs.append(mm_df)
        total_rewards.append(res["total_reward"])
        total_updates += res["n_updates"]
        total_entropy_accum += res.get("entropy_accum", 0.0)
        total_batch_size_accum += res.get("batch_size_accum", 0.0)
        total_spread_capture.append(res.get("spread_capture", 0.0))
        total_inv_penalty.append(res.get("inv_penalty", 0.0))

        if mm_df is not None:
            if "MM_TotalPnL" in mm_df.columns:
                pnls.append(float(mm_df["MM_TotalPnL"].iloc[-1]))
            if "MM_Inventory" in mm_df.columns:
                inv_series = mm_df["MM_Inventory"].values.astype(float)
                abs_inv = np.abs(inv_series)
                abs_invs_mean.append(float(np.mean(abs_inv)))
                abs_invs_max.append(float(np.max(abs_inv)))
                if inv_limit is not None and inv_limit > 0:
                    pcts_at_limit.append(float(np.mean(abs_inv >= inv_limit)))
                else:
                    pcts_at_limit.append(0.0)

    metrics = {
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_abs_inv": float(np.mean(abs_invs_mean)) if abs_invs_mean else 0.0,
        "max_abs_inv": float(np.max(abs_invs_max)) if abs_invs_max else 0.0,
        "pct_at_limit": float(np.mean(pcts_at_limit)) if pcts_at_limit else 0.0,
        "n_updates": total_updates,
        "mean_entropy": total_entropy_accum / max(1, total_updates),
        "mean_batch_size": total_batch_size_accum / max(1, total_updates),
        "mean_spread_capture": float(np.mean(total_spread_capture)) if total_spread_capture else 0.0,
        "mean_inv_penalty": float(np.mean(total_inv_penalty)) if total_inv_penalty else 0.0,
        "mm_dfs": mm_dfs,
    }

    return metrics
