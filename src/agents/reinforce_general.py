#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy-gradient (REINFORCE) controller for the MarketMaker + LOB simulation.

Overview
========
This module implements a Monte Carlo REINFORCE controller with:

    - Advantage baseline (per-time-step return standardization)
    - Entropy regularization (encourages exploration)
    - Two update frequencies: "time_step" (one grad step per t) or "episode"
      (accumulated gradient, single step per episode)
    - Three independent throttle gates (TOB, Event, Time) with OR-blocking
    - Two bypass priorities (mode change, fill replenishment) that can be
      disabled via the ``use_mdp`` flag for strict MDP semantics
    - SMDP (Semi-Markov Decision Process) reward aggregation between decisions
    - Constrained action selection via ``_get_valid_action_mask()`` — prevents
      inventory-violating actions in generic mode and masks degenerate
      (canonically equivalent) actions in pure-MM mode at inventory limits
    - enable_learning toggle (train/eval mode switching)
    - Checkpoint save/load for training resumption
    - LR scheduling (StepLR decay)

REINFORCE vs DQN — Key Algorithmic Differences
===============================================
REINFORCE is an ON-POLICY Monte Carlo method:

    1. Collect a FULL episode trajectory {(s_0, a_0, r_0), ..., (s_T, a_T, r_T)}.
    2. Compute discounted returns G_t = sum_{k>=t} gamma^{k-t} r_k.
    3. Compute advantage A_t = (G_t - mean(G)) / std(G).
    4. Policy gradient: nabla J = E[ -log pi(a_t|s_t) * A_t ].
    5. Update parameters with a single pass over the trajectory.
    6. Discard the trajectory (no replay buffer).

DQN is OFF-POLICY with experience replay:
    - Pushes transitions to a replay buffer immediately.
    - Samples mini-batches from the buffer for each gradient step.
    - Can reuse old transitions (sample efficiency).

This fundamental difference affects how throttle + SMDP integrate:
    - DQN commits each SMDP transition to the replay buffer as soon as
      the next decision arrives.
    - REINFORCE accumulates SMDP transitions in an episode buffer and
      only processes them at episode end (in finish_episode()).

State Representation
====================
GENERIC MODE (pure_mm=False):
    s = [spread, asksize, bidsize, inventory, has_bid, has_ask]
    Dimension: 6

PURE MM MODE (pure_mm=True):
    s = [spread, inventory, bid_sizes[0..K], ask_sizes[0..K]]
    Dimension: 2 + 2*(K+1)  where K = max non-negative offset

Action Space
============
GENERIC ACTIONS (pure_mm=False):
    Index | Action
    ------+---------------------------
      0   | post_bid
      1   | post_ask
      2   | post_bid_ask
      3   | cancel_bid
      4   | cancel_ask
      5   | hold
    (Optional: 6=inside_spread_both, 7=inside_spread_bid, 8=inside_spread_ask)

PURE MM ACTIONS (pure_mm=True):
    Each action index maps to an offset pair (delta_bid, delta_ask) in ticks:
        delta > 0 : more passive (deeper in the book)
        delta = 0 : quote at L1 (best bid / best ask)
        delta < 0 : improve inside the spread (more aggressive),
                     but ONLY if spread >= 2 ticks, and NEVER crossing.

Action Masking (Constrained Selection)
======================================
At each decision step, ``_get_valid_action_mask()`` builds a boolean mask
that restricts the policy to only valid actions.  Invalid logits are set
to ``-inf`` before softmax (training) or argmax (eval), so the policy
can never select a forbidden action.

    GENERIC MODE at inventory limits:
        inv >= +limit  →  only post_ask, hold, ask_inside allowed
        inv <= -limit  →  only post_bid, hold, bid_inside allowed

    PURE MM MODE at inventory limits (Canonical Degenerate Masking):
        At +limit, the bid side is dropped by the execution layer.
        Actions that differ only on the bid offset produce identical
        outcomes (same ask price, same reward, same next state).  For
        each equivalence group (same ask offset), only the smallest-
        index representative (the "canonical" action) is allowed.
        Away from limits, mask is all-True.

        Example with grid4_passive [(0,0), (0,1), (1,0), (1,1)]:
          At +limit (ask-only):
            - a=0 (ask_off=0) and a=2 (ask_off=0) are equivalent → mask = [T, T, F, F]
            - a=1 (ask_off=1) and a=3 (ask_off=1) are equivalent

Throttle Gate Hierarchy
=======================
When throttling is enabled, the controller does NOT query the neural
network on every micro-step. Instead, it follows a priority-based
decision hierarchy that determines whether to act or hold:

    Priority A (BYPASS): Mode changes — inventory crossed inv_limit.
                          Force immediate decision to adjust orders.
                          Disabled when ``use_mdp=True``.

    Priority B (BYPASS): Fill detected — inventory changed since last
                          decision. Force decision to replace the
                          filled order quickly.
                          Disabled when ``use_mdp=True``.

    Priority C (THROTTLE): Hard throttle check with OR-blocking semantics.
                            Three independent gates, any one blocking is
                            enough to suppress the action:
                              - Event gate: not enough events since last decision
                              - Time gate: not enough simulated time elapsed
                              - TOB gate: not enough L1 changes observed
                            The controller only passes when ALL enabled
                            gates are simultaneously satisfied.

    First-Action Bypass: Before the first decision of each episode,
                          all gates are bypassed so initial quotes are
                          placed immediately.  Active even in MDP mode.

MDP Mode (use_mdp=True)
=======================
Disables BOTH bypass priorities (A and B).  The agent strictly
respects the throttle gates — no early decisions for fills, inventory
emergencies, or mode changes.  This removes the variable decision
timing that makes the problem an SMDP:

    - Event gate: K is exactly fixed (every N events) — true MDP.
    - Time gate:  Δt is fixed, but K varies because inter-arrival
      times are stochastic.  Still MDP (state sampled at constant Δt).
    - TOB gate:   K varies (depends on how often L1 moves).

In all cases, bypasses no longer inject extra decisions, so the
decision process is governed solely by the chosen gate.

SMDP Reward Aggregation
========================
When throttled, the runner still calls learn() every micro-step. The
controller accumulates these micro-step rewards into a single SMDP
transition using gamma-discounted cumulative rewards:

    R_cum = r_0 + gamma^1 * r_1 + gamma^2 * r_2 + ... + gamma^{K-1} * r_{K-1}

where K is the number of micro-steps between two consecutive decisions.

At decision time, the previous pending SMDP transition is committed
to the episode buffer with its aggregated reward R_cum AND its holding
time K.  The holding time is critical for correct Monte Carlo returns:

    G_t = R_cum_t + gamma^{K_t} * G_{t+1}      (SMDP-correct)
    G_t = R_cum_t + gamma^1    * G_{t+1}      (WRONG — old code)

Without storing K, the inter-transition discount defaults to gamma^1,
which under-discounts future rewards by a factor that grows
exponentially with the holding period length.  For example, with
K=100 and gamma=0.99 the correct discount is 0.99^100 ~ 0.366 but
the old code used 0.99 — a 2.7x overweight on future rewards.

Inventory Band (Pure MM)
========================
  inv_limit is None:
      Always two-sided quoting (BID + ASK).

  inv_limit is not None:
      inventory >= +inv_limit --> only ASK is posted
      inventory <= -inv_limit --> only BID is posted
      otherwise              --> two-sided BID+ASK
"""

from typing import Dict, Any, Optional, List, Tuple

import os
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
# Policy Network: pi(a | s)
# ============================================================

class PolicyNetwork(nn.Module):
    """
    Simple MLP policy network that outputs action logits.

    Architecture (shared trunk with linear readout):

        input_dim --> Linear --> ReLU --> ...  (n_hidden layers)
                  --> Linear --> logits (n_actions)

    The logits are unnormalized log-probabilities. The caller
    applies softmax to obtain a valid probability distribution
    over actions.

    Input Dimensions
    ----------------
    Generic mode (pure_mm=False):
        D = 6: [spread, asksize, bidsize, inventory, has_bid, has_ask]

    Pure-MM mode (pure_mm=True):
        D = 2 + 2*(K+1): [spread, inventory, bid_sizes[0..K], ask_sizes[0..K]]
        where K is the maximum non-negative offset in the pure-mm action grid.

    Output Dimensions
    -----------------
    n_actions logits (unnormalized). These get fed into:
        - Softmax for sampling (training)
        - Argmax for greedy selection (evaluation)

    Why No Noisy Layers?
    --------------------
    Unlike the DQN controller which uses NoisyNet linear layers for
    exploration, REINFORCE explores naturally through its stochastic
    policy (softmax sampling). The entropy regularization term in the
    loss function provides additional exploration pressure, making
    noisy layers unnecessary.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        n_hidden: int = 2,
        n_neurons: int = 128,
    ):
        super().__init__()

        # Argument validation — catch silent config mistakes early
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")

        # Build the shared MLP trunk: input_dim -> n_neurons -> ... (n_hidden layers)
        # Each hidden layer is Linear + ReLU. ReLU is chosen for simplicity
        # and proven effectiveness in RL settings. More exotic activations
        # (GELU, SiLU) could be explored but offer marginal gains here.
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_hidden)):
            layers.append(nn.Linear(in_dim, int(n_neurons)))
            layers.append(nn.ReLU())
            in_dim = int(n_neurons)

        self.feature = nn.Sequential(*layers)

        # Final linear readout: n_neurons -> n_actions (no activation)
        # The absence of a final activation is intentional: these are logits,
        # not probabilities. Softmax is applied externally as needed.
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
# Episode Batch Container
# ============================================================

class EpisodeBatch:
    """
    Container for a batch of episode trajectories.

    Each episode is a dict with keys:
        'states'  : list of T_i tensors of shape (state_dim,)
        'actions' : list of T_i ints (action indices)
        'rewards' : list of T_i floats (scalar rewards)
        'k_steps' : list of T_i ints (SMDP holding times per decision)

    When throttling + SMDP are active, each "step" in the trajectory
    corresponds to one SMDP-level decision (not one micro-step).
    The rewards are already gamma-discounted aggregates from the
    SMDP accumulation in learn().  The k_steps field stores how many
    micro-steps each SMDP transition spans, which is needed for
    correct inter-transition discounting (gamma^K instead of gamma).
    """

    def __init__(self, episodes: List[Dict[str, Any]]):
        self.episodes = episodes


# ============================================================
# REINFORCE Update with Advantage and Configurable Frequency
# ============================================================

def reinforce_update_batch(
    policy_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: EpisodeBatch,
    gamma: float,
    entropy_coef: float,
    device: torch.device,
    update_frequency: str = "time_step",
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    """
    REINFORCE policy-gradient update over a batch of episodes.

    Algorithm
    ---------
    1) For each episode i, compute Monte Carlo returns G_t^i:

           G_t = r_t + gamma * G_{t+1}

       (backward recursion from the last time step).

    2) For each time step t (iterated backwards):

       a) Gather all (s_t^i, a_t^i, G_t^i) across episodes alive at t.
       b) Compute per-step ADVANTAGE:

              A_t^i = (G_t^i - mean_i(G_t^i)) / (std_i(G_t^i) + eps)

          This acts as a per-time-step baseline, centering and scaling
          the returns to reduce variance without introducing bias.

       c) Policy gradient loss:

              pg_loss_t = - log pi(a_t | s_t) * A_t

       d) Entropy regularization:

              H(pi(.|s_t)) = - sum_a pi(a|s) log pi(a|s)
              total_loss_t = mean( pg_loss_t - entropy_coef * H )

          The entropy term encourages exploration by penalizing
          peaked distributions. The negative sign means MAXIMIZING
          entropy (subtracting it from the loss to minimize).

    Update Frequency
    ----------------
    "time_step":
        One optimizer step per time step t. Each t gets its own
        zero_grad -> backward -> clip -> step cycle. This is more
        frequent but can be unstable for very short episodes.

    "episode":
        Accumulate gradients across ALL time steps in the episode,
        then perform a SINGLE optimizer step. This is more stable
        and memory-efficient (we use gradient accumulation to avoid
        storing computation graphs for all time steps).

        MEMORY OPTIMIZATION (Bug 4 fix):
        Instead of storing loss tensors with their computation graphs
        (which causes OOM on 200k+ step episodes), we call backward()
        on each time step immediately with a scaled loss (1/T), then
        do a single optimizer step at the end. This accumulates
        gradients in-place without retaining computation graphs.

    Parameters
    ----------
    policy_net : nn.Module
        The policy network pi(a|s).
    optimizer : torch.optim.Optimizer
        The optimizer (e.g., AdamW).
    batch : EpisodeBatch
        Container with one or more episode trajectories.
    gamma : float
        Discount factor for Monte Carlo returns.
    entropy_coef : float
        Weight of the entropy regularization term.
    device : torch.device
        Device for tensor computations.
    update_frequency : str
        "time_step" or "episode".
    grad_clip_norm : float
        Maximum L2 norm for gradient clipping (torch.nn.utils.clip_grad_norm_).
        Previously hardcoded to 1.0; now exposed for tuning.  Gradient
        clipping prevents catastrophic updates from high-variance REINFORCE
        gradients, especially in early training when returns vary wildly.

    Returns
    -------
    stats : dict
        Scalar statistics for logging:
            mean_loss, mean_pg_loss, mean_entropy, mean_return
    """
    if update_frequency not in ("time_step", "episode"):
        raise ValueError(
            f"update_frequency must be 'time_step' or 'episode', "
            f"got '{update_frequency}'"
        )

    episodes = batch.episodes
    if len(episodes) == 0:
        return {
            "mean_loss": 0.0,
            "mean_pg_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_return": 0.0,
        }

    # ----------------------------------------------------------------
    # Pass 1: Precompute per-episode Monte Carlo returns G_t
    # ----------------------------------------------------------------
    # For each episode, we compute returns backward in time:
    #     G_T = r_T
    #     G_t = r_t + gamma * G_{t+1}
    #
    # This is done with plain tensors (no grad needed here).
    # ----------------------------------------------------------------
    ep_returns: List[torch.Tensor] = []
    ep_lengths: List[int] = []

    for ep in episodes:
        rewards = ep["rewards"]
        L = len(rewards)
        ep_lengths.append(L)

        if L == 0:
            ep_returns.append(torch.zeros(0, 1, device=device))
            continue

        # Rewards as column vector (L, 1)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device).view(L, 1)
        G_ep = torch.zeros_like(r_t)

        # SMDP holding times: K_t = number of micro-steps spanned by
        # decision t.  When throttling is active K_t >> 1; when inactive
        # (or k_steps not provided) K_t = 1 and we recover the standard
        # single-step discount.
        k_steps = ep.get("k_steps", None)

        # Backward Monte Carlo sum with SMDP-correct discounting:
        #
        #     G_t = R_cum_t + gamma^{K_t} * G_{t+1}
        #
        # BUG FIX: Previously used gamma^1 unconditionally between SMDP
        # transitions, which under-discounts future rewards when K_t > 1.
        # For example, with K=100 and gamma=0.99, the old code used
        # 0.99 where the correct value is 0.99^100 ~ 0.366 — a 2.7x
        # overweight on future rewards that corrupts the policy gradient
        # whenever throttling is active.
        G = 0.0
        for t in reversed(range(L)):
            if k_steps is not None:
                discount = gamma ** k_steps[t]
            else:
                discount = gamma
            G = r_t[t] + discount * G
            G_ep[t] = G

        ep_returns.append(G_ep)

    # Edge case: all episodes are empty (no transitions at all)
    if all(L == 0 for L in ep_lengths):
        return {
            "mean_loss": 0.0,
            "mean_pg_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_return": 0.0,
        }

    # Global statistics for logging and single-episode fallback
    all_returns_concat = torch.cat(
        [G_ep for G_ep in ep_returns if G_ep.shape[0] > 0],
        dim=0,
    )
    mean_return = float(all_returns_concat.mean().item())
    std_return = float(all_returns_concat.std(unbiased=False).item()) if all_returns_concat.numel() > 1 else 1.0

    max_len = max(ep_lengths)

    # ----------------------------------------------------------------
    # Pass 2: Backward-time loop — compute losses and update
    # ----------------------------------------------------------------
    # We iterate over time steps t in REVERSE order (from max_len-1 to 0).
    # At each t, we gather all episodes that are still alive (length > t)
    # and compute the policy gradient loss for that time step.
    #
    # For "episode" mode, we use GRADIENT ACCUMULATION:
    #   - Zero gradients ONCE before the loop.
    #   - At each t, compute loss_t / max_len and call backward().
    #   - Gradients accumulate in-place (no computation graph storage).
    #   - After the loop, clip and step ONCE.
    #
    # This avoids the OOM bug where storing all loss tensors with their
    # computation graphs for 200k+ step episodes exhausted GPU memory.
    # ----------------------------------------------------------------
    sum_total_loss = 0.0
    sum_pg_loss = 0.0
    sum_entropy = 0.0
    n_steps = 0

    # For "episode" mode: zero gradients once before the accumulation loop
    if update_frequency == "episode":
        optimizer.zero_grad()

    for t in reversed(range(max_len)):
        states_t_list: List[torch.Tensor] = []
        actions_t_list: List[int] = []
        G_t_list: List[torch.Tensor] = []
        masks_t_list: List[Optional[np.ndarray]] = []  # P1 FIX

        # Collect all episodes contributing at time t
        for ep_idx, ep in enumerate(episodes):
            L = ep_lengths[ep_idx]
            if t >= L:
                continue

            state_t = ep["states"][t]    # May be on CPU (memory optimization)
            action_t = ep["actions"][t]
            G_t = ep_returns[ep_idx][t]

            # P1 FIX: Retrieve the mask used at decision time (if stored)
            ep_masks = ep.get("masks", None)
            mask_t = ep_masks[t] if ep_masks is not None else None

            states_t_list.append(state_t)
            actions_t_list.append(action_t)
            G_t_list.append(G_t)
            masks_t_list.append(mask_t)

        if len(states_t_list) == 0:
            continue

        # Build batch tensors and move to device (GPU)
        # States were stored on CPU in learn() to save GPU memory
        # during long episodes. We move them back to GPU here.
        states_batch = torch.stack(states_t_list, dim=0).to(device)   # (B_t, D)
        actions_batch = torch.tensor(
            actions_t_list, dtype=torch.long, device=device
        ).view(-1, 1)                                                  # (B_t, 1)
        G_batch = torch.stack(G_t_list, dim=0).to(device)             # (B_t, 1)

        # ----------------------------------------------------------
        # ADVANTAGE COMPUTATION
        # ----------------------------------------------------------
        # Standard case (B_t > 1):
        #   A_t = (G_t - mean(G_t)) / (std(G_t) + eps)
        #
        # Single-episode fallback (B_t = 1):
        #   A_t = (G_t - mean_return) / (std_return + eps)
        #   Uses global return statistics as baseline + scale, preventing
        #   the degenerate case (G - G.mean()) = 0 when B = 1.
        # ----------------------------------------------------------
        if G_batch.shape[0] > 1:
            adv_batch = G_batch - G_batch.mean()
            std = G_batch.std(unbiased=False)
            if std > 1e-8 and not torch.isnan(std):
                adv_batch = adv_batch / (std + 1e-8)
        else:
            adv_batch = G_batch - mean_return
            if std_return > 1e-8:
                adv_batch = adv_batch / (std_return + 1e-8)

        # Forward pass through the policy network
        logits = policy_net(states_batch)                  # (B_t, n_actions)

        # P1 FIX: Reapply the valid action mask that was used during
        # action sampling.  Without this, the log_softmax denominator
        # includes probability mass on invalid actions, making
        # log π_update(a|s) ≠ log π_collect(a|s).  This inconsistency
        # corrupts the policy gradient direction.
        #
        # Build (B_t, n_actions) mask tensor.  If any episode in the
        # batch has a mask, apply it; episodes without masks get all-True.
        has_masks = any(m is not None for m in masks_t_list)
        if has_masks:
            n_act = logits.shape[-1]
            mask_np = np.stack([
                m if m is not None else np.ones(n_act, dtype=bool)
                for m in masks_t_list
            ], axis=0)  # (B_t, n_actions)
            mask_tensor = torch.tensor(
                mask_np, dtype=torch.bool, device=logits.device
            )
            logits = logits.masked_fill(~mask_tensor, float("-inf"))

        log_probs_all = F.log_softmax(logits, dim=-1)      # (B_t, n_actions)
        probs_all = torch.exp(log_probs_all)               # (B_t, n_actions)

        # Log-probability of the chosen action: (B_t, 1)
        action_log_probs = log_probs_all.gather(1, actions_batch)

        # Entropy of the VALID action distribution: (B_t, 1)
        # H = -sum_a pi(a|s) * log pi(a|s)
        # After masking, invalid actions have prob=0 and log_prob=-inf.
        # The product 0 * (-inf) = nan, so we replace nan with 0.
        ent_terms = probs_all * log_probs_all
        ent_terms = torch.nan_to_num(ent_terms, nan=0.0)
        entropy_t = -torch.sum(ent_terms, dim=-1, keepdim=True)

        # Policy gradient loss: L_pg = -log pi(a|s) * A
        pg_loss_t = -action_log_probs * adv_batch  # (B_t, 1)

        # Total loss: L = mean(L_pg - entropy_coef * H)
        # The minus sign on entropy means we MAXIMIZE entropy (exploration).
        total_loss_t = (pg_loss_t - entropy_coef * entropy_t).mean()

        # Accumulate scalar statistics for logging
        sum_total_loss += float(total_loss_t.item())
        sum_pg_loss += float(pg_loss_t.mean().item())
        sum_entropy += float(entropy_t.mean().item())
        n_steps += 1

        if update_frequency == "time_step":
            # One optimizer step per time step
            optimizer.zero_grad()
            total_loss_t.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        else:
            # "episode" mode: GRADIENT ACCUMULATION
            # Scale the loss by 1/max_len so that accumulated gradients
            # are averaged over the episode length. Then call backward()
            # immediately to free the computation graph.
            scaled_loss = total_loss_t / float(max_len)
            scaled_loss.backward()
            # Computation graph is released here — no OOM risk.

    # -----------------------------------------------------------------
    # Apply the single optimizer step for "episode" mode
    # -----------------------------------------------------------------
    if update_frequency == "episode" and n_steps > 0:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

    # Compute mean statistics for logging
    if n_steps > 0:
        mean_loss = sum_total_loss / n_steps
        mean_pg_loss = sum_pg_loss / n_steps
        mean_entropy = sum_entropy / n_steps
    else:
        mean_loss = 0.0
        mean_pg_loss = 0.0
        mean_entropy = 0.0

    return {
        "mean_loss": mean_loss,
        "mean_pg_loss": mean_pg_loss,
        "mean_entropy": mean_entropy,
        "mean_return": mean_return,
    }


# ============================================================
# REINFORCE Controller with Throttle + SMDP
# ============================================================

class PolicyGradientController(RLController):
    """
    Episodic REINFORCE controller (policy-gradient) for the MarketMaker.

    This controller implements Monte Carlo REINFORCE with:
        - Advantage baseline (per-time-step normalization)
        - Entropy regularization
        - Three independent throttle gates (TOB, Event, Time)
        - SMDP reward aggregation between decisions
        - enable_learning toggle for train/eval mode
        - Checkpoint save/load
        - LR scheduling (StepLR)

    Structural Modes
    ----------------
    1) Generic mode (pure_mm=False):
        - 6 discrete actions (post_bid, post_ask, post_bid_ask,
          cancel_bid, cancel_ask, hold). Optional extensions >= 7
          for inside-spread actions.
        - At most one BID LO and one ASK LO.
        - Inventory band: inv >= +limit -> no new BIDs,
                          inv <= -limit -> no new ASKs.

    2) Pure MM mode (pure_mm=True):
        - Each action selects an offset pair (delta_bid, delta_ask).
        - Macro-actions: cancel_all_then_place_bid_ask / cancel_all_then_place.
        - Inventory band controls single-sided quoting.

    Usage Pattern
    -------------
    The runner calls the controller at every micro-step:

        action = controller.act(state_before)
        # ... environment processes action ...
        controller.learn(step_idx, mm, lob, state_before, state_after, reward, info)

    At episode end:

        controller.finish_episode(total_reward)

    When throttling is active:
        - act() may return ("hold",) without querying the network.
        - learn() accumulates rewards via SMDP aggregation.
        - finish_episode() processes the coarsened SMDP-level trajectory.
    """

    def __init__(
        self,
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        entropy_coef: float = 0.01,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: str = "runs/reinforce_mm",
        update_frequency: str = "time_step",
        # ----- MM PURE mode flags / config -----
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        # ----- Policy network architecture -----
        n_hidden: int = 2,
        n_neurons: int = 128,
        # ----- Learning toggle -----
        enable_learning: bool = True,
        # ----- Throttle gating parameters -----
        # These mirror the DQN controller's throttle interface exactly.
        # All gates default to OFF, so the controller queries the network
        # on every micro-step (original behavior) unless explicitly enabled.
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_event_update: bool = False,
        n_events: int = 100,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,
        # ----- MDP mode (disable all bypass) -----
        use_mdp: bool = False,
        # ----- LR scheduling -----
        lr_scheduler_step_size: int = 100,
        lr_scheduler_gamma: float = 0.95,
    ):
        # =============================================================
        # CORE HYPERPARAMETERS
        # =============================================================
        self.level_offset = int(level_offset)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)

        # BUG FIX: weight_decay and grad_clip_norm were previously
        # hardcoded (weight_decay=0.01 in AdamW default, grad_clip=1.0
        # inside reinforce_update_batch). Now exposed for tuning.
        # weight_decay provides L2 regularization on the policy network
        # weights via AdamW's decoupled weight decay. grad_clip_norm
        # prevents catastrophic updates from high-variance REINFORCE
        # gradients (common in early training when returns vary wildly).
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)

        if update_frequency not in ("time_step", "episode"):
            raise ValueError(
                f"update_frequency must be 'time_step' or 'episode', "
                f"got '{update_frequency}'"
            )
        self.update_frequency = update_frequency
        self.pi_n_hidden = int(n_hidden)
        self.pi_n_neurons = int(n_neurons)

        # =============================================================
        # PURE-MM CONFIGURATION
        # =============================================================
        self.pure_mm = bool(pure_mm)
        self.inv_limit = None if inv_limit is None else int(inv_limit)

        if self.pure_mm:
            # Default grid: matches the DQN controller's default grid
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

            # BUG FIX (Codex P1): Validate that the offsets list is non-empty.
            # With pure_mm_offsets=[], n_actions would be 0, the policy network
            # would have an empty output layer (nn.Linear(n_neurons, 0)), and
            # _sample_action_idx() would crash on softmax/argmax over an
            # empty dimension.  Catch this early with a clear error message.
            if len(self.pure_mm_offsets) == 0:
                raise ValueError(
                    "pure_mm_offsets cannot be empty: need at least one "
                    "(bid_offset, ask_offset) pair to define the action space. "
                    "Pass pure_mm_offsets=None to use the default 4-action grid."
                )

            # In pure_mm, n_actions is defined by the grid size
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
            # At inv_limit, one side of the (bid_off, ask_off) pair is
            # dropped by the execution layer.  Actions that differ only
            # on the dropped side produce IDENTICAL outcomes.  For each
            # equivalence group (actions sharing the same recovery-side
            # offset), the smallest-index member is the canonical
            # representative.  The boolean masks below are True only for
            # canonical reps and are used in _get_valid_action_mask() to
            # prevent degenerate actions from being sampled.
            # ---------------------------------------------------------
            _canon_ask: Dict[int, int] = {}  # ask_off → first a_idx
            _canon_bid: Dict[int, int] = {}  # bid_off → first a_idx
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
                    "[PolicyGradientController] WARNING: pure_mm=False but "
                    f"n_actions < 6 (n_actions={self.n_actions}). "
                    "Overriding to 6 generic actions."
                )
                self.n_actions = 6
            elif self.n_actions > 9:
                # BUG FIX (Codex P1): The generic action mapping only defines
                # 9 actions (indices 0-8).  If n_actions > 9, the extra network
                # outputs would be clamped to index 8 (post_ask_inside_spread)
                # inside _mm_action_from_idx_generic(), meaning multiple distinct
                # network outputs map to the same physical action.  This creates
                # dead neurons and distorts the policy gradient.  Cap to 9 with
                # an explicit warning instead of silently clamping downstream.
                print(
                    f"[PolicyGradientController] WARNING: n_actions={self.n_actions} "
                    "exceeds the 9 defined generic actions (indices 0-8). "
                    "Clamping to 9 to prevent dead output neurons and "
                    "distorted policy gradients."
                )
                self.n_actions = 9
            elif self.n_actions > 6:
                print(
                    "[PolicyGradientController] INFO: pure_mm=False with "
                    f"extended actions (n_actions={self.n_actions}). "
                    "Indices >= 6 are inside-spread actions."
                )

        # =============================================================
        # DEVICE SELECTION
        # =============================================================
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # =============================================================
        # POLICY NETWORK pi(a|s)
        #
        # Generic mode state:
        #   [spread, asksize, bidsize, inventory, has_bid, has_ask] -> 6
        #
        # Pure-MM state (matches DQN controller):
        #   [spread, inventory, bid_sizes[0..K], ask_sizes[0..K]]
        #   where K = self.max_offset, so total = 2 + 2*(K+1)
        # =============================================================
        if self.pure_mm:
            input_dim = 2 + 2 * (self.max_offset + 1)
        else:
            input_dim = 6

        self.policy_net = PolicyNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            n_hidden=self.pi_n_hidden,
            n_neurons=self.pi_n_neurons,
        ).to(self.device)

        # =============================================================
        # ENABLE LEARNING TOGGLE
        #
        # Controls whether the controller trains or just infers.
        # When disabled:
        #   - policy_net is in eval() mode (deterministic)
        #   - act() uses greedy argmax instead of sampling
        #   - learn() becomes a no-op
        # =============================================================
        self._enable_learning = bool(enable_learning)
        if self._enable_learning:
            self.policy_net.train()
        else:
            self.policy_net.eval()

        # =============================================================
        # OPTIMIZER + LR SCHEDULER
        # =============================================================
        # BUG FIX: weight_decay is now an explicit parameter instead of
        # relying on AdamW's default (0.01).  This allows tuning L2
        # regularization strength — some RL practitioners prefer 0.0
        # (equivalent to plain Adam) to avoid biasing the policy weights.
        self.optimizer = AdamW(
            self.policy_net.parameters(), lr=lr, weight_decay=self.weight_decay,
        )

        self.lr_scheduler_step_size = int(lr_scheduler_step_size)
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        self.lr_scheduler = StepLR(
            self.optimizer,
            step_size=self.lr_scheduler_step_size,
            gamma=self.lr_scheduler_gamma,
        )

        # =============================================================
        # EPISODE BUFFERS
        #
        # These store the SMDP-level trajectory for the current episode.
        # When no throttle is active, each entry corresponds to one
        # micro-step (original behavior). When throttle is active, each
        # entry corresponds to one SMDP decision with an aggregated
        # reward R_cum that spans multiple micro-steps.
        # =============================================================
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        # BUG FIX (BUG 1): Store the SMDP holding time K for each
        # decision so that reinforce_update_batch() can compute
        # gamma^K between SMDP transitions instead of gamma^1.
        # Without this, future rewards are under-discounted when K > 1.
        self.episode_k_steps: List[int] = []
        # P1 FIX: Store the valid action mask used at each decision so
        # that reinforce_update_batch() can reapply it when computing
        # log_softmax.  Without this, the gradient uses the unmasked
        # distribution π(a|s) instead of the masked distribution
        # π_mask(a|s) that was actually used to sample the action.
        self.episode_masks: List[np.ndarray] = []

        # Last valid mask (set by act(), stored in SMDP pending by learn())
        self._last_valid_mask: Optional[np.ndarray] = None

        # Last action index (set by act(), read by learn())
        self.last_action_idx: Optional[int] = None

        # =============================================================
        # THROTTLE GATE CONFIGURATION
        # =============================================================
        # Hard throttling limits HOW OFTEN the RL agent queries the
        # neural network for a new decision.  Between decisions, the
        # agent simply holds its current position (returns "hold").
        #
        # THREE INDEPENDENT GATES (OR logic — any gate blocking = throttled):
        #
        #   Gate 1: TOB Gating (use_tob_update)
        #     - Counts Top-Of-Book (L1) changes since last decision.
        #     - If moves_tob < threshold_tob -> BLOCKING.
        #     - "TOB change" = any change in (best_bid, best_ask,
        #       bid_size, ask_size). Both price moves and queue size
        #       changes count as TOB events.
        #     - Rationale: Don't react to events that don't change L1.
        #
        #   Gate 2: Event Gating (use_event_update)
        #     - Counts simulator micro-steps since last decision.
        #     - If event_steps < threshold_events -> BLOCKING.
        #     - Rationale: Batch multiple events before re-evaluating.
        #
        #   Gate 3: Time Gating (use_time_update)
        #     - Measures elapsed simulation time since last decision.
        #     - If (now - last_update_time) < min_time_interval -> BLOCKING.
        #     - Rationale: Model realistic latency constraints.
        #
        # OR-blocking / AND-to-unblock semantics:
        #   The controller is throttled if ANY enabled gate is blocking.
        #   It is only unthrottled when ALL enabled gates are satisfied.
        #   This matches the DQN controller's throttle behavior exactly.
        #
        # TWO BYPASS PRIORITIES (override all gates):
        #
        #   Priority A: Mode Change
        #     - Inventory crossed a limit (e.g., two_sided → ask_only).
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
        # =============================================================
        self.use_tob_update = bool(use_tob_update)
        self.threshold_tob = max(1, int(n_tob_moves))

        self.use_event_update = bool(use_event_update)
        self.threshold_events = max(1, int(n_events))

        self.use_time_update = bool(use_time_update)
        self.min_time_interval = float(min_time_interval)

        # MDP mode: disable all bypass (fills, mode changes).
        # The agent strictly respects the throttle — fixed decision timing.
        #   - Event gate: K is exactly fixed (every N events).
        #   - Time gate:  Δt is fixed, but K varies (stochastic arrivals).
        #   - TOB gate:   K varies (depends on L1 move frequency).
        # In all cases, bypasses no longer inject extra decisions.
        self.use_mdp = bool(use_mdp)

        # Throttle counters (reset after each decision via _reset_clocks())
        self.moves_tob: int = 0
        self.last_env_tob_key: Optional[Tuple[int, int, int, int]] = None
        self.event_steps: int = 0
        self.last_update_time: float = -1.0   # -1 = no baseline yet

        # First-action bypass flag.  Set to True after the first real
        # decision is made (inside _reset_clocks).  While False, ALL
        # throttle gates are bypassed so the agent can place its initial
        # quotes immediately.  This replaces the previous approach of
        # testing last_update_time < 0, which broke when use_time_update
        # was False (last_update_time stayed -1 forever, permanently
        # bypassing Event and TOB gates).
        self._has_acted_once: bool = False

        # Bypass trackers (detect fills and mode changes)
        self.last_inventory: Optional[int] = None
        self.last_mode: Optional[str] = None

        # =============================================================
        # SMDP (Semi-Markov Decision Process) STATE
        #
        # These variables track the "pending" SMDP transition that is
        # being built up between consecutive decisions.
        #
        #   _smdp_pending     : True if we have an open (uncommitted)
        #                       transition being accumulated.
        #   _smdp_s_decision  : State dict at the time of the decision.
        #   _smdp_a_decision  : Action index chosen at decision time.
        #   _smdp_cum_reward  : Gamma-discounted cumulative reward:
        #                       R = r_0 + gamma^1*r_1 + ... + gamma^{k-1}*r_{k-1}
        #   _smdp_k_steps     : Number of micro-steps accumulated so far.
        #   _last_was_decision: Flag set by act() to tell learn() whether
        #                       this step was a real decision (True) or a
        #                       throttled hold (False).
        # =============================================================
        self._smdp_pending: bool = False
        self._smdp_s_decision: Optional[Dict[str, Any]] = None
        self._smdp_a_decision: Optional[int] = None
        self._smdp_mask_decision: Optional[np.ndarray] = None  # P1 FIX
        self._smdp_cum_reward: float = 0.0
        self._smdp_k_steps: int = 0
        self._last_was_decision: bool = False

        # =============================================================
        # TENSORBOARD WRITER + EPISODE COUNTER
        # =============================================================
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_idx: int = 0

        # =============================================================
        # DEBUG PRINTOUT — mirrors DQN controller style
        # =============================================================
        print("\n================ REINFORCE CONTROLLER SETUP ================")
        print(f"pure_mm          : {self.pure_mm}")
        print(f"inv_limit        : {self.inv_limit}")
        if self.pure_mm:
            print(f"pure_mm_offsets  : {self.pure_mm_offsets}")
            print(f"max_offset       : {self.max_offset}")
            print(f"input_dim        : {input_dim}")

        print("\n--- Throttle Configuration ---")
        print(f"Time Gating      : {self.use_time_update} "
              f"(min {self.min_time_interval}s)")
        print(f"Event Gating     : {self.use_event_update} "
              f"(min {self.threshold_events} steps)")
        print(f"TOB Gating       : {self.use_tob_update} "
              f"(min {self.threshold_tob} moves)")
        print(f"MDP Mode (no bypass): {self.use_mdp}")

        print("\n--- Core Hyperparameters ---")
        print(f"gamma            : {self.gamma}")
        print(f"lr               : {lr}")
        print(f"entropy_coef     : {self.entropy_coef}")
        print(f"update_frequency : {self.update_frequency}")
        print(f"enable_learning  : {self._enable_learning}")
        print(f"device           : {self.device}")
        print(f"pi_n_hidden      : {self.pi_n_hidden}")
        print(f"pi_n_neurons     : {self.pi_n_neurons}")
        print(f"n_actions        : {self.n_actions}")
        print(f"weight_decay     : {self.weight_decay}")
        print(f"grad_clip_norm   : {self.grad_clip_norm}")

        print("\n--- LR Scheduler ---")
        print(f"step_size        : {self.lr_scheduler_step_size}")
        print(f"gamma            : {self.lr_scheduler_gamma}")
        print("============================================================\n")

    # ================================================================
    # STATE CONVERSION
    # ================================================================

    def _state_to_tensor(self, s: Dict[str, Any]) -> torch.Tensor:
        """
        Convert the MarketMaker state dictionary into a (1, D) float tensor.

        Normalization Strategy
        ----------------------
        1. Spread: math.log1p(max(0.0, spread)) to handle large spreads
           and prevent crashes on negative/crossed spreads.

           WHY max(0.0, ...)?
           When the book is crossed (best_ask < best_bid) or empty,
           raw_spread can be negative or undefined. math.log1p(x) requires
           x > -1, so a negative spread would crash. Clamping to 0.0
           maps crossed books to log1p(0) = 0.0, which is distinct from
           a 1-tick spread (log1p(1) = 0.693). This is a safe encoding
           that the network can learn to interpret as "degenerate book".
           This matches the DQN controller's normalization.

        2. Volumes: math.log1p(volume) to compress the distribution.
           A queue depth of 5000 becomes ~8.5, keeping the input range
           manageable for the neural network.

        3. Inventory: inventory / inv_limit to map approximately to [-1, 1].
           If inv_limit is None, we use 10.0 as a fallback denominator.
        """
        # --- Normalization Constants ---
        inv_denom = float(self.inv_limit) if self.inv_limit is not None else 10.0
        inv_denom = max(inv_denom, 1.0)  # Guard against division by zero

        if not self.pure_mm:
            # ==========================================================
            # GENERIC MODE (pure_mm=False)
            # State: [spread, asksize, bidsize, inventory, has_bid, has_ask]
            # ==========================================================

            # 1. Spread (Log1p with negative clamp)
            raw_spread = float(s["spread"])

            if raw_spread <= -1.0:
                # Diagnostic warning for severely crossed books.
                # This should be rare in a well-configured simulation.
                print(
                    "[BAD SPREAD]", raw_spread,
                    "best_bid", s.get("best_bid"),
                    "best_ask", s.get("best_ask"),
                    "mid", s.get("mid"),
                )

            # BUG 1 FIX: clamp negative spread to 0.0 before log1p.
            # Without this, math.log1p(raw_spread) crashes when
            # raw_spread < -1 (ValueError: math domain error).
            spread = math.log1p(max(0.0, raw_spread))

            # 2. Volumes (Log1p compression)
            asksize = math.log1p(max(0.0, float(s.get("asksize", 0.0))))
            bidsize = math.log1p(max(0.0, float(s.get("bidsize", 0.0))))

            # 3. Inventory (Scaled to approximately [-1, 1])
            raw_inv = float(s["inventory"])
            inventory = raw_inv / inv_denom

            # 4. Binary flags (order presence)
            has_bid = 1.0 if bool(s.get("has_bid", False)) else 0.0
            has_ask = 1.0 if bool(s.get("has_ask", False)) else 0.0

            arr = np.array(
                [spread, asksize, bidsize, inventory, has_bid, has_ask],
                dtype=np.float32,
            )
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        # ==========================================================
        # PURE MM MODE (pure_mm=True)
        # State: [spread, inventory, bid_sizes[0..K], ask_sizes[0..K]]
        # ==========================================================

        # 1. Spread (Log1p with negative clamp)
        raw_spread = float(s["spread"])
        # BUG 1 FIX: same clamp as generic mode
        spread = math.log1p(max(0.0, raw_spread))

        # 2. Inventory (Scaled)
        raw_inv = float(s["inventory"])
        inventory = raw_inv / inv_denom

        K = int(self.max_offset)

        # Try to read aggregated sizes from the state dict.
        # These are provided by the runner when it computes per-level
        # queue depths. If unavailable, fall back to L1 only.
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

        # Pad / truncate to exactly K+1 entries
        if len(bid_sizes) < K + 1:
            bid_sizes += [0.0] * (K + 1 - len(bid_sizes))
        if len(ask_sizes) < K + 1:
            ask_sizes += [0.0] * (K + 1 - len(ask_sizes))

        # 3. Apply Log1p to volume features (critical for neural net scaling)
        bid_sizes = [math.log1p(max(0.0, x)) for x in bid_sizes[: K + 1]]
        ask_sizes = [math.log1p(max(0.0, x)) for x in ask_sizes[: K + 1]]

        # Final state vector: [spread, inventory, bid_sizes..., ask_sizes...]
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
        Build a Top-Of-Book (TOB) fingerprint from the state dictionary.

        This key is used to detect market structure changes. A "TOB change"
        is defined as any change in the best bid/ask prices OR in the
        queue sizes at those levels. Both price moves and size changes
        (e.g., new orders joining or leaving the queue) count as TOB events.

        IMPLEMENTATION NOTE: bid_size and ask_size are rounded to int
        before comparison (``int(round(...))``).  This means sub-unit
        size changes (e.g., 5.1 -> 5.4) are NOT detected as TOB events.
        In the Santa Fe model this is benign (all queue sizes are integer
        multiples of order_size=1), but should be revisited if adapting
        to real-data simulators with fractional sizes.

        We prefer the ``_env`` variants (which exclude the MM's own orders)
        and fall back to standard keys if unavailable.

        Returns
        -------
        tuple of (int, int, int, int)
            (best_bid_price, best_ask_price, bid_size, ask_size)
            as a hashable key for change detection.
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

        Called from act() whenever should_act=True. This starts a fresh
        counting window for all three gates:

            TOB gate:   reset moves_tob to 0, re-anchor baseline key.
            Event gate: reset event_steps to 0.
            Time gate:  record current_time as the new baseline.

        The next action will only be allowed once all enabled gates
        accumulate enough activity to exceed their thresholds again.
        """
        # Mark that the agent has now acted at least once.  This
        # disables the first-action bypass for all subsequent steps.
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

        Called at the start of each new episode (inside start_episode())
        and after episode termination cleanup.  Prepares the controller
        for a fresh episode trajectory by zeroing all SMDP accumulators.

        REINFORCE vs DQN difference:
            DQN resets SMDP after committing each transition to the
            replay buffer. REINFORCE resets once per episode because
            the full trajectory is needed for Monte Carlo returns.
        """
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_mask_decision = None   # P1 FIX
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0
        self._last_was_decision = False

    def _reset_throttle_state(self) -> None:
        """
        P2 FIX: Reset all throttle gate counters and bypass trackers
        to their initial (start-of-episode) values.

        Without this reset, from episode 2 onward:
          - _has_acted_once remains True, so the first-action bypass
            never fires — the agent stays throttled with stale counters
            and cannot place initial quotes immediately.
          - event_steps / moves_tob / last_update_time carry over from
            the previous episode's final values, corrupting the first
            throttle window of the new episode.
          - last_inventory / last_mode carry stale values that can
            spuriously trigger fill/mode-change bypasses on the very
            first step of the new episode.

        Called by:
          - finish_episode() — normal per-episode cleanup.
          - ReinforceRunner — manual inter-env cleanup (when the runner
            bypasses finish_episode() for batch updates).
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

        This mirrors the DQN controller's _get_valid_action_mask() exactly.
        The mask is consumed by _sample_action_idx() to restrict both
        sampling (training) and greedy (eval) selection to the valid subset.

        GENERIC MODE masking rules:
            inv >= +inv_limit (long):
                BLOCK  0 (post_bid)        — would increase long exposure
                BLOCK  2 (post_bid_ask)    — includes a bid
                BLOCK  3 (cancel_bid)      — no-op (bid consumed by fill)
                BLOCK  4 (cancel_ask)      — would remove recovery order
                BLOCK  6 (bid_ask_inside)  — includes a bid  (if exists)
                BLOCK  7 (bid_inside)      — pure bid        (if exists)
                ALLOW  1 (post_ask), 5 (hold)
                ALLOW  8 (ask_inside)      (if exists)

            inv <= -inv_limit (short):
                BLOCK  1 (post_ask)        — would increase short exposure
                BLOCK  2 (post_bid_ask)    — includes an ask
                BLOCK  3 (cancel_bid)      — would remove recovery order
                BLOCK  4 (cancel_ask)      — no-op (ask consumed by fill)
                BLOCK  6 (bid_ask_inside)  — includes an ask (if exists)
                BLOCK  8 (ask_inside)      — pure ask        (if exists)
                ALLOW  0 (post_bid), 5 (hold)
                ALLOW  7 (bid_inside)      (if exists)

        PURE MM MODE:
            At inventory limits, actions differing only on the dropped
            side are degenerate.  We mask non-canonical duplicates,
            keeping only the smallest-index representative per
            recovery-side offset.  Away from limits, mask is all-True.
        """
        # Start with all actions allowed.  We'll flip specific
        # entries to False when the inventory state forbids them.
        mask = np.ones(self.n_actions, dtype=bool)

        # No inventory limit configured → all actions always valid
        if self.inv_limit is None:
            return mask

        inv = float(mm_state.get("inventory", 0.0))

        # ---------------------------------------------------------------
        # PURE MM mode — canonical degenerate masking
        # ---------------------------------------------------------------
        # At inv_limit, one side of the (bid_off, ask_off) pair is
        # dropped by the execution layer.  Actions that differ only
        # on the dropped side produce identical outcomes (same price,
        # same reward, same next state).  We keep only one canonical
        # representative per equivalence group (the smallest-index
        # action with the same recovery-side offset).
        #
        # Example with grid4_passive [(0,0), (0,1), (1,0), (1,1)]:
        #   At +limit (ask-only), bid offset is dropped:
        #     Group ask_off=0: a=0 (canonical), a=2 (duplicate) → mask[2]=F
        #     Group ask_off=1: a=1 (canonical), a=3 (duplicate) → mask[3]=F
        #     Result: mask = [True, True, False, False]
        # ---------------------------------------------------------------
        if self.pure_mm:
            if inv >= self.inv_limit:
                mask = self._canonical_mask_long.copy()
            elif inv <= -self.inv_limit:
                mask = self._canonical_mask_short.copy()
            return mask

        # ---------------------------------------------------------------
        # GENERIC mode — block inventory-violating actions
        # ---------------------------------------------------------------
        # At the long limit, the only useful action is post_ask (recover
        # inventory by selling).  We also allow hold (do nothing) as a
        # safe default.  Everything else is either dangerous (would
        # increase exposure) or a no-op (the relevant order doesn't exist
        # because it was consumed by the fill that pushed us to the limit).
        # ---------------------------------------------------------------
        if inv >= self.inv_limit:
            # Long limit reached: only recovery-side actions allowed.
            mask[0] = False   # post_bid          — would worsen long exposure
            mask[2] = False   # post_bid_ask      — includes a bid
            mask[3] = False   # cancel_bid        — no-op (bid was consumed by fill)
            mask[4] = False   # cancel_ask        — would remove recovery order
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside — includes a bid
            if self.n_actions >= 8:
                mask[7] = False   # bid_inside     — pure bid

        elif inv <= -self.inv_limit:
            # Short limit reached: only recovery-side actions allowed.
            mask[1] = False   # post_ask          — would worsen short exposure
            mask[2] = False   # post_bid_ask      — includes an ask
            mask[3] = False   # cancel_bid        — would remove recovery order
            mask[4] = False   # cancel_ask        — no-op (ask was consumed by fill)
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside — includes an ask
            if self.n_actions >= 9:
                mask[8] = False   # ask_inside     — pure ask

        # Safety fallback: if all actions got masked (should not happen
        # with the rules above), allow everything to prevent a crash.
        if not mask.any():
            mask[:] = True

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
        Select an action index from the current policy pi(.|s),
        constrained by the optional valid_mask.

        Masking Mechanism
        -----------------
        When valid_mask is provided, invalid actions get their logits
        set to ``-inf`` before softmax/argmax.  After softmax, these
        entries become exactly 0.0, so the Categorical distribution
        assigns zero probability to forbidden actions.  This guarantees
        the policy NEVER selects an inventory-violating or degenerate
        action, regardless of what the network outputs.

        This is the REINFORCE equivalent of the DQN controller's
        ``_epsilon_greedy_action(valid_mask=...)`` — both controllers
        now enforce identical constraints on the action space.

        Two modes:
        ----------
        Training (enable_learning=True):
            Sample from the Categorical distribution defined by
            softmax(masked_logits).  The stochastic sampling provides
            natural exploration (no epsilon needed).

        Evaluation (enable_learning=False):
            Greedy argmax over masked_logits with torch.no_grad().
            Deterministic policy execution.

        Why torch.no_grad()?
        --------------------
        REINFORCE computes its policy gradient in reinforce_update_batch()
        via a SEPARATE forward pass on stored states.  The computation
        graph built here would be immediately discarded — wasting GPU
        memory and compute.  no_grad() only suppresses gradient tracking,
        not sampling randomness.

        Parameters
        ----------
        state_tensor : torch.Tensor
            Tensor of shape (1, D).
        valid_mask : np.ndarray or None
            Boolean mask, shape (n_actions,). True = action allowed,
            False = action forbidden.  If None, all actions are valid.

        Returns
        -------
        a_idx : int
            Discrete action index in [0, n_actions - 1], guaranteed
            to be a valid action if valid_mask was provided.
        """
        with torch.no_grad():
            logits = self.policy_net(state_tensor)    # (1, n_actions)

            # Apply action mask: set invalid logits to -inf so that
            # softmax maps them to exactly 0.0 probability.
            if valid_mask is not None:
                mask_tensor = torch.tensor(
                    valid_mask, dtype=torch.bool, device=logits.device
                )
                logits = logits.masked_fill(~mask_tensor, float("-inf"))

            if not self._enable_learning:
                # Eval mode: greedy argmax (deterministic)
                a_idx = int(torch.argmax(logits, dim=-1).item())
            else:
                # Training mode: sample from masked softmax distribution.
                # Invalid actions have -inf logits → 0.0 probability →
                # never sampled.  Valid actions keep their relative
                # probabilities unchanged (softmax is invariant to
                # additive constants on the valid subset).
                probs = torch.softmax(logits, dim=-1)
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

        Here the action space is a list of (bid_offset, ask_offset) pairs:
            - Positive offsets: more passive (further from mid, wider spread).
            - Zero offset: quote at L1 (best bid / best ask).
            - Negative offsets: improve inside the spread (more aggressive),
              but ONLY if spread >= 2 ticks to prevent crossing.

        Price Computation (3 tiers)
        ---------------------------
        Tier 1: Both L1 known (best_bid and best_ask available).
            For positive offsets: move AWAY from L1 (bid_price = best_bid - offset).
            For zero offset: quote AT L1.
            For negative offsets: move TOWARD mid (bid_price = best_bid + |offset|),
                clamped to best_ask - 1 to prevent crossing.

        Tier 2: Only one side of L1 known.
            Allow passive moves only. Negative offsets fall back to L1.

        Tier 3: No L1 available (empty book).
            Use mid-based fallback: bid_price = mid_px - offset.
            Note: negative offsets in this fallback CAN produce crossing,
            but the safety check at the bottom of this method (line "enforce
            bid < ask") catches and corrects this.

        Inventory Band Logic
        --------------------
        If inv_limit is set:
            inv >= +inv_limit -> quote ASK only (reduce long exposure)
            inv <= -inv_limit -> quote BID only (reduce short exposure)
            otherwise         -> quote both sides

        Final Safety Check
        ------------------
        If ask_price <= bid_price after all computations, we force
        ask_price = bid_price + 1. This guarantees no locked/crossed quotes
        reach the simulator, regardless of the input state.
        """
        a_idx = int(a_idx)
        inv = float(state.get("inventory", 0.0))

        # Extract L1 prices (may be None if the book is empty on that side)
        best_bid_raw = state.get("best_bid", None)
        best_ask_raw = state.get("best_ask", None)
        best_bid = int(best_bid_raw) if best_bid_raw is not None else None
        best_ask = int(best_ask_raw) if best_ask_raw is not None else None

        # SENTINEL FILTER: The simulator may use -1 (or any negative value)
        # as a sentinel to signal "no L1 price available on this side".
        # Since -1 is not None, the check above would pass and we'd treat
        # -1 as a valid price, leading to arithmetic like:
        #     bid_price = best_bid - offset = -1 - offset = NEGATIVE PRICE
        # We filter out negative sentinels by treating them as None (empty book).
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

        # ---------------------------------------------------------
        # Compute BID price
        # ---------------------------------------------------------
        if best_bid is not None and best_ask is not None:
            # Tier 1: both sides of L1 known
            if bid_off == 0:
                bid_price = best_bid
            elif bid_off > 0:
                bid_price = best_bid - bid_off
            else:
                # Negative offset: attempt inside-spread improvement
                if spread >= 2:
                    candidate = best_bid - bid_off  # bid_off < 0 => increases
                    candidate = min(candidate, best_ask - 1)  # no crossing
                    bid_price = best_bid if candidate <= best_bid else candidate
                else:
                    bid_price = best_bid
        elif best_bid is not None:
            # Tier 2: only bid side known
            bid_price = best_bid - bid_off if bid_off >= 0 else best_bid
        else:
            # Tier 3: no L1 -> mid-based fallback
            # Note: negative offsets here may produce bid_price > mid_px.
            # The safety check below (ask_price <= bid_price) handles this.
            bid_price = mid_px - bid_off

        # ---------------------------------------------------------
        # Compute ASK price
        # ---------------------------------------------------------
        if best_bid is not None and best_ask is not None:
            # Tier 1: both sides of L1 known
            if ask_off == 0:
                ask_price = best_ask
            elif ask_off > 0:
                ask_price = best_ask + ask_off
            else:
                # Negative offset: attempt inside-spread improvement
                if spread >= 2:
                    candidate = best_ask + ask_off  # ask_off < 0 => decreases
                    candidate = max(candidate, best_bid + 1)  # no crossing
                    ask_price = best_ask if candidate >= best_ask else candidate
                else:
                    ask_price = best_ask
        elif best_ask is not None:
            # Tier 2: only ask side known
            ask_price = best_ask + ask_off if ask_off >= 0 else best_ask
        else:
            # Tier 3: no L1 -> mid-based fallback
            # Same caveat as bid side (safety check below handles crossing).
            ask_price = mid_px + ask_off

        bid_price = int(bid_price)
        ask_price = int(ask_price)

        # ---------------------------------------------------------
        # Inventory band logic
        # ---------------------------------------------------------
        if self.inv_limit is not None:
            if inv >= self.inv_limit:
                # Too long: quote ASK only (reduce long exposure)
                return ("cancel_all_then_place", -1, ask_price)
            if inv <= -self.inv_limit:
                # Too short: quote BID only (reduce short exposure)
                return ("cancel_all_then_place", +1, bid_price)

        # ---------------------------------------------------------
        # SAFETY: Enforce bid < ask (no locked/crossed quotes)
        # ---------------------------------------------------------
        # This catches all edge cases including:
        #   - Tier 3 mid-based fallback with negative offsets
        #   - Rounding artifacts
        #   - Empty/crossed book states
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

        Action Space
        ------------
        Base 6 actions (always present):
            0: post_bid             — Place a new bid limit order
            1: post_ask             — Place a new ask limit order
            2: post_bid_ask         — Place both bid and ask
            3: cancel_bid           — Cancel active bid order
            4: cancel_ask           — Cancel active ask order
            5: hold                 — Do nothing

        Extended actions (only if n_actions >= 7):
            6: post_bid_ask_inside_spread  — Quote inside the spread
            7: post_bid_inside_spread      — Bid inside the spread
            8: post_ask_inside_spread      — Ask inside the spread

        Structural Constraints
        ----------------------
        1. At most one working BID LO and one working ASK LO.
           If the agent tries to post a bid when one is already active,
           the action degrades to "hold" (prevents duplicate orders).

        2. Inventory band (if inv_limit is not None):
           inv >= +inv_limit -> do not post NEW bids (risk capped)
           inv <= -inv_limit -> do not post NEW asks (risk capped)

        3. Locked/crossed quote protection (BUG 2 FIX):
           Before returning place_bid_ask, we verify ask > bid.
           If not, we force ask = bid + 1 to prevent the simulator
           from receiving an invalid order pair.

        Price Logic
        -----------
        All limit orders are placed at L1 +/- level_offset:
            bid_price = best_bid - level_offset
            ask_price = best_ask + level_offset

        When L1 is unavailable (empty book), we fall back to mid:
            bid_price = mid_px - max(1, level_offset)
            ask_price = mid_px + max(1, level_offset)
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

        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))
        inv = float(state.get("inventory", 0.0))

        # Translate action name into desired intent
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

        # Inventory constraints (mirror DQN controller)
        if self.inv_limit is not None and (want_bid or want_ask):
            if inv >= self.inv_limit:
                want_bid = False
                use_inside_bid = False
            if inv <= -self.inv_limit:
                want_ask = False
                use_inside_ask = False

        # NOTE: duplicate-order suppression removed for consistency with
        # A2C/PPO/SAC controllers. The MM's _smart_place() handles repricing.

        # If we ended up wanting nothing from a posting action, hold
        if not want_bid and not want_ask and a_name.startswith("post_"):
            return ("hold",)

        # Inside-spread actions are delegated to MM helper commands
        if "inside_spread" in a_name:
            if want_bid and want_ask and use_inside_bid and use_inside_ask:
                return ("place_bid_ask_inside_spread",)
            if want_bid and use_inside_bid and not want_ask:
                return ("place_bid_inside_spread",)
            if want_ask and use_inside_ask and not want_bid:
                return ("place_ask_inside_spread",)
            return ("hold",)

        # Compute prices: L1 +/- level_offset, with mid fallback
        best_bid_raw = state.get("best_bid", None)
        best_ask_raw = state.get("best_ask", None)
        best_bid = int(best_bid_raw) if best_bid_raw is not None else None
        best_ask = int(best_ask_raw) if best_ask_raw is not None else None

        # SENTINEL FILTER: The simulator may use -1 (or any negative value)
        # as a sentinel for "no L1 on this side". Without this filter, -1
        # would pass the `is not None` check and be used arithmetically:
        #     bid_price_l1 = -1 - level_offset = NEGATIVE PRICE
        # We convert negative sentinels to None so the mid-based fallback
        # is used instead (which always produces valid prices).
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

        # Non-posting actions
        if a_name == "hold":
            return ("hold",)
        if a_name == "cancel_bid":
            return ("cancel_bid",)
        if a_name == "cancel_ask":
            return ("cancel_ask",)

        # Posting actions
        if want_bid and want_ask:
            # BUG 2 FIX: Locked/crossed quote protection.
            # Without this check, the simulator could receive an invalid
            # order pair where ask <= bid, causing undefined behavior.
            # This can happen when level_offset=0 and the book is crossed,
            # or when the mid-based fallback produces degenerate prices.
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

        Decision Flow (7-Step Pipeline)
        ===============================

        Step 1: Update throttle counters.
            - Increment event_steps (if Event gate enabled).
            - Check TOB signature for changes (if TOB gate enabled).

        Step 2: Check BYPASS PRIORITY A — Mode Change.
            - If inventory crossed inv_limit boundary since last decision,
              the quoting mode changed (e.g., two_sided -> ask_only).
            - Bypass all throttle gates (unless use_mdp=True).

        Step 3: Check BYPASS PRIORITY B — Fill Replenishment.
            - If inventory changed at all, a fill happened.
            - Bypass throttle to replace filled order (unless use_mdp=True).

        Step 4: Throttle Gate Check (OR-blocking).
            - Event gate:  event_steps < threshold_events -> BLOCKING
            - Time gate:   elapsed_time < min_time_interval -> BLOCKING
            - TOB gate:    moves_tob < threshold_tob -> BLOCKING
            - Any single gate blocking -> agent is throttled.
            - Only passes when ALL enabled gates are satisfied.

        Step 5: Execute decision or hold.
            - should_act=True  → Steps 5a-5d below
            - should_act=False → Return ("hold",)

        Step 5a: Compute valid action mask.
            - Generic mode: blocks inventory-violating actions at limits.
            - Pure MM mode: masks degenerate (canonically equivalent)
              actions at inventory limits.

        Step 5b: Select action (constrained by mask).
            - Training: sample from softmax(masked_logits).
            - Eval: argmax over masked_logits.

        Step 5c: Map action index to MarketMaker command tuple.

        Step 5d: Reset throttle clocks, signal decision to learn().

        Parameters
        ----------
        mm_state : dict
            State dictionary from MarketMaker.build_state().

        Returns
        -------
        tuple
            MarketMaker command, e.g. ("place_bid_ask", 100, 102)
            or ("hold",).
        """
        # ==============================================================
        # STEP 1: UPDATE THROTTLE COUNTERS
        # ==============================================================
        current_time = float(mm_state.get("time", 0.0))
        inv = float(mm_state.get("inventory", 0.0))

        # Event counter: increments on every call (if gate enabled)
        if self.use_event_update:
            self.event_steps += 1

        # TOB counter: check if L1 signature changed
        tob_key = self._get_env_tob(mm_state)
        if self.use_tob_update:
            if self.last_env_tob_key is None:
                # First call: anchor baseline without counting
                self.last_env_tob_key = tob_key
            elif self.last_env_tob_key != tob_key:
                self.moves_tob += 1
                self.last_env_tob_key = tob_key

        # ==============================================================
        # STEP 2: CHECK BYPASS PRIORITY A — MODE CHANGE
        # ==============================================================
        # Determine desired quoting mode from inventory limits
        if self.inv_limit is not None:
            if inv >= self.inv_limit:
                desired_mode = "ask_only"
            elif inv <= -self.inv_limit:
                desired_mode = "bid_only"
            else:
                desired_mode = "two_sided"
        else:
            desired_mode = "two_sided"

        # Detect mode transitions (e.g., two_sided -> ask_only)
        mode_changed = (
            (self.last_mode is not None) and (self.last_mode != desired_mode)
        )
        self.last_mode = desired_mode

        # ==============================================================
        # STEP 3: CHECK BYPASS PRIORITY B — FILL REPLENISHMENT
        # ==============================================================
        # Any inventory change since the last decision indicates a fill.
        # We need to replace the filled order quickly.
        has_fill = False
        if self.last_inventory is not None:
            if int(inv) != int(self.last_inventory):
                has_fill = True
        self.last_inventory = int(inv)

        # ==============================================================
        # STEP 4: DECISION LOGIC — SHOULD WE QUERY THE NETWORK?
        # ==============================================================
        # In MDP mode (use_mdp=True), bypasses A and B are disabled.
        # The agent strictly respects the throttle gates.  With event
        # gating, K is exactly fixed; with time/TOB gating, K varies
        # but the decision rule is still gate-only (no bypass).
        # ==============================================================
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

            # ---------------------------------------------------------
            # First-action bypass (applies to ALL gates at once).
            # Before the agent has acted for the first time in this
            # episode, every gate is treated as satisfied so the
            # agent can place initial quotes immediately.
            #
            # BUG FIX (P0 regression): The previous implementation
            # used `last_update_time < 0` as the "never acted"
            # sentinel.  That broke when use_time_update=False
            # because _reset_clocks() only updates last_update_time
            # when the Time gate is enabled — leaving it at -1
            # forever and permanently bypassing Event/TOB gates.
            # Now we use the dedicated `_has_acted_once` flag,
            # which is unconditionally set in _reset_clocks().
            # ---------------------------------------------------------
            if not self._has_acted_once:
                pass  # Bypass all gates on first action
            else:
                # Event gate check
                if self.use_event_update:
                    if self.event_steps < self.threshold_events:
                        is_throttled = True

                # Time gate check
                if self.use_time_update:
                    if self.last_update_time < 0:
                        pass  # No baseline yet (should not happen after first action)
                    elif (current_time - self.last_update_time) < self.min_time_interval:
                        is_throttled = True

                # TOB gate check
                if self.use_tob_update:
                    if self.moves_tob < self.threshold_tob:
                        is_throttled = True

            # Only act if NOT throttled (all enabled gates satisfied)
            if not is_throttled:
                should_act = True

        # ==============================================================
        # STEP 5: EXECUTE DECISION OR HOLD
        # ==============================================================

        if should_act:
            # === NEW DECISION: Query the neural network ===

            # 1. State to Tensor
            state_tensor = self._state_to_tensor(mm_state)

            # 2. Compute Valid Action Mask (Constrained Selection)
            # ---------------------------------------------------
            # Build a boolean mask marking inventory-violating actions
            # as False.  In generic mode, this blocks bid-side actions
            # at the long limit and ask-side actions at the short limit.
            # In pure_mm mode, it masks degenerate (canonically equivalent)
            # actions at inventory limits.  The mask is applied inside
            # _sample_action_idx() by setting invalid logits to -inf
            # before softmax (training) or argmax (eval).
            valid_mask = self._get_valid_action_mask(mm_state)

            # 3. Select Action (Constrained)
            # The returned a_idx is guaranteed to be valid given the
            # current inventory state — the mask ensures the policy
            # never samples or selects a forbidden action.
            a_idx = self._sample_action_idx(state_tensor, valid_mask=valid_mask)

            # 4. Map Action → MarketMaker Command Tuple
            # Safety nets in _mm_action_from_idx still exist as a
            # defensive fallback (crossed-quote check, inventory band)
            # but should rarely override now that selection is constrained.
            act_tuple = self._mm_action_from_idx(a_idx, mm_state)

            # 5. Store Action + Mask for Learn
            # The mask already ensures that in pure_mm at inv_limit,
            # only canonical representatives (one per recovery-side
            # offset) are selectable — a_idx is canonical by construction.
            # P1 FIX: Also store the valid_mask so that learn() can
            # commit it alongside the transition.  reinforce_update_batch()
            # needs it to reapply the same masking when computing
            # log_softmax during the policy gradient update.
            self.last_action_idx = a_idx
            self._last_valid_mask = valid_mask

            # 6. Reset Throttling Clocks (Action taken)
            self._reset_clocks(current_time, tob_key)

            # 7. SMDP flag: signal to learn() that this was a true
            #    decision.  learn() will commit any pending SMDP
            #    transition and start a new one with this state/action.
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
        SMDP-aware learning step for episodic REINFORCE.

        Overview
        --------
        The runner calls this method EVERY micro-step, regardless of
        whether the controller was throttled or made a real decision.
        This method implements Semi-Markov Decision Process (SMDP)
        reward aggregation to produce correct episode trajectories.

        The Problem (without SMDP)
        --------------------------
        When throttled, act() returns ("hold",) without updating
        last_action_idx. If we naively buffered every micro-step,
        we would create episode transitions where the (state, action)
        pair does not correspond to any real decision. This corrupts
        the policy gradient because it attributes rewards to actions
        that were never actually chosen.

        The Solution (SMDP aggregation)
        --------------------------------
        Instead of buffering one transition per micro-step, we
        aggregate all micro-steps between two consecutive decisions
        into a SINGLE SMDP-level transition:

            (s_decision, a_decision, R_cum)

        where R_cum = sum_{k=0}^{K-1} gamma^k * r_{t+k} is the
        discounted cumulative reward over the K micro-steps.

        At episode end, finish_episode() computes Monte Carlo returns
        from these SMDP-level transitions.

        Two Cases
        ---------
        CASE A — This step was a DECISION (act() queried the network):
            1. If there is a pending SMDP transition from a previous
               decision, commit it to episode buffers now.
            2. Start a NEW pending SMDP transition with:
               s_decision = state_before, a_decision = last_action_idx,
               R_cum = reward, k_steps = 1.

        CASE B — This step was THROTTLED (act() returned "hold"):
            1. Accumulate reward into pending transition:
               R_cum += gamma^k * reward; k_steps += 1.

        SPECIAL CASE — Episode termination (done=True):
            After handling Case A or B, commit any pending transition.

        Parameters
        ----------
        step_idx : int
            Global step counter from the runner.
        mm, lob : objects
            MarketMaker and LOB instances (unused by REINFORCE but
            required by the RLController interface).
        state_before : dict
            State at the START of this micro-step.
        state_after : dict
            State at the END of this micro-step.
        reward : float
            Scalar reward for this micro-step.
        info : dict
            Environment info (contains "done" flag for episode end).
        """
        if self.last_action_idx is None:
            # No action chosen yet (very first step of the simulation)
            return

        done_flag = bool(info.get("done", False))

        # ==============================================================
        # EVAL MODE: freeze learning (no buffer updates, no gradients)
        # ==============================================================
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

            # Step A.1: Commit any previous pending SMDP transition
            # to the episode buffer. The "next state" for the old
            # transition is state_before of THIS step (the state at
            # the moment the new decision was made).
            if self._smdp_pending:
                s_t = self._state_to_tensor(self._smdp_s_decision)
                # MEMORY OPTIMIZATION: move state tensor to CPU to avoid
                # GPU OOM during long episodes. It will be moved back to
                # GPU in reinforce_update_batch() during the forward pass.
                self.episode_states.append(s_t.squeeze(0).detach().cpu())
                self.episode_actions.append(int(self._smdp_a_decision))
                self.episode_rewards.append(float(self._smdp_cum_reward))
                # BUG FIX (BUG 1): Store the SMDP holding time K so that
                # reinforce_update_batch() can compute gamma^K as the
                # inter-transition discount instead of gamma^1.
                self.episode_k_steps.append(int(self._smdp_k_steps))
                # P1 FIX: Store the valid mask that was used at decision
                # time so reinforce_update_batch() can reapply it.
                self.episode_masks.append(self._smdp_mask_decision)

            # Step A.2: Start a new pending SMDP transition
            self._smdp_pending = True
            self._smdp_s_decision = state_before
            self._smdp_a_decision = self.last_action_idx
            self._smdp_mask_decision = self._last_valid_mask  # P1 FIX
            self._smdp_cum_reward = float(reward)
            self._smdp_k_steps = 1

        else:
            # ----- CASE B: Throttled micro-step -----
            # Accumulate reward with gamma^k discounting into the
            # current pending SMDP transition.
            if self._smdp_pending:
                k = self._smdp_k_steps
                self._smdp_cum_reward += (self.gamma ** k) * float(reward)
                self._smdp_k_steps += 1

        # ==============================================================
        # SPECIAL CASE: EPISODE TERMINATION
        # ==============================================================
        # If done=True, commit the pending SMDP transition NOW.
        # This ensures the final rewards are not lost.
        if done_flag:
            if self._smdp_pending:
                s_t = self._state_to_tensor(self._smdp_s_decision)
                self.episode_states.append(s_t.squeeze(0).detach().cpu())
                self.episode_actions.append(int(self._smdp_a_decision))
                self.episode_rewards.append(float(self._smdp_cum_reward))
                # BUG FIX (BUG 1): Store holding time for the final
                # transition too.  Although the last transition's K doesn't
                # affect the Monte Carlo return (there's no G_{T+1} to
                # discount), storing it keeps the buffer lengths consistent
                # and enables diagnostics on SMDP holding time statistics.
                self.episode_k_steps.append(int(self._smdp_k_steps))
                # P1 FIX: Store the mask for the final transition too.
                self.episode_masks.append(self._smdp_mask_decision)
                self._smdp_pending = False

            # Defensive cleanup: reset SMDP + throttle state here AND in
            # finish_episode().  The double-reset is intentional — if the
            # runner fails to call finish_episode(), this ensures the
            # controller is in a clean state for the next episode.
            self.last_action_idx = None
            self._smdp_reset()
            self._reset_throttle_state()

    # ================================================================
    # EPISODIC REINFORCE UPDATE
    # ================================================================

    def finish_episode(self, total_reward: float = 0.0) -> None:
        """
        Perform the REINFORCE policy update using the accumulated trajectory.

        The episode buffer (episode_states, episode_actions, episode_rewards)
        contains SMDP-level transitions when throttling is active, or
        micro-step transitions when no throttle is used.

        In either case, the REINFORCE algorithm processes them identically:
        compute Monte Carlo returns, compute advantages, update the policy.

        SMDP-CORRECT DISCOUNTING (BUG FIX):
        The SMDP aggregation already applies gamma^k within each holding
        period, so the inter-transition discount must be gamma^{K_t} (where
        K_t = holding time of decision t), NOT gamma^1.  Without this
        correction, future rewards are under-discounted by a factor that
        grows exponentially with the holding period length.  The episode
        buffer now stores K_t alongside each transition for this purpose.

        After the update, this method:
            1. Steps the LR scheduler (decay learning rate per episode).
            2. Logs statistics to TensorBoard.
            3. Clears episode buffers.
            4. Resets SMDP state for the next episode.

        Parameters
        ----------
        total_reward : float
            Total environment reward for logging purposes only.
            The actual returns used in the update are computed from
            the episode_rewards buffer.
        """
        T = len(self.episode_rewards)
        if T == 0:
            return

        # Package trajectory into an EpisodeBatch.
        # BUG FIX (BUG 1): Include k_steps (SMDP holding times) so that
        # reinforce_update_batch() can compute SMDP-correct Monte Carlo
        # returns using gamma^{K_t} between transitions instead of gamma^1.
        episode = {
            "states": self.episode_states,
            "actions": self.episode_actions,
            "rewards": self.episode_rewards,
            "k_steps": self.episode_k_steps,
            "masks": self.episode_masks,  # P1 FIX
        }
        batch = EpisodeBatch([episode])

        # Run the REINFORCE update
        stats = reinforce_update_batch(
            policy_net=self.policy_net,
            optimizer=self.optimizer,
            batch=batch,
            gamma=self.gamma,
            entropy_coef=self.entropy_coef,
            device=self.device,
            update_frequency=self.update_frequency,
            grad_clip_norm=self.grad_clip_norm,
        )

        # Step the LR scheduler (once per episode)
        self.lr_scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Extract statistics
        mean_loss = stats["mean_loss"]
        mean_pg_loss = stats["mean_pg_loss"]
        mean_entropy = stats["mean_entropy"]
        mean_return = stats["mean_return"]

        # Console logging
        print(
            f"[REINFORCE] Episode {self.episode_idx} | "
            f"pure_mm={self.pure_mm} | "
            f"update_freq={self.update_frequency} | "
            f"mean_loss={mean_loss:.4f}, "
            f"mean_pg_loss={mean_pg_loss:.4f}, "
            f"mean_entropy={mean_entropy:.4f}, "
            f"mean_return={mean_return:.4f}, "
            f"env_return={total_reward:.4f}, "
            f"T={T} | "
            f"lr={current_lr:.6f}"
        )

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar("loss/total", mean_loss, self.episode_idx)
            self.writer.add_scalar("loss/pg", mean_pg_loss, self.episode_idx)
            self.writer.add_scalar("loss/entropy", mean_entropy, self.episode_idx)
            self.writer.add_scalar("return/mean", mean_return, self.episode_idx)
            self.writer.add_scalar(
                "return/env_total", total_reward, self.episode_idx
            )
            self.writer.add_scalar("lr", current_lr, self.episode_idx)

        # Prepare for next episode
        self.episode_idx += 1
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_k_steps.clear()
        self.episode_masks.clear()        # P1 FIX
        self.last_action_idx = None
        self._smdp_reset()
        self._reset_throttle_state()      # P2 FIX

    # ================================================================
    # ENABLE LEARNING TOGGLE (property)
    # ================================================================

    @property
    def enable_learning(self) -> bool:
        """Get current learning state (True = training, False = eval)."""
        return self._enable_learning

    @enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        """
        Toggle learning mode.

        When learning is ENABLED (True):
            - policy_net is set to train() mode.
            - act() samples actions stochastically (exploration).
            - learn() buffers transitions and accumulates SMDP rewards.

        When learning is DISABLED (False):
            - policy_net is set to eval() mode (deterministic if using
              dropout or batch norm, though our simple MLP is unaffected).
            - act() uses greedy argmax (no exploration).
            - learn() becomes a no-op (no buffer updates, no gradients).

        Use Cases:
            - Backtesting: evaluate a trained policy without further training.
            - Tournament evaluation: frozen policy across all episodes.
            - Curriculum learning: alternate training and evaluation phases.
        """
        self._enable_learning = bool(value)
        if hasattr(self, "policy_net"):
            if self._enable_learning:
                self.policy_net.train()
            else:
                self.policy_net.eval()

    # ================================================================
    # CHECKPOINT SAVE / LOAD
    # ================================================================

    def save_checkpoint(self, path: str) -> None:
        """
        Save controller state to disk for training resumption.

        Saves:
            - policy_net state_dict (network weights)
            - optimizer state_dict (momentum buffers, etc.)
            - lr_scheduler state_dict (current step, last LR)
            - episode_idx (episode counter for logging continuity)

        Parameters
        ----------
        path : str
            File path, e.g. "checkpoints/reinforce_ep100.pt".
            Directories are created automatically if missing.

        Example
        -------
        >>> controller.save_checkpoint("checkpoints/reinforce_ep100.pt")
        [REINFORCE] Checkpoint saved to checkpoints/reinforce_ep100.pt
        """
        # Safety: create parent directory if needed
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Include controller configuration so that make_reinforce_eval_controller()
        # can reconstruct the architecture without manual specification.
        # This also serves as provenance metadata for the checkpoint.
        # BUG FIX (BUG 3): Include training hyperparameters in the
        # checkpoint.  Previously only architecture metadata was saved,
        # so load_checkpoint() for training resumption would silently
        # use whatever defaults the caller passed to the constructor,
        # potentially corrupting training if the original values of
        # gamma, entropy_coef, etc. were different.
        controller_config = {
            # --- Architecture metadata ---
            "pure_mm": self.pure_mm,
            "inv_limit": self.inv_limit,
            "n_actions": self.n_actions,
            "pure_mm_offsets": list(self.pure_mm_offsets) if self.pure_mm_offsets else None,
            "level_offset": self.level_offset,
            "n_hidden": self.pi_n_hidden,
            "n_neurons": self.pi_n_neurons,
            "max_offset": self.max_offset,
            # --- Training hyperparameters (BUG FIX: previously missing) ---
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "update_frequency": self.update_frequency,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "episode_idx": self.episode_idx,
            "controller_config": controller_config,
        }

        torch.save(checkpoint, path)
        print(f"[REINFORCE] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load controller state from disk for training resumption.

        Restores:
            - policy_net weights
            - optimizer state (learning rate, momentum buffers)
            - lr_scheduler state (current step)
            - episode_idx counter

        After loading, the controller can continue training seamlessly
        from where it left off.

        Parameters
        ----------
        path : str
            Path to a checkpoint file saved by save_checkpoint().

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.

        Example
        -------
        >>> controller.load_checkpoint("checkpoints/reinforce_ep100.pt")
        [REINFORCE] Checkpoint loaded from checkpoints/reinforce_ep100.pt
        [REINFORCE] Resuming from episode 100
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # BUG FIX (BUG 5): Explicit weights_only=False.  Since PyTorch 2.0,
        # torch.load() without this parameter emits a FutureWarning.
        # We use False (not True) because the checkpoint contains non-tensor
        # objects (controller_config dict, episode_idx int).
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if "episode_idx" in checkpoint:
            self.episode_idx = int(checkpoint["episode_idx"])

        # BUG FIX (P1): Restore training hyperparameters from the
        # checkpoint's controller_config if present.  Without this,
        # load_checkpoint() silently kept whatever defaults the caller
        # passed to the constructor — potentially corrupting training
        # resumption if the original gamma, entropy_coef, etc. differed.
        if "controller_config" in checkpoint:
            cfg = checkpoint["controller_config"]
            if "gamma" in cfg:
                self.gamma = float(cfg["gamma"])
            if "entropy_coef" in cfg:
                self.entropy_coef = float(cfg["entropy_coef"])
            if "update_frequency" in cfg:
                self.update_frequency = str(cfg["update_frequency"])
            if "grad_clip_norm" in cfg:
                self.grad_clip_norm = float(cfg["grad_clip_norm"])
            if "weight_decay" in cfg:
                self.weight_decay = float(cfg["weight_decay"])
            print(f"[REINFORCE] Restored hyperparams from checkpoint config")

        print(f"[REINFORCE] Checkpoint loaded from {path}")
        print(f"[REINFORCE] Resuming from episode {self.episode_idx}")

    # ================================================================
    # OPTIONAL: External Logging Helper (PnL, env reward)
    # ================================================================

    def log_episode_stats(
        self, episode_idx: int, total_reward: float, final_pnl: float
    ) -> None:
        """
        Log external episode statistics to TensorBoard.

        Call this in addition to finish_episode() if you want PnL and
        reward plots aligned with episode index. This is useful when
        the runner tracks metrics outside the controller.

        Parameters
        ----------
        episode_idx : int
            Episode number for the x-axis.
        total_reward : float
            Total reward from the environment.
        final_pnl : float
            Final PnL at episode end.
        """
        if self.writer is None:
            return
        self.writer.add_scalar(
            "episode/total_reward_env", total_reward, episode_idx
        )
        self.writer.add_scalar("episode/final_pnl", final_pnl, episode_idx)


# ====================================================================
# EVALUATION FACTORY — Load checkpoint and return eval-mode controller
# ====================================================================

def make_reinforce_eval_controller(
    ckpt_path: str,
    device: str = "cpu",
    log_dir: str = "runs/reinforce_eval",
) -> PolicyGradientController:
    """
    Load a REINFORCE checkpoint and return a PolicyGradientController
    configured for pure evaluation (inference only, no training).

    This is the SINGLE ENTRY POINT for loading trained REINFORCE policies
    for evaluation, backtesting, or comparison studies. It replaces the
    former ``ReinforceEvalPolicy`` wrapper class that duplicated the
    state encoding and action mapping logic.

    Why a factory instead of a wrapper class?
    ------------------------------------------
    Previously, ``MM_GLFT_naive_comparison.py`` contained a separate
    ``ReinforceEvalPolicy`` class (~420 lines) that re-implemented
    ``_state_to_tensor()``, ``_mm_action_from_idx_generic()``, and
    ``_mm_action_from_idx_pure_mm()`` — all of which were already
    implemented in ``PolicyGradientController``. This duplication was
    a maintenance hazard: any bug fix or feature addition in the
    controller (e.g., the log1p clamp fix, the crossed-quote check)
    had to be manually mirrored in the wrapper, or the two would drift.

    This factory eliminates that duplication by returning a real
    ``PolicyGradientController`` with ``enable_learning=False``.
    The controller's own ``act()`` method handles state encoding,
    action selection (greedy argmax in eval mode), and action mapping
    — all from a single source of truth.

    Backward Compatibility — Two Checkpoint Formats
    -------------------------------------------------
    This loader transparently handles BOTH checkpoint formats:

    **Format A — "New" format** (from ``PolicyGradientController.save_checkpoint()``):
        checkpoint = {
            "policy_net_state_dict": ...,
            "optimizer_state_dict": ...,
            "lr_scheduler_state_dict": ...,
            "episode_idx": ...,
            "controller_config": {      # <-- architecture metadata
                "pure_mm": bool,
                "inv_limit": int | None,
                "n_actions": int,
                "pure_mm_offsets": list | None,
                "level_offset": int,
                "n_hidden": int,
                "n_neurons": int,
                "max_offset": int,
            },
        }

    **Format B — "Legacy" format** (from older training scripts):
        checkpoint = {
            "policy_state_dict": ...,   # <-- different key name
            "config": {                 # <-- different config key
                "USE_PURE_MM": bool,
                "INV_LIMIT": int | None,
                "n_actions": int,
                "input_dim": int,
                "pure_mm_offsets": list | None,
            },
        }

    The factory auto-detects which format is present and extracts the
    architecture metadata accordingly.

    What the returned controller provides
    ---------------------------------------
    - ``enable_learning = False``: no gradient computation, no buffer
      updates, learn() is a no-op.
    - ``act(state)`` returns a MarketMaker command tuple using greedy
      argmax (deterministic policy).
    - ``learn(...)`` is a harmless no-op (safe to call from the runner).
    - All throttle gates are OFF by default (acts every micro-step),
      which is appropriate for evaluation.

    Parameters
    ----------
    ckpt_path : str
        Path to the ``.pt`` checkpoint file.
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    log_dir : str
        TensorBoard log directory (usually unused in eval mode, but
        required by the controller constructor).

    Returns
    -------
    PolicyGradientController
        A fully-configured controller in evaluation mode, ready to be
        passed as ``controller=`` to ``simulate_LOB_with_MM`` or to
        the comparison runner.

    Raises
    ------
    FileNotFoundError
        If ``ckpt_path`` does not exist.

    Example
    -------
    >>> ctrl = make_reinforce_eval_controller("checkpoints/reinforce_generic_final.pt")
    >>> action = ctrl.act(mm_state)  # greedy argmax, no gradients
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"REINFORCE checkpoint not found: {ckpt_path}")

    dev = torch.device(device)
    # BUG FIX (BUG 5): Explicit weights_only=False (see load_checkpoint).
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    # ==================================================================
    # STEP 1: Detect checkpoint format and extract configuration
    # ==================================================================
    # We probe for keys that distinguish the two formats:
    #   - "controller_config" → Format A (new)
    #   - "config"            → Format B (legacy)
    #   - Neither             → Fall back to sensible defaults

    if "controller_config" in ckpt:
        # ----- FORMAT A: New checkpoint (from save_checkpoint) -----
        cfg = ckpt["controller_config"]
        weights_key = "policy_net_state_dict"

        pure_mm = bool(cfg.get("pure_mm", False))
        inv_limit = cfg.get("inv_limit", None)
        n_actions = int(cfg.get("n_actions", 6))
        pure_mm_offsets = cfg.get("pure_mm_offsets", None)
        level_offset = int(cfg.get("level_offset", 0))
        n_hidden = int(cfg.get("n_hidden", 2))
        n_neurons = int(cfg.get("n_neurons", 128))

    elif "config" in ckpt:
        # ----- FORMAT B: Legacy checkpoint (older training scripts) -----
        cfg = ckpt["config"]
        weights_key = "policy_state_dict"

        pure_mm = bool(cfg.get("USE_PURE_MM", False))
        inv_limit = cfg.get("INV_LIMIT", None)
        n_actions = int(cfg.get("n_actions", 6))
        pure_mm_offsets = cfg.get("pure_mm_offsets", None)
        level_offset = 0  # Legacy format didn't store this
        n_hidden = int(cfg.get("n_hidden", 2))
        n_neurons = int(cfg.get("n_neurons", 128))

    else:
        # ----- FALLBACK: Bare checkpoint with just weights -----
        # Try the new key first, then the legacy key.
        if "policy_net_state_dict" in ckpt:
            weights_key = "policy_net_state_dict"
        elif "policy_state_dict" in ckpt:
            weights_key = "policy_state_dict"
        else:
            raise KeyError(
                "Checkpoint does not contain 'policy_net_state_dict' or "
                "'policy_state_dict'. Cannot load weights."
            )

        # Use conservative defaults
        pure_mm = False
        inv_limit = None
        n_actions = 6
        pure_mm_offsets = None
        level_offset = 0
        n_hidden = 2
        n_neurons = 128

    # ==================================================================
    # STEP 2: Construct the controller in evaluation mode
    # ==================================================================
    # All throttle gates are OFF (default), so the controller will act
    # on every micro-step. This is the standard behavior for evaluation:
    # we want the policy to respond to every market event.

    ctrl = PolicyGradientController(
        pure_mm=pure_mm,
        inv_limit=inv_limit,
        n_actions=n_actions,
        pure_mm_offsets=pure_mm_offsets,
        level_offset=level_offset,
        n_hidden=n_hidden,
        n_neurons=n_neurons,
        enable_learning=False,   # Eval mode: greedy argmax, no gradients
        device=dev,
        log_dir=log_dir,
    )

    # ==================================================================
    # STEP 3: Load the trained weights into the policy network
    # ==================================================================
    ctrl.policy_net.load_state_dict(ckpt[weights_key])
    ctrl.policy_net.eval()  # Redundant (enable_learning=False does this),
    #                         but explicit for clarity.

    print(f"[REINFORCE] Eval controller loaded from {ckpt_path}")
    print(f"[REINFORCE] pure_mm={pure_mm}, n_actions={n_actions}, "
          f"inv_limit={inv_limit}, enable_learning=False")

    return ctrl
