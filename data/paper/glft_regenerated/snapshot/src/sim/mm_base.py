#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared Base Controller for LOB Market-Making RL Agents
=======================================================

Provides the common infrastructure used by both PPO and SAC-Discrete
controllers:

    1. ``ActorNetwork``          — MLP policy network π_θ(a|s) → logits
    2. ``ObservationNormalizer``  — Per-feature RunningMeanStd (Welford)
    3. ``BaseMMController``      — Shared agent: act, state encoding,
                                   throttle gates, SMDP, action masking

Algorithm-specific components (critic networks, replay buffers, training
loops) live in their respective modules (ppo.py, sac.py).
"""

# ============================================================
# Imports
# ============================================================

from typing import Dict, Any, Optional, List, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from RLController import RLController


# ============================================================
# Section 1: Actor Network — Policy Function Approximator
# ============================================================
#
# MLP policy: S → ℝ^|A|  (unnormalised logits → softmax → π(a|s))
#
# ReLU is chosen over tanh/sigmoid because:
#   (1) it avoids the vanishing-gradient problem in deeper networks,
#   (2) it produces sparse activations (implicit regularisation).
# ============================================================


class ActorNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) policy network.

    Maps a state vector s ∈ ℝ^D to unnormalised action logits z ∈ ℝ^|A|.
    The policy distribution is obtained via the softmax transformation:

        π_θ(a | s) = exp(z_a) / Σ_{a'} exp(z_{a'})

    where z = f_θ(s) is the forward pass through this network.

    Architecture
    ------------
        Input(D) → [Linear(D, H) → ReLU]^L → Linear(H, |A|)

    where D = input_dim, H = n_neurons, L = n_hidden, |A| = n_actions.
    Total parameters: D·H + H + (L-1)·(H² + H) + H·|A| + |A|.

    Note: The output layer has NO activation — raw logits are returned.
    Action masking (setting invalid logits to -∞) is applied externally
    before the softmax, ensuring zero probability for forbidden actions.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
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
        # Output layer: logits z ∈ ℝ^|A| (no activation)
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: s → z = f_θ(s), unnormalised logits."""
        return self.net(x)


# ============================================================
# Section 2: Observation Normalizer — RunningMeanStd (Welford)
# ============================================================
#
# Per-feature running mean/variance normalizer for state observations.
# Uses the parallel Welford algorithm to maintain running statistics
# across all episodes. Normalises each feature to approximately
# zero-mean, unit-variance:
#
#     s_norm[d] = (s[d] - μ[d]) / sqrt(σ²[d] + ε)
#
# This is the same RunningMeanStd approach used in standard PPO
# implementations (CleanRL, SB3, OpenAI baselines).
# ============================================================


class ObservationNormalizer:
    """
    Per-feature running mean/variance normalizer for state observations.

    Uses the parallel Welford algorithm to maintain running statistics
    across all episodes. Normalises each feature to approximately
    zero-mean, unit-variance:

        s_norm[d] = (s[d] - μ[d]) / sqrt(σ²[d] + ε)

    Parameters
    ----------
    shape : int or tuple
        Dimensionality of the observation vector.
    epsilon : float
        Small constant added to variance for numerical stability.
    """

    def __init__(self, shape: int, epsilon: float = 1e-8):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.epsilon = epsilon
        # Running statistics (float64 for numerical precision)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count: float = 1e-4  # Small initial count for stability

    def update(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a batch of observations.

        Parameters
        ----------
        x : torch.Tensor, shape (B, D) or (D,)
            Batch of raw (un-normalised) state vectors.
        """
        batch = x.detach().cpu().numpy().astype(np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Parallel algorithm for merging batch statistics into running stats."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations using current running statistics.

        Parameters
        ----------
        x : torch.Tensor, shape (..., D)
            Raw state tensor(s).

        Returns
        -------
        torch.Tensor
            Normalised state tensor(s), same shape as input.
        """
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std_t = torch.sqrt(
            torch.tensor(self.var, dtype=torch.float32, device=x.device)
            + self.epsilon
        )
        return (x - mean_t) / std_t

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ============================================================
# Section 3: Base MM Controller — Shared Agent Infrastructure
# ============================================================
#
# The BaseMMController provides the common infrastructure shared by
# PPO and SAC-Discrete controllers:
#
#   (1) State representation (raw LOB state → normalised tensor)
#   (2) Action selection (logits → masked softmax → categorical sample)
#   (3) Action mapping (action index → MM command tuple)
#   (4) Throttle gating (SMDP: reduce decision frequency)
#   (5) SMDP state management (cumulative reward aggregation)
#
# Algorithm-specific components (critic, replay buffer, training
# logic) are added by subclasses (PPOController, SACDiscreteController).
#
# Operating Modes
# ================
#
# PURE MM MODE (pure_mm=True):
#   - Actions are (bid_offset, ask_offset) pairs relative to the BBO.
#   - State: [log(1+spread), inv/inv_limit, LOB sizes at each offset level]
#   - Inventory-limit masking uses "canonical equivalence" to avoid
#     redundant actions.
#
# GENERIC MODE (pure_mm=False):
#   - Actions are named commands: post_bid, post_ask, post_bid_ask,
#     cancel_bid, cancel_ask, hold, (+ inside-spread variants).
#   - State: [log(1+spread), log(1+asksize), log(1+bidsize),
#             inv/inv_limit, has_bid, has_ask]
#
# Throttle Gating (SMDP)
# =======================
#
# To reduce decision frequency and create an SMDP structure, the
# controller supports three throttle gates (can be combined):
#
#   (a) TOB-update gate: act after N_tob changes to the top-of-book.
#   (b) Event-update gate: act after N_events simulation events.
#   (c) Time-update gate: act after Δt ≥ min_time_interval.
#
# Between decisions, the controller returns ("hold",) and accumulates
# the SMDP cumulative reward:
#   R^{cum} = Σ_{k=0}^{K-1} γ^k · r_k
#
# If use_mdp=True, fill/mode-change bypasses are disabled.
# ============================================================


class BaseMMController(RLController):
    """
    Shared base controller for market-making RL agents.

    Implements the RLController protocol with:
      - Categorical policy π_θ(a|s) via softmax over masked logits
      - SMDP throttle gating with cumulative reward aggregation
      - Inventory-limit action masking (canonical equivalence in pure MM)

    Subclasses must implement learn() and optionally override
    enable_learning setter to toggle additional networks.

    Parameters
    ----------
    level_offset : int
        Price offset for generic mode quotes.
    n_actions : int
        Number of discrete actions |A| (used in generic mode only).
    gamma : float
        SMDP discount factor γ ∈ (0, 1).
    lr_actor : float
        Learning rate for the actor AdamW optimiser.
    weight_decay : float
        L2 regularisation coefficient for AdamW.
    grad_clip_norm : float
        Maximum global gradient norm.
    device : torch.device
        Computation device.
    pure_mm : bool
        If True, use PURE MM mode with offset-pair action space.
    inv_limit : int or None
        Maximum absolute inventory.
    pure_mm_offsets : list of (int, int)
        Offset pairs defining the action space in PURE MM mode.
    n_hidden_actor, n_neurons_actor : int
        Actor MLP architecture.
    enable_learning : bool
        If False, act() uses argmax (greedy policy).
    use_tob_update, n_tob_moves : bool, int
        TOB-update throttle gate configuration.
    use_event_update, n_events : bool, int
        Event-update throttle gate configuration.
    use_time_update, min_time_interval : bool, float
        Time-update throttle gate configuration.
    use_mdp : bool
        If True, disable fill/mode-change bypasses.
    use_obs_normalizer : bool
        If True (default), apply Welford running mean/std normalisation
        to state features. If False, return raw log1p-scaled features.
    """

    def __init__(
        self,
        level_offset: int = 0,
        n_actions: int = 6,
        gamma: float = 0.97,
        lr_actor: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        # ----- MM PURE mode -----
        pure_mm: bool = False,
        inv_limit: Optional[int] = None,
        pure_mm_offsets: Optional[List[Tuple[int, int]]] = None,
        # ----- Actor architecture -----
        n_hidden_actor: int = 2,
        n_neurons_actor: int = 128,
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
        # =============================================================
        # CORE HYPERPARAMETERS
        # =============================================================
        self.level_offset = int(level_offset)
        self.gamma = float(gamma)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.actor_n_hidden = int(n_hidden_actor)
        self.actor_n_neurons = int(n_neurons_actor)

        # =============================================================
        # PURE-MM CONFIGURATION
        # =============================================================
        self.pure_mm = bool(pure_mm)
        self.inv_limit = None if inv_limit is None else int(inv_limit)

        if self.pure_mm:
            if pure_mm_offsets is None:
                self.pure_mm_offsets: List[Tuple[int, int]] = [
                    (0, 0), (0, 1), (1, 0), (1, 1),
                ]
            else:
                self.pure_mm_offsets = [
                    (int(x[0]), int(x[1])) for x in list(pure_mm_offsets)
                ]

            if len(self.pure_mm_offsets) == 0:
                raise ValueError("pure_mm_offsets cannot be empty.")

            self.n_actions = len(self.pure_mm_offsets)

            bid_offs = [bo for (bo, _) in self.pure_mm_offsets]
            ask_offs = [ao for (_, ao) in self.pure_mm_offsets]
            max_bid_offset = max([bo for bo in bid_offs if bo >= 0], default=0)
            max_ask_offset = max([ao for ao in ask_offs if ao >= 0], default=0)
            self.max_offset = int(max(max_bid_offset, max_ask_offset))

            # Canonical equivalence masks for inventory-limit states
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
            self.pure_mm_offsets = None
            self.max_offset = 0
            self.n_actions = int(n_actions)
            if self.n_actions < 6:
                self.n_actions = 6
            elif self.n_actions > 9:
                self.n_actions = 9

        # =============================================================
        # DEVICE SELECTION
        # =============================================================
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # =============================================================
        # NETWORKS — Actor π_θ
        # =============================================================
        if self.pure_mm:
            input_dim = 2 + 2 * (self.max_offset + 1)
        else:
            input_dim = 6

        self.input_dim = input_dim

        self.actor_net = ActorNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            n_hidden=self.actor_n_hidden,
            n_neurons=self.actor_n_neurons,
        ).to(self.device)

        # =============================================================
        # OBSERVATION NORMALIZER — Per-feature RunningMeanStd
        # =============================================================
        self.use_obs_normalizer = bool(use_obs_normalizer)
        self.obs_normalizer = ObservationNormalizer(shape=input_dim)

        # =============================================================
        # ENABLE LEARNING TOGGLE
        # =============================================================
        self._enable_learning = bool(enable_learning)
        if self._enable_learning:
            self.actor_net.train()
        else:
            self.actor_net.eval()

        # =============================================================
        # ACTOR OPTIMIZER
        # =============================================================
        self.actor_optimizer = AdamW(
            self.actor_net.parameters(), lr=lr_actor, weight_decay=self.weight_decay,
        )

        # =============================================================
        # ACTION TRACKING
        # =============================================================
        self._last_valid_mask: Optional[np.ndarray] = None
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
        self._has_acted_once: bool = False
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
        # Normalized state tensor cached at act()-time for consistency
        # with old_log_prob computation (avoids re-encoding with stale
        # obs_normalizer stats).
        self._last_act_state_tensor: Optional[torch.Tensor] = None

    # ================================================================
    # STATE CONVERSION — LOB State Dict → Neural Network Input
    # ================================================================

    def _state_to_tensor(self, s: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a MarketMaker state dictionary to a normalised (1, D) tensor.

        Two-stage normalisation pipeline:

        Stage 1 — Feature engineering (deterministic transforms):
            Spread, sizes:  x → log(1 + max(0, x))
            Inventory:      x → x / inv_limit

        Stage 2 — RunningMeanStd observation normalisation:
            s_norm = (s_raw - μ_running) / sqrt(σ²_running + ε)

            The obs_normalizer tracks per-feature running statistics
            (updated during data collection) and normalises each
            dimension to approximately zero-mean, unit-variance.

        PURE MM STATE VECTOR (before obs norm):
            s = [log1p(spread), inv/inv_limit,
                 log1p(bid_size_0), ..., log1p(bid_size_K),
                 log1p(ask_size_0), ..., log1p(ask_size_K)]

        GENERIC STATE VECTOR (before obs norm):
            s = [log1p(spread), log1p(asksize), log1p(bidsize),
                 inv/inv_limit, has_bid ∈ {0,1}, has_ask ∈ {0,1}]
        """
        inv_denom = float(self.inv_limit) if self.inv_limit is not None else 10.0
        inv_denom = max(inv_denom, 1.0)

        if not self.pure_mm:
            raw_spread = float(s["spread"])
            spread = math.log1p(max(0.0, raw_spread))
            asksize = math.log1p(max(0.0, float(s.get("asksize", 0.0))))
            bidsize = math.log1p(max(0.0, float(s.get("bidsize", 0.0))))
            inventory = float(s["inventory"]) / inv_denom
            has_bid = 1.0 if bool(s.get("has_bid", False)) else 0.0
            has_ask = 1.0 if bool(s.get("has_ask", False)) else 0.0

            arr = np.array(
                [spread, asksize, bidsize, inventory, has_bid, has_ask],
                dtype=np.float32,
            )
            raw_t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        else:
            # PURE MM MODE
            raw_spread = float(s["spread"])
            spread = math.log1p(max(0.0, raw_spread))
            inventory = float(s["inventory"]) / inv_denom
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

            bid_sizes = [math.log1p(max(0.0, x)) for x in bid_sizes[:K + 1]]
            ask_sizes = [math.log1p(max(0.0, x)) for x in ask_sizes[:K + 1]]

            arr = np.array(
                [spread, inventory] + bid_sizes + ask_sizes,
                dtype=np.float32,
            )
            raw_t = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        # Stage 2: RunningMeanStd observation normalisation (optional)
        if self.use_obs_normalizer:
            if self._enable_learning:
                self.obs_normalizer.update(raw_t)
            return self.obs_normalizer.normalize(raw_t)

        return raw_t

    # ================================================================
    # THROTTLE HELPERS
    # ================================================================

    def _get_env_tob(self, s: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Build Top-Of-Book fingerprint from state dict."""
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
        """Reset throttle gate counters after making a decision."""
        self._has_acted_once = True
        if self.use_tob_update:
            self.moves_tob = 0
            self.last_env_tob_key = tob_key
        if self.use_event_update:
            self.event_steps = 0
        if self.use_time_update:
            self.last_update_time = current_time

    def _smdp_reset(self) -> None:
        """Clear all SMDP aggregation state."""
        self._smdp_pending = False
        self._smdp_s_decision = None
        self._smdp_a_decision = None
        self._smdp_mask_decision = None
        self._smdp_cum_reward = 0.0
        self._smdp_k_steps = 0
        self._last_was_decision = False
        self._last_act_state_tensor = None
        self._smdp_state_tensor_decision = None

    def _reset_throttle_state(self) -> None:
        """Reset throttle trackers to start-of-episode values."""
        self._has_acted_once = False
        self.event_steps = 0
        self.moves_tob = 0
        self.last_env_tob_key = None
        self.last_update_time = -1.0
        self.last_inventory = None
        self.last_mode = None

    # ================================================================
    # ACTION MASK — Inventory-Limit Safety Constraint
    # ================================================================

    def _get_valid_action_mask(self, mm_state: dict) -> np.ndarray:
        """
        Compute a boolean action mask enforcing inventory limits.

        The mask is a vector m ∈ {True, False}^{|A|} where m[a] = True
        iff action a is allowed given the current inventory position.
        """
        mask = np.ones(self.n_actions, dtype=bool)
        if self.inv_limit is None:
            return mask

        inv = float(mm_state.get("inventory", 0.0))

        if self.pure_mm:
            if inv >= self.inv_limit:
                mask = self._canonical_mask_long.copy()
            elif inv <= -self.inv_limit:
                mask = self._canonical_mask_short.copy()
            return mask

        # Generic mode
        if inv >= self.inv_limit:
            mask[0] = False   # post_bid   (would increase long exposure)
            mask[2] = False   # post_bid_ask
            # cancel_bid (3) stays True — allow canceling stale bids
            # cancel_ask (4) stays True — allow managing asks
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside
            if self.n_actions >= 8:
                mask[7] = False   # bid_inside
        elif inv <= -self.inv_limit:
            mask[1] = False   # post_ask   (would increase short exposure)
            mask[2] = False   # post_bid_ask
            # cancel_bid (3) stays True — allow managing bids
            # cancel_ask (4) stays True — allow canceling stale asks
            if self.n_actions >= 7:
                mask[6] = False   # bid_ask_inside
            if self.n_actions >= 9:
                mask[8] = False   # ask_inside

        return mask

    # ================================================================
    # ACTION SAMPLING — Categorical Policy π_θ(a|s)
    # ================================================================

    def _sample_action_idx(
        self,
        state_tensor: torch.Tensor,
        valid_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Sample an action index from the policy distribution.

        Two modes:
          1. eval (enable_learning=False): a = argmax(z)  — pure greedy.
          2. Categorical (default): a ~ Categorical(softmax(z)).
        """
        with torch.no_grad():
            logits = self.actor_net(state_tensor)

            if valid_mask is not None:
                mask_tensor = torch.tensor(
                    valid_mask, dtype=torch.bool, device=logits.device
                )
                logits = logits.masked_fill(~mask_tensor, float("-inf"))

            if not self._enable_learning:
                # Eval mode — pure greedy
                a_idx = int(torch.argmax(logits, dim=-1).item())
            else:
                # Categorical sampling (standard PPO/SAC)
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
    # ACTION MAPPING — PURE MM Mode
    # ================================================================

    def _mm_action_from_idx_pure_mm(
        self, a_idx: int, state: Dict[str, Any]
    ) -> tuple:
        """
        Convert a pure-MM action index to a MarketMaker command tuple.

        The action index selects an (bid_offset, ask_offset) pair from
        self.pure_mm_offsets. The offsets are applied to the current BBO
        to compute absolute bid and ask prices.
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
    # ACTION MAPPING — GENERIC Mode
    # ================================================================

    def _mm_action_from_idx_generic(
        self, a_idx: int, state: Dict[str, Any]
    ) -> tuple:
        """
        Convert a generic-mode action index to a MarketMaker command tuple.
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

        if self.inv_limit is not None and (want_bid or want_ask):
            if inv >= self.inv_limit:
                want_bid = False
                use_inside_bid = False
            elif inv <= -self.inv_limit:
                want_ask = False
                use_inside_ask = False

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

        Implements the SMDP throttle gate logic:
        1. Update throttle counters.
        2. Check bypass priorities (mode change, fill replenishment).
        3. If throttled → return ("hold",).
        4. If decision:
           a. s_t = _state_to_tensor(mm_state)
           b. m_t = _get_valid_action_mask(mm_state)
           c. a_t ~ π_θ(·|s_t, m_t)
           d. cmd = _mm_action_from_idx(a_t)
           e. Reset clocks, mark _last_was_decision = True
        """
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

        # Bypass Priority A — Mode Change
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

        # Bypass Priority B — Fill Replenishment
        has_fill = False
        if self.last_inventory is not None:
            if int(inv) != int(self.last_inventory):
                has_fill = True
        self.last_inventory = int(inv)

        # Decision logic
        should_act = False

        if mode_changed and not self.use_mdp:
            should_act = True
        elif has_fill and not self.use_mdp:
            should_act = True
        else:
            is_throttled = False

            if not self._has_acted_once:
                pass
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

        if should_act:
            state_tensor = self._state_to_tensor(mm_state)
            valid_mask = self._get_valid_action_mask(mm_state)
            a_idx = self._sample_action_idx(state_tensor, valid_mask=valid_mask)
            act_tuple = self._mm_action_from_idx(a_idx, mm_state)

            self.last_action_idx = a_idx
            self._last_valid_mask = valid_mask
            # Cache the normalized tensor for learn() — ensures old_log_prob
            # is computed on the exact same state the actor saw.
            self._last_act_state_tensor = state_tensor.squeeze(0).detach().cpu()
            self._reset_clocks(current_time, tob_key)
            self._last_was_decision = True

            return act_tuple
        else:
            self._last_was_decision = False
            return ("hold",)

    # ================================================================
    # RLController INTERFACE: learn() — to be implemented by subclass
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
        """Subclasses must implement SMDP-aware learning logic."""
        raise NotImplementedError("Subclasses must implement learn()")

    # ================================================================
    # ENABLE LEARNING TOGGLE
    # ================================================================

    @property
    def enable_learning(self) -> bool:
        """Whether gradient updates and stochastic action sampling are enabled."""
        return self._enable_learning

    @enable_learning.setter
    def enable_learning(self, value: bool) -> None:
        """
        Toggle between training and evaluation mode.

        Training (True): actor in train() mode, stochastic sampling.
        Evaluation (False): actor in eval() mode, argmax (greedy).

        Subclasses should override to toggle additional networks.
        """
        self._enable_learning = bool(value)
        if self._enable_learning:
            self.actor_net.train()
        else:
            self.actor_net.eval()
