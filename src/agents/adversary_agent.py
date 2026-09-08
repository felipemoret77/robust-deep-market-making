"""
Adversarial agent for robust market-making training (Glielmo-style).

Overview
--------
This module implements the **adversary player** in a two-player zero-sum game
for training robust market-making agents.  The idea follows the adversarial
reinforcement learning framework of Glielmo et al., where an adversary is
trained *jointly* with the primary agent (the market maker) to expose the
MM to worst-case order flow, thereby improving the MM's robustness to
non-stationary market conditions.

Architecture
~~~~~~~~~~~~
The adversary is a lightweight **DQN (Deep Q-Network)** that acts on a
**semi-MDP** (semi-Markov Decision Process): it only takes actions at
*regime boundaries*, not at every simulation step.  This matches the
intuition that market microstructure conditions (e.g., the proportion of
aggressive buy vs sell flow) change on slower timescales than individual
order events.

At each regime boundary, the adversary observes:
    1. ``mm_inventory / inv_limit``  — normalised MM inventory position
    2. ``prev_p_buy``                — the buy MO probability from the previous regime

It then selects a new ``p_buy`` from a 13-point discrete grid
[0.20, 0.25, ..., 0.80], seeking to *minimise* the MM's shaped reward
(equivalently, maximise the negative of the MM's reward).

Regime durations are drawn from a **Pareto distribution** with shape
parameter ``alpha`` and scale ``L_min``, matching the empirically observed
power-law tails in metaorder durations (Lillo, Mike & Farmer 2005).
The adversary controls only *what* p_buy is (the flow direction), not
*how long* each regime lasts — duration remains an environment constraint.

Training Protocol
~~~~~~~~~~~~~~~~~
An **alternating training** scheme is used to stabilise learning:
    - **MM phase** (N episodes): the MM learns via its own DQN/SARSA updates
      while the adversary's policy is frozen (but still collects experience).
    - **Adversary phase** (N episodes): the adversary trains from its replay
      buffer while the MM's policy is frozen.

Both players always *act* (to collect transitions), but only one *learns*
at a time.  This avoids the non-stationary oscillations that plague
simultaneous training of both players.

Integration with the LOB Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The LOB simulation engine (``LOB_SIM_SANTA_FE.simulate_order``) accepts
``buy_mo_prob`` as either a float or a callable ``step_idx -> float``.
The callable produced by ``make_adversarial_schedule`` plugs directly into
this interface.  However, the engine has no access to the ``MarketMaker``
object, so reward and inventory information is fed back to the schedule
via an ``on_step_reward(reward, mm, is_mo)`` hook injected into
``simulate_LOB_with_MM`` (in ``MM_LOB_SIM.py``).

Usage
-----
1.  Create an ``AdversaryAgent`` once before the training loop.
2.  Each episode, call ``make_adversarial_schedule(adversary, ...)`` to get a
    callable that plugs directly into ``buy_mo_prob`` of
    ``simulate_LOB_with_MM``.
3.  After the episode, call ``schedule.flush_final_regime(done=True)``, push
    transitions into the adversary's replay buffer, and optionally update.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Discrete p_buy grid
# ──────────────────────────────────────────────────────────────────────
# The adversary's action space is a 13-point uniform grid over the
# interval [0.20, 0.80].  This range is symmetric around the fair value
# 0.50 and wide enough to represent meaningful directional bias without
# reaching degenerate extremes (e.g., p_buy = 0 or 1).
#
# Each action index maps to a specific p_buy:
#   idx 0  -> 0.20  (strong sell pressure)
#   idx 6  -> 0.50  (balanced / fair)
#   idx 12 -> 0.80  (strong buy pressure)
#
# The grid spacing of 0.05 provides sufficient granularity for the
# adversary to modulate flow asymmetry while keeping the action space
# small enough for a simple DQN to learn efficiently.
P_BUY_GRID = [round(0.20 + i * 0.05, 2) for i in range(13)]


# ──────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────
class AdversaryDQN(nn.Module):
    """
    Lightweight Q-network for the adversary.

    Architecture: a single-hidden-layer MLP  (3 -> 64 -> 13).

    The network is intentionally kept small for two reasons:
      1. The adversary's observation space is low-dimensional (3 features),
         so a deeper/wider network would overfit with the limited number
         of transitions collected per episode (~6 regime boundaries).
      2. A simpler adversary reduces the risk of the adversary "winning"
         too easily and collapsing the MM's learning signal.  The goal is
         a balanced zero-sum game, not an overwhelmingly strong adversary.

    Parameters
    ----------
    state_dim : int
        Dimension of the observation vector (default 2: normalised inventory,
        previous p_buy).
    n_actions : int
        Size of the discrete action space (default 13: one per P_BUY_GRID value).
    hidden : int
        Width of the single hidden layer (default 64).
    """

    def __init__(self, state_dim: int = 2, n_actions: int = 13, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for all actions given state x."""
        return self.fc2(F.relu(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────
class AdversaryAgent:
    """
    Epsilon-greedy DQN agent for the adversary.

    Parameters
    ----------
    state_dim : int
        Observation dimension (default 2: normalised inventory, previous p_buy).
    n_actions : int
        Number of discrete p_buy choices (default 13).
    hidden : int
        Hidden layer width.
    gamma : float
        Discount factor for the adversary's semi-MDP.
    lr : float
        Learning rate for AdamW.
    epsilon_start, epsilon_min, epsilon_decay : float
        Epsilon-greedy schedule.  Epsilon is multiplied by ``epsilon_decay``
        every episode.
    batch_size : int
        Mini-batch size for replay updates.
    replay_capacity : int
        Maximum transitions in the circular replay buffer.
    target_update_freq : int
        Hard-copy q_net -> target_net every this many episodes.
    inv_limit : float
        Inventory limit used for normalising the MM inventory observation.
    device : str
        Torch device.
    """

    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = 13,
        hidden: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        replay_capacity: int = 10_000,
        target_update_freq: int = 20,
        inv_limit: float = 8.0,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.inv_limit = inv_limit
        self.device = torch.device(device)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = AdversaryDQN(state_dim, n_actions, hidden).to(self.device)
        self.target_net = AdversaryDQN(state_dim, n_actions, hidden).to(self.device)
        self.update_target()  # sync weights

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)

        # Replay buffer (list-based circular, same pattern as ReplayMemory)
        self.replay: deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=replay_capacity
        )

    # ── action selection ──────────────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability ``epsilon``, a uniformly random action is chosen
        (exploration); otherwise, the action with the highest Q-value is
        selected (exploitation).  Epsilon decays multiplicatively each
        episode via ``decay_epsilon()``.

        Parameters
        ----------
        obs : np.ndarray, shape (2,)
            Observation vector: [inventory/inv_limit, prev_p_buy].

        Returns
        -------
        int
            Index into ``P_BUY_GRID`` representing the chosen p_buy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(t)
            return int(q.argmax(dim=1).item())

    @staticmethod
    def get_p_buy(action_idx: int) -> float:
        """Map discrete action index to the corresponding p_buy float value."""
        return P_BUY_GRID[action_idx]

    # ── replay buffer ─────────────────────────────────────────────────
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append((state, action, reward, next_state, done))

    def can_sample(self) -> bool:
        return len(self.replay) >= self.batch_size * 5

    # ── learning ──────────────────────────────────────────────────────
    def update(self) -> Optional[float]:
        """
        Perform one gradient step of standard DQN from a uniformly-sampled
        mini-batch out of the experience replay buffer.

        The Bellman target uses the **target network** (hard-copied every
        ``target_update_freq`` episodes) for stability:

            y = r + gamma * max_a' Q_target(s', a') * (1 - done)

        We use standard (non-double) DQN here because the adversary's
        problem is simple enough (3-dim state, 13 actions) that the
        overestimation bias of vanilla DQN is not a practical concern.

        Returns
        -------
        float or None
            The MSE loss value, or None if the buffer does not yet contain
            enough transitions (``batch_size * 5``) to begin learning.
        """
        if not self.can_sample():
            return None

        batch = random.sample(list(self.replay), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) for the actions actually taken
        q_vals = self.q_net(s).gather(1, a)

        # Bellman target using the frozen target network
        with torch.no_grad():
            q_next = self.target_net(s2).max(dim=1, keepdim=True).values
            target = r + self.gamma * q_next * (1.0 - d)

        loss = F.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        """Hard-copy weights from q_net to target_net (periodic stabilisation)."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        """Multiplicatively decay epsilon, clamped at ``epsilon_min``."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── persistence ───────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """
        Serialise the adversary state to disk.

        The checkpoint includes both networks (q_net and target_net),
        the AdamW optimizer state (for seamless training resumption),
        and the current epsilon value (so exploration schedule continues
        from where it left off).
        """
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Restore adversary state from a previously saved checkpoint.

        Uses ``weights_only=True`` for safety against pickle-based
        code execution attacks in untrusted checkpoint files.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]


# ──────────────────────────────────────────────────────────────────────
# Adversarial schedule callable
# ──────────────────────────────────────────────────────────────────────
def make_adversarial_schedule(
    adversary: AdversaryAgent,
    n_mo_events: int,
    inv_limit: float,
    seed: int,
    L_min: int = 10,
    alpha: float = 1.5,
) -> Callable[[int], float]:
    """
    Build a callable ``step_idx -> buy_mo_prob`` controlled by the adversary.

    This is the central integration point between the adversary agent and the
    LOB simulation engine.  The returned callable is passed as ``buy_mo_prob``
    to ``simulate_LOB_with_MM``, where the engine invokes it at each market
    order event to determine the probability that the incoming MO is a buy.

    Design Pattern — Closure with Mutable State
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The callable is implemented as a closure over a shared mutable ``state``
    dictionary.  This pattern is necessary because:
      - The engine calls ``schedule(step_idx)`` with no access to the MM object.
      - The adversary needs the MM's inventory and accumulated reward to make
        decisions and compute its own reward signal.
      - The ``on_step_reward`` hook (called from ``simulate_LOB_with_MM``)
        writes reward/inventory data into the shared state, which the schedule
        closure reads at regime boundaries.

    Semi-MDP Structure
    ~~~~~~~~~~~~~~~~~~
    The adversary operates on a **semi-MDP**: it only takes decisions at
    regime boundaries, not at every step.  Between boundaries, the same
    p_buy is returned.  Regime durations are drawn from Pareto(alpha, L_min)
    and are **not** controlled by the adversary — they are an environment
    constraint.  Typical values (L_min=10, alpha=1.5) yield ~6 regimes per
    episode with ~192 MO events per 5000-event simulation.

    Adversary Reward
    ~~~~~~~~~~~~~~~~
    The adversary's reward for each regime is the **negative of the MM's
    mean per-MO-event reward** accumulated during that regime:

        r_adv = -sum(r_MM) / n_mo_events_in_regime

    The normalisation by regime length prevents longer regimes from having
    disproportionate influence on the adversary's Q-values — critical for
    stable learning in a semi-MDP where regime durations are heavy-tailed.

    Parameters
    ----------
    adversary : AdversaryAgent
        The adversary DQN agent that will select p_buy at regime boundaries.
    n_mo_events : int
        Expected number of MO events per episode.  Used to generate
        enough Pareto regime boundaries to cover the full episode.
    inv_limit : float
        The MM's inventory limit, used to normalise the inventory observation
        into [-1, +1] for the adversary's neural network input.
    seed : int
        RNG seed for the Pareto duration draws (ensures reproducibility
        of the regime structure for a given episode).
    L_min : int
        Pareto scale parameter — minimum regime duration in MO events.
    alpha : float
        Pareto shape parameter.  alpha=1.5 produces heavy-tailed durations
        consistent with empirical metaorder length distributions.

    Returns
    -------
    schedule : Callable[[int], float]
        Maps MO event index to buy_mo_prob.  Also exposes:
          - ``.on_step_reward(reward, mm, is_mo)`` — reward/inventory hook
          - ``.flush_final_regime(done)``          — close the last regime
          - ``.reset()``                           — clear warmup state
          - ``.transitions``                       — collected (s, a, r, s', done)
          - ``.boundaries``                        — Pareto regime boundary list
          - ``.state``                             — internal mutable state dict
    """

    rng = np.random.default_rng(seed)

    # ── Pre-generate Pareto regime boundaries ─────────────────────────
    # We generate boundaries beyond n_mo_events (with a buffer of +1000)
    # to guarantee full coverage even if the actual MO count exceeds
    # the expected value.  Pareto draws use the inverse-CDF method:
    #     L = ceil(L_min / U^(1/alpha)),  U ~ Uniform(0, 1)
    # which produces E[L] = alpha * L_min / (alpha - 1) for alpha > 1.
    # With alpha=1.5 and L_min=10, E[L] = 30 MO events per regime.
    boundaries: List[int] = [0]
    while boundaries[-1] < n_mo_events + 1000:
        L = int(np.ceil(L_min / rng.uniform() ** (1.0 / alpha)))
        boundaries.append(boundaries[-1] + L)
    boundaries_arr = np.array(boundaries)

    # ── Mutable state shared between schedule / on_step_reward / flush ─
    # This dictionary is the communication channel between the three
    # closures (schedule, on_step_reward, flush_final_regime).  It tracks:
    #   - Which regime we are currently in (current_regime_idx)
    #   - The active p_buy and the action index that produced it
    #   - Accumulated MM reward and MO event count within the current regime
    #   - The MM's last-known inventory (updated by on_step_reward)
    #   - The observation and action at regime start (for building transitions)
    #   - The list of completed (s, a, r, s', done) transitions
    state = {
        "current_regime_idx": -1,
        "current_p_buy": 0.5,
        "current_action": 6,          # index for 0.50 in P_BUY_GRID
        "prev_p_buy": 0.5,
        "regime_reward_accum": 0.0,
        "regime_step_count": 0,
        "mm_inventory": 0.0,
        "prev_obs": None,             # observation at start of current regime
        "prev_action": None,          # action taken at start of current regime
        "transitions": [],            # collected (s, a, r, s', done)
    }

    # ── The callable invoked by the LOB engine at each MO event ───────
    def schedule(step_idx: int) -> float:
        """
        Called by the LOB engine (via ``_buy_mo_prob_fn``) at each MO event.

        Uses ``np.searchsorted`` on the pre-computed boundary array to
        determine which regime ``step_idx`` falls into.  If a new regime
        is entered, the adversary is queried for a fresh action.

        The transition for the *previous* regime is only emitted here
        (not at the end of each step), because the adversary's reward
        must be fully accumulated over the regime's duration before being
        committed.
        """
        # O(log n) lookup into the sorted boundary array
        regime_idx = int(np.searchsorted(boundaries_arr, step_idx, side="right")) - 1
        regime_idx = min(regime_idx, len(boundaries_arr) - 2)

        if regime_idx > state["current_regime_idx"]:
            # ── Regime boundary crossed ───────────────────────────────

            # 1. Close out previous regime: emit a (s, a, r, s', done=False)
            #    transition with the normalised adversary reward.
            if state["prev_obs"] is not None:
                steps = max(1, state["regime_step_count"])
                r_adv = -state["regime_reward_accum"] / steps
                next_obs = np.array(
                    [
                        state["mm_inventory"] / inv_limit,
                        state["current_p_buy"],
                    ],
                    dtype=np.float32,
                )
                state["transitions"].append(
                    (state["prev_obs"], state["prev_action"], r_adv, next_obs, False)
                )

            # 2. Build observation for the new regime
            obs = np.array(
                [
                    state["mm_inventory"] / inv_limit,
                    state["current_p_buy"],  # previous regime's p_buy as context
                ],
                dtype=np.float32,
            )

            # 3. Adversary selects a new p_buy action
            action_idx = adversary.select_action(obs)
            new_p_buy = adversary.get_p_buy(action_idx)

            # 4. Update mutable state for the new regime
            state["prev_obs"] = obs
            state["prev_action"] = action_idx
            state["prev_p_buy"] = state["current_p_buy"]
            state["current_p_buy"] = new_p_buy
            state["current_action"] = action_idx
            state["current_regime_idx"] = regime_idx
            state["regime_reward_accum"] = 0.0
            state["regime_step_count"] = 0

        return state["current_p_buy"]

    # ── Hook called by simulate_LOB_with_MM after each reward ─────────
    def on_step_reward(reward: float, mm, is_mo: bool = False) -> None:
        """
        Reward and inventory feedback hook.

        Called from ``simulate_LOB_with_MM`` (in ``MM_LOB_SIM.py``) after
        every event that produces a reward — including limit order fills,
        cancellations, and market orders.  This solves the architectural
        constraint that the LOB engine's inner loop has no direct access
        to the MarketMaker object.

        The reward is accumulated over the entire regime.  However,
        ``regime_step_count`` is only incremented for MO events (when
        ``is_mo=True``), because regime boundaries are defined in
        MO-event space, not total-event space.  This ensures that the
        per-MO-event normalisation (``r_adv / regime_step_count``) is
        consistent with the regime duration semantics.

        Parameters
        ----------
        reward : float
            The MM's shaped reward for this simulation step.
        mm : MarketMaker
            Reference to the market maker object (used to snapshot inventory).
        is_mo : bool
            True if this event was a market order, False otherwise.
        """
        state["regime_reward_accum"] += reward
        if is_mo:
            state["regime_step_count"] += 1
        state["mm_inventory"] = float(mm.inventory)

    # ── Close the final regime after the episode ends ─────────────────
    def flush_final_regime(done: bool = True) -> None:
        """
        Emit the terminal transition for the last regime.

        Must be called after the simulation episode ends, because the
        schedule callable only emits transitions at regime *boundaries*,
        and the final regime has no subsequent boundary to trigger it.
        Sets ``done = True`` to signal episode termination to the
        adversary's Bellman target.
        """
        if state["prev_obs"] is not None:
            steps = max(1, state["regime_step_count"])
            r_adv = -state["regime_reward_accum"] / steps
            next_obs = np.array(
                [
                    state["mm_inventory"] / inv_limit,
                    state["current_p_buy"],
                ],
                dtype=np.float32,
            )
            state["transitions"].append(
                (state["prev_obs"], state["prev_action"], r_adv, next_obs, done)
            )

    # ── Reset schedule state (e.g. after engine warmup) ─────────────
    def reset() -> None:
        """
        Discard any state accumulated during engine warmup.

        During ``iterations_to_equilibrium``, the LOB engine generates MO
        events that advance ``_event_step`` and trigger adversary decisions
        without meaningful reward feedback (the MM is not yet active).
        This reset is called after warmup to ensure the adversary starts
        the actual episode with a clean slate — no stale observations,
        no phantom transitions, and regime indexing restarted from -1.
        """
        state["current_regime_idx"] = -1
        state["current_p_buy"] = 0.5
        state["current_action"] = 6
        state["prev_p_buy"] = 0.5
        state["regime_reward_accum"] = 0.0
        state["regime_step_count"] = 0
        state["mm_inventory"] = 0.0
        state["prev_obs"] = None
        state["prev_action"] = None
        state["transitions"].clear()

    # ── Attach closures and metadata to the schedule callable ────────
    # We use function attribute assignment (with type: ignore) to make
    # the returned callable a "rich callable" — it behaves like a plain
    # function for the engine, but carries extra methods and metadata
    # for the training loop.  This avoids creating a full class just to
    # satisfy the engine's simple Callable[[int], float] interface.
    schedule.on_step_reward = on_step_reward          # type: ignore[attr-defined]
    schedule.flush_final_regime = flush_final_regime   # type: ignore[attr-defined]
    schedule.reset = reset                             # type: ignore[attr-defined]
    schedule.transitions = state["transitions"]        # type: ignore[attr-defined]
    schedule.boundaries = boundaries                   # type: ignore[attr-defined]
    schedule.state = state                             # type: ignore[attr-defined]

    return schedule
