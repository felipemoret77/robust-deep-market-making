#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adversary_tau_agent.py — Adversarial Tau Bandit for Regime Persistence
======================================================================

This module implements a multi-armed bandit agent that adversarially
controls the **regime persistence parameter** (tau) of an exponential
regime-switching environment.  The bandit acts at regime boundaries:
after each completed regime, it selects the tau used to draw the next
regime duration.

Motivation
----------
A market maker trained under a fixed regime persistence (e.g., tau=20)
may be fragile when the true persistence differs.  By letting an
adversary choose tau from a discrete grid, we can:

  1. **Diagnose fragility**: identify which persistence scales hurt the
     MM most (the adversary's preferred arm).
  2. **Train robustness**: the MM sees a distribution of taus weighted
     toward the hardest ones, forcing it to generalise.

Design Choices
--------------
- **Bandit, not DQN.**  With one discrete tau choice per regime and only
  a few arms, a neural network is overkill.  Epsilon-greedy with EWMA
  scoring per arm is simpler, faster to learn, and easier to interpret.

- **EWMA score, not running mean.**  The MM's policy changes during
  training (non-stationary rewards).  A global running mean would weigh
  episode 1 equally with episode 2000, masking the adversary's ability
  to adapt.  An EWMA with recency factor beta gives higher weight to
  recent outcomes, allowing the adversary to track the MM's evolving
  weaknesses.

- **Burn-in phase.**  Each arm is pulled at least ``burn_in`` times
  before the agent starts exploiting.  This prevents premature
  convergence to a suboptimal arm due to noisy early estimates.

- **High epsilon floor (0.10).**  Even after convergence, the adversary
  explores 10% of the time.  This prevents collapse to a single tau
  (which would make training effectively stationary) and allows the
  adversary to detect if the MM has improved against its preferred arm.

References
----------
- Guéant, Lehalle & Fernandez-Tapia (2013): optimal market making
- Sutton & Barto (2018), Ch. 2: multi-armed bandits

Author: Felipe Moret (ETH Zürich)
"""

import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

TAU_GRID: List[int] = [8, 12, 20, 30, 45]
"""
Discrete set of candidate mean regime durations (in MO events) for the
exponential regime-switching distribution.

- tau=8:  fast switching (~8 MO mean), tests MM reaction speed
- tau=12: moderately fast
- tau=20: the default training value
- tau=30: slow switching, longer directional pressure
- tau=45: very persistent regimes, strong inventory stress
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Adversary Tau Bandit Agent
# ═══════════════════════════════════════════════════════════════════════════════

class AdversaryTauAgent:
    """
    Multi-armed bandit that selects the regime persistence parameter tau
    from a discrete grid.

    The agent maintains an EWMA score for each arm (tau value).  Regime-level
    schedules update the selected arm after each completed regime using the
    MM's PnL rate over that regime.

    Parameters
    ----------
    tau_grid : list of int, optional
        Discrete tau values to choose from.  Defaults to ``TAU_GRID``.
    beta : float
        EWMA recency factor for score updates.  Higher values give more
        weight to recent episodes.  With beta=0.1, the effective memory
        is ~10 pulls per arm (half-life ≈ 7 pulls).
    epsilon_start : float
        Initial exploration rate for epsilon-greedy action selection.
    epsilon_min : float
        Floor for epsilon after decay.  Set to 0.10 to ensure the
        adversary never stops exploring entirely (prevents collapse
        to a single tau and allows detection of MM adaptation).
    burn_in : int
        Minimum number of times each arm must be pulled before the
        agent starts exploiting (epsilon-greedy kicks in only after
        every arm has been pulled ``burn_in`` times).
    """

    def __init__(
        self,
        tau_grid: Optional[List[int]] = None,
        beta: float = 0.1,
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.10,
        burn_in: int = 5,
    ) -> None:
        self.tau_grid: List[int] = list(tau_grid or TAU_GRID)
        self.n_arms: int = len(self.tau_grid)
        self.beta: float = beta
        self.epsilon_start: float = epsilon_start
        self.epsilon: float = epsilon_start
        self.epsilon_min: float = epsilon_min
        self.burn_in: int = burn_in

        # Per-arm state: EWMA score and pull count.
        # scores[i] tracks the EWMA of adv_reward = -mm_objective for arm i.
        # Higher score means this tau hurts the MM more.
        self.scores: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.int64)
        self.total_pulls: int = 0

        # Last action tracking (set by select_action, read by get_tau/update).
        self._last_arm: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────
    #  Action Selection
    # ──────────────────────────────────────────────────────────────────

    def select_action(self) -> int:
        """
        Select the next arm (tau index) to pull.

        Selection logic:
          1. **Burn-in**: if any arm has been pulled fewer than
             ``self.burn_in`` times, pick the arm with the lowest count
             (ties broken randomly).  This ensures every tau is explored
             before exploitation begins.
          2. **Epsilon-greedy**: with probability epsilon, pick a random
             arm.  Otherwise, pick the arm with the highest EWMA score
             (ties broken randomly to avoid systematic bias toward
             lower-index arms).

        Returns
        -------
        int
            Index into ``self.tau_grid`` (0 to n_arms-1).
        """
        # Phase 1: Burn-in — ensure each arm has at least `burn_in` pulls.
        min_count = int(self.counts.min())
        if min_count < self.burn_in:
            # Find all arms tied at the minimum count and pick randomly.
            candidates = [i for i in range(self.n_arms) if self.counts[i] == min_count]
            arm = random.choice(candidates)
        elif random.random() < self.epsilon:
            # Phase 2a: Explore — uniform random.
            arm = random.randrange(self.n_arms)
        else:
            # Phase 2b: Exploit — pick arm with highest EWMA score.
            # Random tie-breaking among arms with the max score.
            max_score = float(self.scores.max())
            candidates = [i for i in range(self.n_arms)
                          if abs(self.scores[i] - max_score) < 1e-12]
            arm = random.choice(candidates)

        self._last_arm = arm
        return arm

    # ──────────────────────────────────────────────────────────────────
    #  Accessors
    # ──────────────────────────────────────────────────────────────────

    def get_tau(self) -> int:
        """
        Return the tau value corresponding to the last selected arm.

        Must be called after ``select_action()``.

        Returns
        -------
        int
            The chosen tau (mean regime duration in MO events).
        """
        if self._last_arm is None:
            raise RuntimeError("get_tau() called before select_action()")
        return self.tau_grid[self._last_arm]

    # ──────────────────────────────────────────────────────────────────
    #  Learning
    # ──────────────────────────────────────────────────────────────────

    def update_arm(self, arm: int, mm_objective_value: float) -> None:
        """
        Update one arm after observing the MM's objective.

        The adversary's reward is the *negative* of the MM's objective:
        the adversary wants to find the tau that minimises the MM's
        performance.

        Parameters
        ----------
        mm_objective_value : float
            The MM objective to minimise.  For regime-level tau control this
            should be the regime PnL rate, ``delta_pnl / regime_length_mo``.
            Positive values mean the MM profited; negative values mean the MM
            lost money.  The adversary receives ``-mm_objective`` as reward.
        """
        arm = int(arm)
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm index out of range: {arm}")
        adv_reward = -mm_objective_value

        # Incremental EWMA update:
        #   score[arm] = (1 - beta) * score[arm] + beta * adv_reward
        #
        # This gives exponentially decaying weight to older observations.
        # With beta=0.1, the effective half-life is ~7 pulls, so the
        # score reflects roughly the last 10 outcomes for this arm.
        self.counts[arm] += 1
        self.total_pulls += 1
        self.scores[arm] = (1.0 - self.beta) * self.scores[arm] + self.beta * adv_reward

    def update(self, mm_objective_value: float) -> None:
        """
        Update the EWMA score of the last-pulled arm.

        Kept for single-action callers; regime-level schedules should prefer
        ``update_arms`` with the list of arms used in the episode.
        """
        if self._last_arm is None:
            raise RuntimeError("update() called before select_action()")
        self.update_arm(self._last_arm, mm_objective_value)

    def update_arms(self, arms: List[int], mm_objective_value: float) -> None:
        """Update every regime-level arm selected during an episode."""
        for arm in arms:
            self.update_arm(int(arm), mm_objective_value)

    def set_linear_epsilon(self, step: int, total_steps: int) -> None:
        """
        Set epsilon on a linear schedule from epsilon_start to epsilon_min.

        ``step`` is zero-indexed.  At step 0 epsilon is epsilon_start; at
        the final step it reaches epsilon_min.
        """
        denom = max(1, int(total_steps) - 1)
        progress = min(1.0, max(0.0, float(step) / float(denom)))
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - progress * (self.epsilon_start - self.epsilon_min),
        )

    # ──────────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """
        Return a dict summarising the bandit's current state.

        Useful for structured logging, checkpoint metadata, and
        TensorBoard.  The runner is responsible for formatting this
        into prints or scalar logs.

        Returns
        -------
        dict
            Keys: tau_grid, scores, counts, epsilon, total_pulls,
            best_arm (index), best_tau (value), best_score (float).
        """
        best_arm = int(np.argmax(self.scores))
        return {
            "tau_grid": self.tau_grid,
            "scores": self.scores.tolist(),
            "counts": self.counts.tolist(),
            "epsilon": self.epsilon,
            "total_pulls": self.total_pulls,
            "best_arm": best_arm,
            "best_tau": self.tau_grid[best_arm],
            "best_score": float(self.scores[best_arm]),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Serialisation
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the bandit's state to a checkpoint file.

        Uses ``torch.save`` for consistency with the rest of the
        codebase (all other agents serialise via torch).  The payload
        contains only Python/numpy primitives — no tensors.

        Parameters
        ----------
        path : str
            File path for the checkpoint (typically .pt extension).
        """
        torch.save({
            "tau_grid": self.tau_grid,
            "scores": self.scores.copy(),
            "counts": self.counts.copy(),
            "epsilon_start": self.epsilon_start,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "beta": self.beta,
            "burn_in": self.burn_in,
            "total_pulls": self.total_pulls,
        }, path)

    def load(self, path: str) -> None:
        """
        Restore the bandit's state from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to a checkpoint saved by ``save()``.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.tau_grid = ckpt["tau_grid"]
        self.n_arms = len(self.tau_grid)
        self.scores = np.array(ckpt["scores"], dtype=np.float64)
        self.counts = np.array(ckpt["counts"], dtype=np.int64)
        self.epsilon_start = ckpt.get("epsilon_start", self.epsilon_start)
        self.epsilon = ckpt["epsilon"]
        self.epsilon_min = ckpt.get("epsilon_min", 0.10)
        self.beta = ckpt["beta"]
        self.burn_in = ckpt["burn_in"]
        self.total_pulls = ckpt["total_pulls"]


class BanditTauRegimeSchedule:
    """
    Callable regime schedule where a tau bandit acts at regime boundaries.

    The LOB simulator calls this object with the MO-event index and receives
    the current p_buy.  Whenever the current regime is exhausted, the bandit
    chooses a tau for the next regime, the regime length is drawn from
    Exp(mean=tau), and p_buy is drawn uniformly from [p_lo, p_hi].
    """

    def __init__(
        self,
        bandit: AdversaryTauAgent,
        n_mo_events: int,
        p_lo: float,
        p_hi: float,
        seed: int,
        train: bool = False,
    ) -> None:
        self.bandit = bandit
        self.n_mo_events = int(n_mo_events)
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.rng = np.random.default_rng(seed)
        self.train = bool(train)

        self.boundaries: List[int] = [0]
        self.p_values: List[float] = []
        self.taus: List[float] = []
        self.arms: List[int] = []
        self.regime_rewards: List[float] = []
        self.regime_pnl_deltas: List[float] = []
        self.regime_pnl_rates: List[float] = []
        self.regime_lengths_mo: List[float] = []
        self.regimes: List[Dict[str, float]] = []
        self._current_end_mo = -1
        self._current_p_buy = 0.5
        self._current_arm: Optional[int] = None
        self._current_start_pnl: Optional[float] = None
        self._latest_pnl: float = 0.0
        self._finalized_current = True

    def _finish_current_regime(self, end_mo: int, terminal_pnl: Optional[float] = None) -> None:
        if self._current_arm is None or self._finalized_current:
            return

        end_mo = min(int(end_mo), self.n_mo_events)
        start_mo = 0
        if self.regimes:
            start_mo = int(self.regimes[-1].get("mo_start", 0.0))
        regime_length_mo = max(1, end_mo - start_mo)

        pnl_end = self._latest_pnl if terminal_pnl is None else float(terminal_pnl)
        pnl_start = 0.0 if self._current_start_pnl is None else float(self._current_start_pnl)
        pnl_delta = pnl_end - pnl_start
        pnl_rate = pnl_delta / float(regime_length_mo)
        adv_reward = -pnl_rate

        self.regime_pnl_deltas.append(float(pnl_delta))
        self.regime_pnl_rates.append(float(pnl_rate))
        self.regime_lengths_mo.append(float(regime_length_mo))
        self.regime_rewards.append(float(adv_reward))
        if self.regimes:
            self.regimes[-1]["mo_end_actual"] = float(end_mo)
            self.regimes[-1]["length_actual_mo"] = float(regime_length_mo)
            self.regimes[-1]["pnl_start"] = float(pnl_start)
            self.regimes[-1]["pnl_end"] = float(pnl_end)
            self.regimes[-1]["pnl_delta"] = float(pnl_delta)
            self.regimes[-1]["pnl_rate"] = float(pnl_rate)
            self.regimes[-1]["adv_reward"] = float(adv_reward)

        if self.train:
            self.bandit.update_arm(self._current_arm, mm_objective_value=pnl_rate)

        self._finalized_current = True

    def _start_new_regime(self, mo_idx: int) -> None:
        self._finish_current_regime(end_mo=int(mo_idx))

        arm = int(self.bandit.select_action())
        tau = float(self.bandit.get_tau())
        length = max(1, int(np.ceil(self.rng.exponential(tau))))
        end_mo = int(mo_idx + length)
        p_buy = float(self.rng.uniform(self.p_lo, self.p_hi))

        self._current_end_mo = end_mo
        self._current_p_buy = p_buy
        self._current_arm = arm
        self._current_start_pnl = float(self._latest_pnl)
        self._finalized_current = False

        if self.boundaries[-1] != int(mo_idx):
            self.boundaries.append(int(mo_idx))
        self.boundaries.append(end_mo)
        self.p_values.append(p_buy)
        self.taus.append(tau)
        self.arms.append(arm)
        self.regimes.append(
            {
                "mo_start": float(mo_idx),
                "mo_end": float(end_mo),
                "length": float(length),
                "tau": tau,
                "arm": float(arm),
                "p_buy": p_buy,
                "expected_p_buy": p_buy,
                "pnl_start": float(self._current_start_pnl),
            }
        )

    def __call__(self, step_idx: int) -> float:
        mo_idx = int(step_idx)
        if self._current_end_mo < 0 or mo_idx >= self._current_end_mo:
            self._start_new_regime(mo_idx)
        return self._current_p_buy

    def on_step_reward(self, reward: float, mm, is_mo: bool = False) -> None:
        if mm is not None and hasattr(mm, "total_pnl"):
            self._latest_pnl = float(mm.total_pnl())

    def finish_episode(self, terminal_pnl: float, train: Optional[bool] = None) -> Dict[str, float]:
        if train is not None:
            self.train = bool(train)
        self._finish_current_regime(end_mo=self._current_end_mo, terminal_pnl=float(terminal_pnl))
        return {
            "n_regimes": float(len(self.taus)),
            "mean_tau": float(np.mean(self.taus)) if self.taus else float("nan"),
            "mean_regime_length_mo": (
                float(np.mean(self.regime_lengths_mo)) if self.regime_lengths_mo else 0.0
            ),
            "mean_regime_pnl_rate": (
                float(np.mean(self.regime_pnl_rates)) if self.regime_pnl_rates else 0.0
            ),
            "mean_regime_adv_reward": (
                float(np.mean(self.regime_rewards)) if self.regime_rewards else 0.0
            ),
            "sum_regime_adv_reward": (
                float(np.sum(self.regime_rewards)) if self.regime_rewards else 0.0
            ),
        }

    def finalize_episode(self, terminal_pnl: float, train: Optional[bool] = None) -> Dict[str, float]:
        return self.finish_episode(terminal_pnl=terminal_pnl, train=train)


def make_bandit_tau_regime_schedule(
    bandit: AdversaryTauAgent,
    n_mo_events: int,
    p_lo: float,
    p_hi: float,
    seed: int,
    train: bool = False,
) -> Callable[[int], float]:
    return BanditTauRegimeSchedule(
        bandit=bandit,
        n_mo_events=n_mo_events,
        p_lo=p_lo,
        p_hi=p_hi,
        seed=seed,
        train=train,
    )
