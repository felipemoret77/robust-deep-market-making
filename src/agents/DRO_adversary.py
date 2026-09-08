#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRO_adversary.py
================

Hard-path adversary for regime-switching market-making training.

This is not a neural adversary.  It keeps the market exogenous by sampling a
finite pool of regime trajectories from the natural regime distribution P0,
then reweights that pool toward trajectories on which the market maker recently
performed poorly.

Natural trajectory law:

    tau_k   ~ Uniform choice(tau_grid)
    L_k     ~ Exponential(mean=tau_k)
    p_buy,k ~ Uniform[p_low, p_high]

Optionally, a fixed fraction of the pool can be sampled from a persistent-side
law that keeps the same p_buy band:

    side_k      in {-1, +1},    P(side_k = side_{k-1}) = rho
    intensity_k ~ Uniform[0, (p_high - p_low) / 2]
    p_buy,k     = (p_low + p_high) / 2 + side_k * intensity_k

For the symmetric band [0.20, 0.80], this preserves the same marginal p_buy
support while making directional flow pressure persistent across regimes.

The adversary chooses which pre-generated natural trajectory to replay.  Its
difficulty score is an EWMA of -terminal_pnl.  A beta-softmax over normalized
difficulties implements the distributionally-robust tilt:

    prob_j ∝ exp(beta * zscore(difficulty_j)).

The difficulty can be updated either from terminal PnL via ``update`` or from
an already-computed robust loss via ``update_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


DEFAULT_DRO_TAU_GRID: List[float] = [15.0, 30.0, 60.0, 120.0, 240.0]


@dataclass
class DRORegimeTrajectory:
    """One pre-generated DRO regime trajectory."""

    seed: int
    boundaries: List[int]
    p_values: List[float]
    tau_values: List[float]
    lengths: List[int]
    persistent_side: bool = False
    rho: Optional[float] = None
    sides: Optional[List[int]] = None
    intensity_values: Optional[List[float]] = None

    def make_schedule(self) -> Callable[[int], float]:
        return DRORegimeSchedule(self)


class DRORegimeSchedule:
    """Callable p_buy schedule backed by a fixed DRO trajectory."""

    def __init__(self, trajectory: DRORegimeTrajectory) -> None:
        self.trajectory = trajectory
        self.boundaries = list(trajectory.boundaries)
        self.p_values = list(trajectory.p_values)
        self.tau_values = list(trajectory.tau_values)
        self.taus = self.tau_values
        self.lengths = list(trajectory.lengths)
        self.persistent_side = bool(trajectory.persistent_side)
        self.rho = trajectory.rho
        self.sides = [] if trajectory.sides is None else list(trajectory.sides)
        self.intensity_values = (
            [] if trajectory.intensity_values is None else list(trajectory.intensity_values)
        )
        self.transitions = []
        self._boundaries_arr = np.asarray(self.boundaries, dtype=np.int64)
        self._p_values_arr = np.asarray(self.p_values, dtype=np.float64)

    def __call__(self, step_idx: int) -> float:
        idx = int(np.searchsorted(self._boundaries_arr, int(step_idx), side="right")) - 1
        idx = max(0, min(idx, len(self._p_values_arr) - 1))
        return float(self._p_values_arr[idx])

    def reset(self) -> None:
        """Compatibility no-op for simulator hooks used by learned schedules."""
        return None

    def on_step_reward(self, reward: float, mm, is_mo: bool = False) -> None:
        """Compatibility no-op; DRO difficulty is updated after the episode."""
        return None


def sample_natural_trajectory(
    seed: int,
    n_mo_events: int,
    tau_grid: Sequence[float] = DEFAULT_DRO_TAU_GRID,
    p_low: float = 0.2,
    p_high: float = 0.8,
    buffer_mo_events: int = 1000,
) -> DRORegimeTrajectory:
    """Sample one natural multi-scale regime trajectory."""
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must contain at least one value")
    if not p_low < p_high:
        raise ValueError(f"p_low must be < p_high, got {p_low} >= {p_high}")

    rng = np.random.default_rng(int(seed))
    tau_grid = [float(x) for x in tau_grid]

    boundaries: List[int] = [0]
    p_values: List[float] = []
    tau_values: List[float] = []
    lengths: List[int] = []

    target = int(n_mo_events) + int(buffer_mo_events)
    while boundaries[-1] < target:
        tau = float(rng.choice(tau_grid))
        length = max(1, int(np.ceil(rng.exponential(tau))))
        p_buy = float(rng.uniform(p_low, p_high))

        boundaries.append(boundaries[-1] + length)
        tau_values.append(tau)
        lengths.append(length)
        p_values.append(p_buy)

    return DRORegimeTrajectory(
        seed=int(seed),
        boundaries=boundaries,
        p_values=p_values,
        tau_values=tau_values,
        lengths=lengths,
    )


def sample_persistent_side_trajectory(
    seed: int,
    n_mo_events: int,
    tau_grid: Sequence[float] = DEFAULT_DRO_TAU_GRID,
    p_low: float = 0.2,
    p_high: float = 0.8,
    rho: float = 0.85,
    buffer_mo_events: int = 1000,
) -> DRORegimeTrajectory:
    """Sample a multi-tau trajectory with persistent directional side."""
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must contain at least one value")
    if not p_low < p_high:
        raise ValueError(f"p_low must be < p_high, got {p_low} >= {p_high}")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    rng = np.random.default_rng(int(seed))
    tau_grid = [float(x) for x in tau_grid]
    p_mid = 0.5 * (float(p_low) + float(p_high))
    p_half_width = 0.5 * (float(p_high) - float(p_low))

    boundaries: List[int] = [0]
    p_values: List[float] = []
    tau_values: List[float] = []
    lengths: List[int] = []
    sides: List[int] = []
    intensity_values: List[float] = []

    side = int(rng.choice([-1, 1]))
    target = int(n_mo_events) + int(buffer_mo_events)
    while boundaries[-1] < target:
        tau = float(rng.choice(tau_grid))
        length = max(1, int(np.ceil(rng.exponential(tau))))
        if sides and float(rng.random()) >= float(rho):
            side = -side
        intensity = float(rng.uniform(0.0, p_half_width))
        p_buy = float(np.clip(p_mid + side * intensity, p_low, p_high))

        boundaries.append(boundaries[-1] + length)
        tau_values.append(tau)
        lengths.append(length)
        p_values.append(p_buy)
        sides.append(int(side))
        intensity_values.append(intensity)

    return DRORegimeTrajectory(
        seed=int(seed),
        boundaries=boundaries,
        p_values=p_values,
        tau_values=tau_values,
        lengths=lengths,
        persistent_side=True,
        rho=float(rho),
        sides=sides,
        intensity_values=intensity_values,
    )


class DROHardPathAdversary:
    """
    Pool-based hard-path adversary.

    Parameters
    ----------
    pool_size:
        Number of natural trajectories in the training pool.
    beta:
        Strength of hard-path reweighting. beta=0 gives uniform sampling.
    eta:
        EWMA update rate for difficulty[j] <- (1-eta)*difficulty[j] + eta*(-pnl).
    eps_uniform:
        Uniform exploration mass mixed into the tilted sampling probabilities.
    max_prob:
        Optional cap on any single trajectory's sampling probability.  This
        prevents the robust tilt from collapsing onto one or two paths.
    refresh_every, refresh_frac:
        Optional pool refresh.  Every refresh_every updates, the easiest
        refresh_frac fraction of trajectories is replaced by fresh natural
        draws.  Set refresh_every <= 0 or refresh_frac <= 0 to disable.
    """

    def __init__(
        self,
        pool_size: int,
        n_mo_events: int,
        tau_grid: Sequence[float] = DEFAULT_DRO_TAU_GRID,
        p_low: float = 0.2,
        p_high: float = 0.8,
        beta: float = 2.0,
        eta: float = 0.10,
        eps_uniform: float = 0.05,
        max_prob: Optional[float] = None,
        seed: int = 0,
        refresh_every: int = 500,
        refresh_frac: float = 0.10,
        buffer_mo_events: int = 1000,
        persistent_mix_prob: float = 0.0,
        persistent_rho: float = 0.85,
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if max_prob is not None and float(max_prob) < 1.0 / float(pool_size):
            raise ValueError(
                f"max_prob={max_prob} is infeasible for pool_size={pool_size}; "
                f"it must be at least {1.0 / float(pool_size):.6f}"
            )
        if not 0.0 <= float(persistent_mix_prob) <= 1.0:
            raise ValueError(
                f"persistent_mix_prob must be in [0, 1], got {persistent_mix_prob}"
            )
        if not 0.0 <= float(persistent_rho) <= 1.0:
            raise ValueError(f"persistent_rho must be in [0, 1], got {persistent_rho}")

        self.pool_size = int(pool_size)
        self.n_mo_events = int(n_mo_events)
        self.tau_grid = [float(x) for x in tau_grid]
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.beta = float(beta)
        self.eta = float(eta)
        self.eps_uniform = float(eps_uniform)
        self.max_prob = None if max_prob is None else float(max_prob)
        self.refresh_every = int(refresh_every)
        self.refresh_frac = float(refresh_frac)
        self.buffer_mo_events = int(buffer_mo_events)
        self.persistent_mix_prob = float(persistent_mix_prob)
        self.persistent_rho = float(persistent_rho)
        self.rng = np.random.default_rng(int(seed))

        self.pool: List[DRORegimeTrajectory] = []
        self.difficulty = np.zeros(self.pool_size, dtype=np.float64)
        self.counts = np.zeros(self.pool_size, dtype=np.int64)
        self.last_probs = np.full(self.pool_size, 1.0 / self.pool_size, dtype=np.float64)
        self.last_index: Optional[int] = None
        self.n_updates = 0

        n_persistent = int(round(self.pool_size * self.persistent_mix_prob))
        initial_persistent_flags = [True] * n_persistent + [False] * (self.pool_size - n_persistent)
        self.rng.shuffle(initial_persistent_flags)
        for use_persistent in initial_persistent_flags:
            self.pool.append(self._new_trajectory(force_persistent=bool(use_persistent)))

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, np.iinfo(np.int32).max))

    def _new_trajectory(self, force_persistent: Optional[bool] = None) -> DRORegimeTrajectory:
        use_persistent = (
            bool(force_persistent)
            if force_persistent is not None
            else self.persistent_mix_prob > 0.0 and float(self.rng.random()) < self.persistent_mix_prob
        )
        if use_persistent:
            return sample_persistent_side_trajectory(
                seed=self._next_seed(),
                n_mo_events=self.n_mo_events,
                tau_grid=self.tau_grid,
                p_low=self.p_low,
                p_high=self.p_high,
                rho=self.persistent_rho,
                buffer_mo_events=self.buffer_mo_events,
            )
        return sample_natural_trajectory(
            seed=self._next_seed(),
            n_mo_events=self.n_mo_events,
            tau_grid=self.tau_grid,
            p_low=self.p_low,
            p_high=self.p_high,
            buffer_mo_events=self.buffer_mo_events,
        )

    def _sampling_probs(self) -> np.ndarray:
        if self.beta <= 0:
            probs = np.full(self.pool_size, 1.0 / self.pool_size, dtype=np.float64)
        else:
            std = float(self.difficulty.std())
            if std < 1e-12:
                z = np.zeros_like(self.difficulty)
            else:
                z = (self.difficulty - float(self.difficulty.mean())) / (std + 1e-12)
            logits = self.beta * z
            logits = logits - float(np.max(logits))
            weights = np.exp(logits)
            probs = weights / max(float(np.sum(weights)), 1e-12)

        eps = min(max(self.eps_uniform, 0.0), 1.0)
        probs = (1.0 - eps) * probs + eps / self.pool_size
        probs = probs / max(float(np.sum(probs)), 1e-12)
        probs = self._apply_max_prob_cap(probs)
        return probs

    def _apply_max_prob_cap(self, probs: np.ndarray) -> np.ndarray:
        """Cap large probabilities and redistribute excess mass."""
        if self.max_prob is None:
            return probs

        cap = min(max(float(self.max_prob), 1.0 / self.pool_size), 1.0)
        capped = np.asarray(probs, dtype=np.float64).copy()

        # A few passes are enough here because the pool is finite and the cap
        # is loose relative to 1/pool_size.  The loop handles the case where
        # redistributing excess pushes another trajectory above the cap.
        for _ in range(self.pool_size):
            over = capped > cap
            if not bool(np.any(over)):
                break
            excess = float(np.sum(capped[over] - cap))
            capped[over] = cap
            under = ~over
            room = cap - capped[under]
            total_room = float(np.sum(room))
            if total_room <= 1e-12:
                break
            capped[under] += excess * room / total_room

        capped = np.minimum(capped, cap)
        total = float(np.sum(capped))
        if total <= 1e-12:
            return np.full(self.pool_size, 1.0 / self.pool_size, dtype=np.float64)

        # If the final normalization would break the cap due to tiny numerical
        # residuals, redistribute once more from capped to uncapped entries.
        capped = capped / total
        over = capped > cap
        if bool(np.any(over)):
            excess = float(np.sum(capped[over] - cap))
            capped[over] = cap
            under = ~over
            if bool(np.any(under)):
                capped[under] += excess * capped[under] / max(float(np.sum(capped[under])), 1e-12)
        return capped / max(float(np.sum(capped)), 1e-12)

    def select_schedule(self) -> Callable[[int], float]:
        probs = self._sampling_probs()
        idx = int(self.rng.choice(self.pool_size, p=probs))
        self.last_probs = probs
        self.last_index = idx
        self.counts[idx] += 1
        schedule = self.pool[idx].make_schedule()
        setattr(schedule, "dro_index", idx)
        setattr(schedule, "dro_prob", float(probs[idx]))
        setattr(schedule, "dro_persistent_side", bool(schedule.persistent_side))
        setattr(schedule, "dro_rho", schedule.rho)
        return schedule

    def update_loss(self, loss: float, index: Optional[int] = None) -> Dict[str, float]:
        """Update the selected trajectory from an externally computed loss.

        Higher loss means the trajectory was harder for the market maker and
        should receive more sampling mass.
        """
        idx = self.last_index if index is None else int(index)
        if idx is None:
            raise RuntimeError("DROHardPathAdversary.update_loss called before select_schedule")

        loss = float(loss)
        self.difficulty[idx] = (1.0 - self.eta) * self.difficulty[idx] + self.eta * loss
        self.n_updates += 1
        traj = self.pool[idx]

        if (
            self.refresh_every > 0
            and self.refresh_frac > 0.0
            and self.n_updates % self.refresh_every == 0
        ):
            self.refresh_easiest()

        probs = self.last_probs
        return {
            "index": float(idx),
            "loss": float(loss),
            "difficulty": float(self.difficulty[idx]),
            "prob": float(probs[idx]),
            "mean_difficulty": float(np.mean(self.difficulty)),
            "max_difficulty": float(np.max(self.difficulty)),
            "min_difficulty": float(np.min(self.difficulty)),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "selected_count": float(self.counts[idx]),
            "max_prob": float("nan") if self.max_prob is None else float(self.max_prob),
            "persistent_side": float(bool(traj.persistent_side)),
            "rho": float("nan") if traj.rho is None else float(traj.rho),
        }

    def update(self, terminal_pnl: float, index: Optional[int] = None) -> Dict[str, float]:
        """Backward-compatible update from terminal PnL."""
        return self.update_loss(-float(terminal_pnl), index=index)

    def refresh_easiest(self) -> None:
        n_refresh = int(np.floor(self.pool_size * self.refresh_frac))
        n_refresh = max(0, min(n_refresh, self.pool_size))
        if n_refresh == 0:
            return

        easy_idx = np.argsort(self.difficulty)[:n_refresh]
        warm_start = float(np.median(self.difficulty))
        for idx in easy_idx:
            self.pool[int(idx)] = self._new_trajectory()
            self.difficulty[int(idx)] = warm_start
            self.counts[int(idx)] = 0

    def summary(self) -> Dict[str, float]:
        probs = self._sampling_probs()
        return {
            "pool_size": float(self.pool_size),
            "beta": float(self.beta),
            "eta": float(self.eta),
            "eps_uniform": float(self.eps_uniform),
            "mean_difficulty": float(np.mean(self.difficulty)),
            "max_difficulty": float(np.max(self.difficulty)),
            "min_difficulty": float(np.min(self.difficulty)),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "max_prob": float(np.max(probs)),
            "min_prob": float(np.min(probs)),
            "persistent_mix_prob": float(self.persistent_mix_prob),
            "persistent_rho": float(self.persistent_rho),
            "persistent_pool_frac": float(np.mean([traj.persistent_side for traj in self.pool])),
        }
