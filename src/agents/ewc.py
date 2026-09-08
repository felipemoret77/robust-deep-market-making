#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ewc.py  —  Elastic Weight Consolidation for catastrophic-forgetting prevention
================================================================================

Implements Fisher-weighted anchor regularization per Kirkpatrick et al. 2017:

    L_EWC(θ) = (λ/2) · Σ_i F_ii · (θ_i - θ*_i)²

where F_ii is the diagonal Fisher Information Matrix estimated under the
previous task's policy.  F_ii measures how much each parameter θ_i matters
for preserving Phase A behavior; parameters with large F_ii are strongly
anchored, while redundant ones are free to adapt.

vs plain L2 anchor (current `_compute_anchor_loss` in the controller):

    L_L2(θ) = λ · Σ_i (θ_i - θ*_i)²

EWC applies SELECTIVE preservation, which typically allows a much larger λ
without destroying plasticity — the "soft freeze" is applied only where it
matters.

Reference
---------
Kirkpatrick et al. 2017, "Overcoming Catastrophic Forgetting in Neural
Networks", PNAS 114(13): 3521–3526.  arXiv:1612.00796

Design notes
------------

1) FISHER ESTIMATION METHOD — Boltzmann policy log-likelihood

   For a Q-network, we define a smoothed policy

        π(a|s) = softmax(Q(s, ·) / τ)
        log π(a|s) = Q(s,a)/τ - logsumexp(Q(s,·)/τ)

   and estimate

        F_ii ≈ (1/N) · Σ_{k=1}^N (∂ log π(a_k|s_k) / ∂ θ_i)²

   where (s_k, a_k) are state-action pairs collected from running the
   frozen Phase A policy on the Phase A environment (p_buy = 0.5).  The
   action a_k is the greedy argmax action of the same policy.

   This "Boltzmann Fisher" requires only a single forward pass through
   the Q-network per sample, and does not depend on TD-loss machinery
   (targets, projections, n-step returns, etc).

2) TEMPERATURE τ

   Hyperparameter controlling the sharpness of the softmax policy.
   Default τ=1.0 works well when Q-values live in roughly [-2, 2], which
   is the case for this project's C51 support.  Larger τ → flatter π →
   smaller gradients → smaller F_ii.  Smaller τ → one-hot policy →
   gradients dominated by argmax action, noisier Fisher.

3) PER-SAMPLE GRADIENT ACCUMULATION

   We iterate over samples one at a time (O(N × forward+backward)),
   never batched.  Reason: batching would give (Σ_i ∇L_i)² ≠ Σ_i(∇L_i)²
   and the batched quantity is not a valid Fisher estimator.  With N ≈
   500-2000 samples this is a one-time cost of minutes, acceptable.

4) STATE-ACTION SOURCE

   The caller provides state-action samples, typically collected by
   running the frozen Phase A policy in Phase A's environment (p_buy=0.5
   for the MM case).  Sampling *under the Phase A policy* is what makes
   F an estimator of the Fisher for the task we want to preserve.

5) INTEGRATION

   The resulting FisherDiagonal attaches to a DeepRLController via
   `deep_controller.set_ewc_fisher(fisher.fisher)`.  The controller's
   `_compute_anchor_loss` automatically becomes Fisher-weighted when
   `_ewc_fisher` is set, with the lambda coming from the existing
   `set_anchor_lambda()` knob.  See DeepSarsaQRunner_REGIME.py for the
   full wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FisherDiagonal:
    """Container for the diagonal Fisher Information Matrix and the anchor
    point (θ*) at which it was estimated.

    Attributes
    ----------
    fisher
        Mapping parameter name → tensor with same shape as the parameter,
        holding the estimated F_ii values.  All entries are non-negative.
    anchor
        Mapping parameter name → tensor, snapshot of θ* at estimation time.
        Stored on CPU for portability; the controller moves it to the
        training device.
    n_samples
        Number of (s, a) pairs used in the estimation.
    temperature
        τ used in the Boltzmann policy during estimation.
    normalization
        'mean' (default) — fisher values were divided by n_samples.
        'none' — raw sum of squared gradients.
    metadata
        Configuration snapshot under which this Fisher was estimated
        (checkpoint path, input_dim, feature flags, support bounds,
        regime distribution, etc).  When loading a cached Fisher, the
        caller MUST validate this via `check_compatibility` to avoid
        silently regularizing against a stale baseline.
    """

    fisher:        Dict[str, torch.Tensor]
    anchor:        Dict[str, torch.Tensor]
    n_samples:     int
    temperature:   float = 1.0
    normalization: str   = "mean"
    metadata:      dict  = field(default_factory=dict)

    # ----------- compatibility validation ----------------------------
    def check_compatibility(
        self,
        current_metadata: dict,
        strict_keys: Optional[list] = None,
        warn_keys: Optional[list] = None,
        tol: float = 1e-9,
    ) -> tuple:
        """Validate that the cached Fisher is compatible with the current
        training setup.

        Mirrors the interface of `adr_lite.BaselineCurve.check_compatibility`.

        Returns
        -------
        (ok, errors, warnings) : tuple[bool, list[str], list[str]]
            `ok` is False if any `strict_keys` mismatch.  The caller
            should refuse to use the cache when ok is False.
        """
        saved = self.metadata or {}
        strict_keys = list(strict_keys) if strict_keys is not None else sorted(
            set(saved.keys()) & set(current_metadata.keys())
        )
        warn_keys = list(warn_keys) if warn_keys is not None else []

        errors: list = []
        warnings: list = []

        def _mismatch(key: str) -> Optional[str]:
            if key not in saved:
                return f"{key}: missing in cached Fisher"
            if key not in current_metadata:
                return None
            s_val = saved[key]
            c_val = current_metadata[key]
            try:
                if isinstance(s_val, float) or isinstance(c_val, float):
                    if abs(float(s_val) - float(c_val)) > tol:
                        return f"{key}: cached={s_val} vs current={c_val}"
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(s_val, (list, tuple)) or isinstance(c_val, (list, tuple)):
                if list(s_val) != list(c_val):
                    return f"{key}: cached={list(s_val)} vs current={list(c_val)}"
                return None
            if s_val != c_val:
                return f"{key}: cached={s_val!r} vs current={c_val!r}"
            return None

        for key in strict_keys:
            m = _mismatch(key)
            if m is not None:
                errors.append(m)
        for key in warn_keys:
            if key in strict_keys:
                continue
            m = _mismatch(key)
            if m is not None:
                warnings.append(m)

        return (len(errors) == 0, errors, warnings)

    # ----------- sanity / introspection -------------------------------
    def stats(self) -> Dict[str, dict]:
        """Per-parameter summary statistics of the Fisher diagonal."""
        out = {}
        for name, f in self.fisher.items():
            flat = f.detach().cpu().flatten().float()
            out[name] = dict(
                shape=tuple(f.shape),
                mean=float(flat.mean()),
                std=float(flat.std()),
                max=float(flat.max()),
                min=float(flat.min()),
                nonzero_frac=float((flat > 0).float().mean()),
            )
        return out

    def summary(self, max_lines: int = 20) -> str:
        """Human-readable summary for logging."""
        lines = [
            f"FisherDiagonal(n_samples={self.n_samples}, "
            f"τ={self.temperature}, norm={self.normalization})",
            f"  {len(self.fisher)} parameter groups",
            "",
            f"{'name':<48s} {'shape':<18s} {'mean':>10s} "
            f"{'max':>10s} {'nnz%':>7s}",
            "-" * 95,
        ]
        stats = self.stats()
        for i, (name, st) in enumerate(stats.items()):
            if i >= max_lines:
                lines.append(f"  ... ({len(stats) - max_lines} more parameter groups)")
                break
            short = (name if len(name) <= 48 else "..." + name[-45:])
            lines.append(
                f"{short:<48s} {str(st['shape']):<18s} "
                f"{st['mean']:>10.2e} {st['max']:>10.2e} "
                f"{st['nonzero_frac']:>6.1%}"
            )
        return "\n".join(lines)

    def total_mass(self) -> float:
        """Sum of all F_ii — useful as a scalar diagnostic."""
        return float(sum(f.sum().item() for f in self.fisher.values()))


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_fisher(path: str, fd: FisherDiagonal) -> None:
    """Save a FisherDiagonal to disk (torch checkpoint format)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(
        fisher={k: v.detach().cpu() for k, v in fd.fisher.items()},
        anchor={k: v.detach().cpu() for k, v in fd.anchor.items()},
        n_samples=int(fd.n_samples),
        temperature=float(fd.temperature),
        normalization=str(fd.normalization),
        metadata=dict(fd.metadata) if fd.metadata else {},
        version=2,
    )
    torch.save(payload, path)


def load_fisher(
    path: str,
    device: Optional[torch.device] = None,
) -> FisherDiagonal:
    """Load a FisherDiagonal from disk.  Move tensors to `device` if given.

    Gracefully handles legacy v1 files without metadata (metadata becomes
    an empty dict, so `check_compatibility` will flag every strict key
    as missing and the caller recalibrates).
    """
    d = torch.load(path, map_location="cpu", weights_only=False)

    def _maybe_to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device) if device is not None else t

    return FisherDiagonal(
        fisher={k: _maybe_to(v) for k, v in d["fisher"].items()},
        anchor={k: _maybe_to(v) for k, v in d["anchor"].items()},
        n_samples=int(d["n_samples"]),
        temperature=float(d.get("temperature", 1.0)),
        normalization=str(d.get("normalization", "mean")),
        metadata=dict(d.get("metadata", {})),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FISHER ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

# Callable that takes a batch tensor [B, obs_dim] and returns expected
# Q-values [B, n_actions].  For a C51 distributional network, this should
# project the atom distribution to its expectation, not return the raw
# distribution.
QValueFn = Callable[[torch.Tensor], torch.Tensor]


def estimate_fisher_diagonal_boltzmann(
    q_net: nn.Module,
    q_value_fn: QValueFn,
    state_action_samples: List[Tuple[torch.Tensor, int]],
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    metadata: Optional[dict] = None,
) -> FisherDiagonal:
    """Estimate the diagonal Fisher Information Matrix via Boltzmann policy.

    For each (s, a) pair, computes log π(a|s) under the softmax policy
        π(a|s) = softmax(Q(s, ·) / τ),
    backwards through `q_net`, squares the gradients, and accumulates.

    Parameters
    ----------
    q_net
        The network whose parameters to estimate Fisher for.  Must be in
        the same architectural state it will be in during fine-tuning
        (typically: post-warmstart, NoisyNet→Linear already converted,
        `eval()` mode).
    q_value_fn
        Callable mapping a state batch [B, obs_dim] → expected Q-values
        [B, n_actions].  For a C51 distributional net, this should compute
        E[Z_a(s)] = Σ_i z_i · p_i(s, a).
    state_action_samples
        List of (state_tensor_1D, action_int) pairs.  Each state is a 1-D
        tensor of shape (obs_dim,).
    temperature
        τ for the softmax policy.  Default 1.0.
    device
        Device for the computation.  Default: q_net's device.
    verbose
        Print progress every ~5% of samples.

    Returns
    -------
    FisherDiagonal
        Estimated diagonal Fisher and anchor snapshot (θ*).  Anchor is
        stored on CPU; fisher tensors are on CPU.
    """
    if device is None:
        device = next(q_net.parameters()).device

    n = len(state_action_samples)
    if n == 0:
        raise ValueError("state_action_samples must be non-empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Snapshot anchor weights (θ*) on CPU for portability
    anchor: Dict[str, torch.Tensor] = {
        name: p.detach().clone().cpu()
        for name, p in q_net.named_parameters()
        if p.requires_grad
    }

    # Fisher accumulators on the compute device
    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(p, device=device)
        for name, p in q_net.named_parameters()
        if p.requires_grad
    }

    was_training = q_net.training
    q_net.eval()

    if verbose:
        print(f"[EWC] Estimating Fisher over {n} samples (τ={temperature})")
    progress_every = max(1, n // 20)

    inv_tau = 1.0 / float(temperature)

    try:
        for idx, (state, action) in enumerate(state_action_samples):
            # Ensure state is a tensor of shape [obs_dim]
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            state = state.to(device).float()
            if state.dim() == 1:
                state_batch = state.unsqueeze(0)
            elif state.dim() == 2 and state.shape[0] == 1:
                state_batch = state
            else:
                raise RuntimeError(
                    f"Each state must be 1-D or [1, obs_dim]; got {tuple(state.shape)}"
                )

            a_int = int(action)

            # Fresh zero-grad per sample
            q_net.zero_grad(set_to_none=True)

            q_vals = q_value_fn(state_batch)   # [1, n_actions]
            if q_vals.dim() != 2 or q_vals.shape[0] != 1:
                raise RuntimeError(
                    f"q_value_fn must return [1, n_actions]; got {tuple(q_vals.shape)}"
                )
            n_actions = int(q_vals.shape[1])
            if not (0 <= a_int < n_actions):
                raise RuntimeError(
                    f"Sample {idx}: action {a_int} out of range [0, {n_actions})"
                )

            logits = q_vals * inv_tau
            log_pi = torch.log_softmax(logits, dim=-1)   # [1, n_actions]
            log_pi_a = log_pi[0, a_int]                  # scalar

            # Backward to populate .grad on trainable params
            log_pi_a.backward()

            # Accumulate squared gradients
            for name, p in q_net.named_parameters():
                if p.grad is None:
                    continue
                if name in fisher:
                    fisher[name] += p.grad.detach() ** 2

            if verbose and (idx + 1) % progress_every == 0:
                print(f"[EWC]   sample {idx + 1}/{n}")

        q_net.zero_grad(set_to_none=True)

    finally:
        if was_training:
            q_net.train()

    # Normalize to mean squared gradient
    for name in fisher:
        fisher[name] /= float(n)

    # Move to CPU for portability
    fisher_cpu = {k: v.detach().cpu() for k, v in fisher.items()}

    if verbose:
        all_vals = torch.cat([f.flatten() for f in fisher_cpu.values()]).float()
        print(
            f"[EWC] Fisher done: "
            f"mean={all_vals.mean().item():.3e}  "
            f"std={all_vals.std().item():.3e}  "
            f"max={all_vals.max().item():.3e}  "
            f"nnz%={(all_vals > 0).float().mean().item():.1%}  "
            f"mass={all_vals.sum().item():.3e}"
        )

    return FisherDiagonal(
        fisher=fisher_cpu,
        anchor=anchor,
        n_samples=int(n),
        temperature=float(temperature),
        normalization="mean",
        metadata=dict(metadata) if metadata else {},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE EWC PENALTY (for callers that don't go through the controller)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ewc_penalty(
    model: nn.Module,
    fisher_diagonal: FisherDiagonal,
    lambda_ewc: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the EWC regularization term outside the controller.

        L_EWC = (λ/2) · Σ_i F_ii · (θ_i - θ*_i)²

    This is a drop-in replacement for a plain L2 anchor loss.  Callers
    that integrate via `DeepRLController.set_ewc_fisher(...)` do not need
    this function — the controller's `_compute_anchor_loss` handles it
    automatically when Fisher is set.

    Parameters
    ----------
    model
        The network whose parameters are being regularized (θ).
    fisher_diagonal
        Precomputed FisherDiagonal (provides both F and θ*).
    lambda_ewc
        Regularization strength.  EWC typically tolerates larger λ than
        plain L2 because the Fisher weighting makes the penalty selective.
    device
        Device where the loss should be accumulated.  Default: model's device.

    Returns
    -------
    torch.Tensor
        Scalar loss to add to the main TD loss.  Returns exactly 0 if
        lambda_ewc <= 0.
    """
    if lambda_ewc <= 0:
        dev = device or next(model.parameters()).device
        return torch.tensor(0.0, device=dev)

    device = device or next(model.parameters()).device
    total = torch.tensor(0.0, device=device)

    fisher_map = fisher_diagonal.fisher
    anchor_map = fisher_diagonal.anchor

    for name, p in model.named_parameters():
        if name not in fisher_map or name not in anchor_map:
            continue
        F = fisher_map[name].to(device)
        theta_star = anchor_map[name].to(device)
        total = total + (F * (p - theta_star).pow(2)).sum()

    return 0.5 * float(lambda_ewc) * total
