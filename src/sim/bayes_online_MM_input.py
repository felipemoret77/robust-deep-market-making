from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np


@dataclass
class BayesOnlineMMInputConfig:
    """Configuration for the online Bayesian MO-flow filter.

    The filter is meant to replace fixed slow/fast MO-flow EWMAs.  It maintains
    a posterior over the current regime run length and a Beta posterior over the
    current buy-MO probability under each run-length hypothesis.
    """

    # Mean regime duration measured on the MO clock.
    tau_r: float = 60.0

    # Prior for p_buy.  With prior_mode="beta", this is a standard Beta prior
    # on [0, 1].  With prior_mode="truncated_beta_grid", the same Beta kernel is
    # truncated to [p_low, p_high].
    prior_a: float = 1.0
    prior_b: float = 1.0
    prior_mode: str = "truncated_beta_grid"  # "beta" or "truncated_beta_grid"
    p_low: float = 0.2
    p_high: float = 0.8

    # Constant changepoint probability per MO.  If None, use
    # h = 1 - exp(-1/tau_r), matching an exponential duration model.
    hazard: Optional[float] = None

    # Computational controls.  Keeping about 5*tau_r run-length hypotheses is
    # usually enough for an exponential hazard.
    max_run_length: Optional[int] = None
    max_run_multiple: float = 5.0
    prune_threshold: float = 1e-10

    # Feature controls.
    cp_window: int = 5
    run_feature_multiple: float = 5.0

    # Numerical quadrature controls for truncated_beta_grid mode.
    trunc_grid_size: int = 401


class BayesOnlineMMInput:
    """Online Bayesian input features for regime-switching market making.

    Observation convention
    ----------------------
    y_t = 1 for a buy market order and y_t = 0 for a sell market order.

    Regime model
    ------------
    Within a regime, y_t ~ Bernoulli(p).  The latent signed flow bias is

        m = 2*p - 1.

    The filter keeps q_t(r) = P(run_length_t = r | y_1:t).  For each run length
    r it also keeps a Beta posterior over p.  The resulting features are:

        bayes_m_hat
            Posterior mean of m = 2*p - 1.

        bayes_cp_prob
            Posterior probability that the current regime is very young,
            sum_{r <= cp_window} q_t(r).

        bayes_expected_run_length
            Normalized expected run length, suitable as a neural-network input.

        bayes_uncertainty
            Posterior standard deviation of m.

    This module is deliberately standalone.  It does not modify MM_LOB_SIM or
    the RL controller; those can call update_from_order_direction(...) and then
    append feature_dict() to the state later.
    """

    def __init__(self, config: Optional[BayesOnlineMMInputConfig] = None) -> None:
        self.config = config or BayesOnlineMMInputConfig()
        self._validate_config()
        self._moments_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._grid: Optional[np.ndarray] = None
        if self.config.prior_mode == "truncated_beta_grid":
            self._grid = np.linspace(
                float(self.config.p_low),
                float(self.config.p_high),
                int(self.config.trunc_grid_size),
                dtype=np.float64,
            )
        self.reset()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the filter to its prior state."""
        self.weights = np.zeros(0, dtype=np.float64)
        self.a_params = np.zeros(0, dtype=np.float64)
        self.b_params = np.zeros(0, dtype=np.float64)
        self.run_lengths = np.zeros(0, dtype=np.int64)
        self.n_mo_updates = 0
        self._refresh_features()

    def update_from_order_direction(
        self,
        order_direction: int,
        order_size: float = 1.0,
    ) -> Dict[str, float]:
        """Update from simulator order_direction.

        Positive direction is interpreted as a buy MO; negative direction as a
        sell MO.  A zero direction carries no Bernoulli observation and leaves
        the filter unchanged.  order_size is accepted for API compatibility with
        the current EWMA updater, but the Beta-Bernoulli filter counts MO signs,
        not sizes.
        """
        side = int(np.sign(order_direction))
        if side == 0:
            return self.feature_dict()
        y = 1 if side > 0 else 0
        return self.update_y(y)

    def update_from_signed_mo(self, xi: float) -> Dict[str, float]:
        """Update from xi in {-1, +1}; +1 is buy and -1 is sell."""
        side = int(np.sign(xi))
        if side == 0:
            return self.feature_dict()
        return self.update_y(1 if side > 0 else 0)

    def update_y(self, y: int) -> Dict[str, float]:
        """Update the BOCPD recursion with y in {0, 1}."""
        y_int = int(y)
        if y_int not in (0, 1):
            raise ValueError(f"y must be 0 or 1, got {y!r}")

        if len(self.weights) == 0:
            self.weights = np.array([1.0], dtype=np.float64)
            self.a_params = np.array([self.config.prior_a + y_int], dtype=np.float64)
            self.b_params = np.array([self.config.prior_b + 1 - y_int], dtype=np.float64)
            self.run_lengths = np.array([1], dtype=np.int64)
            self.n_mo_updates += 1
            self._refresh_features()
            return self.feature_dict()

        h_vals = np.array([self._hazard(int(r)) for r in self.run_lengths], dtype=np.float64)
        h_vals = np.clip(h_vals, 0.0, 1.0)

        # Changepoint branch: a new regime starts before this observation.
        prior_mu, _ = self._posterior_moments(self.config.prior_a, self.config.prior_b)
        prior_pred = prior_mu if y_int == 1 else 1.0 - prior_mu
        cp_mass = float(np.sum(self.weights * h_vals))
        cp_weight = cp_mass * prior_pred

        new_weights: List[float] = [cp_weight]
        new_a: List[float] = [float(self.config.prior_a + y_int)]
        new_b: List[float] = [float(self.config.prior_b + 1 - y_int)]
        new_r: List[int] = [1]

        # Continuation branches: old run length r grows to r+1.
        for w_i, a_i, b_i, r_i, h_i in zip(
            self.weights,
            self.a_params,
            self.b_params,
            self.run_lengths,
            h_vals,
        ):
            mu_i, _ = self._posterior_moments(float(a_i), float(b_i))
            pred_i = mu_i if y_int == 1 else 1.0 - mu_i
            grow_weight = float(w_i) * (1.0 - float(h_i)) * pred_i
            new_weights.append(grow_weight)
            new_a.append(float(a_i + y_int))
            new_b.append(float(b_i + 1 - y_int))
            new_r.append(int(r_i) + 1)

        weights = np.asarray(new_weights, dtype=np.float64)
        a_params = np.asarray(new_a, dtype=np.float64)
        b_params = np.asarray(new_b, dtype=np.float64)
        run_lengths = np.asarray(new_r, dtype=np.int64)

        weights, a_params, b_params, run_lengths = self._prune_and_normalize(
            weights=weights,
            a_params=a_params,
            b_params=b_params,
            run_lengths=run_lengths,
        )

        self.weights = weights
        self.a_params = a_params
        self.b_params = b_params
        self.run_lengths = run_lengths
        self.n_mo_updates += 1
        self._refresh_features()
        return self.feature_dict()

    def predictive_prob_y(self, y: int) -> float:
        """Return P(y_t=y | y_1:t-1) under this filter."""
        y_int = int(y)
        if y_int not in (0, 1):
            raise ValueError(f"y must be 0 or 1, got {y!r}")

        prior_mu, _ = self._posterior_moments(self.config.prior_a, self.config.prior_b)
        prior_pred = prior_mu if y_int == 1 else 1.0 - prior_mu
        if len(self.weights) == 0:
            return float(np.clip(prior_pred, 1e-12, 1.0))

        h_vals = np.array([self._hazard(int(r)) for r in self.run_lengths], dtype=np.float64)
        h_vals = np.clip(h_vals, 0.0, 1.0)
        pred = float(np.sum(self.weights * h_vals) * prior_pred)

        for w_i, a_i, b_i, h_i in zip(
            self.weights,
            self.a_params,
            self.b_params,
            h_vals,
        ):
            mu_i, _ = self._posterior_moments(float(a_i), float(b_i))
            pred_i = mu_i if y_int == 1 else 1.0 - mu_i
            pred += float(w_i) * (1.0 - float(h_i)) * pred_i
        return float(np.clip(pred, 1e-12, 1.0))

    # ------------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------------
    def feature_dict(self, prefix: str = "bayes") -> Dict[str, float]:
        """Return normalized features ready to append to an RL state dict."""
        return {
            f"{prefix}_m_hat": float(self.bayes_m_hat),
            f"{prefix}_cp_prob": float(self.bayes_cp_prob),
            # This one is normalized to [0, 1] for neural-network input.
            f"{prefix}_expected_run_length": float(self.bayes_expected_run_length),
            f"{prefix}_uncertainty": float(self.bayes_uncertainty),
        }

    def telemetry_dict(self, prefix: str = "bayes") -> Dict[str, float]:
        """Return features plus raw diagnostics for logging."""
        out = self.feature_dict(prefix=prefix)
        out.update(
            {
                f"{prefix}_p_hat": float(self.bayes_p_hat),
                f"{prefix}_expected_run_length_raw": float(self.expected_run_length_raw),
                f"{prefix}_n_components": float(len(self.weights)),
                f"{prefix}_n_mo_updates": float(self.n_mo_updates),
                f"{prefix}_hazard": float(self._hazard(1)),
            }
        )
        return out

    def state_dict(self) -> Dict[str, Any]:
        """Serializable internal state, useful for checkpointing."""
        return {
            "config": asdict(self.config),
            "weights": self.weights.tolist(),
            "a_params": self.a_params.tolist(),
            "b_params": self.b_params.tolist(),
            "run_lengths": self.run_lengths.tolist(),
            "n_mo_updates": int(self.n_mo_updates),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load a state produced by state_dict()."""
        if "config" in state:
            valid_keys = set(BayesOnlineMMInputConfig.__dataclass_fields__.keys())
            cfg_dict = {k: v for k, v in dict(state["config"]).items() if k in valid_keys}
            self.config = BayesOnlineMMInputConfig(**cfg_dict)
            self._validate_config()
            self._moments_cache.clear()
            self._grid = None
            if self.config.prior_mode == "truncated_beta_grid":
                self._grid = np.linspace(
                    float(self.config.p_low),
                    float(self.config.p_high),
                    int(self.config.trunc_grid_size),
                    dtype=np.float64,
                )

        self.weights = np.asarray(state.get("weights", []), dtype=np.float64)
        self.a_params = np.asarray(state.get("a_params", []), dtype=np.float64)
        self.b_params = np.asarray(state.get("b_params", []), dtype=np.float64)
        self.run_lengths = np.asarray(state.get("run_lengths", []), dtype=np.int64)
        self.n_mo_updates = int(state.get("n_mo_updates", 0))
        if len(self.weights) > 0:
            self.weights, self.a_params, self.b_params, self.run_lengths = (
                self._prune_and_normalize(
                    self.weights,
                    self.a_params,
                    self.b_params,
                    self.run_lengths,
                )
            )
        self._refresh_features()

    # ------------------------------------------------------------------
    # Internal math
    # ------------------------------------------------------------------
    def _refresh_features(self) -> None:
        if len(self.weights) == 0:
            prior_mu, prior_var = self._posterior_moments(
                self.config.prior_a,
                self.config.prior_b,
            )
            self.bayes_p_hat = float(prior_mu)
            self.bayes_m_hat = float(2.0 * prior_mu - 1.0)
            self.bayes_cp_prob = 0.0
            self.expected_run_length_raw = 0.0
            self.bayes_expected_run_length = 0.0
            self.bayes_uncertainty = float(2.0 * math.sqrt(max(prior_var, 0.0)))
            return

        mus = np.zeros_like(self.weights, dtype=np.float64)
        vars_p = np.zeros_like(self.weights, dtype=np.float64)
        for i, (a_i, b_i) in enumerate(zip(self.a_params, self.b_params)):
            mus[i], vars_p[i] = self._posterior_moments(float(a_i), float(b_i))

        p_hat = float(np.sum(self.weights * mus))
        var_p = float(np.sum(self.weights * (vars_p + (mus - p_hat) ** 2)))

        cp_window = max(int(self.config.cp_window), 1)
        cp_prob = float(np.sum(self.weights[self.run_lengths <= cp_window]))

        expected_run = float(np.sum(self.weights * self.run_lengths.astype(np.float64)))
        run_scale = max(float(self.config.run_feature_multiple) * float(self.config.tau_r), 1.0)
        run_feature = math.log1p(max(expected_run, 0.0)) / math.log1p(run_scale)
        run_feature = min(max(run_feature, 0.0), 1.0)

        self.bayes_p_hat = p_hat
        self.bayes_m_hat = float(2.0 * p_hat - 1.0)
        self.bayes_cp_prob = min(max(cp_prob, 0.0), 1.0)
        self.expected_run_length_raw = expected_run
        self.bayes_expected_run_length = run_feature
        self.bayes_uncertainty = float(2.0 * math.sqrt(max(var_p, 0.0)))

    def _posterior_moments(self, a: float, b: float) -> Tuple[float, float]:
        """Return posterior mean and variance of p."""
        if self.config.prior_mode == "beta":
            total = float(a + b)
            if total <= 0.0:
                return 0.5, 0.25
            mean = float(a / total)
            var = float((a * b) / (total * total * (total + 1.0)))
            return mean, max(var, 0.0)

        # Truncated Beta grid mode.  Cache because the same (a, b) pairs occur
        # frequently across episodes and run-length hypotheses.
        key = (float(a), float(b))
        cached = self._moments_cache.get(key)
        if cached is not None:
            return cached

        if self._grid is None:
            raise RuntimeError("Internal grid is missing for truncated_beta_grid mode")
        p = self._grid
        eps = 1e-12
        logp = np.log(np.clip(p, eps, 1.0))
        log1mp = np.log(np.clip(1.0 - p, eps, 1.0))
        log_kernel = (float(a) - 1.0) * logp + (float(b) - 1.0) * log1mp
        shift = float(np.max(log_kernel))
        density = np.exp(log_kernel - shift)

        den = self._trapz(density, p)
        if (not np.isfinite(den)) or den <= 0.0:
            mean = 0.5 * (float(self.config.p_low) + float(self.config.p_high))
            var = ((float(self.config.p_high) - float(self.config.p_low)) ** 2) / 12.0
        else:
            mean = float(self._trapz(p * density, p) / den)
            second = float(self._trapz((p ** 2) * density, p) / den)
            var = max(second - mean * mean, 0.0)

        out = (mean, var)
        self._moments_cache[key] = out
        return out

    def _hazard(self, run_length: int) -> float:
        if self.config.hazard is not None:
            return float(np.clip(float(self.config.hazard), 0.0, 1.0))
        tau = max(float(self.config.tau_r), 1.0)
        return float(1.0 - math.exp(-1.0 / tau))

    def _max_run_length(self) -> int:
        if self.config.max_run_length is not None:
            return max(int(self.config.max_run_length), 1)
        return max(int(round(float(self.config.max_run_multiple) * float(self.config.tau_r))), 1)

    def _prune_and_normalize(
        self,
        weights: np.ndarray,
        a_params: np.ndarray,
        b_params: np.ndarray,
        run_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_run = self._max_run_length()
        threshold = max(float(self.config.prune_threshold), 0.0)

        finite = np.isfinite(weights) & (weights >= 0.0)
        keep = finite & (run_lengths <= max_run) & (weights > threshold)

        if not np.any(keep):
            # Keep the largest finite component so the filter never collapses.
            if np.any(finite):
                idx = int(np.argmax(np.where(finite, weights, -1.0)))
            else:
                idx = 0
            keep = np.zeros_like(weights, dtype=bool)
            keep[idx] = True

        weights = weights[keep]
        a_params = a_params[keep]
        b_params = b_params[keep]
        run_lengths = run_lengths[keep]

        total = float(np.sum(weights))
        if (not np.isfinite(total)) or total <= 0.0:
            weights = np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float64)
        else:
            weights = weights / total
        return weights, a_params, b_params, run_lengths

    @staticmethod
    def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(values, grid))
        return float(np.trapz(values, grid))

    def _validate_config(self) -> None:
        if self.config.prior_a <= 0.0 or self.config.prior_b <= 0.0:
            raise ValueError("prior_a and prior_b must be positive")
        if self.config.tau_r <= 0.0:
            raise ValueError("tau_r must be positive")
        if self.config.prior_mode not in {"beta", "truncated_beta_grid"}:
            raise ValueError("prior_mode must be 'beta' or 'truncated_beta_grid'")
        if not (0.0 <= self.config.p_low < self.config.p_high <= 1.0):
            raise ValueError("Require 0 <= p_low < p_high <= 1")
        if int(self.config.trunc_grid_size) < 25:
            raise ValueError("trunc_grid_size must be at least 25")


class BayesOnlineMMMixtureInput:
    """Bayesian model average over several fixed-hazard BOCPD filters.

    Each component is a regular BayesOnlineMMInput with its own tau_r, hence
    its own changepoint hazard.  Component weights are updated online by the
    one-step predictive likelihood of the observed MO sign.  This avoids giving
    the market maker a single oracle tau while keeping the exposed RL features
    identical to the single-filter API.
    """

    def __init__(
        self,
        configs: List[BayesOnlineMMInputConfig],
        prior_weights: Optional[List[float]] = None,
    ) -> None:
        if not configs:
            raise ValueError("BayesOnlineMMMixtureInput requires at least one config")
        self.filters = [BayesOnlineMMInput(cfg) for cfg in configs]
        if prior_weights is None:
            weights = np.full(len(self.filters), 1.0 / len(self.filters), dtype=np.float64)
        else:
            weights = np.asarray(prior_weights, dtype=np.float64)
            if weights.shape != (len(self.filters),):
                raise ValueError("prior_weights must have one entry per mixture component")
            weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
            total = float(np.sum(weights))
            if total <= 0.0:
                weights = np.full(len(self.filters), 1.0 / len(self.filters), dtype=np.float64)
            else:
                weights = weights / total
        self.prior_model_weights = weights.copy()
        self.model_weights = weights.copy()

    @classmethod
    def from_base_config(
        cls,
        base_config: BayesOnlineMMInputConfig,
        tau_grid: Optional[List[float]] = None,
        prior_weights: Optional[List[float]] = None,
    ) -> "BayesOnlineMMMixtureInput":
        taus = tau_grid if tau_grid is not None else [15.0, 30.0, 60.0, 120.0]
        clean_taus = []
        for tau in taus:
            tau_f = float(tau)
            if tau_f <= 0.0:
                raise ValueError("All mixture tau values must be positive")
            if tau_f not in clean_taus:
                clean_taus.append(tau_f)
        base = asdict(base_config)
        configs = []
        for tau in clean_taus:
            cfg_dict = dict(base)
            cfg_dict["tau_r"] = tau
            # Mixture mode is explicitly a mixture over hazards, so each
            # component should derive its hazard from its own tau.
            cfg_dict["hazard"] = None
            configs.append(BayesOnlineMMInputConfig(**cfg_dict))
        return cls(configs=configs, prior_weights=prior_weights)

    def reset(self) -> None:
        for filt in self.filters:
            filt.reset()
        self.model_weights = self.prior_model_weights.copy()

    def update_from_order_direction(
        self,
        order_direction: int,
        order_size: float = 1.0,
    ) -> Dict[str, float]:
        side = int(np.sign(order_direction))
        if side == 0:
            return self.feature_dict()
        return self.update_y(1 if side > 0 else 0)

    def update_from_signed_mo(self, xi: float) -> Dict[str, float]:
        side = int(np.sign(xi))
        if side == 0:
            return self.feature_dict()
        return self.update_y(1 if side > 0 else 0)

    def update_y(self, y: int) -> Dict[str, float]:
        y_int = int(y)
        if y_int not in (0, 1):
            raise ValueError(f"y must be 0 or 1, got {y!r}")

        preds = np.asarray([f.predictive_prob_y(y_int) for f in self.filters], dtype=np.float64)
        for filt in self.filters:
            filt.update_y(y_int)

        weights = self.model_weights * np.clip(preds, 1e-12, 1.0)
        total = float(np.sum(weights))
        if (not np.isfinite(total)) or total <= 0.0:
            self.model_weights = np.full(len(self.filters), 1.0 / len(self.filters), dtype=np.float64)
        else:
            self.model_weights = weights / total
        return self.feature_dict()

    def feature_dict(self, prefix: str = "bayes") -> Dict[str, float]:
        combined = self._combined()
        return {
            f"{prefix}_m_hat": combined["m_hat"],
            f"{prefix}_cp_prob": combined["cp_prob"],
            f"{prefix}_expected_run_length": combined["expected_run_length_feature"],
            f"{prefix}_uncertainty": combined["uncertainty"],
        }

    def telemetry_dict(self, prefix: str = "bayes") -> Dict[str, float]:
        combined = self._combined()
        out = self.feature_dict(prefix=prefix)
        out.update(
            {
                f"{prefix}_p_hat": combined["p_hat"],
                f"{prefix}_expected_run_length_raw": combined["expected_run_length_raw"],
                f"{prefix}_n_components": combined["n_components"],
                f"{prefix}_n_mo_updates": combined["n_mo_updates"],
                f"{prefix}_hazard": combined["hazard"],
                f"{prefix}_mixture_enabled": 1.0,
                f"{prefix}_mixture_n_models": float(len(self.filters)),
                f"{prefix}_mixture_tau_mean": combined["tau_mean"],
            }
        )
        for filt, weight in zip(self.filters, self.model_weights):
            tau_key = self._tau_key(float(filt.config.tau_r))
            out[f"{prefix}_mix_w_tau_{tau_key}"] = float(weight)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_weights": self.model_weights.tolist(),
            "filters": [f.state_dict() for f in self.filters],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        filter_states = list(state.get("filters", []))
        for filt, filt_state in zip(self.filters, filter_states):
            filt.load_state_dict(filt_state)
        weights = np.asarray(state.get("model_weights", self.model_weights), dtype=np.float64)
        if weights.shape == self.model_weights.shape:
            weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
            total = float(np.sum(weights))
            if total > 0.0:
                self.model_weights = weights / total

    def _combined(self) -> Dict[str, float]:
        weights = np.asarray(self.model_weights, dtype=np.float64)
        weights = weights / max(float(np.sum(weights)), 1e-12)

        p_vals = np.asarray([f.bayes_p_hat for f in self.filters], dtype=np.float64)
        m_vals = np.asarray([f.bayes_m_hat for f in self.filters], dtype=np.float64)
        cp_vals = np.asarray([f.bayes_cp_prob for f in self.filters], dtype=np.float64)
        run_raw_vals = np.asarray([f.expected_run_length_raw for f in self.filters], dtype=np.float64)
        run_feat_vals = np.asarray([f.bayes_expected_run_length for f in self.filters], dtype=np.float64)
        unc_vals = np.asarray([f.bayes_uncertainty for f in self.filters], dtype=np.float64)
        hazard_vals = np.asarray([f._hazard(1) for f in self.filters], dtype=np.float64)
        tau_vals = np.asarray([f.config.tau_r for f in self.filters], dtype=np.float64)
        n_mo_vals = np.asarray([f.n_mo_updates for f in self.filters], dtype=np.float64)
        n_comp_vals = np.asarray([len(f.weights) for f in self.filters], dtype=np.float64)

        m_hat = float(np.sum(weights * m_vals))
        var_m = float(np.sum(weights * ((unc_vals ** 2) + (m_vals - m_hat) ** 2)))
        return {
            "p_hat": float(np.sum(weights * p_vals)),
            "m_hat": m_hat,
            "cp_prob": float(np.clip(np.sum(weights * cp_vals), 0.0, 1.0)),
            "expected_run_length_raw": float(np.sum(weights * run_raw_vals)),
            "expected_run_length_feature": float(np.clip(np.sum(weights * run_feat_vals), 0.0, 1.0)),
            "uncertainty": float(math.sqrt(max(var_m, 0.0))),
            "hazard": float(np.sum(weights * hazard_vals)),
            "tau_mean": float(np.sum(weights * tau_vals)),
            "n_mo_updates": float(np.max(n_mo_vals)) if len(n_mo_vals) else 0.0,
            "n_components": float(np.sum(n_comp_vals)),
        }

    @staticmethod
    def _tau_key(tau: float) -> str:
        text = f"{float(tau):g}".replace("-", "m").replace(".", "p")
        return text


if __name__ == "__main__":
    # Small smoke demo: a buy-heavy stretch followed by a sell-heavy stretch.
    filt = BayesOnlineMMInput(
        BayesOnlineMMInputConfig(
            tau_r=60.0,
            cp_window=5,
        )
    )
    seq = [1] * 20 + [0] * 20
    for obs in seq:
        filt.update_y(obs)
    print(filt.telemetry_dict())
