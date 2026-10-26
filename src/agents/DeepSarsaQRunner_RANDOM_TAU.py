"""Paper training entry point. Run with make train-a, train-b or train-c.

Outputs are relative to the launch directory. Historical training metadata
is recorded in config/paper_experiments.json. Experimental ADR, EWC and
legacy adversaries and post-training animations are excluded.
"""

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import shutil
from functools import partial
from pathlib import Path
import os

if __name__ == "__main__":
    from LOB_SIM_SANTA_FE import simulate_LOB
    from MM_LOB_SIM import simulate_LOB_with_MM
    from CONFIG_MM import (
        lam,
        mu,
        delta,
        qrm_params,
        mean_size_LO,
        mean_size_MO,
        USE_QRM,
    )
    from regime_schedule import make_regime_schedule
    from DRO_adversary import DROHardPathAdversary
    from MM_LOB_SIM import (
        reward_spread_capture_inv_quadratic,
        reward_pnl_dampened_inv_quadratic,
    )

    INV_PENALTY_COEFF = 0.001
    USE_DAMPENED_REWARD = True
    from pathlib import Path as _PaperPath
    import json as _paper_json

    _paper_root = _PaperPath(__file__).resolve().parents[2]
    _paper_spec = _paper_json.loads(
        (_paper_root / "config/paper_experiments.json").read_text()
    )
    WARMSTART_CKPT = str(_paper_root / _paper_spec["models"]["B"]["checkpoint"])
    FORCE_NOISY_NET = True
    RESET_SIGMAS = False
    USE_FACTORED_NOISE = True
    USE_FULLY_NOISY = True
    NOISY_WEIGHT_DECAY = 0.0
    USE_FINE_TUNING_TECHNIQUES = True
    ANCHOR_LAMBDA = 0.001
    LR_FEATURE_FACTOR = 0.5
    REPLAY_RESTORE_FRACTION = 0.0
    SAVE_FINAL_REPLAY_BUFFER = False
    CHECKPOINT_SAVE_RETRIES = 3
    CHECKPOINT_RETRY_DELAY_SEC = 2.0
    from dqn_distributional_with_throttle import DeepRLController
    import pandas as pd

    mpl.rcParams["animation.embed_limit"] = 1024
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    import torch

    def _torch_save_atomic(obj, path: str) -> None:
        """
        Save a PyTorch object atomically with a small retry loop.

        This avoids partially-written checkpoint files and mitigates transient
        APFS/macOS write failures that may surface as ENOSPC.
        """
        import tempfile
        import time

        dirname = os.path.dirname(path)
        dirpath = dirname if dirname else "."
        os.makedirs(dirpath, exist_ok=True)
        last_err = None
        for attempt in range(1, CHECKPOINT_SAVE_RETRIES + 1):
            tmp_path = None
            try:
                prefix = ".ckpt_tmp_"
                fd, tmp_path = tempfile.mkstemp(
                    prefix=prefix, suffix=".tmp", dir=dirpath
                )
                os.close(fd)
                torch.save(obj, tmp_path)
                os.replace(tmp_path, path)
                return
            except Exception as exc:
                last_err = exc
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                err_txt = str(exc).lower()
                retryable = (
                    "no space left on device" in err_txt
                    or "open file failed" in err_txt
                    or "inline_container.cc" in err_txt
                    or ("file write failed" in err_txt)
                )
                if retryable and attempt < CHECKPOINT_SAVE_RETRIES:
                    print(
                        f"[CHECKPOINT] Save failed on attempt {attempt}/{CHECKPOINT_SAVE_RETRIES}: {exc}"
                    )
                    print(
                        f"[CHECKPOINT] Retrying in {CHECKPOINT_RETRY_DELAY_SEC:.1f}s..."
                    )
                    time.sleep(CHECKPOINT_RETRY_DELAY_SEC)
                    continue
                raise
        if last_err is not None:
            raise last_err

    def _safe_checkpoint_filename(filename: str, max_len: int = 220) -> str:
        """Shorten long checkpoint basenames while preserving key readable tags."""
        base = os.path.basename(filename)
        if len(base.encode("utf-8")) <= max_len:
            return filename
        import hashlib

        root, ext = os.path.splitext(base)
        digest = hashlib.blake2b(base.encode("utf-8"), digest_size=5).hexdigest()
        suffix = f"_h{digest}{ext}"
        budget = max_len - len(suffix) - 1
        head_len = max(40, int(budget * 0.7))
        tail_len = max(20, budget - head_len)
        short = f"{root[:head_len]}_{root[-tail_len:]}{suffix}"
        return os.path.join(os.path.dirname(filename), short)

    def save_deep_rl_checkpoint(controller, path: str, meta: dict):
        """
        Save a Deep RL training checkpoint to disk.

        The checkpoint contains everything needed to resume training or to
        deploy the trained policy in evaluation mode:

            meta       -- hyperparameters, LOB config, and flags needed to
                         reconstruct the DeepRLController from scratch.
            q_net      -- state_dict of the online Q-network (the trained
                         policy weights).
            epsilon    -- current exploration rate (only relevant for
                         epsilon-greedy; ignored if NoisyNets are enabled).
            target_net -- state_dict of the frozen target network (optional;
                         only present if the controller has one).
            optimizer  -- optimizer state_dict (learning rates, momentum
                         buffers, Adam second-moment estimates, etc.).
                         Needed to resume training without a warm-up phase.

        Parameters
        --------------------------------------------------------------
        controller : DeepRLController
            The RL controller whose state we want to persist.
        path : str
            File path for the .pt checkpoint file.
        meta : dict
            Arbitrary metadata (hyperparams, offsets, seeds, etc.).
        """
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        ckpt = {
            "meta": meta,
            "q_net": controller.q_net.state_dict(),
            "epsilon": float(getattr(controller, "epsilon", 0.0)),
        }
        if hasattr(controller, "target_net") and controller.target_net is not None:
            ckpt["target_net"] = controller.target_net.state_dict()
        if hasattr(controller, "optimizer") and controller.optimizer is not None:
            ckpt["optimizer"] = controller.optimizer.state_dict()
        _torch_save_atomic(ckpt, path)
        print(f"[CHECKPOINT] Saved to: {path}")

    import random
    import os

    def seed_everything(seed: int):
        """
        Lock all random number generators to a fixed seed for reproducibility.

        IMPORTANT: cuDNN deterministic mode is slower (~10-20%) because it
        disables the auto-tuner that picks the fastest convolution algorithm.
        This trade-off is acceptable for research / debugging but should be
        disabled for production deployment where speed matters.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    GLOBAL_SEED = 123
    seed_everything(GLOBAL_SEED)
    print(f"[System] Global Seed locked to {GLOBAL_SEED}. Deterministic Mode ON.")
    NUMBER_TICK_LEVELS = 50
    N_PRIORITY_RANKS = 100
    print("Running offline LOB simulation (no MM) just to inspect environment...")
    msg_df_offline, ob_df_offline, ewma_trace = simulate_LOB(
        lam=lam,
        mu=mu,
        delta=delta,
        number_tick_levels=NUMBER_TICK_LEVELS,
        n_priority_ranks=N_PRIORITY_RANKS,
        number_levels_to_store=20,
        p0=100,
        mean_size_LO=mean_size_LO,
        iterations=50000,
        iterations_to_equilibrium=10000,
        path_save_files=None,
        label_simulation=None,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        random_seed=GLOBAL_SEED,
    )
    print("\n=== OFFLINE ENVIRONMENT STATS ===")
    print("Env Type counts:\n", msg_df_offline["Type"].value_counts())
    print("Env Direction counts:\n", msg_df_offline["Direction"].value_counts())
    print(
        "Spread range:", msg_df_offline["Spread"].min(), msg_df_offline["Spread"].max()
    )
    print(
        "MidPrice range:",
        msg_df_offline["MidPrice"].min(),
        msg_df_offline["MidPrice"].max(),
    )
    print("==============================================================\n")
    print("=== LOB SIMULATOR CONFIG ===")
    print(f"Engine             = {('QRM' if USE_QRM else 'Santa Fe')}")
    print(f"lam                = {lam}")
    print(f"mu                 = {mu}")
    print(f"delta              = {delta}")
    print(f"mean_size_LO       = {mean_size_LO}")
    print(f"mean_size_MO       = {mean_size_MO}")
    if USE_QRM:
        print(f"intensity_model    = {qrm_params.get('intensity_model', 'N/A')}")
        print(f"use_dynamic_pref   = {qrm_params.get('use_dynamic_pref', 'N/A')}")
        print(f"size_q             = {qrm_params.get('size_q', 'N/A')}")
        print(f"aes                = {qrm_params.get('aes', 'N/A')}")
        print(f"theta              = {qrm_params.get('theta', 'N/A')}")
        print(f"theta_reinit       = {qrm_params.get('theta_reinit', 'N/A')}")
    print("==============================================================\n")
    USE_PURE_MM = True
    PURE_MM_OFFSETS = [(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    best_params = {
        "lr": 0.0003,
        "weight_decay": 0.01,
        "lr_decay_type": "linear",
        "gamma": 0.999,
        "epsilon_start": 0.05,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "epsilon_decay_type": "linear",
        "batch_size": 64,
        "target_update_steps": 2000,
        "inv_limit": 8,
        "n_neurons": 256,
        "n_hidden": 1,
        "activation": "relu",
        "elu_alpha": None,
        "dropout_level": 0.0,
        "replay_capacity": 100000,
        "n_steps": 3,
        "use_sarsa": False,
        "use_double": True,
        "use_dueling": True,
        "use_distributional": True,
        "dist_v_min": -3.0,
        "dist_v_max": 3.0,
        "dist_atoms": 101,
        "use_per": True,
        "use_noisy_net": FORCE_NOISY_NET or WARMSTART_CKPT is None,
        "per_alpha_start": 0.6,
        "per_alpha_end": 0.4,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "inv_penalty_coeff": INV_PENALTY_COEFF,
        "n_actions": 4,
        "use_tob_update": False,
        "n_tob_moves": 10,
        "use_event_update": False,
        "n_events": 100,
        "use_time_update": True,
        "min_time_interval": 1.0,
    }
    if WARMSTART_CKPT is not None:
        LR_START = best_params["lr"]
    else:
        LR_START = best_params["lr"]
    LR_END_FACTOR = 0.1

    def get_lr_for_episode(
        ep_index: int, n_episodes: int, use_exp: bool = True
    ) -> float:
        """
        Compute the learning rate to be used AFTER finishing episode `ep_index`,
        using either exponential decay (default) or linear decay.

        Parameters
        --------------------------------------------------------------
        ep_index : int
            Episode index in [0, n_episodes - 1].
        n_episodes : int
            Total number of training episodes.
        use_exp : bool
            If True  -> exponential decay
            If False -> linear decay

        Returns
        --------------------------------------------------------------
        float
            Learning rate to be used for the NEXT episode.
        """
        progress = (ep_index + 1) / n_episodes
        LR_END = LR_START * LR_END_FACTOR
        if use_exp:
            lr = LR_START * LR_END_FACTOR**progress
        else:
            lr = (1.0 - progress) * LR_START + progress * LR_END
        return lr

    def get_curriculum_p_bounds(
        progress: float, delta_start: float = 0.15, delta_end: float = 0.3
    ) -> tuple:
        """
        Compute regime-switching buy_mo_prob bounds for the given progress.

        The bounds are symmetric around 0.5 and controlled by a single
        parameter delta (half-width of the allowed p_buy range):

            p_lo = 0.5 - delta(progress)
            p_hi = 0.5 + delta(progress)

        where delta ramps linearly from delta_start to delta_end.

        Parameters
        --------------------------------------------------------------
        progress : float
            Training progress in [0.0, 1.0].
            0.0 = first episode, 1.0 = last episode.
        delta_start : float
            Half-width at the start of training.  Default 0.15 gives
            p in [0.35, 0.65] -- mild asymmetry, not much harder than
            the stationary baseline (p=0.5).
        delta_end : float
            Half-width at the end of training.  Default 0.30 gives
            p in [0.20, 0.80] -- full asymmetry matching the static
            regime bounds.

        Returns
        --------------------------------------------------------------
        (p_lo, p_hi) : tuple of float
            Regime-switching bounds for this episode.

        Example
        --------------------------------------------------------------
        >>> get_curriculum_p_bounds(0.0)   # start of training
        (0.35, 0.65)
        >>> get_curriculum_p_bounds(0.5)   # midpoint
        (0.275, 0.725)
        >>> get_curriculum_p_bounds(1.0)   # end of training
        (0.2, 0.8)
        """
        progress = max(0.0, min(1.0, progress))
        delta = delta_start + (delta_end - delta_start) * progress
        return (0.5 - delta, 0.5 + delta)

    def get_anchor_lambda(
        progress: float, lambda_start: float = 0.05, lambda_end: float = 0.02
    ) -> float:
        """
        Compute the L2 anchor regularization strength for the given progress.

        Linear decay from lambda_start to lambda_end:

            lambda(progress) = lambda_start + (lambda_end - lambda_start) * progress

        Parameters
        --------------------------------------------------------------
        progress : float
            Training progress in [0.0, 1.0].
        lambda_start : float
            Anchor strength at ep=0.  Higher = stronger pull toward
            pre-trained weights (more preservation, less adaptation).
        lambda_end : float
            Anchor strength at ep=N.  Lower = weaker pull (more freedom
            to adapt to new regimes).

        Returns
        --------------------------------------------------------------
        float
            Anchor lambda for this episode.

        Example
        --------------------------------------------------------------
        >>> get_anchor_lambda(0.0)   # start: strong anchor
        0.05
        >>> get_anchor_lambda(0.5)   # midpoint
        0.035
        >>> get_anchor_lambda(1.0)   # end: relaxed anchor
        0.02
        """
        progress = max(0.0, min(1.0, progress))
        return lambda_start + (lambda_end - lambda_start) * progress

    def get_tied_anchor_lambda(
        delta: float, delta_max: float, base: float, slope: float, gate: float
    ) -> float:
        """
        Compute the curriculum-tied L2 anchor strength for the given ADR delta.

        This is a GATED piecewise-linear function of the CURRENT curriculum
        difficulty (not of elapsed time).  The anchor stays at `base` while
        delta <= gate, then climbs linearly to `base + slope` as delta -> delta_max.
        The rationale is that the demand for weight preservation scales
        with curriculum difficulty -- once the agent is exploring hard
        regimes, gradients from negative experiences can dominate the
        replay buffer (via PER) and degrade skills learned at easier
        levels.  The tied anchor escalates exactly where this risk kicks
        in, without penalising adaptation in the easy/medium plateau
        where a lighter anchor is known to work well.

            u(delta) = clip((delta - gate) / (delta_max - gate), 0, 1)
            lambda(delta) = base + slope  *  u(delta)

        Properties:
            lambda(delta <= gate)  = base                (constant plateau)
            lambda(delta = delta_max) = base + slope        (maximum at extreme)
            lambda is monotone non-decreasing in delta, continuous at the gate.

        Parameters
        --------------------------------------------------------------
        delta : float
            Current ADR curriculum delta.
        delta_max : float
            Maximum delta of the curriculum (upper bound).
        base : float
            Anchor strength for delta <= gate.  Should match the value that
            works empirically well in the easy/medium regime (typically
            the same as the constant baseline).
        slope : float
            Additional anchor strength ramped linearly over [gate, delta_max].
            `base + slope` is the maximum lambda applied, at delta = delta_max.
        gate : float
            delta threshold below which lambda stays at `base`.  Above this value
            lambda starts ramping toward `base + slope`.

        Returns
        --------------------------------------------------------------
        float
            Anchor lambda for this episode, given its current delta.

        Example
        --------------------------------------------------------------
        With (base=0.05, slope=0.10, gate=0.18, delta_max=0.30):
        >>> get_tied_anchor_lambda(0.10, 0.30, 0.05, 0.10, 0.18)
        0.05
        >>> get_tied_anchor_lambda(0.18, 0.30, 0.05, 0.10, 0.18)
        0.05
        >>> get_tied_anchor_lambda(0.24, 0.30, 0.05, 0.10, 0.18)
        0.1
        >>> get_tied_anchor_lambda(0.30, 0.30, 0.05, 0.10, 0.18)
        0.15
        """
        if delta_max <= gate:
            return float(base)
        u = (float(delta) - float(gate)) / (float(delta_max) - float(gate))
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        return float(base) + float(slope) * u

    def get_cyclical_lr_multiplier(
        ep_index: int, period: int = 200, min_factor: float = 0.1
    ) -> float:
        """
        Compute a cosine-annealing warm-restart multiplier for the LR.

        The multiplier follows a cosine half-cycle that resets every
        `period` episodes:

            cycle_progress = (ep_index % period) / period
            multiplier = min_factor + (1 - min_factor) * 0.5 * (1 + cos(pi * cycle_progress))

        At the start of each cycle the multiplier is 1.0 (warm restart);
        at the trough it drops to min_factor.

        The final effective LR is:  lr_base  x  multiplier
        where lr_base comes from the monotonic decay schedule.

        Parameters
        --------------------------------------------------------------
        ep_index : int
            Current episode index.
        period : int
            Length of one cosine half-cycle in episodes.
        min_factor : float
            Minimum multiplier at the trough of each cycle.
            0.1 means the LR drops to 10% of base at the trough.

        Returns
        --------------------------------------------------------------
        float
            Multiplier in [min_factor, 1.0].

        Example
        --------------------------------------------------------------
        >>> get_cyclical_lr_multiplier(0, period=200)    # cycle start
        1.0
        >>> get_cyclical_lr_multiplier(100, period=200)  # cycle midpoint
        0.55
        >>> get_cyclical_lr_multiplier(200, period=200)  # new cycle start
        1.0
        """
        import math as _math_clr

        cycle_progress = ep_index % period / max(period, 1)
        return min_factor + (1.0 - min_factor) * 0.5 * (
            1.0 + _math_clr.cos(_math_clr.pi * cycle_progress)
        )

    EPISODE_LENGTH = 5000
    ITER_TO_EQUILIBRIUM = 1000
    N_EPISODES = 2000
    VERBOSE_LOGGING = False
    EARLY_STOPPING = False
    EARLY_STOP_WINDOW = 50
    EARLY_STOP_DROP = 0.35
    EARLY_STOP_MIN_EPISODES = 200
    BEST_CKPT_WINDOW = 100
    if EARLY_STOPPING:
        assert EARLY_STOP_WINDOW > 0, (
            f"EARLY_STOP_WINDOW must be > 0, got {EARLY_STOP_WINDOW}"
        )
        assert 0 < EARLY_STOP_DROP <= 1.0, (
            f"EARLY_STOP_DROP must be in (0, 1], got {EARLY_STOP_DROP}"
        )
        assert EARLY_STOP_MIN_EPISODES >= EARLY_STOP_WINDOW, (
            f"EARLY_STOP_MIN_EPISODES ({EARLY_STOP_MIN_EPISODES}) must be >= EARLY_STOP_WINDOW ({EARLY_STOP_WINDOW})"
        )
    assert BEST_CKPT_WINDOW > 0, f"BEST_CKPT_WINDOW must be > 0, got {BEST_CKPT_WINDOW}"
    USE_REGIME_SWITCH = True
    REGIME_DISTRIBUTION = "exponential"
    REGIME_ALPHA = 1.5
    REGIME_L_MIN = 10
    REGIME_EXP_TAU = 60
    REGIME_P_LO = 0.2
    REGIME_P_HI = 0.8
    USE_RANDOM_TAU_REGIME = True
    RANDOM_TAU_GRID = [15.0, 30.0, 60.0, 120.0, 240.0]
    RANDOM_TAU_PROBS_START = [0.05, 0.15, 0.6, 0.15, 0.05]
    RANDOM_TAU_PROBS_END = [0.15, 0.2, 0.3, 0.2, 0.15]
    RANDOM_TAU_RAMP_FRAC = 1.0
    USE_DRO_TAU = True
    DRO_TAU_MIX_PROB = 1.0
    DRO_POOL_SIZE = 256
    DRO_BETA_START = 0.0
    DRO_BETA_END = 0.5
    DRO_ETA = 0.05
    DRO_EPS_UNIFORM_START = 1.0
    DRO_EPS_UNIFORM_END = 0.5
    DRO_RAMP_FRAC = 0.5
    DRO_BETA = DRO_BETA_START
    DRO_EPS_UNIFORM = DRO_EPS_UNIFORM_START
    DRO_MAX_PROB = 0.05
    DRO_REFRESH_EVERY = 500
    DRO_REFRESH_FRAC = 0.1
    DRO_BUFFER_MO_EVENTS = 1000
    DRO_PERSISTENT_MIX_PROB = 0.5
    DRO_PERSISTENT_RHO = 0.85
    CURRICULUM_DELTA_START = 0.05
    CURRICULUM_DELTA_END = 0.3
    CURRICULUM_RAMP_FRAC = 0.7
    USE_ANCHOR_DECAY = False
    ANCHOR_LAMBDA_START = 0.05
    ANCHOR_LAMBDA_END = 0.01
    TIED_ANCHOR_BASE = 0.001
    TIED_ANCHOR_SLOPE = 0.004
    TIED_ANCHOR_GATE = 0.08
    USE_CYCLICAL_LR = True
    CYCLICAL_LR_PERIOD = 100
    CYCLICAL_LR_MIN_FACTOR = 0.3
    USE_MO_FLOW_SLOW_EWMA = False
    USE_MO_FLOW_FAST_EWMA = False
    FAST_FLOW_TAU_DIVISOR = 4.0
    USE_BAYES_FLOW_SIGNAL = True
    USE_MIXTURE = False
    BAYES_MIXTURE_TAUS = [15.0, 30.0, 60.0, 120.0, 240.0]
    USE_TRUNCATED_BETA = False
    BAYES_FLOW_PRIOR_MODE = "truncated_beta_grid" if USE_TRUNCATED_BETA else "beta"
    BAYES_FLOW_CP_WINDOW = 5
    BAYES_FLOW_FEATURE_KEYS = ["bayes_m_hat", "bayes_expected_run_length"]
    USE_AUTO_GAMMA = False
    AUTO_GAMMA_METHOD = "exact"
    USE_EMPIRICAL_MO_FRAC_CALIBRATION = True
    N_MO_FRAC_CALIB_EPISODES = 5
    FORCE_BAYES_TAU_R_FIXED = True
    USE_FLOW_EWMA = USE_MO_FLOW_SLOW_EWMA
    USE_FAST_FLOW_EWMA = USE_MO_FLOW_FAST_EWMA
    FAST_FLOW_EWMA_ALPHA = None
    PLOT_EWMA = False
    PLOT_EWMA_INTERVAL = 500
    PLOT_FILL_IMBALANCE = False
    PLOT_FILL_IMBALANCE_INTERVAL = 500
    PLOT_BAYES_METRICS = False
    PLOT_BAYES_METRICS_INTERVAL = 500
    ADV_TAU_MODE = "alpha_recalc"
    ADV_TAU_OBJECTIVE = "final_pnl"
    RECALIB_INTERVAL = 250
    _VALID_ADV_TAU_OBJECTIVES = ("final_pnl", "total_reward")
    if ADV_TAU_OBJECTIVE not in _VALID_ADV_TAU_OBJECTIVES:
        raise ValueError(
            f"ADV_TAU_OBJECTIVE must be one of {_VALID_ADV_TAU_OBJECTIVES}, got '{ADV_TAU_OBJECTIVE}'"
        )
    import math

    def _compute_mo_flow_alpha(tau: float) -> float:
        """Compute MO-flow EWMA alpha from a target memory tau on the MO clock."""
        tau_eff = max(float(tau), 1.0)
        return 1.0 - math.exp(-1.0 / tau_eff)

    def _compute_regime_matched_gamma(
        tau_mo: float, mo_frac: float, method: str = "exact"
    ) -> float:
        """
        Compute a gamma whose effective horizon matches roughly one regime.

        Parameters
        --------------------------------------------------------------
        tau_mo : float
            Mean regime duration measured in MO events.
        mo_frac : float
            Approximate fraction of environment steps that are MOs.
        method : {"exact", "approx"}
            - "approx": gamma = exp(-mo_frac / tau_mo)
            - "exact" : solves E[gamma^N] = exp(-1) for
                        N ~ NegBin(tau_mo, mo_frac), giving
                        gamma = a / (p + (1-p)a), a = exp(-1/tau_mo)
        """
        tau_eff = max(float(tau_mo), 1.0)
        p = min(max(float(mo_frac), 1e-09), 1.0)
        if str(method).lower() == "approx":
            gamma = math.exp(-p / tau_eff)
        else:
            a = math.exp(-1.0 / tau_eff)
            gamma = a / (p + (1.0 - p) * a)
        return min(max(float(gamma), 0.0), 0.999999)

    def _compute_fill_alpha(tau: float, mo_frac: float = 1.0) -> float:
        """Compute fill_imbalance_alpha on the all-event clock.

        Parameters
        --------------------------------------------------------------
        tau : float
            Desired memory in MO events (the regime clock).
        mo_frac : float
            Expected fraction of environment steps that are market orders.
        """
        tau_mo = max(float(tau), 1.0)
        mo_frac_eff = max(float(mo_frac), 1e-06)
        tau_steps = tau_mo / mo_frac_eff
        return 1.0 - math.exp(-1.0 / tau_steps)

    def _normalize_probs(probs, name: str = "probs") -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 1D probability vector")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} cannot contain negative entries: {arr}")
        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError(f"{name} must have positive sum")
        return arr / total

    def get_random_tau_probs(ep_index: int) -> np.ndarray:
        """Linear curriculum from RANDOM_TAU_PROBS_START to RANDOM_TAU_PROBS_END."""
        start = _normalize_probs(RANDOM_TAU_PROBS_START, "RANDOM_TAU_PROBS_START")
        end = _normalize_probs(RANDOM_TAU_PROBS_END, "RANDOM_TAU_PROBS_END")
        if start.size != len(RANDOM_TAU_GRID) or end.size != len(RANDOM_TAU_GRID):
            raise ValueError(
                "RANDOM_TAU probability vectors must match RANDOM_TAU_GRID length"
            )
        denom = max(1.0, float(N_EPISODES) * float(RANDOM_TAU_RAMP_FRAC) - 1.0)
        progress = min(1.0, max(0.0, float(ep_index) / denom))
        probs = (1.0 - progress) * start + progress * end
        return _normalize_probs(probs, "random_tau_curriculum_probs")

    def get_dro_tau_params(ep_index: int) -> tuple[float, float]:
        """Ramp DRO tilt from uniform sampling to soft hard-path reweighting."""
        denom = max(1.0, float(N_EPISODES) * float(DRO_RAMP_FRAC) - 1.0)
        progress = min(1.0, max(0.0, float(ep_index) / denom))
        beta = (1.0 - progress) * float(DRO_BETA_START) + progress * float(DRO_BETA_END)
        eps_uniform = (1.0 - progress) * float(
            DRO_EPS_UNIFORM_START
        ) + progress * float(DRO_EPS_UNIFORM_END)
        return (float(beta), float(eps_uniform))

    def make_random_tau_regime_schedule(
        seed: int, n_mo_events: int, tau_grid, tau_probs, p_lo: float, p_hi: float
    ):
        """
        Build a regime schedule that draws a fresh exponential tau per regime.

        The returned callable maps MO-event index -> p_buy and exposes metadata
        for TensorBoard: boundaries, p_values, taus, tau_probs, and tau_grid.
        """
        rng = np.random.default_rng(int(seed))
        tau_grid_arr = np.asarray(tau_grid, dtype=np.float64)
        tau_probs_arr = _normalize_probs(tau_probs, "tau_probs")
        if tau_grid_arr.ndim != 1 or tau_grid_arr.size != tau_probs_arr.size:
            raise ValueError("tau_grid and tau_probs must be same-length 1D arrays")
        if np.any(tau_grid_arr <= 0.0):
            raise ValueError(f"tau_grid must contain positive taus: {tau_grid_arr}")
        boundaries = [0]
        p_values = []
        taus = []
        tau_indices = []
        boundaries_arr = np.asarray(boundaries, dtype=np.int64)
        p_values_arr = np.asarray(p_values, dtype=np.float64)

        def _append_regime() -> None:
            nonlocal boundaries_arr, p_values_arr
            tau_idx = int(rng.choice(len(tau_grid_arr), p=tau_probs_arr))
            tau = float(tau_grid_arr[tau_idx])
            length = max(1, int(np.ceil(rng.exponential(tau))))
            boundaries.append(int(boundaries[-1] + length))
            p_values.append(float(rng.uniform(p_lo, p_hi)))
            taus.append(tau)
            tau_indices.append(tau_idx)
            boundaries_arr = np.asarray(boundaries, dtype=np.int64)
            p_values_arr = np.asarray(p_values, dtype=np.float64)

        while boundaries[-1] < int(n_mo_events) + 1000:
            _append_regime()

        def schedule(step_idx: int) -> float:
            nonlocal boundaries_arr, p_values_arr
            step_idx = int(step_idx)
            while step_idx >= boundaries[-1]:
                _append_regime()
            regime_idx = (
                int(np.searchsorted(boundaries_arr, step_idx, side="right")) - 1
            )
            regime_idx = min(regime_idx, len(p_values_arr) - 1)
            return float(p_values_arr[regime_idx])

        schedule.boundaries = boundaries
        schedule.p_values = p_values
        schedule.taus = taus
        schedule.tau_indices = tau_indices
        schedule.tau_grid = tau_grid_arr.tolist()
        schedule.tau_probs = tau_probs_arr.tolist()
        return schedule

    FILL_IMBALANCE_ALPHA = None
    FAST_FLOW_EWMA_ALPHA = None
    if REGIME_DISTRIBUTION == "exponential" and USE_REGIME_SWITCH:
        EWMA_ALPHA = _compute_mo_flow_alpha(REGIME_EXP_TAU)
        _tau_for_fill_alpha = REGIME_EXP_TAU
        print(
            f"[EWMA] mo_flow alpha derived from tau={REGIME_EXP_TAU}: alpha={EWMA_ALPHA:.6f}"
        )
    elif REGIME_DISTRIBUTION == "pareto" and USE_REGIME_SWITCH:
        _pareto_median = REGIME_L_MIN * 2.0 ** (1.0 / REGIME_ALPHA)
        EWMA_ALPHA = _compute_mo_flow_alpha(_pareto_median)
        _tau_for_fill_alpha = _pareto_median
        print(
            f"[EWMA] Pareto (alpha={REGIME_ALPHA}, L_min={REGIME_L_MIN}): median={_pareto_median:.1f} MOs"
        )
        print(f"[EWMA] mo_flow alpha (from median): alpha={EWMA_ALPHA:.6f}")
    else:
        EWMA_ALPHA = 0.1
        _tau_for_fill_alpha = None
    _tau_fast_base = 20.0 if _tau_for_fill_alpha is None else float(_tau_for_fill_alpha)
    _tau_fast = max(_tau_fast_base / max(float(FAST_FLOW_TAU_DIVISOR), 1.0), 1.0)
    FAST_FLOW_EWMA_ALPHA = _compute_mo_flow_alpha(_tau_fast)
    BAYES_FLOW_TAU_R = (
        float(_tau_for_fill_alpha) if _tau_for_fill_alpha is not None else 60.0
    )

    def _count_mo_events_from_msg(msg_df):
        """Return (n_mo_events, n_logged_events) from a simulator message tape."""
        if msg_df is None or "Type" not in msg_df.columns:
            return (0, 0)
        type_col = msg_df["Type"]
        total_events = int(len(type_col))
        if total_events <= 0:
            return (0, 0)
        if np.issubdtype(type_col.dtype, np.number):
            n_mo = int((type_col.to_numpy() == 1).sum())
        else:
            n_mo = int(type_col.astype(str).str.upper().eq("MO").sum())
        return (n_mo, total_events)

    def _calibrate_mo_event_fraction(n_episodes: int):
        """
        Empirically estimate p_MO = P(env event is a market order).

        The calibration runs the same LOB engine path used by training, but with
        no controller/policy, so it measures the environment event clock before
        MM training begins.
        """
        n_episodes = max(int(n_episodes), 0)
        if n_episodes <= 0:
            return None
        print(f"\n{'=' * 60}")
        print(f"MO EVENT FRACTION CALIBRATION -- {n_episodes} offline episodes (no MM)")
        print(f"{'=' * 60}")
        total_mo = 0
        total_events = 0
        for calib_ep in range(n_episodes):
            calib_seed = GLOBAL_SEED + 750000 + calib_ep
            calib_msg, _, _ = simulate_LOB_with_MM(
                lam=lam,
                mu=mu,
                delta=delta,
                number_tick_levels=NUMBER_TICK_LEVELS,
                n_priority_ranks=N_PRIORITY_RANKS,
                number_levels_to_store=20,
                p0=100,
                mean_size_LO=mean_size_LO,
                mean_size_MO=mean_size_MO,
                iterations=EPISODE_LENGTH,
                iterations_to_equilibrium=ITER_TO_EQUILIBRIUM,
                path_save_files=None,
                label_simulation=None,
                beta_exp_weighted_return=0.0,
                intensity_exp_weighted_return=0.0,
                controller=None,
                mm_policy=None,
                exclude_self_from_state=False,
                reward_fn=None,
                random_seed=calib_seed,
                buy_mo_prob=0.5,
                qrm_params=qrm_params,
            )
            n_mo, n_events = _count_mo_events_from_msg(calib_msg)
            total_mo += n_mo
            total_events += n_events
            frac = n_mo / max(n_events, 1)
            print(
                f"  [MO_CALIB] ep {calib_ep + 1}: {n_mo}/{n_events} MO events  p_MO={frac:.6f}"
            )
        if total_events <= 0:
            print("[MO_CALIB] WARNING: no events observed; keeping analytic p_MO.")
            print(f"{'=' * 60}\n")
            return None
        p_mo = total_mo / total_events
        print(f"[MO_CALIB] p_MO empirical = {total_mo}/{total_events} = {p_mo:.6f}")
        print(f"{'=' * 60}\n")
        return (float(p_mo), int(total_mo), int(total_events))

    _Lam = lam * NUMBER_TICK_LEVELS
    _Mu = 2.0 * mu
    _Delta_approx = delta * 100.0
    _MO_EVENT_FRACTION_ANALYTIC = _Mu / max(_Lam + _Mu + _Delta_approx, 1e-12)
    _MO_EVENT_FRACTION = float(_MO_EVENT_FRACTION_ANALYTIC)
    _MO_EVENT_FRACTION_SOURCE = "analytic"
    if USE_EMPIRICAL_MO_FRAC_CALIBRATION:
        _mo_frac_calib = _calibrate_mo_event_fraction(N_MO_FRAC_CALIB_EPISODES)
        if _mo_frac_calib is not None:
            _MO_EVENT_FRACTION, _mo_calib_count, _event_calib_count = _mo_frac_calib
            _MO_EVENT_FRACTION_SOURCE = "empirical"
    EXPECTED_MO_PER_EPISODE = int(round(EPISODE_LENGTH * _MO_EVENT_FRACTION))
    print(
        f"[MO_CLOCK] p_MO analytic={_MO_EVENT_FRACTION_ANALYTIC:.6f}  active={_MO_EVENT_FRACTION:.6f} ({_MO_EVENT_FRACTION_SOURCE})  expected_MO/episode~={EXPECTED_MO_PER_EPISODE}"
    )
    _AUTO_GAMMA_VALUE = None
    _AUTO_GAMMA_H_STEPS = None
    if USE_AUTO_GAMMA:
        _gamma_tau = 20.0 if _tau_for_fill_alpha is None else float(_tau_for_fill_alpha)
        _AUTO_GAMMA_VALUE = _compute_regime_matched_gamma(
            tau_mo=_gamma_tau, mo_frac=_MO_EVENT_FRACTION, method=AUTO_GAMMA_METHOD
        )
        _AUTO_GAMMA_H_STEPS = _gamma_tau / max(_MO_EVENT_FRACTION, 1e-09)
        best_params["gamma"] = float(_AUTO_GAMMA_VALUE)
        print(
            f"[AUTO_GAMMA] method={AUTO_GAMMA_METHOD}  tau_mo={_gamma_tau:.1f}  mo_frac={_MO_EVENT_FRACTION:.6f}  H_steps~={_AUTO_GAMMA_H_STEPS:.1f}  gamma={_AUTO_GAMMA_VALUE:.6f}"
        )
    USE_INVENTORY_WALL = True
    USE_TERMINAL_BONUS = False
    USE_FILL_IMBALANCE = True
    USE_QUOTE_EXPOSURE_IMBALANCE = True
    USE_G_CLIP = True
    USE_PER_PRIORITY_CLIP = True
    _inv_limit_val = best_params.get("inv_limit", 8)
    _reward_kwargs = dict(inv_penalty_coeff=INV_PENALTY_COEFF)
    if USE_INVENTORY_WALL:
        _reward_kwargs["use_inv_wall"] = True
        _reward_kwargs["inv_wall_threshold"] = _inv_limit_val / 2.0
        print(
            f"[REWARD] Inventory wall: threshold={_reward_kwargs['inv_wall_threshold']}, coeff=phi={INV_PENALTY_COEFF}"
        )
    if USE_TERMINAL_BONUS:
        _reward_kwargs["use_terminal_bonus"] = True
        print("[REWARD] Terminal PnL bonus: ON (PnL in ticks, no attenuation)")
    if USE_DAMPENED_REWARD:
        reward_fn_spread = partial(reward_pnl_dampened_inv_quadratic, **_reward_kwargs)
        print("[REWARD] Using reward_pnl_dampened_inv_quadratic (dampened PnL)")
    else:
        reward_fn_spread = partial(
            reward_spread_capture_inv_quadratic, **_reward_kwargs
        )
        print("[REWARD] Using reward_spread_capture_inv_quadratic (spread capture)")
    ADV_FREEZE_MM = 100
    ADV_FREEZE_ADV = 15
    ADV_WARMUP_EPISODES = 200
    if USE_RANDOM_TAU_REGIME:
        if not USE_REGIME_SWITCH or REGIME_DISTRIBUTION != "exponential":
            raise ValueError(
                "USE_RANDOM_TAU_REGIME requires exponential USE_REGIME_SWITCH"
            )
        assert (
            len(RANDOM_TAU_GRID)
            == len(RANDOM_TAU_PROBS_START)
            == len(RANDOM_TAU_PROBS_END)
        ), (
            "RANDOM_TAU_GRID, RANDOM_TAU_PROBS_START, and RANDOM_TAU_PROBS_END must have the same length"
        )
        assert 0 < RANDOM_TAU_RAMP_FRAC <= 1.0, (
            f"RANDOM_TAU_RAMP_FRAC must be in (0, 1], got {RANDOM_TAU_RAMP_FRAC}"
        )
    if USE_DRO_TAU:
        if not USE_RANDOM_TAU_REGIME:
            raise ValueError("USE_DRO_TAU requires USE_RANDOM_TAU_REGIME=True")
        assert 0.0 <= DRO_TAU_MIX_PROB <= 1.0, (
            f"DRO_TAU_MIX_PROB must be in [0, 1], got {DRO_TAU_MIX_PROB}"
        )
        assert DRO_POOL_SIZE > 0, f"DRO_POOL_SIZE must be positive, got {DRO_POOL_SIZE}"
        assert DRO_REFRESH_FRAC >= 0.0, (
            f"DRO_REFRESH_FRAC must be >= 0, got {DRO_REFRESH_FRAC}"
        )
        assert 0.0 < DRO_RAMP_FRAC <= 1.0, (
            f"DRO_RAMP_FRAC must be in (0, 1], got {DRO_RAMP_FRAC}"
        )
        assert 0.0 <= DRO_EPS_UNIFORM_START <= 1.0, (
            f"DRO_EPS_UNIFORM_START must be in [0, 1], got {DRO_EPS_UNIFORM_START}"
        )
        assert 0.0 <= DRO_EPS_UNIFORM_END <= 1.0, (
            f"DRO_EPS_UNIFORM_END must be in [0, 1], got {DRO_EPS_UNIFORM_END}"
        )
        assert DRO_BETA_START >= 0.0 and DRO_BETA_END >= 0.0, (
            "DRO beta ramp endpoints must be non-negative"
        )
        assert 0.0 <= DRO_PERSISTENT_MIX_PROB <= 1.0, (
            f"DRO_PERSISTENT_MIX_PROB must be in [0, 1], got {DRO_PERSISTENT_MIX_PROB}"
        )
        assert 0.0 <= DRO_PERSISTENT_RHO <= 1.0, (
            f"DRO_PERSISTENT_RHO must be in [0, 1], got {DRO_PERSISTENT_RHO}"
        )
        if DRO_MAX_PROB is not None:
            assert 1.0 / DRO_POOL_SIZE <= DRO_MAX_PROB <= 1.0, (
                f"DRO_MAX_PROB must be in [1/pool_size, 1], got {DRO_MAX_PROB}"
            )

    def _active_fill_imbalance_feature_tag() -> str:
        if not bool(USE_FILL_IMBALANCE):
            return "none"
        if bool(USE_QUOTE_EXPOSURE_IMBALANCE):
            return "quote_exposure"
        return "ewma"

    def _checkpoint_feature_tag() -> str:
        bayes_tag = (
            f"bayes_on{len(BAYES_FLOW_FEATURE_KEYS)}"
            if USE_BAYES_FLOW_SIGNAL
            else "bayes_off"
        )
        mix_tag = "mix_on" if USE_BAYES_FLOW_SIGNAL and USE_MIXTURE else "mix_off"
        prior_tag = (
            f"_truncated_{('on' if USE_TRUNCATED_BETA else 'off')}"
            if USE_BAYES_FLOW_SIGNAL
            else ""
        )
        return f"fill_{_active_fill_imbalance_feature_tag()}_{bayes_tag}_{mix_tag}{prior_tag}"

    def _build_train_suffix() -> str:

        def _fmt_suffix_float(value: float) -> str:
            text = f"{float(value):.6f}".rstrip("0").rstrip(".")
            return text.replace("-", "m").replace(".", "p")

        _n_act = (
            len(PURE_MM_OFFSETS) if USE_PURE_MM else best_params.get("n_actions", 6)
        )
        _gamma_tag = _fmt_suffix_float(best_params.get("gamma", 0.0))
        _n_steps = int(best_params.get("n_steps", 1))
        _n_hidden = int(best_params.get("n_hidden", 0))
        _n_neurons = int(best_params.get("n_neurons", 0))
        suffix = (
            f"_g{_gamma_tag}_nstep{_n_steps}_h{_n_hidden}x{_n_neurons}_a{int(_n_act)}"
        )
        _dropout = float(best_params.get("dropout_level", 0.0))
        if _dropout > 0.0:
            suffix += f"_do{_fmt_suffix_float(_dropout)}"
        if USE_REGIME_SWITCH:
            suffix += f"_regime_{REGIME_DISTRIBUTION}"
        if USE_RANDOM_TAU_REGIME:
            suffix += "_random_tau_curriculum"
        if USE_DRO_TAU:
            _dro_cap_tag = (
                "capnone"
                if DRO_MAX_PROB is None
                else f"cap{_fmt_suffix_float(DRO_MAX_PROB)}"
            )
            _dro_persist_tag = ""
            if DRO_PERSISTENT_MIX_PROB > 0.0:
                _dro_persist_tag = f"_pmix{_fmt_suffix_float(DRO_PERSISTENT_MIX_PROB)}_rho{_fmt_suffix_float(DRO_PERSISTENT_RHO)}"
            suffix += f"_dro_tau_p{int(DRO_POOL_SIZE)}_b{_fmt_suffix_float(DRO_BETA_END)}_eps{_fmt_suffix_float(DRO_EPS_UNIFORM_END)}_{_dro_cap_tag}_ramp{_fmt_suffix_float(DRO_RAMP_FRAC)}{_dro_persist_tag}"
        if USE_FLOW_EWMA:
            suffix += "_with_ewma_mo"
        if USE_FACTORED_NOISE:
            suffix += "_factored_noise"
        if USE_FULLY_NOISY:
            suffix += "_fully_noisy"
        if USE_INVENTORY_WALL:
            suffix += "_inv_wall"
        if USE_DAMPENED_REWARD:
            suffix += "_dampened_reward"
        if WARMSTART_CKPT is not None:
            suffix += "_warmstart"
        if USE_FINE_TUNING_TECHNIQUES and WARMSTART_CKPT is not None:
            if ANCHOR_LAMBDA > 0:
                suffix += f"_ft_anchor{_fmt_suffix_float(ANCHOR_LAMBDA)}"
            if LR_FEATURE_FACTOR != 1.0:
                suffix += "_ft_difflr"
            _replay_path = WARMSTART_CKPT.replace(".pt", "_replay.pt")
            if REPLAY_RESTORE_FRACTION > 0 and os.path.isfile(_replay_path):
                suffix += "_ft_replay"
        if USE_CYCLICAL_LR:
            suffix += "_cyc_lr"
        suffix += f"_{_checkpoint_feature_tag()}"
        return suffix

    _mode_suffix = _build_train_suffix()
    MODE_DIR = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_mode_suffix}"
    if USE_DRO_TAU:

        def _compact_float_tag(value: float) -> str:
            text = f"{float(value):.6f}".rstrip("0").rstrip(".")
            return text.replace("-", "m").replace(".", "p")

        _mix_tag = "mixon" if USE_BAYES_FLOW_SIGNAL and USE_MIXTURE else "mixoff"
        _tau_tag = (
            f"taufix{_compact_float_tag(BAYES_FLOW_TAU_R)}"
            if FORCE_BAYES_TAU_R_FIXED
            else "taufree"
        )
        _cap_tag = (
            "capnone"
            if DRO_MAX_PROB is None
            else f"cap{_compact_float_tag(DRO_MAX_PROB)}"
        )
        _persist_tag = ""
        if DRO_PERSISTENT_MIX_PROB > 0.0:
            _persist_tag = f"_pmix{_compact_float_tag(DRO_PERSISTENT_MIX_PROB)}_rho{_compact_float_tag(DRO_PERSISTENT_RHO)}"
        MODE_DIR = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}_drosoft_p{DRO_POOL_SIZE}_b{_compact_float_tag(DRO_BETA_END)}_eps{_compact_float_tag(DRO_EPS_UNIFORM_END)}_{_cap_tag}_ramp{_compact_float_tag(DRO_RAMP_FRAC)}{_persist_tag}_{_tau_tag}_{_mix_tag}"
    try:
        BASE = Path(__file__).resolve().parent
    except NameError:
        BASE = Path.cwd()
    RUN_ROOT = BASE / "runs_spyder" / MODE_DIR
    print("[TB] RUN_ROOT abs:", RUN_ROOT)
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT, ignore_errors=False)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    pure_mm_offsets = None
    if not USE_PURE_MM:
        deep_controller = DeepRLController(
            level_offset=0,
            n_actions=best_params["n_actions"],
            gamma=best_params["gamma"],
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0.0),
            epsilon_start=best_params["epsilon_start"],
            epsilon_min=best_params["epsilon_min"],
            epsilon_decay=best_params["epsilon_decay"],
            batch_size=best_params["batch_size"],
            replay_capacity=best_params["replay_capacity"],
            target_update_steps=best_params["target_update_steps"],
            use_sarsa=best_params["use_sarsa"],
            use_double=best_params["use_double"],
            use_dueling=best_params["use_dueling"],
            use_prioritized_experience=best_params["use_per"],
            use_noisy_net=best_params["use_noisy_net"],
            use_factored_noise=USE_FACTORED_NOISE,
            use_fully_noisy=USE_FULLY_NOISY,
            use_distributional=best_params.get("use_distributional", False),
            v_min=best_params.get("dist_v_min", -10.0),
            v_max=best_params.get("dist_v_max", 10.0),
            atoms=best_params.get("dist_atoms", 51),
            n_neurons=best_params["n_neurons"],
            n_hidden=best_params["n_hidden"],
            activation=best_params.get("activation", "relu"),
            elu_alpha=best_params.get("elu_alpha", None),
            dropout_level=best_params.get("dropout_level", 0.0),
            per_alpha_start=best_params.get("per_alpha_start", 0.5386),
            per_alpha_end=best_params.get("per_alpha_end", 0.375),
            per_alpha_last_episode=N_EPISODES,
            per_beta_start=best_params.get("per_beta_start", 0.3043),
            per_beta_end=best_params.get("per_beta_end", 0.5481),
            per_beta_last_episode=N_EPISODES,
            n_steps=best_params["n_steps"],
            log_dir=str(RUN_ROOT),
            use_tob_update=bool(best_params.get("use_tob_update", False)),
            n_tob_moves=int(best_params.get("n_tob_moves", 10)),
            use_event_update=bool(best_params.get("use_event_update", False)),
            n_events=int(best_params.get("n_events", 100)),
            use_time_update=bool(best_params.get("use_time_update", False)),
            min_time_interval=float(best_params.get("min_time_interval", 1.0)),
            use_mdp=True,
            use_flow_signal=USE_FLOW_EWMA,
            use_fast_flow_signal=USE_FAST_FLOW_EWMA,
            use_bayes_flow_signal=USE_BAYES_FLOW_SIGNAL,
            bayes_flow_feature_keys=BAYES_FLOW_FEATURE_KEYS,
            use_fill_imbalance=USE_FILL_IMBALANCE,
            use_g_clip=USE_G_CLIP,
            use_per_priority_clip=USE_PER_PRIORITY_CLIP,
            pure_mm=False,
            inv_limit=best_params.get("inv_limit", None),
            pure_mm_offsets=None,
        )
    else:
        pure_mm_offsets = list(PURE_MM_OFFSETS)
        deep_controller = DeepRLController(
            level_offset=0,
            n_actions=len(pure_mm_offsets),
            gamma=best_params["gamma"],
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0.0),
            epsilon_start=best_params["epsilon_start"],
            epsilon_min=best_params["epsilon_min"],
            epsilon_decay=best_params["epsilon_decay"],
            batch_size=best_params["batch_size"],
            replay_capacity=best_params["replay_capacity"],
            target_update_steps=best_params["target_update_steps"],
            use_sarsa=best_params["use_sarsa"],
            use_double=best_params["use_double"],
            use_dueling=best_params["use_dueling"],
            use_prioritized_experience=best_params["use_per"],
            use_noisy_net=best_params["use_noisy_net"],
            use_factored_noise=USE_FACTORED_NOISE,
            use_fully_noisy=USE_FULLY_NOISY,
            use_distributional=best_params.get("use_distributional", False),
            v_min=best_params.get("dist_v_min", -10.0),
            v_max=best_params.get("dist_v_max", 10.0),
            atoms=best_params.get("dist_atoms", 51),
            n_neurons=best_params["n_neurons"],
            n_hidden=best_params["n_hidden"],
            activation=best_params.get("activation", "relu"),
            elu_alpha=best_params.get("elu_alpha", None),
            dropout_level=best_params.get("dropout_level", 0.0),
            per_alpha_start=best_params.get("per_alpha_start", 0.5386),
            per_alpha_end=best_params.get("per_alpha_end", 0.375),
            per_alpha_last_episode=N_EPISODES,
            per_beta_start=best_params.get("per_beta_start", 0.3043),
            per_beta_end=best_params.get("per_beta_end", 0.5481),
            per_beta_last_episode=N_EPISODES,
            n_steps=best_params["n_steps"],
            log_dir=str(RUN_ROOT),
            use_tob_update=bool(best_params.get("use_tob_update", False)),
            n_tob_moves=int(best_params.get("n_tob_moves", 10)),
            use_event_update=bool(best_params.get("use_event_update", False)),
            n_events=int(best_params.get("n_events", 100)),
            use_time_update=bool(best_params.get("use_time_update", False)),
            min_time_interval=float(best_params.get("min_time_interval", 1.0)),
            use_mdp=True,
            use_flow_signal=USE_FLOW_EWMA,
            use_fast_flow_signal=USE_FAST_FLOW_EWMA,
            use_bayes_flow_signal=USE_BAYES_FLOW_SIGNAL,
            bayes_flow_feature_keys=BAYES_FLOW_FEATURE_KEYS,
            use_fill_imbalance=USE_FILL_IMBALANCE,
            use_g_clip=USE_G_CLIP,
            use_per_priority_clip=USE_PER_PRIORITY_CLIP,
            pure_mm=True,
            inv_limit=best_params.get("inv_limit", None),
            pure_mm_offsets=pure_mm_offsets,
        )
    if WARMSTART_CKPT is not None and (not os.path.isfile(WARMSTART_CKPT)):
        raise FileNotFoundError(
            f"[WARMSTART] Checkpoint not found: {WARMSTART_CKPT}\nFix the path or set WARMSTART_CKPT = None to train from scratch."
        )
    if WARMSTART_CKPT is not None and os.path.isfile(WARMSTART_CKPT):
        _ws = torch.load(WARMSTART_CKPT, map_location="cpu", weights_only=True)
        _ws_raw_keys = {
            net_key: list(_ws[net_key].keys())
            for net_key in ("q_net", "target_net")
            if net_key in _ws
        }

        def _first_feature_weight_key(state_dict):
            for _cand in ("feature.0.weight", "feature.0.w_mu"):
                if _cand in state_dict:
                    return _cand
            raise KeyError(
                "Could not locate first feature-layer weight key in state dict. Expected one of: feature.0.weight, feature.0.w_mu"
            )

        def _adapt_linear_noisy_schema(state_dict, model_state_dict):
            """
            Rename/drop linear-vs-noisy keys so the checkpoint schema matches the
            current model schema, including the feature trunk when fully noisy.
            """
            _model_keys = set(model_state_dict.keys())
            _renamed = 0
            _dropped = 0
            for _k in list(state_dict.keys()):
                if _k.endswith(".weight"):
                    _noisy_k = _k.replace(".weight", ".w_mu")
                    if _noisy_k in _model_keys and _k not in _model_keys:
                        state_dict[_noisy_k] = state_dict.pop(_k)
                        _renamed += 1
                elif _k.endswith(".bias"):
                    _noisy_k = _k.replace(".bias", ".b_mu")
                    if _noisy_k in _model_keys and _k not in _model_keys:
                        state_dict[_noisy_k] = state_dict.pop(_k)
                        _renamed += 1
                elif _k.endswith(".w_mu"):
                    _linear_k = _k.replace(".w_mu", ".weight")
                    if _linear_k in _model_keys and _k not in _model_keys:
                        state_dict[_linear_k] = state_dict.pop(_k)
                        _renamed += 1
                elif _k.endswith(".b_mu"):
                    _linear_k = _k.replace(".b_mu", ".bias")
                    if _linear_k in _model_keys and _k not in _model_keys:
                        state_dict[_linear_k] = state_dict.pop(_k)
                        _renamed += 1
                elif _k.endswith(".w_sigma"):
                    _linear_k = _k.replace(".w_sigma", ".weight")
                    if _k not in _model_keys and _linear_k in _model_keys:
                        del state_dict[_k]
                        _dropped += 1
                elif _k.endswith(".b_sigma"):
                    _linear_k = _k.replace(".b_sigma", ".bias")
                    if _k not in _model_keys and _linear_k in _model_keys:
                        del state_dict[_k]
                        _dropped += 1
            return (_renamed, _dropped)

        _ws_input_key = _first_feature_weight_key(_ws["q_net"])
        _cur_input_key = _first_feature_weight_key(deep_controller.q_net.state_dict())
        _ws_input_dim = _ws["q_net"][_ws_input_key].shape[1]
        _cur_input_dim = deep_controller.q_net.state_dict()[_cur_input_key].shape[1]
        _input_pad_info = None
        if _ws_input_dim != _cur_input_dim:
            print(
                f"[WARMSTART] Input dim mismatch: checkpoint={_ws_input_dim}, model={_cur_input_dim}. Padding first layer with zeros."
            )
            for net_key in ("q_net", "target_net"):
                if net_key not in _ws:
                    continue
                sd = _ws[net_key]
                _sd_input_key = _first_feature_weight_key(sd)
                w = sd[_sd_input_key]
                pad = torch.zeros(w.shape[0], _cur_input_dim - _ws_input_dim)
                sd[_sd_input_key] = torch.cat([w, pad], dim=1)
            _input_pad_info = (_ws_input_dim, _cur_input_dim)
        _schema_renamed = 0
        _schema_dropped = 0
        _schema_exact = True
        for net_key, model_net in (
            ("q_net", deep_controller.q_net),
            ("target_net", deep_controller.target_net),
        ):
            if net_key not in _ws or model_net is None:
                continue
            _renamed, _dropped = _adapt_linear_noisy_schema(
                _ws[net_key], model_net.state_dict()
            )
            _schema_renamed += _renamed
            _schema_dropped += _dropped
            if set(_ws[net_key].keys()) != set(model_net.state_dict().keys()):
                _schema_exact = False
        if _schema_renamed or _schema_dropped:
            print(
                f"[WARMSTART] Adapted Linear/Noisy schema to current model ({_schema_renamed} renamed, {_schema_dropped} sigma dropped)"
            )
        _arch_changed = _ws_input_dim != _cur_input_dim or not _schema_exact
        _strict = not _arch_changed
        deep_controller.q_net.load_state_dict(_ws["q_net"], strict=_strict)
        if "target_net" in _ws and deep_controller.target_net is not None:
            deep_controller.target_net.load_state_dict(
                _ws["target_net"], strict=_strict
            )
        if _arch_changed:
            print(
                "[WARMSTART] Skipping optimizer state (architecture changed, Adam moments incompatible)."
            )
        elif "optimizer" in _ws and deep_controller.optimizer is not None:
            deep_controller.optimizer.load_state_dict(_ws["optimizer"])
        print(f"[WARMSTART] Loaded weights from: {WARMSTART_CKPT}")
        if best_params["use_noisy_net"]:
            _missing_sigma_by_net = {}
            _sigma_expected = 0
            _sigma_present = 0
            for net_key, model_net in (
                ("q_net", deep_controller.q_net),
                ("target_net", deep_controller.target_net),
            ):
                if model_net is None:
                    continue
                _model_sigma_keys = {
                    k
                    for k in model_net.state_dict().keys()
                    if ".w_sigma" in k or ".b_sigma" in k
                }
                _raw_keys = set(_ws_raw_keys.get(net_key, []))
                _missing = _model_sigma_keys - _raw_keys
                _missing_sigma_by_net[net_key] = _missing
                _sigma_expected += len(_model_sigma_keys)
                _sigma_present += len(_model_sigma_keys) - len(_missing)
            _all_sigma_covered = (
                _sigma_expected > 0 and _sigma_present == _sigma_expected
            )
            if _all_sigma_covered and (not RESET_SIGMAS):
                print(
                    f"[WARMSTART] NoisyNet sigmas loaded from checkpoint (pre-calibrated)"
                )
            elif RESET_SIGMAS:
                import math as _math

                _sigma_init = 0.5
                _reset_count = 0
                for net in (deep_controller.q_net, deep_controller.target_net):
                    if net is None:
                        continue
                    for name, param in net.named_parameters():
                        if "w_sigma" in name:
                            fan_in = (
                                param.shape[1] if param.dim() >= 2 else param.shape[0]
                            )
                            with torch.no_grad():
                                torch.nn.init.kaiming_uniform_(param, a=_math.sqrt(5))
                                param.mul_(_sigma_init)
                            _reset_count += 1
                        elif "b_sigma" in name:
                            fan_in = param.shape[0]
                            with torch.no_grad():
                                torch.nn.init.constant_(
                                    param, _sigma_init / _math.sqrt(max(1, fan_in))
                                )
                            _reset_count += 1
                print(
                    f"[WARMSTART] Reset {_reset_count} NoisyNet sigma params to sigma_init={_sigma_init}"
                )
            else:
                _SIGMA_FRACTION = 0.05
                _reset_count = 0
                for net_key, net in (
                    ("q_net", deep_controller.q_net),
                    ("target_net", deep_controller.target_net),
                ):
                    if net is None:
                        continue
                    _missing_sigma_keys = _missing_sigma_by_net.get(net_key, set())
                    for name, param in net.named_parameters():
                        if name not in _missing_sigma_keys:
                            continue
                        if "w_sigma" in name:
                            mu_name = name.replace("w_sigma", "w_mu")
                            mu_abs_mean = 0.1
                            for n2, p2 in net.named_parameters():
                                if n2 == mu_name:
                                    mu_abs_mean = p2.abs().mean().item()
                                    break
                            _target_sigma = max(mu_abs_mean * _SIGMA_FRACTION, 0.01)
                            with torch.no_grad():
                                param.fill_(_target_sigma)
                            _reset_count += 1
                        elif "b_sigma" in name:
                            mu_name = name.replace("b_sigma", "b_mu")
                            mu_abs_mean = 0.1
                            for n2, p2 in net.named_parameters():
                                if n2 == mu_name:
                                    mu_abs_mean = p2.abs().mean().item()
                                    break
                            _target_sigma = max(mu_abs_mean * _SIGMA_FRACTION, 0.005)
                            with torch.no_grad():
                                param.fill_(_target_sigma)
                            _reset_count += 1
                if _sigma_present > 0:
                    print(
                        f"[WARMSTART] NoisyNet sigmas partially loaded from checkpoint; calibrated {_reset_count} missing sigma params to {_SIGMA_FRACTION:.0%} of |w_mu| ({_sigma_present}/{_sigma_expected} already present)"
                    )
                else:
                    print(
                        f"[WARMSTART] NoisyNet sigmas calibrated to {_SIGMA_FRACTION:.0%} of |w_mu| ({_reset_count} params, checkpoint had no sigmas)"
                    )
            from dqn_distributional_with_throttle import NoisyLinear

            print(f"[WARMSTART] NoisyNet layer diagnostics (q_net):")
            for name, module in deep_controller.q_net.named_modules():
                if isinstance(module, NoisyLinear):
                    _mu_abs = module.w_mu.abs().mean().item()
                    _sigma_abs = module.w_sigma.abs().mean().item()
                    _ratio = _sigma_abs / max(_mu_abs, 1e-08)
                    print(
                        f"  {name:20s}  mean|w_mu|={_mu_abs:.4f}  mean|w_sigma|={_sigma_abs:.4f}  sigma/mu ratio={_ratio:.1%}"
                    )
        else:
            print(
                f"[WARMSTART] NoisyNet disabled for fine-tuning -> using epsilon-greedy={best_params['epsilon_start']}"
            )
        del _ws
        if USE_FINE_TUNING_TECHNIQUES and WARMSTART_CKPT is not None:
            deep_controller.set_anchor_weights(lambda_anchor=ANCHOR_LAMBDA)
            print(f"[FINE-TUNE] L2 anchor set (lambda={ANCHOR_LAMBDA})")
            if _input_pad_info is not None:
                _old_dim, _new_dim = _input_pad_info
                _mask_dict = {}
                for _name, _ref in list(deep_controller._anchor_weights.items()):
                    if not _name.startswith("feature.0."):
                        continue
                    if _ref.dim() != 2 or _ref.shape[1] != _new_dim:
                        continue
                    _m = torch.ones_like(_ref)
                    _m[:, _old_dim:] = 0.0
                    _mask_dict[_name] = _m
                if _mask_dict:
                    deep_controller.set_anchor_masks(_mask_dict)
                    print(
                        f"[FINE-TUNE] Masked anchor applied to {sorted(_mask_dict.keys())} (columns {_old_dim}:{_new_dim} excluded)"
                    )
            _lr_feature = LR_START * LR_FEATURE_FACTOR
            _lr_head = LR_START
            deep_controller.setup_differential_lr(_lr_feature, _lr_head)
            print(
                f"[FINE-TUNE] Differential LR: feature={_lr_feature:.6f}, head={_lr_head:.6f}"
            )
            _replay_path = WARMSTART_CKPT.replace(".pt", "_replay.pt")
            if os.path.isfile(_replay_path):
                _max_load = int(
                    best_params.get("replay_capacity", 150000) * REPLAY_RESTORE_FRACTION
                )
                _loaded = deep_controller.load_replay_buffer(
                    _replay_path, max_load=_max_load, reset_priorities=True
                )
                print(
                    f"[FINE-TUNE] Loaded {_loaded} transitions from {_replay_path} ({REPLAY_RESTORE_FRACTION:.0%} of capacity)"
                )
            else:
                print(f"[FINE-TUNE] No replay buffer found at {_replay_path}")
    if NOISY_WEIGHT_DECAY > 0 and best_params.get("use_noisy_net", False):
        deep_controller.setup_noisy_weight_decay(sigma_weight_decay=NOISY_WEIGHT_DECAY)
    dro_tau_adversary = None
    if USE_DRO_TAU:
        dro_tau_adversary = DROHardPathAdversary(
            pool_size=DRO_POOL_SIZE,
            n_mo_events=EXPECTED_MO_PER_EPISODE,
            tau_grid=RANDOM_TAU_GRID,
            p_low=REGIME_P_LO,
            p_high=REGIME_P_HI,
            beta=DRO_BETA,
            eta=DRO_ETA,
            eps_uniform=DRO_EPS_UNIFORM,
            max_prob=DRO_MAX_PROB,
            seed=GLOBAL_SEED + 900000,
            refresh_every=DRO_REFRESH_EVERY,
            refresh_frac=DRO_REFRESH_FRAC,
            buffer_mo_events=DRO_BUFFER_MO_EVENTS,
            persistent_mix_prob=DRO_PERSISTENT_MIX_PROB,
            persistent_rho=DRO_PERSISTENT_RHO,
        )
        print(
            f"[DRO_TAU] Initialised hard-path sampler: pool={DRO_POOL_SIZE}, mix_prob={DRO_TAU_MIX_PROB:.2f}, tau_grid={RANDOM_TAU_GRID}, persistent_mix={DRO_PERSISTENT_MIX_PROB:.2f}, rho={DRO_PERSISTENT_RHO:.2f}"
        )

    def _recalibrate_fill_alpha(tau: float) -> float:
        """Recompute FILL_IMBALANCE_ALPHA from tau after clock rescaling."""
        tau_steps = max(float(tau), 1.0) / max(_MO_EVENT_FRACTION, 1e-06)
        alpha = _compute_fill_alpha(tau, mo_frac=_MO_EVENT_FRACTION)
        print(
            f"[CALIB] fill_imbalance alpha = {alpha:.6f}  (tau_mo={tau:.1f}, tau_steps~={tau_steps:.1f})"
        )
        return alpha

    FILL_IMBALANCE_ALPHA = _recalibrate_fill_alpha(_tau_for_fill_alpha)
    print(f"\n{'-' * 60}")
    print(f"CALIBRATED EWMA PARAMETERS")
    print(
        f"  mo_event_fraction        = {_MO_EVENT_FRACTION:.6f} ({_MO_EVENT_FRACTION_SOURCE}; analytic={_MO_EVENT_FRACTION_ANALYTIC:.6f})"
    )
    print(f"  mo_flow_ewma_alpha       = {EWMA_ALPHA:.6f}")
    if FAST_FLOW_EWMA_ALPHA is None:
        print(f"  mo_flow_fast_alpha       = auto(4x slow, clip<=1.0)")
    else:
        print(f"  mo_flow_fast_alpha       = {FAST_FLOW_EWMA_ALPHA:.6f}")
    print(f"  fill_imbalance_alpha     = {FILL_IMBALANCE_ALPHA:.6f}")
    print(f"  fill_imbalance_tau_mo    = {_tau_for_fill_alpha:.1f}")
    print(
        f"  fill_imbalance_tau_steps ~= {max(float(_tau_for_fill_alpha), 1.0) / max(_MO_EVENT_FRACTION, 1e-06):.1f}"
    )
    print(f"{'-' * 60}\n")
    if False or USE_RANDOM_TAU_REGIME or False or False or USE_CYCLICAL_LR:
        print(f"{'-' * 60}")
        print(f"CURRICULUM SCHEDULE CONFIG")
        print(f"  [PROGRESSIVE] OFF (static p in [{REGIME_P_LO}, {REGIME_P_HI}])")
        if USE_RANDOM_TAU_REGIME:
            _p_start = get_random_tau_probs(0)
            _p_end = get_random_tau_probs(N_EPISODES - 1)
            print(f"  [RANDOM_TAU] grid={RANDOM_TAU_GRID}")
            print(f"               start_probs={np.round(_p_start, 4).tolist()}")
            print(f"               end_probs={np.round(_p_end, 4).tolist()}")
            print(
                f"               mean_tau: {float(np.dot(np.asarray(RANDOM_TAU_GRID), _p_start)):.2f} -> {float(np.dot(np.asarray(RANDOM_TAU_GRID), _p_end)):.2f}"
            )
        if USE_DRO_TAU:
            print(
                f"  [DRO_TAU] pool={DRO_POOL_SIZE}, mix_prob={DRO_TAU_MIX_PROB:.2f}, eta={DRO_ETA}"
            )
            print(
                f"            beta ramp: {DRO_BETA_START} -> {DRO_BETA_END} over {DRO_RAMP_FRAC:.0%} of training"
            )
            print(
                f"            eps_uniform ramp: {DRO_EPS_UNIFORM_START} -> {DRO_EPS_UNIFORM_END} over {DRO_RAMP_FRAC:.0%} of training"
            )
            print(f"            max path probability cap: {DRO_MAX_PROB}")
            print(
                f"            persistent-side mix: {DRO_PERSISTENT_MIX_PROB:.2f}, rho={DRO_PERSISTENT_RHO:.2f}, same p band=[{REGIME_P_LO}, {REGIME_P_HI}]"
            )
            print(
                f"            update loss = -penalized_pnl (PnL - phi*sum(q^2) - active wall penalty), phi={INV_PENALTY_COEFF}"
            )
        print(f"  [ANCHOR DECAY] OFF (static lambda)")
        if USE_CYCLICAL_LR:
            print(
                f"  [CYCLICAL LR] period={CYCLICAL_LR_PERIOD}, min_factor={CYCLICAL_LR_MIN_FACTOR}"
            )
        else:
            print(f"  [CYCLICAL LR] OFF (monotonic decay)")
        print(f"{'-' * 60}\n")
    episode_stats = []
    _es_best_ma = float("-inf")
    _es_best_ep = 0
    _es_triggered = False
    _best_ckpt_ma = float("-inf")
    _best_ckpt_ep = -1
    _best_ckpt_path = None
    for ep in range(N_EPISODES):
        if VERBOSE_LOGGING:
            print("=" * 60)
            print(f"Starting episode {ep + 1}/{N_EPISODES}")
            print(f"Current epsilon BEFORE episode: {deep_controller.epsilon:.4f}")
            print(
                f"Controller mode: {('PURE MM' if deep_controller.pure_mm else 'GENERIC')}"
            )
            if USE_REGIME_SWITCH:
                if USE_RANDOM_TAU_REGIME:
                    _p_preview = get_random_tau_probs(ep)
                    _tau_mean_preview = float(
                        np.dot(np.asarray(RANDOM_TAU_GRID, dtype=float), _p_preview)
                    )
                    print(
                        f"MO flow: RANDOM-TAU REGIME (mean_tau={_tau_mean_preview:.1f}, grid={RANDOM_TAU_GRID}, p in [{REGIME_P_LO},{REGIME_P_HI}])"
                    )
                elif REGIME_DISTRIBUTION == "exponential":
                    print(
                        f"MO flow: REGIME-SWITCHING (exponential, tau={REGIME_EXP_TAU}, p in [{REGIME_P_LO},{REGIME_P_HI}])"
                    )
                else:
                    print(
                        f"MO flow: REGIME-SWITCHING (pareto, alpha={REGIME_ALPHA}, L_min={REGIME_L_MIN}, p in [{REGIME_P_LO},{REGIME_P_HI}])"
                    )
                if ep == 0:
                    print(
                        f"  [EWMA] mo_flow_alpha={EWMA_ALPHA:.6f}, fill_imbalance_alpha={FILL_IMBALANCE_ALPHA:.6f} (fixed)"
                    )
            else:
                print(f"MO flow: STATIONARY (buy_mo_prob=0.5)")
            if hasattr(deep_controller, "optimizer"):
                current_lr = deep_controller.optimizer.param_groups[0]["lr"]
                print(f"Current learning rate: {current_lr:.8f}")
        current_ep_seed = GLOBAL_SEED + 1000 + ep
        np.random.seed(current_ep_seed)
        random.seed(current_ep_seed)
        torch.manual_seed(current_ep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_ep_seed)
        deep_controller.last_action_idx = None
        deep_controller.nstep_buffer.clear()
        deep_controller._smdp_reset()
        deep_controller.moves_tob = 0
        deep_controller.last_env_tob_key = None
        deep_controller.event_steps = 0
        deep_controller.last_update_time = -1.0
        deep_controller._has_acted_once = False
        deep_controller.last_inventory = None
        deep_controller.last_mode = None
        _episode_mo_flow_alpha = None
        _episode_mo_flow_fast_alpha = None
        _episode_fill_alpha = None
        _episode_bayes_tau = BAYES_FLOW_TAU_R
        _episode_tau_probs = None
        _episode_is_dro_tau = False
        _episode_dro_beta = float("nan")
        _episode_dro_eps_uniform = float("nan")
        if USE_REGIME_SWITCH and ep > 0 and (ep % RECALIB_INTERVAL == 0):
            FILL_IMBALANCE_ALPHA = _recalibrate_fill_alpha(_tau_for_fill_alpha)
        if USE_REGIME_SWITCH:
            _ep_p_lo = REGIME_P_LO
            _ep_p_hi = REGIME_P_HI
            if USE_RANDOM_TAU_REGIME:
                _episode_tau_probs = get_random_tau_probs(ep)
                if not FORCE_BAYES_TAU_R_FIXED:
                    _episode_bayes_tau = float(
                        np.dot(
                            np.asarray(RANDOM_TAU_GRID, dtype=float), _episode_tau_probs
                        )
                    )
                _dro_draw = np.random.default_rng(current_ep_seed + 700000).random()
                _episode_is_dro_tau = (
                    bool(USE_DRO_TAU)
                    and dro_tau_adversary is not None
                    and (_dro_draw < float(DRO_TAU_MIX_PROB))
                )
                if _episode_is_dro_tau:
                    _episode_dro_beta, _episode_dro_eps_uniform = get_dro_tau_params(ep)
                    dro_tau_adversary.beta = _episode_dro_beta
                    dro_tau_adversary.eps_uniform = _episode_dro_eps_uniform
                    ep_buy_mo_prob = dro_tau_adversary.select_schedule()
                    _dro_taus = np.asarray(
                        getattr(ep_buy_mo_prob, "taus", []), dtype=float
                    )
                    if _dro_taus.size and (not FORCE_BAYES_TAU_R_FIXED):
                        _episode_bayes_tau = float(np.mean(_dro_taus))
                else:
                    ep_buy_mo_prob = make_random_tau_regime_schedule(
                        seed=current_ep_seed + 500000,
                        n_mo_events=EXPECTED_MO_PER_EPISODE,
                        tau_grid=RANDOM_TAU_GRID,
                        tau_probs=_episode_tau_probs,
                        p_lo=_ep_p_lo,
                        p_hi=_ep_p_hi,
                    )
            else:
                _regime_kwargs = dict(
                    seed=current_ep_seed + 500000,
                    n_mo_events=EXPECTED_MO_PER_EPISODE,
                    p_lo=_ep_p_lo,
                    p_hi=_ep_p_hi,
                    distribution=REGIME_DISTRIBUTION,
                )
                if REGIME_DISTRIBUTION == "pareto":
                    _regime_kwargs["L_min"] = REGIME_L_MIN
                    _regime_kwargs["alpha"] = REGIME_ALPHA
                    _episode_bayes_tau = BAYES_FLOW_TAU_R
                else:
                    _regime_kwargs["exp_rate"] = 1.0 / REGIME_EXP_TAU
                    _episode_bayes_tau = float(REGIME_EXP_TAU)
                ep_buy_mo_prob = make_regime_schedule(**_regime_kwargs)
        else:
            ep_buy_mo_prob = 0.5
        _replay_delta = float("nan")
        if USE_REGIME_SWITCH:
            _replay_delta = float((_ep_p_hi - _ep_p_lo) / 2.0)
        else:
            _replay_delta = 0.0
        deep_controller.set_replay_context(ep, _replay_delta)
        msg_df, ob_df, mm_df = simulate_LOB_with_MM(
            lam=lam,
            mu=mu,
            delta=delta,
            number_tick_levels=NUMBER_TICK_LEVELS,
            n_priority_ranks=N_PRIORITY_RANKS,
            number_levels_to_store=20,
            p0=100,
            mean_size_LO=mean_size_LO,
            mean_size_MO=mean_size_MO,
            iterations=EPISODE_LENGTH,
            iterations_to_equilibrium=ITER_TO_EQUILIBRIUM,
            path_save_files=None,
            label_simulation=None,
            beta_exp_weighted_return=0.0,
            intensity_exp_weighted_return=0.0,
            controller=deep_controller,
            mm_policy=None,
            exclude_self_from_state=False,
            reward_fn=reward_fn_spread,
            random_seed=current_ep_seed,
            buy_mo_prob=ep_buy_mo_prob,
            qrm_params=qrm_params,
            ewma_alpha=EWMA_ALPHA,
            mo_flow_ewma_alpha=_episode_mo_flow_alpha,
            mo_flow_fast_ewma_alpha=_episode_mo_flow_fast_alpha
            if _episode_mo_flow_fast_alpha is not None
            else FAST_FLOW_EWMA_ALPHA,
            fill_imbalance_alpha=_episode_fill_alpha
            if _episode_fill_alpha is not None
            else FILL_IMBALANCE_ALPHA,
            quote_exposure_imbalance_signal=bool(USE_QUOTE_EXPOSURE_IMBALANCE),
            bayes_flow_enabled=USE_BAYES_FLOW_SIGNAL,
            bayes_flow_tau_r=float(_episode_bayes_tau),
            bayes_flow_p_low=float(REGIME_P_LO),
            bayes_flow_p_high=float(REGIME_P_HI),
            bayes_flow_prior_mode=BAYES_FLOW_PRIOR_MODE,
            bayes_flow_cp_window=int(BAYES_FLOW_CP_WINDOW),
            bayes_flow_use_mixture=bool(USE_MIXTURE),
            bayes_flow_mixture_taus=list(BAYES_MIXTURE_TAUS),
        )
        if USE_REGIME_SWITCH and callable(ep_buy_mo_prob):
            _rs_n_regimes = len(getattr(ep_buy_mo_prob, "p_values", []))
            _rs_p_values = getattr(ep_buy_mo_prob, "p_values", [])
            _rs_mean_pb = float(np.mean(_rs_p_values)) if len(_rs_p_values) > 0 else 0.5
            deep_controller.writer.add_scalar(
                "regime_switch/n_regimes", _rs_n_regimes, ep
            )
            deep_controller.writer.add_scalar(
                "regime_switch/mean_p_buy", _rs_mean_pb, ep
            )
            if USE_RANDOM_TAU_REGIME:
                _rs_taus = np.asarray(getattr(ep_buy_mo_prob, "taus", []), dtype=float)
                if _episode_is_dro_tau:
                    _target_mean_tau = float("nan")
                    _realized_mean_tau = (
                        float(np.mean(_rs_taus)) if _rs_taus.size else float("nan")
                    )
                    _last_tau = float(_rs_taus[-1]) if _rs_taus.size else float("nan")
                    deep_controller.writer.add_scalar("dro_tau/used", 1.0, ep)
                    deep_controller.writer.add_scalar(
                        "dro_tau/selected_index",
                        float(getattr(ep_buy_mo_prob, "dro_index", -1)),
                        ep,
                    )
                    deep_controller.writer.add_scalar(
                        "dro_tau/selected_prob",
                        float(getattr(ep_buy_mo_prob, "dro_prob", float("nan"))),
                        ep,
                    )
                    deep_controller.writer.add_scalar(
                        "dro_tau/realized_mean_tau", _realized_mean_tau, ep
                    )
                    deep_controller.writer.add_scalar("dro_tau/last_tau", _last_tau, ep)
                    deep_controller.writer.add_scalar(
                        "dro_tau/beta", _episode_dro_beta, ep
                    )
                    deep_controller.writer.add_scalar(
                        "dro_tau/eps_uniform", _episode_dro_eps_uniform, ep
                    )
                    deep_controller.writer.add_scalar(
                        "dro_tau/selected_persistent_side",
                        float(
                            bool(getattr(ep_buy_mo_prob, "dro_persistent_side", False))
                        ),
                        ep,
                    )
                    if _rs_taus.size:
                        deep_controller.writer.add_histogram(
                            "dro_tau/tau_hist", _rs_taus, ep
                        )
                else:
                    deep_controller.writer.add_scalar("dro_tau/used", 0.0, ep)
                    if _episode_tau_probs is None:
                        _episode_tau_probs = get_random_tau_probs(ep)
                    _tau_probs_arr = np.asarray(_episode_tau_probs, dtype=float)
                    _tau_grid_arr = np.asarray(RANDOM_TAU_GRID, dtype=float)
                    _target_mean_tau = float(np.dot(_tau_grid_arr, _tau_probs_arr))
                    _realized_mean_tau = (
                        float(np.mean(_rs_taus)) if _rs_taus.size else float("nan")
                    )
                    _last_tau = float(_rs_taus[-1]) if _rs_taus.size else float("nan")
                    deep_controller.writer.add_scalar(
                        "random_tau/target_mean_tau", _target_mean_tau, ep
                    )
                    deep_controller.writer.add_scalar(
                        "random_tau/realized_mean_tau", _realized_mean_tau, ep
                    )
                    deep_controller.writer.add_scalar(
                        "random_tau/last_tau", _last_tau, ep
                    )
                    for _i, _tau in enumerate(_tau_grid_arr):
                        _count_tau = (
                            int(np.sum(_rs_taus == _tau)) if _rs_taus.size else 0
                        )
                        deep_controller.writer.add_scalar(
                            f"random_tau/prob_tau{_tau:g}",
                            float(_tau_probs_arr[_i]),
                            ep,
                        )
                        deep_controller.writer.add_scalar(
                            f"random_tau/count_tau{_tau:g}", float(_count_tau), ep
                        )
                    if _rs_taus.size:
                        deep_controller.writer.add_histogram(
                            "random_tau/tau_hist", _rs_taus, ep
                        )
            else:
                _target_mean_tau = float("nan")
                _realized_mean_tau = float("nan")
                _last_tau = float("nan")
            if VERBOSE_LOGGING:
                _bounds_str = ""
                _tau_str = ""
                if USE_RANDOM_TAU_REGIME:
                    if _episode_is_dro_tau:
                        _tau_str = f", DRO mean_tau={_realized_mean_tau:.1f}, beta={_episode_dro_beta:.2f}, eps_u={_episode_dro_eps_uniform:.2f}"
                    else:
                        _tau_str = f", mean_tau={_realized_mean_tau:.1f}, target_mean_tau={_target_mean_tau:.1f}"
                print(
                    f"  [RS] regimes={_rs_n_regimes}, mean_p_buy={_rs_mean_pb:.3f}{_bounds_str}{_tau_str}"
                )
        if VERBOSE_LOGGING:
            print("\n=== ENVIRONMENT STATS (EPISODE) ===")
            print("Env Type counts:\n", msg_df["Type"].value_counts())
            print("Env Direction counts:\n", msg_df["Direction"].value_counts())
            print("Spread range:", msg_df["Spread"].min(), msg_df["Spread"].max())
            print("MidPrice range:", msg_df["MidPrice"].min(), msg_df["MidPrice"].max())
            print("\n=== MARKET MAKER STATS (EPISODE, ENV) ===")
            print(
                "MM inventory range:",
                mm_df["MM_Inventory"].min(),
                mm_df["MM_Inventory"].max(),
            )
            print("Any MM PnL movement?:", mm_df["MM_TotalPnL"].ne(0).any())
        if "MM_CumReward" in mm_df.columns:
            total_reward = float(mm_df["MM_CumReward"].iloc[-1])
        elif "MM_Reward" in mm_df.columns:
            total_reward = float(mm_df["MM_Reward"].sum())
        else:
            total_reward = 0.0
        final_pnl = float(mm_df["MM_TotalPnL"].iloc[-1])
        if "MM_Inventory" in mm_df.columns:
            inv_series = mm_df["MM_Inventory"].values.astype(float)
            abs_inv = np.abs(inv_series)
            max_abs_inv = float(np.max(abs_inv))
            mean_abs_inv = float(np.mean(abs_inv))
            sum_inv_sq = float(np.sum(inv_series**2))
            mean_inv_sq = float(np.mean(inv_series**2))
            _inv_limit = best_params.get("inv_limit", 6)
            pct_at_limit = float(np.mean(abs_inv >= _inv_limit))
        else:
            max_abs_inv = 0.0
            mean_abs_inv = 0.0
            sum_inv_sq = 0.0
            mean_inv_sq = 0.0
            pct_at_limit = 0.0
        _inv_quad_penalty_pnl = float(INV_PENALTY_COEFF) * sum_inv_sq
        _inv_wall_penalty_pnl = 0.0
        if bool(USE_INVENTORY_WALL):
            _wall_threshold = float(_reward_kwargs.get("inv_wall_threshold", 0.0))
            if _wall_threshold > 0.0 and "MM_Inventory" in mm_df.columns:
                _excess_inv = np.maximum(abs_inv - _wall_threshold, 0.0)
                _inv_wall_penalty_pnl = float(INV_PENALTY_COEFF) * float(
                    np.sum(_excess_inv**2)
                )
        penalized_pnl = float(final_pnl - _inv_quad_penalty_pnl - _inv_wall_penalty_pnl)
        gamma = best_params.get("gamma", 0.995)
        if "MM_Reward" in mm_df.columns:
            rewards = mm_df["MM_Reward"].values.astype(float)
            gammas = gamma ** np.arange(len(rewards))
            discounted_return = float(np.dot(gammas, rewards))
        else:
            discounted_return = 0.0
        try:
            first_row = mm_df.iloc[0]
            last_row = mm_df.iloc[-1]
            has_rl_meta = all(
                (
                    col in mm_df.columns
                    for col in [
                        "MM_RL_State_Mode",
                        "MM_RL_State_Dim",
                        "MM_RL_State_MaxOffset",
                    ]
                )
            )
            rl_mode = None
            rl_dim = None
            rl_max_off = None
            if has_rl_meta:
                rl_mode = first_row["MM_RL_State_Mode"]
                rl_dim = first_row["MM_RL_State_Dim"]
                rl_max_off = first_row["MM_RL_State_MaxOffset"]
            if rl_mode == "pure_mm" and "MM_RL_State_Vector" in mm_df.columns:
                rl_vec_col = mm_df["MM_RL_State_Vector"]
                valid_mask_vec = rl_vec_col.apply(
                    lambda v: isinstance(v, (list, np.ndarray))
                )
                if valid_mask_vec.any():
                    vec_start = rl_vec_col[valid_mask_vec].iloc[0]
                    vec_end = rl_vec_col[valid_mask_vec].iloc[-1]
                else:
                    vec_start = first_row["MM_RL_State_Vector"]
                    vec_end = last_row["MM_RL_State_Vector"]
                if isinstance(vec_start, (list, np.ndarray)) and isinstance(
                    vec_end, (list, np.ndarray)
                ):
                    K = (
                        int(rl_max_off)
                        if rl_max_off is not None
                        else max(0, (int(rl_dim) - 2) // 2 - 1)
                    )

                    def decode_pure_mm(vec):
                        vec = list(vec)
                        spread = float(vec[0])
                        inv = float(vec[1])
                        bid_vec = vec[2 : 2 + (K + 1)]
                        ask_vec = vec[2 + (K + 1) : 2 + 2 * (K + 1)]
                        return (spread, inv, bid_vec, ask_vec)

                    s_spread, s_inv, s_bid, s_ask = decode_pure_mm(vec_start)
                    e_spread, e_inv, e_bid, e_ask = decode_pure_mm(vec_end)
                    if VERBOSE_LOGGING:
                        print("\n=== RL STATE SNAPSHOT (EPISODE, PURE MM NN INPUT) ===")
                    if VERBOSE_LOGGING:
                        print(
                            "  (spread = log1p(raw), inventory = raw/inv_limit, depths = log1p(volume))"
                        )
                    if VERBOSE_LOGGING:
                        print("Start NN state:")
                    if VERBOSE_LOGGING:
                        print(f"  spread={s_spread:.2f}, inventory={s_inv:.2f}")
                    if VERBOSE_LOGGING:
                        print("  bid_depth (log1p vol):")
                    for j, val in enumerate(s_bid):
                        if VERBOSE_LOGGING:
                            print(f"    level_{j}= {val:.2f}")
                    if VERBOSE_LOGGING:
                        print("  ask_depth (log1p vol):")
                    for j, val in enumerate(s_ask):
                        if VERBOSE_LOGGING:
                            print(f"    level_{j}= {val:.2f}")
                    if VERBOSE_LOGGING:
                        print("End NN state:")
                    if VERBOSE_LOGGING:
                        print(f"  spread={e_spread:.2f}, inventory={e_inv:.2f}")
                    if VERBOSE_LOGGING:
                        print("  bid_depth (log1p vol):")
                    for j, val in enumerate(e_bid):
                        if VERBOSE_LOGGING:
                            print(f"    level_{j}= {val:.2f}")
                    if VERBOSE_LOGGING:
                        print("  ask_depth (log1p vol):")
                    for j, val in enumerate(e_ask):
                        if VERBOSE_LOGGING:
                            print(f"    level_{j}= {val:.2f}")
                elif VERBOSE_LOGGING:
                    print(
                        "\n[Warning] MM_RL_State_Vector not list/ndarray; skipping NN-state decode."
                    )
            else:
                has_state_cols = all(
                    (
                        col in mm_df.columns
                        for col in [
                            "MM_State_Spread",
                            "MM_State_AskSize",
                            "MM_State_BidSize",
                            "MM_State_Inventory",
                            "MM_State_HasBid",
                            "MM_State_HasAsk",
                        ]
                    )
                )
                if has_state_cols:
                    start_spread = float(first_row["MM_State_Spread"])
                    start_asksize = float(first_row["MM_State_AskSize"])
                    start_bidsize = float(first_row["MM_State_BidSize"])
                    start_inv = float(first_row["MM_State_Inventory"])
                    start_has_bid = int(first_row["MM_State_HasBid"])
                    start_has_ask = int(first_row["MM_State_HasAsk"])
                    end_spread = float(last_row["MM_State_Spread"])
                    end_asksize = float(last_row["MM_State_AskSize"])
                    end_bidsize = float(last_row["MM_State_BidSize"])
                    end_inv = float(last_row["MM_State_Inventory"])
                    end_has_bid = int(last_row["MM_State_HasBid"])
                    end_has_ask = int(last_row["MM_State_HasAsk"])
                    if VERBOSE_LOGGING:
                        print("\n=== RL STATE SNAPSHOT (EPISODE, L1 VIEW) ===")
                    if VERBOSE_LOGGING:
                        print("Start state (first action state):")
                    print(
                        f"  spread={start_spread:.1f}, ask_size={start_asksize:.1f}, bid_size={start_bidsize:.1f}, inventory={start_inv:.1f}, has_bid={start_has_bid}, has_ask={start_has_ask}"
                    )
                    if VERBOSE_LOGGING:
                        print("End state (last action state):")
                    print(
                        f"  spread={end_spread:.1f}, ask_size={end_asksize:.1f}, bid_size={end_bidsize:.1f}, inventory={end_inv:.1f}, has_bid={end_has_bid}, has_ask={end_has_ask}"
                    )
                else:
                    print(
                        "\n[Warning] MM_State_* columns not found in mm_df; RL state snapshot for first/last step is skipped."
                    )
            if "MM_HadFill" in mm_df.columns and "MM_LastFillSide" in mm_df.columns:
                had_fill = mm_df["MM_HadFill"] == True
                bid_fills = had_fill & (mm_df["MM_LastFillSide"] == +1)
                ask_fills = had_fill & (mm_df["MM_LastFillSide"] == -1)
                n_bid_fills = int(bid_fills.sum())
                n_ask_fills = int(ask_fills.sum())
                n_total_fills = n_bid_fills + n_ask_fills
                if VERBOSE_LOGGING:
                    print("\n=== FILL STATS (EPISODE) ===")
                if VERBOSE_LOGGING:
                    print(f"Total fills on BID side : {n_bid_fills}")
                if VERBOSE_LOGGING:
                    print(f"Total fills on ASK side : {n_ask_fills}")
                if VERBOSE_LOGGING:
                    print(f"Total fills (bid+ask)   : {n_total_fills}")
            else:
                print(
                    "\n[Warning] MM_HadFill / MM_LastFillSide not found in mm_df; fill statistics are skipped."
                )
        except Exception as e:
            if VERBOSE_LOGGING:
                print(
                    f"\n[Warning] Exception while computing RL state / fill stats: {e}"
                )
        if (
            VERBOSE_LOGGING
            and deep_controller.pure_mm
            and (deep_controller.pure_mm_offsets is not None)
        ):
            print(f"\n=== RL POLICY STATS (EPISODE, PURE MM OFFSETS) ===")
            print(f"Discrete action usage (episode {ep + 1}):")
            for i, (bo, ao) in enumerate(deep_controller.pure_mm_offsets):
                count_i = int(deep_controller.action_counts[i])
                print(
                    f"  a={i:2d}  offsets(bid,ask)=({bo},{ao})  -> used {count_i} times"
                )
        elif VERBOSE_LOGGING:
            print(f"\n=== RL POLICY STATS (EPISODE, GENERIC ACTIONS) ===")
            print(f"Discrete action usage (episode {ep + 1}):")
            generic_names = [
                "post_bid",
                "post_ask",
                "post_bid_ask",
                "cancel_bid",
                "cancel_ask",
                "hold",
                "post_bid_ask_inside_spread",
                "post_bid_inside_spread",
                "post_ask_inside_spread",
            ]
            if VERBOSE_LOGGING:
                for i, name in enumerate(generic_names):
                    if i >= len(deep_controller.action_counts):
                        break
                    count_i = int(deep_controller.action_counts[i])
                    print(f"  a={i:2d}  {name:>26s}  -> used {count_i} times")
        _qmin = deep_controller.episode_q_min
        _qmax = deep_controller.episode_q_max
        _tdmin = deep_controller.episode_td_target_min
        _tdmax = deep_controller.episode_td_target_max
        _td_count = deep_controller.episode_td_target_count
        _td_sum = deep_controller.episode_td_target_sum
        _tz_clip_count = deep_controller.episode_tz_clip_count
        _tz_clip_sum = deep_controller.episode_tz_clip_frac_sum
        deep_controller.log_episode_stats(
            ep, total_reward, final_pnl, discounted_return=discounted_return
        )
        old_eps = deep_controller.epsilon
        _mm_ep_for_decay = ep
        _mm_total_for_decay = N_EPISODES
        eps_start = best_params["epsilon_start"]
        eps_min = best_params["epsilon_min"]
        eps_decay_type = best_params.get("epsilon_decay_type", "linear")
        progress = _mm_ep_for_decay / _mm_total_for_decay
        if eps_decay_type == "exponential":
            eps_mult = best_params.get("epsilon_decay", 0.98)
            new_eps = eps_start * eps_mult**_mm_ep_for_decay
        else:
            new_eps = eps_start - progress * (eps_start - eps_min)
        deep_controller.epsilon = max(eps_min, new_eps)
        if VERBOSE_LOGGING:
            print(f"\nTotal reward (episode {ep + 1}): {total_reward:.4f}")
            print(f"Discounted return G_0 (episode {ep + 1}): {discounted_return:.6f}")
            print(f"Final total PnL (episode {ep + 1}): {final_pnl:.4f}")
            print(f"Penalized PnL (episode {ep + 1}): {penalized_pnl:.4f}")
            print(
                f"|inv|_max={max_abs_inv:.0f}  |inv|_mean={mean_abs_inv:.2f}  at_limit={pct_at_limit:.1%}"
            )
            _qmin_s = f"{_qmin:.6f}" if _qmin != float("inf") else "N/A"
            _qmax_s = f"{_qmax:.6f}" if _qmax != float("-inf") else "N/A"
            _tdmin_s = f"{_tdmin:.6f}" if _tdmin != float("inf") else "N/A"
            _tdmax_s = f"{_tdmax:.6f}" if _tdmax != float("-inf") else "N/A"
            _tdmean_s = f"{_td_sum / _td_count:.6f}" if _td_count > 0 else "N/A"
            _tzclip_s = (
                f"{_tz_clip_sum / _tz_clip_count:.1%}" if _tz_clip_count > 0 else "N/A"
            )
            print(
                f"Q(s,a) range: [{_qmin_s}, {_qmax_s}]  (C51 support: [{deep_controller.dist_v_min}, {deep_controller.dist_v_max}])"
            )
            print(
                f"TD target: [{_tdmin_s}, {_tdmax_s}]  mean={_tdmean_s}  Tz_clip={_tzclip_s}"
            )
            print(f"Replay buffer size: {len(deep_controller.memory)}")
            print(
                f"Epsilon decayed from {old_eps:.4f} to {deep_controller.epsilon:.4f}"
            )
        episode_stats.append(
            {
                "episode": ep,
                "final_pnl": final_pnl,
                "penalized_pnl": penalized_pnl,
                "total_reward": total_reward,
                "discounted_return": discounted_return,
                "epsilon": deep_controller.epsilon,
                "max_abs_inv": max_abs_inv,
                "mean_abs_inv": mean_abs_inv,
                "mean_inv_sq": mean_inv_sq,
                "sum_inv_sq": sum_inv_sq,
                "pct_at_limit": pct_at_limit,
            }
        )
        window = 10
        if len(episode_stats) >= window:
            last_rewards = [e["total_reward"] for e in episode_stats[-window:]]
            ma_10 = float(np.mean(last_rewards))
            deep_controller.writer.add_scalar("episode/total_reward_ma_10", ma_10, ep)
        pnl_ma_10 = float("nan")
        if len(episode_stats) >= window:
            last_pnls = [e["final_pnl"] for e in episode_stats[-window:]]
            pnl_ma_10 = float(np.mean(last_pnls))
            deep_controller.writer.add_scalar("episode/final_pnl_ma_10", pnl_ma_10, ep)
            last_pen_pnls = [e["penalized_pnl"] for e in episode_stats[-window:]]
            deep_controller.writer.add_scalar(
                "episode/penalized_pnl_ma_10", float(np.mean(last_pen_pnls)), ep
            )
        pnl_ma_50 = float("nan")
        pnl_ma_100 = float("nan")
        if len(episode_stats) >= 50:
            _last_pnls_50 = [e["final_pnl"] for e in episode_stats[-50:]]
            pnl_ma_50 = float(np.mean(_last_pnls_50))
            deep_controller.writer.add_scalar("episode/final_pnl_ma_50", pnl_ma_50, ep)
            _last_pen_pnls_50 = [e["penalized_pnl"] for e in episode_stats[-50:]]
            deep_controller.writer.add_scalar(
                "episode/penalized_pnl_ma_50", float(np.mean(_last_pen_pnls_50)), ep
            )
        if len(episode_stats) >= 100:
            _last_pnls_100 = [e["final_pnl"] for e in episode_stats[-100:]]
            pnl_ma_100 = float(np.mean(_last_pnls_100))
            deep_controller.writer.add_scalar(
                "episode/final_pnl_ma_100", pnl_ma_100, ep
            )
            _last_pen_pnls_100 = [e["penalized_pnl"] for e in episode_stats[-100:]]
            deep_controller.writer.add_scalar(
                "episode/penalized_pnl_ma_100", float(np.mean(_last_pen_pnls_100)), ep
            )
        pnl_ma_best = float("nan")
        if len(episode_stats) >= BEST_CKPT_WINDOW:
            _last_pnls_best = [
                e["final_pnl"] for e in episode_stats[-BEST_CKPT_WINDOW:]
            ]
            pnl_ma_best = float(np.mean(_last_pnls_best))
            deep_controller.writer.add_scalar(
                f"episode/final_pnl_ma_{BEST_CKPT_WINDOW}", pnl_ma_best, ep
            )
        if not VERBOSE_LOGGING:
            _ma_parts = []
            if not np.isnan(pnl_ma_10):
                _ma_parts.append(f"MA10={pnl_ma_10:+.4f}")
            if not np.isnan(pnl_ma_50):
                _ma_parts.append(f"MA50={pnl_ma_50:+.4f}")
            if not np.isnan(pnl_ma_100):
                _ma_parts.append(f"MA100={pnl_ma_100:+.4f}")
            _ma_str = f"  {'  '.join(_ma_parts)}" if _ma_parts else ""
            print(f"[{ep + 1:4d}/{N_EPISODES}]  PnL={final_pnl:+.4f}{_ma_str}")
        deep_controller.writer.add_scalar(
            "episode/epsilon_outer", deep_controller.epsilon, ep
        )
        deep_controller.writer.add_scalar("episode/max_abs_inventory", max_abs_inv, ep)
        deep_controller.writer.add_scalar(
            "episode/mean_abs_inventory", mean_abs_inv, ep
        )
        deep_controller.writer.add_scalar("episode/mean_inventory_sq", mean_inv_sq, ep)
        deep_controller.writer.add_scalar("episode/pct_at_inv_limit", pct_at_limit, ep)
        deep_controller.writer.add_scalar("episode/penalized_pnl", penalized_pnl, ep)
        deep_controller.writer.add_scalar(
            "episode/inv_quad_penalty_pnl", _inv_quad_penalty_pnl, ep
        )
        deep_controller.writer.add_scalar(
            "episode/inv_wall_penalty_pnl", _inv_wall_penalty_pnl, ep
        )
        if "MM_Fill_Imbalance_EWMA" in mm_df.columns:
            _fill_bias = mm_df["MM_Fill_Imbalance_EWMA"].values.astype(float)
            deep_controller.writer.add_scalar(
                "episode/fill_bias_abs_mean", float(np.mean(np.abs(_fill_bias))), ep
            )
            deep_controller.writer.add_scalar(
                "episode/fill_bias_abs_max", float(np.max(np.abs(_fill_bias))), ep
            )
        if "MM_Fill_Bid_EWMA" in mm_df.columns and "MM_Fill_Ask_EWMA" in mm_df.columns:
            _fill_bid = mm_df["MM_Fill_Bid_EWMA"].values.astype(float)
            _fill_ask = mm_df["MM_Fill_Ask_EWMA"].values.astype(float)
            deep_controller.writer.add_scalar(
                "episode/fill_activity_mean", float(np.mean(_fill_bid + _fill_ask)), ep
            )
        if USE_DRO_TAU and _episode_is_dro_tau and (dro_tau_adversary is not None):
            _dro_loss = -float(penalized_pnl)
            _dro_stats = dro_tau_adversary.update_loss(_dro_loss)
            deep_controller.writer.add_scalar("dro_tau/loss", _dro_stats["loss"], ep)
            deep_controller.writer.add_scalar(
                "dro_tau/difficulty", _dro_stats["difficulty"], ep
            )
            deep_controller.writer.add_scalar("dro_tau/prob", _dro_stats["prob"], ep)
            deep_controller.writer.add_scalar(
                "dro_tau/mean_difficulty", _dro_stats["mean_difficulty"], ep
            )
            deep_controller.writer.add_scalar(
                "dro_tau/max_difficulty", _dro_stats["max_difficulty"], ep
            )
            deep_controller.writer.add_scalar(
                "dro_tau/min_difficulty", _dro_stats["min_difficulty"], ep
            )
            deep_controller.writer.add_scalar(
                "dro_tau/entropy", _dro_stats["entropy"], ep
            )
            deep_controller.writer.add_scalar(
                "dro_tau/selected_count", _dro_stats["selected_count"], ep
            )
            deep_controller.writer.add_scalar(
                "dro_tau/persistent_side", _dro_stats["persistent_side"], ep
            )
            if not np.isnan(_dro_stats.get("max_prob", float("nan"))):
                deep_controller.writer.add_scalar(
                    "dro_tau/max_prob_cap", _dro_stats["max_prob"], ep
                )
            if VERBOSE_LOGGING:
                print(
                    f"  [DRO_TAU] loss={_dro_stats['loss']:+.4f}, difficulty={_dro_stats['difficulty']:+.4f}, prob={_dro_stats['prob']:.4f}, persistent={bool(_dro_stats['persistent_side'])}"
                )
        lr_decay_type = best_params.get("lr_decay_type", "exponential")
        use_exp_lr = lr_decay_type == "exponential"
        if hasattr(deep_controller, "optimizer"):
            new_lr = get_lr_for_episode(
                _mm_ep_for_decay, _mm_total_for_decay, use_exp=use_exp_lr
            )
            _lr_cycle_mult = 1.0
            if USE_CYCLICAL_LR:
                _lr_cycle_mult = get_cyclical_lr_multiplier(
                    ep, period=CYCLICAL_LR_PERIOD, min_factor=CYCLICAL_LR_MIN_FACTOR
                )
                new_lr *= _lr_cycle_mult
            n_groups = len(deep_controller.optimizer.param_groups)
            if n_groups > 1 and WARMSTART_CKPT is not None:
                _feat_factor = (
                    LR_FEATURE_FACTOR if "LR_FEATURE_FACTOR" in dir() else 0.2
                )
                for i, param_group in enumerate(deep_controller.optimizer.param_groups):
                    if i < n_groups - 1:
                        param_group["lr"] = new_lr * _feat_factor
                    else:
                        param_group["lr"] = new_lr
            else:
                for param_group in deep_controller.optimizer.param_groups:
                    param_group["lr"] = new_lr
            deep_controller.writer.add_scalar("train/lr", new_lr, ep)
            if USE_CYCLICAL_LR:
                deep_controller.writer.add_scalar(
                    "train/lr_cycle_mult", _lr_cycle_mult, ep
                )
            if VERBOSE_LOGGING:
                _clr_str = (
                    f", cycle_mult={_lr_cycle_mult:.4f}" if USE_CYCLICAL_LR else ""
                )
                print(
                    f"Updated learning rate to {new_lr:.8f} (mm_ep={_mm_ep_for_decay}{_clr_str})"
                )
        elif VERBOSE_LOGGING:
            print("Skipping LR update: deep_controller has no 'optimizer'.")
        CKPT_INTERVAL = 50
        _skip_ckpt = False
        if (ep + 1) % CKPT_INTERVAL == 0 and (not _skip_ckpt):
            _ckpt_dir = "checkpoints"
            os.makedirs(_ckpt_dir, exist_ok=True)
            _train_suffix = _build_train_suffix()
            _ckpt_name = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_final.pt"
            _ckpt_path = _safe_checkpoint_filename(os.path.join(_ckpt_dir, _ckpt_name))
            _ckpt_name = os.path.basename(_ckpt_path)
            _ckpt_meta = {
                "USE_PURE_MM": bool(USE_PURE_MM),
                "best_params": dict(best_params),
                "pure_mm_offsets": pure_mm_offsets,
                "lam": lam,
                "mu": mu,
                "delta": delta,
                "GLOBAL_SEED": GLOBAL_SEED,
                "EPISODE_LENGTH": EPISODE_LENGTH,
                "ITER_TO_EQUILIBRIUM": ITER_TO_EQUILIBRIUM,
                "N_EPISODES": N_EPISODES,
                "per_alpha_last_episode": N_EPISODES,
                "per_beta_last_episode": N_EPISODES,
                "episode": ep + 1,
                "USE_REGIME_SWITCH": bool(USE_REGIME_SWITCH),
                "REGIME_DISTRIBUTION": REGIME_DISTRIBUTION,
                "REGIME_ALPHA": REGIME_ALPHA,
                "REGIME_L_MIN": REGIME_L_MIN,
                "REGIME_EXP_TAU": REGIME_EXP_TAU,
                "USE_RANDOM_TAU_REGIME": bool(USE_RANDOM_TAU_REGIME),
                "RANDOM_TAU_GRID": list(RANDOM_TAU_GRID),
                "RANDOM_TAU_PROBS_START": list(RANDOM_TAU_PROBS_START),
                "RANDOM_TAU_PROBS_END": list(RANDOM_TAU_PROBS_END),
                "RANDOM_TAU_RAMP_FRAC": float(RANDOM_TAU_RAMP_FRAC),
                "USE_DRO_TAU": bool(USE_DRO_TAU),
                "DRO_TAU_MIX_PROB": float(DRO_TAU_MIX_PROB),
                "DRO_POOL_SIZE": int(DRO_POOL_SIZE),
                "DRO_BETA": float(DRO_BETA),
                "DRO_BETA_START": float(DRO_BETA_START),
                "DRO_BETA_END": float(DRO_BETA_END),
                "DRO_ETA": float(DRO_ETA),
                "DRO_EPS_UNIFORM": float(DRO_EPS_UNIFORM),
                "DRO_EPS_UNIFORM_START": float(DRO_EPS_UNIFORM_START),
                "DRO_EPS_UNIFORM_END": float(DRO_EPS_UNIFORM_END),
                "DRO_RAMP_FRAC": float(DRO_RAMP_FRAC),
                "DRO_MAX_PROB": None if DRO_MAX_PROB is None else float(DRO_MAX_PROB),
                "DRO_REFRESH_EVERY": int(DRO_REFRESH_EVERY),
                "DRO_REFRESH_FRAC": float(DRO_REFRESH_FRAC),
                "DRO_BUFFER_MO_EVENTS": int(DRO_BUFFER_MO_EVENTS),
                "DRO_PERSISTENT_MIX_PROB": float(DRO_PERSISTENT_MIX_PROB),
                "DRO_PERSISTENT_RHO": float(DRO_PERSISTENT_RHO),
                "REGIME_P_LO": REGIME_P_LO,
                "REGIME_P_HI": REGIME_P_HI,
                "EWMA_ALPHA": float(EWMA_ALPHA),
                "FAST_FLOW_EWMA_ALPHA": None
                if FAST_FLOW_EWMA_ALPHA is None
                else float(FAST_FLOW_EWMA_ALPHA),
                "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
                "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
                "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
                "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
                "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
                "USE_ADVERSARIAL": bool(False),
                "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
                "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
                "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
                "USE_MIXTURE": bool(USE_MIXTURE),
                "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
                "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
                "FORCE_BAYES_TAU_R_FIXED": bool(FORCE_BAYES_TAU_R_FIXED),
                "USE_TRUNCATED_BETA": bool(USE_TRUNCATED_BETA),
                "BAYES_FLOW_PRIOR_MODE": BAYES_FLOW_PRIOR_MODE,
                "BAYES_FLOW_CP_WINDOW": int(BAYES_FLOW_CP_WINDOW),
                "BAYES_FLOW_FEATURE_KEYS": list(BAYES_FLOW_FEATURE_KEYS),
                "PLOT_FILL_IMBALANCE": bool(PLOT_FILL_IMBALANCE),
                "PLOT_FILL_IMBALANCE_INTERVAL": int(PLOT_FILL_IMBALANCE_INTERVAL),
                "PLOT_BAYES_METRICS": bool(PLOT_BAYES_METRICS),
                "PLOT_BAYES_METRICS_INTERVAL": int(PLOT_BAYES_METRICS_INTERVAL),
                "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
                "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
                "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
                "USE_FILL_IMBALANCE": bool(USE_FILL_IMBALANCE),
                "USE_QUOTE_EXPOSURE_IMBALANCE": bool(USE_QUOTE_EXPOSURE_IMBALANCE),
                "ACTIVE_FILL_IMBALANCE_FEATURE": _active_fill_imbalance_feature_tag(),
                "CHECKPOINT_FEATURE_TAG": _checkpoint_feature_tag(),
                "USE_G_CLIP": bool(USE_G_CLIP),
                "USE_PER_PRIORITY_CLIP": bool(USE_PER_PRIORITY_CLIP),
                "WARMSTART_CKPT": WARMSTART_CKPT,
                "USE_FINE_TUNING_TECHNIQUES": bool(USE_FINE_TUNING_TECHNIQUES),
                "ANCHOR_LAMBDA": float(ANCHOR_LAMBDA),
                "LR_FEATURE_FACTOR": float(LR_FEATURE_FACTOR),
                "REPLAY_RESTORE_FRACTION": float(REPLAY_RESTORE_FRACTION),
                "USE_ANCHOR_DECAY": bool(False),
                "USE_CURRICULUM_TIED_ANCHOR": bool(False),
                "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
                "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
                "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
                "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
                "USE_ADV_TAU": bool(False),
                "ADV_TAU_MODE": ADV_TAU_MODE,
                "ADV_TAU_OBJECTIVE": ADV_TAU_OBJECTIVE,
                "ADV_FREEZE_MM": ADV_FREEZE_MM,
                "ADV_FREEZE_ADV": ADV_FREEZE_ADV,
                "ADV_WARMUP_EPISODES": ADV_WARMUP_EPISODES,
            }
            save_deep_rl_checkpoint(deep_controller, _ckpt_path, _ckpt_meta)
            print(f"[CHECKPOINT] Saved checkpoint (ep {ep + 1}): {_ckpt_name}")
        _skip_best_ckpt = False
        if (
            not np.isnan(pnl_ma_best)
            and (not _skip_best_ckpt)
            and (pnl_ma_best > _best_ckpt_ma)
        ):
            _best_ckpt_ma = pnl_ma_best
            _best_ckpt_ep = ep
            _ckpt_dir = "checkpoints"
            os.makedirs(_ckpt_dir, exist_ok=True)
            _train_suffix = _build_train_suffix()
            _best_ckpt_name = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_best_ma{BEST_CKPT_WINDOW}.pt"
            _best_ckpt_path = _safe_checkpoint_filename(
                os.path.join(_ckpt_dir, _best_ckpt_name)
            )
            _best_ckpt_name = os.path.basename(_best_ckpt_path)
            _best_ckpt_meta = {
                "USE_PURE_MM": bool(USE_PURE_MM),
                "best_params": dict(best_params),
                "pure_mm_offsets": pure_mm_offsets,
                "lam": lam,
                "mu": mu,
                "delta": delta,
                "GLOBAL_SEED": GLOBAL_SEED,
                "EPISODE_LENGTH": EPISODE_LENGTH,
                "ITER_TO_EQUILIBRIUM": ITER_TO_EQUILIBRIUM,
                "N_EPISODES": N_EPISODES,
                "episode": ep + 1,
                "best_ma_window": BEST_CKPT_WINDOW,
                "best_ma_value": float(_best_ckpt_ma),
                "best_ma_episode": ep + 1,
                "USE_REGIME_SWITCH": bool(USE_REGIME_SWITCH),
                "REGIME_DISTRIBUTION": REGIME_DISTRIBUTION,
                "REGIME_ALPHA": REGIME_ALPHA,
                "REGIME_L_MIN": REGIME_L_MIN,
                "REGIME_EXP_TAU": REGIME_EXP_TAU,
                "USE_RANDOM_TAU_REGIME": bool(USE_RANDOM_TAU_REGIME),
                "RANDOM_TAU_GRID": list(RANDOM_TAU_GRID),
                "RANDOM_TAU_PROBS_START": list(RANDOM_TAU_PROBS_START),
                "RANDOM_TAU_PROBS_END": list(RANDOM_TAU_PROBS_END),
                "RANDOM_TAU_RAMP_FRAC": float(RANDOM_TAU_RAMP_FRAC),
                "USE_DRO_TAU": bool(USE_DRO_TAU),
                "DRO_TAU_MIX_PROB": float(DRO_TAU_MIX_PROB),
                "DRO_POOL_SIZE": int(DRO_POOL_SIZE),
                "DRO_BETA": float(DRO_BETA),
                "DRO_BETA_START": float(DRO_BETA_START),
                "DRO_BETA_END": float(DRO_BETA_END),
                "DRO_ETA": float(DRO_ETA),
                "DRO_EPS_UNIFORM": float(DRO_EPS_UNIFORM),
                "DRO_EPS_UNIFORM_START": float(DRO_EPS_UNIFORM_START),
                "DRO_EPS_UNIFORM_END": float(DRO_EPS_UNIFORM_END),
                "DRO_RAMP_FRAC": float(DRO_RAMP_FRAC),
                "DRO_MAX_PROB": None if DRO_MAX_PROB is None else float(DRO_MAX_PROB),
                "DRO_REFRESH_EVERY": int(DRO_REFRESH_EVERY),
                "DRO_REFRESH_FRAC": float(DRO_REFRESH_FRAC),
                "DRO_BUFFER_MO_EVENTS": int(DRO_BUFFER_MO_EVENTS),
                "DRO_PERSISTENT_MIX_PROB": float(DRO_PERSISTENT_MIX_PROB),
                "DRO_PERSISTENT_RHO": float(DRO_PERSISTENT_RHO),
                "REGIME_P_LO": REGIME_P_LO,
                "REGIME_P_HI": REGIME_P_HI,
                "EWMA_ALPHA": float(EWMA_ALPHA),
                "FAST_FLOW_EWMA_ALPHA": None
                if FAST_FLOW_EWMA_ALPHA is None
                else float(FAST_FLOW_EWMA_ALPHA),
                "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
                "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
                "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
                "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
                "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
                "USE_ADVERSARIAL": bool(False),
                "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
                "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
                "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
                "USE_MIXTURE": bool(USE_MIXTURE),
                "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
                "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
                "FORCE_BAYES_TAU_R_FIXED": bool(FORCE_BAYES_TAU_R_FIXED),
                "USE_TRUNCATED_BETA": bool(USE_TRUNCATED_BETA),
                "BAYES_FLOW_PRIOR_MODE": BAYES_FLOW_PRIOR_MODE,
                "BAYES_FLOW_CP_WINDOW": int(BAYES_FLOW_CP_WINDOW),
                "BAYES_FLOW_FEATURE_KEYS": list(BAYES_FLOW_FEATURE_KEYS),
                "PLOT_FILL_IMBALANCE": bool(PLOT_FILL_IMBALANCE),
                "PLOT_FILL_IMBALANCE_INTERVAL": int(PLOT_FILL_IMBALANCE_INTERVAL),
                "PLOT_BAYES_METRICS": bool(PLOT_BAYES_METRICS),
                "PLOT_BAYES_METRICS_INTERVAL": int(PLOT_BAYES_METRICS_INTERVAL),
                "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
                "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
                "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
                "USE_FILL_IMBALANCE": bool(USE_FILL_IMBALANCE),
                "USE_QUOTE_EXPOSURE_IMBALANCE": bool(USE_QUOTE_EXPOSURE_IMBALANCE),
                "ACTIVE_FILL_IMBALANCE_FEATURE": _active_fill_imbalance_feature_tag(),
                "CHECKPOINT_FEATURE_TAG": _checkpoint_feature_tag(),
                "USE_G_CLIP": bool(USE_G_CLIP),
                "USE_PER_PRIORITY_CLIP": bool(USE_PER_PRIORITY_CLIP),
                "WARMSTART_CKPT": WARMSTART_CKPT,
                "USE_FINE_TUNING_TECHNIQUES": bool(USE_FINE_TUNING_TECHNIQUES),
                "ANCHOR_LAMBDA": float(ANCHOR_LAMBDA),
                "LR_FEATURE_FACTOR": float(LR_FEATURE_FACTOR),
                "REPLAY_RESTORE_FRACTION": float(REPLAY_RESTORE_FRACTION),
                "USE_ANCHOR_DECAY": bool(False),
                "USE_CURRICULUM_TIED_ANCHOR": bool(False),
                "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
                "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
                "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
                "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
                "USE_ADV_TAU": bool(False),
                "ADV_TAU_MODE": ADV_TAU_MODE,
                "ADV_TAU_OBJECTIVE": ADV_TAU_OBJECTIVE,
                "ADV_FREEZE_MM": ADV_FREEZE_MM,
                "ADV_FREEZE_ADV": ADV_FREEZE_ADV,
                "ADV_WARMUP_EPISODES": ADV_WARMUP_EPISODES,
            }
            save_deep_rl_checkpoint(deep_controller, _best_ckpt_path, _best_ckpt_meta)
            print(
                f"[CHECKPOINT] Updated BEST CHECKPOINT (criterion=MA{BEST_CKPT_WINDOW}) (ep {ep + 1}, MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f}): {_best_ckpt_name}"
            )
        if PLOT_EWMA and USE_FLOW_EWMA:
            if (
                ep + 1
            ) % PLOT_EWMA_INTERVAL == 0 and "MM_MO_Flow_EWMA" in mm_df.columns:
                fig_ewma, ax_ewma = plt.subplots(figsize=(10, 3))
                ax_ewma.plot(
                    mm_df["MM_MO_Flow_EWMA"].values,
                    linewidth=0.8,
                    color="steelblue",
                    label="slow",
                )
                _has_fast_flow = (
                    USE_FAST_FLOW_EWMA and "MM_MO_Flow_Fast_EWMA" in mm_df.columns
                )
                if _has_fast_flow:
                    ax_ewma.plot(
                        mm_df["MM_MO_Flow_Fast_EWMA"].values,
                        linewidth=0.8,
                        color="crimson",
                        alpha=0.9,
                        label="fast",
                    )
                ax_ewma.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax_ewma.set_ylim(-1.05, 1.05)
                ax_ewma.set_xlabel("Simulation Step")
                ax_ewma.set_ylabel("MO Flow EWMA")
                ax_ewma.set_title(
                    f"MO Flow EWMA{(' (slow + fast)' if _has_fast_flow else '')} -- Episode {ep + 1}"
                )
                if _has_fast_flow:
                    ax_ewma.legend(loc="upper right", frameon=False)
                plt.tight_layout()
                plt.show()
                plt.close(fig_ewma)
        if PLOT_FILL_IMBALANCE and USE_FILL_IMBALANCE:
            _fill_interval = max(int(PLOT_FILL_IMBALANCE_INTERVAL), 1)
            if (
                ep + 1
            ) % _fill_interval == 0 and "MM_Fill_Imbalance_EWMA" in mm_df.columns:
                fig_fi, ax_fi = plt.subplots(figsize=(10, 3))
                ax_fi.plot(
                    mm_df["MM_Fill_Imbalance_EWMA"].values,
                    linewidth=0.6,
                    color="gray",
                    alpha=0.75,
                    label="old EWMA",
                )
                if (
                    USE_QUOTE_EXPOSURE_IMBALANCE
                    and "MM_Fill_Imbalance_Feature" in mm_df.columns
                ):
                    ax_fi.plot(
                        mm_df["MM_Fill_Imbalance_Feature"].values,
                        linewidth=0.8,
                        color="teal",
                        alpha=0.95,
                        label="quote exposure imbalance (DQN)",
                    )
                ax_fi.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax_fi.set_ylim(-1.05, 1.05)
                ax_fi.set_xlabel("Simulation Step")
                ax_fi.set_ylabel("Fill Imbalance")
                ax_fi.set_title(f"Fill Imbalance -- Episode {ep + 1}")
                ax_fi.legend(loc="upper right", frameon=False, fontsize=8)
                plt.tight_layout()
                plt.show()
                plt.close(fig_fi)
        if PLOT_BAYES_METRICS and USE_BAYES_FLOW_SIGNAL:
            _bayes_cols = (
                "MM_Bayes_M_Hat",
                "MM_Bayes_CP_Prob",
                "MM_Bayes_Run_Length",
                "MM_Bayes_Uncertainty",
            )
            if (ep + 1) % PLOT_BAYES_METRICS_INTERVAL == 0 and all(
                (c in mm_df.columns for c in _bayes_cols)
            ):
                fig_bayes, ax_bayes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                _x = np.arange(len(mm_df))
                _true_m_hat = None
                _true_regime_age = None
                _true_regime_len = None
                if (
                    callable(ep_buy_mo_prob)
                    and hasattr(ep_buy_mo_prob, "boundaries")
                    and ("Type" in msg_df.columns)
                ):
                    _boundaries = np.asarray(
                        getattr(ep_buy_mo_prob, "boundaries", []), dtype=int
                    )
                    _p_values = np.asarray(
                        getattr(ep_buy_mo_prob, "p_values", []), dtype=float
                    )
                    _types = msg_df["Type"].values
                    if (
                        len(_boundaries) >= 2
                        and len(_p_values) >= 1
                        and (len(_types) == len(mm_df))
                    ):
                        _true_m_hat = np.full(len(mm_df), np.nan, dtype=float)
                        _true_regime_age = np.zeros(len(mm_df), dtype=float)
                        _true_regime_len = np.full(len(mm_df), np.nan, dtype=float)
                        _n_mo_seen = 0
                        for _i, _typ in enumerate(_types):
                            if int(_typ) == 1:
                                _n_mo_seen += 1
                            _mo_idx = max(_n_mo_seen - 1, 0)
                            while _mo_idx >= _boundaries[-1] and hasattr(
                                ep_buy_mo_prob, "__call__"
                            ):
                                ep_buy_mo_prob(_mo_idx)
                                _boundaries = np.asarray(
                                    getattr(ep_buy_mo_prob, "boundaries", []), dtype=int
                                )
                                _p_values = np.asarray(
                                    getattr(ep_buy_mo_prob, "p_values", []), dtype=float
                                )
                            _reg_idx = (
                                int(np.searchsorted(_boundaries, _mo_idx, side="right"))
                                - 1
                            )
                            _reg_idx = min(max(_reg_idx, 0), len(_boundaries) - 2)
                            _p_idx = min(_reg_idx, len(_p_values) - 1)
                            _true_m_hat[_i] = 2.0 * float(_p_values[_p_idx]) - 1.0
                            if _n_mo_seen <= 0:
                                continue
                            _reg_start = int(_boundaries[_reg_idx])
                            _reg_end = int(_boundaries[_reg_idx + 1])
                            _true_regime_age[_i] = float(_mo_idx - _reg_start + 1)
                            _true_regime_len[_i] = float(_reg_end - _reg_start)
                if _true_m_hat is not None:
                    ax_bayes[0].plot(
                        _x,
                        _true_m_hat,
                        linewidth=0.8,
                        color="blue",
                        alpha=0.9,
                        label="true m",
                    )
                ax_bayes[0].plot(
                    _x,
                    mm_df["MM_Bayes_M_Hat"].values,
                    linewidth=0.8,
                    color="red",
                    alpha=0.9,
                    label="Bayes m_hat",
                )
                if "MM_MO_Flow_EWMA" in mm_df.columns:
                    ax_bayes[0].plot(
                        _x,
                        mm_df["MM_MO_Flow_EWMA"].values,
                        linewidth=0.7,
                        color="darkorange",
                        alpha=0.75,
                        linestyle="--",
                        label="slow EWMA",
                    )
                if "MM_MO_Flow_Fast_EWMA" in mm_df.columns:
                    ax_bayes[0].plot(
                        _x,
                        mm_df["MM_MO_Flow_Fast_EWMA"].values,
                        linewidth=0.7,
                        color="green",
                        alpha=0.65,
                        linestyle=":",
                        label="fast EWMA",
                    )
                ax_bayes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax_bayes[0].set_ylim(-1.05, 1.05)
                ax_bayes[0].set_ylabel("m_hat")
                ax_bayes[0].legend(loc="upper right", frameon=False, fontsize=8)
                ax_bayes[1].plot(
                    _x,
                    mm_df["MM_Bayes_CP_Prob"].values,
                    linewidth=0.8,
                    color="crimson",
                    alpha=0.9,
                    label=f"P(r<={BAYES_FLOW_CP_WINDOW})",
                )
                ax_bayes[1].set_ylim(-0.05, 1.05)
                ax_bayes[1].set_ylabel("CP mass")
                ax_bayes[1].legend(loc="upper right", frameon=False, fontsize=8)
                ax_bayes[2].plot(
                    _x,
                    mm_df["MM_Bayes_Run_Length"].values,
                    linewidth=0.8,
                    color="steelblue",
                    label="Bayes E[run]",
                )
                if _true_regime_age is not None:
                    ax_bayes[2].plot(
                        _x,
                        _true_regime_age,
                        linewidth=0.8,
                        color="black",
                        alpha=0.8,
                        linestyle="--",
                        label="true age",
                    )
                if _true_regime_len is not None:
                    ax_bayes[2].step(
                        _x,
                        _true_regime_len,
                        where="post",
                        linewidth=0.8,
                        color="gray",
                        alpha=0.7,
                        linestyle=":",
                        label="true regime length",
                    )
                ax_bayes[2].axhline(
                    float(BAYES_FLOW_TAU_R),
                    color="gray",
                    linewidth=0.6,
                    linestyle="-.",
                    label="nominal tau",
                )
                ax_bayes[2].set_ylabel("run length (MO)")
                ax_bayes[2].legend(loc="upper right", frameon=False, fontsize=8)
                ax_bayes[3].plot(
                    _x,
                    mm_df["MM_Bayes_Uncertainty"].values,
                    linewidth=0.8,
                    color="darkorange",
                )
                ax_bayes[3].set_ylim(bottom=0.0)
                ax_bayes[3].set_ylabel("uncert.")
                ax_bayes[3].set_xlabel("Simulation Step")
                fig_bayes.suptitle(f"Bayesian Flow Metrics -- Episode {ep + 1}")
                plt.tight_layout()
                plt.show()
                plt.close(fig_bayes)
        if EARLY_STOPPING and ep >= EARLY_STOP_MIN_EPISODES:
            if len(episode_stats) >= EARLY_STOP_WINDOW:
                _es_pnls = [e["final_pnl"] for e in episode_stats[-EARLY_STOP_WINDOW:]]
                _es_current_ma = float(np.mean(_es_pnls))
                if _es_current_ma > _es_best_ma:
                    _es_best_ma = _es_current_ma
                    _es_best_ep = ep
                if _es_best_ma > 0:
                    _es_dropped = _es_current_ma < _es_best_ma * (1.0 - EARLY_STOP_DROP)
                else:
                    _es_dropped = (
                        _es_current_ma
                        < _es_best_ma - abs(_es_best_ma) * EARLY_STOP_DROP
                    )
                if _es_dropped:
                    print(f"\n{'!' * 60}")
                    print(f"EARLY STOPPING at episode {ep + 1}")
                    print(
                        f"  MA{EARLY_STOP_WINDOW} PnL peaked at {_es_best_ma:+.4f} (ep {_es_best_ep + 1})"
                    )
                    print(f"  Current MA{EARLY_STOP_WINDOW} PnL: {_es_current_ma:+.4f}")
                    print(f"  Drop threshold: {EARLY_STOP_DROP:.0%}")
                    print(f"{'!' * 60}\n")
                    _es_triggered = True
                    break
    if _es_triggered:
        print(
            f"Training stopped early at episode {ep + 1}/{N_EPISODES} (peak MA{EARLY_STOP_WINDOW} PnL was {_es_best_ma:+.4f} at ep {_es_best_ep + 1})"
        )
    else:
        print(f"Training completed: {N_EPISODES} episodes")
    if _best_ckpt_ep >= 0 and _best_ckpt_path is not None:
        print(
            f"[CHECKPOINT] Best checkpoint summary (criterion=MA{BEST_CKPT_WINDOW}): best MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f} at ep {_best_ckpt_ep + 1}: {_best_ckpt_path}"
        )
    CKPT_DIR = "checkpoints"
    _train_suffix = _build_train_suffix()
    ckpt_name = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_final.pt"
    CKPT_PATH = _safe_checkpoint_filename(os.path.join(CKPT_DIR, ckpt_name))
    meta = {
        "USE_PURE_MM": bool(USE_PURE_MM),
        "best_params": dict(best_params),
        "pure_mm_offsets": pure_mm_offsets if USE_PURE_MM else None,
        "lam": lam,
        "mu": mu,
        "delta": delta,
        "GLOBAL_SEED": GLOBAL_SEED,
        "EPISODE_LENGTH": EPISODE_LENGTH,
        "ITER_TO_EQUILIBRIUM": ITER_TO_EQUILIBRIUM,
        "N_EPISODES": N_EPISODES,
        "episodes_completed": ep + 1,
        "early_stopped": _es_triggered,
        "per_alpha_last_episode": ep + 1,
        "per_beta_last_episode": ep + 1,
        "USE_REGIME_SWITCH": bool(USE_REGIME_SWITCH),
        "REGIME_DISTRIBUTION": REGIME_DISTRIBUTION,
        "REGIME_ALPHA": REGIME_ALPHA,
        "REGIME_L_MIN": REGIME_L_MIN,
        "REGIME_EXP_TAU": REGIME_EXP_TAU,
        "USE_RANDOM_TAU_REGIME": bool(USE_RANDOM_TAU_REGIME),
        "RANDOM_TAU_GRID": list(RANDOM_TAU_GRID),
        "RANDOM_TAU_PROBS_START": list(RANDOM_TAU_PROBS_START),
        "RANDOM_TAU_PROBS_END": list(RANDOM_TAU_PROBS_END),
        "RANDOM_TAU_RAMP_FRAC": float(RANDOM_TAU_RAMP_FRAC),
        "USE_DRO_TAU": bool(USE_DRO_TAU),
        "DRO_TAU_MIX_PROB": float(DRO_TAU_MIX_PROB),
        "DRO_POOL_SIZE": int(DRO_POOL_SIZE),
        "DRO_BETA": float(DRO_BETA),
        "DRO_BETA_START": float(DRO_BETA_START),
        "DRO_BETA_END": float(DRO_BETA_END),
        "DRO_ETA": float(DRO_ETA),
        "DRO_EPS_UNIFORM": float(DRO_EPS_UNIFORM),
        "DRO_EPS_UNIFORM_START": float(DRO_EPS_UNIFORM_START),
        "DRO_EPS_UNIFORM_END": float(DRO_EPS_UNIFORM_END),
        "DRO_RAMP_FRAC": float(DRO_RAMP_FRAC),
        "DRO_MAX_PROB": None if DRO_MAX_PROB is None else float(DRO_MAX_PROB),
        "DRO_REFRESH_EVERY": int(DRO_REFRESH_EVERY),
        "DRO_REFRESH_FRAC": float(DRO_REFRESH_FRAC),
        "DRO_BUFFER_MO_EVENTS": int(DRO_BUFFER_MO_EVENTS),
        "DRO_PERSISTENT_MIX_PROB": float(DRO_PERSISTENT_MIX_PROB),
        "DRO_PERSISTENT_RHO": float(DRO_PERSISTENT_RHO),
        "REGIME_P_LO": REGIME_P_LO,
        "REGIME_P_HI": REGIME_P_HI,
        "EWMA_ALPHA": float(EWMA_ALPHA),
        "FAST_FLOW_EWMA_ALPHA": None
        if FAST_FLOW_EWMA_ALPHA is None
        else float(FAST_FLOW_EWMA_ALPHA),
        "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
        "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
        "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
        "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
        "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
        "USE_ADVERSARIAL": bool(False),
        "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
        "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
        "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
        "USE_MIXTURE": bool(USE_MIXTURE),
        "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
        "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
        "FORCE_BAYES_TAU_R_FIXED": bool(FORCE_BAYES_TAU_R_FIXED),
        "USE_TRUNCATED_BETA": bool(USE_TRUNCATED_BETA),
        "BAYES_FLOW_PRIOR_MODE": BAYES_FLOW_PRIOR_MODE,
        "BAYES_FLOW_CP_WINDOW": int(BAYES_FLOW_CP_WINDOW),
        "BAYES_FLOW_FEATURE_KEYS": list(BAYES_FLOW_FEATURE_KEYS),
        "PLOT_FILL_IMBALANCE": bool(PLOT_FILL_IMBALANCE),
        "PLOT_FILL_IMBALANCE_INTERVAL": int(PLOT_FILL_IMBALANCE_INTERVAL),
        "PLOT_BAYES_METRICS": bool(PLOT_BAYES_METRICS),
        "PLOT_BAYES_METRICS_INTERVAL": int(PLOT_BAYES_METRICS_INTERVAL),
        "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
        "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
        "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
        "USE_FILL_IMBALANCE": bool(USE_FILL_IMBALANCE),
        "USE_QUOTE_EXPOSURE_IMBALANCE": bool(USE_QUOTE_EXPOSURE_IMBALANCE),
        "ACTIVE_FILL_IMBALANCE_FEATURE": _active_fill_imbalance_feature_tag(),
        "CHECKPOINT_FEATURE_TAG": _checkpoint_feature_tag(),
        "USE_G_CLIP": bool(USE_G_CLIP),
        "USE_PER_PRIORITY_CLIP": bool(USE_PER_PRIORITY_CLIP),
        "WARMSTART_CKPT": WARMSTART_CKPT,
        "USE_FINE_TUNING_TECHNIQUES": bool(USE_FINE_TUNING_TECHNIQUES),
        "ANCHOR_LAMBDA": float(ANCHOR_LAMBDA),
        "LR_FEATURE_FACTOR": float(LR_FEATURE_FACTOR),
        "REPLAY_RESTORE_FRACTION": float(REPLAY_RESTORE_FRACTION),
        "USE_ANCHOR_DECAY": bool(False),
        "USE_CURRICULUM_TIED_ANCHOR": bool(False),
        "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
        "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
        "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
        "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
        "USE_ADV_TAU": bool(False),
        "ADV_TAU_MODE": ADV_TAU_MODE,
        "ADV_TAU_OBJECTIVE": ADV_TAU_OBJECTIVE,
        "ADV_FREEZE_MM": ADV_FREEZE_MM,
        "ADV_FREEZE_ADV": ADV_FREEZE_ADV,
        "ADV_WARMUP_EPISODES": ADV_WARMUP_EPISODES,
    }
    save_deep_rl_checkpoint(deep_controller, CKPT_PATH, meta)
    if SAVE_FINAL_REPLAY_BUFFER:
        _replay_save_path = CKPT_PATH.replace(".pt", "_replay.pt")
        _n_saved = deep_controller.save_replay_buffer(_replay_save_path)
        print(
            f"[CHECKPOINT] Saved replay buffer ({_n_saved} transitions) to {_replay_save_path}"
        )
    else:
        print(
            "[CHECKPOINT] Skipped final replay buffer save (SAVE_FINAL_REPLAY_BUFFER=False)"
        )
