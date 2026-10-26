"""Paper training entry point. Run with make train-a, train-b or train-c.

Outputs are relative to the launch directory. Historical training metadata
is recorded in config/paper_experiments.json. Experimental ADR, EWC and
legacy adversaries and post-training animations are excluded.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
    }
)
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
    from MM_LOB_SIM import (
        reward_spread_capture_inv_quadratic,
        reward_pnl_dampened_inv_quadratic,
    )

    INV_PENALTY_COEFF = 0.001
    USE_DAMPENED_REWARD = True
    WARMSTART_CKPT = None
    FORCE_NOISY_NET = True
    RESET_SIGMAS = True
    USE_FACTORED_NOISE = True
    USE_FULLY_NOISY = True
    NOISY_WEIGHT_DECAY = 0.0
    EXPERIMENT_TAG = "g0p999"
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
                prefix = f".{os.path.basename(path)}."
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

    def save_deep_rl_checkpoint(controller, path: str, meta: dict):
        """
        Save a Deep RL training checkpoint to disk.

        The checkpoint contains everything needed to resume training or to
        deploy the trained policy in evaluation mode:

            meta       — hyperparameters, LOB config, and flags needed to
                         reconstruct the DeepRLController from scratch.
            q_net      — state_dict of the online Q-network (the trained
                         policy weights).
            epsilon    — current exploration rate (only relevant for
                         ε-greedy; ignored if NoisyNets are enabled).
            target_net — state_dict of the frozen target network (optional;
                         only present if the controller has one).
            optimizer  — optimizer state_dict (learning rates, momentum
                         buffers, Adam second-moment estimates, etc.).
                         Needed to resume training without a warm-up phase.

        Parameters
        ----------
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
    print("=================================\n")
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
    print("============================\n")
    USE_PURE_MM = True
    USE_INTRA_EVENT_GAMMA = True
    PURE_MM_OFFSETS = [(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    best_params = {
        "lr": 0.0015,
        "weight_decay": 0.01,
        "lr_decay_type": "linear",
        "gamma": 0.999,
        "epsilon_start": 0.05,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "epsilon_decay_type": "linear",
        "batch_size": 128,
        "target_update_steps": 2000,
        "inv_limit": 8,
        "n_neurons": 256,
        "n_hidden": 1,
        "activation": "relu",
        "elu_alpha": None,
        "replay_capacity": 150000,
        "n_steps": 5,
        "use_intra_event_gamma": USE_INTRA_EVENT_GAMMA,
        "use_sarsa": False,
        "use_double": True,
        "use_dueling": True,
        "use_distributional": True,
        "dist_v_min": -2.0,
        "dist_v_max": 2.0,
        "dist_atoms": 51,
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
        ----------
        ep_index : int
            Episode index in [0, n_episodes - 1].
        n_episodes : int
            Total number of training episodes.
        use_exp : bool
            If True  -> exponential decay
            If False -> linear decay

        Returns
        -------
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

    def get_cyclical_lr_multiplier(
        ep_index: int, period: int = 200, min_factor: float = 0.1
    ) -> float:
        """
        Compute a cosine-annealing warm-restart multiplier for the LR.

        The multiplier follows a cosine half-cycle that resets every
        `period` episodes:

            cycle_progress = (ep_index % period) / period
            multiplier = min_factor + (1 - min_factor) * 0.5 * (1 + cos(π * cycle_progress))

        At the start of each cycle the multiplier is 1.0 (warm restart);
        at the trough it drops to min_factor.

        The final effective LR is:  lr_base × multiplier
        where lr_base comes from the monotonic decay schedule.

        Parameters
        ----------
        ep_index : int
            Current episode index.
        period : int
            Length of one cosine half-cycle in episodes.
        min_factor : float
            Minimum multiplier at the trough of each cycle.
            0.1 means the LR drops to 10% of base at the trough.

        Returns
        -------
        float
            Multiplier in [min_factor, 1.0].

        Example
        -------
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
    BEST_CKPT_WINDOW = 50
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
    USE_CYCLICAL_LR = True
    CYCLICAL_LR_PERIOD = 200
    CYCLICAL_LR_MIN_FACTOR = 0.3
    USE_INVENTORY_WALL = True
    USE_TERMINAL_BONUS = False
    USE_G_CLIP = True
    USE_PER_PRIORITY_CLIP = True
    _inv_limit_val = best_params.get("inv_limit", 8)
    _reward_kwargs = dict(inv_penalty_coeff=INV_PENALTY_COEFF)
    if USE_INVENTORY_WALL:
        _reward_kwargs["use_inv_wall"] = True
        _reward_kwargs["inv_wall_threshold"] = _inv_limit_val / 2.0
        print(
            f"[REWARD] Inventory wall: threshold={_reward_kwargs['inv_wall_threshold']}, coeff=φ={INV_PENALTY_COEFF}"
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

    def _build_train_suffix() -> str:
        _n_act = (
            len(PURE_MM_OFFSETS) if USE_PURE_MM else best_params.get("n_actions", 6)
        )
        suffix = f"_a{_n_act}" if _n_act != 6 else ""
        if USE_FACTORED_NOISE:
            suffix += "_factored_noise"
        if USE_FULLY_NOISY:
            suffix += "_fully_noisy"
        if USE_INVENTORY_WALL:
            suffix += "_inv_wall"
        if USE_DAMPENED_REWARD:
            suffix += "_dampened_reward"
        if USE_CYCLICAL_LR:
            _clr_tag = f"_clr{int(CYCLICAL_LR_PERIOD)}f{CYCLICAL_LR_MIN_FACTOR:.2f}"
            suffix += _clr_tag.replace(".", "p")
        if str(EXPERIMENT_TAG).strip():
            _tag = str(EXPERIMENT_TAG).strip()
            suffix += _tag if _tag.startswith("_") else f"_{_tag}"
        return suffix

    _mode_suffix = _build_train_suffix()
    MODE_DIR = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_mode_suffix}"
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
            use_flow_signal=False,
            use_fast_flow_signal=False,
            use_fill_imbalance=False,
            use_g_clip=USE_G_CLIP,
            use_per_priority_clip=USE_PER_PRIORITY_CLIP,
            pure_mm=False,
            inv_limit=best_params.get("inv_limit", None),
            pure_mm_offsets=None,
            use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),
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
            use_flow_signal=False,
            use_fast_flow_signal=False,
            use_fill_imbalance=False,
            use_g_clip=USE_G_CLIP,
            use_per_priority_clip=USE_PER_PRIORITY_CLIP,
            pure_mm=True,
            inv_limit=best_params.get("inv_limit", None),
            pure_mm_offsets=pure_mm_offsets,
            use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),
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
        _ws_input_dim = _ws["q_net"]["feature.0.weight"].shape[1]
        _cur_input_dim = deep_controller.q_net.state_dict()["feature.0.weight"].shape[1]
        if _ws_input_dim != _cur_input_dim:
            print(
                f"[WARMSTART] Input dim mismatch: checkpoint={_ws_input_dim}, model={_cur_input_dim}. Padding first layer with zeros."
            )
            for net_key in ("q_net", "target_net"):
                if net_key not in _ws:
                    continue
                sd = _ws[net_key]
                w = sd["feature.0.weight"]
                pad = torch.zeros(w.shape[0], _cur_input_dim - _ws_input_dim)
                sd["feature.0.weight"] = torch.cat([w, pad], dim=1)
        _ckpt_is_noisy = any(("w_mu" in k for k in _ws["q_net"].keys()))
        _model_is_noisy = best_params["use_noisy_net"]
        if _ckpt_is_noisy and (not _model_is_noisy):
            for net_key in ("q_net", "target_net"):
                if net_key not in _ws:
                    continue
                sd = _ws[net_key]
                keys_to_rename = []
                keys_to_drop = []
                for k in list(sd.keys()):
                    if ".w_mu" in k:
                        keys_to_rename.append((k, k.replace(".w_mu", ".weight")))
                    elif ".b_mu" in k:
                        keys_to_rename.append((k, k.replace(".b_mu", ".bias")))
                    elif ".w_sigma" in k or ".b_sigma" in k:
                        keys_to_drop.append(k)
                for old_k, new_k in keys_to_rename:
                    sd[new_k] = sd.pop(old_k)
                for k in keys_to_drop:
                    del sd[k]
            print(
                "[WARMSTART] Converted NoisyLinear→Linear (w_mu→weight, b_mu→bias, dropped sigma)"
            )
        elif not _ckpt_is_noisy and _model_is_noisy:
            for net_key in ("q_net", "target_net"):
                if net_key not in _ws:
                    continue
                sd = _ws[net_key]
                keys_to_rename = []
                for k in list(sd.keys()):
                    if any((head in k for head in ("fc_value.", "fc_adv.", "fc_out."))):
                        if ".weight" in k and ".w_mu" not in k:
                            keys_to_rename.append((k, k.replace(".weight", ".w_mu")))
                        elif ".bias" in k and ".b_mu" not in k:
                            keys_to_rename.append((k, k.replace(".bias", ".b_mu")))
                for old_k, new_k in keys_to_rename:
                    sd[new_k] = sd.pop(old_k)
            print(
                "[WARMSTART] Converted Linear→NoisyLinear (weight→w_mu, bias→b_mu in heads)"
            )
        _strict = not (not _ckpt_is_noisy and _model_is_noisy)
        deep_controller.q_net.load_state_dict(_ws["q_net"], strict=_strict)
        if "target_net" in _ws and deep_controller.target_net is not None:
            deep_controller.target_net.load_state_dict(
                _ws["target_net"], strict=_strict
            )
        if _ws_input_dim != _cur_input_dim or _ckpt_is_noisy != _model_is_noisy:
            print(
                "[WARMSTART] Skipping optimizer state (architecture changed, Adam moments incompatible)."
            )
        elif "optimizer" in _ws and deep_controller.optimizer is not None:
            deep_controller.optimizer.load_state_dict(_ws["optimizer"])
        print(f"[WARMSTART] Loaded weights from: {WARMSTART_CKPT}")
        if best_params["use_noisy_net"]:
            _ckpt_had_sigma = any(
                (
                    "w_sigma" in k
                    for net_key in ("q_net", "target_net")
                    if net_key in _ws_raw_keys
                    for k in _ws_raw_keys[net_key]
                )
            )
            if _ckpt_had_sigma and (not RESET_SIGMAS):
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
                import math as _math

                _SIGMA_FRACTION = 0.05
                _reset_count = 0
                for net in (deep_controller.q_net, deep_controller.target_net):
                    if net is None:
                        continue
                    for name, param in net.named_parameters():
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
                        f"  {name:20s}  mean|w_mu|={_mu_abs:.4f}  mean|w_sigma|={_sigma_abs:.4f}  σ/μ ratio={_ratio:.1%}"
                    )
        else:
            print(
                f"[WARMSTART] NoisyNet disabled → using ε-greedy={best_params['epsilon_start']}"
            )
        del _ws
    if NOISY_WEIGHT_DECAY > 0 and best_params.get("use_noisy_net", False):
        deep_controller.setup_noisy_weight_decay(sigma_weight_decay=NOISY_WEIGHT_DECAY)
    if USE_CYCLICAL_LR:
        print(f"{'─' * 60}")
        print("PHASE A STATIONARY CONFIG")
        print("  MO flow        = stationary p_buy=0.5")
        print(
            f"  [CYCLICAL LR]  period={CYCLICAL_LR_PERIOD}, min_factor={CYCLICAL_LR_MIN_FACTOR}"
        )
        print(f"{'─' * 60}\n")
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
            print("MO flow: STATIONARY (buy_mo_prob=0.5)")
            if hasattr(deep_controller, "optimizer"):
                current_lr = deep_controller.optimizer.param_groups[0]["lr"]
                print(f"Current learning rate: {current_lr:.8f}")
        current_ep_seed = GLOBAL_SEED + 1 + ep
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
        ep_buy_mo_prob = 0.5
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
            random_seed=GLOBAL_SEED + 1 + ep,
            buy_mo_prob=ep_buy_mo_prob,
            qrm_params=qrm_params,
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
            _inv_limit = best_params.get("inv_limit", 6)
            pct_at_limit = float(np.mean(abs_inv >= _inv_limit))
        else:
            max_abs_inv = 0.0
            mean_abs_inv = 0.0
            pct_at_limit = 0.0
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
        _mm_ep_for_decay = ep + 1
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
                "total_reward": total_reward,
                "discounted_return": discounted_return,
                "epsilon": deep_controller.epsilon,
                "max_abs_inv": max_abs_inv,
                "mean_abs_inv": mean_abs_inv,
                "pct_at_limit": pct_at_limit,
            }
        )
        reward_ma_window = 10
        if len(episode_stats) >= reward_ma_window:
            last_rewards = [
                e["total_reward"] for e in episode_stats[-reward_ma_window:]
            ]
            ma_10 = float(np.mean(last_rewards))
            deep_controller.writer.add_scalar("episode/total_reward_ma_10", ma_10, ep)
        pnl_ma_10 = float("nan")
        pnl_ma_50 = float("nan")
        pnl_ma_100 = float("nan")
        if len(episode_stats) >= 10:
            _last_pnls_10 = [e["final_pnl"] for e in episode_stats[-10:]]
            pnl_ma_10 = float(np.mean(_last_pnls_10))
            deep_controller.writer.add_scalar("episode/final_pnl_ma_10", pnl_ma_10, ep)
        if len(episode_stats) >= 50:
            _last_pnls_50 = [e["final_pnl"] for e in episode_stats[-50:]]
            pnl_ma_50 = float(np.mean(_last_pnls_50))
            deep_controller.writer.add_scalar("episode/final_pnl_ma_50", pnl_ma_50, ep)
        if len(episode_stats) >= 100:
            _last_pnls_100 = [e["final_pnl"] for e in episode_stats[-100:]]
            pnl_ma_100 = float(np.mean(_last_pnls_100))
            deep_controller.writer.add_scalar(
                "episode/final_pnl_ma_100", pnl_ma_100, ep
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
        deep_controller.writer.add_scalar("episode/pct_at_inv_limit", pct_at_limit, ep)
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
        if (ep + 1) % CKPT_INTERVAL == 0:
            _ckpt_dir = "checkpoints"
            os.makedirs(_ckpt_dir, exist_ok=True)
            _train_suffix = _build_train_suffix()
            _ckpt_name = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_final.pt"
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
                "BUY_MO_PROB": 0.5,
                "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
                "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
                "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
            }
            save_deep_rl_checkpoint(
                deep_controller, os.path.join(_ckpt_dir, _ckpt_name), _ckpt_meta
            )
            print(f"[CHECKPOINT] Saved checkpoint (ep {ep + 1}): {_ckpt_name}")
        if not np.isnan(pnl_ma_best) and pnl_ma_best > _best_ckpt_ma:
            _best_ckpt_ma = pnl_ma_best
            _best_ckpt_ep = ep
            _ckpt_dir = "checkpoints"
            os.makedirs(_ckpt_dir, exist_ok=True)
            _train_suffix = _build_train_suffix()
            _intra_event_suffix = "_gevent" if USE_INTRA_EVENT_GAMMA else "_gflat"
            _best_ckpt_name = f"deep_mm_mtm_{('pure' if USE_PURE_MM else 'generic')}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}{_intra_event_suffix}_best_ma{BEST_CKPT_WINDOW}.pt"
            _best_ckpt_path = os.path.join(_ckpt_dir, _best_ckpt_name)
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
                "BUY_MO_PROB": 0.5,
                "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
                "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
                "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
            }
            save_deep_rl_checkpoint(deep_controller, _best_ckpt_path, _best_ckpt_meta)
            print(
                f"[CHECKPOINT] Updated BEST CHECKPOINT (criterion=MA{BEST_CKPT_WINDOW}) (ep {ep + 1}, MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f}): {_best_ckpt_name}"
            )
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
    CKPT_PATH = os.path.join(CKPT_DIR, ckpt_name)
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
        "BUY_MO_PROB": 0.5,
        "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
        "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
        "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
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
