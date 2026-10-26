"""Load the paper DQN checkpoints, preserving the original recovery rules."""

from typing import Optional
import torch
from dqn_distributional_with_throttle import DeepRLController


def make_controller_from_checkpoint(
    ckpt_path: str,
    log_dir: str = "runs_spyder/deep_mm_eval",
    *,
    use_time_update: Optional[bool] = None,
    min_time_interval: Optional[float] = None,
    inv_limit_override: Optional[int] = None,
):
    """
    Load a DeepRLController from a training checkpoint and configure it
    for pure evaluation (no learning, greedy actions).

    The checkpoint is expected to contain:
        ckpt["q_net"]         — state_dict of the Q-network
        ckpt["target_net"]    — state_dict of the target network (optional)
        ckpt["meta"]          — dict with training config:
            meta["best_params"]      — hyperparameters used during training
            meta["USE_PURE_MM"]      — whether training used pure-MM action space
            meta["pure_mm_offsets"]   — list of (bid_off, ask_off) tuples

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt checkpoint file.
    log_dir : str
        TensorBoard log directory (unused in eval, but required by constructor).

    Returns
    -------
    DeepRLController
        Controller in eval mode with loaded weights, epsilon=0, learning disabled.
    """
    # weights_only=True is safe if checkpoint meta contains only Python builtins.
    # If you ever get missing meta/best_params, set weights_only=False.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    meta = ckpt.get("meta", {})
    bp = meta.get("best_params", {})

    # Whether training was pure-mm or generic is saved in the checkpoint meta.
    USE_PURE_MM = bool(meta.get("USE_PURE_MM", False))
    pure_mm_offsets = meta.get("pure_mm_offsets", None)

    # Auto-detect architecture / feature flags from checkpoint metadata
    # or raw state-dict keys.
    #
    # Important:
    # - Standard trunk checkpoint:
    #       feature.0.weight / feature.0.bias
    # - Fully-noisy trunk checkpoint:
    #       feature.0.w_mu / feature.0.w_sigma / ...
    #
    # The eval factory must rebuild the controller with the SAME trunk
    # architecture as training, otherwise load_state_dict() fails with
    # "missing feature.0.weight / unexpected feature.0.w_mu".
    q_sd = ckpt.get("q_net", {})
    ckpt_has_noisy_params = any(
        (".w_mu" in k) or (".w_sigma" in k) for k in q_sd.keys()
    )
    ckpt_is_fully_noisy = any(
        k.startswith("feature.") and ".w_mu" in k for k in q_sd.keys()
    )

    # Auto-detect flow/imbalance feature flags from checkpoint metadata or,
    # for older checkpoints, from the first-layer input dimension.
    use_flow_signal = bool(meta.get("USE_FLOW_EWMA", False))
    use_fast_flow_signal = bool(
        meta.get(
            "USE_FAST_FLOW_EWMA",
            meta.get("USE_MO_FLOW_FAST_EWMA", False),
        )
    )
    use_bayes_flow_signal = bool(meta.get("USE_BAYES_FLOW_SIGNAL", False))
    bayes_flow_feature_keys = meta.get("BAYES_FLOW_FEATURE_KEYS", None)
    if bayes_flow_feature_keys is not None:
        bayes_flow_feature_keys = list(bayes_flow_feature_keys)
    elif use_bayes_flow_signal:
        # Phase-B regime agents trained with the current runner use these two
        # features. Older checkpoints without explicit metadata should not
        # silently fall back to the controller's four-feature Bayes default.
        bayes_flow_feature_keys = ["bayes_m_hat", "bayes_expected_run_length"]
    use_fill_imbalance = bool(meta.get("USE_FILL_IMBALANCE", False))
    if (
        not use_flow_signal
        or not use_fast_flow_signal
        or not use_bayes_flow_signal
        or not use_fill_imbalance
    ) and q_sd:
        first_weight_key = (
            "feature.0.weight" if "feature.0.weight" in q_sd else "feature.0.w_mu"
        )
        if first_weight_key in q_sd:
            actual_input_dim = q_sd[first_weight_key].shape[1]
            if USE_PURE_MM:
                max_offset = (
                    max(max(abs(b), abs(a)) for b, a in pure_mm_offsets)
                    if pure_mm_offsets
                    else 1
                )
                expected_dim = 2 + 2 * (max_offset + 1)
            else:
                expected_dim = 6
            extra = actual_input_dim - expected_dim
            explicit_extra = (
                int(use_flow_signal)
                + int(use_fast_flow_signal)
                + (len(bayes_flow_feature_keys or []) if use_bayes_flow_signal else 0)
                + int(use_fill_imbalance)
            )

            if (
                "USE_BAYES_FLOW_SIGNAL" not in meta
                and "BAYES_FLOW_FEATURE_KEYS" not in meta
            ):
                # Legacy checkpoints before Bayes metadata used the extra dims
                # in this order: slow flow, fast flow, fill imbalance.
                known_extra = (
                    int(use_flow_signal)
                    + int(use_fast_flow_signal)
                    + int(use_fill_imbalance)
                )
                if extra >= 1 and not use_flow_signal:
                    use_flow_signal = True
                if extra >= 2:
                    if (
                        "USE_FILL_IMBALANCE" in meta
                        and meta.get("USE_FILL_IMBALANCE")
                        and not use_fill_imbalance
                    ):
                        use_fill_imbalance = True
                    elif not use_fast_flow_signal and known_extra < extra:
                        use_fast_flow_signal = True
                    elif not use_fill_imbalance and known_extra < extra:
                        use_fill_imbalance = True
                if extra >= 3 and not use_fill_imbalance:
                    use_fill_imbalance = True
                explicit_extra = (
                    int(use_flow_signal)
                    + int(use_fast_flow_signal)
                    + int(use_fill_imbalance)
                )

            if extra > 0:
                print(
                    f"[Auto-detect] use_flow_signal={use_flow_signal}, "
                    f"use_fast_flow_signal={use_fast_flow_signal}, "
                    f"use_bayes_flow_signal={use_bayes_flow_signal}, "
                    f"bayes_flow_feature_keys={bayes_flow_feature_keys}, "
                    f"use_fill_imbalance={use_fill_imbalance} "
                    f"(input_dim={actual_input_dim} vs base={expected_dim})"
                )
            if explicit_extra != extra:
                raise RuntimeError(
                    "Checkpoint feature metadata does not match the Q-network input dimension: "
                    f"actual_input_dim={actual_input_dim}, base_dim={expected_dim}, "
                    f"expected_extra={explicit_extra}, observed_extra={extra}, "
                    f"use_flow_signal={use_flow_signal}, "
                    f"use_fast_flow_signal={use_fast_flow_signal}, "
                    f"use_bayes_flow_signal={use_bayes_flow_signal}, "
                    f"bayes_flow_feature_keys={bayes_flow_feature_keys}, "
                    f"use_fill_imbalance={use_fill_imbalance}."
                )

    # Rebuild controller with the same hyperparameters used during training.
    # (LR, replay_capacity, etc. are irrelevant in eval but required by the constructor.)
    ctrl_use_time_update = (
        bool(bp.get("use_time_update", False))
        if use_time_update is None
        else bool(use_time_update)
    )
    ctrl_min_time_interval = (
        float(bp.get("min_time_interval", 1.0))
        if min_time_interval is None
        else float(min_time_interval)
    )
    ctrl_inv_limit = bp.get("inv_limit", None)
    if inv_limit_override is not None:
        ctrl_inv_limit = int(inv_limit_override)

    ctrl = DeepRLController(
        level_offset=0,
        n_actions=(
            len(pure_mm_offsets) if USE_PURE_MM else int(bp.get("n_actions", 6))
        ),
        gamma=float(bp.get("gamma", 0.99)),
        lr=float(bp.get("lr", 1e-4)),
        epsilon_start=float(bp.get("epsilon_start", 0.0)),
        epsilon_min=float(bp.get("epsilon_min", 0.0)),
        epsilon_decay=float(bp.get("epsilon_decay", 1.0)),
        batch_size=int(bp.get("batch_size", 128)),
        replay_capacity=int(bp.get("replay_capacity", 1_000)),
        target_update_steps=int(bp.get("target_update_steps", 1000)),
        use_sarsa=bool(bp.get("use_sarsa", False)),
        use_double=bool(bp.get("use_double", True)),
        use_dueling=bool(bp.get("use_dueling", False)),
        use_prioritized_experience=bool(bp.get("use_per", False)),
        use_noisy_net=bool(bp.get("use_noisy_net", False) or ckpt_has_noisy_params),
        use_fully_noisy=bool(ckpt_is_fully_noisy),
        n_neurons=int(bp.get("n_neurons", 128)),
        n_hidden=int(bp.get("n_hidden", 1)),
        n_steps=int(bp.get("n_steps", 1)),
        # PER schedule: use actual training values from checkpoint, not defaults.
        per_alpha_start=float(bp.get("per_alpha_start", 0.6)),
        per_alpha_end=float(bp.get("per_alpha_end", 0.4)),
        per_alpha_last_episode=int(
            meta.get("per_alpha_last_episode", meta.get("N_EPISODES", 100))
        ),
        per_beta_start=float(bp.get("per_beta_start", 0.4)),
        per_beta_end=float(bp.get("per_beta_end", 1.0)),
        per_beta_last_episode=int(
            meta.get("per_beta_last_episode", meta.get("N_EPISODES", 100))
        ),
        # Distributional DQN (C51) settings must match training architecture.
        use_distributional=bool(bp.get("use_distributional", False)),
        v_min=float(bp.get("dist_v_min", bp.get("v_min", -10.0))),
        v_max=float(bp.get("dist_v_max", bp.get("v_max", 10.0))),
        atoms=int(bp.get("dist_atoms", bp.get("atoms", 51))),
        log_dir=log_dir,
        pure_mm=USE_PURE_MM,
        inv_limit=ctrl_inv_limit,
        pure_mm_offsets=pure_mm_offsets,
        # Evaluation throttle can be forced independently from the checkpoint.
        use_time_update=ctrl_use_time_update,
        min_time_interval=ctrl_min_time_interval,
        use_mdp=True,
        use_flow_signal=use_flow_signal,
        use_fast_flow_signal=use_fast_flow_signal,
        use_bayes_flow_signal=use_bayes_flow_signal,
        bayes_flow_feature_keys=bayes_flow_feature_keys,
        use_fill_imbalance=use_fill_imbalance,
    )

    # Load trained weights into the controller's networks.
    ctrl.q_net.load_state_dict(ckpt["q_net"])
    if (
        "target_net" in ckpt
        and hasattr(ctrl, "target_net")
        and ctrl.target_net is not None
    ):
        ctrl.target_net.load_state_dict(ckpt["target_net"])

    # Force pure evaluation mode:
    #   - Disable learning updates (no replay buffer sampling, no gradient steps)
    #   - Set epsilon to 0 for fully greedy action selection
    #   - Switch networks to eval mode (disables dropout, batchnorm training, etc.)
    ctrl.enable_learning = False
    ctrl.epsilon = 0.0
    ctrl.q_net.eval()
    if hasattr(ctrl, "target_net") and ctrl.target_net is not None:
        ctrl.target_net.eval()

    return ctrl
