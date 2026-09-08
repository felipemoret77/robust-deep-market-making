#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 02:30:00 2025

@author: felipemoret

Deep RL LOB simulation:
    - Multi-episode Online Deep SARSA / Deep Q-Learning
      MarketMaker training with continuous state features.

GENERIC STATE (pure_mm = False)
--------------------------------------------------------------
    s = [spread, asksize, bidsize, inventory, has_bid, has_ask]

GENERIC ACTIONS (discrete indices)
--------------------------------------------------------------
    0: post_bid
    1: post_ask
    2: post_bid_ask
    3: cancel_bid
    4: cancel_ask
    5: hold
    6: post_bid_ask_inside_spread   (only if n_actions >= 7 in DeepRLController)
    7: post_bid_inside_spread       (only if n_actions >= 8 in DeepRLController)
    8: post_ask_inside_spread       (only if n_actions >= 9 in DeepRLController)

PURE MM MODE (pure_mm = True)
-------------------------------------------------------------- Discrete actions index into a grid of offsets:
          a -> (bid_offset, ask_offset)  in ticks

    - The controller internally sees an extended state:
          [spread, inventory,
           pure_mm_bid_sizes[0..K],
           pure_mm_ask_sizes[0..K]]

      where K = max_offset inferred from pure_mm_offsets.

    - The MarketMaker always tries to maintain quotes:
          * if |inventory| < inv_limit (or inv_limit is None):
                -> two-sided: cancel_all_then_place_bid_ask
          * if inventory >= inv_limit:
                -> only ASK side
          * if inventory <= -inv_limit:
                -> only BID side


INSIDE-SPREAD QUOTING: HOW IT WORKS
--------------------------------------------------------------

1) Generic mode (pure_mm = False)
-------------------------------------------------------------- The discrete action space can be extended beyond the 6 core actions
      by increasing `n_actions` in the DeepRLController constructor.

    - When n_actions >= 7, the controller interprets indices as:
          0: post_bid
          1: post_ask
          2: post_bid_ask
          3: cancel_bid
          4: cancel_ask
          5: hold
          6: post_bid_ask_inside_spread   (two-sided inside-spread quoting)
      If n_actions >= 8:
          7: post_bid_inside_spread       (bid-only inside-spread quoting)
      If n_actions >= 9:
          8: post_ask_inside_spread       (ask-only inside-spread quoting)

    - Semantics in generic mode:
        * For the inside-spread variants, the controller sets internal flags
          (e.g. `use_inside_bid`, `use_inside_ask`) and applies the same
          inventory band logic as for normal posting actions:
              - If inv >= inv_limit  -> only ASK posting is allowed.
              - If inv <= -inv_limit -> only BID posting is allowed.
              - If |inv| < inv_limit -> two-sided posting (when requested).
        * The actual inside-spread prices are computed in
          DeepRLController._mm_action_from_idx(), using:
              - best bid / best ask
              - current spread
          and enforcing:
              - no crossed quotes (bid < ask)
              - minimal spread of 1 tick for two-sided quotes
              - only move inside the spread if it is wide enough
                (e.g. spread >= 2 ticks).

        * If, after inventory constraints and existing orders (has_bid/has_ask),
          there is no valid side to post, the action gracefully falls back
          to "hold" to avoid inconsistent states.

    - How to enable/use in this runner:
        * In the generic branch below, we call:
              deep_controller = DeepRLController(
                  ...,
                  n_actions=9,   # 6 core actions + 3 inside-spread variants
                  pure_mm=False,
                  ...
              )
        * Example interpretation during training:
              - If the network selects a_idx = 6:
                    -> mapped to ("place_bid_ask_inside_spread",)
              - If the network selects a_idx = 7:
                    -> mapped to ("place_bid_inside_spread",)
              - If the network selects a_idx = 8:
                    -> mapped to ("place_ask_inside_spread",)


2) PURE MM mode (pure_mm = True)
-------------------------------------------------------------- In PURE MM mode, **inside-spread behavior is controlled by offsets** in
      the grid `pure_mm_offsets`:

          pure_mm_offsets = [
              (bid_offset, ask_offset),
              ...
          ]

      where each offset is in ticks relative to best bid / best ask.

    - Interpretation of offsets:
        * bid_offset > 0  -> more passive bid (deeper in the book)
        * bid_offset = 0  -> quote at best bid
        * bid_offset < 0  -> try to move the bid **inside the spread**
                            (more aggressive, closer to or above mid),
                            while ensuring:
                                - bid_price < best_ask
                                - no crossing of the ask side

        * ask_offset > 0  -> more passive ask (further from mid)
        * ask_offset = 0  -> quote at best ask
        * ask_offset < 0  -> try to move the ask **inside the spread**
                            (more aggressive, closer to or below mid),
                            while ensuring:
                                - ask_price > best_bid
                                - no crossing of the bid side

    - The controller checks the current spread:
        * If the spread is wide enough (e.g. spread >= 2 ticks), negative
          offsets are allowed to move inside the spread while enforcing
          the constraints above.
        * If the spread is too narrow, the controller falls back to quoting
          at L1 (best bid / best ask).

    - Inventory band (PURE MM):
        * As in generic mode:
              if inv >= inv_limit  -> only ASK side is posted
              if inv <= -inv_limit -> only BID side is posted
              otherwise            -> two-sided quoting (bid + ask)
        * The chosen offsets (which may imply inside-spread) are applied
          to the current best bid / best ask or mid price to compute
          the final quotes.

    - How to enable/use in this runner:
        * In the PURE MM branch below, we set:
              pure_mm_offsets = [
                  (0, 0),
                  (0, 1),
                  (1, 0),
                  (1, 1),
                  (-1, 1),
                  (1, -1),
                  (-1, 0),
                  (0, -1),
                  (-1, -1),
              ]
          so that some actions already encode more aggressive,
          inside-spread behavior (negative offsets).
        * The network then chooses indices over this grid, and the controller
          takes care of:
              - translating offsets into valid prices
              - enforcing no-cross conditions
              - applying inventory band constraints.

--------------------------------------------------------------
How to launch TensorBoard while running this script
--------------------------------------------------------------

This script writes TensorBoard logs to a mode-specific subdirectory:

    runs_spyder/deep_mm_pure/      (if USE_PURE_MM = True)
    runs_spyder/deep_mm_generic/   (if USE_PURE_MM = False)

The log path is passed as `log_dir=` when building the DeepRLController.
The controller owns the SummaryWriter and writes episode-level metrics
from log_episode_stats().  The runner adds a few extra scalars (LR,
moving averages, inventory diagnostics) on top.

IMPORTANT:
- In this script we call:
      os.chdir(os.path.dirname(os.path.abspath(__file__)))
  so all relative log paths are resolved from that project folder.
- If you run TensorBoard from a different working directory, use the
  ABSOLUTE paths shown below.
- The runner wipes stale event files at startup (shutil.rmtree on
  RUN_ROOT), so TensorBoard will only show the current training run.

1) Open a separate terminal (outside Spyder) and run:

   tensorboard --logdir runs_spyder/deep_mm_mtm_pure_invp0.0100 --port 6010

   For absolute paths (works from any folder):

   tensorboard --logdir /Users/felipemoret/Desktop/MM_LOB_SIM/runs_spyder/deep_mm_mtm_pure_invp0.0100 --port 6010

Notes:
- This script can use reward_pnl_dampened_inv_quadratic (dampened PnL version).
- Port 6010 is reserved for this MTM runner.
- Adjust the logdir if INV_PENALTY_COEFF changes (e.g., invp0.0100 for phi=0.01).

2) Then open the dashboard in your browser:

   http://localhost:6010

TensorBoard Metrics Logged (all x-axis = episode index)
--------------------------------------------------------------

  REWARD / PNL (logged by DeepRLController.log_episode_stats):
    episode/total_reward        -- Sum of per-step rewards for the episode
    episode/discounted_return   -- G_0 = sum(gamma^t * r_t), micro-step discount
    episode/final_pnl           -- Terminal PnL (cash + mark-to-market)

  REWARD MOVING AVERAGES (logged by the runner):
    episode/total_reward_ma_10  -- 10-episode moving average of total_reward
    episode/final_pnl_ma_10    -- 10-episode moving average of final_pnl

  TD LOSS (logged by DeepRLController.log_episode_stats):
    episode/mean_td_loss        -- Mean Huber/KL loss per gradient step this episode

  EXPLORATION (logged by DeepRLController.log_episode_stats):
    episode/epsilon             -- Current epsilon (inside controller)
    episode/epsilon_outer       -- Current epsilon (logged by runner after decay)
    episode/mean_noisy_sigma    -- Mean |sigma| across NoisyLinear layers
                                  (only if use_noisy_net=True).
                                  If it drops to ~0 early, exploration died.

  PER SCHEDULE (logged by DeepRLController.log_episode_stats, only if use_per=True):
    episode/per_alpha           -- Current PER prioritization exponent
                                  (annealed: typically 0.6 -> 0.4)
    episode/per_beta            -- Current IS-correction exponent
                                  (annealed: typically 0.4 -> 1.0)

  Q-VALUE DIAGNOSTICS (logged by DeepRLController.log_episode_stats):
    episode/mean_Q_a{i}         -- Mean Q(s, a=i) across all decisions this episode,
                                  one scalar per action index i in [0, n_actions)
    episode/Q_spread            -- max(mean_Q) - min(mean_Q) across actions.
                                  Measures action differentiation.
                                  Good: 0.05-0.10 (20-25 C51 atoms of separation)
                                  Bad:  ~0 (policy collapse, all actions equal)
    episode/Q_min               -- Min Q(s,a) across all valid actions this episode
    episode/Q_max               -- Max Q(s,a) across all valid actions this episode
    episode/Q_mean              -- Mean Q(s,a) across all actions

  TD TARGET + C51 SUPPORT DIAGNOSTICS (logged by DeepRLController.log_episode_stats):
    episode/td_target_min       -- Min scalar TD target r + gamma^n * Q(s',a*)
    episode/td_target_max       -- Max scalar TD target
    episode/td_target_mean      -- Mean scalar TD target
                                  Compare with [V_min, V_max]: if td_target extremes
                                  far exceed V_min/V_max, the C51 support is too narrow.
    episode/tz_clip_frac        -- Fraction of C51 atoms clipped at [V_min, V_max]
                                  during Tz projection (only if use_distributional=True).
                                  Good: <5% and decreasing. Bad: >20% (widen support).

  INVENTORY DIAGNOSTICS (logged by the runner):
    episode/max_abs_inventory   -- Worst-case |inventory| during the episode
    episode/mean_abs_inventory  -- Mean |inventory| (lower = tighter control)
    episode/pct_at_inv_limit    -- Fraction of steps with |inv| >= inv_limit

  LEARNING RATE (logged by the runner):
    train/lr                    -- Current optimizer learning rate (after decay)

This is the most stable way to monitor DQN training when running from Spyder.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# === Paper figure style: larger labels/legends, no embedded titles ===
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})
import shutil
from functools import partial
from pathlib import Path


import os

# -- Import safety: this file is a training script, not a library --------------------------------------------------------------
# All side-effect code (chdir, log cleanup, training loop) lives inside
# the __main__ guard below so that importing this file is safe.

if __name__ == "__main__":

  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # ==============================================================
  # LOB ENGINE -- pure limit-order-book simulator (no market-maker)
  # ==============================================================
  # LOB_SIM_SANTA_FE contains the Santa-Fe-style LOB simulator used to
  # generate synthetic order flow (Poisson arrivals, geometric cancels).
  # We import its top-level entry point `simulate_LOB` which returns
  # (msg_df, ob_df, ewma_trace) for a single episode.
  # ==============================================================
  from LOB_SIM_SANTA_FE import simulate_LOB

  # ==============================================================
  # MM + LOB ENGINE -- coupled market-maker / LOB simulator
  # ==============================================================
  # MM_LOB_SIM extends the pure LOB simulator with a pluggable
  # MarketMaker agent.  `simulate_LOB_with_MM` runs one full episode,
  # returning (msg_df, ob_df, mm_df).
  #
  # We also import a collection of reward functions.  Each function has
  # the signature reward_fn(mm, lob, state_before, state_after, info)
  # and returns a scalar float.  The user can swap them in the training
  # loop to experiment with different shaping strategies.
  # ==============================================================
  from MM_LOB_SIM import simulate_LOB_with_MM
  from CONFIG_MM import lam, mu, delta, qrm_params, mean_size_LO, mean_size_MO, USE_QRM
  from regime_switching_stress_test import make_regime_schedule
  try:
      from adversary_agent import AdversaryAgent, make_adversarial_schedule
  except ModuleNotFoundError:
      AdversaryAgent = None
      make_adversarial_schedule = None
  from adversary_tau_agent import AdversaryTauAgent, TAU_GRID

  from MM_LOB_SIM import reward_spread_capture_inv_quadratic, reward_pnl_dampened_inv_quadratic

  # ==============================================================
  # REWARD CONFIG
  # ==============================================================
  INV_PENALTY_COEFF = 0.001
  USE_DAMPENED_REWARD = True       # False = spread capture reward, True = dampened PnL reward

  # -- Transfer learning: warm-start from a pre-trained checkpoint --------------------------------------------------------------
  #WARMSTART_CKPT = "checkpoints/deep_mm_mtm_pure_invp0.0010_final.pt"      # <- Set to checkpoint path to warm-start, e.g.
  #WARMSTART_CKPT = "checkpoints/deep_mm_mtm_pure_invp0.0030_final.pt"
  #WARMSTART_CKPT = "checkpoints/deep_mm_mtm_pure_invp0.0010_regime_switch_with_ewma_final.pt"
  
  
  # -- Fase B: Regime switching (warmstart from Fase A checkpoint) --------------------------------------------------------------
  WARMSTART_CKPT = None
  FORCE_NOISY_NET = True         # checkpoint has NoisyNet -- keep sigmas from checkpoint
  RESET_SIGMAS = True            # preserve pre-calibrated sigmas from Fase A
  USE_FACTORED_NOISE = True       # factored NoisyNet noise (winner in noisy-mode ablation)
  USE_FULLY_NOISY    = True       # True = NoisyLinear on ALL layers (paper), False = heads only
  NOISY_WEIGHT_DECAY = 0.0        # match checkpoint (weight_decay=0.0)
  USE_FINE_TUNING_TECHNIQUES = True
  ANCHOR_LAMBDA = 0.001           # small fixed L2 anchor; matches controller recommendation
  LR_FEATURE_FACTOR = 0.50        # trunk LR = LR_START  x  0.50 (faster trunk adaptation in Phase B)
  REPLAY_RESTORE_FRACTION = 0.0  # only used if a warmstart replay file exists
  SAVE_FINAL_REPLAY_BUFFER = False # replay dumps are large; keep opt-in by default
  CHECKPOINT_SAVE_RETRIES = 3      # retry transient filesystem write failures
  CHECKPOINT_RETRY_DELAY_SEC = 2.0

  # reward_fn_spread is built after best_params / EPISODE_LENGTH are defined (see below)

  # ==============================================================
  # ANIMATION -- LOB + MM visual replay
  # ==============================================================
  from animate_LOB_sim import animate_lob

  # ==============================================================
  # DEEP RL CONTROLLER -- DQN / SARSA with Rainbow extensions
  # ==============================================================
  # DeepRLController implements the RLController protocol and supports:
  #   - DQN or Deep SARSA (on-policy / off-policy)
  #   - Double Q-learning (decoupled selection / evaluation)
  #   - Dueling architecture (separate V and A streams)
  #   - Prioritized Experience Replay (PER with alpha/beta annealing)
  #   - NoisyNets (parameter-space exploration, replaces epsilon-greedy)
  #   - Distributional C51 (categorical return distribution)
  #   - n-step TD returns with SMDP reward aggregation
  #   - Hard throttling (event / time / TOB gating)
  # ==============================================================
  from dqn_distributional_with_throttle import DeepRLController

  import pandas as pd
  mpl.rcParams["animation.embed_limit"] = 1024  # MB (e.g. 1 GB)

  # Set the maximum number of columns to "None" (unlimited)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.width', None)

  import torch

  _SCRIPT_DIR = Path(__file__).resolve().parent
  VIZ_OUTPUT_DIR = Path(
      os.environ.get("MM_LOB_SIM_VIZ_OUTPUT_DIR", _SCRIPT_DIR / "viz_current_run")
  ).expanduser()
  VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  def _safe_plot_name(title: str, prefix: str = "figure") -> str:
      raw = str(title or prefix).strip().lower()
      keep = []
      for ch in raw:
          if ch.isalnum():
              keep.append(ch)
          elif ch in (" ", "_", "-", "/", "\\", "$", "{", "}", "(", ")", ",", ":"):
              keep.append("_")
      name = "".join(keep).strip("_")
      while "__" in name:
          name = name.replace("__", "_")
      return f"{prefix}_{name or 'figure'}"

  def _save_and_maybe_show(fig, title: str, prefix: str = "figure", dpi: int = 200) -> Path:
      out_path = VIZ_OUTPUT_DIR / f"{_safe_plot_name(title, prefix)}.png"
      fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
      print(f"[Plot saved] {out_path}")
      if "agg" not in plt.get_backend().lower():
          plt.show(block=False)
      return out_path

  def _save_animation_and_maybe_show(anim, filename: str = "last_episode_animation.gif", fps: int = 5) -> Path:
      out_path = VIZ_OUTPUT_DIR / filename
      try:
          anim.save(out_path, writer="pillow", fps=fps)
          print(f"[Animation saved] {out_path}")
      except Exception as exc:
          print(f"[Animation WARNING] Could not save animation to {out_path}: {exc}")
      if "agg" not in plt.get_backend().lower():
          plt.show(block=False)
      return out_path

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
              fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=dirpath)
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
                  or "file write failed" in err_txt
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

      # Include the target network if it exists (needed for seamless
      # training resumption -- avoids re-initializing from q_net).
      if hasattr(controller, "target_net") and controller.target_net is not None:
          ckpt["target_net"] = controller.target_net.state_dict()

      # Include the optimizer state (Adam momentum / second-moment buffers)
      # so that training can continue without a learning-rate warm-up phase.
      if hasattr(controller, "optimizer") and controller.optimizer is not None:
          ckpt["optimizer"] = controller.optimizer.state_dict()

      _torch_save_atomic(ckpt, path)
      print(f"[CHECKPOINT] Saved to: {path}")
    
  import random
  import os


  # --------------------------------------------------------------
  # 0) REPRODUCIBILITY SETUP
  # --------------------------------------------------------------
  # Deterministic execution is critical for debugging RL training.
  # A single non-deterministic operation (hash ordering, cuDNN algo
  # selection, thread scheduling) can make training runs diverge even
  # with the same hyperparameters.
  #
  # seed_everything() locks EVERY source of randomness we control:
  #   1. Python's built-in random module + hash seed (dict ordering)
  #   2. NumPy's global PRNG (used by the LOB simulator)
  #   3. PyTorch CPU PRNG (weight init, dropout, NoisyNets noise)
  #   4. PyTorch CUDA PRNG + cuDNN determinism (GPU training)
  # --------------------------------------------------------------
  def seed_everything(seed: int):
      """
      Lock all random number generators to a fixed seed for reproducibility.

      IMPORTANT: cuDNN deterministic mode is slower (~10-20%) because it
      disables the auto-tuner that picks the fastest convolution algorithm.
      This trade-off is acceptable for research / debugging but should be
      disabled for production deployment where speed matters.
      """
      # 1. Python built-in random + OS hash seed
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)

      # 2. NumPy global PRNG (controls LOB event sampling, epsilon-greedy, etc.)
      np.random.seed(seed)

      # 3. PyTorch CPU PRNG (controls weight initialization, NoisyLinear noise)
      torch.manual_seed(seed)

      # 4. PyTorch CUDA PRNG (all GPUs) + cuDNN deterministic mode
      if torch.cuda.is_available():
          torch.cuda.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)  # seed ALL GPUs if multi-GPU

          # Force cuDNN to use deterministic algorithms (slower but reproducible)
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False

  # --------------------------------------------------------------
  # NOTE (2026-06): Phase B reproducibility audit
  # --------------------------------------------------------------
  # Investigation of duplicate best_ma100 checkpoints across separate
  # Phase B runs (all hashing to 1ac6c70c...) confirmed this is NOT a
  # warm-start bug -- the save path correctly serialises live weights.
  # The identical hashes are a direct consequence of full determinism:
  # same GLOBAL_SEED + seed_everything() + cuDNN deterministic ->
  # bit-exact reproducible training -> identical best-MA100 episode
  # (ep 3885, MA100=+0.14395) -> identical state_dict tensors.
  #
  # To obtain distinguishable Phase B runs without altering the model
  # or training loop, override GLOBAL_SEED at launch time via the
  # MM_GLOBAL_SEED environment variable, e.g.:
  #     MM_GLOBAL_SEED=124 python DeepSarsaQRunner_REGIME.py
  # The default (123) is preserved so existing reproducible runs are
  # unaffected.
  # --------------------------------------------------------------
  GLOBAL_SEED = int(os.environ.get("MM_GLOBAL_SEED", "123"))
  seed_everything(GLOBAL_SEED)
  print(f"[System] Global Seed locked to {GLOBAL_SEED}. Deterministic Mode ON.")

  # ==============================================================
  # LOB parameters (calibrated from AMZN LOBSTER data)
  # ==============================================================
  # These parameters define the Santa-Fe-style LOB model dynamics:
  #
  #   lam   -- limit order arrival intensity (orders per unit time per
  #           tick level).  Higher lam = denser book.
  #   mu    -- cancellation intensity (probability per unit time that
  #           an existing limit order is cancelled).
  #   delta -- market order arrival intensity.  Controls how often
  #           aggressive orders sweep through the book.
  #   mo_size -- size of each market order (in lots).
  #
  #   NUMBER_TICK_LEVELS -- how many price ticks the simulator tracks
  #           on each side of the mid.  Tick 0 = best bid/ask.
  #   N_PRIORITY_RANKS  -- maximum number of orders that can queue at
  #           a single price level (FIFO priority).
  # ==============================================================

  NUMBER_TICK_LEVELS = 50
  N_PRIORITY_RANKS = 100

  # --------------------------------------------------------------
  # 1) OPTIONAL: offline synthetic LOB simulation (NO MM)
  #    Useful if you want to inspect basic LOB stats before training.
  # --------------------------------------------------------------
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
      iterations=50_000,
      iterations_to_equilibrium=10_000,
      path_save_files=None,
      label_simulation=None,
      beta_exp_weighted_return=0.0,
      intensity_exp_weighted_return=0.0,
      random_seed=GLOBAL_SEED,
  )

  print("\n=== OFFLINE ENVIRONMENT STATS ===")
  print("Env Type counts:\n", msg_df_offline["Type"].value_counts())
  print("Env Direction counts:\n", msg_df_offline["Direction"].value_counts())
  print("Spread range:", msg_df_offline["Spread"].min(), msg_df_offline["Spread"].max())
  print("MidPrice range:", msg_df_offline["MidPrice"].min(), msg_df_offline["MidPrice"].max())
  print("==============================================================\n")

  print("=== LOB SIMULATOR CONFIG ===")
  print(f"Engine             = {'QRM' if USE_QRM else 'Santa Fe'}")
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


  # ==============================================================
  # Optuna v3 study -- mm_dqn_tuning_v3_pure_mm
  # Best trial for long runs: T55 (rank #5 by objective = 41.81)
  # PnL = +0.78 | mean|inv| = 3.69 | max|inv| = 8 | %@limit = 11.6%
  # Low gamma + large net + high LR -> best scaling to 500 episodes.
  # ==============================================================


  # --------------------------------------------------------------
  # 2) Experiment configuration: choose controller mode
  # --------------------------------------------------------------
  #
  USE_PURE_MM = True    # <--- PURE MM controller (offset grid)
  #
  #USE_PURE_MM = False  # <--- GENERIC controller (discrete 6-9 actions)

  # --------------------------------------------------------------
  # Intra-decision (SMDP) reward aggregation toggle
  # --------------------------------------------------------------
  # USE_INTRA_EVENT_GAMMA controls how per-micro-step rewards are
  # summed inside ONE SMDP decision interval (i.e. between two
  # consecutive controller decisions).
  #
  #   True  (default, paper convention):
  #     r_t = Σ_n γ^n · r_{t,n}
  #     Each throttled micro-step reward is discounted by γ^k where
  #     k is the number of micro-steps since the last decision.
  #
  #   False (undiscounted intra-decision sum):
  #     r_t = Σ_n r_{t,n}
  #     All micro-step rewards inside one decision interval are
  #     added with unit weight (γ = 1 intra-event).
  #
  # NOTE: The OUTER bootstrap discount γ^{N_t} on Q(s_{t+1}, a') is
  # UNCHANGED in both modes. Only the intra-decision reward
  # aggregation flips.
  # --------------------------------------------------------------
  USE_INTRA_EVENT_GAMMA = os.environ.get("MM_USE_INTRA_EVENT_GAMMA", "true").lower() in ("true", "1", "yes")

  # Source of truth for the PURE MM action grid. The actual controller uses
  # len(PURE_MM_OFFSETS) as n_actions, and the run/checkpoint naming should
  # reflect that same effective action count.
  PURE_MM_OFFSETS = [
      (-1, -1),  # 0. Aggressive both (inside spread)
      (-1,  0),  # 1. Aggressive bid / at-best ask
      ( 0, -1),  # 2. At-best bid / aggressive ask
      ( 0,  0),  # 3. Symmetric at-best (L1)
      ( 0,  1),  # 4. At-best bid / passive ask (+1 tick)
      ( 1,  0),  # 5. Passive bid (+1 tick) / at-best ask
  ]

  # --------------------------------------------------------------
  # Best hyperparameters (Optuna-tuned baseline + manual overrides)
  # --------------------------------------------------------------
  # This dictionary centralizes ALL RL hyperparameters in one place.
  # Both GENERIC and PURE MM branches read from it, so changes here
  # propagate to both controller modes automatically.
  #
  # IMPORTANT: some keys are mode-specific:
  #   "n_actions" -- only used by the GENERIC branch (pure_mm branch
  #                 derives n_actions from len(pure_mm_offsets)).
  #
  # Rainbow components enabled via boolean flags:
  #   use_double       -- Double Q-learning (anti-overestimation)
  #   use_dueling      -- Dueling architecture (separate V + A streams)
  #   use_per          -- Prioritized Experience Replay
  #   use_noisy_net    -- NoisyNets (replaces epsilon-greedy exploration)
  #   use_distributional -- C51 (categorical return distribution)
  # --------------------------------------------------------------
  best_params = {
      # --- Source: Optuna v3 pure_mm Trial 55 (rank #5, objective = 41.81) ---
      # PnL = +0.78 | mean|inv| = 3.69 | max|inv| = 8 | %@limit = 11.6%
      # Best candidate for long runs: low gamma, large net, high LR, tight v_bound.

      # --- Core RL ---
      "lr": 1.5e-3,                # Optuna T55: 3.49e-4 (highest LR in top-5 consensus cluster)
      "weight_decay": 0.01,        # AdamW decoupled weight decay for the Q-network
      "lr_decay_type": "linear", # "linear" or "exponential"
      "gamma": 0.999,             # Signal ablation winner config: fixed gamma used in tuner
      "epsilon_start": 0.05,            # decays to epsilon_min over N_EPISODES
      "epsilon_min": 0.05,             # slightly higher exploration floor for ADR-based adaptation
      "epsilon_decay": 0.995,         # per-episode multiplicative decay
      "epsilon_decay_type": "linear",  # linear decay for fine-tuning: smooth epsilon from start->min
      "batch_size": 64,               # Optuna batch_size grid: best late-stage convergence for long runs
      "target_update_steps": 2000,    # hard target-net sync every 2000 grad steps
      "inv_limit": 8,                 # T55: inv_limit=8, agent reaches mean|inv|approx3.69
      "n_neurons": 256,               # T55: 256 neurons -- full capacity for long training
      "n_hidden": 1,                  # 1 hidden layer x 256 neurons
      "activation": "relu",           # MLP trunk activation: "relu" or "elu"
      "elu_alpha": None,              # ELU alpha; None -> PyTorch default (=1.0) when activation="elu"
      "dropout_level": 0.0,           # Optional MLP trunk dropout; 0.0 preserves the original setup
      "replay_capacity": 100_000,      # ~200 recent episodes
      "n_steps": 3,                   # Signal ablation winner config: fixed 3-step TD returns
      # Intra-decision (SMDP) γ^n reward aggregation toggle. See
      # USE_INTRA_EVENT_GAMMA module-level constant above. True reproduces the
      # paper-faithful r_t = Σ γ^n r_{t,n}; False uses an undiscounted
      # intra-event sum r_t = Σ r_{t,n}. The outer γ^{N_t} bootstrap is
      # unchanged in both modes.
      "use_intra_event_gamma": USE_INTRA_EVENT_GAMMA,

      # --- Rainbow flavor switches ---
      "use_sarsa": False,             # off-policy DQN
      "use_double": True,             # Double Q-learning (decouple select/eval)
      "use_dueling": True,            # Dueling V+A heads
      "use_distributional": True,     # C51 categorical return distribution
      "dist_v_min": -3.0,            # C51 support (matches Fase A checkpoint)
      "dist_v_max":  3.0,            # C51 support (matches Fase A checkpoint)
      "dist_atoms": 101,              # C51 number of atoms
      "use_per": True,               # Prioritized Experience Replay ON
      # NoisyNet: OFF by default for fine-tuning (sigma drifts up).
      # Set FORCE_NOISY_NET = True to override and use NoisyNet even with warmstart.
      "use_noisy_net": FORCE_NOISY_NET or (WARMSTART_CKPT is None),

      # --- PER annealing schedule ---
      # alpha controls prioritization strength: 1.0 = full priority, 0 = uniform.
      # Anneal alpha DOWN (0.6->0.4): start with moderate prioritization, relax over time.
      # beta controls importance-sampling correction: must reach 1.0 for unbiased convergence.
      # Anneal beta UP (0.4->1.0): standard schedule from Schaul et al. 2016.
      "per_alpha_start": 0.6,
      "per_alpha_end": 0.4,
      "per_beta_start": 0.4,
      "per_beta_end": 1.0,

      # --- Reward shaping ---
      "inv_penalty_coeff": INV_PENALTY_COEFF, # synced with global INV_PENALTY_COEFF

      # --- Action space (GENERIC mode only) ---
      "n_actions": 4,                 # 4 actions (grid4_passive in pure_mm mode)

      # --- Throttle gates ---
      "use_tob_update": False,        # Gate 1: react after N top-of-book changes
      "n_tob_moves": 10,              #   threshold for TOB gate
      "use_event_update": False,      # Gate 2: react after N simulation events
      "n_events": 100,                #   threshold for event gate
      "use_time_update": True,        # Gate 3: react after T sim-time seconds
      "min_time_interval": 1.0,       #   minimum seconds between decisions
  }


  # --------------------------------------------------------------
  # Learning-rate schedule (episode-based)
  # --------------------------------------------------------------
  # Initial learning rate comes from best_params.
  if WARMSTART_CKPT is not None:
      LR_START = best_params["lr"]     # Fase B: 3e-4 for fine-tuning
  else:
      LR_START = best_params["lr"]       # full LR for training from scratch

  # Final LR will be LR_START * LR_END_FACTOR.
  LR_END_FACTOR = 0.1

  def get_lr_for_episode(ep_index: int, n_episodes: int, use_exp: bool = True) -> float:
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
      progress = (ep_index + 1) / n_episodes  # goes from 0 to 1
      LR_END = LR_START * LR_END_FACTOR

      if use_exp:
          # Exponential decay:
          # LR = LR_START * (LR_END_FACTOR ** progress)
          lr = LR_START * (LR_END_FACTOR ** progress)
      else:
          # Linear decay:
          # LR = (1-progress)*LR_START + progress*LR_END
          lr = (1.0 - progress) * LR_START + progress * LR_END

      return lr

  # --------------------------------------------------------------
  # CURRICULUM SCHEDULE HELPERS
  # --------------------------------------------------------------
  # These functions compute episode-dependent hyperparameters for
  # the progressive curriculum.  They all take `progress` in [0, 1]
  # (linear interpolation from ep=0 to ep=N-1) and return the
  # parameter value for that point in training.
  #
  # Design rationale:
  #   - All schedules are CONTINUOUS (no step-function jumps) to avoid
  #     sudden shocks to the agent's value function.
  #   - All schedules are LINEAR (simplest possible ramp).  Non-linear
  #     schedules (cosine, exponential) can be added later if needed,
  #     but linear is the right starting point because we don't yet
  #     know the optimal shape -- starting simple avoids overfitting
  #     the schedule itself.
  #   - Each function is PURE (no side effects, no global mutation).
  #     The caller is responsible for applying the returned values
  #     to the controller / episode config.
  # --------------------------------------------------------------

  def get_curriculum_p_bounds(progress: float,
                              delta_start: float = 0.15,
                              delta_end: float = 0.30) -> tuple:
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
      progress = max(0.0, min(1.0, progress))  # clamp to [0, 1]
      delta = delta_start + (delta_end - delta_start) * progress
      return (0.5 - delta, 0.5 + delta)

  def get_anchor_lambda(progress: float,
                        lambda_start: float = 0.05,
                        lambda_end: float = 0.02) -> float:
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

  def get_tied_anchor_lambda(delta: float,
                             delta_max: float,
                             base: float,
                             slope: float,
                             gate: float) -> float:
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
          # Degenerate range -> always use base (log-safe fallback).
          return float(base)
      u = (float(delta) - float(gate)) / (float(delta_max) - float(gate))
      if u < 0.0:
          u = 0.0
      elif u > 1.0:
          u = 1.0
      return float(base) + float(slope) * u

  def get_cyclical_lr_multiplier(ep_index: int,
                                 period: int = 200,
                                 min_factor: float = 0.1) -> float:
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
      cycle_progress = (ep_index % period) / max(period, 1)
      return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + _math_clr.cos(_math_clr.pi * cycle_progress))

  # --------------------------------------------------------------
  # 4) MULTI-EPISODE ONLINE TRAINING CONFIG
  # --------------------------------------------------------------
  # Each episode runs a fresh LOB simulation of EPISODE_LENGTH events.
  # The first ITER_TO_EQUILIBRIUM events use the warm-up phase (the LOB
  # starts empty and needs time to build a realistic book shape before
  # the MM can meaningfully interact with it).
  #
  # N_EPISODES controls how many episodes the agent trains for.
  # Epsilon, learning rate, and PER schedules are all tied to this count.
  # --------------------------------------------------------------

  EPISODE_LENGTH = 5_000           # environment events per episode (~16 min sim time)
  ITER_TO_EQUILIBRIUM = 1000       # warm-up events (reduced proportionally, LOB stabilizes in ~500)
  N_EPISODES = 4000              # training budget (curriculum disabled, static max difficulty from ep 1)

  # Logging verbosity: when False, prints only episode number, final PnL,
  # and PnL MA(10).  When True, prints full diagnostics (epsilon, LR,
  # Q-values, TD targets, inventory stats, action usage, etc.).
  VERBOSE_LOGGING = False

  # Early stopping: stop training if the rolling PnL drops significantly
  # below its peak.  This prevents catastrophic forgetting during fine-tuning
  # by halting before the policy degrades too far.
  #   EARLY_STOP_WINDOW: number of episodes for the rolling average
  #   EARLY_STOP_DROP:   fraction drop from peak MA that triggers stop
  #                      (e.g., 0.5 = stop if MA fell 50% below peak)
  #   EARLY_STOP_MIN_EPISODES: don't check before this many episodes
  #                            (allow initial exploration/warmup)
  EARLY_STOPPING = False              # v3 config: no early stopping
  EARLY_STOP_WINDOW = 50
  EARLY_STOP_DROP = 0.35
  EARLY_STOP_MIN_EPISODES = 200

  # Best-checkpoint tracking (independent of early stopping):
  # save a dedicated checkpoint whenever MA(window) of final_pnl reaches a new
  # peak.  This prevents losing the best policy when the final episodes drift.
  BEST_CKPT_WINDOW = 100

  # Validate early stopping config
  if EARLY_STOPPING:
      assert EARLY_STOP_WINDOW > 0, f"EARLY_STOP_WINDOW must be > 0, got {EARLY_STOP_WINDOW}"
      assert 0 < EARLY_STOP_DROP <= 1.0, f"EARLY_STOP_DROP must be in (0, 1], got {EARLY_STOP_DROP}"
      assert EARLY_STOP_MIN_EPISODES >= EARLY_STOP_WINDOW, (
          f"EARLY_STOP_MIN_EPISODES ({EARLY_STOP_MIN_EPISODES}) must be >= "
          f"EARLY_STOP_WINDOW ({EARLY_STOP_WINDOW})")
  assert BEST_CKPT_WINDOW > 0, f"BEST_CKPT_WINDOW must be > 0, got {BEST_CKPT_WINDOW}"

  # --------------------------------------------------------------
  # REGIME-SWITCHING MO FLOW (Domain Randomization)
  # --------------------------------------------------------------
  # When USE_REGIME_SWITCH = True, each training episode uses a dynamic
  # buy_mo_prob that switches between regimes within the episode.
  #
  #   - Regime duration:  L ~ Pareto(alpha, L_min)  or  L ~ Exp(lambda)
  #   - Per-regime p_buy: U[p_lo, p_hi]         (symmetric around 0.5)
  #
  # REGIME_DISTRIBUTION selects the duration model:
  #   "pareto"      -- heavy-tailed (Lillo, Mike & Farmer 2005)
  #   "exponential" -- memoryless; EWMA alpha is derived from lambda (no knob)
  #
  # This forces the agent to learn policies robust to non-stationary MO flow.
  # Each episode gets a unique regime schedule (seeded for reproducibility).
  #
  # When USE_REGIME_SWITCH = False, buy_mo_prob = 0.5 (constant, original behaviour).
  # --------------------------------------------------------------
  USE_REGIME_SWITCH = True           # Fase B: regime switching exponencial
  REGIME_DISTRIBUTION = "exponential"   # "pareto" or "exponential"
  # Pareto parameters (used when REGIME_DISTRIBUTION == "pareto")
  REGIME_ALPHA      = 1.5          # Pareto shape (metaorder duration exponent)
  REGIME_L_MIN      = 10           # Pareto scale (minimum regime length in MO events)
  # Exponential parameters (used when REGIME_DISTRIBUTION == "exponential")
  REGIME_EXP_TAU    = 60           # tau: mean regime duration in MO events (rate lambda = 1/tau)
  # Common
  REGIME_P_LO       = 0.20         # static maximum-difficulty regime floor
  REGIME_P_HI       = 0.80         # static maximum-difficulty regime cap

  # --------------------------------------------------------------
  # PROGRESSIVE CURRICULUM -- gradual difficulty ramp
  # --------------------------------------------------------------
  # When USE_PROGRESSIVE_CURRICULUM = True, the regime-switching bounds
  # (p_lo, p_hi) are NOT fixed at REGIME_P_LO / REGIME_P_HI for the
  # entire run.  Instead, they follow a continuous linear ramp:
  #
  #   progress = ep / (N_EPISODES - 1)          (0.0 at ep=0, 1.0 at last ep)
  #   delta_p  = CURRICULUM_DELTA_START + (CURRICULUM_DELTA_END - CURRICULUM_DELTA_START) * progress
  #   p_lo     = 0.5 - delta_p
  #   p_hi     = 0.5 + delta_p
  #
  # Example with defaults (DELTA_START=0.15, DELTA_END=0.30):
  #   ep=0:    p in [0.35, 0.65]  -- mild asymmetry, close to stationary
  #   ep=N/2:  p in [0.275, 0.725] -- moderate asymmetry
  #   ep=N:    p in [0.20, 0.80]  -- full asymmetry (matches REGIME_P_LO/HI)
  #
  # WHY: dropping the agent straight into extreme regime switching
  # (p in [0.2, 0.8]) can cause a mid-training performance dip (the "Q3 valley"
  # observed in Fase B v3) because the policy hasn't yet learned to handle
  # large directional flow.  A gradual ramp gives the agent time to adapt
  # its value function to progressively harder regimes, reducing the shock.
  #
  # REQUIRES: USE_REGIME_SWITCH = True (validated below).
  # --------------------------------------------------------------
  USE_PROGRESSIVE_CURRICULUM = False
  CURRICULUM_DELTA_START     = 0.05  # half-width of p_buy range at ep=0     -> [0.45, 0.55]
  CURRICULUM_DELTA_END       = 0.30  # half-width of p_buy range at ramp end -> [0.20, 0.80]
  CURRICULUM_RAMP_FRAC       = 0.70  # fraction of N_EPISODES over which the ramp runs
                                     # (remaining 30% stays at DELTA_END = "hold" phase)
                                     # Example with N=2000: ramp ends at ep 1400,
                                     # eps 1400-2000 train at full [0.20, 0.80]

  # --------------------------------------------------------------
  # ADR-LITE ADAPTIVE CURRICULUM (v3 -- faithful to Akkaya et al. 2019,
  # Algorithm 1 + Table 15)
  # --------------------------------------------------------------
  # When USE_ADR_LITE_CURRICULUM = True, the linear progressive
  # curriculum above is REPLACED by an adaptive thermostat that
  # implements the ADR algorithm directly:
  #
  #   1. Each episode runs over the full curriculum interval
  #      [0.5 - delta, 0.5 + delta] (no boundary sampling).
  #   2. Per-episode metric = clip(final_pnl, -PNL_CLIP, +PNL_CLIP).
  #   3. A single shared buffer of size m collects episode metrics.
  #   4. When full, the buffer mean p_bar is compared to absolute
  #      thresholds: advance if p_bar >= t_H_abs, hold if inventory veto blocks
  #      promotion, and retreat if p_bar <= t_L_abs or the emergency inventory
  #      brake fires.
  #   5. Buffer is cleared after every decision (faithful to paper).
  #
  # No baseline calibration, no Phase A reference, no boundary
  # sampling, no per-side dual buffers. The thermostat is
  # structurally identical to Algorithm 1 of the paper, with
  # final_pnl replacing "number of successes per episode".
  #
  # Mutually exclusive with USE_PROGRESSIVE_CURRICULUM (validated below).
  # Requires USE_REGIME_SWITCH = True.
  # Does NOT require WARMSTART_CKPT (works from scratch too).
  #
  # See adr_lite.py for the full algorithm description.
  # --------------------------------------------------------------
  USE_ADR_LITE_CURRICULUM    = False  # static max difficulty from episode 1
  ADR_LITE_DELTA_START       = 0.00
  ADR_LITE_DELTA_MAX         = 0.25
  ADR_LITE_DELTA_MIN         = 0.00
  ADR_LITE_DELTA_STEP        = 0.01
  ADR_LITE_BUFFER_SIZE       = 50      # m -- episodes per decision batch (ADR paper uses 240)

  # -- Threshold mode --------------------------------------------------------------
  # "constant"
  #     Same absolute thresholds at every delta.  Simple, ADR-paper faithful.
  #
  # "linear_phase_a_anchored" (deprecated -- first attempt)
  #     t_H decays linearly from `phase_a_best_ma50 - 3sigma` to a floor.
  #     Anchors on the LUCKY PEAK of Phase A and treats `t_L` as an
  #     independent line.  Superseded by `linear_mean_gap`.
  #
  # "linear_mean_gap" (current recommendation)
  #     t_H decays linearly from `phase_a_mean_sustained` at delta_start to
  #     `ADR_LITE_T_H_ABS_END` at delta_max.  Anchors on the SUSTAINED mean
  #     of a frozen Phase A evaluation (not the lucky peak), which is a
  #     reproducible reference.  t_L is derived from t_H via a fixed
  #     statistical gap:
  #         t_L(delta) = max(convergence_floor, t_H(delta) - gap_sigma * sigma_block50)
  #     The floor ensures the retreat line never drops below break-even
  #     (agent is never asked to "accept losses").  sigma_block50 is the
  #     std of non-overlapping buffer_size-blocks of final_pnl measured
  #     offline on the frozen Phase A policy -- it is the NOISE of the
  #     exact variable the thermostat decides on.
  ADR_LITE_THRESHOLD_MODE    = "linear_mean_gap"

  # -- Constant-mode params (used iff THRESHOLD_MODE == "constant") --
  # Absolute thresholds on the per-episode performance metric.  The metric
  # is `clip(final_pnl, -PNL_CLIP, +PNL_CLIP)`.  These are PnL units (ticks).
  # Mirrors ADR's "absolute t_H/t_L on success counts" structure.
  ADR_LITE_T_H_ABS           = +0.10   # advance if avg PnL >= +0.10 ticks/episode
  ADR_LITE_T_L_ABS           =  0.00   # retreat if avg PnL <= 0 (no longer profitable)

  # -- Linear-anchored (peak) params (used iff THRESHOLD_MODE == "linear_phase_a_anchored") --
  # LEGACY -- kept for backward compatibility.  The `linear_mean_gap`
  # block below is the current design.
  ADR_LITE_PHASE_A_BEST_MA50 = None    # None -> read from WARMSTART_CKPT
  ADR_LITE_SIGMA_MA50        = 0.04
  ADR_LITE_SIGMA_MARGIN      = 3.0
  ADR_LITE_T_L_ABS_START     = 0.01
  ADR_LITE_T_L_ABS_END       = 0.01

  # -- Mean-gap linear params (used iff THRESHOLD_MODE == "linear_mean_gap") --
  #
  # These are the TWO NUMBERS that must be measured offline from a
  # frozen evaluation of the PREVIOUS CURRICULUM STEP (the reference
  # policy -- typically whatever warmstart checkpoint this run is
  # fine-tuning from).  You get them by running the return-risk frontier
  # (or equivalent long eval) at p_buy=0.5 with the reference policy
  # frozen, and computing:
  #
  #     per_ep = np.clip(final_pnl_series, -ADR_LITE_PNL_CLIP, +ADR_LITE_PNL_CLIP)
  #     prev_curriculum_mean = per_ep.mean()
  #
  #     # Non-overlapping blocks of size ADR_LITE_BUFFER_SIZE -- NOTE:
  #     # this is the std of the BLOCK MEAN, NOT the per-episode std.
  #     # The thermostat decides on buffer-mean PnL, so the sigma it sees
  #     # is the std of that buffer mean under stationary evaluation.
  #     n = len(per_ep) // ADR_LITE_BUFFER_SIZE
  #     blocks = per_ep[:n*ADR_LITE_BUFFER_SIZE].reshape(n, -1).mean(axis=1)
  #     prev_curriculum_buffer_std = blocks.std(ddof=1)
  #
  # Using the per-episode std here (by mistake) would understate the
  # effective noise by sqrt(buffer_size) and collapse the dead zone.
  #
  # These live in `adr_lite_config.py` as a single source of truth --
  # both this runner and the Optuna tuner read from the same file so
  # they can never drift.  Paste the two numbers there once, not here.
  from adr_lite_config import (
      PREV_CURRICULUM_MEAN       as ADR_LITE_PREV_CURRICULUM_MEAN,
      PREV_CURRICULUM_BUFFER_STD as ADR_LITE_PREV_CURRICULUM_BUFFER_STD,
      # -- Opt-in extension flags --------------------------------------------------------------
      # Both default to "off"/"continuous" in adr_lite_config.py, which
      # reproduces the legacy thermostat behavior exactly.  Flip either
      # of them there to experiment with the discrete-ladder or
      # multi-confirmation mechanisms without touching this file.
      DELTA_SCHEDULE_MODE        as ADR_LITE_DELTA_SCHEDULE_MODE,
      LADDER_AUTO_ALPHA          as ADR_LITE_LADDER_AUTO_ALPHA,
      LADDER_AUTO_MAX_GAP        as ADR_LITE_LADDER_AUTO_MAX_GAP,
      LADDER_AUTO_MIN_GAP        as ADR_LITE_LADDER_AUTO_MIN_GAP,
      LADDER_MANUAL              as ADR_LITE_LADDER_MANUAL,
      ADVANCE_CONFIRMATION_MODE  as ADR_LITE_ADVANCE_CONFIRMATION_MODE,
      ADVANCE_CONFIRMATION_K_MAX as ADR_LITE_ADVANCE_CONFIRMATION_K_MAX,
      ADVANCE_CONFIRMATION_GAMMA as ADR_LITE_ADVANCE_CONFIRMATION_GAMMA,
      ADVANCE_CONFIRMATION_RESET_MODE as ADR_LITE_ADVANCE_CONFIRMATION_RESET_MODE,
      generate_auto_ladder,
      make_advance_confirmation_fn,
  )

  # Gap width in units of sigma_block50.  2.0 gives a dead zone of ~32%
  # action rate at midpoint (responsive); 3.0 is more conservative
  # (~13%, less flip-flop, slower to react).
  ADR_LITE_GAP_SIGMA              = 2.0

  # Retreat floor -- t_L is clamped to NEVER fall below this value.
  # Default 0.0 means strict break-even (the curriculum will never ask
  # the agent to tolerate negative mean PnL).  Also gates the
  # convergence detector: consecutive_at_max does not increment when
  # the buffer mean is below this floor, so "converged at delta_max" is
  # only declared in-the-black.
  ADR_LITE_CONVERGENCE_FLOOR      = 0.0

  # Floor of the t_H decay at delta_max.  Set to None to DERIVE automatically
  # from the noise scale:
  #
  #   t_H_end = convergence_floor + gap_sigma  *  sigma_block50
  #
  # With the defaults (floor=0, gap_sigma=2) this gives a uniform dead
  # zone of gap_sigma * sigma across every delta -- i.e. the hardest curriculum
  # level demands EXACTLY the same statistical margin as the easiest,
  # just shifted down to absorb the reduced capability.  At delta_max the
  # retreat line falls exactly on the convergence_floor (floor just
  # touching t_L, never needing to activate).  This is the most
  # principled choice and has zero magic numbers.
  #
  # Override with an explicit positive float only if you want a
  # different philosophy (narrower dead zone at the top, or stricter
  # advance requirement).  The default (None -> derived) is the right
  # starting point.
  ADR_LITE_T_H_ABS_END            = None

  # -- Common --------------------------------------------------------------
  ADR_LITE_PNL_CLIP              = 1.0     # clip per-episode PnL to [-1, +1] before averaging
  ADR_LITE_INV_ADVANCE_VETO_MIN  = 0.05    # block advance when rolling wall-touch rate exceeds 5%
  ADR_LITE_INV_RETREAT_MIN       = 0.10    # emergency retreat brake on inventory stress
  ADR_LITE_INV_HISTORY_LEN       = 100     # rolling window for inventory veto / retreat checks

  # -- Convergence-based early stop (only meaningful with ADR-lite) -
  # The number of training episodes is no longer fixed.  Instead we
  # train until ONE of:
  #   1. convergence: curriculum reached delta_max and stayed there for
  #      N consecutive decisions ("agent fully dominated the curriculum")
  #   2. plateau:    curriculum has not advanced for K consecutive
  #      decisions ("agent hit its capacity ceiling")
  #   3. safety cap: N_EPISODES is reached as a hard wall
  #
  # Set USE_ADR_EARLY_STOP=False to disable and revert to the legacy
  # "train exactly N_EPISODES" behaviour.
  USE_ADR_EARLY_STOP                 = True
  ADR_CONVERGENCE_DECISIONS_AT_MAX   = 3      # consecutive @ delta_max -> done
  ADR_PLATEAU_MAX_DECISIONS_NO_ADV   = 15     # decisions w/o advance -> plateau
  # N_EPISODES below now functions as a safety cap.  Bump it generously.

  # --------------------------------------------------------------
  # ELASTIC WEIGHT CONSOLIDATION (Kirkpatrick et al. 2017)
  # --------------------------------------------------------------
  # When USE_EWC_ANCHOR = True, the existing L2 anchor loss is upgraded
  # to a Fisher-weighted penalty:
  #
  #     L_EWC = (lambda/2)  *  Sigma_i F_ii  *  (theta_i - theta*_i)^2
  #
  # F_ii is the diagonal Fisher Information Matrix estimated under the
  # frozen Phase A policy (Boltzmann log-likelihood method; see ewc.py
  # for the estimator details).  EWC lets us use a larger lambda without
  # destroying plasticity because the penalty is selective -- only
  # parameters that matter to the Phase A policy get strong anchoring.
  #
  # Requires:
  #   - WARMSTART_CKPT set (anchor weights come from the checkpoint)
  #   - USE_FINE_TUNING_TECHNIQUES = True (anchor machinery is active)
  #
  # The Fisher is estimated ONCE before training starts, cached to disk,
  # and reused across runs.
  # --------------------------------------------------------------
  USE_EWC_ANCHOR          = False          # isolate ADR first; add EWC back only if forgetting is the bottleneck
  EWC_FISHER_PATH         = "calibration/phase_a_fisher_ewc.pt"
  EWC_N_FISHER_SAMPLES    = 1000            # (s, a) pairs used to estimate Fisher
  EWC_N_COLLECT_EPISODES  = 20              # eps of Phase A rollouts at p=0.5 to fill the sample pool
  EWC_LAMBDA              = 0.5             # lambda for EWC (much larger than plain L2's 0.05)
  EWC_TEMPERATURE         = 1.0             # tau of the softmax policy used for Fisher
  EWC_CALIBRATION_P_BUY   = 0.5             # Phase A environment (symmetric)

  # With progressive curriculum enabled, only activate early stopping after
  # the difficulty ramp has finished (plus one MA window). This avoids
  # prematurely stopping during the intended transition to harder regimes.
  if USE_PROGRESSIVE_CURRICULUM:
      EARLY_STOP_MIN_EPISODES = int(N_EPISODES * CURRICULUM_RAMP_FRAC) + EARLY_STOP_WINDOW

  # --------------------------------------------------------------
  # ANCHOR DECAY -- linear decay of L2 anchor strength
  # --------------------------------------------------------------
  # When USE_ANCHOR_DECAY = True, the L2 anchor regularization strength
  # (lambda_anchor) decays linearly from ANCHOR_LAMBDA_START to
  # ANCHOR_LAMBDA_END over the training run:
  #
  #   lambda(progress) = START + (END - START) * progress
  #
  # Example with defaults (START=0.05, END=0.02):
  #   ep=0:  lambda = 0.05  -- strong pull toward Fase A weights (preserve)
  #   ep=N:  lambda = 0.02  -- relaxed pull (allow adaptation to new regimes)
  #
  # WHY: at the beginning of fine-tuning, the pre-trained weights encode
  # a good policy for stationary environments.  A strong anchor prevents
  # catastrophic forgetting in the early phase when the agent encounters
  # regime switching for the first time.  As training progresses and the
  # agent accumulates experience under regime switching, we relax the
  # anchor so the policy can fully adapt to the non-stationary dynamics
  # without being overly constrained by the stationary-trained baseline.
  #
  # REQUIRES: WARMSTART_CKPT is not None and anchor already initialised.
  # --------------------------------------------------------------
  USE_ANCHOR_DECAY       = False  # keep anchor fixed; active constant comes from ANCHOR_LAMBDA above
  ANCHOR_LAMBDA_START    = 0.05   # initial anchor strength when time-decay is enabled
  ANCHOR_LAMBDA_END      = 0.01   # final anchor strength when time-decay is enabled

  # --------------------------------------------------------------
  # CURRICULUM-TIED ANCHOR (tied to ADR delta, not time)
  # --------------------------------------------------------------
  # NOTE: this is NOT a "decay" -- the anchor strength GROWS with delta
  # (from `base` in easy territory to `base + slope` at delta_max).  The
  # naming reflects the fact that the lambda schedule is TIED to the
  # curriculum state, not any assumption about whether it increases
  # or decreases over time.  The existing USE_ANCHOR_DECAY is a
  # separate mechanism that truly decays with training progress.
  #
  # When USE_CURRICULUM_TIED_ANCHOR = True, the L2 anchor strength is
  # computed as a GATED LINEAR FUNCTION OF THE CURRENT ADR delta.  The
  # core insight: the demand for weight preservation scales with the
  # difficulty of the current curriculum step, not with elapsed time.
  # An agent retreating from delta=0.25 to delta=0.18 no longer needs strong
  # preservation, because it is back in territory it handles well.
  #
  # The formula is a piecewise linear function with a gate:
  #
  #     u(delta) = clip((delta - gate) / (delta_max - gate), 0, 1)
  #     lambda(delta) = base + slope  *  u(delta)
  #
  # Properties:
  #     lambda(delta <= gate)  = base         (constant plateau in easy/medium)
  #     lambda(delta = delta_max) = base + slope (maximum preservation at extreme)
  #     lambda is monotone non-decreasing in delta, continuous at the gate.
  #
  # Defaults (base=0.05, slope=0.10, gate=0.18) give:
  #     delta <= 0.18: lambda = 0.05 (matches the known-good constant baseline)
  #     delta = 0.20: lambda approx 0.067
  #     delta = 0.22: lambda approx 0.083
  #     delta = 0.25: lambda approx 0.108
  #     delta = 0.28: lambda approx 0.133
  #     delta = 0.30: lambda = 0.15
  #
  # Empirical motivation
  # --------------------------------------------------------------
  # The constant-lambda sweep revealed that:
  #   * lambda = 0.01 destabilises Phase A even in easy regimes
  #   * lambda = 0.05 is the sweet spot in easy/medium but permits
  #     catastrophic forgetting after prolonged exposure to the
  #     hardest regimes (delta >= 0.24)
  #   * lambda = 0.10 constant throughout is slightly too restrictive in
  #     the adaptation phase
  # The gated-tied formulation resolves all three failure modes:
  # matches the sweet spot in adaptation regimes and escalates
  # preservation exactly where forgetting becomes likely.
  #
  # Exclusivity
  # --------------------------------------------------------------
  # USE_CURRICULUM_TIED_ANCHOR and USE_ANCHOR_DECAY are MUTUALLY EXCLUSIVE.
  # Enable at most one at a time.  When both are False, the L2 anchor
  # stays fixed at ANCHOR_LAMBDA (the constant set by the fine-tuning
  # block), which is the current default.
  #
  # REQUIRES: USE_ADR_LITE_CURRICULUM = True (needs `adr_curriculum.delta`)
  # --------------------------------------------------------------
  USE_CURRICULUM_TIED_ANCHOR    = False   # ADR is disabled; keep fixed anchor strength
  TIED_ANCHOR_BASE         = 0.001   # base anchor for easy curriculum (delta <= gate)
  TIED_ANCHOR_SLOPE        = 0.004   # linear increase up to lambda=0.005 at delta_max
  TIED_ANCHOR_GATE         = 0.08    # start ramping once the curriculum leaves easy territory

  # --------------------------------------------------------------
  # CYCLICAL LEARNING RATE -- cosine annealing with warm restarts
  # --------------------------------------------------------------
  # When USE_CYCLICAL_LR = True, the base learning rate (from the
  # monotonic decay schedule) is modulated by a cosine warm-restart
  # cycle.  This can help the agent escape local optima when the
  # environment difficulty changes mid-training.
  #
  #   lr_effective = lr_base  x  (0.5 + 0.5 * cos(pi * cycle_progress))
  #
  # where cycle_progress resets to 0 at the start of each cycle.
  #
  # NOTE: with USE_PROGRESSIVE_CURRICULUM enabled, this is usually
  # unnecessary -- the gradual difficulty ramp avoids the mid-training
  # dip that cyclical LR was meant to fix.  Included for ablation.
  #
  # CAUTION: cyclical LR + strong anchor decay can cause instability
  # (LR spikes while anchor is relaxed).  Test carefully.
  # --------------------------------------------------------------
  USE_CYCLICAL_LR        = True
  CYCLICAL_LR_PERIOD     = 100    # episodes per cosine half-cycle (T_0)
  CYCLICAL_LR_MIN_FACTOR = 0.3   # minimum multiplier at cycle trough

  # -- EWMA flow signal --------------------------------------------------------------
  USE_MO_FLOW_SLOW_EWMA = False      # slow MO-flow EWMA aligned with regime duration
  USE_MO_FLOW_FAST_EWMA = False      # fast MO-flow EWMA for sharper transition timing
  FAST_FLOW_TAU_DIVISOR = 4.0       # tau_fast = tau_regime / divisor
  USE_BAYES_FLOW_SIGNAL = True     # optional Bayesian changepoint MO-flow features
  USE_MIXTURE = False               # if True, Bayes filter averages fixed internal tau hypotheses
  BAYES_MIXTURE_TAUS = [15.0, 30.0, 60.0, 120.0 , 240.0]
  USE_TRUNCATED_BETA = False        # True: truncate p_buy prior to [REGIME_P_LO, REGIME_P_HI]; False: full Beta on [0, 1]
  BAYES_FLOW_PRIOR_MODE = "truncated_beta_grid" if USE_TRUNCATED_BETA else "beta"
  BAYES_FLOW_CP_WINDOW = 5
  BAYES_FLOW_FEATURE_KEYS = ["bayes_m_hat", "bayes_expected_run_length"]
  USE_AUTO_GAMMA = False            # keep the same gamma used in Phase A
  AUTO_GAMMA_METHOD = "exact"       # "exact" (NegBin) or "approx" (exp(-p/tau))
  USE_EMPIRICAL_MO_FRAC_CALIBRATION = True
  N_MO_FRAC_CALIB_EPISODES = 5      # offline no-MM episodes before MM training
  # Backward-compatible aliases; prefer the explicit names above in new code.
  USE_FLOW_EWMA = USE_MO_FLOW_SLOW_EWMA
  USE_FAST_FLOW_EWMA = USE_MO_FLOW_FAST_EWMA
  FAST_FLOW_EWMA_ALPHA = None       # computed from regime tau unless overridden
  PLOT_EWMA          = True       # flow-EWMA diagnostic plots
  PLOT_EWMA_INTERVAL = 100        # episodes between EWMA plots
  PLOT_FILL_IMBALANCE = True        # plot fill imbalance diagnostics
  PLOT_FILL_IMBALANCE_INTERVAL = 100
  PLOT_BAYES_METRICS = True        # Bayesian flow diagnostic plots
  PLOT_BAYES_METRICS_INTERVAL = 100 # episodes between Bayesian diagnostic plots

  # --------------------------------------------------------------
  # Adversarial Tau Bandit (regime duration adversary)
  # --------------------------------------------------------------
  # When USE_ADV_TAU = True, a bandit agent selects the exponential regime
  # duration parameter tau from TAU_GRID = [8, 12, 20, 30, 45] ONCE per
  # episode.  The bandit maximises the negative of the MM's final PnL,
  # i.e., it searches for the regime persistence that hurts the MM most.
  #
  # ADV_TAU_MODE controls EWMA alpha behavior:
  #   "alpha_recalc" -- mo_flow_ewma_alpha = 1-exp(-1/tau) each episode
  #   "alpha_fixed"  -- mo_flow_ewma_alpha stays at global EWMA_ALPHA
  #
  # Mutually exclusive with USE_ADVERSARIAL and USE_REGIME_SWITCH.
  # The mode internally builds exponential regime schedules -- it does NOT
  # require USE_REGIME_SWITCH to be True.
  # --------------------------------------------------------------
  USE_ADV_TAU           = False
  ADV_TAU_MODE          = "alpha_recalc"    # "alpha_recalc" or "alpha_fixed"
  ADV_TAU_OBJECTIVE     = "final_pnl"       # "final_pnl"
  # Alternating freeze schedule (same principle as the p_buy adversary):
  # MM trains for ADV_TAU_FREEZE_MM episodes, then adversary trains for
  # ADV_TAU_FREEZE_ADV episodes.  During the adversary phase, MM learning
  # is disabled and the bandit explores/exploits freely.  During the MM
  # phase, the bandit still selects tau (so the MM sees varied persistence)
  # but does not update its scores.
  ADV_TAU_FREEZE_MM     = 100       # MM trains for this many episodes per cycle
  ADV_TAU_FREEZE_ADV    = 100        # adversary trains for this many episodes per cycle
  ADV_TAU_WARMUP_EPISODES = 200     # MM trains alone (random tau) before adversary starts
  # Fill imbalance alpha uses the same tau target as the MO-flow EWMA,
  # but it is RESCALED to the all-event clock because fill imbalance is
  # updated on every environment step (LO / MO / Cancel), not only on MOs.
  # No extra p_fill calibration is applied.
  RECALIB_INTERVAL      = 250       # legacy interval used to refresh alpha prints/state

  _VALID_ADV_TAU_OBJECTIVES = ("final_pnl", "total_reward")
  if ADV_TAU_OBJECTIVE not in _VALID_ADV_TAU_OBJECTIVES:
      raise ValueError(f"ADV_TAU_OBJECTIVE must be one of {_VALID_ADV_TAU_OBJECTIVES}, "
                       f"got '{ADV_TAU_OBJECTIVE}'")

  # EWMA alpha: derived from tau (time constant) when using exponential regime switching.
  # alpha = 1 - exp(-1/tau)  makes the EWMA time constant match the mean regime duration.
  #
  # For USE_ADV_TAU with alpha_fixed mode, the baseline EWMA_ALPHA must still
  # be calibrated to the default tau (REGIME_EXP_TAU=20), not the fallback 0.10.
  # The per-episode mo_flow_ewma_alpha override handles alpha_recalc mode.
  import math

  def _compute_mo_flow_alpha(tau: float) -> float:
      """Compute MO-flow EWMA alpha from a target memory tau on the MO clock."""
      tau_eff = max(float(tau), 1.0)
      return 1.0 - math.exp(-1.0 / tau_eff)

  def _compute_regime_matched_gamma(
      tau_mo: float,
      mo_frac: float,
      method: str = "exact",
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
      p = min(max(float(mo_frac), 1e-9), 1.0)
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
      mo_frac_eff = max(float(mo_frac), 1e-6)
      tau_steps = tau_mo / mo_frac_eff
      return 1.0 - math.exp(-1.0 / tau_steps)

  # -- EWMA_ALPHA (mo_flow) -- set now --------------------------------------------------------------
  FILL_IMBALANCE_ALPHA = None  # computed after tau selection
  FAST_FLOW_EWMA_ALPHA = None  # computed from tau_fast = tau_regime / FAST_FLOW_TAU_DIVISOR

  if REGIME_DISTRIBUTION == "exponential" and USE_REGIME_SWITCH:
      EWMA_ALPHA = _compute_mo_flow_alpha(REGIME_EXP_TAU)
      _tau_for_fill_alpha = REGIME_EXP_TAU
      print(f"[EWMA] mo_flow alpha derived from tau={REGIME_EXP_TAU}: alpha={EWMA_ALPHA:.6f}")

  elif REGIME_DISTRIBUTION == "pareto" and USE_REGIME_SWITCH:
      # Pareto: use MEDIAN regime duration (more robust than mean for
      # heavy-tailed distributions).
      # Pareto median:  L_med = L_min * 2^(1/alpha_pareto)
      _pareto_median = REGIME_L_MIN * (2.0 ** (1.0 / REGIME_ALPHA))
      EWMA_ALPHA = _compute_mo_flow_alpha(_pareto_median)
      _tau_for_fill_alpha = _pareto_median
      print(f"[EWMA] Pareto (alpha={REGIME_ALPHA}, L_min={REGIME_L_MIN}): "
            f"median={_pareto_median:.1f} MOs")
      print(f"[EWMA] mo_flow alpha (from median): alpha={EWMA_ALPHA:.6f}")

  elif USE_ADV_TAU:
      _adv_tau_baseline = 20
      EWMA_ALPHA = _compute_mo_flow_alpha(_adv_tau_baseline)
      _tau_for_fill_alpha = _adv_tau_baseline
      print(f"[EWMA] ADV_TAU baseline (tau={_adv_tau_baseline}): "
            f"mo_flow alpha={EWMA_ALPHA:.6f}")

  else:
      EWMA_ALPHA = 0.10
      _tau_for_fill_alpha = None

  # Always compute the fast tracker for diagnostics/plots.  The
  # USE_MO_FLOW_FAST_EWMA flag only controls whether the DQN consumes it.
  _tau_fast_base = 20.0 if _tau_for_fill_alpha is None else float(_tau_for_fill_alpha)
  _tau_fast = max(_tau_fast_base / max(float(FAST_FLOW_TAU_DIVISOR), 1.0), 1.0)
  FAST_FLOW_EWMA_ALPHA = _compute_mo_flow_alpha(_tau_fast)

  BAYES_FLOW_TAU_R = float(_tau_for_fill_alpha) if _tau_for_fill_alpha is not None else 60.0

  def _count_mo_events_from_msg(msg_df):
      """Return (n_mo_events, n_logged_events) from a simulator message tape."""
      if msg_df is None or "Type" not in msg_df.columns:
          return 0, 0
      type_col = msg_df["Type"]
      total_events = int(len(type_col))
      if total_events <= 0:
          return 0, 0
      if np.issubdtype(type_col.dtype, np.number):
          n_mo = int((type_col.to_numpy() == 1).sum())
      else:
          n_mo = int(type_col.astype(str).str.upper().eq("MO").sum())
      return n_mo, total_events

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

      print(f"\n{'='*60}")
      print(f"MO EVENT FRACTION CALIBRATION -- {n_episodes} offline episodes (no MM)")
      print(f"{'='*60}")

      total_mo = 0
      total_events = 0
      for calib_ep in range(n_episodes):
          calib_seed = GLOBAL_SEED + 750_000 + calib_ep
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
          print(f"  [MO_CALIB] ep {calib_ep + 1}: "
                f"{n_mo}/{n_events} MO events  p_MO={frac:.6f}")

      if total_events <= 0:
          print("[MO_CALIB] WARNING: no events observed; keeping analytic p_MO.")
          print(f"{'='*60}\n")
          return None

      p_mo = total_mo / total_events
      print(f"[MO_CALIB] p_MO empirical = {total_mo}/{total_events} = {p_mo:.6f}")
      print(f"{'='*60}\n")
      return float(p_mo), int(total_mo), int(total_events)

  # Estimated number of MO events per episode.
  # Used by regime/adversarial schedule boundaries and frac_time_remaining.
  #
  # The analytic Santa Fe estimate is a fallback.  When enabled, the empirical
  # no-MM calibration below replaces it with the realized event fraction from
  # the same simulator path used by training.
  _Lam = lam * NUMBER_TICK_LEVELS
  _Mu  = 2.0 * mu
  _Delta_approx = delta * 100.0     # rough steady-state order count
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
      f"[MO_CLOCK] p_MO analytic={_MO_EVENT_FRACTION_ANALYTIC:.6f}  "
      f"active={_MO_EVENT_FRACTION:.6f} ({_MO_EVENT_FRACTION_SOURCE})  "
      f"expected_MO/episode~={EXPECTED_MO_PER_EPISODE}"
  )
  _AUTO_GAMMA_VALUE = None
  _AUTO_GAMMA_H_STEPS = None
  if USE_AUTO_GAMMA:
      _gamma_tau = 20.0 if _tau_for_fill_alpha is None else float(_tau_for_fill_alpha)
      _AUTO_GAMMA_VALUE = _compute_regime_matched_gamma(
          tau_mo=_gamma_tau,
          mo_frac=_MO_EVENT_FRACTION,
          method=AUTO_GAMMA_METHOD,
      )
      _AUTO_GAMMA_H_STEPS = _gamma_tau / max(_MO_EVENT_FRACTION, 1e-9)
      best_params["gamma"] = float(_AUTO_GAMMA_VALUE)
      print(
          f"[AUTO_GAMMA] method={AUTO_GAMMA_METHOD}  "
          f"tau_mo={_gamma_tau:.1f}  mo_frac={_MO_EVENT_FRACTION:.6f}  "
          f"H_steps~={_AUTO_GAMMA_H_STEPS:.1f}  gamma={_AUTO_GAMMA_VALUE:.6f}"
      )

  # -- New reward / state features --------------------------------------------------------------
  USE_INVENTORY_WALL   = True     # matches Fase A checkpoint
  USE_TERMINAL_BONUS   = False    # terminal PnL bonus (disabled for baseline)
  USE_FILL_IMBALANCE   = True     # keep the fill-feature input slot enabled
  USE_QUOTE_EXPOSURE_IMBALANCE = True  # if True, DQN fill feature = quote_exposure_imbalance

  # -- Robustness clipping flags --------------------------------------------------------------
  USE_G_CLIP           = True    # clip n-step return G to [v_min, v_max]
  USE_PER_PRIORITY_CLIP = True   # cap PER priorities at median_ema * 3

  # --------------------------------------------------------------
  # Reward function
  # --------------------------------------------------------------
  _inv_limit_val = best_params.get("inv_limit", 8)
  _reward_kwargs = dict(inv_penalty_coeff=INV_PENALTY_COEFF)

  if USE_INVENTORY_WALL:
      _reward_kwargs["use_inv_wall"] = True
      _reward_kwargs["inv_wall_threshold"] = _inv_limit_val / 2.0
      print(f"[REWARD] Inventory wall: threshold={_reward_kwargs['inv_wall_threshold']}, "
            f"coeff=phi={INV_PENALTY_COEFF}")

  if USE_TERMINAL_BONUS:
      _reward_kwargs["use_terminal_bonus"] = True
      print("[REWARD] Terminal PnL bonus: ON (PnL in ticks, no attenuation)")

  if USE_DAMPENED_REWARD:
      reward_fn_spread = partial(reward_pnl_dampened_inv_quadratic, **_reward_kwargs)
      print("[REWARD] Using reward_pnl_dampened_inv_quadratic (dampened PnL)")
  else:
      reward_fn_spread = partial(reward_spread_capture_inv_quadratic, **_reward_kwargs)
      print("[REWARD] Using reward_spread_capture_inv_quadratic (spread capture)")

  # --------------------------------------------------------------
  # Adversarial RL training (Glielmo-style zero-sum game)
  # --------------------------------------------------------------
  # When USE_ADVERSARIAL = True, a small DQN adversary chooses buy_mo_prob
  # at each regime boundary to *minimise* the MM's shaped reward.
  # Regime durations remain Pareto-distributed (environment constraint).
  # Training alternates: MM trains for ADV_FREEZE_MM episodes, then the
  # adversary trains for ADV_FREEZE_ADV episodes (asymmetric freeze).
  # The first ADV_WARMUP_EPISODES are vanilla (p=0.5) so the MM has a
  # reasonable starting policy before the adversary begins exploiting.
  #
  # Mutually exclusive with USE_REGIME_SWITCH.
  # --------------------------------------------------------------
  USE_ADVERSARIAL      = False    # <- set True to enable adversarial training
  ADV_FREEZE_MM        = 100        # MM trains for this many episodes per cycle
  ADV_FREEZE_ADV       = 15        # adversary trains for this many episodes per cycle
  ADV_WARMUP_EPISODES  = 200       # train MM alone (p=0.5) before adversary starts

  # Mutual exclusivity: only one MO flow mode can be active at a time.
  if USE_ADVERSARIAL and USE_REGIME_SWITCH:
      raise ValueError("USE_ADVERSARIAL and USE_REGIME_SWITCH are mutually exclusive")
  if USE_ADVERSARIAL and (AdversaryAgent is None or make_adversarial_schedule is None):
      raise RuntimeError(
          "USE_ADVERSARIAL=True requires the legacy adversary_agent.py module, "
          "which is not present. Use DeepSarsaQRunner_ADV.py for current "
          "adversarial training modes."
      )
  if USE_ADV_TAU and USE_ADVERSARIAL:
      raise ValueError("USE_ADV_TAU and USE_ADVERSARIAL are mutually exclusive")
  if USE_ADV_TAU and USE_REGIME_SWITCH:
      raise ValueError("USE_ADV_TAU and USE_REGIME_SWITCH are mutually exclusive")

  # -- Progressive curriculum / anchor decay / cyclical LR validation --
  if USE_PROGRESSIVE_CURRICULUM and not USE_REGIME_SWITCH:
      raise ValueError(
          "USE_PROGRESSIVE_CURRICULUM requires USE_REGIME_SWITCH = True "
          "(the curriculum modulates p_lo/p_hi which only apply under regime switching)")
  if USE_ADR_LITE_CURRICULUM and not USE_REGIME_SWITCH:
      raise ValueError(
          "USE_ADR_LITE_CURRICULUM requires USE_REGIME_SWITCH = True "
          "(ADR-lite thermostat only modulates regime-switching p_buy bounds)")
  if USE_ADR_LITE_CURRICULUM and USE_PROGRESSIVE_CURRICULUM:
      raise ValueError(
          "USE_ADR_LITE_CURRICULUM and USE_PROGRESSIVE_CURRICULUM are mutually "
          "exclusive -- ADR-lite replaces the linear ramp with an adaptive gate")
  # NOTE: ADR-lite v3 does NOT require WARMSTART_CKPT -- the new
  # algorithm has no calibration phase and no Phase A baseline.  It
  # works just as well from a freshly initialised controller.  The
  # earlier requirement (v1/v2) was justified because we calibrated
  # against the frozen Phase A policy; with the absolute-threshold
  # design that is no longer needed.
  if USE_EWC_ANCHOR and WARMSTART_CKPT is None:
      raise ValueError(
          "USE_EWC_ANCHOR requires WARMSTART_CKPT to be set -- the Fisher is "
          "estimated from the frozen Phase A policy, and the anchor is the "
          "pre-trained baseline")
  if USE_ANCHOR_DECAY and WARMSTART_CKPT is None:
      raise ValueError(
          "USE_ANCHOR_DECAY requires WARMSTART_CKPT to be set "
          "(anchor decay is meaningless without a pre-trained anchor)")
  if USE_PROGRESSIVE_CURRICULUM:
      assert 0 < CURRICULUM_DELTA_START <= 0.5, (
          f"CURRICULUM_DELTA_START must be in (0, 0.5], got {CURRICULUM_DELTA_START}")
      assert 0 < CURRICULUM_DELTA_END <= 0.5, (
          f"CURRICULUM_DELTA_END must be in (0, 0.5], got {CURRICULUM_DELTA_END}")
      assert CURRICULUM_DELTA_START <= CURRICULUM_DELTA_END, (
          f"CURRICULUM_DELTA_START ({CURRICULUM_DELTA_START}) must be <= "
          f"CURRICULUM_DELTA_END ({CURRICULUM_DELTA_END})")
      assert 0 < CURRICULUM_RAMP_FRAC <= 1.0, (
          f"CURRICULUM_RAMP_FRAC must be in (0, 1], got {CURRICULUM_RAMP_FRAC}")
      # Verify that DELTA_END matches the static bounds for consistency
      _expected_delta_end = (REGIME_P_HI - REGIME_P_LO) / 2.0
      if abs(CURRICULUM_DELTA_END - _expected_delta_end) > 1e-6:
          print(f"[WARNING] CURRICULUM_DELTA_END ({CURRICULUM_DELTA_END}) does not match "
                f"(REGIME_P_HI - REGIME_P_LO)/2 = {_expected_delta_end:.4f}. "
                f"The final curriculum bounds will differ from the static regime bounds.")
  if USE_ANCHOR_DECAY:
      assert ANCHOR_LAMBDA_START > 0, f"ANCHOR_LAMBDA_START must be > 0, got {ANCHOR_LAMBDA_START}"
      assert ANCHOR_LAMBDA_END > 0, f"ANCHOR_LAMBDA_END must be > 0, got {ANCHOR_LAMBDA_END}"
      assert ANCHOR_LAMBDA_START >= ANCHOR_LAMBDA_END, (
          f"ANCHOR_LAMBDA_START ({ANCHOR_LAMBDA_START}) must be >= "
          f"ANCHOR_LAMBDA_END ({ANCHOR_LAMBDA_END}) (decay, not growth)")

  # -- Curriculum-tied anchor validation --------------------------------------------------------------
  if USE_CURRICULUM_TIED_ANCHOR and USE_ANCHOR_DECAY:
      raise ValueError(
          "USE_CURRICULUM_TIED_ANCHOR and USE_ANCHOR_DECAY are mutually exclusive. "
          "Enable at most one anchor-decay mechanism at a time.")
  if USE_CURRICULUM_TIED_ANCHOR and not USE_ADR_LITE_CURRICULUM:
      raise ValueError(
          "USE_CURRICULUM_TIED_ANCHOR requires USE_ADR_LITE_CURRICULUM = True "
          "(the tied anchor reads adr_curriculum.delta as its input)")
  if USE_CURRICULUM_TIED_ANCHOR and WARMSTART_CKPT is None:
      raise ValueError(
          "USE_CURRICULUM_TIED_ANCHOR requires WARMSTART_CKPT to be set "
          "(anchor-based preservation is meaningless without a pre-trained baseline)")
  if USE_CURRICULUM_TIED_ANCHOR and not USE_FINE_TUNING_TECHNIQUES:
      raise ValueError(
          "USE_CURRICULUM_TIED_ANCHOR requires USE_FINE_TUNING_TECHNIQUES = True. "
          "The tied anchor calls deep_controller.set_anchor_lambda(), which is "
          "only meaningful after the fine-tuning block has snapshotted the "
          "Phase A weights via deep_controller.set_anchor_weights(...). "
          "Without the snapshot, set_anchor_lambda() has nothing to pull "
          "towards and the run will crash at runtime.")
  if USE_CURRICULUM_TIED_ANCHOR:
      assert TIED_ANCHOR_BASE > 0, (
          f"TIED_ANCHOR_BASE must be > 0, got {TIED_ANCHOR_BASE}")
      assert TIED_ANCHOR_SLOPE >= 0, (
          f"TIED_ANCHOR_SLOPE must be >= 0, got {TIED_ANCHOR_SLOPE}")
      assert ADR_LITE_DELTA_MIN <= TIED_ANCHOR_GATE <= ADR_LITE_DELTA_MAX, (
          f"TIED_ANCHOR_GATE ({TIED_ANCHOR_GATE}) must lie in "
          f"[ADR_LITE_DELTA_MIN={ADR_LITE_DELTA_MIN}, "
          f"ADR_LITE_DELTA_MAX={ADR_LITE_DELTA_MAX}]")

  # If warm-starting from a pre-trained MM, skip the warmup -- the MM
  # already has a reasonable policy, so the adversary can start immediately.
  if USE_ADVERSARIAL and WARMSTART_CKPT is not None:
      ADV_WARMUP_EPISODES = 0
      print(f"[ADVERSARIAL] WARMSTART_CKPT set -> ADV_WARMUP_EPISODES forced to 0")
  if USE_ADV_TAU and WARMSTART_CKPT is not None:
      ADV_TAU_WARMUP_EPISODES = 0
      print(f"[ADV_TAU] WARMSTART_CKPT set -> ADV_TAU_WARMUP_EPISODES forced to 0")

  # --------------------------------------------------------------
  # 3) Clean previous TensorBoard runs for this experiment
  # --------------------------------------------------------------
  def _active_fill_imbalance_feature_tag() -> str:
      if not bool(USE_FILL_IMBALANCE):
          return "none"
      if bool(USE_QUOTE_EXPOSURE_IMBALANCE):
          return "quote_exposure"
      return "ewma"

  def _checkpoint_feature_tag() -> str:
      bayes_tag = f"bayes_on{len(BAYES_FLOW_FEATURE_KEYS)}" if USE_BAYES_FLOW_SIGNAL else "bayes_off"
      mix_tag = "mix_on" if (USE_BAYES_FLOW_SIGNAL and USE_MIXTURE) else "mix_off"
      prior_tag = f"_truncated_{'on' if USE_TRUNCATED_BETA else 'off'}" if USE_BAYES_FLOW_SIGNAL else ""
      return f"fill_{_active_fill_imbalance_feature_tag()}_{bayes_tag}_{mix_tag}{prior_tag}"

  def _build_train_suffix() -> str:
      def _fmt_suffix_float(value: float) -> str:
          text = f"{float(value):.6f}".rstrip("0").rstrip(".")
          return text.replace("-", "m").replace(".", "p")

      _n_act = len(PURE_MM_OFFSETS) if USE_PURE_MM else best_params.get("n_actions", 6)
      _gamma_tag = _fmt_suffix_float(best_params.get("gamma", 0.0))
      _n_steps = int(best_params.get("n_steps", 1))
      _n_hidden = int(best_params.get("n_hidden", 0))
      _n_neurons = int(best_params.get("n_neurons", 0))
      suffix = f"_g{_gamma_tag}_nstep{_n_steps}_h{_n_hidden}x{_n_neurons}_a{int(_n_act)}"
      _dropout = float(best_params.get("dropout_level", 0.0))
      if _dropout > 0.0:
          suffix += f"_do{_fmt_suffix_float(_dropout)}"
      if USE_ADVERSARIAL:
          suffix += "_adversarial"
      if USE_REGIME_SWITCH:
          suffix += f"_regime_{REGIME_DISTRIBUTION}"
      if USE_ADR_LITE_CURRICULUM:
          suffix += "_adr_lite"
      if USE_ADV_TAU:
          suffix += f"_adv_tau_{ADV_TAU_MODE}"
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
      if USE_EWC_ANCHOR:
          suffix += "_ft_ewc"
      if USE_ANCHOR_DECAY:
          suffix += "_anchor_decay"
      if USE_CURRICULUM_TIED_ANCHOR:
          suffix += (
              f"_anchor_tied"
              f"_b{TIED_ANCHOR_BASE:.2f}"
              f"_s{TIED_ANCHOR_SLOPE:.2f}"
              f"_g{TIED_ANCHOR_GATE:.2f}"
          )
      if USE_CYCLICAL_LR:
          suffix += "_cyc_lr"
      suffix += f"_{_checkpoint_feature_tag()}"
      # Intra-decision (SMDP) γ^n reward aggregation tag.
      #   USE_INTRA_EVENT_GAMMA=True  -> "_gevent" (paper convention, current run)
      #   USE_INTRA_EVENT_GAMMA=False -> "_gflat"  (undiscounted intra-event sum, ablation)
      # Inserted here so it lands before "_best_ma{N}" / "_final" in checkpoint names
      # and prevents overwriting checkpoints from runs with the opposite setting.
      _intra_event_suffix = "_gevent" if USE_INTRA_EVENT_GAMMA else "_gflat"
      suffix += _intra_event_suffix
      return suffix

  _mode_suffix = _build_train_suffix()
  MODE_DIR = f"deep_mm_mtm_{'pure' if USE_PURE_MM else 'generic'}_invp{INV_PENALTY_COEFF:.4f}{_mode_suffix}"

  try:
      BASE = Path(__file__).resolve().parent
  except NameError:
      BASE = Path.cwd()
  RUN_ROOT = BASE / "runs_spyder" / MODE_DIR

  print("[TB] RUN_ROOT abs:", RUN_ROOT)

  if RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT, ignore_errors=False)

  RUN_ROOT.mkdir(parents=True, exist_ok=True)

  # --------------------------------------------------------------
  # 5) Build Deep RL controller (GENERIC or PURE MM)
  # --------------------------------------------------------------

  pure_mm_offsets = None

  if not USE_PURE_MM:
      # ==============================================================
      # GENERIC MODE:
      #   - 9 discrete actions:
      #       0: post_bid
      #       1: post_ask
      #       2: post_bid_ask
      #       3: cancel_bid
      #       4: cancel_ask
      #       5: hold
      #       6: post_bid_ask_inside_spread
      #       7: post_bid_inside_spread
      #       8: post_ask_inside_spread
      #
      #   - state: [spread, asksize, bidsize, inventory, has_bid, has_ask]
      #   - The controller directly chooses post/cancel/hold actions.
      #   - Inside-spread logic is implemented inside DeepRLController.
      # ==============================================================
      deep_controller = DeepRLController(
          level_offset=0,
          n_actions=best_params["n_actions"],  # 6 core + up to 3 inside-spread variants

          # --- Core RL hyperparameters ---
          gamma=best_params["gamma"],
          lr=best_params["lr"],
          weight_decay=best_params.get("weight_decay", 0.0),
          epsilon_start=best_params["epsilon_start"],
          epsilon_min=best_params["epsilon_min"],
          epsilon_decay=best_params["epsilon_decay"],
          batch_size=best_params["batch_size"],
          replay_capacity=best_params["replay_capacity"],
          target_update_steps=best_params["target_update_steps"],

          # --- Algorithm flavor switches (Rainbow components) ---
          use_sarsa=best_params["use_sarsa"],
          use_double=best_params["use_double"],
          use_dueling=best_params["use_dueling"],
          use_prioritized_experience=best_params["use_per"],
          use_noisy_net=best_params["use_noisy_net"],
          use_factored_noise=USE_FACTORED_NOISE,
          use_fully_noisy=USE_FULLY_NOISY,

          # --- Distributional DQN (C51) ---
          # When use_distributional=True the Q-network outputs a categorical
          # distribution over `atoms` support points in [v_min, v_max] instead
          # of a single scalar Q(s,a).  Action selection uses E[Z(s,a)].
          use_distributional=best_params.get("use_distributional", False),
          v_min=best_params.get("dist_v_min", -10.0),
          v_max=best_params.get("dist_v_max", 10.0),
          atoms=best_params.get("dist_atoms", 51),

          # --- Network architecture ---
          n_neurons=best_params["n_neurons"],
          n_hidden=best_params["n_hidden"],
          activation=best_params.get("activation", "relu"),
          elu_alpha=best_params.get("elu_alpha", None),
          dropout_level=best_params.get("dropout_level", 0.0),

          # --- Prioritized Experience Replay (PER) annealing schedules ---
          # T44-tuned: moderate -> mild alpha; partial -> fuller beta correction.
          per_alpha_start=best_params.get("per_alpha_start", 0.5386),
          per_alpha_end=best_params.get("per_alpha_end", 0.3750),
          per_alpha_last_episode=N_EPISODES,
          per_beta_start=best_params.get("per_beta_start", 0.3043),
          per_beta_end=best_params.get("per_beta_end", 0.5481),
          per_beta_last_episode=N_EPISODES,

          # --- n-step TD ---
          # n_steps controls how many SMDP transitions are chained to build
          # the multi-step return G^{(n)} before bootstrapping with Q(s',a').
          n_steps=best_params["n_steps"],

          # --- Intra-decision (SMDP) γ^n reward aggregation ---
          # True (default): r_t = Σ γ^n r_{t,n} (paper convention).
          # False: r_t = Σ r_{t,n} (undiscounted intra-event sum).
          # Outer γ^{N_t} bootstrap is unchanged in both modes.
          use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),

          log_dir=str(RUN_ROOT),

          # --- Throttle gates ---
          # Controls how often the controller queries the network for a new
          # action. Between decisions, act() returns ("hold",) and SMDP
          # accumulates reward into the pending transition.
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

          # --- Robustness clipping ---
          use_g_clip=USE_G_CLIP,
          use_per_priority_clip=USE_PER_PRIORITY_CLIP,

          # --- MM mode ---
          pure_mm=False,                         # <--- GENERIC mode
          inv_limit=best_params.get("inv_limit", None),
          pure_mm_offsets=None,                  # ignored in generic mode
      )
  else:
      # ==============================================================
      # PURE MM MODE:
      #   - Discrete actions index a grid of offsets (bid_off, ask_off).
      #   - Example grid below (9 actions) includes both passive and
      #     aggressive/inside-spread behaviors through negative offsets.
      #
      #   - The RL agent *only* chooses how tight/wide the quotes are.
      #   - Inventory band enforced via inv_limit:
      #         |inv| < inv_limit -> 2-sided quoting
      #         inv >= inv_limit  -> only ASK side
      #         inv <= -inv_limit -> only BID side
      # ==============================================================
      # pure_mm_offsets = [
      #     (0, 0),
      #     (-1, 0),
      #     (0, -1),
      #     (-1, -1),
      # ]
    
    
      # ==============================================================
      # grid6_with_aggressive -- 6-action pure-MM grid
      # Matches the Fase A warm-start checkpoint exactly.
      # ==============================================================
      pure_mm_offsets = list(PURE_MM_OFFSETS)


      deep_controller = DeepRLController(
          level_offset=0,
          # In PURE MM mode, n_actions must match the length of the offset
          # grid -- each action index maps to a (bid_offset, ask_offset) pair.
          n_actions=len(pure_mm_offsets),

          # --- Core RL hyperparameters ---
          gamma=best_params["gamma"],
          lr=best_params["lr"],
          weight_decay=best_params.get("weight_decay", 0.0),
          epsilon_start=best_params["epsilon_start"],
          epsilon_min=best_params["epsilon_min"],
          epsilon_decay=best_params["epsilon_decay"],
          batch_size=best_params["batch_size"],
          replay_capacity=best_params["replay_capacity"],
          target_update_steps=best_params["target_update_steps"],

          # --- Algorithm flavor switches (Rainbow components) ---
          use_sarsa=best_params["use_sarsa"],
          use_double=best_params["use_double"],
          use_dueling=best_params["use_dueling"],
          use_prioritized_experience=best_params["use_per"],
          use_noisy_net=best_params["use_noisy_net"],
          use_factored_noise=USE_FACTORED_NOISE,
          use_fully_noisy=USE_FULLY_NOISY,

          # --- Distributional DQN (C51) ---
          # Same C51 configuration as in GENERIC mode.  The distributional
          # head learns a full return distribution Z(s,a) over [v_min, v_max].
          use_distributional=best_params.get("use_distributional", False),
          v_min=best_params.get("dist_v_min", -10.0),
          v_max=best_params.get("dist_v_max", 10.0),
          atoms=best_params.get("dist_atoms", 51),

          # --- Network architecture ---
          n_neurons=best_params["n_neurons"],
          n_hidden=best_params["n_hidden"],
          activation=best_params.get("activation", "relu"),
          elu_alpha=best_params.get("elu_alpha", None),
          dropout_level=best_params.get("dropout_level", 0.0),

          # --- PER annealing schedules (T44-tuned) ---
          per_alpha_start=best_params.get("per_alpha_start", 0.5386),
          per_alpha_end=best_params.get("per_alpha_end", 0.3750),
          per_alpha_last_episode=N_EPISODES,
          per_beta_start=best_params.get("per_beta_start", 0.3043),
          per_beta_end=best_params.get("per_beta_end", 0.5481),
          per_beta_last_episode=N_EPISODES,

          # --- n-step TD ---
          n_steps=best_params["n_steps"],

          # --- Intra-decision (SMDP) γ^n reward aggregation ---
          # See USE_INTRA_EVENT_GAMMA in the runner config block.
          use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),

          log_dir=str(RUN_ROOT),

          # --- Throttle gates ---
          # Same throttle configuration as GENERIC mode (from best_params).
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

          # --- Robustness clipping ---
          use_g_clip=USE_G_CLIP,
          use_per_priority_clip=USE_PER_PRIORITY_CLIP,

          # --- MM mode ---
          pure_mm=True,                         # <--- PURE MM mode enabled
          inv_limit=best_params.get("inv_limit", None),
          pure_mm_offsets=pure_mm_offsets,
      )

  # -- Warm-start: load weights from a pre-trained checkpoint --------------------------------------------------------------
  # Fail-fast if the checkpoint path is set but the file doesn't exist.
  # Without this, the script silently falls back to training from scratch
  # with the wrong exploration config (NoisyNet disabled, epsilon too low).
  if WARMSTART_CKPT is not None and not os.path.isfile(WARMSTART_CKPT):
      raise FileNotFoundError(
          f"[WARMSTART] Checkpoint not found: {WARMSTART_CKPT}\n"
          f"Fix the path or set WARMSTART_CKPT = None to train from scratch."
      )

  if WARMSTART_CKPT is not None and os.path.isfile(WARMSTART_CKPT):
      _ws = torch.load(WARMSTART_CKPT, map_location="cpu", weights_only=True)

      # Snapshot original checkpoint keys BEFORE any renaming/dropping,
      # so we can later detect if the checkpoint had NoisyNet sigmas.
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
              "Could not locate first feature-layer weight key in state dict. "
              "Expected one of: feature.0.weight, feature.0.w_mu"
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
          return _renamed, _dropped

      # Check if warmstart checkpoint has a different input dim (e.g. no EWMA)
      _ws_input_key = _first_feature_weight_key(_ws["q_net"])
      _cur_input_key = _first_feature_weight_key(deep_controller.q_net.state_dict())
      _ws_input_dim = _ws["q_net"][_ws_input_key].shape[1]
      _cur_input_dim = deep_controller.q_net.state_dict()[_cur_input_key].shape[1]

      _input_pad_info = None  # set below if zero-padding is applied
      if _ws_input_dim != _cur_input_dim:
          print(f"[WARMSTART] Input dim mismatch: checkpoint={_ws_input_dim}, "
                f"model={_cur_input_dim}. Padding first layer with zeros.")
          for net_key in ("q_net", "target_net"):
              if net_key not in _ws:
                  continue
              sd = _ws[net_key]
              _sd_input_key = _first_feature_weight_key(sd)
              w = sd[_sd_input_key]                               # [n_neurons, old_dim]
              pad = torch.zeros(w.shape[0], _cur_input_dim - _ws_input_dim)
              sd[_sd_input_key] = torch.cat([w, pad], dim=1)      # [n_neurons, new_dim]
          _input_pad_info = (_ws_input_dim, _cur_input_dim)

      # Adapt checkpoint schema to the current model. This handles:
      #   - heads-only noisy <-> fully linear
      #   - fully noisy <-> linear
      #   - heads-only noisy <-> fully noisy (feature trunk conversion)
      _schema_renamed = 0
      _schema_dropped = 0
      _schema_exact = True
      for net_key, model_net in (("q_net", deep_controller.q_net),
                                 ("target_net", deep_controller.target_net)):
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
              "[WARMSTART] Adapted Linear/Noisy schema to current model "
              f"({_schema_renamed} renamed, {_schema_dropped} sigma dropped)"
          )

      # Load state dicts. Use strict=False when architecture differs
      # (e.g. some sigma keys are missing and will keep constructor defaults
      # until the calibration block below adjusts them).
      _arch_changed = (_ws_input_dim != _cur_input_dim) or (not _schema_exact)
      _strict = not _arch_changed
      deep_controller.q_net.load_state_dict(_ws["q_net"], strict=_strict)
      if "target_net" in _ws and deep_controller.target_net is not None:
          deep_controller.target_net.load_state_dict(_ws["target_net"], strict=_strict)
      if _arch_changed:
          print("[WARMSTART] Skipping optimizer state (architecture changed, Adam moments incompatible).")
      elif "optimizer" in _ws and deep_controller.optimizer is not None:
          deep_controller.optimizer.load_state_dict(_ws["optimizer"])
      print(f"[WARMSTART] Loaded weights from: {WARMSTART_CKPT}")

      # -- NoisyNet sigma handling at warmstart --------------------------------------------------------------
      # Three scenarios when loading a checkpoint into a NoisyNet model:
      #
      #   A) Checkpoint HAS w_sigma/b_sigma (trained with NoisyNet):
      #      -> sigmas are already loaded by load_state_dict.  They are
      #        calibrated to the weight magnitudes from training.
      #        DO NOT reset -- just keep them as-is.
      #
      #   B) Checkpoint does NOT have w_sigma/b_sigma (trained without
      #      NoisyNet, e.g. Fase A with use_noisy_net=False):
      #      -> load_state_dict(strict=False) leaves w_sigma/b_sigma at
      #        their constructor-initialised values (sigma_init=0.5).
      #        sigma=0.5 is WAY too large relative to the trained weights
      #        (mean|w|approx0.09-0.16) and would make the agent nearly random.
      #        FIX: scale sigmas DOWN to a fraction of the loaded weight
      #        magnitudes, so exploration is gentle (not destructive).
      #
      #   C) NoisyNet is disabled (use_noisy_net=False):
      #      -> skip entirely, use epsilon-greedy instead.
      #
      # The key insight: for fine-tuning, we want SMALL noise relative
      # to the pre-trained weights -- just enough to explore nearby
      # policies, not enough to scramble the learned value function.
      # A good heuristic is sigma approx 3-5% of mean|w_mu|, matching
      # what NoisyNet converges to after full training.
      # --------------------------------------------------------------
      if best_params["use_noisy_net"]:
          # Check sigma coverage against the CURRENT model schema.
          # This matters for heads-only noisy -> fully noisy warmstarts,
          # where the checkpoint may contain calibrated head sigmas but no
          # feature-trunk sigmas.
          _missing_sigma_by_net = {}
          _sigma_expected = 0
          _sigma_present = 0
          for net_key, model_net in (("q_net", deep_controller.q_net),
                                     ("target_net", deep_controller.target_net)):
              if model_net is None:
                  continue
              _model_sigma_keys = {
                  k for k in model_net.state_dict().keys()
                  if (".w_sigma" in k or ".b_sigma" in k)
              }
              _raw_keys = set(_ws_raw_keys.get(net_key, []))
              _missing = _model_sigma_keys - _raw_keys
              _missing_sigma_by_net[net_key] = _missing
              _sigma_expected += len(_model_sigma_keys)
              _sigma_present += len(_model_sigma_keys) - len(_missing)

          _all_sigma_covered = (_sigma_expected > 0) and (_sigma_present == _sigma_expected)

          if _all_sigma_covered and not RESET_SIGMAS:
              # Scenario A: checkpoint already has calibrated sigmas.
              # They were loaded by load_state_dict - nothing to do.
              print(f"[WARMSTART] NoisyNet sigmas loaded from checkpoint (pre-calibrated)")
          elif RESET_SIGMAS:
              # Scenario A-reset: checkpoint has sigmas but user wants fresh
              # sigma_init=0.5 (high exploration from scratch).
              import math as _math
              _sigma_init = 0.5
              _reset_count = 0
              for net in (deep_controller.q_net, deep_controller.target_net):
                  if net is None:
                      continue
                  for name, param in net.named_parameters():
                      if "w_sigma" in name:
                          fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
                          with torch.no_grad():
                              torch.nn.init.kaiming_uniform_(param, a=_math.sqrt(5))
                              param.mul_(_sigma_init)
                          _reset_count += 1
                      elif "b_sigma" in name:
                          fan_in = param.shape[0]
                          with torch.no_grad():
                              torch.nn.init.constant_(param, _sigma_init / _math.sqrt(max(1, fan_in)))
                          _reset_count += 1
              print(f"[WARMSTART] Reset {_reset_count} NoisyNet sigma params to sigma_init={_sigma_init}")
          else:
              # Scenario B / partial-B:
              # Some or all sigma params are missing from the checkpoint.
              # Scale ONLY the missing ones to ~5% of w_mu magnitude while
              # preserving any sigma params that were actually present.
              _SIGMA_FRACTION = 0.05  # target: sigma ~= 5% of |w_mu|
              _reset_count = 0
              for net_key, net in (("q_net", deep_controller.q_net),
                                   ("target_net", deep_controller.target_net)):
                  if net is None:
                      continue
                  _missing_sigma_keys = _missing_sigma_by_net.get(net_key, set())
                  for name, param in net.named_parameters():
                      if name not in _missing_sigma_keys:
                          continue
                      if "w_sigma" in name:
                          # Find the corresponding w_mu to calibrate sigma
                          mu_name = name.replace("w_sigma", "w_mu")
                          mu_abs_mean = 0.1  # fallback
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
                      f"[WARMSTART] NoisyNet sigmas partially loaded from checkpoint; "
                      f"calibrated {_reset_count} missing sigma params to {_SIGMA_FRACTION:.0%} of |w_mu| "
                      f"({_sigma_present}/{_sigma_expected} already present)"
                  )
              else:
                  print(f"[WARMSTART] NoisyNet sigmas calibrated to {_SIGMA_FRACTION:.0%} of |w_mu| "
                        f"({_reset_count} params, checkpoint had no sigmas)")
          # Print per-layer sigma/mu diagnostics for both scenarios A and B
          from dqn_distributional_with_throttle import NoisyLinear
          print(f"[WARMSTART] NoisyNet layer diagnostics (q_net):")
          for name, module in deep_controller.q_net.named_modules():
              if isinstance(module, NoisyLinear):
                  _mu_abs = module.w_mu.abs().mean().item()
                  _sigma_abs = module.w_sigma.abs().mean().item()
                  _ratio = _sigma_abs / max(_mu_abs, 1e-8)
                  print(f"  {name:20s}  mean|w_mu|={_mu_abs:.4f}  "
                        f"mean|w_sigma|={_sigma_abs:.4f}  "
                        f"sigma/mu ratio={_ratio:.1%}")
      else:
          print(f"[WARMSTART] NoisyNet disabled for fine-tuning -> using epsilon-greedy={best_params['epsilon_start']}")

      del _ws

      # -- Fine-tuning techniques (applied after warmstart) --------------------------------------------------------------
      # These implement the continual RL best practices for domain shift:
      #   1. L2 anchor: penalises drift from pre-trained weights
      #   2. LR differential: lower LR for trunk, higher for heads
      #   3. Replay buffer restore: partial, with PER priority reset
      #
      # All three are gated on USE_FINE_TUNING_TECHNIQUES to allow
      # disabling them for ablation.
      if USE_FINE_TUNING_TECHNIQUES and WARMSTART_CKPT is not None:
          # 1. L2 anchor: snapshot current (pre-trained) weights as baseline
          deep_controller.set_anchor_weights(lambda_anchor=ANCHOR_LAMBDA)
          print(f"[FINE-TUNE] L2 anchor set (lambda={ANCHOR_LAMBDA})")

          # 1b. Masked anchor: when the first feature layer was zero-padded
          #     because the checkpoint had fewer input features than the
          #     current model (e.g. Phase A without EWMAs -> Phase B with
          #     EWMAs), the new columns are currently at zero. A global
          #     anchor would keep pulling them back to zero and prevent
          #     the network from ever learning those new features.
          #     We therefore register a binary mask that is 1 on the
          #     original input columns and 0 on the padded ones, so the
          #     anchor term is switched off only on the new-feature slice.
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
                  print(f"[FINE-TUNE] Masked anchor applied to "
                        f"{sorted(_mask_dict.keys())} "
                        f"(columns {_old_dim}:{_new_dim} excluded)")

          # 2. LR differential: lower LR for feature trunk, higher for heads.
          # MUST be done AFTER optimizer.load_state_dict() which overwrites
          # param_groups with the old LR.
          _lr_feature = LR_START * LR_FEATURE_FACTOR
          _lr_head = LR_START
          deep_controller.setup_differential_lr(_lr_feature, _lr_head)
          print(f"[FINE-TUNE] Differential LR: feature={_lr_feature:.6f}, "
                f"head={_lr_head:.6f}")

          # 3. Replay buffer restore: load partial transitions from base training
          _replay_path = WARMSTART_CKPT.replace(".pt", "_replay.pt")
          if os.path.isfile(_replay_path):
              _max_load = int(best_params.get("replay_capacity", 150_000) * REPLAY_RESTORE_FRACTION)
              _loaded = deep_controller.load_replay_buffer(
                  _replay_path, max_load=_max_load, reset_priorities=True)
              print(f"[FINE-TUNE] Loaded {_loaded} transitions from {_replay_path} "
                    f"({REPLAY_RESTORE_FRACTION:.0%} of capacity)")
          else:
              print(f"[FINE-TUNE] No replay buffer found at {_replay_path}")

  # -- NoisyNet sigma weight decay (prevents random-walk growth) --------------------------------------------------------------
  # Must be called AFTER optimizer is set up (including differential LR).
  if NOISY_WEIGHT_DECAY > 0 and best_params.get("use_noisy_net", False):
      deep_controller.setup_noisy_weight_decay(sigma_weight_decay=NOISY_WEIGHT_DECAY)

  # -- Adversary agent (only when USE_ADVERSARIAL is True) --------------------------------------------------------------
  if USE_ADVERSARIAL:
      adversary = AdversaryAgent(
          inv_limit=best_params.get("inv_limit") or 8,
      )
      adv_phase = "mm"           # start by training the MM
      adv_phase_counter = 0
      mm_train_ep_count = 0      # counts only episodes where MM trains (for LR/epsilon decay)
      adv_train_ep_count = 0     # counts only episodes where adversary trains (for target sync)
      print(f"[ADVERSARY] Initialised adversary DQN "
            f"(warmup={ADV_WARMUP_EPISODES}, freeze_mm={ADV_FREEZE_MM}, freeze_adv={ADV_FREEZE_ADV})")

  # -- Adversary tau bandit (only when USE_ADV_TAU is True) --------------------------------------------------------------
  if USE_ADV_TAU:
      adv_tau = AdversaryTauAgent(
          mode=ADV_TAU_MODE,
          mo_frac=_MO_EVENT_FRACTION,
      )
      # Restore bandit state from a previous run if a checkpoint exists
      # alongside the MM warmstart checkpoint.
      if WARMSTART_CKPT is not None:
          _adv_tau_ckpt = os.path.join(os.path.dirname(WARMSTART_CKPT), "adv_tau_final.pt")
          if os.path.isfile(_adv_tau_ckpt):
              adv_tau.load(_adv_tau_ckpt)
              print(f"[ADV_TAU] Restored bandit state from: {_adv_tau_ckpt}")
          else:
              print(f"[ADV_TAU] No bandit checkpoint found at {_adv_tau_ckpt} -- starting fresh")
      # Alternating freeze schedule state (same pattern as p_buy adversary)
      adv_tau_phase = "mm"            # start by training the MM
      adv_tau_phase_counter = 0
      print(f"[ADV_TAU] Initialised tau bandit: grid={adv_tau.tau_grid}, "
            f"mode={ADV_TAU_MODE}, objective={ADV_TAU_OBJECTIVE}, "
            f"warmup={ADV_TAU_WARMUP_EPISODES}, "
            f"freeze_mm={ADV_TAU_FREEZE_MM}, freeze_adv={ADV_TAU_FREEZE_ADV}")


  # ==============================================================
  # 6) MAIN TRAINING LOOP -- Multi-Episode Online RL
  # ==============================================================
  # For each episode:
  #   1. Reset episode-level controller state (SMDP, throttle, nstep)
  #   2. Run a full LOB + MM simulation (simulate_LOB_with_MM)
  #   3. The controller's act() / learn() are called at every micro-step
  #      inside the simulation loop, accumulating SMDP transitions
  #   4. After the episode ends, log stats, decay epsilon, update LR
  #
  # The training loop is "online" -- the agent trains while interacting
  # with the environment (no separate data-collection phase).
  def _recalibrate_fill_alpha(tau: float) -> float:
      """Recompute FILL_IMBALANCE_ALPHA from tau after clock rescaling."""
      tau_steps = max(float(tau), 1.0) / max(_MO_EVENT_FRACTION, 1e-6)
      alpha = _compute_fill_alpha(tau, mo_frac=_MO_EVENT_FRACTION)
      print(f"[CALIB] fill_imbalance alpha = {alpha:.6f}  "
            f"(tau_mo={tau:.1f}, tau_steps~={tau_steps:.1f})")
      return alpha

  # -- Initial calibration (always, before training) --------------------------------------------------------------
  FILL_IMBALANCE_ALPHA = _recalibrate_fill_alpha(_tau_for_fill_alpha)

  # Print final calibrated parameters for the training run
  print(f"\n{'-'*60}")
  print(f"CALIBRATED EWMA PARAMETERS")
  print(f"  mo_event_fraction        = {_MO_EVENT_FRACTION:.6f} "
        f"({_MO_EVENT_FRACTION_SOURCE}; analytic={_MO_EVENT_FRACTION_ANALYTIC:.6f})")
  print(f"  mo_flow_ewma_alpha       = {EWMA_ALPHA:.6f}")
  if FAST_FLOW_EWMA_ALPHA is None:
      print(f"  mo_flow_fast_alpha       = auto(4x slow, clip<=1.0)")
  else:
      print(f"  mo_flow_fast_alpha       = {FAST_FLOW_EWMA_ALPHA:.6f}")
  print(f"  fill_imbalance_alpha     = {FILL_IMBALANCE_ALPHA:.6f}")
  print(f"  fill_imbalance_tau_mo    = {_tau_for_fill_alpha:.1f}")
  print(f"  fill_imbalance_tau_steps ~= "
        f"{max(float(_tau_for_fill_alpha), 1.0) / max(_MO_EVENT_FRACTION, 1e-6):.1f}")
  if USE_ADV_TAU:
      print(f"  [ADV_TAU] Recalibration at start of each MM phase")
  print(f"{'-'*60}\n")

  # -- Print progressive curriculum / anchor decay / cyclical LR config --
  if USE_PROGRESSIVE_CURRICULUM or USE_ANCHOR_DECAY or USE_CURRICULUM_TIED_ANCHOR or USE_CYCLICAL_LR:
      print(f"{'-'*60}")
      print(f"CURRICULUM SCHEDULE CONFIG")
      if USE_PROGRESSIVE_CURRICULUM:
          _p0_lo, _p0_hi = get_curriculum_p_bounds(0.0, CURRICULUM_DELTA_START, CURRICULUM_DELTA_END)
          _pN_lo, _pN_hi = get_curriculum_p_bounds(1.0, CURRICULUM_DELTA_START, CURRICULUM_DELTA_END)
          print(f"  [PROGRESSIVE] p_buy ramp: [{_p0_lo:.2f}, {_p0_hi:.2f}] -> [{_pN_lo:.2f}, {_pN_hi:.2f}]")
          _ramp_ep = int(N_EPISODES * CURRICULUM_RAMP_FRAC)
          print(f"                delta: {CURRICULUM_DELTA_START} -> {CURRICULUM_DELTA_END} "
                f"(linear ramp over {_ramp_ep} eps, then hold for {N_EPISODES - _ramp_ep} eps)")
      else:
          print(f"  [PROGRESSIVE] OFF (static p in [{REGIME_P_LO}, {REGIME_P_HI}])")
      if USE_ANCHOR_DECAY:
          print(f"  [ANCHOR DECAY-TIME] lambda: {ANCHOR_LAMBDA_START} -> {ANCHOR_LAMBDA_END} (linear in episodes)")
      elif USE_CURRICULUM_TIED_ANCHOR:
          print(f"  [ANCHOR TIED] gated linear in delta (grows with difficulty)")
          print(f"                    base  = {TIED_ANCHOR_BASE}  (lambda for delta <= gate)")
          print(f"                    slope = {TIED_ANCHOR_SLOPE}  (added over [gate, delta_max])")
          print(f"                    gate  = {TIED_ANCHOR_GATE}")
          print(f"                    lambda(delta_max) = {TIED_ANCHOR_BASE + TIED_ANCHOR_SLOPE}")
          _gp = sorted({ADR_LITE_DELTA_MIN, TIED_ANCHOR_GATE,
                        (TIED_ANCHOR_GATE + ADR_LITE_DELTA_MAX) / 2,
                        ADR_LITE_DELTA_MAX})
          _gp_strs = []
          for _d in _gp:
              _lam = get_tied_anchor_lambda(
                  delta=_d, delta_max=ADR_LITE_DELTA_MAX,
                  base=TIED_ANCHOR_BASE, slope=TIED_ANCHOR_SLOPE, gate=TIED_ANCHOR_GATE,
              )
              _gp_strs.append(f"delta={_d:.3f}->lambda={_lam:.4f}")
          print(f"                    schedule: {'  '.join(_gp_strs)}")
      else:
          print(f"  [ANCHOR DECAY] OFF (static lambda)")
      if USE_CYCLICAL_LR:
          print(f"  [CYCLICAL LR] period={CYCLICAL_LR_PERIOD}, min_factor={CYCLICAL_LR_MIN_FACTOR}")
      else:
          print(f"  [CYCLICAL LR] OFF (monotonic decay)")
      print(f"{'-'*60}\n")

  # ==============================================================
  # ADR-LITE CURRICULUM SETUP (v3 -- faithful to Akkaya 2019)
  # ==============================================================
  # Instantiate the v3 thermostat: single shared buffer, absolute
  # thresholds on clipped final_pnl, no calibration phase, no Phase A
  # baseline. The thermostat is fully self-contained and does not
  # depend on a warmstart checkpoint (it works from a freshly
  # initialised controller too).
  # --------------------------------------------------------------
  adr_curriculum = None
  # Per-episode trackers populated by `_adr_reward_wrapper` below.
  # Both are DIAGNOSTIC ONLY in v3 -- the thermostat consumes
  # `final_pnl` directly via report_episode().  We keep these
  # accumulators so that TensorBoard can still display the per-episode
  # gate signal (Sigma_t r_t) and the spread-capture component for
  # interpretability, even though they no longer drive any decision.
  #   _adr_gate_tracker -> Sigma_t r_t = episode_total_reward  (TB diagnostic)
  #   _adr_sc_tracker   -> Sigma_t spread_capture_t            (TB diagnostic)
  _adr_gate_tracker = {"sum": 0.0}
  _adr_sc_tracker   = {"sum": 0.0}

  if USE_ADR_LITE_CURRICULUM:
      # ADR-lite v3 (faithful to Akkaya et al. 2019, Algorithm 1):
      #   - one shared buffer of clipped final_pnl
      #   - p_bar = mean(buffer) compared against absolute thresholds
      #   - no calibration, no baseline, no boundary sampling
      #   - inventory used only as a retreat-side emergency brake
      #   - threshold can be CONSTANT or LINEAR_PHASE_A_ANCHORED
      from adr_lite import ADRLiteCurriculum

      # Resolve linear-peak-mode params (read Phase A best MA50 from ckpt if
      # not explicitly provided in the config block).  Only matters for
      # the legacy `linear_phase_a_anchored` mode.
      _phase_a_best_for_adr = ADR_LITE_PHASE_A_BEST_MA50
      if (ADR_LITE_THRESHOLD_MODE == "linear_phase_a_anchored"
              and _phase_a_best_for_adr is None):
          if WARMSTART_CKPT and os.path.isfile(WARMSTART_CKPT):
              try:
                  _ws_meta_only = torch.load(
                      WARMSTART_CKPT, map_location="cpu", weights_only=False,
                  )
                  _phase_a_best_for_adr = float(
                      _ws_meta_only.get("meta", {}).get("best_ma_value", 0.20)
                  )
                  del _ws_meta_only
                  print(f"[ADR-LITE] Read phase_a_best_ma50 = "
                        f"{_phase_a_best_for_adr:.4f} from {WARMSTART_CKPT}")
              except Exception as _e:
                  print(f"[ADR-LITE WARN] Could not read phase_a_best_ma50 "
                        f"from {WARMSTART_CKPT}: {_e}.  Falling back to 0.20.")
                  _phase_a_best_for_adr = 0.20
          else:
              print(f"[ADR-LITE WARN] linear_phase_a_anchored mode requested but "
                    f"WARMSTART_CKPT is None or missing.  Falling back to "
                    f"phase_a_best_ma50 = 0.20.")
              _phase_a_best_for_adr = 0.20

      # Fail-fast guard for linear_mean_gap: both measured constants
      # must be pasted in `adr_lite_config.py` (the shared source of
      # truth for runner + tuner) before training.
      if ADR_LITE_THRESHOLD_MODE == "linear_mean_gap":
          if (ADR_LITE_PREV_CURRICULUM_MEAN is None
                  or ADR_LITE_PREV_CURRICULUM_BUFFER_STD is None):
              raise RuntimeError(
                  "ADR_LITE_THRESHOLD_MODE='linear_mean_gap' requires both "
                  "PREV_CURRICULUM_MEAN and PREV_CURRICULUM_BUFFER_STD to "
                  "be set in adr_lite_config.py (measured offline from the "
                  "frozen reference policy's return-risk frontier eval at "
                  "p_buy=0.5). See the recipe in the docstring of "
                  "adr_lite_config.py."
              )

      # --------------------------------------------------------------
      # Resolve opt-in extensions from adr_lite_config.py
      # --------------------------------------------------------------
      #
      # Both default to disabled ("continuous" / "off") -- in that case
      # the kwargs passed below are None and the ADR runs in its legacy
      # backward-compatible mode.  Any other setting activates one or
      # both mechanisms without requiring changes here.
      if ADR_LITE_DELTA_SCHEDULE_MODE == "continuous":
          _adr_delta_ladder = None
      elif ADR_LITE_DELTA_SCHEDULE_MODE == "ladder_auto":
          _adr_delta_ladder = generate_auto_ladder(
              d_min=ADR_LITE_DELTA_MIN,
              d_max=ADR_LITE_DELTA_MAX,
              alpha=ADR_LITE_LADDER_AUTO_ALPHA,
              max_gap=ADR_LITE_LADDER_AUTO_MAX_GAP,
              min_gap=ADR_LITE_LADDER_AUTO_MIN_GAP,
          )
          print(f"[ADR-LITE] Auto-ladder generated: n={len(_adr_delta_ladder)}, "
                f"alpha={ADR_LITE_LADDER_AUTO_ALPHA}, "
                f"max_gap={ADR_LITE_LADDER_AUTO_MAX_GAP}, "
                f"min_gap={ADR_LITE_LADDER_AUTO_MIN_GAP}")
      elif ADR_LITE_DELTA_SCHEDULE_MODE == "ladder_manual":
          if not ADR_LITE_LADDER_MANUAL:
              raise RuntimeError(
                  "DELTA_SCHEDULE_MODE='ladder_manual' requires "
                  "LADDER_MANUAL to be a non-empty list in "
                  "adr_lite_config.py."
              )
          _adr_delta_ladder = list(ADR_LITE_LADDER_MANUAL)
          print(f"[ADR-LITE] Manual ladder: n={len(_adr_delta_ladder)}")
      else:
          raise ValueError(
              f"Unknown DELTA_SCHEDULE_MODE: {ADR_LITE_DELTA_SCHEDULE_MODE!r}. "
              f"Valid values: 'continuous', 'ladder_auto', 'ladder_manual'."
          )

      if ADR_LITE_ADVANCE_CONFIRMATION_MODE == "off":
          _adr_advance_confirmations_fn = None
      elif ADR_LITE_ADVANCE_CONFIRMATION_MODE == "formula":
          _adr_advance_confirmations_fn = make_advance_confirmation_fn(
              d_min=ADR_LITE_DELTA_MIN,
              d_max=ADR_LITE_DELTA_MAX,
              k_max=ADR_LITE_ADVANCE_CONFIRMATION_K_MAX,
              gamma=ADR_LITE_ADVANCE_CONFIRMATION_GAMMA,
          )
          print(f"[ADR-LITE] Advance confirmation formula: "
                f"K_max={ADR_LITE_ADVANCE_CONFIRMATION_K_MAX}, "
                f"gamma={ADR_LITE_ADVANCE_CONFIRMATION_GAMMA}")
      else:
          raise ValueError(
              f"Unknown ADVANCE_CONFIRMATION_MODE: "
              f"{ADR_LITE_ADVANCE_CONFIRMATION_MODE!r}. "
              f"Valid values: 'off', 'formula'."
          )

      _adr_kwargs = dict(
          delta_start=ADR_LITE_DELTA_START,
          delta_max=ADR_LITE_DELTA_MAX,
          delta_min=ADR_LITE_DELTA_MIN,
          delta_step=ADR_LITE_DELTA_STEP,
          buffer_size=ADR_LITE_BUFFER_SIZE,
          threshold_mode=ADR_LITE_THRESHOLD_MODE,
          # constant-mode kwargs
          t_H_abs=ADR_LITE_T_H_ABS,
          t_L_abs=ADR_LITE_T_L_ABS,
          # linear-peak-mode kwargs (ignored outside linear_phase_a_anchored)
          phase_a_best_ma50=(
              _phase_a_best_for_adr if _phase_a_best_for_adr is not None else 0.314
          ),
          sigma_ma50=ADR_LITE_SIGMA_MA50,
          sigma_margin=ADR_LITE_SIGMA_MARGIN,
          t_L_abs_start=ADR_LITE_T_L_ABS_START,
          t_L_abs_end=ADR_LITE_T_L_ABS_END,
          # linear-mean-gap kwargs (ignored outside linear_mean_gap)
          prev_curriculum_mean=ADR_LITE_PREV_CURRICULUM_MEAN,
          prev_curriculum_buffer_std=ADR_LITE_PREV_CURRICULUM_BUFFER_STD,
          gap_sigma=ADR_LITE_GAP_SIGMA,
          convergence_floor=ADR_LITE_CONVERGENCE_FLOOR,
          # Opt-in extensions (None = legacy behavior)
          delta_ladder=_adr_delta_ladder,
          advance_confirmations_fn=_adr_advance_confirmations_fn,
          advance_confirmation_reset_mode=ADR_LITE_ADVANCE_CONFIRMATION_RESET_MODE,
          # shared between both linear modes
          t_H_abs_end=ADR_LITE_T_H_ABS_END,
          # common
          pnl_clip=ADR_LITE_PNL_CLIP,
          inv_advance_veto_min=ADR_LITE_INV_ADVANCE_VETO_MIN,
          inv_retreat_min=ADR_LITE_INV_RETREAT_MIN,
          inv_history_len=ADR_LITE_INV_HISTORY_LEN,
          rng=np.random.default_rng(GLOBAL_SEED),
      )
      adr_curriculum = ADRLiteCurriculum(**_adr_kwargs)

      print(f"\n{'-'*60}")
      print(f"ADR-LITE CURRICULUM (v3 -- faithful to Akkaya 2019 Algorithm 1)")
      print(f"  threshold_mode      = {ADR_LITE_THRESHOLD_MODE}")
      print(f"  m (buffer)          = {ADR_LITE_BUFFER_SIZE}")
      print(f"  delta range         = [{ADR_LITE_DELTA_MIN}, {ADR_LITE_DELTA_MAX}], step={ADR_LITE_DELTA_STEP}")
      print(f"  pnl_clip            = +/-{ADR_LITE_PNL_CLIP}")
      print(f"  inv advance veto    = inv_ma > {ADR_LITE_INV_ADVANCE_VETO_MIN:.2%}  (blocks advance only)")
      print(f"  inv retreat brake   = inv_ma > {ADR_LITE_INV_RETREAT_MIN:.2%}")
      print(f"  metric              = clip(final_pnl, -{ADR_LITE_PNL_CLIP}, +{ADR_LITE_PNL_CLIP})")
      if ADR_LITE_THRESHOLD_MODE == "constant":
          print(f"  t_H_abs / t_L_abs   = {ADR_LITE_T_H_ABS:+.3f} / {ADR_LITE_T_L_ABS:+.3f}  (constant, in PnL units)")
      elif ADR_LITE_THRESHOLD_MODE == "linear_phase_a_anchored":
          print(f"  phase_a_best_ma50   = {adr_curriculum.phase_a_best_ma50:+.4f}  "
                f"(read from ckpt meta)")
          print(f"  sigma_MA50          = {ADR_LITE_SIGMA_MA50:.4f}")
          print(f"  sigma_margin        = {ADR_LITE_SIGMA_MARGIN}")
          print(f"  t_H schedule        = {adr_curriculum._t_H_abs_start:+.4f}  ->  "
                f"{adr_curriculum._t_H_abs_end:+.4f}  (linear in delta)")
          print(f"  t_L schedule        = {adr_curriculum._t_L_abs_start:+.4f}  ->  "
                f"{adr_curriculum._t_L_abs_end:+.4f}  (linear in delta)")
      elif ADR_LITE_THRESHOLD_MODE == "linear_mean_gap":
          print(f"  prev_curriculum_mean       = {adr_curriculum.prev_curriculum_mean:+.4f}  "
                f"(offline reference eval)")
          print(f"  prev_curriculum_buffer_std = {adr_curriculum.prev_curriculum_buffer_std:.4f}  "
                f"(std of {ADR_LITE_BUFFER_SIZE}-ep block means)")
          print(f"  gap_sigma                  = {ADR_LITE_GAP_SIGMA}  "
                f"(-> gap = {adr_curriculum._mean_gap_abs:+.4f})")
          print(f"  convergence_floor          = {ADR_LITE_CONVERGENCE_FLOOR:+.4f}  "
                f"(break-even floor for t_L)")
          print(f"  t_H schedule               = {adr_curriculum._t_H_abs_start:+.4f}  ->  "
                f"{adr_curriculum._t_H_abs_end:+.4f}  (linear in delta)")
          print(f"  t_L schedule               = max(floor, t_H - gap)  (coupled to t_H)")

      if ADR_LITE_THRESHOLD_MODE != "constant":
          print(f"  schedule preview    :")
          for _d in [
              ADR_LITE_DELTA_START,
              (ADR_LITE_DELTA_START + ADR_LITE_DELTA_MAX) / 2,
              ADR_LITE_DELTA_MAX,
          ]:
              _saved_d = adr_curriculum.delta
              adr_curriculum.delta = _d
              _tH, _tL = adr_curriculum._current_thresholds()
              print(f"      delta={_d:.3f}: t_H={_tH:+.4f}  t_L={_tL:+.4f}")
              adr_curriculum.delta = _saved_d

      # -- Opt-in extensions banner --------------------------------------------------------------
      print(f"  delta_schedule_mode = {ADR_LITE_DELTA_SCHEDULE_MODE}")
      if _adr_delta_ladder is not None:
          if ADR_LITE_DELTA_SCHEDULE_MODE == "ladder_auto":
              print(f"    alpha   = {ADR_LITE_LADDER_AUTO_ALPHA}")
              print(f"    max_gap = {ADR_LITE_LADDER_AUTO_MAX_GAP}")
              print(f"    min_gap = {ADR_LITE_LADDER_AUTO_MIN_GAP}")
          print(f"    n_points = {len(_adr_delta_ladder)}")
          print(f"    ladder[0:5]  = {[round(d,4) for d in _adr_delta_ladder[:5]]}")
          print(f"    ladder[-5:]  = {[round(d,4) for d in _adr_delta_ladder[-5:]]}")
          _gaps = [
              _adr_delta_ladder[i+1] - _adr_delta_ladder[i]
              for i in range(len(_adr_delta_ladder) - 1)
          ]
          print(f"    gap stats   : min={min(_gaps):.4f}  "
                f"max={max(_gaps):.4f}  "
                f"mean={sum(_gaps)/len(_gaps):.4f}")

      print(f"  advance_confirmation = {ADR_LITE_ADVANCE_CONFIRMATION_MODE}")
      if _adr_advance_confirmations_fn is not None:
          print(f"    K_max = {ADR_LITE_ADVANCE_CONFIRMATION_K_MAX}")
          print(f"    gamma = {ADR_LITE_ADVANCE_CONFIRMATION_GAMMA}")
          print(f"    reset = {ADR_LITE_ADVANCE_CONFIRMATION_RESET_MODE}")
          print(f"    K(delta) preview:")
          for _d in [
              ADR_LITE_DELTA_MIN,
              ADR_LITE_DELTA_MIN + 0.25 * (ADR_LITE_DELTA_MAX - ADR_LITE_DELTA_MIN),
              ADR_LITE_DELTA_MIN + 0.50 * (ADR_LITE_DELTA_MAX - ADR_LITE_DELTA_MIN),
              ADR_LITE_DELTA_MIN + 0.75 * (ADR_LITE_DELTA_MAX - ADR_LITE_DELTA_MIN),
              ADR_LITE_DELTA_MAX,
          ]:
              _k = _adr_advance_confirmations_fn(_d)
              print(f"      delta={_d:.3f} -> K={_k}")
      print(f"  -> {adr_curriculum}")
      print(f"{'-'*60}\n")

      # Wrap the reward function only to track per-episode `spread_capture`
      # for diagnostic TensorBoard logging (final_pnl is the actual gate).
      _adr_base_reward_fn = reward_fn_spread

      def _adr_reward_wrapper(step_idx, mm, lob, sb, sa, info, **kw):
          r = _adr_base_reward_fn(step_idx, mm, lob, sb, sa, info, **kw)
          _adr_sc_tracker["sum"]   += float(info.get("_reward_spread_capture", 0.0))
          _adr_gate_tracker["sum"] += float(r)   # diagnostic only
          return r

      reward_fn_spread = _adr_reward_wrapper

      # Defensive -- there's no calibration phase any more, but if the
      # warmstart block left any stale per-episode accumulators around,
      # zero them so episode 1 starts clean.
      deep_controller.reset_episode_accumulators()

  # ==============================================================
  # ELASTIC WEIGHT CONSOLIDATION SETUP (opt-in, Fase C or later)
  # ==============================================================
  # Estimate the diagonal Fisher Information Matrix from the frozen
  # Phase A policy and attach it to the controller, upgrading the
  # existing L2 anchor into an EWC Fisher-weighted penalty.  This
  # happens AFTER the warmstart anchor has been set (inside the
  # USE_FINE_TUNING_TECHNIQUES block above) and BEFORE training starts.
  # --------------------------------------------------------------
  if USE_EWC_ANCHOR:
      from ewc import (
          FisherDiagonal, estimate_fisher_diagonal_boltzmann,
          save_fisher, load_fisher,
      )

      if deep_controller._anchor_weights is None:
          raise RuntimeError(
              "USE_EWC_ANCHOR=True requires that an L2 anchor has been set "
              "already (via the fine-tuning block).  Check that "
              "USE_FINE_TUNING_TECHNIQUES=True and WARMSTART_CKPT is valid."
          )

      # Build a metadata snapshot of the current EWC setup -- must reject
      # any cached Fisher that was estimated under a different checkpoint,
      # input dim, or feature configuration.
      def _ewc_build_metadata() -> dict:
          _input_dim = int(
              deep_controller.q_net.state_dict()["feature.0.weight"].shape[1]
          )
          meta = dict(
              warmstart_ckpt=str(WARMSTART_CKPT) if WARMSTART_CKPT else None,
              input_dim=_input_dim,
              n_actions=int(deep_controller.n_actions),
              use_distributional=bool(getattr(deep_controller, "use_distributional", False)),
              dist_atoms=int(getattr(deep_controller, "dist_atoms", 0) or 0),
              dist_v_min=float(getattr(deep_controller, "dist_v_min", 0.0) or 0.0),
              dist_v_max=float(getattr(deep_controller, "dist_v_max", 0.0) or 0.0),
              n_hidden=int(best_params.get("n_hidden", 0)),
              n_neurons=int(best_params.get("n_neurons", 0)),
              activation=str(best_params.get("activation", "relu")),
              elu_alpha=best_params.get("elu_alpha", None),
              dropout_level=float(best_params.get("dropout_level", 0.0)),
              use_dueling=bool(best_params.get("use_dueling", False)),
              pure_mm=bool(best_params.get("pure_mm", True)),
              use_flow_ewma=bool(USE_FLOW_EWMA),
              use_fast_flow_ewma=bool(USE_FAST_FLOW_EWMA),
              use_bayes_flow_signal=bool(USE_BAYES_FLOW_SIGNAL),
              bayes_flow_feature_keys=list(BAYES_FLOW_FEATURE_KEYS),
              use_fill_imbalance=bool(USE_FILL_IMBALANCE),
              ewc_calibration_p_buy=float(EWC_CALIBRATION_P_BUY),
              ewc_temperature=float(EWC_TEMPERATURE),
              ewc_n_fisher_samples=int(EWC_N_FISHER_SAMPLES),
              episode_length=int(EPISODE_LENGTH),
              iter_to_equilibrium=int(ITER_TO_EQUILIBRIUM),
              regime_distribution=str(REGIME_DISTRIBUTION),
          )
          if WARMSTART_CKPT and os.path.isfile(WARMSTART_CKPT):
              try:
                  meta["warmstart_ckpt_mtime"] = float(os.path.getmtime(WARMSTART_CKPT))
                  meta["warmstart_ckpt_size"] = int(os.path.getsize(WARMSTART_CKPT))
              except OSError:
                  pass
          return meta

      _ewc_current_meta = _ewc_build_metadata()
      _ewc_strict_keys = [
          "warmstart_ckpt", "warmstart_ckpt_mtime", "warmstart_ckpt_size",
          "input_dim", "n_actions",
          "use_distributional", "dist_atoms", "dist_v_min", "dist_v_max",
          "n_hidden", "n_neurons", "activation", "elu_alpha", "dropout_level",
          "use_dueling", "pure_mm",
          "use_flow_ewma", "use_fill_imbalance",
          "ewc_calibration_p_buy", "ewc_temperature",
          "episode_length",
      ]
      _ewc_warn_keys = [
          "ewc_n_fisher_samples", "iter_to_equilibrium", "regime_distribution",
      ]

      _ewc_fd = None
      if os.path.isfile(EWC_FISHER_PATH):
          _cached_fd = load_fisher(EWC_FISHER_PATH)
          ok_ewc, errs_ewc, warns_ewc = _cached_fd.check_compatibility(
              _ewc_current_meta,
              strict_keys=_ewc_strict_keys,
              warn_keys=_ewc_warn_keys,
          )
          # Also validate that cached anchor shapes match the current q_net
          # (defense against loading a Fisher from a different architecture).
          if ok_ewc:
              for _nm, _p in deep_controller.q_net.named_parameters():
                  if _nm in _cached_fd.fisher and tuple(_cached_fd.fisher[_nm].shape) != tuple(_p.shape):
                      ok_ewc = False
                      errs_ewc.append(
                          f"fisher shape mismatch for '{_nm}': "
                          f"cached={tuple(_cached_fd.fisher[_nm].shape)} "
                          f"vs current={tuple(_p.shape)}"
                      )
                      break

          if ok_ewc:
              _ewc_fd = _cached_fd
              print(f"[EWC] Loaded cached Fisher from {EWC_FISHER_PATH}")
              print(f"[EWC] n_samples={_ewc_fd.n_samples} tau={_ewc_fd.temperature}")
              for w in warns_ewc:
                  print(f"[EWC WARN] metadata drift: {w}")
          else:
              print(f"[EWC] Cached Fisher at {EWC_FISHER_PATH} is INCOMPATIBLE "
                    f"with the current setup:")
              for e in errs_ewc:
                  print(f"    x {e}")
              print(f"[EWC] Discarding cache and recomputing. "
                    f"(Delete the file manually if you want to keep it.)")

      if _ewc_fd is None:
          if not os.path.isfile(EWC_FISHER_PATH):
              print(f"[EWC] No cached Fisher at {EWC_FISHER_PATH}")
          print(f"[EWC] Collecting state-action samples from frozen Phase A policy...")

          # -- Step 1: collect (state, action) pairs from Phase A --------------------------------------------------------------
          # Run EWC_N_COLLECT_EPISODES episodes of the frozen Phase A
          # policy at p_buy = EWC_CALIBRATION_P_BUY (symmetric environment),
          # pooling the RL state vectors and the actions taken.  Stop
          # once we have EWC_N_FISHER_SAMPLES samples.
          _saved_learning_ewc = deep_controller.enable_learning
          _saved_epsilon_ewc = float(getattr(deep_controller, "epsilon", 0.0))
          _saved_qnet_training_ewc = deep_controller.q_net.training
          _saved_target_training_ewc = (
              deep_controller.target_net.training
              if getattr(deep_controller, "target_net", None) is not None
              else None
          )
          deep_controller.enable_learning = False
          deep_controller.q_net.eval()
          if _saved_target_training_ewc is not None:
              deep_controller.target_net.eval()
          try:
              deep_controller.epsilon = 0.0
          except Exception:
              pass

          _sa_pool: list = []   # list of (state_tensor_1d, action_int)

          try:
              _ewc_ep = 0
              while len(_sa_pool) < EWC_N_FISHER_SAMPLES and _ewc_ep < EWC_N_COLLECT_EPISODES * 10:
                  _ewc_seed = GLOBAL_SEED + 10_000 + _ewc_ep

                  # Flat symmetric p_buy for the Phase A environment
                  _ewc_regime_kwargs = dict(
                      seed=_ewc_seed + 500_000,
                      n_mo_events=EXPECTED_MO_PER_EPISODE,
                      p_lo=float(EWC_CALIBRATION_P_BUY),
                      p_hi=float(EWC_CALIBRATION_P_BUY),
                      distribution=REGIME_DISTRIBUTION,
                  )
                  if REGIME_DISTRIBUTION == "pareto":
                      _ewc_regime_kwargs["L_min"] = REGIME_L_MIN
                      _ewc_regime_kwargs["alpha"] = REGIME_ALPHA
                  else:
                      _ewc_regime_kwargs["exp_rate"] = 1.0 / REGIME_EXP_TAU
                  _ewc_buy_mo_prob = make_regime_schedule(**_ewc_regime_kwargs)

                  _, _, _ewc_mm_df = simulate_LOB_with_MM(
                      lam=lam, mu=mu, delta=delta,
                      number_tick_levels=NUMBER_TICK_LEVELS,
                      n_priority_ranks=N_PRIORITY_RANKS,
                      number_levels_to_store=20,
                      p0=100, mean_size_LO=mean_size_LO, mean_size_MO=mean_size_MO,
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
                      random_seed=_ewc_seed,
                      buy_mo_prob=_ewc_buy_mo_prob,
                      qrm_params=qrm_params,
                      ewma_alpha=EWMA_ALPHA,
                      mo_flow_ewma_alpha=None,
                      mo_flow_fast_ewma_alpha=FAST_FLOW_EWMA_ALPHA,
                      fill_imbalance_alpha=FILL_IMBALANCE_ALPHA,
                      quote_exposure_imbalance_signal=bool(USE_QUOTE_EXPOSURE_IMBALANCE),
                      bayes_flow_enabled=USE_BAYES_FLOW_SIGNAL,
                      bayes_flow_tau_r=float(BAYES_FLOW_TAU_R),
                      bayes_flow_p_low=float(REGIME_P_LO),
                      bayes_flow_p_high=float(REGIME_P_HI),
                      bayes_flow_prior_mode=BAYES_FLOW_PRIOR_MODE,
                      bayes_flow_cp_window=int(BAYES_FLOW_CP_WINDOW),
                      bayes_flow_use_mixture=bool(USE_MIXTURE),
                      bayes_flow_mixture_taus=list(BAYES_MIXTURE_TAUS),
                  )

                  # Extract (state_vector, action) pairs from this episode.
                  # Rows without a valid state vector are skipped, along
                  # with rows where the action index is -1 (sentinel for
                  # throttled steps where the controller did not act).
                  if "MM_RL_State_Vector" in _ewc_mm_df.columns:
                      _svecs = _ewc_mm_df["MM_RL_State_Vector"].values
                  else:
                      _svecs = np.array([], dtype=object)

                  # MM_LOB_SIM writes the integer action index as
                  # "MM_ActionIdx" (no underscore); other column names
                  # tried here cover historic variants.  Check all in
                  # priority order.
                  _acts = None
                  _acts_col = None
                  for _col in ("MM_ActionIdx", "MM_RL_Action_Idx",
                               "MM_Action_Idx", "MM_Action_Index"):
                      if _col in _ewc_mm_df.columns:
                          _acts = _ewc_mm_df[_col].values
                          _acts_col = _col
                          break
                  if _acts is None:
                      raise RuntimeError(
                          "[EWC] mm_df does not expose an action index column. "
                          f"Available columns: {list(_ewc_mm_df.columns)[:20]}..."
                      )
                  if _ewc_ep == 0:
                      print(f"[EWC] using action column '{_acts_col}'")

                  _added = 0
                  for _sv, _a in zip(_svecs, _acts):
                      if not isinstance(_sv, (list, np.ndarray)):
                          continue
                      if _a is None:
                          continue
                      try:
                          _a_int = int(_a)
                      except (TypeError, ValueError):
                          continue
                      # -1 = throttled step (MM did not act).  Skip.
                      if _a_int < 0:
                          continue
                      _state_t = torch.as_tensor(_sv, dtype=torch.float32)
                      _sa_pool.append((_state_t, _a_int))
                      _added += 1
                      if len(_sa_pool) >= EWC_N_FISHER_SAMPLES:
                          break
                  _ewc_ep += 1
                  print(f"[EWC] collect ep {_ewc_ep}: +{_added} samples "
                        f"(pool: {len(_sa_pool)}/{EWC_N_FISHER_SAMPLES})")

          finally:
              deep_controller.enable_learning = _saved_learning_ewc
              try:
                  deep_controller.epsilon = _saved_epsilon_ewc
              except Exception:
                  pass
              if _saved_qnet_training_ewc:
                  deep_controller.q_net.train()
              else:
                  deep_controller.q_net.eval()
              if _saved_target_training_ewc is not None:
                  if _saved_target_training_ewc:
                      deep_controller.target_net.train()
                  else:
                      deep_controller.target_net.eval()

          if len(_sa_pool) == 0:
              raise RuntimeError(
                  "[EWC] Sample collection yielded 0 valid (state, action) pairs. "
                  "Check that the MM controller is writing MM_RL_State_Vector and "
                  "an action index column to mm_df."
              )
          if len(_sa_pool) < EWC_N_FISHER_SAMPLES:
              print(f"[EWC WARNING] Only collected {len(_sa_pool)} / "
                    f"{EWC_N_FISHER_SAMPLES} samples -- estimating Fisher on what we have.")

          # Randomly downsample if we over-collected (should not happen
          # with current budget, but defensive).
          if len(_sa_pool) > EWC_N_FISHER_SAMPLES:
              _rng_ewc = np.random.default_rng(GLOBAL_SEED)
              _idx = _rng_ewc.choice(len(_sa_pool), size=EWC_N_FISHER_SAMPLES,
                                     replace=False)
              _sa_pool = [_sa_pool[int(i)] for i in _idx]

          # -- Step 2: build q_value_fn closure over q_net --------------------------------------------------------------
          # For distributional C51, DeepQNetwork.forward ALREADY applies
          # softmax over atoms and returns q_probs of shape
          # [B, n_actions, n_atoms] (see dqn_distributional_with_throttle.py
          # line 726).  We must NOT apply softmax again -- the adapter
          # just multiplies by the support and sums along the atom axis.
          #
          # For the scalar Q head, forward returns [B, n_actions] directly.
          _device_qn = next(deep_controller.q_net.parameters()).device
          _probe = _sa_pool[0][0].unsqueeze(0).to(_device_qn)
          with torch.no_grad():
              _probe_out = deep_controller.q_net(_probe)
          if isinstance(_probe_out, tuple):
              _probe_out = _probe_out[0]

          if _probe_out.dim() == 3:
              # Use the controller's own support buffer -- this is the
              # ground truth for the C51 atom positions.  We do NOT
              # rebuild it from best_params because a checkpoint could
              # have been trained with different v_min/v_max than the
              # current config.
              if not getattr(deep_controller, "use_distributional", False):
                  raise RuntimeError(
                      "[EWC] q_net output is 3D but controller.use_distributional is False")
              _support_t = deep_controller.support
              if _support_t is None:
                  raise RuntimeError(
                      "[EWC] deep_controller.support is None -- cannot compute E[Z_a]")
              _support_t = _support_t.to(_device_qn).view(1, 1, -1)  # [1, 1, n_atoms]
              _n_atoms = int(_probe_out.shape[-1])
              if _support_t.shape[-1] != _n_atoms:
                  raise RuntimeError(
                      f"[EWC] support size {_support_t.shape[-1]} does not match "
                      f"n_atoms {_n_atoms} returned by q_net")
              print(f"[EWC] C51 distributional head detected: "
                    f"n_atoms={_n_atoms}, "
                    f"support=[{float(deep_controller.support.min()):.2f}, "
                    f"{float(deep_controller.support.max()):.2f}]")

              def _q_value_fn(state_batch):
                  out = deep_controller.q_net(state_batch)
                  if isinstance(out, tuple):
                      out = out[0]
                  # out is ALREADY probabilities (softmax applied in forward).
                  # Compute E[Z_a(s)] = Sigma_i z_i  *  p_i(s, a) directly.
                  if out.dim() != 3:
                      raise RuntimeError(
                          f"[EWC] expected 3D C51 probs, got {tuple(out.shape)}")
                  return (out * _support_t).sum(dim=-1)    # [B, n_actions]

          elif _probe_out.dim() == 2:
              # [B, n_actions] -- scalar Q head, return directly
              print(f"[EWC] Scalar Q head detected: n_actions={_probe_out.shape[1]}")
              def _q_value_fn(state_batch):
                  out = deep_controller.q_net(state_batch)
                  if isinstance(out, tuple):
                      out = out[0]
                  return out
          else:
              raise RuntimeError(
                  f"[EWC] Unexpected q_net output shape: {tuple(_probe_out.shape)}"
              )

          # -- Step 3: estimate Fisher diagonal --------------------------------------------------------------
          _ewc_fd = estimate_fisher_diagonal_boltzmann(
              q_net=deep_controller.q_net,
              q_value_fn=_q_value_fn,
              state_action_samples=_sa_pool,
              temperature=EWC_TEMPERATURE,
              device=next(deep_controller.q_net.parameters()).device,
              verbose=True,
              metadata=_ewc_current_meta,
          )
          save_fisher(EWC_FISHER_PATH, _ewc_fd)
          print(f"[EWC] Saved Fisher to {EWC_FISHER_PATH}")

      # -- Step 4: attach to controller (anchor loss becomes EWC) --------------------------------------------------------------
      deep_controller.set_ewc_fisher(_ewc_fd.fisher)
      deep_controller.set_anchor_lambda(EWC_LAMBDA)
      print(f"[EWC] Anchor loss upgraded to Fisher-weighted EWC with lambda={EWC_LAMBDA}")
      print(_ewc_fd.summary())

      # Clear per-episode accumulators polluted by the sample-collection
      # rollouts so episode 1 of training starts with clean stats.
      deep_controller.reset_episode_accumulators()

  episode_stats = []

  # Early stopping state
  _es_best_ma = float("-inf")
  _es_best_ep = 0
  _es_triggered = False

  # Best-checkpoint state (tracked via MA(BEST_CKPT_WINDOW) of final_pnl)
  _best_ckpt_ma = float("-inf")
  _best_ckpt_ep = -1
  _best_ckpt_path = None

  # --- Pre-loop anchor initialisation --------------------------------------------------------------
  # If the curriculum-tied anchor mechanism is active, we must set lambda
  # BEFORE the first training step so episode 1 trains under the
  # tied-derived value (not the constant ANCHOR_LAMBDA that was
  # assigned during warmstart).  Without this, a user who sets
  # TIED_ANCHOR_BASE != ANCHOR_LAMBDA would silently train episode 1
  # with a stale lambda.  The tied update inside the training loop fires
  # AFTER report_episode, so it only affects episodes 2+ on its own.
  if USE_CURRICULUM_TIED_ANCHOR and adr_curriculum is not None:
      _init_delta = float(adr_curriculum.delta)
      _init_tied_lambda = get_tied_anchor_lambda(
          delta=_init_delta,
          delta_max=ADR_LITE_DELTA_MAX,
          base=TIED_ANCHOR_BASE,
          slope=TIED_ANCHOR_SLOPE,
          gate=TIED_ANCHOR_GATE,
      )
      deep_controller.set_anchor_lambda(_init_tied_lambda)
      print(f"[ANCHOR TIED] pre-loop init: delta={_init_delta:.3f}  "
            f"lambda={_init_tied_lambda:.4f}  "
            f"(overrides ANCHOR_LAMBDA={ANCHOR_LAMBDA} for ep 1)")

  for ep in range(N_EPISODES):
      if VERBOSE_LOGGING:
          print("=" * 60)
          print(f"Starting episode {ep + 1}/{N_EPISODES}")
          print(f"Current epsilon BEFORE episode: {deep_controller.epsilon:.4f}")
          print(f"Controller mode: {'PURE MM' if deep_controller.pure_mm else 'GENERIC'}")
          if USE_ADVERSARIAL:
              if ep < ADV_WARMUP_EPISODES:
                  print(f"MO flow: ADVERSARIAL (warmup -- p=0.5, ep {ep+1}/{ADV_WARMUP_EPISODES})")
              else:
                  print(f"MO flow: ADVERSARIAL (phase={adv_phase}, eps_adv={adversary.epsilon:.3f})")
          elif USE_REGIME_SWITCH:
              if REGIME_DISTRIBUTION == "exponential":
                  print(f"MO flow: REGIME-SWITCHING (exponential, tau={REGIME_EXP_TAU}, p in [{REGIME_P_LO},{REGIME_P_HI}])")
              else:
                  print(f"MO flow: REGIME-SWITCHING (pareto, alpha={REGIME_ALPHA}, L_min={REGIME_L_MIN}, p in [{REGIME_P_LO},{REGIME_P_HI}])")
              if ep == 0:
                  print(f"  [EWMA] mo_flow_alpha={EWMA_ALPHA:.6f}, fill_imbalance_alpha={FILL_IMBALANCE_ALPHA:.6f} (fixed)")
          elif USE_ADV_TAU:
              print(f"MO flow: ADV_TAU BANDIT (mode={ADV_TAU_MODE}, eps={adv_tau.epsilon:.3f})")
          else:
              print(f"MO flow: STATIONARY (buy_mo_prob=0.5)")

          if hasattr(deep_controller, "optimizer"):
              current_lr = deep_controller.optimizer.param_groups[0]["lr"]
              print(f"Current learning rate: {current_lr:.8f}")

      # --------------------------------------------------------------
      # Per-episode random seed
      # --------------------------------------------------------------
      # Each episode gets a unique seed = GLOBAL_SEED + 1 + ep.
      # This ensures:
      #   (a) Different episodes see different LOB environments (order
      #       flow, cancels, market orders all differ).
      #   (b) Given the same GLOBAL_SEED, the SEQUENCE of episodes is
      #       perfectly reproducible across runs.
      #   (c) The RL exploration noise (epsilon-greedy, NoisyNets) is also
      #       reproducible per episode.
      # --------------------------------------------------------------
      current_ep_seed = GLOBAL_SEED + 1_000 + ep

      # 1. NumPy -- controls LOB event sampling inside simulate_LOB_with_MM
      np.random.seed(current_ep_seed)

      # 2. Python built-in -- used by any stdlib random calls
      random.seed(current_ep_seed)

      # 3. PyTorch -- controls NoisyLinear noise, dropout, and any GPU ops
      torch.manual_seed(current_ep_seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed(current_ep_seed)

      # ==============================================================
      # EPISODE BOUNDARY RESET -- Clean controller state between episodes
      # ==============================================================
      # Without a full reset, state from episode N leaks into episode N+1:
      #
      #   last_action_idx : stale action index -> learn() creates a phantom
      #       transition at the start of the next episode using old action.
      #
      #   nstep_buffer : leftover SMDP-level transitions from the previous
      #       episode would be mixed with new episode data, producing
      #       cross-episode n-step returns that are mathematically wrong.
      #
      #   SMDP state : a pending cumulative-reward aggregation from the
      #       end of episode N would carry over, tainting the first
      #       transition of episode N+1.
      #
      #   Throttle counters : stale values for event_steps, moves_tob,
      #       last_update_time, last_inventory, and last_mode from the
      #       end of episode N can cause:
      #       - False fill-replenishment bypass at the start of N+1
      #         (last_inventory != new starting inventory -> act thinks
      #          a fill happened when it didn't).
      #       - False mode-change bypass (last_mode from old episode
      #         differs from the new starting mode).
      #       - Time gate blocking if last_update_time is far in the
      #         future relative to the new episode's time=0.
      #
      # NOTE: If simulate_LOB_with_MM sends done=True on the final step,
      # learn() already clears last_action_idx, nstep_buffer, and SMDP.
      # We reset them again here DEFENSIVELY -- the cost is negligible and
      # it protects against edge cases where done might not fire.
      # ==============================================================

      # --- RL pipeline state ---
      deep_controller.last_action_idx = None
      deep_controller.nstep_buffer.clear()
      deep_controller._smdp_reset()

      # --- Throttle gate counters ---
      # Reset event / TOB / time gate counters so the first event in the
      # new episode is not gated by stale counters from the previous one.
      deep_controller.moves_tob = 0
      deep_controller.last_env_tob_key = None
      deep_controller.event_steps = 0
      deep_controller.last_update_time = -1.0  # -1 = "never decided" sentinel

      # --- First-action bypass ---
      # BUG FIX (A): reset so the first step of each episode bypasses
      # all throttle gates, allowing the agent to place initial quotes.
      deep_controller._has_acted_once = False

      # --- Bypass priority trackers ---
      # Setting these to None ensures act() does not trigger a false
      # fill-replenishment or mode-change bypass on the very first step.
      deep_controller.last_inventory = None
      deep_controller.last_mode = None

      # -- MO flow schedule --------------------------------------------------------------
      # Four modes (mutually exclusive):
      #   1. USE_ADVERSARIAL:  adversary DQN picks p_buy at regime boundaries
      #   2. USE_REGIME_SWITCH: random domain-randomization schedule
      #   3. USE_ADV_TAU:      bandit picks tau per episode, builds exponential schedule
      #   4. None:             stationary p=0.5
      _episode_mo_flow_alpha = None       # set by USE_ADV_TAU if active
      _episode_mo_flow_fast_alpha = None  # set by USE_ADV_TAU if active
      _episode_fill_alpha = None          # set by USE_ADV_TAU if active
      _episode_bayes_tau = BAYES_FLOW_TAU_R

      # -- Periodic alpha refresh (USE_REGIME_SWITCH) --------------------------------------------------------------
      if USE_REGIME_SWITCH and ep > 0 and ep % RECALIB_INTERVAL == 0:
          FILL_IMBALANCE_ALPHA = _recalibrate_fill_alpha(_tau_for_fill_alpha)

      if USE_ADVERSARIAL and ep >= ADV_WARMUP_EPISODES:
          # Alternating training: asymmetric freeze (MM gets more episodes)
          adv_phase_counter += 1
          _current_freeze = ADV_FREEZE_MM if adv_phase == "mm" else ADV_FREEZE_ADV
          if adv_phase_counter > _current_freeze:
              adv_phase = "adversary" if adv_phase == "mm" else "mm"
              adv_phase_counter = 1
          deep_controller.enable_learning = (adv_phase == "mm")

          ep_buy_mo_prob = make_adversarial_schedule(
              adversary=adversary,
              n_mo_events=EXPECTED_MO_PER_EPISODE,
              inv_limit=best_params.get("inv_limit") or 8,
              seed=current_ep_seed + 500_000,
          )
      elif USE_REGIME_SWITCH:
          # -- Progressive / ADR-lite curriculum: per-episode p_bounds --
          # - USE_ADR_LITE_CURRICULUM: adaptive thermostat always trains on
          #   the full interval [0.5-delta, 0.5+delta], replacing the linear ramp
          #   with a performance-gated advance/retreat loop.
          # - USE_PROGRESSIVE_CURRICULUM: linear ramp from narrow -> wide.
          # - Otherwise: static REGIME_P_LO / REGIME_P_HI every episode.
          _adr_mode = None
          if USE_ADR_LITE_CURRICULUM:
              # Reset per-episode gate + SC trackers BEFORE running
              _adr_gate_tracker["sum"] = 0.0
              _adr_sc_tracker["sum"]   = 0.0
              _ep_p_lo, _ep_p_hi, _adr_mode = adr_curriculum.sample_episode_config()
          elif USE_PROGRESSIVE_CURRICULUM:
              _curriculum_progress = min(1.0, (ep + 1) / (N_EPISODES * CURRICULUM_RAMP_FRAC))
              _ep_p_lo, _ep_p_hi = get_curriculum_p_bounds(
                  _curriculum_progress,
                  delta_start=CURRICULUM_DELTA_START,
                  delta_end=CURRICULUM_DELTA_END,
              )
          else:
              _ep_p_lo = REGIME_P_LO
              _ep_p_hi = REGIME_P_HI

          _regime_kwargs = dict(
              seed=current_ep_seed + 500_000,
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
      elif USE_ADV_TAU:
          # -- Alternating freeze schedule --------------------------------------------------------------
          # Same principle as the p_buy adversary:
          #   - During warmup (ep < ADV_TAU_WARMUP_EPISODES): MM trains
          #     with random tau (uniform from TAU_GRID), bandit inactive.
          #   - After warmup: alternating phases.
          #     MM phase: MM learns, bandit selects tau but does NOT update.
          #     ADV phase: MM frozen, bandit updates scores from episode outcomes.
          #   - At the START of each MM phase: refresh the baseline
          #     fill_imbalance alpha using a representative tau.
          if ep >= ADV_TAU_WARMUP_EPISODES:
              adv_tau_phase_counter += 1
              _current_freeze = (ADV_TAU_FREEZE_MM if adv_tau_phase == "mm"
                                 else ADV_TAU_FREEZE_ADV)
              if adv_tau_phase_counter > _current_freeze:
                  adv_tau_phase = "adversary" if adv_tau_phase == "mm" else "mm"
                  adv_tau_phase_counter = 1
                  # Use the adversary's current best tau as the representative
                  # baseline tau for the next MM phase.
                  if adv_tau_phase == "mm":
                      _repr_tau = float(adv_tau.summary()["best_tau"])
                      FILL_IMBALANCE_ALPHA = _recalibrate_fill_alpha(_repr_tau)
              deep_controller.enable_learning = (adv_tau_phase == "mm")

          # Bandit selects tau (active in all phases -- the MM always sees
          # varied persistence, but the bandit only updates during ADV phase).
          _adv_tau_arm = adv_tau.select_action()
          _chosen_tau = adv_tau.get_tau()
          _episode_bayes_tau = float(_chosen_tau)
          _episode_mo_flow_alpha = adv_tau.get_ewma_alpha()  # None if alpha_fixed
          _episode_mo_flow_fast_alpha = None
          if _episode_mo_flow_alpha is not None:
              _episode_mo_flow_fast_alpha = _compute_mo_flow_alpha(
                  max(float(_chosen_tau) / max(float(FAST_FLOW_TAU_DIVISOR), 1.0), 1.0)
              )

          # Calibrate fill_imbalance_alpha to the chosen tau, rescaled to the
          # all-event clock used by update_fill_imbalance_ewma(...).
          _episode_fill_alpha = adv_tau.get_fill_imbalance_alpha()  # None if alpha_fixed

          # Build exponential regime schedule with the chosen tau.
          ep_buy_mo_prob = make_regime_schedule(
              seed=current_ep_seed + 500_000,
              n_mo_events=EXPECTED_MO_PER_EPISODE,
              p_lo=REGIME_P_LO,
              p_hi=REGIME_P_HI,
              distribution="exponential",
              exp_rate=1.0 / _chosen_tau,
          )
      else:
          ep_buy_mo_prob = 0.5

      _replay_delta = float("nan")
      if USE_REGIME_SWITCH:
          _replay_delta = float((_ep_p_hi - _ep_p_lo) / 2.0)
      elif USE_ADV_TAU:
          _replay_delta = float((REGIME_P_HI - REGIME_P_LO) / 2.0)
      elif not USE_ADVERSARIAL:
          _replay_delta = 0.0
      deep_controller.set_replay_context(ep, _replay_delta)

      # Run one training episode with Deep RL controller
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
          path_save_files=None,         # or a path if you want CSVs
          label_simulation=None,
          beta_exp_weighted_return=0.0,
          intensity_exp_weighted_return=0.0,
          controller=deep_controller,   # DeepRLController acts as MM policy
          mm_policy=None,               # not used when controller is given
          exclude_self_from_state=False,
          reward_fn = reward_fn_spread,
          random_seed=current_ep_seed,
          buy_mo_prob=ep_buy_mo_prob,
          qrm_params=qrm_params,
          ewma_alpha=EWMA_ALPHA,
          mo_flow_ewma_alpha=_episode_mo_flow_alpha,
          mo_flow_fast_ewma_alpha=(_episode_mo_flow_fast_alpha
                                   if _episode_mo_flow_fast_alpha is not None
                                   else FAST_FLOW_EWMA_ALPHA),
          fill_imbalance_alpha=(_episode_fill_alpha if _episode_fill_alpha is not None
                                else FILL_IMBALANCE_ALPHA),
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

      # -- Post-episode adversary update --------------------------------------------------------------
      if USE_ADVERSARIAL and ep >= ADV_WARMUP_EPISODES:
          # Flush the terminal regime transition
          ep_buy_mo_prob.flush_final_regime(done=True)

          # Store all regime transitions into the adversary replay buffer
          for t in ep_buy_mo_prob.transitions:
              adversary.store_transition(*t)

          n_regimes = len(ep_buy_mo_prob.transitions)
          adv_loss_sum = 0.0
          adv_updates = 0

          # Train adversary only during its phase
          if adv_phase == "adversary" and adversary.can_sample():
              for _ in range(n_regimes):
                  loss = adversary.update()
                  if loss is not None:
                      adv_loss_sum += loss
                      adv_updates += 1

          # Target network sync -- count only adversary-learning episodes
          if adv_phase == "adversary" and adv_updates > 0:
              adv_train_ep_count += 1
              if adv_train_ep_count % adversary.target_update_freq == 0:
                  adversary.update_target()

          if adv_phase == "adversary":
              adversary.decay_epsilon()

          # Re-enable MM learning (in case it was disabled during adversary phase)
          deep_controller.enable_learning = True

          # Log adversary stats to TensorBoard
          p_buys_chosen = [adversary.get_p_buy(t[1]) for t in ep_buy_mo_prob.transitions]
          deep_controller.writer.add_scalar("adversary/phase", 0.0 if adv_phase == "mm" else 1.0, ep)
          deep_controller.writer.add_scalar("adversary/epsilon", adversary.epsilon, ep)
          deep_controller.writer.add_scalar("adversary/n_regimes", n_regimes, ep)
          deep_controller.writer.add_scalar("adversary/mean_p_buy",
                                            float(np.mean(p_buys_chosen)) if p_buys_chosen else 0.5, ep)
          if adv_updates > 0:
              deep_controller.writer.add_scalar("adversary/loss", adv_loss_sum / adv_updates, ep)
          _mean_pb = float(np.mean(p_buys_chosen)) if p_buys_chosen else 0.5
          if VERBOSE_LOGGING:
              print(f"  [ADV] phase={adv_phase}, regimes={n_regimes}, "
                    f"mean_p_buy={_mean_pb:.3f}, epsilon={adversary.epsilon:.3f}")

      elif USE_REGIME_SWITCH and callable(ep_buy_mo_prob):
          _rs_n_regimes = len(getattr(ep_buy_mo_prob, "p_values", []))
          _rs_p_values = getattr(ep_buy_mo_prob, "p_values", [])
          _rs_mean_pb = float(np.mean(_rs_p_values)) if len(_rs_p_values) > 0 else 0.5
          deep_controller.writer.add_scalar("regime_switch/n_regimes", _rs_n_regimes, ep)
          deep_controller.writer.add_scalar("regime_switch/mean_p_buy", _rs_mean_pb, ep)

          # -- Progressive curriculum TensorBoard signals --------------------------------------------------------------
          # These scalars let you visualize the curriculum ramp in TB
          # and correlate difficulty changes with PnL / TD loss / inventory.
          #
          # Logged signals:
          #   curriculum/p_lo       -- lower bound of p_buy range this episode
          #   curriculum/p_hi       -- upper bound of p_buy range this episode
          #   curriculum/delta_p    -- half-width = (p_hi - p_lo) / 2
          #
          # Only logged when USE_PROGRESSIVE_CURRICULUM is True (the static
          # bounds are already visible in the config printout at startup).
          if USE_PROGRESSIVE_CURRICULUM:
              deep_controller.writer.add_scalar("curriculum/p_lo", _ep_p_lo, ep)
              deep_controller.writer.add_scalar("curriculum/p_hi", _ep_p_hi, ep)
              deep_controller.writer.add_scalar("curriculum/delta_p",
                                                (_ep_p_hi - _ep_p_lo) / 2.0, ep)

          if VERBOSE_LOGGING:
              _bounds_str = (f", pin[{_ep_p_lo:.3f},{_ep_p_hi:.3f}]"
                             if USE_PROGRESSIVE_CURRICULUM else "")
              print(f"  [RS] regimes={_rs_n_regimes}, mean_p_buy={_rs_mean_pb:.3f}{_bounds_str}")

      # --------------------------------------------------------------
      # ENVIRONMENT-LEVEL STATS (what actually happened)
      # --------------------------------------------------------------
      if VERBOSE_LOGGING:
          print("\n=== ENVIRONMENT STATS (EPISODE) ===")
          print("Env Type counts:\n", msg_df["Type"].value_counts())
          print("Env Direction counts:\n", msg_df["Direction"].value_counts())
          print("Spread range:", msg_df["Spread"].min(), msg_df["Spread"].max())
          print("MidPrice range:", msg_df["MidPrice"].min(), msg_df["MidPrice"].max())

          print("\n=== MARKET MAKER STATS (EPISODE, ENV) ===")
          print("MM inventory range:", mm_df["MM_Inventory"].min(), mm_df["MM_Inventory"].max())
          print("Any MM PnL movement?:", mm_df["MM_TotalPnL"].ne(0).any())

      # Episode reward and PnL
      if "MM_CumReward" in mm_df.columns:
          total_reward = float(mm_df["MM_CumReward"].iloc[-1])
      elif "MM_Reward" in mm_df.columns:
          total_reward = float(mm_df["MM_Reward"].sum())
      else:
          total_reward = 0.0

      final_pnl = float(mm_df["MM_TotalPnL"].iloc[-1])

      # --------------------------------------------------------------
      # Inventory diagnostics
      # --------------------------------------------------------------
      #   max_abs_inv   -- worst-case inventory exposure during the episode
      #   mean_abs_inv  -- average absolute inventory (lower = tighter control)
      #   pct_at_limit  -- fraction of steps where |inv| >= inv_limit (breach %)
      # --------------------------------------------------------------
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

      # -- ADR-lite v3: report episode result, log state, maybe advance/retreat -
      # The thermostat consumes `final_pnl` directly (clipped internally
      # to ADR_LITE_PNL_CLIP).  No baseline, no gate signal, no relative
      # threshold -- see adr_lite.py for the faithful Akkaya 2019
      # Algorithm 1 adaptation.
      if USE_ADR_LITE_CURRICULUM and adr_curriculum is not None:
          _ep_gate_signal    = float(_adr_gate_tracker["sum"])   # diagnostic only
          _ep_spread_capture = float(_adr_sc_tracker["sum"])     # diagnostic only

          _adr_events = adr_curriculum.report_episode(
              final_pnl=float(final_pnl),
              pct_at_inv_limit=pct_at_limit,
          )

          # TensorBoard: log curriculum state + per-episode diagnostics.
          # The thermostat is driven by clipped final_pnl; gate_signal and
          # spread_capture are kept as auxiliary diagnostics.
          _adr_state = adr_curriculum.get_state()
          for _k, _v in _adr_state.items():
              deep_controller.writer.add_scalar(f"adr_lite/{_k}", _v, ep)
          deep_controller.writer.add_scalar("adr_lite/episode_final_pnl",
                                            float(final_pnl), ep)
          deep_controller.writer.add_scalar("adr_lite/episode_metric_clipped",
                                            float(_adr_events.get("metric_clipped", 0.0)),
                                            ep)
          deep_controller.writer.add_scalar("adr_lite/episode_gate_signal",
                                            _ep_gate_signal, ep)
          deep_controller.writer.add_scalar("adr_lite/episode_spread_capture",
                                            _ep_spread_capture, ep)
          if _adr_events.get("p_bar") is not None:
              deep_controller.writer.add_scalar("adr_lite/p_bar_at_decision",
                                                float(_adr_events["p_bar"]), ep)

          # Stdout event log when a batch decision fires.
          #
          # There are four possible outcomes of a decision once the
          # opt-in extensions are considered:
          #   ADVANCE  -- delta actually moved up
          #   RETREAT  -- delta actually moved down
          #   PENDING  -- advance signal fired but the multi-confirmation
          #              counter has not yet reached K(delta), so delta stayed
          #              where it was (only possible when
          #              ADVANCE_CONFIRMATION_MODE == "formula" and
          #              K(delta) > 1 at the current delta)
          #   HOLD     -- p_bar inside the dead zone, nothing happened
          # When the multi-confirmation extension is disabled (the
          # default), PENDING is unreachable and the log falls back to
          # the legacy three-state labeling.
          if _adr_events.get("buffer_evaluated"):
              if _adr_events["advance"]:
                  _action_str = "ADVANCE"
              elif _adr_events["retreat"]:
                  _action_str = "RETREAT"
              elif _adr_events.get("advance_pending"):
                  _k_needed = _adr_events.get("advances_needed", 1)
                  _csa = _adr_events.get("consecutive_advance_signals", 0)
                  _action_str = f"PENDING({_csa}/{_k_needed})"
              else:
                  _action_str = "HOLD   "
              _t_H_now = _adr_events.get("t_H_at_decision")
              _t_L_now = _adr_events.get("t_L_at_decision")
              print(f"[ADR-LITE] ep={ep+1:4d}  {_action_str}  "
                    f"p_bar={_adr_events['p_bar']:+.4f}  "
                    f"(t_H={_t_H_now:+.3f}, t_L={_t_L_now:+.3f})  "
                    f"inv_ma={_adr_events['inv_ma']:.2%}  "
                    f"-> delta={adr_curriculum.delta:.3f} "
                    f"(pin[{0.5 - adr_curriculum.delta:.2f}, "
                    f"{0.5 + adr_curriculum.delta:.2f}])")

      # Discounted return: G_0 = Sigma_{t=0}^{T-1} gamma^t r_t  (micro-step discount)
      # This is what V(s_0) should converge to under the current policy, making
      # it a direct diagnostic for the C51 support bounds [-2, +2].
      gamma = best_params.get("gamma", 0.995)
      if "MM_Reward" in mm_df.columns:
          rewards = mm_df["MM_Reward"].values.astype(float)
          gammas = gamma ** np.arange(len(rewards))
          discounted_return = float(np.dot(gammas, rewards))
      else:
          discounted_return = 0.0

      # --------------------------------------------------------------
      # NEW: RL STATE SNAPSHOT (FIRST/LAST) + FILL COUNTS
      #      Only computed and printed in verbose mode.
      # --------------------------------------------------------------
      try:
          first_row = mm_df.iloc[0]
          last_row = mm_df.iloc[-1]

          # --------------------------------------------------------------
          # 1) READ RL STATE METADATA (mode, dim, max_offset)
          # --------------------------------------------------------------
          has_rl_meta = all(
              col in mm_df.columns
              for col in [
                  "MM_RL_State_Mode",
                  "MM_RL_State_Dim",
                  "MM_RL_State_MaxOffset",
              ]
          )

          rl_mode = None
          rl_dim = None
          rl_max_off = None

          if has_rl_meta:
              rl_mode = first_row["MM_RL_State_Mode"]
              rl_dim = first_row["MM_RL_State_Dim"]
              rl_max_off = first_row["MM_RL_State_MaxOffset"]

          # --------------------------------------------------------------
          # 2) PURE MM MODE -> DECODE NN INPUT VECTOR
          # --------------------------------------------------------------
          if (rl_mode == "pure_mm") and ("MM_RL_State_Vector" in mm_df.columns):
              # Find first/last rows with a valid state vector (skip None from
              # pre-decision steps where the controller was throttled or hadn't
              # acted yet).
              rl_vec_col = mm_df["MM_RL_State_Vector"]
              valid_mask_vec = rl_vec_col.apply(lambda v: isinstance(v, (list, np.ndarray)))
              if valid_mask_vec.any():
                  vec_start = rl_vec_col[valid_mask_vec].iloc[0]
                  vec_end = rl_vec_col[valid_mask_vec].iloc[-1]
              else:
                  vec_start = first_row["MM_RL_State_Vector"]
                  vec_end = last_row["MM_RL_State_Vector"]

              if isinstance(vec_start, (list, np.ndarray)) and isinstance(vec_end, (list, np.ndarray)):
                  K = int(rl_max_off) if rl_max_off is not None else max(0, (int(rl_dim) - 2) // 2 - 1)

                  def decode_pure_mm(vec):
                      vec = list(vec)
                      spread = float(vec[0])
                      inv = float(vec[1])

                      # [spread, inventory, bid_0..bid_K, ask_0..ask_K]
                      bid_vec = vec[2 : 2 + (K + 1)]
                      ask_vec = vec[2 + (K + 1) : 2 + 2 * (K + 1)]

                      return spread, inv, bid_vec, ask_vec

                  s_spread, s_inv, s_bid, s_ask = decode_pure_mm(vec_start)
                  e_spread, e_inv, e_bid, e_ask = decode_pure_mm(vec_end)

                  if VERBOSE_LOGGING: print("\n=== RL STATE SNAPSHOT (EPISODE, PURE MM NN INPUT) ===")
                  if VERBOSE_LOGGING: print("  (spread = log1p(raw), inventory = raw/inv_limit, depths = log1p(volume))")
                  if VERBOSE_LOGGING: print("Start NN state:")
                  if VERBOSE_LOGGING: print(f"  spread={s_spread:.2f}, inventory={s_inv:.2f}")
                  if VERBOSE_LOGGING: print("  bid_depth (log1p vol):")
                  for j, val in enumerate(s_bid):
                      if VERBOSE_LOGGING: print(f"    level_{j}= {val:.2f}")
                  if VERBOSE_LOGGING: print("  ask_depth (log1p vol):")
                  for j, val in enumerate(s_ask):
                      if VERBOSE_LOGGING: print(f"    level_{j}= {val:.2f}")

                  if VERBOSE_LOGGING: print("End NN state:")
                  if VERBOSE_LOGGING: print(f"  spread={e_spread:.2f}, inventory={e_inv:.2f}")
                  if VERBOSE_LOGGING: print("  bid_depth (log1p vol):")
                  for j, val in enumerate(e_bid):
                      if VERBOSE_LOGGING: print(f"    level_{j}= {val:.2f}")
                  if VERBOSE_LOGGING: print("  ask_depth (log1p vol):")
                  for j, val in enumerate(e_ask):
                      if VERBOSE_LOGGING: print(f"    level_{j}= {val:.2f}")
              else:
                  if VERBOSE_LOGGING: print("\n[Warning] MM_RL_State_Vector not list/ndarray; skipping NN-state decode.")

          else:
              # --------------------------------------------------------------
              # 3) GENERIC MODE -> L1 VIEW
              # --------------------------------------------------------------
              has_state_cols = all(
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

                  if VERBOSE_LOGGING: print("\n=== RL STATE SNAPSHOT (EPISODE, L1 VIEW) ===")
                  if VERBOSE_LOGGING: print("Start state (first action state):")
                  print(
                      f"  spread={start_spread:.1f}, ask_size={start_asksize:.1f}, "
                      f"bid_size={start_bidsize:.1f}, inventory={start_inv:.1f}, "
                      f"has_bid={start_has_bid}, has_ask={start_has_ask}"
                  )
                  if VERBOSE_LOGGING: print("End state (last action state):")
                  print(
                      f"  spread={end_spread:.1f}, ask_size={end_asksize:.1f}, "
                      f"bid_size={end_bidsize:.1f}, inventory={end_inv:.1f}, "
                      f"has_bid={end_has_bid}, has_ask={end_has_ask}"
                  )
              else:
                  print("\n[Warning] MM_State_* columns not found in mm_df; "
                        "RL state snapshot for first/last step is skipped.")

          # --------------------------------------------------------------
          # 4) ALWAYS: FILL STATS
          # --------------------------------------------------------------
          if "MM_HadFill" in mm_df.columns and "MM_LastFillSide" in mm_df.columns:
              had_fill = mm_df["MM_HadFill"] == True
              bid_fills = had_fill & (mm_df["MM_LastFillSide"] == +1)
              ask_fills = had_fill & (mm_df["MM_LastFillSide"] == -1)

              n_bid_fills = int(bid_fills.sum())
              n_ask_fills = int(ask_fills.sum())
              n_total_fills = n_bid_fills + n_ask_fills

              if VERBOSE_LOGGING: print("\n=== FILL STATS (EPISODE) ===")
              if VERBOSE_LOGGING: print(f"Total fills on BID side : {n_bid_fills}")
              if VERBOSE_LOGGING: print(f"Total fills on ASK side : {n_ask_fills}")
              if VERBOSE_LOGGING: print(f"Total fills (bid+ask)   : {n_total_fills}")
          else:
              print("\n[Warning] MM_HadFill / MM_LastFillSide not found in mm_df; "
                    "fill statistics are skipped.")
      except Exception as e:
          # Defensive: never break training because of debug code
          if VERBOSE_LOGGING: print(f"\n[Warning] Exception while computing RL state / fill stats: {e}")

      # --------------------------------------------------------------
      # RL POLICY-LEVEL STATS (what the network chose)
      # --------------------------------------------------------------
      if VERBOSE_LOGGING and deep_controller.pure_mm and deep_controller.pure_mm_offsets is not None:
          print(f"\n=== RL POLICY STATS (EPISODE, PURE MM OFFSETS) ===")
          print(f"Discrete action usage (episode {ep + 1}):")
          for i, (bo, ao) in enumerate(deep_controller.pure_mm_offsets):
              count_i = int(deep_controller.action_counts[i])
              print(f"  a={i:2d}  offsets(bid,ask)=({bo},{ao})  -> used {count_i} times")
      elif VERBOSE_LOGGING:
          print(f"\n=== RL POLICY STATS (EPISODE, GENERIC ACTIONS) ===")
          print(f"Discrete action usage (episode {ep + 1}):")
          # NOTE:
          #   0..5 = core actions
          #   6    = post_bid_ask_inside_spread  (if n_actions >= 7)
          #   7    = post_bid_inside_spread      (if n_actions >= 8)
          #   8    = post_ask_inside_spread      (if n_actions >= 9)
          generic_names = [
              "post_bid",                    # 0
              "post_ask",                    # 1
              "post_bid_ask",                # 2
              "cancel_bid",                  # 3
              "cancel_ask",                  # 4
              "hold",                        # 5
              "post_bid_ask_inside_spread",  # 6
              "post_bid_inside_spread",      # 7
              "post_ask_inside_spread",      # 8
          ]
          if VERBOSE_LOGGING:
              for i, name in enumerate(generic_names):
                  if i >= len(deep_controller.action_counts):
                      break
                  count_i = int(deep_controller.action_counts[i])
                  print(f"  a={i:2d}  {name:>26s}  -> used {count_i} times")

      # Snapshot Q-value / TD-target diagnostics BEFORE log_episode_stats resets them
      _qmin = deep_controller.episode_q_min
      _qmax = deep_controller.episode_q_max
      _tdmin = deep_controller.episode_td_target_min
      _tdmax = deep_controller.episode_td_target_max
      _td_count = deep_controller.episode_td_target_count
      _td_sum = deep_controller.episode_td_target_sum
      _tz_clip_count = deep_controller.episode_tz_clip_count
      _tz_clip_sum = deep_controller.episode_tz_clip_frac_sum

      # Log high-level stats (this will also reset episode_loss_* and action_counts)
      deep_controller.log_episode_stats(ep, total_reward, final_pnl, discounted_return=discounted_return)

      # ==============================================================
      # EPSILON DECAY -- done ONCE per episode, OUTSIDE the environment
      # ==============================================================
      # We decay epsilon BETWEEN episodes (not inside learn()) to ensure
      # the exploration rate stays constant throughout a single episode.
      # This is important because:
      #   (a) Within an episode, the agent commits to a single epsilon-greedy
      #       policy.  Changing epsilon mid-episode breaks the stationarity
      #       assumption for the on-policy SARSA variant.
      #   (b) It makes episode-level statistics comparable -- every step
      #       within the same episode used the same exploration rate.
      #
      # When NoisyNets are enabled (use_noisy_net=True), epsilon is
      # IRRELEVANT -- exploration is driven by learned parameter noise.
      # ==============================================================
      old_eps = deep_controller.epsilon

      # Track MM training episodes for decay schedules (LR, epsilon).
      # When adversarial training is active, only count episodes where
      # the MM actually trained -- otherwise decay budgets are wasted
      # on frozen phases.
      if USE_ADVERSARIAL:
          if adv_phase == "mm":
              mm_train_ep_count += 1
          _mm_ep_for_decay = mm_train_ep_count
          _mm_frac = ADV_FREEZE_MM / (ADV_FREEZE_MM + ADV_FREEZE_ADV)
          _mm_total_for_decay = int(ADV_WARMUP_EPISODES + (N_EPISODES - ADV_WARMUP_EPISODES) * _mm_frac)
      else:
          _mm_ep_for_decay = ep
          _mm_total_for_decay = N_EPISODES

      if True:  # epsilon decay always active (even with NoisyNet)
          # --------------------------------------------------------------
          # EPSILON DECAY -- two strategies selectable via best_params
          # --------------------------------------------------------------
          # "linear"      : epsilon(ep) = epsilon_start - progress  x  (epsilon_start - epsilon_min)
          #                 Straight line from epsilon_start -> epsilon_min over N episodes.
          #                 Keeps exploration HIGH for longer -- recommended when
          #                 the agent is prone to early policy collapse.
          #
          # "exponential" : epsilon(ep) = max(epsilon_min, epsilon_start  x  decay^(ep+1))
          #                 Multiplicative per-episode factor read from
          #                 best_params["epsilon_decay"].  Drops faster in the
          #                 first episodes -- useful when the environment is
          #                 simple and the agent converges quickly.
          # --------------------------------------------------------------
          eps_start = best_params["epsilon_start"]
          eps_min = best_params["epsilon_min"]
          eps_decay_type = best_params.get("epsilon_decay_type", "linear")

          progress = _mm_ep_for_decay / _mm_total_for_decay

          if eps_decay_type == "exponential":
              eps_mult = best_params.get("epsilon_decay", 0.98)
              new_eps = eps_start * (eps_mult ** _mm_ep_for_decay)
          else:
              # Default: linear decay
              new_eps = eps_start - (progress * (eps_start - eps_min))

          # Clamp to floor (guards against floating-point undershoot)
          deep_controller.epsilon = max(eps_min, new_eps)

      if VERBOSE_LOGGING:
          print(f"\nTotal reward (episode {ep + 1}): {total_reward:.4f}")
          print(f"Discounted return G_0 (episode {ep + 1}): {discounted_return:.6f}")
          print(f"Final total PnL (episode {ep + 1}): {final_pnl:.4f}")
          print(f"|inv|_max={max_abs_inv:.0f}  |inv|_mean={mean_abs_inv:.2f}  at_limit={pct_at_limit:.1%}")
          _qmin_s = f"{_qmin:.6f}" if _qmin != float('inf') else "N/A"
          _qmax_s = f"{_qmax:.6f}" if _qmax != float('-inf') else "N/A"
          _tdmin_s = f"{_tdmin:.6f}" if _tdmin != float('inf') else "N/A"
          _tdmax_s = f"{_tdmax:.6f}" if _tdmax != float('-inf') else "N/A"
          _tdmean_s = f"{_td_sum / _td_count:.6f}" if _td_count > 0 else "N/A"
          _tzclip_s = f"{_tz_clip_sum / _tz_clip_count:.1%}" if _tz_clip_count > 0 else "N/A"
          print(f"Q(s,a) range: [{_qmin_s}, {_qmax_s}]  (C51 support: [{deep_controller.dist_v_min}, {deep_controller.dist_v_max}])")
          print(f"TD target: [{_tdmin_s}, {_tdmax_s}]  mean={_tdmean_s}  Tz_clip={_tzclip_s}")
          print(f"Replay buffer size: {len(deep_controller.memory)}")
          print(f"Epsilon decayed from {old_eps:.4f} to {deep_controller.epsilon:.4f}")

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

      # --------------------------------------------------------------
      # MOVING AVERAGE -- smoothed reward trend for TensorBoard
      # --------------------------------------------------------------
      # Raw episode rewards are noisy; the 10-episode moving average gives
      # a cleaner signal that is easier to read in TensorBoard.
      window = 10
      if len(episode_stats) >= window:
          last_rewards = [e["total_reward"] for e in episode_stats[-window:]]
          ma_10 = float(np.mean(last_rewards))
          deep_controller.writer.add_scalar("episode/total_reward_ma_10", ma_10, ep)

      # Moving average of PnL (same window as reward)
      pnl_ma_10 = float("nan")
      if len(episode_stats) >= window:
          last_pnls = [e["final_pnl"] for e in episode_stats[-window:]]
          pnl_ma_10 = float(np.mean(last_pnls))
          deep_controller.writer.add_scalar("episode/final_pnl_ma_10", pnl_ma_10, ep)

      # PnL moving averages used for console monitoring and TensorBoard.
      pnl_ma_50 = float("nan")
      pnl_ma_100 = float("nan")

      if len(episode_stats) >= 50:
          _last_pnls_50 = [e["final_pnl"] for e in episode_stats[-50:]]
          pnl_ma_50 = float(np.mean(_last_pnls_50))
          deep_controller.writer.add_scalar("episode/final_pnl_ma_50", pnl_ma_50, ep)

      if len(episode_stats) >= 100:
          _last_pnls_100 = [e["final_pnl"] for e in episode_stats[-100:]]
          pnl_ma_100 = float(np.mean(_last_pnls_100))
          deep_controller.writer.add_scalar("episode/final_pnl_ma_100", pnl_ma_100, ep)

      # MA for best-checkpoint tracking (now default: MA100)
      pnl_ma_best = float("nan")
      if len(episode_stats) >= BEST_CKPT_WINDOW:
          _last_pnls_best = [e["final_pnl"] for e in episode_stats[-BEST_CKPT_WINDOW:]]
          pnl_ma_best = float(np.mean(_last_pnls_best))
          deep_controller.writer.add_scalar(
              f"episode/final_pnl_ma_{BEST_CKPT_WINDOW}", pnl_ma_best, ep
          )

      # Compact logging (non-verbose): one line per episode
      if not VERBOSE_LOGGING:
          _ma_parts = []
          if not np.isnan(pnl_ma_10):
              _ma_parts.append(f"MA10={pnl_ma_10:+.4f}")
          if not np.isnan(pnl_ma_50):
              _ma_parts.append(f"MA50={pnl_ma_50:+.4f}")
          if not np.isnan(pnl_ma_100):
              _ma_parts.append(f"MA100={pnl_ma_100:+.4f}")
          _ma_str = f"  {'  '.join(_ma_parts)}" if _ma_parts else ""
          print(f"[{ep+1:4d}/{N_EPISODES}]  PnL={final_pnl:+.4f}{_ma_str}")

      # Log the outer epsilon (the one actually used, post-decay) so we can
      # verify the schedule visually alongside reward curves.
      deep_controller.writer.add_scalar("episode/epsilon_outer", deep_controller.epsilon, ep)

      # --------------------------------------------------------------
      # INVENTORY DIAGNOSTICS -- TensorBoard scalars
      # --------------------------------------------------------------
      deep_controller.writer.add_scalar("episode/max_abs_inventory", max_abs_inv, ep)
      deep_controller.writer.add_scalar("episode/mean_abs_inventory", mean_abs_inv, ep)
      deep_controller.writer.add_scalar("episode/pct_at_inv_limit", pct_at_limit, ep)
      if "MM_Fill_Imbalance_EWMA" in mm_df.columns:
          _fill_bias = mm_df["MM_Fill_Imbalance_EWMA"].values.astype(float)
          deep_controller.writer.add_scalar(
              "episode/fill_bias_abs_mean", float(np.mean(np.abs(_fill_bias))), ep
          )
          deep_controller.writer.add_scalar(
              "episode/fill_bias_abs_max", float(np.max(np.abs(_fill_bias))), ep
          )
      if ("MM_Fill_Bid_EWMA" in mm_df.columns) and ("MM_Fill_Ask_EWMA" in mm_df.columns):
          _fill_bid = mm_df["MM_Fill_Bid_EWMA"].values.astype(float)
          _fill_ask = mm_df["MM_Fill_Ask_EWMA"].values.astype(float)
          deep_controller.writer.add_scalar(
              "episode/fill_activity_mean", float(np.mean(_fill_bid + _fill_ask)), ep
          )

      # --------------------------------------------------------------
      # ADVERSARY TAU BANDIT -- post-episode update & logging
      # --------------------------------------------------------------
      # The bandit only updates its scores during the ADVERSARY phase.
      # During the MM phase, the bandit still selects tau (so the MM sees
      # varied persistence), but it does NOT update -- this prevents the
      # bandit from learning on stale policy data while the MM is changing.
      #
      # Epsilon decay happens only during the adversary phase.
      #
      # During warmup (ep < ADV_TAU_WARMUP_EPISODES), neither the bandit
      # nor the freeze schedule is active -- the MM trains alone with
      # random tau selection.
      if USE_ADV_TAU:
          _adv_tau_is_learning = (ep >= ADV_TAU_WARMUP_EPISODES
                                  and adv_tau_phase == "adversary")

          # Update the bandit only during adversary phase
          if _adv_tau_is_learning:
              _adv_tau_obj = final_pnl if ADV_TAU_OBJECTIVE == "final_pnl" else total_reward
              adv_tau.update(_adv_tau_obj)
              adv_tau.decay_epsilon()

          # Re-enable MM learning after the episode (in case it was disabled)
          deep_controller.enable_learning = True

          # -- TensorBoard scalars (always, regardless of phase) --------------------------------------------------------------
          deep_controller.writer.add_scalar("adv_tau/chosen_tau", float(_chosen_tau), ep)
          deep_controller.writer.add_scalar("adv_tau/arm_idx", float(_adv_tau_arm), ep)
          deep_controller.writer.add_scalar("adv_tau/epsilon", adv_tau.epsilon, ep)
          deep_controller.writer.add_scalar("adv_tau/phase",
              0.0 if (ep < ADV_TAU_WARMUP_EPISODES) else
              (1.0 if adv_tau_phase == "mm" else 2.0), ep)
          if _episode_mo_flow_alpha is not None:
              deep_controller.writer.add_scalar("adv_tau/mo_flow_alpha", _episode_mo_flow_alpha, ep)
          if _episode_fill_alpha is not None:
              deep_controller.writer.add_scalar("adv_tau/fill_imbalance_alpha", _episode_fill_alpha, ep)

          # Per-arm diagnostics
          for _ai, _tv in enumerate(adv_tau.tau_grid):
              deep_controller.writer.add_scalar(f"adv_tau/score_tau{_tv}", adv_tau.scores[_ai], ep)
              deep_controller.writer.add_scalar(f"adv_tau/count_tau{_tv}", float(adv_tau.counts[_ai]), ep)

          # Regime schedule stats
          _rs_p_values = getattr(ep_buy_mo_prob, "p_values", [])
          _rs_n_regimes = len(_rs_p_values)
          _rs_mean_pb = float(np.mean(_rs_p_values)) if _rs_p_values else 0.5
          deep_controller.writer.add_scalar("adv_tau/n_regimes", _rs_n_regimes, ep)
          deep_controller.writer.add_scalar("adv_tau/mean_p_buy", _rs_mean_pb, ep)

          # -- Per-episode print: alphas used, bandit state --------------------------------------------------------------
          _s = adv_tau.summary()
          _mo_alpha_used = _episode_mo_flow_alpha if _episode_mo_flow_alpha is not None else EWMA_ALPHA
          _fi_alpha_used = _episode_fill_alpha if _episode_fill_alpha is not None else FILL_IMBALANCE_ALPHA
          _phase_str = ("warmup" if ep < ADV_TAU_WARMUP_EPISODES
                        else f"phase={adv_tau_phase}")
          if VERBOSE_LOGGING:
              print(f"  [ADV_TAU] tau={_chosen_tau}, {_phase_str}, epsilon={adv_tau.epsilon:.3f}, "
                    f"best_tau={_s['best_tau']} (score={_s['best_score']:.4f})")
              print(f"  [ADV_TAU] mo_flow_alpha={_mo_alpha_used:.6f}, "
                    f"fill_imbalance_alpha={_fi_alpha_used:.6f}")

      # --------------------------------------------------------------
      # LEARNING RATE SCHEDULE -- configurable via best_params["lr_decay_type"]
      # --------------------------------------------------------------
      # We compute the LR that should be used AFTER finishing episode `ep`,
      # i.e. during episode `ep + 1`.
      #
      #   "exponential" : lr(ep) = LR_START  x  (LR_END_FACTOR ** progress)
      #       Smooth log-linear ramp.  Large updates early, fine-tuning late.
      #
      #   "linear"      : lr(ep) = (1 - progress)  x  LR_START + progress  x  LR_END
      #       Straight-line ramp.  Gentler initial drop than exponential.
      #
      # A decaying LR is critical in RL: early on, large updates push the
      # Q-network toward the correct basin; later, small updates refine
      # without oscillation.
      # --------------------------------------------------------------
      lr_decay_type = best_params.get("lr_decay_type", "exponential")
      use_exp_lr = (lr_decay_type == "exponential")

      if hasattr(deep_controller, "optimizer"):
          new_lr = get_lr_for_episode(_mm_ep_for_decay, _mm_total_for_decay, use_exp=use_exp_lr)

          # -- Cyclical LR modulation --------------------------------------------------------------
          # When USE_CYCLICAL_LR is active, the monotonically-decayed LR
          # (new_lr) is multiplied by a cosine warm-restart factor that
          # oscillates between 1.0 (cycle start) and CYCLICAL_LR_MIN_FACTOR
          # (cycle trough).
          #
          # The combined effect is a DECAYING ENVELOPE with periodic warm
          # restarts -- the restarts get progressively smaller because the
          # base LR is shrinking.  This is gentler than pure cyclical LR
          # (no monotonic decay) which can cause late-training instability.
          #
          # Example (period=200, min_factor=0.1, LR decaying 3e-4 -> 6e-5):
          #   ep=0:   lr = 3e-4  x  1.0 = 3e-4   (restart peak)
          #   ep=100: lr = 2.5e-4  x  0.55 = 1.4e-4 (mid-cycle)
          #   ep=200: lr = 2e-4  x  1.0 = 2e-4   (restart peak, but lower base)
          #   ep=300: lr = 1.5e-4  x  0.55 = 8e-5 (mid-cycle, smaller)
          _lr_cycle_mult = 1.0
          if USE_CYCLICAL_LR:
              _lr_cycle_mult = get_cyclical_lr_multiplier(
                  ep, period=CYCLICAL_LR_PERIOD, min_factor=CYCLICAL_LR_MIN_FACTOR)
              new_lr *= _lr_cycle_mult

          # When using differential LR (multiple param_groups with different
          # LRs), preserve the RATIO between groups instead of setting all
          # to the same value.  This maintains "trunk slow, head fast"
          # throughout the decay schedule.
          n_groups = len(deep_controller.optimizer.param_groups)
          if n_groups > 1 and WARMSTART_CKPT is not None:
              # Differential mode: scale each group proportionally.
              # The head group (last) gets new_lr; feature groups get
              # new_lr * LR_FEATURE_FACTOR.
              _feat_factor = LR_FEATURE_FACTOR if 'LR_FEATURE_FACTOR' in dir() else 0.2
              for i, param_group in enumerate(deep_controller.optimizer.param_groups):
                  if i < n_groups - 1:
                      param_group["lr"] = new_lr * _feat_factor
                  else:
                      param_group["lr"] = new_lr
          else:
              # Uniform mode: all groups get the same LR.
              for param_group in deep_controller.optimizer.param_groups:
                  param_group["lr"] = new_lr

          deep_controller.writer.add_scalar("train/lr", new_lr, ep)
          if USE_CYCLICAL_LR:
              deep_controller.writer.add_scalar("train/lr_cycle_mult", _lr_cycle_mult, ep)
          if VERBOSE_LOGGING:
              _clr_str = f", cycle_mult={_lr_cycle_mult:.4f}" if USE_CYCLICAL_LR else ""
              print(f"Updated learning rate to {new_lr:.8f} (mm_ep={_mm_ep_for_decay}{_clr_str})")
      else:
          if VERBOSE_LOGGING:
              print("Skipping LR update: deep_controller has no 'optimizer'.")

      # --------------------------------------------------------------
      # ANCHOR DECAY -- update L2 anchor strength per episode
      # --------------------------------------------------------------
      # When USE_ANCHOR_DECAY is active, the anchor lambda linearly decays
      # from ANCHOR_LAMBDA_START (strong preservation) to ANCHOR_LAMBDA_END
      # (relaxed, allow adaptation) over the training run.
      #
      # This is applied AFTER the LR update because the anchor and LR
      # schedules are independent -- the anchor controls HOW MUCH the weights
      # can drift from the pre-trained baseline, while the LR controls
      # HOW FAST they move in general.
      #
      # NOTE: we only update the lambda (strength), NOT the anchor weights
      # themselves.  The anchor weights are always the pre-trained Fase A
      # snapshot taken at warmstart time.
      # --------------------------------------------------------------
      if USE_ANCHOR_DECAY:
          _anchor_progress = (ep + 1) / N_EPISODES
          _current_anchor_lambda = get_anchor_lambda(
              _anchor_progress,
              lambda_start=ANCHOR_LAMBDA_START,
              lambda_end=ANCHOR_LAMBDA_END,
          )
          deep_controller.set_anchor_lambda(_current_anchor_lambda)
          deep_controller.writer.add_scalar("train/anchor_lambda", _current_anchor_lambda, ep)
          if VERBOSE_LOGGING:
              print(f"Anchor lambda updated to {_current_anchor_lambda:.6f} "
                    f"(progress={_anchor_progress:.3f})")

      # --------------------------------------------------------------
      # CURRICULUM-TIED ANCHOR -- update L2 anchor strength from ADR delta
      # --------------------------------------------------------------
      # The anchor GROWS with curriculum difficulty (not a time-based
      # decay).  Mutually exclusive with USE_ANCHOR_DECAY (validated at
      # startup).  When active, the anchor strength is a gated linear
      # function of the CURRENT curriculum difficulty (delta), not of
      # elapsed time.
      #
      # Applied once per episode, AFTER report_episode() has potentially
      # updated delta.  This ensures the NEXT episode is trained under an
      # anchor that reflects the new delta (not the delta of the episode that
      # just finished).
      # --------------------------------------------------------------
      if USE_CURRICULUM_TIED_ANCHOR and adr_curriculum is not None:
          _tied_delta = float(adr_curriculum.delta)
          _current_anchor_lambda = get_tied_anchor_lambda(
              delta=_tied_delta,
              delta_max=ADR_LITE_DELTA_MAX,
              base=TIED_ANCHOR_BASE,
              slope=TIED_ANCHOR_SLOPE,
              gate=TIED_ANCHOR_GATE,
          )
          deep_controller.set_anchor_lambda(_current_anchor_lambda)
          deep_controller.writer.add_scalar(
              "train/anchor_lambda", _current_anchor_lambda, ep
          )
          deep_controller.writer.add_scalar(
              "train/anchor_lambda_delta", _tied_delta, ep
          )
          if VERBOSE_LOGGING:
              print(f"Tied anchor lambda = {_current_anchor_lambda:.4f} "
                    f"(delta={_tied_delta:.3f})")

      # --------------------------------------------------------------
      # Periodic checkpoint saving (every 50 episodes)
      # --------------------------------------------------------------
      # Saves an intermediate checkpoint so that multi-hour training runs
      # can be recovered after crashes or interruptions.  The final
      # checkpoint below overwrites the "best" slot after training ends.
      # --------------------------------------------------------------
      CKPT_INTERVAL = 50
      _skip_ckpt = USE_ADVERSARIAL and adv_phase == "adversary"
      if (ep + 1) % CKPT_INTERVAL == 0 and not _skip_ckpt:
          _ckpt_dir = "checkpoints"
          os.makedirs(_ckpt_dir, exist_ok=True)
          _train_suffix = _build_train_suffix()
          _ckpt_name = f"deep_mm_mtm_{'pure' if USE_PURE_MM else 'generic'}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_final.pt"
          _ckpt_meta = {
              "USE_PURE_MM": bool(USE_PURE_MM),
              "best_params": dict(best_params),
              "pure_mm_offsets": pure_mm_offsets,
              "lam": lam, "mu": mu, "delta": delta,
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
              "REGIME_P_LO": REGIME_P_LO,
              "REGIME_P_HI": REGIME_P_HI,
              "EWMA_ALPHA": float(EWMA_ALPHA),
              "FAST_FLOW_EWMA_ALPHA": (None if FAST_FLOW_EWMA_ALPHA is None else float(FAST_FLOW_EWMA_ALPHA)),
              "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
              "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
              "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
              "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
              "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
              "USE_ADVERSARIAL": bool(USE_ADVERSARIAL),
              "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
              "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
              "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
              "USE_MIXTURE": bool(USE_MIXTURE),
              "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
              "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
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
              "USE_ANCHOR_DECAY": bool(USE_ANCHOR_DECAY),
              "USE_CURRICULUM_TIED_ANCHOR": bool(USE_CURRICULUM_TIED_ANCHOR),
              "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
              "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
              "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
              "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
              "USE_ADV_TAU": bool(USE_ADV_TAU),
              "ADV_TAU_MODE": ADV_TAU_MODE,
              "ADV_TAU_OBJECTIVE": ADV_TAU_OBJECTIVE,
              "ADV_FREEZE_MM": ADV_FREEZE_MM,
              "ADV_FREEZE_ADV": ADV_FREEZE_ADV,
              "ADV_WARMUP_EPISODES": ADV_WARMUP_EPISODES,
          }
          save_deep_rl_checkpoint(
              deep_controller, os.path.join(_ckpt_dir, _ckpt_name), _ckpt_meta
          )
          if USE_ADVERSARIAL:
              adversary.save(os.path.join(_ckpt_dir, f"adversary_ep{ep+1:04d}.pt"))
          if USE_ADV_TAU:
              adv_tau.save(os.path.join(_ckpt_dir, f"adv_tau_ep{ep+1:04d}.pt"))
          print(f"[CHECKPOINT] Saved checkpoint (ep {ep+1}): {_ckpt_name}")

      # --------------------------------------------------------------
      # BEST-CHECKPOINT update (single rolling "best so far" slot)
      # --------------------------------------------------------------
      # Save whenever MA(BEST_CKPT_WINDOW) of final_pnl improves.
      _skip_best_ckpt = USE_ADVERSARIAL and adv_phase == "adversary"
      # When using progressive curriculum, only save best checkpoint after
      # the ramp phase ends.  This prevents selecting a model that only
      # excelled on the easy initial curriculum (which happened in earlier
      # runs where best MA50 was at ep ~200 on the easiest difficulty).
      if USE_PROGRESSIVE_CURRICULUM:
          _hold_start_ep = int(N_EPISODES * CURRICULUM_RAMP_FRAC)
          _skip_best_ckpt = _skip_best_ckpt or (ep < _hold_start_ep)
      if not np.isnan(pnl_ma_best) and not _skip_best_ckpt and pnl_ma_best > _best_ckpt_ma:
          _best_ckpt_ma = pnl_ma_best
          _best_ckpt_ep = ep
          _ckpt_dir = "checkpoints"
          os.makedirs(_ckpt_dir, exist_ok=True)
          _train_suffix = _build_train_suffix()
          _best_ckpt_name = (
              f"deep_mm_mtm_{'pure' if USE_PURE_MM else 'generic'}_"
              f"invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_best_ma{BEST_CKPT_WINDOW}.pt"
          )
          _best_ckpt_path = os.path.join(_ckpt_dir, _best_ckpt_name)
          _best_ckpt_meta = {
              "USE_PURE_MM": bool(USE_PURE_MM),
              "best_params": dict(best_params),
              "pure_mm_offsets": pure_mm_offsets,
              "lam": lam, "mu": mu, "delta": delta,
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
              "REGIME_P_LO": REGIME_P_LO,
              "REGIME_P_HI": REGIME_P_HI,
              "EWMA_ALPHA": float(EWMA_ALPHA),
              "FAST_FLOW_EWMA_ALPHA": (None if FAST_FLOW_EWMA_ALPHA is None else float(FAST_FLOW_EWMA_ALPHA)),
              "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
              "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
              "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
              "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
              "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
              "USE_ADVERSARIAL": bool(USE_ADVERSARIAL),
              "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
              "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
              "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
              "USE_MIXTURE": bool(USE_MIXTURE),
              "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
              "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
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
              "USE_ANCHOR_DECAY": bool(USE_ANCHOR_DECAY),
              "USE_CURRICULUM_TIED_ANCHOR": bool(USE_CURRICULUM_TIED_ANCHOR),
              "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
              "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
              "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
              "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
              "USE_ADV_TAU": bool(USE_ADV_TAU),
              "ADV_TAU_MODE": ADV_TAU_MODE,
              "ADV_TAU_OBJECTIVE": ADV_TAU_OBJECTIVE,
              "ADV_FREEZE_MM": ADV_FREEZE_MM,
              "ADV_FREEZE_ADV": ADV_FREEZE_ADV,
              "ADV_WARMUP_EPISODES": ADV_WARMUP_EPISODES,
          }
          save_deep_rl_checkpoint(deep_controller, _best_ckpt_path, _best_ckpt_meta)
          print(
              f"[CHECKPOINT] Updated BEST CHECKPOINT (criterion=MA{BEST_CKPT_WINDOW}) "
              f"(ep {ep+1}, MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f}): {_best_ckpt_name}"
          )

      # -- EWMA diagnostic plot (every PLOT_EWMA_INTERVAL episodes) --------------------------------------------------------------
      if PLOT_EWMA and USE_FLOW_EWMA:
          if (ep + 1) % PLOT_EWMA_INTERVAL == 0 and "MM_MO_Flow_EWMA" in mm_df.columns:
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
              # set_title removed for paper export (caption describes the figure)
              if _has_fast_flow:
                  ax_ewma.legend(loc="upper right", frameon=False)
              plt.tight_layout()
              _save_and_maybe_show(fig_ewma, "mo_flow_ewma_diagnostic", prefix="diagnostic")
              plt.close(fig_ewma)

      # -- Fill imbalance diagnostic plot --------------------------------------------------------------
      if PLOT_FILL_IMBALANCE and USE_FILL_IMBALANCE:
          _fill_interval = max(int(PLOT_FILL_IMBALANCE_INTERVAL), 1)
          _fill_plot_col = None
          _fill_plot_label = None
          _fill_plot_color = None
          if USE_QUOTE_EXPOSURE_IMBALANCE and "MM_Fill_Imbalance_Feature" in mm_df.columns:
              _fill_plot_col = "MM_Fill_Imbalance_Feature"
              _fill_plot_label = "quote exposure imbalance (DQN)"
              _fill_plot_color = "#0072B2"
          elif "MM_Fill_Imbalance_EWMA" in mm_df.columns:
              _fill_plot_col = "MM_Fill_Imbalance_EWMA"
              _fill_plot_label = "fill imbalance EWMA"
              _fill_plot_color = "#999999"

          if (ep + 1) % _fill_interval == 0 and _fill_plot_col is not None:
              fig_fi, ax_fi = plt.subplots(figsize=(10, 3))
              ax_fi.plot(
                  mm_df[_fill_plot_col].values,
                  linewidth=1.0,
                  color=_fill_plot_color,
                  alpha=0.95,
                  label=_fill_plot_label,
              )
              ax_fi.axhline(0, color="gray", linewidth=0.5, linestyle="--")
              ax_fi.set_ylim(-1.05, 1.05)
              ax_fi.set_xlabel("Simulation Step", fontsize=15)
              ax_fi.set_ylabel("Fill Imbalance", fontsize=15)
              # ax_fi.set_title removed for paper export
              ax_fi.legend(loc="upper right", frameon=False, fontsize=13)
              plt.tight_layout()
              _save_and_maybe_show(fig_fi, "fill_imbalance_diagnostic", prefix="diagnostic")
              plt.close(fig_fi)

      # -- Bayesian flow diagnostic plot --------------------------------------------------------------
      if PLOT_BAYES_METRICS and USE_BAYES_FLOW_SIGNAL:
          _bayes_cols = (
              "MM_Bayes_M_Hat",
              "MM_Bayes_Run_Length",
          )
          if (ep + 1) % PLOT_BAYES_METRICS_INTERVAL == 0 and all(c in mm_df.columns for c in _bayes_cols):
              fig_bayes, ax_bayes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
              _x = np.arange(len(mm_df))
              _true_m_hat = None
              _true_regime_age = None
              _true_regime_len = None
              if callable(ep_buy_mo_prob) and hasattr(ep_buy_mo_prob, "boundaries") and "Type" in msg_df.columns:
                  _boundaries = np.asarray(getattr(ep_buy_mo_prob, "boundaries", []), dtype=int)
                  _p_values = np.asarray(getattr(ep_buy_mo_prob, "p_values", []), dtype=float)
                  _types = msg_df["Type"].values
                  if len(_boundaries) >= 2 and len(_p_values) >= 1 and len(_types) == len(mm_df):
                      _true_m_hat = np.full(len(mm_df), np.nan, dtype=float)
                      _true_regime_age = np.zeros(len(mm_df), dtype=float)
                      _true_regime_len = np.full(len(mm_df), np.nan, dtype=float)
                      _n_mo_seen = 0
                      for _i, _typ in enumerate(_types):
                          if int(_typ) == 1:
                              _n_mo_seen += 1
                          _mo_idx = max(_n_mo_seen - 1, 0)
                          while _mo_idx >= _boundaries[-1] and hasattr(ep_buy_mo_prob, "__call__"):
                              ep_buy_mo_prob(_mo_idx)
                              _boundaries = np.asarray(getattr(ep_buy_mo_prob, "boundaries", []), dtype=int)
                              _p_values = np.asarray(getattr(ep_buy_mo_prob, "p_values", []), dtype=float)
                          _reg_idx = int(np.searchsorted(_boundaries, _mo_idx, side="right")) - 1
                          _reg_idx = min(max(_reg_idx, 0), len(_boundaries) - 2)
                          _p_idx = min(_reg_idx, len(_p_values) - 1)
                          _true_m_hat[_i] = 2.0 * float(_p_values[_p_idx]) - 1.0
                          if _n_mo_seen <= 0:
                              continue
                          _reg_start = int(_boundaries[_reg_idx])
                          _reg_end = int(_boundaries[_reg_idx + 1])
                          _true_regime_age[_i] = float(_mo_idx - _reg_start + 1)
                          _true_regime_len[_i] = float(_reg_end - _reg_start)

              # Panel 1: latent directional bias --- true vs Bayesian estimate
              if _true_m_hat is not None:
                  ax_bayes[0].plot(
                      _x,
                      _true_m_hat,
                      linewidth=1.0,
                      color="#444444",
                      alpha=0.9,
                      label="true $m$",
                  )
              ax_bayes[0].plot(
                  _x,
                  mm_df["MM_Bayes_M_Hat"].values,
                  linewidth=1.0,
                  color="#0072B2",
                  alpha=0.95,
                  label=r"Bayes $\widehat{m}$",
              )
              ax_bayes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
              ax_bayes[0].set_ylim(-1.05, 1.05)
              ax_bayes[0].set_ylabel("directional bias", fontsize=15)
              ax_bayes[0].legend(loc="upper right", frameon=False, fontsize=13)

              # Panel 2: run length --- Bayesian expected vs true
              ax_bayes[1].plot(
                  _x,
                  mm_df["MM_Bayes_Run_Length"].values,
                  linewidth=1.0,
                  color="#0072B2",
                  label=r"Bayes $E[\mathrm{run}]$",
              )
              if _true_regime_age is not None:
                  ax_bayes[1].plot(
                      _x,
                      _true_regime_age,
                      linewidth=1.0,
                      color="#444444",
                      alpha=0.85,
                      linestyle="--",
                      label="true run length",
                  )
              ax_bayes[1].set_ylabel("run length (MO)", fontsize=15)
              ax_bayes[1].set_xlabel("Simulation Step", fontsize=15)

              # fig_bayes.suptitle removed for paper export
              plt.tight_layout()
              _save_and_maybe_show(fig_bayes, "bayes_flow_diagnostic", prefix="diagnostic")
              plt.close(fig_bayes)

      # -- ADR-lite curriculum-based early stop --------------------------------------------------------------
      # When the ADR-lite v3 thermostat is active, the natural notion of
      # "training is done" is curriculum-state-based, NOT episode-count-
      # based.  We stop on:
      #
      #   (A) CONVERGENCE: curriculum reached delta_max and stayed there for
      #       ADR_CONVERGENCE_DECISIONS_AT_MAX consecutive decisions
      #       -> "agent dominated the entire curriculum"
      #
      #   (B) PLATEAU: curriculum has not advanced for
      #       ADR_PLATEAU_MAX_DECISIONS_NO_ADV consecutive decisions
      #       -> "agent hit its capacity ceiling"
      #
      # N_EPISODES still acts as a hard safety cap (see the for-loop
      # range), but in normal operation the run terminates here.
      if (USE_ADR_LITE_CURRICULUM and adr_curriculum is not None
              and USE_ADR_EARLY_STOP):
          _adr_state_eos = adr_curriculum.get_state()
          _csm = int(_adr_state_eos["consecutive_at_max"])
          _dsla = int(_adr_state_eos["decisions_since_last_advance"])

          # (A) Convergence at delta_max
          if _csm >= ADR_CONVERGENCE_DECISIONS_AT_MAX:
              print(f"\n{'='*60}")
              print(f"ADR EARLY STOP: CONVERGED AT delta_max")
              print(f"  episode = {ep+1}")
              print(f"  delta_max   = {adr_curriculum.delta_max}")
              print(f"  consecutive decisions at delta_max = {_csm}")
              print(f"  total decisions = {int(_adr_state_eos['decisions'])}")
              print(f"  total advances  = {int(_adr_state_eos['advances'])}")
              print(f"  total holds     = {int(_adr_state_eos['holds'])}")
              print(f"  total retreats  = {int(_adr_state_eos['retreats'])}")
              print(f"{'='*60}\n")
              _es_triggered = True
              break  # exit training loop

          # (B) Plateau detection
          if (_adr_state_eos["decisions"] > 0
                  and _dsla >= ADR_PLATEAU_MAX_DECISIONS_NO_ADV):
              print(f"\n{'='*60}")
              print(f"ADR EARLY STOP: PLATEAUED")
              print(f"  episode = {ep+1}")
              print(f"  current delta = {adr_curriculum.delta:.4f} "
                    f"(out of [{adr_curriculum.delta_min}, {adr_curriculum.delta_max}])")
              print(f"  decisions without advance = {_dsla}")
              print(f"  total decisions = {int(_adr_state_eos['decisions'])}")
              print(f"  total advances  = {int(_adr_state_eos['advances'])}")
              print(f"  total holds     = {int(_adr_state_eos['holds'])}")
              print(f"  total retreats  = {int(_adr_state_eos['retreats'])}")
              print(f"{'='*60}\n")
              _es_triggered = True
              break

      # -- Generic MA50-based early stopping check --------------------------------------------------------------
      # Placed AFTER all logging, TB scalars, checkpoints, LR updates,
      # and ADV_TAU updates so the last episode is fully complete.
      if EARLY_STOPPING and ep >= EARLY_STOP_MIN_EPISODES:
          if len(episode_stats) >= EARLY_STOP_WINDOW:
              _es_pnls = [e["final_pnl"] for e in episode_stats[-EARLY_STOP_WINDOW:]]
              _es_current_ma = float(np.mean(_es_pnls))

              if _es_current_ma > _es_best_ma:
                  _es_best_ma = _es_current_ma
                  _es_best_ep = ep

              # For positive peaks: fractional drop.
              # For negative peaks: absolute drop (avoids division issues).
              if _es_best_ma > 0:
                  _es_dropped = _es_current_ma < _es_best_ma * (1.0 - EARLY_STOP_DROP)
              else:
                  _es_dropped = _es_current_ma < _es_best_ma - abs(_es_best_ma) * EARLY_STOP_DROP

              if _es_dropped:
                  print(f"\n{'!'*60}")
                  print(f"EARLY STOPPING at episode {ep+1}")
                  print(f"  MA{EARLY_STOP_WINDOW} PnL peaked at {_es_best_ma:+.4f} (ep {_es_best_ep+1})")
                  print(f"  Current MA{EARLY_STOP_WINDOW} PnL: {_es_current_ma:+.4f}")
                  print(f"  Drop threshold: {EARLY_STOP_DROP:.0%}")
                  print(f"{'!'*60}\n")
                  _es_triggered = True
                  break  # exit training loop

  # Training loop ended -- either completed all episodes or early-stopped.
  if _es_triggered:
      print(f"Training stopped early at episode {ep+1}/{N_EPISODES} "
            f"(peak MA{EARLY_STOP_WINDOW} PnL was {_es_best_ma:+.4f} at ep {_es_best_ep+1})")
  else:
      print(f"Training completed: {N_EPISODES} episodes")
  if _best_ckpt_ep >= 0 and _best_ckpt_path is not None:
      print(
          f"[CHECKPOINT] Best checkpoint summary (criterion=MA{BEST_CKPT_WINDOW}): "
          f"best MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f} "
          f"at ep {_best_ckpt_ep+1}: {_best_ckpt_path}"
      )

  # Final checkpoint (same file -- last periodic save is already up-to-date,
  # but we save once more to guarantee the final state is captured even if
  # N_EPISODES is not a multiple of CKPT_INTERVAL or training was early-stopped).
  CKPT_DIR = "checkpoints"
  _train_suffix = _build_train_suffix()
  ckpt_name = f"deep_mm_mtm_{'pure' if USE_PURE_MM else 'generic'}_invp{INV_PENALTY_COEFF:.4f}{_train_suffix}_final.pt"
  CKPT_PATH = os.path.join(CKPT_DIR, ckpt_name)

  meta = {
      "USE_PURE_MM": bool(USE_PURE_MM),
      "best_params": dict(best_params),
      "pure_mm_offsets": (pure_mm_offsets if USE_PURE_MM else None),
      "lam": lam, "mu": mu, "delta": delta,
      "GLOBAL_SEED": GLOBAL_SEED,
      "EPISODE_LENGTH": EPISODE_LENGTH,
      "ITER_TO_EQUILIBRIUM": ITER_TO_EQUILIBRIUM,
      "N_EPISODES": N_EPISODES,
      "episodes_completed": ep + 1,  # actual episodes run (may be < N_EPISODES if early-stopped)
      "early_stopped": _es_triggered,
      "per_alpha_last_episode": ep + 1,
      "per_beta_last_episode": ep + 1,
      "USE_REGIME_SWITCH": bool(USE_REGIME_SWITCH),
      "REGIME_DISTRIBUTION": REGIME_DISTRIBUTION,
      "REGIME_ALPHA": REGIME_ALPHA,
      "REGIME_L_MIN": REGIME_L_MIN,
      "REGIME_EXP_TAU": REGIME_EXP_TAU,
      "REGIME_P_LO": REGIME_P_LO,
      "REGIME_P_HI": REGIME_P_HI,
      "EWMA_ALPHA": float(EWMA_ALPHA),
      "FAST_FLOW_EWMA_ALPHA": (None if FAST_FLOW_EWMA_ALPHA is None else float(FAST_FLOW_EWMA_ALPHA)),
      "FILL_IMBALANCE_ALPHA": float(FILL_IMBALANCE_ALPHA),
      "MO_EVENT_FRACTION": float(_MO_EVENT_FRACTION),
      "MO_EVENT_FRACTION_ANALYTIC": float(_MO_EVENT_FRACTION_ANALYTIC),
      "MO_EVENT_FRACTION_SOURCE": _MO_EVENT_FRACTION_SOURCE,
      "N_MO_FRAC_CALIB_EPISODES": int(N_MO_FRAC_CALIB_EPISODES),
      "USE_ADVERSARIAL": bool(USE_ADVERSARIAL),
      "USE_FLOW_EWMA": bool(USE_FLOW_EWMA),
      "USE_FAST_FLOW_EWMA": bool(USE_FAST_FLOW_EWMA),
      "USE_BAYES_FLOW_SIGNAL": bool(USE_BAYES_FLOW_SIGNAL),
      "USE_MIXTURE": bool(USE_MIXTURE),
      "BAYES_MIXTURE_TAUS": list(BAYES_MIXTURE_TAUS),
      "BAYES_FLOW_TAU_R": float(BAYES_FLOW_TAU_R),
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
      "USE_ANCHOR_DECAY": bool(USE_ANCHOR_DECAY),
      "USE_CURRICULUM_TIED_ANCHOR": bool(USE_CURRICULUM_TIED_ANCHOR),
      "TIED_ANCHOR_BASE": float(TIED_ANCHOR_BASE),
      "TIED_ANCHOR_SLOPE": float(TIED_ANCHOR_SLOPE),
      "TIED_ANCHOR_GATE": float(TIED_ANCHOR_GATE),
      "USE_CYCLICAL_LR": bool(USE_CYCLICAL_LR),
      "USE_ADV_TAU": bool(USE_ADV_TAU),
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
      print(f"[CHECKPOINT] Saved replay buffer ({_n_saved} transitions) to {_replay_save_path}")
  else:
      print("[CHECKPOINT] Skipped final replay buffer save (SAVE_FINAL_REPLAY_BUFFER=False)")
  if USE_ADVERSARIAL:
      adversary.save(os.path.join(CKPT_DIR, "adversary_final.pt"))
      print(f"[ADVERSARY] Final adversary checkpoint saved.")
  if USE_ADV_TAU:
      adv_tau.save(os.path.join(CKPT_DIR, "adv_tau_final.pt"))
      print(f"[ADV_TAU] Final bandit checkpoint saved.")

  # ==============================================================
  # POST-TRAINING: ANIMATION + DATA PERSISTENCE (last episode only)
  # ==============================================================
  # After the training loop completes, we generate a visual animation of
  # the LOB + MM overlay for the LAST episode.  This lets us visually
  # inspect how the trained policy manages quotes, fills, and inventory.
  #
  # We also persist the DataFrames (msg_df, ob_df, mm_df) as pickle files
  # so they can be loaded offline for further analysis without re-running
  # the full training loop.
  # ==============================================================
  print(f"\nGenerating animation for LAST episode {ep + 1}...")
  anim = animate_lob(
      msg_df,
      ob_df,
      mm_df=mm_df,
      max_levels=6,
      interval_ms=200,
      step=1,
      price_step=1,
      bar_height=0.8,
  )
  _save_animation_and_maybe_show(anim, filename=f"last_episode_{ep + 1}_animation.gif", fps=5)

  # --------------------------------------------------------------
  # 12) Interpretability Studies - 2D Value Slices
  # --------------------------------------------------------------

  import copy
  import numpy as np
  import matplotlib.pyplot as plt
  import torch
  from typing import Dict, Any, Optional


  def _finite_int_or_none(x) -> Optional[int]:
      try:
          if x is None:
              return None
          x_float = float(x)
          if not np.isfinite(x_float):
              return None
          return int(x_float)
      except (TypeError, ValueError):
          return None


  def _last_valid_rl_state_row(mm_df, mode="pure_mm"):
      required = ["MM_RL_State_Mode", "MM_RL_State_Vector"]
      if not all(c in mm_df.columns for c in required):
          return None

      for idx in reversed(mm_df.index):
          row = mm_df.loc[idx]
          vec = row.get("MM_RL_State_Vector", None)
          row_mode = row.get("MM_RL_State_Mode", None)

          if row_mode == mode and isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
              return row

      return None


  def _decode_pure_mm_network_vector(vec, k_offset: int, deep_controller) -> Dict[str, Any]:
      vec = list(vec)
      k_offset = int(k_offset)
      depth_len = k_offset + 1
      base_len = 2 + 2 * depth_len

      if len(vec) < base_len:
          raise ValueError(
              f"PURE-MM state vector too short: len={len(vec)}, expected at least {base_len}"
          )

      def _inv_log1p(x: float) -> float:
          return float(np.expm1(max(0.0, float(x))))

      inv_denom = float(deep_controller.inv_limit) if deep_controller.inv_limit is not None else 10.0
      inv_denom = max(inv_denom, 1.0)

      state = {
          "spread": _inv_log1p(vec[0]),
          "inventory": float(vec[1]) * inv_denom,
          "pure_mm_bid_sizes": [_inv_log1p(x) for x in vec[2 : 2 + depth_len]],
          "pure_mm_ask_sizes": [_inv_log1p(x) for x in vec[2 + depth_len : 2 + 2 * depth_len]],
      }

      cursor = base_len

      if getattr(deep_controller, "use_flow_signal", False) and cursor < len(vec):
          state["mo_flow_p_hat"] = float(vec[cursor])
          cursor += 1

      if getattr(deep_controller, "use_fast_flow_signal", False) and cursor < len(vec):
          state["mo_flow_fast_p_hat"] = float(vec[cursor])
          cursor += 1

      if getattr(deep_controller, "use_bayes_flow_signal", False):
          bayes_keys = tuple(getattr(deep_controller, "bayes_flow_feature_keys", (
              "bayes_m_hat",
              "bayes_cp_prob",
              "bayes_expected_run_length",
              "bayes_uncertainty",
          )))
          for key in bayes_keys:
              if cursor < len(vec):
                  state[key] = float(vec[cursor])
                  cursor += 1

      if getattr(deep_controller, "use_fill_imbalance", False) and cursor < len(vec):
          state["fill_imbalance_ewma"] = float(vec[cursor])

      return state


  def _attach_flow_features_from_row(base_state: Dict[str, Any], row) -> Dict[str, Any]:
      if getattr(deep_controller, "use_flow_signal", False) and "mo_flow_p_hat" not in base_state:
          if "MM_MO_Flow_PHat" in row:
              base_state["mo_flow_p_hat"] = float(row["MM_MO_Flow_PHat"])

      if getattr(deep_controller, "use_fast_flow_signal", False) and "mo_flow_fast_p_hat" not in base_state:
          if "MM_MO_Flow_Fast_PHat" in row:
              base_state["mo_flow_fast_p_hat"] = float(row["MM_MO_Flow_Fast_PHat"])

      if getattr(deep_controller, "use_bayes_flow_signal", False):
          bayes_col_map = {
              "bayes_m_hat": "MM_Bayes_M_Hat",
              "bayes_cp_prob": "MM_Bayes_CP_Prob",
              "bayes_expected_run_length": "MM_Bayes_Run_Length_Feature",
              "bayes_uncertainty": "MM_Bayes_Uncertainty",
          }
          for key, col in bayes_col_map.items():
              if key not in base_state and col in row:
                  base_state[key] = float(row[col])

      if getattr(deep_controller, "use_fill_imbalance", False) and "fill_imbalance_ewma" not in base_state:
          if USE_QUOTE_EXPOSURE_IMBALANCE and "MM_Fill_Imbalance_Feature" in row:
              base_state["fill_imbalance_ewma"] = float(row["MM_Fill_Imbalance_Feature"])
          elif "MM_Fill_Imbalance_EWMA" in row:
              base_state["fill_imbalance_ewma"] = float(row["MM_Fill_Imbalance_EWMA"])

      return base_state


  def plot_value_slices_2d(
      deep_controller,
      base_state: Dict[str, Any],
      inv_values,
      spread_values,
      bid0_values=None,
      ask0_values=None,
      bid_size_values=None,
      ask_size_values=None,
      bayes_m_values=None,
      bayes_run_values=None,
      quote_exposure_values=None,
      as_cost: bool = False,
  ):
      q_net = deep_controller.q_net
      q_net.eval()

      def evaluate_state(mm_state: Dict[str, Any]) -> float:
          with torch.no_grad():
              s_tensor = deep_controller._state_to_tensor(mm_state)
              q_values = deep_controller._q_values_from_net(q_net, s_tensor)
              v = torch.max(q_values, dim=1).values.item()
          return -v if as_cost else v

      def set_feature(mm_state: Dict[str, Any], feature_name: str, value: float):
          if feature_name == "inventory":
              mm_state["inventory"] = float(value)
          elif feature_name == "spread":
              mm_state["spread"] = float(value)
          elif feature_name == "bid0":
              bid_sizes = list(mm_state["pure_mm_bid_sizes"])
              bid_sizes[0] = float(value)
              mm_state["pure_mm_bid_sizes"] = bid_sizes
          elif feature_name == "ask0":
              ask_sizes = list(mm_state["pure_mm_ask_sizes"])
              ask_sizes[0] = float(value)
              mm_state["pure_mm_ask_sizes"] = ask_sizes
          elif feature_name == "bidsize":
              mm_state["bidsize"] = float(value)
          elif feature_name == "asksize":
              mm_state["asksize"] = float(value)
          elif feature_name == "bayes_m_hat":
              mm_state["bayes_m_hat"] = float(value)
          elif feature_name == "bayes_expected_run_length":
              mm_state["bayes_expected_run_length"] = float(value)
          elif feature_name == "quote_exposure":
              mm_state["fill_imbalance_ewma"] = float(value)
          else:
              raise ValueError(f"Unknown feature: {feature_name}")

      def compute_slice(x_values, y_values, x_feature: str, y_feature: str):
          Z = np.zeros((len(y_values), len(x_values)), dtype=np.float32)

          for iy, y in enumerate(y_values):
              for ix, x in enumerate(x_values):
                  mm_state = copy.deepcopy(base_state)
                  set_feature(mm_state, x_feature, x)
                  set_feature(mm_state, y_feature, y)
                  Z[iy, ix] = evaluate_state(mm_state)

          return Z

      def plot_heatmap(x_values, y_values, Z, xlabel: str, ylabel: str, title: str):
          fig, ax = plt.subplots(figsize=(8.5, 5.8))
          im = ax.imshow(
              Z,
              origin="lower",
              aspect="auto",
              extent=[
                  float(np.min(x_values)),
                  float(np.max(x_values)),
                  float(np.min(y_values)),
                  float(np.max(y_values)),
              ],
              cmap="viridis",
          )
          cbar_label = "Cost-to-go J(s)" if as_cost else "Value V(s)"
          fig.colorbar(im, ax=ax, label=cbar_label)

          ax.set_xlabel(xlabel, fontsize=16)
          ax.set_ylabel(ylabel, fontsize=16)
          ax.tick_params(labelsize=13)
          # ax.set_title(title)  # removed for paper export
          ax.grid(False)
          plt.tight_layout()
          _save_and_maybe_show(fig, title, prefix="heatmap")
          plt.close(fig)

      def maybe_plot(x_values, y_values, x_feature, y_feature, xlabel, ylabel, title):
          if x_values is None or y_values is None:
              return
          if len(x_values) == 0 or len(y_values) == 0:
              return

          Z = compute_slice(x_values, y_values, x_feature, y_feature)
          plot_heatmap(x_values, y_values, Z, xlabel, ylabel, title)

      is_pure_mm_state = (
          deep_controller.pure_mm
          and "pure_mm_bid_sizes" in base_state
          and "pure_mm_ask_sizes" in base_state
      )

      value_label = "cost" if as_cost else "value"

      if is_pure_mm_state:
          maybe_plot(
              spread_values,
              inv_values,
              "spread",
              "inventory",
              "Spread",
              "Inventory",
              f"MM {value_label}: inventory x spread",
          )
          maybe_plot(
              bid0_values,
              inv_values,
              "bid0",
              "inventory",
              "Best-bid queue size",
              "Inventory",
              f"MM {value_label}: inventory x best-bid queue",
          )
          maybe_plot(
              ask0_values,
              inv_values,
              "ask0",
              "inventory",
              "Best-ask queue size",
              "Inventory",
              f"MM {value_label}: inventory x best-ask queue",
          )
          maybe_plot(
              ask0_values,
              bid0_values,
              "ask0",
              "bid0",
              "Best-ask queue size",
              "Best-bid queue size",
              f"MM {value_label}: best-bid x best-ask queue",
          )
      else:
          maybe_plot(
              spread_values,
              inv_values,
              "spread",
              "inventory",
              "Spread",
              "Inventory",
              f"MM {value_label}: inventory x spread",
          )
          maybe_plot(
              bid_size_values,
              inv_values,
              "bidsize",
              "inventory",
              "Bid size",
              "Inventory",
              f"MM {value_label}: inventory x bid size",
          )
          maybe_plot(
              ask_size_values,
              inv_values,
              "asksize",
              "inventory",
              "Ask size",
              "Inventory",
              f"MM {value_label}: inventory x ask size",
          )
          maybe_plot(
              ask_size_values,
              bid_size_values,
              "asksize",
              "bidsize",
              "Ask size",
              "Bid size",
              f"MM {value_label}: bid size x ask size",
          )

      has_bayes_m = (
          getattr(deep_controller, "use_bayes_flow_signal", False)
          and "bayes_m_hat" in base_state
      )
      has_bayes_run = (
          getattr(deep_controller, "use_bayes_flow_signal", False)
          and "bayes_expected_run_length" in base_state
      )
      has_quote_exposure = (
          getattr(deep_controller, "use_fill_imbalance", False)
          and "fill_imbalance_ewma" in base_state
      )

      if has_bayes_m:
          maybe_plot(
              bayes_m_values,
              inv_values,
              "bayes_m_hat",
              "inventory",
              "Bayes directional bias m_hat",
              "Inventory",
              f"Regime {value_label}: inventory x Bayes m_hat",
          )

      if has_bayes_run:
          maybe_plot(
              bayes_run_values,
              inv_values,
              "bayes_expected_run_length",
              "inventory",
              "Bayes expected run-length feature",
              "Inventory",
              f"Regime {value_label}: inventory x Bayes run length",
          )

      if has_quote_exposure:
          maybe_plot(
              quote_exposure_values,
              inv_values,
              "quote_exposure",
              "inventory",
              "Quote-exposure imbalance",
              "Inventory",
              f"Exposure {value_label}: inventory x quote exposure",
          )

      if has_bayes_m and has_quote_exposure:
          maybe_plot(
              quote_exposure_values,
              bayes_m_values,
              "quote_exposure",
              "bayes_m_hat",
              "Quote-exposure imbalance",
              "Bayes directional bias m_hat",
              f"Regime/exposure {value_label}: Bayes m_hat x quote exposure",
          )

      if has_bayes_m:
          maybe_plot(
              spread_values,
              bayes_m_values,
              "spread",
              "bayes_m_hat",
              "Spread",
              "Bayes directional bias m_hat",
              f"Regime/spread {value_label}: Bayes m_hat x spread",
          )


  try:
      if deep_controller.pure_mm:
          anchor_row = _last_valid_rl_state_row(mm_df, mode="pure_mm")

          if anchor_row is None:
              print(
                  "\n[Interpretability WARNING] No valid PURE-MM RL decision state found "
                  "in mm_df. Skipping interpretability plots."
              )
          else:
              rl_dim = anchor_row.get("MM_RL_State_Dim", None)
              rl_max_off = anchor_row.get("MM_RL_State_MaxOffset", None)
              rl_vec = anchor_row.get("MM_RL_State_Vector", None)

              k_from_meta = _finite_int_or_none(rl_max_off)
              if k_from_meta is not None:
                  K = k_from_meta
              else:
                  dim_from_meta = _finite_int_or_none(rl_dim)
                  L = dim_from_meta if dim_from_meta is not None else len(rl_vec)
                  K = int(getattr(deep_controller, "max_offset", max(0, (L - 2) // 2 - 1)))

              base_state = _decode_pure_mm_network_vector(rl_vec, K, deep_controller)
              base_state = _attach_flow_features_from_row(base_state, anchor_row)

              print("\n[Interpretability] PURE-MM base_state used for 2D plots:")
              print("  spread:", base_state["spread"])
              print("  inventory:", base_state["inventory"])
              print("  bid_sizes:", base_state["pure_mm_bid_sizes"])
              print("  ask_sizes:", base_state["pure_mm_ask_sizes"])
              for key in ["bayes_m_hat", "bayes_expected_run_length", "fill_imbalance_ewma"]:
                  if key in base_state:
                      print(f"  {key}:", base_state[key])

              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  inv_min = int(np.floor(mm_df["MM_Inventory"].min())) if "MM_Inventory" in mm_df.columns else -20
                  inv_max = int(np.ceil(mm_df["MM_Inventory"].max())) if "MM_Inventory" in mm_df.columns else 20
                  if inv_min == inv_max:
                      inv_min -= 1
                      inv_max += 1
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              q0_all = []
              if "MM_RL_State_Vector" in mm_df.columns and "MM_RL_State_Mode" in mm_df.columns:
                  for vec, mode in zip(mm_df["MM_RL_State_Vector"], mm_df["MM_RL_State_Mode"]):
                      if mode != "pure_mm" or vec is None or not isinstance(vec, (list, np.ndarray)):
                          continue
                      try:
                          decoded_row = _decode_pure_mm_network_vector(vec, K, deep_controller)
                      except (TypeError, ValueError):
                          continue
                      q0_all.append(float(decoded_row["pure_mm_bid_sizes"][0]))
                      q0_all.append(float(decoded_row["pure_mm_ask_sizes"][0]))

              if len(q0_all) > 0:
                  q0_all = np.asarray(q0_all, dtype=float)
                  q_min = max(0.0, float(np.percentile(q0_all, 5)))
                  q_max = float(np.percentile(q0_all, 95))
                  if q_max <= q_min:
                      q_max = q_min + 1.0
                  offset_values = np.linspace(q_min, q_max, num=35)
              else:
                  base0 = max(
                      float(base_state["pure_mm_bid_sizes"][0]),
                      float(base_state["pure_mm_ask_sizes"][0]),
                      1.0,
                  )
                  offset_values = np.linspace(0.0, max(5.0, 2.0 * base0), num=35)

              base_spread = float(base_state.get("spread", 1.0))
              spread_min = max(1.0, base_spread - 3)
              spread_max = base_spread + 3
              spread_values = np.arange(spread_min, spread_max + 1, 1)

              bayes_m_values = np.linspace(-1.0, 1.0, 41)
              bayes_run_values = np.linspace(0.0, 1.0, 41)
              quote_exposure_values = np.linspace(-1.0, 1.0, 41)

              print("\n[Interpretability] 2D grids:")
              print("  inv:", inv_values[0], "->", inv_values[-1])
              print("  spread:", spread_values[0], "->", spread_values[-1])
              print("  queue:", offset_values[0], "->", offset_values[-1])

              plot_value_slices_2d(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  spread_values=spread_values,
                  bid0_values=offset_values,
                  ask0_values=offset_values,
                  bayes_m_values=bayes_m_values,
                  bayes_run_values=bayes_run_values,
                  quote_exposure_values=quote_exposure_values,
                  as_cost=False,
              )

      else:
          anchor_row = mm_df.iloc[-1]

          required_cols = [
              "MM_State_Spread",
              "MM_State_AskSize",
              "MM_State_BidSize",
              "MM_State_Inventory",
              "MM_State_HasBid",
              "MM_State_HasAsk",
          ]

          if all(c in mm_df.columns for c in required_cols):
              base_state = {
                  "spread": float(anchor_row["MM_State_Spread"]),
                  "asksize": float(anchor_row["MM_State_AskSize"]),
                  "bidsize": float(anchor_row["MM_State_BidSize"]),
                  "inventory": float(anchor_row["MM_State_Inventory"]),
                  "has_bid": bool(anchor_row["MM_State_HasBid"]),
                  "has_ask": bool(anchor_row["MM_State_HasAsk"]),
              }
              base_state = _attach_flow_features_from_row(base_state, anchor_row)

              print("\n[Interpretability] GENERIC base_state used for 2D plots:")
              print(base_state)

              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  inv_min = int(np.floor(mm_df["MM_State_Inventory"].min()))
                  inv_max = int(np.ceil(mm_df["MM_State_Inventory"].max()))
                  if inv_min == inv_max:
                      inv_min -= 1
                      inv_max += 1
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              s_min = max(1, int(np.floor(mm_df["MM_State_Spread"].min())))
              s_max = int(np.ceil(mm_df["MM_State_Spread"].max()))
              if s_min == s_max:
                  s_max = s_min + 5
              spread_values = np.arange(s_min, s_max + 1, 1)

              a_min = max(0.0, float(mm_df["MM_State_AskSize"].min()))
              a_max = float(mm_df["MM_State_AskSize"].max())
              if a_max <= a_min:
                  a_max = a_min + 1.0
              ask_values = np.linspace(a_min, a_max, num=35)

              b_min = max(0.0, float(mm_df["MM_State_BidSize"].min()))
              b_max = float(mm_df["MM_State_BidSize"].max())
              if b_max <= b_min:
                  b_max = b_min + 1.0
              bid_values = np.linspace(b_min, b_max, num=35)

              bayes_m_values = np.linspace(-1.0, 1.0, 41)
              bayes_run_values = np.linspace(0.0, 1.0, 41)
              quote_exposure_values = np.linspace(-1.0, 1.0, 41)

              plot_value_slices_2d(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  spread_values=spread_values,
                  bid_size_values=bid_values,
                  ask_size_values=ask_values,
                  bayes_m_values=bayes_m_values,
                  bayes_run_values=bayes_run_values,
                  quote_exposure_values=quote_exposure_values,
                  as_cost=False,
              )
          else:
              print(
                  "\n[Interpretability WARNING] MM_State_* columns not found in mm_df; "
                  "skipping generic-mode interpretability plots."
              )

  except Exception as e:
      print(f"\n[Interpretability ERROR] Exception while building 2D value slices: {e}")



  # --------------------------------------------------------------
  # 12) Interpretability Studies - 3D Value Surfaces
  # --------------------------------------------------------------



  def _finite_int_or_none(x) -> Optional[int]:
      try:
          if x is None:
              return None
          x_float = float(x)
          if not np.isfinite(x_float):
              return None
          return int(x_float)
      except (TypeError, ValueError):
          return None


  def _last_valid_rl_state_row(mm_df, mode="pure_mm"):
      required = ["MM_RL_State_Mode", "MM_RL_State_Vector"]
      if not all(c in mm_df.columns for c in required):
          return None

      for idx in reversed(mm_df.index):
          row = mm_df.loc[idx]
          vec = row.get("MM_RL_State_Vector", None)
          row_mode = row.get("MM_RL_State_Mode", None)

          if row_mode == mode and isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
              return row

      return None


  def _decode_pure_mm_network_vector(vec, k_offset: int, deep_controller) -> Dict[str, Any]:
      vec = list(vec)
      k_offset = int(k_offset)
      depth_len = k_offset + 1
      base_len = 2 + 2 * depth_len

      if len(vec) < base_len:
          raise ValueError(
              f"PURE-MM state vector too short: len={len(vec)}, expected at least {base_len}"
          )

      def _inv_log1p(x: float) -> float:
          return float(np.expm1(max(0.0, float(x))))

      inv_denom = float(deep_controller.inv_limit) if deep_controller.inv_limit is not None else 10.0
      inv_denom = max(inv_denom, 1.0)

      state = {
          "spread": _inv_log1p(vec[0]),
          "inventory": float(vec[1]) * inv_denom,
          "pure_mm_bid_sizes": [_inv_log1p(x) for x in vec[2 : 2 + depth_len]],
          "pure_mm_ask_sizes": [_inv_log1p(x) for x in vec[2 + depth_len : 2 + 2 * depth_len]],
      }

      cursor = base_len

      if getattr(deep_controller, "use_flow_signal", False) and cursor < len(vec):
          state["mo_flow_p_hat"] = float(vec[cursor])
          cursor += 1

      if getattr(deep_controller, "use_fast_flow_signal", False) and cursor < len(vec):
          state["mo_flow_fast_p_hat"] = float(vec[cursor])
          cursor += 1

      if getattr(deep_controller, "use_bayes_flow_signal", False):
          bayes_keys = tuple(getattr(deep_controller, "bayes_flow_feature_keys", (
              "bayes_m_hat",
              "bayes_cp_prob",
              "bayes_expected_run_length",
              "bayes_uncertainty",
          )))
          for key in bayes_keys:
              if cursor < len(vec):
                  state[key] = float(vec[cursor])
                  cursor += 1

      if getattr(deep_controller, "use_fill_imbalance", False) and cursor < len(vec):
          state["fill_imbalance_ewma"] = float(vec[cursor])

      return state


  def _attach_flow_features_from_row(base_state: Dict[str, Any], row) -> Dict[str, Any]:
      if getattr(deep_controller, "use_flow_signal", False) and "mo_flow_p_hat" not in base_state:
          if "MM_MO_Flow_PHat" in row:
              base_state["mo_flow_p_hat"] = float(row["MM_MO_Flow_PHat"])

      if getattr(deep_controller, "use_fast_flow_signal", False) and "mo_flow_fast_p_hat" not in base_state:
          if "MM_MO_Flow_Fast_PHat" in row:
              base_state["mo_flow_fast_p_hat"] = float(row["MM_MO_Flow_Fast_PHat"])

      if getattr(deep_controller, "use_bayes_flow_signal", False):
          bayes_col_map = {
              "bayes_m_hat": "MM_Bayes_M_Hat",
              "bayes_cp_prob": "MM_Bayes_CP_Prob",
              "bayes_expected_run_length": "MM_Bayes_Run_Length_Feature",
              "bayes_uncertainty": "MM_Bayes_Uncertainty",
          }
          for key, col in bayes_col_map.items():
              if key not in base_state and col in row:
                  base_state[key] = float(row[col])

      if getattr(deep_controller, "use_fill_imbalance", False) and "fill_imbalance_ewma" not in base_state:
          if USE_QUOTE_EXPOSURE_IMBALANCE and "MM_Fill_Imbalance_Feature" in row:
              base_state["fill_imbalance_ewma"] = float(row["MM_Fill_Imbalance_Feature"])
          elif "MM_Fill_Imbalance_EWMA" in row:
              base_state["fill_imbalance_ewma"] = float(row["MM_Fill_Imbalance_EWMA"])

      return base_state


  def plot_value_surfaces_3d(
      deep_controller,
      base_state: Dict[str, Any],
      inv_values,
      spread_values,
      bid0_values=None,
      ask0_values=None,
      bid_size_values=None,
      ask_size_values=None,
      bayes_m_values=None,
      bayes_run_values=None,
      quote_exposure_values=None,
      as_cost: bool = False,
  ):
      q_net = deep_controller.q_net
      q_net.eval()

      def evaluate_state(mm_state: Dict[str, Any]) -> float:
          with torch.no_grad():
              s_tensor = deep_controller._state_to_tensor(mm_state)
              q_values = deep_controller._q_values_from_net(q_net, s_tensor)
              v = torch.max(q_values, dim=1).values.item()
          return -v if as_cost else v

      def set_feature(mm_state: Dict[str, Any], feature_name: str, value: float):
          if feature_name == "inventory":
              mm_state["inventory"] = float(value)
          elif feature_name == "spread":
              mm_state["spread"] = float(value)
          elif feature_name == "bid0":
              bid_sizes = list(mm_state["pure_mm_bid_sizes"])
              bid_sizes[0] = float(value)
              mm_state["pure_mm_bid_sizes"] = bid_sizes
          elif feature_name == "ask0":
              ask_sizes = list(mm_state["pure_mm_ask_sizes"])
              ask_sizes[0] = float(value)
              mm_state["pure_mm_ask_sizes"] = ask_sizes
          elif feature_name == "bidsize":
              mm_state["bidsize"] = float(value)
          elif feature_name == "asksize":
              mm_state["asksize"] = float(value)
          elif feature_name == "bayes_m_hat":
              mm_state["bayes_m_hat"] = float(value)
          elif feature_name == "bayes_expected_run_length":
              mm_state["bayes_expected_run_length"] = float(value)
          elif feature_name == "quote_exposure":
              mm_state["fill_imbalance_ewma"] = float(value)
          else:
              raise ValueError(f"Unknown feature: {feature_name}")

      def compute_surface(x_values, y_values, x_feature: str, y_feature: str):
          Z = np.zeros((len(y_values), len(x_values)), dtype=np.float32)

          for iy, y in enumerate(y_values):
              for ix, x in enumerate(x_values):
                  mm_state = copy.deepcopy(base_state)
                  set_feature(mm_state, x_feature, x)
                  set_feature(mm_state, y_feature, y)
                  Z[iy, ix] = evaluate_state(mm_state)

          X, Y = np.meshgrid(x_values, y_values)
          return X, Y, Z

      def plot_surface(x_values, y_values, Z, xlabel: str, ylabel: str, title: str):
          X, Y = np.meshgrid(x_values, y_values)

          fig = plt.figure(figsize=(10, 7))
          ax = fig.add_subplot(111, projection="3d")

          surf = ax.plot_surface(
              X,
              Y,
              Z,
              cmap="viridis",
              edgecolor="none",
              alpha=0.92,
              linewidth=0,
              antialiased=True,
          )

          cbar_label = "Cost-to-go J(s)" if as_cost else "Value V(s)"
          fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label=cbar_label)

          ax.set_xlabel(xlabel, fontsize=16)
          ax.set_ylabel(ylabel, fontsize=16)
          ax.set_zlabel(cbar_label, fontsize=16)
          ax.tick_params(labelsize=13)
          # ax.set_title(title)  # removed for paper export
          ax.view_init(elev=28, azim=230)

          plt.tight_layout()
          _save_and_maybe_show(fig, title, prefix="surface3d")
          plt.close(fig)

      def maybe_plot(x_values, y_values, x_feature, y_feature, xlabel, ylabel, title):
          if x_values is None or y_values is None:
              return
          if len(x_values) == 0 or len(y_values) == 0:
              return

          _, _, Z = compute_surface(x_values, y_values, x_feature, y_feature)
          plot_surface(x_values, y_values, Z, xlabel, ylabel, title)

      is_pure_mm_state = (
          deep_controller.pure_mm
          and "pure_mm_bid_sizes" in base_state
          and "pure_mm_ask_sizes" in base_state
      )

      value_label = "cost" if as_cost else "value"

      if is_pure_mm_state:
          maybe_plot(
              spread_values,
              inv_values,
              "spread",
              "inventory",
              "Spread",
              "Inventory",
              f"MM {value_label}: inventory x spread",
          )
          maybe_plot(
              bid0_values,
              inv_values,
              "bid0",
              "inventory",
              "Best-bid queue size",
              "Inventory",
              f"MM {value_label}: inventory x best-bid queue",
          )
          maybe_plot(
              ask0_values,
              inv_values,
              "ask0",
              "inventory",
              "Best-ask queue size",
              "Inventory",
              f"MM {value_label}: inventory x best-ask queue",
          )
          maybe_plot(
              ask0_values,
              bid0_values,
              "ask0",
              "bid0",
              "Best-ask queue size",
              "Best-bid queue size",
              f"MM {value_label}: best-bid x best-ask queue",
          )
      else:
          maybe_plot(
              spread_values,
              inv_values,
              "spread",
              "inventory",
              "Spread",
              "Inventory",
              f"MM {value_label}: inventory x spread",
          )
          maybe_plot(
              bid_size_values,
              inv_values,
              "bidsize",
              "inventory",
              "Bid size",
              "Inventory",
              f"MM {value_label}: inventory x bid size",
          )
          maybe_plot(
              ask_size_values,
              inv_values,
              "asksize",
              "inventory",
              "Ask size",
              "Inventory",
              f"MM {value_label}: inventory x ask size",
          )
          maybe_plot(
              ask_size_values,
              bid_size_values,
              "asksize",
              "bidsize",
              "Ask size",
              "Bid size",
              f"MM {value_label}: bid size x ask size",
          )

      has_bayes_m = (
          getattr(deep_controller, "use_bayes_flow_signal", False)
          and "bayes_m_hat" in base_state
      )
      has_bayes_run = (
          getattr(deep_controller, "use_bayes_flow_signal", False)
          and "bayes_expected_run_length" in base_state
      )
      has_quote_exposure = (
          getattr(deep_controller, "use_fill_imbalance", False)
          and "fill_imbalance_ewma" in base_state
      )

      if has_bayes_m:
          maybe_plot(
              bayes_m_values,
              inv_values,
              "bayes_m_hat",
              "inventory",
              "Bayes directional bias m_hat",
              "Inventory",
              f"Regime {value_label}: inventory x Bayes m_hat",
          )

      if has_bayes_run:
          maybe_plot(
              bayes_run_values,
              inv_values,
              "bayes_expected_run_length",
              "inventory",
              "Bayes expected run-length feature",
              "Inventory",
              f"Regime {value_label}: inventory x Bayes run length",
          )

      if has_quote_exposure:
          maybe_plot(
              quote_exposure_values,
              inv_values,
              "quote_exposure",
              "inventory",
              "Quote-exposure imbalance",
              "Inventory",
              f"Exposure {value_label}: inventory x quote exposure",
          )

      if has_bayes_m and has_quote_exposure:
          maybe_plot(
              quote_exposure_values,
              bayes_m_values,
              "quote_exposure",
              "bayes_m_hat",
              "Quote-exposure imbalance",
              "Bayes directional bias m_hat",
              f"Regime/exposure {value_label}: Bayes m_hat x quote exposure",
          )

      if has_bayes_m:
          maybe_plot(
              spread_values,
              bayes_m_values,
              "spread",
              "bayes_m_hat",
              "Spread",
              "Bayes directional bias m_hat",
              f"Regime/spread {value_label}: Bayes m_hat x spread",
          )


  try:
      if deep_controller.pure_mm:
          anchor_row = _last_valid_rl_state_row(mm_df, mode="pure_mm")

          if anchor_row is None:
              print(
                  "\n[Interpretability WARNING] No valid PURE-MM RL decision state found "
                  "in mm_df. Skipping interpretability plots."
              )
          else:
              rl_dim = anchor_row.get("MM_RL_State_Dim", None)
              rl_max_off = anchor_row.get("MM_RL_State_MaxOffset", None)
              rl_vec = anchor_row.get("MM_RL_State_Vector", None)

              k_from_meta = _finite_int_or_none(rl_max_off)
              if k_from_meta is not None:
                  K = k_from_meta
              else:
                  dim_from_meta = _finite_int_or_none(rl_dim)
                  L = dim_from_meta if dim_from_meta is not None else len(rl_vec)
                  K = int(getattr(deep_controller, "max_offset", max(0, (L - 2) // 2 - 1)))

              base_state = _decode_pure_mm_network_vector(rl_vec, K, deep_controller)
              base_state = _attach_flow_features_from_row(base_state, anchor_row)

              print("\n[Interpretability] PURE-MM base_state used for 3D surfaces:")
              print("  spread:", base_state["spread"])
              print("  inventory:", base_state["inventory"])
              print("  bid_sizes:", base_state["pure_mm_bid_sizes"])
              print("  ask_sizes:", base_state["pure_mm_ask_sizes"])
              for key in ["bayes_m_hat", "bayes_expected_run_length", "fill_imbalance_ewma"]:
                  if key in base_state:
                      print(f"  {key}:", base_state[key])

              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  inv_min = int(np.floor(mm_df["MM_Inventory"].min())) if "MM_Inventory" in mm_df.columns else -20
                  inv_max = int(np.ceil(mm_df["MM_Inventory"].max())) if "MM_Inventory" in mm_df.columns else 20
                  if inv_min == inv_max:
                      inv_min -= 1
                      inv_max += 1
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              q0_all = []
              if "MM_RL_State_Vector" in mm_df.columns and "MM_RL_State_Mode" in mm_df.columns:
                  for vec, mode in zip(mm_df["MM_RL_State_Vector"], mm_df["MM_RL_State_Mode"]):
                      if mode != "pure_mm" or vec is None or not isinstance(vec, (list, np.ndarray)):
                          continue
                      try:
                          decoded_row = _decode_pure_mm_network_vector(vec, K, deep_controller)
                      except (TypeError, ValueError):
                          continue
                      q0_all.append(float(decoded_row["pure_mm_bid_sizes"][0]))
                      q0_all.append(float(decoded_row["pure_mm_ask_sizes"][0]))

              if len(q0_all) > 0:
                  q0_all = np.asarray(q0_all, dtype=float)
                  q_min = max(0.0, float(np.percentile(q0_all, 5)))
                  q_max = float(np.percentile(q0_all, 95))
                  if q_max <= q_min:
                      q_max = q_min + 1.0
                  offset_values = np.linspace(q_min, q_max, num=35)
              else:
                  base0 = max(
                      float(base_state["pure_mm_bid_sizes"][0]),
                      float(base_state["pure_mm_ask_sizes"][0]),
                      1.0,
                  )
                  offset_values = np.linspace(0.0, max(5.0, 2.0 * base0), num=35)

              base_spread = float(base_state.get("spread", 1.0))
              spread_min = max(1.0, base_spread - 3)
              spread_max = base_spread + 3
              spread_values = np.arange(spread_min, spread_max + 1, 1)

              bayes_m_values = np.linspace(-1.0, 1.0, 41)
              bayes_run_values = np.linspace(0.0, 1.0, 41)
              quote_exposure_values = np.linspace(-1.0, 1.0, 41)

              print("\n[Interpretability] 3D grids:")
              print("  inv:", inv_values[0], "->", inv_values[-1])
              print("  spread:", spread_values[0], "->", spread_values[-1])
              print("  queue:", offset_values[0], "->", offset_values[-1])

              plot_value_surfaces_3d(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  spread_values=spread_values,
                  bid0_values=offset_values,
                  ask0_values=offset_values,
                  bayes_m_values=bayes_m_values,
                  bayes_run_values=bayes_run_values,
                  quote_exposure_values=quote_exposure_values,
                  as_cost=False,
              )

      else:
          anchor_row = mm_df.iloc[-1]

          required_cols = [
              "MM_State_Spread",
              "MM_State_AskSize",
              "MM_State_BidSize",
              "MM_State_Inventory",
              "MM_State_HasBid",
              "MM_State_HasAsk",
          ]

          if all(c in mm_df.columns for c in required_cols):
              base_state = {
                  "spread": float(anchor_row["MM_State_Spread"]),
                  "asksize": float(anchor_row["MM_State_AskSize"]),
                  "bidsize": float(anchor_row["MM_State_BidSize"]),
                  "inventory": float(anchor_row["MM_State_Inventory"]),
                  "has_bid": bool(anchor_row["MM_State_HasBid"]),
                  "has_ask": bool(anchor_row["MM_State_HasAsk"]),
              }
              base_state = _attach_flow_features_from_row(base_state, anchor_row)

              print("\n[Interpretability] GENERIC base_state used for 3D surfaces:")
              print(base_state)

              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  inv_min = int(np.floor(mm_df["MM_State_Inventory"].min()))
                  inv_max = int(np.ceil(mm_df["MM_State_Inventory"].max()))
                  if inv_min == inv_max:
                      inv_min -= 1
                      inv_max += 1
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              s_min = max(1, int(np.floor(mm_df["MM_State_Spread"].min())))
              s_max = int(np.ceil(mm_df["MM_State_Spread"].max()))
              if s_min == s_max:
                  s_max = s_min + 5
              spread_values = np.arange(s_min, s_max + 1, 1)

              a_min = max(0.0, float(mm_df["MM_State_AskSize"].min()))
              a_max = float(mm_df["MM_State_AskSize"].max())
              if a_max <= a_min:
                  a_max = a_min + 1.0
              ask_values = np.linspace(a_min, a_max, num=35)

              b_min = max(0.0, float(mm_df["MM_State_BidSize"].min()))
              b_max = float(mm_df["MM_State_BidSize"].max())
              if b_max <= b_min:
                  b_max = b_min + 1.0
              bid_values = np.linspace(b_min, b_max, num=35)

              bayes_m_values = np.linspace(-1.0, 1.0, 41)
              bayes_run_values = np.linspace(0.0, 1.0, 41)
              quote_exposure_values = np.linspace(-1.0, 1.0, 41)

              plot_value_surfaces_3d(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  spread_values=spread_values,
                  bid_size_values=bid_values,
                  ask_size_values=ask_values,
                  bayes_m_values=bayes_m_values,
                  bayes_run_values=bayes_run_values,
                  quote_exposure_values=quote_exposure_values,
                  as_cost=False,
              )
          else:
              print(
                  "\n[Interpretability WARNING] MM_State_* columns not found in mm_df; "
                  "skipping generic-mode interpretability plots."
              )

  except Exception as e:
      print(f"\n[Interpretability ERROR] Exception while building 3D value surfaces: {e}")
