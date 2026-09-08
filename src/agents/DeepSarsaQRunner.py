#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 02:30:00 2025

@author: felipemoret

Deep RL LOB simulation:
    - Multi-episode Online Deep SARSA / Deep Q-Learning
      MarketMaker training with continuous state features.

GENERIC STATE (pure_mm = False)
-------------------------------
    s = [spread, asksize, bidsize, inventory, has_bid, has_ask]

GENERIC ACTIONS (discrete indices)
----------------------------------
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
-----------------------------
    - Discrete actions index into a grid of offsets:
          a -> (bid_offset, ask_offset)  in ticks

    - The controller internally sees an extended state:
          [spread, inventory,
           pure_mm_bid_sizes[0..K],
           pure_mm_ask_sizes[0..K]]

      where K = max_offset inferred from pure_mm_offsets.

    - The MarketMaker always tries to maintain quotes:
          * if |inventory| < inv_limit (or inv_limit is None):
                → two-sided: cancel_all_then_place_bid_ask
          * if inventory >= inv_limit:
                → only ASK side
          * if inventory <= -inv_limit:
                → only BID side


INSIDE-SPREAD QUOTING: HOW IT WORKS
-----------------------------------

1) Generic mode (pure_mm = False)
---------------------------------
    - The discrete action space can be extended beyond the 6 core actions
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
              - If inv >= inv_limit  → only ASK posting is allowed.
              - If inv <= -inv_limit → only BID posting is allowed.
              - If |inv| < inv_limit → two-sided posting (when requested).
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
                    → mapped to ("place_bid_ask_inside_spread",)
              - If the network selects a_idx = 7:
                    → mapped to ("place_bid_inside_spread",)
              - If the network selects a_idx = 8:
                    → mapped to ("place_ask_inside_spread",)


2) PURE MM mode (pure_mm = True)
--------------------------------
    - In PURE MM mode, **inside-spread behavior is controlled by offsets** in
      the grid `pure_mm_offsets`:

          pure_mm_offsets = [
              (bid_offset, ask_offset),
              ...
          ]

      where each offset is in ticks relative to best bid / best ask.

    - Interpretation of offsets:
        * bid_offset > 0  → more passive bid (deeper in the book)
        * bid_offset = 0  → quote at best bid
        * bid_offset < 0  → try to move the bid **inside the spread**
                            (more aggressive, closer to or above mid),
                            while ensuring:
                                - bid_price < best_ask
                                - no crossing of the ask side

        * ask_offset > 0  → more passive ask (further from mid)
        * ask_offset = 0  → quote at best ask
        * ask_offset < 0  → try to move the ask **inside the spread**
                            (more aggressive, closer to or below mid),
                            while ensuring:
                                - ask_price > best_bid
                                - no crossing of the bid side

    - The controller checks the current spread:
        * If the spread is wide enough (e.g. spread ≥ 2 ticks), negative
          offsets are allowed to move inside the spread while enforcing
          the constraints above.
        * If the spread is too narrow, the controller falls back to quoting
          at L1 (best bid / best ask).

    - Inventory band (PURE MM):
        * As in generic mode:
              if inv >= inv_limit  → only ASK side is posted
              if inv <= -inv_limit → only BID side is posted
              otherwise            → two-sided quoting (bid + ask)
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

----------------------------------------------------
How to launch TensorBoard while running this script
----------------------------------------------------

This script writes TensorBoard logs to a mode-specific subdirectory:

    runs_spyder/deep_mm_pure/      (if USE_PURE_MM = True)
    runs_spyder/deep_mm_generic/   (if USE_PURE_MM = False)

The log path is passed as `log_dir=` when building the DeepRLController.
The controller owns the SummaryWriter and writes episode-level metrics
from log_episode_stats().  The runner adds a few extra scalars (LR,
moving averages, inventory diagnostics) on top.

IMPORTANT:
- In this script we call:
      os.chdir("/Users/felipemoret/Desktop/MM_LOB_SIM/")
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
- Adjust the logdir if INV_PENALTY_COEFF changes (e.g., invp0.0100 for φ=0.01).

2) Then open the dashboard in your browser:

   http://localhost:6010

TensorBoard Metrics Logged (all x-axis = episode index)
--------------------------------------------------------

  REWARD / PNL (logged by DeepRLController.log_episode_stats):
    episode/total_reward        — Sum of per-step rewards for the episode
    episode/discounted_return   — G_0 = sum(gamma^t * r_t), micro-step discount
    episode/final_pnl           — Terminal PnL (cash + mark-to-market)

  REWARD MOVING AVERAGES (logged by the runner):
    episode/total_reward_ma_10  — 10-episode moving average of total_reward
    episode/final_pnl_ma_10    — 10-episode moving average of final_pnl

  TD LOSS (logged by DeepRLController.log_episode_stats):
    episode/mean_td_loss        — Mean Huber/KL loss per gradient step this episode

  EXPLORATION (logged by DeepRLController.log_episode_stats):
    episode/epsilon             — Current epsilon (inside controller)
    episode/epsilon_outer       — Current epsilon (logged by runner after decay)
    episode/mean_noisy_sigma    — Mean |sigma| across NoisyLinear layers
                                  (only if use_noisy_net=True).
                                  If it drops to ~0 early, exploration died.

  PER SCHEDULE (logged by DeepRLController.log_episode_stats, only if use_per=True):
    episode/per_alpha           — Current PER prioritization exponent
                                  (annealed: typically 0.6 -> 0.4)
    episode/per_beta            — Current IS-correction exponent
                                  (annealed: typically 0.4 -> 1.0)

  Q-VALUE DIAGNOSTICS (logged by DeepRLController.log_episode_stats):
    episode/mean_Q_a{i}         — Mean Q(s, a=i) across all decisions this episode,
                                  one scalar per action index i in [0, n_actions)
    episode/Q_spread            — max(mean_Q) - min(mean_Q) across actions.
                                  Measures action differentiation.
                                  Good: 0.05-0.10 (20-25 C51 atoms of separation)
                                  Bad:  ~0 (policy collapse, all actions equal)
    episode/Q_min               — Min Q(s,a) across all valid actions this episode
    episode/Q_max               — Max Q(s,a) across all valid actions this episode
    episode/Q_mean              — Mean Q(s,a) across all actions

  TD TARGET + C51 SUPPORT DIAGNOSTICS (logged by DeepRLController.log_episode_stats):
    episode/td_target_min       — Min scalar TD target r + gamma^n * Q(s',a*)
    episode/td_target_max       — Max scalar TD target
    episode/td_target_mean      — Mean scalar TD target
                                  Compare with [V_min, V_max]: if td_target extremes
                                  far exceed V_min/V_max, the C51 support is too narrow.
    episode/tz_clip_frac        — Fraction of C51 atoms clipped at [V_min, V_max]
                                  during Tz projection (only if use_distributional=True).
                                  Good: <5% and decreasing. Bad: >20% (widen support).

  INVENTORY DIAGNOSTICS (logged by the runner):
    episode/max_abs_inventory   — Worst-case |inventory| during the episode
    episode/mean_abs_inventory  — Mean |inventory| (lower = tighter control)
    episode/pct_at_inv_limit    — Fraction of steps with |inv| >= inv_limit

  LEARNING RATE (logged by the runner):
    train/lr                    — Current optimizer learning rate (after decay)

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

# ── Import safety: this file is a training script, not a library ──────
# All side-effect code (chdir, log cleanup, training loop) lives inside
# the __main__ guard below so that importing this file is safe.

if __name__ == "__main__":

  os.chdir("/Users/felipemoret/Desktop/MM_LOB_SIM/")

  # =====================================================================
  # LOB ENGINE — pure limit-order-book simulator (no market-maker)
  # =====================================================================
  # LOB_SIM_SANTA_FE contains the Santa-Fe-style LOB simulator used to
  # generate synthetic order flow (Poisson arrivals, geometric cancels).
  # We import its top-level entry point `simulate_LOB` which returns
  # (msg_df, ob_df, ewma_trace) for a single episode.
  # =====================================================================
  from LOB_SIM_SANTA_FE import simulate_LOB

  # =====================================================================
  # MM + LOB ENGINE — coupled market-maker / LOB simulator
  # =====================================================================
  # MM_LOB_SIM extends the pure LOB simulator with a pluggable
  # MarketMaker agent.  `simulate_LOB_with_MM` runs one full episode,
  # returning (msg_df, ob_df, mm_df).
  #
  # We also import a collection of reward functions.  Each function has
  # the signature reward_fn(mm, lob, state_before, state_after, info)
  # and returns a scalar float.  The user can swap them in the training
  # loop to experiment with different shaping strategies.
  # =====================================================================
  from MM_LOB_SIM import simulate_LOB_with_MM
  from CONFIG_MM import lam, mu, delta, qrm_params, mean_size_LO, mean_size_MO, USE_QRM

  from MM_LOB_SIM import reward_spread_capture_inv_quadratic, reward_pnl_dampened_inv_quadratic

  # =====================================================================
  # REWARD CONFIG
  # =====================================================================
  INV_PENALTY_COEFF = 0.001
  USE_DAMPENED_REWARD = True      # False = spread capture reward, True = dampened PnL reward

  # ── Transfer learning: warm-start from a pre-trained checkpoint ──────
  #WARMSTART_CKPT = "checkpoints/deep_mm_mtm_pure_invp0.0010_final.pt"      # ← Set to checkpoint path to warm-start, e.g.
  #WARMSTART_CKPT = "checkpoints/deep_mm_mtm_pure_invp0.0030_final.pt"
  
  
  WARMSTART_CKPT = None
  
  FORCE_NOISY_NET = True          # set True to force NoisyNet even with warmstart
  RESET_SIGMAS = True             # set True to reset NoisyNet sigmas to sigma_init=0.5
                                   # instead of keeping the pre-calibrated values from checkpoint
  USE_FACTORED_NOISE = True        # [EXPERIMENT] factored noise ON for Phase A invp0.001 test
  USE_FULLY_NOISY    = True         # [EXPERIMENT] fully-noisy ON for Phase A invp0.001 test
                                   # Factored noise stabilises sigma training and prevents
                                   # random-walk growth observed with independent noise on
                                   # high-dimensional outputs (C51 with 51 atoms × 7 actions)
  NOISY_WEIGHT_DECAY = 0.0         # L2 weight decay applied ONLY to sigma params (w_sigma, b_sigma)
                                   # Creates a restoring force that prevents random-walk growth:
                                   #   σ_new = σ * (1 - lr * wd)  per step
                                   # Recommended: 0.01 for C51+NoisyNet.  0.0 = disabled (legacy).
  EXPERIMENT_TAG = "g0p999"        # [EXPERIMENT] distinguishes factored+fully gamma=0.999 run
                                   # from the gamma=0.97 checkpoints (independent + factored) already on disk
                                   # examples: "factored", "wd0010", "testA"
  SAVE_FINAL_REPLAY_BUFFER = False # replay dumps are large; keep opt-in by default
  CHECKPOINT_SAVE_RETRIES = 3      # retry transient filesystem write failures
  CHECKPOINT_RETRY_DELAY_SEC = 2.0

  # reward_fn_spread is built after best_params / EPISODE_LENGTH are defined (see below)

  # =====================================================================
  # ANIMATION — LOB + MM visual replay
  # =====================================================================
  from animate_LOB_sim import animate_lob

  # =====================================================================
  # DEEP RL CONTROLLER — DQN / SARSA with Rainbow extensions
  # =====================================================================
  # DeepRLController implements the RLController protocol and supports:
  #   - DQN or Deep SARSA (on-policy / off-policy)
  #   - Double Q-learning (decoupled selection / evaluation)
  #   - Dueling architecture (separate V and A streams)
  #   - Prioritized Experience Replay (PER with alpha/beta annealing)
  #   - NoisyNets (parameter-space exploration, replaces ε-greedy)
  #   - Distributional C51 (categorical return distribution)
  #   - n-step TD returns with SMDP reward aggregation
  #   - Hard throttling (event / time / TOB gating)
  # =====================================================================
  from dqn_distributional_with_throttle import DeepRLController

  import pandas as pd
  mpl.rcParams["animation.embed_limit"] = 1024  # MB (e.g. 1 GB)

  # Set the maximum number of columns to "None" (unlimited)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.width', None)

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

      # Include the target network if it exists (needed for seamless
      # training resumption — avoids re-initializing from q_net).
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


  # --------------------------------------------------
  # 0) REPRODUCIBILITY SETUP
  # --------------------------------------------------
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
  # --------------------------------------------------
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

  GLOBAL_SEED = 123
  seed_everything(GLOBAL_SEED)
  print(f"[System] Global Seed locked to {GLOBAL_SEED}. Deterministic Mode ON.")

  # ============================================================
  # LOB parameters (calibrated from AMZN LOBSTER data)
  # ============================================================
  # These parameters define the Santa-Fe-style LOB model dynamics:
  #
  #   lam   — limit order arrival intensity (orders per unit time per
  #           tick level).  Higher lam = denser book.
  #   mu    — cancellation intensity (probability per unit time that
  #           an existing limit order is cancelled).
  #   delta — market order arrival intensity.  Controls how often
  #           aggressive orders sweep through the book.
  #   mo_size — size of each market order (in lots).
  #
  #   NUMBER_TICK_LEVELS — how many price ticks the simulator tracks
  #           on each side of the mid.  Tick 0 = best bid/ask.
  #   N_PRIORITY_RANKS  — maximum number of orders that can queue at
  #           a single price level (FIFO priority).
  # ============================================================

  NUMBER_TICK_LEVELS = 50
  N_PRIORITY_RANKS = 100

  # --------------------------------------------------
  # 1) OPTIONAL: offline synthetic LOB simulation (NO MM)
  #    Useful if you want to inspect basic LOB stats before training.
  # --------------------------------------------------
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
  print("=================================\n")

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
  print("============================\n")


  # ============================================================
  # Optuna v3 study — mm_dqn_tuning_v3_pure_mm
  # Best trial for long runs: T55 (rank #5 by objective = 41.81)
  # PnL = +0.78 | mean|inv| = 3.69 | max|inv| = 8 | %@limit = 11.6%
  # Low gamma + large net + high LR → best scaling to 500 episodes.
  # ============================================================


  # --------------------------------------------------
  # 2) Experiment configuration: choose controller mode
  # --------------------------------------------------
  #
  USE_PURE_MM = True    # <--- PURE MM controller (offset grid)
  #
  #USE_PURE_MM = False  # <--- GENERIC controller (discrete 6-9 actions)

  # --------------------------------------------------
  # Intra-decision (SMDP) reward aggregation toggle
  # --------------------------------------------------
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
  # --------------------------------------------------
  USE_INTRA_EVENT_GAMMA = True

  # Single source of truth for the PURE MM action grid.
  # This is used both by the controller construction and by run/checkpoint
  # naming so the `_aN` suffix always reflects the REAL number of actions
  # in the offset grid, not the legacy generic `best_params["n_actions"]`.
  PURE_MM_OFFSETS = [
      (-1, -1),  # 0. Aggressive both (inside spread)
      (-1,  0),  # 1. Aggressive bid / at-best ask
      ( 0, -1),  # 2. At-best bid / aggressive ask
      ( 0,  0),  # 3. Symmetric at-best (L1)
      ( 0,  1),  # 4. At-best bid / passive ask (+1 tick)
      ( 1,  0),  # 5. Passive bid (+1 tick) / at-best ask
      #(-1, 1),
      #(1, -1),
  ]

  # --------------------------------------------------
  # Best hyperparameters (Optuna-tuned baseline + manual overrides)
  # --------------------------------------------------
  # This dictionary centralizes ALL RL hyperparameters in one place.
  # Both GENERIC and PURE MM branches read from it, so changes here
  # propagate to both controller modes automatically.
  #
  # IMPORTANT: some keys are mode-specific:
  #   "n_actions" — only used by the GENERIC branch (pure_mm branch
  #                 derives n_actions from len(pure_mm_offsets)).
  #
  # Rainbow components enabled via boolean flags:
  #   use_double       — Double Q-learning (anti-overestimation)
  #   use_dueling      — Dueling architecture (separate V + A streams)
  #   use_per          — Prioritized Experience Replay
  #   use_noisy_net    — NoisyNets (replaces ε-greedy exploration)
  #   use_distributional — C51 (categorical return distribution)
  # --------------------------------------------------
  best_params = {
      # --- Source: Optuna v3 pure_mm Trial 55 (rank #5, objective = 41.81) ---
      # PnL = +0.78 | mean|inv| = 3.69 | max|inv| = 8 | %@limit = 11.6%
      # Best candidate for long runs: low gamma, large net, high LR, tight v_bound.

      # --- Core RL ---
      "lr": 1.5e-3,                # Optuna T55: 3.49e-4 (highest LR in top-5 consensus cluster)
      "weight_decay": 0.01,         # Explicitly disable AdamW weight decay in this run config
      "lr_decay_type": "linear", # "linear" or "exponential"
      "gamma": 0.999,               # [EXPERIMENT] Phase-B gamma (1-1e-3) for factored+fully retrain; Phase A default was 0.97
      "epsilon_start": 0.05,            # decays to epsilon_min over N_EPISODES
      "epsilon_min": 0.05,            # exploration floor
      "epsilon_decay": 0.995,         # per-episode multiplicative decay
      "epsilon_decay_type": "linear",  # smooth ε from start→min
      "batch_size": 128,              # Optuna batch_size grid: best late-stage convergence for long runs
      "target_update_steps": 2000,    # hard target-net sync every 2000 grad steps
      "inv_limit": 8,                 # T55: inv_limit=8, agent reaches mean|inv|≈3.69
      "n_neurons": 256,               # T55: 256 neurons — full capacity for long training
      "n_hidden": 1,                  # 1 hidden layer × 256 neurons
      "activation": "relu",           # MLP trunk activation: "relu" or "elu"
      "elu_alpha": None,              # ELU alpha; None -> PyTorch default (=1.0) when activation="elu"
      "replay_capacity": 150_000,      # ~200 recent episodes
      "n_steps": 5,                   # 5-step TD returns
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
      "dist_v_min": -2.0,            # C51 support lower bound (v3 config)
      "dist_v_max":  2.0,            # C51 support upper bound (v3 config)
      "dist_atoms": 51,               # C51 number of atoms
      "use_per": True,               # Prioritized Experience Replay ON
      # NoisyNet: ON by default for training from scratch.
      # Set FORCE_NOISY_NET = True to override and use NoisyNet even with warmstart.
      "use_noisy_net": FORCE_NOISY_NET or (WARMSTART_CKPT is None),

      # --- PER annealing schedule ---
      # α controls prioritization strength: 1.0 = full priority, 0 = uniform.
      # Anneal α DOWN (0.6→0.4): start with moderate prioritization, relax over time.
      # β controls importance-sampling correction: must reach 1.0 for unbiased convergence.
      # Anneal β UP (0.4→1.0): standard schedule from Schaul et al. 2016.
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


  # --------------------------------------------------
  # Learning-rate schedule (episode-based)
  # --------------------------------------------------
  # Initial learning rate comes from best_params.
  if WARMSTART_CKPT is not None:
      LR_START = best_params["lr"]
  else:
      LR_START = best_params["lr"]       # full LR for training from scratch

  # Final LR will be LR_START * LR_END_FACTOR.
  LR_END_FACTOR = 0.1

  def get_lr_for_episode(ep_index: int, n_episodes: int, use_exp: bool = True) -> float:
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

  def get_cyclical_lr_multiplier(ep_index: int,
                                 period: int = 200,
                                 min_factor: float = 0.1) -> float:
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
      cycle_progress = (ep_index % period) / max(period, 1)
      return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + _math_clr.cos(_math_clr.pi * cycle_progress))

  # --------------------------------------------------
  # 4) MULTI-EPISODE ONLINE TRAINING CONFIG
  # --------------------------------------------------
  # Each episode runs a fresh LOB simulation of EPISODE_LENGTH events.
  # The first ITER_TO_EQUILIBRIUM events use the warm-up phase (the LOB
  # starts empty and needs time to build a realistic book shape before
  # the MM can meaningfully interact with it).
  #
  # N_EPISODES controls how many episodes the agent trains for.
  # Epsilon, learning rate, and PER schedules are all tied to this count.
  # --------------------------------------------------

  EPISODE_LENGTH = 5_000           # environment events per episode (~16 min sim time)
  ITER_TO_EQUILIBRIUM = 1000       # warm-up events (reduced proportionally, LOB stabilizes in ~500)
  N_EPISODES = 2000

  # Logging verbosity: when False, prints only episode number, final PnL,
  # and PnL MA(10).  When True, prints full diagnostics (epsilon, LR,
  # Q-values, TD targets, inventory stats, action usage, etc.).
  VERBOSE_LOGGING = False

  # Early stopping: stop training if the rolling PnL drops significantly
  # below its peak, halting before the policy degrades too far.
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
  BEST_CKPT_WINDOW = 50

  # Validate early stopping config
  if EARLY_STOPPING:
      assert EARLY_STOP_WINDOW > 0, f"EARLY_STOP_WINDOW must be > 0, got {EARLY_STOP_WINDOW}"
      assert 0 < EARLY_STOP_DROP <= 1.0, f"EARLY_STOP_DROP must be in (0, 1], got {EARLY_STOP_DROP}"
      assert EARLY_STOP_MIN_EPISODES >= EARLY_STOP_WINDOW, (
          f"EARLY_STOP_MIN_EPISODES ({EARLY_STOP_MIN_EPISODES}) must be >= "
          f"EARLY_STOP_WINDOW ({EARLY_STOP_WINDOW})")
  assert BEST_CKPT_WINDOW > 0, f"BEST_CKPT_WINDOW must be > 0, got {BEST_CKPT_WINDOW}"

  # --------------------------------------------------
  # CYCLICAL LEARNING RATE — cosine annealing with warm restarts
  # --------------------------------------------------
  # When USE_CYCLICAL_LR = True, the base learning rate (from the
  # monotonic decay schedule) is modulated by a cosine warm-restart
  # cycle.  This can help the agent escape local optima when the
  # environment difficulty changes mid-training.
  #
  #   lr_effective = lr_base × (0.5 + 0.5 * cos(π * cycle_progress))
  #
  # where cycle_progress resets to 0 at the start of each cycle.
  #
  # This is a Phase A stationary runner.  The only MO-flow distribution
  # used below is the balanced baseline buy_mo_prob = 0.5.
  # --------------------------------------------------
  USE_CYCLICAL_LR        = True
  CYCLICAL_LR_PERIOD     = 200    # episodes per cosine half-cycle (T_0)
  CYCLICAL_LR_MIN_FACTOR = 0.3   # minimum multiplier at cycle trough

  # ── Reward / robustness features ──────────────────────────────────────
  USE_INVENTORY_WALL   = True    # DQN SC: no inv wall (not in checkpoint meta)
  USE_TERMINAL_BONUS   = False    # terminal PnL bonus (disabled for baseline)
  USE_G_CLIP           = True    # clip n-step return G to [v_min, v_max]
  USE_PER_PRIORITY_CLIP = True   # cap PER priorities at median_ema * 3

  # --------------------------------------------------
  # Reward function
  # --------------------------------------------------
  _inv_limit_val = best_params.get("inv_limit", 8)
  _reward_kwargs = dict(inv_penalty_coeff=INV_PENALTY_COEFF)

  if USE_INVENTORY_WALL:
      _reward_kwargs["use_inv_wall"] = True
      _reward_kwargs["inv_wall_threshold"] = _inv_limit_val / 2.0
      print(f"[REWARD] Inventory wall: threshold={_reward_kwargs['inv_wall_threshold']}, "
            f"coeff=φ={INV_PENALTY_COEFF}")

  if USE_TERMINAL_BONUS:
      _reward_kwargs["use_terminal_bonus"] = True
      print("[REWARD] Terminal PnL bonus: ON (PnL in ticks, no attenuation)")

  if USE_DAMPENED_REWARD:
      reward_fn_spread = partial(reward_pnl_dampened_inv_quadratic, **_reward_kwargs)
      print("[REWARD] Using reward_pnl_dampened_inv_quadratic (dampened PnL)")
  else:
      reward_fn_spread = partial(reward_spread_capture_inv_quadratic, **_reward_kwargs)
      print("[REWARD] Using reward_spread_capture_inv_quadratic (spread capture)")

  def _build_train_suffix() -> str:
      _n_act = len(PURE_MM_OFFSETS) if USE_PURE_MM else best_params.get("n_actions", 6)
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

  # --------------------------------------------------
  # 3) Clean previous TensorBoard runs for this experiment
  # --------------------------------------------------
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

  # --------------------------------------------------
  # 5) Build Deep RL controller (GENERIC or PURE MM)
  # --------------------------------------------------

  pure_mm_offsets = None

  if not USE_PURE_MM:
      # ======================================================
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
      # ======================================================
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

          # --- Prioritized Experience Replay (PER) annealing schedules ---
          # T44-tuned: moderate → mild alpha; partial → fuller beta correction.
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
          use_flow_signal=False,
          use_fast_flow_signal=False,
          use_fill_imbalance=False,

          # --- Robustness clipping ---
          use_g_clip=USE_G_CLIP,
          use_per_priority_clip=USE_PER_PRIORITY_CLIP,

          # --- MM mode ---
          pure_mm=False,                         # <--- GENERIC mode
          inv_limit=best_params.get("inv_limit", None),
          pure_mm_offsets=None,                  # ignored in generic mode

          # --- Intra-decision (SMDP) reward aggregation ---
          # See USE_INTRA_EVENT_GAMMA in the runner config block.
          use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),
      )
  else:
      # ======================================================
      # PURE MM MODE:
      #   - Discrete actions index a grid of offsets (bid_off, ask_off).
      #   - Example grid below (9 actions) includes both passive and
      #     aggressive/inside-spread behaviors through negative offsets.
      #
      #   - The RL agent *only* chooses how tight/wide the quotes are.
      #   - Inventory band enforced via inv_limit:
      #         |inv| < inv_limit → 2-sided quoting
      #         inv >= inv_limit  → only ASK side
      #         inv <= -inv_limit → only BID side
      # ======================================================
      # pure_mm_offsets = [
      #     (0, 0),
      #     (-1, 0),
      #     (0, -1),
      #     (-1, -1),
      # ]
    
    
      # =============================================================
      # grid6_with_aggressive — 6-action grid with inside-spread
      # Adds aggressive (offset -1) placements for active inventory
      # reduction. Removes passive-both (1,1) which was never used.
      # =============================================================
      pure_mm_offsets = list(PURE_MM_OFFSETS)


      deep_controller = DeepRLController(
          level_offset=0,
          # In PURE MM mode, n_actions must match the length of the offset
          # grid — each action index maps to a (bid_offset, ask_offset) pair.
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

          # --- PER annealing schedules (T44-tuned) ---
          per_alpha_start=best_params.get("per_alpha_start", 0.5386),
          per_alpha_end=best_params.get("per_alpha_end", 0.3750),
          per_alpha_last_episode=N_EPISODES,
          per_beta_start=best_params.get("per_beta_start", 0.3043),
          per_beta_end=best_params.get("per_beta_end", 0.5481),
          per_beta_last_episode=N_EPISODES,

          # --- n-step TD ---
          n_steps=best_params["n_steps"],

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
          use_flow_signal=False,
          use_fast_flow_signal=False,
          use_fill_imbalance=False,

          # --- Robustness clipping ---
          use_g_clip=USE_G_CLIP,
          use_per_priority_clip=USE_PER_PRIORITY_CLIP,

          # --- MM mode ---
          pure_mm=True,                         # <--- PURE MM mode enabled
          inv_limit=best_params.get("inv_limit", None),
          pure_mm_offsets=pure_mm_offsets,

          # --- Intra-decision (SMDP) reward aggregation ---
          # See USE_INTRA_EVENT_GAMMA in the runner config block.
          use_intra_event_gamma=bool(best_params.get("use_intra_event_gamma", True)),
      )

  # ── Warm-start: load weights from a pre-trained checkpoint ────────────
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

      # Check if warmstart checkpoint has a different input dim.
      _ws_input_dim = _ws["q_net"]["feature.0.weight"].shape[1]
      _cur_input_dim = deep_controller.q_net.state_dict()["feature.0.weight"].shape[1]

      if _ws_input_dim != _cur_input_dim:
          print(f"[WARMSTART] Input dim mismatch: checkpoint={_ws_input_dim}, "
                f"model={_cur_input_dim}. Padding first layer with zeros.")
          for net_key in ("q_net", "target_net"):
              if net_key not in _ws:
                  continue
              sd = _ws[net_key]
              w = sd["feature.0.weight"]                          # [n_neurons, old_dim]
              pad = torch.zeros(w.shape[0], _cur_input_dim - _ws_input_dim)
              sd["feature.0.weight"] = torch.cat([w, pad], dim=1) # [n_neurons, new_dim]

      # ── Key conversion for architecture mismatch ─────────────────────
      # The checkpoint and current model may differ in NoisyNet usage.
      # We detect this by checking if the checkpoint has w_mu keys
      # (NoisyLinear) or weight keys (nn.Linear) in the head layers.
      _ckpt_is_noisy = any("w_mu" in k for k in _ws["q_net"].keys())
      _model_is_noisy = best_params["use_noisy_net"]

      if _ckpt_is_noisy and not _model_is_noisy:
          # Checkpoint: NoisyLinear → Model: nn.Linear
          # Convert w_mu→weight, b_mu→bias, drop w_sigma/b_sigma.
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
          print("[WARMSTART] Converted NoisyLinear→Linear (w_mu→weight, b_mu→bias, dropped sigma)")

      elif not _ckpt_is_noisy and _model_is_noisy:
          # Checkpoint: nn.Linear → Model: NoisyLinear
          # Convert weight→w_mu, bias→b_mu in head layers.
          # w_sigma/b_sigma are NOT in the checkpoint — they will keep
          # their constructor-initialised values (overwritten below in
          # the sigma calibration block).
          for net_key in ("q_net", "target_net"):
              if net_key not in _ws:
                  continue
              sd = _ws[net_key]
              keys_to_rename = []
              for k in list(sd.keys()):
                  # Only rename head layers (fc_value, fc_adv, fc_out),
                  # not feature layers which use standard nn.Linear.
                  if any(head in k for head in ("fc_value.", "fc_adv.", "fc_out.")):
                      if ".weight" in k and ".w_mu" not in k:
                          keys_to_rename.append((k, k.replace(".weight", ".w_mu")))
                      elif ".bias" in k and ".b_mu" not in k:
                          keys_to_rename.append((k, k.replace(".bias", ".b_mu")))
              for old_k, new_k in keys_to_rename:
                  sd[new_k] = sd.pop(old_k)
          print("[WARMSTART] Converted Linear→NoisyLinear (weight→w_mu, bias→b_mu in heads)")

      # Load state dicts.  Use strict=False when architecture differs
      # (e.g. non-noisy checkpoint → noisy model: sigma keys are missing
      # from checkpoint and will keep their constructor-initialised values).
      _strict = not (not _ckpt_is_noisy and _model_is_noisy)
      deep_controller.q_net.load_state_dict(_ws["q_net"], strict=_strict)
      if "target_net" in _ws and deep_controller.target_net is not None:
          deep_controller.target_net.load_state_dict(_ws["target_net"], strict=_strict)
      if _ws_input_dim != _cur_input_dim or (_ckpt_is_noisy != _model_is_noisy):
          print("[WARMSTART] Skipping optimizer state (architecture changed, Adam moments incompatible).")
      elif "optimizer" in _ws and deep_controller.optimizer is not None:
          deep_controller.optimizer.load_state_dict(_ws["optimizer"])
      print(f"[WARMSTART] Loaded weights from: {WARMSTART_CKPT}")

      # ── NoisyNet sigma handling at warmstart ──────────────────────────
      # Three scenarios when loading a checkpoint into a NoisyNet model:
      #
      #   A) Checkpoint HAS w_sigma/b_sigma (trained with NoisyNet):
      #      → sigmas are already loaded by load_state_dict.  They are
      #        calibrated to the weight magnitudes from training.
      #        DO NOT reset — just keep them as-is.
      #
      #   B) Checkpoint does NOT have w_sigma/b_sigma (trained without
      #      NoisyNet, e.g. Fase A with use_noisy_net=False):
      #      → load_state_dict(strict=False) leaves w_sigma/b_sigma at
      #        their constructor-initialised values (sigma_init=0.5).
      #        sigma=0.5 is WAY too large relative to the trained weights
      #        (mean|w|≈0.09–0.16) and would make the agent nearly random.
      #        FIX: scale sigmas DOWN to a fraction of the loaded weight
      #        magnitudes, so exploration is gentle (not destructive).
      #
      #   C) NoisyNet is disabled (use_noisy_net=False):
      #      → skip entirely, use ε-greedy instead.
      #
      # The key insight: for warmstarts, we want SMALL noise relative
      # to the pre-trained weights — just enough to explore nearby
      # policies, not enough to scramble the learned value function.
      # A good heuristic is sigma ≈ 3-5% of mean|w_mu|, matching
      # what NoisyNet converges to after full training.
      # ------------------------------------------------------------------
      if best_params["use_noisy_net"]:
          # Check if the checkpoint contained sigma params by looking
          # at the raw checkpoint keys (before any renaming/dropping).
          _ckpt_had_sigma = any(
              "w_sigma" in k
              for net_key in ("q_net", "target_net")
              if net_key in _ws_raw_keys
              for k in _ws_raw_keys[net_key]
          )

          if _ckpt_had_sigma and not RESET_SIGMAS:
              # Scenario A: checkpoint already has calibrated sigmas.
              # They were loaded by load_state_dict — nothing to do.
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
              # Scenario B: checkpoint was non-noisy.  Scale sigmas to
              # ~5% of w_mu magnitude (matches post-convergence levels).
              import math as _math
              _SIGMA_FRACTION = 0.05  # target: sigma ≈ 5% of |w_mu|
              _reset_count = 0
              for net in (deep_controller.q_net, deep_controller.target_net):
                  if net is None:
                      continue
                  for name, param in net.named_parameters():
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
                        f"σ/μ ratio={_ratio:.1%}")
      else:
          print(f"[WARMSTART] NoisyNet disabled → using ε-greedy={best_params['epsilon_start']}")

      del _ws

  # ── NoisyNet sigma weight decay (prevents random-walk growth) ────────
  # Must be called after optimizer setup.
  if NOISY_WEIGHT_DECAY > 0 and best_params.get("use_noisy_net", False):
      deep_controller.setup_noisy_weight_decay(sigma_weight_decay=NOISY_WEIGHT_DECAY)

  # =====================================================================
  # 6) MAIN TRAINING LOOP — Multi-Episode Online RL
  # =====================================================================
  # For each episode:
  #   1. Reset episode-level controller state (SMDP, throttle, nstep)
  #   2. Run a full LOB + MM simulation (simulate_LOB_with_MM)
  #   3. The controller's act() / learn() are called at every micro-step
  #      inside the simulation loop, accumulating SMDP transitions
  #   4. After the episode ends, log stats, decay epsilon, update LR
  #
  # The training loop is "online" — the agent trains while interacting
  # with the environment (no separate data-collection phase).
  # =====================================================================

  if USE_CYCLICAL_LR:
      print(f"{'─'*60}")
      print("PHASE A STATIONARY CONFIG")
      print("  MO flow        = stationary p_buy=0.5")
      print(f"  [CYCLICAL LR]  period={CYCLICAL_LR_PERIOD}, min_factor={CYCLICAL_LR_MIN_FACTOR}")
      print(f"{'─'*60}\n")

  episode_stats = []

  # Early stopping state
  _es_best_ma = float("-inf")
  _es_best_ep = 0
  _es_triggered = False

  # Best-checkpoint state (tracked via MA(BEST_CKPT_WINDOW) of final_pnl)
  _best_ckpt_ma = float("-inf")
  _best_ckpt_ep = -1
  _best_ckpt_path = None

  for ep in range(N_EPISODES):
      if VERBOSE_LOGGING:
          print("=" * 60)
          print(f"Starting episode {ep + 1}/{N_EPISODES}")
          print(f"Current epsilon BEFORE episode: {deep_controller.epsilon:.4f}")
          print(f"Controller mode: {'PURE MM' if deep_controller.pure_mm else 'GENERIC'}")
          print("MO flow: STATIONARY (buy_mo_prob=0.5)")

          if hasattr(deep_controller, "optimizer"):
              current_lr = deep_controller.optimizer.param_groups[0]["lr"]
              print(f"Current learning rate: {current_lr:.8f}")

      # ---------------------------------------------------------------
      # Per-episode random seed
      # ---------------------------------------------------------------
      # Each episode gets a unique seed = GLOBAL_SEED + 1 + ep.
      # This ensures:
      #   (a) Different episodes see different LOB environments (order
      #       flow, cancels, market orders all differ).
      #   (b) Given the same GLOBAL_SEED, the SEQUENCE of episodes is
      #       perfectly reproducible across runs.
      #   (c) The RL exploration noise (ε-greedy, NoisyNets) is also
      #       reproducible per episode.
      # ---------------------------------------------------------------
      current_ep_seed = GLOBAL_SEED + 1 + ep

      # 1. NumPy — controls LOB event sampling inside simulate_LOB_with_MM
      np.random.seed(current_ep_seed)

      # 2. Python built-in — used by any stdlib random calls
      random.seed(current_ep_seed)

      # 3. PyTorch — controls NoisyLinear noise, dropout, and any GPU ops
      torch.manual_seed(current_ep_seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed(current_ep_seed)

      # =================================================================
      # EPISODE BOUNDARY RESET — Clean controller state between episodes
      # =================================================================
      # Without a full reset, state from episode N leaks into episode N+1:
      #
      #   last_action_idx : stale action index → learn() creates a phantom
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
      #         (last_inventory != new starting inventory → act thinks
      #          a fill happened when it didn't).
      #       - False mode-change bypass (last_mode from old episode
      #         differs from the new starting mode).
      #       - Time gate blocking if last_update_time is far in the
      #         future relative to the new episode's time=0.
      #
      # NOTE: If simulate_LOB_with_MM sends done=True on the final step,
      # learn() already clears last_action_idx, nstep_buffer, and SMDP.
      # We reset them again here DEFENSIVELY — the cost is negligible and
      # it protects against edge cases where done might not fire.
      # =================================================================

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

      ep_buy_mo_prob = 0.5

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
          random_seed=GLOBAL_SEED + 1 + ep,
          buy_mo_prob=ep_buy_mo_prob,
          qrm_params=qrm_params,
      )

      # --------------------------------------------------
      # ENVIRONMENT-LEVEL STATS (what actually happened)
      # --------------------------------------------------
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

      # ------------------------------------------------------------------
      # Inventory diagnostics
      # ------------------------------------------------------------------
      #   max_abs_inv   — worst-case inventory exposure during the episode
      #   mean_abs_inv  — average absolute inventory (lower = tighter control)
      #   pct_at_limit  — fraction of steps where |inv| >= inv_limit (breach %)
      # ------------------------------------------------------------------
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

      # Discounted return: G_0 = Σ_{t=0}^{T-1} γ^t r_t  (micro-step discount)
      # This is what V(s_0) should converge to under the current policy, making
      # it a direct diagnostic for the C51 support bounds [-2, +2].
      gamma = best_params.get("gamma", 0.995)
      if "MM_Reward" in mm_df.columns:
          rewards = mm_df["MM_Reward"].values.astype(float)
          gammas = gamma ** np.arange(len(rewards))
          discounted_return = float(np.dot(gammas, rewards))
      else:
          discounted_return = 0.0

      # --------------------------------------------------
      # NEW: RL STATE SNAPSHOT (FIRST/LAST) + FILL COUNTS
      #      Only computed and printed in verbose mode.
      # --------------------------------------------------
      try:
          first_row = mm_df.iloc[0]
          last_row = mm_df.iloc[-1]

          # --------------------------------------------------------
          # 1) READ RL STATE METADATA (mode, dim, max_offset)
          # --------------------------------------------------------
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

          # --------------------------------------------------------
          # 2) PURE MM MODE → DECODE NN INPUT VECTOR
          # --------------------------------------------------------
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
              # ----------------------------------------------------
              # 3) GENERIC MODE → L1 VIEW
              # ----------------------------------------------------
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

          # --------------------------------------------------------
          # 4) ALWAYS: FILL STATS
          # --------------------------------------------------------
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

      # --------------------------------------------------
      # RL POLICY-LEVEL STATS (what the network chose)
      # --------------------------------------------------
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

      # =================================================================
      # EPSILON DECAY — done ONCE per episode, OUTSIDE the environment
      # =================================================================
      # We decay epsilon BETWEEN episodes (not inside learn()) to ensure
      # the exploration rate stays constant throughout a single episode.
      # This is important because:
      #   (a) Within an episode, the agent commits to a single ε-greedy
      #       policy.  Changing ε mid-episode breaks the stationarity
      #       assumption for the on-policy SARSA variant.
      #   (b) It makes episode-level statistics comparable — every step
      #       within the same episode used the same exploration rate.
      #
      # When NoisyNets are enabled (use_noisy_net=True), epsilon is
      # IRRELEVANT — exploration is driven by learned parameter noise.
      # =================================================================
      old_eps = deep_controller.epsilon

      _mm_ep_for_decay = ep + 1
      _mm_total_for_decay = N_EPISODES

      if True:  # epsilon decay always active (even with NoisyNet)
          # ---------------------------------------------------------
          # EPSILON DECAY — two strategies selectable via best_params
          # ---------------------------------------------------------
          # "linear"      : ε(ep) = ε_start − progress × (ε_start − ε_min)
          #                 Straight line from ε_start → ε_min over N episodes.
          #                 Keeps exploration HIGH for longer — recommended when
          #                 the agent is prone to early policy collapse.
          #
          # "exponential" : ε(ep) = max(ε_min, ε_start × decay^(ep+1))
          #                 Multiplicative per-episode factor read from
          #                 best_params["epsilon_decay"].  Drops faster in the
          #                 first episodes — useful when the environment is
          #                 simple and the agent converges quickly.
          # ---------------------------------------------------------
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

      # -----------------------------------------------------------------
      # MOVING AVERAGE — smoothed reward trend for TensorBoard
      # -----------------------------------------------------------------
      # Raw episode rewards are noisy; the moving averages below provide a
      # smoother read of recent training quality at short / medium / long
      # horizons.
      reward_ma_window = 10
      if len(episode_stats) >= reward_ma_window:
          last_rewards = [e["total_reward"] for e in episode_stats[-reward_ma_window:]]
          ma_10 = float(np.mean(last_rewards))
          deep_controller.writer.add_scalar("episode/total_reward_ma_10", ma_10, ep)

      # PnL moving averages used for console monitoring and TensorBoard.
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

      # -----------------------------------------------------------------
      # INVENTORY DIAGNOSTICS — TensorBoard scalars
      # -----------------------------------------------------------------
      deep_controller.writer.add_scalar("episode/max_abs_inventory", max_abs_inv, ep)
      deep_controller.writer.add_scalar("episode/mean_abs_inventory", mean_abs_inv, ep)
      deep_controller.writer.add_scalar("episode/pct_at_inv_limit", pct_at_limit, ep)

      # -----------------------------------------------------------------
      # LEARNING RATE SCHEDULE — configurable via best_params["lr_decay_type"]
      # -----------------------------------------------------------------
      # We compute the LR that should be used AFTER finishing episode `ep`,
      # i.e. during episode `ep + 1`.
      #
      #   "exponential" : lr(ep) = LR_START × (LR_END_FACTOR ** progress)
      #       Smooth log-linear ramp.  Large updates early, smaller updates late.
      #
      #   "linear"      : lr(ep) = (1 − progress) × LR_START + progress × LR_END
      #       Straight-line ramp.  Gentler initial drop than exponential.
      #
      # A decaying LR is critical in RL: early on, large updates push the
      # Q-network toward the correct basin; later, small updates refine
      # without oscillation.
      # -----------------------------------------------------------------
      lr_decay_type = best_params.get("lr_decay_type", "exponential")
      use_exp_lr = (lr_decay_type == "exponential")

      if hasattr(deep_controller, "optimizer"):
          new_lr = get_lr_for_episode(_mm_ep_for_decay, _mm_total_for_decay, use_exp=use_exp_lr)

          # ── Cyclical LR modulation ──────────────────────────────────
          # When USE_CYCLICAL_LR is active, the monotonically-decayed LR
          # (new_lr) is multiplied by a cosine warm-restart factor that
          # oscillates between 1.0 (cycle start) and CYCLICAL_LR_MIN_FACTOR
          # (cycle trough).
          #
          # The combined effect is a DECAYING ENVELOPE with periodic warm
          # restarts — the restarts get progressively smaller because the
          # base LR is shrinking.  This is gentler than pure cyclical LR
          # (no monotonic decay) which can cause late-training instability.
          #
          # Example (period=200, min_factor=0.1, LR decaying 3e-4 → 6e-5):
          #   ep=0:   lr = 3e-4 × 1.0 = 3e-4   (restart peak)
          #   ep=100: lr = 2.5e-4 × 0.55 = 1.4e-4 (mid-cycle)
          #   ep=200: lr = 2e-4 × 1.0 = 2e-4   (restart peak, but lower base)
          #   ep=300: lr = 1.5e-4 × 0.55 = 8e-5 (mid-cycle, smaller)
          _lr_cycle_mult = 1.0
          if USE_CYCLICAL_LR:
              _lr_cycle_mult = get_cyclical_lr_multiplier(
                  ep, period=CYCLICAL_LR_PERIOD, min_factor=CYCLICAL_LR_MIN_FACTOR)
              new_lr *= _lr_cycle_mult

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

      # -----------------------------------------------------------------
      # Periodic checkpoint saving (every 50 episodes)
      # -----------------------------------------------------------------
      # Saves an intermediate checkpoint so that multi-hour training runs
      # can be recovered after crashes or interruptions.  The final
      # checkpoint below overwrites the "best" slot after training ends.
      # -----------------------------------------------------------------
      CKPT_INTERVAL = 50
      if (ep + 1) % CKPT_INTERVAL == 0:
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
              "BUY_MO_PROB": 0.5,
              "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
              "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
              "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
          }
          save_deep_rl_checkpoint(
              deep_controller, os.path.join(_ckpt_dir, _ckpt_name), _ckpt_meta
          )
          print(f"[CHECKPOINT] Saved checkpoint (ep {ep+1}): {_ckpt_name}")

      # -----------------------------------------------------------------
      # BEST-CHECKPOINT update (single rolling "best so far" slot)
      # -----------------------------------------------------------------
      # Save whenever MA(BEST_CKPT_WINDOW) of final_pnl improves.
      if not np.isnan(pnl_ma_best) and pnl_ma_best > _best_ckpt_ma:
          _best_ckpt_ma = pnl_ma_best
          _best_ckpt_ep = ep
          _ckpt_dir = "checkpoints"
          os.makedirs(_ckpt_dir, exist_ok=True)
          _train_suffix = _build_train_suffix()
          # Intra-decision γ aggregation tag — keeps gevent/gflat ablation
          # checkpoints from overwriting each other (and the legacy file
          # without the suffix).
          _intra_event_suffix = "_gevent" if USE_INTRA_EVENT_GAMMA else "_gflat"
          _best_ckpt_name = (
              f"deep_mm_mtm_{'pure' if USE_PURE_MM else 'generic'}_"
              f"invp{INV_PENALTY_COEFF:.4f}{_train_suffix}{_intra_event_suffix}_best_ma{BEST_CKPT_WINDOW}.pt"
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
              "BUY_MO_PROB": 0.5,
              "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
              "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
              "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
          }
          save_deep_rl_checkpoint(deep_controller, _best_ckpt_path, _best_ckpt_meta)
          print(
              f"[CHECKPOINT] Updated BEST CHECKPOINT (criterion=MA{BEST_CKPT_WINDOW}) "
              f"(ep {ep+1}, MA{BEST_CKPT_WINDOW}={_best_ckpt_ma:+.4f}): {_best_ckpt_name}"
          )

      # ── Early stopping check (LAST thing in the episode) ────────────
      # Placed AFTER all logging, TB scalars, checkpoints, LR updates,
      # so the last episode is fully complete.
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

  # Training loop ended — either completed all episodes or early-stopped.
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

  # Final checkpoint (same file — last periodic save is already up-to-date,
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
      "BUY_MO_PROB": 0.5,
      "USE_INVENTORY_WALL": bool(USE_INVENTORY_WALL),
      "USE_TERMINAL_BONUS": bool(USE_TERMINAL_BONUS),
      "USE_DAMPENED_REWARD": bool(USE_DAMPENED_REWARD),
  }
  save_deep_rl_checkpoint(deep_controller, CKPT_PATH, meta)
  if SAVE_FINAL_REPLAY_BUFFER:
      _replay_save_path = CKPT_PATH.replace(".pt", "_replay.pt")
      _n_saved = deep_controller.save_replay_buffer(_replay_save_path)
      print(f"[CHECKPOINT] Saved replay buffer ({_n_saved} transitions) to {_replay_save_path}")
  else:
      print("[CHECKPOINT] Skipped final replay buffer save (SAVE_FINAL_REPLAY_BUFFER=False)")

  # =====================================================================
  # POST-TRAINING: ANIMATION + DATA PERSISTENCE (last episode only)
  # =====================================================================
  # After the training loop completes, we generate a visual animation of
  # the LOB + MM overlay for the LAST episode.  This lets us visually
  # inspect how the trained policy manages quotes, fills, and inventory.
  #
  # We also persist the DataFrames (msg_df, ob_df, mm_df) as pickle files
  # so they can be loaded offline for further analysis without re-running
  # the full training loop.
  # =====================================================================
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
  plt.show()

  # Persist last episode DataFrames for offline analysis
  mm_df.to_pickle("mm_sim_df")
  ob_df.to_pickle("ob_sim_df")
  msg_df.to_pickle("msg_sim_df")
    
    
  # --------------------------------------------------
  # 7) Plot total reward per episode + polynomial trend
  # --------------------------------------------------
  episodes = np.array([d["episode"] for d in episode_stats])
  total_rewards = np.array([d["total_reward"] for d in episode_stats])


  def best_poly_degree(x, y, max_degree=8):
      """
      Automatically choose the best polynomial degree (1..max_degree)
      using AIC as the selection criterion.
      """
      best_deg = None
      best_aic = np.inf
      best_coeffs = None

      n = len(x)

      for deg in range(1, max_degree + 1):
          coeffs = np.polyfit(x, y, deg)
          p = np.poly1d(coeffs)
          y_pred = p(x)
          resid = y - y_pred
          sse = np.sum(resid ** 2)

          k = deg + 1  # number of parameters
          eps = 1e-12  # avoid log(0)

          aic = n * np.log(sse / n + eps) + 2 * k

          if aic < best_aic:
              best_aic = aic
              best_deg = deg
              best_coeffs = coeffs

      return best_deg, best_coeffs, best_aic


  best_deg, best_coeffs, best_aic = best_poly_degree(
      episodes,
      total_rewards,
      max_degree=8,
  )
  poly = np.poly1d(best_coeffs)

  print(f"\nBest polynomial degree = {best_deg}, AIC = {best_aic:.2f}")

  plt.figure(figsize=(10, 5))
  plt.plot(episodes, total_rewards, marker="o", label="Total Reward")

  # Smooth curve using the best polynomial
  x_smooth = np.linspace(episodes.min(), episodes.max(), 500)
  y_smooth = poly(x_smooth)

  plt.plot(
      x_smooth,
      y_smooth,
      "r-",
      linewidth=2,
      label="Best poly nomial fit",
  )

  plt.xlabel("Episode")
  plt.ylabel("Total Reward")
  # plt.title("Total Reward per Episode")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.show()

  # --------------------------------------------------
  # 8) Plot MM inventory for the last played episode
  # --------------------------------------------------

  plt.plot(mm_df["MM_Inventory"])
  # plt.title("MM Inventory")
  plt.grid(True, alpha=0.3)
  plt.show()

  # --------------------------------------------------
  # 9) Plot MM Total PnL for the last played episode
  # --------------------------------------------------

  plt.plot(mm_df["MM_TotalPnL"])
  # plt.title("MM Total PnL")
  plt.grid(True, alpha=0.3)
  plt.show()

  # --------------------------------------------------
  # 10) Plot MM UPnL for the last played episode
  # --------------------------------------------------

  plt.plot(mm_df["MM_UPnL"])
  plt.title("MM Unrealized PnL")
  plt.grid(True, alpha=0.3)
  plt.show()

  # --------------------------------------------------
  # 11) Plot MM Cash PnL for the last played episode
  # --------------------------------------------------

  plt.plot(mm_df["MM_CashPnL"])
  plt.title("MM Cash PnL")
  plt.grid(True, alpha=0.3)
  plt.show()


  # --------------------------------------------------
  # 12) Interpretability Studies - Optimal Policy
  # --------------------------------------------------

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


  def _decode_pure_mm_network_vector(
      vec,
      k_offset: int,
      deep_controller,
  ) -> Dict[str, Any]:
      """
      Convert the stored PURE-MM NN input vector back to a raw MM state.

      `MM_RL_State_Vector` stores the tensor after `_state_to_tensor()`, so
      spread/depths are log1p-scaled and inventory is divided by inv_limit.
      The interpretability helper expects raw state values because it calls
      `_state_to_tensor()` again before evaluating the network.
      """
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

      spread_raw = _inv_log1p(vec[0])
      inventory_raw = float(vec[1]) * inv_denom
      bid_vec_raw = [_inv_log1p(x) for x in vec[2 : 2 + depth_len]]
      ask_start = 2 + depth_len
      ask_vec_raw = [_inv_log1p(x) for x in vec[ask_start : ask_start + depth_len]]

      state = {
          "spread": spread_raw,
          "inventory": inventory_raw,
          "pure_mm_bid_sizes": bid_vec_raw,
          "pure_mm_ask_sizes": ask_vec_raw,
      }

      cursor = base_len
      if getattr(deep_controller, "use_flow_signal", False) and cursor < len(vec):
          state["mo_flow_p_hat"] = float(vec[cursor])
          cursor += 1
      if getattr(deep_controller, "use_fast_flow_signal", False) and cursor < len(vec):
          state["mo_flow_fast_p_hat"] = float(vec[cursor])
          cursor += 1
      if getattr(deep_controller, "use_fill_imbalance", False) and cursor < len(vec):
          state["fill_imbalance_ewma"] = float(vec[cursor])

      return state


  def plot_cost_to_go_interpretability(
      deep_controller,
      base_state: Dict[str, Any],
      # Grids for generic mode (pure_MM == False)
      inv_values=np.arange(-10, 11, 1),
      spread_values=np.arange(1, 7, 1),
      ask_values=np.arange(1, 21, 1),
      bid_values=np.arange(1, 21, 1),
      # Grid of queue sizes for pure-MM offsets
      offset_values: Optional[np.ndarray] = None,
      as_cost: bool = True,
  ):
      """
      Interpretability plots for the learned value/cost function.

      The function automatically adapts to the type of state and controller:

      Case 1: PURE-MM (state has 'pure_mm_bid_sizes' and 'pure_mm_ask_sizes')
          - Plots 3D surfaces based only on the features that the RL actually uses:
              1) inventory × bid_size_offset_0
              2) inventory × ask_size_offset_0
              3) bid_size_offset_0 × bid_size_offset_1   (inventory fixed)
              4) ask_size_offset_0 × ask_size_offset_1   (inventory fixed)
              5) bid_size_offset_0 × ask_size_offset_0   (inventory fixed)
              6) inventory × spread (NEW)

      Case 2: GENERIC (pure_MM == False; no pure_mm_* in state)
          - Plots 3D surfaces in terms of:
              1) inventory × spread
              2) inventory × ask_size
              3) inventory × bid_size
              4) bid_size × ask_size     (inventory fixed)
              5) bid_size × spread       (inventory fixed)
              6) ask_size × spread       (inventory fixed)

      In both cases:
        - All other state fields in base_state are kept untouched.
        - State encoding is exactly the same as used in training,
          via DeepRLController._state_to_tensor().
      """

      q_net = deep_controller.q_net
      q_net.eval()

      # ------------------------------------------------------------------
      # Helper: evaluate V(s) or J(s) for a single state dict
      # ------------------------------------------------------------------
      # IMPORTANT: We use `_q_values_from_net` instead of calling q_net()
      # directly because the network architecture depends on whether
      # Distributional DQN (C51) is enabled:
      #
      #   Standard DQN:  q_net(s) → (1, n_actions)  — scalar Q per action
      #   C51:           q_net(s) → (1, n_actions, atoms) — probability
      #                  distribution over atoms per action.  We need the
      #                  expected value E[Z] = Σ p_i · z_i to get a scalar.
      #
      # `_q_values_from_net` handles both cases transparently, always
      # returning (B, n_actions) scalar Q-values suitable for argmax/max.
      # ------------------------------------------------------------------
      def evaluate_state(mm_state: Dict[str, Any]) -> float:
          with torch.no_grad():
              s_tensor = deep_controller._state_to_tensor(mm_state)
              q_values = deep_controller._q_values_from_net(q_net, s_tensor)  # (1, n_actions)
              v = torch.max(q_values, dim=1).values.item()
          return -v if as_cost else v

      # ------------------------------------------------------------------
      # Helper: 3D surface plot
      # ------------------------------------------------------------------
      def plot_surface(X, Y, Z, xlabel: str, ylabel: str, title_suffix: str):
          fig = plt.figure(figsize=(10, 7))
          ax = fig.add_subplot(111, projection="3d")

          surf = ax.plot_surface(
              X, Y, Z,
              cmap="viridis",
              edgecolor="none",
              alpha=0.9,
          )

          cbar_label = "Cost-to-go J(s)" if as_cost else "Value V(s)"
          fig.colorbar(surf, shrink=0.5, aspect=10, label=cbar_label)

          ax.set_xlabel(xlabel)
          ax.set_ylabel(ylabel)

          if as_cost:
              ax.set_zlabel("J(s) = -max_a Q(s,a)")
              ax.set_title(f"Cost-to-go J(s) = -max_a Q(s,a)\n{title_suffix}")
          else:
              ax.set_zlabel("V(s) = max_a Q(s,a)")
              ax.set_title(f"Value V(s) = max_a Q(s,a)\n{title_suffix}")

          ax.view_init(elev=30, azim=230)
          plt.tight_layout()
          plt.show()

      # ======================================================
      # CASE 1: PURE-MM (queue offsets)
      # ======================================================
      is_pure_mm_state = (
          deep_controller.pure_mm and
          ("pure_mm_bid_sizes" in base_state) and
          ("pure_mm_ask_sizes" in base_state)
      )

      if is_pure_mm_state:
          base_bid_sizes = np.array(base_state["pure_mm_bid_sizes"], dtype=float)
          base_ask_sizes = np.array(base_state["pure_mm_ask_sizes"], dtype=float)

          # Default offset grid if not provided
          if offset_values is None:
              base0 = 0.0
              if len(base_bid_sizes) > 0:
                  base0 = max(base0, base_bid_sizes[0])
              if len(base_ask_sizes) > 0:
                  base0 = max(base0, base_ask_sizes[0])
              vmax = max(5.0, base0 * 2.0)
              offset_values = np.linspace(0.0, vmax, num=21)
          offset_values = np.asarray(offset_values, dtype=float)

          # --------------- 1) inventory × bid_size_offset_0 ---------------
          if len(base_bid_sizes) > 0:
              n_inv = len(inv_values)
              n_off = len(offset_values)
              value_map = np.zeros((n_inv, n_off), dtype=np.float32)

              for i, inv in enumerate(inv_values):
                  for j, q0 in enumerate(offset_values):
                      mm_state = dict(base_state)

                      bid_sizes = np.array(base_bid_sizes, copy=True)
                      bid_sizes[0] = float(q0)

                      mm_state["inventory"] = float(inv)
                      mm_state["pure_mm_bid_sizes"] = bid_sizes.tolist()

                      value_map[i, j] = evaluate_state(mm_state)

              X, Y = np.meshgrid(offset_values, inv_values)  # X: q0, Y: inventory
              plot_surface(
                  X, Y, value_map,
                  xlabel="Bid size offset_0",
                  ylabel="Inventory",
                  title_suffix="inventory × bid_size_offset_0 (pure-MM)",
              )

          # --------------- 2) inventory × ask_size_offset_0 ---------------
          if len(base_ask_sizes) > 0:
              n_inv = len(inv_values)
              n_off = len(offset_values)
              value_map = np.zeros((n_inv, n_off), dtype=np.float32)

              for i, inv in enumerate(inv_values):
                  for j, q0 in enumerate(offset_values):
                      mm_state = dict(base_state)

                      ask_sizes = np.array(base_ask_sizes, copy=True)
                      ask_sizes[0] = float(q0)

                      mm_state["inventory"] = float(inv)
                      mm_state["pure_mm_ask_sizes"] = ask_sizes.tolist()

                      value_map[i, j] = evaluate_state(mm_state)

              X, Y = np.meshgrid(offset_values, inv_values)  # X: q0, Y: inventory
              plot_surface(
                  X, Y, value_map,
                  xlabel="Ask size offset_0",
                  ylabel="Inventory",
                  title_suffix="inventory × ask_size_offset_0 (pure-MM)",
              )

          # Fixed inventory for offset × offset maps
          inv_fixed = float(base_state.get("inventory", 0.0))

          # --------------- 3) bid_size_offset_0 × ask_size_offset_0 -------
          if (len(base_bid_sizes) > 0) and (len(base_ask_sizes) > 0):
              n1 = len(offset_values)
              n2 = len(offset_values)
              value_map = np.zeros((n1, n2), dtype=np.float32)

              for i, qb in enumerate(offset_values):
                  for j, qa in enumerate(offset_values):
                      mm_state = dict(base_state)

                      bid_sizes = np.array(base_bid_sizes, copy=True)
                      ask_sizes = np.array(base_ask_sizes, copy=True)
                      bid_sizes[0] = float(qb)
                      ask_sizes[0] = float(qa)

                      mm_state["inventory"] = inv_fixed
                      mm_state["pure_mm_bid_sizes"] = bid_sizes.tolist()
                      mm_state["pure_mm_ask_sizes"] = ask_sizes.tolist()

                      value_map[i, j] = evaluate_state(mm_state)

              X, Y = np.meshgrid(offset_values, offset_values, indexing="ij")
              plot_surface(
                  X, Y, value_map,
                  xlabel="Bid size offset_0",
                  ylabel="Ask size offset_0",
                  title_suffix="bid_size_offset_0 × ask_size_offset_0 (pure-MM, fixed inventory)",
              )

          # --------------- 4) inventory × spread (PURE-MM) -----------------
          # Use a spread grid around the base spread (or a reasonable range)
          base_spread = float(base_state.get("spread", 1.0))
          # Small band around the base spread; you can tune this if needed
          spread_min = max(1.0, base_spread - 3)
          spread_max = base_spread + 3
          spread_grid = np.arange(spread_min, spread_max + 1, 1)

          n_inv = len(inv_values)
          n_sp = len(spread_grid)
          value_map = np.zeros((n_inv, n_sp), dtype=np.float32)

          for i, inv in enumerate(inv_values):
              for j, sp in enumerate(spread_grid):
                  mm_state = dict(base_state)
                  mm_state["inventory"] = float(inv)
                  mm_state["spread"] = float(sp)
                  value_map[i, j] = evaluate_state(mm_state)

          X, Y = np.meshgrid(spread_grid, inv_values)  # X: spread, Y: inventory
          plot_surface(
              X, Y, value_map,
              xlabel="Spread (ticks)",
              ylabel="Inventory",
              title_suffix="inventory × spread (pure-MM)",
          )

          return  # end of pure-MM case

      # ======================================================
      # CASE 2: GENERIC (pure_MM == False)
      # ======================================================

      fixed_spread = float(base_state.get("spread", 1.0))
      fixed_ask_size = float(base_state.get("asksize", 1.0))
      fixed_bid_size = float(base_state.get("bidsize", 1.0))
      fixed_inventory = float(base_state.get("inventory", 0.0))

      # ---------- value map for (inventory, one feature) ----------
      def compute_value_map_inventory_feature(inv_vals, other_vals, feature_name: str):
          n_inv = len(inv_vals)
          n_other = len(other_vals)
          value_map = np.zeros((n_inv, n_other), dtype=np.float32)

          for i, inv in enumerate(inv_vals):
              for j, other in enumerate(other_vals):

                  mm_state = dict(base_state)

                  spread = fixed_spread
                  ask_size = fixed_ask_size
                  bid_size = fixed_bid_size

                  if feature_name == "spread":
                      spread = float(other)
                  elif feature_name == "ask":
                      ask_size = float(other)
                  elif feature_name == "bid":
                      bid_size = float(other)
                  else:
                      raise ValueError(f"Unknown feature_name: {feature_name}")

                  mm_state["spread"] = spread
                  mm_state["asksize"] = ask_size
                  mm_state["bidsize"] = bid_size
                  mm_state["inventory"] = float(inv)

                  value_map[i, j] = evaluate_state(mm_state)

          return value_map

      # ---------- value map for (feature1, feature2), fixed inventory ----------
      def compute_value_map_feature_feature(vals1, vals2, feature1_name: str, feature2_name: str):
          n1 = len(vals1)
          n2 = len(vals2)
          value_map = np.zeros((n1, n2), dtype=np.float32)

          for i, v1 in enumerate(vals1):
              for j, v2 in enumerate(vals2):

                  mm_state = dict(base_state)

                  spread = fixed_spread
                  ask_size = fixed_ask_size
                  bid_size = fixed_bid_size

                  if feature1_name == "spread":
                      spread = float(v1)
                  elif feature1_name == "ask":
                      ask_size = float(v1)
                  elif feature1_name == "bid":
                      bid_size = float(v1)
                  else:
                      raise ValueError(f"Unknown feature1_name: {feature1_name}")

                  if feature2_name == "spread":
                      spread = float(v2)
                  elif feature2_name == "ask":
                      ask_size = float(v2)
                  elif feature2_name == "bid":
                      bid_size = float(v2)
                  else:
                      raise ValueError(f"Unknown feature2_name: {feature2_name}")

                  mm_state["spread"] = spread
                  mm_state["asksize"] = ask_size
                  mm_state["bidsize"] = bid_size
                  mm_state["inventory"] = fixed_inventory

                  value_map[i, j] = evaluate_state(mm_state)

          return value_map

      # ------------------ 1) inventory × spread ------------------
      value_map_spread = compute_value_map_inventory_feature(
          inv_values, spread_values, feature_name="spread"
      )
      X, Y = np.meshgrid(spread_values, inv_values)  # X: spread, Y: inventory
      plot_surface(
          X, Y, value_map_spread,
          xlabel="Spread (ticks)",
          ylabel="Inventory",
          title_suffix="inventory × spread (generic mode)",
      )

      # ------------------ 2) inventory × ask_size ----------------
      value_map_ask = compute_value_map_inventory_feature(
          inv_values, ask_values, feature_name="ask"
      )
      X, Y = np.meshgrid(ask_values, inv_values)  # X: ask_size, Y: inventory
      plot_surface(
          X, Y, value_map_ask,
          xlabel="Ask size",
          ylabel="Inventory",
          title_suffix="inventory × ask_size (generic mode)",
      )

      # ------------------ 3) inventory × bid_size ----------------
      value_map_bid = compute_value_map_inventory_feature(
          inv_values, bid_values, feature_name="bid"
      )
      X, Y = np.meshgrid(bid_values, inv_values)  # X: bid_size, Y: inventory
      plot_surface(
          X, Y, value_map_bid,
          xlabel="Bid size",
          ylabel="Inventory",
          title_suffix="inventory × bid_size (generic mode)",
      )

      # ------------------ 4) bid_size × ask_size -----------------
      value_map_bid_ask = compute_value_map_feature_feature(
          bid_values, ask_values, feature1_name="bid", feature2_name="ask"
      )
      X, Y = np.meshgrid(bid_values, ask_values, indexing="ij")
      plot_surface(
          X, Y, value_map_bid_ask,
          xlabel="Bid size",
          ylabel="Ask size",
          title_suffix="bid_size × ask_size (generic mode, fixed inventory)",
      )

      # ------------------ 5) bid_size × spread -------------------
      value_map_bid_spread = compute_value_map_feature_feature(
          bid_values, spread_values, feature1_name="bid", feature2_name="spread"
      )
      X, Y = np.meshgrid(bid_values, spread_values, indexing="ij")
      plot_surface(
          X, Y, value_map_bid_spread,
          xlabel="Bid size",
          ylabel="Spread (ticks)",
          title_suffix="bid_size × spread (generic mode, fixed inventory)",
      )

      # ------------------ 6) ask_size × spread -------------------
      value_map_ask_spread = compute_value_map_feature_feature(
          ask_values, spread_values, feature1_name="ask", feature2_name="spread"
      )
      X, Y = np.meshgrid(ask_values, spread_values, indexing="ij")
      plot_surface(
          X, Y, value_map_ask_spread,
          xlabel="Ask size",
          ylabel="Spread (ticks)",
          title_suffix="ask_size × spread (generic mode, fixed inventory)",
      )


  # --------------------------------------------------
  # 12) Interpretability Studies - Optimal Policy
  # --------------------------------------------------

  try:
      # Use the LAST row of the last episode as anchor state
      last_row = mm_df.iloc[-1]

      # ======================================================
      # PURE-MM MODE
      # ======================================================
      if deep_controller.pure_mm:
          rl_mode = last_row.get("MM_RL_State_Mode", None)
          rl_dim = last_row.get("MM_RL_State_Dim", None)
          rl_max_off = last_row.get("MM_RL_State_MaxOffset", None)
          rl_vec = last_row.get("MM_RL_State_Vector", None)

          if (rl_mode == "pure_mm") and isinstance(rl_vec, (list, np.ndarray)):
              # Decode K from last state's metadata (fallback from rl_dim if needed)
              k_from_meta = _finite_int_or_none(rl_max_off)
              if k_from_meta is not None:
                  K = k_from_meta
              else:
                  dim_from_meta = _finite_int_or_none(rl_dim)
                  L = dim_from_meta if dim_from_meta is not None else len(rl_vec)
                  K = int(getattr(deep_controller, "max_offset", max(0, (L - 2) // 2 - 1)))

              base_state = _decode_pure_mm_network_vector(
                  rl_vec,
                  K,
                  deep_controller,
              )

              print("\n[Interpretability] PURE-MM base_state used for plots:")
              print("  (decoded from normalized MM_RL_State_Vector)")
              print("  spread:", base_state["spread"])
              print("  inventory:", base_state["inventory"])
              print("  bid_sizes:", base_state["pure_mm_bid_sizes"])
              print("  ask_sizes:", base_state["pure_mm_ask_sizes"])

              # ----------------------------------------
              # 1) Inventory grid (based on inv_limit or episode stats)
              # ----------------------------------------
              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  if "MM_Inventory" in mm_df.columns:
                      inv_min = int(np.floor(mm_df["MM_Inventory"].min()))
                      inv_max = int(np.ceil(mm_df["MM_Inventory"].max()))
                      if inv_min == inv_max:
                          inv_min -= 1
                          inv_max += 1
                  else:
                      inv_min, inv_max = -20, 20
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              # ----------------------------------------
              # 2) Dynamic queue-size grid for offset_0
              #    (based on all PURE-MM RL_State_Vectors)
              # ----------------------------------------
              offset_values_dyn = None
              if (
                  "MM_RL_State_Vector" in mm_df.columns
                  and "MM_RL_State_Mode" in mm_df.columns
              ):
                  rl_vec_col = mm_df["MM_RL_State_Vector"]
                  rl_mode_col = mm_df["MM_RL_State_Mode"]

                  q0_all = []

                  for vec, mode in zip(rl_vec_col, rl_mode_col):
                      if mode != "pure_mm":
                          continue
                      if vec is None:
                          continue
                      if not isinstance(vec, (list, np.ndarray)):
                          continue

                      vec_list = list(vec)
                      L = len(vec_list)
                      if L < 4:
                          continue

                      # Vector is already normalized NN input; decode it to
                      # raw queue volumes before building the plot grid.
                      K_row = int(getattr(deep_controller, "max_offset", K))
                      base_len = 2 + 2 * (K_row + 1)
                      if K_row < 0 or L < base_len:
                          continue

                      try:
                          decoded_row = _decode_pure_mm_network_vector(
                              vec_list,
                              K_row,
                              deep_controller,
                          )
                      except (TypeError, ValueError):
                          continue

                      q_bid0 = float(decoded_row["pure_mm_bid_sizes"][0])
                      q_ask0 = float(decoded_row["pure_mm_ask_sizes"][0])

                      q0_all.append(q_bid0)
                      q0_all.append(q_ask0)

                  if len(q0_all) > 0:
                      q0_all = np.array(q0_all, dtype=float)

                      # Use percentiles to avoid extreme outliers
                      q_min = max(0.0, float(np.percentile(q0_all, 5)))
                      q_max = float(np.percentile(q0_all, 95))
                      if q_max <= q_min:
                          q_max = q_min + 1.0

                      offset_values_dyn = np.linspace(q_min, q_max, num=31)

              # Fallback: local heuristic around base_state if episode stats are missing
              if offset_values_dyn is None:
                  base_bid_sizes = base_state["pure_mm_bid_sizes"]
                  base_ask_sizes = base_state["pure_mm_ask_sizes"]
                  q0_candidates = []
                  if len(base_bid_sizes) > 0:
                      q0_candidates.append(float(base_bid_sizes[0]))
                  if len(base_ask_sizes) > 0:
                      q0_candidates.append(float(base_ask_sizes[0]))
                  if len(q0_candidates) > 0:
                      base0 = max(q0_candidates)
                  else:
                      base0 = 1.0
                  vmax = max(5.0, 2.0 * base0)
                  offset_values_dyn = np.linspace(0.0, vmax, num=31)

              print("\n[Interpretability] PURE-MM grids:")
              print("  inv_values range:", inv_values[0], "→", inv_values[-1])
              print("  offset_values range:", offset_values_dyn[0], "→", offset_values_dyn[-1])

              # ----------------------------------------
              # Call interpretability plot for PURE-MM
              # ----------------------------------------
              plot_cost_to_go_interpretability(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  offset_values=offset_values_dyn,
                  as_cost=False,
              )
          else:
              print(
                  "\n[Interpretability WARNING] Could not reconstruct PURE-MM base_state "
                  "(missing RL_State_* info). Skipping interpretability plots."
              )

      # ======================================================
      # GENERIC MODE
      # ======================================================
      else:
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
                  "spread": float(last_row["MM_State_Spread"]),
                  "asksize": float(last_row["MM_State_AskSize"]),
                  "bidsize": float(last_row["MM_State_BidSize"]),
                  "inventory": float(last_row["MM_State_Inventory"]),
                  "has_bid": bool(last_row["MM_State_HasBid"]),
                  "has_ask": bool(last_row["MM_State_HasAsk"]),
              }

              print("\n[Interpretability] GENERIC base_state used for plots:")
              print(base_state)

              # ---------------- Inventory grid ----------------
              if deep_controller.inv_limit is not None:
                  inv_limit = int(deep_controller.inv_limit)
                  inv_values = np.arange(-inv_limit, inv_limit + 1, 1)
              else:
                  inv_col = "MM_State_Inventory"
                  if inv_col in mm_df.columns:
                      inv_min = int(np.floor(mm_df[inv_col].min()))
                      inv_max = int(np.ceil(mm_df[inv_col].max()))
                      if inv_min == inv_max:
                          inv_min -= 1
                          inv_max += 1
                  else:
                      inv_min, inv_max = -20, 20
                  inv_values = np.arange(inv_min, inv_max + 1, 1)

              # ---------------- Spread grid -------------------
              if "MM_State_Spread" in mm_df.columns:
                  s_min = int(np.floor(mm_df["MM_State_Spread"].min()))
                  s_max = int(np.ceil(mm_df["MM_State_Spread"].max()))
              elif "Spread" in msg_df.columns:
                  s_min = int(np.floor(msg_df["Spread"].min()))
                  s_max = int(np.ceil(msg_df["Spread"].max()))
              else:
                  s_min, s_max = 1, 6

              s_min = max(1, s_min)
              if s_min == s_max:
                  s_max = s_min + 5
              spread_values = np.arange(s_min, s_max + 1, 1)

              # ---------------- Ask size grid -----------------
              if "MM_State_AskSize" in mm_df.columns:
                  a_min = max(0.0, float(mm_df["MM_State_AskSize"].min()))
                  a_max = float(mm_df["MM_State_AskSize"].max())
              else:
                  a_min, a_max = 0.0, 20.0
              if a_max <= a_min:
                  a_max = a_min + 1.0
              ask_values = np.linspace(a_min, a_max, num=21)

              # ---------------- Bid size grid -----------------
              if "MM_State_BidSize" in mm_df.columns:
                  b_min = max(0.0, float(mm_df["MM_State_BidSize"].min()))
                  b_max = float(mm_df["MM_State_BidSize"].max())
              else:
                  b_min, b_max = 0.0, 20.0
              if b_max <= b_min:
                  b_max = b_min + 1.0
              bid_values = np.linspace(b_min, b_max, num=21)

              print("\n[Interpretability] GENERIC grids:")
              print("  inv_values range:", inv_values[0], "→", inv_values[-1])
              print("  spread_values range:", spread_values[0], "→", spread_values[-1])
              print("  ask_values range:", ask_values[0], "→", ask_values[-1])
              print("  bid_values range:", bid_values[0], "→", bid_values[-1])

              # Call interpretability plot for GENERIC mode
              plot_cost_to_go_interpretability(
                  deep_controller=deep_controller,
                  base_state=base_state,
                  inv_values=inv_values,
                  spread_values=spread_values,
                  ask_values=ask_values,
                  bid_values=bid_values,
                  as_cost=False,
              )
          else:
              print(
                  "\n[Interpretability WARNING] MM_State_* columns not found in mm_df; "
                  "skipping generic-mode interpretability plots."
              )

  except Exception as e:
      print(f"\n[Interpretability ERROR] Exception while building base_state: {e}")
