"""Archived paper evaluation helpers, extracted without executing notebook cells.

Function bodies match MM_LOB_SIM/regime_switching_diagnostic.ipynb on
2026-10-02; see data/paper/helper_provenance.json. Runtime globals are supplied
by paper_runtime.py, as in the original experiment.
"""
from __future__ import annotations
import random
from typing import Callable
import numpy as np
import torch
from CONFIG_MM import lam, mu, delta, qrm_params, mean_size_LO, mean_size_MO, USE_QRM
from MM_LOB_SIM import simulate_LOB_with_MM
from regime_switching_stress_test import make_regime_schedule
SEED = 42
LONG_SIM_STEPS = 100000
LONG_SIM_STEPS_TO_EQUIL = 10000
REGIME_SEED = 1000041
REGIME_DISTRIBUTION = 'exponential'
REGIME_EXP_TAU = 60
REGIME_EXP_RATE = 1 / 60
REGIME_L_MIN = 10
REGIME_ALPHA = 1.5
REGIME_P_LO = .2
REGIME_P_HI = .8
MO_EVENT_FRACTION = .03488
ctrl_base = None
SIGNAL_BASE = None
RANDOM_TAU_GRID = [15, 30, 60, 120, 240]
CLUSTERED_TAU_GRID = RANDOM_TAU_GRID
CLUSTERED_RHO = .85
CLUSTERED_CHUNK_LEN_MO = None
ORDER_FLOW_INTENSITY_LOW = 0.
ORDER_FLOW_INTENSITY_HIGH = .3


def meta_bool(meta, key, default=False):
    return bool(meta.get(key, default))

def meta_float(meta, key, default):
    value = meta.get(key, default)
    return default if value is None else float(value)

def checkpoint_signal_config(meta):
    """Read every signal-related flag this checkpoint was trained with.

    Phase A checkpoints have no Bayes / quote_exposure entries in their meta;
    the defaults below recover the legacy EWMA-only setup.  Phase B checkpoints
    carry the full set, including the BOCPD mixture parameters and the
    queue-adjusted quote-exposure flag.
    """
    slow = meta_float(meta, 'EWMA_ALPHA', 0.1)
    fast_raw = meta.get('FAST_FLOW_EWMA_ALPHA', None)
    fill_raw = meta.get('FILL_IMBALANCE_ALPHA', None)
    mixture_taus_raw = meta.get('BAYES_MIXTURE_TAUS', None)
    return {'use_flow': meta_bool(meta, 'USE_FLOW_EWMA', False), 'use_fast_flow': meta_bool(meta, 'USE_FAST_FLOW_EWMA', meta_bool(meta, 'USE_MO_FLOW_FAST_EWMA', False)), 'use_fill_imbalance': meta_bool(meta, 'USE_FILL_IMBALANCE', False), 'mo_flow_ewma_alpha': slow, 'mo_flow_fast_ewma_alpha': None if fast_raw is None else float(fast_raw), 'fill_imbalance_alpha': slow if fill_raw is None else float(fill_raw), 'use_bayes': meta_bool(meta, 'USE_BAYES_FLOW_SIGNAL', False), 'bayes_use_mixture': meta_bool(meta, 'USE_MIXTURE', False), 'bayes_mixture_taus': [float(t) for t in mixture_taus_raw] if mixture_taus_raw is not None else None, 'bayes_tau_r': meta_float(meta, 'BAYES_FLOW_TAU_R', 60.0), 'bayes_prior_mode': str(meta.get('BAYES_FLOW_PRIOR_MODE', 'truncated_beta_grid')), 'bayes_cp_window': int(meta.get('BAYES_FLOW_CP_WINDOW', 5)), 'bayes_p_low': meta_float(meta, 'REGIME_P_LO', 0.2), 'bayes_p_high': meta_float(meta, 'REGIME_P_HI', 0.8), 'use_quote_exposure': meta_bool(meta, 'USE_QUOTE_EXPOSURE_IMBALANCE', False)}

def estimate_expected_mo_events(total_events):
    return max(1, int(round(total_events * MO_EVENT_FRACTION)))

def build_regime_schedule(total_events, seed=REGIME_SEED, distribution=None, tau=None, p_lo=None, p_hi=None):
    distribution = REGIME_DISTRIBUTION if distribution is None else distribution
    p_lo = REGIME_P_LO if p_lo is None else p_lo
    p_hi = REGIME_P_HI if p_hi is None else p_hi
    kwargs = dict(seed=seed, n_mo_events=estimate_expected_mo_events(total_events), distribution=distribution, L_min=REGIME_L_MIN, alpha=REGIME_ALPHA, p_lo=p_lo, p_hi=p_hi)
    if distribution == 'exponential':
        tau = REGIME_EXP_TAU if tau is None else tau
        kwargs['exp_rate'] = 1.0 / tau
    return make_regime_schedule(**kwargs)

def build_sim_kwargs(total_events, equil_steps, signal_config=None):
    """Translate a checkpoint-derived `signal_config` into simulator kwargs.

    Phase A: no auxiliary signals -> none of the optional kwargs are passed.
    Phase B: Bayes BOCPD (with mixture over candidate tau) + quote-exposure
    imbalance are forwarded so that the simulator computes the exact features
    the Phase B agent expects in its state vector.
    """
    kw = dict(lam=lam, mu=mu, delta=delta, number_tick_levels=50, n_priority_ranks=100, number_levels_to_store=20, p0=100, mean_size_LO=mean_size_LO, mean_size_MO=mean_size_MO, iterations=total_events, iterations_to_equilibrium=equil_steps, **{'qrm_params': qrm_params} if USE_QRM else {})
    if signal_config is not None:
        kw['mo_flow_ewma_alpha'] = signal_config['mo_flow_ewma_alpha']
        if signal_config.get('mo_flow_fast_ewma_alpha') is not None:
            kw['mo_flow_fast_ewma_alpha'] = signal_config['mo_flow_fast_ewma_alpha']
        if signal_config.get('fill_imbalance_alpha') is not None:
            kw['fill_imbalance_alpha'] = signal_config['fill_imbalance_alpha']
        if signal_config.get('use_bayes'):
            kw['bayes_flow_enabled'] = True
            kw['bayes_flow_tau_r'] = float(signal_config['bayes_tau_r'])
            kw['bayes_flow_p_low'] = float(signal_config['bayes_p_low'])
            kw['bayes_flow_p_high'] = float(signal_config['bayes_p_high'])
            kw['bayes_flow_use_mixture'] = bool(signal_config.get('bayes_use_mixture', False))
            if signal_config.get('bayes_mixture_taus') is not None:
                kw['bayes_flow_mixture_taus'] = [float(t) for t in signal_config['bayes_mixture_taus']]
            kw['bayes_flow_prior_mode'] = str(signal_config.get('bayes_prior_mode', 'truncated_beta_grid'))
            kw['bayes_flow_cp_window'] = int(signal_config.get('bayes_cp_window', 5))
        if signal_config.get('use_quote_exposure'):
            kw['quote_exposure_imbalance_signal'] = True
    return kw

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def reset_eval_controller(controller):
    if controller is None:
        return
    controller.last_action_idx = None
    if hasattr(controller, 'nstep_buffer'):
        controller.nstep_buffer.clear()
    if hasattr(controller, '_smdp_reset'):
        controller._smdp_reset()
    controller._has_acted_once = False
    controller._last_was_decision = False
    controller.moves_tob = 0
    controller.last_env_tob_key = None
    controller.event_steps = 0
    controller.last_update_time = -1.0
    controller.last_inventory = None
    controller.last_mode = None

def run_one_sim(buy_mo_prob, seed=SEED, controller=None, total_events=LONG_SIM_STEPS, equil_steps=LONG_SIM_STEPS_TO_EQUIL, signal_config=None, metrics_only=False, metrics_trace_max_points=None):
    if controller is None:
        controller = ctrl_base
    reset_eval_controller(controller)
    set_seeds(seed)
    msg_df, _, mm_out = simulate_LOB_with_MM(**build_sim_kwargs(total_events, equil_steps, signal_config=signal_config), controller=controller, random_seed=seed, buy_mo_prob=buy_mo_prob, metrics_only=bool(metrics_only), metrics_trace_max_points=metrics_trace_max_points)
    return (msg_df, mm_out)

def make_random_tau_regime_schedule(total_events, seed, tau_grid=RANDOM_TAU_GRID, p_lo=REGIME_P_LO, p_hi=REGIME_P_HI):
    """Callable p_buy schedule: each regime samples tau, length, and p_buy.

        Regime clock is the MO clock.  At the start of each regime:
          1. tau_k is drawn uniformly from tau_grid;
          2. L_k is drawn from Exp(mean=tau_k), rounded up to at least 1 MO;
          3. p_buy,k is drawn uniformly from [p_lo, p_hi].
        """
    rng = np.random.default_rng(int(seed))
    n_mo_events = estimate_expected_mo_events(total_events)
    boundaries = [0]
    p_values = []
    tau_values = []
    lengths = []
    boundaries_arr = np.array(boundaries, dtype=int)
    p_values_arr = np.array(p_values, dtype=float)

    def _append_regime():
        nonlocal boundaries_arr, p_values_arr
        tau = float(rng.choice(tau_grid))
        length = max(1, int(np.ceil(rng.exponential(tau))))
        p_buy = float(rng.uniform(p_lo, p_hi))
        boundaries.append(boundaries[-1] + length)
        lengths.append(length)
        tau_values.append(tau)
        p_values.append(p_buy)
        boundaries_arr = np.array(boundaries, dtype=int)
        p_values_arr = np.array(p_values, dtype=float)
    while boundaries[-1] < n_mo_events + 1000:
        _append_regime()

    def schedule(mo_idx):
        nonlocal boundaries_arr, p_values_arr
        mo_idx = int(mo_idx)
        while mo_idx >= boundaries[-1]:
            _append_regime()
        regime_idx = int(np.searchsorted(boundaries_arr, mo_idx, side='right')) - 1
        regime_idx = min(regime_idx, len(p_values_arr) - 1)
        return float(p_values_arr[regime_idx])
    schedule.boundaries = boundaries
    schedule.p_values = p_values
    schedule.tau_values = tau_values
    schedule.lengths = lengths
    schedule.tau_grid = list(map(float, tau_grid))
    return schedule

def make_clustered_intensity_regime_schedule(total_events, seed, tau_grid=CLUSTERED_TAU_GRID, rho=CLUSTERED_RHO, chunk_len_mo=CLUSTERED_CHUNK_LEN_MO, intensity_low=ORDER_FLOW_INTENSITY_LOW, intensity_high=ORDER_FLOW_INTENSITY_HIGH):
    rng = np.random.default_rng(int(seed))
    n_mo_events = estimate_expected_mo_events(total_events)
    tau_grid = np.asarray(tau_grid, dtype=float)
    intensity_low = float(intensity_low)
    intensity_high = float(intensity_high)
    if chunk_len_mo is None:
        chunk_len_mo_eff = None
    else:
        chunk_len_mo_eff = max(1, int(chunk_len_mo))
    if rho is not None:
        rho = float(rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f'Require rho in [0, 1] or None, got {rho}')
    if not 0.0 <= intensity_low <= intensity_high <= 0.5:
        raise ValueError(f'Require 0 <= intensity_low <= intensity_high <= 0.5, got {intensity_low}, {intensity_high}')
    boundaries = [0]
    tau_values = []
    lengths = []
    sides = []
    p_values = []
    intensity_values = []
    chunk_boundaries = [0]
    chunk_p_values = []
    chunk_intensity_values = []
    chunk_regime_idx = []
    chunk_boundaries_arr = np.asarray(chunk_boundaries, dtype=int)
    chunk_p_values_arr = np.asarray(chunk_p_values, dtype=float)
    current_side = int(rng.choice([-1, 1]))

    def _refresh_arrays():
        nonlocal chunk_boundaries_arr, chunk_p_values_arr
        chunk_boundaries_arr = np.asarray(chunk_boundaries, dtype=int)
        chunk_p_values_arr = np.asarray(chunk_p_values, dtype=float)

    def _sample_p_buy_for_side(side):
        intensity = float(rng.uniform(intensity_low, intensity_high))
        p_buy = float(np.clip(0.5 + int(side) * intensity, 0.0, 1.0))
        return (p_buy, intensity)

    def _append_regime():
        nonlocal current_side
        regime_idx = len(tau_values)
        if regime_idx == 0:
            current_side = int(rng.choice([-1, 1]))
        elif rho is None:
            current_side = int(rng.choice([-1, 1]))
        elif rng.random() >= rho:
            current_side *= -1
        tau = float(rng.choice(tau_grid))
        length = max(1, int(np.ceil(rng.exponential(tau))))
        start = int(boundaries[-1])
        end = start + length
        local_ps = []
        local_intensities = []
        if chunk_len_mo_eff is None:
            p_buy, intensity = _sample_p_buy_for_side(current_side)
            if chunk_boundaries[-1] != start:
                chunk_boundaries.append(start)
            chunk_boundaries.append(end)
            chunk_p_values.append(p_buy)
            chunk_intensity_values.append(intensity)
            chunk_regime_idx.append(regime_idx)
            local_ps.append(p_buy)
            local_intensities.append(intensity)
        else:
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(end, chunk_start + chunk_len_mo_eff)
                p_buy, intensity = _sample_p_buy_for_side(current_side)
                if chunk_boundaries[-1] != chunk_start:
                    chunk_boundaries.append(chunk_start)
                chunk_boundaries.append(chunk_end)
                chunk_p_values.append(p_buy)
                chunk_intensity_values.append(intensity)
                chunk_regime_idx.append(regime_idx)
                local_ps.append(p_buy)
                local_intensities.append(intensity)
                chunk_start = chunk_end
        boundaries.append(end)
        tau_values.append(tau)
        lengths.append(length)
        sides.append(current_side)
        p_values.append(float(np.mean(local_ps)))
        intensity_values.append(float(np.mean(local_intensities)))
        _refresh_arrays()
    while boundaries[-1] < n_mo_events + 1000:
        _append_regime()

    def schedule(mo_idx):
        mo_idx = int(mo_idx)
        while mo_idx >= chunk_boundaries[-1]:
            _append_regime()
        chunk_idx = int(np.searchsorted(chunk_boundaries_arr, mo_idx, side='right')) - 1
        chunk_idx = min(max(chunk_idx, 0), len(chunk_p_values_arr) - 1)
        return float(chunk_p_values_arr[chunk_idx])
    schedule.boundaries = boundaries
    schedule.tau_values = tau_values
    schedule.lengths = lengths
    schedule.sides = sides
    schedule.p_values = p_values
    schedule.intensity_values = intensity_values
    schedule.chunk_boundaries = chunk_boundaries
    schedule.chunk_p_values = chunk_p_values
    schedule.chunk_intensity_values = chunk_intensity_values
    schedule.chunk_regime_idx = chunk_regime_idx
    schedule.tau_grid = list(map(float, tau_grid))
    schedule.rho = rho
    schedule.chunk_len_mo = chunk_len_mo
    schedule.intensity_low = intensity_low
    schedule.intensity_high = intensity_high
    return schedule
