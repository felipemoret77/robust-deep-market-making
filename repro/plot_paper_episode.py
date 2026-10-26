#!/usr/bin/env python3
"""Replot corrected episode traces and the canonical-B signal example.

The two 100,000-event episode figures use the complete corrected evaluations.
The two 5,000-event signal illustrations use a new frozen canonical-B rollout
with seed 123; they do not reconstruct the lost historical training traces.
Use --verify-rerun to rerun that signal example and compare every saved array.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/paper/episode'
CANONICAL_B = '3a4649c860f91ffe2ca3747d071880508e953aa45f5a0d14d6b18cdfcff30200'
COL_BASELINE = '#444444'
COL_REGIME_A = '#D55E00'
COL_REGIME_B = '#0072B2'
BASE_LABEL = 'Algorithm A stationary DQN'
INV_LIMIT = 8
EPISODE_STYLE = {
    'figure.dpi': 120, 'axes.labelsize': 15, 'axes.titlesize': 14,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11,
    'legend.title_fontsize': 11, 'legend.framealpha': .9,
    'legend.fancybox': True, 'legend.borderpad': .3, 'legend.handlelength': 1.6,
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_episode(data, include_phaseb):
    """Three-panel single-episode diagnostic; optionally overlays Phase B."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2.0]},
    )

    steps = np.arange(len(data["A_regime_MM_TotalPnL"]))

    # Panel 1: regime schedule
    ax1.fill_between(
        steps, 0.5, data["p_buy"],
        where=data["p_buy"] > 0.5,
        alpha=0.3, color="green",
        label="buy-heavy (p > 0.5)",
    )
    ax1.fill_between(
        steps, 0.5, data["p_buy"],
        where=data["p_buy"] < 0.5,
        alpha=0.3, color="red",
        label="sell-heavy (p < 0.5)",
    )
    ax1.plot(steps, data["p_buy"], linewidth=0.5, color="steelblue")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6)
    ax1.set_ylabel("P(buy MO)")
    ax1.set_ylim(0.1, 0.9)
    # ax1.set_title("Regime schedule")  # supervisor: no inline titles; caption describes
    ax1.legend(loc="upper right", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Panel 2: cumulative PnL
    pnl_base = data["A_baseline_MM_TotalPnL"]
    pnl_regime = data["A_regime_MM_TotalPnL"]
    ax2.plot(
        np.arange(len(pnl_base)), pnl_base,
        linewidth=1.0, color=COL_BASELINE, alpha=0.85,
        label=f"{BASE_LABEL}: p=0.5",
    )
    ax2.plot(
        steps, pnl_regime,
        linewidth=1.0, color=COL_REGIME_A,
        label=f"{BASE_LABEL}: regime-switching",
    )
    if include_phaseb:
        pnl_phaseb = data["B_regime_MM_TotalPnL"]
        ax2.plot(
            np.arange(len(pnl_phaseb)), pnl_phaseb,
            linewidth=1.0, color=COL_REGIME_B,
            label="Algorithm B: regime-switching",
        )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax2.set_ylabel("Cumulative PnL")
    # (title removed; caption describes)
    ax2.legend(loc="upper left", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Panel 3: inventory
    inv_base = data["A_baseline_MM_Inventory"]
    inv_regime = data["A_regime_MM_Inventory"]
    ax3.plot(
        np.arange(len(inv_base)), inv_base,
        linewidth=0.7, color=COL_BASELINE, alpha=0.6,
        label=f"{BASE_LABEL}: p=0.5",
    )
    ax3.plot(
        steps, inv_regime,
        linewidth=0.7, color=COL_REGIME_A, alpha=0.85,
        label=f"{BASE_LABEL}: regime-switching",
    )
    if include_phaseb:
        inv_phaseb = data["B_regime_MM_Inventory"]
        ax3.plot(
            np.arange(len(inv_phaseb)), inv_phaseb,
            linewidth=0.7, color=COL_REGIME_B, alpha=0.85,
            label="Algorithm B: regime-switching",
        )
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax3.axhline(y=INV_LIMIT, color="gray", linestyle=":", alpha=0.4,
                label=f"±INV_LIMIT={INV_LIMIT}")
    ax3.axhline(y=-INV_LIMIT, color="gray", linestyle=":", alpha=0.4)
    ax3.set_ylabel("Inventory (q)")
    ax3.set_xlabel("Event step")
    # (title removed; caption describes)
    ax3.legend(loc="upper right", fontsize=14, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_signal_example(output):
    """Replay the selected signal example with the recorded C51/FIFO protocol."""
    import torch
    from evaluate_corrected import PROTOCOL, replay
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    case=next(c for c in PROTOCOL['cases'] if c['name']=='illustration_signals_B_123')
    row,a=replay(case,123,output,verify=True)
    arrays=dict(a,step=a['trace_step'],inventory=a['trace_inventory'],pnl=a['trace_pnl'])
    return arrays,row


def plot_signals(data):
    """Use the original diagnostic plot design with current iota notation."""
    x = data['step']
    fig_bayes, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(x, data['latent_bias'], linewidth=1, color='#444444', alpha=.9,
                 label=r'true $\iota$')
    axes[0].plot(x, data['posterior_bias'], linewidth=1, color='#0072B2', alpha=.95,
                 label=r'Bayes $\widehat{\iota}$')
    axes[0].axhline(0, color='gray', linewidth=.5, linestyle='--')
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel('directional bias', fontsize=15)
    axes[0].legend(loc='upper right', frameon=False, fontsize=13)
    axes[1].plot(x, data['expected_age_mo'], linewidth=1, color='#0072B2',
                 label=r'Bayes $E[\mathrm{run}]$')
    axes[1].plot(x, data['true_age_mo'], linewidth=1, color='#444444', alpha=.85,
                 linestyle='--', label='true run length')
    axes[1].set_ylabel('Regime age (MOs)', fontsize=15)
    axes[1].set_xlabel('LOB event', fontsize=15)
    fig_bayes.tight_layout()

    fig_exposure, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, data['quote_exposure'], linewidth=1, color='#0072B2', alpha=.95,
            label='quote exposure imbalance (DQN)')
    ax.axhline(0, color='gray', linewidth=.5, linestyle='--')
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('LOB event', fontsize=15)
    ax.set_ylabel('Quote exposure', fontsize=15)
    ax.legend(loc='upper right', frameon=False, fontsize=13)
    fig_exposure.tight_layout()
    return [('bayes_single_estimation_example', fig_bayes),
            ('quote_exposure_single_estimation_example', fig_exposure)]


def load_arrays(name, provenance):
    path = ARCHIVE / name
    if file_hash(path) != provenance['archives'][name]['sha256']:
        raise ValueError(f'Archived array hash differs: {name}')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError(f'Non-finite archived array: {name}')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures')
    parser.add_argument('--verify-rerun', action='store_true',
                        help='Rerun the new 5000-event signal illustration and verify all numeric arrays.')
    args = parser.parse_args()
    output = args.output.resolve()
    for folder in ('paper', 'data', 'src', 'config', 'checkpoints', 'repro'):
        protected = ROOT / folder
        if output == protected or protected in output.parents:
            parser.error('Use a separate output directory; archived inputs and paper assets are protected.')
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ARCHIVE / 'provenance.json').read_text())
    episode = load_arrays('single_episode.npz', provenance)
    signals = load_arrays('signal_example.npz', provenance)
    verification = None
    if args.verify_rerun:
        actual, metadata = generate_signal_example(output / 'signal_verification')
        if set(actual) != set(signals):
            raise ValueError('Rerun signal array names differ.')
        checks = {}
        for key, value in actual.items():
            if value.dtype != signals[key].dtype:
                raise ValueError(f'Rerun signal dtype differs: {key}')
            np.testing.assert_array_equal(value, signals[key], err_msg=key)
            checks[key] = dict(exact=True, max_abs_difference=0.0)
        verification = dict(all_exact=True, arrays=checks, rollout=metadata)

    rendered = {}
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update(EPISODE_STYLE)
        for include_b, name in [(False, 'regime_switching_one_episode_example'),
                                (True, 'regime_trained_final_phase_b')]:
            fig = plot_episode(episode, include_b)
            for suffix in ('png', 'pdf'):
                path = output / f'{name}.{suffix}'
                fig.savefig(path, dpi=150)
                rendered[path.name] = file_hash(path)
            plt.close(fig)
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update(EPISODE_STYLE)
        for name, fig in plot_signals(signals):
            for suffix in ('png', 'pdf'):
                path = output / f'{name}.{suffix}'
                fig.savefig(path, dpi=100, bbox_inches='tight')
                rendered[path.name] = file_hash(path)
            plt.close(fig)
    historical = {name: rendered[name] == spec['sha256']
                  for name, spec in provenance['selected_episode_figures'].items()}
    manifest = dict(
        archive_provenance_sha256=file_hash(ARCHIVE / 'provenance.json'),
        script_sha256=file_hash(__file__), outputs_sha256=rendered,
        selected_episode_png_byte_identical=historical,
        auxiliary_figures_kind=provenance['signal_example']['kind'],
        verification=verification, numpy=np.__version__, matplotlib=matplotlib.__version__,
    )
    (output / 'episode_reproduction.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), selected_episode_png_byte_identical=historical,
                          new_signal_example_verified=verification is not None), indent=2))


if __name__ == '__main__':
    main()
