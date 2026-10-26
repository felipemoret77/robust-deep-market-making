#!/usr/bin/env python3
"""Replot five selected paper figures from complete corrected evaluations.

python repro/plot_policy_results.py --output outputs/policy_figures
python repro/plot_policy_results.py --verify-rerun --strict-images

The optional verification replays the corrected 5000-event stationary A example.
The default loads numeric archives with no simulator or PyTorch import.
Degradation summaries use all 100 recorded individual outcomes per condition.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paper/policy_results'
PAPER_STYLE = {'axes.labelsize': 14, 'xtick.labelsize': 12,
               'ytick.labelsize': 12, 'legend.fontsize': 12,
               'legend.title_fontsize': 12}
# Notebook cell 2 imports regime_switching_stress_test after its initial style
# cell, so these module-level settings govern the two selected histograms.
STRESS_STYLE = {'axes.labelsize': 15, 'axes.titlesize': 14,
                'xtick.labelsize': 13, 'ytick.labelsize': 13,
                'legend.fontsize': 11, 'legend.title_fontsize': 11,
                'legend.framealpha': .9, 'legend.fancybox': True,
                'legend.borderpad': .3, 'legend.handlelength': 1.6,
                'figure.dpi': 120}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    with np.load(DATA / name, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def save(fig, output, name, *, tight=False):
    kwargs = {'bbox_inches': 'tight'} if tight else {}
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{Path(name).stem}.{ext}', dpi=150, **kwargs)
    plt.close(fig)


def plot_episode(output):
    data = load('episode_seed124.npz')
    assert len(data['event']) == 5000
    np.testing.assert_array_equal(data['event'], np.arange(5000))
    assert np.max(np.abs(data['inventory'])) <= 8
    with plt.rc_context({'axes.labelsize': 15, 'axes.titlesize': 14,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13}):
        color = plt.cm.viridis(.05)
        for name, key, label in [('total_pnl_0001.png', 'pnl', 'Cumulative total PnL'),
                                 ('MM_inv_0001.png', 'inventory', r'Inventory $q_t$')]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(data['event'], data[key], color=color, linewidth=2)
            if key == 'inventory':
                ax.axhline(8, color='0.5', linestyle='--', linewidth=1)
                ax.axhline(-8, color='0.5', linestyle='--', linewidth=1)
            ax.set_xlabel('LOB event')
            ax.set_ylabel(label)
            ax.grid(True, alpha=.3)
            fig.tight_layout()
            save(fig, output, name)
    return dict(events=len(data['event']), final_pnl=float(data['pnl'][-1]),
                inventory_min=float(data['inventory'].min()),
                inventory_max=float(data['inventory'].max()))


def plot_stress(output, family):
    data = load(f'{family}.npz')
    fixed, stress = data['fixed_pnl'], data['stress_pnl']
    assert len(fixed) == len(stress) == 50
    np.testing.assert_array_equal(stress-fixed, data['paired_pnl_difference'])
    with plt.rc_context(STRESS_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        all_pnls = np.concatenate([fixed, stress])
        bins = np.linspace(all_pnls.min()-.05, all_pnls.max()+.05, 30)
        ax.hist(fixed, bins=bins, alpha=.45, color='#444444', edgecolor='black',
                linewidth=.3, label=f'fixed tau=60, mu={fixed.mean():+.3f}')
        stem = 'random tau' if family == 'random' else 'stress rho=0.85'
        ax.hist(stress, bins=bins, alpha=.45, color='#009E73', edgecolor='black',
                linewidth=.3, label=f'{stem}, mu={stress.mean():+.3f}')
        ax.axvline(fixed.mean(), color='#444444', linestyle='--', linewidth=1.5)
        ax.axvline(stress.mean(), color='#009E73', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Terminal PnL')
        ax.set_ylabel('Count')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=.3)
        plt.tight_layout()
        name = 'random_tau_stress.png' if family == 'random' else 'correlated_regime_stress.png'
        save(fig, output, name, tight=True)
    return dict(episodes=50, fixed_mean=float(fixed.mean()), stress_mean=float(stress.mean()),
                mean_paired_difference=float((stress-fixed).mean()),
                bins=bins.tolist(), fixed_counts=np.histogram(fixed, bins=bins)[0].tolist(),
                stress_counts=np.histogram(stress, bins=bins)[0].tolist())


def plot_degradation(output):
    with (DATA / 'degradation_summary.csv').open() as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    groups = {}
    for phi in (.001, .005):
        selected = [row for row in rows if row['phi'] == phi]
        assert [row['tau'] for row in selected] == [0, 15, 30, 60, 120, 240]
        assert all(row['n'] == 100 for row in selected)
        groups[phi] = dict(means=np.array([row['mean'] for row in selected]),
                           sems=np.array([row['std'] for row in selected])/np.sqrt(100))
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(5)
        styles = [(.001, '#E69F00', 'o'), (.005, 'midnightblue', 's')]
        for phi, color, marker in styles:
            ax.axhline(groups[phi]['means'][0], color=color, linestyle=':',
                linewidth=2., alpha=.9, label=f'Algorithm A invp={phi:.3f} no regime')
        for phi, color, marker in styles:
            ax.errorbar(x, groups[phi]['means'][1:], yerr=groups[phi]['sems'][1:],
                fmt=marker+'-', capsize=4, color=color, linewidth=1.9, markersize=6,
                label=f'Algorithm A invp={phi:.3f} regime')
        ax.set_ylabel('Terminal PnL mean +/- SEM')
        ax.grid(True, linestyle='--', alpha=.35)
        ax.legend(loc='best', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([rf'$\tau_r={tau}$' for tau in (15, 30, 60, 120, 240)])
        ax.set_xlabel(r'$\tau_r$ [MO events]')
        plt.tight_layout()
        save(fig, output, 'degradation_vs_tau_regime.png')
    return dict(precision='Full precision individual corrected outcomes',
                series={str(phi): {key: values.tolist() for key, values in group.items()}
                        for phi, group in groups.items()})


def verify_rerun(output, provenance):
    """Replay the selected corrected 5000-event A example; compare all arrays."""
    import torch
    from evaluate_corrected import PROTOCOL, replay
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    case=next(c for c in PROTOCOL['cases'] if c['name']=='illustration_A_stationary_124')
    row, arrays=replay(case,124,output/'stationary_rerun',verify=True)
    return dict(status='passed',seed=124,events=5000,archive_verified=row['archive_verified'])


def compare_images(output, references):
    checks = {}
    for name, expected in references.items():
        with Image.open(output / name) as original:
            image = original.convert('RGBA')
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        checks[name] = dict(byte_identical=sha(output/name) == expected['sha256'],
            pixel_identical=pixel_sha == expected['rgba_sha256'] and list(image.size) == expected['size'],
            sha256=sha(output/name), rgba_sha256=pixel_sha, size=list(image.size),
            reference_size=expected['size'])
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/policy_figures'))
    parser.add_argument('--verify-rerun', action='store_true')
    parser.add_argument('--strict-images', action='store_true',
                        help='Require identical decoded RGBA pixels to selected paper PNGs.')
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    protected = [ROOT / name for name in ('paper', 'data', 'checkpoints', 'src', 'config', 'repro')]
    if output == ROOT or any(output == folder or folder in output.parents for folder in protected):
        parser.error('--output must be outside the source, paper, checkpoint and archived-data directories')
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((DATA / 'provenance.json').read_text())
    for name, expected in provenance['archive_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Final archive hash mismatch: {name}')
    numerical = dict(episode=plot_episode(output), random=plot_stress(output, 'random'),
                     correlated=plot_stress(output, 'correlated'), degradation=plot_degradation(output))
    images = compare_images(output, provenance['selected_outputs'])
    result = dict(status='passed', archive_files_verified=len(provenance['archive_sha256']),
        figures=images, numeric_checks=numerical,
        all_pixels_identical=all(record['pixel_identical'] for record in images.values()),
        software=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        limitation=provenance['degradation']['limitation'])
    if args.verify_rerun:
        result['stationary_rerun'] = verify_rerun(output, provenance)
    (output / 'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.strict_images and not result['all_pixels_identical']:
        raise AssertionError('Image pixels differ; inspect validation.json and plotting-stack versions.')
    print(json.dumps(dict(status='passed', output=str(output),
                         all_pixels_identical=result['all_pixels_identical'],
                         stationary_rerun=result.get('stationary_rerun', 'not requested')), indent=2))


if __name__ == '__main__':
    main()
