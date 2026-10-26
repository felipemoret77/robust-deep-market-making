#!/usr/bin/env python3
"""Replot archived corrected probes with unweighted per-seed normalization.

Only writes to --output. No simulator, benchmark, manuscript, or slide edits.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DISTANCES = np.array([0.5, 1., 1.5, 2., 2.5])
WINDOWS = (0.5, 1., 10., 60., 300.)
FIGURE = 'censored_waiting_different_throttles'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool(runs):
    counts, exposure = np.zeros(5, dtype=np.int64), np.zeros(5)
    for run in runs:
        for row in run['actual_distance_bins']:
            ix = np.flatnonzero(DISTANCES == row['distance'])
            if ix.size:
                counts[ix[0]] += row['n_exec']
                exposure[ix[0]] += row['exposure']
    assert np.all(counts > 0) and np.all(exposure > 0)
    return counts, exposure


def fit(rates):
    design = np.column_stack((np.ones(5), -DISTANCES))
    log_rates = np.log(rates)
    intercept, kappa = np.linalg.lstsq(design, log_rates, rcond=None)[0]
    # Independent slope formula and OLS normal equations.
    centered = DISTANCES - DISTANCES.mean()
    direct_kappa = -np.dot(centered, log_rates - log_rates.mean()) / np.dot(centered, centered)
    np.testing.assert_allclose(kappa, direct_kappa, rtol=1e-12)
    np.testing.assert_allclose(design.T @ (log_rates - intercept + kappa*DISTANCES), 0., atol=1e-12)
    return float(intercept), float(kappa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root/'data/paper/calibration')
    parser.add_argument('--manifest', type=Path, default=root/'data/paper/calibration_windows_unweighted/calibration_plot_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    original_fit = dict(manifest['full_fit'])
    old_windows = {r['window']: r for r in manifest['window_fits']}
    new_windows, graph_rows, seed_rows = [], [], []
    source_hashes = {}

    for window in WINDOWS:
        source = args.data / f'window_{window:g}s_runs.json.gz'
        summary_path = args.data / f'window_{window:g}s_summary.json'
        source_hashes[str(source)] = sha(source)
        source_hashes[str(summary_path)] = sha(summary_path)
        summary = json.loads(summary_path.read_text())
        with gzip.open(source, 'rt') as f:
            runs = [r for r in json.load(f) if r['provenance']['variant'] == 'corrected']
        seeds = sorted({r['provenance']['config']['random_seed'] for r in runs})
        assert len(runs) == 100 and len(seeds) == 20
        assert seeds == sorted(summary['seeds'])
        for r in runs:
            cfg = r['provenance']['config']
            assert cfg['n_steps'] == 1_000_000
            assert cfg['time_interval'] == cfg['posting_interval'] == window
        counts, exposure = pool(runs)
        reference = {r['distance']: r for r in summary['curves']['corrected_actual']}
        np.testing.assert_array_equal(counts, [reference[d]['n_exec'] for d in DISTANCES])
        np.testing.assert_allclose(exposure, [reference[d]['exposure'] for d in DISTANCES], rtol=2e-14)
        intercept, kappa = fit(counts / exposure)
        np.testing.assert_allclose([np.exp(intercept), kappa],
                                   [old_windows[window]['A_fit'], old_windows[window]['kappa']], rtol=1e-13)
        values = []
        for seed in seeds:
            subset = [r for r in runs if r['provenance']['config']['random_seed'] == seed]
            assert len(subset) == 5
            assert sorted(r['provenance']['distance'] for r in subset) == DISTANCES.tolist()
            seed_counts, seed_exposure = pool(subset)
            rates = seed_counts / seed_exposure
            seed_intercept, seed_kappa = fit(rates)
            normalized = np.log(rates) - seed_intercept
            values.append(normalized)
            for i, distance in enumerate(DISTANCES):
                seed_rows.append(dict(window=window, seed=seed, distance=float(distance),
                                      fills=int(seed_counts[i]), exposure=float(seed_exposure[i]),
                                      intensity=float(rates[i]), A_fit=float(np.exp(seed_intercept)),
                                      kappa=seed_kappa, normalized_log_intensity=float(normalized[i])))
        values = np.array(values)
        means = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        np.testing.assert_allclose(sem**2, ((values-means)**2).sum(axis=0)/(20*19), atol=1e-15)
        new_windows.append(dict(window=window, A_fit=float(np.exp(intercept)), kappa=kappa,
                                distances=DISTANCES.tolist(), normalized_log_intensity=means.tolist(), sem=sem.tolist()))
        for i, distance in enumerate(DISTANCES):
            graph_rows.append(dict(window=window, distance=float(distance), normalized_log_intensity=float(means[i]),
                                   sem=float(sem[i]), n_seeds=20, fitted_line=float(-kappa*distance)))

    # Preserve the approved plot design, axes, colors, markers, and legend.
    plt.rcParams.update({'axes.labelsize':15, 'axes.titlesize':14, 'xtick.labelsize':13,
                         'ytick.labelsize':13, 'legend.fontsize':11, 'legend.title_fontsize':11,
                         'legend.framealpha':.9, 'legend.fancybox':True, 'legend.borderpad':.3,
                         'legend.handlelength':1.6})
    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    for color, row in zip(plt.cm.viridis(np.linspace(.05,.95,5)), new_windows):
        window = row['window']
        ax.errorbar(DISTANCES, row['normalized_log_intensity'], yerr=row['sem'], fmt='o',
                    markersize=4, capsize=4, color=color, label=rf'$\Delta T={window:g}$ s')
        ax.plot(DISTANCES, -row['kappa']*DISTANCES, '--', color=color, linewidth=2)
    ax.set_xlabel(r'$\delta$ [tick]')
    ax.set_ylabel(r'$\log(\hat\Lambda(\delta)/\hat A_{\mathrm{fit}})$')
    ax.grid(alpha=.3)
    ax.legend(loc='best')
    for extension in ('png','pdf'):
        fig.savefig(args.output/f'{FIGURE}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    method = dict(regression='Unweighted OLS of log intensity against effective distance, both pooled and per seed.',
                  intensity='Observed fills divided by observed exposure; observations pooled within each distance.',
                  points='Equal-weight mean over 20 seeds of log(intensity_seed / A_fit_seed).',
                  bars='One standard error across seeds: sample standard deviation (ddof=1) / sqrt(20).',
                  lines='Negative pooled unweighted slope times distance.',
                  simulations_rerun=False, selected_full_range_parameters_changed=False,
                  selection='current paper and slides')
    manifest['window_fits'] = new_windows
    manifest['window_comparison_method'] = method
    manifest['input_sha256'].update(source_hashes)
    assert manifest['full_fit'] == original_fit
    write_json(args.output/'calibration_plot_manifest.json', manifest)
    table(args.output/'window_points_unweighted.csv', graph_rows)
    table(args.output/'window_seed_statistics_unweighted.csv', seed_rows)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), method=method,
                  validation=dict(source_counts_and_exposure='passed', twenty_seeds_per_window='passed',
                                  pooled_unweighted_parameters_unchanged='passed', independent_OLS_check='passed',
                                  SEM_check='passed'),
                  source_sha256=source_hashes, source_manifest_sha256=sha(args.manifest),
                  script_sha256=sha(Path(__file__)), numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                  selected_full_range_fit=original_fit,
                  output_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    write_json(args.output/'validation.json', report)
    print(json.dumps(dict(output=str(args.output), validation=report['validation']), indent=2))


if __name__ == '__main__':
    main()
