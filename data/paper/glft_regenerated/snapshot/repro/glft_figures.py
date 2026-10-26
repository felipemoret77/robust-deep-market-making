#!/usr/bin/env python3
"""Reproduce the corrected GLFT paper panels from compact archived observations.

python repro/glft_figures.py --output outputs/glft_figures
python repro/glft_figures.py --strict-images

Recomputes the stationary model and q0=0 finite-horizon occupation, checks all
reference numerical tables, and renders 11 selected paper PNG/PDF panels plus
one diagnostic comparison. No FIFO simulation, training, or paper writes.
PNG hashes may depend on the Matplotlib/font version; numerical checks do not.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
import glft_plot_helpers as plots
import glft_transient as transient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paper/glft'
OUT = ROOT / 'outputs/glft_figures'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_archive():
    provenance = json.loads((DATA / 'provenance.json').read_text())
    for name, expected in provenance['archive_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'GLFT archive hash mismatch: {name}')
    if sha(ROOT / 'config/glft_experiments.json') != provenance['config_sha256']:
        raise ValueError('GLFT experiment configuration hash mismatch')
    return provenance


def save(fig, name, **kwargs):
    kwargs.pop('local_dir', None)
    kwargs.pop('dpi', None)
    kwargs.pop('bbox_inches', None)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{Path(name).stem}.{ext}', dpi=200, bbox_inches='tight', **kwargs)


def load(name):
    with np.load(DATA / f'{name}.npz') as source:
        return {key: source[key] for key in source.files}


def fig2():
    store, summaries = {}, []
    for gamma in (1e-6, 1e-3, 1e-1):
        d = load(f'fig2_gamma_{gamma:g}')
        store[gamma] = [pd.DataFrame({'MM_TotalPnL': p, 'MM_Inventory': q})
            for p, q in zip(d['pnl_trace'], d['inventory_trace'])]
        summaries.extend(dict(gamma=gamma, final_total_pnl=float(p)) for p in d['terminal_pnl'])
    base = load('fig2_at_best')
    benchmark = [pd.DataFrame({'MM_TotalPnL': p, 'MM_Inventory': q})
        for p, q in zip(base['pnl_trace'], base['inventory_trace'])]
    plots.plot_mean_total_pnl_trajectory_by_gamma(store, benchmark_store=benchmark)
    plots.plot_cartea_style_distributions(pd.DataFrame(summaries), store,
        benchmark_summary_df=pd.DataFrame({'final_total_pnl': base['terminal_pnl']}),
        benchmark_store=benchmark)
    plt.close('all')


def stationary(parameters):
    fields = ('A', 'kappa', 'sigma', 'gamma', 'T', 'inv_limit', 'r_fill_factor', 'tick_size', 'v0')
    probs = np.linspace(.2, .8, 121)
    curves = plots.compute_bundle_curves(probs, **{key: parameters[key] for key in fields})
    expected = load('analytical_curves')
    max_error = 0.
    for key, value in curves.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(np.asarray(value)-expected[key]))))
    np.savez_compressed(OUT / 'stationary_curves.npz', probs=probs, **curves)
    plots.save_plot_pi_lines([.3, .4, .5, .6, .7], curves)
    plots.save_plot_pi_heatmap(probs, curves)
    plots.save_plot_qbar(probs, curves)
    plots.save_plot_boundary_mass(probs, curves)
    rates = load('eta')['fill_rate']
    estimated_eta = rates.mean() / (2*parameters['A']*np.exp(-parameters['kappa']*curves['delta_bar']))
    np.testing.assert_allclose(estimated_eta, parameters['r_fill_factor'], atol=1e-14, rtol=0)
    return dict(max_absolute_error=max_error, eta_recomputed=float(estimated_eta))


def fig4():
    taus, means, sems = [15, 30, 60, 120, 240], [], []
    for tau in taus:
        y = load(f'fig4_new_tau_{tau}')['terminal_pnl']
        means.append(y.mean())
        sems.append(y.std(ddof=1)/np.sqrt(len(y)))
    base = load('fig4_new_baseline')['terminal_pnl'].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    color = '#E69F00'
    ax.errorbar(taus, means, yerr=sems, marker='o', capsize=4, color=color,
        linewidth=1.6, markersize=7, label=r'GLFT $\gamma=10^{-6}$ regime')
    ax.axhline(base, color=color, linestyle=':', linewidth=1.6,
        label=r'GLFT $\gamma=10^{-6}$ no regime')
    ax.set_xticks(taus)
    ax.set_xticklabels([rf'$\tau_r={t}$' for t in taus])
    ax.set_xlabel(r'$\tau_r$ [MO events]')
    ax.set_ylabel('Terminal PnL mean +/- SEM')
    ax.grid(alpha=.35)
    ax.legend(fontsize=14)
    fig.tight_layout()
    save(fig, 'glft_degradation_vs_tau_regime.png')
    plt.close(fig)


def fig1():
    data = pd.read_csv(DATA / 'calibration_windows_plot.csv')
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for color, window in zip(plt.cm.viridis(np.linspace(.05, .95, 5)), (.5, 1., 10., 60., 300.)):
        rows = data[data.window == window]
        summary = json.loads((DATA / f'calibration/window_{window:g}s_summary.json').read_text())
        x = rows.distance.to_numpy()
        ax.errorbar(x, rows.normalized_log_intensity, yerr=rows['sem'], fmt='o',
            markersize=4, capsize=4, color=color, label=rf'$\Delta T={window:g}$ s')
        ax.plot(x, -summary['fits']['corrected_actual']['kappa']*x, '--', color=color, linewidth=2)
    ax.set_xlabel(r'$\delta$ [tick]')
    ax.set_ylabel(r'$\log(\hat{\Lambda}(\delta)/\hat A)$')
    ax.grid(alpha=.3)
    ax.legend(loc='best')
    save(fig, 'censored_waiting_different_throttles.png')
    plt.close(fig)


def check_table(name, rows):
    expected = pd.read_csv(DATA / 'transient' / name)
    actual = pd.DataFrame(rows)
    if list(actual.columns) != list(expected.columns):
        raise AssertionError(f'Column mismatch in {name}')
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                               atol=2e-11, rtol=2e-12)
    return float(np.max(np.abs(actual.to_numpy(dtype=float)-expected.to_numpy(dtype=float))))


def finite_horizon(parameters):
    transient.ROOT = OUT
    with (DATA / 'expected_pnl_vs_buy_MO_updated.csv').open() as stream:
        source_rows = [{k: float(v) for k, v in row.items() if k != 'analytical_use_v0'}
                       for row in csv.DictReader(stream)]
    rows, raw, common_seeds = [], [], None
    for source in source_rows:
        p = source['p_buy_mo']
        row, _, _, _ = transient.calculate(p, parameters)
        for key in ('dqn_empirical_mean', 'dqn_empirical_std', 'dqn_empirical_se_mean',
                    'glft_empirical_mean', 'glft_empirical_std', 'glft_empirical_se_mean'):
            row[key] = source[key]
        np.testing.assert_allclose(row['stationary_full_pnl'],
                                   source['glft_misspecified_analytical_mean'], atol=1e-12)
        d = load(f'fig3_p_{p:g}')
        seeds, y = d['seeds'], d['terminal_pnl']
        if common_seeds is None:
            common_seeds = seeds.copy()
        np.testing.assert_array_equal(common_seeds, seeds)
        np.testing.assert_allclose([y.mean(), y.std(ddof=1)/np.sqrt(len(y))],
            [row['glft_empirical_mean'], row['glft_empirical_se_mean']], atol=1e-13)
        raw.append(y)
        row['n_glft_episodes'] = len(y)
        for name in ('stationary', 'transient'):
            residual = row[name+'_full_pnl']-row['glft_empirical_mean']
            row[name+'_residual'] = residual
            row[name+'_residual_in_se'] = residual/row['glft_empirical_se_mean']
        rows.append(row)
    dense, distributions = [], []
    for p in np.linspace(.2, .8, 121):
        row, model, terminal, occupation = transient.calculate(p, parameters)
        dense.append(row)
        for q, pi, end, average in zip(model['q'], model['pi'], terminal, occupation):
            distributions.append(dict(p_buy_mo=float(p), inventory=int(q),
                stationary_probability=float(pi), terminal_probability=float(end),
                time_average_probability=float(average)))
    errors = {}
    for name, table in [('comparison_all_probabilities.csv', rows),
                        ('analytical_curves_dense.csv', dense),
                        ('distributions_for_diagnostics.csv', distributions)]:
        errors[name] = check_table(name, table)
        transient.table(OUT / name, table)
    validation = transient.validate(parameters)
    metrics = transient.metrics(rows, raw)
    expected_metrics = json.loads((DATA / 'transient/metrics.json').read_text())
    for name in ('stationary', 'transient'):
        for key, value in metrics[name].items():
            np.testing.assert_allclose(value, expected_metrics[name][key], atol=1e-12, rtol=1e-12)
    transient.figures(rows, dense, parameters)
    dump(OUT / 'transient_metrics.json', metrics)
    dump(OUT / 'transient_validation.json', validation)
    return dict(reference_table_max_absolute_errors=errors, independent_validation=validation)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/glft_figures'))
    parser.add_argument('--strict-images', action='store_true',
        help='Also require PNG byte hashes to match the archived selected campaign images.')
    args = parser.parse_args()
    OUT = args.output if args.output.is_absolute() else ROOT / args.output
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = verify_archive()
    parameters = json.loads((ROOT / 'config/glft_experiments.json').read_text())['analytical_parameters']
    plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 14,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11,
        'legend.title_fontsize': 11, 'legend.framealpha': .9, 'legend.fancybox': True,
        'legend.borderpad': .3, 'legend.handlelength': 1.6})
    plt.show = lambda *args, **kwargs: None
    plots._save_paper_figure = save
    fig2()
    gc.collect()
    plt.rcParams['axes.prop_cycle'] = cycler(color=[tuple(c) for c in plt.cm.viridis(np.linspace(.05, .95, 6))])
    stationary_check = stationary(parameters)
    fig4()
    fig1()
    transient_check = finite_horizon(parameters)
    image_checks = {}
    for name, reference in provenance['selected_outputs'].items():
        actual = sha(OUT / name)
        image_checks[name] = dict(expected_sha256=reference['sha256'], actual_sha256=actual,
                                  byte_identical=actual == reference['sha256'])
    validation = dict(status='passed', archive_files_verified=len(provenance['archive_hashes']),
        stationary=stationary_check, transient=transient_check, selected_images=image_checks,
        all_selected_images_identical=all(row['byte_identical'] for row in image_checks.values()),
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                      pandas=pd.__version__, matplotlib=matplotlib.__version__),
        rendering_note='Image hashes depend on the plotting stack and fonts. All numeric references are checked independently.')
    dump(OUT / 'validation.json', validation)
    if args.strict_images and not validation['all_selected_images_identical']:
        raise AssertionError('Selected PNG hashes differ; see validation.json')
    print(json.dumps(dict(status='passed', selected_panels=len(image_checks),
        all_selected_images_identical=validation['all_selected_images_identical'],
        output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
