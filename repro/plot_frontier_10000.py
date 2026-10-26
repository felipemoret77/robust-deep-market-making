#!/usr/bin/env python3
"""Validate and replot the selected 110,000-outcome frontier without simulation.

The archived standard errors are from the campaign's 10,000 paired bootstrap
resamples. Replotting reuses them; it does not substitute normal-theory errors.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frontier_corrected_matplotlib')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/paper/frontier_10000'


def validate():
    provenance = json.loads((ARCHIVE / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == expected, name
    protocol = json.loads((ARCHIVE / 'campaign_manifest.json').read_text())['protocol']
    policies, seeds = protocol['policies'], protocol['seeds']
    rows = list(csv.DictReader(io.StringIO(gzip.decompress((ARCHIVE / 'per_run.csv.gz').read_bytes()).decode())))
    lookup = {(r['policy'], int(r['seed'])): r for r in rows}
    assert len(rows) == len(lookup) == len(seeds) * len(policies) == 110000
    assert set(lookup) == {(p['name'], s) for p in policies for s in seeds}
    assert all(int(r['n_events']) == 5000 for r in rows)
    assert all(r['variant'] == ('both' if p['family'] == 'RLMM' else 'queue_only')
               for p in policies for s in seeds for r in [lookup[p['name'], s]])
    x = np.array([[float(lookup[p['name'], s]['terminal_pnl']) for p in policies] for s in seeds])
    assert np.isfinite(x).all()
    summaries = {r['policy']: r for r in csv.DictReader((ARCHIVE / 'summary.csv').open())
                 if r['sample'] == 'corrected_10000'}
    means, sds = x.mean(0), x.std(0, ddof=1)
    for i, policy in enumerate(policies):
        row = summaries[policy['name']]
        assert int(row['episodes']) == len(seeds)
        np.testing.assert_allclose([means[i], sds[i]], [float(row['mean_pnl']), float(row['sd_pnl'])], atol=1e-12, rtol=0)
        if 'checkpoint' in policy:
            assert hashlib.sha256((ROOT / policy['checkpoint']).read_bytes()).hexdigest() == policy['sha256']
            training = policy['declared_training_seeds']
            assert not set(seeds).intersection(range(training['start'], training['start'] + training['count']))
    return provenance, policies, x, summaries


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    provenance, policies, x, summaries = validate()
    output = Path(output).resolve()
    for name in ('paper', 'data', 'config', 'src', 'checkpoints', 'repro'):
        protected = ROOT / name
        if output == protected or protected in output.parents:
            raise ValueError('Choose a separate output directory.')
    output.mkdir(parents=True, exist_ok=True)
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'RLMM': '#E69F00', 'GLFT': '#0072B2', 'at_best': '#999999'}
    labels = {'RLMM': 'RLMM (Algorithm A)', 'GLFT': 'GLFT', 'at_best': 'At-best baseline'}
    for family in ('GLFT', 'RLMM', 'at_best'):
        ids = sorted([i for i, p in enumerate(policies) if p['family'] == family], key=lambda i: sd[i])
        ax.errorbar(sd[ids], mu[ids],
                    xerr=[float(summaries[policies[i]['name']]['sd_se']) for i in ids],
                    yerr=[float(summaries[policies[i]['name']]['mean_se']) for i in ids],
                    color=colors[family], marker='D' if family == 'at_best' else 'o',
                    linestyle='none' if family == 'at_best' else '-', capsize=3, linewidth=1.5,
                    markersize=8 if family == 'at_best' else 6, label=labels[family], zorder=3)
        if family == 'at_best':
            continue
        groups = {}
        for i in ids:
            groups.setdefault((round(float(sd[i]), 12), round(float(mu[i]), 12)), []).append(i)
        for (xx, yy), members in groups.items():
            if family == 'RLMM':
                label = r'$\varphi=' + f"{policies[members[0]]['phi']:.3f}" + '$'
            else:
                powers = [r'10^{' + str(round(math.log10(policies[i]['gamma']))) + '}' for i in members]
                label = r'$\gamma=' + powers[0] + '$' if len(members) == 1 else r'$\gamma\in\{' + ','.join(powers) + r'\}$'
            offset = (8, -16) if family == 'GLFT' and policies[members[0]]['gamma'] == .01 else (8, 7)
            ax.annotate(label, (xx, yy), xytext=offset, textcoords='offset points', fontsize=9, color=colors[family])
    ax.set_xlabel(r'Risk: $\sqrt{\mathrm{Var}(\mathrm{PnL}_T)}$', fontsize=12)
    ax.set_ylabel(r'Return: $\mathbb{E}[\mathrm{PnL}_T]$', fontsize=12)
    ax.grid(alpha=.3)
    ax.margins(x=.24, y=.12)
    low, high = ax.get_ylim()
    ax.set_ylim(low, max(high, float(mu.max() + .30*np.ptp(mu))))
    ax.legend(loc='upper left', fontsize=11)
    fig.text(.5, .033, f'{len(x):,} paired episodes; 5,000 LOB events + 1,000 warmup; bars: ±1 bootstrap SE', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'glft_frontier_complete.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    actual = hashlib.sha256((output / 'glft_frontier_complete.png').read_bytes()).hexdigest()
    (output / 'frontier_10000_plot_validation.json').write_text(json.dumps({
        'runs': len(x)*len(policies), 'png_sha256': actual,
        'published_png_sha256': provenance['published_png_sha256'],
        'png_byte_identical': actual == provenance['published_png_sha256'],
        'uncertainty': 'archived bootstrap standard errors; 10000 resamples'}, indent=2)+'\n')
    print(f'Replotted 110000 outcomes; PNG byte-identical: {actual == provenance["published_png_sha256"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    plot(parser.parse_args().output)
