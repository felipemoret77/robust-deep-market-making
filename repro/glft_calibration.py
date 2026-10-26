#!/usr/bin/env python3
"""Corrected GLFT calibration: bounded smoke by default, explicit full rerun.

python repro/glft_calibration.py
python repro/glft_calibration.py --mode aggregate
python repro/glft_calibration.py --mode full --stage all --workers 2

Full primary: 20 seeds x 5 distances x 1,000,000 events.
Full windows: 20 seeds x 5 distances x 4 windows x 1,000,000 events.
All outputs remain under --output; the archived selected parameters are fixed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
import glft_calibration_core as core
from glft_calibration_stats import fit_runs, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paper'


class CompressedRecorder(core.Recorder):
    """Preserve all observation arithmetic while compressing lifecycle evidence."""
    def __init__(self, config, distance, policy, csv_path, progress_path):
        super().__init__(config, distance, policy, csv_path, progress_path)
        fields = self.writer.fieldnames
        self.file.close()
        Path(csv_path).unlink()
        self.csv_path = Path(str(csv_path) + '.gz')
        self.file = gzip.open(self.csv_path, 'wt', newline='', compresslevel=1)
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()


def run_job(job):
    core.Recorder = CompressedRecorder
    return core.run_distance(*job)


def smoke(output):
    core.self_test(output)
    comparisons = {}
    for variant in ('corrected',):
        observed = json.loads((output / 'self_test' /
            f'{variant}_seed1234_delta0.5_n5000.json').read_text())
        archived = json.loads((DATA / 'calibration' /
            f'{variant}_smoke_reference.json').read_text())
        fields = ('actual_distance_bins',
                  'requested_distance_bins', 'diagnostics')
        for key in fields:
            if observed[key] != archived[key]:
                raise AssertionError(f'{variant}: archived smoke differs in {key}')
        comparisons[variant] = 'exact counts, exposures, rates, and diagnostics'
    result = dict(status='passed', events=5000,
                  note='One 5000-event simulation, with 1000 warmup events; no parameter selection.',
                  reference_checks=comparisons)
    core.write_json(output / 'smoke_validation.json', result)
    print(json.dumps(result, indent=2))


def aggregate(output, bootstrap):
    """Recalculate selected fits from archived sufficient statistics, no simulation."""
    checks = {}
    for window in (.5, 1., 10., 60., 300.):
        stem = f'window_{window:g}s'
        with gzip.open(DATA / f'calibration/{stem}_runs.json.gz', 'rt') as f:
            runs = json.load(f)
        expected = json.loads((DATA / f'calibration/{stem}_summary.json').read_text())
        cfg = core.Config(**expected['config'])
        fits, curves = fit_runs(runs, cfg)
        for method, fit in fits.items():
            np.testing.assert_allclose([fit['A'], fit['kappa']],
                [expected['fits'][method]['A'], expected['fits'][method]['kappa']],
                rtol=2e-13, atol=1e-14)
            for row, reference in zip(curves[method], expected['curves'][method]):
                for key in ('distance', 'n_exec', 'exposure', 'lambda_hat'):
                    np.testing.assert_allclose(row[key], reference[key], rtol=2e-13, atol=1e-12)
        result = dict(fits=fits, curves=curves)
        if bootstrap:
            result = summarize(runs, cfg, expected['seeds'], 5000)
            for method in fits:
                for key in ('A', 'kappa'):
                    np.testing.assert_allclose(result['fits'][method]['cluster_bootstrap_95ci'][key],
                        expected['fits'][method]['cluster_bootstrap_95ci'][key], rtol=2e-13, atol=1e-13)
        core.write_json(output / f'{stem}_summary.json', result)
        checks[stem] = dict(runs=len(runs), fit='matches archived coefficients',
                           bootstrap='matches archived 5000 resamples' if bootstrap else 'not rerun')
    core.write_json(output / 'aggregation_validation.json', dict(status='passed', windows=checks))
    print(json.dumps(checks, indent=2))


def full(output, stage, workers):
    protocol = json.loads((ROOT / 'config/glft_experiments.json').read_text())['calibration']
    seeds = protocol['seeds']
    windows = [.5] if stage == 'primary' else [1., 10., 60., 300.]
    if stage == 'all':
        windows = protocol['windows_seconds']
    jobs = []
    for window in windows:
        folder = output / f'window_{window:g}s'
        folder.mkdir(parents=True, exist_ok=True)
        variants = protocol['primary_variants'] if window == .5 else protocol['other_window_variants']
        for seed in seeds:
            cfg = replace(core.Config(), n_steps=protocol['events_per_seed_distance'],
                          random_seed=seed, time_interval=window, posting_interval=window)
            for variant in variants:
                for distance in protocol['requested_distances']:
                    jobs.append((cfg, distance, variant, folder))
    sources = core.source_hashes()
    for filename in ('glft_calibration.py', 'glft_calibration_stats.py', 'glft_runtime.py'):
        path = ROOT / 'repro' / filename
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    specification = dict(protocol=protocol, stage=stage,
        workers=workers, jobs=len(jobs), total_events=len(jobs)*protocol['events_per_seed_distance'],
        sources=sources)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if any(previous.get(key) != specification[key] for key in ('protocol', 'stage', 'sources')):
            raise ValueError('Existing calibration protocol or source differs. Choose a new --output directory.')
    elif any(output.glob('window_*s/*_seed*_delta*_n*.json')):
        raise ValueError('Existing calibration results have no protocol. Choose a new --output directory.')
    core.write_json(protocol_path, specification)
    results = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            status = dict(completed=len(results), total=len(jobs), elapsed_s=time.monotonic()-started)
            core.write_json(output / 'status.json', status)
            print(json.dumps(status), flush=True)
    for window in windows:
        chosen = [r for r in results if r['provenance']['config']['time_interval'] == window]
        cfg = replace(core.Config(), n_steps=protocol['events_per_seed_distance'],
                      time_interval=window, posting_interval=window)
        summary = summarize(chosen, cfg, seeds, protocol['bootstrap_repetitions'])
        core.write_json(output / f'window_{window:g}s/summary.json', summary)
    core.write_json(output / 'status.json', dict(status='complete', completed=len(results),
        total=len(jobs), elapsed_s=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'aggregate', 'full'), default='smoke')
    parser.add_argument('--stage', choices=('primary', 'windows', 'all'), default='primary')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--bootstrap', action='store_true', help='Recheck all 5000 bootstrap resamples in aggregate mode.')
    parser.add_argument('--output', type=Path, default=Path('outputs/glft_calibration'))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'smoke':
        smoke(output)
    elif args.mode == 'aggregate':
        aggregate(output, args.bootstrap)
    else:
        full(output, args.stage, args.workers)


if __name__ == '__main__':
    main()
