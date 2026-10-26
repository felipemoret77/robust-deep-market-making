#!/usr/bin/env python3
"""Rerun frozen corrected-GLFT simulator experiments, with bounded default smoke.

python repro/glft_evaluate.py
python repro/glft_evaluate.py --mode smoke --experiment fig4_new_tau_60
python repro/glft_evaluate.py --mode full --workers 2

Default: one original 5000-event episode each for gamma=1e-6, p=0.2, and eta.
Full: original seeds, horizons, policies and experiments in glft_experiments.json.
Only the classical GLFT/at-best policies run here. Archived DQN means are reused.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
import glft_evaluation_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paper/glft_regenerated/data'


def run_chunk(job):
    output, specification, parameters, seeds = job
    runtime.ROOT = Path(output)
    return runtime.run_chunk((specification, parameters, seeds))


def verify_group(name, values):
    reference = DATA / f'{name}.npz'
    with np.load(reference) as expected:
        indices = np.searchsorted(expected['seeds'], values['seeds'])
        np.testing.assert_array_equal(expected['seeds'][indices], values['seeds'])
        errors = {}
        for key, actual in values.items():
            target = expected[key]
            if len(target) == len(expected['seeds']):
                target = target[indices]
            np.testing.assert_allclose(actual, target, rtol=0, atol=1e-12,
                                       err_msg=f'{name}: {key}')
            errors[key] = float(np.max(np.abs(actual-target))) if actual.size else 0.
    return dict(episodes=len(values['seeds']), maximum_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--experiment', action='append',
        help='Exact experiment name in config/glft_experiments.json; repeat for several.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.workers, args.chunk_size) < 1:
        parser.error('--workers and --chunk-size must be positive')
    config = json.loads((ROOT / 'config/glft_experiments.json').read_text())
    experiments = config['evaluation_experiments']
    names = args.experiment
    if names is None and args.mode == 'smoke':
        names = ['fig2_gamma_1e-06', 'fig3_p_0.2', 'eta']
    if names:
        known = {entry['spec']['name'] for entry in experiments}
        if set(names)-known:
            parser.error(f'Unknown experiments: {sorted(set(names)-known)}')
        experiments = [entry for entry in experiments if entry['spec']['name'] in names]
    if args.mode == 'smoke':
        experiments = [dict(entry, seeds=entry['seeds'][:1]) for entry in experiments]
    output = args.output or Path(f'outputs/glft_evaluation_{args.mode}')
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    source_paths = ['repro/glft_evaluation_runtime.py', 'src/sim/MM_LOB_SIM.py',
                    'src/glft/GLFT_policy_factory.py', 'src/glft/MM_policy_5.py',
                    'src/sim/glft_regime_schedule.py']
    protocol = dict(mode=args.mode, experiments=experiments,
        total_events=sum(len(entry['seeds'])*entry['spec'].get('events', 5000) for entry in experiments),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths})
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing evaluation protocol or source differs; choose a new --output directory.')
    runtime.atomic_json(protocol_path, protocol)
    jobs = []
    paths = {entry['spec']['name']: [] for entry in experiments}
    for entry in experiments:
        for start in range(0, len(entry['seeds']), args.chunk_size):
            jobs.append((str(output), entry['spec'], entry['params'],
                         entry['seeds'][start:start+args.chunk_size]))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(run_chunk, job): job[1]['name'] for job in jobs}
        for future in as_completed(futures):
            name = futures[future]
            paths[name].append(future.result())
            status = dict(completed_chunks=sum(map(len, paths.values())), total_chunks=len(jobs),
                          elapsed_s=time.monotonic()-started)
            runtime.atomic_json(output / 'status.json', status)
            print(json.dumps(status), flush=True)
    checks = {}
    for name, chunks in paths.items():
        values = runtime.load_group(chunks)
        checks[name] = verify_group(name, values)
        np.savez_compressed(output / 'data' / f'{name}.npz', **values)
    runtime.atomic_json(output / 'validation.json', dict(status='passed', experiments=checks,
        total_events=protocol['total_events'], elapsed_s=time.monotonic()-started,
        comparison='Every output array checked against archived corresponding seeds at absolute tolerance 1e-12.'))
    runtime.atomic_json(output / 'status.json', dict(status='complete', elapsed_s=time.monotonic()-started))
    print(json.dumps(dict(status='passed', total_events=protocol['total_events'],
                         experiments=list(checks), output=str(output)), indent=2))


if __name__ == '__main__':
    main()
