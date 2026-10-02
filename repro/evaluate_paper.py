#!/usr/bin/env python3
"""Evaluate frozen A/B/C controllers using the paper's explicit configuration."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

from paper_runtime import (checkpoint, evaluation_helpers, file_hash,
                           load_controller, make_schedule, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--scenario', choices=['stationary', 'recovery', 'sweep', 'random', 'correlated'], default='recovery')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int)
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--tau', type=int, help='Run one duration instead of the full sweep.')
    parser.add_argument('--original-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--audit-trace', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = manifest()['evaluation'].get(args.scenario, manifest()['evaluation']['recovery'])
    values = {k: getattr(args, k.replace('-', '_')) for k in ('seeds', 'seed-start', 'steps', 'warmup')}
    for k in list(values):
        if values[k] is None:
            values[k] = cfg[k.replace('-', '_')]
    if values['seeds'] < 1 or values['steps'] < 1 or values['warmup'] < 0:
        parser.error('seeds and steps must be positive; warmup must be nonnegative.')
    if args.output.exists():
        parser.error('Use a new output directory to preserve completed results.')
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (args.output / 'run.log').open('w') as log, contextlib.redirect_stdout(log):
        ctrl, meta = load_controller(args.algorithm, args.output, original_root=args.original_root)
        helpers, signals = evaluation_helpers(meta)
        taus = ([args.tau] if args.tau is not None else
                cfg.get('taus', [0 if args.scenario == 'stationary' else 60]))
        rows = []
        traces = []
        original_act = ctrl.act
        if args.audit_trace:
            def traced_act(state, mm=None):
                command = original_act(state)
                if ctrl._last_was_decision:
                    traces.append(dict(state=state, action=int(ctrl.last_action_idx), command=command))
                return command
            ctrl.act = traced_act
        try:
            for tau in taus:
                for seed in range(values['seed-start'], values['seed-start'] + values['seeds']):
                    schedule = make_schedule(helpers, args.scenario, seed, values['steps'], tau)
                    _, metrics = helpers.run_one_sim(
                        schedule, seed=seed, controller=ctrl, total_events=values['steps'],
                        equil_steps=values['warmup'], signal_config=signals, metrics_only=True)
                    rows.append(dict(algorithm=args.algorithm, scenario=args.scenario, tau=tau,
                                     seed=seed, terminal_pnl=float(metrics['MM_TotalPnL']),
                                     terminal_inventory=float(metrics['MM_Inventory'])))
                    print(rows[-1], flush=True)
        finally:
            ctrl.writer.close()
    with (args.output / 'per_run.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(algorithm=args.algorithm, scenario=args.scenario, **values,
                    taus=taus, checkpoint=str(checkpoint(args.algorithm)),
                    checkpoint_sha256=file_hash(checkpoint(args.algorithm)), signals=signals,
                    original_root=str(args.original_root) if args.original_root else None)
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.audit_trace:
        def encode(value):
            if isinstance(value, np.ndarray): return value.tolist()
            if isinstance(value, np.generic): return value.item()
            raise TypeError(type(value).__name__)
        (args.output / 'decisions.json').write_text(json.dumps(traces, default=encode, sort_keys=True))
    print(f'Completed {len(rows)} rollouts: {args.output / "per_run.csv"}')


if __name__ == '__main__':
    main()
