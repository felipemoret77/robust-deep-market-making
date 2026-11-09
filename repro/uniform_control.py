"""Verify the matched scenario-allocation control and replay its final policies.

Verification recomputes the paper claim from the 400 shipped outcomes. Replay
uses the same frozen C51/FIFO protocol as the selected paper evaluations.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from scipy import stats
import paper_runtime as paper

ROOT = paper.ROOT
CONFIG = ROOT / 'config/uniform_control.json'
METRICS = ('terminal_pnl', 'terminal_inventory', 'mean_abs_inventory',
           'inventory_limit_fraction', 'decisions', 'preserved_recovery_quotes')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def records(config=None):
    config = read_json(CONFIG) if config is None else config
    return [json.loads(line) for line in (ROOT / config['archive']['outcomes.jsonl']).read_text().splitlines()
            if line.strip()]


def summarize(rows, config):
    summaries, comparisons = [], []
    evaluation = config['evaluation']
    for family in ('random', 'correlated'):
        arrays = {}
        for arm in ('adaptive', 'uniform'):
            subset = sorted((row for row in rows if row['family'] == family and row['arm'] == arm),
                            key=lambda row: row['seed'])
            assert [row['seed'] for row in subset] == evaluation['families'][family]['seeds']
            pnl = np.array([row['terminal_pnl'] for row in subset])
            arrays[arm] = subset, pnl
            summaries.append(dict(family=family, arm=arm, episodes=len(pnl),
                mean_terminal_pnl=float(pnl.mean()), sd_terminal_pnl=float(pnl.std(ddof=1)),
                loss_fraction=float((pnl < 0).mean()),
                mean_abs_inventory=float(np.mean([row['mean_abs_inventory'] for row in subset])),
                mean_inventory_limit_fraction=float(np.mean([row['inventory_limit_fraction'] for row in subset]))))
        a_rows, a = arrays['adaptive']; b_rows, b = arrays['uniform']
        assert all(left['schedule_sha256_before'] == right['schedule_sha256_before']
                   for left, right in zip(a_rows, b_rows)), 'Unpaired regime schedules'
        delta = a - b
        rng = np.random.default_rng(evaluation['bootstrap_seed'])
        indices = rng.integers(0, len(delta), size=(evaluation['bootstrap_resamples'], len(delta)))
        means = delta[indices].mean(axis=1)
        comparisons.append(dict(family=family, contrast='adaptive minus uniform',
            episodes=len(delta), mean_pnl_difference=float(delta.mean()),
            paired_schedule_initial_hashes_equal=True,
            inference_scope='conditional on this single checkpoint pair',
            paired_standard_error=float(delta.std(ddof=1) / np.sqrt(len(delta))),
            paired_bootstrap_ci95=[float(x) for x in np.quantile(means, [.025, .975])],
            sd_ratio_left_over_right=float(a.std(ddof=1) / b.std(ddof=1)),
            paired_t_pvalue=float(stats.ttest_rel(a, b).pvalue)))
    return dict(status='complete', smoke=False, arm_summaries=summaries, paired_comparisons=comparisons)


def assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    else:
        assert actual == expected, (actual, expected)


def verify():
    """Check identities, matching, seed completeness and all reported statistics."""
    import torch
    config = read_json(CONFIG)
    data = {name: ROOT / relative for name, relative in config['archive'].items()}
    provenance = read_json(data['provenance.json'])
    for name, expected in provenance['data_sha256'].items():
        assert paper.file_hash(data[name]) == expected, name
    B = config['source_B']
    assert B['sha256'] == paper.manifest()['models']['B']['sha256']
    assert paper.file_hash(ROOT / B['checkpoint']) == B['sha256']
    for arm, item in config['models'].items():
        path = ROOT / item['checkpoint']
        assert paper.file_hash(path) == item['sha256'], arm
        saved = torch.load(path, map_location='cpu', weights_only=True)
        meta = saved['meta']
        assert meta['episodes_completed'] == meta['N_EPISODES'] == item['episodes_completed'] == 2000
        assert meta['GLOBAL_SEED'] == config['training']['master_seed']
        assert meta['MATCHED_BANDIT_PILOT']['arm'] == arm
        assert meta['USE_MIXTURE'] is False and meta['BAYES_FLOW_TAU_R'] == 60
    initial = read_json(data['matched_initialization.json'])
    for field in initial['adaptive']:
        if field != 'arm':
            assert initial['adaptive'][field] == initial['uniform'][field], field
    assert initial['adaptive']['replay_size'] == 0
    assert initial['adaptive']['optimizer_state_entries'] == 12
    assert initial['adaptive']['initial_weights_equal_B']
    assert initial['adaptive']['optimizer_moments_equal_B']
    assert len(initial['adaptive']['initial_pool_hashes']) == 256
    refreshes = read_json(data['refreshes.json'])
    for arm in ('adaptive', 'uniform'):
        assert [row['update'] for row in refreshes[arm]] == [500, 1000, 1500, 2000]
        assert all(len(set(row['replacement_indices'])) == len(row['incoming_hashes']) == 25
                   for row in refreshes[arm])
    for left, right in zip(refreshes['adaptive'], refreshes['uniform']):
        assert left['incoming_hashes'] == right['incoming_hashes']
    rows = records(config)
    assert len(rows) == len({(r['family'], r['arm'], r['seed']) for r in rows}) == 400
    for row in rows:
        assert row['events'] == config['evaluation']['events'] == 500000
        assert row['warmup'] == config['evaluation']['warmup'] == 10000
        assert row['checkpoint_sha256'] == config['models'][row['arm']]['sha256']
        assert row['networks_unchanged'] and not row['schedule_extended']
        assert row['schedule_sha256_before'] == row['schedule_sha256_after']
        assert all(math.isfinite(row[key]) for key in METRICS)
        assert abs(row['terminal_inventory']) <= 8 and 0 <= row['inventory_limit_fraction'] <= 1
    result = summarize(rows, config)
    assert_equal(result, read_json(data['summary.json']))
    for family, rounded in [('random', 9.4), ('correlated', 7.0)]:
        selected = {r['arm']: r for r in result['arm_summaries'] if r['family'] == family}
        gain = 100 * (selected['adaptive']['mean_terminal_pnl'] / selected['uniform']['mean_terminal_pnl'] - 1)
        assert round(gain, 1) == rounded
        print(f'{family}: mean terminal PnL gain {gain:.8f}%; 100 matched evaluation seeds')
    print('Uniform control: 400 outcomes, final models, matched initialization, refresh candidates and paired statistics verified')
    return result


def network_digest(controller):
    value = hashlib.sha256()
    for network in ('q_net', 'target_net'):
        for name, tensor in getattr(controller, network).state_dict().items():
            value.update((network + name).encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_controller(checkpoint, output):
    import torch
    from paper_evaluation_corrections import instrument_controller
    factory_path = ROOT / 'src/agents/checkpoint_io.py'
    tree = ast.parse(factory_path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'make_controller_from_checkpoint']
    assert len(nodes) == 1
    namespace = dict(torch=torch, Optional=Optional,
                     DeepRLController=partial(paper.controller_class(), device=torch.device('cpu')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(factory_path), 'exec'), namespace)
    controller = namespace['make_controller_from_checkpoint'](
        str(checkpoint), log_dir=str(output / 'tensorboard'),
        use_time_update=True, min_time_interval=1.0, inv_limit_override=8)
    controller.enable_learning = False
    controller.epsilon = 0.0
    for network in (controller.q_net, controller.target_net):
        network.eval(); network.requires_grad_(False)
    counters = dict(decisions=0, preserved_recovery_quotes=0)
    instrument_controller(controller, 'both', counters)
    return controller, torch.load(checkpoint, map_location='cpu', weights_only=True)['meta'], counters


def schedule_digest(schedule):
    fields = ('boundaries', 'p_values', 'tau_values', 'lengths', 'sides',
              'intensity_values', 'chunk_boundaries', 'chunk_p_values')
    record = {key: list(getattr(schedule, key)) for key in fields if hasattr(schedule, key)}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replay(args):
    import torch
    from evaluation_environment import check_environment
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = read_json(CONFIG); evaluation = config['evaluation']
    seeds = evaluation['families'][args.family]['seeds'][:args.seeds]
    events = evaluation['events'] if args.events is None else args.events
    warmup = evaluation['warmup'] if args.warmup is None else args.warmup
    if events <= 0 or warmup < 0:
        raise ValueError('events must be positive; warmup must be nonnegative')
    if (events != evaluation['events'] or warmup != evaluation['warmup']) and not args.smoke:
        raise ValueError('Horizon changes require --smoke')
    if args.verify_archive and args.smoke:
        raise ValueError('Archive replay verification requires the full recorded horizon')
    output = args.output.resolve()
    if args.smoke and 'smoke' not in str(output).lower():
        raise ValueError('Use a clearly named smoke output directory')
    if output == ROOT or any(output == ROOT / name or ROOT / name in output.parents
                             for name in ('paper', 'src', 'data', 'config', 'repro', 'checkpoints')):
        raise ValueError('Choose a separate output directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'environment.json', check_environment(allow_mismatch=args.allow_environment_mismatch))
    checkpoint = ROOT / config['models'][args.arm]['checkpoint']
    assert paper.file_hash(checkpoint) == config['models'][args.arm]['sha256']
    smoke_reference = read_json(ROOT / config['archive']['runtime_smoke_reference.json'])['reference'][args.arm]
    results = []
    with (output / 'rollout.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        controller, meta, counters = load_controller(checkpoint, output)
        before = network_digest(controller)
        helpers, signals = paper.evaluation_helpers(meta)
        try:
            for seed in seeds:
                counters.update(decisions=0, preserved_recovery_quotes=0)
                schedule = paper.make_schedule(helpers, args.family, seed, events)
                schedule_before = schedule_digest(schedule)
                with torch.inference_mode():
                    _, result = helpers.run_one_sim(schedule, seed=seed, controller=controller,
                        total_events=events, equil_steps=warmup, signal_config=signals, metrics_only=True)
                assert result['n_steps'] == events and network_digest(controller) == before
                assert not controller.enable_learning and len(controller.memory) == 0
                row = dict(arm=args.arm, family=args.family, seed=seed, events=events, warmup=warmup,
                    checkpoint_sha256=config['models'][args.arm]['sha256'],
                    terminal_pnl=float(result['MM_TotalPnL']), terminal_inventory=float(result['MM_Inventory']),
                    mean_abs_inventory=float(result['mean_abs_inventory']),
                    inventory_limit_fraction=float(result['limit_frac']),
                    schedule_sha256_before=schedule_before, schedule_sha256_after=schedule_digest(schedule),
                    networks_unchanged=True, **counters)
                assert all(math.isfinite(row[key]) for key in METRICS)
                if args.verify_archive:
                    saved = next(r for r in records(config) if (r['family'], r['arm'], r['seed']) == (args.family, args.arm, seed))
                    for key in METRICS:
                        np.testing.assert_allclose(row[key], saved[key], rtol=0, atol=1e-10, err_msg=key)
                    for key in ('schedule_sha256_before', 'schedule_sha256_after'):
                        assert row[key] == saved[key], key
                    row['archive_verified'] = True
                if args.smoke and all(row[key] == smoke_reference[key]
                                      for key in ('family', 'seed', 'events', 'warmup')):
                    for key in METRICS:
                        np.testing.assert_allclose(row[key], smoke_reference[key], rtol=0, atol=1e-10, err_msg=key)
                    for key in ('schedule_sha256_before', 'schedule_sha256_after'):
                        assert row[key] == smoke_reference[key], key
                    row['smoke_reference_verified'] = True
                results.append(row)
        finally:
            controller.writer.close()
    assert paper.file_hash(checkpoint) == config['models'][args.arm]['sha256']
    write_json(output / 'result.json', dict(status='complete', smoke=args.smoke, episodes=results))
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Recompute the published control from its archived outcomes')
    command = sub.add_parser('replay', help='Replay final policies on the prescribed evaluation seeds')
    command.add_argument('--family', choices=('random', 'correlated'), required=True)
    command.add_argument('--arm', choices=('adaptive', 'uniform'), required=True)
    command.add_argument('--seeds', type=int, choices=range(1, 101), default=1)
    command.add_argument('--events', type=int)
    command.add_argument('--warmup', type=int)
    command.add_argument('--smoke', action='store_true')
    command.add_argument('--verify-archive', action='store_true')
    command.add_argument('--allow-environment-mismatch', action='store_true')
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    else:
        replay(args)


if __name__ == '__main__':
    main()
