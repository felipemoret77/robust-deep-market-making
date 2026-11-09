"""Isolated instrumentation for the paired scenario-sampling pilot.

Learner, reward, simulator and original C update order remain untouched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np
from DRO_adversary import DROHardPathAdversary


def config():
    return json.loads(Path(os.environ['UNIFORM_CONTROL_CONFIG']).read_text())


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(jsonable(value), allow_nan=False) + '\n')


def object_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        import torch
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def trajectory_hash(trajectory):
    return hashlib.sha256(json.dumps(dataclasses.asdict(trajectory), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


class PairedScenarioSampler(DROHardPathAdversary):
    """Original adaptive arithmetic with explicit independent random streams."""
    def __init__(self, *args, **kwargs):
        self.experiment = config()
        self.output = Path(self.experiment['arm_dir'])
        self.arm = self.experiment['arm']
        self.stream_seeds = self.experiment['streams']
        # Original pool construction has its own stream and exact balanced mix.
        kwargs['seed'] = self.stream_seeds['initial_pool']
        super().__init__(*args, **kwargs)
        self.rng_initial = self.rng
        self.rng_selection = np.random.default_rng(self.stream_seeds['scenario_selection'])
        self.rng_candidates = np.random.default_rng(self.stream_seeds['incoming_candidates'])
        self.rng_replacement = np.random.default_rng(self.stream_seeds['pool_replacement'])
        self.rng = self.rng_selection
        self.last_selected_hash = None
        initial = [dataclasses.asdict(item) for item in self.pool]
        write_json(self.output / 'initial_pool.json', initial)
        write_json(self.output / 'initial_pool_hashes.json', [trajectory_hash(item) for item in self.pool])

    def _sampling_probs(self):
        if getattr(self, 'arm', None) == 'uniform':
            return np.full(self.pool_size, 1.0 / self.pool_size, dtype=np.float64)
        return super()._sampling_probs()

    def select_schedule(self):
        schedule = super().select_schedule()
        self.last_selected_hash = trajectory_hash(self.pool[self.last_index])
        return schedule

    def refresh_easiest(self):
        n = max(0, min(int(np.floor(self.pool_size * self.refresh_frac)), self.pool_size))
        if not n:
            return
        # Draw candidates before any slot choice, using a dedicated stream.
        selected_rng = self.rng
        self.rng = self.rng_candidates
        try:
            incoming = [self._new_trajectory() for _ in range(n)]
        finally:
            self.rng = selected_rng
        if self.arm == 'uniform':
            indices = self.rng_replacement.choice(self.pool_size, n, replace=False)
        else:
            # Preserve the original adaptive runner's NumPy argsort tie rule.
            indices = np.argsort(self.difficulty)[:n]
        median = float(np.median(self.difficulty))
        for index, trajectory in zip(indices, incoming):
            self.pool[int(index)] = trajectory
            self.difficulty[int(index)] = median
            self.counts[int(index)] = 0
        append_json(self.output / 'refreshes.jsonl', {
            'update': self.n_updates, 'arm': self.arm, 'replacement_indices': indices,
            'incoming_hashes': [trajectory_hash(item) for item in incoming],
            'pool_hashes': [trajectory_hash(item) for item in self.pool],
            'median_difficulty': median,
        })

    def update_loss(self, loss, index=None):
        selected_index = self.last_index if index is None else int(index)
        selected_hash = trajectory_hash(self.pool[selected_index])
        result = super().update_loss(loss, index=index)
        append_json(self.output / 'sampler_updates.jsonl', {
            'update': self.n_updates, 'index': selected_index, 'scenario_hash': selected_hash,
            'probability': float(self.last_probs[selected_index]), 'loss': float(loss),
            'beta': float(self.beta), 'eps_uniform': float(self.eps_uniform),
            'probability_sum': float(self.last_probs.sum()),
            'probability_min': float(self.last_probs.min()),
            'probability_max': float(self.last_probs.max()),
        })
        return result

    def save_state(self):
        write_json(self.output / 'sampler_state.json', {
            'pool': [dataclasses.asdict(item) for item in self.pool],
            'difficulty': self.difficulty, 'counts': self.counts,
            'n_updates': self.n_updates, 'last_index': self.last_index,
            'last_probs': self.last_probs,
            'rng_initial': self.rng_initial.bit_generator.state,
            'rng_selection': self.rng_selection.bit_generator.state,
            'rng_candidates': self.rng_candidates.bit_generator.state,
            'rng_replacement': self.rng_replacement.bit_generator.state,
        })


def before_training(ctrl, variables, sampler):
    import torch
    cfg = config()
    output = Path(cfg['arm_dir'])
    checkpoint = torch.load(variables['WARMSTART_CKPT'], map_location='cpu', weights_only=False)
    assert len(ctrl.memory) == 0
    assert not variables['_arch_changed']
    assert int(variables['EXPECTED_MO_PER_EPISODE']) == round(variables['EPISODE_LENGTH'] * 0.03488)
    for key in ('q_net', 'target_net'):
        actual = getattr(ctrl, key).state_dict()
        assert actual.keys() == checkpoint[key].keys()
        for name in actual:
            assert torch.equal(actual[name].cpu(), checkpoint[key][name].cpu()), (key, name)
    names_and_params = list(ctrl.q_net.named_parameters())
    old_ids = [index for group in checkpoint['optimizer']['param_groups'] for index in group['params']]
    assert len(old_ids) == len(names_and_params) == 12
    for old_id, (name, parameter) in zip(old_ids, names_and_params):
        old_state = checkpoint['optimizer']['state'][old_id]
        current_state = ctrl.optimizer.state[parameter]
        for key, old_value in old_state.items():
            value = current_state[key]
            assert torch.equal(value.cpu(), old_value.cpu()) if isinstance(value, torch.Tensor) else value == old_value
        assert torch.equal(ctrl._anchor_weights[name].cpu(), checkpoint['q_net'][name].cpu())
    assert variables['ANCHOR_LAMBDA'] == 0.001
    initial = {
        'arm': cfg['arm'], 'warmstart': variables['WARMSTART_CKPT'],
        'q_net_hash': object_hash(ctrl.q_net.state_dict()),
        'target_net_hash': object_hash(ctrl.target_net.state_dict()),
        'optimizer_hash': object_hash(ctrl.optimizer.state_dict()),
        'anchor_hash': object_hash(ctrl._anchor_weights), 'replay_size': len(ctrl.memory),
        'optimizer_state_entries': len(ctrl.optimizer.state),
        'optimizer_moments_equal_B': True, 'initial_weights_equal_B': True,
        'learning_rates': [group['lr'] for group in ctrl.optimizer.param_groups],
        'best_params': variables['best_params'], 'episodes': variables['N_EPISODES'],
        'events_per_episode': variables['EPISODE_LENGTH'],
        'warmup_events': variables['ITER_TO_EQUILIBRIUM'],
        'initial_pool_hashes': [trajectory_hash(item) for item in sampler.pool],
    }
    write_json(output / 'initial_state.json', initial)
    write_json(output / 'progress.json', {'status': 'training', 'episodes_completed': 0,
                                         'started_unix': cfg['started_unix']})


def record_episode(variables):
    cfg = config()
    output = Path(cfg['arm_dir'])
    controller = variables['deep_controller']
    sampler = variables['dro_tau_adversary']
    episode = int(variables['ep']) + 1
    row = {
        'episode': episode, 'environment_seed': int(variables['current_ep_seed']),
        'learning_seed': int(variables['learning_ep_seed']),
        'scenario_index': sampler.last_index, 'scenario_hash': sampler.last_selected_hash,
        'scenario_probability': float(sampler.last_probs[sampler.last_index]),
        'persistent_side': bool(variables['ep_buy_mo_prob'].persistent_side),
        'terminal_pnl': float(variables['final_pnl']),
        'penalized_pnl': float(variables['penalized_pnl']),
        'mean_abs_inventory': float(variables['mean_abs_inv']),
        'mean_inventory_sq': float(variables['mean_inv_sq']),
        'fraction_at_limit': float(variables['pct_at_limit']),
        'gradient_steps_cumulative': int(controller.train_steps),
        'replay_size': len(controller.memory),
        'learning_rates_used': [float(group['lr']) for group in controller.optimizer.param_groups],
        'elapsed_seconds': time.time() - cfg['started_unix'],
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024),
    }
    append_json(output / 'episodes.jsonl', row)
    write_json(output / 'progress.json', {'status': 'training', 'episodes_completed': episode,
                                         'requested_episodes': cfg['episodes'], **row})


def finish_training(variables):
    import torch
    cfg = config()
    output = Path(cfg['arm_dir'])
    ctrl = variables['deep_controller']
    sampler = variables['dro_tau_adversary']
    source = Path(variables['CKPT_PATH']).resolve()
    final = output / 'final.pt'
    shutil.copy2(source, final)
    sampler.save_state()
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state()}, output / 'learning_rng_state.pt')
    ctrl.writer.flush(); ctrl.writer.close()
    completed = int(variables['ep']) + 1
    assert completed == cfg['episodes']
    assert sampler.n_updates == completed
    info = {'status': 'completed', 'arm': cfg['arm'], 'episodes_completed': completed,
            'checkpoint': str(final), 'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
            'gradient_steps': int(ctrl.train_steps), 'selection_rule': 'final',
            'elapsed_seconds': time.time() - cfg['started_unix'],
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)}
    write_json(output / 'final_checkpoint.json', info)
    write_json(output / 'progress.json', info)
    if cfg['smoke']:
        verify_smoke(output)
    print(json.dumps(info), flush=True)


def verify_smoke(output):
    """Check a short real training run against the executed pilot reference."""
    import torch
    output = Path(output)
    cfg = json.loads((output / 'experiment_config.json').read_text())
    assert cfg['smoke'] and cfg['episodes'] == 3
    reference = Path(cfg['repository']) / 'data/paper/uniform_control/training_smoke_reference.json'
    expected = json.loads(reference.read_text())['arms'][cfg['arm']]
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    rows = [{key: value for key, value in row.items()
             if key not in ('elapsed_seconds', 'peak_rss_mib')} for row in rows]
    assert rows == expected['episodes'], 'Early training metrics differ from the pilot reference'
    saved = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
    for key in ('q_net', 'target_net', 'optimizer'):
        assert object_hash(saved[key]) == expected[key + '_hash'], key
    info = json.loads((output / 'final_checkpoint.json').read_text())
    assert info['gradient_steps'] == expected['gradient_steps']
    write_json(output / 'smoke_verification.json', {
        'status': 'passed', 'arm': cfg['arm'], 'episodes': len(rows),
        'episode_metrics_equal_reference': True,
        'online_target_and_optimizer_equal_reference': True,
    })
