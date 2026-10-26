"""Portable, frozen-policy loading for the paper's named experiments."""
from __future__ import annotations

import ast
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for folder in ('sim', 'agents', 'glft', 'eval'):
    sys.path.insert(0, str(ROOT / 'src' / folder))


def manifest():
    return json.loads((ROOT / 'config/paper_experiments.json').read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(algorithm):
    spec = manifest()['models'][algorithm]
    path = ROOT / spec['checkpoint']
    if file_hash(path) != spec['sha256']:
        raise ValueError(f'Checkpoint hash differs from paper manifest: {path}')
    return path


def load_controller(algorithm, output, *, original_root=None):
    """Load the original factory without importing unrelated PPO/SAC tooling.

    original_root is used only by the equivalence audit in a fresh process.
    Normal reproduction imports the shipped source and checkpoint.
    """
    import torch
    if original_root is not None:
        original_root = Path(original_root).resolve()
        sys.path.insert(0, str(original_root))
    from dqn_distributional_with_throttle import DeepRLController

    factory_path = ((original_root / 'MM_GLFT_naive_comparison.py')
                    if original_root else ROOT / 'src/glft/MM_GLFT_naive_comparison.py')
    tree = ast.parse(factory_path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == 'make_controller_from_checkpoint']
    if len(nodes) != 1:
        raise RuntimeError('Expected one checkpoint factory.')
    namespace = dict(torch=torch, Optional=Optional,
                     DeepRLController=partial(DeepRLController, device=torch.device('cpu')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(factory_path), 'exec'), namespace)
    path = checkpoint(algorithm)
    ctrl = namespace['make_controller_from_checkpoint'](
        str(path), log_dir=str(Path(output) / 'tensorboard'),
        use_time_update=True, min_time_interval=1.0, inv_limit_override=8)
    ctrl.enable_learning = False
    ctrl.epsilon = 0.0
    ctrl.q_net.eval()
    ctrl.target_net.eval()
    ctrl.q_net.requires_grad_(False)
    meta = torch.load(path, map_location='cpu', weights_only=True)['meta']
    if algorithm in ('B', 'C'):
        assert meta['USE_MIXTURE'] is False
        assert meta['BAYES_FLOW_PRIOR_MODE'] == 'beta'
        assert meta['BAYES_FLOW_TAU_R'] == 60
    return ctrl, meta


def evaluation_helpers(meta):
    import paper_helpers as helpers
    helpers.MO_EVENT_FRACTION = float(meta.get('MO_EVENT_FRACTION', .03488))
    return helpers, helpers.checkpoint_signal_config(meta)


def make_schedule(helpers, family, seed, steps, tau=60):
    if family == 'stationary' or (family == 'sweep' and tau == 0):
        return .5
    if family in ('sweep', 'recovery'):
        schedule_seed = seed + 1000000 + round(10000 * tau) if family == 'sweep' else seed + 999999
        return helpers.build_regime_schedule(total_events=steps, seed=schedule_seed,
                                             distribution='exponential', tau=float(tau))
    schedule_seed = seed + 20000000
    if family == 'random':
        return helpers.make_random_tau_regime_schedule(steps, schedule_seed,
                                                       [15, 30, 60, 120, 240], .2, .8)
    if family == 'correlated':
        return helpers.make_clustered_intensity_regime_schedule(
            steps, schedule_seed, [15, 30, 60, 120, 240], .85, None, 0., .3)
    raise ValueError(f'Unknown scenario: {family}')
