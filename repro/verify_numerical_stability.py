#!/usr/bin/env python3
"""Regression tests for frozen C51 action ranking (no training).

The optional full episode reproduces the diagnosed B/500128 case using the
current shared controller, without forcing any action or applying FIFO changes.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch
import paper_runtime as runtime
from c51_evaluation import action_scores_from_logits
from dqn_distributional_with_throttle import DeepRLController
from evaluation_environment import check_environment, environment_report

FIXTURES = runtime.ROOT / 'data/paper/numerical_regression'


def digest(ctrl):
    result = hashlib.sha256()
    for network in ('q_net', 'target_net'):
        for name, value in getattr(ctrl, network).state_dict().items():
            result.update((network + name).encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class NumericalTests(unittest.TestCase):
    def test_fixture_identity(self):
        info = json.loads((FIXTURES / 'provenance.json').read_text())
        for name, expected in info['files'].items():
            self.assertEqual(runtime.file_hash(FIXTURES / name), expected)

    def test_real_near_tie_uses_logits_before_softmax(self):
        with np.load(FIXTURES / 'predictions.npz', allow_pickle=False) as f:
            logits = torch.from_numpy(f['reference_logits'])
            support = torch.from_numpy(f['support'])
            actual = action_scores_from_logits(logits, support)
            self.assertEqual(actual.dtype, torch.float64)
            self.assertEqual(actual.argmax(-1).item(), 5)
            torch.testing.assert_close(actual, torch.from_numpy(f['reference_stable']),
                                       rtol=0, atol=2e-12)
            # Casting the archived rounded probabilities is a different operation.
            rounded = (torch.from_numpy(f['reference_probs']).double() *
                       (support.double() - support[0].double())).sum(-1)
            self.assertGreater(float(torch.max(torch.abs(actual - rounded))), 1e-9)

    def test_exact_tie_and_support_translation(self):
        logits = torch.tensor([[[0., 1., 2.], [0., 1., 2.]]])
        z = torch.tensor([-1., 0., 1.])
        first = action_scores_from_logits(logits, z)
        shifted = action_scores_from_logits(logits, z + 2.)
        self.assertTrue(torch.equal(first, shifted))
        self.assertEqual(first.argmax(-1).item(), 0)

    def test_training_and_scalar_models_keep_existing_calculation(self):
        for learning, distributional in ((True, True), (True, False), (False, False)):
            sentinel = torch.tensor([[1., 2.]])
            ctrl = SimpleNamespace(enable_learning=learning, use_distributional=distributional,
                                   _q_values_from_net=lambda net, state: sentinel)
            with patch('dqn_distributional_with_throttle.action_scores_from_logits',
                       side_effect=AssertionError('Stable evaluation leaked into training')):
                self.assertIs(DeepRLController._action_scores_from_net(ctrl, None, None), sentinel)

    def test_frozen_mode_rejects_noisy_training_network(self):
        ctrl = SimpleNamespace(enable_learning=False, use_distributional=True)
        with self.assertRaisesRegex(RuntimeError, 'net.eval'):
            DeepRLController._action_scores_from_net(ctrl, SimpleNamespace(training=True), None)

    def test_learning_and_unshifted_value_methods_are_unchanged(self):
        old = ast.parse(runtime.LEGACY_CONTROLLER.read_text())
        new = ast.parse((runtime.ROOT / 'src/agents/dqn_distributional_with_throttle.py').read_text())
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepRLController')
            return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
        a, b = methods(old), methods(new)
        changed = {name for name in a if a[name] != b[name]}
        self.assertEqual(changed, {'_epsilon_greedy_action', '_epsilon_greedy_action_batch_from_net'})
        # In these two methods the only change is routing the ranking call.
        for name in changed:
            self.assertEqual(a[name].replace("attr='_q_values_from_net'", "attr='_action_scores_from_net'"), b[name])

    def test_environment_drift_is_rejected_and_recorded(self):
        bad = dict(actual={}, matches_reference=False,
                   differences={'torch': {'expected': '2.10.0', 'actual': 'future'}})
        with patch('evaluation_environment.environment_report', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'Unvalidated evaluation environment'):
                check_environment()
            self.assertTrue(check_environment(allow_mismatch=True)['mismatch_explicitly_allowed'])

    def test_abc_checkpoints_actions_masks_and_unshifted_values(self):
        with tempfile.TemporaryDirectory() as folder, np.load(FIXTURES / 'action_reference.npz', allow_pickle=False) as f:
            for algorithm in ('A', 'B', 'C'):
                with self.subTest(algorithm=algorithm), contextlib.redirect_stdout(io.StringIO()):
                    ctrl, _ = runtime.load_controller(algorithm, Path(folder) / algorithm)
                    before = digest(ctrl)
                    try:
                        for index, state in enumerate(f[algorithm + '_states']):
                            x = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                            for q_index, q in enumerate((-8, 0, 8)):
                                mask = ctrl._get_valid_action_mask({'inventory': q})
                                self.assertEqual(int(mask.sum()), 6 if q == 0 else 3)
                                for noisy in (True, False):
                                    ctrl.use_noisy_net = noisy
                                    a = ctrl._epsilon_greedy_action(x, mask)
                                    self.assertTrue(mask[a])
                                    self.assertEqual(a, int(f[algorithm + '_actions'][index, q_index]))
                                batch_a = ctrl._epsilon_greedy_action_batch_from_net(
                                    ctrl.q_net, x, torch.tensor(mask).reshape(1, -1))
                                self.assertEqual(batch_a.item(), a)
                            with torch.no_grad():
                                # The Q used in heatmaps/logging is still E[Z], not E[Z-Zmin].
                                old_q = (ctrl.q_net(x) * ctrl.support.view(1, 1, -1)).sum(-1)
                                self.assertTrue(torch.equal(ctrl._q_values_from_net(ctrl.q_net, x), old_q))
                        self.assertEqual(digest(ctrl), before)
                        self.assertFalse(ctrl.enable_learning)
                    finally:
                        ctrl.writer.close()


def full_episode(output):
    with (output / 'episode.log').open('w') as log, contextlib.redirect_stdout(log):
        ctrl, meta = runtime.load_controller('B', output / 'tensorboard')
        before = digest(ctrl)
        checkpoint_before = runtime.file_hash(runtime.checkpoint('B'))
        try:
            helpers, signals = runtime.evaluation_helpers(meta)
            schedule = runtime.make_schedule(helpers, 'random', 500128, 500000)
            with torch.inference_mode():
                _, metrics = helpers.run_one_sim(schedule, seed=500128, controller=ctrl,
                    total_events=500000, equil_steps=10000, signal_config=signals, metrics_only=True)
            actual = dict(terminal_pnl=float(metrics['MM_TotalPnL']),
                          terminal_inventory=float(metrics['MM_Inventory']),
                          mean_abs_inventory=float(metrics['mean_abs_inventory']),
                          inventory_limit_fraction=float(metrics['limit_frac']))
            expected = json.loads((FIXTURES / 'candidate_current.json').read_text())
            errors = {key: abs(value - expected[key]) for key, value in actual.items()}
            unchanged = before == digest(ctrl) and checkpoint_before == runtime.file_hash(runtime.checkpoint('B'))
            return dict(actual=actual, errors=errors, seed=500128, events=500000, warmup=10000,
                        passed=max(errors.values()) < 1e-9 and unchanged,
                        weights_unchanged=unchanged, forced_actions=0, fifo_intervention=False)
        finally:
            ctrl.writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-episode', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  environment=environment_report(), mode='stable_float64_v1')
    if result.wasSuccessful() and args.full_episode:
        report['full_episode'] = full_episode(args.output)
        report['passed'] = report['passed'] and report['full_episode']['passed']
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Numerical regression report:', args.output / 'report.json', flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
