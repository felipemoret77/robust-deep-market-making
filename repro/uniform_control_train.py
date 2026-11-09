#!/usr/bin/env python3
"""Repeat one arm of the matched scenario-allocation control.

The published C runner supplies the learner, reward and episode updates. This
wrapper changes only the paired seed streams, scenario allocation and output
paths. A smoke keeps the 2,000-episode schedules and runs their first 3 episodes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'config/uniform_control.json'


def instrument_runner(source: Path) -> tuple[str, list[dict]]:
    text = source.read_text()
    changes = []

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, (old, text.count(old))
        text = text.replace(old, new)
        changes.append({'old': old, 'new': new})

    replace('if __name__ == "__main__":', '''if __name__ == "__main__":
    from uniform_control_support import config as _control_config
    from uniform_control_support import before_training, record_episode, finish_training
    _pilot = _control_config()
    os.chdir(_pilot['arm_dir'])''')
    replace('    from DRO_adversary import DROHardPathAdversary',
            '    from uniform_control_support import PairedScenarioSampler as DROHardPathAdversary')
    replace('    _paper_root = _PaperPath(__file__).resolve().parents[2]',
            "    _paper_root = _PaperPath(_pilot['repository'])")
    replace('    WARMSTART_CKPT = str(_paper_root / _paper_spec["models"]["B"]["checkpoint"])',
            "    WARMSTART_CKPT = str(_paper_root / _pilot['source_B']['checkpoint'])")
    replace('    GLOBAL_SEED = 123', "    GLOBAL_SEED = int(_pilot['master_seed'])")
    start = text.index('    print("Running offline LOB simulation (no MM) just to inspect environment...")')
    end = text.index('    print("=== LOB SIMULATOR CONFIG ===")', start)
    text = text[:start] + '    # Display-only offline simulation omitted.\n\n' + text[end:]
    changes.append({'omit_display_only_offline_simulation': True})
    replace('    USE_EMPIRICAL_MO_FRAC_CALIBRATION = True',
            '    USE_EMPIRICAL_MO_FRAC_CALIBRATION = False')
    replace('    _MO_EVENT_FRACTION = float(_MO_EVENT_FRACTION_ANALYTIC)',
            '    _MO_EVENT_FRACTION = 0.03488')
    replace('    _MO_EVENT_FRACTION_SOURCE = "analytic"',
            '    _MO_EVENT_FRACTION_SOURCE = "fixed_canonical_C_empirical"')
    replace('    DRO_BETA_END = 0.5',
            "    DRO_BETA_END = 0.0 if _pilot['arm'] == 'uniform' else 0.5")
    replace('    DRO_EPS_UNIFORM_END = 0.5',
            "    DRO_EPS_UNIFORM_END = 1.0 if _pilot['arm'] == 'uniform' else 0.5")
    replace('    RUN_ROOT = BASE / "runs_spyder" / MODE_DIR',
            '    RUN_ROOT = Path.cwd() / "tensorboard"')
    replace('        shutil.rmtree(RUN_ROOT, ignore_errors=False)',
            '        raise FileExistsError(f"Refusing existing output: {RUN_ROOT}")')
    replace('    for ep in range(N_EPISODES):',
            "    before_training(deep_controller, globals(), dro_tau_adversary)\n\n    for ep in range(int(_pilot['episodes'])):")
    replace('        np.random.seed(current_ep_seed)',
            "        learning_ep_seed = int(_pilot['streams']['torch_and_python']) + ep\n        np.random.seed(learning_ep_seed)")
    replace('        random.seed(current_ep_seed)', '        random.seed(learning_ep_seed)')
    replace('        torch.manual_seed(current_ep_seed)', '        torch.manual_seed(learning_ep_seed)')
    replace('            torch.cuda.manual_seed(current_ep_seed)',
            '            torch.cuda.manual_seed(learning_ep_seed)')
    replace('        if USE_DRO_TAU and _episode_is_dro_tau and (dro_tau_adversary is not None):',
            '        record_episode(globals())\n        if USE_DRO_TAU and _episode_is_dro_tau and (dro_tau_adversary is not None):')
    replace('        ckpt = {\n            "meta": meta,',
            '        meta = dict(meta)\n        meta["MATCHED_BANDIT_PILOT"] = dict(_pilot)\n        ckpt = {\n            "meta": meta,')
    replace('f"Training completed: {N_EPISODES} episodes"',
            'f"Training completed: {ep + 1} episodes"')
    text += '\n    finish_training(globals())\n'
    ast.parse(text)
    return text, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('adaptive', 'uniform'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    training = protocol['training']
    assert training['episodes'] == 2000 and training['events'] == 5000
    assert training['warmup'] == 1000
    source_B = protocol['source_B']
    checkpoint = ROOT / source_B['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_B['sha256']
    output = args.output_root.resolve()
    protected = [ROOT / name for name in ('src', 'repro', 'config', 'data', 'paper', 'checkpoints')]
    if output == ROOT or any(output == path or path in output.parents for path in protected):
        parser.error('Choose a separate output directory')
    arm_dir = output / ('smoke' if args.smoke else 'runs/pilot_0') / args.arm
    if arm_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {arm_dir}')
    arm_dir.mkdir(parents=True)
    source = ROOT / 'src/agents/DeepSarsaQRunner_RANDOM_TAU.py'
    runner_text, changes = instrument_runner(source)
    runner = arm_dir / 'instrumented_C_runner.py'
    runner.write_text(runner_text)
    (arm_dir / 'runner_transformations.json').write_text(json.dumps(changes, indent=2) + '\n')
    import numpy as np
    roles = {'initial_pool': 10, 'incoming_candidates': 20, 'scenario_selection': 30,
             'pool_replacement': 40, 'torch_and_python': 50}
    seeds = {name: int(np.random.SeedSequence(training['master_seed'],
                      spawn_key=(training.get('replicate_id', 0), role)).generate_state(1)[0])
             for name, role in roles.items()}
    if 'streams' in training:
        assert seeds == training['streams']
    cfg = {'arm': args.arm, 'arm_dir': str(arm_dir), 'repository': str(ROOT),
           'master_seed': training['master_seed'], 'source_B': source_B,
           'episodes': 3 if args.smoke else training['episodes'],
           'schedule_episodes': training['episodes'], 'smoke': args.smoke,
           'streams': seeds, 'started_unix': time.time(), 'pid': os.getpid(),
           'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
           'adaptive_refresh_tie_rule': 'original NumPy argsort default'}
    config_path = arm_dir / 'experiment_config.json'
    config_path.write_text(json.dumps(cfg, indent=2) + '\n')
    os.environ.update(UNIFORM_CONTROL_CONFIG=str(config_path), PYTHONDONTWRITEBYTECODE='1',
                      MPLBACKEND='Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for folder in ('sim', 'agents', 'glft'):
        sys.path.insert(0, str(ROOT / 'src' / folder))
    sys.path.insert(0, str(ROOT / 'repro'))
    namespace = {'__name__': '__main__', '__file__': str(runner)}
    try:
        exec(compile(runner_text, str(runner), 'exec'), namespace)
    except BaseException as error:
        (arm_dir / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
