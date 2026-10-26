#!/usr/bin/env python3
"""Verify a clean export with access to the original research workspace blocked."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/portability')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Check identities only; skip generating and comparing figures.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'commands': [], 'original_workspace_access': 'blocked'}
    spec = json.loads((ROOT / 'config/paper_experiments.json').read_text())
    with tempfile.TemporaryDirectory(prefix='paper-reproduction-') as temp:
        base = Path(temp)
        exported = base / 'export'
        exported.mkdir()
        for directory in ('src', 'config', 'data/paper', 'paper/figures'):
            shutil.copytree(ROOT / directory, exported / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (exported / 'repro').mkdir()
        for script in (ROOT / 'repro').glob('*.py'):
            shutil.copy2(script, exported / 'repro' / script.name)
        # Supported GLFT helpers live in a package; local audit directories do not.
        if (ROOT / 'repro/glft').is_dir():
            shutil.copytree(ROOT / 'repro/glft', exported / 'repro/glft',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('Makefile', 'requirements.txt', 'requirements-tested.txt',
                     'README.md', 'data/README.md', 'repro/README.md'):
            shutil.copy2(ROOT / name, exported / name)
        (exported / 'checkpoints').mkdir()
        checkpoints = {entry['checkpoint'] for entry in spec['models'].values()}
        checkpoints |= {entry['checkpoint'] for entry in spec['frontier_checkpoints'].values()}
        for checkpoint in checkpoints:
            shutil.copy2(ROOT / checkpoint, exported / checkpoint)

        hooks = base / 'audit_hooks'
        hooks.mkdir()
        # Audit both file opens and directory operations. Python/site libraries
        # stay accessible; inputs must come from the exported repository.
        (hooks / 'sitecustomize.py').write_text('''import os, sys
blocked = tuple(os.path.realpath(p) for p in os.environ['PAPER_BLOCK_PATHS'].split(os.pathsep))
def audit(event, args):
    if event in ('open', 'os.listdir', 'os.scandir', 'os.chdir') and args:
        value = args[0]
        if isinstance(value, (str, bytes)):
            path = os.path.realpath(os.fsdecode(value))
            if any(path == root or path.startswith(root + os.sep) for root in blocked):
                raise RuntimeError('External research input accessed: ' + path)
sys.addaudithook(audit)
''')
        blocked = [str(ROOT)]
        desktop = Path.home() / 'Desktop'
        if desktop.exists():
            blocked.append(str(desktop))
        env = dict(os.environ, PYTHONPATH=str(hooks), MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(base / 'matplotlib'),
                   PAPER_BLOCK_PATHS=os.pathsep.join(blocked), PYTHONDONTWRITEBYTECODE='1')
        guard_check = ('from pathlib import Path\n'
                       'try:\n'
                       f'    Path({str(ROOT / "README.md")!r}).read_bytes()\n'
                       'except RuntimeError as error:\n'
                       '    assert "External research input accessed" in str(error)\n'
                       '    print("Original-workspace access is blocked")\n'
                       'else:\n'
                       '    raise AssertionError("Workspace guard is not active")\n')
        commands = [[sys.executable, '-c', guard_check],
                    [sys.executable, str(exported / 'repro/verify_paper.py')],
                    [sys.executable, str(exported / 'repro/verify_numerical_stability.py'),
                     '--output', str(base / 'numerical_regression')]]
        if not args.skip_figures:
            commands += [['make', '-C', str(exported), 'figures', f'PY={sys.executable}'],
                         [sys.executable, str(exported / 'repro/verify_paper.py'),
                          '--figures-dir', str(exported / 'outputs/paper_figures')],
                         [sys.executable, str(exported / 'repro/frontier.py'), 'evaluate',
                          '--seeds', '1', '--compare-archive', '--output', str(base / 'frontier_smoke')],
                         [sys.executable, str(exported / 'repro/glft_calibration.py'),
                          '--output', str(base / 'calibration_smoke')],
                         [sys.executable, str(exported / 'repro/glft_evaluate.py'),
                          '--output', str(base / 'glft_evaluation_smoke')],
                         [sys.executable, str(exported / 'repro/plot_policy_results.py'),
                          '--verify-rerun', '--output', str(base / 'policy_rerun')],
                         [sys.executable, str(exported / 'repro/plot_paper_episode.py'),
                          '--verify-rerun', '--output', str(base / 'episode_rerun')]]
            commands.append([sys.executable, str(exported / 'repro/evaluate_corrected.py'),
                             '--case','correlated_C','--seeds','1','--verify-archive',
                             '--output',str(base/'C_corrected_replay')])
        for number, command in enumerate(commands):
            print('Checking:', ' '.join(command), flush=True)
            result = subprocess.run(command, cwd=base, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = args.output / f'command_{number + 1}.log'
            log.write_text(result.stdout)
            report['commands'].append({'command': [str(x).replace(str(exported), '<export>')
                                                     for x in command],
                                       'exit_code': result.returncode, 'log': log.name})
            if result.returncode:
                report['status'] = 'failed'
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                raise RuntimeError(f'Export check failed; see {log}')
        report['status'] = 'passed'
        report['figures_regenerated'] = not args.skip_figures
        report['checkpoints_copied'] = len(checkpoints)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Portable export verified:', args.output / 'report.json')


if __name__ == '__main__':
    main()
