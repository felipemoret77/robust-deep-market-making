"""Record and check the numerical environment before frozen evaluations."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


def environment_report():
    import torch
    expected = json.loads((ROOT / 'config/evaluation_environment.json').read_text())
    actual = dict(python=platform.python_version(), system=platform.system(),
                  machine=platform.machine(), device='cpu',
                  packages={name: importlib.metadata.version(name)
                            for name in expected['packages']},
                  torch_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(),
                  torch_build=torch.__config__.show())
    differences = {key: dict(expected=value, actual=actual[key])
                   for key, value in expected.items()
                   if key not in ('packages', 'description') and actual.get(key) != value}
    for name, value in expected['packages'].items():
        if actual['packages'][name] != value:
            differences[name] = dict(expected=value, actual=actual['packages'][name])
    return dict(actual=actual, matches_reference=not differences, differences=differences)


def check_environment(*, allow_mismatch=False):
    report = environment_report()
    if report['differences'] and not allow_mismatch:
        raise RuntimeError(
            'Unvalidated evaluation environment: ' + json.dumps(report['differences']) +
            '. Use requirements-tested.txt and the recorded Python/CPU environment. '
            'For an intentional compatibility test, pass --allow-environment-mismatch; '
            'this does not certify the new environment.')
    report['mismatch_explicitly_allowed'] = bool(allow_mismatch)
    return report
