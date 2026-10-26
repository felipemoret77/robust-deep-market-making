"""Resolve the repository's frozen simulator, never a separate checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in ('sim', 'glft'):
    sys.path.insert(0, str(ROOT / 'src' / folder))

import MM_LOB_SIM as simulator
import censored_waiting_times_calib as original

for module in (simulator, original):
    if not Path(module.__file__).resolve().is_relative_to(ROOT / 'src'):
        raise ImportError(f'Expected shipped source module, got {module.__file__}')
