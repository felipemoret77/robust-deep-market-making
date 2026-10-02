#!/usr/bin/env python3
"""Check checkpoint identity, signal settings, archived data and helper provenance."""
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from paper_runtime import ROOT, checkpoint, file_hash, manifest


def main():
    spec = manifest()
    for algorithm, info in spec['models'].items():
        saved = torch.load(checkpoint(algorithm), map_location='cpu', weights_only=True)
        assert json.loads(json.dumps(saved['meta'])) == info['metadata'], f'{algorithm}: metadata changed'
        if algorithm in ('B', 'C'):
            assert saved['meta']['USE_MIXTURE'] is False
            assert saved['meta']['BAYES_FLOW_PRIOR_MODE'] == 'beta'
            assert saved['meta']['BAYES_FLOW_TAU_R'] == 60
        print(f'{algorithm}: checkpoint hash and metadata verified')
    assert spec['models']['B']['metadata']['WARMSTART_CKPT'] is None
    assert Path(spec['models']['C']['metadata']['WARMSTART_CKPT']).name == Path(spec['models']['B']['checkpoint']).name
    for label, info in spec['frontier_checkpoints'].items():
        path = ROOT / info['checkpoint']
        assert file_hash(path) == info['sha256'], label
        saved_meta = torch.load(path, map_location='cpu', weights_only=True)['meta']
        assert json.loads(json.dumps(saved_meta)) == info['metadata'], label
    print('Frontier checkpoints: hashes and metadata verified')

    data_dir = ROOT / 'data/paper'
    for name, info in json.loads((data_dir / 'data_manifest.json').read_text()).items():
        assert file_hash(data_dir / name) == info['sha256'], name
        with np.load(data_dir / name, allow_pickle=False) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
    rows = list(csv.DictReader((data_dir / 'tau_sweep_B_runs.csv').open()))
    summary = list(csv.DictReader((data_dir / 'tau_sweep_summary.csv').open()))
    assert len(rows) == 600
    for line in summary:
        subset = [r for r in rows if int(r['tau']) == int(line['tau'])]
        assert sorted(int(r['seed']) for r in subset) == list(range(20123, 20223))
        pnls = np.array([float(r['terminal_pnl']) for r in subset])
        np.testing.assert_allclose([pnls.mean(), pnls.std(ddof=1)/10],
                                   [float(line['B_new_mean_pnl']), float(line['B_new_sem_pnl'])],
                                   rtol=0, atol=1e-12)
    print('Archived data: hashes, 600 sweep outcomes, means and SEM verified')

    tree = ast.parse((ROOT / 'repro/paper_helpers.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    provenance = json.loads((data_dir / 'helper_provenance.json').read_text())
    for item in provenance['functions']:
        body = ast.unparse(functions[item['function']])
        assert hashlib.sha256(body.encode()).hexdigest() == item['ast_sha256'], item['function']
    print('Evaluation helpers: unchanged function bodies verified')
    print('This check does not retrain models or validate every historical figure.')


if __name__ == '__main__':
    main()
