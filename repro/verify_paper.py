#!/usr/bin/env python3
"""Check checkpoint identity, signal settings, archived data and helper provenance."""
import ast
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from paper_runtime import ROOT, checkpoint, file_hash, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-dir', type=Path,
                        help='Also check generated figures against the selected manuscript PNGs.')
    args = parser.parse_args()
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
    heatmaps = json.loads((ROOT / 'data/paper/heatmaps/provenance.json').read_text())
    assert heatmaps['checkpoint_sha256'] == spec['models']['B']['sha256']
    assert heatmaps['USE_MIXTURE'] is False
    from plot_frontier_10000 import validate as validate_selected_frontier
    validate_selected_frontier()
    print('Selected frontier: 110000 outcomes; checkpoint hashes; fresh seeds; final summary identities verified')

    from plot_glft_regenerated import validate as validate_selected_glft
    validate_selected_glft()
    print('Selected GLFT panels: 4810 outcomes; seeds, calibration and summary statistics verified')

    from corrected_evaluation import validate as validate_corrected
    current=validate_corrected()
    print(f"Selected corrected evaluation: {current['episodes']} outcomes and {current['traces']} trajectories verified")

    from uniform_control import verify as verify_uniform_control
    verify_uniform_control()
    print('Matched adaptive/uniform control: 400 outcomes, final policies and paired summaries verified')

    data_dir = ROOT / 'data/paper'
    for name, info in json.loads((data_dir / 'data_manifest.json').read_text()).items():
        assert file_hash(data_dir / name) == info['sha256'], name
        if Path(name).suffix == '.npz':
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
    package = json.loads((ROOT / 'config/reproduction_manifest.json').read_text())
    for relative, expected in package['files'].items():
        path = ROOT / relative
        assert path.is_file(), f'Missing reproduction input: {relative}'
        assert file_hash(path) == expected, f'Reproduction input changed: {relative}'
    print(f"Reproduction package: {len(package['files'])} input/source hashes verified")
    inventory = json.loads((data_dir / 'figure_inventory.json').read_text())
    assert len({item['path'] for item in inventory}) == len(inventory), 'Duplicate figure inventory entry'
    for item in inventory:
        assert file_hash(ROOT / item['path']) == item['sha256'], item['path']
        if item.get('generation_provenance'):
            assert (ROOT / item['generation_provenance']).is_file(), item['path']
    portable = sum(bool(item.get('portable_replot')) for item in inventory)
    print(f'Figure coverage: {portable}/{len(inventory)} selected panels have portable generators')
    if args.figures_dir is not None:
        from PIL import Image
        exact = 0
        for item in inventory:
            if not item.get('portable_replot'):
                continue
            name = Path(item['path']).name
            generated = args.figures_dir / name
            assert generated.is_file(), f'Missing generated figure: {name}'
            with Image.open(generated) as first, Image.open(ROOT / item['path']) as second:
                equal = np.array_equal(np.asarray(first.convert('RGBA')),
                                       np.asarray(second.convert('RGBA')))
            if item.get('pixel_match_expected', True):
                assert equal, f'Figure pixels differ: {name}'
            if equal:
                exact += 1
            else:
                print(f'{name}: archived numeric inputs verified; documented style difference')
        print(f'Generated figures: {exact} exact pixel matches')
    print('These checks verify model identities, archived inputs and optional figure pixels; they do not rerun training or simulations.')


if __name__ == '__main__':
    main()
