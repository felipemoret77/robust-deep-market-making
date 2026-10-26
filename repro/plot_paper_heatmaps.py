"""Regenerate the paper's three masked value slices from frozen canonical B.

Run ``python repro/plot_paper_heatmaps.py --output outputs/paper_heatmaps
--verify-reference`` to also compare every numeric array with the archived
final heatmaps. The default comparison is exact; --atol/--rtol can allow
explicit floating-point tolerances on value arrays on another platform.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import paper_runtime


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'data/paper/heatmaps'
CANONICAL_B_SHA256 = '3a4649c860f91ffe2ca3747d071880508e953aa45f5a0d14d6b18cdfcff30200'
BIAS_LABEL = r'Bayesian directional bias $\widehat{\iota}$'


def panel_specs():
    return [
        ('p_buy_mhat_vs_inv', 'bayes_m_hat', np.linspace(-1, 1, 41),
         'inventory', np.arange(-8, 9), BIAS_LABEL, 'Inventory'),
        ('inventory_vs_spread', 'spread', np.linspace(1, 5, 41),
         'inventory', np.arange(-8, 9), 'Spread (ticks)', 'Inventory'),
        ('bayes_m_vs_spread', 'spread', np.linspace(1, 5, 41),
         'bayes_m_hat', np.linspace(-1, 1, 41), 'Spread (ticks)', BIAS_LABEL),
    ]


def evaluate_slices(ctrl, base):
    """Use the controller's state transform, Q expectation, and action mask."""
    arrays, stats = {}, {}
    with torch.inference_mode():
        for name, x, xs, y, ys, _, _ in panel_specs():
            z = np.empty((len(ys), len(xs)), dtype=np.float32)
            raw = np.empty_like(z)
            counts = np.empty_like(z, dtype=np.int8)
            for iy, yv in enumerate(ys):
                for ix, xv in enumerate(xs):
                    state = copy.deepcopy(base)
                    state[x], state[y] = float(xv), float(yv)
                    q_values = ctrl._q_values_from_net(
                        ctrl.q_net, ctrl._state_to_tensor(state)).reshape(-1)
                    mask = torch.as_tensor(ctrl._get_valid_action_mask(state),
                                           dtype=torch.bool, device=q_values.device)
                    expected_count = 3 if abs(state['inventory']) >= ctrl.inv_limit else 6
                    if mask.shape != q_values.shape or int(mask.sum()) != expected_count:
                        raise ValueError(f'Unexpected action mask in {name}: {state}')
                    raw[iy, ix] = q_values.max().item()
                    z[iy, ix] = q_values[mask].max().item()
                    counts[iy, ix] = int(mask.sum())
            if not np.isfinite(z).all() or not np.isfinite(raw).all():
                raise ValueError(f'Non-finite Q values in {name}')
            if not np.array_equal(z[counts == 6], raw[counts == 6]) or not np.all(z <= raw):
                raise ValueError(f'Unexpected masked value relationship in {name}')
            arrays.update({name + '_x': xs, name + '_y': ys,
                           name + '_value': z, name + '_unmasked_value': raw,
                           name + '_retained_action_count': counts})
            stats[name] = dict(min=float(z.min()), max=float(z.max()), mean=float(z.mean()))
    return arrays, stats


def verify_reference(arrays, provenance, *, atol=0.0, rtol=0.0):
    path = REFERENCE / 'heatmap_data.npz'
    expected_sha256 = provenance['outputs_sha256'][path.name]
    if paper_runtime.file_hash(path) != expected_sha256:
        raise ValueError('Archived heatmap_data.npz hash differs from its provenance')
    comparisons = {}
    with np.load(path, allow_pickle=False) as archived:
        if set(arrays) != set(archived.files):
            raise ValueError('Generated and archived heatmap array names differ')
        for name, actual in arrays.items():
            expected = archived[name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f'Archived shape/dtype mismatch for {name}')
            if name.endswith('_value'):
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                                           err_msg=f'Archived values differ: {name}')
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=name)
            comparisons[name] = dict(exact=bool(np.array_equal(actual, expected)),
                                     max_abs_difference=float(np.max(np.abs(actual - expected))))
    return dict(atol=atol, rtol=rtol, arrays=comparisons)


def plot_slices(arrays, output):
    """Preserve the final paper panels' size, labels, scales, and rendering."""
    paths = []
    for name, _, xs, _, ys, xlabel, ylabel in panel_specs():
        fig, ax = plt.subplots(figsize=(8.5, 6.4))
        im = ax.imshow(arrays[name + '_value'], origin='lower', aspect='auto',
                       extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
                       cmap='viridis')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Value V(s)', fontsize=16)
        cb.ax.tick_params(labelsize=13)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        ax.grid(False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            path = output / f'{name}.{ext}'
            fig.savefig(path, dpi=240, bbox_inches='tight')
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.3), layout='constrained')
    for index, (ax, spec) in enumerate(zip(axes, panel_specs())):
        name, _, xs, _, ys, xlabel, ylabel = spec
        im = ax.imshow(arrays[name + '_value'], origin='lower', aspect='auto',
                       extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
                       cmap='viridis')
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title('(' + chr(97 + index) + ')', loc='left', fontsize=14)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cb.set_label('Value V(s)')
    for ext in ('png', 'pdf'):
        path = output / f'Heatmaps_B_mascara.{ext}'
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_heatmaps')
    parser.add_argument('--verify-reference', action='store_true',
                        help='Compare all 15 generated arrays with the archived final arrays.')
    parser.add_argument('--atol', type=float, default=0.0, help='Value comparison absolute tolerance.')
    parser.add_argument('--rtol', type=float, default=0.0, help='Value comparison relative tolerance.')
    args = parser.parse_args()
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error('--atol and --rtol must be finite and nonnegative')
    output = args.output.resolve()
    if output == REFERENCE.resolve():
        parser.error('--output must differ from the archived reference directory')
    output.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((REFERENCE / 'provenance.json').read_text())
    base_path = REFERENCE / 'reference_state.json'
    if paper_runtime.file_hash(base_path) != provenance['source_sha256']['data/paper/heatmaps/reference_state.json']:
        raise ValueError('Reference state hash differs from its provenance')
    base = json.loads(base_path.read_text())
    checkpoint = paper_runtime.checkpoint('B')
    if paper_runtime.file_hash(checkpoint) != CANONICAL_B_SHA256:
        raise ValueError('The paper heatmaps require the frozen canonical B checkpoint')
    torch.set_num_threads(1)
    with (output / 'controller_load.log').open('w') as log, contextlib.redirect_stdout(log):
        ctrl, metadata = paper_runtime.load_controller('B', output / 'runtime')
    try:
        if (ctrl.q_net.training or ctrl.target_net.training or ctrl.enable_learning
                or any(p.requires_grad for p in ctrl.q_net.parameters())):
            raise ValueError('Controller must be frozen and in evaluation mode')
        if ctrl.inv_limit != 8 or ctrl.n_actions != 6 or metadata['USE_MIXTURE'] is not False:
            raise ValueError('Unexpected canonical B architecture or inventory limit')
        retained = {label: np.flatnonzero(ctrl._get_valid_action_mask(dict(inventory=inventory))).tolist()
                    for label, inventory in [('long_cap', 8), ('short_cap', -8)]}
        if retained != {'long_cap': [0, 1, 4], 'short_cap': [0, 2, 5]}:
            raise ValueError(f'Unexpected canonical B boundary masks: {retained}')
        weights_before = {k: v.detach().clone() for k, v in ctrl.q_net.state_dict().items()}
        arrays, stats = evaluate_slices(ctrl, base)
        if not all(torch.equal(weights_before[k], v) for k, v in ctrl.q_net.state_dict().items()):
            raise ValueError('Controller weights changed during evaluation')
        if paper_runtime.file_hash(checkpoint) != CANONICAL_B_SHA256:
            raise ValueError('Checkpoint file changed during evaluation')
    finally:
        ctrl.writer.close()

    verification = (verify_reference(arrays, provenance, atol=args.atol, rtol=args.rtol)
                    if args.verify_reference else None)
    paths = plot_slices(arrays, output)
    arrays_path = output / 'heatmap_data.npz'
    np.savez_compressed(arrays_path, **arrays)
    paths.append(arrays_path)
    generated = dict(
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=CANONICAL_B_SHA256,
        algorithm='B', USE_MIXTURE=False, reference_state=base,
        evaluation=provenance['evaluation'], mask_method='_get_valid_action_mask',
        mask_semantics=provenance['mask_semantics'],
        retained_indices_at_long_cap=retained['long_cap'],
        retained_indices_at_short_cap=retained['short_cap'],
        age_coordinate=provenance['age_coordinate'], stats=stats,
        validation=dict(unchanged_weights=True, unchanged_checkpoint=True,
                        frozen_eval_mode=True, reference_comparison=verification),
        source_sha256={str(p.relative_to(ROOT)): paper_runtime.file_hash(p) for p in [
            Path(__file__).resolve(), ROOT / 'repro/paper_runtime.py',
            ROOT / 'src/agents/dqn_distributional_with_throttle.py',
            base_path, REFERENCE / 'provenance.json']},
        outputs_sha256={p.name: paper_runtime.file_hash(p) for p in paths},
        versions=dict(numpy=np.__version__, torch=torch.__version__, matplotlib=matplotlib.__version__),
    )
    (output / 'provenance.json').write_text(json.dumps(generated, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), checkpoint_sha256=CANONICAL_B_SHA256,
                          stats=stats, reference_comparison=verification), indent=2))


if __name__ == '__main__':
    main()
