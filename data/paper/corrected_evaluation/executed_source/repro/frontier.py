#!/usr/bin/env python3
"""Replot the selected paper frontier, or reevaluate the legacy 1000-seed protocol.

Examples (from the repository root):
  python repro/frontier.py plot --output outputs/frontier
  python repro/frontier.py evaluate --seeds 1 --compare-archive --output outputs/frontier_smoke
  python repro/frontier.py evaluate --seeds 1000 --output outputs/frontier_rerun

Evaluation is sequential on CPU, with one numerical thread. Plotting requires
only NumPy/Matplotlib; evaluation additionally uses the shipped simulator and
PyTorch checkpoints. Outputs never replace the paper figure or archived data.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from functools import partial
import gzip
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Optional

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import paper_runtime as runtime

ROOT = runtime.ROOT
CONFIG = ROOT / "config/frontier_fresh_seeds.json"
ARCHIVE = ROOT / "data/paper/frontier"
FIELDS = ("policy", "family", "parameter", "seed", "terminal_pnl",
          "terminal_inventory", "mean_abs_inventory")
METRICS = FIELDS[4:]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config():
    config = json.loads(CONFIG.read_text())
    provenance = json.loads((ARCHIVE / "manifest.json").read_text())
    if runtime.file_hash(CONFIG) != provenance["config_sha256"]:
        raise ValueError("Frontier config differs from the archived experiment.")
    return config, provenance


def load_archive(config, provenance):
    import hashlib
    path = ARCHIVE / provenance["outcomes"]["file"]
    if runtime.file_hash(path) != provenance["outcomes"]["sha256"]:
        raise ValueError("Archived frontier CSV hash differs.")
    data = gzip.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != provenance["outcomes"]["uncompressed_sha256"]:
        raise ValueError("Uncompressed frontier CSV hash differs.")
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    policies = [p["name"] for p in config["policies"]]
    seeds = range(config["seed_start"], config["seed_start"] + config["seeds"])
    expected = {(p, s) for p in policies for s in seeds}
    observed = {(r["policy"], int(r["seed"])) for r in rows}
    if len(rows) != provenance["runs"] or observed != expected or len(observed) != len(rows):
        raise ValueError("Archived frontier must contain exactly 1000 seeds for each of 11 policies.")
    if not np.isfinite([[float(r[m]) for m in METRICS] for r in rows]).all():
        raise ValueError("Non-finite archived outcome.")
    return rows


def prepare_output(path):
    path = Path(path).resolve()
    for protected in (ROOT / "paper", ROOT / "data", ROOT / "config", ROOT / "src", ROOT / "checkpoints", ROOT / "repro"):
        if path == protected or protected in path.parents:
            raise ValueError(f"Use a separate output directory, not {protected}.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot(output):
    from plot_frontier_10000 import plot as selected_plot
    return selected_plot(output)


def plot_legacy(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, provenance = load_config()
    rows = load_archive(config, provenance)
    policies = config["policies"]
    lookup = {(r["policy"], int(r["seed"])): float(r["terminal_pnl"]) for r in rows}
    seeds = range(config["seed_start"], config["seed_start"] + config["seeds"])
    matrix = np.array([[lookup[p["name"], seed] for p in policies] for seed in seeds])
    means, stds = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
    colors = {"RLMM": "#E69F00", "GLFT": "#0072B2", "at_best": "#999999"}
    labels = {"RLMM": "RLMM (Algorithm A)", "GLFT": "GLFT", "at_best": "At-best baseline"}
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 7))
    for family, marker in (("GLFT", "o"), ("RLMM", "o"), ("at_best", "D")):
        indices = sorted([i for i, p in enumerate(policies) if p["family"] == family], key=lambda i: stds[i])
        xs, ys = stds[indices], means[indices]
        ax.errorbar(xs, ys, xerr=xs / np.sqrt(2 * (len(matrix) - 1)),
                    yerr=xs / np.sqrt(len(matrix)), color=colors[family], marker=marker,
                    linestyle="none" if family == "at_best" else "-", capsize=3,
                    linewidth=1.5, markersize=9 if family == "at_best" else 7,
                    alpha=1., label=labels[family], zorder=3)
        if family == "at_best":
            continue
        duplicates = {}
        for i in indices:
            duplicates.setdefault((round(float(stds[i]), 12), round(float(means[i]), 12)), []).append(i)
        for (x, y), members in duplicates.items():
            if family == "RLMM":
                label = r"$\varphi=" + f"{policies[members[0]]['phi']:.3f}" + "$"
            else:
                powers = [r"10^{" + str(round(math.log10(policies[i]["gamma"]))) + "}" for i in members]
                label = (r"$\gamma=" + powers[0] + "$" if len(members) == 1 else
                         r"$\gamma\in\{" + ",".join(powers) + r"\}$")
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10, color=colors[family])
    ax.set_xlabel(r"Risk: $\sqrt{\mathrm{Var}(\mathrm{PnL}_T)}$", fontsize=13)
    ax.set_ylabel(r"Return: $\mathbb{E}[\mathrm{PnL}_T]$", fontsize=13)
    ax.grid(alpha=.3)
    ax.margins(x=.18, y=.10)
    ax.legend(fontsize=13)
    fig.text(.5, .015, "Fresh seeds 9000000–9000999; N=1000; mean ±1 SEM; SD ±approximate 1 SE", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"glft_frontier_complete.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy", "n", "mean_pnl", "sd_pnl", "sem_pnl", "se_sd_approx"))
        writer.writeheader()
        for i, policy in enumerate(policies):
            writer.writerow(dict(policy=policy["name"], n=len(matrix), mean_pnl=means[i], sd_pnl=stds[i],
                                 sem_pnl=stds[i] / np.sqrt(len(matrix)), se_sd_approx=stds[i] / np.sqrt(2 * (len(matrix) - 1))))
    actual_hash = runtime.file_hash(output / "glft_frontier_complete.png")
    write_json(output / "plot_manifest.json", dict(
        archive_sha256=provenance["outcomes"]["sha256"], config_sha256=runtime.file_hash(CONFIG),
        script_sha256=runtime.file_hash(__file__), numpy=np.__version__, matplotlib=matplotlib.__version__,
        rendered_png_sha256=actual_hash, published_png_sha256=provenance["published_figure"]["sha256"],
        png_byte_identical=actual_hash == provenance["published_figure"]["sha256"],
        note="Exact raster bytes can depend on library/font versions; the pinned environment reproduces them. PDF metadata can differ."))
    print(f"Replotted {len(rows)} archived outcomes in {output}; PNG byte-identical: {actual_hash == provenance['published_figure']['sha256']}")


def load_controller(spec, output):
    """Same AST-extracted shipped factory as paper_runtime, for frontier A files."""
    import torch
    from dqn_distributional_with_throttle import DeepRLController
    path = ROOT / spec["checkpoint"]
    if runtime.file_hash(path) != spec["sha256"]:
        raise ValueError(f"Checkpoint hash differs: {path}")
    canonical = [p for p in runtime.manifest()["frontier_checkpoints"].values() if p["checkpoint"] == spec["checkpoint"]]
    if len(canonical) != 1 or canonical[0]["sha256"] != spec["sha256"]:
        raise ValueError("Frontier checkpoint does not match the canonical paper manifest.")
    factory_path = ROOT / "src/glft/MM_GLFT_naive_comparison.py"
    nodes = [n for n in ast.parse(factory_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "make_controller_from_checkpoint"]
    if len(nodes) != 1:
        raise RuntimeError("Expected one checkpoint factory.")
    namespace = dict(torch=torch, Optional=Optional, DeepRLController=partial(DeepRLController, device=torch.device("cpu")))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(factory_path), "exec"), namespace)
    ctrl = namespace["make_controller_from_checkpoint"](str(path), log_dir=str(output),
            use_time_update=True, min_time_interval=1., inv_limit_override=8)
    ctrl.q_net.requires_grad_(False)
    ctrl.target_net.requires_grad_(False)
    if ctrl.enable_learning or ctrl.epsilon != 0 or ctrl.q_net.training or ctrl.target_net.training:
        raise RuntimeError("Controller is not in frozen evaluation mode.")
    return ctrl


def simulate(config, spec, seed, events, warmup, ctrl=None):
    import torch
    from MM_LOB_SIM import simulate_LOB_with_MM, reward_spread_capture_inv_quadratic
    from GLFT_policy_factory import glft_policy_factory
    from MM_policy_5 import always_best_bid_ask_mm_policy_factory
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = dict(config["simulator"], iterations=events, iterations_to_equilibrium=warmup, random_seed=seed)
    if ctrl is not None:
        def reward(step, mm, lob, before, after, info, **extra):
            reward_spread_capture_inv_quadratic(step, mm, lob, before, after, info,
                                               inv_penalty_coeff=spec["phi"], **extra)
            return 0.
        kwargs.update(controller=ctrl, reward_fn=reward)
    elif spec["family"] == "at_best":
        kwargs["mm_policy"] = always_best_bid_ask_mm_policy_factory(**config["policy"])
    else:
        kwargs["mm_policy"] = glft_policy_factory(**config["parameters"], gamma=spec["gamma"],
                                                 **config["glft_policy"], **config["policy"])
    metrics = simulate_LOB_with_MM(**kwargs)[2]
    return dict(terminal_pnl=float(metrics["MM_TotalPnL"]), terminal_inventory=float(metrics["MM_Inventory"]),
                mean_abs_inventory=float(metrics["mean_abs_inventory"]))


def evaluate(args, output):
    existing = [name for name in ("manifest.json", "per_run.csv", "archive_comparison.json")
                if (output / name).exists() or (output / name).is_symlink()]
    if existing:
        raise FileExistsError(
            f"Evaluation output already contains {', '.join(existing)}; use a fresh --output directory.")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, provenance = load_config()
    policies = [p for p in config["policies"] if not args.policies or p["name"] in args.policies]
    if not policies or (args.policies and set(args.policies) != {p["name"] for p in policies}):
        raise ValueError("Unknown policy name; use names from config/frontier_fresh_seeds.json.")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for name, expected in provenance["sources"].items():
        if runtime.file_hash(ROOT / name) != expected:
            raise ValueError(f"Shipped source differs from the archived run: {name}")
    for spec in policies:
        if "declared_training_seeds" in spec:
            train = spec["declared_training_seeds"]
            if set(seeds).intersection(range(train["start"], train["start"] + train["count"])):
                raise ValueError("Evaluation seeds overlap the checkpoint's declared training seed block.")
    expected = None
    if args.compare_archive:
        if args.events != config["events"] or args.warmup != config["warmup"]:
            raise ValueError("Archive comparison requires the paper's 5000 events and 1000 warmup.")
        expected = {(r["policy"], int(r["seed"])): r for r in load_archive(config, provenance)}
        if any((p["name"], s) not in expected for p in policies for s in seeds):
            raise ValueError("Archive comparison requires seeds in 9000000–9000999.")
    manifest = dict(completed=False, seeds=seeds, events=args.events, warmup=args.warmup, policies=policies,
        config=config, config_sha256=runtime.file_hash(CONFIG), source_hashes=provenance["sources"],
        runner_sha256=runtime.file_hash(__file__), paper_runtime_sha256=runtime.file_hash(runtime.__file__),
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        weights_frozen=True, paired_seeds=True, training_seed_overlap=0, workers=1)
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    checks, rows = [], []
    with (output / "per_run.csv").open("w", newline="") as file, (output / "evaluation.log").open("w") as log:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for spec in policies:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                ctrl = load_controller(spec, output / "tensorboard" / spec["name"]) if spec["family"] == "RLMM" else None
                before = {name: {k: v.detach().clone() for k, v in getattr(ctrl, name).state_dict().items()}
                          for name in ("q_net", "target_net")} if ctrl else {}
                try:
                    for seed in seeds:
                        result = simulate(config, spec, seed, args.events, args.warmup, ctrl)
                        if not all(math.isfinite(v) for v in result.values()):
                            raise ValueError("Non-finite simulated outcome.")
                        if expected is not None:
                            reference = expected[spec["name"], seed]
                            errors = {m: abs(result[m] - float(reference[m])) for m in METRICS}
                            checks.append(dict(policy=spec["name"], seed=seed, errors=errors,
                                               exactly_equal=all(error == 0 for error in errors.values())))
                            if max(errors.values()) > 1e-9:
                                raise AssertionError(f"Archived outcome differs: {checks[-1]}")
                        row = dict(policy=spec["name"], family=spec["family"], parameter=spec.get("phi", spec.get("gamma", "")), seed=seed, **result)
                        writer.writerow(row)
                        file.flush()
                        rows.append(row)
                    if ctrl:
                        for name, state in before.items():
                            if not all(torch.equal(state[k], v) for k, v in getattr(ctrl, name).state_dict().items()):
                                raise AssertionError(f"{name} state changed during evaluation.")
                finally:
                    if ctrl:
                        ctrl.writer.close()
            print(f"Completed {spec['name']}: {len(seeds)} episodes ({len(rows)}/{len(seeds) * len(policies)} total)", flush=True)
    manifest.update(completed=True, runs=len(rows), elapsed_seconds=time.monotonic() - started,
                    per_run_sha256=runtime.file_hash(output / "per_run.csv"), weights_unchanged=True)
    write_json(output / "manifest.json", manifest)
    if expected is not None:
        write_json(output / "archive_comparison.json", dict(passed=True, tolerance=1e-9,
                   checked_runs=len(checks), all_exactly_equal=all(c["exactly_equal"] for c in checks), checks=checks))
    print(f"Completed frozen-policy evaluation in {manifest['elapsed_seconds']:.1f}s: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plotting = commands.add_parser("plot", aliases=["replot"], help="Replot the selected 110000 archived outcomes without simulation")
    plotting.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Reevaluate the legacy 1000-seed protocol on CPU (not the selected corrected frontier)")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--seeds", type=int, default=1000, help="Number of consecutive paired seeds")
    evaluation.add_argument("--seed-start", type=int, default=9000000)
    evaluation.add_argument("--events", type=int, default=5000)
    evaluation.add_argument("--warmup", type=int, default=1000)
    evaluation.add_argument("--policies", nargs="+", help="Optional policy names from the frontier config")
    evaluation.add_argument("--compare-archive", action="store_true", help="Compare all three outcomes against archived seeds (absolute tolerance 1e-9)")
    args = parser.parse_args()
    if args.command == "evaluate" and (args.seeds <= 0 or args.events <= 0 or args.warmup < 0 or args.seed_start < 0 or args.seed_start + args.seeds > 2**32):
        parser.error("Require positive seeds/events, nonnegative warmup, and seeds below 2**32.")
    output = prepare_output(args.output)
    if args.command in ("plot", "replot"):
        plot(output)
    else:
        evaluate(args, output)


if __name__ == "__main__":
    main()
