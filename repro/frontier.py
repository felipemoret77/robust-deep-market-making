"""Replot or replay the final paper frontier: 10,000 seeds, 11 policies."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse, ast, contextlib, csv, gzip, io, json, random
from functools import partial
from pathlib import Path
from typing import Optional
import numpy as np
import paper_runtime as runtime
from paper_evaluation_corrections import instrument_controller, instrument_benchmark

ROOT = runtime.ROOT
ARCHIVE = ROOT / "data/paper/frontier_10000"


def evaluate(args):
    import torch
    from plot_frontier_10000 import validate

    validate()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((ARCHIVE / "campaign_manifest.json").read_text())["protocol"]
    config["parameters"] = json.loads((ARCHIVE / "calibration_used.json").read_text())[
        "parameters"
    ]
    seeds = config["seeds"][: args.seeds]
    if not 1 <= args.seeds <= len(config["seeds"]):
        raise ValueError("Choose between 1 and 10000 recorded seeds.")
    policies = [
        p for p in config["policies"] if not args.policies or p["name"] in args.policies
    ]
    if not policies or (
        args.policies and set(args.policies) != {p["name"] for p in policies}
    ):
        raise ValueError("Unknown policy name.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory.")
    if any(
        output == ROOT / f or ROOT / f in output.parents
        for f in ("src", "data", "paper", "repro", "config", "checkpoints")
    ):
        raise ValueError("Choose an output directory outside paper inputs.")
    output.mkdir(parents=True, exist_ok=True)
    rows = csv.DictReader(
        io.StringIO(gzip.decompress((ARCHIVE / "per_run.csv.gz").read_bytes()).decode())
    )
    reference = {(r["policy"], int(r["seed"])): r for r in rows}
    results = []
    with (output / "evaluation.log").open("w") as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        for policy in policies:
            for seed in seeds:
                ctrl = (
                    load_controller(policy, output / "runtime" / policy["name"])
                    if policy["family"] == "RLMM"
                    else None
                )
                try:
                    values = simulate(
                        config, policy, seed, config["events"], config["warmup"], ctrl
                    )
                    if args.compare_archive:
                        for key, value in values.items():
                            np.testing.assert_allclose(
                                value,
                                float(reference[policy["name"], seed][key]),
                                rtol=0,
                                atol=1e-10,
                                err_msg=f'{policy["name"]}/{seed}/{key}',
                            )
                    results.append(dict(policy=policy["name"], seed=seed, **values))
                finally:
                    if ctrl is not None:
                        ctrl.writer.close()
    with (output / "per_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    report = dict(
        status="passed",
        runs=len(results),
        archive_verified=args.compare_archive,
        events=config["events"],
        warmup=config["warmup"],
        seeds=seeds,
        protocol_sha256=runtime.file_hash(ARCHIVE / "campaign_manifest.json"),
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Final frontier: {len(results)} runs; archive verified: {args.compare_archive}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plot", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--compare-archive", action="store_true")
    args = parser.parse_args()
    if args.command == "plot":
        from plot_frontier_10000 import plot

        plot(args.output)
    else:
        evaluate(args)


def load_controller(spec, output, *, c51_mode="stable_float64_v1"):
    """Same AST-extracted shipped factory as paper_runtime, for frontier A files."""
    import torch

    DeepRLController = runtime.controller_class(c51_mode)
    path = ROOT / spec["checkpoint"]
    if runtime.file_hash(path) != spec["sha256"]:
        raise ValueError(f"Checkpoint hash differs: {path}")
    canonical = [
        p
        for p in runtime.manifest()["frontier_checkpoints"].values()
        if p["checkpoint"] == spec["checkpoint"]
    ]
    if len(canonical) != 1 or canonical[0]["sha256"] != spec["sha256"]:
        raise ValueError(
            "Frontier checkpoint does not match the canonical paper manifest."
        )
    factory_path = ROOT / "src/agents/checkpoint_io.py"
    nodes = [
        n
        for n in ast.parse(factory_path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "make_controller_from_checkpoint"
    ]
    if len(nodes) != 1:
        raise RuntimeError("Expected one checkpoint factory.")
    namespace = dict(
        torch=torch,
        Optional=Optional,
        DeepRLController=partial(DeepRLController, device=torch.device("cpu")),
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(factory_path), "exec"),
        namespace,
    )
    ctrl = namespace["make_controller_from_checkpoint"](
        str(path),
        log_dir=str(output),
        use_time_update=True,
        min_time_interval=1.0,
        inv_limit_override=8,
    )
    ctrl.evaluation_c51_mode = c51_mode
    ctrl.q_net.requires_grad_(False)
    ctrl.target_net.requires_grad_(False)
    if (
        ctrl.enable_learning
        or ctrl.epsilon != 0
        or ctrl.q_net.training
        or ctrl.target_net.training
    ):
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
    kwargs = dict(
        config["simulator"],
        iterations=events,
        iterations_to_equilibrium=warmup,
        random_seed=seed,
    )
    if ctrl is not None:

        def reward(step, mm, lob, before, after, info, **extra):
            reward_spread_capture_inv_quadratic(
                step,
                mm,
                lob,
                before,
                after,
                info,
                inv_penalty_coeff=spec["phi"],
                **extra,
            )
            return 0.0

        instrument_controller(
            ctrl, "both", dict(decisions=0, preserved_recovery_quotes=0)
        )
        kwargs.update(controller=ctrl, reward_fn=reward)
    elif spec["family"] == "at_best":
        kwargs["mm_policy"] = always_best_bid_ask_mm_policy_factory(**config["policy"])
    else:
        kwargs["mm_policy"] = glft_policy_factory(
            **config["parameters"],
            gamma=spec["gamma"],
            **config["glft_policy"],
            **config["policy"],
        )
    if ctrl is None:
        kwargs["mm_policy"] = instrument_benchmark(
            kwargs["mm_policy"], 8, dict(preserved_recovery_quotes=0)
        )
    metrics = simulate_LOB_with_MM(**kwargs)[2]
    return dict(
        terminal_pnl=float(metrics["MM_TotalPnL"]),
        terminal_inventory=float(metrics["MM_Inventory"]),
        mean_abs_inventory=float(metrics["mean_abs_inventory"]),
    )


if __name__ == "__main__":
    main()
