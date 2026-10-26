"""Small-memory GLFT evaluations using unmodified frozen simulator/policy code."""

import os

os.environ.update(
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg"
)
import ast
from contextlib import redirect_stdout, redirect_stderr
import contextlib
from paper_evaluation_corrections import instrument_benchmark
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Callable, List
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/glft_evaluation"
import glft_runtime

simulator = glft_runtime.simulator
from GLFT_policy_factory import glft_policy_factory
from MM_policy_5 import always_best_bid_ask_mm_policy_factory


def load_regime_function():
    # Execute the exact frozen function without importing the unrelated training
    # networks and their large optional dependencies into every worker.
    path = REPO / "src/sim/glft_regime_schedule.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "make_regime_schedule"
    )
    env = dict(
        np=np,
        Callable=Callable,
        List=List,
        REGIME_L_MIN=10,
        REGIME_ALPHA=1.5,
        REGIME_P_LO=0.2,
        REGIME_P_HI=0.8,
        REGIME_DISTRIBUTION="exponential",
        REGIME_EXP_TAU=60.0,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env["make_regime_schedule"]


make_regime_schedule = load_regime_function()


COUNTS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


atomic = lambda path, value: atomic_json(path, value)


def make_policy(params, gamma=1e-6, naive=False):
    common = dict(
        inv_limit=8,
        use_tob_update=False,
        use_event_update=False,
        use_time_update=True,
        min_time_interval=1.0,
        use_mdp=True,
    )
    if naive:
        policy = always_best_bid_ask_mm_policy_factory(**common)
    else:
        policy = glft_policy_factory(
            A=params["A"],
            kappa=params["kappa"],
            sigma=params["sigma"],
            gamma=gamma,
            round_to_int=True,
            delta=1.0,
            epsilon=None,
            n_tob_moves=1,
            n_events=1000,
            **common,
        )
    COUNTS.clear()
    COUNTS["preserved_recovery_quotes"] = 0
    return instrument_benchmark(policy, 8, COUNTS)


def simulate(
    params,
    seed,
    events=5000,
    gamma=1e-6,
    prob=0.5,
    tau=None,
    p_lo=0.2,
    p_hi=0.8,
    naive=False,
    trace=False,
    full=False,
):
    random.seed(int(seed))
    np.random.seed(int(seed))
    policy = make_policy(params, gamma, naive)
    if tau is not None:
        prob = make_regime_schedule(
            seed=int(seed) + 999999,
            n_mo_events=events,
            p_lo=p_lo,
            p_hi=p_hi,
            distribution="exponential",
            exp_rate=1.0 / tau,
        )
    return simulator.simulate_LOB_with_MM(
        lam=0.06,
        mu=0.10,
        delta=0.02,
        random_seed=int(seed),
        number_tick_levels=50,
        n_priority_ranks=100,
        iterations=events,
        iterations_to_equilibrium=1000,
        mm_policy=policy,
        exclude_self_from_state=False,
        beta_exp_weighted_return=0.0,
        intensity_exp_weighted_return=0.0,
        buy_mo_prob=prob,
        metrics_only=not full,
        metrics_trace_max_points=events if trace and not full else None,
    )[2]


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run_chunk(job):
    spec, params, seeds = job
    signature = dict(
        spec=spec,
        parameters=params,
        seeds=seeds,
        runtime_sha256=sha(__file__),
        queue_sha256=sha(REPO / "repro/paper_evaluation_corrections.py"),
    )
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    folder = ROOT / "chunks" / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{seeds[0]}_{digest}.npz"
    meta = file.with_suffix(".json")
    if file.exists() and meta.exists():
        m = json.loads(meta.read_text())
        assert m["signature"] == signature and m["sha256"] == sha(file)
        return str(file)
    arr = {
        k: []
        for k in [
            "terminal_pnl",
            "terminal_inventory",
            "mean_abs_inventory",
            "limit_fraction",
            "fill_rate",
            "preserved_recovery_quotes",
        ]
    }
    traces = []
    inventories = []
    started = time.monotonic()
    with file.with_suffix(".log").open("w") as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        for seed in seeds:
            m = simulate(
                params,
                seed,
                events=spec.get("events", 5000),
                gamma=spec.get("gamma", 1e-6),
                prob=spec.get("prob", 0.5),
                tau=spec.get("tau"),
                p_lo=spec.get("p_lo", 0.2),
                p_hi=spec.get("p_hi", 0.8),
                naive=spec.get("naive", False),
                trace=spec.get("trace", False),
                full=spec.get("eta", False),
            )
            arr["preserved_recovery_quotes"].append(COUNTS["preserved_recovery_quotes"])
            if spec.get("eta"):
                if "MM_NumBidFills" in m and "MM_NumAskFills" in m:
                    nf = int(
                        m["MM_NumBidFills"].iloc[-1] + m["MM_NumAskFills"].iloc[-1]
                    )
                else:
                    assert m["MM_Inventory"].iloc[0] == 0
                    changes = m["MM_Inventory"].diff().fillna(0).abs()
                    assert changes.max() <= 1
                    nf = int((changes > 0).sum())
                dt = float(m["Time"].iloc[-1] - m["Time"].iloc[0])
                assert dt > 0
                arr["fill_rate"].append(nf / dt)
                arr["terminal_pnl"].append(float(m["MM_TotalPnL"].iloc[-1]))
                arr["terminal_inventory"].append(float(m["MM_Inventory"].iloc[-1]))
            else:
                assert m["n_steps"] == spec.get("events", 5000)
                arr["terminal_pnl"].append(m["MM_TotalPnL"])
                arr["terminal_inventory"].append(m["MM_Inventory"])
                arr["mean_abs_inventory"].append(m["mean_abs_inventory"])
                arr["limit_fraction"].append(m["limit_frac"])
                if spec.get("trace"):
                    assert len(m["trace_pnl"]) == spec.get("events", 5000)
                    np.testing.assert_allclose(
                        m["trace_pnl"][-1], m["MM_TotalPnL"], rtol=1e-6, atol=1e-6
                    )
                    traces.append(m["trace_pnl"])
                    inventories.append(m["trace_inventory"])
            assert abs(arr["terminal_inventory"][-1]) <= 8
            print("completed", int(seed), flush=True)
    arrays = {k: np.asarray(v) for k, v in arr.items()}
    arrays["seeds"] = np.asarray(seeds, dtype=np.int64)
    if traces:
        arrays.update(pnl_trace=np.stack(traces), inventory_trace=np.stack(inventories))
    for v in arrays.values():
        assert np.isfinite(v).all()
    temp = file.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(file)
    atomic(
        meta,
        dict(
            signature=signature,
            episodes=len(seeds),
            evaluation_events=len(seeds) * spec.get("events", 5000),
            elapsed_s=time.monotonic() - started,
            sha256=sha(file),
        ),
    )
    return str(file)


def load_group(paths):
    groups = {}
    for path in paths:
        with np.load(path) as d:
            for k in d.files:
                groups.setdefault(k, []).append(d[k])
    arrays = {k: np.concatenate(v, axis=0) for k, v in groups.items()}
    order = np.argsort(arrays["seeds"])
    return {k: (v[order] if len(v) == len(order) else v) for k, v in arrays.items()}
