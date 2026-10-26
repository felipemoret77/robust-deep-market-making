#!/usr/bin/env python3
"""Frozen order-lifecycle calibration core. Use glft_calibration.py for safe CLI.

Each successful probe has its own timestamp, ID, effective posting distance and
observation deadline. Reposting is gated from actual posting. Late fills do not
enter the fit; partial exposures and tick rounding are retained explicitly.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import csv
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import glft_runtime

simulator = glft_runtime.simulator
original = glft_runtime.original


@dataclass(frozen=True)
class Config:
    lam: float = 0.06
    mu: float = 0.10
    delta: float = 0.02
    number_tick_levels: int = 50
    n_priority_ranks: int = 100
    n_steps: int = 500_000
    n_steps_to_equilibrium: int = 1_000
    random_seed: int = 1234
    time_interval: float = 0.5
    posting_interval: float = 0.5
    max_levels: int = 5
    half_tick: float = 0.5
    side_mode: str = 'buy'
    burn_in_windows: int = 5
    exclude_self_from_state: bool = True
    mo_size: int = 1

    def validate(self):
        if self.time_interval <= 0 or self.posting_interval < self.time_interval:
            raise ValueError('Require posting_interval >= time_interval > 0.')
        if self.side_mode not in ('buy', 'sell'):
            raise ValueError('side_mode must be buy or sell.')
        if self.mo_size != 1:
            raise ValueError('This comparison is validated for unit market orders.')
        if self.n_steps <= 0 or self.max_levels <= 0 or self.half_tick <= 0:
            raise ValueError('Positive simulation horizon and distance grid required.')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes():
    paths = [Path(__file__), Path(original.__file__), Path(simulator.__file__)]
    return {str(p.relative_to(glft_runtime.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def current_time(mm):
    tape = mm.lob.message_dict.get('Time', [])
    return float(tape[-1]) if len(tape) else 0.0


class ProbePolicy:
    def __init__(self, distance, config):
        self.distance = float(distance)
        self.config = config
        self.next_post = -math.inf

    def __call__(self, state, mm=None):
        if mm is None or current_time(mm) < self.next_post:
            return ('hold', 0, -1)
        # Removing our previous quote can change the mid. Take the reference
        # after that removal, immediately before inserting the next probe.
        # No market event or RNG draw occurs between cancellation and posting.
        mm.cancel_all()
        mid = float(mm.lob.compute_mid_price())
        if not np.isfinite(mid):
            return ('hold', 0, -1)
        if self.config.side_mode == 'buy':
            return ('cancel_all_then_place', +1, int(math.floor(mid-self.distance)))
        return ('cancel_all_then_place', -1, int(math.ceil(mid+self.distance)))

    def on_successful_post(self, time_posted):
        # A failed placement creates no observation and does not advance this gate.
        self.next_post = float(time_posted) + self.config.posting_interval


def observation(posted, ended, reason, horizon):
    """Return (exposure, event indicator), keeping first-event positive waits."""
    if ended < posted:
        raise AssertionError('Order ended before posting.')
    deadline = posted + horizon
    exposure = min(ended, deadline) - posted
    indicator = int(reason == 'fill' and ended < deadline)
    return float(exposure), indicator


def empty_bin():
    return dict(n_probes=0, n_exec=0, exposure=0.0)


class Recorder:
    def __init__(self, config, distance, policy, csv_path, progress_path):
        self.config, self.distance, self.policy = config, distance, policy
        self.csv_path, self.progress_path = csv_path, progress_path
        self.active = {}
        self.clock = array('d')
        self.fill_sides = array('b')
        self.actual = defaultdict(empty_bin)
        self.nominal = defaultdict(empty_bin)
        self.reasons = Counter()
        self.n_posted = self.n_failed = self.n_fills = self.n_late = 0
        self.n_early_removed = self.n_truncated = 0
        self.step = -1
        self.end_time = 0.0
        self.started = time.monotonic()
        self.file = Path(csv_path).open('w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'oid', 'post_index', 'post_step', 'posted_at', 'reference_mid', 'price',
            'requested_distance', 'actual_distance', 'side', 'ended_at', 'end_step',
            'reason', 'deadline', 'exposure', 'I', 'used_after_burnin'])
        self.writer.writeheader()

    def post(self, oid, info, mid, timestamp):
        if oid in self.active:
            raise AssertionError('Duplicate live probe ID.')
        price, side = int(info['price']), int(info['side'])
        actual = mid-price if side > 0 else price-mid
        if isinstance(self.policy, ProbePolicy) and actual < 0:
            raise AssertionError('Passive probe has negative posting distance.')
        rec = dict(oid=int(oid), post_index=self.n_posted, post_step=self.step,
            posted_at=timestamp, reference_mid=mid, price=price,
            requested_distance=self.distance, actual_distance=float(actual), side=side)
        self.n_posted += 1
        self.active[int(oid)] = rec
        if isinstance(self.policy, ProbePolicy):
            self.policy.on_successful_post(timestamp)

    def close(self, oid, timestamp, reason):
        rec = self.active.pop(int(oid))
        exposure, indicator = observation(rec['posted_at'], timestamp, reason, self.config.time_interval)
        deadline = rec['posted_at'] + self.config.time_interval
        self.reasons[reason] += 1
        if reason == 'fill':
            self.n_fills += 1
            self.n_late += int(timestamp >= deadline)
        if reason in ('cancel', 'removed_by_engine') and timestamp < deadline-1e-12:
            self.n_early_removed += 1
        if reason == 'end_of_sample' and timestamp < deadline:
            self.n_truncated += 1
        used = rec['post_index'] >= self.config.burn_in_windows
        rec.update(ended_at=timestamp, end_step=self.step, reason=reason,
            deadline=deadline, exposure=exposure, I=indicator, used_after_burnin=used)
        self.writer.writerow(rec)
        if used:
            for target, key in [(self.actual, rec['actual_distance']), (self.nominal, self.distance)]:
                b = target[float(key)]
                b['n_probes'] += 1
                b['n_exec'] += indicator
                b['exposure'] += exposure

    def reconcile(self, mm, timestamp):
        for oid in set(self.active)-set(mm.orders):
            self.close(oid, timestamp, 'removed_by_engine')
        if set(self.active) != set(mm.orders):
            raise AssertionError('Unregistered resting probe.')

    def finish(self):
        for oid in list(self.active):
            self.close(oid, self.end_time, 'end_of_sample')
        self.file.close()
        if len(self.clock) != self.config.n_steps:
            raise AssertionError('Event clock length mismatch.')
        if sum(v != 0 for v in self.fill_sides) != self.n_fills:
            raise AssertionError('Fill telemetry disagrees with probe lifecycles.')


@contextmanager
def tracked_market_maker(recorder):
    """Process-local instrumentation; original class restored even on failure."""
    base = simulator.MarketMaker
    r = recorder

    class TrackedMarketMaker(base):
        def pre_step(self, *args, **kwargs):
            r.step += 1
            timestamp = current_time(self)
            r.reconcile(self, timestamp)
            action = super().pre_step(*args, **kwargs)
            r.reconcile(self, timestamp)
            return action

        def place_limit(self, side, price_idx, *args, **kwargs):
            # This hook runs after old quotes have been cancelled, immediately
            # before the simulator inserts the new limit order.
            mid = float(self.lob.compute_mid_price())
            timestamp = current_time(self)
            result = super().place_limit(side, price_idx, *args, **kwargs)
            oid = int(result[0])
            if oid >= 0 and oid in self.orders:
                r.post(oid, self.orders[oid], mid, timestamp)
            else:
                r.n_failed += 1
            return result

        def cancel(self, oid, *args, **kwargs):
            existed = oid in self.orders
            result = super().cancel(oid, *args, **kwargs)
            if existed and oid not in self.orders and oid in r.active:
                r.close(oid, current_time(self), 'cancel')
            return result

        def post_step_detect_fill_by_env_mo(self, *args, **kwargs):
            result = super().post_step_detect_fill_by_env_mo(*args, **kwargs)
            fills = result.get('fills_list', [])
            if not fills and result.get('qty', 0) > 0:
                fills = [result]
            if len(fills) > 1:
                raise AssertionError('Expected one unit probe at a time.')
            side = 0
            for fill in fills:
                side = int(fill['side'])
                r.close(int(fill['filled_oid']), current_time(self), 'fill')
            r.fill_sides.append(side)
            return result

        def log_step(self, step_idx, time_val, *args, **kwargs):
            result = super().log_step(step_idx, time_val, *args, **kwargs)
            r.clock.append(float(time_val))
            r.end_time = float(time_val)
            r.reconcile(self, float(time_val))
            if (step_idx+1) % 25_000 == 0:
                write_json(r.progress_path, dict(events=step_idx+1, total=r.config.n_steps,
                    elapsed_s=time.monotonic()-r.started, probes=r.n_posted, fills=r.n_fills))
            return result

    simulator.MarketMaker = TrackedMarketMaker
    try:
        yield
    finally:
        simulator.MarketMaker = base




def bin_rows(bins):
    return [dict(distance=float(d), **b,
        lambda_hat=b['n_exec']/b['exposure'] if b['exposure'] else 0.)
        for d,b in sorted(bins.items())]


def run_distance(config, distance, variant, output):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tag = f'{variant}_seed{config.random_seed}_delta{distance:g}_n{config.n_steps}'
    path = output/(tag+'.json')
    provenance = dict(config=asdict(config), distance=distance, variant=variant, source_sha256=source_hashes())
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['provenance'] == provenance:
            return saved
    if variant == 'corrected':
        policy = ProbePolicy(distance, config)
    else:
        raise ValueError(variant)
    recorder = Recorder(config, distance, policy, output/(tag+'_orders.csv'), output/(tag+'_progress.json'))
    started = time.monotonic()
    try:
        with (output/(tag+'.log')).open('w') as log, redirect_stdout(log), redirect_stderr(log), tracked_market_maker(recorder):
            original._seed_everything(config.random_seed)
            simulator.simulate_LOB_with_MM(lam=config.lam, mu=config.mu, delta=config.delta,
                random_seed=config.random_seed, number_tick_levels=config.number_tick_levels,
                n_priority_ranks=config.n_priority_ranks, iterations=config.n_steps,
                iterations_to_equilibrium=config.n_steps_to_equilibrium, mm_policy=policy,
                exclude_self_from_state=config.exclude_self_from_state, mean_size_MO=config.mo_size,
                buy_mo_prob=0.5,
                split_sweeps=False, beta_exp_weighted_return=0.0, intensity_exp_weighted_return=0.0,
                metrics_only=True, metrics_trace_max_points=0)
        recorder.finish()
    finally:
        recorder.file.close()
    result = dict(provenance=provenance, elapsed_s=time.monotonic()-started,
        actual_distance_bins=bin_rows(recorder.actual), requested_distance_bins=bin_rows(recorder.nominal),
        diagnostics=dict(posted=recorder.n_posted, failed_postings=recorder.n_failed,
            all_fills=recorder.n_fills, fills_after_observation_deadline=recorder.n_late,
            removals_before_deadline=recorder.n_early_removed,
            sample_end_partial_observations=recorder.n_truncated,
            end_reasons=dict(recorder.reasons), simulation_end=recorder.end_time),
        csv_file=str(recorder.csv_path))
    if variant == 'corrected' and recorder.n_early_removed:
        raise AssertionError('Unexpected early removal in corrected calibration; inspect CSV.')
    write_json(path,result)
    return result


def fit_curve(rows, config):
    # Keep the same requested fit range; rounding may create extra bins at 3 ticks.
    chosen = [r for r in rows if config.half_tick-1e-10 <= r['distance'] <= config.max_levels*config.half_tick+1e-10]
    d = np.array([r['distance'] for r in chosen])
    rates = np.array([r['lambda_hat'] for r in chosen])
    weights = np.array([r['exposure'] for r in chosen])
    A,kappa,info = original.fit_A_kappa_loglinear(d,rates,weights=weights)
    return dict(A=A,kappa=kappa,fit_diagnostics={k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in info.items()},
        distances_used=[float(x) for x,r in zip(d,rates) if r > 1e-15],
        fit_range=[config.half_tick,config.max_levels*config.half_tick],
        weighting='total observed exposure, as in GLFT_studies.py')






def self_test(output):
    output=Path(output)/'self_test';output.mkdir(parents=True,exist_ok=True)
    cases=[(0.,.312,'fill',.5,.312,1),(0.,.6,'fill',.5,.5,0),
           (0.,.5,'fill',.5,.5,0),(0.,.8,'cancel',.5,.5,0),
           (0.,.2,'removed_by_engine',.5,.2,0),(0.,.1,'end_of_sample',.5,.1,0)]
    for posted,ended,reason,horizon,want_time,want_i in cases:
        got_time,got_i=observation(posted,ended,reason,horizon)
        assert abs(got_time-want_time)<1e-12 and got_i==want_i
    cfg=replace(Config(),n_steps=5000,max_levels=1)
    new=run_distance(cfg,.5,'corrected',output)
    assert new['diagnostics']['removals_before_deadline']==0
    with Path(new['csv_file']).open() as f:
        records=list(csv.DictReader(f))
    assert len(records)==new['diagnostics']['posted']
    assert len({row['oid'] for row in records})==len(records)
    posts=[float(row['posted_at']) for row in sorted(records,key=lambda r:int(r['post_index']))]
    assert np.all(np.diff(posts)>=cfg.posting_interval-1e-12)
    for row in records:
        t_post,t_end=float(row['posted_at']),float(row['ended_at'])
        assert 0<=float(row['exposure'])<=cfg.time_interval+1e-12
        if int(row['I']):assert row['reason']=='fill' and t_end<t_post+cfg.time_interval
        assert float(row['actual_distance'])==float(row['reference_mid'])-float(row['price'])
        assert float(row['requested_distance'])<=float(row['actual_distance'])<float(row['requested_distance'])+1
    result=dict(status='passed',synthetic_cases=len(cases),
        corrected_lifecycle_invariants='passed')
    write_json(output/'tests.json',result)
    print(json.dumps(result),flush=True)
