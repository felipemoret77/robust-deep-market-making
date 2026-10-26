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


def legacy_statistics(clock, sides, config):
    """Exact original arithmetic, with first-fill search on recorded event indices.

    Validated against the public unchanged estimator. Unit one-sided probes
    have no other inventory/cash jumps, so side flags identify the same fills.
    """
    clock = np.asarray(clock, dtype=float)
    sides = np.asarray(sides, dtype=np.int8)
    intervals, delta_t, _ = original.build_intervals(aggregation_mode='time',
        clock_t=clock, n_events_interval=1, time_interval=config.time_interval, n_tob_moves=5)
    fills = np.flatnonzero(sides == (1 if config.side_mode == 'buy' else -1))
    n_exec, exposure, n_windows, zero_wait = 0, 0.0, 0, 0
    for i0, i1, c0, c1 in intervals[config.burn_in_windows:]:
        effective = c1-float(clock[i0])
        if not effective > 0:
            effective = delta_t
        at = int(np.searchsorted(fills, i0))
        found = at < len(fills) and int(fills[at]) < i1
        duration, indicator = effective, 0
        if found:
            tau = float(clock[fills[at]])-float(clock[i0])
            if np.isfinite(tau) and tau > 0:
                duration = min(tau, effective)
                indicator = int(tau < effective)
            else:
                zero_wait += 1
        n_exec += indicator
        exposure += duration
        n_windows += 1
    return dict(n_exec=n_exec, exposure=exposure, n_windows=n_windows,
        lambda_hat=n_exec/exposure if exposure > 0 else 0., zero_wait_fills_discarded=zero_wait)


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
    elif variant == 'original':
        policy = original.censored_wait_probe_policy_factory(delta_value=distance,
            half_tick=config.half_tick, aggregation_mode='time', n_events_interval=1,
            time_interval=config.time_interval, n_tob_moves=5, side_mode=config.side_mode)
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
        original_estimator_on_this_tape=legacy_statistics(recorder.clock,recorder.fill_sides,config),
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


def combine(results, config):
    old, nominal, actual = [], defaultdict(empty_bin), defaultdict(empty_bin)
    for r in results:
        p = r['provenance']
        if p['variant']=='original':
            old.append(dict(distance=p['distance'],**r['original_estimator_on_this_tape']))
        else:
            for key,target in [('requested_distance_bins',nominal),('actual_distance_bins',actual)]:
                for row in r[key]:
                    b = target[row['distance']]
                    for k in b:
                        b[k] += row[k]
    curves = dict(original=sorted(old,key=lambda r:r['distance']),
                  corrected_requested=bin_rows(nominal),corrected_actual=bin_rows(actual))
    fits = {name:fit_curve(rows,config) for name,rows in curves.items() if rows}
    return dict(config=asdict(config),curves=curves,fits=fits,
        notes=[
            'Same configuration and seeds; paths can differ because actual-time reposting changes the policy.',
            'Original discards the first five reconstructed windows; corrected discards the first five successfully posted probes in each distance run.',
            'corrected_requested and corrected_actual use exactly the same corrected simulations; only grouping differs.',
            'Only distances inside the original fit range enter the fit; extra realized bins remain in curves.',
            'Observation ends at posting+time_interval; physical cancellation occurs at the next eligible simulator call.',
            'Corrected quotes use the mid immediately after cancelling the previous probe and before inserting the next one.',
            'No change to sigma, GLFT benchmark parameters, original files, checkpoints, paper or figures.'])


def save_outputs(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    write_json(output/'comparison.json',summary)
    fields=['method','distance','n_exec','exposure','lambda_hat']
    with (output/'intensities.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name,rows in summary['curves'].items():
            for row in rows:
                writer.writerow(dict(method=name,**{k:row[k] for k in fields[1:]}))
    labels=dict(original='Original',corrected_requested='Protocolo corrigido; distância solicitada',
                corrected_actual='Protocolo corrigido; distância efetiva')
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for name,rows in summary['curves'].items():
        if not rows:continue
        positive=[r for r in rows if r['lambda_hat']>0 and r['distance']<=summary['config']['max_levels']*summary['config']['half_tick']]
        line,=ax.semilogy([r['distance'] for r in positive],[r['lambda_hat'] for r in positive],
            'o',label=labels[name])
        fit=summary['fits'][name]
        xs=np.linspace(*fit['fit_range'],200)
        ax.semilogy(xs,fit['A']*np.exp(-fit['kappa']*xs),'--',color=line.get_color())
    ax.set_xlabel('Distância ao mid (ticks)')
    ax.set_ylabel('Intensidade estimada (s⁻¹)')
    ax.set_title(f"Calibração GLFT: original versus corrigida\n{summary['config']['n_steps']:,} eventos por distância; seed {summary['config']['random_seed']}")
    ax.grid(alpha=.2);ax.legend(fontsize=10)
    for ext in ['png','pdf']:fig.savefig(output/('comparacao_calibracao_GLFT.'+ext),dpi=200)
    plt.close(fig)


def self_test(output):
    output=Path(output)/'self_test';output.mkdir(parents=True,exist_ok=True)
    cases=[(0.,.312,'fill',.5,.312,1),(0.,.6,'fill',.5,.5,0),
           (0.,.5,'fill',.5,.5,0),(0.,.8,'cancel',.5,.5,0),
           (0.,.2,'removed_by_engine',.5,.2,0),(0.,.1,'end_of_sample',.5,.1,0)]
    for posted,ended,reason,horizon,want_time,want_i in cases:
        got_time,got_i=observation(posted,ended,reason,horizon)
        assert abs(got_time-want_time)<1e-12 and got_i==want_i
    cfg=replace(Config(),n_steps=5000,max_levels=1)
    old=run_distance(cfg,.5,'original',output)
    with (output/'public_original.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        public=original.fit_execution_intensity_censored_waiting_times(
            aggregation_mode='time',time_interval=cfg.time_interval,max_levels=1,half_tick=.5,
            side_mode='buy',burn_in_windows=cfg.burn_in_windows,
            lam=cfg.lam,mu=cfg.mu,delta=cfg.delta,number_tick_levels=50,n_priority_ranks=100,
            n_steps=cfg.n_steps,n_steps_to_equilibrium=1000,random_seed=1234,
            exclude_self_from_state=True,mo_size=1,split_sweeps=False,plot=False,use_tqdm=False)
    old=old['original_estimator_on_this_tape']
    for key,src in [('n_exec','n_exec'),('exposure','denom_sum_tau'),('lambda_hat','lambda_hat')]:
        assert abs(old[key]-float(public[src][0]))<1e-10,(key,old,public)
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
        original_full_vs_lightweight='exact agreement in counts, exposure and lambda',
        corrected_lifecycle_invariants='passed',original_control=old)
    write_json(output/'tests.json',result)
    print(json.dumps(result),flush=True)
