#!/usr/bin/env python3
"""Recover per-seed A outcomes by rerunning the two recorded historical sweeps.

Frozen checkpoints, original seeds and schedules. Resumable per-episode JSON.
Every completed group is checked against the four-decimal historical summary;
a mismatch stops further work instead of replacing the published results.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import contextlib
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
WORKER = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(ctrl):
    h = hashlib.sha256()
    for name in ('q_net','target_net'):
        for key, value in getattr(ctrl,name).state_dict().items():
            h.update((name+key).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(output, protocol_hash):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    WORKER.update(output=Path(output), protocol_hash=protocol_hash, controllers={})


def simulate(job):
    import torch
    import frontier
    import paper_runtime as runtime
    family, phi, tau, seed = job
    out = WORKER['output']; start = time.monotonic()
    with (out/'logs'/f'{family}_{phi}_{tau}_{seed}.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        cache = WORKER['controllers']
        if phi not in cache:
            config = json.loads((ROOT/'config/frontier_fresh_seeds.json').read_text())
            spec = next(p for p in config['policies'] if p.get('family')=='RLMM' and p['phi']==float(phi))
            ctrl = frontier.load_controller(spec, out/'workers'/f'{os.getpid()}_{phi}', c51_mode='legacy_float32')
            meta = torch.load(ROOT/spec['checkpoint'], map_location='cpu', weights_only=True)['meta']
            cache[phi] = ctrl, meta, spec
        ctrl, meta, spec = cache[phi]
        assert not ctrl.enable_learning and ctrl.epsilon==0
        assert not ctrl.q_net.training and not ctrl.target_net.training
        before = digest(ctrl)
        helpers, signals = runtime.evaluation_helpers(meta)
        schedule = runtime.make_schedule(helpers,'sweep',seed,100000,tau)
        with torch.inference_mode():
            _, metrics = helpers.run_one_sim(schedule, seed=seed, controller=ctrl,
                total_events=100000, equil_steps=10000, signal_config=signals, metrics_only=True)
        assert digest(ctrl)==before
        return dict(family=family,phi=phi,tau=tau,seed=seed,
                    schedule_seed=None if tau==0 else seed+1000000+10000*tau,
                    events=100000,warmup=10000,terminal_pnl=float(metrics['MM_TotalPnL']),
                    terminal_inventory=float(metrics['MM_Inventory']),
                    checkpoint_sha256=spec['sha256'],networks_unchanged=True,
                    protocol_sha256=WORKER['protocol_hash'],elapsed_seconds=time.monotonic()-start)


def groups():
    for family, phis, start in [('degradation',('0.001','0.005'),10123),('sweep',('0.001',),20123)]:
        for phi in phis:
            for tau in (0,15,30,60,120,240):
                yield family,phi,tau,list(range(start,start+100))


def reference():
    expected = {}
    for row in csv.DictReader((ROOT/'data/paper/historical_results/degradation_summary.csv').open()):
        expected['degradation',row['phi'],int(row['tau'])] = float(row['mean']),float(row['std'])
    for row in csv.DictReader((ROOT/'data/paper/tau_sweep_summary.csv').open()):
        expected['sweep','0.001',int(row['tau'])] = float(row['A_mean_pnl']),float(row['A_std_pnl'])
    return expected


def run(args):
    import numpy as np
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    for name in ('episodes','logs','workers'): (out/name).mkdir(exist_ok=True)
    sources = [Path(__file__),ROOT/'repro/frontier.py',ROOT/'repro/paper_runtime.py',ROOT/'repro/paper_helpers.py',
               ROOT/'config/paper_experiments.json',ROOT/'config/frontier_fresh_seeds.json',
               ROOT/'data/paper/tau_sweep_summary.csv',ROOT/'data/paper/historical_results/degradation_summary.csv',
               ROOT/'src/agents/checkpoint_io.py']
    sources += sorted((ROOT/'src/sim').glob('*.py'))+sorted((ROOT/'src/agents').glob('*.py'))
    model_spec = json.loads((ROOT/'config/paper_experiments.json').read_text())
    selected = {k:v for k,v in model_spec['frontier_checkpoints'].items()
                if v['metadata']['best_params']['inv_penalty_coeff'] in (.001,.005)}
    for spec in selected.values(): assert sha(ROOT/spec['checkpoint'])==spec['sha256']
    protocol = dict(events=100000,warmup=10000,episodes_per_group=100,total_rollouts=1800,
        taus=[0,15,30,60,120,240],seed_ranges={'degradation':[10123,10222],'sweep':[20123,20222]},
        schedule_seed_rule='seed+1000000+10000*tau; stationary probability0.5 for tau0',
        models={k:{j:v[j] for j in ('checkpoint','sha256')} for k,v in selected.items()},
        learning=False,epsilon=0,device='cpu',numerical_threads=1,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    path=out/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol,'Protocol changed'
    else: save(path,protocol)
    ph=sha(path); expected=reference(); completed=[]; summaries=[]; start=time.monotonic()
    def record_path(job):return out/'episodes'/f'{job[0]}_{job[1]}_{job[2]}_{job[3]}.json'
    def check(record,job):
        for key,value in zip(('family','phi','tau','seed'),job):assert record[key]==value
        assert record['protocol_sha256']==ph and record['networks_unchanged']
        assert np.isfinite(record['terminal_pnl'])
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(str(out),ph)) as pool:
        for family,phi,tau,seeds in groups():
            jobs=[(family,phi,tau,s) for s in seeds]; pending=[]
            for job in jobs:
                if record_path(job).exists():check(json.loads(record_path(job).read_text()),job)
                else:pending.append(job)
            futures={pool.submit(simulate,job):job for job in pending}
            for future in cf.as_completed(futures):
                job=futures[future];record=future.result();check(record,job);save(record_path(job),record)
                n=len(list((out/'episodes').glob('*.json')))
                save(out/'progress.json',dict(state='running',completed=n,total=1800,workers=args.workers,updated_unix=time.time()))
                print(f'{n}/1800 {family} phi={phi} tau={tau} seed={job[3]} pnl={record["terminal_pnl"]:+.4f} {record["elapsed_seconds"]:.1f}s',flush=True)
            rows=[json.loads(record_path(job).read_text()) for job in jobs]
            values=np.array([row['terminal_pnl'] for row in rows]); m,s=values.mean(),values.std(ddof=1)
            oldm,olds=expected[family,phi,tau]
            result=dict(family=family,phi=phi,tau=tau,n=100,mean=float(m),std=float(s),
                        historical_mean=oldm,historical_std=olds,mean_error=float(m-oldm),std_error=float(s-olds),
                        matches_historical_rounding=bool(abs(m-oldm)<=.00005001 and abs(s-olds)<=.00005001))
            summaries.append(result);save(out/'group_validation.json',summaries)
            if not result['matches_historical_rounding']:
                save(out/'mismatch.json',result)
                raise RuntimeError(f'Historical summary mismatch: {result}')
            completed.extend(rows)
            print('Historical mean/std reproduced:',family,phi,tau,flush=True)
    assert len(completed)==1800
    for name,h in protocol['source_sha256'].items():assert sha(ROOT/name)==h,name
    for spec in selected.values():assert sha(ROOT/spec['checkpoint'])==spec['sha256']
    with (out/'per_episode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(completed[0]));writer.writeheader();writer.writerows(completed)
    save(out/'completion.json',dict(status='passed',rollouts=1800,groups=18,
        all_historical_means_and_stds_reproduced=True,checkpoints_and_sources_unchanged=True,
        protocol_sha256=ph,per_episode_sha256=sha(out/'per_episode.csv'),elapsed_seconds=time.monotonic()-start))
    save(out/'progress.json',dict(state='complete',completed=1800,total=1800,updated_unix=time.time()))
    print('All1800 outcomes recovered and checked against historical aggregates.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('Use1–4workers')
    try:run(args)
    except Exception as e:
        if args.output.exists():save(args.output/'error.json',dict(error=repr(e),time=time.time()))
        raise
