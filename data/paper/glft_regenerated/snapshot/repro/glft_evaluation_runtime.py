"""Small-memory GLFT evaluations using unmodified frozen simulator/policy code."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Callable, List
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/glft_evaluation'
import glft_runtime
simulator=glft_runtime.simulator
from GLFT_policy_factory import glft_policy_factory
from MM_policy_5 import always_best_bid_ask_mm_policy_factory


def load_regime_function():
    # Execute the exact frozen function without importing the unrelated training
    # networks and their large optional dependencies into every worker.
    path=REPO/'src/eval/regime_switching_stress_test_2.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_regime_schedule')
    env=dict(np=np,Callable=Callable,List=List,REGIME_L_MIN=10,REGIME_ALPHA=1.5,
             REGIME_P_LO=.2,REGIME_P_HI=.8,REGIME_DISTRIBUTION='exponential',REGIME_EXP_TAU=60.)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['make_regime_schedule']


make_regime_schedule=load_regime_function()


def make_policy(params,gamma=1e-6,naive=False):
    common=dict(inv_limit=8,use_tob_update=False,use_event_update=False,
                use_time_update=True,min_time_interval=1.,use_mdp=True)
    if naive:return always_best_bid_ask_mm_policy_factory(**common)
    return glft_policy_factory(A=params['A'],kappa=params['kappa'],sigma=params['sigma'],
                              gamma=gamma,round_to_int=True,delta=1.,epsilon=None,
                              n_tob_moves=1,n_events=1000,**common)


def simulate(params,seed,events=5000,gamma=1e-6,prob=.5,tau=None,p_lo=.2,p_hi=.8,
             naive=False,trace=False,full=False):
    random.seed(int(seed));np.random.seed(int(seed))
    policy=make_policy(params,gamma,naive)
    if tau is not None:
        prob=make_regime_schedule(seed=int(seed)+999999,n_mo_events=events,
                                 p_lo=p_lo,p_hi=p_hi,distribution='exponential',exp_rate=1./tau)
    return simulator.simulate_LOB_with_MM(lam=.06,mu=.10,delta=.02,random_seed=int(seed),
        number_tick_levels=50,n_priority_ranks=100,iterations=events,iterations_to_equilibrium=1000,
        mm_policy=policy,exclude_self_from_state=False,beta_exp_weighted_return=0.,intensity_exp_weighted_return=0.,
        buy_mo_prob=prob,metrics_only=not full,metrics_trace_max_points=events if trace and not full else None)[2]


def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def run_chunk(job):
    """Each job is independently restartable; cache signature includes all inputs."""
    spec,params,seeds=job
    signature=dict(spec=spec,params=params,seeds=seeds,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),REPO/'src/sim/MM_LOB_SIM.py',REPO/'src/glft/GLFT_policy_factory.py',REPO/'src/eval/regime_switching_stress_test_2.py']})
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'data'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    output=folder/f'chunk_{seeds[0]}_{digest}.npz'
    manifest=output.with_suffix('.json')
    if output.exists() and manifest.exists() and json.loads(manifest.read_text()).get('signature')==signature:
        return str(output)
    pnls=[];inventories=[];absinv=[];limit=[];pnl_traces=[];inv_traces=[];fill_rates=[]
    started=time.monotonic()
    with output.with_suffix('.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        for seed in seeds:
            m=simulate(params,seed,events=spec.get('events',5000),gamma=spec.get('gamma',1e-6),
                       prob=spec.get('prob',.5),tau=spec.get('tau'),p_lo=spec.get('p_lo',.2),p_hi=spec.get('p_hi',.8),
                       naive=spec.get('naive',False),trace=spec.get('trace',False),full=spec.get('eta',False))
            if spec.get('eta',False):
                if 'MM_NumBidFills' in m and 'MM_NumAskFills' in m:
                    nf=int(m['MM_NumBidFills'].iloc[-1]+m['MM_NumAskFills'].iloc[-1])
                else:nf=int((m['MM_Inventory'].diff().abs()>0).sum())
                dt=float(m['Time'].iloc[-1]-m['Time'].iloc[0])
                assert dt>0
                fill_rates.append(nf/dt)
                pnls.append(float(m['MM_TotalPnL'].iloc[-1]));inventories.append(float(m['MM_Inventory'].iloc[-1]))
            else:
                pnls.append(m['MM_TotalPnL']);inventories.append(m['MM_Inventory'])
                absinv.append(m['mean_abs_inventory']);limit.append(m['limit_frac'])
                if spec.get('trace',False):
                    assert len(m['trace_pnl'])==spec.get('events',5000)
                    pnl_traces.append(m['trace_pnl']);inv_traces.append(m['trace_inventory'])
    arrays=dict(seeds=np.asarray(seeds),terminal_pnl=np.asarray(pnls),terminal_inventory=np.asarray(inventories),
                mean_abs_inventory=np.asarray(absinv),limit_fraction=np.asarray(limit),fill_rate=np.asarray(fill_rates))
    if pnl_traces:arrays.update(pnl_trace=np.stack(pnl_traces),inventory_trace=np.stack(inv_traces))
    assert np.isfinite(arrays['terminal_pnl']).all()
    tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(output)
    atomic_json(manifest,dict(signature=signature,episodes=len(seeds),elapsed_s=time.monotonic()-started))
    return str(output)


def load_group(paths):
    groups={}
    for path in paths:
        with np.load(path) as d:
            for k in d.files:groups.setdefault(k,[]).append(d[k])
    arrays={k:np.concatenate(v,axis=0) for k,v in groups.items()}
    order=np.argsort(arrays['seeds'])
    return {k:(v[order] if len(v)==len(order) else v) for k,v in arrays.items()}
