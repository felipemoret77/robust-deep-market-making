"""Validate the selected 4,405 corrected evaluations and all derived plot inputs.

Uses individual results and sampled trajectories. Does not simulate or train.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/paper'
ARCHIVE = DATA/'corrected_evaluation'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))

def records():
    return read_csv(ARCHIVE/'per_episode.csv')

def grouped():
    rows={}
    for row in records():rows.setdefault(row['case'],[]).append(row)
    return {k:sorted(v,key=lambda r:int(r['seed'])) for k,v in rows.items()}

def load_npz(path):
    with np.load(path,allow_pickle=False) as arrays:return {k:arrays[k].copy() for k in arrays.files}

def derive():
    groups=grouped()
    values=lambda case:np.array([float(r['terminal_pnl']) for r in groups[case]])
    trace=lambda case,seed:load_npz(ARCHIVE/next(r['trace_file'] for r in groups[case] if int(r['seed'])==seed))
    arrays={};tables={}
    mc={}
    for case,label in [('recovery_A_stationary','base'),('recovery_A_001','regime'),('recovery_B','phaseb')]:
        paths=[load_npz(ARCHIVE/r['trace_file']) for r in groups[case]]
        for d in paths:np.testing.assert_array_equal(d['trace_step'],np.r_[np.arange(0,100000,20),99999])
        mc['pnls_'+label+'_all']=values(case)
        mc['pnl_traj_'+label]=np.stack([d['trace_pnl'] for d in paths])
        mc['inv_traj_'+label]=np.stack([d['trace_inventory'] for d in paths])
    mc['trace_step']=paths[0]['trace_step'];arrays['regime_mc.npz']=mc
    for family in ('random','correlated'):
        fixed=values(family+'_fixed_B');b=values(family+'_B');c=values(family+'_C')
        arrays['historical_results/'+family+'.npz']=dict(fixed_pnl=fixed,stress_pnl=b[:50],paired_pnl_difference=b[:50]-fixed)
        arrays['bandit_'+family+'.npz']=dict(B_pnl=b,C_pnl=c)
    a=trace('illustration_A_stationary_124',124)
    arrays['historical_results/episode_seed124.npz']=dict(event=a['trace_step'],pnl=a['trace_pnl'],inventory=a['trace_inventory'])
    ep={}
    for case,label in [('illustration_A_001_stationary_42','A_baseline'),('illustration_A_001_regime_42','A_regime'),('illustration_B_regime_42','B_regime')]:
        a=trace(case,42)
        ep[label+'_MM_TotalPnL']=a['trace_pnl'];ep[label+'_MM_Inventory']=a['trace_inventory']
        if label=='A_regime':ep['p_buy']=a['p_buy']
    arrays['episode/single_episode.npz']=ep
    a=trace('illustration_signals_B_123',123)
    arrays['episode/signal_example.npz']=dict(a,step=a['trace_step'],inventory=a['trace_inventory'],pnl=a['trace_pnl'])
    tables['historical_results/degradation_summary.csv']=[]
    for phi,model in [(.001,'A_001'),(.005,'A_005')]:
        for tau in [0,15,30,60,120,240]:
            y=values(f'degradation_{model}_tau{tau}')
            tables['historical_results/degradation_summary.csv'].append(dict(phi=phi,tau=tau,n=len(y),mean=float(y.mean()),std=float(y.std(ddof=1))))
    tables['tau_sweep_summary.csv']=[];tables['tau_sweep_B_runs.csv']=[]
    for tau in [0,15,30,60,120,240]:
        a=values(f'sweep_A_001_tau{tau}');b=values(f'sweep_B_tau{tau}')
        tables['tau_sweep_summary.csv'].append(dict(tau=tau,A_mean_pnl=float(a.mean()),A_sem_pnl=float(a.std(ddof=1)/10),B_new_mean_pnl=float(b.mean()),B_new_sem_pnl=float(b.std(ddof=1)/10)))
        for r in groups[f'sweep_B_tau{tau}']:
            tables['tau_sweep_B_runs.csv'].append(dict(tau=tau,seed=int(r['seed']),terminal_pnl=float(r['terminal_pnl'])))
    return arrays,tables

def validate(output=None):
    p=json.loads((ARCHIVE/'protocol.json').read_text())
    provenance=json.loads((ARCHIVE/'provenance.json').read_text())
    for rel,h in provenance['files'].items():assert sha(ROOT/rel)==h,rel
    for original,item in provenance['executed_source_mapping'].items():
        assert sha(ROOT/item['path'])==item['sha256'],original
    canonical=json.loads((ROOT/'config/paper_experiments.json').read_text())
    specs=list(canonical['models'].values())+list(canonical['frontier_checkpoints'].values())
    for model,info in p['models'].items():
        assert sha(ROOT/info['checkpoint'])==info['sha256'],model
        assert any(x['checkpoint']==info['checkpoint'] and x['sha256']==info['sha256'] and x['metadata']==info['metadata'] for x in specs),model
    groups=grouped();assert set(groups)=={c['name'] for c in p['cases']}
    n=0;events=0;traces=0
    summaries={r['case']:r for r in read_csv(ARCHIVE/'reports/summary.csv')}
    for case in p['cases']:
        rows=groups[case['name']]
        assert [int(r['seed']) for r in rows]==case['seeds'],case['name']
        for r in rows:
            assert r['variant']=='both' and r['networks_unchanged']=='True'
            assert r['protocol_id']==sha(ARCHIVE/'protocol.json')
            assert r['model']==case['model'] and r['family']==case['family']
            assert (float(r['tau']) if r['tau'] else None)==case['tau']
            if 'probability' in case:assert float(r['probability'])==case['probability']
            assert r['checkpoint_sha256']==p['models'][case['model']]['sha256']
            assert int(r['events'])==case['events'] and int(r['warmup'])==case['warmup']
            for k in ['terminal_pnl','terminal_inventory','mean_abs_inventory','inventory_limit_fraction','decisions','preserved_recovery_quotes']:
                assert np.isfinite(float(r[k])),(case['name'],k)
            assert abs(float(r['terminal_inventory']))<=8 and 0<=float(r['inventory_limit_fraction'])<=1
            if case['trace']:
                path=ARCHIVE/r['trace_file'];assert sha(path)==r['trace_sha256'],str(path)
                arrays=load_npz(path)
                assert all(np.isfinite(v).all() for v in arrays.values())
                assert arrays['trace_step'][0]==0 and arrays['trace_step'][-1]==case['events']-1
                assert (np.diff(arrays['trace_step'])>0).all()
                assert len(arrays['trace_step'])==int(r['trace_points'])
                assert np.max(abs(arrays['trace_inventory']))<=8
                np.testing.assert_allclose(arrays['trace_pnl'][-1],float(r['terminal_pnl']),rtol=1e-6,atol=1e-7)
                traces+=1
        y=np.array([float(r['terminal_pnl']) for r in rows]);summary=summaries[case['name']]
        for key,value in [('mean_pnl',y.mean()),('std_pnl',y.std(ddof=1) if len(y)>1 else 0),('sem_pnl',y.std(ddof=1)/np.sqrt(len(y)) if len(y)>1 else 0)]:
            np.testing.assert_allclose(value,float(summary[key]),atol=1e-12,rtol=1e-12)
        assert int(summary['n'])==len(rows) and int(summary['events'])==case['events'] and int(summary['warmup'])==case['warmup']
        n+=len(rows);events+=len(rows)*case['events']
    assert n==p['total_episodes']==4405 and events==p['total_events']==526310000
    # Compare every selected plot input against independent reconstruction.
    arrays,tables=derive()
    for rel,want in arrays.items():
        got=load_npz(DATA/rel);assert set(got)==set(want),rel
        for key in want:np.testing.assert_array_equal(got[key],want[key],err_msg=rel+'/'+key)
    for rel,want in tables.items():
        got=read_csv(DATA/rel);assert len(want)==len(got)
        for first,second in zip(got,want):
            for key,value in second.items():np.testing.assert_allclose(float(first[key]),value,rtol=1e-12,atol=1e-12,err_msg=rel+'/'+key)
    for row in read_csv(ARCHIVE/'asymmetry_summary.csv'):
        y=np.array([float(r['terminal_pnl']) for r in groups[f'asymmetry_A_p{float(row["p_buy_mo"]):g}']])
        np.testing.assert_allclose([y.mean(),y.std(ddof=1),y.std(ddof=1)/10],[float(row[k]) for k in ['dqn_empirical_mean','dqn_empirical_std','dqn_empirical_se_mean']],rtol=1e-12,atol=1e-12)
    if output is not None:
        output=Path(output).resolve()
        if output==ROOT or any(output==ROOT/f or ROOT/f in output.parents for f in ('paper','data','src','config','checkpoints','repro')):raise ValueError('Use a separate output directory.')
        output.mkdir(parents=True,exist_ok=True)
        for rel,a in arrays.items():
            path=output/rel;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
        for rel,rows in tables.items():
            path=output/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(status='passed',episodes=n,evaluation_events=events,traces=traces,derived_inputs=len(arrays)+len(tables),training_rerun=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    print(json.dumps(validate(parser.parse_args().output),indent=2))
