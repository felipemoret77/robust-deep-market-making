import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import contextlib, gc, json, math, time
import legacy_runtime as r
import numpy as np
import torch
ROOT=r.ROOT
P=r.PROTOCOL
METRICS=['terminal_pnl','terminal_inventory','mean_abs_inventory','inventory_limit_fraction','decisions','preserved_recovery_quotes']

def schedule_for(case,helpers,seed,events):
 if case['family']=='asymmetry':return float(case['probability'])
 if case['family']=='fixed_stress':return helpers.build_regime_schedule(events,seed=seed+10000000,distribution='exponential',tau=60)
 if case['family']=='illustration_regime':return helpers.build_regime_schedule(events,seed=case['schedule_seed'],distribution='exponential',tau=60)
 return r.paper.make_schedule(helpers,'sweep' if case['family']=='degradation' else case['family'],seed,events,case['tau'])

def simulate(case,seed,*,variant='both',events=None,warmup=None,force_full=None):
 events=case['events'] if events is None else events
 warmup=case['warmup'] if warmup is None else warmup
 full=case.get('full_frame',False) if force_full is None else force_full
 folder=ROOT/'runtime'/f'{os.getpid()}_{case["model"]}_{variant}'
 ctrl,meta,counters,before=r.load(case['model'],variant,folder)
 try:
  helpers,signals=r.paper.evaluation_helpers(meta)
  schedule=schedule_for(case,helpers,seed,events)
  max_points=(events if case['name'].startswith('illustration_') else 5000) if case['trace'] else None
  with torch.inference_mode():
   messages,out=helpers.run_one_sim(schedule,seed=seed,controller=ctrl,total_events=events,equil_steps=warmup,signal_config=signals,metrics_only=not full,metrics_trace_max_points=max_points)
  assert r.digest(ctrl)==before and not ctrl.enable_learning and len(ctrl.memory)==0
  assert not ctrl.q_net.training and not ctrl.target_net.training
  arrays={}
  if full:
   assert len(out)==len(messages)==events
   pnl=out['MM_TotalPnL'].to_numpy(dtype=np.float64);inv=out['MM_Inventory'].to_numpy(dtype=np.float64)
   result=dict(terminal_pnl=float(pnl[-1]),terminal_inventory=float(inv[-1]),mean_abs_inventory=float(abs(inv).mean()),inventory_limit_fraction=float((abs(inv)>=8).mean()))
   arrays.update(trace_step=np.arange(events),trace_pnl=pnl,trace_inventory=inv)
   types=messages['Type'].to_numpy(dtype=np.int64);arrays['message_type']=types
   if callable(schedule):
    n=np.cumsum(types==1);idx=np.maximum(n-1,0)
    per_mo=np.array([schedule(int(j)) for j in range(int(n[-1])+1)])
    arrays['p_buy']=np.where(n>0,per_mo[idx],.5)
    arrays['regime_boundaries']=np.array(schedule.boundaries,dtype=np.int64)
    arrays['regime_probabilities']=np.array(schedule.p_values,dtype=np.float64)
    ri=np.searchsorted(arrays['regime_boundaries'],idx,side='right')-1
    arrays['latent_bias']=2*arrays['regime_probabilities'][ri]-1
    arrays['true_age_mo']=np.where(n>0,idx-arrays['regime_boundaries'][ri]+1,0)
   if case['name']=='illustration_signals_B_123':
    for key,col in [('posterior_bias','MM_Bayes_M_Hat'),('expected_age_mo','MM_Bayes_Run_Length'),('quote_exposure','MM_Fill_Imbalance_Feature')]:arrays[key]=out[col].to_numpy(dtype=np.float64)
  else:
   assert out['n_steps']==events
   result=dict(terminal_pnl=float(out['MM_TotalPnL']),terminal_inventory=float(out['MM_Inventory']),mean_abs_inventory=float(out['mean_abs_inventory']),inventory_limit_fraction=float(out['limit_frac']))
   if case['trace']:arrays={key:out[key] for key in ['trace_step','trace_pnl','trace_inventory']}
  assert all(math.isfinite(x) for x in result.values()) and abs(result['terminal_inventory'])<=8
  assert 0<=result['inventory_limit_fraction']<=1
  if arrays:
   assert arrays['trace_step'][0]==0 and arrays['trace_step'][-1]==events-1
   assert np.isfinite(arrays['trace_pnl']).all() and np.max(abs(arrays['trace_inventory']))<=8
   np.testing.assert_allclose(arrays['trace_pnl'][-1],result['terminal_pnl'],rtol=1e-6,atol=1e-7)
  result.update(events=events,warmup=warmup,networks_unchanged=True,**counters)
  return result,arrays
 finally:
  ctrl.writer.close()
  del ctrl
  gc.collect()

def record_path(case,seed):return ROOT/'data'/case['name']/f'{seed}.json'
def valid_record(case,seed,protocol_id):
 path=record_path(case,seed)
 if not path.exists():return False
 row=json.loads(path.read_text())
 assert row['protocol_id']==protocol_id and row['events']==case['events'] and row['warmup']==case['warmup']
 assert row['variant']=='both' and row['networks_unchanged'] and row['checkpoint_sha256']==P['models'][case['model']]['sha256']
 if case['trace']:
  path=ROOT/row['trace_file'];assert path.exists() and r.sha(path)==row['trace_sha256']
 return True

def run_job(job):
 name,seed,protocol_id=job;case=next(x for x in P['cases'] if x['name']==name)
 if valid_record(case,seed,protocol_id):return json.loads(record_path(case,seed).read_text())
 with (ROOT/'logs'/f'{name}_{seed}.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  r.atomic(ROOT/'runtime'/f'worker_{os.getpid()}.json',dict(state='running',case=name,seed=seed,events=case['events'],started_unix=time.time()))
  started=time.monotonic();metrics,arrays=simulate(case,seed)
  row=dict(case=name,model=case['model'],family=case['family'],tau=case['tau'],seed=seed,variant='both',protocol_id=protocol_id,source='new_corrected_evaluation',elapsed_seconds=time.monotonic()-started,checkpoint_sha256=P['models'][case['model']]['sha256'],**metrics)
  if 'probability' in case:row['probability']=case['probability']
  if arrays:
   path=ROOT/'traces'/name/f'{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True)
   tmp=path.with_suffix(f'.{os.getpid()}.tmp')
   with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
   tmp.replace(path);row.update(trace_file=str(path.relative_to(ROOT)),trace_sha256=r.sha(path),trace_points=len(arrays['trace_step']))
  # Existing pilot points with newly recorded trajectories must reproduce the same metrics.
  ref=ROOT/'reference/pilot_recovery'/name/f'{seed}.json'
  if ref.exists():
   old=json.loads(ref.read_text())
   for key in METRICS:np.testing.assert_allclose(row[key],old[key],rtol=0,atol=1e-10,err_msg=f'Pilot replay {name}/{seed}: {key}')
   row['pilot_metrics_reproduced']=True
  r.atomic(record_path(case,seed),row)
  r.atomic(ROOT/'runtime'/f'worker_{os.getpid()}.json',dict(state='idle',last_case=name,last_seed=seed,updated_unix=time.time()))
  return row
