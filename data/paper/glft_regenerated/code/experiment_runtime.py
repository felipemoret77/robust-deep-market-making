import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'snapshot/repro'))
import glft_evaluation_runtime as base
from queue_correction import instrument_benchmark
_original_make=base.make_policy
COUNTS={}
def make_policy(params,gamma=1e-6,naive=False):
 COUNTS.clear();COUNTS['preserved_recovery_quotes']=0
 return instrument_benchmark(_original_make(params,gamma,naive),8,COUNTS)
base.make_policy=make_policy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def run_chunk(job):
 spec,params,seeds=job
 signature=dict(spec=spec,parameters=params,seeds=seeds,source_manifest_sha256=sha(ROOT/'source_manifest.json'),runtime_sha256=sha(__file__),queue_sha256=sha(ROOT/'code/queue_correction.py'))
 digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
 folder=ROOT/'chunks'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
 file=folder/f'{seeds[0]}_{digest}.npz';meta=file.with_suffix('.json')
 if file.exists() and meta.exists():
  m=json.loads(meta.read_text());assert m['signature']==signature and m['sha256']==sha(file)
  return str(file)
 arr={k:[] for k in ['terminal_pnl','terminal_inventory','mean_abs_inventory','limit_fraction','fill_rate','preserved_recovery_quotes']}
 traces=[];inventories=[];started=time.monotonic()
 with file.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  for seed in seeds:
   m=base.simulate(params,seed,events=spec.get('events',5000),gamma=spec.get('gamma',1e-6),prob=spec.get('prob',.5),tau=spec.get('tau'),p_lo=spec.get('p_lo',.2),p_hi=spec.get('p_hi',.8),naive=spec.get('naive',False),trace=spec.get('trace',False),full=spec.get('eta',False))
   arr['preserved_recovery_quotes'].append(COUNTS['preserved_recovery_quotes'])
   if spec.get('eta'):
    if 'MM_NumBidFills' in m and 'MM_NumAskFills' in m:
     nf=int(m['MM_NumBidFills'].iloc[-1]+m['MM_NumAskFills'].iloc[-1])
    else:
     assert m['MM_Inventory'].iloc[0]==0
     changes=m['MM_Inventory'].diff().fillna(0).abs();assert changes.max()<=1
     nf=int((changes>0).sum())
    dt=float(m['Time'].iloc[-1]-m['Time'].iloc[0]);assert dt>0
    arr['fill_rate'].append(nf/dt)
    arr['terminal_pnl'].append(float(m['MM_TotalPnL'].iloc[-1]));arr['terminal_inventory'].append(float(m['MM_Inventory'].iloc[-1]))
   else:
    assert m['n_steps']==spec.get('events',5000)
    arr['terminal_pnl'].append(m['MM_TotalPnL']);arr['terminal_inventory'].append(m['MM_Inventory'])
    arr['mean_abs_inventory'].append(m['mean_abs_inventory']);arr['limit_fraction'].append(m['limit_frac'])
    if spec.get('trace'):
     assert len(m['trace_pnl'])==spec.get('events',5000)
     np.testing.assert_allclose(m['trace_pnl'][-1],m['MM_TotalPnL'],rtol=1e-6,atol=1e-6)
     traces.append(m['trace_pnl']);inventories.append(m['trace_inventory'])
   assert abs(arr['terminal_inventory'][-1])<=8
   print('completed',int(seed),flush=True)
 arrays={k:np.asarray(v) for k,v in arr.items()};arrays['seeds']=np.asarray(seeds,dtype=np.int64)
 if traces:arrays.update(pnl_trace=np.stack(traces),inventory_trace=np.stack(inventories))
 for v in arrays.values():assert np.isfinite(v).all()
 temp=file.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(file)
 atomic(meta,dict(signature=signature,episodes=len(seeds),evaluation_events=len(seeds)*spec.get('events',5000),elapsed_s=time.monotonic()-started,sha256=sha(file)))
 return str(file)

def merge(entry,paths):
 d=base.load_group(paths)
 np.testing.assert_array_equal(d['seeds'],np.sort(entry['seeds']))
 file=ROOT/'data'/f"{entry['spec']['name']}.npz"
 temp=file.with_suffix('.tmp.npz');np.savez_compressed(temp,**d);temp.replace(file)
 return str(file)
