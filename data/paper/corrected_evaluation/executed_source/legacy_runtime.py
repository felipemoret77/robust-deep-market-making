"""Frozen-source paired evaluation; writes only inside this campaign."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import contextlib,hashlib,json,math,sys,time
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib')
sys.path.insert(0,str(ROOT/'source_snapshot/repro'))
import paper_runtime as paper
import frontier
import numpy as np
import torch
from candidate import instrument_controller
try:torch.set_num_threads(1);torch.set_num_interop_threads(1)
except RuntimeError:pass
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(ctrl):
 h=hashlib.sha256()
 for name in ['q_net','target_net']:
  for key,value in getattr(ctrl,name).state_dict().items():h.update((name+key).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load(model,variant,out):
 if model=='A_005':
  spec=PROTOCOL['models'][model];ctrl=frontier.load_controller(spec,out)
  meta=torch.load(ROOT/'source_snapshot'/spec['checkpoint'],map_location='cpu',weights_only=True)['meta']
 else:ctrl,meta=paper.load_controller('A' if model=='A_001' else model,out)
 assert not ctrl.enable_learning and ctrl.epsilon==0 and not ctrl.q_net.training and not ctrl.target_net.training
 counters=dict(decisions=0,preserved_recovery_quotes=0);instrument_controller(ctrl,variant,counters)
 return ctrl,meta,counters,digest(ctrl)
def simulate(case,variant,seed,*,events=None,warmup=None,full=False,folder=None):
 events=case['events'] if events is None else events;warmup=case['warmup'] if warmup is None else warmup
 folder=Path(folder) if folder else ROOT/'runtime'/f'{os.getpid()}_{case["model"]}_{variant}'
 ctrl,meta,counters,before=load(case['model'],variant,folder)
 try:
  helpers,signals=paper.evaluation_helpers(meta)
  family='sweep' if case['family']=='degradation' else case['family']
  schedule=paper.make_schedule(helpers,family,seed,events,case['tau'])
  with torch.inference_mode():
   _,out=helpers.run_one_sim(schedule,seed=seed,controller=ctrl,total_events=events,equil_steps=warmup,signal_config=signals,metrics_only=not full,metrics_trace_max_points=1001 if not full else None)
  assert before==digest(ctrl),'Network state changed'
  assert not ctrl.enable_learning and len(ctrl.memory)==0
  if full:
   result=dict(terminal_pnl=float(out['MM_TotalPnL'].iloc[-1]),terminal_inventory=float(out['MM_Inventory'].iloc[-1]),mean_abs_inventory=float(out['MM_Inventory'].abs().mean()),inventory_limit_fraction=float((out['MM_Inventory'].abs()>=8).mean()))
  else:
   assert out['n_steps']==events
   result=dict(terminal_pnl=float(out['MM_TotalPnL']),terminal_inventory=float(out['MM_Inventory']),mean_abs_inventory=float(out['mean_abs_inventory']),inventory_limit_fraction=float(out['limit_frac']))
  assert all(math.isfinite(v) for v in result.values())
  assert 0<=result['inventory_limit_fraction']<=1 and abs(result['terminal_inventory'])<=8
  result.update(events=events,warmup=warmup,networks_unchanged=True,**counters)
  return result
 finally:ctrl.writer.close()
def record_path(case,variant,seed):return ROOT/'data'/case['name']/variant/f'{seed}.json'
def validate_references(case,seed,record):
 expected=json.loads((ROOT/'references.json').read_text())[f'{case["name"]}/{seed}']
 differences={key:float(record[key]-value) for key,value in expected.items()}
 assert all(math.isfinite(value) for value in differences.values())
 matches=all(abs(value)<1e-9 for value in differences.values())
 if not matches:
  atomic(ROOT/'diagnostics/historical_reference_differences'/f'{case["name"]}_{seed}.json',dict(case=case['name'],seed=seed,expected=expected,observed={key:record[key] for key in expected},differences=differences,tolerance=1e-9,status='unresolved_reference_discrepancy',comparison_baseline='fresh original execution; archived outcome is not substituted'))
 return dict(historical_terminal_outcome_reproduced=matches,historical_reference_deltas=differences)
def pair(job):
 case_name,seed,protocol_id=job;case=next(c for c in PROTOCOL['cases'] if c['name']==case_name);rows=[]
 with (ROOT/'logs'/f'{case_name}_{seed}.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  for variant in PROTOCOL['variants']:
   path=record_path(case,variant,seed)
   if path.exists():
    row=json.loads(path.read_text());assert row['protocol_id']==protocol_id and row['networks_unchanged']
   else:
    atomic(ROOT/'runtime'/f'worker_{os.getpid()}.json',dict(state='running',case=case_name,seed=seed,variant=variant,events=case['events'],started_unix=time.time()))
    started=time.monotonic();result=simulate(case,variant,seed)
    row=dict(case=case_name,model=case['model'],family=case['family'],tau=case['tau'],seed=seed,variant=variant,protocol_id=protocol_id,elapsed_seconds=time.monotonic()-started,checkpoint_sha256=PROTOCOL['models'][case['model']]['sha256'],**result)
    if variant=='original':row.update(validate_references(case,seed,row))
    atomic(path,row)
   assert row['case']==case_name and row['seed']==seed and row['variant']==variant and row['events']==case['events']
   if variant=='original':
    checked=validate_references(case,seed,row)
    assert row['historical_terminal_outcome_reproduced']==checked['historical_terminal_outcome_reproduced']
   rows.append(row)
 return rows
