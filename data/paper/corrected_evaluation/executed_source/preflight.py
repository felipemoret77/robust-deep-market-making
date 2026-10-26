from pathlib import Path
import contextlib,json,platform,sys,time
import evaluate as e
import numpy as np
import torch
R=e.ROOT
def main():
 checks=[]
 for name,h in json.loads((R/'source_manifest.json').read_text()).items():assert e.r.sha(R/name)==h,name
 assert torch.__version__=='2.10.0' and np.__version__=='2.4.2'
 with (R/'logs/preflight.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  for model in ['A_001','A_005','B','C']:
   case=dict(name='trace_test',family='recovery',model=model,tau=60,events=3000,warmup=1000,trace=True)
   compact,ca=e.simulate(case,123)
   full,fa=e.simulate(case,123,force_full=True)
   for key in e.METRICS:np.testing.assert_allclose(compact[key],full[key],rtol=0,atol=1e-10)
   np.testing.assert_array_equal(ca['trace_step'],fa['trace_step'])
   np.testing.assert_array_equal(ca['trace_inventory'],fa['trace_inventory'])
   np.testing.assert_array_equal(ca['trace_pnl'],fa['trace_pnl'].astype(np.float32))
   checks.append(dict(model=model,compact_and_full_equal=True,weights_unchanged=compact['networks_unchanged']))
  # Exact original/corrected pilot agreement when adding dense MC traces.
  for name in ['recovery_A_001','recovery_B']:
   e.run_job((name,123,e.r.sha(R/'protocol.json')))
   case=next(c for c in e.P['cases'] if c['name']==name)
   assert e.valid_record(case,123,e.r.sha(R/'protocol.json'))
   row=json.loads(e.record_path(case,123).read_text());assert row['pilot_metrics_reproduced'] and row['trace_points']==5001
   checks.append(dict(case=name,seed=123,exact_pilot_metrics_reproduced=True,trace_points=row['trace_points']))
 reuse=json.loads((R/'validation/reuse.json').read_text())
 for row in reuse['records']:
  case=next(c for c in e.P['cases'] if c['name']==row['case']);assert e.valid_record(case,row['seed'],e.r.sha(R/'protocol.json'))
 # Every stress pair uses identical scheduled environments by seed, without equating realized paths.
 helpers,signals=e.r.paper.evaluation_helpers(e.P['models']['B']['metadata'])
 for family in ['random','correlated']:
  left=next(c for c in e.P['cases'] if c['name']==family+'_B');right=next(c for c in e.P['cases'] if c['name']==family+'_C')
  assert left['seeds']==right['seeds'] and left['events']==right['events']==500000
  s1=e.schedule_for(left,helpers,left['seeds'][0],500000);s2=e.schedule_for(right,helpers,right['seeds'][0],500000)
  np.testing.assert_array_equal([s1(i) for i in range(1000)],[s2(i) for i in range(1000)])
 result=dict(status='passed',checks=checks,reused_records_verified=len(reuse['records']),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,protocol_sha256=e.r.sha(R/'protocol.json'))
 e.r.atomic(R/'validation/preflight.json',result);print(json.dumps(result))
if __name__=='__main__':main()
