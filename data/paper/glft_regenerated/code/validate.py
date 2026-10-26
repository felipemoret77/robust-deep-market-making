from pathlib import Path
import contextlib,json,time
import numpy as np
import experiment_runtime as r
ROOT=r.ROOT

def run():
 for rel,digest in json.loads((ROOT/'source_manifest.json').read_text()).items():assert r.sha(ROOT/rel)==digest,rel
 old=json.loads((ROOT/'inputs/original_experiments.json').read_text());new=json.loads((ROOT/'protocol.json').read_text())['parameters']
 checked=[];start=time.monotonic()
 with (ROOT/'validation/smoke.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  for name in ['fig2_gamma_1e-06','fig3_p_0.2','fig4_new_tau_60','fig2_at_best','eta']:
   e=next(e for e in old['evaluation_experiments'] if e['spec']['name']==name)
   file=r.run_chunk((e['spec'],e['params'],e['seeds'][:1]));got=np.load(file);ref=np.load(ROOT/'validation/reference'/f'{name}.npz')
   for k in ref.files:
    a=got[k];b=ref[k][:1] if len(ref[k]) else ref[k]
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,err_msg=name+':'+k)
   assert got['preserved_recovery_quotes'].sum()==0
   checked.append(dict(experiment=name,old_parameters_match_archive=True))
  # New-calibration compact metrics vs full event log, and deterministic restart.
  a=r.base.simulate(new,123456,events=5000,trace=True)
  b=r.base.simulate(new,123456,events=5000,full=True)
  np.testing.assert_allclose(a['MM_TotalPnL'],b['MM_TotalPnL'].iloc[-1],atol=1e-12,rtol=0)
  np.testing.assert_array_equal(a['trace_inventory'],b['MM_Inventory'].to_numpy())
  np.testing.assert_allclose(a['trace_pnl'],b['MM_TotalPnL'].to_numpy(),atol=1e-6,rtol=1e-6)
  c=r.base.simulate(new,123456,events=5000,trace=True)
  np.testing.assert_array_equal(a['trace_pnl'],c['trace_pnl'])
  np.testing.assert_array_equal(a['trace_inventory'],c['trace_inventory'])
  # Naive is parameter-independent: compare complete new-parameter trace to archive.
  e=next(e for e in old['evaluation_experiments'] if e['spec'].get('naive'))
  f=r.run_chunk((e['spec'],new,e['seeds'][:2]));d=np.load(f);ref=np.load(ROOT/'validation/reference/fig2_at_best.npz')
  for k in ref.files:np.testing.assert_allclose(d[k],ref[k],atol=1e-12,rtol=0)
  assert d['preserved_recovery_quotes'].sum()==0
 result=dict(status='passed',historical_checks=checked,new_compact_vs_full_log=True,new_deterministic_repeat=True,naive_reuse_validated=True,elapsed_seconds=time.monotonic()-start)
 r.atomic(ROOT/'validation/preflight.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
