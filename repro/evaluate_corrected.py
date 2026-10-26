"""Replay selected corrected paper evaluations using frozen checkpoints.

No training; no writes to paper assets. --verify-archive requires full horizons.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse, contextlib, gc, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import paper_runtime as paper
import frontier
from paper_evaluation_corrections import instrument_controller
ROOT = paper.ROOT
ARCHIVE = ROOT / 'data/paper/corrected_evaluation'
PROTOCOL = json.loads((ARCHIVE / 'protocol.json').read_text())
OUTPUT = ROOT / 'outputs/corrected_evaluation'

def network_digest(ctrl):
    h = hashlib.sha256()
    for name in ('q_net', 'target_net'):
        for key, value in getattr(ctrl, name).state_dict().items():
            h.update((name+key).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_controller(model, variant, output):
    spec = PROTOCOL['models'][model]
    assert paper.file_hash(ROOT / spec['checkpoint']) == spec['sha256']
    if model == 'A_005':
        ctrl = frontier.load_controller(spec, output, c51_mode='stable_float64_v1')
        meta = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=True)['meta']
    else:
        ctrl, meta = paper.load_controller('A' if model == 'A_001' else model, output, c51_mode='stable_float64_v1')
    assert not ctrl.enable_learning and ctrl.epsilon == 0
    assert not ctrl.q_net.training and not ctrl.target_net.training
    counters = dict(decisions=0, preserved_recovery_quotes=0)
    instrument_controller(ctrl, variant, counters)
    return ctrl, meta, counters, network_digest(ctrl)

def schedule_for(case,helpers,seed,events):
 if case['family']=='asymmetry':return float(case['probability'])
 if case['family']=='fixed_stress':return helpers.build_regime_schedule(events,seed=seed+10000000,distribution='exponential',tau=60)
 if case['family']=='illustration_regime':return helpers.build_regime_schedule(events,seed=case['schedule_seed'],distribution='exponential',tau=60)
 return paper.make_schedule(helpers,'sweep' if case['family']=='degradation' else case['family'],seed,events,case['tau'])

def simulate(case,seed,*,variant='both',events=None,warmup=None,force_full=None):
 events=case['events'] if events is None else events
 warmup=case['warmup'] if warmup is None else warmup
 full=case.get('full_frame',False) if force_full is None else force_full
 folder=OUTPUT/'runtime'/f'{os.getpid()}_{case["model"]}_{variant}'
 ctrl,meta,counters,before=load_controller(case['model'],variant,folder)
 try:
  helpers,signals=paper.evaluation_helpers(meta)
  schedule=schedule_for(case,helpers,seed,events)
  max_points=(events if case['name'].startswith('illustration_') else 5000) if case['trace'] else None
  with torch.inference_mode():
   messages,out=helpers.run_one_sim(schedule,seed=seed,controller=ctrl,total_events=events,equil_steps=warmup,signal_config=signals,metrics_only=not full,metrics_trace_max_points=max_points)
  assert network_digest(ctrl)==before and not ctrl.enable_learning and len(ctrl.memory)==0
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


def replay(case, seed, output, *, verify=False, events=None, warmup=None, allow_environment_mismatch=False):
    global OUTPUT
    OUTPUT = Path(output)
    from evaluation_environment import check_environment
    environment=check_environment(allow_mismatch=allow_environment_mismatch)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    with (OUTPUT / 'rollout.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        metrics, arrays = simulate(case, seed, events=events, warmup=warmup)
    result = dict(case=case['name'], seed=seed, variant='both',
                  checkpoint_sha256=PROTOCOL['models'][case['model']]['sha256'],
                  protocol_sha256=paper.file_hash(ARCHIVE/'protocol.json'), **metrics)
    if verify:
        from corrected_evaluation import records
        reference = next(r for r in records() if r['case'] == case['name'] and int(r['seed']) == seed)
        for key in ('terminal_pnl','terminal_inventory','mean_abs_inventory','inventory_limit_fraction','decisions','preserved_recovery_quotes'):
            np.testing.assert_allclose(metrics[key], float(reference[key]), rtol=0, atol=1e-10, err_msg=key)
        if arrays:
            with np.load(ARCHIVE / reference['trace_file'], allow_pickle=False) as saved:
                assert set(arrays) == set(saved.files)
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], saved[key], err_msg=key)
        result['archive_verified'] = True
    if arrays:
        np.savez_compressed(OUTPUT/'trace.npz', **arrays)
    (OUTPUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result, arrays

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[c['name'] for c in PROTOCOL['cases']], required=True)
    parser.add_argument('--seeds', type=int, default=1, help='First N recorded seeds for the selected case; default 1.')
    parser.add_argument('--events', type=int, help='Override horizon for a smoke check only.')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--verify-archive', action='store_true')
    parser.add_argument('--allow-environment-mismatch', action='store_true', help='Record an explicit compatibility check outside the reference environment.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = next(c for c in PROTOCOL['cases'] if c['name']==args.case)
    if not 1 <= args.seeds <= len(case['seeds']):parser.error('Requested seed count outside the recorded protocol.')
    if args.verify_archive and (args.events is not None or args.warmup is not None):
        parser.error('Archive verification requires the original event horizon and warmup.')
    if args.events is not None and args.events <= 0:parser.error('--events must be positive.')
    if args.warmup is not None and args.warmup < 0:parser.error('--warmup must be nonnegative.')
    dest = args.output.resolve()
    if dest == ROOT or any(dest == ROOT/f or ROOT/f in dest.parents for f in ('paper','data','src','repro','config','checkpoints')):
        parser.error('Choose a separate output directory.')
    if dest.exists() and any(dest.iterdir()):parser.error('Use a new output directory to preserve previous results.')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from corrected_evaluation import validate
    validate()
    results=[]
    for seed in case['seeds'][:args.seeds]:
        row,_=replay(case,seed,dest/str(seed),verify=args.verify_archive,events=args.events,warmup=args.warmup,allow_environment_mismatch=args.allow_environment_mismatch)
        results.append(row);print(json.dumps(row),flush=True)
    (dest/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1),indent=2)+'\n')
if __name__ == '__main__':main()
