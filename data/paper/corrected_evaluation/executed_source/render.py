"""Render complete fixed-sample results, then publish only to Desktop staging folder."""
import os
os.environ['MPLBACKEND']='Agg'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,importlib.util,json,shutil,subprocess,sys,datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parent
P=json.loads((R/'protocol.json').read_text())
W=R/'plot_workspace';D=W/'data/paper';REF=R/'reference/data/paper'
SMOKE='--smoke' in sys.argv
OUT=R/'validation/plot_smoke' if SMOKE else R/'figures'
NAMES=['dqn_vs_as_vs_glft','total_pnl_0001','MM_inv_0001','regime_switching_one_episode_example','regime_trained_final_phase_b','bayes_single_estimation_example','quote_exposure_single_estimation_example','degradation_vs_tau_regime','regime_trained_pnl_mean_across_time','inventory_distribution_phaseA_phaseB','pnl_distribution_phaseA_phaseB','regime_tau_sweep_decay_phasea_vs_phaseb','random_tau_stress','correlated_regime_stress','scenario_bandit_random_tau','scenario_bandit_correlated_regime_stress']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def table(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadnp(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def module(name):
 spec=importlib.util.spec_from_file_location(name,W/'repro'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(fig,name,dpi=200,tight=True):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=dpi,**({'bbox_inches':'tight'} if tight else {}))
 plt.close(fig)
def case_rows(name):
 c=next(c for c in P['cases'] if c['name']==name)
 rows=[json.loads((R/'data'/name/f'{s}.json').read_text()) for s in c['seeds']]
 assert len(rows)==len(c['seeds']) and [r['seed'] for r in rows]==c['seeds']
 for row in rows:
  assert row['events']==c['events'] and row['warmup']==c['warmup'] and row['networks_unchanged'] and row['variant']=='both'
  assert row['protocol_id']==sha(R/'protocol.json') and row['checkpoint_sha256']==P['models'][c['model']]['sha256']
 return rows
def values(name,key='terminal_pnl'):return np.array([x[key] for x in case_rows(name)])
def trace(name,seed):
 row=json.loads((R/'data'/name/f'{seed}.json').read_text());path=R/row['trace_file'];assert sha(path)==row['trace_sha256'];return loadnp(path)
def prepare():
 if SMOKE:
  for rel in ['historical_results/episode_seed124.npz','historical_results/random.npz','historical_results/correlated.npz','historical_results/degradation_summary.csv','regime_mc.npz','bandit_random.npz','bandit_correlated.npz']:
   source=REF/rel
   if source.exists():shutil.copy2(source,D/rel)
  # Stationary example is an archived genuine 5000-event trace.
  source=REF/'historical_results/episode_seed124.npz';shutil.copy2(source,D/'historical_results/episode_seed124.npz')
  mc=loadnp(D/'regime_mc.npz');mc['trace_step']=np.r_[np.arange(0,100000,20),99999];np.savez_compressed(D/'regime_mc.npz',**mc)
  source=REF/'tau_sweep_summary.csv';shutil.copy2(source,D/'tau_sweep_summary.csv')
  return loadnp(REF/'episode/single_episode.npz'),loadnp(REF/'episode/signal_example.npz')
 # All summaries use actual complete episode vectors, without rounded intermediate values.
 summaries=[]
 for c in P['cases']:
  rows=case_rows(c['name']);y=np.array([x['terminal_pnl'] for x in rows]);n=len(y)
  summaries.append(dict(case=c['name'],model=c['model'],family=c['family'],tau=c['tau'],n=n,events=c['events'],warmup=c['warmup'],mean_pnl=float(y.mean()),std_pnl=float(y.std(ddof=1)) if n>1 else 0.,sem_pnl=float(y.std(ddof=1)/np.sqrt(n)) if n>1 else 0.,minimum_pnl=float(y.min()),maximum_pnl=float(y.max()),mean_abs_inventory=float(np.mean([x['mean_abs_inventory'] for x in rows])),inventory_limit_fraction=float(np.mean([x['inventory_limit_fraction'] for x in rows]))))
 table(R/'reports/summary.csv',summaries)
 degradation=[]
 for phi,model in [(.001,'A_001'),(.005,'A_005')]:
  for tau in [0,15,30,60,120,240]:
   y=values(f'degradation_{model}_tau{tau}');assert len(y)==100
   degradation.append(dict(phi=phi,tau=tau,n=len(y),mean=float(y.mean()),std=float(y.std(ddof=1))))
 table(D/'historical_results/degradation_summary.csv',degradation)
 sweep=[]
 for tau in [0,15,30,60,120,240]:
  a=values(f'sweep_A_001_tau{tau}');b=values(f'sweep_B_tau{tau}');assert len(a)==len(b)==100
  sweep.append(dict(tau=tau,A_mean_pnl=float(a.mean()),A_sem_pnl=float(a.std(ddof=1)/10),B_new_mean_pnl=float(b.mean()),B_new_sem_pnl=float(b.std(ddof=1)/10)))
 table(D/'tau_sweep_summary.csv',sweep)
 mc={}
 for name,label in [('recovery_A_stationary','base'),('recovery_A_001','regime'),('recovery_B','phaseb')]:
  rows=case_rows(name);assert len(rows)==100;arrays=[trace(name,x['seed']) for x in rows]
  for a in arrays:np.testing.assert_array_equal(a['trace_step'],np.r_[np.arange(0,100000,20),99999])
  mc['pnls_'+label+'_all']=values(name)
  mc['pnl_traj_'+label]=np.stack([a['trace_pnl'] for a in arrays]);mc['inv_traj_'+label]=np.stack([a['trace_inventory'] for a in arrays])
 mc['trace_step']=arrays[0]['trace_step'];np.savez_compressed(D/'regime_mc.npz',**mc)
 for family in ['random','correlated']:
  fixed=values(f'{family}_fixed_B');b=values(f'{family}_B');c=values(f'{family}_C');assert len(fixed)==50 and len(b)==len(c)==100
  np.savez_compressed(D/'historical_results'/f'{family}.npz',fixed_pnl=fixed,stress_pnl=b[:50],paired_pnl_difference=b[:50]-fixed)
  np.savez_compressed(D/f'bandit_{family}.npz',B_pnl=b,C_pnl=c)
 a=trace('illustration_A_stationary_124',124)
 np.savez_compressed(D/'historical_results/episode_seed124.npz',event=a['trace_step'],pnl=a['trace_pnl'],inventory=a['trace_inventory'])
 episode={}
 for case,label in [('illustration_A_001_stationary_42','A_baseline'),('illustration_A_001_regime_42','A_regime'),('illustration_B_regime_42','B_regime')]:
  a=trace(case,42);assert len(a['trace_step'])==100000
  episode[label+'_MM_TotalPnL']=a['trace_pnl'];episode[label+'_MM_Inventory']=a['trace_inventory']
  if label=='A_regime':episode['p_buy']=a['p_buy']
 a=trace('illustration_signals_B_123',123)
 signals=dict(a,step=a['trace_step'],inventory=a['trace_inventory'],pnl=a['trace_pnl'])
 np.savez_compressed(R/'reports/single_episode.npz',**episode);np.savez_compressed(R/'reports/signal_example.npz',**signals)
 return episode,signals
def plotting(episode,signals):
 historical=module('plot_historical_results')
 historical.plot_episode(OUT);historical.plot_degradation(OUT)
 for family in ['random','correlated']:historical.plot_stress(OUT,family)
 for name in ['plot_paper_sweep.py','regen_regime_mc_figs.py']:
  subprocess.run([sys.executable,'-B',str(W/'repro'/name),'--output',str(OUT)],check=True,cwd=W)
 ep=module('plot_paper_episode')
 with plt.rc_context(ep.EPISODE_STYLE):
  for include,name in [(False,'regime_switching_one_episode_example'),(True,'regime_trained_final_phase_b')]:save(ep.plot_episode(episode,include),name,dpi=150,tight=False)
  for name,fig in ep.plot_signals(signals):save(fig,name,dpi=100)
 bandit=module('regen_scenario_bandit_shaded')
 for family,name in [('random','scenario_bandit_random_tau'),('correlated','scenario_bandit_correlated_regime_stress')]:
  a=loadnp(D/f'bandit_{family}.npz');bandit._plot_pair(a['B_pnl'],a['C_pnl'],phaseb_label_stem='Algorithm B',out_path=OUT/f'{name}.png')
 # The GLFT results and analytical transient are unchanged; replace only the RLMM series.
 rows=list(csv.DictReader((R/'reference/glft/comparison_all_probabilities.csv').open()))
 rows=[{k:float(v) for k,v in row.items()} for row in rows]
 for row in rows:
  if not SMOKE:
   y=values(f'asymmetry_A_p{row["p_buy_mo"]:g}');assert len(y)==100
   row.update(dqn_empirical_mean=float(y.mean()),dqn_empirical_std=float(y.std(ddof=1)),dqn_empirical_se_mean=float(y.std(ddof=1)/10))
 table(OUT/'asymmetry_summary.csv',rows)
 tr=module('glft_transient');fig,ax=plt.subplots(figsize=(9,5));x=np.array([r['p_buy_mo'] for r in rows]);arr=lambda k:np.array([r[k] for r in rows])
 ax.errorbar(x,arr('dqn_empirical_mean'),yerr=1.96*arr('dqn_empirical_se_mean'),fmt='o-',color=tr.ORANGE,capsize=4,linewidth=2,markersize=7,label='DQN (φ=1.0e-03)')
 ax.errorbar(x,arr('glft_empirical_mean'),yerr=1.96*arr('glft_empirical_se_mean'),fmt='s-',color=tr.BLUE,capsize=4,linewidth=2,markersize=6,label='GLFT (γ=1.0e-06)')
 ax.plot(x,arr('transient_full_pnl'),'o--',color=tr.BLUE,linewidth=2,markersize=6,label=r'GLFT analytical ($q_0=0$)');tr.style_axes(ax);ax.legend(fontsize=11,loc='best');fig.tight_layout();save(fig,'dqn_vs_as_vs_glft')
def report():
 from scipy.stats import ttest_rel
 comparisons=[]
 for a,b,label in [('recovery_A_001','recovery_B','Recovery B minus A'),('random_B','random_C','Random C minus B'),('correlated_B','correlated_C','Correlated C minus B')]+[(f'sweep_A_001_tau{t}',f'sweep_B_tau{t}',f'Sweep tau={t}: B minus A') for t in [0,15,30,60,120,240]]:
  x=values(a);y=values(b);d=y-x
  comparisons.append(dict(comparison=label,n=len(d),left_mean=float(x.mean()),right_mean=float(y.mean()),paired_difference=float(d.mean()),paired_difference_sem=float(d.std(ddof=1)/np.sqrt(len(d))),paired_t_pvalue=float(ttest_rel(y,x).pvalue)))
 table(R/'reports/algorithm_comparisons.csv',comparisons)
 old={}
 for x in csv.DictReader((REF/'historical_results/A_sweeps/per_episode.csv').open()):
  model='A_001' if float(x['phi'])==.001 else 'A_005';name=f'{x["family"]}_{model}_tau{int(float(x["tau"]))}';old[(name,int(x['seed']))]=float(x['terminal_pnl'])
 for x in csv.DictReader((REF/'tau_sweep_B_runs.csv').open()):
  tau=int(float(x.get('tau',x.get('tau_r','0'))));old[(f'sweep_B_tau{tau}',int(x['seed']))]=float(x.get('final_pnl',x.get('terminal_pnl')))
 mc=loadnp(REF/'regime_mc.npz')
 for name,key in [('recovery_A_stationary','pnls_base_all'),('recovery_A_001','pnls_regime_all'),('recovery_B','pnls_phaseb_all')]:
  for seed,value in zip(range(123,223),mc[key]):old[(name,seed)]=float(value)
 for family,start in [('random',500123),('correlated',700123)]:
  a=loadnp(REF/f'bandit_{family}.npz')
  for model in ['B','C']:
   for seed,value in zip(range(start,start+100),a[model+'_pnl']):old[(f'{family}_{model}',seed)]=float(value)
  a=loadnp(REF/'historical_results'/f'{family}.npz')
  for seed,value in zip(a['seeds'],a['fixed_pnl']):old[(f'{family}_fixed_B',int(seed))]=float(value)
 changed=[]
 for c in P['cases']:
  if not all((c['name'],s) in old for s in c['seeds']):continue
  before=np.array([old[(c['name'],s)] for s in c['seeds']]);after=values(c['name']);d=after-before
  changed.append(dict(case=c['name'],n=len(after),historical_mean=float(before.mean()),corrected_mean=float(after.mean()),mean_change=float(d.mean()),historical_sd=float(before.std(ddof=1)),corrected_sd=float(after.std(ddof=1)),changed_episodes=int((abs(d)>1e-9).sum()),comparison='Historical reference versus new corrected evaluation; not a fresh-original ablation'))
 table(R/'reports/historical_vs_corrected.csv',changed)
 text=['AVALIAÇÕES COMPLETAS FINALIZADAS','',f"{P['total_episodes']} episódios; {P['total_events']:,} eventos avaliados, além do aquecimento.",'Correções: avaliação estável do C51 e preservação FIFO. Pesos congelados.','Paper e apresentação não foram alterados.','Figuras PNG/PDF na raiz de glft_regeneration, com os nomes do paper.','','Comparações entre algoritmos:']
 for x in comparisons:text.append(f"{x['comparison']}: {x['left_mean']:+.5f} -> {x['right_mean']:+.5f}; n={x['n']}; p pareado={x['paired_t_pvalue']:.4g}.")
 text+=['','Mudanças frente ao histórico (mesmas seeds, históricos não reexecutados):']
 for x in changed:text.append(f"{x['case']}: média {x['historical_mean']:+.5f} -> {x['corrected_mean']:+.5f}; desvio {x['historical_sd']:.5f} -> {x['corrected_sd']:.5f}.")
 text+=['','Tabelas e dados individuais estão em full_evaluation_20261008/reports e per_episode.csv.','Os testes são condicionais aos checkpoints congelados, sem avaliar a variabilidade entre treinamentos.']
 (R/'RESULTADOS.txt').write_text('\n'.join(text)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 episode,signals=prepare();plotting(episode,signals)
 images={}
 for name in NAMES:
  for ext in ['png','pdf']:assert (OUT/f'{name}.{ext}').is_file(),(name,ext)
  with Image.open(OUT/f'{name}.png') as im:im.verify()
  images[name]={ext:sha(OUT/f'{name}.{ext}') for ext in ['png','pdf']}
 dump(OUT/'validation.json',dict(status='passed',uses_historical_data_for_smoke_only=SMOKE,figures=images))
 if SMOKE:print('PLOT SMOKE PASSED');return
 report()
 for name,h in json.loads((R/'source_manifest.json').read_text()).items():assert sha(R/name)==h,name
 # Publish only after every final figure and complete sample have passed validation.
 backup=R/'previous_figures';backup.mkdir(exist_ok=True)
 for name in NAMES:
  for ext in ['png','pdf']:
   dest=R.parent/f'{name}.{ext}'
   if dest.exists() and not (backup/dest.name).exists():shutil.copy2(dest,backup/dest.name)
   tmp=dest.with_suffix('.publishing'+dest.suffix);shutil.copy2(OUT/dest.name,tmp);tmp.replace(dest)
 shutil.copy2(R/'RESULTADOS.txt',R.parent/'RESULTADOS_AVALIACOES_COMPLETAS.txt')
 old_manifest=json.loads((R.parent/'reports/figure_manifest.json').read_text())
 for name in NAMES:
  if name not in old_manifest['expected']:old_manifest['expected'].append(name)
  old_manifest['completed'][name]=images[name]['png']
 old_manifest['complete']=all((R.parent/(n+'.png')).exists() for n in old_manifest['expected'])
 dump(R.parent/'reports/figure_manifest.json',old_manifest)
 import pymupdf
 doc=pymupdf.open()
 for name in old_manifest['expected']:
  with pymupdf.open(R.parent/(name+'.pdf')) as part:doc.insert_pdf(part)
 target=R.parent/'TODAS_AS_FIGURAS.pdf'
 if target.exists() and not (backup/target.name).exists():shutil.copy2(target,backup/target.name)
 doc.save(R/'TODAS_AS_FIGURAS.pdf');doc.close();shutil.copy2(R/'TODAS_AS_FIGURAS.pdf',target)
 dump(R/'publication.json',dict(status='complete',published_at=datetime.datetime.now().astimezone().isoformat(),destination=str(R.parent),figures=images,paper_and_slides_modified=False))
 print('FULL FIGURES PUBLISHED TO DESKTOP FOLDER')
if __name__=='__main__':main()
