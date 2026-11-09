import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import csv,gc,hashlib,json,sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data/paper/glft_regenerated'
OUT=ROOT/'outputs/glft_figures'
import glft_figures as g
import glft_transient as t
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def configure():
 plt.rcParams.update({'axes.labelsize':15,'axes.titlesize':14,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':11,'legend.title_fontsize':11,'legend.framealpha':.9,'legend.fancybox':True,'legend.borderpad':.3,'legend.handlelength':1.6})
 plt.show=lambda *a,**k:None
 g.OUT=OUT;g.DATA=ARCHIVE/'data';g.plots._save_paper_figure=g.save

def calibration():
 full=json.loads((ARCHIVE/'inputs/calibration_summary.json').read_text());fit=full['fit_05a5']
 points=[x for x in full['empirical_rows'] if .5<=x['distance']<=5]
 x=np.array([z['distance'] for z in points]);y=np.array([z['intensity'] for z in points])
 b,k=np.linalg.lstsq(np.c_[np.ones(len(x)),-x],np.log(y),rcond=None)[0]
 np.testing.assert_allclose([np.exp(b),k],[fit['A_fit'],fit['kappa']],rtol=1e-13)
 params=json.loads((ARCHIVE/'inputs/calibration_used.json').read_text())['parameters'];np.testing.assert_allclose([fit['A'],fit['kappa']],[params['A'],params['kappa']],rtol=1e-13)
 plt.rcParams.update({'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'legend.framealpha':.9})
 fig,(top,bottom)=plt.subplots(2,1,figsize=(7,5.25),sharex=True)
 xx=np.linspace(.5,5,700);yy=fit['A_fit']*np.exp(-fit['kappa']*xx)
 top.plot(x,y,'o-',color='#1f77b4',linewidth=1.5,markersize=5,label='Pooled empirical intensity')
 top.plot(xx,yy,'--',color='#ff7f0e',linewidth=1.5,label=rf'Exponential fit: $A_{{\mathrm{{fit}}}}={fit["A_fit"]:.4f},\ \kappa={fit["kappa"]:.4f}$')
 top.set_title(r'Censored waiting times, $\Delta T=0.5\,\mathrm{u}$');top.set_ylabel(r'$\widehat\Lambda(\delta)$ ($\mathrm{u}^{-1}$)');top.set_ylim(bottom=0);top.legend(loc='upper right')
 bottom.plot(x,np.log(y),'o-',color='#1f77b4',linewidth=1.5,markersize=5);bottom.plot(xx,np.log(yy),'--',color='#ff7f0e',linewidth=1.5)
 bottom.set_ylabel(r'$\log\widehat\Lambda(\delta)$');bottom.set_xlabel(r'Effective quote distance $\delta$ (ticks)');bottom.set_xticks([.5,1,2,3,4,5])
 for ax in (top,bottom):ax.grid(alpha=.3)
 fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.92,hspace=.20)
 for ext in ['png','pdf']:fig.savefig(OUT/f'GLFT_censored_waiting_times_calib_half_second.{ext}',dpi=220)
 plt.close(fig);configure()
 # Recompute points and SEM from per-seed probe counts/exposures with OLS.
 import subprocess, shutil
 target=OUT/'reports/window_normalization'
 subprocess.run([sys.executable,str(ROOT/'repro/plot_calibration_windows.py'),'--output',str(target)],check=True)
 reference=ROOT/'data/paper/calibration_windows_unweighted'
 for name in ['window_points_unweighted.csv','window_seed_statistics_unweighted.csv']:
  actual=list(csv.DictReader((target/name).open()));expected=list(csv.DictReader((reference/name).open()))
  assert len(actual)==len(expected)
  for a,b in zip(actual,expected):
   for key in b:np.testing.assert_allclose(float(a[key]),float(b[key]),rtol=1e-12,atol=1e-12)
 for ext in ['png','pdf']:shutil.copy2(target/f'censored_waiting_different_throttles.{ext}',OUT/f'censored_waiting_different_throttles.{ext}')
 atomic(OUT/'reports/calibration_validation.json',dict(status='passed',fit=fit,simulation_reused=True,per_seed_normalization='unweighted OLS',bars='sample standard deviation / sqrt(20)'))

def parameters():
 old=json.loads((ARCHIVE/'reports/analytical_parameters.json').read_text())
 par=dict(old);par.update(json.loads((ARCHIVE/'inputs/calibration_used.json').read_text())['parameters'])
 with np.load(ARCHIVE/'data/eta.npz') as d:rates=d['fill_rate'];assert len(rates)==10
 model=t.make_model(.5,dict(par,r_fill_factor=1.))
 reference=2*par['A']*np.exp(-par['kappa']*model['half_spread'])
 par.update(r_fill_factor=float(rates.mean()/reference),eta_empirical_rate=float(rates.mean()),eta_reference_rate=float(reference),delta_bar=model['half_spread'])
 atomic(OUT/'reports/analytical_parameters.json',par)
 return par

def analytical(par):
 configure();plt.rcParams['axes.prop_cycle']=cycler(color=[tuple(c) for c in plt.cm.viridis(np.linspace(.05,.95,6))])
 probs=np.linspace(.2,.8,121)
 fields=('A','kappa','sigma','gamma','T','inv_limit','r_fill_factor','tick_size','v0')
 curves=g.plots.compute_bundle_curves(probs,**{k:par[k] for k in fields})
 for p,pi in zip(probs,curves['pi']):np.testing.assert_allclose(pi,t.make_model(float(p),par)['pi'],atol=1e-12,rtol=0)
 (OUT/'data').mkdir(exist_ok=True)
 np.savez_compressed(OUT/'data/analytical_curves.npz',probs=probs,**curves)
 g.plots.save_plot_pi_lines([.3,.4,.5,.6,.7],curves);g.plots.save_plot_pi_heatmap(probs,curves);g.plots.save_plot_qbar(probs,curves);g.plots.save_plot_boundary_mass(probs,curves)
 dense=[t.calculate(float(p),par)[0] for p in probs]
 t.table(OUT/'reports/analytical_curves_dense.csv',dense)
 # PnL panels preserve the already-corrected finite-horizon convention.
 for name in ['as_miss_expected_pnl_curves','as_miss_pnl_decomposition']:
  fig,ax=plt.subplots(figsize=(8.8,5.2))
  arr=lambda k:np.array([r[k] for r in dense])
  if name=='as_miss_expected_pnl_curves':
   ax.plot(probs,arr('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread-only');ax.plot(probs,arr('transient_full_pnl'),color='crimson',lw=2.5,label='Full (with drift)')
  else:
   ax.plot(probs,arr('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread capture');ax.plot(probs,-arr('transient_drift_pnl'),color='crimson',lw=2.5,label='MtM loss');ax.plot(probs,arr('transient_full_pnl'),color='black',lw=2.5,label='Net full PnL')
  t.style_axes(ax);ax.set_title(r'Finite horizon, $q_0=0$',fontsize=13);ax.legend();fig.tight_layout();g.save(fig,name+'.png');plt.close(fig)
 t.ROOT=OUT/'reports';atomic(OUT/'reports/analytical_validation.json',t.validate(par))
 return dense

def empirical_comparison(par,dense):
 sources=list(csv.DictReader((ROOT/'data/paper/corrected_evaluation/asymmetry_summary.csv').open()))
 rows=[];raw=[];seeds=None
 for source in sources:
  p=float(source['p_buy_mo']);d=g.load(f'fig3_p_{p:g}');y=d['terminal_pnl']
  if seeds is None:seeds=d['seeds']
  np.testing.assert_array_equal(seeds,d['seeds']);assert len(y)==100
  row=t.calculate(p,par)[0]
  for k in ['dqn_empirical_mean','dqn_empirical_std','dqn_empirical_se_mean']:row[k]=float(source[k])
  row.update(glft_empirical_mean=float(y.mean()),glft_empirical_std=float(y.std(ddof=1)),glft_empirical_se_mean=float(y.std(ddof=1)/np.sqrt(len(y))),n_glft_episodes=len(y))
  for kind in ['stationary','transient']:
   residual=row[kind+'_full_pnl']-row['glft_empirical_mean'];row[kind+'_residual']=residual;row[kind+'_residual_in_se']=residual/row['glft_empirical_se_mean']
  rows.append(row);raw.append(y)
 t.table(OUT/'reports/comparison_all_probabilities.csv',rows)
 dest=OUT/'reports/transient_diagnostics';dest.mkdir(exist_ok=True);t.ROOT=dest
 t.figures(rows,dense,par)
 import shutil
 for name in ['dqn_vs_as_vs_glft','as_miss_expected_pnl_curves','as_miss_pnl_decomposition']:
  for ext in ['png','pdf']:shutil.copy2(dest/f'{name}.{ext}',OUT/f'{name}.{ext}')
 atomic(OUT/'reports/transient_metrics.json',t.metrics(rows,raw))

def result_summary():
 rows=[]
 for e in json.loads((ARCHIVE/'protocol.json').read_text())['experiments']:
  name=e['spec']['name']
  with np.load(ARCHIVE/'data'/f'{name}.npz') as d:
   y=d['terminal_pnl'];n=len(y);queue=int(d['preserved_recovery_quotes'].sum())
  rows.append(dict(experiment=name,episodes=n,events=e['spec'].get('events',5000),mean=float(y.mean()),sd=float(y.std(ddof=1)),se=float(y.std(ddof=1)/np.sqrt(n)),preserved_recovery_quotes=queue))
 t.table(OUT/'reports/summary.csv',rows)
 return rows

EXPECTED=['censored_waiting_different_throttles','pnl_cumulated_pnl_glft_vs_naive','glft_naive_pnl_lifetime_hist','dqn_vs_as_vs_glft','glft_degradation_vs_tau_regime','as_miss_pi_q_lines','as_miss_pi_heatmap','as_miss_qbar_vs_p','as_miss_boundary_mass_vs_p','as_miss_expected_pnl_curves','as_miss_pnl_decomposition']
def render(final=False):
 configure();calibration()
 config=json.loads((ARCHIVE/'protocol.json').read_text())
 present=lambda names:all((ARCHIVE/'data'/f'{n}.npz').exists() for n in names)
 if present(['eta']):
  par=parameters();dense=analytical(par)
  names=[e['spec']['name'] for e in config['experiments'] if e['spec']['name'].startswith('fig3')]
  if present(names):empirical_comparison(par,dense)
 if present(['fig2_gamma_1e-06','fig2_gamma_0.001','fig2_gamma_0.1','fig2_at_best']):configure();g.fig2();gc.collect()
 if present(['fig4_new_baseline']+[f'fig4_new_tau_{tau}' for tau in [15,30,60,120,240]]):configure();g.fig4()
 rows=result_summary();images={n:sha(OUT/f'{n}.png') for n in EXPECTED if (OUT/f'{n}.png').exists()}
 atomic(OUT/'reports/figure_manifest.json',dict(expected=EXPECTED,completed=images,complete=len(images)==len(EXPECTED)))
 print(json.dumps(dict(figures_ready=len(images),figures_total=len(EXPECTED),final=final)),flush=True)


def validate():
 provenance=json.loads((ARCHIVE/'provenance.json').read_text())
 for rel,digest in provenance['archive_hashes'].items():
  assert sha(ARCHIVE/rel)==digest,rel
 protocol=json.loads((ARCHIVE/'protocol.json').read_text())
 complete=json.loads((ARCHIVE/'completion.json').read_text())
 assert complete['status']=='complete' and complete['episodes']==protocol['total_episodes']==4810
 assert complete['evaluation_events']==protocol['total_evaluation_events']==81050000
 summary={r['experiment']:r for r in csv.DictReader((ARCHIVE/'reports/summary.csv').open())}
 for item in protocol['experiments']:
  name=item['spec']['name']
  with np.load(ARCHIVE/'data'/f'{name}.npz',allow_pickle=False) as data:
   np.testing.assert_array_equal(data['seeds'],item['seeds'])
   assert all(np.isfinite(data[k]).all() for k in data.files)
   y=data['terminal_pnl'];assert len(y)==len(item['seeds'])
   np.testing.assert_allclose([y.mean(),y.std(ddof=1)],[float(summary[name]['mean']),float(summary[name]['sd'])],rtol=1e-12,atol=1e-12)
 params=json.loads((ARCHIVE/'reports/analytical_parameters.json').read_text())
 for key,value in protocol['parameters'].items():assert params[key]==value
 frontier=json.loads((ROOT/'data/paper/frontier_10000/calibration_used.json').read_text())
 # The exact selected calibration is recorded independently in both campaigns.
 assert all(params[key]==frontier['parameters'][key] for key in ['A','kappa','sigma'])
 rates=np.load(ARCHIVE/'data/eta.npz',allow_pickle=False)['fill_rate']
 np.testing.assert_allclose(rates.mean()/params['eta_reference_rate'],params['r_fill_factor'],rtol=1e-12)
 return provenance


def main():
 global OUT
 import argparse
 parser=argparse.ArgumentParser(description='Render the selected GLFT calibration and benchmark panels from the completed 4810-episode archive.')
 parser.add_argument('--output',type=Path,default=OUT)
 parser.add_argument('--strict-images',action='store_true')
 args=parser.parse_args();OUT=args.output if args.output.is_absolute() else ROOT/args.output
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'reports').mkdir(exist_ok=True)
 provenance=validate();render()
 with np.load(OUT/'data/analytical_curves.npz') as actual,np.load(ARCHIVE/'data/analytical_curves.npz') as expected:
  for key in expected.files:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-12)
 for name in ['analytical_curves_dense.csv','comparison_all_probabilities.csv']:
  actual=list(csv.DictReader((OUT/'reports'/name).open()));expected=list(csv.DictReader((ARCHIVE/'reports'/name).open()))
  assert len(actual)==len(expected)
  for a,b in zip(actual,expected):
   for key in b:
    if key.startswith('dqn_empirical_') and name=='comparison_all_probabilities.csv':continue
    np.testing.assert_allclose(float(a[key]),float(b[key]),rtol=1e-11,atol=1e-11)
 references=dict(provenance['selected_outputs'])
 selected=json.loads((ROOT/'data/paper/figure_inventory.json').read_text())
 for item in selected:
  if Path(item['path']).name in references:references[Path(item['path']).name]=item['sha256']
 checks={name:sha(OUT/name)==digest for name,digest in references.items()}
 atomic(OUT/'validation.json',dict(status='passed',numerical_reference_verified=True,selected_images=checks,all_selected_images_identical=all(checks.values())))
 if args.strict_images:assert all(checks.values()),checks
 print(json.dumps(dict(status='passed',panels=len(checks),output=str(OUT),all_selected_images_identical=all(checks.values()))))

if __name__=='__main__':main()
