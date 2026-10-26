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
sys.path.insert(0,str(ROOT/'snapshot/repro'))
import glft_figures as g
import glft_transient as t
from experiment_runtime import atomic,sha

def configure():
 plt.rcParams.update({'axes.labelsize':15,'axes.titlesize':14,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':11,'legend.title_fontsize':11,'legend.framealpha':.9,'legend.fancybox':True,'legend.borderpad':.3,'legend.handlelength':1.6})
 plt.show=lambda *a,**k:None
 g.OUT=ROOT;g.DATA=ROOT/'data';g.plots._save_paper_figure=g.save

def calibration():
 info=json.loads((ROOT/'inputs/calibration_plot_manifest.json').read_text());full=json.loads((ROOT/'inputs/calibration_summary.json').read_text());fit=full['fit_05a5']
 points=[x for x in full['empirical_rows'] if .5<=x['distance']<=5]
 x=np.array([z['distance'] for z in points]);y=np.array([z['intensity'] for z in points])
 b,k=np.linalg.lstsq(np.c_[np.ones(len(x)),-x],np.log(y),rcond=None)[0]
 np.testing.assert_allclose([np.exp(b),k],[fit['A_fit'],fit['kappa']],rtol=1e-13)
 params=json.loads((ROOT/'inputs/calibration_used.json').read_text())['parameters'];np.testing.assert_allclose([fit['A'],fit['kappa']],[params['A'],params['kappa']],rtol=1e-13)
 plt.rcParams.update({'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'legend.framealpha':.9})
 fig,(top,bottom)=plt.subplots(2,1,figsize=(7,5.25),sharex=True)
 xx=np.linspace(.5,5,700);yy=fit['A_fit']*np.exp(-fit['kappa']*xx)
 top.plot(x,y,'o-',color='#1f77b4',linewidth=1.5,markersize=5,label='Pooled empirical intensity')
 top.plot(xx,yy,'--',color='#ff7f0e',linewidth=1.5,label=rf'Exponential fit: $A_{{\mathrm{{fit}}}}={fit["A_fit"]:.4f},\ \kappa={fit["kappa"]:.4f}$')
 top.set_title(r'Censored waiting times, $\Delta T=0.5$ s');top.set_ylabel(r'$\widehat\Lambda(\delta)$ (s$^{-1}$)');top.set_ylim(bottom=0);top.legend(loc='upper right')
 bottom.plot(x,np.log(y),'o-',color='#1f77b4',linewidth=1.5,markersize=5);bottom.plot(xx,np.log(yy),'--',color='#ff7f0e',linewidth=1.5)
 bottom.set_ylabel(r'$\log\widehat\Lambda(\delta)$');bottom.set_xlabel(r'Effective quote distance $\delta$ (ticks)');bottom.set_xticks([.5,1,2,3,4,5])
 for ax in (top,bottom):ax.grid(alpha=.3)
 fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.92,hspace=.20)
 for ext in ['png','pdf']:fig.savefig(ROOT/f'GLFT_censored_waiting_times_calib_half_second.{ext}',dpi=220)
 plt.close(fig);configure()
 fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
 for color,row in zip(plt.cm.viridis(np.linspace(.05,.95,5)),info['window_fits']):
  x=np.array(row['distances']);w=row['window']
  ax.errorbar(x,row['normalized_log_intensity'],yerr=row['sem'],fmt='o',markersize=4,capsize=4,color=color,label=rf'$\Delta T={w:g}$ s')
  ax.plot(x,-row['kappa']*x,'--',color=color,linewidth=2)
 ax.set_xlabel(r'$\delta$ [tick]');ax.set_ylabel(r'$\log(\hat\Lambda(\delta)/\hat A_{\mathrm{fit}})$');ax.grid(alpha=.3);ax.legend(loc='best')
 g.save(fig,'censored_waiting_different_throttles.png');plt.close(fig)
 atomic(ROOT/'reports/calibration_validation.json',dict(status='passed',fit=fit,simulation_reused=True))

def parameters():
 old=json.loads((ROOT/'inputs/original_experiments.json').read_text())['analytical_parameters']
 par=dict(old);par.update(json.loads((ROOT/'inputs/calibration_used.json').read_text())['parameters'])
 with np.load(ROOT/'data/eta.npz') as d:rates=d['fill_rate'];assert len(rates)==10
 model=t.make_model(.5,dict(par,r_fill_factor=1.))
 reference=2*par['A']*np.exp(-par['kappa']*model['half_spread'])
 par.update(r_fill_factor=float(rates.mean()/reference),eta_empirical_rate=float(rates.mean()),eta_reference_rate=float(reference),delta_bar=model['half_spread'])
 atomic(ROOT/'reports/analytical_parameters.json',par)
 atomic(ROOT/'reports/eta_comparison.json',dict(old=old,new=par,empirical_fill_rates=rates.tolist(),note='T and v0 are properties measured on the bare LOB and are retained. Eta is reestimated with the new GLFT policy.'))
 return par

def analytical(par):
 configure();plt.rcParams['axes.prop_cycle']=cycler(color=[tuple(c) for c in plt.cm.viridis(np.linspace(.05,.95,6))])
 probs=np.linspace(.2,.8,121)
 fields=('A','kappa','sigma','gamma','T','inv_limit','r_fill_factor','tick_size','v0')
 curves=g.plots.compute_bundle_curves(probs,**{k:par[k] for k in fields})
 for p,pi in zip(probs,curves['pi']):np.testing.assert_allclose(pi,t.make_model(float(p),par)['pi'],atol=1e-12,rtol=0)
 np.savez_compressed(ROOT/'data/analytical_curves.npz',probs=probs,**curves)
 g.plots.save_plot_pi_lines([.3,.4,.5,.6,.7],curves);g.plots.save_plot_pi_heatmap(probs,curves);g.plots.save_plot_qbar(probs,curves);g.plots.save_plot_boundary_mass(probs,curves)
 dense=[t.calculate(float(p),par)[0] for p in probs]
 t.table(ROOT/'reports/analytical_curves_dense.csv',dense)
 # PnL panels preserve the already-corrected finite-horizon convention.
 for name in ['as_miss_expected_pnl_curves','as_miss_pnl_decomposition']:
  fig,ax=plt.subplots(figsize=(8.8,5.2))
  arr=lambda k:np.array([r[k] for r in dense])
  if name=='as_miss_expected_pnl_curves':
   ax.plot(probs,arr('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread-only');ax.plot(probs,arr('transient_full_pnl'),color='crimson',lw=2.5,label='Full (with drift)')
  else:
   ax.plot(probs,arr('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread capture');ax.plot(probs,-arr('transient_drift_pnl'),color='crimson',lw=2.5,label='MtM loss');ax.plot(probs,arr('transient_full_pnl'),color='black',lw=2.5,label='Net full PnL')
  t.style_axes(ax);ax.set_title(r'Finite horizon, $q_0=0$',fontsize=13);ax.legend();fig.tight_layout();g.save(fig,name+'.png');plt.close(fig)
 t.ROOT=ROOT/'reports';atomic(ROOT/'reports/analytical_validation.json',t.validate(par))
 return dense

def empirical_comparison(par,dense):
 sources=list(csv.DictReader((ROOT/'inputs/expected_pnl_vs_buy_MO_updated.csv').open()))
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
 t.table(ROOT/'reports/comparison_all_probabilities.csv',rows)
 dest=ROOT/'reports/transient_diagnostics';dest.mkdir(exist_ok=True);t.ROOT=dest
 t.figures(rows,dense,par)
 import shutil
 for name in ['dqn_vs_as_vs_glft','as_miss_expected_pnl_curves','as_miss_pnl_decomposition']:
  for ext in ['png','pdf']:shutil.copy2(dest/f'{name}.{ext}',ROOT/f'{name}.{ext}')
 atomic(ROOT/'reports/transient_metrics.json',t.metrics(rows,raw))

def comparison_summary():
 previous={r['experiment']:r for r in json.loads((ROOT/'inputs/old_summary.json').read_text())};rows=[]
 for e in json.loads((ROOT/'protocol.json').read_text())['experiments']:
  name=e['spec']['name'];path=ROOT/'data'/f'{name}.npz'
  if not path.exists():continue
  with np.load(path) as d:y=d['terminal_pnl'];n=len(y);queue=int(d['preserved_recovery_quotes'].sum())
  old=previous[name];rows.append(dict(experiment=name,episodes=n,events=e['spec'].get('events',5000),old_mean=old['mean'],new_mean=float(y.mean()),mean_change=float(y.mean()-old['mean']),old_sd=old['sd'],new_sd=float(y.std(ddof=1)),new_se=float(y.std(ddof=1)/np.sqrt(n)),preserved_recovery_quotes=queue))
 if rows:t.table(ROOT/'reports/old_vs_new.csv',rows)
 return rows

EXPECTED=['censored_waiting_different_throttles','GLFT_censored_waiting_times_calib_half_second','pnl_cumulated_pnl_glft_vs_naive','glft_naive_pnl_lifetime_hist','dqn_vs_as_vs_glft','glft_degradation_vs_tau_regime','as_miss_pi_q_lines','as_miss_pi_heatmap','as_miss_qbar_vs_p','as_miss_boundary_mass_vs_p','as_miss_expected_pnl_curves','as_miss_pnl_decomposition','glft_frontier_complete']
def render(final=False):
 configure();calibration()
 config=json.loads((ROOT/'protocol.json').read_text())
 present=lambda names:all((ROOT/'data'/f'{n}.npz').exists() for n in names)
 if present(['eta']):
  par=parameters();dense=analytical(par)
  names=[e['spec']['name'] for e in config['experiments'] if e['spec']['name'].startswith('fig3')]
  if present(names):empirical_comparison(par,dense)
 if present(['fig2_gamma_1e-06','fig2_gamma_0.001','fig2_gamma_0.1','fig2_at_best']):configure();g.fig2();gc.collect()
 if present(['fig4_new_baseline']+[f'fig4_new_tau_{tau}' for tau in [15,30,60,120,240]]):configure();g.fig4()
 rows=comparison_summary();images={n:sha(ROOT/f'{n}.png') for n in EXPECTED if (ROOT/f'{n}.png').exists()}
 atomic(ROOT/'reports/figure_manifest.json',dict(expected=EXPECTED,completed=images,complete=len(images)==len(EXPECTED)))
 if final:
  assert len(images)==len(EXPECTED)
  assert all((ROOT/f'{n}.pdf').exists() for n in EXPECTED)
  assert len(rows)==len(config['experiments'])
  import pymupdf as fitz
  pdf=fitz.open()
  for name in EXPECTED:
   with fitz.open(ROOT/f'{name}.pdf') as panel:pdf.insert_pdf(panel)
  pdf.save(ROOT/'TODAS_AS_FIGURAS.pdf');pdf.close()
  content=['Regeneração concluída.','',f"GLFT: A={config['parameters']['A']:.9f}; kappa={config['parameters']['kappa']:.9f}; sigma=0.30.",f"{config['total_episodes']} episódios GLFT; 1000 episódios naïve arquivados reaproveitados.",'Figuras PNG e PDF na raiz, com os nomes usados no paper.','TODAS_AS_FIGURAS.pdf reúne os 13 gráficos.','Comparação das médias e dispersões: reports/old_vs_new.csv.','Parâmetros analíticos e novo eta: reports/analytical_parameters.json.','Paper, slides e checkpoints preservados.']
  (ROOT/'RESULTADOS.txt').write_text('\n'.join(content)+'\n')
 print(json.dumps(dict(figures_ready=len(images),figures_total=len(EXPECTED),final=final)),flush=True)
if __name__=='__main__':render('--final' in sys.argv)
