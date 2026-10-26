from pathlib import Path
import csv, hashlib, json, shutil, sys
DESK=Path('/Users/felipemoret/Desktop')
REPO=DESK/'robust-deep-market-making'
PILOT=DESK/'teste1_c51_fifo_regimes_20261008'
DEST=DESK/'glft_regeneration'
R=DEST/'full_evaluation_20261008'
STAGE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not R.exists(),R
R.mkdir()
for d in ['data','traces','logs','runtime','validation','reports','figures','reference','plot_workspace/repro','plot_workspace/data/paper/historical_results']:(R/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(PILOT/'source_snapshot',R/'source_snapshot')
shutil.copy2(PILOT/'runtime.py',R/'legacy_runtime.py')
shutil.copy2(PILOT/'candidate.py',R/'candidate.py')
for name in ['evaluate.py','run.py','render.py','preflight.py','setup.py']:shutil.copy2(STAGE/name,R/name)
old=json.loads((PILOT/'protocol.json').read_text())
cases=[]
for c in old['cases']:
 c=dict(c);c['seeds']=list(range(c['seeds'][0],c['seeds'][0]+100));c['trace']=c['name'].startswith('recovery_');cases.append(c)
for family,start in [('random',500123),('correlated',700123)]:
 cases.append(dict(name=f'{family}_fixed_B',family='fixed_stress',stress_family=family,model='B',tau=60,seeds=list(range(start,start+50)),events=500000,warmup=10000,trace=False))
g=json.loads((DEST/'protocol.json').read_text())
for spec in g['experiments']:
 if spec['spec']['name'].startswith('fig3_p_'):
  p=spec['spec']['prob'];cases.append(dict(name=f'asymmetry_A_p{p:g}',family='asymmetry',model='A_001',tau=None,probability=p,seeds=spec['seeds'],events=5000,warmup=1000,trace=False))
cases.append(dict(name='illustration_A_stationary_124',family='stationary',model='A_001',tau=0,seeds=[124],events=5000,warmup=1000,trace=True))
for model,scenario in [('A_001','stationary'),('A_001','regime'),('B','regime')]:
 cases.append(dict(name=f'illustration_{model}_{scenario}_42',family='stationary' if scenario=='stationary' else 'illustration_regime',model=model,tau=60,seeds=[42],events=100000,warmup=10000,trace=True,full_frame=scenario=='regime' and model=='A_001',schedule_seed=1000042))
cases.append(dict(name='illustration_signals_B_123',family='recovery',model='B',tau=60,seeds=[123],events=5000,warmup=1000,trace=True,full_frame=True))
cfg=dict(name='full_paper_evaluation_c51_fifo',phase='full',models=old['models'],cases=cases,workers=2,variant='both',learning=False,epsilon=0,device='cpu',numerical_threads=1,
 total_episodes=sum(len(c['seeds']) for c in cases),total_events=sum(len(c['seeds'])*c['events'] for c in cases),
 corrections=['float64 softmax and centered expected return for action ranking','preserve unchanged valid inventory-reducing quote at inventory caps'],
 pairing='Same published seeds and scenario construction across policies; realized paths may depend on actions.',
 stopping='Fixed sample size from paper. No outcome-based selection or extension.',
 reference_comparison='Historical same-seed outcomes are reported separately from the new corrected evaluations. Historical outcomes are not fresh original-code runs.',
 paper_and_slides_modified=False)
assert cfg['total_episodes']==4405
dump(R/'protocol.json',cfg)
protocol_id=sha(R/'protocol.json')
for rel in ['config/paper_experiments.json','data/paper/historical_results/provenance.json','data/paper/historical_results/degradation_summary.csv','data/paper/historical_results/A_sweeps/per_episode.csv','data/paper/historical_results/random.npz','data/paper/historical_results/correlated.npz','data/paper/regime_mc.npz','data/paper/bandit_random.npz','data/paper/bandit_correlated.npz','data/paper/tau_sweep_B_runs.csv','data/paper/episode/provenance.json','data/paper/episode/single_episode.npz','data/paper/episode/signal_example.npz']:
 p=R/'reference'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,p)
for rel in ['reports/comparison_all_probabilities.csv','reports/analytical_curves_dense.csv','reports/analytical_parameters.json','reports/figure_manifest.json']:
 p=R/'reference/glft'/Path(rel).name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DEST/rel,p)
for name in ['plot_historical_results.py','plot_paper_episode.py','plot_paper_sweep.py','regen_regime_mc_figs.py','regen_scenario_bandit_shaded.py','glft_transient.py']:
 shutil.copy2(REPO/'repro'/name,R/'plot_workspace/repro'/name)
for rel in ['historical_results/episode_seed124.npz','tau_sweep_summary.csv']:
 dest=R/'reference/data/paper'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/'data/paper'/rel,dest)
p=R/'plot_workspace/repro/regen_scenario_bandit_shaded.py';s=p.read_text().replace('plt.close(fig)','fig.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")\n    plt.close(fig)');p.write_text(s)
# Recovery plot's x coordinates use the actual recorded event indices.
p=R/'plot_workspace/repro/regen_regime_mc_figs.py';s=p.read_text().replace('t = np.linspace(0, N_STEPS - 1, T)','t = trace_step[:T]')
s=s.replace('fig.savefig(REPO / name, dpi=200, bbox_inches="tight")','fig.savefig(REPO / name, dpi=200, bbox_inches="tight")\n    fig.savefig((REPO / name).with_suffix(".pdf"), dpi=200, bbox_inches="tight")')
s=s.replace('print("  saved pnl_distribution_phaseA_phaseB.png -> output")','fig = None\nprint("  saved pnl_distribution_phaseA_phaseB.png -> output")') if False else s
s=s.replace('print("  saved pnl_distribution_phaseA_phaseB.png -> output"); plt.close(fig)','fig.savefig(REPO / "pnl_distribution_phaseA_phaseB.pdf", dpi=150, bbox_inches="tight")\nprint("  saved pnl_distribution_phaseA_phaseB.png -> output"); plt.close(fig)')
p.write_text(s)
p=R/'plot_workspace/repro/plot_paper_sweep.py';s=p.read_text().replace('plt.close(fig)','plt.savefig(args.output / "regime_tau_sweep_decay_phasea_vs_phaseb.pdf", dpi=150)\nplt.close(fig)');p.write_text(s)
reused=[]
for c in cases:
 if c['trace']:continue
 for seed in c['seeds']:
  src=PILOT/'data'/c['name']/'both'/f'{seed}.json'
  if not src.exists():continue
  row=json.loads(src.read_text())
  assert row['networks_unchanged'] and row['events']==c['events'] and row['warmup']==c['warmup'] and row['checkpoint_sha256']==cfg['models'][c['model']]['sha256']
  row.update(protocol_id=protocol_id,source='validated_corrected_pilot',source_record=str(src),source_record_sha256=sha(src))
  dump(R/'data'/c['name']/f'{seed}.json',row);reused.append(dict(case=c['name'],seed=seed,sha256=sha(src)))
assert len(reused)==560
dump(R/'validation/reuse.json',dict(count=len(reused),records=reused,pilot_protocol_sha256=sha(PILOT/'protocol.json'),candidate_sha256=sha(R/'candidate.py')))
for c in cases:
 if not c['name'].startswith('recovery_'):continue
 for seed in c['seeds'][:20]:
  src=PILOT/'data'/c['name']/'both'/f'{seed}.json'
  dest=R/'reference/pilot_recovery'/c['name']/f'{seed}.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
protected={str(p):sha(p) for base,patterns in [(REPO/'paper',['*.tex','*.pdf','figures/*']), (DESK/'algodefi_prese',['*.tex','*.pdf','figs_algodefi/*']), (REPO/'checkpoints',['*.pt'])] for pattern in patterns for p in base.glob(pattern) if p.is_file()}
dump(R/'validation/protected_files.json',protected)
manifest={str(p.relative_to(R)):sha(p) for folder in ['source_snapshot','reference','plot_workspace/repro'] for p in (R/folder).rglob('*') if p.is_file()}
manifest.update({n:sha(R/n) for n in ['evaluate.py','run.py','render.py','preflight.py','legacy_runtime.py','candidate.py','protocol.json']})
dump(R/'source_manifest.json',manifest)
text='AVALIAÇÕES COMPLETAS — C51 ESTÁVEL E PRESERVAÇÃO FIFO\n\n4.405 avaliações finais; 560 reaproveitadas do piloto validado; 3.845 novas execuções.\n'
text+='Figuras finais serão instaladas na raiz de glft_regeneration, com os nomes do paper, somente após validação da amostra completa.\nPaper, slides e pesos ficam preservados.\n\n100 x 100.000 eventos: degradação, recuperação A/B e sweeps.\n100 x 500.000: B/C sob persistência aleatória e direção correlacionada.\n50 x 500.000 por família: B com duração fixa para comparação ao stress (B stress usa as primeiras 50 seeds pré-definidas da avaliação de 100).\n100 x 5.000 por probabilidade: assimetria constante de A, mantendo os benchmarks GLFT já corrigidos.\nIlustrações: mesmas seeds, horizontes e aquecimentos do paper.\n\nA avaliação corrigida usa pesos congelados e as duas intervenções do piloto. O histórico é uma referência separada; não se apresenta a comparação histórica como nova execução original.\n'
(R/'LEIA.txt').write_text(text)
(DEST/'AVALIACOES_COMPLETAS.txt').write_text(text+'\nProgresso: status_avaliacoes_completas.json\nDados, códigos e logs: full_evaluation_20261008/\n')
print(json.dumps(dict(folder=str(R),episodes=cfg['total_episodes'],events=cfg['total_events'],reused=len(reused),new=cfg['total_episodes']-len(reused))))
