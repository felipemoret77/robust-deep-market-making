"""Analyze the fixed-size 5,000-event campaign; retain every evaluated point."""
import argparse,csv,gzip,hashlib,io,json,math,os
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/frontier_corrected_matplotlib')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def boot(x,n,seed):
    rng=np.random.default_rng(seed);means=[];sds=[]
    for start in range(0,n,25):
        indices=rng.integers(0,len(x),size=(min(25,n-start),len(x)))
        y=x[indices]
        means.append(y.mean(axis=1));sds.append(y.std(axis=1,ddof=1))
    return np.concatenate(means),np.concatenate(sds)

def draw(ax,x,bm,bs,policies,title=None):
    mu,sd=x.mean(axis=0),x.std(axis=0,ddof=1)
    colors={'RLMM':'#E69F00','GLFT':'#0072B2','at_best':'#999999'}
    labels={'RLMM':'RLMM (Algorithm A)','GLFT':'GLFT','at_best':'At-best baseline'}
    for family in ('GLFT','RLMM','at_best'):
        ids=sorted([i for i,p in enumerate(policies) if p['family']==family],key=lambda i:sd[i])
        ax.errorbar(sd[ids],mu[ids],xerr=bs[:,ids].std(axis=0,ddof=1),yerr=bm[:,ids].std(axis=0,ddof=1),
            color=colors[family],marker='D' if family=='at_best' else 'o',
            linestyle='none' if family=='at_best' else '-',capsize=3,linewidth=1.5,
            markersize=8 if family=='at_best' else 6,label=labels[family],zorder=3)
        if family=='at_best':continue
        groups={}
        for i in ids:groups.setdefault((round(float(sd[i]),12),round(float(mu[i]),12)),[]).append(i)
        for (xx,yy),members in groups.items():
            if family=='RLMM':text=r'$\varphi='+f"{policies[members[0]]['phi']:.3f}"+'$'
            else:
                powers=[r'10^{'+str(round(math.log10(policies[i]['gamma'])))+'}' for i in members]
                text=r'$\gamma='+powers[0]+'$' if len(members)==1 else r'$\gamma\in\{'+','.join(powers)+r'\}$'
            offset=(8,-16) if family=='GLFT' and policies[members[0]]['gamma']==.01 else (8,7)
            ax.annotate(text,(xx,yy),xytext=offset,textcoords='offset points',fontsize=9,color=colors[family])
    ax.set_xlabel(r'Risk: $\sqrt{\mathrm{Var}(\mathrm{PnL}_T)}$',fontsize=12)
    ax.set_ylabel(r'Return: $\mathbb{E}[\mathrm{PnL}_T]$',fontsize=12)
    if title:ax.set_title(title,fontsize=12)
    ax.grid(alpha=.3);ax.margins(x=.24,y=.12)
    low,high=ax.get_ylim();ax.set_ylim(low,max(high,float(mu.max()+.30*np.ptp(mu))))
    ax.legend(loc='upper left',fontsize=11)

def figures(x,subset,old,bx,bs,bo,policies,out):
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,7))
    draw(ax,x,*bx,policies)
    fig.text(.5,.033,f'{len(x):,} paired episodes; 5,000 LOB events + 1,000 warmup; bars: ±1 bootstrap SE',ha='center',fontsize=10)
    fig.text(.5,.010,'Stable C51 expectation + FIFO preservation; lines connect all evaluated points.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','pdf'):fig.savefig(out/f'frontier_corrected_{len(x)}x5000.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(20,6.5),sharex=True,sharey=True)
    for ax,values,b,caption in zip(axes,(old,subset,x),(bo,bs,bx),
        (f'Paper: original {len(old):,} seeds',f'Corrected: same {len(subset):,} seeds',f'Corrected: all {len(x):,} seeds')):
        draw(ax,values,*b,policies,title=caption)
    fig.text(.5,.022,'Same 5,000-event horizon, warmup and checkpoints. Bars: ±1 bootstrap SE; all evaluated points retained.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ('png','pdf'):fig.savefig(out/f'comparison_paper_paired_and_full.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def analyze(root,resamples=None):
    manifest=json.loads((root/'manifest.json').read_text());protocol=manifest['protocol']
    complete=json.loads((root/'evaluation_complete.json').read_text())
    assert sha(root/'per_run.csv')==complete['per_run_sha256']
    with (root/'per_run.csv').open() as f:rows=list(csv.DictReader(f))
    policies=protocol['policies'];seeds=protocol['seeds']
    expected={(p['name'],s) for p in policies for s in seeds}
    lookup={(r['policy'],int(r['seed'])):r for r in rows}
    assert len(rows)==len(lookup)==len(expected) and set(lookup)==expected
    assert all(int(r['n_events'])==5000 for r in rows)
    assert all(r['variant']==('both' if next(p for p in policies if p['name']==r['policy'])['family']=='RLMM' else 'queue_only') for r in rows)
    x=np.array([[float(lookup[p['name'],s]['terminal_pnl']) for p in policies] for s in seeds])
    assert np.isfinite(x).all()
    source=root/'source_snapshot/data/paper/frontier'
    provenance=json.loads((source/'manifest.json').read_text());meta=provenance['outcomes']
    assert sha(source/meta['file'])==meta['sha256']
    raw=gzip.decompress((source/meta['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==meta['uncompressed_sha256']
    old_rows=list(csv.DictReader(io.StringIO(raw.decode())))
    old_lookup={(r['policy'],int(r['seed'])):float(r['terminal_pnl']) for r in old_rows}
    common=sorted(set(seeds).intersection({s for _,s in old_lookup}))
    old=np.array([[old_lookup[p['name'],s] for p in policies] for s in common])
    subset=np.array([[float(lookup[p['name'],s]['terminal_pnl']) for p in policies] for s in common])
    n=resamples or protocol['bootstrap_resamples'];seed=protocol['bootstrap_seed']
    bx=boot(x,n,seed);bs=boot(subset,n,seed);bo=boot(old,n,seed)
    out=root/'results';out.mkdir(exist_ok=True)
    records=[];changes=[]
    for label,values,b in [(f'corrected_{len(x)}',x,bx),('corrected_paired_1000',subset,bs),('paper_original_1000',old,bo)]:
        for i,p in enumerate(policies):
            mu,sd=values[:,i].mean(),values[:,i].std(ddof=1)
            records.append(dict(sample=label,policy=p['name'],episodes=len(values),mean_pnl=mu,sd_pnl=sd,
                mean_se=b[0][:,i].std(ddof=1),sd_se=b[1][:,i].std(ddof=1)))
    for i,p in enumerate(policies):
        for metric,k,observed in [('mean_pnl',0,subset[:,i].mean()-old[:,i].mean()),('sd_pnl',1,subset[:,i].std(ddof=1)-old[:,i].std(ddof=1))]:
            ci=np.quantile(bs[k][:,i]-bo[k][:,i],[.025,.975])
            changes.append(dict(policy=p['name'],metric=metric,paired_episodes=len(common),difference=observed,ci95_low=ci[0],ci95_high=ci[1]))
    for name,record in [('summary.csv',records),('paired_changes_vs_paper.csv',changes)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(record[0]));w.writeheader();w.writerows(record)
    counts=[]
    for p in policies:
        rr=[lookup[p['name'],s] for s in seeds]
        counts.append(dict(policy=p['name'],preserved_quotes=sum(int(r['preserved_recovery_quotes']) for r in rr),
            affected_episodes=sum(int(r['preserved_recovery_quotes'])>0 for r in rr),
            mean_cap_fraction=float(np.mean([float(r['inventory_limit_fraction']) for r in rr]))))
    write_json(out/'fifo_interventions.json',counts)
    figures(x,subset,old,bx,bs,bo,policies,out)
    lines=[f'FRONTEIRA CORRIGIDA: {len(x)} EPISÓDIOS DE 5.000 EVENTOS POR PONTO',
        '11 políticas, fluxo estacionário p_buy=0.5, 1.000 eventos de aquecimento.',
        'Pesos congelados. Mesmos checkpoints e horizonte; GLFT usa a calibração sem pesos de 0,5 a 5 ticks registrada em calibration_used.json.',
        'A: expectativa C51 em dupla precisão + preservação FIFO da ordem de recuperação com preço inalterado.',
        'GLFT e at-best: mesmo predicado de preservação FIFO aplicado.',
        f'Seeds {seeds[0]}–{seeds[-1]}; as primeiras 1.000 coincidem com a avaliação do paper e permitem comparação pareada.',
        'Linhas ligam todos os pontos por risco crescente; nenhum ponto excluído e nenhuma suavização.',
        'A comparação pareada de A avalia o pacote C51/FIFO. A comparação GLFT também inclui a nova calibração.',
        'O confronto da amostra completa corrigida versus 1.000 originais também muda a amostra.',
        'ICs bootstrap de 95% individuais, sem ajuste por comparações múltiplas; condicionais aos pesos congelados.',
        'Paper, slides e checkpoints não foram modificados.', '', 'Média e SD:']
    lines.extend(f"{r['sample']} | {r['policy']} | n={r['episodes']} | mean={r['mean_pnl']:.8f} | sd={r['sd_pnl']:.8f}" for r in records)
    lines+=['','Diferenças corrigido menos original nas mesmas seeds:']
    lines.extend(f"{r['policy']} | {r['metric']} | delta={r['difference']:+.8f} | IC95=[{r['ci95_low']:+.8f}, {r['ci95_high']:+.8f}]" for r in changes)
    (root/'RESULTADOS.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'analysis_manifest.json',dict(protocol_id=manifest['protocol_id'],rows=len(rows),
        paired_reference_episodes=len(common),bootstrap_resamples=n,bootstrap_seed=seed,
        source_sha256=sha(root/'per_run.csv'),script_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))
    print('Analysis and connected-point paper-style figures completed.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();analyze(args.root)
