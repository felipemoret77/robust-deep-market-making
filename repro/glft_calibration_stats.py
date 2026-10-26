"""Frozen pooled exposure fit and paired seed bootstrap used by the campaign."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import gzip
import numpy as np
import glft_calibration_core as cc

def pooled_curve(runs, method):
    bins = defaultdict(lambda: dict(n_exec=0, exposure=0.0, n_probes=0))
    for r in runs:
        if method == 'original':
            if r['provenance']['variant'] != 'original': continue
            rows = [dict(distance=r['provenance']['distance'], **r['original_estimator_on_this_tape'])]
        else:
            if r['provenance']['variant'] != 'corrected': continue
            rows = r['actual_distance_bins' if method == 'corrected_actual' else 'requested_distance_bins']
        for row in rows:
            b = bins[float(row['distance'])]
            b['n_exec'] += row['n_exec']
            b['exposure'] += row['exposure']
            b['n_probes'] += row.get('n_probes', row.get('n_windows', 0))
    return cc.bin_rows(bins)

def fit_runs(runs, cfg):
    curves = {m: pooled_curve(runs, m) for m in ('original','corrected_requested','corrected_actual')}
    return {m:cc.fit_curve(v,cfg) for m,v in curves.items() if v},curves

def summarize(runs, cfg, seeds, bootstrap_reps=5000):
    fits,curves = fit_runs(runs,cfg)
    methods = list(fits)
    by_seed = {s:[r for r in runs if r['provenance']['config']['random_seed']==s] for s in seeds}
    per_seed = {str(s):fit_runs(rs,cfg)[0] for s,rs in by_seed.items()}
    # Cluster bootstrap: resample complete paired seed blocks, not individual probes.
    # Each seed block includes all five requested distances and both protocols.
    rng=np.random.default_rng(20261003)
    boot={m:[] for m in methods}
    # Sufficient statistics avoid repeatedly constructing millions of observations.
    ds=sorted({r['distance'] for rows in curves.values() for r in rows})
    counts={m:np.zeros((len(seeds),len(ds))) for m in methods}
    exposure={m:np.zeros((len(seeds),len(ds))) for m in methods}
    for i,s in enumerate(seeds):
        for m in methods:
            for r in pooled_curve(by_seed[s],m):
                j=ds.index(r['distance']);counts[m][i,j]=r['n_exec'];exposure[m][i,j]=r['exposure']
    for _ in range(bootstrap_reps):
        ix=rng.integers(0,len(seeds),len(seeds))
        for m in methods:
            ns=counts[m][ix].sum(axis=0); ts=exposure[m][ix].sum(axis=0)
            rows=[dict(distance=d,n_exec=int(n),exposure=float(t),lambda_hat=float(n/t))
                  for d,n,t in zip(ds,ns,ts) if t>0]
            f=cc.fit_curve(rows,cfg);boot[m].append([f['A'],f['kappa']])
    for m in methods:
        arr=np.asarray(boot[m])
        fits[m]['cluster_bootstrap_95ci']={k:np.quantile(arr[:,j],[.025,.975]).tolist() for j,k in enumerate(('A','kappa'))}
    comparisons={}
    if 'original' in methods:
        for m in methods:
            if m=='original':continue
            delta=np.asarray(boot[m])-np.asarray(boot['original'])
            relative=np.asarray(boot[m])/np.asarray(boot['original'])-1
            comparisons[m]={k:dict(difference=fits[m][k]-fits['original'][k],
                difference_95ci=np.quantile(delta[:,j],[.025,.975]).tolist(),
                relative_change_percent=100*(fits[m][k]/fits['original'][k]-1),
                relative_change_percent_95ci=(100*np.quantile(relative[:,j],[.025,.975])).tolist())
                for j,k in enumerate(('A','kappa'))}
    published=dict(A=.15070716752258323,kappa=2.3353417388646154,sigma=.3)
    significant = bool(comparisons) and any(
        lo>0 or hi<0 for lo,hi in [comparisons['corrected_actual'][k]['difference_95ci'] for k in ('A','kappa')])
    return dict(config=asdict(cfg),seeds=seeds,runs=len(runs),curves=curves,fits=fits,
        per_seed_fits=per_seed,paired_comparisons=comparisons,published_parameters=published,
        difference_confirmed=significant,selected_method='corrected_actual',
        selected_parameters={k:fits['corrected_actual'][k] for k in ('A','kappa')}|dict(sigma=.3),
        bootstrap=dict(repetitions=bootstrap_reps,unit='paired seed, including all requested distances',seed=20261003),
        notes=[
            'Coefficients fitted to pooled executions / pooled exposure, weighted by pooled exposure.',
            'Requested and effective fits use identical corrected order observations.',
            'Actual distances refer to rounded prices and the mid after previous-probe cancellation, immediately before insertion.',
            'Only distances 0.5 through 2.5 ticks enter the fit; extra bins remain available.',
            'Original and corrected policies share seeds and LOB parameters, not identical realized trajectories.',
            'Changing observation-window length also changes posting interval, as in the original figure.',
            'Zero-event bins are reported; original log-linear fitter omits their logarithms.',
            'No claim that the exponential intensity exactly describes the FIFO simulator.'])
