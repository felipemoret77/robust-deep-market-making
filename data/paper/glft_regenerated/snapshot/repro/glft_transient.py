#!/usr/bin/env python3
"""Exact finite-horizon GLFT reduced model; portable core used by glft_figures.py.

The matrix-exponential occupation calculation, independent numerical checks,
conditional seed bootstrap, and plot functions follow the validated campaign.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE = '#0072B2', '#E69F00'
plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 14,
                    'xtick.labelsize': 13, 'ytick.labelsize': 13,
                    'legend.fontsize': 11, 'legend.framealpha': .9,
                    'legend.fancybox': True, 'legend.borderpad': .3,
                    'legend.handlelength': 1.6})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n')


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_model(p, par):
    """Row-generator convention: rho'(t) = rho(t) G."""
    A, k, gamma = par['A'], par['kappa'], par['gamma']
    q = np.arange(-par['inv_limit'], par['inv_limit'] + 1, dtype=float)
    base = np.log1p(gamma / k) / gamma
    skew = np.sqrt(par['sigma']**2 * gamma / (2*k*A)) * np.exp(
        .5 * (1 + k/gamma) * np.log1p(gamma/k))
    ask, bid = base + skew/2 - skew*q, base + skew/2 + skew*q
    assert min(ask.min(), bid.min()) > 0
    down = 2*p*par['r_fill_factor']*A*np.exp(-k*ask)
    up = 2*(1-p)*par['r_fill_factor']*A*np.exp(-k*bid)
    down[0], up[-1] = 0., 0.
    G = np.diag(-down-up) + np.diag(up[:-1], 1) + np.diag(down[1:], -1)
    log_pi = -k*skew*q*q - np.log(p/(1-p))*q
    pi = np.exp(log_pi-log_pi.max()); pi /= pi.sum()
    spread_rate = down*ask + up*bid
    drift_rate = par['v0']*(2*p-1)*q
    relaxation = -1/np.sort(np.linalg.eigvals(G).real)[-2]
    np.testing.assert_allclose(G.sum(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(pi@G, 0, atol=1e-14)
    np.testing.assert_allclose(pi[:-1]*up[:-1], pi[1:]*down[1:], atol=1e-14)
    return dict(p=float(p), q=q, G=G, pi=pi, spread_rate=spread_rate,
                drift_rate=drift_rate, relaxation=float(relaxation),
                skew=float(skew), half_spread=float(base+skew/2))


def exact_occupation(model, horizon, initial=None):
    n = len(model['q'])
    initial = np.eye(n)[n//2] if initial is None else initial
    block = np.zeros((2*n, 2*n))
    block[:n, :n] = horizon*model['G']
    block[:n, n:] = np.eye(n)
    result = expm(block)
    terminal = initial@result[:n, :n]
    occupation = initial@result[:n, n:]
    np.testing.assert_allclose([terminal.sum(), occupation.sum()], 1., atol=1e-10)
    assert min(occupation.min(), terminal.min()) >= -1e-12
    return terminal, occupation


def quantities(model, occ, horizon, par):
    spread = par['tick_size']*horizon*(occ@model['spread_rate'])
    drift = par['tick_size']*horizon*(occ@model['drift_rate'])
    return dict(spread_pnl=float(spread), drift_pnl=float(drift), full_pnl=float(spread+drift),
                mean_inventory=float(occ@model['q']), boundary_mass=float(occ[0]+occ[-1]))


def calculate(p, par):
    model = make_model(p, par)
    terminal, occ = exact_occupation(model, par['T'])
    trans = quantities(model, occ, par['T'], par)
    stationary = quantities(model, model['pi'], par['T'], par)
    row = dict(p_buy_mo=float(p), horizon=par['T'], q0=0,
               relaxation_time=model['relaxation'])
    row.update({f'transient_{k}': v for k, v in trans.items()})
    row.update({f'stationary_{k}': v for k, v in stationary.items()})
    return row, model, terminal, occ


def validate(par):
    checks, convergence = [], []
    nodes, weights = leggauss(96)
    for p in [.2, .3, .4, .45, .5, .55, .6, .7, .8]:
        m = make_model(p, par)
        rho0 = np.eye(len(m['q']))[par['inv_limit']]
        terminal, avg = exact_occupation(m, par['T'])
        # Independent construction: solve forward ODE plus accumulated occupancy.
        n = len(rho0)
        def rhs(t, y):
            return np.concatenate([y[:n]@m['G'], y[:n]])
        sol = solve_ivp(rhs, (0., par['T']), np.r_[rho0, np.zeros(n)],
                        method='DOP853', rtol=1e-11, atol=1e-13)
        assert sol.success
        ode_term, ode_avg = sol.y[:n, -1], sol.y[n:, -1]/par['T']
        # Independent numerical integration of exp(tG), no augmented block.
        gauss_avg = sum(w*(rho0@expm(.5*(x+1)*par['T']*m['G']))
                        for x, w in zip(nodes, weights))/2
        err_ode = float(max(np.max(abs(avg-ode_avg)), np.max(abs(terminal-ode_term))))
        err_gauss = float(np.max(abs(avg-gauss_avg)))
        np.testing.assert_allclose(avg, ode_avg, atol=2e-10, rtol=0)
        np.testing.assert_allclose(avg, gauss_avg, atol=2e-10, rtol=0)
        _, stationary_avg = exact_occupation(m, par['T'], m['pi'])
        np.testing.assert_allclose(stationary_avg, m['pi'], atol=1e-11, rtol=0)
        mirror = make_model(1-p, par)
        _, mirrored_avg = exact_occupation(mirror, par['T'])
        np.testing.assert_allclose(avg, mirrored_avg[::-1], atol=1e-11, rtol=0)
        _, tiny = exact_occupation(m, 1e-7)
        assert np.max(abs(tiny-rho0)) < 1e-7
        tv_list = []
        for multiplier in [1, 2, 5, 10, 50, 100, 1000]:
            t = multiplier*m['relaxation']
            end, average = exact_occupation(m, t)
            tv = float(.5*np.abs(average-m['pi']).sum()); tv_list.append(tv)
            convergence.append(dict(p_buy_mo=p, relaxation_multiples=multiplier,
                horizon=t, mean_occupation_tv=tv,
                terminal_tv=float(.5*np.abs(end-m['pi']).sum()),
                average_pnl_rate=float(quantities(m, average, t, par)['full_pnl']/t),
                stationary_pnl_rate=float(quantities(m, m['pi'], t, par)['full_pnl']/t)))
        # The terminal law converges exponentially; the time average retains
        # the finite initial transient and converges as 1/T, not exponentially.
        assert .5*np.abs(end-m['pi']).sum() < 1e-8
        assert tv_list[-1] < .003 and tv_list[-1] < tv_list[0]
        assert 9.99 < tv_list[-2]/tv_list[-1] < 10.01
        checks.append(dict(p_buy_mo=p, max_error_independent_ode=err_ode,
                           max_error_gauss_integration=err_gauss,
                           long_horizon_occupation_tv=tv_list[-1]))
    table(ROOT/'validation_independent.csv', checks)
    table(ROOT/'horizon_convergence.csv', convergence)
    return dict(status='passed', checks=['generator', 'detailed balance', 'probability mass',
        'nonnegative probabilities', 'independent forward ODE', 'Gauss-Legendre integration',
        'stationary initialization', 'mirror symmetry', 'short horizon', 'long horizon'],
        max_ode_error=max(r['max_error_independent_ode'] for r in checks),
        max_gauss_error=max(r['max_error_gauss_integration'] for r in checks))


def metrics(rows, raw):
    empirical = np.array([r['glft_empirical_mean'] for r in rows])
    se = np.array([r['glft_empirical_se_mean'] for r in rows])
    result = {}
    for name in ['stationary', 'transient']:
        predicted = np.array([r[f'{name}_full_pnl'] for r in rows])
        residual = predicted-empirical
        result[name] = dict(rmse=float(np.sqrt(np.mean(residual**2))),
            mean_absolute_error=float(np.abs(residual).mean()), mean_bias=float(residual.mean()),
            largest_absolute_error=float(np.abs(residual).max()),
            rms_residual_in_empirical_se=float(np.sqrt(np.mean((residual/se)**2))),
            largest_absolute_residual_in_empirical_se=float(np.max(abs(residual/se))),
            outside_empirical_pointwise_95ci=int(np.count_nonzero(abs(residual)>1.96*se)))
    old, new = result['stationary']['rmse'], result['transient']['rmse']
    result['rmse_reduction_fraction'] = 1-new/old
    # Re-use the common evaluation seeds as bootstrap clusters across p.
    # Calibration parameters are held fixed: this does not include model/calibration uncertainty.
    stack = np.stack(raw, axis=1)
    rng = np.random.default_rng(20261003)
    indices = rng.integers(0, stack.shape[0], size=(10000, stack.shape[0]))
    means = stack[indices].mean(axis=1)
    models = {name: np.array([r[f'{name}_full_pnl'] for r in rows]) for name in result if name in ('stationary','transient')}
    rmse = {name: np.sqrt(np.mean((means-pred)**2, axis=1)) for name, pred in models.items()}
    for name in models:
        result[name]['conditional_bootstrap_rmse_95ci'] = np.quantile(rmse[name], [.025,.975]).tolist()
    result['conditional_bootstrap_rmse_improvement_95ci'] = np.quantile(
        rmse['stationary']-rmse['transient'], [.025,.975]).tolist()
    result['uncertainty_note'] = ('Residual/SE treats the analytical curves as fixed; empirical SE only. '
        'Bootstrap clusters are common evaluation seeds, resampled jointly across p. '
        'No calibration, horizon or reduced-model uncertainty is included. These are descriptive diagnostics, '
        'not a formal model-validation p-value or equivalence test.')
    return result


def style_axes(ax):
    ax.axvline(.5, color='gray', linestyle='--', alpha=.5)
    ax.axhline(0, color='black', alpha=.2)
    ax.set_xlabel('P(buy MO)')
    ax.set_ylabel(r'$\mathbb{E}[\mathrm{PnL}]$')
    ax.grid(alpha=.3)


def save(fig, name, pdf):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT/f'{name}.{ext}', dpi=200, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def figures(rows, dense, par):
    x = np.array([r['p_buy_mo'] for r in rows])
    def arr(key): return np.array([r[key] for r in rows])
    dx = np.array([r['p_buy_mo'] for r in dense])
    def curve(key): return np.array([r[key] for r in dense])
    with PdfPages(ROOT/'TESTE2_figuras.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(9,5))
        ax.errorbar(x, arr('dqn_empirical_mean'), yerr=1.96*arr('dqn_empirical_se_mean'),
            fmt='o-', color=ORANGE, capsize=4, linewidth=2, markersize=7, label='DQN (φ=1.0e-03)')
        ax.errorbar(x, arr('glft_empirical_mean'), yerr=1.96*arr('glft_empirical_se_mean'),
            fmt='s-', color=BLUE, capsize=4, linewidth=2, markersize=6, label='GLFT (γ=1.0e-06)')
        ax.plot(x, arr('transient_full_pnl'), 'o--', color=BLUE, linewidth=2, markersize=6,
            label=r'GLFT analytical ($q_0=0$)')
        style_axes(ax); ax.legend(fontsize=11, loc='best')
        save(fig, 'dqn_vs_as_vs_glft', pdf)

        fig, (ax, residual) = plt.subplots(2,1,figsize=(9,7.2),sharex=True,
                                           gridspec_kw={'height_ratios':[2.4,1]})
        ax.errorbar(x, arr('glft_empirical_mean'), yerr=1.96*arr('glft_empirical_se_mean'),
            fmt='s', color=BLUE, capsize=4, markersize=6, label='FIFO simulation (95% CI)')
        ax.plot(dx, curve('stationary_full_pnl'), '--', color='#999999', linewidth=2.2,
                label='Stationary approximation')
        ax.plot(dx, curve('transient_full_pnl'), color='#D55E00', linewidth=2.2,
                label=r'Finite horizon, $q_0=0$')
        style_axes(ax); ax.set_xlabel(''); ax.legend(loc='best')
        residual.axhspan(-1.96,1.96,color='#009E73',alpha=.09)
        residual.axhline(0,color='black',alpha=.4)
        for name, color, marker in [('stationary','#999999','o'),('transient','#D55E00','s')]:
            residual.plot(x, (arr(name+'_full_pnl')-arr('glft_empirical_mean'))/arr('glft_empirical_se_mean'),
                marker+'-',color=color,linewidth=1.6)
        residual.set_xlabel('P(buy MO)'); residual.set_ylabel('Residual / SE',fontsize=13)
        residual.grid(alpha=.3)
        save(fig,'stationary_transient_fifo_comparison',pdf)

        fig,ax=plt.subplots(figsize=(8.8,5.2))
        ax.plot(dx,curve('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread-only')
        ax.plot(dx,curve('transient_full_pnl'),color='crimson',lw=2.5,label='Full (with drift)')
        style_axes(ax);ax.set_title(r'Finite horizon, $q_0=0$',fontsize=13);ax.legend()
        save(fig,'as_miss_expected_pnl_curves',pdf)

        fig,ax=plt.subplots(figsize=(8.8,5.2))
        ax.plot(dx,curve('transient_spread_pnl'),color='forestgreen',lw=2.5,label='Spread capture')
        ax.plot(dx,-curve('transient_drift_pnl'),color='crimson',lw=2.5,label='MtM loss')
        ax.plot(dx,curve('transient_full_pnl'),color='black',lw=2.5,label='Net full PnL')
        style_axes(ax);ax.set_title(r'Finite horizon, $q_0=0$',fontsize=13);ax.legend()
        save(fig,'as_miss_pnl_decomposition',pdf)

