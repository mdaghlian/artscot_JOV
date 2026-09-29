"""Simulation check of spatial_scpc_euclidean.py / artscot_JOV.scpc_stats: does
scpc-python on Euclidean pial coordinates do what we expect on our ROI
geometries?

Questions
---------
1. Size: under a true null (mean d = 0) with spatially correlated noise, is the
   false-positive rate <= 5% whenever the true correlation is no stronger than
   the stated worst case, and how fast does it rise beyond it?
2. Robustness to the shape of d: same, with noise that has the skewed,
   heavy-tailed distribution of the real d (the SCPC guarantee is for Gaussian
   data).
3. Power: how often is a known non-zero mean detected?

Design
------
Geometry  : real thresholded ROIs (AS1 threshold), raw pial x, y, z (mm), both
            hemispheres together - exactly the input of spatial_scpc_euclidean.py.
True noise: correlation nu * I + (1 - nu) * exp(-D / len_true), D = Euclidean
            distance, nugget nu = 0.4, len_true in {2, 5, 10, 20} mm. The
            exponential kernel is positive definite in 3-D, so fields are drawn
            exactly (Cholesky).
Marginals : "gaussian" (unit variance), or "empirical": the Gaussian field is
            mapped through the quantiles of that ROI's real d minus its mean
            (Gaussian copula), so each vertex has the real distribution of d
            with mean 0. The mapping slightly weakens the correlation.
Options   : the four worst-case options of spatial_scpc_euclidean.py
            (len_5mm, len_10mm, avc_0.03, avc_0.10).
Power     : Gaussian noise, len_true = 5 mm, mean shifted by delta (in noise SD
            units), delta in {0.25, 0.5, 1.0}.

How scpc-python is used
-----------------------
Calling scpc() for every simulated dataset rebuilds the spatial setup each
time (~1 s), which is too slow for tens of thousands of datasets. The setup
depends only on the coordinates and avc, so it is built once per geometry x
option with the package's own set_oms_wfin(). Each dataset then uses the
package's max_rp() for the p-value. For an intercept-only OLS model ('d ~ 1'),
scpc() computes the statistic from wj = n * (X'X)^-1 * resid + mean = d itself,
so tau = sqrt(q) * w0'd / ||W1'd|| (scpc/core.py). The script checks this
shortcut against a full scpc() call on the first CHECK_EVERY-th datasets of
every condition and stops if p differs by more than 1e-8.

Usage (artscot env, Python >= 3.11 with scpc-python; see README). ~12 min.
-----
cd analysis/r0_reports
python spatial_scpc_euclidean_simulate.py
python spatial_scpc_euclidean_simulate.py --n-sim 100

Outputs (prf_for_pub/spatial_scpc_euclidean/simulate/)
------------------------------------------------------
sim_fpr.csv, sim_fpr.png     : false-positive rates
sim_power.csv, sim_power.png : detection rates for known shifts
sim_summary.txt              : expectation checks
"""
import os
import sys
import math
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

from scpc import scpc
from scpc.utils.spatial import set_oms_wfin, max_rp, get_distmat

from artscot_JOV.load_saved_info import get_roi
from artscot_JOV.scpc_stats import (  # same data loading and options as the main analysis
    load_pial_coords, load_shift_diff, avc_for_length, DEFAULT_OPTIONS as OPTIONS,
    AVC_MIN, AVC_MAX,
)
from spatial_scpc_euclidean import min_vx, out_dir as main_out_dir

opj = os.path.join
out_dir = opj(main_out_dir, 'simulate')
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GEOMETRIES = [                   # (subject, task used for the threshold, ROI)
    ('sub-02', 'AS1', 'v1custom'),     # large, flat-ish
    ('sub-01', 'AS1', 'v4custom'),     # small
    ('sub-02', 'AS1', 'TOcustom'),     # medium
    ('sub-05', 'AS1', 'IPScustom'),    # large, folded
]
LEN_TRUE = [2.0, 5.0, 10.0, 20.0]      # true correlation length (mm)
NUGGET = 0.4
MARGINALS = ['gaussian', 'empirical']
POWER_LEN, POWER_DELTAS = 5.0, [0.25, 0.5, 1.0]
ALPHA = 0.05
N_SIM = 500
CHECK_EVERY = 250                      # compare with full scpc() every N datasets
SEED = 0


# ---------------------------------------------------------------------------
# scpc-python: setup once per geometry x option, p-value per dataset
# ---------------------------------------------------------------------------
def package_setup(xyz, avc):
    '''scpc-python's spatial setup (projection W, critical value, Omega grid).'''
    return set_oms_wfin(xyz, avc, latlong=False)


def package_pvalue(y, setup):
    '''p-value of scpc() for OLS 'd ~ 1', from the precomputed setup
    (same arithmetic as scpc/core.py: wj = y for an intercept-only model).'''
    w = np.asarray(setup.wfin, dtype=float)
    q = w.shape[1] - 1
    wj = y[np.asarray(setup.perm, dtype=int)]
    proj = w[:, 1:].T @ wj
    tau = math.sqrt(q) * float(w[:, 0] @ wj) / math.sqrt(float(np.sum(proj ** 2)))
    return max_rp(setup.omsfin, q, abs(tau) / math.sqrt(q))[0]


def package_pvalue_full(y, xyz, avc):
    '''The public scpc() call used in spatial_scpc_euclidean.py.'''
    df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    df['d'] = y
    res = scpc(smf.ols('d ~ 1', data=df).fit(), df, coords_euclidean=['x', 'y', 'z'],
               avc=avc)
    return res.scpcstats[0][3]


# ---------------------------------------------------------------------------
# Null fields
# ---------------------------------------------------------------------------
def field_sampler(xyz, len_true, nu=NUGGET):
    '''Returns draw(n_sim, rng) -> (n_sim, n) unit-variance Gaussian fields with
    correlation nu * I + (1 - nu) * exp(-D / len_true), D = Euclidean distance.'''
    D = np.asarray(get_distmat(xyz, latlong=False), dtype=float)
    R = (1 - nu) * np.exp(-D / len_true)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    return lambda n_sim, rng: rng.standard_normal((n_sim, len(xyz))) @ L.T


def to_empirical(Z, d_roi):
    '''Gaussian copula: map each N(0, 1) value to the quantile of (d - mean d).'''
    r = np.sort(d_roi - d_roi.mean())
    return np.quantile(r, stats.norm.cdf(Z))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(n_sim=N_SIM):
    rng = np.random.default_rng(SEED)
    fpr_rows, pow_rows, max_check = [], [], 0.0
    for sub, task, roi in GEOMETRIES:
        t0 = time.time()
        d, vx_mask = load_shift_diff(sub, task)
        m = vx_mask & get_roi(sub, roi, combine_matches=True)
        assert m.sum() >= min_vx
        vx = np.where(m)[0]
        xyz, d_roi = load_pial_coords(sub)[vx], d[vx]
        # one scpc-python setup per worst-case option
        setups, avcs, len0 = {}, {}, {}
        for name, (kind, val) in OPTIONS.items():
            avc = avc_for_length(xyz, val) if kind == 'len' else val
            avcs[name] = float(np.clip(avc, AVC_MIN, AVC_MAX))
            setups[name] = package_setup(xyz, avcs[name])
            len0[name] = 1.0 / setups[name].c0   # worst-case length in mm (exact branch)
        geo = f'{sub} {roi.replace("custom", "")} {task} (n={len(vx)})'

        def evaluate(Y, cond):
            nonlocal max_check
            rej = {name: 0 for name in OPTIONS}
            for i, y in enumerate(Y):
                for name, S in setups.items():
                    p = package_pvalue(y, S)
                    rej[name] += p < ALPHA
                    if i % CHECK_EVERY == 0:
                        p_full = package_pvalue_full(y, xyz, avcs[name])
                        max_check = max(max_check, abs(p - p_full))
                        if abs(p - p_full) > 1e-8:
                            raise RuntimeError(f'shortcut != scpc() ({cond}, {name}): '
                                               f'{p} vs {p_full}')
            return rej

        # 1-2. size, for each true length and marginal
        for len_true in LEN_TRUE:
            Z = field_sampler(xyz, len_true)(n_sim, rng)
            for marg in MARGINALS:
                Y = Z if marg == 'gaussian' else to_empirical(Z, d_roi)
                rej = evaluate(Y, f'{geo} len={len_true} {marg}')
                for name, k in rej.items():
                    lo, hi = stats.binomtest(int(k), n_sim).proportion_ci()
                    fpr_rows.append({'geometry': geo, 'sub': sub, 'roi': roi, 'n_vx': len(vx),
                                     'marginal': marg, 'len_true': len_true, 'option': name,
                                     'avc': avcs[name], 'len0_mm': len0[name],
                                     'fpr': k / n_sim, 'ci_lo': lo, 'ci_hi': hi})
        # 3. power
        Z = field_sampler(xyz, POWER_LEN)(n_sim, rng)
        for delta in POWER_DELTAS:
            rej = evaluate(Z + delta, f'{geo} power delta={delta}')
            for name, k in rej.items():
                lo, hi = stats.binomtest(int(k), n_sim).proportion_ci()
                pow_rows.append({'geometry': geo, 'option': name, 'len0_mm': len0[name],
                                 'len_true': POWER_LEN, 'delta_sd': delta,
                                 'power': k / n_sim, 'ci_lo': lo, 'ci_hi': hi})
        print(f'  {geo} done ({time.time() - t0:.0f}s)', flush=True)
    return pd.DataFrame(fpr_rows), pd.DataFrame(pow_rows), max_check


# ---------------------------------------------------------------------------
# Checks and figures
# ---------------------------------------------------------------------------
def summarise(fpr, max_check, n_sim):
    '''Expectation: for each option, FPR not significantly above 5% whenever the
    true length <= its worst-case length len0 (Gaussian noise).'''
    lines = [f'n_sim = {n_sim} per condition; shortcut vs scpc(): max |dp| = {max_check:.1e}']
    for marg in MARGINALS:
        f = fpr[fpr.marginal == marg]
        inside = f[f.len_true <= f.len0_mm + 1e-6]
        bad = inside[inside.ci_lo > ALPHA]
        lines.append(f'[{marg}] within worst case (len_true <= len0): {len(inside)} conditions, '
                     f'FPR {inside.fpr.min():.3f}-{inside.fpr.max():.3f}; '
                     f'significantly above 5%: {len(bad)}'
                     + ('' if bad.empty else ' -> ' + '; '.join(
                         f'{r.geometry} {r.option} len={r.len_true:g}: {r.fpr:.3f}'
                         for r in bad.itertuples())))
        outside = f[f.len_true > f.len0_mm + 1e-6]
        lines.append(f'[{marg}] beyond worst case: {len(outside)} conditions, '
                     f'FPR {outside.fpr.min():.3f}-{outside.fpr.max():.3f}')
    txt = '\n'.join(lines)
    print(txt)
    open(opj(out_dir, 'sim_summary.txt'), 'w').write(txt + '\n')


OPT_COL = {'len_5mm': '#2a6fdb', 'len_10mm': '#173f7f', 'avc_0.03': '#d9730d',
           'avc_0.10': '#8c4a0a'}


def plot_fpr(fpr):
    geoms = list(dict.fromkeys(fpr.geometry))
    fig, axs = plt.subplots(len(MARGINALS), len(geoms), figsize=(4.2 * len(geoms), 3.6 * len(MARGINALS)),
                            sharex=True, sharey=True, squeeze=False)
    for i, marg in enumerate(MARGINALS):
        for j, geo in enumerate(geoms):
            ax = axs[i, j]
            for name in OPTIONS:
                dt = fpr[(fpr.marginal == marg) & (fpr.geometry == geo) & (fpr.option == name)]
                l0 = dt.len0_mm.iloc[0]
                ax.errorbar(dt.len_true, dt.fpr.clip(lower=1e-3),
                            yerr=[dt.fpr - dt.ci_lo, dt.ci_hi - dt.fpr],
                            fmt='-o', ms=4, lw=2, capsize=2, color=OPT_COL[name],
                            label=f'{name} (len0 = {l0:.1f} mm)')
                ax.axvline(l0, color=OPT_COL[name], lw=0.7, ls='--')
            ax.axhline(ALPHA, color='#1f1f1f', lw=0.8, ls=':')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(LEN_TRUE)
            ax.set_xticklabels([f'{v:g}' for v in LEN_TRUE])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f'{geo}\n{marg} noise', fontsize=9)
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            if i == len(MARGINALS) - 1:
                ax.set_xlabel('true correlation length (mm)')
        axs[i, 0].set_ylabel('false-positive rate (alpha = 0.05)')
    fig.suptitle('scpc-python, Euclidean coordinates: null simulation (dashed = worst-case '
                 'length of each option; dotted = 5%)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(opj(out_dir, 'sim_fpr.png'), dpi=120)
    plt.close(fig)


def plot_power(pw):
    geoms = list(dict.fromkeys(pw.geometry))
    fig, axs = plt.subplots(1, len(geoms), figsize=(4.2 * len(geoms), 3.4), sharey=True,
                            squeeze=False)
    for j, geo in enumerate(geoms):
        ax = axs[0, j]
        for name in OPTIONS:
            dt = pw[(pw.geometry == geo) & (pw.option == name)]
            ax.errorbar(dt.delta_sd, dt.power, yerr=[dt.power - dt.ci_lo, dt.ci_hi - dt.power],
                        fmt='-o', ms=4, lw=2, capsize=2, color=OPT_COL[name], label=name)
        ax.set_ylim(0, 1)
        ax.set_xticks(POWER_DELTAS)
        ax.set_xlabel('true mean shift (noise SD)')
        ax.set_title(geo, fontsize=9)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel(f'detection rate (true length {POWER_LEN:g} mm)')
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(opj(out_dir, 'sim_power.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    n_sim = N_SIM
    if '--n-sim' in sys.argv:
        n_sim = int(sys.argv[sys.argv.index('--n-sim') + 1])
    fpr, pw, max_check = run(n_sim)
    fpr.to_csv(opj(out_dir, 'sim_fpr.csv'), index=False)
    pw.to_csv(opj(out_dir, 'sim_power.csv'), index=False)
    summarise(fpr, max_check, n_sim)
    plot_fpr(fpr)
    plot_power(pw)
