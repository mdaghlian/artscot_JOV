"""SCPC test of mean(d) = 0 within subject x ROI x task, using the scpc-python
package on Euclidean (x, y, z) pial vertex coordinates.

d = shift_mag(AS0 -> ASx, gauss) - shift_mag(AS0 -> ASx, norm)

All SCPC calculations are done by scpc-python (Becker, Boll & Voth 2026,
https://github.com/spatial-spur/scpc-python; Mueller & Watson 2022), called via
artscot_JOV.scpc_stats. See r0_reports/scpc_explained.md.

Worst case (each run for every subject x ROI x task;
artscot_JOV.scpc_stats.DEFAULT_LENGTHS)
------------------------------------------------------
len_5mm, len_10mm : worst-case correlation length l0 = 5 / 10 mm, i.e.
                    correlation exp(-distance / l0). scpc-python takes the worst
                    case as avc (average pairwise correlation), so avc is set per
                    ROI to the value l0 implies on that ROI's Euclidean
                    distances, computed with the package's own helpers.

Coordinates: raw pial surface coordinates (mm) of the thresholded ROI
vertices, both hemispheres together (no offset).

Usage (artscot env, Python >= 3.11 with scpc-python; see README)
-----
cd analysis/r0_reports
python spatial_scpc_euclidean.py                 # all subjects
python spatial_scpc_euclidean.py sub-01 sub-02

Outputs (prf_for_pub/spatial_scpc_euclidean/)
---------------------------------------------
scpc_euclidean_results.csv : one row per subject x task x ROI x worst-case length
scpc_euclidean_bars.png    : per-ROI mean d, per-subject significance
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from artscot_JOV.load_saved_info import default_prf_dir, get_roi
from artscot_JOV.scpc_stats import (
    load_pial_coords, load_shift_diff, scpc_mean_test, DEFAULT_LENGTHS,
)

opj = os.path.join
out_dir = opj(default_prf_dir, 'spatial_scpc_euclidean')   # outputs live with the data, not in git
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
sub_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
roi_list = ['v1custom', 'v2custom', 'v3custom', 'v3abcustom', 'v4custom',
            'LOcustom', 'TOcustom', 'IPScustom']
task_list = ['AS1', 'AS2']
min_rsq, max_ecc, min_vx = 0.1, 5, 50          # same as PAIRED_df
OPTIONS = DEFAULT_LENGTHS                      # name -> worst-case length (mm)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_case(y, xyz):
    '''All worst-case lengths for one subject x task x ROI.'''
    rows = []
    for name, len0 in OPTIONS.items():
        r = scpc_mean_test(y, xyz, len0=len0)
        rows.append({'option': name, 'mean_d': r.pop('mean'), **r})
    return rows


def run_subject(sub):
    t0 = time.time()
    xyz_all = load_pial_coords(sub)
    rows = []
    for task in task_list:
        d, vx_mask = load_shift_diff(sub, task, min_rsq=min_rsq, max_ecc=max_ecc)
        for roi in roi_list:
            m = vx_mask & get_roi(sub, roi, combine_matches=True)
            if m.sum() < min_vx:
                continue
            vx = np.where(m)[0]
            for r in run_case(d[vx], xyz_all[vx]):
                rows.append({'sub': sub, 'task': task, 'roi': roi, 'n_vx': vx.size, **r})
        print(f'  {sub} {task} done ({time.time() - t0:.0f}s)', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_bars(df):
    '''One panel per option x task. Bar = group mean over subjects of the
    per-subject mean d; dots = subjects (x = p < 0.05, o = n.s.); whiskers =
    SCPC 95% CI; numbers = subjects significant / tested.'''
    options = list(OPTIONS)
    rois = roi_list
    subs = sorted(df['sub'].unique())
    offs = dict(zip(subs, np.linspace(-0.25, 0.25, len(subs))))
    fig, axs = plt.subplots(len(options), len(task_list),
                            figsize=(11, 2.4 * len(options)),
                            sharex=True, sharey=True, squeeze=False)
    for i, opt in enumerate(options):
        do = df[df.option == opt].set_index(['sub', 'task', 'roi'])
        for j, task in enumerate(task_list):
            ax = axs[i, j]
            for k, roi in enumerate(rois):
                keys = [(s, task, roi) for s in subs if (s, task, roi) in do.index]
                if not keys:
                    continue
                ax.bar(k, np.mean([do.loc[key].mean_d for key in keys]),
                       width=0.7, color='#fdbb84', zorder=1)
                n_sig = 0
                for key in keys:
                    r = do.loc[key]
                    x = k + offs[key[0]]
                    sig = r.pvalue < 0.05
                    n_sig += sig
                    ax.plot([x, x], [r.ci_lo, r.ci_hi], color='#6b6b6b', lw=0.8,
                            alpha=0.9 if sig else 0.4, zorder=2)
                    if sig:
                        ax.scatter(x, r.mean_d, marker='x', s=22, color='#1f1f1f', zorder=3)
                    else:
                        ax.scatter(x, r.mean_d, marker='o', s=16, facecolors='none',
                                   edgecolors='#6b6b6b', zorder=3)
                ax.text(k, 1.22, f'{n_sig}/{len(keys)}', ha='center', fontsize=7,
                        color='#1f1f1f')
            ax.axhline(0, color='#1f1f1f', lw=0.8, ls='--', alpha=0.5)
            ax.set_ylim(-0.6, 1.35)
            ax.set_xticks(range(len(rois)))
            ax.set_xticklabels([r.replace('custom', '') for r in rois], fontsize=8)
            ax.set_title(f'Euclidean xyz, {opt} - {task}', fontsize=9)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
        axs[i, 0].set_ylabel('mean d (deg)', fontsize=8)
    fig.suptitle('scpc-python on Euclidean pial coordinates. Bar = group mean of per-subject '
                 'mean d; x = p < 0.05, o = n.s.; whiskers = 95% CI (clipped); '
                 'numbers = significant / tested', fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(opj(out_dir, 'scpc_euclidean_bars.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    subs = sys.argv[1:] or sub_list
    df = pd.concat([run_subject(s) for s in subs], ignore_index=True)
    df.to_csv(opj(out_dir, 'scpc_euclidean_results.csv'), index=False)
    plot_bars(df)
    print(df.groupby('option').pvalue.apply(lambda p: f'{(p < 0.05).sum()}/{len(p)} p<0.05'))
