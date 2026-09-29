"""Within-subject test of mean(d) = 0 across the vertices of an ROI, robust to
spatial autocorrelation: SCPC (Mueller & Watson 2022) on Euclidean pial
vertex coordinates, computed with the scpc-python package
(https://github.com/spatial-spur/scpc-python).

d = shift_mag(AS0 -> ASx, gauss) - shift_mag(AS0 -> ASx, norm)

All SCPC calculations are done by scpc-python; this module only prepares its
inputs. See r0_reports/scpc_explained.md.

The worst-case spatial correlation is stated as a correlation length len0 (mm):
correlation exp(-distance / len0). scpc-python takes the worst case as avc (the
average pairwise correlation), so avc is set per ROI to the value len0 implies
on that ROI's distances, using the package's own helpers.

Typical use
-----------
>>> from artscot_JOV.scpc_stats import load_pial_coords, load_shift_diff, scpc_mean_test
>>> from artscot_JOV.load_saved_info import get_roi
>>> xyz_all = load_pial_coords('sub-02')
>>> d, vx_mask = load_shift_diff('sub-02', 'AS1')
>>> vx = np.where(vx_mask & get_roi('sub-02', 'v1custom'))[0]
>>> res = scpc_mean_test(d[vx], xyz_all[vx], len0=5.0)   # mean, se, pvalue, ci, ...
"""
import os
import pickle
import numpy as np
import pandas as pd

from artscot_JOV.load_saved_info import default_prf_dir, load_data_prf

opj = os.path.join
default_mesh_path = opj(default_prf_dir, 'mesh_info.pkl')

# scpc-python requires avc in (0.001, 0.99)
AVC_MIN, AVC_MAX = 0.0011, 0.989

# Worst-case options used in the analysis: ('len', mm) -> avc matched per ROI,
# ('avc', value) -> the same avc for every ROI (for comparison only; a fixed
# avc implies a different correlation length in every ROI).
DEFAULT_OPTIONS = {
    'len_5mm': ('len', 5.0),     # just beyond the measurement-noise range
    'len_10mm': ('len', 10.0),   # conservative
}

_MESH_CACHE = {}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_pial_coords(sub, mesh_path=default_mesh_path):
    '''Pial vertex coordinates (mm) for one subject, lh then rh (the same
    vertex order as the ROI masks from get_roi). From prf_for_pub/mesh_info.pkl.'''
    if mesh_path not in _MESH_CACHE:
        with open(mesh_path, 'rb') as f:
            _MESH_CACHE[mesh_path] = pickle.load(f)
    mesh = _MESH_CACHE[mesh_path][sub]
    return np.concatenate([mesh['lh']['coords'], mesh['rh']['coords']])


def load_shift_diff(sub, task, min_rsq=0.1, max_ecc=5, look_in=default_prf_dir):
    '''d = shift_mag(AS0 -> task, gauss) - shift_mag(AS0 -> task, norm) at every
    vertex, and the threshold mask (rsq > min_rsq and ecc < max_ecc in all four
    fits) - the same quantities as PAIRED_df in artscot_stats.ipynb.

    Returns
    -------
    d       : (n_vx,) float
    vx_mask : (n_vx,) bool
    '''
    from artscot_JOV.prfpy_functions import PrfMulti, Prf1T1M
    tasks, models = ['AS0', task], ['gauss', 'norm']
    pars = load_data_prf(sub=sub, task_list=tasks, model_list=models,
                         roi_fit='all', include='tc', look_in=look_in)
    short = {'gauss': 'G', 'norm': 'N'}
    objs, ids = [], []
    for t in tasks:
        for mdl in models:
            objs.append(Prf1T1M(prf_params=pars[t][mdl], model=mdl))
            ids.append(f'{t}f_{short[mdl]}')
    pm = PrfMulti(objs, ids)
    th = {}
    for i in ids:
        th[f'{i}-min-rsq'] = min_rsq
        th[f'{i}-max-ecc'] = max_ecc
    vx_mask = np.asarray(pm.return_vx_mask(th))
    sG = pm.return_diff_params('AS0f_G', f'{task}f_G', 'shift_mag')['shift_mag']
    sN = pm.return_diff_params('AS0f_N', f'{task}f_N', 'shift_mag')['shift_mag']
    return np.asarray(sG) - np.asarray(sN), vx_mask


# ---------------------------------------------------------------------------
# SCPC via scpc-python
# ---------------------------------------------------------------------------
def avc_for_length(xyz, len0):
    '''avc implied by a worst-case correlation length len0 (mm): the average of
    exp(-distance / len0) over all vertex pairs, using the package's own
    distance matrix (get_distmat), pair vector (lvech) and average (get_avc).'''
    from scpc.utils.spatial import get_distmat, get_avc
    from scpc.utils.matrix import lvech
    return get_avc(1.0 / len0, lvech(get_distmat(xyz, latlong=False)))


def scpc_mean_test(y, xyz, len0=None, avc=None):
    '''SCPC test of mean(y) = 0 with scpc-python, on Euclidean coordinates.

    Give the worst case as a correlation length len0 (mm; preferred) OR
    as a fixed avc.

    Parameters
    ----------
    y    : (n,) values at the ROI vertices (e.g. d[vx])
    xyz  : (n, 3) pial coordinates of the same vertices (mm)

    Returns
    -------
    dict: mean, se, tau, pvalue, ci_lo, ci_hi (95%), q (number of spatial
    components), cv (critical value), avc (used), avc_clipped, c0, method
    ('exact' below 4500 vertices), len0_mm (worst-case length; exact branch)
    '''
    import statsmodels.formula.api as smf
    from scpc import scpc
    if (len0 is None) == (avc is None):
        raise ValueError('give exactly one of len0 or avc')
    if avc is None:
        avc = avc_for_length(xyz, len0)
    avc_use = float(np.clip(avc, AVC_MIN, AVC_MAX))
    df = pd.DataFrame(np.asarray(xyz, dtype=float), columns=['x', 'y', 'z'])
    df['d'] = np.asarray(y, dtype=float)
    model = smf.ols('d ~ 1', data=df).fit()  # intercept = mean of y
    res = scpc(model, df, coords_euclidean=['x', 'y', 'z'], avc=avc_use)
    est, se, tau, p, ci_lo, ci_hi = res.scpcstats[0]
    return {
        'mean': est, 'se': se, 'tau': tau, 'pvalue': p, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'q': res.q, 'cv': res.cv, 'avc': avc_use, 'avc_clipped': avc_use != avc,
        'c0': res.c0, 'method': res.method,
        # the exact branch works in raw mm distances, so 1 / c0 is a length in mm
        'len0_mm': 1.0 / res.c0 if res.method == 'exact' else np.nan,
    }
