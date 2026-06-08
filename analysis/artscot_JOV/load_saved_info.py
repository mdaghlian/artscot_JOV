import numpy as np
import scipy.io
import yaml
import pickle
import os
import sys

from prfpy_csenf.stimulus import PRFStimulus2D
import pandas as pd
from dpu_mini.utils import *
from dpu_mini.fs_tools import *
from artscot_JOV.utils import *

opj = os.path.join

# Derive all paths from this file's location — no machine-specific paths needed.
_PKG_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../analysis/artscot_JOV
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))   # repo root

default_prf_dir = opj(_REPO_ROOT, 'prf_for_pub')
default_ses     = 'ses-1'


def get_yml_settings_path(yml_name='s0_prf_analysis.yml'):
    yml_path = opj(os.path.dirname(_PKG_DIR), 's0_analysis_steps', yml_name)
    return yml_path


def load_data_tc(sub, task_list, ses=default_ses, look_in=default_prf_dir, do_demo=False, n_timepts=225):
    '''Load PSC time courses from prf_for_pub.

    Parameters
    ----------
    sub         str     subject ID, e.g. 'sub-01'
    task_list   str or list   task(s) to load, e.g. ['AS0', 'AS1', 'AS2']
    ses         str     session, default 'ses-1'
    look_in     str     root of data directory (default: prf_for_pub/)
    do_demo     bool    if True, trim to first 100 vertices
    n_timepts   int     expected number of time points (used to orient the array)

    Returns
    -------
    data_tc     dict    {task: ndarray (vertices x TRs)}
    '''
    if isinstance(task_list, str):
        task_list = [task_list]

    data_tc  = {}
    this_dir = opj(look_in, sub, ses)
    for task in task_list:
        try:
            data_tc_path = dag_find_file_in_folder(
                [sub, ses, dag_hyphen_parse('task', task), 'hemi-LR', '.npy'],
                this_dir, exclude=['correlation', 'mean_epi'])
        except:
            data_tc_path = dag_find_file_in_folder(
                [sub, ses, dag_hyphen_parse('task', task), 'hemi-lr', '.npy'],
                this_dir, exclude=['correlation', 'mean_epi'])
        data_tc[task] = set_tc_shape(np.load(data_tc_path), n_timepts=n_timepts)
        if do_demo:
            data_tc[task] = data_tc[task][0:100, :]
    return data_tc


def load_data_prf(sub, task_list, model_list, var_to_load='pars', roi_fit='all',
                  fit_stage='iter', ses=default_ses, look_in=default_prf_dir, **kwargs):
    '''Load pRF fit parameters from prf_for_pub pkl files.

    Parameters
    ----------
    sub         str
    task_list   str or list    e.g. 'AS0' or ['AS0', 'AS1']
    model_list  str or list    e.g. 'gauss' or ['gauss', 'norm']
    var_to_load str            key to extract from pkl dict: 'pars', 'preds', 'settings'
    roi_fit     str            ROI label used when fitting, typically 'all'
    fit_stage   str            'iter' (default) or 'grid'
    ses         str            session label
    look_in     str            root of data directory (default: prf_for_pub/)

    Returns
    -------
    prf_vars    dict           {task: {model: ndarray}}
    '''
    include = kwargs.get('include', [])
    if isinstance(include, str):
        include = [include]
    exclude = kwargs.get('exclude', None)
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(task_list, str):
        task_list = [task_list]
    if isinstance(model_list, str):
        model_list = [model_list]

    prf_vars  = {}
    this_dir = opj(look_in, sub, ses)
    for task in task_list:
        prf_vars[task] = {}
        for model in model_list:
            this_include = include + [sub, dag_hyphen_parse('task', task), model, roi_fit, fit_stage, '.pkl']
            prf_vars_path = dag_find_file_in_folder(this_include, this_dir, exclude=exclude)
            print(prf_vars_path)
            with open(prf_vars_path, 'rb') as pkl_file:
                pkl_data = pickle.load(pkl_file)
            if 'pred' in var_to_load:
                prf_vars[task][model] = set_tc_shape(pkl_data[var_to_load])
            else:
                prf_vars[task][model] = pkl_data[var_to_load]
    return prf_vars


def get_roi(sub, label, **kwargs):
    '''Return a boolean surface mask for a named ROI.

    Loads from prf_for_pub/<sub>/<sub>_roi.npz.
    Available ROI keys: all, v1custom, v2custom, v3custom, v3abcustom,
    v4custom, LOcustom, TOcustom, IPScustom.

    Parameters
    ----------
    sub     str   subject ID, e.g. 'sub-01'
    label   str   ROI name, e.g. 'v1custom'

    Returns
    -------
    mask    np.ndarray (bool), shape (n_vertices,)
    '''
    look_in = kwargs.get('look_in', default_prf_dir)
    f = opj(look_in, sub, f'{sub}_roi.npz')
    return np.load(f)[label]

def get_roi_FS(sub, label, fs_dir, **kwargs):
    '''Return a boolean surface mask for a named ROI using FreeSurfer label files.
    Requires FreeSurfer outputs — not part of prf_for_pub.
    Use get_roi() instead when working with prf_for_pub data.
    '''
    roi_idx = dag_load_roi(sub=sub, roi=label, fs_dir=fs_dir, **kwargs)
    return roi_idx


def get_design_matrix_npy(task_list, prf_dir=[]):
    '''Load stimulus design matrices bundled with the package.

    Returns
    -------
    dm_npy  dict   {task: ndarray (n_pix x n_pix x n_TRs)}
    '''
    if not isinstance(task_list, list):
        task_list = [task_list]
    dm_npy = {}
    for task in task_list:
        dm_path = dag_find_file_in_folder(['design', task], _PKG_DIR)
        dm_npy[task] = scipy.io.loadmat(dm_path)['stim']
    return dm_npy


def get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir, cut_vols=5):
    '''Construct PRFStimulus2D objects from the bundled design matrices.

    Settings (screen size, TR, etc.) are read from the first subject's
    stored fit settings in prf_dir.
    '''
    if not isinstance(task_list, list):
        task_list = [task_list]
    dm_npy = get_design_matrix_npy(task_list, prf_dir=prf_dir)
    with open(get_yml_settings_path()) as f:
        fit_settings = yaml.safe_load(f)
    prfpy_stim = {}
    for task in task_list:
        print(task)
        prfpy_stim[task] = PRFStimulus2D(
            screen_size_cm=fit_settings['screen_size_cm'],
            screen_distance_cm=fit_settings['screen_distance_cm'],
            design_matrix=dm_npy[task][:, :, cut_vols:],
            axis=0,
            TR=fit_settings['TR']
        )
    return prfpy_stim


def get_scotoma_info(sub):
    '''Return scotoma geometry for all tasks (AS0, AS1, AS2).

    Builds coordinate grids from the stimulus design matrix.
    Values are in visual degrees, corrected for the screen-distance
    discrepancy between exptools (210 cm) and fitting (196 cm).

    Returns
    -------
    scotoma_info    dict  keyed by 'task-AS0', 'task-AS1', 'task-AS2'
        Each value has: scotoma_centre, scotoma_radius, aperture_rad,
        n_pix, grid (x_deg, y_deg meshgrids)
    '''
    scotoma_info = {}
    task_list = 'task-AS0'
    prfpy_stim = get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir)[task_list]
    aperture_rad = prfpy_stim.screen_size_degrees / 2
    n_pix = prfpy_stim.design_matrix.shape[0]
    x_deg = np.tile(np.linspace(-aperture_rad, aperture_rad, n_pix), (n_pix, 1))
    y_deg = np.tile(np.linspace(-aperture_rad, aperture_rad, n_pix), (n_pix, 1)).T
    grid = {'x_deg': x_deg, 'y_deg': y_deg}

    # Screen-distance correction: exptools used 210 cm, fitting used 196 cm
    exptools_ssize = np.degrees(2 * np.arctan((39.3 / 2) / 210))
    fitting_ssize  = np.degrees(2 * np.arctan((39.3 / 2) / 196))
    conversion_factor = fitting_ssize / exptools_ssize

    scotoma_info['task-AS0'] = {
        'scotoma_centre': [],
        'scotoma_radius': [],
        'aperture_rad': aperture_rad,
        'n_pix': n_pix,
        'grid': grid,
        'name': 'task-AS0',
    }
    scotoma_info['task-AS1'] = {
        'scotoma_centre': [0.8284 * conversion_factor, 0.8284 * conversion_factor],
        'scotoma_radius': 0.8284 * conversion_factor,
        'aperture_rad': aperture_rad,
        'n_pix': n_pix,
        'grid': grid,
        'name': 'task-AS1',
    }
    scotoma_info['task-AS2'] = {
        'scotoma_centre': [0 * conversion_factor, 0 * conversion_factor],
        'scotoma_radius': 2 * conversion_factor,
        'aperture_rad': aperture_rad,
        'n_pix': n_pix,
        'grid': grid,
        'name': 'task-AS2',
    }
    return scotoma_info


def load_params_generic(params_file, load_all=False, load_var=[]):
    '''Load pRF parameters from a .npy or .pkl file into a numpy array.'''
    if isinstance(params_file, str):
        if params_file.endswith('npy'):
            params = np.load(params_file)
        elif params_file.endswith('pkl'):
            with open(params_file, 'rb') as f:
                data = pickle.load(f)
            if len(load_var) == 1:
                params = data[load_var[0]]
            elif len(load_var) > 1:
                params = {v: data[v] for v in load_var}
            elif load_all:
                params = {v: data[v] for v in data.keys()}
            else:
                params = data['pars']
    elif isinstance(params_file, np.ndarray):
        params = params_file.copy()
    elif isinstance(params_file, pd.DataFrame):
        dict_keys = list(params_file.keys())
        if 'hemi' not in dict_keys:
            params = np.array((
                params_file['x'][0], params_file['y'][0],
                params_file['prf_size'][0], params_file['A'][0],
                params_file['bold_bsl'][0], params_file['B'][0],
                params_file['C'][0], params_file['surr_size'][0],
                params_file['D'][0], params_file['r2'][0]))
        else:
            raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized input type for '{params_file}'")
    return params
