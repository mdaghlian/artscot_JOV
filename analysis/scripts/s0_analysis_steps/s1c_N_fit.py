#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V


import sys
import os
opj = os.path.join
import numpy as np
import yaml
import pickle
from datetime import datetime, timedelta
import time

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, DoG_Iso2DGaussianFitter, CSS_Iso2DGaussianFitter,Norm_Iso2DGaussianFitter

from pfa_scripts.load_saved_info import get_design_matrix_npy, load_data_tc, load_params_generic,get_roi, get_number_of_vx, get_yml_settings_path

from dag_prf_utils.utils import *
from dag_prf_utils.prfpy_functions import *


def main(argv):

    """
---------------------------------------------------------------------------------------------------

Fit the time series using the extended model

Args:
    -s (--sub=)         e.g., 01
    -m (--model=)       e.g., norm, css, dog
    -t (--task=)        e.g., AS0, AS1, AS2
    -r (--roi_fit=)     e.g., all, V1_exvivo
    --nr_jobs           number of jobs
    --verbose
    --tc                
    --bgfs
    --ow               overwrite

Example:


---------------------------------------------------------------------------------------------------
    """
    derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'    
    print('\n\n\n\n')

    # default
    ses = 'ses-1'
    model = None
    fit_hrf = False
    verbose = True
    cut_vols = 5
    n_timepts = 225 - cut_vols

    # Specify
    sub = None
    task = None
    roi_fit = None
    constraints = None
    nr_jobs = None
    prf_out = 'prf'    
    ignore_mask = False
    overwrite = False
    rsq_threshold = None
    yml_name = 's0_prf_analysis.yml'
    ow_prf_settings = {} # overwrite prf settings from the yml file with these settings
    for i,arg in enumerate(argv):
        if arg in ('-s', '--sub'):
            sub = dag_hyphen_parse('sub', argv[i+1])
        elif arg in ('--ses'):
            ses = dag_hyphen_parse('ses', argv[i+1])            
        elif arg in ('-t', '--task'):
            task = dag_hyphen_parse('task', argv[i+1])
        elif arg in ("-m", "--model"):
            model = argv[i+1]            
        elif '--prf_out' in arg:
            prf_out = argv[i+1]                
        elif arg in ("-r", "--roi_fit"):
            roi_fit = argv[i+1]
        elif arg in ("--nr_jobs"):
            nr_jobs = int(argv[i+1])  
        elif arg in ("--tc"):
            constraints = "tc"
        elif arg in ("--bgfs"):
            constraints = "bgfs"
        elif arg in ("--ignore_mask"):
            ignore_mask = True
            zero_pad = True
        elif arg in ("--rsq_threshold"):
            rsq_threshold = float(argv[i+1])                        
        elif arg in ("--yml_name"):
            yml_name = argv[i+1]            
        elif arg in ("--ow" or "--overwrite"):
            overwrite = True
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit(2)
        elif '--' in arg:
            ow_prf_settings[arg.split('--')[-1]] = dag_arg_checker(argv[i+1])    


    # Where to save everything
    prf_dir = opj(derivatives_dir, prf_out)    
    output_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-fits"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    # load basic settings from the yml file
    prf_settings_file = get_yml_settings_path(yml_name)
    print(prf_settings_file)
    with open(prf_settings_file) as f:
        prf_settings = yaml.safe_load(f)
    dm_task = task +''
    dm_task = dm_task.split('_run')[0] # 
    dm_task = dm_task.split('_fold')[0] 
    # Add important info to settings
    prf_settings['sub'] = sub
    prf_settings['model'] = model
    prf_settings['task'] = task
    prf_settings['roi_fit'] = roi_fit
    prf_settings['nr_jobs'] = nr_jobs
    prf_settings['constraints'] = constraints
    prf_settings['ses'] = ses
    prf_settings['task'] = task
    prf_settings['fit_hrf'] = fit_hrf
    prf_settings['verbose'] = verbose
    prf_settings['prf_out'] = out
    prf_settings['prf_dir'] = prf_dir
    prf_settings['ignore_mask'] = ignore_mask
    prf_settings['cut_vols'] = cut_vols
    prf_settings['n_timepts'] = n_timepts        
    if rsq_threshold!=None:        
        prf_settings['rsq_threshold'] = rsq_threshold   
    if len(ow_prf_settings)>0:
        for key in ow_prf_settings.keys():
            prf_settings[key] = ow_prf_settings[key]
            print(f'Overwriting {key} with {ow_prf_settings[key]}')

    # ****************************************************

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD TIME SERIES & MASK THE ROI   
    num_vx = get_number_of_vx(sub=sub)
    print(prf_dir)

    m_prf_tc_data = load_data_tc(
        sub=sub, ses=ses, task_list=task, look_in=prf_dir, n_timepts=n_timepts)[task]
    roi_mask = get_roi(sub=sub, label=roi_fit)
    # Are we limiting the fits to an roi?
    num_vx_in_roi = roi_mask.sum()
    print(roi_mask.shape)
    print(f'Fitting {roi_fit} {num_vx_in_roi} voxels out of {num_vx} voxels in total')
    print('Removing irrelevant times courses')
    print('These will be added in later, as zero fits in the parameters')
    zero_pad = False
    print(m_prf_tc_data.shape)
    print(f'RSQ THRESHOLD {prf_settings["rsq_threshold"]}')
    if not ignore_mask:        
        m_prf_tc_data = mask_time_series(ts=m_prf_tc_data, mask=roi_mask, zero_pad=zero_pad, )

    # ************************************************************************


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   LOAD DESIGN MATRIX   
    design_matrix = get_design_matrix_npy([dm_task])[dm_task]         
    design_matrix = design_matrix[:,:,cut_vols:]
    assert design_matrix.shape[-1]==m_prf_tc_data.shape[-1]
    prf_stim = PRFStimulus2D(
        screen_size_cm=prf_settings['screen_size_cm'],          # Distance of screen to eye
        screen_distance_cm=prf_settings['screen_distance_cm'],  # height of the screen (i.e., the diameter of the stimulated region)
        design_matrix=design_matrix,                            # dm (npix x npix x time_points)
        TR=prf_settings['TR'],                                  # TR
        )   
    max_eccentricity = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region 
    # ************************************************************************
    
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD "previous" GAUSSIAN MODEL & fitter   
    gg = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf=prf_settings['hrf']['pars'],                    # These are the parameters for the HRF that we normally use at Spinoza (with 7T data). (we can fit it, this will be done later...)
        normalize_RFs=prf_settings['normalize_RFs'],        # Normalize the volume of the RF (so that RFs w/ different sizes have the same volume. Generally not needed, as this can be solved using the beta values i.e.,amplitude)
        )
    gf = Iso2DGaussianFitter(
        data=m_prf_tc_data,             # time series
        model=gg,                       # model (see above)
        n_jobs=prf_settings['nr_jobs'], # number of jobs to use in parallelization 
        )
    iter_gauss = dag_find_file_in_folder([sub, 'gauss', roi_fit, 'iter', task, constraints, '.pkl'], output_dir, return_msg=None)        
    if iter_gauss is None:
        # -> gauss is faster than the extended, so we may have the 'all' fit already...
        # -> check for this and use it if appropriate (make sure the correct constraints are applied)
        # iter_gauss = dag_find_file_in_folder([sub, 'gauss', 'all', 'iter', task, constraints], output_dir, return_msg=None)        
        iter_gauss = dag_find_file_in_folder([sub, 'gauss', 'all', 'iter', task, 'bgfs', '.pkl'], output_dir, return_msg=None)        
        print('**** USING ALL *****')

    iter_gauss_params = load_params_generic(iter_gauss)
    # Apply the same mask to the gaussian parameters
    if not zero_pad:
        iter_gauss_params = iter_gauss_params[roi_mask,:]

    gf.iterative_search_params = iter_gauss_params
    gf.rsq_mask = iter_gauss_params[:,-1] > prf_settings['rsq_threshold']
    # CREATE GAUSSIAN MODEL<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ************************************************************************


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE & RUN EXTENDED MODEL
    if model=='norm': # ******************************** NORM
        gg_ext = Norm_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf=prf_settings['hrf']['pars'],                    
            normalize_RFs=prf_settings['normalize_RFs'],        
            )
        gf_ext = Norm_Iso2DGaussianFitter(
            data=m_prf_tc_data,           
            model=gg_ext,                  
            n_jobs=prf_settings['nr_jobs'],
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = prf_settings['use_previous_gaussian_fitter_hrf'], 
            )
        ext_grid_bounds = [
            prf_settings['prf_ampl'],
            prf_settings['norm']['surround_baseline_bound']
        ]
        ext_grids = [
            np.array(prf_settings['norm']['surround_amplitude_grid'], dtype='float32'),
            np.array(prf_settings['norm']['surround_size_grid'], dtype='float32'),
            np.array(prf_settings['norm']['neural_baseline_grid'], dtype='float32'),
            np.array(prf_settings['norm']['surround_baseline_grid'], dtype='float32'),            
        ]
        ext_custom_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            (prf_settings['norm']['neural_baseline_bound']),        # neural baseline (b) 
            (prf_settings['norm']['surround_baseline_bound']),      # surround baseline (d)
            ] 
        
    elif model=='dog': # ******************************** DOG
        gg_ext = DoG_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf=prf_settings['hrf']['pars'],                    
            normalize_RFs=prf_settings['normalize_RFs'],        
            )
        gf_ext = DoG_Iso2DGaussianFitter(
            data=m_prf_tc_data,           
            model=gg_ext,                  
            n_jobs=prf_settings['nr_jobs'],
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = prf_settings['use_previous_gaussian_fitter_hrf'], 
            )
        ext_grid_bounds = [
            prf_settings['prf_ampl'],
            prf_settings['dog']['surround_amplitude_bound']
        ]
        ext_grids = [
            np.array(prf_settings['dog']['dog_surround_amplitude_grid'], dtype='float32'),
            np.array(prf_settings['dog']['dog_surround_size_grid'], dtype='float32'),
        ]
        ext_custom_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            ]

    elif model=='css': # ******************************** CSS
        gg_ext = CSS_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf=prf_settings['hrf']['pars'],                    
            normalize_RFs=prf_settings['normalize_RFs'],        
            )
        gf_ext = CSS_Iso2DGaussianFitter(
            data=m_prf_tc_data,           
            model=gg_ext,                  
            n_jobs=prf_settings['nr_jobs'],
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = prf_settings['use_previous_gaussian_fitter_hrf'], 
            )
        ext_grid_bounds = [
            prf_settings['prf_ampl']
        ]
        ext_grids = [
            np.array(prf_settings['css']['css_exponent_grid'], dtype='float32'),
        ]
        ext_custom_bounds = [
            (prf_settings['css']['css_exponent_bound']),  # css exponent 
            ]
    # Combine the bounds 
    # first create the standard bounds
    standard_bounds = [
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
        (1e-1, max_eccentricity*3),                             # prf size bounds
        (prf_settings['prf_ampl']),                             # prf amplitude
        (prf_settings['bold_bsl']),                             # bold baseline (fixed)
    ]    
    # & the hrf bounds. these will be overwritten later by the vx wise hrf parameters
    # ( inherited from previous fits)
    hrf_bounds = [
        (prf_settings['hrf']['deriv_bound']),                   # hrf_1 bound
        (prf_settings['hrf']['disp_bound']),                    # hrf_2 bound
    ]
    ext_bounds = standard_bounds.copy() + ext_custom_bounds.copy() + hrf_bounds.copy()

    # Make sure we don't accidentally save gf stuff
    gf = []
    # ************************************************************************


    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IF NOT DONE - DO GRID FIT
    grid_ext = dag_find_file_in_folder([out, model, 'grid'], output_dir, return_msg=None)
    if grid_ext is None:
        print('Not done grid fit - doing that now')
        g_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f'Starting grid {g_start_time}')
        start = time.time()
        #        
        gf_ext.grid_fit(
            *ext_grids,
            verbose=True,
            n_batches=prf_settings['nr_jobs'],
            rsq_threshold=prf_settings['rsq_threshold'],
            fixed_grid_baseline=prf_settings['fixed_grid_baseline'],
            grid_bounds=ext_grid_bounds,
        )

        # Fiter for nans
        gf_ext.gridsearch_params = dag_filter_for_nans(gf_ext.gridsearch_params)
        g_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        elapsed = (time.time() - start)
        
        # Stuff to print:         
        print(f'Finished grid {g_end_time}')
        print(f'Took {timedelta(seconds=elapsed)}')
        # Save everything as a pickle...
        # Put them in the correct format to save
        grid_pkl_file = opj(output_dir, f'{out}_stage-grid_desc-prf_params.pkl')
        if (zero_pad) or (ignore_mask):
            grid_pars_to_save = gf_ext.gridsearch_params
        else:
            grid_pars_to_save = process_prfpy_out(gf_ext.gridsearch_params, roi_mask)         
        grid_dict = {}
        grid_dict['pars'] = grid_pars_to_save
        grid_dict['settings'] = prf_settings
        grid_dict['start_time'] = g_start_time
        grid_dict['end_time'] = g_end_time
        f = open(grid_pkl_file, "wb")
        pickle.dump(grid_dict, f)
        f.close()

    else:
        print('Loading old grid parameters')
        g_params = load_params_generic(grid_ext)
        # Apply the mask 
        if (not zero_pad) and (not ignore_mask):
            g_params = g_params[roi_mask,:]
        gf_ext.gridsearch_params = g_params        
    # Stuff to print:         
    vx_gt_rsq_th = gf_ext.gridsearch_params[:,-1]>prf_settings['rsq_threshold']
    nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
    mean_vx_gt_rsq_th = np.mean(gf_ext.gridsearch_params[vx_gt_rsq_th,-1])
    print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')
    # ************************************************************************



    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DO ITERATIVE FIT
    iter_check = dag_find_file_in_folder([out, model, 'iter', constraints], output_dir, return_msg=None)
    if (iter_check is not None) and (not overwrite):
        print(f'Already done {iter_check}')
        sys.exit()        

    prf_settings['ext_bounds'] = ext_bounds
    # model_idx = prfpy_params_dict()[model]
    # # Need to fix HRF, using HRF bounds
    # if zero_pad:
    #     num_vx_for_bounds = num_vx
    # else:
    #     num_vx_for_bounds = num_vx_in_roi

    # model_vx_bounds = make_vx_wise_bounds(
    #     num_vx_for_bounds, ext_bounds, model=model, 
    #     fix_param_dict = {
    #         'hrf_deriv' : gf_ext.gridsearch_params[:,model_idx['hrf_deriv']],
    #         'hrf_disp' : gf_ext.gridsearch_params[:,model_idx['hrf_disp']],
    #     })
    model_vx_bounds = ext_bounds
    
    # Constraints determines which scipy fitter is used
    # -> can also be used to make certain parameters interdependent (e.g. size depening on eccentricity... not normally done)
    if prf_settings['constraints']=='tc':
        n_constraints = []   # uses trust-constraint (slower, but moves further from grid
    elif prf_settings['constraints']=='bgfs':
        n_constraints = None # uses l-BFGS (which is faster)

    i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'Starting iter {i_start_time}, constraints = {n_constraints}')
    start = time.time()

    gf_ext.iterative_fit(
        rsq_threshold=prf_settings['rsq_threshold'],    # Minimum variance explained. Puts a lower bound on the quality of PRF fits. Any fits worse than this are thrown away...     
        verbose=False,
        bounds=model_vx_bounds,       # Bounds (on parameters)
        constraints=n_constraints, # Constraints
        xtol=float(prf_settings['xtol']),     # float, passed to fitting routine numerical tolerance on x
        ftol=float(prf_settings['ftol']),     # float, passed to fitting routine numerical tolerance on function
        )

    # Fiter for nans
    gf_ext.iterative_search_params = dag_filter_for_nans(gf_ext.iterative_search_params)    
    i_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'End iter {i_end_time}')           
    elapsed = (time.time() - start)
    
    # Stuff to print:         
    print(f'Finished iter {i_end_time}')
    print(f'Took {timedelta(seconds=elapsed)}')
    vx_gt_rsq_th = gf_ext.iterative_search_params[:,-1]>prf_settings['rsq_threshold']
    nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
    mean_vx_gt_rsq_th = np.mean(gf_ext.iterative_search_params[vx_gt_rsq_th,-1]) 
    print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')
    
    # CREATE PREDICTIONS
    preds = gg_ext.return_prediction(*list(gf_ext.iterative_search_params[:,:-1].T))
    preds = dag_filter_for_nans(preds)    
    # *************************************************************    
    
    
    # Save everything as a pickle...
    if zero_pad:
        iter_pars_to_save = gf_ext.iterative_search_params
        preds_to_save = preds
    else:
        iter_pars_to_save = process_prfpy_out(gf_ext.iterative_search_params, roi_mask)
        preds_to_save = process_prfpy_out(preds, roi_mask)    
    
    iter_pkl_file = opj(output_dir, f'{out}_stage-iter_constr-{constraints}_desc-prf_params.pkl')
    iter_dict = {}
    iter_dict['pars'] = iter_pars_to_save
    iter_dict['settings'] = prf_settings
    iter_dict['preds'] = preds_to_save
    iter_dict['start_time'] = i_start_time
    iter_dict['end_time'] = i_end_time
    iter_dict['prfpy_model'] = gg_ext
    
    from figure_finder.utils import get_running_path, get_running_code_string
    iter_dict['running_code_string'] = get_running_code_string(get_running_path())
    # Dump running code 
    run_code_file = iter_pkl_file.replace('prf_params.pkl', 'running_code.py')
    with open(run_code_file, 'w') as f:
        f.write(iter_dict['running_code_string'])
    
    # Also dump the settings as a separate yaml file for ease of reading 
    settings_file = iter_pkl_file.replace('prf_params.pkl', 'settings.yml')
    with open(settings_file, 'w') as f:
        yaml.dump(prf_settings, f)

    # Dump everything!!! into pickle
    f = open(iter_pkl_file, "wb")
    pickle.dump(iter_dict, f)
    f.close()

    print('DONE!!!')




if __name__ == "__main__":
    main(sys.argv[1:])
