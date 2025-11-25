import numpy as np
import scipy.io
# import linescanning.utils as lsutils
import nibabel as nb
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
source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'
code_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/code/artscot_JOV'
default_prf_dir = opj(derivatives_dir, 'prf_for_pub')
default_ses = 'ses-1'

def get_yml_settings_path(yml_name='s0_prf_analysis.yml'):
    yml_folder_path = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/code/artscot_JOV/analysis/s0_analysis_steps/'
    yml_path = opj(yml_folder_path, yml_name) 
    return yml_path

def load_data_tc(sub, task_list, ses=default_ses, look_in=default_prf_dir, do_demo=False, n_timepts=225):
    '''
    Loads real data
    '''
    if isinstance(task_list, str):
        task_list = [task_list]

    data_tc  = {}
    this_dir = opj(look_in, sub, ses)
    for task in task_list:
        # data_tc_path = dag_find_file_in_folder([sub, ses, dag_hyphen_parse('task', task), 'hemi-LR', 'desc-avg_bold', '.npy'], this_dir)
        try:
            data_tc_path = dag_find_file_in_folder([sub, ses, dag_hyphen_parse('task', task), 'hemi-LR', '.npy'], this_dir, exclude=['correlation', 'mean_epi'])
        except:
            data_tc_path = dag_find_file_in_folder([sub, ses, dag_hyphen_parse('task', task), 'hemi-lr', '.npy'], this_dir, exclude=['correlation', 'mean_epi'])
        data_tc[task] = set_tc_shape(np.load(data_tc_path), n_timepts=n_timepts)
        if do_demo:
            data_tc[task] = data_tc[task][0:100,:]
    return data_tc

def load_data_prf(sub, task_list, model_list, var_to_load='pars', roi_fit='all', fit_stage='iter', ses=default_ses, look_in=default_prf_dir, **kwargs):
    '''
    Load PRF model fits on * DATA *  (pkl file)
    output a dict 
        prf_vars[task][model]
    Default loads 'pars' (prf params)
    Can also specify settings or preds 
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
            pkl_file = open(prf_vars_path,'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()     
            if 'pred' in var_to_load:
                prf_vars[task][model] = set_tc_shape(pkl_data[var_to_load])
            else:
                prf_vars[task][model] = pkl_data[var_to_load]

    return prf_vars  

def get_number_of_vx(sub):
    num_vx = np.sum(load_nverts(sub))
    return num_vx

def load_nverts(sub):
    n_verts = []
    for i in ['lh', 'rh']:
        surf = opj(derivatives_dir, 'freesurfer', sub, 'surf', f'{i}.white')
        verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
        n_verts.append(verts)
    return n_verts

def get_roi(sub, label, **kwargs):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - conjunctive statements (not)
    '''
    # roi = label
    roi_idx = dag_load_roi(sub=sub, roi=label, fs_dir=opj(derivatives_dir, 'freesurfer'), **kwargs)
    return roi_idx

def get_design_matrix_npy(task_list, prf_dir=[]):

    if not isinstance(task_list, list):
        task_list = [task_list]    
    dm_npy  = {}    
    for task in task_list:
        dm_path = dag_find_file_in_folder(['design', task], opj(code_dir, 'analysis', 'artscot_JOV'))        
        dm_npy[task] = scipy.io.loadmat(dm_path)['stim']
    return dm_npy

def get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir,cut_vols=5):
    if not isinstance(task_list, list):
        task_list = [task_list]
    dm_npy = get_design_matrix_npy(task_list, prf_dir=prf_dir)
    model_list = 'gauss'     # stimulus settings are the same for both norm & gauss models  (so only use gauss)     
    fit_settings = load_data_prf(sub, task_list, model_list, var_to_load='settings', roi_fit='all', ses=default_ses, look_in=default_prf_dir)
    prfpy_stim = {}
    for task in task_list:
        prfpy_stim[task] = PRFStimulus2D(
            screen_size_cm=fit_settings[task][model_list]['screen_size_cm'],
            screen_distance_cm=fit_settings[task][model_list]['screen_distance_cm'],
            design_matrix=dm_npy[task][:,:,cut_vols:], 
            axis=0,
            TR=fit_settings[task][model_list]['TR']
            )    
    return prfpy_stim

def get_exp_settings(sub, task_list, run='run-1', source_data_dir=source_data_dir):    
    exp_settings = {}
    for task in task_list:
        ses='ses-1'
        this_dir = opj(source_data_dir, sub, ses)

        log_dir = dag_find_file_in_folder([sub, ses, task, run, 'Log'], this_dir)
        settings_path = dag_find_file_in_folder([sub, ses, task, run, 'expsettings.yml'], log_dir)
        with open(settings_path) as f:
            this_settings = yaml.safe_load(f)
        exp_settings[task] = this_settings 
    return exp_settings

def get_scot_centre(scot):
    exptools_ssize = np.degrees(2 *np.arctan((39.3/2)/210))
    fitting_ssize = np.degrees(2 *np.arctan((39.3/2)/196))                
    conversion_factor = fitting_ssize / exptools_ssize

    scot_coords = {}

    scot_coords['AS1'] = {
        'scotoma_centre' : [0.8284* conversion_factor,0.8284* conversion_factor] ,
        'scotoma_radius' : 0.8284* conversion_factor,
        }
    scot_coords['AS2'] = {
        'scotoma_centre' : [0* conversion_factor,0* conversion_factor] ,
        'scotoma_radius' : 2* conversion_factor,
        }        
    if 'AS1' in scot:
        return scot_coords['AS1']
    elif 'AS2' in scot:
        return scot_coords['AS2']

def get_scotoma_info(sub):    
    scotoma_info = {}
    # First get the info which is the same for all tasks
    # - so just take task AS0, model gauss
    task_list = 'task-AS0'    
    prfpy_stim = get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir)[task_list]
    aperture_rad = prfpy_stim.screen_size_degrees / 2    
    n_pix = prfpy_stim.design_matrix.shape[0]
    # Also create grid of coordinates
    x_deg = np.tile(np.linspace(-aperture_rad, aperture_rad, n_pix), (n_pix,1))
    y_deg = np.tile(np.linspace(-aperture_rad, aperture_rad, n_pix), (n_pix,1)).T
    grid = {
        "x_deg": x_deg,
        "y_deg": y_deg
        }
    # Now get task specific info, and sort out the dodgy stuff...
    scotoma_info['task-AS0'] = {
        'scotoma_centre' : [],
        'scotoma_radius' : [],
        'aperture_rad' : aperture_rad,
        'n_pix' : n_pix,
        'grid' : grid,
        'name' :'task-AS0'
    }

    scotoma_info['task-2R'] = {
        'scotoma_centre' : [],
        'scotoma_radius' : [],
        'aperture_rad' : aperture_rad,
        'n_pix' : n_pix,
        'grid' : grid,
    }    
    # Is the distance to the screen the same?... 
    # need to convert from exp (I accidentally set screen distance to be 210, not 196)
    exptools_ssize = np.degrees(2 *np.arctan((39.3/2)/210))
    fitting_ssize = np.degrees(2 *np.arctan((39.3/2)/196))                
    conversion_factor = fitting_ssize / exptools_ssize

    scotoma_info['task-AS1'] = {
        'scotoma_centre' : [0.8284* conversion_factor,0.8284* conversion_factor] ,
        'scotoma_radius' : 0.8284* conversion_factor,
        'aperture_rad' : aperture_rad,
        'n_pix' : n_pix,
        'grid' : grid,        
        'name' :'task-AS1'
    }
    scotoma_info['task-AS2'] = {
        'scotoma_centre' : [0* conversion_factor,0* conversion_factor] ,
        'scotoma_radius' : 2* conversion_factor,
        'aperture_rad' : aperture_rad,
        'n_pix' : n_pix,
        'grid' : grid,       
        'name' :'task-AS2' 
    }    
    return scotoma_info

# ********************
def load_params_generic(params_file, load_all=False, load_var=[]):
    """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

    if isinstance(params_file, str):
        if params_file.endswith('npy'):
            params = np.load(params_file)
        elif params_file.endswith('pkl'):
            with open(params_file, 'rb') as input:
                data = pickle.load(input)
            
            if len(load_var)==1:
                params = data[load_var[0]]
            elif len(load_var)>1:
                params = {}
                # Load the specified variables
                for this_var in load_var:
                    params[this_var] = data[this_var]
            elif load_all:
                params = {}
                for this_var in data.keys():
                    params[this_var] = data[this_var]
            else:
                params = data['pars']

    elif isinstance(params_file, np.ndarray):
        params = params_file.copy()
    elif isinstance(params_file, pd.DataFrame):
        dict_keys = list(params_file.keys())
        if not "hemi" in dict_keys:
            # got normalization parameter file
            params = np.array((params_file['x'][0],
                                params_file['y'][0],
                                params_file['prf_size'][0],
                                params_file['A'][0],
                                params_file['bold_bsl'][0],
                                params_file['B'][0],
                                params_file['C'][0],
                                params_file['surr_size'][0],
                                params_file['D'][0],
                                params_file['r2'][0]))
        else:
            raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized input type for '{params_file}'")

    return params







































# class ScotPrf1T1M(Prf1T1M):
#     '''
#     Used to hold parameters for 1 subject, 1 task & 1 model
#     & To return user specified masks 

#     __init__ will set up the useful information into 3 pandas data frames
#     >> including: all the parameters in the numpy arrays input model specific
#         gauss: "x", "y", "a_sigma", "a_val", "bold_baseline", "rsq"
#         norm : "x", "y", "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"
#     >> & eccentricit, polar angle, 
#         "ecc", "pol",
    
#     Functions:
#     return_vx_mask: returns a mask for voxels, specified by the user
#     return_th_param: returns the specified parameters, masked
#     '''
#     def __init__(self,sub, task, model, **kwargs):
#         '''
#         params_LE/X        np array, of all the parameters in the LE/X condition
#         model               str, model: e.g., gauss or norm
#         '''        
#         # [1] Load numpy values
#         roi_fit = kwargs.get('roi_fit', 'all')
#         fit_stage = kwargs.get('fit_stage', 'iter')
#         sim_or_data = kwargs.get('sim_or_data', 'data')
#         if sim_or_data=='data':
#             prf_params = load_data_prf(
#                 sub=sub, 
#                 task_list=task, 
#                 model_list=model, 
#                 roi_fit=roi_fit, 
#                 fit_stage=fit_stage)[task][model]
#             self.sim_model = 'data'
#             self.fit_model = model

#         elif sim_or_data=='sim':
#             prf_params = load_sim_prf(
#                 sub=sub, 
#                 task_list=task, 
#                 model_list=model, 
#                 roi_fit=roi_fit, 
#                 fit_stage=fit_stage)[task][model]
#             self.sim_model = model
#             self.fit_model = 'gauss'                        
            

#         super().__init__(
#             prf_params=prf_params, 
#             model = self.fit_model, 
#             task=task, **kwargs)
#         self.sub = sub
#         # Add distance to scotoma:
#         if not 'AS0' in task:
#             self.scotoma_info = get_scotoma_info(self.sub)[dag_hyphen_parse('task', task)]
#             self.pd_params['d2s_centre'] = get_d2_target(
#                 x=self.pd_params['x'], 
#                 y=self.pd_params['y'], 
#                 scotoma_info=self.scotoma_info, 
#                 edge_or_com='com')        
#         self.n_vox = np.sum(load_nverts(sub))
#         self.n_vox_L,self.n_vox_R = load_nverts(sub)
#         self.roi_fit = roi_fit
#         self.fit_stage = fit_stage

# class ScotPrf2T1M(Prf2T1M):
#     '''
#     Used to hold parameters for 1 subject, ***2 tasks***, for 1model
#     & To return user specified masks 

#     __init__ will set up the useful information into 3 pandas data frames
#     >> including: all the parameters in the numpy arrays input model specific
#         gauss: "x", "y", "a_sigma", "a_val", "bold_baseline", "rsq"
#         norm : "x", "y", "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"
#     >> & eccentricit, polar angle, 
#         "ecc", "pol",

#     >> In addition we will also add the mean and difference of the different tasks...
    
#     Functions:
#     return_vx_mask: returns a mask for voxels, specified by the user

#     '''
#     def __init__(self,sub, task_list, model, **kwargs):
#         '''
#         params_LE/X        np array, of all the parameters in the LE/X condition
#         model               str, model: e.g., gauss or norm
#         '''        
#         # [1] Load numpy values
#         roi_fit = kwargs.get('roi_fit', 'all')
#         fit_stage = kwargs.get('fit_stage', 'iter')
#         sim_or_data = kwargs.get('sim_or_data', 'data')
#         if sim_or_data == 'data':
#             prf_params = load_data_prf(
#                 sub=sub, 
#                 task_list=task_list, 
#                 model_list=model, 
#                 roi_fit=roi_fit, 
#                 fit_stage=fit_stage)
#             self.sim_model = 'data'
#             self.fit_model = model
#         elif sim_or_data == 'sim':        
#             prf_params = load_sim_prf(
#                 sub=sub, 
#                 task_list=task_list, 
#                 model_list=model, 
#                 roi_fit=roi_fit, 
#                 fit_stage=fit_stage)
#             self.sim_model = model
#             self.fit_model = 'gauss'                        
#         task1 = task_list[0]
#         task2 = task_list[1]

#         super().__init__(
#             prf_params1=prf_params[task1][model], 
#             prf_params2=prf_params[task2][model],             
#             model = self.fit_model, 
#             task1=task1,
#             task2=task2,
#             )
        
#         self.n_vox_L,self.n_vox_R = load_nverts(sub)
#         self.roi_fit = roi_fit
#         self.fit_stage = fit_stage

# class ScotPrf1T1Mx2(object):
#     def __init__(self,prf_obj1, prf_obj2, **kwargs):
#         self.task1 = prf_obj1.task
#         self.task2 = prf_obj2.task
#         self.model1 = prf_obj1.model
#         self.model2 = prf_obj2.model
#         self.model_labels1 = list(prf_obj1.pd_params.keys())
#         self.model_labels2 = list(prf_obj2.pd_params.keys())
#         self.id1 = kwargs.get('id1', 'id1')#f'{self.task1}_{self.model1}')
#         self.id2 = kwargs.get('id2', 'id2')#f'{self.task2}_{self.model2}')
#         self.n_vox = prf_obj1.n_vox 
#         self.pd_params = {}
#         self.pd_params[self.id1] = prf_obj1.pd_params
#         self.pd_params[self.id2] = prf_obj2.pd_params
        
#         # Make mean and difference:
#         comp_dict = {'mean':{}, 'diff':{}}
#         for i_label in self.model_labels1:
#             if i_label not in self.model_labels2:
#                 continue
#             comp_dict['mean'][i_label] = (self.pd_params[self.id1][i_label] +  self.pd_params[self.id2][i_label]) / 2
#             comp_dict['diff'][i_label] = self.pd_params[self.id2][i_label] -  self.pd_params[self.id1][i_label]
#         # some stuff needs to be recalculated: (because they don't scale linearly... e.g. polar angle)
#         # -> check which models...
#         xy_model_list = ['gauss', 'norm', 'css', 'dog']
#         both_with_xy =  (self.model1 in xy_model_list) & (self.model2 in xy_model_list)
#         both_norm = (self.model1=='norm') & (self.model2=='norm')
#         both_dog = (self.model1=='dog') & (self.model2=='dog')
#         both_csf = (self.model1=='CSF') & (self.model2=='CSF')
#         for i_comp in ['mean', 'diff']:
#             # Now add other interesting stuff:
#             if both_with_xy:
#                 # Ecc, pol
#                 comp_dict[i_comp]['ecc'], comp_dict[i_comp]['pol'] = dag_coord_convert(
#                     comp_dict[i_comp]['x'], comp_dict[i_comp]['y'], 'cart2pol'
#                 )
#             if both_norm or both_dog:
#                 # -> size ratio:
#                 comp_dict[i_comp]['size_ratio'] = comp_dict[i_comp]['size_2'] / comp_dict[i_comp]['size_1']
#                 comp_dict[i_comp]['amp_ratio']  = comp_dict[i_comp]['amp_1']  / comp_dict[i_comp]['amp_2']
#             if both_norm:
#                 comp_dict[i_comp]['bd_ratio'] = comp_dict[i_comp]['b_val'] / comp_dict[i_comp]['d_val']
#             if both_csf:
#                 comp_dict[i_comp]['log10_sf0']  = np.log10(comp_dict[i_comp]['sf0'])
#                 comp_dict[i_comp]['log10_maxC'] = np.log10(comp_dict[i_comp]['maxC'])
#                 comp_dict[i_comp]['sfmax'] = np.nan_to_num(
#                     10**(np.sqrt(comp_dict[i_comp]['log10_maxC'] / (comp_dict[i_comp]['width_r']**2)) + \
#                                                 comp_dict[i_comp]['log10_sf0']))            
#                 comp_dict[i_comp]['sfmax'][comp_dict[i_comp]['sfmax']>100] = 100 # MAX VALUE
#                 comp_dict[i_comp]['log10_sfmax'] = np.log10(comp_dict[i_comp]['sfmax'])            
#         # Enter into pd data frame
#         self.pd_params['mean'] = pd.DataFrame(comp_dict['mean'])
#         self.pd_params['diff'] = pd.DataFrame(comp_dict['diff'])
    
#     def return_vx_mask(self, th={}):
#         '''
#         return_vx_mask: returns a mask (boolean array) for voxels, specified by the user        
#         th keys must be split into 3 parts
#         'task-comparison-param' : value
#         e.g.: to exclude gauss fits with rsq less than 0.1
#         th = {'AS0_gauss-min-rsq': 0.1 } 
#         task        -> task1, task2, diff, mean, all. (all means apply the threshold to both task1, and task2)
#         comparison  -> min, max, bound
#         param       -> any of... (model dependent e.g., 'x', 'y', 'ecc'...)
        

#         '''        

#         # Start with EVRYTHING        
#         vx_mask = np.ones(self.n_vox, dtype=bool)
#         for th_key in th.keys():
#             th_key_str = str(th_key) # convert to string... 
#             if 'roi' in th_key_str:
#                 # Input roi specification...
#                 vx_mask &= th[th_key]
#                 continue # now next item in key

#             id, comp, p = th_key_str.split('-')
#             th_val = th[th_key]
#             if id=='all':
#                 # Apply to both task1 and task2:
#                 if p in self.model_labels1:
#                     vx_mask &= self.return_vx_mask({
#                         f'{self.id1}-{comp}-{p}': th_val
#                     })
                
#                 if p in self.model_labels2:
#                     vx_mask &= self.return_vx_mask({
#                         f'{self.id2}-{comp}-{p}': th_val
#                     })

#                 continue # now next item in th_key...
            
#             if comp=='min':
#                 vx_mask &= self.pd_params[id][p].gt(th_val)
#             elif comp=='max':
#                 vx_mask &= self.pd_params[id][p].lt(th_val)
#             elif comp=='bound':
#                 vx_mask &= self.pd_params[id][p].gt(th_val[0])
#                 vx_mask &= self.pd_params[id][p].lt(th_val[1])
#             else:
#                 sys.exit()
#         if not isinstance(vx_mask, np.ndarray):
#             vx_mask = vx_mask.to_numpy()
#         return vx_mask
    
#     def rapid_hist(self, id, param, th={'all-min-rsq':.1}, ax=None, **kwargs):
#         if ax==None:
#             ax = plt.axes()
#         vx_mask = self.return_vx_mask(th)        
#         label = kwargs.get('label', f'{id}-{param}')
#         kwargs['label'] = label
#         ax.hist(self.pd_params[id][param][vx_mask].to_numpy(), **kwargs)
#         ax.set_title(f'{id}-{param}')
#         dag_add_ax_basics(ax=ax, **kwargs)

#     def rapid_arrow(self, ax=None, th={'all-min-rsq':.1, 'all-max-ecc':5}, d2_task=None, **kwargs):
#         if ax==None:
#             ax = plt.gca()
#         vx_mask = self.return_vx_mask(th)        
#         kwargs['title'] = kwargs.get('title', f'{self.id1}-{self.id2}')
#         if d2_task is not None:
#             # [1] Get change in d2 scotoma 
#             q_cmap = mpl.cm.__dict__['bwr_r']
#             q_norm = mpl.colors.Normalize()
#             q_norm.vmin = -1
#             q_norm.vmax = 1
#             arrow_col = q_cmap(q_norm(self.pd_params['diff'][f'd2s_{d2_task}']))
#             kwargs['arrow_col'] = arrow_col[vx_mask,:]
#         # plt.figure()                
#         dag_arrow_plot(
#             ax, 
#             old_x=self.pd_params[self.id1]['x'][vx_mask], 
#             old_y=self.pd_params[self.id1]['y'][vx_mask], 
#             new_x=self.pd_params[self.id2]['x'][vx_mask], 
#             new_y=self.pd_params[self.id2]['y'][vx_mask], 
#             # arrow_col='angle', 
#             **kwargs
#             )
#     # def rapid_p_corr(self, px, py, th={'all-min-rsq':.1}, ax=None, **kwargs):
#     #     dot_col = kwargs.get('dot_col', 'k')
#     #     dot_alpha = kwargs.get('dot_alpha', None)
#     #     if ax==None:
#     #         ax = plt.axes()
#     #     vx_mask = self.return_vx_mask(th)
#     #     px_id, px_p = px.split('-')
#     #     py_id, py_p = py.split('-')
#     #     ax.scatter(
#     #         self.pd_params[px_id][px_p][vx_mask],
#     #         self.pd_params[py_id][py_p][vx_mask],
#     #         c = dot_col,
#     #         alpha=dot_alpha,
#     #     )
#     #     corr_xy = np.corrcoef(
#     #         self.pd_params[px_id][px_p][vx_mask],
#     #         self.pd_params[py_id][py_p][vx_mask],
#     #         )[0,1]
        
#     #     ax.set_title(f'corr {px}, {py} = {corr_xy:.3f}')
#     #     ax.set_xlabel(px)        
#     #     ax.set_ylabel(py)        
        
#     def rapid_p_corr(self, px, py, th={'all-min-rsq':.1}, ax=None, **kwargs):
#         # dot_col = kwargs.get('dot_col', 'k')
#         # dot_alpha = kwargs.get('dot_alpha', None)
#         if ax==None:
#             ax = plt.axes()
#         vx_mask = self.return_vx_mask(th)
#         px_id, px_p = px.split('-')
#         py_id, py_p = py.split('-')
#         # ax.scatter(
#         #     self.pd_params[px_id][px_p][vx_mask],
#         #     self.pd_params[py_id][py_p][vx_mask],
#         #     c = dot_col,
#         #     alpha=dot_alpha,
#         # )
#         # corr_xy = np.corrcoef(
#         #     self.pd_params[px_id][px_p][vx_mask],
#         #     self.pd_params[py_id][py_p][vx_mask],
#         #     )[0,1]
        
#         # ax.set_title(f'corr {px}, {py} = {corr_xy:.3f}')
#         dag_rapid_corr(
#             ax=ax,
#             X=self.pd_params[px_id][px_p][vx_mask],
#             Y=self.pd_params[py_id][py_p][vx_mask],
#             **kwargs
#         )                
#         ax.set_xlabel(px)        
#         ax.set_ylabel(py)        
#         # dag_add_ax_basics(ax=plt.gca(), **kwargs)                

#     def rapid_scatter(self, id, th=None, ax=None, dot_col='k', **kwargs):
#         if ax==None:
#             ax = plt.axes()
#         if th==None:
#             th = {f'{id}-min-rsq':.1, f'{id}-max-ecc':5}
#         vx_mask = self.return_vx_mask(th)
                
#         dag_visual_field_scatter(
#             ax=ax, 
#             dot_x=self.pd_params[id]['x'][vx_mask],
#             dot_y=self.pd_params[id]['y'][vx_mask],
#             dot_col = dot_col,
#             **kwargs
#         )
#     def drop_out_plot(self, ax, param, **kwargs):
#         roi_mask = kwargs.get('roi_mask', np.ones(self.n_vox))
#         ecc_th = kwargs.get("ecc_th", 5)        
#         rsq_th = kwargs.get("rsq_th", 0.1)
#         n_bins = kwargs.get("n_bins", 10)
#         drop_out_idx = self.return_vx_mask({
#             'all-max-ecc':ecc_th, 
#             f'{self.id1}-min-rsq': rsq_th, 
#             f'{self.id2}-max-rsq': rsq_th,
#             'roi':roi_mask})
#         drop_in_idx = self.return_vx_mask({
#             'all-max-ecc':ecc_th, 
#             f'{self.id2}-min-rsq': rsq_th, 
#             f'{self.id1}-max-rsq': rsq_th,
#             'roi':roi_mask})
        
#         d2_drout = self.pd_params[self.id1][param][drop_out_idx]
#         d2_drin = self.pd_params[self.id2][param][drop_in_idx]
#         ax.hist(d2_drout, alpha=0.5, bins=n_bins, label=f'{self.id1} (drop out)')
#         ax.hist(d2_drin, alpha=0.5, bins=n_bins, label=f'{self.id2} (drop in)')
#         ax.set_title(f'{param} of voxels w/ rsq above {rsq_th}')
#         ax.legend()

# def add_d2scotoma_to_obj(prf_obj):
    
#     # Check - is this a comparison thing?    
#     # if 'diff' in prf_obj.pd_params.keys():
#     #     is_comp = True
#     # else:
#     #     is_comp = False
#     for task in ['AS1', 'AS2']:
#         scot_info = get_scot_centre(task)
#         prf_obj.pd_params[f'd2s_{task}'] = dag_get_pos_change(
#             prf_obj.pd_params['x'],
#             prf_obj.pd_params['y'],
#             scot_info['scotoma_centre'][0],
#             scot_info['scotoma_centre'][1],
#         )
#     return prf_obj
    

