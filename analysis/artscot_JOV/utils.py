import numpy as np
import sys
from prfpy_csenf.rf import *

def mask_time_series(ts, mask, ts_axis = 1, zero_pad = False):    
    '''mask_time_series    
    Mask certain voxel time series for later fitting. This is useful if you want to fit a subset of voxels, 
    to speed up fitting.

    Input:
    ----------
    ts          np.ndarray          time series, default is nvx x time
    mask        np.ndarray, bool    which vx to include (=True)
    ts_axis     int                 which axis is time
    zero_pad    bool                return the masked vx as flat time series    
        
    Output:
    ----------
    ts_out      np.ndarray          masked time series
    '''
    if zero_pad:
        # initialize empty array and only keep the timecourses from label; 
        # keeps the original dimensions for simplicity sake!     
        ts_out = np.zeros_like(ts)

        # insert timecourses 
        lbl_true = np.where(mask == True)[0]
        if ts_axis==0:
            ts_out[:,lbl_true] = ts[:,lbl_true]
        elif ts_axis==1:
            ts_out[lbl_true,:] = ts[lbl_true,:]
        else:
            print('Bad ts_axis...')
            sys.exit()
    else:
        # Don't pad everything...
        if ts_axis==0:
            ts_out = np.copy(ts[:,mask])        
        elif ts_axis==1:
            ts_out = np.copy(ts[mask,:])
        else:
            print('Bad ts_axis...')
            sys.exit()
    
    return ts_out

def dag_filter_for_nans(array):
    """
    filter out NaNs from an array
    Copied from JH linescanning toolbox
    """

    if np.isnan(array).any():
        return np.nan_to_num(array)
    else:
        return array

def process_prfpy_out(prfpy_out, mask=None):    
    '''process_prfpy_out
    Fit parameters can come out with nans, and are in the wrong shape
    If we masked certain timeseries, we want to go back and put in empty values for fitted parameters. 
    This is so that the shape of the vector is nice... (i.e., fits on the surface)
    '''
    if mask is None:
        mask = np.ones(prfpy_out.shape[0], dtype=bool)
    total_n_vx = mask.shape[0]
    n_vx_fit = prfpy_out.shape[0]
    n_pars = prfpy_out.shape[1]
    n_vx_in_mask = mask.sum()
    assert n_vx_fit==n_vx_in_mask
    
    filled_pars = np.zeros((total_n_vx, n_pars))

    filled_pars[mask,:] = dag_filter_for_nans(prfpy_out)

    return filled_pars
def set_tc_shape (tc_in, n_timepts = 225):
    '''set_tc_shape
    Force the timecourse to be n_units * n_time
    '''
    # *** ALWAYS n_units * n_time
    if tc_in.shape[0] == n_timepts:
        tc_out = tc_in.T
    else:
        tc_out = tc_in
    return tc_out

def get_d2_target(x,y, scotoma_info, edge_or_com='com'):
    if isinstance(scotoma_info['scotoma_centre'], list):
        # Are we doing it with refrence to COM of scotoma, or EDGE?
        if edge_or_com=='com':
            target_coords = np.tile(scotoma_info['scotoma_centre'], (x.shape[0], 1))
        elif edge_or_com=='edge':
            target_coords = get_nearest_scot_edge(x,y,scotoma_info)

        d2_target = np.sqrt((x-target_coords[:,0])**2+(y-target_coords[:,1])**2)
    else:
        d2_target = np.ones_like(x) * np.NaN
    return d2_target

def get_d2_target_change(x,y,new_x,new_y, scotoma_info, edge_or_com='com'):
    # Are we doing it with refrence to COM of scotoma, or EDGE?
    if edge_or_com=='com':
        target_coords = np.tile(scotoma_info['scotoma_centre'], (x.shape[0], 1))
    elif edge_or_com=='edge':
        target_coords = get_nearest_scot_edge(x,y,scotoma_info)

    d2_target_old = np.sqrt((x-target_coords[:,0])**2+(y-target_coords[:,1])**2)
    d2_target_new = np.sqrt((new_x-target_coords[:,0])**2+(new_y-target_coords[:,1])**2)
    change_in_d2_target = d2_target_new - d2_target_old
    return d2_target_old, d2_target_new, change_in_d2_target

def get_scot_mask(scotoma_info):
        d2_scot_centre = np.sqrt((scotoma_info["grid"]["x_deg"]-scotoma_info["scotoma_centre"][0])**2+(scotoma_info["grid"]["y_deg"]-scotoma_info["scotoma_centre"][1])**2) 
        as_mask = (d2_scot_centre>scotoma_info["scotoma_radius"])  * 1
        as_mask = is_xy_out_scotoma(scotoma_info["grid"]["x_deg"],scotoma_info["grid"]["y_deg"],scotoma_info)

        return as_mask

def get_nearest_scot_edge(x,y,scotoma_info):
        n_coords = x.shape[0]
        nearest_edge = np.zeros((x.shape[0],2)) # output coords of nearest edge

        # [1] Check if it is inside the scotoma
        i_outside = is_xy_out_scotoma(x,y,scotoma_info)

        # [2] find the nearest 0 pixel value in the as_mask
        # ~ get idx of all 0 value pixels in as_mask
        as_mask = get_scot_mask(scotoma_info)
        idx_0_pix_X,idx_0_pix_Y = np.where(as_mask==0)
        n_pix_0 = idx_0_pix_X.shape[0]
        # ~ convert to cartesian coords
        cart_0_pix_X = scotoma_info['grid']['x_deg'][idx_0_pix_X,idx_0_pix_Y] 
        cart_0_pix_Y = scotoma_info['grid']['y_deg'][idx_0_pix_X,idx_0_pix_Y] 
        # ~ find closest pt
        x_tiled = np.tile(x,(n_pix_0,1)) # tile coords to find closest pt simultaneously
        y_tiled = np.tile(y,(n_pix_0,1)) 
        d2_coords_0 = np.sqrt(
                (x_tiled - cart_0_pix_X[...,np.newaxis])**2 +\
                (y_tiled - cart_0_pix_Y[...,np.newaxis])**2
        )
        closest_idx_0 = np.argmin(d2_coords_0, 0)
        closest_x_0,closest_y_0 = cart_0_pix_X[closest_idx_0],cart_0_pix_Y[closest_idx_0]

        # [3] find the nearest 1 pixel value in the as_mask
        # ~ get idx of all 1 value pixels in as_mask
        idx_1_pix_X,idx_1_pix_Y = np.where(as_mask==1)
        n_pix_1 = idx_1_pix_X.shape[0]
        # ~ convert to cartesian coords
        cart_1_pix_X = scotoma_info['grid']['x_deg'][idx_1_pix_X,idx_1_pix_Y] 
        cart_1_pix_Y = scotoma_info['grid']['y_deg'][idx_1_pix_X,idx_1_pix_Y] 
        # ~ find closest pt
        x_tiled = np.tile(x,(n_pix_1,1)) # tile coords to find closest pt simultaneously
        y_tiled = np.tile(y,(n_pix_1,1)) 

        d2_coords_1 = np.sqrt(
                (x_tiled - cart_1_pix_X[...,np.newaxis])**2 +\
                (y_tiled - cart_1_pix_Y[...,np.newaxis])**2
        )
        closest_idx_1 = np.argmin(d2_coords_1, 0)
        closest_x_1,closest_y_1 = cart_1_pix_X[closest_idx_1],cart_1_pix_Y[closest_idx_1]
        
        nearest_edge[i_outside==1,0] = closest_x_0[i_outside==1]
        nearest_edge[i_outside==1,1] = closest_y_0[i_outside==1]
        nearest_edge[i_outside==0,0] = closest_x_1[i_outside==0]
        nearest_edge[i_outside==0,1] = closest_y_1[i_outside==0]
        

        return nearest_edge

def is_xy_out_scotoma(x,y,scotoma_info):
    if isinstance(scotoma_info['scotoma_centre'], list):
        d2_scot_centre = np.sqrt((x-scotoma_info["scotoma_centre"][0])**2+(y-scotoma_info["scotoma_centre"][1])**2)
        outside_scot = d2_scot_centre>scotoma_info["scotoma_radius"] * 1
    else:
        outside_scot = np.ones_like(x) * 1

    return outside_scot

def calculate_sup_idx(prf_obj, prfpy_stim, th=None):
    if th is None:
        th = {'min-rsq':.1}
    vx_mask = prf_obj.return_vx_mask(th)
    n_vox = prf_obj.n_vox
    n_pix = prfpy_stim.x_coordinates.shape[0]
    aperture = prfpy_stim.ecc_coordinates > 5
    prf = np.zeros((n_vox, n_pix, n_pix))                
    print(prfpy_stim.x_coordinates[...,np.newaxis].shape)
    # b = prf_obj.pd_params['amp_1'][vx_mask, np.newaxis,np.newaxis]*
    print(prf_obj.pd_params['x'][vx_mask].shape)
    print(prf_obj.pd_params['x'][vx_mask].shape)
    mu_x = prf_obj.pd_params['x'][vx_mask].to_numpy()
    mu_y = prf_obj.pd_params['y'][vx_mask].to_numpy()
    size_1 = prf_obj.pd_params['size_1'][vx_mask].to_numpy()
    size_2 = prf_obj.pd_params['size_2'][vx_mask].to_numpy()
    amp_1 = prf_obj.pd_params['amp_1'][vx_mask].to_numpy()
    amp_2 = prf_obj.pd_params['amp_2'][vx_mask].to_numpy()
    prf = amp_1[...,np.newaxis, np.newaxis] * np.rot90(
        gauss2D_iso_cart(
            x=prfpy_stim.x_coordinates[...,np.newaxis],
            y=prfpy_stim.y_coordinates[...,np.newaxis],
            mu=(mu_x, mu_y),
            sigma=(size_1),
            normalize_RFs=False).T,axes=(1,2))
    srf = amp_2[...,np.newaxis, np.newaxis] * np.rot90(
        gauss2D_iso_cart(
            x=prfpy_stim.x_coordinates[...,np.newaxis],
            y=prfpy_stim.y_coordinates[...,np.newaxis],
            mu=(mu_x, mu_y),
            sigma=(size_2),
            normalize_RFs=False).T,axes=(1,2))
    prf[:,aperture] = 0
    srf[:,aperture] = 0         

    sup_idx = np.zeros(n_vox)
    sup_idx[vx_mask] = prf.sum(axis=(1,2)) / srf.sum(axis=(1,2))    
    return sup_idx

