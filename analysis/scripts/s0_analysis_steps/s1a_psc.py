#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import sys
import os
opj = os.path.join
import numpy as np

from dag_prf_utils.utils import *
from dag_prf_utils.stats import *

def main(argv):
    """
    ---------------------------    
    Get percent signal change of tasks time series

    Args:
        -s|--sub    <sub number>        number of subject's FreeSurfer directory from which you can 
                                        omit "sub-" (e.g.,for "sub-001", enter "001").
        -t|--task   <task name>         name of the experiment performed (e.g., "2R")
        --prf_out   <prf_out>           Where to put everything 
    ---------------------------
    """
    derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'    
    fs_dir = opj(derivatives_dir, 'freesurfer')
    sub         = None
    task        = None
    prf_out     = 'prf' 

    # Set parameters for psc
    ses         = 'ses-1'    
    file_ending = "desc-denoised_bold.npy"
    space       = "fsnative"
    cut_vols    = 5 # Skip first 5, screen change w/ OFF 
    original_run_length = 225 # Number of TRs
    run_length = original_run_length - cut_vols # 
    print('Removing first 5 TRs')
    print('Using starting and ending for baseline setting')
    baseline_pt = np.arange(cut_vols,15).tolist() + np.arange(225-10,225).tolist()
    baseline_pt = [i-cut_vols for i in baseline_pt]
    print(f'Baseline is {baseline_pt}')

    for i,arg in enumerate(argv):
        if arg in ('-s', '--sub'):
            sub = dag_hyphen_parse('sub', argv[i+1])
        elif arg in ('-t', '--task'):
            task = dag_hyphen_parse('task', argv[i+1])
        elif '--prf_out' in arg:
            prf_out = argv[i+1]                
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit(2)

    # MORE PATHS
    prf_dir = opj(derivatives_dir, prf_out)    
    output_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inputdir    = opj(derivatives_dir, 'pybest', sub, ses, 'unzscored')
    search_for = ["run-", task, file_ending]
    if space != None:
        search_for += [f"space-{space}"]

    # OUTPUT NAMING
    out = f"{sub}"
    if ses != None:
        out += f"_{ses}"    
    if task != None:
        out += f"_{task}"

    # FIND THE FILES
    lh_all_files = dag_find_file_in_folder([*search_for, 'hemi-L'], inputdir)
    rh_all_files = dag_find_file_in_folder([*search_for, 'hemi-R'], inputdir)

    print('ONLY DOING RUNS 1 & 2')
    lh_files = []    
    for ff in lh_all_files:
        if ('run-1' in ff) or ('run-2' in ff):
            lh_files.append(ff)
    rh_files = []    
    for ff in rh_all_files:
        if ('run-1' in ff) or ('run-2' in ff):
            rh_files.append(ff)

    # Should be the same length
    assert len(lh_files)==len(rh_files)

    # Sort them (by run number)
    lh_files.sort()
    rh_files.sort()

    # -> load them
    lh_data = []
    rh_data = []
    for i_file in range(len(lh_files)):
        # ASSUMING LH and RH have same naming...
        lh_run_data = np.load(lh_files[i_file]).T
        lh_data.append(lh_run_data[:,cut_vols:])

        rh_run_data = np.load(rh_files[i_file]).T
        rh_data.append(rh_run_data[:,cut_vols:])
        
    # -> average the runs...
    mean_lh_data = np.mean(np.stack(lh_data, axis=0), axis=0).astype(np.float32)
    mean_rh_data = np.mean(np.stack(rh_data, axis=0), axis=0).astype(np.float32)
    
    # -> mean across runs
    mean_lr_data = np.concatenate([mean_lh_data, mean_rh_data], axis=0)
    print('')
    print(mean_lr_data.shape)
    print('')
    # CHECK NUMBER OF VX
    nvx = dag_load_nverts(sub, fs_dir)

    assert sum(nvx) == mean_lr_data.shape[0]

    # *** Calculate the mean epi (average across runs & time) *** 
    # -> useful for checking for veins and signal dropout later...
    lr_mean_epi = np.mean(mean_lr_data, axis=1)
    # -> save it
    mean_epi_file = opj(output_dir, f'{out}_hemi-LR_mean_epi.npy')
    np.save(mean_epi_file, lr_mean_epi)

    # -> save the averaged runs in PSC [combined] (.npy format)
    # ** percent signal change **
    lr_data_psc = dag_dct_detrending(
        ts_au=mean_lr_data, 
        n_trend_to_remove=False,
        do_psc=True, 
        baseline_pt=baseline_pt,
        )

    # sanity check that the mean is 0
    print('Sanity check...')
    print(np.mean(lr_data_psc, axis=1))
    print(np.mean(mean_lr_data, axis=1).shape)

    lr_out_psc_file = opj(output_dir, f'{out}_hemi-LR_desc-avg_bold.npy')
    np.save(lr_out_psc_file, lr_data_psc)
    print(f'Saved {lr_out_psc_file}')        

    # *** Also calculate the noise ceiling ***
    # -> useful for seeing how good our fits are later...
    # -> Split into 2 halves (half the runs)
    n_runs = len(lh_files)
    if n_runs==1:
        print('Only 1 run, therefore cannot do noise ceiling')
        return
    n_runs_half = int(n_runs / 2)
    lh_data_half1 = np.mean(np.stack(lh_data[:n_runs_half], axis=0), axis=0)
    lh_data_half2 = np.mean(np.stack(lh_data[n_runs_half:], axis=0), axis=0)
    rh_data_half1 = np.mean(np.stack(rh_data[:n_runs_half], axis=0), axis=0)
    rh_data_half2 = np.mean(np.stack(rh_data[n_runs_half:], axis=0), axis=0)
    # -> combine them
    lr_half1_data = np.concatenate([lh_data_half1, rh_data_half1], axis=0)
    lr_half2_data = np.concatenate([lh_data_half2, rh_data_half2], axis=0)

    # -> convert to percent signal change 
    psc_lr_half1 = dag_dct_detrending(
        ts_au=lr_half1_data, 
        n_trend_to_remove=False,
        do_psc=True, 
        baseline_pt=baseline_pt,)
    psc_lr_half2 = dag_dct_detrending(
        ts_au=lr_half2_data,
        n_trend_to_remove=False,
        do_psc=True, 
        baseline_pt=baseline_pt,)        

    # -> calculate the correlation between runs (for each voxel)
    # should be an array of size [n_vx]
    # only do it for vx that are not std = 0 
    std_mask = psc_lr_half1.std(axis=1) != 0
    std_mask &= psc_lr_half2.std(axis=1) != 0
    std_idx = np.where(std_mask)[0]
    run_correlation = np.zeros(psc_lr_half1.shape[0])

    i_count = 0
    for i in std_idx:
        run_correlation[i] = np.corrcoef(psc_lr_half1[i, :], psc_lr_half2[i, :])[0,1]
        i_count += 1
        if i_count % 5000 == 0:
            print(f'Calculating correlation for voxel {i_count} of {psc_lr_half1.shape[0]}')
    # run_correlation[run_correlation<0] = 0
    # run_correlation = run_correlation**2 # R squared
    print(f'Run correlation: {run_correlation}')
    # -> save it
    run_correlation_file = opj(output_dir, f'{out}_hemi-LR_run_correlation.npy')
    np.save(run_correlation_file, run_correlation)
    print(f'Saved {run_correlation_file}')            







if __name__ == "__main__":
    main(sys.argv[1:])    