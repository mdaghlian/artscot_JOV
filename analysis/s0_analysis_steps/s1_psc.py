#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import sys
import os
opj = os.path.join
import numpy as np

from dpu_mini.utils import *
from dpu_mini.stats import *
from dpu_mini.fs_tools import dag_load_nverts

def main(argv):
    """
    ---------------------------
    Get percent signal change of task time series

    Args:
        -s|--sub        subject ID (e.g. 01 or sub-01)
        -t|--task       task name (AS0, AS1, AS2)
        --prf_out       full path to output directory (e.g. /path/to/prf_for_pub)
        --input_dir     full path to directory containing denoised bold .npy files
                        (pybest unzscored output, organised as <input_dir>/<sub>/<ses>/)
        --fs_dir        full path to FreeSurfer subjects directory
    ---------------------------
    """
    prf_out     = None
    input_dir   = None
    fs_dir      = None
    sub         = None
    task        = None

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

    for i, arg in enumerate(argv):
        if arg in ('-s', '--sub'):
            sub = dag_hyphen_parse('sub', argv[i+1])
        elif arg in ('-t', '--task'):
            task = dag_hyphen_parse('task', argv[i+1])
        elif arg == '--prf_out':
            prf_out = argv[i+1]
        elif arg == '--input_dir':
            input_dir = argv[i+1]
        elif arg == '--fs_dir':
            fs_dir = argv[i+1]
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit(2)

    if prf_out is None:
        sys.exit('Error: --prf_out is required')
    if input_dir is None:
        sys.exit('Error: --input_dir is required')
    if fs_dir is None:
        sys.exit('Error: --fs_dir is required')

    prf_dir    = prf_out
    output_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take the output from pybest folder
    inputdir    = opj(input_dir, sub, ses, 'unzscored')
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
    nvx = dag_load_nverts(sub=sub, fs_dir=fs_dir)

    assert sum(nvx) == mean_lr_data.shape[0]

    # *** Calculate the mean epi (average across runs & time) *** 
    # -> useful for checking for veins and signal dropout later...
    lr_mean_epi = np.mean(mean_lr_data, axis=1)
    # -> save it
    mean_epi_file = opj(output_dir, f'{out}_hemi-LR_mean_epi.npy')
    np.save(mean_epi_file, lr_mean_epi)

    # -> save the averaged runs in PSC [combined] (.npy format)
    # ** percent signal change **
    lr_data_psc = dag_psc(
        ts_in=mean_lr_data, 
        baseline_pt=baseline_pt,
        )

    # sanity check that the mean is 0
    print('Sanity check...')
    print(np.mean(lr_data_psc, axis=1))
    print(np.mean(mean_lr_data, axis=1).shape)
    
    lr_out_psc_file = opj(output_dir, f'{out}_hemi-LR_desc-avg_bold.npy')
    np.save(lr_out_psc_file, lr_data_psc)
    print(f'Saved {lr_out_psc_file}')        







if __name__ == "__main__":
    main(sys.argv[1:])    