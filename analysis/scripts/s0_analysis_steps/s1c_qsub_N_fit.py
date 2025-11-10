#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S1c *********
Fit the normalization model 
'''

import os
import sys
opj = os.path.join
from dag_prf_utils.utils import dag_get_cores_used
import time

derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'
prf_out = 'prf_no_hrf'
prf_dir = opj(derivatives_dir, prf_out)

sub_list = ['sub-07'] # , 'sub-06']
task_list = ['AS0', 'AS1' ,'AS2']

roi_fit = 'all'
constraint = '--tc'
nr_jobs = 30
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:
        prf_job_name = f'N-{task}-{sub}-{roi_fit}'            
        job=f"qsub -q long.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
        # job="python"
        script_path = opj(os.path.dirname(__file__),'s1c_N_fit.py')
        script_args = f"--sub {sub} --model norm --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out}"
        n_cores = dag_get_cores_used()
        while n_cores>=60:
            # pause for 
            print('sleep')
            time.sleep(50)
            n_cores = dag_get_cores_used()        
        os.system(f'{job} {script_path} {script_args}')
        # sys.exit()