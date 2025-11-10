#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S1a *********
Percent signal change the outputs from pybest

'''

import os
import sys
opj = os.path.join

derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'
prf_out = 'prf_no_hrf'
prf_dir = opj(derivatives_dir, prf_out)

sub_list = ['sub-07'] # , 'sub-03', 'sub-04', 'sub-06'] 
task_list = ['AS0', 'AS1', 'AS2']

nr_jobs = 1
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(this_dir): 
        os.makedirs(this_dir)    
    # ************ LOOP THROUGH TASKS ***************
    for task in task_list:
        prf_job_name = f'PSC{sub}_{task}'            
        # remove the 
        job=f"qsub -q long.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
        # job="python"
        script_path = opj(os.path.dirname(__file__),'s1a_psc.py')
        script_args = f"--sub {sub} --task {task} --prf_out {prf_out}"
        # print(f'{job} {script_path} {script_args}')
        os.system(f'{job} {script_path} {script_args}')
        # sys.exit()