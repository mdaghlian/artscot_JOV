#!/usr/bin/env python

'''
********* S1 *********
Percent signal change the outputs from pybest
Start jobs
'''

import os
import sys
opj = os.path.join
sys.path.insert(0, os.path.dirname(__file__))
from paths import prf_out, fs_dir, input_dir

prf_dir = prf_out

sub_list = ['sub-01', ] #'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
task_list = ['AS0', 'AS1', 'AS2']

ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
    # ************ LOOP THROUGH TASKS ***************
    for task in task_list:
        script_path = opj(os.path.dirname(__file__),'s1_psc.py')
        script_args = f"--sub {sub} --task {task} --prf_out {prf_out} --input_dir {input_dir} --fs_dir {fs_dir}"
        os.system(f'python {script_path} {script_args}')
