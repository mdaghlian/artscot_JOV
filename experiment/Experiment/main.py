#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:04:44 2019

@author: marcoaqil
"""
import sys
import os
import getopt
from session import PRFSession
from datetime import datetime
datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main(argv):
    '''
    -s/--sub        subject e.g., 03
    -n/--ses        session e.g., 1
    -t/--task       task, AS0, AS1, AS2
    -r/--run        run, e.g., 1
    -e/--eye        True/False (0/1)

    e.g., for sub 999, session 9, task-AS0, run 1, do eyetracking
    >> python main.py -s 999 -n 9 -t AS0 -r 1 -e 1    
    '''

    try:
        opts = getopt.getopt(argv,"h:s:n:t:r:e:",["help", "sub=", "ses=", "task=", "run=", "eye="])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            subject = arg
        elif opt in ("-n", "--ses"):
            sess = arg
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-r", "--run"):
            run = arg
        elif opt in ("-e", "--eye"):
            eyetracker_on = int(arg)==1
            
    print(f"sub-{subject} ses-{sess} task-{task} run-{run} eye-{eyetracker_on}")

    # subject = sys.argv[1]
    # sess =  sys.argv[2]
    # 3 conditions: AS0, AS1, AS2
    # (no scotoma, scotoma at [0,0], scotoma at [2,0])
    # task = sys.argv[3]
    #in the full experiment we would do 3 runs?
    # run = sys.argv[4]
    
    # try:
    #     eye = sys.argv[5]
    # except:
    #     eye = 0

    # if eye == 0:
    #     eyetracker_on = False
    # else:
    #     eyetracker_on = True
    # eyetracker_on =False
    
    output_str= f"sub-{subject}_ses-{sess}_task-{task}_run-{run}"
    
    output_dir = './logs/'+output_str+'_Logs'

    if os.path.exists(output_dir):
        print("Warning: output directory already exists. Renaming to avoid overwriting.")
        output_dir = output_dir + datetime.now().strftime('%Y%m%d%H%M%S')
    
    settings_file='./expsettings_'+task+'.yml'

    ts = PRFSession(output_str=output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
    ts.run()

if __name__ == '__main__':
    main(sys.argv[1:])