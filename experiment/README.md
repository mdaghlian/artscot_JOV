# PRF_Experiment_A_Scotoma 
Essentially copied from https://github.com/marcoaqil/PRF_Experiment_Checkers (Marco Aqil)
Adapted (by M Daghlian) so that stimuli also include an option for a 'scotoma' -> i.e., a hole in screen, which the PRF bar will not stimulate

***
Repository for PRF mapping experiment stimulus

Requirements: psychopy and exptools2

**Usage**

Create setting files named expsettings_*Task*.yml within the Experiment folder. Change *Task* to your actual task name. Run the following line from within the Experient folder. 

- python main.py sub-*xxx* ses-*x* task-*NameTask* run-*x*

Subject SHOULD be specified according the the BIDS convention (sub-001, sub-002 and so on), Task MUST match one of the settings files in the Experiment folder, and Run SHOULD be an integer.

**Marcus's PRF mapping**

All tasks follow the regular PRF stimulus settings: 2 squares per checkerboard bar, Regular speed (20TR bar passes, aka 30 seconds with our standard sequence)
The Experiment folder contains 3 setting files. The Task names are:

- AS0: No scotoma (i.e., normal PRF mapping)
- AS1: Scotoma position 1 (small: r=0.83; x=0.83,y=0.83)
- AS2: Scotoma position 2 (large: r=2;    x=0,   y=0)

This is set up so that AS1 is entirely in the NE quadrant, inside AS2, and has a common border with AS2 

Note that the actual task (fixation dot color discrimination) is identical in all cases.

**Settings file** 

In the settings file you should specify the *operating system* that the code is running on, as "operating system: *your OS*" as 'mac', 'linux' or 'windows'
This is mainly important if you run the stimulus on a mac, as the size of the stimulus needs to be adjusted in that case.

You can change the *task parameters* in the settings file under "Task settings:"
- you can specify how much time you allow for the participant to respond that still counts as correct response (default is 0.8s), as "response interval: *your time*"
- you can specify the timing of the color switches (default is 3.5s), as "color switch interval: *your interval*"
Note: Make sure that the difference between two adjacent color switches is bigger than the time you give the participant to respond. 
The code adds a randomization of max. +1 or -1 to the color switch times, so e.g. in case of a color switch interval of 3.5, the two closest adjacent color switches will be 1.5s apart, well outside the response interval of 0.8s.

# Create order of conditions
import random 
task_list = ['AS0', 'AS1', 'AS2']
random.shuffle(task_list)
print(f'Run 1-> {task_list}')
random.shuffle(task_list)
print(f'Run 2-> {task_list}')


FOR SUB07 ...
Run 1-> ['AS1', 'AS0', 'AS2']
Run 2-> ['AS0', 'AS2', 'AS1']


FOR SUB07 ...
Run 1-> ['AS1', 'AS0', 'AS2']
Run 2-> ['AS0', 'AS2', 'AS1']
run 3,4 -> AS0,AS0



