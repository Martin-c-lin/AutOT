import QD_positioning_0_1 as QD
import tkinter
import SLM
import sys
from common_experiment_parameters import get_default_c_p, append_c_p, get_thread_activation_parameters


c_p = get_default_c_p()

# Create a empty list to put the threads in
thread_list = []
d0x = -80e-6
d0y = -80e-6

# Define experiment to be run. Can also be read from a file nowadays.
xm1, ym1 = SLM.get_xm_ym_rect(nbr_rows=2, nbr_columns=1, d0x=d0x, d0y=d0y, dx=20e-6, dy=20e-6,)
experiment_schedule = [
{'xm':xm1, 'ym':ym1, 'use_LGO':[False],'target_experiment_z':1000,
'LGO_order':4,  'recording_duration':1000,' SLM_iterations':30,'activate_traps_one_by_one':False},
]

c_p['experiment_schedule'] = experiment_schedule
append_c_p(c_p, get_thread_activation_parameters())

# Select which threads to turn on
c_p['motor_x'] = True
c_p['motor_y'] = True
c_p['motor_z'] = True

# Use max--- stage?
c_p['stage_stepper_x'] = False
c_p['stage_stepper_y'] = False
c_p['stage_stepper_z'] = False

# Use piezos of stage?
c_p['stage_piezo_x'] = False
c_p['stage_piezo_y'] = False
c_p['stage_piezo_z'] = False

# Temperature controller
c_p['temperature_controlls'] = True

# Automatic tracking?
c_p['QD_tracking'] = False

# Arduino for control?
c_p['arduino_LED'] = False

# SLM connected?
c_p['SLM'] = True

# Create a window into which to put the interface
T_D = QD.UserInterface(tkinter.Tk(), c_p, thread_list)
#print('Typical number of parameters: ', len(c_p))
sys.exit()
