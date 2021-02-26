import QD_positioning_0_1 as QD
from common_experiment_parameters import get_default_c_p

c_p = get_default_c_p()
c_p['camera_model'] = 'ThorlabsCam'
thread_list = []
d0x = -80e-6
d0y = -80e-6

# Define experiment to be run. Can also be read from a file nowadays.
xm1, ym1 = SLM.get_xm_ym_rect(nbr_rows=2, nbr_columns=1, d0x=d0x, d0y=d0y, dx=20e-6, dy=20e-6,)
experiment_schedule = [
{'xm':xm1, 'ym':ym1, 'use_LGO':[False],'target_experiment_z':1000,
'LGO_order':4,  'recording_duration':1000,'SLM_iterations':30,'activate_traps_one_by_one':False},
]

c_p['experiment_schedule'] = experiment_schedule
T_D = UserInterface(tkinter.Tk(), "Control display", c_p, thread_list)

sys.exit()
