from pynput.mouse import Listener
from threading import Thread
from time import sleep

class mouseInputThread(Thread):

    def __init__(self, threadID, name, c_p, stepper_step_distance=0.0002,
            piezo_step_distance=0.02, sleep_time=0.01):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.setDaemon(True)
        self.c_p = c_p
        self.piezo_step_distance = piezo_step_distance
        self.stepper_step_distance = stepper_step_distance
        self.sleep_time = sleep_time


    def on_scroll(self, x, y, dx, dy):

        if self.c_p['scroll_for_z'].get():
            if self.c_p['stage_piezos']:
                self.c_p['piezo_target_pos'][2] += dy * self.piezo_step_distance
            else:
                self.c_p['stepper_target_position'][2] += dy * self.stepper_step_distance
        sleep(self.sleep_time)
        # print('Scrolled {0}'.format(
        #     (x, y, dx, dy)))
        # print('Piezos target posittion: ',self.c_p['piezo_target_pos'][2] )
        # if dx > 0:
        #     self.c_p['program_running'] = False
        return self.c_p['program_running']

    def run(self):
        with Listener(on_move=None, on_click=None, on_scroll=self.on_scroll) as listener:
            listener.join()
