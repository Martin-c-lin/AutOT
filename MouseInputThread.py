from pynput.mouse import Listener
from threading import Thread

class mouseInputThread(Thread):
    '''
    Thread for getting mouse input. Used to read scroll-wheel and thus
    lower/raise the sample.
    '''
    def __init__(self, threadID, name, c_p, stepper_step_distance=0.0005,
            piezo_step_distance=0.02, sleep_time=0.04):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.setDaemon(True)
        self.c_p = c_p
        self.piezo_step_distance = piezo_step_distance
        self.stepper_step_distance = stepper_step_distance
        self.sleep_time = sleep_time

    def on_move(self, x, y):
        """
        Used to check if the program is still running.
        """
        return self.c_p['program_running']

    def on_scroll(self, x, y, dx, dy):
        '''
        Moves the stage up/down depending on scroll. Will automagically shift
        between using steppers and piezos depending on which are activated.
        '''
        if self.c_p['scroll_for_z']:
            if self.c_p['stage_piezos'] and self.c_p['piezos_activated'].get():
                if dy<0:
                    self.c_p['piezo_target_position'][2] = self.c_p['piezo_current_position'][2] - self.piezo_step_distance
                else:
                    self.c_p['piezo_target_position'][2] = self.c_p['piezo_current_position'][2] + self.piezo_step_distance
            elif self.c_p['stepper_activated'].get():
                if dy<0:
                    self.c_p['stepper_target_position'][2] = self.c_p['stepper_current_position'][2] - self.stepper_step_distance
                else:
                    self.c_p['stepper_target_position'][2] = self.c_p['stepper_current_position'][2] + self.stepper_step_distance

        return self.c_p['program_running']

    def run(self):
        # TODO This function can only stop when the mouse is mooved/scrolled
        with Listener(on_move=self.on_move, on_click=None, on_scroll=self.on_scroll) as listener:
            listener.join()
