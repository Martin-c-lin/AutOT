from pynput.mouse import Listener
from threading import Thread

class mouseInputThread(Thread):
    '''
    Thread for getting mouse input. Used to read scroll-wheel and thus
    lower/raise the sample.
    '''
    def __init__(self, threadID, name, c_p, stepper_step_distance=0.0003,
            piezo_step_distance=0.1, piezo_z=50, sleep_time=0.02):
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
            # TODO piezo elevation is causin more problem than it's worth, remove it!
            if self.c_p['stage_piezos'] and self.c_p['piezos_activated'].get():
                if dy < 0:
                    self.c_p['piezo_elevation'] -= self.piezo_step_distance
                else:
                    self.c_p['piezo_elevation'] += self.piezo_step_distance

            elif self.c_p['using_stepper_motors'] and self.c_p['stepper_activated'].get():
                if dy < 0:
                    self.c_p['stepper_elevation'] -= self.stepper_step_distance
                else:
                    self.c_p['stepper_elevation'] += self.stepper_step_distance

            elif self.c_p['motor_z']:
                if dy < 0:
                    self.c_p['z_movement'] -= 10
                else:
                    self.c_p['z_movement'] += 10

        return self.c_p['program_running']

    def run(self):
        # TODO This function can only stop when the mouse is mooved/scrolled
        # Should probably let main interface handle the mouse positioning.
        with Listener(on_move=self.on_move, on_click=None, on_scroll=self.on_scroll) as listener:
            listener.join()
