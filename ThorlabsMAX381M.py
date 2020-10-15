'''
Author: Martin Selin
'''

import numpy as np
from threading import Thread
import time


class XYZ_stepper_stage_motor(Thread):
    '''
    Class to help a main program of Automagic Trapping interface with a
    thorlabs max381 stage stepper motors.
    '''
    def __init__(self, threadID, name, c_p):
        threading.Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        # TODO initiate contact with stepper stage
        # We may need one thread per axis!
        self.c_p['starting_position_xyz'] = [1,2,3] # TODO fix to correct value

    def update_position(self):
        # Update c_p position
        pass

    def run(self):

        while self.c_p['running']:
            self.update_position()
            time.sleep(1)
        self.__del__()
    def __del__(self):
        self.c_p['running'] = False
        # TODO disconnect


class XYZ_piezo_stage_motor(Thread):
    '''
    Class to help a main program of Automagic Trapping interface with a
    thorlabs max381 stage piezo motors.
    '''
    def __init__(self, threadID, name, c_p):
        threading.Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        # TODO initiate contact with stepper stage
        # We may need one thread per axis!
        self.c_p['starting_position_xyz'] = [1,2,3] # TODO fix to correct value

    def update_position(self):
        # Update c_p position
        pass

    def run(self):

        while self.c_p['running']:
            self.update_position()
            time.sleep(1)
        self.__del__()
    def __del__(self):
        self.c_p['running'] = False
        # TODO disconnect
