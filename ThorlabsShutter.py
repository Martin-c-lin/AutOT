import instruments
from threading import Thread
import time

def get_shutter_c_p():
    shutter_c_p = {
    'shutter_connected':False, # True if shutter is connected
    'shutter_open':False, # Current state of shutter
    'should_shutter_open':False, # True if another open command should be sent
    'shutter_open_time':5000, # Time which the shutter should be opened for after
    # recieving an open command.
    }
    return shutter_c_p

class ShutterThread(Thread):
    # TODO add possibility to force shut
    def __init__(self, threadID, name, c_p, port='COM1'):
        Thread.__init__(self)
        self.name = name
        self.threadID = threadID
        self.open = False
        self.c_p = c_p
        print('Trying to connect shutter')
        self.shutter = instruments.thorlabs.SC10.open_serial(port=port, baud=9600, vid=None, pid=None, serial_number=None, timeout=2, write_timeout=10)
        self.c_p['shutter_connected'] = True
        self.setDaemon(True)

    def open_shutter(self):
        try:
            self.shutter.sendcmd(cmd='ens')
            self.is_open()
        except:
            pass

    def open_for_duration(self, duration):
        try:
            self.shutter.sendcmd(cmd='open='+str(duration))
        except:
            pass
        try:
            self.shutter.sendcmd(cmd='ens')
            self.is_open()
        except:
            pass

    def is_open(self):
        # Function for quering if the shutter is open or not.
        # Returns True if shutter is open(enabled) otherwise False.
        # Automatically updates c_p.
        try:
            # OBS in order for query to work
            # C:\Users\Feynman\Anaconda3\Lib\site-packages\instruments\abstract_instruments\instrument.py
            # lines 139-143 should be commented out, incorrectly gives an error!
            self.c_p['shutter_open'] = True if self.shutter.query(cmd='ens?') == "1" else False
        except:
            pass
        return self.c_p['shutter_open']

    def run(self):
        print('Shutter thread started')
        while self.c_p['program_running']:
            self.is_open()
            if self.c_p['should_shutter_open']:
                print('Opening shutter')
                self.open_for_duration(self.c_p['shutter_open_time'])
                self.c_p['should_shutter_open'] = False

            time.sleep(0.2)
