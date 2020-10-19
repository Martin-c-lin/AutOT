import instruments
from threading import Thread

def get_shutter_c_p():
    #
    pass

class ShutterThread(Thread):

    def __init__(self, threadID, name, c_p, open_time=5, port='COM1'):

        self.open = False
        self.open_time = open_time
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.shutter = instruments.thorlabs.SC10.open_serial(port=port, baud=9600, vid=None, pid=None, serial_number=None, timeout=2, write_timeout=10)

    def set_open_time(self, duration):
        self.open_time = duration
        try:
            self.shutter.sendcmd(cmd='open='+str(duration))
        except:
            # Always get a irrelevant errormessage which is probably caused
            # by the package instruments being old. Not worth the effort to fix
            pass

    def open_shutter(self):
        try:
            self.shutter.sendcmd(cmd='ens')
            self.open = True
        except:
            pass

    def open_for_duration(self, duration):
        try:
            self.shutter.sendcmd(cmd='open='+str(duration))
        except:
            pass
        try:
            self.shutter.sendcmd(cmd='ens')
            self.open = True
        except:
            pass

    def is_open(self):
        # Function for quering if the shutter is open or not.
        # TODO should check this with a proper command.
        self.c_p['shutter_open'] = self.open
        return self.open

    def run(self):

        while self.c_p['running']:
            self.set_open_time(self.c_p['shutter_open_time'])
            if self.c_p['should_open']:
                self.open_shutter()
                self.c_p['should_open'] = False
                while self.is_open() and self.c_p['running']: # add possibility to force shut
                    time.sleep(0.1)
