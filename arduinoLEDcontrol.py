import serial, time
from random import randint
from threading import Thread

def get_arduino_c_p():
    arduino_c_p = {
    'polymerzation_LED':False,
    }
    return arduino_c_p

class ArduinoLEDControlThread(Thread):
    '''
    Thread which controls the arduinos output. Turns on/off the polymerization LED
    '''
    def __init__(self, threadID, name, c_p, sleep_time=0.05, port = 'com4'):
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.sleep_time = sleep_time
        self.ArduinoUnoSerial = serial.Serial(port, 9600)
        self.last_write = False

    # 
    # def __del__(self):
    #     # Turn off LED and close connection.
    #     self.ArduinoUnoSerial.close()

    def run(self):

        while self.c_p['program_running']:
            if self.c_p['polymerzation_LED']:
                self.ArduinoUnoSerial.write(b'H')
            else:
                self.ArduinoUnoSerial.write(b'L')
            # TODO make this thread listen to the change rather than just wait.
            # Check the speed of this function.
            time.sleep(self.sleep_time)
        self.ArduinoUnoSerial.write(b'L')
        self.ArduinoUnoSerial.close()
        #self.__del__()
