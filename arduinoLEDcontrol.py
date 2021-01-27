import serial, time
from random import randint
from threading import Thread

def get_arduino_c_p():
    arduino_c_p = {
    'polymerization_LED':'L',
    'polymerization_LED_status':'OFF'
    }
    return arduino_c_p

# TODO: This probably does not need to be a separate thread.


class ArduinoLEDControlThread(Thread):
    '''
    Thread which controls the arduinos output. Turns on/off the polymerization LED
    '''
    def __init__(self, threadID, name, c_p, sleep_time=0.01, port = 'com4'):
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.sleep_time = sleep_time
        self.ArduinoUnoSerial = serial.Serial(port, 9600)
        self.last_write = False

    def run(self):
        # TODO add radio buttons for different led on times
        while self.c_p['program_running']:
            # This function is made significantly faster by this check.
            if not self.last_write == self.c_p['polymerization_LED']:
                self.last_write = self.c_p['polymerization_LED']
                message = self.last_write.encode('utf-8')
                self.ArduinoUnoSerial.write(message)

                print(self.ArduinoUnoSerial.readline())
                if self.c_p['polymerization_LED'] != 'H' and self.c_p['polymerization_LED'] != 'L':
                    self.c_p['polymerization_LED'] = 'L'
            time.sleep(self.sleep_time)
        self.ArduinoUnoSerial.write(b'L')
        self.ArduinoUnoSerial.close()
        #self.__del__()
