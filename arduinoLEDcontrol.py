import serial, time
from threading import Thread

def get_arduino_c_p():
    arduino_c_p = {
    'polymerization_LED':'L',
    'polymerization_LED_status':'OFF',
    'background_illumination': False,
    }
    return arduino_c_p

# TODO: This probably does not need to be a separate thread.

def toggle_BG_shutter(c_p):

    c_p['background_illumination'] = not c_p['background_illumination']
    if c_p['background_illumination']:
        # Open shutter
        c_p['polymerization_LED'] = 'O'
        c_p['exposure_time'] /= 7
    else:
        # Close shutter
        c_p['polymerization_LED'] = 'C'
        c_p['exposure_time'] *= 7
    c_p['new_settings_camera'] = True

class ArduinoLEDControlThread(Thread):
    '''
    Thread which controls the arduinos output. Turns on/off the polymerization LED
    '''
    def __init__(self, threadID, name, c_p, sleep_time=0.01, port = 'com5'):
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.sleep_time = sleep_time
        self.ArduinoUnoSerial = serial.Serial(port, 9600)
        self.last_write = False

    def run(self):

        # The red LED should be blocked by default.
        self.ArduinoUnoSerial.write(b'C')
        while self.c_p['program_running']:
            # This function is made significantly faster by this check.
            if not self.last_write == self.c_p['polymerization_LED']:
                self.last_write = self.c_p['polymerization_LED']
                message = self.last_write.encode('utf-8')
                self.ArduinoUnoSerial.write(message)

                #print(self.ArduinoUnoSerial.readline())
                if self.c_p['polymerization_LED'] != 'H' and self.c_p['polymerization_LED'] != 'L':
                    self.c_p['polymerization_LED'] = 'L'
            time.sleep(self.sleep_time)
        # Turn off blue LED and block the red before exiting.
        self.ArduinoUnoSerial.write(b'L')
        time.sleep(self.sleep_time)
        self.ArduinoUnoSerial.write(b'C')
        self.ArduinoUnoSerial.close()
