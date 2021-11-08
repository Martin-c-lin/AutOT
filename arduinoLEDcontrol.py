import serial, time
from threading import Thread

def get_arduino_c_p():
    arduino_c_p = {
    'polymerization_LED':'L',
    'polymerization_LED_status':'OFF',
    'background_illumination': False,
    'green_laser':True,
    'polymerization_time':1000, # polymerization time in ms.
    }
    return arduino_c_p

# TODO: This probably does not need to be a separate thread.
# TODO: Should communicate back and forth with the arduino.

def toggle_BG_shutter(c_p):

    c_p['background_illumination'] = not c_p['background_illumination']
    if c_p['background_illumination']:
        # Open shutter
        c_p['polymerization_LED'] = 'O'
    else:
        # Close shutter
        c_p['polymerization_LED'] = 'C'
    set_default_exposure(c_p)

def toggle_green_laser(c_p):

    c_p['green_laser'] = not c_p['green_laser']
    if c_p['green_laser']:
        # Open shutter
        c_p['polymerization_LED'] = 'G'
    else:
        # Close shutter
        c_p['polymerization_LED'] = 'B'
    set_default_exposure(c_p)

def set_default_exposure(c_p):
    # TODO replace this with a single parameter with 3 different values.
    if c_p['background_illumination']:
        c_p['exposure_time'] = 50000
        c_p['downsampling'].set(False)
    elif c_p['green_laser'] and not c_p['background_illumination']:
        c_p['exposure_time'] = 27000
    elif not c_p['green_laser'] and not c_p['background_illumination']:
        c_p['exposure_time'] = 200_000
        c_p['downsampling'].set(True)

    c_p['new_settings_camera'] = True

class ArduinoLEDControlThread(Thread):
    '''
    Thread which controls the arduinos output. Turns on/off the polymerization LED
    '''
    def __init__(self, threadID, name, c_p, sleep_time=0.01, port = 'com3'):
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
        time.sleep(1)
        toggle_green_laser(self.c_p)
        # TODO make it so that this thread listens in on the arduino and has two way communications with it.
        while self.c_p['program_running']:
            # This function is made significantly faster by this check.
            if not self.last_write == self.c_p['polymerization_LED']:
                self.last_write = self.c_p['polymerization_LED']

                # If the polymerization time is updated this needs to be recorded
                if self.last_write[0] == 'S':
                    try:
                        self.c_p['polymerization_time'] = int(self.c_p['polymerization_LED'][1:])
                    except:
                        print('Incorrect message')
                if self.last_write == 'T':
                    self.c_p['polymerization_LED_status'] = 'ON'

                message = self.last_write.encode('utf-8')
                self.ArduinoUnoSerial.write(message)

                #print(self.ArduinoUnoSerial.readline())
                if self.c_p['polymerization_LED'] != 'H' and self.c_p['polymerization_LED'] != 'L':
                    self.c_p['polymerization_LED'] = 'L'
            time.sleep(self.sleep_time)
        # Turn off blue LED and block the red before exiting.
        self.ArduinoUnoSerial.write(b'W')
        time.sleep(self.sleep_time)
        self.ArduinoUnoSerial.write(b'C')
        time.sleep(self.sleep_time)
        self.ArduinoUnoSerial.write(b'B')
        self.ArduinoUnoSerial.close()
