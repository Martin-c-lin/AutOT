# import keyboard  # using module keyboard
# For windows
from pynput import keyboard

class TRD():
    def __init__(self, c_p=None):
        self.speed = 1
        self.x = 10
        self.c_p = c_p
        # Have these here so they can easily be changed for different stages.
        self.current_pos = 'stepper_current_position'
        self.target_pos = 'stepper_target_position'
    def on_press(self, key):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys

        # Use keys to control the motors. Shift for changing between fast and slow movement.
        # When pressed we move a little bit. 
        print('Key pressed: ' + k)
        if k == 'shift':
            self.speed %= 3
            self.speed += 1
            #print(f'speed {self.speed}')
        elif k == 'up':
            next_pos = self.c_p[self.current_pos][0] + self.speed * 1e-3
            self.c_p[self.target_pos ][0] = next_pos
        elif k == 'down':
            next_pos = self.c_p[self.current_pos][0] - self.speed * 1e-3
            self.c_p[self.target_pos ][0] = next_pos

        elif k == 'left':
            next_pos = self.c_p[self.current_pos][1] + self.speed * 1e-3
            self.c_p[self.target_pos ][1] = next_pos
        elif k == 'right':
            next_pos = self.c_p[self.current_pos][1] - self.speed * 1e-3
            self.c_p[self.target_pos ][1] = next_pos

        elif k == 'page_up':
            next_pos = self.c_p[self.current_pos][2] + self.speed * 1e-3
            self.c_p[self.target_pos ][2] = next_pos
        elif k == 'page_down':
            next_pos = self.c_p[self.current_pos][2] - self.speed * 1e-3
            self.c_p[self.target_pos ][2] = next_pos

        elif k == 'q':
            return False
        return self.c_p['program_running'] # Will require the user to press a key to quit

# trd = TRD()
# listener = keyboard.Listener(on_press=trd.on_press)
# listener.start()  # start to listen on a separate thread
# listener.join()  # remove if main thread is polling self.keysq