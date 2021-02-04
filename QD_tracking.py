# Algorithm for locating the quantum dots
import numpy as np
import cv2
from find_particle_threshold import find_particle_centers
from numba import jit
from threading import  Thread
from time import sleep, time
import pyfftw
pyfftw.interfaces.cache.enable()
from math import isclose
from tkinter import BooleanVar
# TODO incorporate also radial symmetry in the tracking.


def get_fft_object(image_shape):
    a = pyfftw.empty_aligned(image_shape, dtype='complex64')
    b = pyfftw.empty_aligned(image_shape, dtype='complex64')

    # Over the both axes
    print('Creating FFTW object')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1))
    return fft_object

image_shape = [3008, 3600]
fft_object = get_fft_object(image_shape)


@jit
def normalize_image(image):
    # Normalizes the image to be in range 0,1
    image -= np.min(image)
    image /= np.max(image)
    image = image -np.mean(image)
    return image

def create_circular_mask(h, w, center=None, radius=20):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def create_circular_mask_inverse(h, w, center=None, radius=100):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask

# Create default mask to speed things up
filter_radii = [20, 100]
s = [0,0]
s[0] = int(image_shape[1]/2)
s[1] = int(image_shape[0]/2)
mask = create_circular_mask(image_shape[0],image_shape[1],s,filter_radii[0])
outer_mask = create_circular_mask_inverse(image_shape[0],image_shape[1],s,filter_radii[1])

def fourier_filter(image, inner_filter_width=20, outer_filter_width=100):
    '''
    Function which filters outs the low frequency componentes of a image.
    Essentially a high-pass filter for the image.

    Inputs:
        image - Image to be filtered.
        inner_filter_width - How many elements in the middle of the fourier-transformed image should be removed
        outer_filter_width -
    Outputs:
        Filtered image
    '''
    # Make use of global object to avoid having to recreate them at each call.
    global image_shape, mask, outer_mask, filter_radii, fft_object

    d = np.shape(image)

    if not d == image_shape or not inner_filter_width == filter_radii[0]\
        or not outer_filter_width == filter_radii[1]:
        s = [0,0]
        s[0] = int(d[1] / 2)
        s[1] = int(d[0] / 2)
        filter_radii[0] = inner_filter_width
        mask = create_circular_mask(d[0], d[1], s, inner_filter_width)

        filter_radii[1] = outer_filter_width
        outer_mask = create_circular_mask_inverse(d[0], d[1], s, outer_filter_width)

    if not d == image_shape:
        image_shape = d
        fft_object = get_fft_object(image_shape)

    ff = fft_object(image)
    fshift = np.fft.fftshift(ff)


    fshift[mask] = 0 # set center to 0
    fshift[outer_mask] = 0 # set outer region to 0

    inv = np.fft.ifftshift(fshift)
    inv = fft_object(inv)
    return inv


def find_QDs(image, inner_filter_width=15, outer_filter_width=300,threshold=0.11,
    particle_size_threshold=60, particle_upper_size_threshold=400, edge=80):
    '''
    Function for detecting the quantum dots.

    Inputs:
            image - image to be tracked
    Outputs:
            px - list of detected quantum dots x-positions. Empty if no QDs were detected
            py - list of detected quantum dots y-positions. Empty if no QDs were detected
            image - The thresholded image used to detect the quantum dots.
                Usefull for debugging and finding suitable paramters to give this
                function for optimizing tracking.
    '''
    # TODO make this function also discover printed areas.
    # NOTE: Threshold lowered since normalize image has been changed to set the mean of the image to 0.
    if image is None:
        return [], []
    else:
        s = np.shape(image)
        if s[0] < edge*2 or s[1] < edge*2:
            return [], []
    image = fourier_filter(image, inner_filter_width=inner_filter_width, outer_filter_width=outer_filter_width)
    image = normalize_image(np.float32(image)) # Can edge removal be added here already?
    x,y,ret_img = find_particle_centers(image[edge:-edge,edge:-edge], threshold=threshold, particle_size_threshold=particle_size_threshold, particle_upper_size_threshold=particle_upper_size_threshold)

    px = [s[1] - x_i - edge for x_i in x]
    py = [s[0] - y_i - edge for y_i in y]
    return px, py, ret_img

def get_QD_tracking_c_p():
    '''
    Function for retrieving default c_p needed for tracking QDs.
    '''
    tracking_params = {
        'tracking_edge': 80,
        'threshold': 0.6,
        'inner_filter_width': 20,
        'outer_filter_width': 100,
        'particle_size_threshold': 30,
        'particle_upper_size_threshold': 5000,
        'max_trapped_counter':3, # Maximum number of missed detections allowed before
        # particle is considered untrapped.
        'QD_trapped': False, # True if a QD has been trapped, will still be true if it has been trapped in the last few frames
        'QD_currently_trapped': False, # True if the QD was in the trap the last frame
        'move_QDs':BooleanVar(),
        'QD_trapped_counter': 0,
        'QD_polymerization_time': 2, # time during which the polymerization laser will be turned on.
        'closest_QD': None, # index of quantum dot closest to the trap
        'QD_target_loc_x': [1, 5], # X - location(in mum) where quantum dots should be placed.
        'QD_target_loc_y': [1, 1],
        'polymerized_x':[],
        'polymerized_y':[],
        'nbr_quantum_dots_stuck': 0, # number of quantum dots already positioned
        'step_size': 0.2, # Step size to move the quantum dots
        'tolerance': 0.01
    }

    return tracking_params


def is_trapped(c_p, trap_dist):
    '''
    Calculate distance between trap and particle centers.
    '''
    if len(c_p['particle_centers'][0]) < 1:
        c_p['QD_currently_trapped'] = False
        c_p['closest_QD'] = None
        return None

    dists_x = np.asarray(c_p['particle_centers'][0] - c_p['traps_relative_pos'][0][0])
    dists_y = np.asarray(c_p['particle_centers'][1] - c_p['traps_relative_pos'][1][0])
    dists_tot = dists_x**2 + dists_y**2
    min_val = min(dists_tot)
    c_p['QD_currently_trapped'] = min_val < trap_dist**2
    c_p['closest_QD'] = np.argmin(dists_tot)#dists_tot.index(min_val)
    #print(dists_x, dists_y)
    return min_val < trap_dist**2

class QD_Tracking_Thread(Thread):
   '''
   Thread which does the tracking and placement of quantum dots automatically.

   '''
   def __init__(self, threadID, name, c_p, sleep_time=0.05, tolerance=0.0005):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.sleep_time = sleep_time
        self.tolerance = tolerance
        self.setDaemon(True)

   def __del__(self):
       self.c_p['tracking_on'] = False

   def trapped_now(self):
       is_trapped(self.c_p, 10)
       if self.c_p['QD_currently_trapped']:
           self.c_p['QD_trapped'] = True
           self.c_p['QD_trapped_counter'] = 0
       else:
           self.c_p['QD_trapped_counter'] += 1
           if self.c_p['QD_trapped_counter'] >= self.c_p['max_trapped_counter']:
               self.c_p['QD_trapped'] = False

   def move_to_target_location(self):
       '''
       Function for transporting a quantum dot to target location.
       '''
       print('Trying to move to target location')
       # TODO add so that this function automatically moves the QD to target location
       # Has been solved by setting the paramter 'piezo_move_to_target' to true
       pass

       # Update target position of piezo in x-direction
       x_move = self.c_p['QD_target_loc_x'][self.c_p['nbr_quantum_dots_stuck']] - \
            self.c_p['piezo_current_position'][0]
       if x_move < 0:
           self.c_p['piezo_target_position'][0] += max(x_move, -self.c_p['step_size'])
       else:
           self.c_p['piezo_target_position'][0] += min(x_move, self.c_p['step_size'])

       # Update target position of piezo in y-direction
       y_move = self.c_p['QD_target_loc_y'][self.c_p['nbr_quantum_dots_stuck']] - \
            self.c_p['piezo_current_position'][1]
       if y_move < 0:
           self.c_p['piezo_target_position'][1] += max(y_move, -self.c_p['step_size'])
       else:
           self.c_p['piezo_target_position'][1] += min(y_move, self.c_p['step_size'])

   def look_for_quantum_dot(self):
       pass

   def move_QD_to_location_rough(self, x, y, step = 0.0005):
       # 1 check if quantum dot is trapped
       # 2.1 quantum dot trapped -> take step towards target location
       # 2.2 QD not trapped.
       # 2.2.1 QDs in view
       self.trapped_now()
       if self.c_p['closest_QD'] is not None:

           # Calcualte distance to target position
           if self.c_p['QD_trapped']:
               d = [x - self.c_p['stepper_current_position'][0], y - self.c_p['stepper_current_position'][1]]
           else:
               # No QD is trapped but there are other visible in frame
               dx = self.c_p['particle_centers'][0][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][0][0]
               dy = self.c_p['particle_centers'][1][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][1][0]
               # TODO finish this function, increase step?
               d = [dx/self.c_p['mmToPixel'], dy/self.c_p['mmToPixel']]

           # If we are close enough to target location, return
           if self.c_p['QD_trapped'] and (d[0]**2 + d[1]**2) < self.tolerance**2:
                print('Done with moving')
                return True

           # Move a small step towards target location or QD
           if d[0] < 0:
               self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0] + max(-step, d[0])
           else:
               self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0] +  min(step, d[0])
           if d[1] < 0:
               self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] + max(-step, d[1])
           else:
               self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] + min(step, d[1])
           #print('QD currently in trap:',self.c_p['QD_currently_trapped'], 'QD recently in trap', self.c_p['QD_trapped'])
           return False

       else:
           # look for other particle
           return None
           pass

   def stick_quantum_dot(self):

       # Check that a quantum dot is trapepd
       if not self.c_p['QD_trapped']:
           return
       # Open shutter to stick the quantum dot
       self.c_p['shutter_open_time'] = 500 # open shutter for 500 ms
       self.c_p['should_shutter_open'] = True
       # Increase the stick count
       self.c_p['nbr_quantum_dots_stuck'] += 1

   def trap_quantum_dot(self):
       '''
       Function for trapping a quantum dot and trapping it in the area around the area it is in now.
       If there are no quantum dots in the area then the program will go look for one.
       '''
       if len(self.c_p['particle_centers'][0]) > 0:
           #if self.c_p[]:
           pass
       else:
           self.look_for_quantum_dot()
           print('Looking for quantum dots')
       pass

   def ready_to_stick(self):
       # Check if a quantum dot is in correct position
       if len(self.c_p['QD_target_loc_y']) >= self.c_p['nbr_quantum_dots_stuck']:
           return False

       y = self.c_p['piezo_current_position'][1] - self.c_p['QD_target_loc_y'][self.c_p['nbr_quantum_dots_stuck']]
       x = self.c_p['piezo_current_position'][0] - self.c_p['QD_target_loc_x'][self.c_p['nbr_quantum_dots_stuck']]

       return np.abs(x) < self.c_p['tolerance'] and np.abs(y) < self.c_p['tolerance']

   def extract_piezo_image(self):
       '''
       Exctracts the part of the image onto which the piezos can move.
       '''
       x_0 = self.c_p['traps_relative_pos'][0][0]
       y_0 = self.c_p['traps_relative_pos'][1][0]

       # Also check what happens when zoomed in
       # should there be a factor 1000 somewhere here?
       x_start = int(x_0 - self.c_p['piezo_current_position'][0] * self.c_p['mmToPixel'] / 1000)
       x_end = int(x_0 + (20 - self.c_p['piezo_current_position'][0]) * self.c_p['mmToPixel'] / 1000)

       y_start = int(y_0 - self.c_p['piezo_current_position'][1] * self.c_p['mmToPixel'] / 1000)
       y_end = int(y_0 + (20 - self.c_p['piezo_current_position'][1]) * self.c_p['mmToPixel'] / 1000)

       return self.c_p['image'][x_start:x_end, y_start:y_end], True

   def run(self):

       while self.c_p['program_running']:
           if self.c_p['tracking_on']:
               # TODO make it possible to adjust tracking parameters on the fly
               start = time()
               # Do the particle tracking.
               # Note that the tracking algorithm can easily be replaced if need be

               x, y, tmp = find_QDs(self.c_p['image'],
                inner_filter_width=12, outer_filter_width=240)

               # This sort of works for the polymerized areas.
               # TODO add parameter for this guy
               self.c_p['polymerized_x'], self.c_p['polymerized_y'], tmp = find_QDs(
               self.c_p['image'], inner_filter_width=5, outer_filter_width=140,
               particle_size_threshold=1000, particle_upper_size_threshold=6400,threshold=0.11)


               self.c_p['particle_centers'] = [x, y]

               if self.c_p['move_QDs'].get():
                   tmp = self.move_QD_to_location_rough(
                    x=self.c_p['stepper_starting_position'][0],
                    y=self.c_p['stepper_starting_position'][1])
                   if tmp:
                       self.c_p['move_QDs'].set(False)

               # Check trapping status
               # self.trapped_now()
               #
               # if self.c_p['QD_trapped']:
               #     self.move_to_target_location()
               #     if self.ready_to_stick():
               #         print('Sticking QD no:', self.c_p['nbr_quantum_dots_stuck'])
               #         self.stick_quantum_dot()
               # else:
               #     self.trap_quantum_dot()

               # Check if particle is trapped or has been trapped in the last number of frames
               # if so then try to move it to target position.
               #print('Tracked in', time()-start, ' seconds.')
               #print("Particles at",x,y)

           sleep(self.sleep_time)
       self.__del__()
