# Algorithm for locating the quantum dots
import numpy as np
import cv2
from find_particle_threshold import find_particle_centers
from numba import jit
from threading import  Thread
from time import sleep, time

@jit
def normalize_image(image):
    #image = image/np.max(image)
    image = image -np.mean(image)
    image -= np.min(image)
    image /= np.max(image)
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

@jit
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

    # TODO replace fft with fftw
    ff = np.fft.fft2(image)
    fshift = np.fft.fftshift(ff)
    d = np.shape(image)
    s = [0,0]
    s[0] = int(d[1] / 2)
    s[1] = int(d[0] / 2)

    # put this on top so I do not need to create it on each call
    mask = create_circular_mask(d[0],d[1],s,inner_filter_width)
    outer_mask = create_circular_mask_inverse(d[0],d[1],s,outer_filter_width)

    fshift[mask] = 0 # set center to 0
    fshift[outer_mask] = 0 # set center to 0

    inv = np.fft.ifftshift(fshift)
    inv = np.fft.fft2(inv)
    return inv


def find_QDs(image, inner_filter_width=20, outer_filter_width=100,threshold=0.6,
    particle_size_threshold=30, particle_upper_size_threshold=5000, edge=80):
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

    if image is None:
        return [], [], []
    else:
        s = np.shape(image)
        if s[0] < edge*2 or s[1] < edge*2:
            return [], [], []
    image = fourier_filter(image, inner_filter_width=inner_filter_width, outer_filter_width=outer_filter_width)
    image = np.float32(normalize_image(image)) # Can edge removal be added here already?

    x,y = find_particle_centers(image[edge:-edge,edge:-edge], threshold=threshold, particle_size_threshold=particle_size_threshold, particle_upper_size_threshold=particle_upper_size_threshold)

    px = [s[1] - x_i - edge for x_i in x]
    py = [s[0] - y_i - edge for y_i in y]
    return px, py, image

def get_QD_tracking_c_p():
    '''
    Function for retrieving default c_p needed for tracking QDs.
    '''
    tracking_params = {
        'tracking_edge':80,
        'threshold':0.6,
        'inner_filter_width':20,
        'outer_filter_width':100,
        'particle_size_threshold':30,
        'particle_upper_size_threshold':5000,
        'max_trapped_counter':5, # Maximum number of missed detections allowed before
        # particle is considered untrapped.
        'QD_trapped':False, # True if a QD has been trapped, will still be true if it has been trapped in the last few frames
        'QD_currently_trapped': False, # True if the QD was in the trap the last frame
        'QD_trapped_counter': 0,
        'QD_target_location':[[],[]], # Location to stick the QDs to. Measured in mm, motor positions.
        'QD_polymerization_time':2, # time during which the polymerization laser will be turned on.
        'closest_QD': None, # index of quantum dot closest to the trap
    }

    return tracking_params


def is_trapped(c_p, trap_dist):
    '''
    Calculate distance between trap and particle centers.
    '''
    if len(c_p['particle_centers'][0]) < 1:
        return False, None
    dists_x = np.asarray(c_p['particle_centers'][0] - c_p['traps_absolute_pos'][0][0])
    dists_y = np.asarray(c_p['particle_centers'][1] - c_p['traps_absolute_pos'][1][0])
    dists_tot = dists_x**2 + dists_y**2
    return min(a) < trap_dist

class QD_Tracking_Thread(Thread):
   '''
   Thread which does the tracking and placement of quantum dots automatically.

   '''
   def __init__(self, threadID, name, c_p, sleep_time=0.01):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.sleep_time = sleep_time
        self.setDaemon(True)

   def __del__(self):
       self.c_p['tracking_on'] = False


   def trapped_now(self):
       self.c_p['QD_currently_trapped'], self.c_p['closest_QD'] = is_trapped(self.c_p, 15)
       if self.c_p['QD_currently_trapped']:
           self.c_p["QD_trapped"] = True
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
       # TODO add so that this function automatically
       pass

   def look_for_quantum_dot(self):
       pass


   def trap_quantum_dot(self):
       '''
       Function for trapping a quantum dot and trapping it in the area around the area it is in now.
       If there are no quantum dots in the area then the program will go look for one.
       '''
       if len(self.c_p['particle_centers'][0]) > 0:
           pass # Should move in and trap that QD
       else:
           self.look_for_quantum_dot()
           print('Looking for quantum dots')
       pass


   def run(self):

       while self.c_p['program_running']:
           if self.c_p['tracking_on']:
               # TODO make it possible to adjust tracking parameters on the fly
               #start = time()
               # Do the particle tracking.
               # Note that the tracking algorithm can easily be replaced if need be
               x, y, tracked_image = find_QDs(self.c_p['image'])
               self.c_p['particle_centers'] = [x, y]

               # Check trapping status
               self.trapped_now()

               if self.c_p['QD_trapped']:
                   self.move_to_target_location()
               else:
                   self.trap_quantum_dot()
               # Check if particle is trapped or has been trapped in the last number of frames
               # if so then try to move it to target position.
               # print('Tracked in', time()-start, ' seconds.')
               #print("Particles at",x,y)

           sleep(self.sleep_time)
       self.__del__()