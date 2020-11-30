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

def remove_edge_detections(x,y, width=1200, height=1000, edge_size=30):
    # Removes false positives close to the edge.
    # TODO make it more elegant by using vectorization properly.
    px = []
    py = []
    for i in range(len(x)):
        if x[i]>edge_size and x[i]<width-edge_size and y[i]>edge_size and y[i]<height-edge_size:
            px.append(1200 - x[i])
            py.append(1000 - y[i])
    return px, py

def find_QDs(image, inner_filter_width=20, outer_filter_width=100,threshold=0.6,
    particle_size_threshold=30, particle_upper_size_threshold=5000, edge=80):
    if image is None:
        return [], [], []
    else:
        s = np.shape(image)
        if s[0] < edge*2 or s[1] < edge*2:
            return [], [], []
    image = fourier_filter(image, inner_filter_width=inner_filter_width, outer_filter_width=outer_filter_width)
    image = np.float32(normalize_image(image)) # Can I add the edge removal here already

    x,y = find_particle_centers(image[edge:-edge,edge:-edge], threshold=threshold, particle_size_threshold=particle_size_threshold, particle_upper_size_threshold=particle_upper_size_threshold)
    #x,y = remove_edge_detections(x,y)
    px = [s[1] - x_i - edge for x_i in x]
    py = [s[0] - y_i - edge for y_i in y]
    return px,py,image

def get_QD_tracking_c_p():
    # Function for retrieving default c_p needed for tracking QDs
    tracking_params = {
    'tracking_edge':80,
    }

    return tracking_params


class QD_Tracking_Thread(Thread):
   '''
   Thread which does the tracking.
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

   def run(self):

       while self.c_p['program_running']:
           if self.c_p['tracking_on']:
               # TODO make it possible to adjust tracking parameters on the fly
               start = time()
               x, y, tracked_image = find_QDs(self.c_p['image']) # Need to send copy?
               self.c_p['particle_centers'] = [x, y]
               # print('Tracked in', time()-start, ' seconds.')
               #print("Particles at",x,y)

           sleep(self.sleep_time)
       self.__del__()
