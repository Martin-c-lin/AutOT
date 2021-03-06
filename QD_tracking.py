# Algorithm for locating the quantum dots
import numpy as np
from find_particle_threshold import find_particle_centers
from arduinoLEDcontrol import toggle_BG_shutter
from numba import jit
from threading import Thread
from time import sleep, time, strftime
import pyfftw
from tkinter import BooleanVar

#from scipy.spatial import distance_matrix
pyfftw.interfaces.cache.enable()


def get_fft_object(image_shape):
    '''
    Creates a pyfftw object for quicker FFTs on CPU.

    Parameters
    ----------
    image_shape : Tuple or array
        Size of input to the fft object. I.e size of images to be transformed.

    Returns
    -------
    fft_object : FFTW object
        Object for faster FFTs.

    '''
    a = pyfftw.empty_aligned(image_shape, dtype='complex64')
    b = pyfftw.empty_aligned(image_shape, dtype='complex64')

    # Over the both axes
    print('Creating FFTW object')
    fft_object = pyfftw.FFTW(a, b, axes=(0, 1))
    return fft_object


image_shape = [3008, 3600]
fft_object = get_fft_object(image_shape)


@jit
def normalize_image(image):
    # Normalizes the image to be in range 0,1
    image -= np.min(image)
    image /= np.max(image)
    image -= np.mean(image)
    return image


def create_circular_mask(h, w, center=None, radius=20):

    if center is None:  # Use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_circular_mask_inverse(h, w, center=None, radius=100):

    if center is None:  # Use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask


# Create default mask to speed things up
filter_radii = [20, 100]
s = [0, 0]
s[0] = int(image_shape[1]/2)
s[1] = int(image_shape[0]/2)
mask = create_circular_mask(image_shape[0], image_shape[1], s, filter_radii[0])
outer_mask = create_circular_mask_inverse(image_shape[0], image_shape[1], s,
                                          filter_radii[1])
# TODO make the mask and outer mask FFT-shifted so we don't need to do that
# withe the for each image. Combine into a single mask

def autoset_laser_position(c_p):
    '''
    Function for calculating the exact position of the laser once a QD (or set of QDs)
    have been trapped.
    '''
    # This is a bit tricky since it requires one to take multiple images and
    # process these to get an average

def fourier_filter(image, inner_filter_width=20, outer_filter_width=100):
    '''
    Function which filters outs the low frequency componentes of a image.
    Essentially a high-pass filter for the image.


    Parameters
    ----------
    image : Array of ints
        Image to be filtered.
    inner_filter_width : Int, optional
        How many elements in the middle of the fourier-transformed image
        should be removed. The default is 20.
    outer_filter_width : Int, optional
        How many elements outside the center should be removed. The default is
        100.

    Returns
    -------
    inv : TYPE
        DESCRIPTION.

    '''
    # Make use of global object to avoid having to recreate them at each call.
    global image_shape, mask, outer_mask, filter_radii, fft_object

    d = np.shape(image)

    if not d == image_shape or not inner_filter_width == filter_radii[0]\
            or not outer_filter_width == filter_radii[1]:
        s = [0, 0]
        s[0] = int(d[1] / 2)
        s[1] = int(d[0] / 2)
        filter_radii[0] = inner_filter_width
        mask = create_circular_mask(d[0], d[1], s, inner_filter_width)

        filter_radii[1] = outer_filter_width
        outer_mask = create_circular_mask_inverse(d[0], d[1], s,
                                                  outer_filter_width)

    if not d == image_shape:
        image_shape = d
        fft_object = get_fft_object(image_shape)

    ff = fft_object(image)
    fshift = np.fft.fftshift(ff)
    fshift[mask] = 0  # Set center to 0
    fshift[outer_mask] = 0  # set outer region to 0

    inv = np.fft.ifftshift(fshift)
    inv = fft_object(inv)
    return inv


def find_QDs(image, inner_filter_width=15, outer_filter_width=300,
             threshold=0.11, particle_size_threshold=60,
             particle_upper_size_threshold=400, edge=80,
             negative_particles= False, fill_holes=False,
             check_circular=False):
    '''
    Function for detecting the quantum dots.

    Parameters
    ----------
    image : Array of ints
        Image to be tracked.
    inner_filter_width : Int, optional
        Parameter of fourier filter. The default is 15.
    outer_filter_width : Int, optional
        Parameter of fourier filter. The default is 300.
    threshold : Float, optional
        Parameter determining the sensitivity of the tracking. The default is
        0.11.
    particle_size_threshold : Int, optional
        Minimum size(in pixels) of detected particle. The default is 60.
    particle_upper_size_threshold : Int, optional
        Maximum size(in pixels) of detected particle. The default is 400.
    edge : Int, optional
        How far close to the edge a particle is allowed to be while still being
        detected. Fourier-transform gives errors close to the edge.
        The default is 80.
    negative_particles : Boolean, optional
        Set to true if the particles are darker than the background.
        The default value is False.
    Returns
    -------
    px : list
        List of detected quantum dots x-positions. Empty if no QDs were
        detected.
    py : list
        List of detected quantum dots y-positions. Empty if no QDs were
        detected.
    image : Array of floats
        The thresholded image used to detect the quantum dots. Usefull for
        debugging and finding suitable paramters to give this function for
        optimizing tracking.

    '''

    # TODO make this function also discover printed areas.

    if image is None:
        return [], [], []
    else:
        s = np.shape(image)
        if s[0] < edge*2 or s[1] < edge*2:
            return [], [], []
    image = fourier_filter(image, inner_filter_width=inner_filter_width,
                           outer_filter_width=outer_filter_width)
    image = normalize_image(np.float32(image))
    if negative_particles:
        x, y, ret_img = find_particle_centers(image[edge:-edge,edge:-edge] * -1,
                                        threshold=threshold,
                                        particle_size_threshold=particle_size_threshold,
                                        particle_upper_size_threshold=particle_upper_size_threshold,
                                        fill_holes=fill_holes, check_circular=check_circular)
    else:
        x, y, ret_img = find_particle_centers(image[edge:-edge,edge:-edge],
                                        threshold=threshold,
                                        particle_size_threshold=particle_size_threshold,
                                        particle_upper_size_threshold=particle_upper_size_threshold,
                                        fill_holes=fill_holes)

    px = [s[1] - x_i - edge for x_i in x]
    py = [s[0] - y_i - edge for y_i in y]
    return px, py, image#ret_img


def find_optimal_move(p1, p2):
    '''
    Finds the distance to move the stage so as to minimize the difference
    between the measured polymerized areas and the true pattern.

    Parameters
    ----------
    p1 : Numpy array
        Target pattern which we are trying to match to.
    p2 : Numpy array
        Seen pattern with which the distance to the target pattern should be
        minimized.

    Returns
    -------
    x : Float
        distance in x direction between target pattern and seen pattern.
    y : Float
        distance in y direction between target pattern and seen pattern.
    '''
    from scipy.spatial import distance_matrix

    if len(p2) < 1 or len(p1) < 1:
        return 0, 0
    min_x = np.zeros(len(p1))
    min_y = np.zeros(len(p1))
    d = distance_matrix(p1, p2)
    for idx, p in enumerate(p1):
        min_idx = np.argmin(d[idx, :])
        min_x[idx] = p1[idx, 0] - p2[min_idx, 0]
        min_y[idx] = p1[idx, 1] - p2[min_idx, 1]
    # minimize max error to avoid finding intermediate stable positions
    max_idx = np.argmax(min_x**2 + min_y**2)
    # TODO dynamically change between using mean and max error
    # Use square of distance instead?
    # TODO can currently not find a good minimum unless the polymerization LED is
    # perfectly aligned with the laser.
    return min_x[max_idx], min_y[max_idx]


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
        'max_trapped_counter': 5,  # Maximum number of missed detections
        # allowed before particle is considered untrapped.
        'QD_trapped': False,  # True if a QD has been trapped, will still be
        # true if it has been trapped in the last few frames
        'QD_currently_trapped': False,
        'move_QDs': BooleanVar(),
        'generate_training_data': BooleanVar(),
        'QD_trapped_counter': 0,
        'QD_polymerization_time': 2,  # time during which the polymerization
        # laser will be turned on.
        'closest_QD': None,  # Index of quantum dot closest to the trap
        'QD_target_loc_x': [1, 5],  # X - location(in mum) where quantum dots
        # should be placed.
        'QD_target_loc_y': [1, 1],
        'QD_target_loc_x_px': [1000],
        'QD_target_loc_y_px': [1000], # Where the QD should be put
        'QD_fine_target':[0,0],
        'anchor_array_dist': [10, -10], # Distance between array and anchor(x,y) in microns
        'polymerized_x': [],
        'polymerized_y': [],
        'polymerized_x_piezo':[],
        'polymerized_y_piezo':[],
        #'QDs_placed': 0,  # number of quantum dots already in
        # position
        'step_size': 0.2,  # Step size to move the quantum dots
        'tolerance': 0.01,
        'position_QD_in_pattern': BooleanVar(),
        'override_QD_trapped': BooleanVar(),
        'Anchor_placed': False,
        'Anchor_position':[5,5], # Position of anchor in stepper coordinates

    }

    return tracking_params


def is_trapped(c_p, trap_dist):
    '''
    Calculate distance between trap and particle centers.
    '''

    # We can override the return of this function for testing purposes
    if c_p['override_QD_trapped'].get():
        c_p['QD_currently_trapped'] = True
        return True

    if len(c_p['particle_centers'][0]) < 1:
        c_p['QD_currently_trapped'] = False
        c_p['closest_QD'] = None
        return None

    dists_x = np.asarray(c_p['particle_centers'][0] - c_p['traps_relative_pos'][0][0])
    dists_y = np.asarray(c_p['particle_centers'][1] - c_p['traps_relative_pos'][1][0])
    dists_tot = dists_x**2 + dists_y**2
    min_val = min(dists_tot)
    c_p['QD_currently_trapped'] = min_val < trap_dist**2
    c_p['closest_QD'] = np.argmin(dists_tot)
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
        self.QD_unseen_counter = 0
        self.previous_pos = [0, 0] # Saved location
        self.piezo_step = 0.3
        self.anchor_time = 30 # Seconds to wait while anchor is being made
        self.polymerization_position_piezo = [-1, -1]
        self.qd_search_height = 2
        self.in_rough_location = False
        self.fine_position_found = False # Have we located the exact position of the next area?
        self.in_z_position = False
        self.in_fine_position = False # Are we in the exact position of the next area?
        self.printing_height = 0
        self.z_push_direction = 'down' # which direction is the QD being pushed, up or down
        self.up_steps = 0 # number of intermeiate lifts which have been made to find QDs while pushing down
        self.setDaemon(True)
        self.last_trapped_pos = [-1, -1] # Last position in which a QD was trapped

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

    def check_for_cured_areas(self):


            # We only look for polymerized areas when we have background_illumination
        try:
            subtracted = self.c_p['image'] - self.c_p['raw_background']
        except:
            subtracted = np.copy(self.c_p['image'])

        self.c_p['polymerized_x'], self.c_p['polymerized_y'],_ = find_QDs(
                            subtracted, inner_filter_width=7, outer_filter_width=80,
                            particle_size_threshold=1_000, # Need to test these sizes
                            particle_upper_size_threshold=100_000,
                            threshold=0.1, #07, # Increased sensitivity
                            edge=120,negative_particles=False, fill_holes=True,
                            check_circular=False)

    def check_for_trapped_QDs(self):
        # TODO let this guy look only at a smalla area of the image if need be
        if not self.c_p['override_QD_trapped'].get():
            if self.c_p['green_laser'] and not self.c_p['background_illumination']:
                x, y, ret_img = find_QDs(self.c_p['image'],
                                    inner_filter_width=7, outer_filter_width=180,
                                    particle_upper_size_threshold=700,
                                    threshold=0.1, #07, # Increased sensitivity
                                    edge=120,
                                    ) # 12 240
            elif not self.c_p['background_illumination']:
                # we can allow ourselves to be a bit more restrictive in this case
                x, y ,ret_img = find_particle_centers(self.c_p['image'], threshold=17, particle_size_threshold=150,
                                particle_upper_size_threshold=5000,
                                bright_particle=True, fill_holes=False)
                # x, y, ret_img = find_QDs(self.c_p['image'], inner_filter_width=15,
                #                  outer_filter_width=160,
                #                  threshold=0.25, particle_size_threshold=60,
                #                  particle_upper_size_threshold=30000, edge=150,
                #                  negative_particles= False,
                #                  fill_holes=True) # 12 240
            else:
                x=[]
                y=[]
                print('Cannot see QDs with background led turnd on!')
            # QDs have been located, now check if one is trapped.
            self.c_p['particle_centers'] = [x, y]
            self.trapped_now()
        else:
            self.trapped_now()

    def look_for_quantum_dot(self, x, y):
        '''
        Takes a move to look for a quantum dot
        '''
        # TODO: Z moves does not play nicely wiht z-xy compensation
        # If we have moved less than 5 microns in the z- direction we may look along z

        print('Looking for quantum dot')
        dx = self.c_p['stepper_current_position'][0] - x
        dy = self.c_p['stepper_current_position'][1] - y
        # dz = self.c_p['stepper_current_position'][2] - self.c_p['stepper_starting_position'][2]
        # Check if we are too far from original position
        if np.abs(dx) > 0.5:
            dx = -dx
        if np.abs(dy) > 0.5:
            dy = -dy
        dx /= np.abs(dx)
        dy /= np.abs(dy)

        choice = np.random.randint(-1,10)
        if choice == 1:
            pass
        # TODO use pizeo for z-moves
        #     self.c_p['stepper_target_position'][2] += 0.0005
        # elif 0<= choice:
        #     self.c_p['stepper_target_position'][2] -= 0.0005
        #     print('Changing focus a bit')
        # Make it move all the way to the QD it has seen.
        else:
            # Move in xy-plane
            if 2 <= choice <=3:
                dy += 2.0 # up
            if 4 <= choice <=5:
                dx += 2.0 # right
            if 6 <= choice <=7:
                dy -= 2.0 # down
            if 8 <= choice <=9:
                dx -= 2.0 # left

            self.c_p['stepper_target_position'][0] += dx * 0.01
            self.c_p['stepper_target_position'][1] += dy * 0.01

        # Too far from focus, move
        # if dz > -0.02 or dz > 0.01:
        #     self.c_p['stepper_target_position'][2] = self.c_p['stepper_starting_position'][2] - 0.005

    def move_QD_to_location_stepper(self, x, y, small_step = 0.0003, large_step=0.001):
        '''
        Algorithm for pulling a quantum dot to the position specified in x,y
        using the stepper only.
        Uses the following method
         Check if quantum dot is trapped:
          QD trapped -> take step towards target location
          QD not trapped:
            check if QD in view:
              QDs in view -> move to trap the QD
              QD not in view -> Look for QD with look_for_quantum_dot function
        '''
        # TODO Reuse this for the fine move but with a parameter which dictates the use of
        # the piezo.
        # TODO make it automatically avoid other QDs along the way.
        self.trapped_now()
        step = small_step
        if self.c_p['closest_QD'] is not None:
            self.QD_unseen_counter = 0
            # Calcualte distance to target position
            if self.c_p['QD_currently_trapped']:
                d = [x - self.c_p['stepper_current_position'][0], y - self.c_p['stepper_current_position'][1]]
                print('Moving QD towards target', d, x , y )
                step = small_step
            elif not self.c_p['QD_currently_trapped'] and self.c_p['QD_trapped']:
                d = [0, 0] # Wait for qd to be trapped
            else:
                # No QD is trapped but there are other visible in frame
                dx = self.c_p['particle_centers'][0][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][0][0]
                dy = self.c_p['particle_centers'][1][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][1][0]
                # TODO finish this function, increase step?
                d = [dx/self.c_p['mmToPixel'], dy/self.c_p['mmToPixel']]
                setp = large_step
                print('Moving to trap a QD', d)

            # If we are close enough to target location, return
            if self.c_p['QD_currently_trapped'] and (d[0]**2 + d[1]**2) < self.tolerance**2:
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
            print('QD currently in trap:',self.c_p['QD_currently_trapped'], 'QD recently in trap', self.c_p['QD_trapped'])
            return False

        else:
            # Could be that tracking is not perfect and a QD is missed one frame or two
            # therefore the cunter is used
            self.QD_unseen_counter += 1
            # look for other particle
            if self.QD_unseen_counter > 20:
                 self.look_for_quantum_dot(x, y)
            return None

    def reset_pushing(self):
        # Sets the variables/parameters used when pushing to their default value
        self.c_p['test_move_down'].set(False)
        self.printing_height = 0
        self.up_steps = 0

    def push_QD_to_glass(self, step=0.05, motor='piezo', tolerance=0.1):

        '''
        Function for pulling a QD down as far as possible.
        Inputs:
            Step - How large steps to take
            motor - Which motor to use
            tolerance - how close we need to move
        # Returns False if not Done, True if done, and None if QD was lost.
        '''
        # TODO integrate the stepper in a good way as well as a reset function.
        if motor == 'stepper':
            current_position = 'stepper_current_position'
            target_position = 'stepper_target_position'
            elevation = 'stepper_elevation'
        elif motor == 'piezo':
            if self.c_p['piezos_activated'].get():
                print('Please deactivate manual piezo move to allow for automatic control.')
                return None
            current_position = 'piezo_current_position'
            target_position = 'piezo_target_position'
            elevation = 'piezo_elevation'
        else:
            return None

        # Check trapped status
        self.trapped_now()

        if self.c_p['QD_currently_trapped']:
            self.QD_unseen_counter = 0
            if self.up_steps > 50: #self.printing_height > self.c_p[current_position][2] + step:
                print('Great height already found at: ', self.printing_height)
                self.reset_pushing()
                return True
            if motor == 'piezo' and self.c_p[current_position][2]  > 19.8:
                print('Error, cannot push further!')
                self.reset_pushing()
                return False
            self.c_p[elevation] += step
            self.printing_height = max(self.printing_height, self.c_p[current_position][2])
            sleep(self.sleep_time)
        else:
            self.QD_unseen_counter += 1
            sleep(self.sleep_time)
            if self.QD_unseen_counter % 10 == 0:
                self.c_p[elevation] -= step
                self.up_steps += 1
                print('Waiting for QD to catch up. Have waited for ',
                    self.QD_unseen_counter, ' iterations')
            # look for other particle
        if self.QD_unseen_counter >= 200:
            # We have not seen QDs for a good while, time to give up
            print('QD lost!')
            self.reset_pushing()
            return False
        return False

    def move_QD_to_location(self, x, y, step = 0.0003, motor='stepper',
            tolerance=None):
        '''
        Algorithm for pulling a quantum dot to the position specified in x,y
        using the stepper only.
        Uses the following method
         Check if quantum dot is trapped:
          QD trapped -> take step towards target location
          QD not trapped:
            check if QD in view:
              QDs in view -> move to trap the QD
              QD not in view -> Look for QD with look_for_quantum_dot function
        '''
        # TODO make it automatically avoid other QDs along the way.
        if motor == 'stepper':
            current_position = 'stepper_current_position'
            target_position = 'stepper_target_position'

        elif motor == 'piezo':
            current_position = 'piezo_current_position'
            target_position = 'piezo_target_position'
        else:
            return None

        if tolerance is None:
            tolerance = self.tolerance

        self.trapped_now()
        if self.c_p['closest_QD'] is not None or self.c_p['override_QD_trapped'].get():
            self.QD_unseen_counter = 0
            # Calcualte distance to target position
            if self.c_p['QD_currently_trapped']:
                d = [x - self.c_p[current_position][0], y - self.c_p[current_position][1]]
                print('Moving QD towards target', d, x , y , ' Motor is ', motor)
            elif not self.c_p['QD_currently_trapped'] and self.c_p['QD_trapped']:
                d = [0, 0] # Wait for qd to be trapped
            else:
                # No QD is trapped but there are other visible in frame
                dx = self.c_p['particle_centers'][0][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][0][0]
                dy = self.c_p['particle_centers'][1][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][1][0]
                # TODO finish this function, increase step?
                if motor == 'stepper':
                    d = [dx/self.c_p['mmToPixel'], dy/self.c_p['mmToPixel']]
                else:
                    d = [1000 * dx/self.c_p['mmToPixel'], 1000 * dy/self.c_p['mmToPixel']]

                print('Moving to trap a QD', d)

            # If we are close enough to target location, return
            if self.c_p['QD_currently_trapped'] and (d[0]**2 + d[1]**2) < tolerance**2:
                 print('Done with moving')
                 return True

            # Move a small step towards target location or QD
            if d[0] < 0:
                t_x = self.c_p[current_position][0] + max(-step, d[0])
            else:
                t_x = self.c_p[current_position][0] +  min(step, d[0])
            if d[1] < 0:
                t_y = self.c_p[current_position][1] + max(-step, d[1])
            else:
                t_y = self.c_p[current_position][1] + min(step, d[1])
            if motor == 'stepper':
                self.c_p[target_position][0] = t_x
                self.c_p[target_position][1] = t_y
                self.last_trapped_pos = [t_x, t_y]
            else:
                if 0 < t_x < 20 and 0 < t_y < 20:
                    self.c_p[target_position][0] = t_x
                    self.c_p[target_position][1] = t_y
                else:
                    # must use steppers
                    print('Must use stepper instead of piezo, out of range')
                    return self.move_QD_to_location(x, y)

            print('QD currently in trap:',self.c_p['QD_currently_trapped'],
                    'QD recently in trap', self.c_p['QD_trapped'])
            return False

        else:
            # Could be that tracking is not perfect and a QD is missed one frame or two
            # therefore the cunter is used
            self.QD_unseen_counter += 1
            # look for other particle
            if self.QD_unseen_counter > 30:
                # TODO test if this works, - it does not
                # find a better way of dealing with losing a QD. That is move back to get it
                #self.c_p[target_position][0:1] = self.last_trapped_pos
                print('Moved to previous position')
            if self.QD_unseen_counter > 40:
                self.c_p['piezo_current_position'][2] = self.qd_search_height
                self.look_for_quantum_dot(x, y)
            return None

    def stick_quantum_dot(self):

        # Check that a quantum dot is trapped
        # Open shutter to stick the quantum dot
        print('Sticking a QD!')
        self.c_p['polymerization_LED'] = 'T'
        sleep(self.c_p['polymerization_time']/1000 + 1)

    def locate_anchor_position(self):
        '''
        Calculates the position of the anchor in the image in microns.
        x=0,y=0 is in the top left corner of the image
        '''
        # Should only be run when zoomed out

        if not self.c_p['background_illumination']:
            toggle_BG_shutter(self.c_p)
            sleep(0.5)

        print('Trying to find anchor')
        x,y,ret_img =find_QDs(self.c_p['image'], inner_filter_width=5, outer_filter_width=70,
                            particle_size_threshold=18_000, # Need to test these sizes
                            particle_upper_size_threshold=100_000,
                            threshold=0.09, #07, # Increased sensitivity
                            edge=120,negative_particles=True, fill_holes=True,
                            check_circular=True)
        if self.c_p['background_illumination']:
            toggle_BG_shutter(self.c_p)
            sleep(0.5)

        if len(x) == 1:
            print('Found anchor at: ', x,y)
            tx = x[0] / self.c_p['mmToPixel'] * 1000
            ty = y[0] / self.c_p['mmToPixel'] * 1000
            return tx, ty
        if len(x) > 1:
            print('Found excessive number of anchors at locs', x, y)
        return None, None

    def locate_next_poly_area(self):
        '''
        Locates where the next polymerized area should be in the image
        (measured in microns relative top left corner).
        Returns True if we can use the piezo to move there and how to move it
        Returns False if we cannot.

        '''
        # TODO remove all QD sightings which are really close to a polymerized area

        # Check
        anchor_x, anchor_y = self.locate_anchor_position()
        if anchor_x is None:
            print('Could not locate anchor in image!')
            return None

        # Calculate laser position in microns in image
        Lx = self.c_p['traps_relative_pos'][0][0] / self.c_p['mmToPixel'] * 1000
        Ly = self.c_p['traps_relative_pos'][1][0] / self.c_p['mmToPixel'] * 1000

        # Calculate the move(microns) we need to make to get to the new array position
        dx = anchor_x + self.c_p['anchor_array_dist'][0] + self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']] - Lx
        dy = anchor_y + self.c_p['anchor_array_dist'][1] + self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']] - Ly

        # Calculate piezo target position
        Tx = self.c_p['piezo_current_position'][0] + dx
        Ty = self.c_p['piezo_current_position'][1] + dy

        motor = 'piezo'

        # Check that targets are ok
        if 0 > Tx or 20 < Tx or Ty < 0 or Ty > 20:
            # We are too far from the target to use the piezo, use the stepper instead
            motor == 'stepper'
            dx *= 1000
            dy *= 1000

        if motor == 'piezo':
            self.fine_position_found = True
            self.polymerization_position_piezo = [self.c_p['piezo_current_position'][0] + dx, self.c_p['piezo_current_position'][1] + dy]
            print('Piezo target locs identified')
        if motor == 'stepper':
            self.move_QD_to_location(
                 x=self.c_p['stepper_current_position'][0] + dx,
                 y=self.c_p['stepper_current_position'][1] + dy,#
                 motor='stepper', tolerance=0.0003)
            print('Stepper move needed ')

        return motor

    def ready_to_stick(self):
        # Check if a quantum dot is in correct position
        pass

    def step_move(self, axis, distance):
        # Makes a smart move in of length distance[mm] along axis
        # axis = 0 => x axis = 1 = y axis = 2 => z
        if self.c_p['stage_piezos']:
            if distance < 0:
                new_position_piezo = self.c_p['piezo_current_position'][axis] + max(-self.piezo_step, distance * 1000)
            else:
                new_position_piezo = self.c_p['piezo_current_position'][axis] + min(self.piezo_step, distance * 1000)
            if 0 < new_position_piezo < 20:
                self.c_p['piezo_target_position'][axis] = new_position_piezo
                print('Step move!', new_position_piezo, distance * 1000)
            else:
                self.c_p['stepper_target_position'][axis] = self.c_p['stepper_current_position'][axis] + max(-self.piezo_step, distance * 1000)
        elif self.c_p['using_stepper_motors']:
            self.c_p['stepper_target_position'][axis] = self.c_p['stepper_current_position'][axis] +  min(self.piezo_step, distance * 1000)

    def find_array_position(self):
        '''


        Returns
        -------
        None.

        '''
        # Adjust tracking parameters based on AOI
        width = self.c_p['AOI'][1] - self.c_p['AOI'][0]
        edge = 80 if width > 1500 else 20
        inner_filter_width = 10 if width > 1500 else 7
        outer_filter_width = 120 if width > 1500 else 80
        area_size_threshold = 7000

        # Open the BG shutter to see better
        if not self.c_p['background_illumination']:
            toggle_BG_shutter(self.c_p)
            sleep(1)
        # Locate the polymerized areas
        self.c_p['polymerized_x'], self.c_p['polymerized_y'], tmp = find_QDs(
                self.c_p['image'], inner_filter_width=inner_filter_width,
                outer_filter_width=outer_filter_width,
                threshold=0.11, particle_size_threshold=area_size_threshold,
                particle_upper_size_threshold=30000, edge=edge, negative_particles=True)

        # Close the BG shutter to see the QDs
        if self.c_p['background_illumination']:
            toggle_BG_shutter(self.c_p)
            sleep(1)

        # TODO make the program avoid already polymerized areas.
        if self.c_p['QDs_placed'] > 0:
            P1 = np.transpose(np.array([self.c_p['polymerized_x'], self.c_p['polymerized_y']]))
            P2 = np.transpose(np.array([self.c_p['QD_position_screen_y'], self.c_p['QD_position_screen_x']]))
            P2 = P2[:self.c_p['QDs_placed'],:]
            # TODO remove those points outside the AOI and edge properly
            s = np.shape(self.c_p['image'])

            P3 = np.array([P for P in P2 if (50<P[0]<s[0]-50 and 50<P[1]<s[1]-50)])
            dx, dy = find_optimal_move(np.array(P3), P1)
            print(dx,dy)

            # Check if there is a polymerized spot where the laser is
            if dx**2 + dy**2 < 100:
              lx = self.c_p['polymerized_x'] - self.c_p['traps_relative_pos'][0][0]
              ly = self.c_p['polymerized_y'] - self.c_p['traps_relative_pos'][1][0]

              dist = lx**2 + ly**2
              # Check that
              if len(dist) < 1:
                  return None, None

              if min(dist) < 600:
                  # The laser is close to an already polymerized area, move away from it
                  dx -= self.c_p['QD_position_screen_y'][self.c_p['QDs_placed']] - self.c_p['QD_position_screen_y'][self.c_p['QDs_placed']-1]
                  dy -= self.c_p['QD_position_screen_x'][self.c_p['QDs_placed']] - self.c_p['QD_position_screen_x'][self.c_p['QDs_placed']-1]
                  print('Moving away from already polymerized area', dx, dy)
            self.c_p['QD_target_loc_x_px'] = int(self.c_p['traps_relative_pos'][0][0] - dx)
            self.c_p['QD_target_loc_y_px'] = int(self.c_p['traps_relative_pos'][1][0] - dy)
            dx /= self.c_p['mmToPixel']
            dy /= self.c_p['mmToPixel']
            # TODO change so that this only returns a target distance to move
            #self.step_move(0, -dx)
            #self.step_move(1, -dy)
            return -dx, -dy
        return 0, 0

        # TODO Can we decrease minimum step in thorlabs motor?

    def update_polymerized_positions(self):
        '''
        Updates the relative position of the polymerized areas in view.
        Relies on the piezo for accurate movement.
        '''
        # TODO replace previous pos with polymerized_x_piezos
        x0p = self.c_p['piezo_current_position'][0]
        y0p = self.c_p['piezo_current_position'][1]
        x0l = self.c_p['traps_relative_pos'][0][0]
        y0l = self.c_p['traps_relative_pos'][1][0]
        self.c_p['polymerized_x'] = [x0l - (x0p - x) * self.c_p['mmToPixel'] / 1000.0 for x in self.c_p['polymerized_x_piezo']]
        self.c_p['polymerized_y'] = [y0l - (y0p - y) * self.c_p['mmToPixel'] / 1000.0 for y in self.c_p['polymerized_y_piezo']]

    def save_polymerization_data(self):
        # Function for saving all the data necessary for training a neural
        # network to detect polymerized locations.
        # TODO save piezos original positions
        # Wait for next frame
        sleep(self.c_p['exposure_time'] / 1e6)
        data_name = self.c_p['recording_path'] + "/training_data-" + \
                    strftime("%d-%m-%Y-%H-%M-%S")
        z_diff = self.c_p['piezo_current_position'][2] - self.c_p['piezo_starting_position'][2]
        data = [self.c_p['image'], self.c_p['polymerized_x'],
                self.c_p['polymerized_y'], z_diff]
        np.save(data_name, data)

    def random_piezo_move(self, safe_distance=110, piezo_tolerance=0.1):
        '''
        Function for moving to a new position where there is nothing polymerized
        using the piezos
        '''
        position_found = False
        # Generate new position
        z0 = self.c_p['piezo_starting_position'][2]
        print('z0 is ', z0)
        while not position_found and self.c_p['program_running']:
            position_ok = True
            x = np.random.uniform(1, 19)
            y = np.random.uniform(1, 19)
            z = np.random.uniform(max(z0-2,1), min(z0+2, 19))

            # Calcualte distance to new positon
            dx = (x - self.c_p['piezo_current_position'][0]) * self.c_p['mmToPixel'] / 1000
            dy = (y - self.c_p['piezo_current_position'][1]) * self.c_p['mmToPixel'] / 1000

            # check if new positon is ok
            for xi, yi in zip(self.c_p['polymerized_x'], self.c_p['polymerized_y']):
                x_sep = self.c_p['traps_relative_pos'][0][0] + dx -xi
                y_sep = self.c_p['traps_relative_pos'][1][0] + dy -yi
                if x_sep**2 + y_sep**2 < safe_distance**2:
                    position_ok = False
                if not self.c_p['program_running']:
                    return False
            # If position is ok then we have can exit the loop
            position_found = position_ok

        # Set new positon as target position
        self.c_p['piezo_target_position'][0] = x
        self.c_p['piezo_target_position'][1] = y
        self.c_p['piezo_target_position'][2] = z


        # Wait for piezos to move to new position
        move_finished = False
        while not move_finished:
            sleep(0.2)
            dx = np.abs(self.c_p['piezo_target_position'][0] - self.c_p['piezo_current_position'][0])
            dy = np.abs(self.c_p['piezo_target_position'][1] - self.c_p['piezo_current_position'][1])
            dz = np.abs(self.c_p['piezo_target_position'][2] - self.c_p['piezo_current_position'][2])
            if dx < piezo_tolerance and dy < piezo_tolerance and dz < piezo_tolerance:
                move_finished = True
            if not self.c_p['program_running']:
                return False
        sleep(0.5)
        return True

    def generate_polymerization_training_data(self):
        # Function for aoutomatically generating training data
        # 1 move to new location.
        # 2

        # If there are too many polymerized areas in place already, then move
        # to a new area.
        if len(self.c_p['polymerized_x']) > 9:
            # Reset polymerized areas since we are moving into new territory
            self.c_p['polymerized_x'] = []
            self.c_p['polymerized_y'] = []
            self.c_p['polymerized_x_piezo'] = []
            self.c_p['polymerized_y_piezo'] = []
            self.c_p['stepper_target_position'][0] += 0.08

            # Wait for stepper to finish moving
            while self.c_p['stepper_target_position'][0] - self.c_p['stepper_current_position'][0] > 0.005:
                sleep(self.sleep_time)
                print('Waiting for stepper to move')
                if not self.c_p['program_running']:
                    return
            # Make sure we have stopped before polymerizing
            sleep(1)

            # Update old position of piezos
            self.previous_pos[0] = self.c_p['piezo_current_position'][0]
            self.previous_pos[1] = self.c_p['piezo_current_position'][1]

        # Save current data on polymerization
        self.save_polymerization_data()

        # Polymerize a new area
        self.c_p['polymerization_LED'] = 'H'
        print('polymerizing')
        sleep(1)
        self.c_p['polymerization_LED'] = 'L'

        # Could be image artefacts from polymerization if we don't wait a little bit
        sleep(1)
        # Save position of new area
        self.c_p['polymerized_x'].append(self.c_p['traps_relative_pos'][0][0])
        self.c_p['polymerized_x_piezo'].append(self.c_p['piezo_current_position'][0])
        self.c_p['polymerized_y'].append(self.c_p['traps_relative_pos'][1][0])
        self.c_p['polymerized_y_piezo'].append(self.c_p['piezo_current_position'][1])
        self.save_polymerization_data()
        self.previous_pos[0] = self.c_p['piezo_current_position'][0]
        self.previous_pos[1] = self.c_p['piezo_current_position'][1]

        # Move to a new position
        self.random_piezo_move()
        print('Moving to new piezo position')
        self.update_polymerized_positions()

    def place_anchor(self):

        # TODO fix polymerization in this function. Do it in a loop so it can be cancelled
        self.c_p['piezo_target_position'][2] = 2
        self.c_p['polymerization_LED'] = 'H'
        start_time = time()
        print('Creating anchor')
        while self.c_p['program_running'] and time() < start_time + self.anchor_time:
            sleep(self.sleep_time)
        self.c_p['Anchor_position'] = self.c_p['stepper_current_position'][0:2]
        self.c_p['Anchor_placed'] = True
        self.c_p['polymerization_LED'] = 'L'
        self.c_p['piezo_target_position'][2] = 18
        self.c_p['stepper_starting_position'][0] = self.c_p['stepper_current_position'][0] + (self.c_p['anchor_array_dist'][0] / 1000)
        self.c_p['stepper_starting_position'][1] = self.c_p['stepper_current_position'][1] + (self.c_p['anchor_array_dist'][1] / 1000)
        print('New anchor created')

    def get_next_starting_position(self):
        dx = self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']] - self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']-1]
        dy = self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']] - self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']-1]
        # This is causing trouble with focus
        self.c_p['stepper_starting_position'][0] += dx / 1000
        self.c_p['stepper_starting_position'][1] += dy / 1000
        dx = (self.c_p['stepper_current_position'][0] - self.c_p['stepper_starting_position'][0])
        dy = (self.c_p['stepper_current_position'][1] - self.c_p['stepper_starting_position'][1])
        # dz = (self.c_p['tilt'][0] * dx) + (self.c_p['tilt'][1] * dy)
        # self.c_p['stepper_starting_position'][2] -= dz

    def run(self):
        self.in_rough_location = False
        while self.c_p['program_running']:
            if self.c_p['tracking_on']:
                # TODO make it possible to adjust tracking parameters on the fly

                # Do the particle tracking.
                # Note that the tracking algorithm can easily be replaced if need be
                if self.c_p['background_illumination']:
                    self.check_for_cured_areas()
                else:
                    self.check_for_trapped_QDs()
                    if self.c_p['move_QDs'].get() and self.c_p['QDs_placed'] < len(self.c_p['QD_target_loc_x']):
                        # If we are far from the starting position simply move towards it
                        dx = (self.c_p['stepper_current_position'][0] - self.c_p['stepper_starting_position'][0])**2
                        dy = (self.c_p['stepper_current_position'][1] - self.c_p['stepper_starting_position'][1])**2
                        if dx + dy > 0.005 **2:
                            self.in_rough_location = self.move_QD_to_location(
                             x=self.c_p['stepper_starting_position'][0],
                             y=self.c_p['stepper_starting_position'][1],
                             motor='stepper')

                        if self.in_rough_location:
                            self.c_p['move_QDs'].set(False)
                    # Test the pushing function.
                    if self.c_p['test_move_down'].get():
                        self.push_QD_to_glass(motor='stepper', step=6e-5)

            elif self.c_p['generate_training_data'].get():
                # TODO handle z position
                # Here we run the test version of the final program
                if not self.c_p['Anchor_placed']:
                    self.place_anchor()

                if self.c_p['QDs_placed'] < len(self.c_p['QD_target_loc_x']):
                    # Check for trapped particles
                    print('Checking for trapped QDs')
                    self.check_for_trapped_QDs()
                    # If we are far from the starting position simply move towards it
                    dx = (self.c_p['stepper_current_position'][0] - self.c_p['stepper_starting_position'][0])**2
                    dy = (self.c_p['stepper_current_position'][1] - self.c_p['stepper_starting_position'][1])**2
                    if dx + dy > 0.001**2:
                        print('Far from target loc')
                        self.in_rough_location = self.move_QD_to_location(
                            x=self.c_p['stepper_starting_position'][0],
                            y=self.c_p['stepper_starting_position'][1],
                            motor='stepper')
                        self.fine_position_found = False
                    else: # TODO add correct condition here
                        self.in_rough_location = True
                    # We are close to starting position and have a QD
                    if self.in_rough_location and not self.fine_position_found and not self.in_z_position:
                        # Locate fine position and update targets
                        print('Moving qd to target height')
                        self.in_z_position = self.push_QD_to_glass()#push_QD_down(target_height=18)

                    if self.in_rough_location and not self.fine_position_found and self.in_z_position:
                        print('Doing fine locating of area')
                        motor = self.locate_next_poly_area()
                        # We currently don't check for the case of the anchor not found
                        # We also automatically move the stepper if need be
                        # TODO add

                    elif self.c_p['QD_currently_trapped'] and self.fine_position_found:
                        # We are close to the target and know where to move
                        dist_2 = (self.polymerization_position_piezo[0] - self.c_p['piezo_current_position'][0])**2\
                            + (self.polymerization_position_piezo[1] - self.c_p['piezo_current_position'][1])**2

                        if not self.in_fine_position and dist_2 < 1:
                            motor = self.locate_next_poly_area()
                            dist_2 = (self.polymerization_position_piezo[0] - self.c_p['piezo_current_position'][0])**2\
                                + (self.polymerization_position_piezo[1] - self.c_p['piezo_current_position'][1])**2
                            print('Almost there, missing', dist_2, self.polymerization_position_piezo[0] - self.c_p['piezo_current_position'][0],
                            self.polymerization_position_piezo[1] - self.c_p['piezo_current_position'][1])
                            if motor == 'piezo' and dist_2 < 0.5**2 and self.c_p['QD_currently_trapped']:
                                # We are in target position and ready to stick
                                self.in_fine_position = True
                                self.stick_quantum_dot()
                                self.c_p['QDs_placed'] += 1
                                # Center the piezo
                                self.c_p['piezo_target_position'][0] = 10
                                self.c_p['piezo_target_position'][1] = 10
                                # We are not in the correct position yet
                                self.in_rough_location = False
                                self.fine_position_found = False
                                self.in_fine_position = False
                                self.polymerization_position_piezo = [-1,-1]

                                # Update stepper starting position
                                if self.c_p['QDs_placed'] < len(self.c_p['QD_target_loc_x']):
                                    # TODO may need to adjust this to compensate for the fact that the piezos are moving
                                    self.get_next_starting_position()
                                    self.c_p['piezo_target_position'][2] = 2
                                    self.in_z_position = False

                            else:
                                self.move_QD_to_location(
                                    x=self.polymerization_position_piezo[0],
                                    y=self.polymerization_position_piezo[1],
                                    motor='piezo',
                                    tolerance=0.2,
                                    step=0.5)
                        else:
                            print('Doing the fine piezo move')
                            self.move_QD_to_location(
                                x=self.polymerization_position_piezo[0],
                                y=self.polymerization_position_piezo[1],
                                motor='piezo',
                                tolerance=0.2,
                                step=0.5)
                            # Do a fine move towards the target

                #self.generate_polymerization_training_data()

            sleep(self.sleep_time)
        self.__del__()


    ################################### OLD FUNCTIONS NO LONGER USED #####################################

    def move_to_target_location(self):
        '''
        Function for transporting a quantum dot to target location.
        '''
        print('Trying to move to target location')
        # TODO add so that this function automatically moves the QD to
        # target location Has been solved by setting the paramter
        # 'piezo_move_to_target' to true.
        pass

        # Update target position of piezo in x-direction
        x_move = self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']] - \
             self.c_p['piezo_current_position'][0]
        if x_move < 0:
            self.c_p['piezo_target_position'][0] += max(x_move, -self.c_p['step_size'])
        else:
            self.c_p['piezo_target_position'][0] += min(x_move, self.c_p['step_size'])

        # Update target position of piezo in y-direction
        y_move = self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']] - \
             self.c_p['piezo_current_position'][1]
        if y_move < 0:
            self.c_p['piezo_target_position'][1] += max(y_move, -self.c_p['step_size'])
        else:
            self.c_p['piezo_target_position'][1] += min(y_move, self.c_p['step_size'])


    def push_QD_down(self, target_height=None, step=0.1, motor='piezo', tolerance=0.1):
        '''
        Function for pulling a QD down to target height.
        Inputs:
            Step - How large steps to take
            motor - Which motor to use
            tolerance - how close we need to move
        # Returns False if not Done, True if done, and None if QD was lost.
        '''
        # TODO change so that it pushes as far as possible
        # Check which motor to use
        if motor == 'stepper':
            current_position = 'stepper_current_position'
            target_position = 'stepper_target_position'
        elif motor == 'piezo':
            if self.c_p['piezos_activated'].get():
                print('Please deactivate manual piezo move to allow for automatic control.')
                return None
            current_position = 'piezo_current_position'
            target_position = 'piezo_target_position'
        else:
            return None

        if target_height is None:
            if motor=='piezo':
                target_height = self.c_p['piezo_starting_position'][2]
            else:
                target_height = self.c_p['stepper_starting_position'][2]

        # Check trapped status
        self.trapped_now()
        d = target_height - self.c_p[current_position][2]
        if self.c_p['QD_currently_trapped']:
            self.QD_unseen_counter = 0

            if d < 0:
                self.c_p[target_position][2] = self.c_p[current_position][2] + max(-step, d)
            else:
                self.c_p[target_position][2] = self.c_p[current_position][2] + min(step, d)
            print('Pizeo z move made towards target')
            if np.abs(d) < tolerance:
                print('Piezo movements are done now')
                return True
            else:
                return False

        # TODO move opposite direction to search for QD
        else:
            self.QD_unseen_counter += 1
            # look for other particle
        if self.QD_unseen_counter > 20:
            if d < 0:
                self.c_p[target_position][2] = self.c_p[current_position][2] - max(-step, d)
            else:
                self.c_p[target_position][2] = self.c_p[current_position][2] - min(step, d)

        if self.QD_unseen_counter > 30:
            # TODO move to predetermined safe position.
            # Look for QD, return to safe state
            return None

        return False

    def trap_quantum_dot(self):
        '''
        Function for trapping a quantum dot and trapping it in the area around
        the area it is in now. If there are no quantum dots in the area then
        the program will go look for one.
        '''
        if len(self.c_p['particle_centers'][0]) > 0:
            #if self.c_p[]:
            pass
        else:
            self.look_for_quantum_dot()
            print('Looking for quantum dots')
        pass

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

    def fine_move_QD(self):
        '''

        '''
        # TODO This should perhaps be a loop
        # Calculate where we should put the QD
        print('Recalculating the QD target position')
        x0, y0 = self.find_array_position()
        print('Array moves', x0, y0)
        if x0 is not None:
            if (x0**2 + y0**2) > 0.002 **2:
                X = self.c_p['stepper_current_position'][0] + x0
                Y = self.c_p['stepper_current_position'][1] + y0
                self.c_p['stepper_target_position'][0] = X
                self.c_p['stepper_target_position'][1] = Y
                # self.in_rough_location = self.move_QD_to_location_stepper(
                #  x=X, y=Y)
            elif (x0**2 + y0**2) > 0.0005 **2:
                self.step_move(0, x0)
                self.step_move(1, y0)
            else:
                print('Sticking!')
                self.stick_quantum_dot()
                if self.c_p['QDs_placed'] == len(self.c_p['QD_target_loc_x']):
                    self.c_p['move_QDs'].set(False)
                # Update starting position to move QDs to
                else:
                    print('Updating home location of the stepper.')
                    dx = self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']] - self.c_p['QD_target_loc_x'][self.c_p['QDs_placed']-1]
                    dy = self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']] - self.c_p['QD_target_loc_y'][self.c_p['QDs_placed']-1]
                    # This is causing trouble with focus
                    self.c_p['stepper_starting_position'][0] += dx / 1000
                    self.c_p['stepper_starting_position'][1] += dy / 1000
                    dx = (self.c_p['stepper_current_position'][0] - self.c_p['stepper_starting_position'][0])
                    dy = (self.c_p['stepper_current_position'][1] - self.c_p['stepper_starting_position'][1])
                    dz = (self.c_p['tilt'][0] * dx) + (self.c_p['tilt'][1] * dy)
                    self.c_p['stepper_starting_position'][2] -= dz
