# Algorithm for locating the quantum dots
import numpy as np
from find_particle_threshold import find_particle_centers
from numba import jit
from threading import Thread
from time import sleep, time, strftime
import pyfftw
from tkinter import BooleanVar
from scipy.spatial import distance_matrix
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
             particle_upper_size_threshold=400, edge=80):
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
    x, y, ret_img = find_particle_centers(image[edge:-edge,edge:-edge],
                                        threshold=threshold,
                                        particle_size_threshold=particle_size_threshold,
                                        particle_upper_size_threshold=particle_upper_size_threshold)

    px = [s[1] - x_i - edge for x_i in x]
    py = [s[0] - y_i - edge for y_i in y]
    return px, py, ret_img


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
        'polymerized_x': [],
        'polymerized_y': [],
        'nbr_quantum_dots_stuck': 0,  # number of quantum dots already in
        # position
        'step_size': 0.2,  # Step size to move the quantum dots
        'tolerance': 0.01,
        'position_QD_in_pattern': BooleanVar(),
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
        # TODO add so that this function automatically moves the QD to
        # target location Has been solved by setting the paramter
        # 'piezo_move_to_target' to true.
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


    def look_for_quantum_dot(self, x, y):
        '''
        Takes a move to look for a quantum dot
        '''
        # If we have moved less than 5 microns in the z- direction we may look along z
        dx = self.c_p['stepper_current_position'][0] - x
        dy = self.c_p['stepper_current_position'][1] - y
        dz = self.c_p['stepper_current_position'][2] - self.c_p['stepper_starting_position'][2]
        # Check if we are too far from original position
        if np.abs(dx) > 0.5:
            dx = -dx
        if np.abs(dy) > 0.5:
            dy = -dy
        dx /= np.abs(dx)
        dy /= np.abs(dy)

        choice = np.random.randint(-1,10)
        if choice == 1:
            self.c_p['stepper_target_position'][2] += 0.0005
        elif 0<= choice:
            self.c_p['stepper_target_position'][2] -= 0.0005
            print('Changing focus a bit')
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
        if dz > -0.02 or dz > 0.01:
            self.c_p['stepper_target_position'][2] = self.c_p['stepper_starting_position'][2] - 0.005

    def move_QD_to_location_rough(self, x, y, step = 0.0003):
        '''
        Algorithm for pulling a quantum dot to the position specified in x,y
        using the stepper only.
        Uses the following method
         Check if quantum dot is trapped:
          QD trapped ->take step towards target location
          QD not trapped:
            check if QD in view:
              QDs in view -> move to trap the QD
              QD not in view -> Look for QD with look_for_quantum_dot function
        '''
        self.trapped_now()
        if self.c_p['closest_QD'] is not None:
            self.QD_unseen_counter = 0
            # Calcualte distance to target position
            if self.c_p['QD_currently_trapped']:
                d = [x - self.c_p['stepper_current_position'][0], y - self.c_p['stepper_current_position'][1]]
                print('Moving QD towards target', d, x , y )
            elif not self.c_p['QD_currently_trapped'] and self.c_p['QD_trapped']:
                d = [0, 0] # Wait for qd to be trapped
            else:
                # No QD is trapped but there are other visible in frame
                dx = self.c_p['particle_centers'][0][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][0][0]
                dy = self.c_p['particle_centers'][1][self.c_p['closest_QD']] - self.c_p['traps_relative_pos'][1][0]
                # TODO finish this function, increase step?
                d = [dx/self.c_p['mmToPixel'], dy/self.c_p['mmToPixel']]

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
            #print('QD currently in trap:',self.c_p['QD_currently_trapped'], 'QD recently in trap', self.c_p['QD_trapped'])
            return False

        else:
            # Could be that tracking is not perfect and a QD is missed one frame or two
            # therefore the cunter is used
            self.QD_unseen_counter += 1
            # look for other particle
            if self.QD_unseen_counter > 15:
                 self.look_for_quantum_dot(x, y)
            return None

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

    def step_move(self, axis, distance):
        # Makes a smart move in of length distance[mm] along axis
        # axis = 0 => x axis = 1 = y axis = 2 => z
        if self.c_p['stage_piezos']:
            new_position_piezo = self.c_p['piezo_current_position'][axis] + (distance/1000)
            if 0 < new_position_piezo < 20:
                self.c_p['piezo_target_position'][axis] = new_position_piezo
            else:
                self.c_p['stepper_target_position'][axis] = self.c_p['stepper_current_position'][axis] + distance
        elif self.c_p['using_stepper_motors']:
            self.c_p['stepper_target_position'][axis] = self.c_p['stepper_current_position'][axis] + distance

    def move_to_array_position(self):
        '''


        Returns
        -------
        None.

        '''
        self.c_p['polymerized_x'], self.c_p['polymerized_y'], tmp = find_QDs(
        self.c_p['image'], inner_filter_width=5, outer_filter_width=140,
        particle_size_threshold=1000, particle_upper_size_threshold=6400,threshold=0.11)
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
              if min(dist) < 600:
                  # The laser is close to an already polymerized area, move away from it
                  dx -= self.c_p['QD_position_screen_y'][self.c_p['QDs_placed']+1] - self.c_p['QD_position_screen_y'][self.c_p['QDs_placed']]
                  dy -= self.c_p['QD_position_screen_x'][self.c_p['QDs_placed']+1] - self.c_p['QD_position_screen_x'][self.c_p['QDs_placed']]
                  print('Moving away from already polymerized area', dx, dy)
            dx /= self.c_p['mmToPixel']
            dy /= self.c_p['mmToPixel']
            self.step_move(0, -dx)
            self.step_move(1, -dy)

        # TODO Can we decrease minimum step in thorlabs motor?

    def update_polymerized_positions(self):
        '''
        Updates the relative position of the polymerized areas in view.
        Relies on the piezo for accurat movement.
        '''
        x_diff = self.c_p['mmToPixel'] / 1000.0 * (self.previous_pos[0] - \
            self.c_p['piezo_current_position'][0])
        y_diff = self.c_p['mmToPixel'] / 1000.0 * (self.previous_pos[1] - \
            self.c_p['piezo_current_position'][1])

        self.c_p['polymerized_x'] = [x + x_diff for x in self.c_p['polymerized_x']]
        self.c_p['polymerized_y'] = [y + y_diff for y in self.c_p['polymerized_y']]

    def save_polymerization_data(self):
        # Function for saving all the data necessary for training a neural
        # network to detect polymerized locations.

        # Wait for next frame
        sleep(self.c_p['exposure_time'] / 1e6)
        data_name = self.c_p['recording_path'] + "/training_data-" + \
                    strftime("%d-%m-%Y-%H-%M-%S")
        z_diff = self.c_p['piezo_current_position'][2] - self.c_p['piezo_starting_position'][2]
        data = [self.c_p['image'], self.c_p['polymerized_x'],
                self.c_p['polymerized_y'], z_diff]
        np.save(data_name, data)

    def random_piezo_move(self, safe_distance=110, piezo_tolerance=0.2):
        '''
        Function for moving to a new position where there is nothing polymerized
        using the piezos
        '''
        position_found = False
        # Generate new position
        while not position_found:
            position_ok = True
            x = np.random.uniform(0, 20)
            y = np.random.uniform(0, 20)
            z = np.random.uniform(0, 20)

            # Calcualte distance to new positon
            dx = (x - self.c_p['piezo_current_position'][0]) * self.c_p['mmToPixel'] / 1000
            dy = (y - self.c_p['piezo_current_position'][1]) * self.c_p['mmToPixel'] / 1000

            # check if new positon is ok
            for xi, yi in zip(self.c_p['polymerized_x'], self.c_p['polymerized_y']):
                x_sep = self.c_p['traps_relative_pos'][0][0] + dx -xi
                y_sep = self.c_p['traps_relative_pos'][1][0] + dy -yi
                if x_sep**2 + y_sep**2 > safe_distance**2:
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
            self.c_p['stepper_target_position'][0] += 0.01

            # Wait for stepper to finish moving
            while self.c_p['stepper_target_position'][0] - self.c_p['stepper_current_position'][0] > 0.01:
                sleep(self.sleep_time)
                print('Waiting for stepper to move')
                if not self.c_p['program_running']:
                    return

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
        # Save position of new area
        self.c_p['polymerized_x'].append(self.c_p['traps_relative_pos'][0][0])
        self.c_p['polymerized_y'].append(self.c_p['traps_relative_pos'][1][0])
        self.save_polymerization_data()
        self.previous_pos[0] = self.c_p['piezo_current_position'][0]
        self.previous_pos[1] = self.c_p['piezo_current_position'][1]

        # Move to a new position
        self.random_piezo_move()
        print('Moving to new piezo position')
        self.update_polymerized_positions()

    def run(self):

        while self.c_p['program_running']:
            if self.c_p['tracking_on']:
                # TODO make it possible to adjust tracking parameters on the fly
                # Do the particle tracking.
                # Note that the tracking algorithm can easily be replaced if need be
                x, y, tmp = find_QDs(self.c_p['image'],
                                    inner_filter_width=7, outer_filter_width=180,
                                    particle_upper_size_threshold=700) # 12 240
                self.c_p['particle_centers'] = [x, y]
                # This sort of works for the polymerized areas.
                if self.c_p['position_QD_in_pattern'].get():
                    self.move_to_array_position()

                if self.c_p['move_QDs'].get():
                    tmp = self.move_QD_to_location_rough(
                     x=self.c_p['stepper_starting_position'][0],
                     y=self.c_p['stepper_starting_position'][1])
                    if tmp:
                        self.c_p['move_QDs'].set(False)
            elif self.c_p['generate_training_data'].get():
                self.generate_polymerization_training_data()

            sleep(self.sleep_time)
        self.__del__()
