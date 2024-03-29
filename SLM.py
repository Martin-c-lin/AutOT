import numpy as np
import matplotlib.pyplot as plt
from math import ceil,pi
from random import random
from time import time, sleep
from math import atan2
from threading import Thread
import PIL.Image, PIL.ImageTk
from tkinter import *

# TODO: investigate if we can change to smaller
# datatypes to improve performance
# See if smaller datatypes and upscaling can be used to improve performance
# TODO use cupy to implement the calculations on GPU

def atan2_vec_2d(Y, X):
    '''
    Function for calculating atan2 of 2 2d arrays of coordinate positions(X,Y)

    Parameters
    ----------
    Y : 2-d array of y coordinates
    X : 2-d array of x-coordinates

    Returns
    -------
    2d-array of same shape as x and y.
    None if x and y are not of the same shape

    '''
    shape = np.shape(X)

    if np.shape(X) == np.shape(Y):

        Y = np.reshape(Y, shape[0]*shape[1])
        X = np.reshape(X, shape[0]*shape[1])
        res = np.zeros(len(X))
        for idx in range(len(X)):
            res[idx] = atan2(Y[idx], X[idx])
        return np.reshape(res, shape)
    else:
        print('Error, vectors not of equal length', len(X),' is not equal to ',len(Y))
        return None


def get_LGO(image_width=1080, order=-8):
    '''
    Parameters
    ----------
    image_width : TYPE, optional
        DESCRIPTION. The default is 1080.

    Returns
    -------
    LGO : TYPE
        LGO, phase shift required to delta to get a laguerre gaussian instead of a gaussian.

    '''

    xc = image_width / 2
    yc = image_width / 2
    xxx,yyy = np.meshgrid(np.linspace(1, image_width, image_width),
                    np.linspace(1, image_width, image_width))
    LGO = np.mod((order * atan2_vec_2d(yyy-yc, xxx-xc)), (2*pi)) # Ther should maybe be a +pi before mod 2pi

    return LGO


def RM(N, M, Delta,image_width):
    # random mask encoding algorithm  (RM)
    return Delta[np.random.randint(0,M,N),range(N)]


def RS(N, M, Delta):
    # Random Superposition Algorithm (SR)
    RN = np.transpose(np.random.uniform(low=0.0, high=2*pi, size=(1,M)))*np.ones((1,N))
    return np.angle(np.sum(np.exp(1j*(Delta+RN)),axis=0))+pi


def GSW(N, M, Delta=None, image_width=1080, nbr_iterations=30):
    # Weighted Gerchberg-Saxton Algorithm (GSW)
    if Delta is None:
        Delta = SLM.get_delta(image_width=image_width)
    Phi = RS(N, M, Delta) # Initial guess
    W = np.ones((M,1))
    I_m =np.uint8(np.ones((M,1)))
    I_N = np.uint8(np.ones((1,N)))
    Delta_J = np.exp(1j*Delta)
    for J in range(nbr_iterations):
        V = np.reshape(np.mean((np.exp(1j*(I_m*Phi)-Delta)), axis=1), (M, 1))
        V_abs = abs(V)
        W = np.mean(V_abs)*np.divide(W,V_abs)
        Phi = np.angle(sum(np.multiply(Delta_J, np.divide(np.multiply(W, V), V_abs)*I_N)))
        print('Iteration: ', J+1, 'of ', nbr_iterations)

    return np.reshape(128+Phi*255/(2*pi), (image_width, image_width))


def  GS(N, M, Delta=None, image_width=1080, nbr_iterations=30):
    if Delta is None:
        Delta = SLM.get_delta(image_width=image_width)
    Phi = RS(N,M,Delta) # Initial guess
    W = np.ones((M,1))
    I_m =np.uint8(np.ones((M,1)))
    I_N = np.uint8(np.ones((1,N)))
    Delta_J = np.exp(1j*Delta)
    for J in range(nbr_iterations):
        V = np.reshape( np.transpose( np.mean((np.exp(1j*(I_m*Phi)-Delta)),axis=1) ),(M,1))
        Phi = np.angle(sum(np.multiply(Delta_J,np.divide(V,abs(V)))*I_N ))
        print('Iteration: ', J+1, 'of ', nbr_iterations)
    result = np.reshape(128+Phi*255/(2*pi),(image_width,image_width))
    return  result


def get_default_xm_ym():
    '''
    Generates default x,y positions for particle.
    Legacy funciton, use get_xm_ym_rect instead
    '''
    M = 9 # Changed to 9 from 16
    xm = np.zeros((M))
    ym = np.zeros((M))

    d = 65e-6
    d0x=-115e-6
    d0y=-115e-6

    fac = 3
    for i in range(fac):
        xm[i*fac+0] = d0x-d/2
        xm[i*fac+1] = d0x
        xm[i*fac+2] = d0x+d/2
        ym[i*fac+0] = d0y+d/2*(i-1)
        ym[i*fac+1] = d0y+d/2*(i-1)
        ym[i*fac+2] = d0y+d/2*(i-1)

    return xm,ym


def get_three_handle_trap(x0=0, y0=0, alpha=0, center=True, d=31e-6):
    """
    Functions which generates SLM trap positions for three handle particles.
    Inputs:
        x0, yo - Coordinates of center of particle trap
        alpha - rotation of trap config
        center - if there should be a trap for the center block of the particle
        d - length of handles
    Outputs:
        x, y - positions of traps
    """

    # Default position of handles
    xm = [0, -d, d]
    ym = [d, 0, 0]

    # Rotate the handles around the center an angle alpha
    c = np.cos(alpha)
    s = np.sin(alpha)
    R = [[c, s], [-s, c]]
    t = np.dot(R, [xm, ym])

    # Add offset and move to separate vectors
    x = [i+x0 for i in t[:][0]]
    y = [j+y0 for j in t[:][1]]

    # Add center trap if it's needed
    if center:
        x.append(x0)
        y.append(y0)

    return x, y


def hexagonal(d0x, d0y, d):
    '''
    Function which generates a hexagonal pattern with a particle in the center.
    The particle-particle distance is d and the center particle is located at
    (d0x,d0y).
    '''
    from math import pi
    from numpy import cos, sin
    xm = np.zeros(7)
    ym = np.zeros(7)
    xm[0] = d0x
    ym[0] = d0y

    for i in range(6):
        xm[i+1] = d0x + cos(pi*i/3)
        ym[i+1] = d0y + sin(pi*i/3)
    return xm, ym


def hexagonal_lattice(nbr_rows, nbr_columns, d0x, d0y, d):
    '''
    Function which calculates the positions in a hexagonal
    lattice with lower left corner in (d0x,d0y) and
    lattice constant d (particle-particle distance d).
    '''
    from numpy import mod, sqrt
    nbr_particles = nbr_rows * nbr_columns
    xm = np.zeros(nbr_particles)
    ym = np.zeros(nbr_particles)
    for r in range(nbr_rows):
        for c in range(nbr_columns):
            ym[r*nbr_columns+c] = d0y + (d * r * sqrt(3) / 2)
            if mod(r,2) == 1:
                xm[r*nbr_columns+c] = d0x + d * c + d/2
            else:
                xm[r*nbr_columns+c] = d0x + d * c
    return xm, ym


def get_xm_ym_rect(nbr_rows,nbr_columns, dx=30e-6,dy=30e-6, d0x=-115e-6, d0y=-115e-6):
    '''
    Generates xm,ym in a rectangular grid with a particle-particle distance of d.
    '''
    if nbr_rows < 1 or nbr_columns < 1:
        return [],[]
    xm = np.zeros((nbr_rows * nbr_columns))
    ym = np.zeros((nbr_rows * nbr_columns))

    for i in range(nbr_rows):
        for j in range(nbr_columns):
            xm[i*nbr_columns+j] = d0x + dx*j
            ym[i*nbr_columns+j] = d0y + dy*i
    return xm,ym


def get_xm_ym_E_L_Triangle(d=30e-6, d0x=-115e-6, d0y=-115e-6):
    '''
    Generates xm,ym in a equilateral triangle with sidelength d and first
    corner placed on (d0x,d0y)
    '''
    xm = np.zeros(3)
    ym = np.zeros(3)

    xm[0] = d0x
    ym[0] = d0y
    xm[1] = d0x
    ym[1] = d0y + d
    xm[2] = d0x + np.sqrt(3/4) * d
    ym[2] = d0y + d/2
    return xm, ym


def get_xm_ym_triangle_with_center(d=30e-6, d0x=-115e-6, d0y=-115e-6):
        '''
        Generates xm,ym in a equilateral triangle with sidelength d and a
        particle in the center at d0x,d0y
        '''
        xm = np.zeros(4)
        ym = np.zeros(4)

        xm[0] = d0x
        ym[0] = d0y

        xm[1] = d0x - d / np.sqrt(12)
        ym[1] = d0y + d / 2
        xm[2] = d0x - d / np.sqrt(12)
        ym[2] = d0y - d / 2
        xm[3] = d0x + d * (np.sqrt(3/4) - np.sqrt(1/12))
        ym[3] = d0y

        return xm, ym


def get_Isaac_xm_ym(d=30e-6, d0x=-115e-6, d0y=-115e-6):
    '''
    Two particles, first placed at [d0x, d0y] other at [d0x, d0y + d]
    '''
    xm = np.zeros((2))
    ym = np.zeros((2))

    xm[0] = d0x
    xm[1] = d0x

    ym[0] = d0y
    ym[1] = d0y+d

    return xm, ym


def compensate_z(xm, ym, zm, x_comp, y_comp):
    # Function for compensating shift in lateral positon of beam due to changes
    # in z.
    xm = [x-x_comp*z for x,z in zip(xm, zm)]
    ym = [y-y_comp*z for y,z in zip(ym, zm)]
    return xm, ym


def get_delta(image_width = 1080, xm=[], ym=[], zm=None, use_LGO=[False], order=-8,
    x_comp=None, y_comp=None):
    """
    Calculates delta in paper. I.e the phase shift of light when travelling from
    the SLM to the trap position for a specific set of points
    Default parameters copied from Allessandros script
    """
    # TODO seems a bit tricky to change the size to non-square :/  could maybe have a
    # bigger screen and cut out a piece of it
    x = np.linspace(1,image_width,image_width)
    y = np.reshape(np.transpose(np.linspace(1,image_width,image_width)),(image_width,1))

    I = np.ones((1,image_width))
    N = image_width**2
    p = 9e-6 # pixel size
    f = np.sqrt(2e-4*0.4) # Focal length of imaging system. Empirically found value
    z = 0
    lambda_ = 532e-9 # Laser wavelength

    if len(xm)<1 or len(ym)<1:
        xm,ym = get_default_xm_ym()
        use_LGO = [False for i in range(len(xm))]
    # TODO make the order into a list
    if True in use_LGO:
        LGO = get_LGO(image_width,order=order)
    M = len(xm) # Total number of traps

    # Initiate zm if not provided by user.
    if zm is None:
        zm = np.zeros((M))
    # Compensate for shift in x-y plane when changing z
    zm = zm[:M]
    if x_comp is not None:
        xm, ym = compensate_z(xm, ym, zm, x_comp, y_comp)
        print('compensated')
    Delta=np.zeros((M,N))
    for m in range(M):

        # Calculate delta according to eq : in paper
        # Using python "%" instead of Matlabs "rem"
        Delta[m,:]=np.reshape(2*pi*p/lambda_/f*((np.transpose(I)*x*xm[m]+(y*I)*ym[m]) + 1/(2*f)*zm[m] * ( (np.transpose(I)*x)**2 + (y*I)**2 )) % (2*pi),(1,N))
        if len(use_LGO)>m and use_LGO[m]:
            Delta[m,:] += np.reshape(LGO,(N))
            Delta[m,:] = Delta[m,:] % (2*pi)
    return Delta, N, M


def setup_fullscreen_plt_image():
    '''
    This script magically sets up pyplot lib so it displays an image on a secondary display
    in full screen.
    Legacy function. USe the SLM_controller script with tkinter windows instead
    '''
    plt.switch_backend('QT4Agg')

    # a little hack to get screen size; from here [1]
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    py = mgr.canvas.height()
    px = mgr.canvas.width()
    mgr.window.close()
    # hack end

    plt.figure()
    plt.rcParams['toolbar'] = 'None'
    fig = plt.gcf()
    fig.canvas.window().statusBar().setVisible(False)
    fig.set_size_inches(8,8)
    figManager = plt.get_current_fig_manager()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # if px=0, plot will display on 1st screen
    figManager.window.move(px, 0)
    figManager.window.showMaximized()
    figManager.window.setFocus()

def SLM_loc_to_trap_loc(c_p, xm, ym):
    '''
    Fucntion for updating the traps position based on their locaitons
    on the SLM.
    '''
    # TODO this should not be both here and in QD positioning
    tmp_x = [x * c_p['slm_to_pixel'] + c_p['slm_x_center'] for x in xm]
    tmp_y = [y * c_p['slm_to_pixel'] + c_p['slm_y_center'] for y in ym]
    tmp = np.asarray([tmp_x, tmp_y])
    c_p['traps_absolute_pos'] = tmp
    print('Traps are at: ', c_p['traps_absolute_pos'] )

class CreatePhasemaskThread(Thread):
    def __init__(self, threadID, name, c_p):
        '''
        Thread for controlling the SLM creation. When new_phasemask set to true
        the phasemask is updated.
        Parameters
        ----------
        threadID : int
            Thread number.
        name : string
            Name of thread.

        Returns
        -------
        None.

        '''
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.setDaemon(True)
        self.c_p = c_p

    def run(self):
        # TODO use cupy if available
        '''
        Thread calculates the new phasemask when the parameter 'new phasemask'
        is set to true. It does this using the control parameters (xm, ym) for
        particle positions. use_LGO to determine if to use LGO or not.

        '''
        c_p = self.c_p
        c_p['xm'], c_p['ym'] = get_default_xm_ym()
        c_p['zm'] = np.zeros(len(c_p['xm']))
        Delta, N, M = get_delta(xm=c_p['xm'], ym=c_p['ym'], zm=c_p['zm'],
            use_LGO=c_p['use_LGO'],
            order=c_p['LGO_order'])

        c_p['phasemask'] = GSW(
            N, M, Delta, nbr_iterations=c_p['SLM_iterations'])

        c_p['phasemask_updated'] = True
        SLM_loc_to_trap_loc(c_p, xm=c_p['xm'], ym=c_p['ym'])

        c_p['traps_occupied'] =\
            [False for i in range(len(c_p['traps_absolute_pos'][0]))]

        while c_p['program_running']:
            if c_p['new_phasemask']:
                # Calcualte new delta and phasemask
                Delta, N, M = get_delta(xm=c_p['xm'], ym=c_p['ym'],
                    zm=c_p['zm'],
                    use_LGO=c_p['use_LGO'],
                    order=c_p['LGO_order'])
                if M==2:
                    print('Using normal Grechbgerg-Saxton since there are 2 traps')
                    c_p['phasemask'] = GS(
                        N, M, Delta,
                        nbr_iterations=c_p['SLM_iterations'])
                else:
                    c_p['phasemask'] = GSW(
                        N, M, Delta,
                        nbr_iterations=c_p['SLM_iterations'])
                # if c_p['save_phasemask']:
                #     save_phasemask()
                c_p['phasemask_updated'] = True
                c_p['new_phasemask'] = False

                # Update the number of traps and their position
                SLM_loc_to_trap_loc(c_p, xm=c_p['xm'], ym=c_p['ym'])
                print(c_p['traps_absolute_pos'])
                c_p['traps_occupied'] =\
                    [False for i in range(len(c_p['traps_absolute_pos'][0]))]
            sleep(0.5)

class SLM_window(Frame):
    def __init__(self, master, c_p, SLM_width=1920, SLM_height=1080):
        Frame.__init__(self, master)
        self.master = master
        self.c_p = c_p
        self.SLM_width, self.SLM_height = SLM_width, SLM_height
        self.master.overrideredirect(1)
        # TODO, have SLM position as parameters
        self.master.geometry("%dx%d+1920+0" % (SLM_width, SLM_height))
        self.master.focus_set()
        self.canvas = Canvas(self.master, width=self.SLM_width, height=self.SLM_height)
        self.canvas.pack()
        self.canvas.configure(background='black')
        self.image=None
        self.update()

    def resize_pilImage(self, pilImage):
        """

        """
        imgWidth, imgHeight = pilImage.size
        if imgWidth > self.SLM_width or imgHeight > self.SLM_height:
            ratio = min(self.SLM_width/imgWidth, self.SLM_height/imgHeight)
            imgWidth = int(imgWidth*ratio)
            imgHeight = int(imgHeight*ratio)
            pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
        return pilImage

    def update(self):
        """
        Updates the image on display to the latest phasemask
        """
        del self.image
        pilImage = PIL.Image.fromarray(self.c_p['phasemask'])
        pilImage = self.resize_pilImage(pilImage)
        self.image = PIL.ImageTk.PhotoImage(pilImage)
        imagesprite = self.canvas.create_image(self.SLM_width/2,self.SLM_height/2,image=self.image)
