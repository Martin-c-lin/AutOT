import numpy as cp
import cupy as cp
from math import pi

def RM(N, M, Delta, image_width):
    pass

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
    shape = cp.shape(X)

    if cp.shape(X) == cp.shape(Y):

        Y = cp.reshape(Y, shape[0]*shape[1])
        X = cp.reshape(X, shape[0]*shape[1])
        res = cp.zeros(len(X))
        for idx in range(len(X)):
            res[idx] = cp.arctan2(Y[idx], X[idx])
        return cp.reshape(res, shape)
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
    xxx,yyy = cp.meshgrid(cp.linspace(1, image_width, image_width),
                    cp.linspace(1, image_width, image_width))
    LGO = cp.mod((order * atan2_vec_2d(yyy-yc, xxx-xc)), (2*pi)) # Ther should maybe be a +pi before mod 2pi

    return LGO


def RS(N, M, Delta):
    # TODO make sure that * is run correctly on GPU
    RN = cp.transpose(cp.random.uniform(low=0.0, high=2*pi, size=(1, M)))*\
                        cp.ones((1, N))
    #return cp.angle(cp.sum(cp.exp(1j*(Delta+RN), axis=0 )))+ pi
    return cp.angle(cp.sum(cp.exp(1j*(Delta+RN))))+ pi

def compensate_z(xm, ym, zm, x_comp, y_comp):
    # Function for compensating shift in lateral positon of beam due to changes
    # in z.
    xm = [x-x_comp*z for x,z in zip(xm, zm)]
    ym = [y-y_comp*z for y,z in zip(ym, zm)]
    return xm, ym


# TODO: A fair bit of the initalization could be moved out from functions to
# save run-time


def get_delta(image_width = 1080, xm=[], ym=[], zm=None, use_LGO=[False], order=-8,
    x_comp=None, y_comp=None):
    x = cp.linspace(1, image_width, image_width)
    y = cp.reshape(cp.transpose(cp.linspace(1, image_width, image_width)),(image_width, 1))

    I = cp.ones((1,image_width))
    N = image_width**2
    p = 9e-6 # pixel size
    f = cp.sqrt(1e-4) #cp.sqrt(2e-4*0.4) # Focal length of imaging system. Empirically found value
    z = 0
    lambda_ = 532e-9 # Laser wavelength

    if len(xm) < 1 or len(ym) < 1:
        xm,ym = get_default_xm_ym()
        use_LGO = [False for i in range(len(xm))]
    # TODO make the order into a list
    if True in use_LGO:
        LGO = get_LGO(image_width,order=order)
    M = len(xm) # Total number of traps

    # Initiate zm if not provided by user.
    if zm is None:
        zm = cp.zeros((M))
    # Compensate for shift in x-y plane when changing z
    zm = zm[:M]
    if x_comp is not None:
        xm, ym = compensate_z(xm, ym, zm, x_comp, y_comp)
        print('compensated')
    Delta = cp.zeros((M,N))
    prefactor = 2*pi*p/lambda_/f
    tmp = cp.multiply(cp.transpose(I), x)
    for m in range(M):
        Delta[m,:] = cp.reshape(prefactor*((cp.transpose(I)*x*xm[m]+(y*I)*ym[m]) \
         + 1/(2*f)*zm[m] * ( (cp.transpose(I)*x)**2 + (y*I)**2 )) % (2*pi),(N))
        if len(use_LGO) > m and use_LGO[m]:
            Delta[m,:] += cp.reshape(LGO,(N))
            Delta[m,:] = cp.mod(Delta[m,:], (2*pi))
        pass
    return Delta, N, M


def GSW(N, M, Delta=None, image_width=1080, nbr_iterations=30):
    # Weighted Gerchberg-Saxton Algorithm (GSW)
    if Delta is None:
        Delta = SLM.get_delta(image_width=image_width)
    Phi = RS(N, M, Delta) # Initial guess
    W = cp.ones((M,1))
    I_m = cp.ones((M,1))# cp.uint8(cp.ones((M,1)))
    I_N = cp.ones((1,N))
    Delta_J = cp.exp(1j*Delta)
    for J in range(nbr_iterations):
        V = cp.reshape(cp.mean((cp.exp(1j * (I_m * Phi) - Delta)), axis=1), (M, 1))
        V_abs = abs(V) # TODO check if we can use cp abs and cp sum to speed up further
        W = cp.mean(V_abs) * cp.divide(W,V_abs)
        Phi = cp.angle(sum(cp.multiply(Delta_J, cp.divide(cp.multiply(W, V), V_abs)*I_N)))
        #print('Iteration: ', J+1, 'of ', nbr_iterations)

    return cp.reshape(128+Phi*255/(2*pi), (image_width, image_width))


def  GS(N, M, Delta=None, image_width=1080, nbr_iterations=30):
    # Grechberg Saxton algorithm
    if Delta is None:
        Delta = SLM.get_delta(image_width=image_width)
    Phi = RS(N,M,Delta) # Initial guess
    W = cp.ones((M,1))
    I_m = cp.ones((M,1))# cp.uint8(cp.ones((M,1)))
    I_N = cp.ones((1,N))
    Delta_J = cp.exp(1j*Delta)
    for J in range(nbr_iterations):
        V = cp.reshape(cp.transpose(cp.mean((cp.exp(1j*(I_m*Phi)-Delta)), axis=1)), (M, 1))
        Phi = cp.angle(sum(cp.multiply(Delta_J, cp.divide(V, abs(V)))*I_N ))
        print('Iteration: ', J+1, 'of ', nbr_iterations)
    result = cp.reshape(128+Phi*255/(2*pi),(image_width,image_width))
    return  result
