import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
from QD_tracking import fourier_filter, normalize_image, find_QDs

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

def create_gpu_mask(h,w,inner_radius=10, outer_radius=180):
    m_i = create_circular_mask(h,w,radius=inner_radius)
    m_o = create_circular_mask_inverse(h,w,radius=outer_radius)
    mask = np.ones((h,w))
    mask[m_i] = 0
    mask[m_o] = 0
    return cp.fft.ifftshift(cp.asarray(mask))

image_shape = [992, 992]
filter_sizes = [10, 180]

gpu_mask = create_gpu_mask(image_shape[0], image_shape[1], filter_sizes[0], filter_sizes[1])

def pycu_fourier_filter(image, filter_bounds=[10, 200]):
    '''
    Uses a "fourier filter" on the image to easier locate particles.
    Also normalizes the result to be within [0,1]
    '''
    global image_shape, filter_sizes, gpu_mask
    shape = cp.shape(image)
    if not shape[0] == image_shape[0] or not shape[1] == image_shape[1] or not \
        filter_sizes[0] == inner_filter_width or not filter_sizes[0] == outer_filter_width:
        image_shape[0] = shape[0]
        image_shape[1] = shape[1]
        filter_sizes[0] = filter_bounds[0]
        filter_sizes[1] = filter_bounds[1]
        gpu_mask = create_gpu_mask(image_shape[0], image_shape[1],
            filter_sizes[0], filter_sizes[1])
        print('new mask created')

    fft_frame_gpu = cp.fft.fft2(image)
    fft_frame_gpu = cp.multiply(fft_frame_gpu, gpu_mask)
    fft_frame_gpu = cp.absolute(cp.fft.ifft2(fft_frame_gpu))
    fft_frame_gpu = fft_frame_gpu.astype(cp.float32)
    fft_frame_gpu = cp.subtract(fft_frame_gpu, cp.amin(fft_frame_gpu))
    fft_frame_gpu = cp.divide(fft_frame_gpu,cp.amax(fft_frame_gpu))

    return fft_frame_gpu

def pycu_extract_particle_positions(image, threshold=0.12, size_bounds=[30,1500]):
    # TODO add edges
    labelled = ndimage.label(cp.greater(image, threshold))
    # Check each different class
    nbr_classes = labelled[1]
    x_positions = []
    y_positions = []
    for idx in range(nbr_classes): # check indexing
        tmp = cp.nonzero(cp.equal(labelled[0], idx+1))
        tot = cp.shape(tmp[1])[0]
        if size_bounds[0] < tot and size_bounds[1] > tot:
            x_positions.append(cp.sum(tmp[0]) / tot)
            y_positions.append(cp.sum(tmp[1]) / tot)
    return x_positions, y_positions, labelled[0]

def pycu_fourier_tracking(image, threshold=0.12, size_bounds=[30,1500], filter_bounds=[10, 200],
        edge=80):
    gpu_frame = cp.asarray(image)
    gpu_frame = pycu_fourier_filter(gpu_frame, filter_bounds)
    x,y,ret = pycu_extract_particle_positions(gpu_frame[edge:-edge,edge:-edge],
        threshold, size_bounds)
    # px = [s[1] - x_i - edge for x_i in x]
    # py = [s[0] - y_i - edge for y_i in y]
    px = [xi + edge for x_i in x]
    py = [xi + edge for y_i in y]
    return px, py
