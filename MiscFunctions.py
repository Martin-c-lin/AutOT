import numpy as np
from skimage.measure import block_reduce

def sum_downsample(image, filter_size=4, lim=255):
    """
    Downsamples an array with the factor given by filter size
    """
    new_array = np.int16(image)
    new_array -= np.uint8(np.mean(new_array))
    new_array = block_reduce(image, (filter_size, filter_size), np.sum)
    new_array[new_array<0] = 0
    new_array[new_array>lim] = lim
    return np.uint8(new_array)
