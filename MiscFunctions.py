import numpy as np
from skimage.measure import block_reduce
import cv2

def sum_downsample(image, filter_size=4, lim=255, contrast=1):
    """
    Downsamples an array with the factor given by filter size
    """
    new_array = np.int16(image)
    new_array = block_reduce(new_array, (filter_size, filter_size), np.sum)
    #
    blur = cv2.blur(new_array,(100, 100))
    new_array -= np.int16(blur) # Why did this have opposite effect?
    new_array -= np.min(new_array)
    new_array = new_array % 256
    return np.uint8(new_array)

# TODO Try Andreas idea about rolling average(or sum)

def max_downsample(image, filter_size=4, lim=255):
    """
    Downsamples an array with the factor given by filter size
    """
    new_array = np.int16(image)
    new_array -= np.uint8(np.mean(new_array))
    new_array = block_reduce(image, (filter_size, filter_size), np.max)
    new_array *= 4 #-50
    new_array -= 50
    new_array[new_array<0] = 0

    new_array[new_array>lim] = 255#lim
    return np.uint8(new_array)
