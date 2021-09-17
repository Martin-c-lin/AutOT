import numpy as np
from skimage.measure import block_reduce
import cv2
import os
import skvideo.io
from scipy import ndimage

from numba import jit
#@jit(nopython=True)
def subtract_bg(I1, I2):
    # could potentially do this in cupy
    assert np.shape(I1) == np.shape(I2)
    image = np.array(I1 - I2)
    image += 120
    image[image<0] = 0
    return image


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

def convolve_Sample(image, filter_size=4,):
    """
    Convolves an image with a filter and then subtracts a ... to make it easier to see quantum dots and similar faint objects
    """
    a = np.ones([filter_size,filter_size])
    new_array = np.int16(image)
    new_array = ndimage.convolve(new_array, a, mode='constant')

    blur = cv2.blur(new_array,(50, 50))
    new_array -= np.int16(blur)
    new_array -= np.min(new_array)
    new_array = new_array % 256
    return np.uint8(new_array)


def create_video_from_np(path, video_name='video.mp4', downsample_factor=4,
    threshold=50, downsample=True, format="AVI"):
    video_name = path + video_name
    video = skvideo.io.FFmpegWriter(video_name,outputdict={
                                         '-b':'6000000000',
                                         '-r':'25', # Does not like this
                                         # specifying codec and bitrate, 'vcodec': 'libx264',
                                        })
    nbr_frames = 0
    directory = os.listdir(path)
    done = False
    num = '0' # First frame to load
    # TODO add subtract b_g option
    # Look for the first file and then successively load more of them
    while not done:
        done = True
        for file in directory:
            idx = file.find('-')
            if file[:idx] == num and file[-4:] == '.npy':
                images = np.load(path+file)
                num = file[idx+1:-4]
                done = False
                print(file)
                for image in images:
                    if np.mean(image)>0:
                        if downsample and np.mean(image) > threshold:
                            tmp = convolve_Sample(image, filter_size=downsample_factor) # [0:1920,0:1080]
                        else:
                            tmp = image
                        video.writeFrame(tmp)
                        nbr_frames += 1

    video.close()
