import numpy as np
from skimage.measure import block_reduce
import cv2
import os
import skvideo.io
from scipy import ndimage
import cupy as cp
from numba import jit


#@jit(nopython=True)
def subtract_bg(I1, I2):
    # TOOD should not rely on cupy exclusively for this.
    assert np.shape(I1) == np.shape(I2), "Images shapes don't match!"
    I1 = cp.asarray(I1)
    I2 = cp.asarray(I2)
    image = I1 - I2
    image += 120
    image[image<0] = 0
    image[image>255] = 255 # TODO fix so that this does not cause overflow in snapshot function.
    # snapshot
    return cp.asnumpy(image)


# TODO replace downsamplename  with binning
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

def convolve_binning(image, filter_size=4,):
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

def npy_generator(path):
    """
    Used to read all the images in a npy image folder one at a time. Takes the
    full path as input and outputs an image. Outputs None if there are no more
    images to read.
    """

    nbr_frames = 0
    directory = os.listdir(path)
    done = False
    num = '0' # First frame to load

    while not done:
        done = True
        for file in directory:
            idx = file.find('-')
            if file[:idx] == num and file[-4:] == '.npy':
                images = np.load(path+file)
                num = file[idx+1:-4]
                done = False
                for image in images:
                    yield image
    while True:
        yield None

def create_video_from_np(path, video_name='video.mp4', downsample_factor=4,
    threshold=50, downsample=True, format="AVI"):
    video_name = path + video_name
    video = skvideo.io.FFmpegWriter(video_name,outputdict={
                                         '-b':'1000000000', # 6000000000
                                         '-r':'25', # Does not like this
                                         # specifying codec and bitrate, 'vcodec': 'libx264',
                                        })
    # TODO rewrite this to use the npy generator instead
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
                        if downsample and np.mean(image) < threshold: # should it  not be less than?
                            tmp = convolve_Sample(image, filter_size=downsample_factor) # [0:1920,0:1080]
                        else:
                            tmp = image
                        video.writeFrame(tmp)
                        nbr_frames += 1

    video.close()


def create_video_from_np_bg_subtract(path, video_name='video.mp4', downsample_factor=4,
    threshold=50, downsample=True, format="AVI"):
    video_name = path + video_name
    video = skvideo.io.FFmpegWriter(video_name,outputdict={
                                         '-b':'1000000000', # 6000000000
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
                #print(file)
                bg = np.int16(images[5])
                print(np.shape(bg), np.mean(bg))
                for image in images:
                    if np.mean(image)>0:
                        if downsample and np.mean(image) < threshold:
                            tmp_0 = np.subtract(np.int16(image), bg)
                            tmp_0[tmp_0<0] = 0
                            print(np.mean(tmp_0), np.mean(image))
                            tmp = convolve_Sample(tmp_0, filter_size=downsample_factor) # [0:1920,0:1080]
                        else:
                            tmp = image
                        video.writeFrame(tmp)
                        nbr_frames += 1
                        #bg = image

    video.close()


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
