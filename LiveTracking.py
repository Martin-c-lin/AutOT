from threading import Thread
import time
import os
from time import sleep
import cv2
import sys
# TODO will need to update to latest deeptrack here and use it instead
sys.path.append("C:/Users/Martin/OneDrive/PhD/Misc/python_environments/dt2_1_autotracker/deeptrack2.1/")
import numpy as np
import deeptrack as dt
import skimage.color


class LiveTrackerThread(Thread):
    """
    A class for continously tracking the input from the camera. Will output it
    to the particle_centers of the c_p

    Uses an arbitrary tracking function but the autotracker is recommended and
    used by default
    """
    def __init__(self , threadID, name, c_p):
        Thread.__init__(self)
        self.c_p = c_p
        self.setDaemon(True)

    def run(self):
        while c_p['program_running']:
            if c_p['tracking_on']:
                try:
                    self.c_p['particle_centers'] = self.c_p['tracking_function'](self.c_p['image'])
                except:
                    pass


def deeptrack_prediction(frames, model, image_normalizer):
    return model.predict(image_normalizer(frames))


def get_image_normalizer():
    image_normalizer = (
    dt.DummyFeature() >>
    skimage.color.rgb2gray >> # may not be needed for some of this
    (lambda x: cv2.resize(np.array(x), (IMAGE_SIZE_1, IMAGE_SIZE))[..., np.newaxis]) >>
    dt.NormalizeMinMax()
    )
    return image_normalizer

def train_auto_tracker(image, batch_size=8, epochs=20):
    """
    Trains a new model to track an arbitrary particle type
    """
    IMAGE_SIZE = np.shape(image)[0]
    IMAGE_SIZE_1 = np.shape(image)[1]

    image_value = dt.Value(image)

    image_loader = dt.OneOfDict({
        "ideal": image_value,
    })

    # Technically not needed, but good practice
    image_augmenter = (
        dt.DummyFeature() >>
        dt.Gaussian(sigma=lambda:np.random.rand() * 0) >>
        dt.Add(lambda: np.random.randn() * 0.03)
    )

    image_normalizer = get_image_normalizer()

    training_dataset = image_loader >> image_augmenter >> image_normalizer

    ideal_training_dataset = dt.Bind(training_dataset, key="ideal")
    model = dt.models.AutoTracker(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE_1, 1),
    mode="tracking",
    )

    model.fit(ideal_training_dataset, batch_size=batch_size, epochs=epochs)

    return model, image_normalizer
