import ThorlabsCam as TC
import numpy as np
import threading
import time
import pickle
from cv2 import VideoWriter, VideoWriter_fourcc
from pypylon import pylon
from datetime import datetime
import skvideo.io

def get_camera_c_p():
    '''
    Function for retrieving the c_p relevant for controlling the camera
    '''
    # TODO make it so that the AOI of the basler camera is not hardcoded. Maybe
    # make a camera model class?
    # Make it easier to change objective?
    camera_c_p = {
        'new_video': False,
        'recording_duration': 3000,
        'exposure_time': 30_000,  # ExposureTime in ms for thorlabs,
        # mus for basler
        'framerate': 15,
        'max_framerate': 10_000, # maximum framerate allowed.
        'recording': False,  # True if recording is on
        'AOI': [0, 480, 0, 480],  # Default for
        'zoomed_in': False,  # Keeps track of whether the image is cropped or
        'camera_model': 'basler_fast',  # basler_fast, thorlabs are options
        'camera_orientatation': 'down',  # direction camera is mounted in.
        'default_offset_x':0, # Used to center the camera on the sample
        'default_offset_y':0,
        'bitrate': '300000000', # Default value 3e8 default
        # Needed for not
    }
# TODO recording with ffmpeg is really slow!
    # Add custom parameters for different cameras.
    if camera_c_p['camera_model'] == 'basler_large':
        camera_c_p['mmToPixel'] = 37_700 # Made a control measurement and found it to be 37.7
        camera_c_p['camera_width'] = 4096
        camera_c_p['camera_height'] = 3040
        camera_c_p['default_offset_x'] = 1000

    elif camera_c_p['camera_model'] == 'thorlabs':
        camera_c_p['mmToPixel'] = 17736/0.7
        camera_c_p['camera_width'] = 1200
        camera_c_p['camera_height'] = 1080

    elif camera_c_p['camera_model'] == 'basler_fast':
        camera_c_p['mmToPixel'] = 21_500#37_700#16140/0.7
        camera_c_p['camera_width'] = 672
        camera_c_p['camera_height'] = 512

    camera_c_p['slm_to_pixel'] = 5_000_000 if camera_c_p['camera_model'] == 'basler_fast' else 4_550_000
    camera_c_p['AOI'] = [0, camera_c_p['camera_width'], 0, camera_c_p['camera_height']]
    print(camera_c_p['AOI'])
    return camera_c_p


class CameraThread(threading.Thread):

    def __init__(self, threadID, name, c_p):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        # Initalize camera and global image
        if c_p['camera_model'] == 'ThorlabsCam':
            # Get a thorlabs camera
            self.cam = TC.get_camera()
            self.cam.set_defaults(left=c_p['AOI'][0], right=c_p['AOI'][1],
                                  top=c_p['AOI'][2], bot=c_p['AOI'][3],
                                  n_frames=1)
            c_p['exposure_time'] = 20
        else:
            # Get a basler camera
            tlf = pylon.TlFactory.GetInstance()
            self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
            self.cam.Open()
        self.setDaemon(True)
        self.video_width = self.c_p['AOI'][1] - self.c_p['AOI'][0]
        self.video_height = self.c_p['AOI'][3] - self.c_p['AOI'][2]
        # Could change here to get color
        c_p['image'] = np.ones((self.c_p['AOI'][3], self.c_p['AOI'][1], 1))

    def __del__(self):
        """
        Closes the camera in preparation of terminating the thread.
        """
        if self.c_p['camera_model'] == 'ThorlabsCam':
            self.cam.close()
        else:
            self.cam.Close()

    def get_important_parameters(self):
        '''
        Gets the parameters needed to describe the experiment. Used for svaing
        experiment setup.
        Returns
        -------
        parameter_dict : Dictionary
            Dictionary containing the control parameters needed to describe
            the experiment.
        '''
        c_p = self.c_p
        parameter_dict = {
            'xm': c_p['xm'],
            'ym': c_p['ym'],
            'zm': c_p['zm'],
            'use_LGO': c_p['use_LGO'],
            'LGO_order': c_p['LGO_order'],
            'max_framerate': c_p['max_framerate'],
            'setpoint_temperature': c_p['setpoint_temperature'],
            'target_experiment_z': c_p['target_experiment_z'],
            'temperature_output_on': c_p['temperature_output_on'],
            'exposure_time': c_p['exposure_time'],
            'starting_temperature': c_p['current_temperature'],
            'phasemask': c_p['phasemask'],
        }
        return parameter_dict

    def create_video_writer(self):
        '''
        Funciton for creating a VideoWriter.
        Will also save the relevant parameters of the experiments.
        Returns
        -------
        video : VideoWriter
            A video writer for creating a video.
        experiment_info_name : String
            Name of experiment being run.
        exp_info_params : Dictionary
            Dictionary with controlparameters describing the experiment.
        '''
        c_p = self.c_p
        now = datetime.now()
        fourcc = VideoWriter_fourcc(*'MJPG')
        image_width = c_p['AOI'][1]-c_p['AOI'][0]
        image_height = c_p['AOI'][3]-c_p['AOI'][2]
        self.video_width = image_width
        self.video_height = image_height
        video_name = c_p['recording_path'] + '/video-'+ c_p['measurement_name'] + \
            '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)+'.avi'

        experiment_info_name = c_p['recording_path'] + '/data-' + c_p['measurement_name'] + \
            str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)

        print('Image width,height,framerate', image_width, image_height, int(c_p['framerate']))
        video = VideoWriter(video_name, fourcc, min(500, c_p['framerate']),
                            (image_width, image_height), isColor=False)

        exp_info_params = self.get_important_parameters()

        return video, experiment_info_name, exp_info_params


    def create_HQ_video_writer(self):
        """
        Crates a high quality video writer for lossless recording.
        """
        c_p = self.c_p
        now = datetime.now()
        frame_rate = str(min(500, int(c_p['framerate'])))

        image_width = c_p['AOI'][1]-c_p['AOI'][0]
        image_height = c_p['AOI'][3]-c_p['AOI'][2]
        self.video_width = image_width
        self.video_height = image_height
        video_name = c_p['recording_path'] + '/video-'+ c_p['measurement_name'] + \
            '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)+'.mp4'

        experiment_info_name = c_p['recording_path'] + '/data-' + c_p['measurement_name'] + \
            str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)

        print('Image width,height,framerate', image_width, image_height, int(c_p['framerate']))

        video = skvideo.io.FFmpegWriter(video_name, outputdict={
                                         '-b':c_p['bitrate'],
                                         '-r':frame_rate, # Does not like this
                                         # specifying codec and bitrate, 'vcodec': 'libx264',
                                        })
        exp_info_params = self.get_important_parameters()

        return video, experiment_info_name, exp_info_params


# TODO make it possible to terminate a video without changing camera settings

    def thorlabs_capture(self):
        number_images_saved = 0
        video_created = False
        c_p = self.c_p

        while c_p['program_running']:
            # Set defaults for camera, aknowledge that this has been done

            self.cam.set_defaults(left=c_p['AOI'][0],
                right=c_p['AOI'][1],
                top=c_p['AOI'][2],
                bot=c_p['AOI'][3],
                exposure_time=TC.number_to_millisecond(c_p['exposure_time']))
            c_p['new_settings_camera'] = False

            image_count = 0

            # Setting  maximum framerate. Will cap it to make it stable
            # Start livefeed from the camera
            self.cam.start_live_video()
            start = time.time()

            # Start continously capturing images
            while c_p['program_running'] and not c_p['new_settings_camera']:
                self.cam.wait_for_frame(timeout=None)
                if c_p['recording']:

                    if not video_created:
                        video, experiment_info_name, exp_info_params = self.create_video_writer()

                        video_created = True
                    video.write(c_p['image'])

                # Capture an image and update the image count
                image_count = image_count+1
                c_p['image'] = self.cam.latest_frame()[:, :, 0]
            # Close the livefeed and calculate the fps of the captures
            end = time.time()
            self.cam.stop_live_video()
            fps = image_count/(end-start)

            if video_created:
                video.release()
                del video
                video_created = False
                # Save the experiment data in a pickled dict.
                outfile = open(experiment_info_name, 'wb')
                exp_info_params['fps'] = fps
                pickle.dump(exp_info_params, outfile)
                outfile.close()

    def set_basler_AOI(self):
        '''
        Function for setting AOI of basler camera to c_p['AOI']
        '''
        c_p = self.c_p
        try:
            '''
            The order in which you set the size and offset parameters matter.
            If you ever get the offset + width greater than max width the
            camera won't accept your valuse. Thereof the if-else-statements
            below. Conditions might need to be changed if the usecase of this
            funciton change
            '''

            width = int(c_p['AOI'][1] - c_p['AOI'][0])
            offset_x = c_p['AOI'][0]
            height = int(c_p['AOI'][3] - c_p['AOI'][2])
            offset_y = c_p['AOI'][2]
            #print(self.cam.Width.GetMax(), self.cam.Height.GetMax())
            self.video_width = width
            self.video_height = height
            self.cam.OffsetX = 0
            self.cam.OffsetY = 0
            time.sleep(0.1)
            self.cam.Width = width
            self.cam.Height = height
            self.cam.OffsetX = c_p['default_offset_x'] + offset_x
            self.cam.OffsetY = c_p['default_offset_y'] + offset_y
            # self.cam.Width.SetValue(width)
            # self.cam.Height.SetValue(height)
            # self.cam.OffsetX.SetValue(c_p['default_offset_x'] + offset_x)
            # self.cam.OffsetY.SetValue(c_p['default_offset_y'] + offset_y)

        except Exception as e:
            print('AOI not accepted', c_p['AOI'], width, height, offset_x, offset_y)
            print(e)

    def update_basler_exposure(self):
        try:
            self.cam.ExposureTime = self.c_p['exposure_time']
            self.c_p['framerate'] = self.cam.ResultingFrameRate.GetValue()
            self.c_p['framerate'] = round(float(self.c_p['framerate']), 1)
        except:
            print('Exposure time not accepted by camera')

    def basler_capture(self):
        '''
        Function for live capture using the basler camera. Also allows for
        change of AOI, exposure and saving video on the fly.
        Returns
        -------
        None.
        '''
        video_created = False
        c_p = self.c_p
        img = pylon.PylonImage()

        while c_p['program_running']:
            # Set defaults for camera, aknowledge that this has been done

            self.set_basler_AOI()
            c_p['new_settings_camera'] = False

            #TODO replace c_p['new_settings_camera'] with two parameters and
            # one for expsore and one for AOI
            self.update_basler_exposure()
            image_count = 0

            global image
            self.cam.StartGrabbing()

            start = time.time()

            # Start continously capturing images
            while c_p['program_running']\
                 and not c_p['new_settings_camera']:
                 with self.cam.RetrieveResult(2000) as result:
                    img.AttachGrabResultBuffer(result)
                    if result.GrabSucceeded():
                        c_p['image'] = np.uint8(img.GetArray())
                        img.Release()
                        if c_p['recording']:
                            if not video_created:
                                #video, experiment_info_name, exp_info_params = self.create_HQ_video_writer()
                                video, experiment_info_name, exp_info_params = self.create_video_writer()
                                video_created = True
                            # video.writeFrame(c_p['image']) # For HQ video with ffmpeg
                            video.write(c_p['image'])
                            # consider adding text to one corner
                        # Capture an image and update the image count
                        image_count = image_count+1
                 if c_p['new_settings_camera']:
                    w = c_p['AOI'][1] - c_p['AOI'][0]
                    h = c_p['AOI'][3] - c_p['AOI'][2]
                    self.cam.AcquisitionFrameRateEnable.SetValue(True);
                    self.cam.AcquisitionFrameRate.SetValue(c_p['max_framerate'])
                    if w == self.video_width and h == self.video_height:
                        self.update_basler_exposure()
                        c_p['new_settings_camera'] = False
            self.cam.StopGrabbing()

            # Close the livefeed and calculate the fps of the captures
            end = time.time()
            try:
                fps = image_count/(end-start)
            except:
                fps = -1

            if video_created:
                #video.close() # HQ video
                video.release()
                del video
                video_created = False
                # Save the experiment data in a pickled dict.
                outfile = open(experiment_info_name, 'wb')
                exp_info_params['fps'] = fps
                print(" Measured fps was:", fps, " Indicated by camera", c_p['framerate'])
                pickle.dump(exp_info_params, outfile)
                outfile.close()

    def run(self):
        if self.c_p['camera_model'] == 'ThorlabsCam':
            self.thorlabs_capture()
        elif self.c_p['camera_model'] == 'basler_large' or 'basler_fast':
            self.basler_capture()
        return


def set_AOI(c_p, left=None, right=None, up=None, down=None):
    '''
    Function for changing the Area Of Interest for the camera to the box
    specified by left,right,top,bottom.
    Parameters
    ----------
    c_p : Dictionary
        Dictionary with control parameters.
    left : INT, optional
        Left position of camera AOI in pixels. The default is None.
    right : INT, optional
        Right position of camera AOI in pixels. The default is None.
    up : INT, optional
        Top position of camera AOI in pixels. The default is None.
    down : INT, optional
        Bottom position of camera AOI in pixels. The default is None.
    Returns
    -------
    None.
    '''

    # If exact values have been provided for all the corners change AOI
    h_max = c_p['camera_width']
    v_max = c_p['camera_height']
    if left is not None and right is not None and up is not None and down is not None:
        if 0<=left<=h_max and left<=right<=h_max and 0<=up<=v_max and up<=down<=v_max:
            c_p['AOI'][0] = left
            c_p['AOI'][1] = right
            c_p['AOI'][2] = up
            c_p['AOI'][3] = down
        else:
            print("Trying to set invalid area")
            return

    # Inform the camera and display thread about the updated AOI
    c_p['new_settings_camera'] = True
    c_p['new_AOI_display'] = True

    # Update trap relative position
    update_traps_relative_pos(c_p)

    # Threads time to catch up
    time.sleep(0.1)


def update_traps_relative_pos(c_p):
    '''
    Updates the relative position of the traps when zooming in/out.
    '''
    tmp_x = [x - c_p['AOI'][0] for x in c_p['traps_absolute_pos'][0]]
    tmp_y = [y - c_p['AOI'][2] for y in c_p['traps_absolute_pos'][1]]
    tmp = np.asarray([tmp_x, tmp_y])
    c_p['traps_relative_pos'] = tmp


def zoom_out(c_p):
    '''
    Zooming out the camera AOI to the maximum allowed.
    '''
    # Reset camera to fullscreen view
    set_AOI(c_p, left=0, right=int(c_p['camera_width']), up=0,
            down=int(c_p['camera_height']))
    c_p['AOI'] = [0, int(c_p['camera_width']), 0, int(c_p['camera_height'])]
    print('Zoomed out', c_p['AOI'])
    c_p['new_settings_camera'] = True
