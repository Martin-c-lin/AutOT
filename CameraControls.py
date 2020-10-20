import ThorlabsCam as TC
import numpy as np
import threading
import time
from cv2 import VideoWriter, VideoWriter_fourcc
import PIL.Image, PIL.ImageTk
from pypylon import pylon
from datetime import  datetime


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
          self.cam.set_defaults(left=c_p['AOI'][0], right=c_p['AOI'][1], top=c_p['AOI'][2], bot=c_p['AOI'][3], n_frames=1)
          exposure_time = TC.find_exposure_time(self.cam, targetIntensity=70) # automagically finds a decent exposure time
          print('Exposure time = ', exposure_time)
          c_p['exposure_time'] = exposure_time
      else:
          # Get a basler camera
          tlf = pylon.TlFactory.GetInstance()
          self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
          self.cam.Open()
          image = np.zeros((672,512,1))
      self.setDaemon(True)
      c_p['image'] = np.ones((512,672,1)) # Need to change here to get color

   def __del__(self):
        if self.c_p['camera_model'] == 'basler':
            self.cam.Close()
        else:
            self.cam.close()

   def get_important_parameters(self):
       #global c_p
       c_p = self.c_p
       parameter_dict = {
       'xm':c_p['xm'],
       'ym':c_p['ym'],
       'zm':c_p['zm'],
       'use_LGO':c_p['use_LGO'],
       'LGO_order':c_p['LGO_order'],
       'setpoint_temperature':c_p['setpoint_temperature'],
       'target_experiment_z':c_p['target_experiment_z'],
       'temperature_output_on':c_p['temperature_output_on'],
       'exposure_time':c_p['exposure_time'],
       'starting_temperature':c_p['current_temperature'],
       'phasemask':c_p['phasemask'],
       }
       return parameter_dict

   def create_video_writer(self):
        '''
        Funciton for creating a VideoWriter.
        Will also save the relevant parameters of the experiments.
        '''
        #global c_p
        c_p = self.c_p
        now = datetime.now()
        fourcc = VideoWriter_fourcc(*'MJPG')
        image_width = c_p['AOI'][1]-c_p['AOI'][0]
        image_height = c_p['AOI'][3]-c_p['AOI'][2]
        video_name = c_p['recording_path'] + '/video-'+ c_p['measurement_name'] + \
            '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)+'.avi'

        experiment_info_name =c_p['recording_path'] + '/data-' + c_p['measurement_name'] + \
            str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        print('Image width,height,framerate',image_width,image_height,int(c_p['framerate']))
        video = VideoWriter(video_name, fourcc,
            min(500,c_p['framerate']), # Format cannot handle high framerates
            (image_width, image_height), isColor=False)
        exp_info_params = self.get_important_parameters()
        return video, experiment_info_name, exp_info_params

   def thorlabs_capture(self):
      number_images_saved = 0 # counts
      video_created = False
      #global c_p
      c_p = self.c_p

      while c_p['program_running']:
          # Set defaults for camera, aknowledge that this has been done
          self.cam.set_defaults(left=c_p['AOI'][0],
              right=c_p['AOI'][1],
              top=c_p['AOI'][2],
              bot=c_p['AOI'][3])
          TC.set_exposure(self.cam, c_p['exposure_time'])
          c_p['new_settings_camera'] = False

          # Grab one example image
          #global image
          image = c_p['image'] # TODO might give me problems
          image = self.cam.grab_image(n_frames=1)
          image_count = 0
          # Start livefeed from the camera

          # Setting  maximum framerate. Will cap it to make it stable
          self.cam.start_live_video(
               framerate=str(c_p['framerate']) + 'hertz' )

          start = time.time()

          # Start continously capturin images now that the camera parameters have been set
          while c_p['program_running'] and not c_p['new_settings_camera']:
              self.cam.wait_for_frame(timeout=None)
              if c_p['recording']:
                  # Create an array to store the images which have been captured in

                  if not video_created:
                      video, experiment_info_name, exp_info_params = self.create_video_writer()
                      video_created = True
                  video.write(image)
              # Capture an image and update the image count
              image_count = image_count+1
              c_p['image'] = self.cam.latest_frame()[:,:,0]
              # c_p['image'] = c_p['image'][:,:,0]
              # print(np.shape(c_p['image']))
          # Close the livefeed and calculate the fps of the captures
          end = time.time()
          self.cam.stop_live_video()
          fps = image_count/(end-start)
          print('Capture sequence finished', image_count,
               'Images captured in ', end-start, 'seconds. \n FPS is ',
               fps)

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
       #global c_p
       c_p = self.c_p
       try:
            # The order in which you set the size and offset parameters matter.
            # If you ever get the offset + width greater than max width the
            # camera won't accept your valuse. Thereof the if-else-statements
            # below. Conditions might need to be changed if the usecase of this
            #  funciton change
            c_p['AOI'][1] -= np.mod(c_p['AOI'][1]-c_p['AOI'][0],16)
            c_p['AOI'][3] -= np.mod(c_p['AOI'][3]-c_p['AOI'][2],16)

            width = int(c_p['AOI'][1] - c_p['AOI'][0])
            offset_x = 672 - width - c_p['AOI'][0]
            height = int(c_p['AOI'][3] - c_p['AOI'][2])
            offset_y = 512 - height - c_p['AOI'][2]

            self.cam.OffsetX = 0
            self.cam.Width = width
            self.cam.OffsetX = offset_x
            self.cam.OffsetY = 0
            self.cam.Height = height
            self.cam.OffsetY = offset_y
       except Exception as e:
           print('AOI not accepted',c_p['AOI'])
           print(e)

   def basler_capture(self):
      number_images_saved = 0 # counts
      video_created = False
      c_p = self.c_p
      img = pylon.PylonImage()

      while c_p['program_running']:
          # Set defaults for camera, aknowledge that this has been done

          self.set_basler_AOI()
          c_p['new_settings_camera'] = False
          try:
              self.cam.ExposureTime = c_p['exposure_time']
              c_p['framerate'] = self.cam.ResultingFrameRate.GetValue()
              c_p['framerate'] = round(float(c_p['framerate']), 1)
              print('Read framerate to ', c_p['framerate'], ' fps.')
          except:
              print('Exposure time not accepted by camera')
          # Grab one example image
          image_count = 0

          global image
          self.cam.StartGrabbing()

          start = time.time()

          # Start continously capturing images now that the camera parameters have been set
          while c_p['program_running']\
               and not c_p['new_settings_camera']:

               with self.cam.RetrieveResult(2000) as result:
                  img.AttachGrabResultBuffer(result)
                  image = np.flip(img.GetArray(),axis=(0,1)) # Testing to flip this guy
                  img.Release()
                  if c_p['recording']:
                      # Create an array to store the images which have been captured in
                      if not video_created:
                            video, experiment_info_name, exp_info_params = self.create_video_writer()
                            video_created = True
                      video.write(image)
                  # Capture an image and update the image count
                  image_count = image_count+1


          self.cam.StopGrabbing()

          # Close the livefeed and calculate the fps of the captures
          end = time.time()

          # Calculate FPS
          fps = image_count/(end-start)
          print('Capture sequence finished', image_count,
               'Images captured in ', end-start, 'seconds. \n FPS is ',
               fps)

          if video_created:
              video.release()
              del video
              video_created = False
              # Save the experiment data in a pickled dict.
              outfile = open(experiment_info_name, 'wb')
              exp_info_params['fps'] = fps
              pickle.dump(exp_info_params, outfile)
              outfile.close()

   def run(self):
       if self.c_p['camera_model'] == 'ThorlabsCam':
           self.thorlabs_capture()
       elif self.c_p['camera_model'] == 'basler':
           self.basler_capture()

def set_AOI(c_p, half_image_width=50, left=None, right=None, up=None, down=None):
    '''
    Function for changing the Area Of Interest for the camera to the box specified by
    left,right,top,bottom
    Assumes global access to c_p
    '''
    #global c_p

    # Do not want motors to be moving when changing AOI!
    # If exact values have been provided for all the corners change AOI
    if c_p['camera_model'] == 'ThorlabsCam':
        if left is not None and right is not None and up is not None and down is not None:
            if 0<=left<=1279 and left<=right<=1280 and 0<=up<=1079 and up<=down<=1080:
                c_p['AOI'][0] = left
                c_p['AOI'][1] = right
                c_p['AOI'][2] = up
                c_p['AOI'][3] = down
            else:
                print("Trying to set invalid area")
    else:
        if left is not None and right is not None and up is not None and down is not None:
            if 0<=left<672 and left<=right<=672 and 0<=up<512 and up<=down<=512:
                c_p['AOI'][0] = left
                c_p['AOI'][1] = right
                c_p['AOI'][2] = up
                c_p['AOI'][3] = down
            else:
                print("Trying to set invalid area")

    print('Setting AOI to ',c_p['AOI'])

    # Inform the camera and display thread about the updated AOI
    c_p['new_settings_camera'] = True
    c_p['new_AOI_display'] = True
    print('AOI changed')
    # Update trap relative position
    #update_traps_relative_pos() TODO fix this problemo

    # Give motor threads time to catch up
    time.sleep(0.5)
