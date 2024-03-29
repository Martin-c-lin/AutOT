# Script for controlling the whole setup automagically
import sys
sys.coinit_flags = 2
import pywinauto # To fix askdirectory error
from tkinter import simpledialog, ttk, filedialog

import ThorlabsCam as TC
import SLM#, QD_tracking
import ThorlabsMotor as TM
import TemperatureControllerTED4015 as TC4015
import find_particle_threshold as fpt
import read_dict_from_file as rdff
import ThorlabsShutter as TS
import CameraControls
from arduinoLEDcontrol import ArduinoLEDControlThread, get_arduino_c_p, toggle_BG_shutter, toggle_green_laser
from CameraControls import update_traps_relative_pos # Moved this function
from common_experiment_parameters import append_c_p, get_save_path
from MiscFunctions import subtract_bg, convolve_binning
from KeyboardControl import TRD

#from instrumental import u
import numpy as np
import threading, time, cv2, queue, copy, tkinter, os, pickle
from tkinter import filedialog as fd
from functools import partial
from datetime import datetime
from tkinter import *  # TODO Should avoid this type of import statements.
from pynput import keyboard


from tkinter.messagebox import showinfo
import PIL.Image, PIL.ImageTk


def terminate_threads(thread_list, c_p):
    '''
    Function for terminating all threads.

    Returns
    -------
    None.

    '''
    c_p['program_running'] = False
    c_p['motor_running'] = False
    c_p['tracking_on'] = False
    c_p['scroll_for_z'] = False
    time.sleep(1)
    for thread in thread_list:
        thread.join()
    for thread in thread_list:
        del thread


def start_threads(c_p, thread_list):
    """
    Function for starting all the threads, should only be called once.

    """

    if c_p['cam']:
        append_c_p(c_p, CameraControls.get_camera_c_p())
        camera_thread = CameraControls.CameraThread(1, 'Thread-camera',c_p=c_p)
        camera_thread.start()
        thread_list.append(camera_thread)

        VideoWriterThread = CameraControls.VideoWriterThread(15, 'Writer thread',c_p=c_p)
        VideoWriterThread.start()
        thread_list.append(VideoWriterThread)

        print('Camera thread started')

    # Indicator to if the standard motors are being used ( non stage)
    c_p['standard_motors'] = False
    if c_p['motor_x']:
        c_p['standard_motors'] = True
        try:
            motor_X_thread = TM.MotorThread(2, 'Thread-motorX', 0, c_p)
            motor_X_thread.start()
            thread_list.append(motor_X_thread)
            print('Motor x thread started')
        except Exception as ex:
            print(f"Could not start motor x thread, {ex}")

    if c_p['motor_y']:
        c_p['standard_motors'] = True
        try:
            motor_Y_thread = TM.MotorThread(3,'Thread-motorY',1,c_p)
            motor_Y_thread.start()
            thread_list.append(motor_Y_thread)
            print('Motor y thread started')
        except Exception as ex:
            print(f"Could not start motor y thread, {ex}")

    if c_p['motor_z']:
        try:
            z_thread = TM.z_movement_thread(4, 'z-thread',serial_no=c_p['serial_nums_motors'][2],channel=c_p['channel'],c_p=c_p)
            z_thread.start()
            thread_list.append(z_thread)
            print('Motor z thread started')
        except Exception as ex:
            print(f"Could not start motor z thread, {ex}")

    if c_p['SLM']:
        slm_thread = SLM.CreatePhasemaskThread(5,'Thread-SLM', c_p=c_p)
        slm_thread.start()
        thread_list.append(slm_thread)
        print('SLM thread started')

    if c_p['tracking']:
        tracking_thread = ExperimentControlThread(6,'Tracker_thread')
        tracking_thread.start()
        thread_list.append(tracking_thread)
        print('Tracking thread started')

    if c_p['temperature_controlls']:

        try:
            temperature_controller = TC4015.TED4015()
        except Exception as ex:
            temperature_controller = None
            print(f"Problem connecting to temperature controller, {ex}")

        temperature_thread = TC4015.TemperatureThread(7, 'Temperature_thread', c_p, temperature_controller=temperature_controller)
        temperature_thread.start()
        thread_list.append(temperature_thread)
        print('Temperature thread started')

    # update c_p to include the necessary parameters
    c_p['stage_piezos'] = False
    if c_p['stage_piezo_x'] or c_p['stage_piezo_y'] or c_p['stage_piezo_z']:
        c_p['stage_piezos'] = True
        append_c_p(c_p, TM.get_default_piezo_c_p())
        try:
            c_p['piezo_controller'] = TM.ConnectBenchtopPiezoController(c_p['piezo_serial_no'])
        except Exception as ex:
            print('Could not connect piezo controller')

    if c_p['stage_piezo_x']:
        # OBS assumes that the x-motor is connected to channel 1
        try:
            thread_piezo_x = TM.XYZ_piezo_stage_motor(8, 'piezo_x', 1,0, c_p,
                target_key='QD_target_loc_x')
            thread_piezo_x.start()
            thread_list.append(thread_piezo_x)

            print('Started piezo x-thread')
        except Exception as ex:
            print('Could not start piezo x-thread')

    if c_p['stage_piezo_y']:
        # OBS assumes that the y-motor is connected to channel 2
        try:
            thread_piezo_y = TM.XYZ_piezo_stage_motor(9, 'piezo_y', 2,1, c_p,
                target_key='QD_target_loc_y')
            thread_piezo_y.start()
            thread_list.append(thread_piezo_y)
            print('Started piezo y-thread')
        except Exception as ex:
            print('Could not start piezo y-thread')

    if c_p['stage_piezo_z']:
        # OBS assumes that the z-motor is connected to channel 3
        try:
            thread_piezo_z = TM.XYZ_piezo_stage_motor(10, 'piezo_z', 3,2, c_p,
                target_key='QD_target_loc_z', step=1)
            thread_piezo_z.start()
            thread_list.append(thread_piezo_z)
            c_p['piezo_elevation'] = 0 # Used by mouse input thread
            print('Started piezo z-thread')
        except Exception as ex:
            print('Could not start piezo z-thread')

    # If there is any stepper motor to connect, then add necessary c_p and
    # connect benchtop controller.
    # TODO: Don't really need the stage_stepper... parameters anymore
    if c_p['stage_stepper_x'] or c_p['stage_stepper_y'] or c_p['stage_stepper_z']:
        c_p['using_stepper_motors'] = True
        append_c_p(c_p, TM.get_default_stepper_c_p())
        try:
            c_p['stepper_controller'] = TM.ConnectBenchtopStepperController(c_p['stepper_serial_no'])
        except Exception as ex:
            print('Could not connect stepper controller.')
    else:
        c_p['using_stepper_motors'] = False

    if c_p['stage_stepper_x']:
        try:
            thread_stepper_x = TM.XYZ_stepper_stage_motor(11, 'stepper_X',1,0,
            c_p, controller_device=c_p['stepper_controller'])
            thread_stepper_x.start()
            thread_list.append(thread_stepper_x)
        except Exception as ex:
            print('Could not connect stepper x')

    if c_p['stage_stepper_y']:
        try:
            thread_stepper_y = TM.XYZ_stepper_stage_motor(12, 'stepper_Y',2,1,
            c_p, controller_device=c_p['stepper_controller'])
            thread_stepper_y.start()
            thread_list.append(thread_stepper_y)
        except Exception as ex:
            print('Could not connect stepper y')

    if c_p['stage_stepper_z']:
        try:
            thread_stepper_z = TM.XYZ_stepper_stage_motor(13, 'stepper_Z',3,2,
            c_p, controller_device=c_p['stepper_controller'])
            thread_stepper_z.start()
            thread_list.append(thread_stepper_z)
            c_p['stepper_elevation'] = 0
        except Exception as ex:
            print('Could not connect stepper z')


    if c_p['shutter']:
        append_c_p(c_p, TS.get_shutter_c_p())
        try:
            shutter_thread = TS.ShutterThread(14, 'shutter_thread', c_p)
            shutter_thread.start()
            thread_list.append(shutter_thread)
            print('Started shutter thread')
        except Exception as ex:
            print('Could not start shutter thread')

    if c_p['QD_tracking']:
        import QD_tracking
        append_c_p(c_p, QD_tracking.get_QD_tracking_c_p())

        try:
            QD_Tracking_Thread = QD_tracking.QD_Tracking_Thread(
                15, 'QD_Tracking_Thread', c_p=c_p)
            QD_Tracking_Thread.start()
            thread_list.append(QD_Tracking_Thread)
            print('Quantum dot tracking thread started')
        except Exception as ex:
            print('Could not start quantum dot tracking thread')

    if c_p['arduino_LED']:
        append_c_p(c_p, get_arduino_c_p())
        try:
            ArcuinoThread = ArduinoLEDControlThread(16, 'ArduinoLEDThread',\
                c_p=c_p)
            ArcuinoThread.start()
            thread_list.append(ArcuinoThread)
            print('Arduino com thread started!')
        except Exception as ex:
            print('Could not start arduino thread')

    if c_p['stage_piezos'] or c_p['using_stepper_motors'] or c_p['standard_motors']:
        # Start thread for controlling z-position using the mouse scroll-wheel.
        try:
            from MouseInputThread import mouseInputThread
            mouseInputTrd = mouseInputThread(17,'mouse thread', c_p)
            mouseInputTrd.start()
            thread_list.append(mouseInputTrd)
            print('Mouse-thread started')
        except Exception as ex:
            print('Could not start mouse input thread')
    


class UserInterface:
    """
    Class which creates the GUI.
    """
    def __init__(self, window, c_p, thread_list):
         # TODO make this thing create it's own window
        self.window = window
        start_threads(c_p, thread_list)
        self.thread_list = thread_list
        # Create a canvas that can fit the above video source size
        # TODO: Add simpler way to label outputs(videos, images etc) from the program
        # TODO make display window able to change size
        self.canvas_width = 1300
        self.canvas_height = 1120

        self.mini_canvas_width = 240
        self.mini_canvas_height = 200
        self.c_p = c_p
        self.canvas = tkinter.Canvas(
            window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.place(x=0, y=0)

        self.mini_canvas = tkinter.Canvas(
            window, width=self.mini_canvas_width, height=self.mini_canvas_height)
        self.mini_canvas.place(x=self.canvas_width, y=self.canvas_height-200)
        self.mini_image = np.zeros((180,240,3))
        self.create_buttons(self.window)
        self.save_bg()
        self.window.geometry(str(self.canvas_width+500)+'x'+str(self.canvas_height))#('1700x1000')
        # After it is called once, the update method will be automatically
        # called every delay milliseconds
        self.image_scale = 1 # scale of image being displayed
        self.delay = 10 #50 standard how often to update view in ms intervals
        self.zoomed_in = False
        if c_p['SLM']:
            self.create_SLM_window(SLM.SLM_window)

        self.update_qd_on_screen_targets(c_p['image'])
        CameraControls.zoom_out(c_p)
        # Computer cannot handle to have menus as well. Make program really slow
        self.menubar = tkinter.Menu(self.window)
        self.control_menu = tkinter.Menu(self.menubar)
        self.create_control_menus()
        self.window.config(menu=self.menubar)
        self.keyboard_c = TRD(c_p)
        self.keyboard_listener = keyboard.Listener(on_press=self.keyboard_c.on_press)
        self.keyboard_listener.start()
        self.update()
        self.window.mainloop()
        self.__del__()

    def __del__(self):
        # Closes the program
        self.c_p['program_running'] = False
        self.c_p['motor_running'] = False
        self.c_p['tracking_on'] = False
        self.keyboard_listener.join()
        terminate_threads(self.thread_list, self.c_p)

    def read_experiment_dictionary(self):
        c_p = self.c_p
        filepath = fd.askopenfilename()
        # TODO make it so that we can handle exceptions from the file better here.
        # Bring up a confirmation menu for the schedule perhaps?
        experiment_list = rdff.ReadFileToExperimentList(filepath)
        if experiment_list is not None and len(experiment_list) > 0:
            c_p['experiment_schedule'] = experiment_list
            print('Starting the following experiment. \n', experiment_list)
            # Reset experiment progress
            if c_p['tracking_on']:
                c_p['tracking_on'] = False
                time.sleep(0.5)
                c_p['experiment_progress'] = 0
                time.sleep(0.2)
                c_p['tracking_on'] = True
            else:
                c_p['experiment_progress'] = 0
            update_c_p(c_p, experiment_list[0])
            c_p['nbr_experiments'] = len(c_p['experiment_schedule'])
            # Update recording path
            name = filepath[filepath.rfind('/')+1:filepath.rfind('.')]
            c_p['recording_path'] = get_save_path(extension_path='_'+name)
        else:
            print('Invalid or empty file.')
        self.update_qd_on_screen_targets(c_p['image'])

    def create_trap_image(self, mini_image_width=240):
        """
        Creates a mini image which shows the AOI and the relative position
        of traps and particles(estimated from tracking thread)
        """

        c_p = self.c_p
        trap_x = c_p['traps_absolute_pos'][0]
        trap_y = c_p['traps_absolute_pos'][1]
        particle_x = c_p['particle_centers'][0]
        particle_y = c_p['particle_centers'][1]
        AOI = c_p['AOI']
        # Define new mini-image

        scale_factor = c_p['camera_width'] / mini_image_width
        mini_image_height = int(scale_factor * c_p['camera_height'])
        mini_image = np.zeros((mini_image_height, mini_image_width, 3))

        l = int(round(AOI[2]/scale_factor))  # left
        r = int(round(AOI[3]/scale_factor))  # right
        u = int(round(AOI[0]/scale_factor))  # up
        d = int(round(AOI[1]/scale_factor))  # down

        # Draw the traps
        if len(trap_x) > 0 and len(trap_x) == len(trap_y):
            for x, y in zip(trap_x, trap_y):
                # Round down and recalculate
                x = int(round(x/scale_factor))
                y = int(round(y/scale_factor))

                if 2 <= x <= mini_image_width-2 and 2 <= y <= mini_image_height-2:
                    mini_image[(y-2):(y+3),x,0] = 255
                    mini_image[y,(x-2):(x+3),0] = 255

        # Draw the particles
        if  len(particle_x) > 0 and len(particle_x) == len(particle_y):
            for x, y in zip(particle_x, particle_y):
                # Round down and recalculate
                x = int(round(x/scale_factor + u))
                y = int(round(y/scale_factor + l))
                if 1 <= x <= mini_image_width-1 and 1 <= y <= mini_image_height:
                    mini_image[y-1:y+2,x-1:x+2,2] = 255

        # Draw the AOI
        try:
            mini_image[l,u:d,1:2] = 255  # Left edge
            mini_image[l:r,u,1:2] = 255  # Upper edge
            mini_image[r,u:d,1:2] = 255  # Right edge
            mini_image[l:r,d,1:2] = 255  # Bottom edge
        except IndexError:
            #print()
        #else:
            mini_image[0,0:-1,1:2] = 255  # Left edge
            mini_image[0:-1,0,1:2] = 255  # Upper edge
            mini_image[-1,0:-1,1:2] = 255  # Right edge
            mini_image[0:-1,-1,1:2] = 255  # Bottom edge

        self.mini_image = mini_image.astype('uint8')

    def get_standard_move_buttons(self, top):
        self.up_button = tkinter.Button(top, text='Move up',
                                   command=partial(move_button, 0))
        self.down_button = tkinter.Button(top, text='Move down',
                                     command=partial(move_button, 1))
        self.right_button = tkinter.Button(top, text='Move right',
                                      command=partial(move_button, 2))
        self.left_button = tkinter.Button(top, text='Move left',
                                     command=partial(move_button, 3))

    def get_stage_move_buttons(self, top):
        self.up_button = tkinter.Button(top, text='Move up',
                                   command=partial(stage_piezo_manual_move, axis=1, distance=0.2))
        self.down_button = tkinter.Button(top, text='Move down',
                                     command=partial(stage_piezo_manual_move, axis=1, distance=-0.2))
        self.right_button = tkinter.Button(top, text='Move right',
                                      command=partial(stage_piezo_manual_move, axis=0, distance=0.2))
        self.left_button = tkinter.Button(top, text='Move left',
                                     command=partial(stage_piezo_manual_move, axis=0, distance=-0.2))

    def screen_click(self, event):
        c_p = self.c_p
        self.get_mouse_position()
        if c_p['mouse_move_allowed']:
            self.mouse_command_move()
        if self.set_laser_position.get():
            if c_p['mouse_move_allowed']:
                print('Cannot choose laser position when mouse movement is on.')
                print('Please turn off mouse moves!')
                self.set_laser_position.set(False)
            else:
                self.mouse_set_laser_position()
                # TODO test if the camera settings update works
                c_p['new_settings_camera'] = True

    def reset_printing_parameters(self):
        self.c_p['Anchor_placed'] = False
        self.c_p['QDs_placed'] = 0

    def mouse_set_laser_position(self):
        c_p = self.c_p
        c_p['traps_absolute_pos'][0][0] = int(self.image_scale * c_p['mouse_position'][0] + c_p['AOI'][0])
        c_p['traps_absolute_pos'][1][0] = int(self.image_scale * c_p['mouse_position'][1] + c_p['AOI'][2])
        print('Setting laser position (x,y) to (',
            c_p['traps_absolute_pos'][0][0], c_p['traps_absolute_pos'][1][0] ,
            ')')
        update_traps_relative_pos(c_p)
        self.update_qd_on_screen_targets()

    def toggle_move_by_clicking(self):
        self.c_p['mouse_move_allowed'] = not self.c_p['mouse_move_allowed']

    def increment_QD_count(self):
        c_p = self.c_p
        if len(c_p['QD_target_loc_x']) > (c_p['QDs_placed'] + 1):
            c_p['QDs_placed'] += 1
        else:
            print('Already at final location')
        self.update_qd_on_screen_targets()

    def decrement_QD_count(self):
        if self.c_p['QDs_placed'] > 0:
            self.c_p['QDs_placed'] -= 1
        else:
            print('Already at first location')
        self.update_qd_on_screen_targets(self.c_p['image'])

    def save_starting_position(self):
        c_p = self.c_p
        print('Positions saved.')

        if c_p['stage_piezos']:
            # Set piezo starting positions to current position and reset elevation offset
            c_p['QD_target_loc_z'][0] = c_p['piezo_current_position'][2]
            c_p['piezo_starting_position'][2] = c_p['piezo_current_position'][2]
            c_p['piezo_starting_position'][1] = c_p['piezo_current_position'][1]
            c_p['piezo_starting_position'][0] = c_p['piezo_current_position'][0]
            c_p['piezo_elevation'] = 0

        if c_p['using_stepper_motors']:
            # Set stepper starting positions to current position and reset elevation offset
            c_p['stepper_starting_position'][0] = c_p['stepper_current_position'][0]
            c_p['stepper_starting_position'][1] = c_p['stepper_current_position'][1]
            c_p['stepper_starting_position'][2] = c_p['stepper_current_position'][2]
            c_p['stepper_elevation'] = 0

    def to_focus(self):
        c_p = self.c_p
        if c_p['stage_piezos']:
            c_p['piezo_move_to_target'][2] = not c_p['piezo_move_to_target'][2]

        if c_p['using_stepper_motors']:
            c_p['stepper_elevation'] = 0
            #c_p['stepper_target_position'][2] = c_p['stepper_starting_position'][2]

    def toggle_move_piezo_to_target(self):
        c_p = self.c_p
        if c_p['piezo_move_to_target'][0] or c_p['piezo_move_to_target'][1]:
            c_p['piezo_move_to_target'] = [False, False, False]
        else:
            c_p['piezo_move_to_target'] = [True, True, False]

    def toggle_polymerization_LED(self):
        c_p = self.c_p
        if c_p['polymerization_LED_status'] == 'OFF':
            c_p['polymerization_LED'] = 'Q'
            c_p['polymerization_LED_status'] = 'ON'
        else:
            c_p['polymerization_LED'] = 'W'
            c_p['polymerization_LED_status'] = 'OFF'

    def connect_disconnect_motorX(self):
        c_p = self.c_p
        c_p['connect_motor'][0] = not c_p['connect_motor'][0]

    def connect_disconnect_motorY(self):
        c_p = self.c_p
        c_p['connect_motor'][1] = not c_p['connect_motor'][1]

    def connect_disconnect_piezo(self):
        c_p = self.c_p
        c_p['connect_motor'][2] = not c_p['connect_motor'][2]

    def open_shutter(self):
        c_p = self.c_p
        c_p['should_shutter_open'] = True

    def get_y_separation(self, start=5, distance=38):
        # Simple generator to avoid printing all the y-positions of the
        # buttons
        index = 0
        while True:
            yield start + (distance * index)
            index += 1

    def pickle_c_p(self):
        """
        Saves the c_p to a file of your choosing.
        """
        files = [('Pickle object', '*.pkl')]
        file = fd.asksavefile(filetypes=files, defaultextension=files, mode='wb')
        pickle.dump(self.c_p, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def timed_polymerization(self):
        c_p = self.c_p
        c_p['polymerization_LED'] = 'E'

    def add_stepper_buttons(self, top, generator_y, position_x):

        c_p = self.c_p
        c_p['stepper_activated'] = tkinter.BooleanVar()
        self.stepper_checkbutton = tkinter.Checkbutton(top, text='Use stepper',\
        variable=c_p['stepper_activated'], onvalue=True, offvalue=False)
        self.stepper_checkbutton.place(x=position_x, y=generator_y.__next__())
        self.place_motor_speed_scale(top, x=position_x, y=generator_y.__next__())
        c_p['stepper_activated'].set(True)

    def connect_stepper_motors(self):
        """
        Function for connecting/disconnecting stepper motors.
        """
        c_p = self.c_p
        c_p['connect_steppers'] = not c_p['connect_steppers']

        if not c_p['connect_steppers']:
            #TODO: This should be executed async
            try:
                c_p['stepper_controller'].Disconnect()
            except Exception as ex:
                print("Steppers already disconnected, {ex}")
            c_p['steppers_connected'] = [False, False, False]
        else:
            try:
                c_p['stepper_controller'] = TM.ConnectBenchtopStepperController(c_p['stepper_serial_no'])
            except Exception as ex:
                pass
            if c_p['stepper_controller'] is not None:
                c_p['connect_steppers'] = c_p['stepper_controller'].IsConnected

    def connect_stage_piezos(self):
        """
        Function for connecting/disconnecing piezos
        """
        c_p = self.c_p
        c_p['connect_piezos'] = not c_p['connect_piezos']

        if not c_p['connect_piezos']:
            try:
                c_p['piezo_controller'].Disconnect()
            except Exception as ex:
                print('Piezo controller not connected')
            c_p['stage_piezo_connected'] = [False,False,False]
        else:
            try:
                c_p['piezo_controller'] = TM.ConnectBenchtopPiezoController(c_p['piezo_serial_no'])
            except Exception as ex:
                print('Error connecting piezo controller')
            c_p['connect_piezos'] = c_p['piezo_controller'].IsConnected

    def motor_control_buttons(self, top, y_position, x_position):

        y_pos = y_position.__next__()

        self.to_focus_button = tkinter.Button(
            top, text='To focus', command=self.to_focus)
        self.to_focus_button.place(x=x_position+70, y=y_pos)

        # TODO make the scrolling work only when mouse is on the canvas
        self.move_by_clicking_button = tkinter.Checkbutton(top, text='move by clicking',\
        variable=self.move_by_clicking, onvalue=True, offvalue=False)
        self.move_by_clicking.set(False)
        self.z_scrolling = tkinter.BooleanVar()
        self.z_scrolling_button = tkinter.Checkbutton(top, text='scroll for z-control',\
        variable=self.z_scrolling, onvalue=True, offvalue=False)
        self.z_scrolling_button.place(x=x_position, y=y_position.__next__())
        self.move_by_clicking_button.place(x=x_position, y=y_position.__next__())

    def add_move_buttons(self, top, x_position, y_position):

        c_p = self.c_p
        if c_p['standard_motors']:
            self.get_standard_move_buttons(top)
        elif c_p['stage_piezos']:
            self.get_stage_move_buttons(top)
        self.up_button.place(x=x_position, y=y_position.__next__())
        self.down_button.place(x=x_position, y=y_position.__next__())
        self.right_button.place(x=x_position, y=y_position.__next__())
        self.left_button.place(x=x_position, y=y_position.__next__())

        focus_up_button = tkinter.Button(
            top, text='Move focus up', command=focus_up)
        focus_down_button = tkinter.Button(
            top, text='Move focus down', command=focus_down)
        focus_up_button.place(x=x_position, y=y_position.__next__())
        focus_down_button.place(x=x_position, y=y_position.__next__())


    def toggle_zoom(self):
        if self.zoomed_in:
            CameraControls.zoom_out(self.c_p)
            self.zoomed_in = False
        else:
            zoom_in(self.c_p)
            self.zoomed_in = True
        self.update_qd_on_screen_targets(self.c_p['image'])

    def motor_scale_command(self, value):
        c_p = self.c_p
        value = float(value)/ 1000
        c_p['stepper_max_speed'] = [value, value, value]
        c_p['stepper_acc'] = [value*2, value*2, value*2]
        c_p['new_stepper_velocity_params'] = [True, True, True]

    def place_motor_speed_scale(self, top, x, y):

        self.motor_scale = tkinter.Scale(top, command=self.motor_scale_command,
        orient=HORIZONTAL, from_=0.5, to=50, resolution=0.5, length=200)
        self.motor_speed_label = Label(self.window, text="Stepper speed [microns/s]")
        self.motor_speed_label.place(x=x-10, y=y-15)
        self.motor_scale.place(x=x, y=y)

    def polymerization_time_command(self, value):
        # Sends the slider value to the arduino to update the polymerization
        # time.
        c_p = self.c_p
        c_p['polymerization_LED'] = 'S ' + str(value)

    def zoom_command(self, value, axis, max_width, indices):
        """
        Changes the AOI of the camera so that it is of width value centerd at
        the target position.
        """
        c_p = self.c_p
        try:
            width = int(value)
        except ValueError:
            print('Massive error! Cannot convert x-zoom value')
            return
        # Calcualte where the right and left position of the AOI is.
        # Needs to be divisble with 16 to be accepted by camera

        if axis == 0:
            center = c_p['traps_absolute_pos'][0][0]
        elif axis == 1:
            center = c_p['traps_absolute_pos'][1][0]
        else:
            print('Choice of axis is incorrect')
            return
        if center > max_width:
            center = int(max_width)/2
        # TODO fix so that we can zoom out properly also when the trap is not in the center!
        tmp = int(center  + width/2)
        right = int(min(tmp - (tmp%16), max_width ))

        tmp = int(center - width/2)
        left = int(max(tmp - (tmp%16), 0))

        # Check if we are at the edge on one side
        if left == 0:
            right = width
        if right == max_width:
            left = max_width - width
        c_p['AOI'][indices[0]] = left
        c_p['AOI'][indices[1]] = right
        c_p['new_settings_camera'] = True
        update_traps_relative_pos(c_p)


    def place_polymerization_time(self, top, x, y):

        self.polymerization_scale = tkinter.Scale(top,
            command=self.polymerization_time_command, from_=50,
            to=5000, resolution=50, orient=HORIZONTAL, length=150)
        self.polymerization_scale.set(1000)
        self.polymerization_label = Label(self.window, text="Polymerization time")
        self.polymerization_label.place(x=x-0, y=y-15)
        self.polymerization_scale.place(x=x, y=y)

    def set_downsample_level(self, value):
        self.c_p['downsample_rate'] = int(value)

    def place_manual_zoom_sliders(self, top, x, y, x1, y1):
        """
        Puts sliders used for controlling the AOI on the GUI.
        """
        c_p = self.c_p
        L = 150 # length of scale in pixels
        # TODO Fix problem here with autodetecting cameras
        self.x_zoom = partial(self.zoom_command, axis=0,
            max_width=c_p['camera_width'], indices=[0, 1])
        # Center does not get updated when using partial!
        self.x_zoom_slider = tkinter.Scale(top,
            command=self.x_zoom, from_=32,
            to=c_p['camera_width'], resolution=16, orient=HORIZONTAL, length=L)
        self.x_zoom_slider.set(c_p['camera_width'])
        self.x_slider_label = Label(self.window, text='x-zoom')
        self.x_slider_label.place(x=x, y=y-15)
        self.x_zoom_slider.place(x=x, y=y)

        self.y_zoom = partial(self.zoom_command, axis=1,
            max_width=c_p['camera_height'], indices=[2, 3])
        self.y_zoom_slider = tkinter.Scale(top,
            command=self.y_zoom, from_=32,
            to=c_p['camera_height'], resolution=16, orient=HORIZONTAL, length=L)
        self.y_zoom_slider.set(c_p['camera_height'])
        self.y_slider_label = Label(self.window, text='y-zoom')
        self.y_slider_label.place(x=x1, y=y1-15)
        self.y_zoom_slider.place(x=x1, y=y1)

    def set_exposure(self, entry=None):
        """
        Sets the exposure time of the camera to the value specified in entry
        """
        c_p = self.c_p
        if entry is None:
            print('No exposure time supplied')
            return

        if c_p['camera_model'] == 'basler_large' or 'basler_fast':
            try:
                exposure_time = int(entry)
            except ValueError:
                print('Cannot convert entry to integer')
            else:
                if 59 < exposure_time < 5e6:
                    c_p['exposure_time'] = exposure_time
                    print("Exposure time set to ", exposure_time)
                    c_p['new_settings_camera'] = True
                else:
                    print('Exposure time out of bounds!')
        elif c_p['camera_model'] == 'ThorlabsCam':
            try:
                exposure_time = float(entry)
            except ValueError:
                print('Cannot convert entry to float')
            else:
                if 0.01 < exposure_time < 120:
                    c_p['exposure_time'] = exposure_time
                    print("Exposure time set to ", exposure_time)
                    c_p['new_settings_camera'] = True
                else:
                    print('Exposure time out of bounds!')

    def set_temperature(self):
        """
        Sets the fixpoint temperature of the temperature controller.
        """
        c_p = self.c_p
        entry = self.temperature_entry.get()
        try:
            temperature = float(entry)
        except ValueError:
            print('Cannot convert entry to integer')
        else:
            if 20 < temperature < 40:
                c_p['setpoint_temperature'] = temperature
                print("Temperature set to ", temperature)
            else:
                print('Temperature out of bounds, it is no good to cook or \
                      freeze your samples')
        self.temperature_entry.delete(0, last=5000)

    def set_tracking_threshold(self):
        entry = threshold_entry.get()
        try:
            threshold = int(entry)
        except ValueError:
            print('Cannot convert entry to integer')
        else:
            if 0 < threshold < 255:
                c_p['particle_threshold'] = threshold
                print("Threshold set to ", threshold)
            else:
                print('Threshold out of bounds')
        threshold_entry.delete(0, last=5000)

    def add_arduino_buttons(self, top, generator_y, generator_y_2, x_position, x_position_2):
        '''
        Creates the buttons needed for the arduino.
        '''
        self.arduino_LED_button = tkinter.Button(
            top, text='Toggle LED', command=self.toggle_polymerization_LED)
        self.arduino_LED_button.place(x=x_position_2, y=generator_y_2.__next__())

        self.arduino_LED_pulse_button = tkinter.Button(
            top, text='Pulse LED', command=self.timed_polymerization)
        self.arduino_LED_pulse_button.place(x=x_position_2, y=generator_y_2.__next__())
        # TODO fix the button color
        self.bg_illumination_button = tkinter.Button(
            top, text='Toggle on BG illumination', command= partial(toggle_BG_shutter, self.c_p))
        self.bg_illumination_button.place(x=x_position, y=generator_y.__next__())
        self.green_laser_button = tkinter.Button(
            top, text='Toggle green laser', command= partial(toggle_green_laser, self.c_p))
        self.green_laser_button.place(x=x_position, y=generator_y.__next__())
        self.place_polymerization_time(top, x=x_position, y=generator_y.__next__())

    def add_piezo_activation_buttons(self, top, x_position, y_position):

        c_p = self.c_p
        c_p['piezos_activated'] = tkinter.BooleanVar()
        self.piezo_checkbutton = tkinter.Checkbutton(top, text='Use piezos',\
        variable=c_p['piezos_activated'], onvalue=True, offvalue=False)
        self.piezo_checkbutton.place(x=x_position, y=y_position.__next__())
        # self.training_data_button = tkinter.Checkbutton(top, text='Generate training data',\
        # variable=c_p['generate_training_data'], onvalue=True, offvalue=False)
        # self.training_data_button.place(x=x_position, y=y_position.__next__())

    def add_qd_tracking_buttons(self, top,x_position, x_position_2, y_position,
                        y_position_2):
        """
        Adds the buttons needed for the qd tracking.
        """

        c_p = self.c_p
        self.next_qd_button = tkinter.Button(top, text='Next QD position',
            command=self.increment_QD_count)
        self.previous_qd_button = tkinter.Button(top, text='Previous QD position',
            command=self.decrement_QD_count)

        self.next_qd_button.place(x=x_position_2, y=y_position_2.__next__())
        self.previous_qd_button.place(x=x_position_2, y=y_position_2.__next__())
        self.auto_move_button = tkinter.Checkbutton(top, text='Move QDs automatically',
            variable=c_p['move_QDs'], onvalue=True, offvalue=False)
        self.auto_move_button.place(x=x_position_2, y=y_position_2.__next__())
        self.to_array_position_button = tkinter.Checkbutton(top, text='Move to array pos',\
        variable=c_p['position_QD_in_pattern'], onvalue=True, offvalue=False)
        self.to_array_position_button.place(x=x_position, y=y_position.__next__())
        self.override_trapped_button = tkinter.Checkbutton(top, text='Set QD trapped',\
        variable=c_p['override_QD_trapped'], onvalue=True, offvalue=False)
        c_p['override_QD_trapped'].set(False)
        self.override_trapped_button.place(x=x_position, y=y_position.__next__())
        self.reset_array_printer_button = tkinter.Button(top, text='Reset printer',
        command=self.reset_printing_parameters)
        self.reset_array_printer_button.place(x=x_position, y=y_position.__next__())

        c_p['test_move_down'] = tkinter.BooleanVar()
        self.move_down_button = tkinter.Checkbutton(top, text='Move down QDs',
            variable=c_p['test_move_down'], onvalue=True, offvalue=False)
        self.move_down_button.place(x=x_position, y=y_position.__next__())

    def save_bg(self):
        # Saves an image to be used as background when subtracting bg
        c_p = self.c_p
        c_p['raw_background'] = np.copy(c_p['image'])
        c_p['background'] = self.resize_display_image(np.int16(np.copy(c_p['image'])))
        print('BG saved')

    def snapshot(self, label=None):
        """
        Saves the latest frame captured by the camera into a jpg image.
        Will label it with date and time if no label is provided. Image is
        put into same folder as videos recorded.
        """
        c_p = self.c_p
        if label == None:
            image_name = c_p['recording_path'] + "/frame-" +\
                        time.strftime("%d-%m-%Y-%H-%M-%S") +\
                        ".jpg"
        else:
            image_name = c_p['recording_path'] + '/' + label + '.jpg'

        tmp = np.copy(c_p['image'])
        if c_p['bg_removal']:
            try:
                tmp = subtract_bg(np.copy(c_p['image']),
                c_p['raw_background'][c_p['AOI'][2]:c_p['AOI'][3], c_p['AOI'][0]:c_p['AOI'][1]])
            except AssertionError as AE:
                # No worries this happens from time to time
                pass

        cv2.imwrite(image_name, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
        # Check if downsamlping is being done
        if self.downsample.get():
           binned_name = c_p['recording_path'] + '/' + image_name[:-4] + 'binned.jpg'
           binned_image =  convolve_binning(np.copy(c_p['image']), c_p['downsample_rate'])
           cv2.imwrite(binned_name, cv2.cvtColor(binned_image, cv2.COLOR_RGB2BGR))

        np.save(image_name[:-4], c_p['image'])
        print('Took a snapshot of the experiment.')

    def toggle_bg_removal(self):
        # TODO have all these toggle functions as lambda functions
        self.c_p['bg_removal'] = not self.c_p['bg_removal']

    def set_exposure_dialog(self):
        user_input = simpledialog.askstring("Exposure time dialog",
        " Set exposure time (microseconds) ")
        self.set_exposure(user_input)

    def set_target_position_dialog(self, axis=2):
        """
        Sends command to move motor to target position.
        """

        labels = ["x", "y", "z"]

        user_input = simpledialog.askstring("Target position dialog",
        f"Set target position of {labels[axis]}-axis motor (millimeter) ")
        try:
            position = float(user_input)
        except ValueError:
            print("Cannot convert input to position")
            return
        if not self.c_p['steppers_connected'][axis]:
            print('Stepper not connected!')
            return
        # Check that position is within the okay bounds
        if 0 >= position or 8 <= position:
            print("Target position out of bounds!")
            return

        # Tell the motors to move to the target position
        self.c_p['stepper_next_move'][axis] = position
        self.c_p['stepper_move_to_target'][axis] = True

    def max_fps_dialog(self):
        """
        Opens a dialog box prompting the user to enter a new fps for the
        camera
        """
        if self.c_p['camera_model'] == 'ThorlabsCam':
            # Not implemented in thorlabs_capture yet :/
            return
        user_input = simpledialog.askstring("fps dialog",
        " Set fps (frames per second) ")
        try:
            FPS = float(user_input)

        except ValueError:
            showinfo("Warning!",
                "Entered fps could not be converted to number!")
        if FPS <= 0:
            # Raise exception
            raise ValueError("Trying to set invalid FPS!")
        self.c_p['max_fps'] = FPS
        self.c_p['new_settings_camera'] = True


    def show_camera_info(self):
        """
        Opens a small pop-up window with basic camera info such as model,
        fps, image size and exposure time.
        """
        c_p = self.c_p
        camera_info = f"Model: {c_p['camera_model']} , fps: {c_p['fps']} ,exposure time: {c_p['exposure_time']}\n"
        camera_AOI_info = f"Current area of interest: {c_p['AOI']}, maximum image width: {c_p['camera_width']} , maximum image height: {c_p['camera_height']}"
        camera_AOI_info += f"\n recording format: .{c_p['video_format']}"
        showinfo(title="Camera feeed info", message=camera_info+camera_AOI_info)

    def set_save_folder(self):
        # Ask user for a folder to save in
        folder_ = filedialog.askdirectory()
        # Check so that the dialog was not cancelled and a proper path was returned
        if len(folder_) > 1:
            # Update recording path
            self.c_p['recording_path'] = folder_
            # This may give errors with the numpy-save function

    def save_c_p(self):
        pass

    def create_control_menus(self):
        """
        Creates the Basic controls and Camera menus on top of the GUI.
        """
        # TODO Save settings? Use pickle for this?

        self.control_menu.add_command(label="Save position", command=self.save_starting_position)
        self.control_menu.add_command(label="Exit program", command=self.window.quit)
        self.control_menu.add_command(label='Connect/disconnect steppers', command=self.connect_stepper_motors)
        self.control_menu.add_command(label='Connect/disconnect piezos', command=self.connect_stage_piezos)
        self.control_menu.add_command(label='Select experiment schedule', command=self.read_experiment_dictionary)
        x_move = partial(self.set_target_position_dialog, 0)
        y_move = partial(self.set_target_position_dialog, 1)
        self.control_menu.add_command(label='Set target position x', command=x_move)
        self.control_menu.add_command(label='Set target position y', command=y_move)
        self.control_menu.add_command(label='Set target position z', command=self.set_target_position_dialog)
        self.control_menu.add_command(label='Set save folder', command=self.set_save_folder)
        self.control_menu.add_separator()
        self.menubar.add_cascade(label="Basic controls", menu=self.control_menu)

        self.camera_menu = Menu(self.menubar)
        self.camera_menu.add_command(label="Snapshot", command=self.snapshot)
        self.camera_menu.add_command(label="Save bg", command=self.save_bg)
        self.camera_menu.add_command(label="Toggle bg removal", command=self.toggle_bg_removal)
        self.camera_menu.add_command(label="Exposure time", command=self.set_exposure_dialog)
        self.camera_menu.add_command(label="Show camera info", command=self.show_camera_info)
        self.camera_menu.add_command(label="Set max fps", command=self.max_fps_dialog)

        c_p = self.c_p

        def set_mp4_format():
            c_p['video_format'] = "mp4"

        def set_avi_format():
            c_p['video_format'] = "avi"

        def set_npy_format():
            c_p['video_format'] = "npy"

        self.video_format_menu = Menu(self.window)
        self.video_format_menu.add_command(label='mp4', command=set_mp4_format)
        self.video_format_menu.add_command(label='avi', command=set_avi_format)
        self.video_format_menu.add_command(label='npy', command=set_npy_format)
        self.camera_menu.add_cascade(label="Recording format",
            menu=self.video_format_menu)
        self.menubar.add_cascade(label="Camera control", menu=self.camera_menu)

    def toggle_recording(self):
        '''
        Button function for starting of recording
        '''
        c_p = self.c_p
        c_p['recording'] = not c_p['recording']
        if c_p['recording']:
            c_p['video_name'] = CameraControls.get_video_name(c_p=c_p)

    def create_buttons(self, top=None):
        '''
        This function generates all the buttons for the interface along with
        the other control elementsn such as entry boxes.
        '''
        # TODO make c_p non global, and change so that only the buttons actually
        # usable are displayed. Split this function into several smaller ones

        c_p = self.c_p
        self.canvas.bind("<Button-1>", self.screen_click)
        if top is None:
            top = self.window
        x_position = 1310
        x_position_2 = 1500
        y_position = self.get_y_separation()
        y_position_2 = self.get_y_separation()

        def home_z_command():
            c_p['return_z_home'] = not c_p['return_z_home']

        self.recording_button = tkinter.Button(top, text='Start recording',
                                             command=self.toggle_recording)
        self.home_z_button = tkinter.Button(top, text='Toggle home z',
                                            command=home_z_command)
        # toggle_bright_particle_button = tkinter.Button(
        #     top, text='Toggle particle brightness',
        #     command=toggle_bright_particle)



        self.zoom_button = tkinter.Button(top, text='Zoom in', command=self.toggle_zoom)

        temperature_output_button = tkinter.Button(top,
            text='toggle temperature output', command=toggle_temperature_output)


        # experiment_schedule_button = tkinter.Button(top,
        #     text='Select experiment schedule',
        #     command=self.read_experiment_dictionary
        #     )
        self.display_laser = tkinter.BooleanVar()
        self.diplay_laser_button = tkinter.Checkbutton(top, text='Laser indicator',\
        variable=self.display_laser, onvalue=True, offvalue=False)
        self.display_laser.set(True)

        # Place all the buttons, starting with first column
        if c_p['standard_motors'] or c_p['stage_piezos']:
            #self.add_move_buttons(top, x_position, y_position)
            self.move_to_target_button = tkinter.Button(top, \
                text='Toggle move to target', command=self.toggle_move_piezo_to_target)
            self.move_to_target_button.place(x=x_position, y=y_position.__next__())
        if c_p['temperature_controlls']:
            self.temperature_entry = tkinter.Entry(top, bd=5)
            self.temperature_entry.place(x=x_position, y=y_position.__next__())
            temperature_button = tkinter.Button(
                top, text='Set setpoint temperature', command=self.set_temperature)
            temperature_button.place(x=x_position, y=y_position.__next__())

        self.recording_button.place(x=x_position, y=y_position.__next__())
        # toggle_bright_particle_button.place(x=x_position, y=y_position.__next__())
        self.tracking_toggled = tkinter.BooleanVar()
        self.tracking_toggled.set(False)

        if c_p['tracking']:
            self.toggle_tracking_button = tkinter.Checkbutton(top, text='Enable tracking',\
                    variable=self.tracking_toggled, onvalue=True, offvalue=False)
            self.toggle_tracking_button.place(x=x_position, y=y_position.__next__())

            threshold_entry = tkinter.Entry(top, bd=5)
            threshold_button = tkinter.Button(
                top, text='Set threshold', command=self.set_tracking_threshold)
            threshold_entry.place(x=x_position, y=y_position.__next__())
            threshold_button.place(x=x_position, y=y_position.__next__())

        self.zoom_button.place(x=x_position, y=y_position.__next__())

        # Downsample button and variable
        self.downsample = tkinter.BooleanVar()
        c_p['downsampling'] = self.downsample
        self.downsample_button = tkinter.Checkbutton(top, text='Binning of livefeed',\
        variable=self.downsample, onvalue=True, offvalue=False)
        self.downsample_button.place(x=x_position, y=y_position.__next__())
        self.downsample.set(False)

        self.downsample_label = tkinter.Label(top, text='Binning factor')
        y = y_position.__next__()
        self.downsample_label.place(x=x_position, y=y)
        self.downsample_slider = tkinter.Scale(top, command=self.set_downsample_level,
            orient=HORIZONTAL, from_=1, to=10, resolution=1, length=150)
        self.downsample_slider.place(x=x_position, y=y+20)
        self.downsample_slider.set(c_p['downsample_rate'])
        y_position.__next__()

        # Laser position checkbutton. Cannot be used at same time as click move
        self.set_laser_position = tkinter.BooleanVar()
        self.laser_position_button = tkinter.Checkbutton(top, text='Set laser position',
            variable=self.set_laser_position, onvalue=True, offvalue=False)
        self.laser_position_button.place(x=x_position_2, y=y_position_2.__next__())

        # Add column separator
        self.columnn_separator = ttk.Separator(top, orient='vertical')
        self.columnn_separator.place(x=x_position_2-10,y=0, width=5, height=self.canvas_height-200)
        # Second column
        self.diplay_laser_button.place(x=x_position_2, y=y_position_2.__next__())

        # Motor buttons. Attributes of UserInterface class os we can easily change
        # the description text of them.
        if c_p['standard_motors']:
            self.toggle_motorX_button = tkinter.Button(
                top, text='Connect motor x', command=self.connect_disconnect_motorX)
            self.toggle_motorY_button = tkinter.Button(
                top, text='Connect motor y', command=self.connect_disconnect_motorY)
            self.toggle_piezo_button = tkinter.Button(
                top, text='Connect piezo motor', command=self.connect_disconnect_piezo)
            self.toggle_motorX_button.place(x=x_position_2, y=y_position_2.__next__())
            self.toggle_motorY_button.place(x=x_position_2, y=y_position_2.__next__())
            self.toggle_piezo_button.place(x=x_position_2, y=y_position_2.__next__())
            self.home_z_button.place(x=x_position_2, y=y_position_2.__next__())

        if c_p['shutter']:
            self.open_shutter_button = tkinter.Button(
                top, text='Open shutter', command=self.open_shutter)
            self.open_shutter_button.place(x=x_position_2, y=y_position_2.__next__())

        if c_p['arduino_LED']:
            self.add_arduino_buttons(top, y_position, y_position_2, x_position,
                            x_position_2)

        if c_p['QD_tracking']:
            self.add_qd_tracking_buttons(top, x_position, x_position_2, y_position,
                        y_position_2)
        if c_p['stage_piezos']:
            self.add_piezo_activation_buttons(top, x_position_2, y_position_2)

        if c_p['using_stepper_motors']:
            self.add_stepper_buttons(top, y_position_2, x_position_2 )
        y_position.__next__()
        y0 = y_position.__next__()
        y_position.__next__()
        y1 = y_position.__next__()
        self.place_manual_zoom_sliders(top, x=x_position, y=y0,
            x1=x_position, y1=y1)
        y1 = y_position_2.__next__()
        y_position_2.__next__()
        y_position_2.__next__()
        y2 = y_position_2.__next__()

        self.create_indicators(x=x_position_2, y1=y1, y2=y2)
        y_position_2.__next__()
        self.move_by_clicking = tkinter.BooleanVar()

        if c_p['stage_piezos'] or c_p['using_stepper_motors'] or c_p['standard_motors']:
            self.motor_control_buttons(top, y_position_2, x_position_2)

    def create_SLM_window(self, _class):
        try:
            if self.new.state() == "normal":
                self.new.focus()
        except Exception as ex:
            self.new = tkinter.Toplevel(self.window)
            self.SLM_Window = _class(self.new, self.c_p)

    def get_temperature_info(self):
        """
        Returns a string containg a description of the statys of the temperature
        controller. Specifically current, temperature, target temperature,
        connection status and if the temperature is stable or not.
        """
        c_p = self.c_p
        if c_p['temperature_controller_connected']:
            temperature_info = 'Current objective temperature is: '
            temperature_info += str(c_p['current_temperature'])+' C'
            temperature_info += '\n setpoint temperature is: '
            temperature_info += str(c_p['setpoint_temperature'])+' C'

            if c_p['temperature_stable']:
                temperature_info += '\nTemperature is stable. '
            else:
                temperature_info += '\nTemperature is not stable. '
            if c_p['temperature_output_on']:
                temperature_info += '\n Temperature controller output is on.'
            else:
                temperature_info += '\n Temperature controller output is off.'
        else:
            temperature_info = 'Temperature controller is not connected.'
        return temperature_info

    def get_position_info(self):
        # TODO: this does not need to be updated all the time...
        c_p = self.c_p
        # Add position info
        target_key_motor = None
        target_key_connection = None
        position_text = ''

        if c_p['standard_motors']:
            target_key_motor = 'motor_current_pos'
            target_key_connection = 'motors_connected'
        elif  c_p['using_stepper_motors']:
            target_key_motor = 'stepper_current_position'
            target_key_connection = 'steppers_connected'

        if target_key_motor is not None:
            position_text = 'x: '+str(c_p[target_key_motor][0])+\
                'mm   y: '+str(c_p[target_key_motor][1])+\
                'mm   z: '+str(c_p[target_key_motor][2])+'\n'
            # Add motor connection info
            x_connected = 'connected. ' if c_p[target_key_connection][0] else 'disconnected.'
            y_connected = 'connected. ' if c_p[target_key_connection][1] else 'disconnected.'
            z_connected = 'connected. ' if c_p[target_key_connection][2] else 'disconnected.'

            position_text += 'Motor-X is ' + x_connected
            position_text += ' Motor-Y is ' + y_connected + '\n'
            position_text += ' Focus (z) motor is ' + z_connected + '\n'

        if c_p['stage_piezos']:
            if c_p['stage_piezo_connected'][0]:
                position_text += 'Piezos are connected:'
            else:
                position_text += 'Piezos are NOT connected.'
            position_text += 'Piezo x: ' + str(c_p['piezo_current_position'][0]) + '\n'
            position_text += 'Piezo y: ' + str(c_p['piezo_current_position'][1]) + '\n'
            position_text += 'Piezo z: ' + str(c_p['piezo_current_position'][2]) + '\n'

        if c_p['tracking']:
            position_text += '\n Experiments run ' + str(c_p['experiment_progress'])
            position_text += ' out of ' + str(c_p['nbr_experiments']) + '  '
            position_text += str(c_p['experiment_runtime']) + 's run out of ' + str(c_p['recording_duration'])
            position_text += '\n Current search direction is: ' + str(c_p['search_direction'] + '\n')

        return position_text

    def update_motor_buttons(self):
        # Motor connection buttons
        c_p = self.c_p
        x_connect = 'Disconnect' if c_p['connect_motor'][0] else 'Connect'
        self.toggle_motorX_button.config(text=x_connect + ' motor x')
        y_connect = 'Disconnect' if c_p['connect_motor'][1] else 'Connect'
        self.toggle_motorY_button.config(text=y_connect + ' motor y')
        piezo_connected = 'Disconnect' if c_p['connect_motor'][2] else 'Connect'
        self.toggle_piezo_button.config(text=piezo_connected + ' piezo motor')

    def update_home_button(self):

        c_p = self.c_p
        if c_p['return_z_home']:
            self.home_z_button.config(text='Z is homing. Press to stop')
        else:
            self.home_z_button.config(text='Press to home z')

    def create_indicators(self, x=1420, y1=760, y2=1100):
        c_p = self.c_p

        self.position_label = Label(self.window, text=self.get_position_info())
        self.position_label.place(x=x, y=y1)
        self.temperature_label = Label(self.window, text=self.get_temperature_info())
        self.temperature_label.place(x=x, y=y2)
        self.saving_video_label = Label(self.window, text="No video is being saved")
        self.saving_video_label.config(fg='green')
        self.saving_video_label.place(x=x, y=y2+100)

    def update_indicators(self):
        '''
        Helper function for updating on-screen indicators
        '''
        c_p = self.c_p
        # Update if recording is turned on or not
        if c_p['recording']:
            self.recording_button.config(text='Turn off recording', bg='green')
        else:
            self.recording_button.config(text='Turn on recording', bg='red')

        if not c_p['saving_video']:
            self.saving_video_label.config(text="No video is being saved",
                fg='green')
        else:
            txt = f"Video being saved, {c_p['frame_queue'].qsize()} frames left to process"
            self.saving_video_label.config(text=txt, fg='red')

        if c_p['arduino_LED']:
            if c_p['polymerization_LED_status'] == 'ON':
                self.arduino_LED_button.config(bg='green')
            else:
                self.arduino_LED_button.config(bg='red')

            if c_p['background_illumination']:
                self.bg_illumination_button.config(bg='green')
            else:
                self.bg_illumination_button.config(bg='red')
            if c_p['green_laser']:
                self.green_laser_button.config(bg='green')
            else:
                self.green_laser_button.config(bg='red')


        # Update "move to target button", may not exist
        if c_p['stage_piezos']:
            if c_p['piezo_move_to_target'][0] or c_p['piezo_move_to_target'][1]:
                self.move_to_target_button.config(bg='green')
            else:
                self.move_to_target_button.config(bg='red')
            if c_p['piezo_move_to_target'][2]:
                self.to_focus_button.config(bg='green')
            else:
                self.to_focus_button.config(bg='red')
        if self.zoomed_in:
            # TODO integrate zoom button with the zoom scrollers
            self.zoom_button.config(text='Zoom out')
        else:
            self.zoom_button.config(text='Zoom in')

        self.temperature_label.config(text=self.get_temperature_info())
        self.position_label.config(text=self.get_position_info())

        # Update the shutter if it is used
        if c_p['shutter']:
            if c_p['shutter_open']:
                self.open_shutter_button.config(bg='green',text='close shutter')
            else:
                self.open_shutter_button.config(bg='red',text='open shutter')

        # If standard motors are use then update these
        if c_p['standard_motors']:
            self.update_motor_buttons()
            self.update_home_button()

        #if c_p['using_stepper_motors'] or c_p['stage_piezos']:
        c_p['mouse_move_allowed'] = self.move_by_clicking.get()

    def resize_display_image(self, img):
        # Reshapes the image img to fit perfectly into the tkninter window.

        c_p = self.c_p
        img_size = np.shape(img)
        if img_size[1]==self.canvas_width or img_size[0] == self.canvas_height:
            return img

        if img_size[1]/self.canvas_width > img_size[0]/self.canvas_height:
            dim = (int(self.canvas_width/img_size[1]*img_size[0]), int(self.canvas_width))
        else:
            dim = ( int(self.canvas_height), int(self.canvas_height/img_size[0]*img_size[1]))
        self.image_scale = max(img_size[1]/self.canvas_width, img_size[0]/self.canvas_height)

        # Compensate for downsampling in image_scale
        if self.downsample.get():
            self.image_scale *= c_p['downsample_rate']

        return cv2.resize(img, (dim[1],dim[0]), interpolation = cv2.INTER_AREA)

    def get_mouse_position(self):

        self.c_p['mouse_position'][0] = self.window.winfo_pointerx() - self.window.winfo_rootx()
        self.c_p['mouse_position'][1] = self.window.winfo_pointery() - self.window.winfo_rooty()

    def mouse_command_move(self):
        # Function to convert a mouse click to a movement.

        # TODO add parameter for camera orientaiton(0,90,180,270 degrees tilt)
        # Update target position for the stepper stage.
        # Camera orientation will affect signs in the following expressions

        # Account for camera rotation
        c_p = self.c_p
        x_rot = 1
        y_rot = 1
        if c_p['camera_orientatation'] == 'up':
            # No change needed
            pass
        elif c_p['camera_orientatation'] == 'down':
            y_rot = -1
            x_rot = -1
        elif c_p['camera_orientatation'] == 'left':
            # Not implemented yet
            pass
        else: # Right
            # Not implemented yet
            pass
        # Calculate travel distance
        dx0 = float(c_p['traps_relative_pos'][0,0] - self.image_scale * c_p['mouse_position'][0])
        dx = dx0 / c_p['mmToPixel']
        dy0 = float(c_p['traps_relative_pos'][1,0] - self.image_scale * c_p['mouse_position'][1])
        dy = dy0 / c_p['mmToPixel']
        # Checks which motors are connected and activated, acts accordingly.
        if c_p['stage_piezos'] and c_p['using_stepper_motors'] and \
        c_p['piezos_activated'].get() and c_p['stepper_activated'].get():

            if 1<c_p['piezo_current_position'][0] - (dx * 1000) < 19:
                c_p['piezo_target_position'][0] -= (dx * 1000.0)
            else:
                c_p['stepper_target_position'][0] = c_p['stepper_current_position'][0] - dx

            if 1<c_p['piezo_current_position'][1] - (dy * 1000) < 19:
                c_p['piezo_target_position'][1] -= (dy * 1000.0)
            else:
                c_p['stepper_target_position'][1] = c_p['stepper_current_position'][1] - dy

        elif c_p['stage_piezos'] and c_p['piezos_activated'].get():
            if 1<c_p['piezo_current_position'][0] - (dx * 1000) < 19:
                c_p['piezo_target_position'][0] -= (dx * 1000.0)
            if 1<c_p['piezo_current_position'][1] - (dy * 1000) < 19:
                c_p['piezo_target_position'][1] -= (dy * 1000.0)

        elif c_p['using_stepper_motors'] and c_p['stepper_activated'].get():
            c_p['stepper_target_position'][0] = c_p['stepper_current_position'][0] - dx
            c_p['stepper_target_position'][1] = c_p['stepper_current_position'][1] - dy

        # TODO add case for the old motors as well
        if c_p['standard_motors']:
            print('Moving standard motors')
            c_p['motor_movements'][0] = - dx0
            c_p['motor_movements'][1] = dy0

    def add_laser_cross(self, image):

        c_p = self.c_p
        try:
            y = int(c_p['traps_relative_pos'][0][0])
            x = int(c_p['traps_relative_pos'][1][0])

        except ValueError:
            print("Could not convert trap positions to integers")
            return
        else:
            try:
                image[x-10:x+10, y] = 0
                image[x, y-10:y+10] = 0
            except IndexError as e:
                print('Could not display laser position ', e)

    def update_qd_on_screen_targets(self, image=None):
        '''
        Draws the locations in which there should be QDs. The locations are drawn
        relative to the laser.
        TODO: Make the markers a different color
        '''
        c_p = self.c_p
        if image is None:
            s = np.shape(c_p['image'])
        else:
            s = np.shape(image)
        # Extract laser position
        x = int(c_p['traps_relative_pos'][0][0])
        y = int(c_p['traps_relative_pos'][1][0])
        c_p['QD_position_screen_x'] = []
        c_p['QD_position_screen_y'] = []
        # Calculate distance from laser to target location
        # Get index so that we don't get an error when QDs_placed == len(QD_target_loc_x)
        index = min(c_p['QDs_placed'], len(c_p['QD_target_loc_x'])-1)

        separation_x = c_p['QD_target_loc_x'][index] * c_p['mmToPixel']/1000 - x
        separation_y = c_p['QD_target_loc_y'][index] * c_p['mmToPixel']/1000 - y

        for x_loc, y_loc in zip(c_p['QD_target_loc_x'], c_p['QD_target_loc_y']):
            # Calcualte where in the image the markers should be put
            yc = int(x_loc * c_p['mmToPixel']/1000 - separation_x)
            xc = int(y_loc * c_p['mmToPixel']/1000 - separation_y)
            c_p['QD_position_screen_x'].append(xc)
            c_p['QD_position_screen_y'].append(yc)

    def add_target_QD_locs(self, image):
        '''
        Draws the locations in which there should be QDs. The locations are drawn
        relative to the laser.
        TODO: Make the markers a different color
        '''
        c_p = self.c_p
        s = np.shape(image)
        cross = np.int16(np.linspace(-5,5,11))
        for xc, yc in zip(c_p['QD_position_screen_x'], c_p['QD_position_screen_y']):

            # Check that the marker lies inside the image
            if 5 < xc < s[0]-5 and 5 < yc < s[1]-5:
                image[xc+cross, yc+cross] = 0
                image[xc+cross, yc-cross] = 0
        if c_p['tracking_on']:
            try:
                # Set pixelvalues of image 0 to mark the spots
                # TODO should check that indices ok earlier
                image[c_p['QD_target_loc_y_px']-3:c_p['QD_target_loc_y_px']+3,
                c_p['QD_target_loc_x_px']-3:c_p['QD_target_loc_x_px']+3] = 0
            except IndexError as IE:
                pass

    def mark_polymerized_areas(self, image):
        """
        Marks the polynerized areas in an image if the positions of these are
        known.
        """
        c_p = self.c_p
        if self.tracking_toggled.get():
            for x, y in zip(c_p['polymerized_x'], c_p['polymerized_y']):
                x = int(x)
                y = int(y)
                try:
                    image[y-4:y+4, x] = 0
                    image[y, x-4:x+4] = 0
                except IndexError:
                    # Locations outside the image, nothihg to worry about
                    pass

    def crop_in(self, image, edge=500):
        """
        Crops in on an area around the laser for easier viewing.
        """
        c_p = self.c_p
        # Check if we can do crop
        top = int(max(c_p['traps_absolute_pos'][0][0]-edge, 0))
        bottom = int(min(c_p['traps_absolute_pos'][0][0]+edge, np.shape(image)[0]))
        left = int(max(c_p['traps_absolute_pos'][1][0]-edge, 0))
        right = int(min(c_p['traps_absolute_pos'][1][0]+edge, np.shape(image)[1]))
        c_p['traps_relative_pos'][0,0] = bottom - c_p['traps_absolute_pos'][0][0]
        c_p['traps_relative_pos'][1,0] = right - c_p['traps_absolute_pos'][1][0]
        print(c_p['traps_relative_pos'][0,0], c_p['traps_relative_pos'][1,0])

        return image[left:right, top:bottom]

    def add_particle_positions_to_image(self, image):
        c_p = self.c_p
        for x,y in zip(c_p['particle_centers'][0], c_p['particle_centers'][1]):
            try:
                x = int(x)
                y = int(y)
                image[x-10:x+10, y] = 0
                image[x, y-10:y+10] = 0
                image[x-3:x+3, y-3:y+3] = 0
            except (ValueError, IndexError):
                print(' Warning could not display particle at position: ', x, y)

    def update(self):
         """
         Updates the live video feed and all monitors(motor positions, temperature etc).
         This function calls itself recursively.
         """
         c_p = self.c_p
         image = np.asarray(c_p['image'])
         image = image.astype('uint16')
         self.update_qd_on_screen_targets()

         if self.display_laser.get():
             self.add_laser_cross(image)
             self.mark_polymerized_areas(image)

         if c_p['display_target_QD_positions']:
             self.add_target_QD_locs(image)

         if c_p['crop_in']:
             image = self.crop_in(image)

         if c_p['phasemask_updated']:
              print('New phasemask')
              self.SLM_Window.update()
              c_p['phasemask_updated'] = False

         c_p['scroll_for_z'] = self.z_scrolling.get()
         if c_p['stage_piezos'] or c_p['using_stepper_motors']:
             # TODO this case statement occurs in several places. Make it better
             compensate_focus_xy_move(c_p)
        # Binn the image to increase effective sensitivity
         if self.downsample.get():
            image = convolve_binning(image, c_p['downsample_rate'])

         self.update_indicators()
         c_p['tracking_on'] = self.tracking_toggled.get()

         image = self.resize_display_image(image)
         # TODO implement convolution alternative to the downsampling
         # Now do the background removal on the significantly smaller image.
         if c_p['bg_removal']:
             try:
                if np.shape(image) == np.shape(c_p['background']):
                    image = subtract_bg(image, c_p['background'])
                elif np.shape(c_p['raw_background']) == (c_p['camera_height'], c_p['camera_width']):
                    tmp = np.copy(c_p['raw_background'][c_p['AOI'][2]:c_p['AOI'][3], c_p['AOI'][0]:c_p['AOI'][1]])
                    c_p['background'] = self.resize_display_image(tmp)
                    # TODO There is a problem here with the resizing sometimes
                    # Suspect it has to do with the fact that the target AOI is not always accepted

             except AssertionError:
                 # Image is not the same size as background
                 c_p['bg_removal'] = False
                 print('Saved background does not match shape of current image!')

         # TODO make it possible to use 16 bit color(ie 12 from camera)
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
         # need to use a compatible image type
         self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

         # Update mini-window
         # TODO THIS MINI WINDOW WAS EATING UP VERY MUCH RESOURCES
         #self.create_trap_image()
         #self.mini_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.mini_image, mode='RGB'))
         # need to use a compatible image type
         #self.mini_canvas.create_image(0, 0, image = self.mini_photo, anchor = tkinter.NW)
         self.window.after(self.delay, self.update)


def compensate_focus_xy_move(c_p):
    '''
    Function for compensating the change in focus caused by x-y movement.
    Returns the positon in ticks which z  should take to compensate for the focus
    '''
    dx = (c_p['stepper_current_position'][0] - c_p['stepper_starting_position'][0])
    dy = (c_p['stepper_current_position'][1] - c_p['stepper_starting_position'][1])
    dz = (c_p['tilt'][0] * dx) + (c_p['tilt'][1] * dy)

    if c_p['stage_piezos']:
        z0 = c_p['piezo_starting_position'][2]
        zt = z0 + c_p['piezo_elevation'] # + dz/1000
        if 0.1 < zt < 19.9:
            c_p['piezo_target_position'][2] = zt
            #return
        elif zt <= 0:
            c_p['piezo_target_position'][2] = 0.1
        else:
            c_p['piezo_target_position'][2] = 19.9

    z0 = c_p['stepper_starting_position'][2]
    target_pos = z0 + dz + c_p['stepper_elevation']
    c_p['stepper_target_position'][2] = target_pos
    return

def compensate_focus():
    '''
    Function for compensating the change in focus caused by x-y movement.
    Returns the positon in ticks which z  should take to compensate for the focus
    '''
    c_p = self.c_p
    new_z_pos = (c_p['z_starting_position']
        +c_p['z_x_diff']*(c_p['motor_starting_pos'][0] - c_p['motor_current_pos'][0])
        +c_p['z_y_diff']*(c_p['motor_starting_pos'][1] - c_p['motor_current_pos'][1]) )
    new_z_pos += c_p['temperature_z_diff']*(c_p['current_temperature']-c_p['starting_temperature'])
    return int(new_z_pos)

# Not used anymore
class ExperimentControlThread(threading.Thread):
   '''
   Thread which does the tracking.
   '''
   def __init__(self, threadID, name, c_p):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.setDaemon(True)

   def __del__(self):
       self.c_p['tracking_on'] = False

   def catch_particle(self, min_index_trap=None, min_index_particle=None):
        '''
        Function for determimning where and how to move when min_index_particle
        has been found
        '''
        c_p = self.c_p
        if min_index_particle is not None:
          c_p['target_trap_pos'] = [c_p['traps_relative_pos'][0][min_index_trap],
                                    c_p['traps_relative_pos'][1][min_index_trap]]
          c_p['target_particle_center'] = [c_p['particle_centers'][0][min_index_particle],
                                            c_p['particle_centers'][1][min_index_particle]]

          if True in c_p['traps_occupied']:
              c_p['xy_movement_limit'] = 40
              # Some traps are occupied. Want to avoid catching more than one
              # particle per trap.
              filled_traps_locs = []
              for idx, occupied in enumerate(c_p['traps_occupied']):
                  if occupied:
                      filled_traps_locs.append([c_p['traps_relative_pos'][0][idx],
                      c_p['traps_relative_pos'][1][idx] ])
              x, y, success = path_search(filled_traps_locs,
                              target_particle_location=c_p['target_particle_center'],
                              target_trap_location=c_p['target_trap_pos'])
          else:
              success = False
              c_p['xy_movement_limit'] = 1200
          if success:
              c_p['motor_movements'][0] = -x
              c_p['motor_movements'][1] = y
          else:
              c_p['motor_movements'][0] = -(c_p['target_trap_pos'][0] - c_p['target_particle_center'][0]) # Note: Sign of this depends on setup
              c_p['motor_movements'][1] = c_p['target_trap_pos'][1] - c_p['target_particle_center'][1]

        else:
            c_p['target_particle_center'] = []

   def lift_for_experiment(self, patiance=3):
       '''
       Assumes that all particles have been caught.
       patiance, how long(s) we allow a trap to be unoccipied for

       Returns true if lift succeded
       '''
       z_starting_pos = compensate_focus()
       patiance_counter = 0
       c_p = self.c_p
       print('Lifting time. Starting from ', z_starting_pos, ' lifting ',
            c_p['target_experiment_z'])
       while c_p['target_experiment_z'] > c_p['motor_current_pos'][2] - z_starting_pos:
            time.sleep(0.2)
            all_filled, nbr_particles, min_index_trap, min_index_particle  =\
                self.check_exp_conditions()
            if all_filled:
                c_p['z_movement'] = 40
                c_p['return_z_home'] = False
                patiance_counter = 0
            else:
                patiance_counter += 1
            if patiance_counter >= patiance or not c_p['tracking_on']:
                c_p['return_z_home'] = True
                c_p['z_movement'] = 0
                return False
       print('Lifting done. Now at',  c_p['motor_current_pos'][2])
       return True

   def check_exp_conditions(self, tracking_func=None):
        '''
        Checks if all traps are occupied. Returns true if this is the case.
        Tries to catch the closes unoccupied particle.
        '''
        c_p = self.c_p
        if tracking_func is None:
            x, y = fpt.find_particle_centers(copy.copy(image),
                      threshold=c_p['particle_threshold'],
                      particle_size_threshold=c_p['particle_size_threshold'],
                      bright_particle=c_p['bright_particle'])
        else:
            x, y = tracking_func(copy.copy(image))

        c_p['particle_centers'] = [x, y]
        c_p['traps_occupied'] = [False for i in range(len(c_p['traps_absolute_pos'][0]))]
        min_index_trap, min_index_particle = find_closest_unoccupied()

        if False not in c_p['traps_occupied']:
            # All traps have been occupied
            return True, -1, min_index_trap, min_index_particle
        # Not all traps have been occupied, might need to go searching
        return False, len(x), min_index_trap, min_index_particle

   def run_experiment(self, duration):
        '''
        Run an experiment for 'duration'.
        Returns 0 if it ran to the end without interruption otherwise it
        returns the amount of time remaining of the experiment.
        '''
        start = time.time()
        now = datetime.now()
        time_stamp = str(now.hour) + '-' + str(now.minute)

        # Taking snapshot before zooming in so user can see if there are
        # particles in the background which may interfere with measurement.

        #snapshot(c_p['measurement_name']+'_pre'+time_stamp)
        zoom_in(self.c_p)
        self.c_p['recording'] = True
        patiance = 50
        patiance_counter = 0
        while time.time() <= start + duration and self.c_p['tracking_on']:
            all_filled, nbr_particles, min_index_trap, min_index_particle  =\
                self.check_exp_conditions()
            if all_filled:
                patiance_counter = 0
                time.sleep(1)
                c_p['experiment_runtime'] = np.round(time.time() - start)
            else:
                patiance_counter += 1
            if patiance_counter > patiance:
                break
        self.c_p['recording'] = False
        zoom_out()
        #snapshot(c_p['measurement_name']+'_after'+time_stamp)
        if time.time() >= start + duration:
            return 0
        return start + duration - time.time()

   def add_ghost_traps(self, ghost_traps_x, ghost_traps_y, ghost_traps_z=None):
       '''
       Function for adding ghost traps after having lifted the particles.
       '''
       # Update number of ghost traps.
       self.c_p['nbr_ghost_traps'] = len(ghost_traps_x)

       # Convert traps positions to SLM positions.
       if min(ghost_traps_x) >= 1:
           ghost_traps_x = pixels_to_SLM_locs(ghost_traps_x, 0)
       if min(ghost_traps_y) >= 1:
           ghost_traps_y = pixels_to_SLM_locs(ghost_traps_y, 1)
       if ghost_traps_z is None:
           ghost_traps_z = np.zeros(self.c_p['nbr_ghost_traps'])

       # Append ghost traps to xm, ym and zm
       self.c_p['xm'] += ghost_traps_x
       self.c_p['ym'] += ghost_traps_y
       self.c_p['zm'] += ghost_traps_z

       # Update the phasemask
       self.c_p['new_phasemask'] = True
       while self.c_p['new_phasemask'] and self.c_p['tracking_on']:
           time.sleep(0.1)

   def run(self):
        '''
        Plan - have an experiment procedure.
        Before each experiment is allowed to start make sure all traps
        which are supposed to be filled are filled.
        Then lift the particles and start recording. If any particle
        is dropped go down and refill the traps and continue* the
        experiment.
        In between each experiment(when experiment parameters are to be changed)
        try to move the particles which are already trapped rather than
        cathing new ones (unless this is necessary). Then change all desired parameters.

        Control the experiments with a experiment Dictionary which
        keeps track of 'c_p' which are to be changed during the experiment.
        For instance might desire to change LGO orders as well as
        particle distance and temperature,then this should be possible

        * Do not record the full length but only the missing part of the video
        '''

        # TODO make the program understand when two particles have been trapped
        # in the same trap. - Possible solution: Train an AI to detect this.
        global image
        c_p = self.c_p
        c_p['nbr_experiments'] = len(c_p['experiment_schedule'])
        c_p['experiment_progress'] = 0

        while c_p['program_running']: # Change to continue tracking?
            time.sleep(0.3)
            # Look through the whole schedule, a list of dictionaries.

            if c_p['tracking_on'] and c_p['experiment_progress'] < c_p['nbr_experiments']:
                setup_dict = c_p['experiment_schedule'][c_p['experiment_progress']]
                print('Next experiment is', setup_dict)
                run_finished = False
                update_c_p(c_p, setup_dict)
                full_xm = copy.copy(c_p['xm']) # for adding one trap at a time.
                full_ym = copy.copy(c_p['ym']) # Using copy since
                time_remaining = c_p['recording_duration']
                all_filled, nbr_particles, min_index_trap, min_index_particle = self.check_exp_conditions()

                # Check if we need to go down and look for more particles in
                # between the experiments
                if all_filled:
                    nbr_active_traps = len(full_xm)
                else:
                    # Not all traps were filled, need to home and activate
                    # only the first trap
                    c_p['return_z_home'] = True
                    time.sleep(1)
                    if c_p['activate_traps_one_by_one']:
                        nbr_active_traps = min(3,len(full_xm))
                        active_traps_dict = {'xm':full_xm[:nbr_active_traps],
                            'ym':full_ym[:nbr_active_traps]}
                        update_c_p(c_p, active_traps_dict)
                # Start looking for particles.
                while not run_finished and c_p['tracking_on']:
                   time.sleep(0.3)
                   # We are (probably) in full frame mode looking for a particle

                   all_filled, nbr_particles, min_index_trap, min_index_particle = self.check_exp_conditions()

                   if not all_filled and nbr_particles <= c_p['traps_occupied'].count(True):
                       # Fewer particles than traps. Look for more particles.
                       c_p['return_z_home'] = True
                       search_for_particles()

                   elif not all_filled and nbr_particles > c_p['traps_occupied'].count(True):
                       # Untrapped particles and unfilled traps. Catch particles
                       self.catch_particle(min_index_trap=min_index_trap,
                            min_index_particle=min_index_particle)

                   elif all_filled:

                       # All active traps are filled, activate a new one if
                       # there are more to activate
                       if len(c_p['xm']) < len(full_xm):
                            nbr_active_traps += 1
                            active_traps_dict = {'xm':full_xm[:nbr_active_traps],
                                'ym':full_ym[:nbr_active_traps]}
                            update_c_p(c_p, active_traps_dict)

                       # No more traps to activate, can start lifting.
                       elif self.lift_for_experiment():
                           print('lifted!')

                           if 'ghost_traps_x' in setup_dict:
                               print('Adding ghost traps')
                               ghost_traps_z = None if 'ghost_traps_z' not in setup_dict else setup_dict['ghost_traps_z']
                               self.add_ghost_traps(setup_dict['ghost_traps_x'],
                                                    setup_dict['ghost_traps_y'],
                                                    ghost_traps_z)
                               # Will currently add ... which will
                           # Particles lifted, can start experiment.
                           time_remaining = self.run_experiment(time_remaining)
                       else:
                           c_p['return_z_home'] = True

                   if time_remaining < 1:
                       run_finished = True
                       c_p['experiment_progress'] += 1

        c_p['return_z_home'] = True
        print('I AM DONE NOW QUITTING!')


def get_adjacency_matrix(nx, ny):
    '''
    Function for calculating the adjacency matrix used in graph theory to
    describe which nodes are neighbours.
    '''
    X, Y = np.meshgrid(
        np.arange(0, nx),
        np.arange(0, ny)
    )

    nbr_nodes = nx*ny
    XF = np.reshape(X, (nbr_nodes, 1))
    YF = np.reshape(Y, (nbr_nodes, 1))
    adjacency_matrix = np.zeros((nbr_nodes, nbr_nodes))
    for idx in range(nx*ny):
        distance_map = (X - XF[idx])**2 + (Y - YF[idx])**2
        adjacency_matrix[idx, :] = np.reshape(distance_map, (nbr_nodes)) <= 3
        adjacency_matrix[idx, idx] = 0
    return adjacency_matrix


def path_search(filled_traps_locs, target_particle_location,
                target_trap_location, c_p):
    '''
    Function for finding paths to move the stage so as to trap more particles
    without accidentally trapping extra particles.
    Divides the AOI into a grid and calculates the shortest path to the trap
    without passing any already occupied traps.

    Parameters
    ----------
    filled_traps_locs : TYPE list of list of
        traps locations [[x1, y1], [x2, y2]...]
        DESCRIPTION.
    target_particle_location : TYPE list [x,y] of target particle location [px]
        DESCRIPTION.
    target_trap_location : TYPE list of target trap location [px]
        DESCRIPTION.

    Returns
    -------
    TYPE move_x, move_y, success
        DESCRIPTION. The move to make to try and trap the particle without it
        getting caught in another trap along the way
        success - True if path was not blocked by other particles and a move
        was found. False otherwise.

    '''

    nx = int( (c_p['AOI'][1]-c_p['AOI'][0]) / c_p['cell_width'])
    ny = int( (c_p['AOI'][3]-c_p['AOI'][2]) / c_p['cell_width'])
    X, Y = np.meshgrid(
        np.arange(0, nx),
        np.arange(0, ny)
    )


    nbr_nodes = nx*ny
    node_weights = 1e6 * np.ones((nbr_nodes, 1))  # Initial large weights
    unvisited_set = np.zeros(nbr_nodes)  # Replace with previous nodes
    previous_nodes = -1 * np.ones(nbr_nodes)

    def matrix_to_array_index(x, y, nx):
        return x + y * nx

    def array_to_matrix_index(idx, nx):
        y = idx // nx
        x = np.mod(idx, nx)
        return x, y

    def loc_to_index(x, y, nx):
        x = int(x/c_p['cell_width'])
        y = int(y/c_p['cell_width'])
        return matrix_to_array_index(x, y, nx)

    adjacency_matrix = get_adjacency_matrix(nx, ny)

    trap_radii = 3
    for location in filled_traps_locs:
        x = location[0] / c_p['cell_width']
        y = location[1] / c_p['cell_width']
        distance_map = (X - x)**2 + (Y - y)**2
        indices = [i for i, e in enumerate(np.reshape(distance_map, (nbr_nodes))) if e < trap_radii]
        adjacency_matrix[:, indices] = 0
        node_weights[indices] = 50
        node_weights[matrix_to_array_index(int(x), int(y), nx)] = 40
        unvisited_set[matrix_to_array_index(int(x), int(y), nx)] = 1

    target_node = loc_to_index(target_trap_location[0],
                               target_trap_location[1], nx)
    current_node = loc_to_index(target_particle_location[0],
                                target_particle_location[1], nx)

    # Djikstra:
    node_weights[current_node] = 0
    unvisited_set[current_node] = 1
    previous_nodes[current_node] = 0

    def update_dist(current_node, adjacency_indices, node_weights,
                    previous_nodes):
        for index in adjacency_indices:
            if node_weights[current_node] + 1 < node_weights[index]:
                node_weights[index] = node_weights[current_node] + 1
                # All distances are either inf or 1
                previous_nodes[index] = current_node

    def find_next_node(unvisited_set, node_weights):
        res_list = [i for i, value in enumerate(unvisited_set) if value == 0]
        min_value = 1e6
        min_idx = -1
        for index in res_list:
            if node_weights[index] < min_value:
                min_idx = index
                min_value = node_weights[index]
        return min_idx

    iterations = 0
    while unvisited_set[target_node] == 0 and iterations <= nbr_nodes:
        adjacency_indices = [i for i, e in enumerate(adjacency_matrix[current_node,:]) if e == 1]
        update_dist(current_node, adjacency_indices,
                    node_weights, previous_nodes)
        current_node = find_next_node(unvisited_set, node_weights)
        unvisited_set[current_node] = 1
        iterations += 1

    previous = target_node
    node_weights[previous]
    prev_x = []
    prev_y = []

    while previous != 0:
        node_weights[previous] = -3
        tmp_x, tmp_y = array_to_matrix_index(previous, nx)
        prev_x.append(tmp_x)
        prev_y.append(tmp_y)
        previous = int(previous_nodes[previous])
        if previous == -1:
            break

    if previous == -1:
        return 0, 0, False
    elif len(prev_x) > 3:
        x_move = prev_x[-3] * c_p['cell_width'] - target_particle_location[0]
        # SHould be -2 but was very slow
        y_move = prev_y[-3] * c_p['cell_width'] - target_particle_location[1]
        return x_move, y_move, True
    else:
        try:
            x_move = prev_x[-2] * c_p['cell_width'] - target_particle_location[0]
            y_move = prev_y[-2] * c_p['cell_width'] - target_particle_location[1]
            return x_move, y_move, True
        except Exception as ex:
            return 0, 0, False


def update_c_p(c_p, update_dict, wait_for_completion=True):
    '''
    Simple function for updating c_p['keys'] with new values 'values'.
    Ensures that all updates where successfull.
    Parameter wait_for_completion should be set to True if there is a need
    to wait for phasemask to be finished updating before continuing the program.

    Highly suggest usage of measurement_name to make it easier to keep track of
    the data.
    '''

    ok_parameters = ['use_LGO', 'LGO_order', 'xm', 'ym', 'zm', 'setpoint_temperature',
    'recording_duration', 'target_experiment_z', 'SLM_iterations',
    'temperature_output_on','activate_traps_one_by_one','need_T_stable',
    'measurement_name','phasemask','QD_target_loc_x','QD_target_loc_y']

    requires_new_phasemask = ['use_LGO', 'LGO_order', 'xm', 'ym', 'zm', 'SLM_iterations']

    for key in update_dict:
        if key in ok_parameters:
            try:
                if key == 'xm' and min(update_dict[key]) > 1:
                    c_p[key] = pixels_to_SLM_locs(update_dict[key], 0)
                    print('xm ' ,update_dict[key])
                elif key == 'ym' and min(update_dict[key]) > 1:
                    c_p[key] = pixels_to_SLM_locs(update_dict[key], 1)
                    print('ym ',update_dict[key])
                else:
                    c_p[key] = update_dict[key]

            except Exception as ex:
                print(f"Could not update control parameter: {key},{ex}")
                return
        else:
            print('Invalid key: ', key)

    # Need to update measurement_name even if it is not in measurement to
    # prevent confusion

    if 'ghost_traps_x' or 'ghost_traps_y' not in update_dict:
        c_p['nbr_ghost_traps'] = 0

    # Check that both xm and ym are updated
    if len(c_p['xm']) > len(c_p['ym']):
        c_p['xm'] = c_p['xm'][:len(c_p['ym'])]
        print(f" WARNING! xm and ym not the same length, cutting off xm!")
    if len(c_p['ym']) > len(c_p['xm']):
        c_p['ym'] = c_p['ym'][:len(c_p['xm'])]
        print(" WARNING! xm and ym not the same length, cutting off ym!")

    # update phasemask. If there was an old one linked use it.
    if 'phasemask' not in update_dict:
        for key in update_dict:
            if key in requires_new_phasemask:
                c_p['new_phasemask'] = True
    else:
        # Manually change phasemask without SLM thread
        c_p['phasemask_updated'] = True
        SLM_loc_to_trap_loc(c_p['xm'], c_p['ym'])

    # Wait for new phasemask if user whishes this
    while c_p['new_phasemask'] and wait_for_completion:
        time.sleep(0.3)

    # Check if there is an old phasemask to be used.
    # Await stable temperature
    while c_p['need_T_stable'] and not c_p['temperature_stable'] and\
        c_p['temperature_controller_connected']:
        time.sleep(0.3)


def count_interior_particles(margin=30):
    '''
    Function for counting the number of particles in the interior of the frame.
    margin
    '''
    c_p = self.c_p
    interior_particles = 0
    for position in c_p['particle_centers']:
        interior_particles += 1

    return interior_particles


def in_focus(margin=40):
    '''
    Function for determining if a image is in focus by looking at the intensity
    close to the trap positions and comparing it to the image median.
    # Highly recommended to change to only one trap before using this function
    # This function is also very unreliable. Better to use the old method +
    # deep learning I believe.
    '''

    global image
    c_p = self.c_p
    median_intesity = np.median(image)


    if c_p['camera_model'] == 'basler_fast':
        image_limits = [672, 512]
    else:
        image_limits = [1200, 1000]

    left = int(max(min(c_p['traps_absolute_pos'][0]) - margin, 0))
    right = int(min(max(c_p['traps_absolute_pos'][0]) + margin, image_limits[0]))
    up = int(max(min(c_p['traps_absolute_pos'][1]) - margin, 0))
    down = int(min(max(c_p['traps_absolute_pos'][1]) + margin, image_limits[1]))

    expected_median = (left - right) * (up - down) * median_intesity
    actual_median = np.sum(image[left:right,up:down])

    print(median_intesity,actual_median)
    print( expected_median + c_p['focus_threshold'])
    if actual_median > (expected_median + c_p['focus_threshold']):
        return True
    return False


def predict_particle_position_network(network,c_p, half_image_width=50,
    network_image_width=101,
    print_position=False):
    '''
    Function for making g a prediciton with a network and automatically updating the center position array.
    inputs :
        network - the network to predict with. Trained with deeptrack
        half_image_width - half the width of the image. needed to convert from deeptrack output to pixels
        network_image_width - Image width that the newtwork expects
        print_position - If the predicted positon should be printed in the console or not
    Outputs :
        Updates the center position of the particle
    '''
    global image
    resized = cv2.resize(copy.copy(image), (network_image_width,network_image_width), interpolation = cv2.INTER_AREA)
    pred = network.predict(np.reshape(resized/255,[1,network_image_width,network_image_width,1]))

    c_p['target_particle_center'][0] = half_image_width + pred[0][1] * half_image_width
    c_p['target_particle_center'][1] = half_image_width + pred[0][0] * half_image_width

    if print_position:
        print('Predicted posiiton is ',c_p['particle_centers'])


def get_particle_trap_distances(c_p):
    '''
    Calcualtes the distance between all particles and traps and returns a
    distance matrix, ordered as distances(traps,particles),
    To clarify the distance between trap n and particle m is distances[n][m].
    '''
    update_traps_relative_pos(c_p) # just in case
    nbr_traps = len(c_p['traps_relative_pos'][0])
    nbr_particles = len(c_p['particle_centers'][0])
    distances = np.ones((nbr_traps, nbr_particles))

    for i in range(nbr_traps):
        for j in range(nbr_particles):
            dx = (c_p['traps_relative_pos'][0][i] - c_p['particle_centers'][0][j])
            dy = (c_p['traps_relative_pos'][1][i] - c_p['particle_centers'][1][j])
            distances[i,j] = np.sqrt(dx * dx + dy * dy)
    return distances


def trap_occupied(distances, trap_index, c_p):
    '''
    Checks if a specific trap is occupied by a particle. If so set that trap to occupied.
    Updates if the trap is occupied or not and returns the index of the particle in the trap
    '''

    # Check that trap index is ok
    if trap_index > len(c_p['traps_occupied']) or trap_index < 0:
        print('Trap index out of range')
        return None
    for i in range(len(distances[trap_index, :])):
        dist_to_trap = distances[trap_index, i]
        if dist_to_trap <= c_p['movement_threshold']:
            c_p['traps_occupied'][trap_index] = True
            return i
    try:
        c_p['traps_occupied'][trap_index] = False
        return None
    except IndexError:
        print(" Indexing error for trap index", str(trap_index),\
        " length is ",len(c_p['traps_occupied']))
        return None


def check_all_traps(c_p, distances=None):
    '''
    Updates all traps to see if they are occupied.
    Returns the indices of the particles which are trapped. Indices refers to their
    position in the c_p['particle_centers'] array.
    Returns an empty array if there are no trapped particles.

    '''

    if distances is None:
        distances = get_particle_trap_distances()
    trapped_particle_indices = []
    nbr_traps = len(distances)
    for trap_index in range(nbr_traps):
        # Check for ghost traps
        if trap_index >= nbr_traps - c_p['nbr_ghost_traps']:
            c_p['traps_occupied'][trap_index] = True
        else:
            # Check occupation of non ghost traps.
            trapped_particle_index = trap_occupied(distances, trap_index)
            if trapped_particle_index is not None:
                trapped_particle_indices.append(trapped_particle_index)
    return trapped_particle_indices


def find_closest_unoccupied(c_p):
    '''
    Function for finding the paricle and (unoccupied) trap which are the closest
    Returns : min_index_trap,min_index_particle.
        Index of the untrapped particle closest to an unoccupied trap.
    '''

    distances = get_particle_trap_distances()
    trapped_particles = check_all_traps(distances)
    distances[:,trapped_particles] = 1e6 # These indices are not ok, put them very large

    min_distance = 2000 # If the particle is not within 2000 pixels then it is not within the frame
    min_index_particle = None # Index of particle which is closes to an unoccupied trap
    min_index_trap = None # Index of unoccupied trap which is closest to a particle

    # Check all the traps
    for trap_idx in range(len(c_p['traps_occupied'])):
        trapped = c_p['traps_occupied'][trap_idx]

        # If there is not a particle trapped in the trap check for the closest particle
        if not trapped and len(distances[0]) > 0:
            # Had problems with distances being [] if there were no particles
            particle_idx = np.argmin(distances[trap_idx])

            # If particle is within the threshold then update min_index and trap index as well as min distance
            if distances[trap_idx,particle_idx]<min_distance:
                min_distance = distances[trap_idx,particle_idx]
                min_index_trap = trap_idx
                min_index_particle = particle_idx

    return min_index_trap, min_index_particle


def move_button(move_direction, c_p):
    '''
    Button function for manually moving the motors a bit
    The direction refers to the direction a particle in the fiel of view will move on the screen
    move_direction = 0 => move up
    move_direction = 1 => move down
    move_direction = 2 => move right
    move_direction = 3 => move left
    '''
    move_distance = 200
    if move_direction==0:
        # Move up (Particles in image move up on the screen)
        c_p['motor_movements'][1] = move_distance
    elif move_direction==1:
        # Move down
        c_p['motor_movements'][1] = -move_distance
    elif move_direction==2:
        # Move right
        c_p['motor_movements'][0] = move_distance
    elif move_direction==3:
        # Move left
        c_p['motor_movements'][0] = -move_distance
    else:
        print('Invalid move direction')


def stage_piezo_manual_move(axis, distance, c_p):
    # Manually move the piezos a given distance(in microns)
    c_p['piezo_target_position'][axis] += distance
    # Check if intended move is out of bounds
    c_p['piezo_target_position'][axis] = max(0.0, c_p['piezo_target_position'][axis])
    c_p['piezo_target_position'][axis] = min(20.0, c_p['piezo_target_position'][axis])

def toggle_temperature_output(c_p):
    '''
    Function for toggling temperature output on/off.
    '''
    c_p['temperature_output_on'] = not c_p['temperature_output_on']
    print("c_p['temperature_output_on'] set to",c_p['temperature_output_on'])


def toggle_bright_particle(c_p):
    '''
    Function for switching between bright and other particle
    '''
    c_p['bright_particle'] = not c_p['bright_particle']
    print("c_p['bright_particle'] set to",c_p['bright_particle'])


def focus_up(c_p):
    '''
    Used for focus button to shift focus slightly up
    '''
    c_p['piezo_target_position'][2] += 0.1


def focus_down(c_p):
    '''
    Used for focus button to shift focus slightly up
    '''
    c_p['piezo_target_position'][2] -= 0.1

def zoom_in(c_p, margin=60, use_traps=False):
    '''
    Helper function for zoom button and zoom function.
    Zooms in on an area around the traps
    '''
    if c_p['camera_model'] == 'ThorlabsCam':
        left = max(min(c_p['traps_absolute_pos'][0]) - margin, 0)
        left = int(left // 20 * 20)
        right = min(max(c_p['traps_absolute_pos'][0]) + margin, 1200)
        right = int(right // 20 * 20)
        up = max(min(c_p['traps_absolute_pos'][1]) - margin, 0)
        up = int(up // 20 * 20)
        down = min(max(c_p['traps_absolute_pos'][1]) + margin, 1000)
        down = int(down // 20 * 20)

    elif c_p['camera_model'] == 'basler_large':
        margin = 500
        left = max(min(c_p['traps_absolute_pos'][0]) - margin, 0)
        left = int(left // 16 * 16)
        right = min(max(c_p['traps_absolute_pos'][0]) + margin, 3600)
        right = int(right // 16 * 16)
        up = max(min(c_p['traps_absolute_pos'][1]) - margin, 0)
        up = int(up // 16 * 16)
        down = min(max(c_p['traps_absolute_pos'][1]) + margin, 3008)
        down = int(down // 16 * 16)
        print('basler large zoom in')
    else:
        left = max(min(c_p['traps_absolute_pos'][0]) - margin, 0)
        left = int(left // 16 * 16)
        right = min(max(c_p['traps_absolute_pos'][0]) + margin, 672)
        right = int(right // 16 * 16)
        up = max(min(c_p['traps_absolute_pos'][1]) - margin, 0)
        up = int(up // 16 * 16)
        down = min(max(c_p['traps_absolute_pos'][1]) + margin, 512)
        down = int(down // 16 * 16)
    # Note calculated fps is automagically saved.
    CameraControls.set_AOI(c_p, left=left, right=right, up=up, down=down)
    update_traps_relative_pos(c_p)
    print(c_p['traps_relative_pos'][0][0], c_p['traps_relative_pos'][1][0])

def stepper_button_move_down(c_p, distance=0.002):
    # Moves the z-motor of the stepper up a tiny bit
    c_p['stepper_target_position'][2] = c_p['stepper_current_position'][2] - distance


def stepper_button_move_upp(c_p, distance=0.002):
    # Moves the z-motor of the stepper up a tiny bit
    c_p['stepper_target_position'][2] = c_p['stepper_current_position'][2] + distance

def search_for_particles(c_p):
    '''
    Function for searching after particles. Threats the sample as a grid and
    systmatically searches it
    '''
    x_max = 0.005 # [mm]
    delta_y = 3 # [mm]
    # Make movement
    if c_p['search_direction']== 'right':
        c_p['motor_movements'][0] = 300 # [px]

    elif c_p['search_direction']== 'left':
        c_p['motor_movements'][0] = -300

    elif c_p['search_direction']== 'up':
        c_p['motor_movements'][1] = 300

    elif c_p['search_direction']== 'down': # currently not used
        c_p['motor_movements'][1] = -300

    if c_p['search_direction']== 'up' and \
        (c_p['motor_current_pos'][1]-c_p['motor_starting_pos'][1])>delta_y:
        c_p['search_direction']= 'right'
        c_p['x_start'] = c_p['motor_current_pos'][0]
        print('changing search direction to right, x_start is ',c_p['x_start'] )

    if c_p['search_direction']== 'right' and \
        (c_p['motor_current_pos'][0]-c_p['x_start'])>x_max:

        if c_p['motor_current_pos'][1] - c_p['motor_starting_pos'][1]>delta_y/2:
            c_p['search_direction'] = 'down'
            print('changing search direction to down')
        else:
            c_p['search_direction'] = 'up'
            print('changing search direction to up')
    if c_p['search_direction']== 'down' \
        and (c_p['motor_current_pos'][1] - c_p['motor_starting_pos'][1])<0:
        c_p['search_direction']=='right'
        print('changing search direction to right')


def pixels_to_SLM_locs(c_p, locs, axis):
    '''
    Function for converting from PIXELS to SLM locations.
    '''
    if axis != 0 and axis != 1:
        print('cannot perform conversion, incorrect choice of axis')
        return locs
    offset = c_p['slm_x_center'] if not axis else c_p['slm_y_center']
    new_locs = [((x - offset) / c_p['slm_to_pixel']) for x in locs]
    return new_locs


def SLM_loc_to_trap_loc(c_p, xm, ym):
    '''
    Fucntion for updating the traps position based on their locaitons
    on the SLM.
    '''
    tmp_x = [x * c_p['slm_to_pixel'] + c_p['slm_x_center'] for x in xm]
    tmp_y = [y * c_p['slm_to_pixel'] + c_p['slm_y_center'] for y in ym]
    tmp = np.asarray([tmp_x, tmp_y])
    c_p['traps_absolute_pos'] = tmp
    print('Traps are at: ', c_p['traps_absolute_pos'] )
    update_traps_relative_pos(c_p)


def save_phasemask(c_p):
    # Helperfunction for saving the SLM.
    # Should probably save parameters of this at same time as well.

    now = datetime.now()
    phasemask_name = c_p['recording_path'] + '/phasemask-'+\
        c_p['measurement_name'] + '-' + str(now.hour) + '-' + str(now.minute) +\
        '-' + str(now.second) + '.npy'
    np.save(phasemask_name, c_p['phasemask'], allow_pickle=True)

def move_particles_slowly(c_p, last_d=30e-6):
    # Function for moving the particles between the center and the edges
    # without dropping then
    if last_d>40e-6 or last_d<0:
        print('Too large distance.')
        return
    if last_d>c_p['trap_separation']:
        while c_p['trap_separation']<last_d:
            if c_p['new_phasemask']==False:
                if last_d - c_p['trap_separation'] < 1e-6:
                    c_p['trap_separation'] = last_d
                else:
                    c_p['trap_separation'] += 1e-6
                c_p['new_phasemask'] = True
                print(c_p['trap_separation'])
                time.sleep(0.5)
            time.sleep(1)
    else:
        while c_p['trap_separation']>last_d:
            if c_p['new_phasemask']==False:
                if c_p['trap_separation'] - last_d < 1e-6:
                    c_p['trap_separation'] = last_d
                else:
                    c_p['trap_separation'] -= 1e-6
                c_p['new_phasemask'] = True
                print(c_p['trap_separation'])
                time.sleep(0.5)
            time.sleep(1)
    return
