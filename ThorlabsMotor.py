'''
File containing a simplified interface for thorlabs motors. Supports
the max381 stage as well as kcube motors. The motors are handled as threads
'''

# Import packages
from ctypes import *
import clr,sys
from System import Decimal,Int32
from time import sleep
from threading import Thread

# Import DLLs
"""
Note when usin this code on other computer than the one in the biophysics lab these paths may need changing.
"""
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.DeviceManagerCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.GenericMotorCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.DCServoCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.InertialMotorCLI.dll ')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.GenericPiezoCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.PiezoCLI.dll')

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *

timeoutVal = 30000

class PiezoMotor():
    '''
    Piezo motor class.
    '''

    def __init__(self, serialNumber, channel, pollingRate=250, timeout=10000):
        self.serial_number = serialNumber
        self.polling_rate = pollingRate
        self.connect_piezo_motor()
        self.channel = channel
        self.timeout = timeout

    def connect_piezo_motor(self):
        self.motor = InitiatePiezoMotor(self.serial_number, self.polling_rate)
        self.is_connected = True if not self.motor is None else False

    def move_to_position(self, position):
        '''
        Function for moving the motor to a specified position

        Parameters
        ----------
        position : Float, given in units of mm.
            Target position to move motor to.

        Returns
        -------
        bool
            True if move was successfull otherwise false.

        '''
        try:
            self.motor.MoveTo(self.channel, position, self.timeout)
            return True
        except:
            print('Could not move piezo to target position')
            return False

    def set_timeout(self, timeout):
        '''
        Function for setting the timeout of the piezo motor

        Parameters
        ----------
        timeout : int
            Timeout of motot in ms.

        Returns
        -------
        bool
            True if the timeout was okay and set.

        '''
        if timeout >= 1:
            self.timeout = timeout
            return True
        else:
            print("Timeout NOK")
            return False

    def get_timeout(self):
        '''
        Returns
        -------
        int
            Timeout of piezo motor.

        '''
        return self.timeout

    def move_relative(self, distance):
        '''
        Function for moving the piezo a fixed distance relative to it's
        current position.

        Parameters
        ----------
        distance : Int
            Distance in ticks to move.

        Returns
        -------
        boolean
            True if move was successfull otherwise false.

        '''
        target_position = self.get_position()+distance
        return self.move_to_position(target_position)

    def get_position(self):
        '''
        Returns
        -------
        int
            Current position of piezo(in ticks).

        '''
        try:
            return self.motor.GetPosition(self.channel)
        except:
            print('Could not find piezo position')
            return 0

    def disconnect_piezo(self):
        if self.is_connected:
            self.motor.StopPolling()
            self.motor.Disconnect()
            self.is_connected = False

    def __del__(self):
        self.disconnect_piezo()


def InitiatePiezoMotor(serialNumber, pollingRate=250):
    '''
    Function for initalizing a piezo motor.

    Parameters
    ----------
    serialNumber : String
        Serialnumber of controller which is being contacted.
    pollingRate : float, optional
        The default is 250. Polling rate of controller.

    Returns
    -------
    motor :
        A PiezoMotor controller object. None if initalization failed.

    '''
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    motor = KCubeInertialMotor.CreateKCubeInertialMotor(serialNumber)
    for attempts in range(3):
        try:
            motor.Connect(serialNumber)
        except:
            print("Connection attempt", attempts, "failed")
            if attempts < 2:
                print("Will wait 2 seconds and try again")
                sleep(2)
            else:
                print("Cannot connect to device.\n Please ensure that the \
                      device is connected to your computer and not in use in\
                          any other program!")
                return None
    motor.WaitForSettingsInitialized(5000)
    # configure the stage
    motorSettings = motor.GetInertialMotorConfiguration(serialNumber)
    motorSettings.DeviceSettingsName = 'PIA'
    # update the RealToDeviceUnit converter
    motorSettings.UpdateCurrentConfiguration()
    # push the settings down to the device
    currentDeviceSettings = ThorlabsInertialMotorSettings.GetSettings(motorSettings)

    motor.SetSettings(currentDeviceSettings, True, False)
    # Start polling and enable the device
    motor.StartPolling(pollingRate)
    motor.EnableDevice()

    return motor


class StageMotor():
    '''
    Class for the motors used by the stage. Class currently not in use
    '''
    def __init__(self, serialNumber, pollingRate=200, mmToPixel=16140,timeoutVal=30000):
        self.serialNumber = serialNumber
        self.pollingRate = pollingRate
        self.timeoutVal = timeoutVal
        self.connect_motor()
        self.mmToPixel = mmToPixel

    def SetJogSpeed(self,jogSpeed,jogAcc=0.1):
        try:
            self.motor.SetJogVelocityParams(Decimal(jogSpeed),Decimal(jogAcc))
        except:
            print('Could not set jog speed.')

    def connect_motor(self):
        self.motor = InitiateMotor(self.serialNumber, self.pollingRate)
        self.is_connected = False if self.motor is None else True
        if self.is_connected:
            self.startingPosition = self.motor.GetPosition()
        else:
            self.startingPosition = 0

    def disconnect_motor(self):
        if self.is_connected:
            motor.StopPolling()
            motor.Disconnect()
            self.is_connected = False

    def MoveMotor(self, distance):
        '''
        Helper function for moving a motor.

        Parameters
        ----------
        motor : thorlabs motor
            Motor to be moved.
        distance : float
            Distance to move the motor.

        Returns
        -------
        bool
            True if the move was a success, otherwise false.

        '''
        if not self.is_connected:
            return False

        if distance > 0.1 or distance < -0.1:
            print("Trying to move too far")
            return False
        # For unknown reason python thinks one first must convert to float but
        # only when running from console...
        self.motor.SetJogStepSize(Decimal(float(distance)))
        try:
            motor.MoveJog(1, timeoutVal)# Jog in forward direction
        except:
            print( "Trying to move motor to NOK position")
            return False
        return True
    def MoveMotorPixels(self, distance):
        # TODO - check if the mmToPixel value is valid for the basler camera.
        '''
        Moves motor a specified number of pixels.

        Parameters
        ----------
        motor : TYPE - thorlabs motor
             Motor to be moved
        distance : TYPE number
             Distance to move the motor
        mmToPixel : TYPE number for converting from mm(motor units) to pixels, optional
             The default is 16140, valid for our 100x objective and setup.

        Returns
        -------
        bool
            True if move was successfull, false otherwise.
        '''
        self.motor.SetJogStepSize(Decimal(float(distance/self.mmToPixel)))
        try:
            self.motor.MoveJog(1, timeoutVal)  # Jog in forward direction
        except:
            print( "Trying to move motor to NOK position")
            return False
        return True


    def MoveMotorToPixel(self, targetPixel, currentPixel, maxPixel=1280):
        '''

        Parameters
        ----------
        motor : TYPE
            DESCRIPTION.
        targetPixel : TYPE
            DESCRIPTION.
        currentPixel : TYPE
            DESCRIPTION.
        maxPixel : TYPE, optional
            DESCRIPTION. The default is 1280.
        mmToPixel : TYPE, optional
            DESCRIPTION. The default is 16140.

        Returns
        -------
        bool
            DESCRIPTION.

        '''
        if(targetPixel < 0 or targetPixel > maxPixel): # Fix correct boundries
            print("Target pixel outside of bounds")
            return False
        if not self.is_connected:
            return False
        # There should be a minus here, this is due to the setup
        dx = -(targetPixel-currentPixel)/self.mmToPixel
        self.motor.SetJogStepSize(Decimal(float(dx)))
        try:
            self.motor.MoveJog(1,timeoutVal)# Jog in forward direction
        except:
            print( "Trying to move motor to NOK position")
            return False
        return True

    def __del__(self):
        self.motor.StopPolling()
        self.motor.Disconnect()


def InitiateMotor(serialNumber, pollingRate=250, DeviceSettingsName='Z812'):
    '''
    Function for initalizing contact with a thorlabs k-cube controller object.

    Parameters
    ----------
    serialNumber : String
        Serial number of device to be connected. Written on the back of the
    pollingRate : int, optional
        Polling rate of device in ms. The default is 250.
    DeviceSettingsName : string, optional
        Indicates which type of motor is connectd to the controller.
        The default is 'Z812'.

    Returns
    -------
    motor : k-cube controller
        k-cube controller which can be used to control a thorlabs motor.

    '''
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()

    motor = KCubeDCServo.CreateKCubeDCServo(serialNumber)
    for attempts in range(3):
        try:
            motor.Connect(serialNumber)
        except:
            print("Connection attempt", attempts, "failed")
            if attempts < 2:
                print("Will wait 2 seconds and try again")
                sleep(2)
            else:
                print("Cannot connect to device.\n Please ensure that the" +\
                      " device is connected to your computer and not in"+\
                          " use by any other program!")
                return None
    motor.WaitForSettingsInitialized(5000)
    # configure the stage
    motorSettings = motor.LoadMotorConfiguration(serialNumber)
    motorSettings.DeviceSettingsName = DeviceSettingsName
    # update the RealToDeviceUnit converter
    motorSettings.UpdateCurrentConfiguration()
    # push the settings down to the device
    MotorDeviceSettings = motor.MotorDeviceSettings
    motor.SetSettings(MotorDeviceSettings, True, False)
    # Start polling the device
    motor.StartPolling(pollingRate)

    motor.EnableDevice()
    # Jogging parameters set to minimum
    motor.SetJogVelocityParams(Decimal(0.01), Decimal(0.01))
    return motor


def DisconnectMotor(motor):
    '''
    Function for safely disconnecting a motor so that other programs may use
    it.
    Parameters
    ----------
    motor : Thorlabs motor.
        Motor to be disconnected.

    Returns
    -------
    None.

    '''
    motor.StopPolling()
    motor.Disconnect()


def MoveMotor(motor, distance):
    '''
    Helper function for moving a motor.

    Parameters
    ----------
    motor : thorlabs motor
        Motor to be moved.
    distance : float
        Distance to move the motor.

    Returns
    -------
    bool
        True if the move was a success, otherwise false.

    '''
    if distance > 0.1 or distance < -0.1:
        print("Trying to move too far")
        return False
    # For unknown reason python thinks one first must convert to float but
    # only when running from console...
    motor.SetJogStepSize(Decimal(float(distance)))
    try:
        motor.MoveJog(1, timeoutVal)# Jog in forward direction
    except:
        print( "Trying to move motor to NOK position")
        return False
    return True


def MoveMotorPixels(motor, distance, mmToPixel=16140):
    # TODO - check if the mmToPixel value is valid for the basler camera.
    '''
    Moves motor a specified number of pixels.

    Parameters
    ----------
    motor : TYPE - thorlabs motor
         Motor to be moved
    distance : TYPE number
         Distance to move the motor
    mmToPixel : TYPE number for converting from mm(motor units) to pixels, optional
         The default is 16140, valid for our 100x objective and setup.

    Returns
    -------
    bool
        True if move was successfull, false otherwise.
    '''
    motor.SetJogStepSize(Decimal(float(distance/mmToPixel)))
    try:
        motor.MoveJog(1, timeoutVal)  # Jog in forward direction
    except:
        print( "Trying to move motor to NOK position")
        return False
    return True


def MoveMotorToPixel(motor, targetPixel,
                     currentPixel, maxPixel=1280, mmToPixel=16140):
    '''


    Parameters
    ----------
    motor : TYPE
        DESCRIPTION.
    targetPixel : TYPE
        DESCRIPTION.
    currentPixel : TYPE
        DESCRIPTION.
    maxPixel : TYPE, optional
        DESCRIPTION. The default is 1280.
    mmToPixel : TYPE, optional
        DESCRIPTION. The default is 16140.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    if(targetPixel<0 or targetPixel>maxPixel): # Fix correct boundries
        print("Target pixel outside of bounds")
        return False
    # There should be a minus here, this is due to the setup
    dx = -(targetPixel-currentPixel)/mmToPixel
    motor.SetJogStepSize(Decimal(float(dx)))
    try:
        motor.MoveJog(1,timeoutVal)# Jog in forward direction
    except:
        print( "Trying to move motor to NOK position")
        return False
    return True

def MoveTrapToPosition(motorX, motorY, targetX, targetY, trapX, trapY):
    '''

    Parameters
    ----------
    motorX : TYPE
        DESCRIPTION.
    motorY : TYPE
        DESCRIPTION.
    targetX : TYPE
        DESCRIPTION.
    targetY : TYPE
        DESCRIPTION.
    trapX : TYPE
        DESCRIPTION.
    trapY : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x = MoveMotorToPixel(motorX, targetX, trapX) # move X
    y = MoveMotorToPixel(motorY, targetY, trapY) # move Y
    return x and y


def setJogSpeed(motor, jog_speed, jog_acc=0.01):
    """
    Sets the jog-speed in mm/s of the motor as well as the jog acceleration
    """
    return motor.SetJogVelocityParams(Decimal(jog_speed), Decimal(jog_acc))


class MotorThread(Thread):
    '''
    Thread in which a motor is controlled. The motor object is available globally.
    '''
    # TODO: Try removing the treadlocks on the motors.
    # Try replacing some of the c_p with events.
    def __init__(self, threadID, name, axis, c_p):

      Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.axis = axis # 0 = x-axis, 1 = y axis
      self.c_p = c_p
      # Initiate contact with motor
      if self.axis == 0 or self.axis == 1:
          self.motor = InitiateMotor(c_p['serial_nums_motors'][self.axis],
            pollingRate=c_p['polling_rate'])
      else:
          raise Exception("Invalid choice of axis, no motor available.")

      # Read motor starting position
      if self.motor is not None:
          c_p['motor_starting_pos'][self.axis] = float(str(self.motor.Position))
          print('Motor is at ', c_p['motor_starting_pos'][self.axis])
          c_p['motors_connected'][self.axis] = True
      else:
          c_p['motors_connected'][self.axis] = False
      self.setDaemon(True)

    def run(self):
        print('Running motor thread')
        # global c_p
        c_p = self.c_p
        while c_p['motor_running']:
            # If motor connected and it should be connected, check for next move
            if c_p['motors_connected'][self.axis] and \
                c_p['connect_motor'][self.axis] and c_p['motors_connected'][self.axis]:
                # Acquire lock to ensure that it is safe to move the motor
                # with c_p['motor_locks'][self.axis]:
                if np.abs(c_p['motor_movements'][self.axis])>0:

                    # The movement limit must be positive
                    c_p['xy_movement_limit'] = np.abs(c_p['xy_movement_limit'])

                    # Check how much the motor is allowed to move
                    if np.abs(c_p['motor_movements'][self.axis])<=c_p['xy_movement_limit']:
                        MoveMotorPixels(self.motor,
                            c_p['motor_movements'][self.axis],
                            mmToPixel=c_p['mmToPixel'])
                    else:
                        if c_p['motor_movements'][self.axis]>0:
                            MoveMotorPixels(self.motor,


                                c_p['xy_movement_limit'],
                                mmToPixel=c_p['mmToPixel'])
                        else:
                            MoveMotorPixels(self.motor,
                                -c_p['xy_movement_limit'],
                                mmToPixel=c_p['mmToPixel'])

                    c_p['motor_movements'][self.axis] = 0
                c_p['motor_current_pos'][self.axis] = float(str(self.motor.Position))
            # Motor is connected but should be disconnected
            elif c_p['motors_connected'][self.axis] and not c_p['connect_motor'][self.axis]:
                DisconnectMotor(self.motor)
                c_p['motors_connected'][self.axis] = False
                self.motor = None
            # Motor is not connected but should be
            elif not c_p['motors_connected'][self.axis] and c_p['connect_motor'][self.axis]:
                self.motor = InitiateMotor(c_p['serial_nums_motors'][self.axis],
                  pollingRate=c_p['polling_rate'])
                # Check if motor was successfully connected.
                if self.motor is not None:
                    c_p['motors_connected'][self.axis] = True
                    c_p['motor_current_pos'][self.axis] = float(str(self.motor.Position))
                    c_p['motor_starting_pos'][self.axis] = c_p['motor_current_pos'][self.axis]
                else:
                    motor_ = 'x' if self.axis == 0 else 'y'
                    print('Failed to connect motor '+motor_)
            sleep(0.1) # To give other threads some time to work
        if c_p['motors_connected'][self.axis]:
            DisconnectMotor(self.motor)

class z_movement_thread(Thread):
    '''
    Thread for controling movement of the objective in z-direction.
    Will also help with automagically adjusting the focus to the sample.
    '''
    def __init__(self, threadID, name, serial_no, channel, c_p, polling_rate=250):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.piezo = PiezoMotor(serial_no, channel=channel, pollingRate=polling_rate)
        if self.piezo.is_connected:
            c_p['z_starting_position'] = self.piezo.get_position()
            c_p['motor_current_pos'][2] = self.piezo.get_position()
            c_p['motors_connected'][2] = self.piezo.is_connected
        self.setDaemon(True)
        self.c_p = c_p

    def run(self):
        # global c_p
        c_p = self.c_p
        lifting_distance = 0
        while c_p['program_running']:
            c_p['motors_connected'][2] = self.piezo.is_connected

            # Check if piezo connected and should be connected
            if self.piezo.is_connected and c_p['connect_motor'][2]:

                # Check if the objective should be moved
                self.piezo.move_to_position(compensate_focus()+lifting_distance)

                if c_p['z_movement'] is not 0:
                        c_p['z_movement'] = int(c_p['z_movement'])
                        # Move up if we are not already up
                        if self.piezo.move_relative(c_p['z_movement']):
                            lifting_distance += c_p['z_movement']
                        c_p['z_movement'] = 0

                elif c_p['return_z_home'] and c_p['motor_current_pos'][2]>compensate_focus():
                    lifting_distance -= min(20,c_p['motor_current_pos'][2]-compensate_focus())
                    # Compensating for hysteresis effect in movement
                    print('homing z')
                if c_p['motor_current_pos'][2]<=compensate_focus() or c_p['z_movement'] != 0:
                    c_p['return_z_home'] = False
                if self.piezo.is_connected:
                    c_p['motor_current_pos'][2] = self.piezo.get_position()

            # Piezomotor not connected but should be
            elif not self.piezo.is_connected and c_p['connect_motor'][2]:
                self.piezo.connect_piezo_motor()
                sleep(0.4)
                if self.piezo.is_connected:
                    # If the motor was just connected then reset positions
                    c_p['motor_current_pos'][2] = self.piezo.get_position()

                    c_p['z_starting_position'] = c_p['motor_current_pos'][2]
                    c_p['motor_starting_pos'][0] = c_p['motor_current_pos'][0]
                    c_p['motor_starting_pos'][1] = c_p['motor_current_pos'][1]
            # Piezo motor connected but should not be
            elif self.piezo.is_connected and not c_p['connect_motor'][2]:
                self.piezo.disconnect_piezo()

            sleep(0.3)
        del(self.piezo)

def ConnectBenchtopPiezoController(serialNo):
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    device = BenchtopPiezo.CreateBenchtopPiezo(serialNo)
    device.Connect(serialNo)
    return device

def ConnectPiezoStageChannel(device, channel):
    # DeviceManagerCLI.BuildDeviceList()
    # DeviceManagerCLI.GetDeviceListSize()
    # device = BenchtopPiezo.CreateBenchtopPiezo(serialNo)
    # device.Connect(serialNo)
    channel = device.GetChannel(channel)

    piezoConfiguration = channel.GetPiezoConfiguration(channel.DeviceID)
    currentDeviceSettings = channel.PiezoDeviceSettings
    #channel.SetSettings(currentDeviceSettings, True, False)

    channel.WaitForSettingsInitialized(5000)

    channel.StartPolling(100)#(250)
    # Needs a delay so that the current enabled state can be obtained

    deviceInfo = channel.GetDeviceInfo()
    # Enable the channel otherwise any move is ignored
    channel.EnableDevice()
    channel.SetPositionControlMode(2) # Set to closed loop mode.

    return channel

def get_default_piezo_c_p():
    piezo_c_p = {
    'piezo_serial_no':'71165844',
    'starting_position_piezo_xyz':[0,0,0],
    'piezo_target_pos':[0,0,0],
    'piezo_current_position':[0,0,0],
    'stage_piezo_connected':[False,False,False],
    'running':True,
    }
    return piezo_c_p

class XYZ_piezo_stage_motor(Thread):
    '''
    Class to help a main program of Automagic Trapping interface with a
    thorlabs max381 stage piezo motors.
    '''

    # TODO make it possible to connect/disconnect these motors on the fly.
    # TODO fix problem with single device
    def __init__(self, threadID, name, channel, axis, c_p, controller_device=None,
        serialNo='71165844', sleep_time=0.3):
        """

        """
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.channel = channel
        self.axis = axis
        print('I am here')
        self.sleep_time = sleep_time
        if controller_device is None:
            controller_device = ConnectBenchtopPiezoController(serialNo)
        self.controller_device = controller_device
        self.piezo_channel = ConnectPiezoStageChannel(controller_device, channel)
        self.c_p['starting_position_piezo_xyz'][self.axis] = self.piezo_channel.GetPosition()
        self.c_p['stage_piezo_connected'][self.axis] = True

    def update_position(self):
        # Update c_p position
        self.piezo_channel.SetPosition(Decimal(self.c_p['piezo_target_pos'][self.axis]))
        self.c_p['piezo_current_position'][self.axis] = self.piezo_channel.GetPosition()

    def run(self):
        '''
        '''
        while self.c_p['program_running']:
            self.update_position()
            sleep(self.sleep_time)
        self.__del__()

    def __del__(self):
        self.piezo_channel.StopPolling()
        self.piezo_channel.Disconnect()

def ConnectBenchtopStepperController(serialNo):
    '''
    Connects a benchtop stepper controller.
    '''

    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    device = BenchtopStepperMotor.CreateBenchtopStepperMotor(serialNo)
    device.Connect(serialNo)
    return device


def ConnectBenchtopStepperChannel(device, channel, polling_rate=100):
    '''
    Connects to the stepper motor of BenchtopStepperController on channel "channel".
    '''

    channel = device.GetChannel(channel)

    channel.WaitForSettingsInitialized(5000)

    channel.StartPolling(polling_rate)
    # Needs a delay so that the current enabled state can be obtained
    motorConfiguration = channel.LoadMotorConfiguration(channel.DeviceID);
    currentDeviceSettings = channel.MotorDeviceSettings# as ThorlabsBenchtopStepperMotorSettings;
    channel.SetSettings(currentDeviceSettings, True, False)

    deviceInfo = channel.GetDeviceInfo()
    # Enable the channel otherwise any move is ignored
    channel.EnableDevice()
    return channel


def get_default_stepper_c_p():
    """
    Returns a dictionary containg default values for all control parameters needed
    for the benchtop stepper controller and it's motors.
    """
    stepper_c_p = {
        'stepper_serial_no':'70167314',
        'starting_position_stepper_xyz':[0, 0, 0],
        'stage_stepper_connected':[False, False, False],
        'stepper_current_pos':[0, 0, 0],
        'stepper_next_move':[0, 0, 0],
    }
    return stepper_c_p


class XYZ_stepper_stage_motor(Thread):

    def __init__(self, threadID, name, channel, axis, c_p, controller_device=None,
        serialNo='70167314', sleep_time=0.3):
        """

        """
        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.channel = channel
        self.axis = axis
        self.sleep_time = sleep_time
        if controller_device is None:
            controller_device = ConnectBenchtopStepperController(serialNo)
        self.controller_device = controller_device
        self.stepper_channel = ConnectBenchtopStepperChannel(controller_device, channel)
        self.c_p['starting_position_stepper_xyz'][self.axis] = self.update_current_position()
        self.c_p['stage_stepper_connected'][self.axis] = True

    def update_current_position(self):
        decimal_pos = self.stepper_channel.Position
        self.c_p['stepper_current_pos'][self.axis] = float(str(decimal_pos))
        return float(str(decimal_pos))

    def move_distance(self, distance):
        self.stepper_channel.MoveRelative(1, Decimal(distance), Int32(10000))
        self.update_current_position()

    def move_to_position(self, position):
        distance = position - self.c_p['stepper_current_pos'][self.axis]
        self.move_distance(distance)

    def run(self):

        while self.c_p['program_running']:
            self.move_distance(self.c_p['stepper_next_move'][self.axis])
            sleep(self.sleep_time)
        self.__del__()

    def __del__(self):
        self.stepper_channel.StopPolling()
        self.stepper_channel.Disconnect()
