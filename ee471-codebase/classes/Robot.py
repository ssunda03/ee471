# (c) 2024 S. Farzan, Electrical Engineering Department, Cal Poly
# Skeleton Robot class for OpenManipulator-X Robot for EE 471

import numpy as np
from OM_X_arm import OM_X_arm
from DX_XM430_W350 import DX_XM430_W350

"""
Robot class for controlling the OpenManipulator-X Robot.
Inherits from OM_X_arm and provides methods specific to the robot's operation.
"""
class Robot(OM_X_arm):
    """
    Initialize the Robot class.
    Creates constants and connects via serial. Sets default mode and state.
    """
    def __init__(self):
        super().__init__()

        # Robot Dimensions (in mm)
        self.mDim = [77, 130, 124, 126]
        self.mOtherDim = [128, 24]
        
        # Set default mode and state
        # Change robot to position mode with torque enabled by default
        # Feel free to change this as desired
        self.write_mode('position')
        self.write_motor_state(True)

        # Set the robot to move between positions with a 5 second trajectory profile
        # change here or call writeTime in scripts to change
        self.write_time(5)

    """
    Sends the joints to the desired angles.
    Parameters:
    goals (list of 1x4 float): Angles (degrees) for each of the joints to go to.
    """
    def write_joints(self, goals):
        goals = [round(goal * DX_XM430_W350.TICKS_PER_DEG + DX_XM430_W350.TICK_POS_OFFSET) % DX_XM430_W350.TICKS_PER_ROT for goal in goals]
        self.bulk_read_write(DX_XM430_W350.POS_LEN, DX_XM430_W350.GOAL_POSITION, goals)

    """
    Creates a time-based profile (trapezoidal) based on the desired times.
    This will cause write_position to take the desired number of seconds to reach the setpoint.
    Parameters:
    time (float): Total profile time in seconds. If 0, the profile will be disabled (be extra careful).
    acc_time (float, optional): Total acceleration time for ramp up and ramp down (individually, not combined). Defaults to time/3.
    """
    def write_time(self, time, acc_time=None):
        if acc_time is None:
            acc_time = time / 3

        time_ms = int(time * DX_XM430_W350.MS_PER_S)
        acc_time_ms = int(acc_time * DX_XM430_W350.MS_PER_S)

        self.bulk_read_write(DX_XM430_W350.PROF_ACC_LEN, DX_XM430_W350.PROF_ACC, [acc_time_ms]*self.motorsNum)
        self.bulk_read_write(DX_XM430_W350.PROF_VEL_LEN, DX_XM430_W350.PROF_VEL, [time_ms]*self.motorsNum)

    """
    Sets the gripper to be open or closed.
    Parameters:
    open (bool): True to set the gripper to open, False to close.
    """
    def write_gripper(self, open):
        if open:
            self.gripper.write_position(-45)
        else:
            self.gripper.write_position(45)

    def read_gripper(self):
        if open:
            pos = self.gripper.read_position()
        else:
            pos = self.gripper.read_position()
        return pos

    """
    Sets position holding for the joints on or off.
    Parameters:
    enable (bool): True to enable torque to hold the last set position for all joints, False to disable.
    """
    def write_motor_state(self, enable):
        state = 1 if enable else 0
        states = [state] * self.motorsNum  # Repeat the state for each motor
        self.bulk_read_write(DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, states)

    """
    Supplies the joints with the desired currents.
    Parameters:
    currents (list of 1x4 float): Currents (mA) for each of the joints to be supplied.
    """
    def write_currents(self, currents):
        current_in_ticks = [round(current * DX_XM430_W350.TICKS_PER_mA) for current in currents]
        self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.GOAL_CURRENT, current_in_ticks)

    """
    Change the operating mode for all joints.
    Parameters:
    mode (str): New operating mode for all joints. Options include:
        "current": Current Control Mode (writeCurrent)
        "velocity": Velocity Control Mode (writeVelocity)
        "position": Position Control Mode (writePosition)
        "ext position": Extended Position Control Mode
        "curr position": Current-based Position Control Mode
        "pwm voltage": PWM Control Mode
    """
    def write_mode(self, mode):
        if mode in ['current', 'c']:
            write_mode = DX_XM430_W350.CURR_CNTR_MD
        elif mode in ['velocity', 'v']:
            write_mode = DX_XM430_W350.VEL_CNTR_MD
        elif mode in ['position', 'p']:
            write_mode = DX_XM430_W350.POS_CNTR_MD
        elif mode in ['ext position', 'ep']:
            write_mode = DX_XM430_W350.EXT_POS_CNTR_MD
        elif mode in ['curr position', 'cp']:
            write_mode = DX_XM430_W350.CURR_POS_CNTR_MD
        elif mode in ['pwm voltage', 'pwm']:
            write_mode = DX_XM430_W350.PWM_CNTR_MD
        else:
            raise ValueError(f"writeMode input cannot be '{mode}'. See implementation in DX_XM430_W350 class.")

        self.write_motor_state(False)
        write_modes = [write_mode] * self.motorsNum  # Create a list with the mode value for each motor
        self.bulk_read_write(DX_XM430_W350.OPR_MODE_LEN, DX_XM430_W350.OPR_MODE, write_modes)
        self.write_motor_state(True)

    """
    Gets the current joint positions, velocities, and currents.
    Returns:
    numpy.ndarray: A 3x4 array containing the joints' positions (deg), velocities (deg/s), and currents (mA).
    """
    def get_joints_readings(self):
        readings = np.zeros((3, 4))
        
        positions = np.array(self.bulk_read_write(DX_XM430_W350.POS_LEN, DX_XM430_W350.CURR_POSITION))
        velocities = np.array(self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.CURR_VELOCITY))
        currents = np.array(self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.CURR_CURRENT))

        # Take two's complement of velocity and current data
        for i in range(4):
            if velocities[i] > 0x7fffffff:
                velocities[i] = velocities[i] - 4294967296
            if currents[i] > 0x7fff:
                currents[i] = currents[i] - 65536

        readings[0, :] = (positions - DX_XM430_W350.TICK_POS_OFFSET) / DX_XM430_W350.TICKS_PER_DEG
        readings[1, :] = velocities / DX_XM430_W350.TICKS_PER_ANGVEL
        readings[2, :] = currents / DX_XM430_W350.TICKS_PER_mA

        return readings

    """
    Sends the joints to the desired velocities.
    Parameters:
    vels (list of 1x4 float): Angular velocities (deg/s) for each of the joints to go at.
    """
    def write_velocities(self, vels):
        vels = [round(vel * DX_XM430_W350.TICKS_PER_ANGVEL) for vel in vels]
        self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.GOAL_VELOCITY, vels)
