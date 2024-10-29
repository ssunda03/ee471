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

        self.dh_table = np.array([
            [0,                                 77,  0,  -np.pi/2],
            [-(np.pi/2 - np.arcsin(24/130)),    0,   130, 0],
            [(np.pi/2 - np.arcsin(24/130)),     0,   124, 0],
            [0,                                 0,   126, 0]
        ])

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
    
    
    """
    Calculates the homogeneous transformation matrix Ai for a given row of DH parameters.
    
    Parameters:
    dh_row: A 1x4 array containing the DH parameters [theta, d, a, alpha].
    Returns:
    A 4x4 numpy array representing the homogeneous transformation matrix Ai.
    """
    def get_dh_row_mat(self, dh_row):
        theta, d, a, alpha = dh_row

        A_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha),  a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha),  -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,             np.sin(alpha),                  np.cos(alpha),                  d],
            [0,             0,                              0,                              1]
        ])

        return A_i
        
    """
    Calculates a 4x4x4 numpy array of transformation matrices for specified joint angles.

    Parameters:
    joint_angles: A 1x4 array containing the joint angles.
    Returns:
    A 4x4x4 numpy array of transformation matrices A1, A2, A3, A4.
    """
    def get_int_mat(self, joint_angles):
        # Ensure joint_angles is a numpy array
        joint_angles = np.asarray(joint_angles)

        # Create an array to hold the transformation matrices
        transformation_matrices = np.zeros((4, 4, 4))

        for i in range(4):
            # Update the theta value in the DH table with the corresponding joint angle
            dh_row = self.dh_table[i].copy()
            dh_row[0] += np.radians(joint_angles[i])

            # Calculate the transformation matrix for the current joint
            transformation_matrices[i] = self.get_dh_row_mat(dh_row)

        return transformation_matrices
    
    def get_acc_mat(self, joint_angles):
        # Ensure joint_angles is a numpy array
        joint_angles = np.asarray(joint_angles)

        # Create array to hold the A matrices
        A_matrices = np.zeros((4, 4, 4))

        # Create array to hold the T matrices
        T_matrices = np.zeros((4, 4, 4))

        for i in range(4):
            # Update the theta value in the DH table with the corresponding joint angle
            dh_row = self.dh_table[i].copy()
            dh_row[0] += np.radians(joint_angles[i])

            # Calculate the transformation matrix for the current joint
            A_matrices[i] = self.get_dh_row_mat(dh_row)

            # Update T_matrix array
            if i == 0:
                T_matrices[i] = A_matrices[i] # Handle base case
            else:
                T_matrices[i] = np.dot(T_matrices[i - 1], A_matrices[i]) # The next T is just the current A multiplied by last T

        return T_matrices
    
    def get_fk(self, joint_angles):
        return self.get_acc_mat(joint_angles)[3]

    def get_current_fk(self):
        return self.get_fk(self.get_joints_readings()[0])
    
    def get_ee_pos(self, joint_angles):
        t_mat = self.get_fk(joint_angles)
        ee_pos = [float(t_mat[0][3]), 
                  float(t_mat[1][3]), 
                  float(t_mat[2][3]), 
                  float(joint_angles[0]), 
                  float(-(joint_angles[1] + joint_angles[2] + joint_angles[3]))]
        return ee_pos
    
    def get_ik(self, ee_pos):

        joint_angles = np.zeros(4)  # Initialize joint angles
        radius = self.mOtherDim[0] + self.mDim[2] + self.mDim[3]
        center = [0, 0, self.mDim[0]]
        test_radius = [i - j for i, j in zip(ee_pos, center)]

        # Check if the end-effector is within reach
        if (radius**2 < sum([i**2 for i in test_radius]) and test_radius[2] > 0):
            raise ValueError("End effector position out of reach")

        r = np.sqrt(np.power(ee_pos[0], 2) + np.power(ee_pos[1], 2))
        rw = r - self.mDim[3] * np.cos(np.radians(ee_pos[3]))
        zw = ee_pos[2] - self.mDim[0] - self.mDim[3] * np.sin(np.radians(ee_pos[3]))
        dw = np.sqrt(np.power(rw,2) + np.power(zw,2))

        mu = np.arctan2(zw, rw)
        
        cosbeta = (np.power(self.mDim[1],2) + np.power(self.mDim[2],2) - np.power(dw,2)) / (2 * self.mDim[1] * self.mDim[2])
        cosbeta = np.clip(cosbeta, -1, 1)
        
        sinbeta = np.sqrt(1 - np.power(cosbeta,2))
        sinbeta = np.clip(sinbeta, -1, 1)
        
        sinbetas = [sinbeta, -1*sinbeta]
        
        beta = np.arctan2(sinbetas[0], cosbeta)  # Use the positive solution for beta

        cosgamma = (np.power(dw,2) + np.power(self.mDim[1],2) - np.power(self.mDim[2],2)) / (2 * dw * self.mDim[1])
        cosgamma = np.clip(cosgamma, -1, 1)
        
        singamma = np.sqrt(1 - np.power(cosgamma,2))
        singamma = np.clip(singamma, -1, 1)
        
        singammas = [singamma, -1*singamma]
        
        gamma = np.arctan2(singammas[0], cosgamma)  # Use the positive solution for gamma

        delta = np.arctan2(self.mOtherDim[1], self.mOtherDim[0])
        
        # Compute joint angles
        joint_angles[0] = np.degrees(np.arctan2(ee_pos[1], ee_pos[0]))
        joint_angles[1] = np.degrees((np.pi / 2) - delta - gamma - mu)
        joint_angles[2] = np.degrees((np.pi / 2) + delta - beta)
        joint_angles[3] = -ee_pos[3] - joint_angles[1] - joint_angles[2]

        return joint_angles
    
    def get_jacobian(self, joint_angles):
        jacobian = np.zeros((6, 4))
        acc_mat = self.get_acc_mat(joint_angles)
        
        z0 = np.array([0, 0, 1])
        
        o4 = acc_mat[-1][:-1,-1]
        jacobian[:,0] = np.hstack((np.radians(np.cross(z0, o4)),z0))
        
        for i in range(1, len(joint_angles)):
            z = acc_mat[i-1][:-1,2]
            o = acc_mat[i-1][:-1,3]
            jacobian[:,i] = np.hstack((np.radians(np.cross(z, o4-o)), z))
            
        return jacobian
    
    def get_forward_diff_kinematics(self, joint_angles, joint_velocities):
        return np.dot(self.get_jacobian(joint_angles), joint_velocities.T)