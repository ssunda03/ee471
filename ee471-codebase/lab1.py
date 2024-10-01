# (c) 2024 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), './classes'))

from Robot import Robot
import time
import numpy as np
import matplotlib.pyplot as mpl

"""
This script demonstrates the basic operation of the OpenManipulator-X using the Robot class.
It initializes the robot, sets a movement time profile, and iteratively moves the joints through a series of predefined waypoints.
Each waypoint adjustment is followed by a pause, during which joint readings are printed to the console.
Finally, the gripper is closed, and the script concludes with a brief pause.
This example serves as a basic demonstration of controlling the robot's joints and reading their positions, velocities, and currents.
"""

# Setup robot
traj_init = 1  # Defines the inittial trajectory time
traj_time = 10  # Defines the trajectory time
robot = Robot()  # Creates robot object

robot.write_time(traj_init)  # Write trajectory time
robot.write_motor_state(True)  # Write position mode

# Program
robot.write_joints([0, 0, 0, 0])  # Write joints to zero position
time.sleep(traj_init)  # Wait for trajectory completion

base_waypoint = 45  # Define base waypoints
joint_positions : list[list[float]] = []
time_stamps = []

robot.write_time(traj_time)  # Write trajectory time

robot.write_joints([base_waypoint, 0, 0, 0])  # Write joint values
start_time = time.time()  # Start timer
elapsed_time = 0

while elapsed_time < traj_time:
    joint_positions.append(robot.get_joints_readings()[0]) # Read joint values
    time_stamps.append(elapsed_time)
    
    print(joint_positions[-1])
    print(time_stamps[-1])
    
    elapsed_time = time.time() - start_time

time.sleep(1)  # Pause for a second before ending

joint_positions_np = np.array(joint_positions)
time_stamps_np = np.array(time_stamps)

