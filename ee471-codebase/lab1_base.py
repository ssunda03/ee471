# (c) 2024 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), './classes'))

from Robot import Robot
import time

"""
This script demonstrates the basic operation of the OpenManipulator-X using the Robot class.
It initializes the robot, sets a movement time profile, and iteratively moves the joints through a series of predefined waypoints.
Each waypoint adjustment is followed by a pause, during which joint readings are printed to the console.
Finally, the gripper is closed, and the script concludes with a brief pause.
This example serves as a basic demonstration of controlling the robot's joints and reading their positions, velocities, and currents.
"""

# Setup robot
traj_time = 3  # Defines the trajectory time
robot = Robot()  # Creates robot object

robot.write_time(traj_time)  # Write trajectory time
robot.write_motor_state(True)  # Write position mode

# Program
robot.write_joints([0, 0, 0, 0])  # Write joints to zero position
time.sleep(traj_time)  # Wait for trajectory completion

base_waypoints = [-45, 45, 0]  # Define base waypoints

for base_waypoint in base_waypoints:  # Iterate through waypoints
    robot.write_joints([base_waypoint, 0, 0, 0])  # Write joint values
    start_time = time.time()  # Start timer
    
    while time.time() - start_time < traj_time:
        print(robot.get_joints_readings())  # Read joint values
        time.sleep(0.5)  # Small delay to avoid flooding the output

pos = robot.read_gripper()
# print(f"pos: {pos}")
if pos < 180:
    robot.write_gripper(True)  # Write gripper state to OPEN
else:
    robot.write_gripper(False)  # Write gripper state to CLOSE

time.sleep(1)  # Pause for a second before ending
