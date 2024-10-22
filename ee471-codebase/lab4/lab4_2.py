# Script for Lab 4.2
# Diego Curiel / Srinivas Sundararaman

import sys
import os
import time
import numpy as np

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from TrajPlanner import TrajPlanner
from Robot import Robot

def init_robot(robot, traj_init):
    # Setup robot
    traj_init = 1  # Defines the initial trajectory time

    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    robot.write_joints([0, 0, 0, 0])  # Write joints to zero position
    time.sleep(traj_init)  # Wait for trajectory completion

def run_robot_trajectory(robot, traj_time, joint_angles):

    robot.write_time(traj_time)  # Write trajectory time
    robot.write_joints(joint_angles)  # Write joint values
    
    # start_time = time.time()  # Start timer
    # elapsed_time = 0

    # while elapsed_time < traj_time:
    #     print(f"Transformation matrix for End Effector to Base @ {elapsed_time}\n")
    #     print(robot.get_current_fk())
    #     print(f"Current end effector position and orientation @ {elapsed_time}\n")
    #     print(robot.get_ee_pos(robot.get_joints_readings()[0])) # Get end effector x,y,z pos and orientation
    #     elapsed_time = time.time() - start_time

    # time.sleep(1)  # Pause for a second before ending

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 1
    robot.init_robot(traj_init)

    # Setpoints for trajectory
    setpoints_taskspace = [
        np.array([25, -100, 150, -60]),
        np.array([150, 80, 300, 0]),
        np.array([250, -115, 75, -45]),
        np.array([25, -100, 150, -60]),
    ]

    # Use IK method to get Joint Angles
    setpoints_jointspace = []
    for i in range(len(setpoints_taskspace)):
        cur_joints = robot.get_ik(setpoints_taskspace[i])
        setpoints_jointspace.append(cur_joints)

    # Make np array
    setpoints_jointspace = np.array(setpoints_jointspace)

    # Create instance of TrajPlanner
    planner = TrajPlanner(setpoints_jointspace)

    # Parameters for cubic trajectory
    traj_time = 5  # 5 seconds between setpoints
    num_waypoints = 998  # 998 intermediate waypoints

    # Generate cubic trajectory for each segment: Pose 1 -> Pose 2 -> Pose 3 -> Pose 1
    trajectory = []
    for i in range(len(setpoints_jointspace) - 1):
        traj_segment = planner.get_cubic_traj(traj_time=traj_time, num_waypoints=num_waypoints)
        trajectory.append(traj_segment)
    


if __name__ == "__main__":
    main()