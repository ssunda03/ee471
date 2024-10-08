# file for 2_3 in lab 2
# les go

import sys
import os
import time
import pickle
import numpy as np

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Robot import Robot

data = {
        'joint_angles' : [],
        'ee_positions' : [],
        'timestamps' : []
    }

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
    
    start_time = time.time()  # Start timer
    elapsed_time = 0

    while elapsed_time < traj_time:
        joints = robot.get_joints_readings()[0]
        
        data['joint_angles'].append(joints)
        data['ee_positions'].append(robot.get_ee_pos(joints))
        data['timestamps'].append(elapsed_time)
        
        print(f"elapsed time: {elapsed_time}")
        print(f"current joint angles: {joints}")
        print(f"current end effector position: {robot.get_ee_pos(joints)}")
        
        elapsed_time = time.time() - start_time

    time.sleep(1)  # Pause for a second before ending

def save_to_pickle(data, filename):
    with open(filename, 'ab') as file:
        pickle.dump(data,file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 1
    
    init_robot(robot, traj_init)

    # Run robot trajectory, printing and displaying info
    # about end effector position as well as the 
    # transformation matrix T4,0

    traj_time = 5
    joint_angles = [[0, -45, 60, 50],[0, 10, 50, -45], [0, 10, 0, -80], [0, -45, 60, 50]] # Define waypoints

    for point in joint_angles:
        run_robot_trajectory(robot, traj_time, point)
        
    save_to_pickle(data, "datafile.pkl")



if __name__ == "__main__":
    main()