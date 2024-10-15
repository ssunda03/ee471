# file for 3_3 in lab 3
# les go

import sys
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))
from Robot import Robot

data = {
        'joint_angles' : [],
        'ee_positions' : [],
        'timestamps' : []
}

total_time = 0
global_start_time = 0

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data,file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def init_robot(robot, traj_init):
    # Setup robot
    traj_init = 1  # Defines the initial trajectory time

    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    robot.write_joints([0, 0, 0, 0])  # Write joints to zero position
    time.sleep(traj_init)  # Wait for trajectory completion

def run_robot_trajectory(robot, traj_time, joint_angles):
    global global_start_time
    global total_time
    
    robot.write_time(traj_time)  # Write trajectory time
    robot.write_joints(joint_angles)  # Write joint values
    
    start_time = time.time()  # Start timer
    elapsed_time = 0

    while elapsed_time < traj_time:
        joints = robot.get_joints_readings()[0]

        data['joint_angles'].append(joints)
        data['ee_positions'].append(robot.get_ee_pos(joints))
        data['timestamps'].append(total_time)
        
        print(f"elapsed time: {total_time}")
        print(f"current joint angles: {joints}")
        print(f"current end effector position: {robot.get_ee_pos(joints)}")

        elapsed_time = time.time() - start_time
        total_time = time.time() - global_start_time



    time.sleep(1)  # Pause for a second before ending

def main():
    # Declare as global
    global global_start_time

    # Initialize Robot instance
    robot = Robot()

    # Test poses for IK triangle movements
    test_poses = [
        np.array([25, -100, 150, -60]),
        np.array([150, 80, 300, 0]),
        np.array([250, -115, 75, -45]),
        np.array([25, -100, 150, -60])
    ]

    joint_angles = [[round(i) for i in robot.get_ik(pose)] for pose in test_poses]

    traj_init = 1
    
    init_robot(robot, traj_init)

    # Run robot trajectory using IK
    traj_time = 5
    
    global_start_time = time.time()
    for j in joint_angles:
        run_robot_trajectory(robot, traj_time, j)

    #.pkl stuff
    save_to_pickle(data, "lab3.pkl")
    




if __name__ == "__main__":
    main()