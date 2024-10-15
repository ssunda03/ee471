# file for 2_3 in lab 2
# les go

import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Robot import Robot

data = {
        'joint_angles' : [],
        'ee_positions' : [],
        'timestamps' : []
    }

dh_table = np.array([
            [0,                                 0.077,  0,      -np.pi/2],
            [-(np.pi/2 - np.arcsin(24/130)),    0,      0.13,   0],
            [(np.pi/2 - np.arcsin(24/130)),     0,      0.124,  0],
            [0,                                 0,      0.126,  0]
        ])

total_time = 0
global_start_time = 0

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
        data['timestamps'].append(total_time)
        
        print(f"elapsed time: {total_time}")
        print(f"current joint angles: {joints}")
        print(f"current end effector position: {robot.get_ee_pos(joints)}")
        
        elapsed_time = time.time() - start_time
        total_time = time.time() - global_start_time

    time.sleep(1)  # Pause for a second before ending

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data,file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    global global_start_time
    # Initialize Robot instance
    robot = Robot()
    traj_init = 1
    
    init_robot(robot, traj_init)

    # Run robot trajectory, printing and displaying info
    # about end effector position as well as the 
    # transformation matrix T4,0

    traj_time = 5
    joint_angles = [[0, -45, 60, 50],[0, 10, 50, -45], [0, 10, 0, -80], [0, -45, 60, 50]] # Define waypoints
    global_start_time = time.time()
    for point in enumerate(joint_angles):
        run_robot_trajectory(robot, traj_time, point)
        
    save_to_pickle(data, "datafile.pkl")
    
    loaded = load_from_pickle("datafile.pkl")
    
    fig, axs = plt.subplots(4, 1)  # 4 subplots vertically aligned

    # Titles for each subplot
    joint_titles = ['Base Joint', 'Joint 2', 'Joint 3', 'Joint 4']
    
    for i in range(4):
        axs[i].plot(loaded["timestamps"], [joints[i] for joints in loaded["joint_angles"]], 'r')
        axs[i].set_title(joint_titles[i])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Position (degrees)')
        axs[i].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    fig.savefig(f"BaseJointTrajTime.png")
    
    
    
    plt.plot(loaded["timestamps"], [ee[0] for ee in loaded["ee_positions"]], 'r')
    plt.plot(loaded["timestamps"], [ee[2] for ee in loaded["ee_positions"]], 'b')
    
    plt.title("x-z")
    plt.xlabel('Time (s)')
    plt.ylabel('x -z')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"x-z.png")
    
    
    joint_angles = [[0, -45, 60, 50],[0, 10, 50, -45], [0, 10, 0, -80], [0, -45, 60, 50]]
    waypoints = [get_ee_pos(j) for j in joint_angles]
    wayx = [w[0] for w in waypoints]
    wayz = [w[2] for w in waypoints]
    
    plt.plot([ee[0] for ee in loaded["ee_positions"]], [ee[2] for ee in loaded["ee_positions"]], 'r')
    plt.scatter(wayx, wayz, label='waypoints', color='blue')
    
    plt.title("x-z ee")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"x-z ee.png")

    joint_angles = [[0, -45, 60, 50], [0, 10, 50, -45], [0, 10, 0, -80], [0, -45, 60, 50]]
    waypoints = [get_ee_pos(j) for j in joint_angles]
    wayx = [w[0] for w in waypoints]
    wayy = [w[1] for w in waypoints]
    
    plt.plot([ee[0] for ee in loaded["ee_positions"]], [ee[1] for ee in loaded["ee_positions"]], 'r')
    plt.scatter(wayx, wayy, label='waypoints', color='blue')
    
    plt.title("x-y ee")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"x-y ee.png")