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

def plot_end_effector_pose(time, poses):
    x = poses[:, 0]  # X position (in mm)
    y = poses[:, 1]  # Y position (in mm)
    z = poses[:, 2]  # Z position (in mm)
    orientation = poses[:, 3]  # Orientation (in degrees)

    plt.figure()
    plt.plot(time, x, label='x (mm)', linestyle='-', color='r')
    plt.plot(time, y, label='y (mm)', linestyle='--', color='g')
    plt.plot(time, z, label='z (mm)', linestyle='-.', color='b')
    plt.plot(time, orientation, label='orientation (deg)', linestyle=':', color='k')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pose')
    plt.legend()
    plt.title('Time Series of End-Effector Pose')
    plt.grid(True)
    plt.show()

def plot_3d_trajectory(poses):
    x = poses[:, 0]  # X position (in mm)
    y = poses[:, 1]  # Y position (in mm)
    z = poses[:, 2]  # Z position (in mm)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='End-Effector Path', color='r')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    plt.title('3D Trajectory of End-Effector')
    plt.show()

def plot_joint_angles_sensor_readings(time, joint_angles):
    q1 = joint_angles[:, 0]
    q2 = joint_angles[:, 1]
    q3 = joint_angles[:, 2]
    q4 = joint_angles[:, 3]

    plt.figure()
    plt.plot(time, q1, label='q1 (deg)', linestyle='-', color='r')
    plt.plot(time, q2, label='q2 (deg)', linestyle='--', color='g')
    plt.plot(time, q3, label='q3 (deg)', linestyle='-.', color='b')
    plt.plot(time, q4, label='q4 (deg)', linestyle=':', color='k')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    plt.title('Joint Angles from Sensor Readings')
    plt.grid(True)
    plt.show()

def plot_joint_angles_inverse_kinematics(robot, time, poses):
    joint_angles_ik = []

    # Loop through each pose and calculate the joint angles using inverse kinematics
    for pose in poses:
        joint_angles = robot.get_ik(pose)  # Assuming get_ik returns [q1, q2, q3, q4]
        joint_angles_ik.append(joint_angles)

    joint_angles_ik = np.array(joint_angles_ik)

    # Plot the joint angles
    q1 = joint_angles_ik[:, 0]
    q2 = joint_angles_ik[:, 1]
    q3 = joint_angles_ik[:, 2]
    q4 = joint_angles_ik[:, 3]

    plt.figure()
    plt.plot(time, q1, label='q1 (deg)', linestyle='-', color='r')
    plt.plot(time, q2, label='q2 (deg)', linestyle='--', color='g')
    plt.plot(time, q3, label='q3 (deg)', linestyle='-.', color='b')
    plt.plot(time, q4, label='q4 (deg)', linestyle=':', color='k')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    plt.title('Joint Angles from Inverse Kinematics')
    plt.grid(True)
    plt.show()

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

    # joint_angles = [[round(i) for i in robot.get_ik(pose)] for pose in test_poses]

    # traj_init = 1
    
    # init_robot(robot, traj_init)

    # # Run robot trajectory using IK
    # traj_time = 5
    
    # global_start_time = time.time()
    # for j in joint_angles:
    #     run_robot_trajectory(robot, traj_time, j)

    # #.pkl stuff
    # save_to_pickle(data, "lab3.pkl")

    # Load data
    file_path = "ee471-codebase/lab3/lab3.pkl"
    data = load_from_pickle(file_path)

    # Assume the data contains time, poses (x, y, z, orientation), and joint_angles
    time = data['timestamps']
    poses = np.array(data['ee_positions'])  # Shape: (n, 4) -> [x, y, z, orientation]
    joint_angles_sensor = np.array(data['joint_angles'])  # Shape: (n, 4) -> [q1, q2, q3, q4]

    # Create the plots
    plot_end_effector_pose(time, poses)
    plot_3d_trajectory(poses)
    plot_joint_angles_sensor_readings(time, joint_angles_sensor)

    # Assuming you have a robot object that can calculate IK
    plot_joint_angles_inverse_kinematics(robot, time, poses)
    




if __name__ == "__main__":
    main()