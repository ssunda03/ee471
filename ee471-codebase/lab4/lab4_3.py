# Script for Lab 4.3
# Diego Curiel / Srinivas Sundararaman

import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from TrajPlanner import TrajPlanner
from Robot import Robot

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data,file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

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

def plot_time_series_of_joint_angles(data):
    plt.plot(data["time"], data["joints"][:,0], label='Joint Angle q1', linestyle='-', color='b')
    plt.plot(data["time"], data["joints"][:,1], label='Joint Angle q2', linestyle='--', color='g')
    plt.plot(data["time"], data["joints"][:,2], label='Joint Angle q3', linestyle='-.', color='r')
    plt.plot(data["time"], data["joints"][:,3], label='Joint Angle q4', linestyle=':', color='m')

    # Adding labels and title
    plt.title('Joint Angles Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angles (degrees)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def init_robot(robot, traj_init):
    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    joints = robot.get_ik([25, -100, 150, -60])
    robot.write_joints(joints)  # Write joints to first setpoint
    time.sleep(traj_init)  # Wait for trajectory completion

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 3 # Traj time
    init_robot(robot, traj_init)

    # Setpoints for trajectory as np array
    setpoints_taskspace = np.array([
        [25, -100, 150, -60],
        [150, 80, 300, 0],
        [250, -115, 75, -45],
        [25, -100, 150, -60],
    ])

    # Create instance of TrajPlanner
    planner = TrajPlanner(setpoints_taskspace)

    # Parameters for cubic trajectory
    traj_time = 5  # 5 seconds between setpoints
    num_waypoints = 998  # 998 intermediate waypoints

    # Generate cubic trajectory for each segment: Pose 1 -> Pose 2 -> Pose 3 -> Pose 1
    trajectory = planner.get_quintic_traj(traj_time, num_waypoints)
    time_step = trajectory[2,0] - trajectory[1,0]

    # Pre-allocate data
    data_time = np.zeros(num_waypoints+2)
    data_ee_poses = np.zeros((num_waypoints+2, 4))
    data_q = np.zeros((num_waypoints+2, 4))
    count = 0

    robot.write_time(time_step)
    start_time = time.time()

    # Move the robot along all trajectories
    for i in range(1, len(trajectory)):
        curjoints = robot.get_ik(trajectory[i, 1:]) # Use IK to get joints
        robot.write_joints(curjoints)  # Write joint values
        # Collect a reading periodically until the waypoint is reached
        while time.time() - start_time < (i * time_step):
            data_q[count, :] = robot.get_joints_readings()[0, :]
            data_time[count] = time.time() - start_time
            data_ee_poses[count, :] = robot.get_ee_pos(data_q[count, :])[0:4]
            count += 1

    # Trim unused space in data
    data_time = data_time[:count]
    data_ee_poses = data_ee_poses[:count, :]
    data_q = data_q[:count, :]
    plot_3d_trajectory(data_ee_poses)
    
    # data = {
    #     "joints" : data_q,
    #     "ee" : data_ee_poses,
    #     "time" : data_time
    # }
    
    # data = load_from_pickle("lab4_cubic_traj.pkl")

    # plot_3d_trajectory(data["ee"])
    # plot_time_series_of_joint_angles(data)




if __name__ == "__main__":
    main()