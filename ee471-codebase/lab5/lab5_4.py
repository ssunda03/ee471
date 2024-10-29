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

def plot_time_series_of_ee_pos(data):
    plt.plot(data["time"], data["ee"][:,0], label='x', linestyle='-', color='b')
    plt.plot(data["time"], data["ee"][:,1], label='y', linestyle='-', color='g')
    plt.plot(data["time"], data["ee"][:,2], label='z', linestyle='-', color='r')
    plt.plot(data["time"], data["ee"][:,3], label='a', linestyle='-', color='m')

    # Adding labels and title
    plt.title('End-Effector Pose Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pose Parameters')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_time_series_of_ee_velocity(data):

    # Compute the time differences and joint angle differences
    time_diffs = np.diff(data["time"])
    x_velocities = np.diff(data["ee"][:,0]) / time_diffs
    y_velocities = np.diff(data["ee"][:,1]) / time_diffs
    z_velocities = np.diff(data["ee"][:,2]) / time_diffs
    a_velocities = np.diff(data["ee"][:,3]) / time_diffs
    
    plt.plot(data["time"][1:], x_velocities, label='x', linestyle='-', color='b')
    plt.plot(data["time"][1:], y_velocities, label='y', linestyle='-', color='g')
    plt.plot(data["time"][1:], z_velocities, label='z', linestyle='-', color='r')
    plt.plot(data["time"][1:], a_velocities, label='a', linestyle='-', color='m')

    # Adding labels and title
    plt.title('End-Effector Velocity Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity Parameters')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_time_series_of_ee_accel(data):

    # Compute the time differences and joint angle differences
    time_diffs = np.diff(data["time"])
    x_accel = np.diff(np.diff(data["ee"][:,0])) / time_diffs[:-1]
    y_accel = np.diff(np.diff(data["ee"][:,1])) / time_diffs[:-1]
    z_accel = np.diff(np.diff(data["ee"][:,2])) / time_diffs[:-1]
    a_accel = np.diff(np.diff(data["ee"][:,3])) / time_diffs[:-1]
    
    plt.plot(data["time"][2:], x_accel, label='x', linestyle='-', color='b')
    plt.plot(data["time"][2:], y_accel, label='y', linestyle='-', color='g')
    plt.plot(data["time"][2:], z_accel, label='z', linestyle='-', color='r')
    plt.plot(data["time"][2:], a_accel, label='a', linestyle='-', color='m')

    # Adding labels and title
    plt.title('End-Effector Acceleration Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration Parameters')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def init_robot(robot, traj_init):
    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    joints = robot.get_ik([25, -100, 150, 0])
    robot.write_joints(joints)  # Write joints to first setpoint
    time.sleep(traj_init)  # Wait for trajectory completion

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 3 # Traj time
    init_robot(robot, traj_init)

    # Setpoints for trajectory as np array
    setpoints_taskspace = np.array([
        [25, -100, 150, 0],
        [150, 80, 300, 0],
        [250, -115, 75, 0],
        [25, -100, 150, 0],
    ])

    # Set robot to velocity control mode
    robot.write_mode("velocity")

    # Define constants
    speed = 50  # Desired speed in mm/s
    tolerance = 5  # Tolerance in mm for reaching each target

    for i in range(len(setpoints_taskspace) - 1):
        # Define current and target positions
        current_pose = setpoints_taskspace[i]
        target_pose = setpoints_taskspace[i + 1]

        # Loop to control robot velocity to reach each target
        while True:
            # Get current end-effector position
            ee_pos = robot.get_ee_pos(robot.get_joints_readings()[0])[:3]

            # Calculate the vector to the target and its distance
            direction_vector = np.array(target_pose[:3]) - np.array(ee_pos)
            distance_to_target = np.linalg.norm(direction_vector)

            # Check if within tolerance
            if distance_to_target <= tolerance:
                print(f"Reached target {i + 1}")
                break

            # Calculate the unit direction vector and scale by the constant speed
            unit_vector = direction_vector / distance_to_target
            task_space_velocity = unit_vector * speed

            # Compute joint velocities via inverse velocity kinematics
            jacobian = robot.get_jacobian(robot.get_joints_readings()[0])
            translational_jacobian = jacobian[:3, :]  # Use the top 3x4 portion for translational motion
            joint_velocities = np.dot(np.linalg.pinv(translational_jacobian), task_space_velocity)

            # Write the computed joint velocities to the robot
            robot.write_velocities(joint_velocities)

            # Optional: Print debug info or call get_forward_diff_kinematics() for verification
            actual_velocities = robot.get_joints_readings()[1]  # Joint velocities in deg/s
            print(f"Moving to target {i + 1}, distance: {distance_to_target:.2f} mm")

            # Safety: Check the maximum allowable velocity
            if np.any(np.abs(task_space_velocity) > 100):
                print("Error: Exceeding maximum task-space velocity limit.")
                robot.write_velocities([0, 0, 0, 0])  # Stop the robot
                break

            time.sleep(0.1)  # Control loop delay for smooth operation

    # Zero all joint velocities after reaching the final target
    robot.write_velocities([0, 0, 0, 0])
    print("Motion completed successfully.")

if __name__ == "__main__":
    main()