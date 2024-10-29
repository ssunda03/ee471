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

    plt.savefig('./ee471-codebase/lab5/plots/3D_Trajectory.png')  # Saves in a "plots" subdirectory of the script’s location
    plt.show()

def plot_time_series_of_joint_velocities(data):
    plt.plot(data["time"], data["q_vel"][:,0], label='Joint Vel q1', linestyle='-', color='b')
    plt.plot(data["time"], data["q_vel"][:,1], label='Joint Vel q2', linestyle='--', color='g')
    plt.plot(data["time"], data["q_vel"][:,2], label='Joint Vel q3', linestyle='-.', color='r')
    plt.plot(data["time"], data["q_vel"][:,3], label='Joint Vel q4', linestyle=':', color='m')

    # Adding labels and title
    plt.title('Joint Velocities Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Velocities (degrees/sec)')
    plt.legend()
    plt.grid(True)

    plt.savefig('./ee471-codebase/lab5/plots/Time_vs_Joint_Velocities.png')  # Saves in a "plots" subdirectory of the script’s location

    # Show the plot
    plt.show()
    
def plot_time_series_of_ee_velocities(data):
    plt.plot(data["time"], data["ee_vel"][:,0], label='X dot', linestyle='-', color='b')
    plt.plot(data["time"], data["ee_vel"][:,1], label='Y dot', linestyle='--', color='g')
    plt.plot(data["time"], data["ee_vel"][:,2], label='Z dot', linestyle='-.', color='r')
    plt.plot(data["time"], data["ee_vel"][:,3], label='A dot', linestyle=':', color='m')

    # Adding labels and title
    plt.title('EE Velocities Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EE Velocities')
    plt.legend()
    plt.grid(True)

    plt.savefig('./ee471-codebase/lab5/plots/Time_vs_EE_Velocities.png')  # Saves in a "plots" subdirectory of the script’s location

    # Show the plot
    plt.show()

def plot_time_series_of_ee_pos(data):
    plt.plot(data["time"], data["ee_pos"][:,0], label='x', linestyle='-', color='b')
    plt.plot(data["time"], data["ee_pos"][:,1], label='y', linestyle='-', color='g')
    plt.plot(data["time"], data["ee_pos"][:,2], label='z', linestyle='-', color='r')
    plt.plot(data["time"], data["ee_pos"][:,3], label='a', linestyle='-', color='m')

    # Adding labels and title
    plt.title('End-Effector Pose Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pose Parameters')
    plt.legend()
    plt.grid(True)

    plt.savefig('./ee471-codebase/lab5/plots/Time_vs_EE_Position.png')  # Saves in a "plots" subdirectory of the script’s location

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
    tolerance = 10  # Tolerance in mm for reaching each target

    # Pre-allocate data
    data_time = np.zeros(400)
    data_ee_poses = np.zeros((400, 4))
    data_q_vel = np.zeros((400, 4))
    data_ee_vel = np.zeros((400, 4))
    count = 0

    # Start time
    start_time = time.time()

    for i in range(len(setpoints_taskspace) - 1):
        # Define target positions
        target_pose = setpoints_taskspace[i + 1]

        distance_to_target = tolerance + 100
        
        # Loop to control robot velocity to reach each target
        while distance_to_target > tolerance:
            data_time[count] = time.time() - start_time # Log Time

            posvel = robot.get_joints_readings()[:2, :]
            # Get current end-effector position
            full_ee_pos = robot.get_ee_pos(posvel[0])
            ee_pos = full_ee_pos[:3]
            data_ee_poses[count][:3] = ee_pos
            data_ee_poses[count][-1] = full_ee_pos[-1]

            # Calculate the vector to the target and its distance
            vector_to_target = np.array(target_pose[:3]) - np.array(ee_pos)
            distance_to_target = np.linalg.norm(vector_to_target)

            # Calculate the unit direction vector and scale by the constant speed
            unit_vector = vector_to_target / distance_to_target
            task_space_velocity = unit_vector * speed

            # Compute joint velocities via inverse velocity kinematics
            jacobian = robot.get_jacobian(posvel[0])
            translational_jacobian = jacobian[:3, :]  # Use the top 3x4 portion for translational motion
            joint_velocities = np.dot(np.linalg.pinv(translational_jacobian), task_space_velocity.T)
            data_q_vel[count] = joint_velocities # Store commanded velocities

            ee_velocities = robot.get_forward_diff_kinematics(
                posvel[0],
                posvel[1])  # Joint velocities in deg/s
            data_ee_vel[count][:3] = ee_velocities[:3] # Store EE velocities
            data_ee_vel[count][-1] = 0 # Pitch always 0

            # Optional: Print debug info or call get_forward_diff_kinematics() for verification
            print(f"Moving to target {i + 1}, distance: {distance_to_target:.2f} mm")
            print(f"Calculated velocity: {joint_velocities} mm/s, actual velocity: {posvel[1]} mm/s")
            

            # Safety: Check the maximum allowable velocity
            if np.any(np.abs(task_space_velocity) > 100):
                print("Error: Exceeding maximum task-space velocity limit.")
                robot.write_velocities([0, 0, 0, 0])  # Stop the robot
                break

            # Write the computed joint velocities to the robot
            robot.write_velocities(joint_velocities)
            count += 1
        
        print(f"waypoint {i+1} reached")
            
    # Zero all joint velocities after reaching the final target
    robot.write_velocities([0, 0, 0, 0])
    print("Motion completed successfully.")

        # Trim unused space in data
    data_time = data_time[:count]
    data_ee_poses = data_ee_poses[:count, :]
    data_q_vel = data_q_vel[:count, :]
    data_ee_vel = data_ee_vel[:count, :]
    
    data = {
        "time": data_time,
        "ee_pos": data_ee_poses,
        "q_vel": data_q_vel,
        "ee_vel": data_ee_vel
    }

    save_to_pickle(data, "lab5_velocity_kinematics.pkl")
    data = load_from_pickle("lab5_velocity_kinematics.pkl")
    plot_3d_trajectory(data["ee_pos"])
    plot_time_series_of_ee_pos(data)
    plot_time_series_of_joint_velocities(data)
    plot_time_series_of_ee_velocities(data)

if __name__ == "__main__":
    main()