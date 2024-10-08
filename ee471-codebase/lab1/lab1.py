#Srinivas Sundararaman and Diego Curiel
#EE 471
#Lab 1
#California Polytechnic State University, San Luis Obispo

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Robot import Robot
import time
import numpy as np
import matplotlib.pyplot as plt

def init_robot(robot, traj_init):
    # Setup robot
    traj_init = 1  # Defines the initial trajectory time

    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    robot.write_joints([0, 0, 0, 0])  # Write joints to zero position
    time.sleep(traj_init)  # Wait for trajectory completion

def create_joint_position_subplots(time_stamps_np, joint_positions_np, traj_time):
    """
    Creates 4 different subplots of the joint positions during the trajectory
    
    Parameters:
    time_stamps_np (ndarray): Array of time stamps.
    joint_positions_np (ndarray): Array of the joint positions over time.
    traj_time (float): The total time for the robot's trajectory motion.
    """
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))  # 4 subplots vertically aligned

    # Titles for each subplot
    joint_titles = ['Base Joint', 'Joint 2', 'Joint 3', 'Joint 4']

    # Plot each joint's motion profile
    for i in range(4):
        axs[i].plot(time_stamps_np, joint_positions_np[:, i], label=f'Joint {i+1}')
        axs[i].set_title(joint_titles[i])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Position (degrees)')
        axs[i].set_ylim(-5, 45)
        axs[i].legend()
        axs[i].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    fig.savefig(f"BaseJointTrajTime_{traj_time}.png")
    
def analyze_time_intervals(time_stamps_np, traj_time):
    """
    Analyzes and plots the time intervals between consecutive readings.
    
    Parameters:
    time_stamps_np (ndarray): Array of time stamps.
    traj_time (float): The total time for the robot's trajectory motion.
    """
    # Calculate time intervals between consecutive readings
    time_intervals = np.diff(time_stamps_np) * 1000  # Convert from seconds to milliseconds

    # Calculate statistics
    mean_interval = np.mean(time_intervals)
    median_interval = np.median(time_intervals)
    max_interval = np.max(time_intervals)
    min_interval = np.min(time_intervals)
    stddev_interval = np.std(time_intervals)

    # Print the statistics with clear labels
    print(f"Time Interval Statistics (in milliseconds):")
    print(f"Mean: {mean_interval:.2f} ms")
    print(f"Median: {median_interval:.2f} ms")
    print(f"Max: {max_interval:.2f} ms")
    print(f"Min: {min_interval:.2f} ms")
    print(f"Standard Deviation: {stddev_interval:.2f} ms")

    # Plot a histogram of the time intervals
    plt.figure(figsize=(8, 6))
    plt.hist(time_intervals, bins=20, edgecolor='black')
    plt.title("Histogram of Time Intervals Between Readings")
    plt.xlabel("Time Interval (ms)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    # Save the histogram
    plt.savefig(f"TimeIntervalsTrajTime_{traj_time}.png")

def run_robot_trajectory(robot, traj_time):
    base_waypoint = 45  # Define base waypoints
    joint_positions: list[list[float]] = []
    time_stamps = []

    robot.write_time(traj_time)  # Write trajectory time

    robot.write_joints([base_waypoint, 0, 0, 0])  # Write joint values
    start_time = time.time()  # Start timer
    elapsed_time = 0

    while elapsed_time < traj_time:
        joint_positions.append(robot.get_joints_readings()[0])  # Read joint values
        time_stamps.append(elapsed_time)
        elapsed_time = time.time() - start_time

    time.sleep(1)  # Pause for a second before ending

    joint_positions_np = np.array(joint_positions)
    time_stamps_np = np.array(time_stamps)

    return time_stamps_np, joint_positions_np

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 1
    
    init_robot(robot, traj_init)
    
    # Call the robot trajectory function with different traj_time values
    traj_time = 10
    tstamp, jpos = run_robot_trajectory(robot, traj_time)
    create_joint_position_subplots(tstamp, jpos, traj_time)
    analyze_time_intervals(tstamp, traj_time)
    
    init_robot(robot, traj_init)
    
    traj_time = 2
    tstamp, jpos = run_robot_trajectory(robot, traj_time)
    create_joint_position_subplots(tstamp, jpos, traj_time)
    analyze_time_intervals(tstamp, traj_time)

if __name__ == '__main__':
    main()
