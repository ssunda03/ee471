# file for 2_3 in lab 2
# les go

import sys
import os
import time

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

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
    
    start_time = time.time()  # Start timer
    elapsed_time = 0

    while elapsed_time < traj_time:
        print(f"Transformation matrix for End Effector to Base @ {elapsed_time}\n")
        print(robot.get_current_fk())
        print(f"Current end effector position and orientation @ {elapsed_time}\n")
        print(robot.get_ee_pos(robot.get_joints_readings()[0])) # Get end effector x,y,z pos and orientation
        elapsed_time = time.time() - start_time

    time.sleep(1)  # Pause for a second before ending

def main():
    # Initialize Robot instance
    robot = Robot()
    traj_init = 1
    
    init_robot(robot, traj_init)

    # Run robot trajectory, printing and displaying info
    # about end effector position as well as the 
    # transformation matrix T4,0

    traj_time = 3
    joint_angles = [-90, 15, 30, -45] # Define waypoints
    run_robot_trajectory(robot, traj_time, joint_angles)
    
    init_robot(robot, traj_init)
    
    joint_angles = [15, -45, -60, 90] # Define waypoints
    run_robot_trajectory(robot, traj_time, joint_angles)
    
    init_robot(robot, traj_init)


if __name__ == "__main__":
    main()