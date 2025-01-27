# file for 2_3 in lab 2
# les go

import sys
import os
import time
import numpy as np

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
    
    # start_time = time.time()  # Start timer
    # elapsed_time = 0

    # while elapsed_time < traj_time:
    #     print(f"Transformation matrix for End Effector to Base @ {elapsed_time}\n")
    #     print(robot.get_current_fk())
    #     print(f"Current end effector position and orientation @ {elapsed_time}\n")
    #     print(robot.get_ee_pos(robot.get_joints_readings()[0])) # Get end effector x,y,z pos and orientation
    #     elapsed_time = time.time() - start_time

    time.sleep(1)  # Pause for a second before ending

def main():
    # Initialize Robot instance
    robot = Robot()

    # Test poses for IK/FK testing
    test_poses = [
        np.array([274, 0, 204, 0]),
        np.array([16, 4, 336, 15]),
        np.array([0, -270, 106, 0])
    ]

<<<<<<< HEAD
    joint_angles = [[round(i) for i in robot.get_ik(pose)] for pose in test_poses]
=======
    joint_angles = [[round(j) for j in robot.get_ik(pose)] for pose in test_poses]
>>>>>>> 44df1e43e53dac577119489002b8d63557a56884

    print(f"Test poses: \n{test_poses}\n")
    print(f"Joint angles:\n{joint_angles}\n")
    
<<<<<<< HEAD
    fk_poses = [[round(j) for j in robot.get_ee_pos(joint_angle)] for joint_angle in joint_angles]
=======
    fk_poses = [[round(d) for d in robot.get_ee_pos(joint_angle)] for joint_angle in joint_angles]
>>>>>>> 44df1e43e53dac577119489002b8d63557a56884
    print(f"FK poses:\n{fk_poses}")


    # traj_init = 1
    
    # init_robot(robot, traj_init)

    # # Run robot trajectory using IK

    # traj_time = 3

    # print()
    # run_robot_trajectory(robot, traj_time, joint_angles[0])
    # init_robot(robot, traj_init)
    
    # run_robot_trajectory(robot, traj_time, joint_angles[1])
    # init_robot(robot, traj_init)

    # run_robot_trajectory(robot, traj_time, joint_angles[2])
    # init_robot(robot, traj_init)

    # Run robot trajectory using FK




if __name__ == "__main__":
    main()