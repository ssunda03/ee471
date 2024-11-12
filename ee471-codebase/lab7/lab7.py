"""
Lab 7: Position-Based Visual Servoing Implementation

This script will implement tracking using a PID controller,
moving the robot arm to the target position (the selected tag)
"""

import sys
import os
import time

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

import pickle
import numpy as np
import cv2

from Robot import Robot
from Controller import PIDController
from Realsense import Realsense
from AprilTags import AprilTags

def init_robot(robot, traj_init):
    robot.write_time(traj_init)  # Write trajectory time
    robot.write_motor_state(True)  # Write position mode

    # Program
    joints = [0, 0, 0, 0] # Home position
    robot.write_joints(joints)  # Write joints to first setpoint
    time.sleep(traj_init)  # Wait for trajectory completion

def main():

    try:
        # Load the saved calibration transform
        camera_robot_transform = np.load('camera_robot_transform.npy')
        ## C:/Users/thede/CALPOLY/CPE471_visionGuidedRoboticManipulation/ee471/ee471-codebase/
        print("Loaded camera-robot transformation matrix:")
        print(camera_robot_transform)

        # Initialize Robot instance
        robot = Robot()
        traj_init = 3 # Traj time
        init_robot(robot, traj_init)

        # Set robot to velocity control mode
        robot.write_mode("velocity")

        # Initialize camera and detector
        camera = Realsense()
        detector = AprilTags()
        intrinsics = camera.get_intrinsics()

        # Initialize the PID Controller
        timestep = 0.025 # ms
        controller = PIDController(dt = timestep)
        Kp = 0.7
        controller.Kp = Kp * np.eye(3)  # Proportional gain
        controller.Kd = (0.05 * Kp) * np.eye(3)  # Derivative gain
        controller.Ki = (0.025 * Kp) * np.eye(3) # Integral gain

        # Constants
        desired_offset = np.array([0, 0, 15])
        TAG_SIZE = 40.0  # mm
        desired_tag = 2
        start_time = 0
        tag_found = False
        
        while True:
            # 1. Get camera frame
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                robot.write_velocities([0, 0, 0, 0])
                continue
            
            # 2. Detect AprilTags in the frame
            tags = detector.detect_tags(color_frame)
            if len(tags) == 0:
                # print("No tags detected.")
                robot.write_velocities([0, 0, 0, 0])
                continue  # No tags detected; skip to next frame

            # 3. Process each detected tag
            tag_found = False

            for tag in tags:
                # Skip if the detected tag is not the one on the stick
                if tag.tag_id != desired_tag:  # Replace YOUR_TAG_ID with the actual tag ID on the stick
                    # robot.write_velocities([0, 0, 0, 0])
                    continue
                
                tag_found = True

                # 3.5 Start time
                start_time = time.time()

                # Get pose of the tag in the camera frame
                corners = tag.corners
                rot_matrix, tvec = detector.get_tag_pose(corners, intrinsics, TAG_SIZE)
                
                if tvec is not None and rot_matrix is not None:              
                    # Construct the 4x4 transformation matrix from tag to camera frame
                    tag_to_camera_transform = np.eye(4)
                    tag_to_camera_transform[:3, :3] = rot_matrix
                    tag_to_camera_transform[:3, 3] = tvec.flatten()
                    
                    # Calculate tag position in the robot frame
                    tag_robot_frame = camera_robot_transform @ tag_to_camera_transform
                    robot_frame_position = tag_robot_frame[:3, 3]
                    # print(robot_frame_position)

                    # Add offset to tag position in robot frame
                    robot_frame_position += desired_offset
                    # print(robot_frame_position)

                    # Get current End Effector position from FK
                    current_joint_readings = robot.get_joints_readings()[0]
                    current_robot_ee_pos = np.array(robot.get_ee_pos(current_joint_readings)[:3])

                    # Get error
                    current_error = robot_frame_position - current_robot_ee_pos
                    # print("error: ")
                    # print(current_error)

                    # Compute control signal (cartesian velocities!)
                    control_signal = controller.compute_pid(current_error)

                    # Turn Cartesian velocities into joint velocities using inverse Jacobian
                    # via inverse velocity kinematics
                    jacobian = robot.get_jacobian(current_joint_readings)
                    translational_jacobian = jacobian[:3, :]  # Use the top 3x4 portion for translational motion
                    joint_velocities = np.dot(np.linalg.pinv(translational_jacobian), control_signal.T)

                    # Limit velocities
                    for i in range(len(joint_velocities)):
                        if np.abs(joint_velocities[i]) > 180:
                            joint_velocities[i] = 180

                    # Write joint velocities to the Robot!
                    robot.write_velocities(joint_velocities)
                
                # Visualize tag detection
                color_frame = detector.draw_tags(color_frame, tag)
                break

            if not (tag_found):
                robot.write_velocities([0, 0, 0, 0])
            
            # 4. Display frame with detections
            cv2.imshow("AprilTag Tracking", color_frame)
            
            # Calculate elapsed time for the loop and add a delay if needed
            elapsed_time = time.time() - start_time
            if elapsed_time < timestep:
                time.sleep(timestep - elapsed_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()