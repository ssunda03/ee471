"""
Lab 6 Part 2: Camera-Robot Calibration
Implements camera-robot calibration using AprilTags and the Kabsch algorithm
"""

import sys
import os
import numpy as np
import cv2

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Realsense import Realsense
from AprilTags import AprilTags
from point_registration import point_registration, rmse  # Import your existing point_registration function

def calibrate_camera():
    try:
        # Step i. Initialize RealSense camera and AprilTag detector
        camera = Realsense()
        detector = AprilTags()
        intrinsics = camera.get_intrinsics()
        TAG_SIZE = 40.0  # in mm

        # Step ii. Define known 3D points in robot base frame
        robot_points = np.array([
            [66, -90, 0],
            [66, -30, 0],
            [66, 30, 0],
            [66, 90, 0],
            [126, -90, 0],
            [126, -30, 0],
            [126, 30, 0],
            [126, 90, 0],
            [186, -90, 0],
            [186, -30, 0],
            [186, 30, 0],
            [186, 90, 0],
        ]).T

        robot_points = np.vstack((robot_points, np.ones((1, robot_points.shape[1]))))  # Convert to homogeneous coordinates
        
        points_camera = np.zeros_like(robot_points)  # Empty array for camera measurements
        measurements_list = [[] for _ in range(robot_points.shape[1])]  # List to store multiple measurements

        # Step iii. Collect multiple measurements
        num_measurements = 5
        counter = 0

        while counter < num_measurements:
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                continue
            
            # Detect AprilTags in the frame
            tags = detector.detect_tags(color_frame)
            
            tags.sort(key=lambda x: x.tag_id)  # Sort tags by ID for correct correspondence

            for index, tag in enumerate(tags):
                corners = tag.corners
                rot_matrix, tvec = detector.get_tag_pose(corners, intrinsics, TAG_SIZE)

                if tvec is not None:
                    measurements_list[index].append(tvec.reshape(3))  # Store translation in mm

            # Visualize tag detection
            for tag in tags:
                color_frame = detector.draw_tags(color_frame, tag)
            cv2.imshow("AprilTag Detection", color_frame)
            
            counter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Average measurements for each tag position
        for i in range(robot_points.shape[1]):
            points_camera[:3, i] = np.mean(measurements_list[i], axis=0)
        points_camera[3, :] = 1  # Homogeneous coordinates

        # Step iv. Calculate transformation using point_registration
        T_camera_to_robot = point_registration(points_camera[:3,:], robot_points[:3,:])
        
        # Step v. Calculate and display calibration error
        transformed_points = T_camera_to_robot @ points_camera
        rmse_error = rmse(transformed_points, robot_points)
        print(f"RMSE Calibration Error: {rmse_error:.2f} mm")

        # Step vi. Save transformation matrix, print transformation matrix
        print("Transformation Matrix (Camera to Robot):\n", T_camera_to_robot)
        np.save("camera_robot_transform.npy", T_camera_to_robot)
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()

def main():
    # Calibrate the camera
    calibrate_camera()

if __name__ == "__main__":
    main()
