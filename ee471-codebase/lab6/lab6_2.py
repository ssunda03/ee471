"""
Lab 6 Part 2: Camera-Robot Calibration
Implements camera-robot calibration using AprilTags and the Kabsch algorithm
"""

import sys
import os
import numpy as np
import cv2
from Realsense import Realsense
from AprilTags import AprilTags

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from prelab import point_registration  # Import your existing point_registration function

def main():
    try:
        # Step i. Initialize RealSense camera and AprilTag detector
        camera = Realsense()
        detector = AprilTags()
        intrinsics = camera.get_intrinsics()
        TAG_SIZE = 40.0  # in mm

        # Step ii. Define known 3D points in robot base frame
        robot_points = np.array([
            [80, -90, 0],  # Replace with all measured points for each tag

            # Add all remaining points here
            
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
            if len(tags) < robot_points.shape[1]:  # Ensure all tags are visible
                continue
            
            tags.sort(key=lambda x: x.tag_id)  # Sort tags by ID for correct correspondence

            for idx, tag in enumerate(tags):
                corners = tag.corners
                rot_matrix, tvec = detector.get_tag_pose(corners, intrinsics, TAG_SIZE)

                if tvec is not None:
                    tvec_mm = tvec * 1000  # Convert from meters to mm
                    measurements_list[idx].append(tvec_mm.reshape(3))  # Store translation in mm

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
        T_camera_to_robot = point_registration(points_camera, robot_points)
        
        # Step v. Calculate and display calibration error
        transformed_points = T_camera_to_robot @ points_camera
        error = np.linalg.norm(transformed_points[:3, :] - robot_points[:3, :], axis=0)
        avg_error = np.mean(error)
        print("Transformation Matrix (Camera to Robot):\n", T_camera_to_robot)
        print(f"Average Calibration Error: {avg_error:.2f} mm")

        # Step vi. Save transformation matrix
        np.save("camera_robot_transform.npy", T_camera_to_robot)
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
