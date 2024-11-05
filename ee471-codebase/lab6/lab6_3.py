"""
Lab 6 Part 3: Validation of Camera-Robot Calibration
Tracks an AprilTag and transforms its position from camera to robot frame
"""

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

import numpy as np
import cv2

from Realsense import Realsense
from AprilTags import AprilTags

def main():
    try:
        # Initialize camera and detector
        camera = Realsense()
        detector = AprilTags()
        intrinsics = camera.get_intrinsics()
        
        # Load the saved calibration transform
        camera_robot_transform = np.load('camera_robot_transform.npy')
        print("Loaded camera-robot transformation matrix:")
        print(camera_robot_transform)
        
        # Constants
        TAG_SIZE = 40.0  # mm
        PRINT_INTERVAL = 10  # frames
        counter = 0
        desired_tag = 2
        
        while True:
            # 1. Get camera frame
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                continue
            
            # 2. Detect AprilTags in the frame
            tags = detector.detect_tags(color_frame)
            if len(tags) == 0:
                print("No tags detected.")
                continue  # No tags detected; skip to next frame
            
            # 3. Process each detected tag
            for tag in tags:
                # Skip if the detected tag is not the one on the stick
                if tag.tag_id != desired_tag:  # Replace YOUR_TAG_ID with the actual tag ID on the stick
                    continue
                
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
                    camera_frame_position = tvec.flatten()
                    robot_frame_position = tag_robot_frame[:3, 3]
                    
                    # Print the position information every PRINT_INTERVAL frames
                    if counter % PRINT_INTERVAL == 0:
                        print(f"Tag ID: {tag.tag_id}")
                        print(f"Position in Camera Frame (mm): {camera_frame_position}")
                        print(f"Position in Robot Frame (mm): {robot_frame_position}")
                        print("Tracking status: Tag detected\n")
                
                # Visualize tag detection
                color_frame = detector.draw_tags(color_frame, tag)

            # 4. Display frame with detections
            cv2.imshow("AprilTag Tracking", color_frame)
            
            counter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()