"""
Lab 6 Part 3: Validation of Camera-Robot Calibration
Tracks an AprilTag and transforms its position from camera to robot frame
"""

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), './classes'))

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
        TAG_SIZE_METERS = TAG_SIZE / 1000.0
        PRINT_INTERVAL = 10  # frames
        counter = 0
        
        while True:
            # 1. Get camera frame
            # YOUR CODE HERE
            
            # 2. Detect AprilTags
            # YOUR CODE HERE
            
            # 3. For each detected tag:
            #    - Get pose using get_tag_pose()
            #    - Create tag-to-camera transform
            #    - Convert to robot frame using camera_robot_transform
            #    - Print every PRINT_INTERVAL frames:
            #      * Camera frame coordinates
            #      * Robot frame coordinates
            # YOUR CODE HERE
            
            # 4. Display frame with detections
            # YOUR CODE HERE
            
            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()