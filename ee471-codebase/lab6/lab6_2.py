"""
Lab 6 Part 2: Camera-Robot Calibration
Implements camera-robot calibration using AprilTags and the Kabsch algorithm
"""

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

import numpy as np
import cv2

from Realsense import Realsense
from AprilTags import AprilTags

def point_registration(points_camera, points_robot):
    """
    Implement Kabsch algorithm to find transformation between camera and robot frames.
    
    Args:
        points_camera: 4xN array of homogeneous points in camera frame
        points_robot: 4xN array of homogeneous points in robot frame
        
    Returns:
        4x4 homogeneous transformation matrix from camera to robot frame
    """
    # Algorithm implemented for students
    pass

def main():
    try:
        # Initialize cameras and detectors
        camera = Realsense()
        detector = AprilTags()
        intrinsics = camera.get_intrinsics()
        
        # AprilTag size in mm (adjust according to your tag)
        TAG_SIZE = 40.0
        TAG_SIZE_METERS = TAG_SIZE / 1000.0
        
        # Define known tag positions in robot frame (mm)
        robot_points = np.array([
            [80, -90, 0],   # Add all your measured points here
            # ... more points
        ])
        
        # 1. Convert robot points to homogeneous coordinates (4xN)
        # YOUR CODE HERE
        
        # 2. Initialize camera points array with same size
        # YOUR CODE HERE
        
        # 3. Collect 3 to 5 measurements:
        #    - Get frames and detect tags
        #    - When all tags visible:
        #      * Get pose for each tag
        #      * Store positions in order
        #    - Average the measurements
        # YOUR CODE HERE
        
        # 4. Calculate transformation using point_registration()
        # YOUR CODE HERE
        
        # 5. Calculate and print calibration error
        # YOUR CODE HERE
        
        # 6. Save transformation matrix
        # YOUR CODE HERE
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()