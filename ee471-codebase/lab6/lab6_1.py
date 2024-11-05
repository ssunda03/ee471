"""
Lab 6 Part 1: AprilTag Detection and Pose Estimation Test Script
Tests the integration of RealSense camera with AprilTag detection and pose estimation.
"""

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), './classes'))

# Required imports
import numpy as np
import cv2

from Realsense import Realsense
from AprilTags import AprilTags

def main():
    try:
        # Initialize RealSense camera
        rs = Realsense()
        intrinsics = rs.get_intrinsics()
        
        # Initialize AprilTag detector
        at = AprilTags()
        
        # Tag size in millimeters (measure your actual tag size)
        TAG_SIZE = 40.0  # Adjust this to match your tag size
        
        # Counter for controlling print frequency
        counter = 0
        
        while True:
            # 1. Get color frame from RealSense camera
            color_frame, _ = rs.get_frames()
            
            # 2. Detect AprilTags in the frame
            tags = at.detect_tags(color_frame)
            
            # 3. For each detected tag:
            #    - Draw tag detection on image
            #    - Get pose estimation using get_tag_pose()
            #    - Print every 10 frames:
            #      * Tag ID
            #      * Distance
            #      * Orientation (roll, pitch, yaw)
            #      * Position (x, y, z)
            # YOUR CODE HERE
            
            # 4. Display the image
            # YOUR CODE HERE
            
            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Cleanup
        rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()