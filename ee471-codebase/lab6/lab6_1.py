"""
Lab 6 Part 1: AprilTag Detection and Pose Estimation Test Script
Tests the integration of RealSense camera with AprilTag detection and pose estimation.
"""

import sys
import os
import numpy as np
import cv2

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Realsense import Realsense
from AprilTags import AprilTags

def main():
    try:
        # Initialize RealSense camera
        rs = Realsense()
        intrinsics = rs.get_intrinsics()
        
        # Initialize AprilTag detector
        at = AprilTags()
        
        # Tag size in millimeters (adjust to match your actual tag size)
        TAG_SIZE = 40.0  
        
        # Counter for controlling print frequency
        counter = 0
        
        while True:
            # 1. Get color frame from RealSense camera
            color_frame, _ = rs.get_frames()
            
            if color_frame is None:
                continue  # skip if frame is not available
            
            # 2. Detect AprilTags in the frame
            tags = at.detect_tags(color_frame)
            
            # 3. Process each detected tag
            for tag in tags:
                # Draw detected tag
                color_frame = at.draw_tags(color_frame, tag)
                
                # Get tag corners and tag pose
                corners = tag.corners
                rot_matrix, tvec = at.get_tag_pose(corners, intrinsics, TAG_SIZE)
                
                if rot_matrix is None or tvec is None:
                    continue  # skip if pose estimation fails
                
                # Calculate the distance to the tag
                distance = np.linalg.norm(tvec)  # Distance in mm
                
                # Calculate orientation angles (roll, pitch, yaw) from rotation matrix
                rpy_angles = cv2.RQDecomp3x3(rot_matrix)[0] # in degrees btw
                
                # Print the tag information every 10 frames
                if counter % 10 == 0:
                    print(f"Tag ID: {tag.tag_id}")
                    print(f"Distance: {distance:.2f} mm")
                    print(f"Orientation: Roll={rpy_angles[0]:.2f}, Pitch={rpy_angles[1]:.2f}, Yaw={rpy_angles[2]:.2f} degrees")
                    print(f"Position: X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f} mm\n")
                    
            # Increment counter
            counter += 1
            
            # 4. Display the image
            cv2.imshow('AprilTag Detection', color_frame)
            
            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Cleanup
        rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
