"""
Lab 8:  Vision-Guided Robotic Pick-and-Place Sorting System

"""

import sys
import os
import numpy as np
import cv2
import pyrealsense2 as rs


# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Realsense import Realsense
from AprilTags import AprilTags
from point_registration import point_registration, rmse  # Import your existing point_registration function

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

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
        # Save the processed image
        script_dir = os.path.abspath(os.path.dirname(__file__))
        transformation_matrix_path = os.path.join(script_dir, "camera_robot_transform.npy") 
        print(f"Saving to: {transformation_matrix_path}")
        np.save(transformation_matrix_path, T_camera_to_robot)
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()

def detect_colored_spheres_live(camera, workspace_mask=None):
    """
    Detects and classifies colored spheres in the live feed from the RealSense camera.

    Args:
        camera (Realsense): RealSense camera object for live feed.
        workspace_mask (np.ndarray, optional): Binary mask to ignore areas outside the robot's workspace.

    Returns:
        list: List of detected spheres with color, centroid coordinates, and radius.
    """
    # Define HSV color ranges for sphere detection
    color_ranges = {
        "red_lower": [(0, 100, 100), (5, 255, 255)],        # Lower range for red
        "red_upper": [(170, 100, 100), (180, 255, 255)],    # Upper range for red
        "orange": [(6, 150, 150), (24, 255, 255)],         # Refined range for orange
        "yellow": [(25, 50, 50), (50, 255, 255)],         # Refined range for yellow
        "blue": [(100, 50, 50), (130, 255, 255)],           # Standard range for blue
    }

    detected_spheres = []

    # Get a frame from the RealSense camera
    color_frame, depth_frame = camera.get_frames()
    if color_frame is None or  depth_frame is None:
        return detected_spheres  # No frame available

    # Apply workspace mask if provided
    if workspace_mask is not None:
        color_frame = cv2.bitwise_and(color_frame, color_frame, mask=workspace_mask)

    # Convert the frame to grayscale for circle detection
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    # Apply CLAHE to the grayscale image
    clahe_image = clahe.apply(gray_frame)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(clahe_image, (7, 7), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred_frame,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,  # Minimum distance between detected centers
        param1=100,  # Canny edge detection threshold
        param2=30,   # Accumulator threshold for circle detection
        minRadius=5,  # Minimum circle radius
        maxRadius=22,  # Maximum circle radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Convert the frame to HSV color space for color detection
        hsv_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

        for circle in circles[0, :]:
            x, y, radius = circle

            # cv2.circle(color_frame, (x, y), radius, (0, 255, 0), 2)

            # # Create a mask for the circle
            # mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            # cv2.circle(mask, (x, y), radius, 255, -1)  # Filled circle mask

            # # Extract HSV values within the circle
            # hsv_values = hsv_image[mask == 255]

            # # Calculate the mean HSV values
            # mean_hsv = cv2.mean(hsv_image, mask=mask)[:3]  # Exclude alpha channel

            # # Print the HSV values
            # print(f"Circle at (x={x}, y={y}, radius={radius}):")
            # print(f"  Mean HSV: H={mean_hsv[0]:.2f}, S={mean_hsv[1]:.2f}, V={mean_hsv[2]:.2f}")

            # Check the color of the circle
            detected_color = None
            for color, (lower, upper) in color_ranges.items():
                lower_bound = np.array(lower, dtype=np.uint8)
                upper_bound = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

                # Focus the mask on the detected circle region
                circle_mask = np.zeros_like(mask)
                cv2.circle(circle_mask, (x, y), radius, 255, -1)
                masked_region = cv2.bitwise_and(mask, mask, mask=circle_mask)

                # Check if the circle region contains sufficient color pixels
                if cv2.countNonZero(masked_region) > 0.5 * np.pi * (radius ** 2):  # At least 60% match
                    detected_color = "red" if color in ["red_lower", "red_upper"] else color
                    break

            if detected_color:
                detected_spheres.append((detected_color, (int(x), int(y)), int(radius)))

                # Annotate the detected circles on the frame
                cv2.circle(color_frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(color_frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(
                    color_frame,
                    detected_color,
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    # Save the processed image
    script_dir = os.path.abspath(os.path.dirname(__file__))
    output_image_path = os.path.join(script_dir, "detected_spheres.jpg") 
    cv2.imwrite(output_image_path, color_frame)

    # Display the annotated frame
    cv2.imshow("Colored Sphere Detection", color_frame)

    return detected_spheres, depth_frame

def get_sphere_coordinates(spheres, camera, transformation_matrix, depth_frame):
    """
    Converts 2D sphere centroids to 3D robot coordinates using depth information.

    Args:
        spheres (list): List of detected spheres with color, centroid coordinates, and radius.
                        Format: [(color, (cx, cy), radius), ...]
        camera (Realsense): RealSense camera object for accessing intrinsics.
        transformation_matrix (np.ndarray): Transformation matrix from camera to robot frame.
        depth_frame: RealSense depth frame object for depth information.

    Returns:
        list: List of spheres with 3D robot coordinates.
              Format: [(color, (x, y, z)), ...]
    """
    sphere_coordinates = []

    # Get depth intrinsics for deprojection
    depth_intrinsics = camera.get_depth_intrinsics()

    for color, (cx, cy), radius in spheres:
        # Get the depth value at the sphere's centroid
        depth = depth_frame.get_distance(cx, cy)

        if depth == 0:  # Skip invalid depth readings
            print(f"Invalid depth for {color} sphere at ({cx}, {cy}). Skipping.")
            continue

        # Convert 2D pixel (cx, cy) and depth to 3D coordinates in the camera frame
        camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)

        # Convert camera frame coordinates to robot frame coordinates
        camera_coords_homogeneous = np.array([1000 * camera_coords[0], 1000 * camera_coords[1], 1000 * camera_coords[2], 1])
        robot_coords_homogeneous = np.dot(transformation_matrix, camera_coords_homogeneous)
        robot_coords = robot_coords_homogeneous[:3]  # Extract 3D coordinates

        print("Camera Coordinates (Homogeneous):", camera_coords_homogeneous)
        # print("Robot Coordinates (Homogeneous):", robot_coords_homogeneous)
        print("Robot Coordinates (Cartesian):", robot_coords)

        # Tag ID: 5
        # Position in Camera Frame (mm): [ -1.15863301  82.59690072 562.16642814]
        # Position in Robot Frame (mm): [121.55167254 -30.16229282  -3.59678858]

        # Apply a geometric correction for the sphere's center (optional)
        # Adjust z-coordinate for radius (viewpoint offset correction)
        robot_coords[2] -= radius

        # Append the sphere's color and its computed robot coordinates
        sphere_coordinates.append((color, tuple(robot_coords)))

    return sphere_coordinates

def main():
    try:
        # Camera calibration
        calibrate_camera()

        # Load the calibration matrix
        script_dir = os.path.abspath(os.path.dirname(__file__))
        transformation_matrix_path = os.path.join(script_dir, "camera_robot_transform.npy")
        print(f"Loading from: {transformation_matrix_path}")
        transformation_matrix = np.load(transformation_matrix_path)

        # Initialize RealSense camera
        camera = Realsense()

        # Define workspace mask (optional)
        workspace_mask = None  # Replace with actual mask if needed

        while True:
            # Detect spheres in live feed
            spheres, depth_frame = detect_colored_spheres_live(camera, workspace_mask)

            # Get 3d coordinates
            robot_sphere_coordinates = get_sphere_coordinates(spheres, camera, transformation_matrix, depth_frame)

            # Log the 3D coordinates
            for color, coords in robot_sphere_coordinates:
                print(f"{color} sphere at robot coordinates: {coords}")

            # PICK AND PLACE LOGIC HERE

            # Print detected spheres to the terminal
            for color, (x, y), radius in spheres:
                print(f"Detected {color} sphere at ({x}, {y}) with radius {radius}")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
