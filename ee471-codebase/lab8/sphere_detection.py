import cv2
import numpy as np
import os


def detect_colored_spheres(image_path):
    """
    Detects and classifies colored spheres in an image.

    Args:
        image_path (str): Path to the image file containing the robot workspace.

    Returns:
        tuple: A tuple containing a list of detected spheres and the path to the processed image file.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define refined HSV color ranges for spheres
    color_ranges = {
        "red": [(0, 100, 100), (10, 255, 255)],
        "orange": [(10, 150, 150), (25, 255, 255)],
        "yellow": [(25, 150, 150), (35, 255, 255)],
        "blue": [(100, 150, 100), (130, 255, 255)]
    }

    detected_spheres = []

    # Process each color range
    for color, (lower, upper) in color_ranges.items():
        # Create a binary mask for the current color
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Apply Gaussian blur to reduce noise
        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

        # Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            cleaned_mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,         # Minimum distance between detected centers
            param1=100,         # Canny edge detection threshold
            param2=20,          # Accumulator threshold for circle detection
            minRadius=10,       # Minimum circle radius
            maxRadius=50        # Maximum circle radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                detected_spheres.append((color, (int(x), int(y))))

                # Annotate the detected circles on the image
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(
                    image,
                    color,
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # Save the processed image
    script_dir = os.path.abspath(os.path.dirname(__file__))
    output_image_path = os.path.join(script_dir, "processed_image_prelab8.jpg") 
    cv2.imwrite(output_image_path, image)

    return detected_spheres, output_image_path


def main():
    # Print the absolute path of the script directory for troubleshooting
    script_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Script is running from: {script_dir}")

    # Load the workspace image
    image_path = os.path.join(script_dir, "image_prelab8.jpg")
    print(f"Looking for image path at: {image_path}")

    try:
        # Detect spheres and save the processed image
        detected_spheres, processed_image_path = detect_colored_spheres(image_path)

        # Print the results to the terminal
        print("Detected Spheres:")
        for color, (x, y) in detected_spheres:
            print(f"{color}: ({x}, {y})")

        print(f"Processed image saved as: {processed_image_path}")

        # Display the processed image
        processed_image = cv2.imread(processed_image_path)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
