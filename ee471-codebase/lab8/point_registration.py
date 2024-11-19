import sys
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))   
    

def point_registration(A, B):
    """
    Performs 3D point set registration using the Kabsch algorithm to find the rigid transformation
    matrix that maps points in A (camera coordinates) to points in B (robot coordinates).

    Args:
    - A (numpy.ndarray): A 3xN array of 3D points in camera coordinates.
    - B (numpy.ndarray): A 3xN array of 3D points in robot coordinates.

    Returns:
    - T (numpy.ndarray): A 4x4 transformation matrix that maps points in A to B.
    """

    # Step 1: Center the point sets by subtracting their centroids
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Step 2: Calculate the covariance matrix H
    H = np.dot(A_centered, B_centered.T)

    # Step 3: Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, S, Vt = np.linalg.svd(H)
    
    # Step 4: Compute the optimal rotation matrix R
    R = np.dot(Vt.T, U.T)

    # Handle reflection case (det(R) should be +1 for proper rotation)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 5: Compute the translation vector t
    t = centroid_B - np.dot(R, centroid_A)

    # Step 6: Build the transformation matrix T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T

def test_point_registration():
    # Define test point sets A (camera coordinates) and B (robot coordinates)
    A = np.array([
        [681.2, 526.9, 914.8],
        [542.3, 381., 876.5],
        [701.2, 466.3, 951.4],
        [598.4, 556.8, 876.9],
        [654.3, 489., 910.2]
    ]).T

    B = np.array([
        [110.1, 856.3, 917.8],
        [115.1, 654.9, 879.5],
        [167.1, 827.5, 954.4],
        [ 30.4, 818.8, 879.9],
        [117.9, 810.4, 913.2]
    ]).T

    # Call the point_registration function to get the transformation matrix
    T = point_registration(A, B)
    
    # Apply the transformation to A
    A_transformed = np.dot(T[:3, :3], A) + T[:3, 3].reshape(3, 1)
    
    # Calculate RMSE between the transformed A and B
    error = rmse(A_transformed, B)
    
    print("Transformation Matrix T:")
    print(T)
    print(f"RMSE between transformed A and B: {error:.6f}")
    print(f"RR^T:")
    print(f"{T[:3, :3] @ T[:3, :3].T}")
    print(f"Determinant of R: {np.linalg.det(T[:3, :3])}")

def rmse(A, B):
    """Computes the Root Mean Squared Error (RMSE) between two sets of points A and B."""
    return np.sqrt(np.mean(np.sum((A - B) ** 2, axis=0)))

def main():
    test_point_registration()
    return

if __name__ == "__main__":
    main()