import sys
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

dh_table = np.array([
        [0,                                 77,  0,  -np.pi/2],
        [-(np.pi/2 - np.arcsin(24/130)),    0,   130, 0],
        [(np.pi/2 - np.arcsin(24/130)),     0,   124, 0],
        [0,                                 0,   126, 0]
    ])

def get_dh_row_mat(dh_row):
    theta, d, a, alpha = dh_row

    A_i = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha),  a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha),  -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,             np.sin(alpha),                  np.cos(alpha),                  d],
        [0,             0,                              0,                              1]
    ])

    return A_i

def get_acc_mat(joint_angles):
    # Ensure joint_angles is a numpy array
    joint_angles = np.asarray(joint_angles)

    # Create array to hold the A matrices
    A_matrices = np.zeros((4, 4, 4))

    # Create array to hold the T matrices
    T_matrices = np.zeros((4, 4, 4))

    for i in range(4):
        # Update the theta value in the DH table with the corresponding joint angle
        dh_row = dh_table[i].copy()
        dh_row[0] += np.radians(joint_angles[i])

        # Calculate the transformation matrix for the current joint
        A_matrices[i] = get_dh_row_mat(dh_row)

        # Update T_matrix array
        if i == 0:
            T_matrices[i] = A_matrices[i] # Handle base case
        else:
            T_matrices[i] = np.dot(T_matrices[i - 1], A_matrices[i]) # The next T is just the current A multiplied by last T

    return T_matrices

def get_jacobian(joint_angles):
    jacobian = np.zeros((6, 4))
    acc_mat = get_acc_mat(joint_angles)
    
    z0 = np.array([0, 0, 1])
    
    o4 = acc_mat[-1][:-1,-1]
    jacobian[:,0] = np.hstack((np.radians(np.cross(z0, o4)),z0))
    
    for i in range(1, len(joint_angles)):
        z = acc_mat[i-1][:-1,2]
        o = acc_mat[i-1][:-1,3]
        jacobian[:,i] = np.hstack((np.radians(np.cross(z, o4-o)), z))
        
    return jacobian

def main():
    configs = [[0, -10.62, -79.38, 0],[0, 0, 0, 0]]
    for c in configs:
        j = get_jacobian(c)
        det = np.linalg.det(j[:3,:3])
        print(f'jacobian for {c}:')
        print(j)
        print(f'det: {det}')
    return

if __name__ == "__main__":
    main()