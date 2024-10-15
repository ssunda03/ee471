import sys
import os

import time
import numpy as np
import matplotlib.pyplot as plt

def test():
    ee = [
        np.array([274, 0, 204, 0]),
        np.array([16, 4, 336, 15]),
        np.array([0, -270, 106, 0])
    ]
    for pos in ee:
        print(get_ik(pos))

def get_ik(ee_pos):
    mDim = [77, 130, 124, 126]  # Dimensional parameters of the robot
    mOtherDim = [128, 24]       # Additional dimensions

    joint_angles = np.zeros(4)  # Initialize joint angles
    radius = mOtherDim[0] + mDim[2] + mDim[3]
    center = [0, 0, mDim[0]]
    test_radius = [i - j for i, j in zip(ee_pos, center)]

    # Check if the end-effector is within reach
    if (radius**2 < sum([i**2 for i in test_radius]) and test_radius[2] > 0):
        raise ValueError("End effector position out of reach")

    r = np.sqrt(np.power(ee_pos[0], 2) + np.power(ee_pos[1], 2))
    rw = r - mDim[3] * np.cos(np.radians(ee_pos[3]))
    zw = ee_pos[2] - mDim[0] - mDim[3] * np.sin(np.radians(ee_pos[3]))
    dw = np.sqrt(np.power(rw,2) + np.power(zw,2))

    mu = np.arctan2(zw, rw)
    
    cosbeta = (np.power(mDim[1],2) + np.power(mDim[2],2) - np.power(dw,2)) / (2 * mDim[1] * mDim[2])
    cosbeta = np.clip(cosbeta, -1, 1)
    
    sinbeta = np.sqrt(1 - np.power(cosbeta,2))
    sinbeta = np.clip(sinbeta, -1, 1)
    
    sinbetas = [sinbeta, -1*sinbeta]
    
    beta = np.arctan2(-1*sinbeta, cosbeta)  # Use the positive solution for beta

    cosgamma = (np.power(dw,2) + np.power(mDim[1],2) - np.power(mDim[2],2)) / (2 * dw * mDim[1])
    cosgamma = np.clip(cosgamma, -1, 1)
    
    singamma = np.sqrt(1 - np.power(cosgamma,2))
    singamma = np.clip(singamma, -1, 1)
    
    singammas = [singamma, -1*singamma]
    
    gamma = np.arctan2(-1*singamma, cosgamma)  # Use the positive solution for gamma

    delta = np.arctan2(mOtherDim[1], mOtherDim[0])
    
    # Compute joint angles
    joint_angles[0] = np.degrees(np.arctan2(ee_pos[1], ee_pos[0]))
    joint_angles[1] = np.degrees((np.pi / 2) - delta - gamma - mu)
    joint_angles[2] = np.degrees((np.pi / 2) + delta - beta)
    joint_angles[3] = -ee_pos[3] - joint_angles[1] - joint_angles[2]

    return joint_angles


if __name__ == "__main__":
    test()