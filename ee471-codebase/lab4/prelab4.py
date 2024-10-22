# pre lab 4
# Diego Curiel
# EE 471: Vision Guided Robotic Manipulation
# Dr. Siavash Farzan

# # Usage Example
# import numpy as np
# from traj_planner import TrajPlanner

# # Define setpoints (example values)
# setpoints = np.array([
#     [15, -45, -60, 90],
#     [-90, 15, 30, -45]
# ])

# # Create a TrajPlanner object
# trajectories = TrajPlanner(setpoints)

# # Generate cubic trajectory
# cubic_traj = trajectories.get_cubic_traj(traj_time=5, points_num=10)
# print(cubic_traj)

import sys
import os
import numpy as np

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from TrajPlanner import TrajPlanner

# Define the setpoints as a numpy array
setpoints = np.array([[15, -45, -60, 90],
                      [-90, 15, 30, -45]])

# Create an object of the TrajPlanner class
trajectories = TrajPlanner(setpoints)

# Call the get_cubic_traj() method with trajectory time and number of waypoints
traj_time = 5
num_waypoints = 6
traj = trajectories.get_cubic_traj(traj_time, num_waypoints)

# Print the planned trajectory
print("Planned trajectory:\n", traj)
