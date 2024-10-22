import numpy as np

class TrajPlanner:
    """
    Trajectory Planner class for calculating trajectories for different polynomial orders and relevant coefficients.
    """

    def __init__(self, setpoints):
        """
        Initialize the TrajPlanner class.

        Parameters:
        setpoints (numpy array): List of setpoints to travel to.
        """
        self.setpoints = setpoints

    def calc_cubic_coeff(self, t0, tf, p0, pf, v0, vf):
        """
        Given the initial time, final time, initial position, final position, initial velocity, and final velocity,
        returns cubic polynomial coefficients.

        Parameters:
        t0 (float): Start time of trajectory
        tf (float): End time of trajectory
        p0 (float): Initial setpoint
        pf (float): Final setpoint
        v0 (float): Initial velocity
        vf (float): Final velocity

        Returns:
        numpy array: The calculated polynomial coefficients.
        """
        coeff_matrix = np.array([
            [1, t0, t0 ** 2, t0 ** 3],
            [0, 1, 2 * t0, 3 * t0 ** 2],
            [1, tf, tf ** 2, tf ** 3],
            [0, 1, 2 * tf, 3 * tf ** 2]
        ])
        qs = np.array([p0, v0, pf, vf])
        coeff = np.linalg.solve(coeff_matrix, qs)

        return coeff


    def calc_cubic_traj(self, traj_time, points_num, coeff):
        """
        Given the time between setpoints, number of points between waypoints, and polynomial coefficients,
        returns the cubic trajectory of waypoints for a single pair of setpoints.

        Parameters:
        traj_time (int): Time between setPoints.
        points_num (int): Number of waypoints between setpoints.
        coeff (numpy array): Polynomial coefficients for trajectory.

        Returns:
        numpy array: The calculated waypoints.
        """
        waypoints = np.zeros(points_num)
        times = np.linspace(0, traj_time, points_num+2)[1:-1]

        for k, t in enumerate(times):
            waypoints[k] = coeff[0] + coeff[1] * t + coeff[2] * t**2 + coeff[3] * t**3
        
        return waypoints


    def get_cubic_traj(self, traj_time, points_num):
        """
        Given the time between setpoints and number of points between waypoints, returns the cubic trajectory.

        Parameters:
        traj_time (int): Time between setPoints.
        points_num (int): Number of waypoints between setpoints.

        Returns:
        numpy array: List of waypoints for the cubic trajectory.
        """
        setpoints = self.setpoints
        waypoints_list = np.zeros(((len(setpoints)-1)*(points_num+1)+1, 5))
        
        for i in range(4):
            count = 0
            for j in range(len(setpoints)-1):
                coeff = self.calc_cubic_coeff(0, traj_time, setpoints[j, i], setpoints[j+1, i], 0, 0)
                waypoints_list[count, 1:] = setpoints[j, :]
                count += 1
                waypoints_list[count:count+points_num, i+1] = self.calc_cubic_traj(traj_time, points_num, coeff)
                count += points_num
            waypoints_list[count, 1:] = setpoints[-1, :]

        time = np.linspace(0, traj_time*(len(setpoints)-1), waypoints_list.shape[0])
        waypoints_list[:, 0] = time
        # print(waypoints_list)
        return waypoints_list


    # def calc_cubic_coeff(self, t0, tf, q0, qf, v0=0, vf=0):
    #     """
    #     Calculate the cubic polynomial coefficients for a trajectory between two points.

    #     Parameters:
    #     t0 (float): Initial time.
    #     tf (float): Final time.
    #     q0 (float): Initial position.
    #     qf (float): Final position.
    #     v0 (float): Initial velocity (default is 0).
    #     vf (float): Final velocity (default is 0).

    #     Returns:
    #     numpy array: Coefficients of the cubic polynomial.
    #     """
    #     # Construct the matrix A and vector b to solve for the coefficients
    #     A = np.array([[1, t0, t0**2, t0**3],
    #                   [0, 1, 2*t0, 3*t0**2],
    #                   [1, tf, tf**2, tf**3],
    #                   [0, 1, 2*tf, 3*tf**2]])

    #     b = np.array([q0, v0, qf, vf])

    #     # Solve for the coefficients
    #     coeff = np.linalg.solve(A, b)
    #     print(f"Cubic coefficients: {coeff}")
    #     return coeff

    # def calc_cubic_traj(self, traj_time, num_waypoints, coeff):
    #     """
    #     Calculate the cubic trajectory for a single joint.

    #     Parameters:
    #     traj_time (float): Total time of the trajectory.
    #     num_waypoints (int): Number of intermediate waypoints.
    #     coeff (numpy array): Coefficients of the cubic polynomial.

    #     Returns:
    #     numpy array: Trajectory points for the joint.
    #     """
    #     # Generate time points for the waypoints
    #     t_points = np.linspace(0, traj_time, num_waypoints + 2)
    #     waypoints = np.zeros_like(t_points) # Empty array for the waypoints

    #     # Calculate the position for each time point using the cubic polynomial
    #     for i, t in enumerate(t_points):
    #         waypoints[i] = coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3

    #     return waypoints

    # def get_cubic_traj(self, traj_time, num_waypoints):
    #     """
    #     Calculate cubic trajectories for all joints.

    #     Parameters:
    #     traj_time (float): Total time for the trajectory.
    #     num_waypoints (int): Number of intermediate waypoints.

    #     Returns:
    #     numpy array: Trajectories for all joints, including time.
    #         The output is an (n+2) x 5 numpy array where:
    #         - The first column represents time values.
    #         - The next four columns represent the cubic trajectory waypoints
    #             for each of the four joints (joint 1 in column 2, joint 2 in column 3, etc.).
            
    #         Example of the output structure:
    #         [
    #         [t0, q1(t0), q2(t0), q3(t0), q4(t0)],   # First row (start time and positions)
    #         [t1, q1(t1), q2(t1), q3(t1), q4(t1)],   # Intermediate waypoints
    #         ...
    #         [tn, q1(tn), q2(tn), q3(tn), q4(tn)]    # Last row (end time and positions)
    #         ]
    #     """
    #     num_joints = self.setpoints.shape[1]  # Get the number of joints
    #     num_setpoints = self.setpoints.shape[0]  # Get the number of setpoints
    #     total_waypoints = (num_setpoints - 1) * num_waypoints + 1  # Including start and end points for all segments
        
    #     traj = np.zeros((total_waypoints, num_joints + 1))  # +1 for time column

    #     current_waypoint_idx = 0  # Index to keep track of where to fill in values

    #     # Calculate trajectory for each joint and each segment between setpoints
    #     for joint in range(num_joints):
    #         for i in range(num_setpoints - 1):
    #             q0, qf = self.setpoints[i, joint], self.setpoints[i + 1, joint]
    #             segment_time = traj_time / (num_setpoints - 1)  # Time for each segment
                
    #             # Generate time points for this segment
    #             t_segment = np.linspace(i * segment_time, (i + 1) * segment_time, num_waypoints + 1)
                
    #             # Get the cubic coefficients for this segment
    #             coeff = self.calc_cubic_coeff(0, segment_time, q0, qf)
                
    #             # Calculate the cubic trajectory for this segment
    #             traj_segment = self.calc_cubic_traj(segment_time, num_waypoints, coeff)
                
    #             # Fill in the time and trajectory for the current segment
    #             traj[current_waypoint_idx:current_waypoint_idx + num_waypoints + 1, 0] = t_segment  # Time column
    #             traj[current_waypoint_idx:current_waypoint_idx + num_waypoints + 1, joint + 1] = traj_segment  # Joint column
                
    #             current_waypoint_idx += num_waypoints  # Move index to the next segment

    #     return traj
