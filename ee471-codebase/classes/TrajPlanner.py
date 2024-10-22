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

    def calc_cubic_coeff(self, t0, tf, q0, qf, v0=0, vf=0):
        """
        Calculate the cubic polynomial coefficients for a trajectory between two points.

        Parameters:
        t0 (float): Initial time.
        tf (float): Final time.
        q0 (float): Initial position.
        qf (float): Final position.
        v0 (float): Initial velocity (default is 0).
        vf (float): Final velocity (default is 0).

        Returns:
        numpy array: Coefficients of the cubic polynomial.
        """
        # Construct the matrix A and vector b to solve for the coefficients
        A = np.array([[1, t0, t0**2, t0**3],
                      [0, 1, 2*t0, 3*t0**2],
                      [1, tf, tf**2, tf**3],
                      [0, 1, 2*tf, 3*tf**2]])

        b = np.array([q0, v0, qf, vf])

        # Solve for the coefficients
        coeff = np.linalg.solve(A, b)
        print(f"Cubic coefficients: {coeff}")
        return coeff

    def calc_cubic_traj(self, traj_time, num_waypoints, coeff):
        """
        Calculate the cubic trajectory for a single joint.

        Parameters:
        traj_time (float): Total time of the trajectory.
        num_waypoints (int): Number of intermediate waypoints.
        coeff (numpy array): Coefficients of the cubic polynomial.

        Returns:
        numpy array: Trajectory points for the joint.
        """
        # Generate time points for the waypoints
        t_points = np.linspace(0, traj_time, num_waypoints + 2)
        waypoints = np.zeros_like(t_points) # Empty array for the waypoints

        # Calculate the position for each time point using the cubic polynomial
        for i, t in enumerate(t_points):
            waypoints[i] = coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3

        return waypoints

    def get_cubic_traj(self, traj_time, num_waypoints):
        """
        Calculate cubic trajectories for all joints.

        Parameters:
        traj_time (float): Total time for the trajectory.
        num_waypoints (int): Number of intermediate waypoints.

        Returns:
        numpy array: Trajectories for all joints, including time.
            The output is an (n+2) x 5 numpy array where:
            - The first column represents time values.
            - The next four columns represent the cubic trajectory waypoints
                for each of the four joints (joint 1 in column 2, joint 2 in column 3, etc.).
            
            Example of the output structure:
            [
            [t0, q1(t0), q2(t0), q3(t0), q4(t0)],   # First row (start time and positions)
            [t1, q1(t1), q2(t1), q3(t1), q4(t1)],   # Intermediate waypoints
            ...
            [tn, q1(tn), q2(tn), q3(tn), q4(tn)]    # Last row (end time and positions)
            ]
        """
        num_joints = self.setpoints.shape[1] # Get the number of joints 
        total_waypoints = num_waypoints + 2  # Including the start and end points
        traj = np.zeros((total_waypoints, num_joints + 1))  # +1 for time column

        # Generate time points
        t_points = np.linspace(0, traj_time, total_waypoints)
        traj[:, 0] = t_points  # First column is time

        # Calculate trajectory for each joint
        for joint in range(num_joints):
            q0, qf = self.setpoints[0, joint], self.setpoints[1, joint]
            coeff = self.calc_cubic_coeff(0, traj_time, q0, qf)
            traj[:, joint + 1] = self.calc_cubic_traj(traj_time, num_waypoints, coeff)

        return traj
