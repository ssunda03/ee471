import numpy as np

class PIDController:
    def __init__(self, dim=3, dt=0.05):
        # Initialize gains (tuned for position control in mm)
        self.Kp = 0.5 * np.eye(dim)  # Proportional gain
        self.Ki = 0.05 * np.eye(dim) # Integral gain
        self.Kd = 0.1 * np.eye(dim)  # Derivative gain

        # Initialize error terms
        self.error_integral = np.zeros(dim)
        self.error_prev = np.zeros(dim)
        self.dt = dt  # Control period in seconds

    def compute_pid(self, error):
        # Compute the integral term by accumulating errors over time
        self.error_integral += error * self.dt

        # Compute the derivative term using the difference between current and previous error
        error_derivative = (error - self.error_prev) / self.dt

        # Combine all terms using the pre-defined gain matrices
        output = (self.Kp @ error) + (self.Ki @ self.error_integral) + (self.Kd @ error_derivative)

        # Update previous error for the next derivative calculation
        self.error_prev = error

        return output