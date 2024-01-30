"""
# A linear state estimation example.

The system state contains the scalars position p(t) and speed s(t):
x(t) = (p(t), s(t))

The linear system model is a constant velocity model:
x(t) = A @ x(t-1) + B @ w
"""

import numpy as np

from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.measurement_models import LinearMeasurementModel
from nonlinear_estimation_toolbox.system_models import LinearSystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


def run_example():
    np.set_printoptions(precision=4)

    # Step size
    T = 0.01

    # Linear system model
    #   x(t) = A @ x(t-1) + B @ w
    A = np.array([[1, T], [0, 1]])
    B = np.array([[T], [1]])
    sys_model = LinearSystemModel(sys_matrix=A, sys_noise_matrix=B)

    # Time-invariant, zero-mean process noise from which w will be sampled
    system_model_noise = Gaussian(
        mean=ColumnVectors(0), covariance=np.array([[0.1**2]])
    )
    sys_model.set_noise(system_model_noise)

    # Linear measurement model
    #   y(t) = H @ x(t) + v
    H = np.array([[1, 0]])
    meas_model = LinearMeasurementModel(meas_matrix=H)

    # Time-invariant, zero-mean Gaussian white noise from which v will be sampled
    measurement_model_noise = Gaussian(
        mean=ColumnVectors(0), covariance=np.array([[0.05**2]])
    )
    meas_model.set_noise(measurement_model_noise)

    # Estimator
    # In our linear example we can use any estimator that derives from
    # LinearGaussianFilter: EKF, UKF, GHKF, etc.
    # They fall back to a standard kalman prediction or standard kalman
    # update step when a linear measurement model or a linear system
    # model is given, respectively
    filter = EKF()

    # Initial state estimate
    #     E[x(t=0)]   = (0, 0)
    #     Cov[x(t=0)] = diag((1, 1))
    initial_state_estimate = Gaussian(
        mean=ColumnVectors(np.zeros(2)), covariance=np.eye(2)
    )
    filter.set_state(state=initial_state_estimate)

    # Perform the prediction step
    filter.predict(sys_model=sys_model)

    # Print state after prediction
    predicted_state = filter.get_state()
    mean, cov, _ = predicted_state.get_mean_and_covariance()
    print(f"Predicted mean:\n {mean}")
    print(f"Predicted covariance:\n {cov}")

    # Assume we receive a measurement
    measurement = ColumnVectors(2.139)

    # Perform the measurement update
    filter.update(meas_model=meas_model, measurement=measurement)

    # Print state after update
    updated_state = filter.get_state()
    mean, cov, _ = updated_state.get_mean_and_covariance()
    print(f"Updated mean:\n {mean}")
    print(f"Updated covariance:\n {cov}")


if __name__ == "__main__":
    run_example()
