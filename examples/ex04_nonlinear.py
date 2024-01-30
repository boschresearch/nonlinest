"""
A nonlinear state estimation example.

This example shows how to track a vehicle state using a constant
turn rate / constant velocity system model, which is nonlinear,
and a position measurement in polar coordinates, which is nonlinear too.

The model is described for example here:
    M. Roth, G. Hendeby and F. Gustafsson, "EKF/UKF maneuvering target tracking using coordinated turn models with
    polar/Cartesian velocity," 17th International Conference on Information Fusion (FUSION), 2014, pp. 1-8.
"""

import numpy as np

from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.ghkf import GHKF
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


# To implement such a system model, we simply need to derive from the generic system model class
# and override the function system_equation(...).
# The SystemModel class comes with handy functions such as Jacobian calculation,
# example value generation / simulation. It is compatible to all filters in this library.
#
# There are also more specific system models like the AdditiveNoiseSystemModel, which should
# be used if applicable. They enable some filters, like the EKF, to take the noise into account
# in closed form.
class ConstantTurnRateVelocitySystemModel(SystemModel):
    def __init__(self, delta_t):
        super(ConstantTurnRateVelocitySystemModel, self).__init__()
        self.delta_t = delta_t
        # A good practice would also be to define the system model noise in the constructor,
        # however in this example we define it outside to show that you can also set it via
        # the set_noise() function afterwards.
        # self.noise = Gaussian(...)

    def system_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        position_x, position_y, direction, speed, turn_rate = state_samples
        noise_speed, noise_turn_rate = noise_samples

        speed_pred = speed + noise_speed
        turn_rate_pred = turn_rate + noise_turn_rate
        direction_pred = direction + self.delta_t * turn_rate_pred
        position_x_pred = (
            position_x + np.cos(direction_pred) * self.delta_t * speed_pred
        )
        position_y_pred = (
            position_y + np.sin(direction_pred) * self.delta_t * speed_pred
        )

        predictions = np.stack(
            [
                position_x_pred,
                position_y_pred,
                direction_pred,
                speed_pred,
                turn_rate_pred,
            ]
        )
        return predictions


# Analogous to the system model, we have to derive from a general measurement model class
# and override the measurement_equation() function.
# In this case, we override the AdditiveNoiseMeasurementModel class, which means that
# filters like the EKf take care of the noise in closed form.
# Here, we set the noise directly in the constructor.
class NaivePolarMeasurementModel(AdditiveNoiseMeasurementModel):
    def __init__(self):
        self.noise = Gaussian(
            mean=ColumnVectors([0, 0]), covariance=np.diag([1e-2, 1e-4])
        )

    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        px, py, _, _, _ = state_samples
        measurements = np.stack([np.sqrt(px**2 + py**2), np.arctan2(py, px)])
        return measurements


def run_example():
    np.set_printoptions(precision=4)

    # System model
    delta_t = 0.1  # Step size
    sys_model = ConstantTurnRateVelocitySystemModel(delta_t)

    # Set system model noise
    system_model_noise = Gaussian(
        mean=ColumnVectors([0, 0]), covariance=np.diag([1e-1, 1e-3])
    )
    sys_model.set_noise(system_model_noise)

    # Measurement model
    meas_model = NaivePolarMeasurementModel()

    # Estimator
    filter = GHKF()

    # The estimator can be changed seamlessly:
    # filter = UKF()

    # Set initial state estimate
    initial_state_estimate = Gaussian(
        mean=ColumnVectors([1, 1, 0, 0, 0]), covariance=np.diag([10, 10, 1e-1, 1, 1e-1])
    )
    filter.set_state(initial_state_estimate)

    # Perform the prediction step
    filter.predict(sys_model=sys_model)

    # Print state after prediction
    mean, cov, _ = filter.get_state().get_mean_and_covariance()
    print(f"Predicted mean:\n {mean}")
    print(f"Predicted covariance:\n {cov}")

    # Perform the update step
    filter.update(meas_model=meas_model, measurement=ColumnVectors([3, np.pi / 5]))

    # Print state after update
    mean, cov, _ = filter.get_state().get_mean_and_covariance()
    print(f"Updated mean:\n {mean}")
    print(f"Updated covariance:\n {cov}")


if __name__ == "__main__":
    run_example()
