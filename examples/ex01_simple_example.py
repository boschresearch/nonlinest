"""
Minimal example for running a prediction and update step with the EKF.

We use the system equation
    x_{k+1} = x_k^2 + w_k
and the measurement equation
    y = x^3 + v_k.

Note that we do not need to give the derivatives of the system and measurement models because the EKF approximates them numerically.
You can easily switch out the EKF for another filter, e.g. the UKF, by replacing EKF with UKF everywhere.
"""

import numpy as np

from nonlinear_estimation_toolbox.distributions import Gaussian

# %%
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import AdditiveNoiseSystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors

filter = EKF()

# %%
initial_state = Gaussian(ColumnVectors([1, 2]), np.eye(2))
filter.set_state(initial_state)


# %%
class SquareSysModel(AdditiveNoiseSystemModel):
    def system_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        return np.square(state_samples)


sys_model = SquareSysModel()
sys_noise = Gaussian(ColumnVectors([0, 0]), np.eye(2))
sys_model.set_noise(sys_noise)

# %%
filter.predict(sys_model)
print(filter.get_state())


# %%
class CubicMeasModel(AdditiveNoiseMeasurementModel):
    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        return np.power(state_samples, 3)


meas_model = CubicMeasModel()
meas_noise = Gaussian(ColumnVectors([0, 0]), np.eye(2))
meas_model.set_noise(meas_noise)
measurement = ColumnVectors([4, 5])

# %%
filter.update(meas_model, measurement)
print(filter.get_state())

# %%
