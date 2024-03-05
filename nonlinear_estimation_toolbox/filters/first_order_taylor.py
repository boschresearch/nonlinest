from typing import Tuple, Union

import numpy as np

from nonlinear_estimation_toolbox.filters.linear_gaussian import LinearGaussianFilter
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import (
    AdditiveNoiseSystemModel,
    SystemModel,
)


class FirstOrderTaylorLinearGaussianFilter(LinearGaussianFilter):
    # TODO: Add doc string
    """
    Toolbox/Interfaces/Filters/LinearGaussianFilters/FirstOrderTaylorLinearGaussianFilter.m
    """

    def __init__(self, name: str = ""):
        super(FirstOrderTaylorLinearGaussianFilter, self).__init__(name)

    def predict_system_model(
        self, sys_model: SystemModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise_mean, _, noise_cov_sqrt = sys_model.noise.get_mean_and_covariance()
        dim_noise = len(noise_mean)

        # linearize system model around current state mean and noise mean
        state_jacobian, noise_jacobian = sys_model.derivative(
            self.state_mean, noise_mean
        )

        # check computed derivatives
        self._check_state_jacobian(state_jacobian, self.dim_state, self.dim_state)
        self._check_noise_jacobian(noise_jacobian, self.dim_state, dim_noise)

        # Compute predicted state mean
        predicted_state_mean = sys_model.system_equation(self.state_mean, noise_mean)

        # Compute predicted state covariance
        A = state_jacobian.dot(self.state_cov_sqrt)
        B = noise_jacobian.dot(noise_cov_sqrt)

        predicted_state_cov = A.dot(A.T) + B.dot(B.T)

        return predicted_state_mean, predicted_state_cov

    def predict_additive_noise_system_model(
        self, sys_model: AdditiveNoiseSystemModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise_mean, noise_cov, _ = sys_model.noise.get_mean_and_covariance()
        dim_noise = len(noise_mean)

        self._check_additive_system_noise(dim_noise)

        # Linearize system model around current state mean
        state_jacobian = sys_model.derivative(self.state_mean)

        # Check computed derivative
        self._check_state_jacobian(state_jacobian, self.dim_state, self.dim_state)

        # Compute predicted state mean
        predicted_state_mean = sys_model.system_equation(self.state_mean) + noise_mean

        # Compute predicted state covariance
        A = state_jacobian.dot(self.state_cov_sqrt)

        predicted_state_cov = A.dot(A.T) + noise_cov

        return predicted_state_mean, predicted_state_cov

    def evaluate_additive_noise_measurement_model(
        self,
        meas_model: AdditiveNoiseMeasurementModel,
        dim_meas: int,
        state_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dim_state = len(state_mean)

        # Linearize measurement model around current state mean
        state_jacobian = meas_model.derivative(state_mean)

        # Check computed derivative
        self._check_state_jacobian(state_jacobian, dim_meas, dim_state)

        h = meas_model.measurement_equation(state_mean)
        return h, state_jacobian

    def evaluate_measurement_model(
        self,
        meas_model: Union[MeasurementModel, MixedNoiseMeasurementModel],
        dim_meas: int,
        dim_noise: int,
        noise_mean: np.ndarray,
        state_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim_state = len(state_mean)

        # Linearize measurement model around current state mean and noise mean
        state_jacobian, noise_jacobian = meas_model.derivative(state_mean, noise_mean)

        # Check computed derivatives
        self._check_state_jacobian(state_jacobian, dim_meas, dim_state)
        self._check_noise_jacobian(noise_jacobian, dim_meas, dim_noise)

        h = meas_model.measurement_equation(state_mean, noise_mean)
        return h, state_jacobian, noise_jacobian
