from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox.filters.first_order_taylor import (
    FirstOrderTaylorLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.filters.iterative_kalman import IterativeKalmanFilter
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)


class EKF(
    IterativeKalmanFilter, FirstOrderTaylorLinearGaussianFilter
):  # TODO: get rid of multi-inheritance
    # TODO: Add doc string
    """
    Toolbox/Filters/IterativeKalmanFilters/EKF.m
    """

    def __init__(self, name: str = ""):
        if name == "":
            name = "EKF"
        super(EKF, self).__init__(name)
        self.linearized_measurement_model_func = None

    def setup_measurement_model(
        self, measurement_model: MeasurementModel, dim_meas: int
    ):
        (
            noise_mean,
            _,
            noise_cov_sqrt,
        ) = measurement_model.noise.get_mean_and_covariance()
        dim_noise = len(noise_mean)

        self.linearized_measurement_model_func = (
            lambda state_mean: self.linearized_measurement_model(
                measurement_model,
                dim_meas,
                dim_noise,
                noise_mean,
                noise_cov_sqrt,
                state_mean,
            )
        )

    def setup_additive_noise_measurement_model(
        self, measurement_model: AdditiveNoiseMeasurementModel, dim_meas: int
    ):
        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = measurement_model.noise.get_mean_and_covariance()
        dim_add_noise = len(add_noise_mean)

        self._check_additive_measurement_noise(dim_meas, dim_add_noise)

        self.linearized_measurement_model_func = (
            lambda state_mean: self.linearized_additive_noise_meas_model(
                measurement_model, dim_meas, add_noise_mean, add_noise_cov, state_mean
            )
        )

    def setup_mixed_noise_measurement_model(
        self, meas_model: MixedNoiseMeasurementModel, dim_meas: int
    ):
        noise_mean, _, noise_cov_sqrt = meas_model.noise.get_mean_and_covariance()
        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = meas_model.additive_noise.get_mean_and_covariance()
        dim_noise = len(noise_mean)
        dim_add_noise = len(add_noise_mean)

        self._check_additive_measurement_noise(dim_meas, dim_add_noise)

        self.linearized_measurement_model_func = (
            lambda state_mean: self.linearized_mixed_noise_meas_model(
                meas_model,
                dim_meas,
                dim_noise,
                noise_mean,
                noise_cov_sqrt,
                add_noise_mean,
                add_noise_cov,
                state_mean,
            )
        )

    def get_meas_moments(
        self,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        meas_mean, H, R = self.linearized_measurement_model_func(prior_state_mean)

        Px = H.dot(prior_state_cov_sqrt)

        meas_cov = Px.dot(Px.T) + R
        state_meas_cross_cov = prior_state_cov.dot(H.T)

        return meas_mean, meas_cov, state_meas_cross_cov

    def get_meas_moments_iteration(
        self,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
        updated_state_mean: np.ndarray,
        *args
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, H, R = self.linearized_measurement_model_func(updated_state_mean)

        Px = H.dot(prior_state_cov_sqrt)

        meas_mean = h + H.dot(prior_state_mean - updated_state_mean)
        meas_cov = Px.dot(Px.T) + R
        state_meas_cross_cov = prior_state_cov.dot(H.T)

        return meas_mean, meas_cov, state_meas_cross_cov

    def linearized_measurement_model(
        self,
        meas_model: MeasurementModel,
        dim_meas: int,
        dim_noise: int,
        noise_mean: np.ndarray,
        noise_cov_sqrt: np.ndarray,
        state_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, H, noise_jacobian = self.evaluate_measurement_model(
            meas_model, dim_meas, dim_noise, noise_mean, state_mean
        )

        Pv = noise_jacobian.dot(noise_cov_sqrt)
        R = Pv.dot(Pv.T)

        return h, H, R

    def linearized_additive_noise_meas_model(
        self,
        meas_model: AdditiveNoiseMeasurementModel,
        dim_meas: int,
        add_noise_mean: np.ndarray,
        add_noise_cov: np.ndarray,
        state_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, H = self.evaluate_additive_noise_measurement_model(
            meas_model, dim_meas, state_mean
        )

        h = h + add_noise_mean

        R = add_noise_cov

        return h, H, R

    def linearized_mixed_noise_meas_model(
        self,
        meas_model: MixedNoiseMeasurementModel,
        dim_meas: int,
        dim_noise: int,
        noise_mean: np.ndarray,
        noise_cov_sqrt: np.ndarray,
        add_noise_mean: np.ndarray,
        add_noise_cov: np.ndarray,
        state_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, H, noise_jacobian = self.evaluate_measurement_model(
            meas_model, dim_meas, dim_noise, noise_mean, state_mean
        )

        h = h + add_noise_mean

        Pv = noise_jacobian.dot(noise_cov_sqrt)
        R = Pv.dot(Pv.T) + add_noise_cov

        return h, H, R
