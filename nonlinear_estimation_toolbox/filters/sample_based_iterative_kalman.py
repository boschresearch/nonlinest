from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox.filters.iterative_kalman import IterativeKalmanFilter
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)


class SampleBasedIterativeKalmanFilter(
    IterativeKalmanFilter, SampleBasedLinearGaussianFilter
):
    def __init__(self, name: str = ""):
        # This filter is not finished
        super(SampleBasedIterativeKalmanFilter, self).__init__(name)
        self.moment_func = None

    def setup_measurement_model(
        self, meas_model: MeasurementModel, dim_meas: int
    ) -> None:
        noise_mean, _, noise_cov_sqrt = meas_model.noise.get_mean_and_covariance()

        self.moment_func = (
            lambda state_mean, state_cov_sqrt: self.moment_func_meas_model(
                meas_model,
                dim_meas,
                noise_mean,
                noise_cov_sqrt,
                state_mean,
                state_cov_sqrt,
            )
        )

    def setup_additive_noise_measurement_model(
        self, meas_model: AdditiveNoiseMeasurementModel, dim_meas: int
    ) -> None:
        add_noise_mean, add_noise_cov, _ = meas_model.noise.get_mean_and_covariance()
        dim_add_noise = add_noise_cov.shape[0]

        self._check_additive_measurement_noise(dim_meas, dim_add_noise)

        self.moment_func = lambda state_mean, state_cov_sqrt: self.moment_func_additive_noise_meas_model(
            meas_model,
            dim_meas,
            add_noise_mean,
            add_noise_cov,
            state_mean,
            state_cov_sqrt,
        )

    def setup_mixed_noise_measurement_model(
        self, meas_model: MixedNoiseMeasurementModel, dim_meas: int
    ) -> None:
        noise_mean, _, noise_cov_sqrt = meas_model.noise.get_mean_and_covariance()
        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = meas_model.additive_noise.get_mean_and_covariance()
        dim_add_noise = add_noise_cov.shape[0]

        self._check_additive_measurement_noise(dim_meas, dim_add_noise)

        self.moment_func = (
            lambda state_mean, state_cov_sqrt: self.moment_func_mixed_noise_meas_model(
                meas_model,
                dim_meas,
                noise_mean,
                noise_cov_sqrt,
                add_noise_mean,
                add_noise_cov,
                state_mean,
                state_cov_sqrt,
            )
        )

    def get_meas_moments(
        self, prior_mean: np.ndarray, _, prior_cov_sqrt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.moment_func(prior_mean, prior_cov_sqrt)

    def get_meas_moments_iteration(
        self,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
        updated_state_mean: np.ndarray,
        updated_state_cov: np.ndarray,
        updated_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        meas_mean, meas_cov, state_meas_cross_cov = self.moment_func(
            updated_state_mean, updated_state_cov_sqrt
        )

        updated_state_cov_sqrt_inv = np.linalg.inv(updated_state_cov_sqrt)
        A = state_meas_cross_cov.T.dot(updated_state_cov_sqrt_inv.T)
        H = A.dot(updated_state_cov_sqrt_inv)
        P = H.dot(prior_state_cov_sqrt)

        meas_mean = meas_mean + H.dot(prior_state_mean - updated_state_mean)
        meas_cov = meas_cov + P.dot(P.T) - A.dot(A.T)
        state_meas_cross_cov = prior_state_cov.dot(H.T)
        return meas_mean, meas_cov, state_meas_cross_cov

    def moment_func_meas_model(
        self,
        meas_model: MeasurementModel,
        dim_meas: int,
        noise_mean: np.ndarray,
        noise_cov_sqrt: np.ndarray,
        state_mean: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (
            h_samples,
            weights,
            zero_mean_state_samples,
            _,
        ) = self.evaluate_meas_model_uncorrelated(
            meas_model, dim_meas, state_mean, state_cov_sqrt, noise_mean, noise_cov_sqrt
        )

        meas_mean, meas_cov, state_meas_cross_cov, _ = self.get_meas_model_moments(
            h_samples,
            weights,
            zero_mean_state_samples,
        )

        return meas_mean, meas_cov, state_meas_cross_cov

    def moment_func_additive_noise_meas_model(
        self,
        meas_model: AdditiveNoiseMeasurementModel,
        dim_meas: int,
        additive_noise_mean: np.ndarray,
        additive_noise_cov: np.ndarray,
        state_mean: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (
            h_samples,
            weights,
            zero_mean_state_samples,
        ) = self.evaluate_additive_noise_meas_model(
            meas_model, dim_meas, state_mean, state_cov_sqrt
        )

        h_mean, h_cov, state_h_cross_cov, _ = self.get_meas_model_moments(
            h_samples, weights, zero_mean_state_samples
        )

        meas_mean = h_mean + additive_noise_mean
        meas_cov = h_cov + additive_noise_cov
        state_meas_cross_cov = state_h_cross_cov

        return meas_mean, meas_cov, state_meas_cross_cov

    def moment_func_mixed_noise_meas_model(
        self,
        meas_model: MixedNoiseMeasurementModel,
        dim_meas: int,
        noise_mean: np.ndarray,
        noise_cov_sqrt: np.ndarray,
        add_noise_mean: np.ndarray,
        add_noise_cov: np.ndarray,
        state_mean: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (
            h_samples,
            weights,
            zero_mean_state_samples,
            _,
        ) = self.evaluate_meas_model_uncorrelated(
            meas_model,
            dim_meas,
            state_mean,
            state_cov_sqrt,
            noise_mean,
            noise_cov_sqrt,
        )

        h_mean, h_cov, state_h_cross_cov, _ = self.get_meas_model_moments(
            h_samples, weights, zero_mean_state_samples
        )

        meas_mean = h_mean + add_noise_mean
        meas_cov = h_cov + add_noise_cov

        state_meas_cross_cov = state_h_cross_cov
        return meas_mean, meas_cov, state_meas_cross_cov
