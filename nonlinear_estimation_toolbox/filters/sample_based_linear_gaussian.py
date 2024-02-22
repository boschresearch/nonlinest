from typing import Tuple, Union

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
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


class SampleBasedLinearGaussianFilter(LinearGaussianFilter):
    def __init__(self, name: str = ""):
        super(SampleBasedLinearGaussianFilter, self).__init__(name)

    def predict_system_model(
        self, sys_model: SystemModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        # todo untested
        noise_mean, _, noise_cov_sqrt = sys_model.noise.get_mean_and_covariance()
        dim_noise = noise_cov_sqrt.shape[0]
        dim_aug_state = self.dim_state + dim_noise

        (
            std_normal_samples,
            weights,
            num_samples,
        ) = self.get_std_normal_samples_prediction(dim_aug_state)

        zero_mean_state_samples = self.state_cov_sqrt.dot(
            std_normal_samples[0 : self.dim_state, :]
        )
        state_samples = zero_mean_state_samples + self.state_mean

        zero_mean_noise_samples = noise_cov_sqrt.dot(
            std_normal_samples[self.dim_state :, :]
        )
        noise_samples = zero_mean_noise_samples + noise_mean

        a_samples = sys_model.system_equation(state_samples, noise_samples)

        self._check_predicted_state_samples(a_samples, num_samples)

        mean, cov = utils.get_sample_mean_and_covariance(a_samples, weights)
        return mean, cov

    def predict_additive_noise_system_model(
        self, sys_model: AdditiveNoiseSystemModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise_mean, noise_cov, _ = sys_model.noise.get_mean_and_covariance()
        dim_noise = noise_cov.shape[0]

        self._check_additive_system_noise(dim_noise)

        (
            std_normal_samples,
            weights,
            num_samples,
        ) = self.get_std_normal_samples_prediction(self.dim_state)

        zero_mean_state_samples = self.state_cov_sqrt.dot(std_normal_samples)
        state_samples = zero_mean_state_samples + self.state_mean

        a_samples = sys_model.system_equation(state_samples)

        self._check_predicted_state_samples(a_samples, num_samples)

        mean, cov = utils.get_sample_mean_and_covariance(a_samples, weights)

        predicted_state_mean = mean + noise_mean
        predicted_state_cov = cov + noise_cov
        return predicted_state_mean, predicted_state_cov

    def evaluate_additive_noise_meas_model(
        self,
        meas_model: AdditiveNoiseMeasurementModel,
        dim_meas: int,
        state_mean: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # this function does not use dimState, state_mean and state_cov_sqrt from the class!
        dim_state = state_cov_sqrt.shape[0]

        (
            std_normal_samples,
            weights,
            num_samples,
        ) = self.get_std_normal_samples_prediction(dim_state)

        zero_mean_state_samples = state_cov_sqrt.dot(std_normal_samples)
        state_samples = zero_mean_state_samples + np.array(state_mean)

        h_samples = meas_model.measurement_equation(state_samples)

        self._check_computed_measurements(h_samples, dim_meas, num_samples)
        return h_samples, weights, zero_mean_state_samples

    def evaluate_meas_model_uncorrelated(
        self,
        meas_model: Union[MeasurementModel, MixedNoiseMeasurementModel],
        dim_meas: int,
        state_mean: np.ndarray,
        state_cov_sqrt: np.ndarray,
        noise_mean: np.ndarray,
        noise_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # this function does not use dim_state, state_mean and state_cov_sqrt from the class!
        dim_state = state_cov_sqrt.shape[0]
        dim_noise = noise_cov_sqrt.shape[0]
        dim_aug_state = dim_state + dim_noise

        (
            std_normal_samples,
            weights,
            num_samples,
        ) = self.get_std_normal_samples_prediction(dim_aug_state)

        zero_mean_state_samples = state_cov_sqrt.dot(std_normal_samples[0:dim_state, :])
        state_samples = zero_mean_state_samples + state_mean

        zero_mean_noise_samples = noise_cov_sqrt.dot(std_normal_samples[dim_state:, :])
        noise_samples = zero_mean_noise_samples + noise_mean

        h_samples = meas_model.measurement_equation(state_samples, noise_samples)

        self._check_computed_measurements(h_samples, dim_meas, num_samples)
        return h_samples, weights, zero_mean_state_samples, zero_mean_noise_samples

    def evaluate_meas_model_correlated(
        self,
        meas_model,
        dim_meas,
        state_mean,
        state_cov,
        noise_mean,
        noise_cov,
        state_noise_cross_cov,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # todo untested
        # this function does not use dim_state, state_mean and state_cov_sqrt from the class!
        # this function is used if measurement noise and state are correlated
        dim_state = state_cov.shape[0]
        dim_noise = noise_cov.shape[0]
        dim_aug_state = dim_state + dim_noise

        state_noise_cov = np.block(
            [[state_cov, state_noise_cross_cov], [state_noise_cross_cov.T, noise_cov]]
        )

        state_noise_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
            state_noise_cov, "Joint system state and measurement noise"
        )

        (
            std_normal_samples,
            weights,
            num_samples,
        ) = self.get_std_normal_samples_prediction(dim_aug_state)

        zero_mean_state_samples = state_noise_cov_sqrt[0:dim_state, :].dot(
            std_normal_samples[0:dim_state, :]
        )
        state_samples = zero_mean_state_samples + state_mean

        zero_mean_noise_samples = state_noise_cov_sqrt[dim_state:, :].dot(
            std_normal_samples[dim_state, :]
        )
        noise_samples = zero_mean_noise_samples + noise_mean

        h_samples = meas_model.measurement_equation(state_samples, noise_samples)

        self._check_computed_measurements(h_samples, dim_meas, num_samples)

        return (
            h_samples,
            weights,
            zero_mean_state_samples,
            zero_mean_noise_samples,
            state_noise_cov,
        )

    def _adjust_sample_signs(
        self, weights: np.ndarray, samples: np.ndarray
    ) -> np.ndarray:
        """
        Returns negated samples where weights are negative

        :param weights: 1 x N weight array
        :param samples: dim_state x N state/noise sample array
        :return: Adjusted samples
        """
        return np.where(weights < 0, -samples, samples)

    def get_meas_model_moments(
        self,
        h_samples: np.ndarray,
        weights: np.ndarray,
        zero_mean_state_samples: np.ndarray,
        zero_mean_noise_samples: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = h_samples.shape[1]

        if np.isscalar(weights):
            weights = np.ones((1, num_samples)) * weights
        else:
            if not checks.is_array_of_vecs(weights, dim=1, num=num_samples):
                raise ValueError("weights must be a vector with one element per sample")

        # samples have different weights
        h_mean = h_samples.dot(weights.T)
        zero_mean_h_samples = h_samples - h_mean

        # Handle negative weights later in covariance calculation
        sqrt_weights = np.sqrt(np.abs(weights))
        weighted_zero_mean_h_samples = zero_mean_h_samples * sqrt_weights
        weighted_zero_mean_state_samples = zero_mean_state_samples * sqrt_weights

        # Add negative sign to weighted samples with negative weights
        h_cov = weighted_zero_mean_h_samples.dot(
            self._adjust_sample_signs(weights, weighted_zero_mean_h_samples).T
        )

        state_h_cross_cov = weighted_zero_mean_state_samples.dot(
            self._adjust_sample_signs(weights, weighted_zero_mean_h_samples).T
        )

        if zero_mean_noise_samples is not None:
            weighted_zero_mean_noise_samples = zero_mean_noise_samples * sqrt_weights
            h_noise_cross_cov = weighted_zero_mean_h_samples.dot(
                self._adjust_sample_signs(weights, weighted_zero_mean_noise_samples).T
            )
        else:
            h_noise_cross_cov = None

        return h_mean, h_cov, state_h_cross_cov, h_noise_cross_cov

    def get_std_normal_samples_prediction(self, num_samples: int):
        raise NotImplementedError
