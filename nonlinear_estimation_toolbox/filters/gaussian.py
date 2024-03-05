from typing import Callable, Tuple, Union

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.filter import Filter
from nonlinear_estimation_toolbox.measurement_models import MeasurementModel
from nonlinear_estimation_toolbox.system_models import (
    AdditiveNoiseSystemModel,
    LinearSystemModel,
    MixedNoiseSystemModel,
    SystemModel,
)
from nonlinear_estimation_toolbox.utils import ColumnVectors


class GaussianFilter(Filter):
    """
    Toolbox/Interfaces/Filters/GaussianFilter.m
    """

    def __init__(self, name: str = ""):
        super(GaussianFilter, self).__init__(name)
        self.stateDecompDim = 0

        self.state_mean = None
        self.state_cov = None
        self.state_cov_sqrt = None

        # By default, no post-processing is enabled
        self.prediction_post_processing = None
        self.update_post_processing = None

    def get_state(self) -> Gaussian:
        return Gaussian(self.state_mean, self.state_cov)

    def get_state_mean_and_covariance(self) -> Tuple[np.array, np.array, np.array]:
        return self.state_mean, self.state_cov, self.state_cov_sqrt

    def set_state_decomposition_dimension(self, dim: int):
        if not np.isscalar(dim):
            raise TypeError("InvalidDimension: dim must be a scalar.")
        if not dim >= 0:
            raise ValueError("InvalidDimension: dim must be a non-negative scalar.")

        self.stateDecompDim = np.ceil(dim)

    def get_state_decomposition_dimension(self) -> int:
        return self.stateDecompDim

    def set_prediction_post_processing(self, prediction_post_processing: Callable):
        # TODO: checks
        self.prediction_post_processing = prediction_post_processing

    def get_prediction_post_processing(self) -> Callable:
        return self.prediction_post_processing

    def set_update_post_processing(self, update_post_processing: Callable):
        # TODO: checks
        self.update_post_processing = update_post_processing

    def get_update_post_processing(self) -> Callable:
        return self.update_post_processing

    def _perform_set_state(self, state: Gaussian):
        (
            self.state_mean,
            self.state_cov,
            self.state_cov_sqrt,
        ) = state.get_mean_and_covariance()

    def _perform_set_state_mean_and_covariance(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ):
        self.state_mean, self.state_cov, self.state_cov_sqrt = (
            state_mean,
            state_cov,
            state_cov_sqrt,
        )

    def _perform_prediction(self, sys_model: SystemModel):
        # TODO: checks
        if isinstance(sys_model, LinearSystemModel):
            (
                predicted_state_mean,
                predicted_state_cov,
            ) = self.predict_linear_system_model(sys_model)
        elif isinstance(sys_model, SystemModel):
            predicted_state_mean, predicted_state_cov = self.predict_system_model(
                sys_model
            )
        elif isinstance(sys_model, AdditiveNoiseSystemModel):
            (
                predicted_state_mean,
                predicted_state_cov,
            ) = self.predict_additive_noise_system_model(sys_model)
        elif isinstance(sys_model, MixedNoiseSystemModel):
            (
                predicted_state_mean,
                predicted_state_cov,
            ) = self.predict_mixed_noise_system_model(sys_model)
        else:
            raise TypeError("GaussianFilter: invalid system model")

        self._check_and_save_prediction(predicted_state_mean, predicted_state_cov)

    def _perform_update(self, meas_model: MeasurementModel, measurement: np.ndarray):
        observable_state_dim = self.get_observable_state_dim()

        if observable_state_dim < self.dim_state:
            # Extract observable part of the system state
            mean = self.state_mean[:observable_state_dim, :]
            cov = self.state_cov[:observable_state_dim, :observable_state_dim]
            cov_sqrt = self.state_cov_sqrt[:observable_state_dim, :observable_state_dim]
            # Update observable state variables
            updated_mean, updated_cov = self.perform_update_observable_state(
                meas_model, measurement, mean, cov, cov_sqrt
            )

            # Check if updated observable state covariance is valid
            updated_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
                updated_cov, "Updated observable state"
            )

            # Update entire system state
            updated_state_mean, updated_state_cov = utils.decomposed_state_update(
                self.state_mean,
                self.state_cov,
                self.state_cov_sqrt,
                updated_mean,
                updated_cov,
                updated_cov_sqrt,
            )
        else:
            # Update entire system state
            (
                updated_state_mean,
                updated_state_cov,
            ) = self.perform_update_observable_state(
                meas_model,
                measurement,
                self.state_mean,
                self.state_cov,
                self.state_cov_sqrt,
            )
        self._check_and_save_update(updated_state_mean, updated_state_cov)

    def _check_and_save_prediction(
        self, predicted_state_mean: np.ndarray, predicted_state_cov: np.ndarray
    ):
        # Check if predicted state covariance is valid
        predicted_state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
            predicted_state_cov, "Predicted state"
        )

        if self.prediction_post_processing is not None:
            # Perform post-processing
            predicted_state_mean, predicted_state_cov = self.prediction_post_processing(
                predicted_state_mean, predicted_state_cov, predicted_state_cov_sqrt
            )

            # TODO: implement checks

            # Check if post-processed state covariance is valid
            predicted_state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
                predicted_state_cov, "Post-processed predicted state"
            )

        # Save predicted state estimate
        self.state_mean = predicted_state_mean
        self.state_cov = predicted_state_cov
        self.state_cov_sqrt = predicted_state_cov_sqrt

    def _check_and_save_update(
        self, updated_state_mean: np.ndarray, updated_state_cov: np.ndarray
    ):
        # Check if updated state covariance is valid
        updated_state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
            updated_state_cov, "Updated state"
        )

        if self.update_post_processing is not None:
            # Perform post-processing
            updated_state_mean, updated_state_cov = self.update_post_processing(
                updated_state_mean, updated_state_cov, updated_state_cov_sqrt
            )

            # TODO: implement checks

            # Check if post-processed state covariance is valid

            updated_state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
                updated_state_cov, "Post-processed updated state"
            )

        # Save updated state estimate
        self.state_mean = updated_state_mean
        self.state_cov = updated_state_cov
        self.state_cov_sqrt = updated_state_cov_sqrt

    def predict_linear_system_model(
        self, sys_model: LinearSystemModel
    ):  # TODO: output type hint
        return sys_model.analytic_moments(
            self.state_mean, self.state_cov, self.state_cov_sqrt
        )  # TODO: resolve sys_model not having analyticModels

    def predict_mixed_noise_system_model(
        self, sys_model: MixedNoiseSystemModel
    ) -> Tuple[np.array, np.array]:
        (
            additive_noise_mean,
            additive_noise_cov,
            _,
        ) = (
            sys_model.additive_noise.get_mean_and_covariance()
        )  # TODO: resolve sys_model not having additiveNoise
        dim_additive_noise = additive_noise_mean.shape[0]

        self._check_additive_system_noise(dim_additive_noise)

        mean, cov = self.predict_system_model(sys_model)

        # Compute predicted state mean
        predicted_state_mean = mean + additive_noise_mean

        # Compute predicted state covariance
        predicted_state_cov = cov + additive_noise_cov

        return predicted_state_mean, predicted_state_cov

    def predict_system_model(
        self, sys_model: Union[SystemModel, MixedNoiseSystemModel]
    ):
        raise NotImplementedError

    def predict_additive_noise_system_model(self, sys_model: AdditiveNoiseSystemModel):
        raise NotImplementedError

    # todo: rename to _perform_update_observable_state, also ColumnVectors consistency
    def perform_update_observable_state(
        self,
        meas_model: MeasurementModel,
        measurement: np.ndarray,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ):
        raise NotImplementedError

    def get_observable_state_dim(self) -> int:
        observable_state_dim = self.dim_state - self.stateDecompDim

        # At least one variable of the system state must be "observable"
        if observable_state_dim <= 0:
            raise ValueError(
                'InvalidUnobservableStateDimension: At least one variable of the system state must be "observable".'
            )

        return int(observable_state_dim)
