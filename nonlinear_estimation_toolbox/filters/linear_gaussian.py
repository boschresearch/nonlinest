from typing import Tuple

import numpy as np
from scipy.stats import chi2

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.filters.gaussian import GaussianFilter
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    LinearMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)


class LinearGaussianFilter(GaussianFilter):
    # TODO: Add doc string
    """
    Toolbox/Interfaces/Filters/LinearGaussianFilter.m
    """

    def __init__(self, name: str = ""):
        super(LinearGaussianFilter, self).__init__(name)
        self.measurement_gating_threshold = 1

    def set_measurement_gating_threshold(self, threshold: float):
        assert checks.is_pos_scalar(threshold) and "Threshold must be a positive scalar"
        self.measurement_gating_threshold = threshold

    def get_measurement_gating_threshold(self) -> float:
        return self.measurement_gating_threshold

    def perform_update_observable_state(
        self,
        meas_model: MeasurementModel,
        measurement: np.ndarray,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: is measurement always one measurement or do we accept multiple meaurements
        self._check_measurement_vector(measurement)

        dim_meas = len(measurement)

        if isinstance(meas_model, LinearMeasurementModel):
            (
                updated_state_mean,
                updated_state_cov,
            ) = self.update_with_linear_measurement_model(
                meas_model,
                measurement,
                prior_state_mean,
                prior_state_cov,
                prior_state_cov_sqrt,
            )
        else:
            if isinstance(meas_model, MeasurementModel):
                self.setup_measurement_model(meas_model, dim_meas)
            elif isinstance(meas_model, AdditiveNoiseMeasurementModel):
                self.setup_additive_noise_measurement_model(meas_model, dim_meas)
            elif isinstance(meas_model, MixedNoiseMeasurementModel):
                self.setup_mixed_noise_measurement_model(meas_model, dim_meas)
            else:
                raise ValueError(
                    "Measurement model must be of type LinearMeasurementModel, MeasurementModel, AdditiveNoiseMeausrementModel, or MixedNoiseMeasurementModel"
                )

            updated_state_mean, updated_state_cov = self.update_nonlinear(
                measurement, prior_state_mean, prior_state_cov, prior_state_cov_sqrt
            )
        return updated_state_mean, updated_state_cov

    def update_with_linear_measurement_model(
        self,
        meas_model: LinearMeasurementModel,
        measurement: np.ndarray,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        meas_mean, meas_cov, state_meas_cross_cov = meas_model.analytic_moments(
            prior_state_mean, prior_state_cov, prior_state_cov_sqrt
        )

        try:
            if self._is_measurement_gating_enabled():
                dim_meas = len(measurement)

                (
                    updated_state_mean,
                    updated_state_cov,
                    sq_meas_mahal_dist,
                ) = utils.kalman_update(
                    prior_state_mean,
                    prior_state_cov,
                    measurement,
                    meas_mean,
                    meas_cov,
                    state_meas_cross_cov,
                )

                self.do_measurement_gating(dim_meas, sq_meas_mahal_dist)
            else:
                updated_state_mean, updated_state_cov, _ = utils.kalman_update(
                    prior_state_mean,
                    prior_state_cov,
                    measurement,
                    meas_mean,
                    meas_cov,
                    state_meas_cross_cov,
                )
        except Exception:
            raise RuntimeError("Skipping linear measurement model update")
        return updated_state_mean, updated_state_cov

    def _is_measurement_gating_enabled(self) -> bool:
        return self.measurement_gating_threshold != 1.0

    def do_measurement_gating(self, dim_meas: int, sq_meas_mahalanobis_distance: float):
        normalized_value = chi2.cdf(sq_meas_mahalanobis_distance, dim_meas)

        if normalized_value > self.measurement_gating_threshold:
            # TODO: Implement discarding mechanism
            raise ValueError("Measurement will be discarded")

    def setup_measurement_model(self, meas_model: MeasurementModel, dim_meas: int):
        raise NotImplementedError

    def setup_additive_noise_measurement_model(
        self, meas_model: AdditiveNoiseMeasurementModel, dim_meas: int
    ):
        raise NotImplementedError

    def setup_mixed_noise_measurement_model(
        self, meas_model: MixedNoiseMeasurementModel, dim_meas: int
    ):
        raise NotImplementedError

    def update_nonlinear(
        self,
        measurement: np.ndarray,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
