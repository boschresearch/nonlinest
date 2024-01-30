from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.distributions import (
    DiracMixture,
    Distribution,
    Gaussian,
)
from nonlinear_estimation_toolbox.filters.particle_filter import ParticleFilter
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


class EnKF(ParticleFilter):
    """The ensemble Kalman filter (EnKF).

    Literature:
        S. Gillijns, O. Barrero Mendoza, J. Chandrasekar, B. L. R. De Moor, D. S. Bernstein, and A. Ridley,
        What Is the Ensemble Kalman Filter and How Well Does It Work?,
        Proceedings of the 2006 American Control Conference (ACC 2006), Minneapolis, USA, Jun. 2006.
    """

    def __init__(self, name="EnKF"):
        super().__init__(name)
        self.ensemble = None
        self.ensemble_size = 1000

    def set_ensemble_size(self, ensemble_size: int) -> None:
        """Set the size of the ensemble (i.e., the number of samples).

        By default, 1000 ensemble members will be used.

        :param ensemble_size: The number of ensemble members used by the filter.
        """
        assert (
            isinstance(ensemble_size, int) and ensemble_size > 0
        ), "ensemble_size must be a positive integer."

        if self.ensemble is None:
            self.ensemble_size = ensemble_size
        else:
            self._resample(ensemble_size)

    def get_ensemble_size(self) -> int:
        """Get the ensemble size of the filter.

        :return: The ensemble size used by the filter.
        """
        return self.ensemble_size

    def get_state(self) -> DiracMixture:
        return DiracMixture(self.ensemble)

    def get_state_mean_and_covariance(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_mean, state_cov = utils.get_sample_mean_and_covariance(self.ensemble)
        state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(state_cov)
        return state_mean, state_cov, state_cov_sqrt

    def _perform_set_state(self, state: Distribution) -> None:
        self.ensemble = state.draw_random_samples(self.ensemble_size)

    def _perform_set_state_mean_and_covariance(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> None:
        self._perform_set_state(Gaussian(mean=state_mean, covariance=state_cov))

    def _perform_prediction(self, sys_model: SystemModel) -> None:
        self.ensemble = self._predict_particles(
            sys_model, self.ensemble, self.ensemble_size
        )

    def _perform_update(
        self, meas_model: MeasurementModel, measurement: ColumnVectors
    ) -> None:
        self._check_measurement_vector(measurement)

        if isinstance(meas_model, MeasurementModel):
            self._update_arbitrary_noise(meas_model, measurement)
        elif isinstance(meas_model, AdditiveNoiseMeasurementModel):
            self._update_additive_noise(meas_model, measurement)
        elif isinstance(meas_model, MixedNoiseMeasurementModel):
            self._update_mixed_noise(meas_model, measurement)
        else:
            raise TypeError(
                "meas_model must be of type MeasurementModel, AdditiveNoiseMeasurementModel, or MixedNoiseMeasurementModel"
            )

    def _update_arbitrary_noise(
        self, meas_model: MeasurementModel, measurement: ColumnVectors
    ) -> None:
        dim_meas = measurement.shape[0]
        noise_samples = meas_model.noise.draw_random_samples(self.ensemble_size)
        meas_samples = meas_model.measurement_equation(self.ensemble, noise_samples)
        self._check_computed_measurements(meas_samples, dim_meas, self.ensemble_size)

        self._update_ensemble(measurement, meas_samples)

    def _update_additive_noise(
        self, meas_model: AdditiveNoiseMeasurementModel, measurement: ColumnVectors
    ) -> None:
        dim_noise = meas_model.noise.get_dimension()
        dim_meas = measurement.shape[0]
        self._check_additive_measurement_noise(dim_meas, dim_noise)

        meas = meas_model.measurement_equation(self.ensemble)
        self._check_computed_measurements(meas, dim_meas, self.ensemble_size)

        add_noise_samples = meas_model.noise.draw_random_samples(self.ensemble_size)
        meas_samples = meas + add_noise_samples

        self._update_ensemble(measurement, meas_samples)

    def _update_mixed_noise(
        self, meas_model: MixedNoiseMeasurementModel, measurement: ColumnVectors
    ) -> None:
        dim_add_noise = meas_model.additive_noise.get_dimension()
        dim_meas = measurement.shape[0]
        self._check_additive_measurement_noise(dim_meas, dim_add_noise)

        noise_samples = meas_model.noise.draw_random_samples(self.ensemble_size)
        meas = meas_model.measurement_equation(self.ensemble, noise_samples)
        self._check_computed_measurements(meas, dim_meas, self.ensemble_size)

        add_noise_samples = meas_model.additive_noise.draw_random_samples(
            self.ensemble_size
        )
        meas_samples = meas + add_noise_samples

        self._update_ensemble(measurement, meas_samples)

    def _update_ensemble(
        self, measurement: ColumnVectors, meas_samples: ColumnVectors
    ) -> None:
        meas_mean = np.sum(meas_samples, axis=1, keepdims=True) / self.ensemble_size
        zero_mean_meas_samples = meas_samples - meas_mean
        meas_cov = (
            zero_mean_meas_samples.dot(zero_mean_meas_samples.T) / self.ensemble_size
        )

        ensemble_mean = (
            np.sum(self.ensemble, axis=1, keepdims=True) / self.ensemble_size
        )
        zero_mean_ensemble = self.ensemble - ensemble_mean

        ensemble_meas_cross_cov = (
            zero_mean_ensemble.dot(zero_mean_meas_samples.T) / self.ensemble_size
        )
        checks.compute_cholesky_if_valid_covariance(meas_cov, "Measurement")

        kalman_gain = ensemble_meas_cross_cov.dot(np.linalg.inv(meas_cov))
        innovation = measurement - meas_samples

        self.ensemble = self.ensemble + kalman_gain.dot(innovation)

    def _resample(self, ensemble_size) -> None:
        idx = np.random.randint(
            low=0, high=self.ensemble_size - 1, size=self.ensemble_size, dtype=int
        )
        self.ensemble_size = ensemble_size
        self.ensemble = self.ensemble[:, idx]
