from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.distributions import (
    DiracMixture,
    Distribution,
    Gaussian,
)
from nonlinear_estimation_toolbox.filters.particle_filter import ParticleFilter
from nonlinear_estimation_toolbox.measurement_models import Likelihood
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


class SIRPF(ParticleFilter):
    """The sampling importance resampling particle filter (SIRPF).

    Literature:
        Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
        Beyond the Kalman Filter: Particle filters for Tracking Applications,
        Artech House Publishers, 2004.
    """

    def __init__(self, name: str = "SIRPF"):
        super(SIRPF, self).__init__(name)

        self.num_particles = 1000
        self.particles = None
        self.weights = None
        self.min_allowed_normalized_ess = 0.5

    def get_state(self) -> DiracMixture:
        return DiracMixture(self.particles, self.weights)

    def get_state_mean_and_covariance(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_mean, state_cov = utils.get_sample_mean_and_covariance(
            self.particles, self.weights
        )
        state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(state_cov)
        return state_mean, state_cov, state_cov_sqrt

    def set_num_particles(self, num_particles: int) -> None:
        """Set the number of particles used by the filter.

        By default, 1000 particles are used.
        :param num_particles: The number of particles used by the filter.
        :return:
        """
        assert (
            isinstance(num_particles, int) and num_particles > 0
        ), "num particles must be a positive integer."

        if self.particles is None:
            self.num_particles = num_particles
        else:
            self._resample(num_particles)

    def get_num_particles(self) -> int:
        """Get the number of particles used by the filter.

        :return: The number of particles used by the filter.
        """
        return self.num_particles

    def set_min_allowed_normalized_ess(self, min_allowed_normalized_ess: float) -> None:
        """Set the minimum allowed normalized effective sample size (ESS).

        Determines how often a resampling will be performed.
        By default, a normalized ESS of 0.5 will be used.

        :param min_allowed_normalized_ess: The minimum allowed noramlized ESS between 0 and 1.
        """
        assert checks.is_scalar(
            min_allowed_normalized_ess
        ), "min_allowed_normalized_ess must be a scalar in [0, 1]"
        assert (
            0 <= min_allowed_normalized_ess <= 1
        ), "min_allowed_normalized_ess must be a scalar in [0, 1]"

        self.min_allowed_normalized_ess = min_allowed_normalized_ess

    def get_min_allowed_normalized_ess(self) -> float:
        """Get the minimum allowed normalized effective sample size (ESS).

        :return: The minimum allowed normalized ESS between 0 and 1.
        """
        return self.min_allowed_normalized_ess

    def _perform_set_state(self, state: Distribution) -> None:
        if isinstance(state, DiracMixture):
            self.particles, self.weights = state.get_components()
            self.num_particles = state.get_num_components()
        else:
            self.particles = state.draw_random_samples(self.num_particles)
            self.weights = np.ones((1, self.num_particles)) / self.num_particles

    def _perform_set_state_mean_and_covariance(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> None:
        self._perform_set_state(Gaussian(mean=state_mean, covariance=state_cov))

    def _perform_prediction(self, sys_model: SystemModel) -> None:
        """
        Implements the prediction described in:
            Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
            Beyond the Kalman Filter: Particle filters for Tracking Applications,
            Artech House Publishers, 2004,
            Section 3.5.1
        """
        self._resample_by_ess()
        self.particles = self._predict_particles(
            sys_model, self.particles, self.num_particles
        )

    def _perform_update(self, meas_model: Likelihood, measurement: np.ndarray) -> None:
        if isinstance(meas_model, Likelihood):
            self._update_likelihood(meas_model, measurement)
        else:
            raise ValueError("meas_model must be of type Likelihood")

    def _update_likelihood(
        self, meas_model: Likelihood, measurement: ColumnVectors
    ) -> None:
        """
        Implements the measurement update described in:
            Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
            Beyond the Kalman Filter: Particle filters for Tracking Applications,
            Artech House Publishers, 2004,
            Section 3.5.1
        """
        self._resample_by_ess()

        values = self._evaluate_likelihood(
            meas_model, measurement, self.particles, self.num_particles
        )
        values = values * self.weights
        sum_weights = np.sum(values)
        assert (
            sum_weights >= 0
        ), "Sum of computed posterior particle weights is not positive."
        self.weights = values / sum_weights

    def _resample(self, num_particles: int = None) -> None:
        if num_particles is None:
            num_particles = self.num_particles

        # Assumption: Particle weights are always normalized
        self.particles, _ = utils.systematic_resampling(
            self.particles, self.weights, num_particles
        )
        self.num_particles = num_particles
        self.weights = np.ones((1, num_particles)) / num_particles

    def _resample_by_ess(self) -> None:
        normalized_ess = self._get_normalized_ess()

        if normalized_ess < self.min_allowed_normalized_ess:
            self._resample()

    def _get_normalized_ess(self) -> float:
        return 1 / (self.num_particles * np.sum(self.weights**2, axis=1))
