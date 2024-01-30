from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox import utils
from nonlinear_estimation_toolbox.distributions import DiracMixture, Gaussian
from nonlinear_estimation_toolbox.filters.gaussian import GaussianFilter
from nonlinear_estimation_toolbox.filters.particle_filter import ParticleFilter
from nonlinear_estimation_toolbox.measurement_models import Likelihood
from nonlinear_estimation_toolbox.system_models import (
    AdditiveNoiseSystemModel,
    SystemModel,
)
from nonlinear_estimation_toolbox.utils import ColumnVectors


class GPF(GaussianFilter, ParticleFilter):
    """The Gaussian particle filter (GPF).

    Literature:
        Jayesh H. Kotecha and Petar M. Djuric,
        Gaussian Particle Filtering,
        IEEE Transactions on Signal Processing, vol. 51, no. 10, pp. 2592-2601, Oct. 2003.
    """

    def __init__(self, name="GPF"):
        super(GPF, self).__init__(name)

        self.num_particles = 1000

    def set_num_particles(self, num_particles: int) -> None:
        """Set the number of particles used by the filter.

        By default, 1000 particles are used.
        :param num_particles: The number of particles used by the filter.
        :return:
        """
        assert (
            isinstance(num_particles, int) and num_particles > 0
        ), "num particles must be a positive integer."

        self.num_particles = num_particles

    def get_num_particles(self) -> int:
        """Get the number of particles used by the filter.

        :return: The number of particles used by the filter.
        """
        return self.num_particles

    def _predict_system_model(
        self, sys_model: SystemModel
    ) -> Tuple[ColumnVectors, np.ndarray]:
        noise = sys_model.noise.draw_random_samples(self.num_particles)
        particles = self.get_state().draw_random_samples(
            self.num_particles
        )  # Get state particles
        predicted_particles = sys_model.system_equation(particles, noise)
        self._check_predicted_state_samples(predicted_particles, self.num_particles)

        return utils.get_sample_mean_and_covariance(predicted_particles)

    def _predict_additive_noise_system_model(
        self, sys_model: AdditiveNoiseSystemModel
    ) -> Tuple[ColumnVectors, np.ndarray]:
        noise_mean, noise_cov, _ = sys_model.noise.get_mean_and_covariance()
        dim_noise = noise_mean.shape[0]
        self._check_additive_system_noise(dim_noise)

        particles = self.get_state().draw_random_samples(
            self.num_particles
        )  # Get state particles
        predicted_particles = sys_model.system_equation(particles)
        self._check_predicted_state_samples(predicted_particles, self.num_particles)

        mean, cov = utils.get_sample_mean_and_covariance(predicted_particles)
        predicted_state_mean = mean + noise_mean
        predicted_state_cov = cov + noise_cov

        return predicted_state_mean, predicted_state_cov

    def perform_update_observable_state(
        self,
        meas_model: Likelihood,
        measurement: ColumnVectors,
        prior_state_mean: ColumnVectors,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[ColumnVectors, np.ndarray]:
        if isinstance(meas_model, Likelihood):
            prior_state = Gaussian(prior_state_mean, prior_state_cov)
            particles = prior_state.draw_random_samples(self.num_particles)

            updated_state_mean, updated_state_cov = self._update_likelihood(
                meas_model, measurement, particles
            )
            return updated_state_mean, updated_state_cov
        else:
            raise ValueError("meas_model must be of type Likelihood")

    def _update_likelihood(
        self,
        meas_model: Likelihood,
        measurement: ColumnVectors,
        particles: ColumnVectors,
    ) -> Tuple[ColumnVectors, np.ndarray]:
        values = self._evaluate_likelihood(
            meas_model, measurement, particles, self.num_particles
        )
        sum_weights = np.sum(values)

        assert (
            sum_weights >= 0
        ), "Sum of computed posterior particle weights is not positive."
        weights = values / sum_weights

        updated_state = DiracMixture(samples=particles, weights=weights)
        (
            updated_state_mean,
            updated_state_cov,
            _,
        ) = updated_state.get_mean_and_covariance()
        return updated_state_mean, updated_state_cov
