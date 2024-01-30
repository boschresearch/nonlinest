from typing import Union

import numpy as np

from nonlinear_estimation_toolbox.filters.filter import Filter
from nonlinear_estimation_toolbox.measurement_models import Likelihood
from nonlinear_estimation_toolbox.system_models import (
    AdditiveNoiseSystemModel,
    MixedNoiseSystemModel,
    SystemModel,
)
from nonlinear_estimation_toolbox.utils import ColumnVectors


class ParticleFilter(Filter):
    """Abstract base class for particle filters."""

    def _predict_particles(
        self,
        sys_model: Union[SystemModel, AdditiveNoiseSystemModel, MixedNoiseSystemModel],
        particles: ColumnVectors,
        num_particles: int,
    ) -> ColumnVectors:
        """Helper function to take care of different system model types."""

        if isinstance(sys_model, SystemModel):
            return self._predict_particles_arbitrary_noise(
                sys_model, particles, num_particles
            )
        elif isinstance(sys_model, AdditiveNoiseSystemModel):
            return self._predict_particles_additive_noise(
                sys_model, particles, num_particles
            )
        elif isinstance(sys_model, MixedNoiseSystemModel):
            return self._predict_particles_mixed_noise(
                sys_model, particles, num_particles
            )
        else:
            raise TypeError(
                "sys_model must be of type SystemModel, AdditiveNoiseSystemModel, or MixedNoiseSystemModel"
            )

    def _predict_particles_arbitrary_noise(
        self, sys_model: SystemModel, particles: ColumnVectors, num_particles: int
    ) -> ColumnVectors:
        noise = sys_model.noise.draw_random_samples(num_particles)
        predicted_particles = sys_model.system_equation(particles, noise)
        self._check_predicted_state_samples(predicted_particles, num_particles)

        return predicted_particles

    def _predict_particles_additive_noise(
        self,
        sys_model: AdditiveNoiseSystemModel,
        particles: ColumnVectors,
        num_particles: int,
    ) -> ColumnVectors:
        noise = sys_model.noise.draw_random_samples(num_particles)
        dim_noise = noise.shape[0]
        self._check_additive_system_noise(dim_noise)

        predicted_particles = sys_model.system_equation(particles)
        self._check_predicted_state_samples(predicted_particles, num_particles)

        predicted_particles = predicted_particles + noise
        return predicted_particles

    def _predict_particles_mixed_noise(
        self,
        sys_model: MixedNoiseSystemModel,
        particles: ColumnVectors,
        num_particles: int,
    ) -> ColumnVectors:
        noise = sys_model.noise.draw_random_samples(num_particles)
        add_noise = sys_model.additive_noise.draw_random_samples(num_particles)
        dim_add_noise = add_noise.shape[0]
        self._check_additive_system_noise(dim_add_noise)

        predicted_particles = sys_model.system_equation(particles, noise)
        self._check_predicted_state_samples(predicted_particles, num_particles)

        predicted_particles = predicted_particles + add_noise
        return predicted_particles

    def _evaluate_likelihood(
        self,
        meas_model: Likelihood,
        measurement: ColumnVectors,
        particles: ColumnVectors,
        num_particles: int,
    ):
        log_values = meas_model.log_likelihood(particles, measurement)
        self._check_log_likelihood_evaluations(log_values, num_particles)

        max_log_value = np.max(log_values)
        log_values = log_values - max_log_value
        values = np.exp(log_values)
        return values
