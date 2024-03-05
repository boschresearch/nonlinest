import numpy as np

from nonlinear_estimation_toolbox import utils
from nonlinear_estimation_toolbox.filters.sirpf import SIRPF
from nonlinear_estimation_toolbox.measurement_models import Likelihood
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


class ASIRPF(SIRPF):
    """
    The auxiliary sampling importance resampling particle filter (ASIRPF).

    This function implements the step() method (combined predict and update)
    to execute the auxiliary SIRPF. The update and predict functions can still
    be called seperately and are equal to the functions from SIRPF.

    According to Ristic et al. the ASIRPF is preferred if the system/process noise is small.

    Literature:
        Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
        Beyond the Kalman Filter: Particle filters for Tracking Applications,
        Artech House Publishers, 2004.
    """

    def _perform_step(
        self,
        sys_model: SystemModel,
        meas_model: Likelihood,
        measurement: ColumnVectors,
    ) -> None:
        self._resample_by_ess()
        predicted_particles = self._predict_particles(
            sys_model, self.particles, self.num_particles
        )

        pred_log_values = meas_model.log_likelihood(predicted_particles, measurement)
        self._check_log_likelihood_evaluations(pred_log_values, self.num_particles)

        values = np.exp(pred_log_values)
        weights = self.weights * values
        sum_weights = np.sum(weights)
        assert sum_weights > 0, "Sum of computed particle weights is not positive."

        weights = weights / sum_weights

        # Resample, predict again using resampled particles
        particles, idx = utils.systematic_resampling(
            self.particles, weights, self.num_particles
        )
        predicted_particles = self._predict_particles(
            sys_model, particles, self.num_particles
        )

        log_values = meas_model.log_likelihood(predicted_particles, measurement)
        self._check_log_likelihood_evaluations(pred_log_values, self.num_particles)

        log_values = log_values - pred_log_values[:, idx]
        log_values = log_values - max(log_values)  # Numeric stability

        weights = np.exp(log_values)
        sum_weights = np.sum(weights)
        assert (
            sum_weights > 0
        ), "Sum of computed posterior particle weights is not positive."

        self.weights = weights / sum_weights
        self.particles = predicted_particles
