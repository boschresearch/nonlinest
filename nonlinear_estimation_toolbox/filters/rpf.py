import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.filters.sirpf import SIRPF


class RPF(SIRPF):
    """The regularized particle filter (RPF).

    This implementation uses Gaussian kernels for resampling.

    Literature:
        Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
        Beyond the Kalman Filter: Particle filters for Tracking Applications,
        Artech House Publishers, 2004.
    """

    def _resample(self, num_particles: int = None) -> None:
        if num_particles is None:
            num_particles = self.num_particles

        _, cov = utils.get_sample_mean_and_covariance(self.particles, self.weights)

        self.particles, _ = utils.systematic_resampling(
            self.particles, self.weights, num_particles
        )

        # Only use regularized PF resampling if sample covariance matrix is valid
        try:
            cov_sqrt = checks.compute_cholesky_if_valid_covariance(cov)
        except ValueError:
            pass
        else:
            # Optimal kernel bandwidth, using Gaussian kernel for regularization
            h_opt = (4 / (self.num_particles * (self.dim_state + 2))) ** (
                1 / (self.dim_state + 4)
            )
            self.particles = self.particles + (h_opt * cov_sqrt).dot(
                np.random.randn(self.dim_state, num_particles)
            )

        self.num_particles = num_particles
        self.weights = np.ones((1, num_particles)) / num_particles
