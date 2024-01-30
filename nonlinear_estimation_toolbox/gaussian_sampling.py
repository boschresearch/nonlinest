import numpy as np

import nonlinear_estimation_toolbox.checks as checks
import nonlinear_estimation_toolbox.distributions as distributions
import nonlinear_estimation_toolbox.utils as utils
from nonlinear_estimation_toolbox import sample_cache


class GaussianSampling:
    def get_samples(
        self, gaussian: distributions.Gaussian
    ) -> (np.ndarray, np.ndarray, int):
        mean, _, cov_sqrt = gaussian.get_mean_and_covariance()
        dim = gaussian.get_dimension()

        std_normal_samples, weights, num_samples = self.get_std_normal_samples(dim)

        samples = cov_sqrt.dot(std_normal_samples)
        samples = samples + mean
        return samples, weights, num_samples

    def get_std_normal_samples(self, dim: int):
        raise NotImplementedError


class GaussianSamplingUKF(GaussianSampling):
    def __init__(self):
        self.scaling = 0.5

    def set_sample_scaling(self, scaling) -> None:
        if not checks.is_scalar(scaling):
            raise ValueError("InvalidScaling: scaling must be a scalar.")
        self.scaling = scaling

    def get_sample_scaling(self) -> float:
        return self.scaling

    def get_std_normal_samples(self, dimension: int) -> (np.ndarray, np.ndarray, int):
        if not checks.is_pos_scalar(dimension):
            raise ValueError("InvalidDimension: dimension must be a positive scalar.")

        if dimension + self.scaling <= 0:
            raise ValueError(
                "InvalidScaling: The sample scaling factor is too small for the requested dimension."
            )

        num_samples = 2 * dimension + 1

        mat = np.sqrt(dimension + self.scaling) * np.eye(dimension)

        samples = np.hstack([np.zeros([dimension, 1]), mat, -mat])

        if self.scaling == 0.5:
            weights = 1 / num_samples
        else:
            weights = np.hstack(
                [
                    utils.ColumnVectors(2 * self.scaling),
                    np.ones([1, num_samples - 1]),
                ]
            ) / (2 * (dimension + self.scaling))

        return samples, weights, num_samples


class GaussianSamplingRUKF(GaussianSampling):
    #   Jindrich Dunik, Ondrej Straka, and Miroslav Simandl,
    #   The Development of a Randomised Unscented Kalman Filter,
    #   Proceedings of the 18th IFAC World Congress, Milano, Italy, pp. 8-13, Aug. 2011.

    def __init__(self):
        self.num_iterations = 5

    def set_numiterations(self, num_interations: int):
        if not checks.is_pos_scalar(num_interations):
            raise ValueError("Number of iterations must be a positive scalar")

        self.num_iterations = num_interations

    def get_numiterations(self):
        return self.num_iterations

    def get_std_normal_samples(self, dim: int) -> (np.ndarray, np.ndarray, int):
        if not checks.is_pos_scalar(dim):
            raise ValueError("Dimension must be a positive scalar")

        num_samples = 2 * dim * self.num_iterations + 1

        samples = np.zeros([dim, num_samples])
        weights = np.zeros([1, num_samples])

        step = 2 * dim - 1  # number of samples added in each iteration
        a = 1  # start index of current iteration

        for i in range(self.num_iterations):
            U = utils.rnd_orthogonal_matrix(dim)

            scaling = np.sqrt(np.random.chisquare(2 + dim))
            b = a + step + 1  # end index of current iteration
            samples[:, a:b] = scaling * np.hstack([U, -U])
            weights[0, 0] = weights[0, 0] + (1 - dim / (scaling * scaling))
            weights[0, a:b] = 1 / (2 * scaling * scaling) * np.ones([1, 2 * dim])
            a = b

        weights = weights / self.num_iterations
        return samples, weights, num_samples


class GaussianSamplingCKF(GaussianSampling):
    def __init__(self):
        self.sample_cache = sample_cache.SampleCacheCKF()

    def get_std_normal_samples(self, dim: int) -> [np.ndarray, np.ndarray, int]:
        if not checks.is_pos_scalar(dim):
            raise ValueError("Dimension must be a positive scalar")

        num_samples = 2 * dim**2 + 1
        samples, weights = self.sample_cache.get_samples(dim, num_samples)
        return samples, weights, num_samples


class GaussianSamplingGHQ(GaussianSampling):
    #   Kazufumi Ito and Kaiqi Xiong,
    #   Gaussian Filters for Nonlinear Filtering Problems,
    #   IEEE Transactions on Automatic Control, vol. 45, no. 5, pp. 910-927, May 2000.

    def __init__(self):
        self.sample_cache = sample_cache.SampleCacheGHQ()
        self.sample_cache.set_num_quadrature_points(2)

    def set_num_quadrature_points(self, num_points: int):
        self.sample_cache.set_num_quadrature_points(num_points)

    def get_num_quadature_points(self) -> int:
        return self.sample_cache.get_num_quadrature_points()

    def get_std_normal_samples(self, dim: int) -> [np.ndarray, np.ndarray, int]:
        if not checks.is_pos_scalar(dim):
            raise ValueError("Dimension must be a positive scalar")

        num_points = self.sample_cache.get_num_quadrature_points()
        num_samples = num_points**dim
        samples, weights = self.sample_cache.get_samples(dim, num_samples)
        return samples, weights, num_samples
