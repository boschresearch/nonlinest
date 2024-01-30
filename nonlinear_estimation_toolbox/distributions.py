import math
from typing import Tuple

import numpy as np

import nonlinear_estimation_toolbox.checks as checks
import nonlinear_estimation_toolbox.utils as utils
from nonlinear_estimation_toolbox.utils import ColumnVectors

"""
Abstract base class for all distributions
"""


class Distribution:
    def get_dimension(self):
        return self.dim

    """
    Obtain mean and covariance

    :returns : mean, covariance ands square root of covariance
    """

    def get_mean_and_covariance(self) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:
        raise NotImplementedError

    """
    Simulates random samples from the distribution.

    :param numSamples: number of samples to draw
    :returns: matrix of samples, one sample per column
    """

    def draw_random_samples(self, num_samples: int) -> ColumnVectors:
        raise NotImplementedError

    """
    Evaluates the logarithm of the probability density function at the given locations.

    :param values: where the evaluate the pdf
    :returns: logarithm of the pdf"""

    def log_pdf(self, values: ColumnVectors) -> ColumnVectors:
        raise NotImplementedError


"""
A Gaussian distribution of arbitrary dimension.
"""


class Gaussian(Distribution):
    def __init__(self, mean: ColumnVectors = None, covariance: np.ndarray = None):
        """
        Constructor

        :param mean: the mean as a vector (n,)
        :param covariance: the covariance as a matrix (n x n) (TODO also support vector for diagonal covariances?)
        """
        # if no mean/cov is given, we create a 1D standard Gaussian
        mean = ColumnVectors(np.array([0])) if mean is None else mean
        covariance = np.array([[1]]) if covariance is None else covariance
        self.set(mean, covariance)

    def set(self, mean: ColumnVectors, covariance: np.ndarray) -> None:
        assert checks.is_single_vec(mean), "mean needs to be a vector"
        covariance_sqrt = checks.compute_cholesky_if_valid_covariance(covariance)
        self.dim = mean.shape[0]
        self.mean = mean
        self.covariance = covariance
        self.covariance_sqrt = covariance_sqrt
        # only compute when needed
        self.inverse_covariance_sqrt = None
        self.log_pdf_constant = None

    def get_mean_and_covariance(self) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:
        return self.mean, self.covariance, self.covariance_sqrt

    def draw_random_samples(self, num_samples: int) -> ColumnVectors:
        assert num_samples >= 0, "numSamples cannot be negative"
        return ColumnVectors(
            np.transpose(
                np.random.multivariate_normal(
                    self.mean.flatten(), self.covariance, num_samples
                )
            )
        )

    def log_pdf(self, values) -> ColumnVectors:
        assert checks.is_fixed_row_mat(values, self.dim), (
            "values must have the same number of rows as the dimension of "
            "the distribution"
        )
        if self.log_pdf_constant is None:
            self.inverse_covariance_sqrt = np.linalg.inv(self.covariance_sqrt)
            log_sqrt_covariance_determinant = np.sum(
                np.log(np.diag(self.covariance_sqrt))
            )
            self.log_pdf_constant = (
                self.dim * 0.5 * np.log(2 * math.pi) * log_sqrt_covariance_determinant
            )

        s = values - self.mean
        v = np.dot(self.inverse_covariance_sqrt, s)

        return ColumnVectors(-0.5 * np.sum(v**2, axis=0) - self.log_pdf_constant).T

    def __str__(self) -> str:
        return (
            "Gaussian with\nmean="
            + str(self.mean.T)
            + "^T\ncovariance=\n"
            + str(self.covariance)
        )


"""
Uniform distribution on an interval in arbitrary dimension
"""


class Uniform(Distribution):
    def __init__(self, a: ColumnVectors = None, b: ColumnVectors = None):
        # if a/b are not given, create a 1D uniform distribution on [0,1]
        a = ColumnVectors(np.array([0])) if a is None else a
        b = ColumnVectors(np.array([1])) if b is None else b
        self.set(a, b)

    def set(self, a: ColumnVectors, b: ColumnVectors) -> None:
        assert checks.is_single_vec(a), "a needs to be a vector"
        assert checks.is_single_vec(b), "b needs to be a vector"
        self.a = a
        self.b = b
        self.dim = a.shape[0]
        self.mean = None
        self.cov = None
        self.covariance_sqrt = None

    def get_mean_and_covariance(self) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:
        if self.mean is None:
            self.mean = 0.5 * (self.a + self.b)
        if self.cov is None:
            self.cov = np.diag(((self.b - self.a) ** 2 / 12).flatten())
        if self.covariance_sqrt is None:
            self.covariance_sqrt = np.sqrt(self.cov)  # cov is a diagonal matrix

        return self.mean, self.cov, self.covariance_sqrt

    def draw_random_samples(self, num_samples: int) -> ColumnVectors:
        assert num_samples >= 0, "numSamples cannot be negative"
        s = np.random.rand(self.dim, num_samples)
        return self.a + s * (self.b - self.a)

    def log_pdf(self, values: ColumnVectors) -> ColumnVectors:
        assert checks.is_fixed_row_mat(values, self.dim), (
            "values must have the same number of rows as the dimension of "
            "the distribution"
        )
        pdfval = -np.sum(np.log(self.b - self.a))
        return ColumnVectors(
            pdfval
            * np.logical_and((values >= self.a).all(0), (values <= self.b).all(0))
        ).T


"""
Dirac mixture distribution (sum of weighted dirac delta functions)
"""


class DiracMixture(Distribution):
    def __init__(self, samples: ColumnVectors = None, weights: ColumnVectors = None):
        self.set(samples, weights)

    def set(self, samples: ColumnVectors, weights: ColumnVectors = None) -> None:
        if samples is None:
            self.samples = ColumnVectors(np.array([]))
            self.dim = 0
            self.num_components = 0
        else:
            # assert checks.isVec(samples), "samples must be given as a matrix" # TODO is colvec...
            self.samples = samples  # samples needs to be d x n, even if d==1
            self.dim, self.num_components = np.shape(samples)

        if weights is None:
            if self.num_components > 0:
                self.weights = (
                    ColumnVectors(np.ones((1, self.num_components)))
                    / self.num_components
                )
            else:
                self.weights = None
        else:
            assert checks.is_array_of_vecs(weights, dim=1, num=self.num_components)
            self.weights = weights / np.sum(weights)

        self.mean = None
        self.cumulated_weights = None
        self.cov = None
        self.covariance_sqrt = None

    def get_mean_and_covariance(self) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:
        if self.num_components == 0:
            return [], [], []
        if self.mean is None:
            self.mean, self.cov = utils.get_sample_mean_and_covariance(
                self.samples, self.weights
            )
            if self.dim == 1:
                self.covariance_sqrt = np.sqrt(self.cov)
            else:
                self.covariance_sqrt = np.linalg.cholesky(self.cov)
        return self.mean, self.cov, self.covariance_sqrt

    def draw_random_samples(self, num_samples) -> ColumnVectors:
        assert num_samples >= 0, "numSamples cannot be negative"
        # if self.cumWeights is None:
        #    self.cumWeights = np.cumsum(self.weights)
        s, ind = utils.random_resampling(self.samples, self.weights, num_samples)
        return s

    def log_pdf(self, values: ColumnVectors) -> ColumnVectors:
        raise Exception("PDF of Dirac mixture is not defined")

    def get_components(self) -> Tuple[ColumnVectors, np.ndarray]:
        """Get the Dirac mixture components.

        :return: Column-wise arranged sample positions, row-wise arranged weights of the samples.
        """
        return self.samples, self.weights

    def get_num_components(self) -> int:
        """Get the number of Dirac mixture components.

        :return: The number of Dirac mixture components.
        """
        return self.num_components


class GaussianMixture(Distribution):
    """Multivariate Gaussian mixture distribution."""

    def __init__(
        self, means: ColumnVectors, covariance: np.ndarray, weights: np.ndarray = None
    ):
        """Initialize a Gaussian mixture distribution.

        :param means: Means of shape (num_variables, num_components).
        :param covariance: Covariances of shape (num_variables, num_variables, num_components).
        :param weights: Weights of shape (1, num_components). If no weights are passed, it is assumed that the components are equally weighted.
        """
        # Definitions of instance variables
        self.dim = 0
        self.num_comps = 0
        self.means = None
        self.cov_sqrts = None
        self.covs = None
        self.inv_cov_sqrts = None
        self.weights = None
        self.cum_weights = None
        self.log_pdf_consts = None
        self.mean = None
        self.cov = None
        self.cov_sqrt = None

        self.set(means, covariance, weights)

    def set(
        self, means: ColumnVectors, covariances: np.ndarray, weights: np.ndarray
    ) -> None:
        """Set the means, covariances and weights of this distribution's components.

        :param means: Means of shape (num_variables, num_components).
        :param covariances: Covariances of shape (num_variables, num_variables, num_components).
        :param weights: Weights of shape (1, num_components). If no weights are passed, it is assumed that the components are equally weighted.
        """
        if not checks.is_mat(means):
            raise ValueError(
                "Means must be a matrix of dimensions num_means × num_variables."
            )

        dim, num_comps = means.shape

        if num_comps == 1:
            cov_sqrts = checks.compute_cholesky_if_valid_covariance(covariances)[
                ..., None
            ]
        else:
            cov_sqrts = checks.compute_cholesky_if_valid_3d_covariance(
                covariances, dim, num_comps
            )

        if weights is not None:
            if not checks.is_single_row_vec(weights, num_comps):
                raise ValueError(
                    "Weights must be given as row vector of form 1 × num_components. "
                    f"Got vector of shape {weights.shape} instead."
                )
            elif np.min(weights) < 0:
                raise ValueError("Weights vector contains negative values.")

            sum_weights = np.sum(weights)

            if sum_weights == 0:
                raise ValueError(
                    "Cannot normalize weights vector because all values are zeros."
                )

            weights = weights / sum_weights
        else:
            # Weight all components equally
            weights = np.ones((1, num_comps)) / num_comps

        # Set the values after all checks have passed
        self.dim, self.num_comps = dim, num_comps
        self.means = means
        self.covs = covariances
        self.cov_sqrts = cov_sqrts
        self.weights = weights

    def get_dimension(self) -> int:
        """Get the number of variables.

        :return: The number of variables.
        """
        return self.dim

    def get_mean_and_covariance(self) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:
        """Get the mean and covariance of the distribution.

        :return: Mean, covariance and Cholesky decomposition of the covariance.
        """
        if self.mean is None:
            self.mean, self.cov = utils.get_gaussian_mixture_mean_and_covariance(
                self.means, self.covs, self.weights
            )

        return self.mean, self.cov, np.linalg.cholesky(self.cov)

    def draw_random_samples_with_comp_ids(
        self, num_samples: int
    ) -> Tuple[ColumnVectors, np.ndarray]:
        """Draw random samples from the distribution and return corresponding component ids.

        :param num_samples: Number of samples.
        :return: Samples of shape (num_variables, num_samples), list with component ids of shape (1, num_samples).
        """
        assert (
            isinstance(num_samples, int) and num_samples > 0
        ), "num_samples must be positive integer."

        if self.cum_weights is None:
            self.cum_weights = np.cumsum(self.weights)

        u = np.random.rand(num_samples)
        u = np.sort(u)

        num_comps_samples = np.zeros(self.num_comps, dtype=int)
        i = 0

        for j in range(num_samples):
            while u[j] > self.cum_weights[i]:
                i += 1

            num_comps_samples[i] += 1

        rnd_samples: ColumnVectors = ColumnVectors(np.empty((self.dim, num_samples)))
        comp_ids = np.empty((1, num_samples))

        a = b = 0
        for i in range(self.num_comps):
            num_comp_samples = num_comps_samples[i]

            if num_comp_samples > 0:
                b = b + num_comp_samples

                component = Gaussian(self.means[:, i : i + 1], self.covs[..., i])
                rnd_samples[:, a:b] = component.draw_random_samples(num_comp_samples)

                comp_ids[:, a:b] = i
                a = b

        return rnd_samples, comp_ids

    def draw_random_samples(self, num_samples: int) -> ColumnVectors:
        """Draw random samples from the distribution.

        :param num_samples: Number of samples.
        :return: Samples of shape (num_variables, num_samples).
        """
        rnd_samples, _ = self.draw_random_samples_with_comp_ids(num_samples)
        return rnd_samples

    def log_pdf(self, values: ColumnVectors) -> ColumnVectors:
        """Evaluate the logarithmic probability density function at given values.

        :param values: Values of shape (num_variables, num_values)
        :return: Evaluations of shape (num_variables, num_values)
        """
        assert checks.is_fixed_row_mat(values, self.dim), (
            "values must have the same number of rows as the dimension of "
            "the distribution"
        )

        if self.log_pdf_consts is None:
            log_norm_const = self.dim * 0.5 * np.log(2 * np.pi)
            log_weights = np.log(self.weights)  # 1 × num_comps

            log_sqrt_det_cov = np.sum(
                np.log(np.diagonal(self.cov_sqrts, axis1=0, axis2=1)), axis=1
            )[
                None, :
            ]  # 1 × num_comps
            self.log_pdf_consts = log_weights - (
                log_sqrt_det_cov + log_norm_const
            )  # 1 × num_comps
            cov_sqrt = self.cov_sqrts.transpose(2, 0, 1)
            self.inv_cov_sqrts = np.linalg.inv(cov_sqrt).transpose(
                1, 2, 0
            )  # num_dims × num_dims × num_comps

        # Substract means from components,
        # then matrix multiply lower cholesky with values for each component
        s = (
            values[:, :, None] - self.means[:, None, :]
        )  # num_dims × num_values × num_comps
        v = np.einsum(
            "ijm,jkm->ikm", self.inv_cov_sqrts, s
        )  # num_dims × num_values × num_comps
        comp_values = self.log_pdf_consts - 0.5 * np.sum(
            v**2, axis=0
        )  # num_values × num_comps
        comp_values = comp_values.T

        # Classic using loops:
        # comp_values2 = np.empty((self.num_comps, values.shape[1]))
        # for i in range(self.num_comps):
        #     s2 = values[:, :] - self.means[:, None, i]
        #     v2 = self.inv_cov_sqrts[..., i].dot(s2)
        #     comp_values2[i, :] = self.log_pdf_consts[:, i] - 0.5*np.sum(v2**2, axis=0, keepdims=True)
        #
        # comp_values = comp_values2

        max_log_comp_values = np.max(
            comp_values, axis=0, keepdims=True
        )  # 1 × num_values
        comp_values = comp_values - max_log_comp_values  # num_comps × num_values
        comp_values = np.exp(comp_values)  # num_comps × num_values

        log_values = max_log_comp_values + np.log(
            np.sum(comp_values, axis=0, keepdims=True)
        )
        return log_values
