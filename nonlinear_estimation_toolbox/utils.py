import random
from typing import Callable, Tuple, Union

import numpy as np

from nonlinear_estimation_toolbox import checks as checks

EPSILON_DIFF = np.finfo(float).eps ** (0.25)


class ColumnVectors(np.ndarray):
    """
    Wrapper class to have all vectors as column vectors and all samples as matrix of column vector. It transforms
    scalars and vectors to two dimensional np.arrays in column representation. This shall prevent confusion with
    matrix products.
    """

    def __new__(cls, value):
        obj = np.asarray(value).view(cls)
        if obj.ndim > 2:
            raise RuntimeError("No tensors are allowed")
        # reshape scalars
        if obj.ndim == 0:
            obj = obj.reshape((1, 1))
        # reshape 1D arrays
        elif obj.ndim == 1:
            obj = obj.reshape((len(obj), 1))
        return obj

    def as_list(self):
        return list(self)

    def as_numpy(self):
        return np.asarray(self)


def get_numeric_jacobians_for_state_and_noise(
    func: Callable[[ColumnVectors, ColumnVectors], ColumnVectors],
    nominal_state: ColumnVectors,
    nominal_noise: ColumnVectors,
    step: float = EPSILON_DIFF,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param func: function to calculate the diff quotient for
    :param nominal_state: numpy array of shape (dimState,) which contains the nominal system state vector.
    :param nominal_noise: numpy array of shape (dimNoise,) which contains the nominal system noise vector.
    optional:
    :param step: Positive scalar. Step size to compute the finite difference.
    """

    dimState = nominal_state.shape[0]
    dimNoise = nominal_noise.shape[0]

    # State Jacobian
    state_samples = get_jacobian_samples(
        dim=dimState, nominal_vecs=nominal_state, step=step
    )
    noise_samples = np.tile(A=nominal_noise, reps=(1, 2 * dimState + 1))
    state_values = func(state_samples, noise_samples)
    state_jacobian = get_jacobian(dim=dimState, values=state_values, step=step)

    # Noise Jacobian
    noise_samples = get_jacobian_samples(
        dim=dimNoise, nominal_vecs=nominal_noise, step=step
    )
    state_samples = np.tile(A=nominal_state, reps=(1, 2 * dimNoise + 1))
    noise_values = func(state_samples, noise_samples)
    noise_jacobian = get_jacobian(dim=dimNoise, values=noise_values, step=step)

    return state_jacobian, noise_jacobian


def get_numeric_jacobian_for_state(
    func: Callable[[ColumnVectors], ColumnVectors],
    nominal_state: ColumnVectors,
    step: float = EPSILON_DIFF,
) -> np.ndarray:
    """
    :param func: function to calculate the diff quotient for
    :param nominal_state: numpy array of shape (dim_state,) which contains the nominal system state vector.
    optional:
    :param step: Positive scalar. Step size to compute the finite difference.
    """
    dim_state = nominal_state.shape[0]

    # State Jacobian
    state_samples = get_jacobian_samples(
        dim=dim_state, nominal_vecs=nominal_state, step=step
    )
    state_values = func(state_samples)

    state_jacobian = get_jacobian(dim=dim_state, values=state_values, step=step)
    return state_jacobian


def decomposed_state_update(
    state_mean: np.ndarray,
    state_cov: np.ndarray,
    state_cov_sqrt: np.ndarray,
    updated_state_mean_a: np.ndarray,
    updated_state_cov_a: np.ndarray,
    updated_state_cov_sqrt_a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    D = updated_state_mean_a.shape[0]

    prior_state_mean_a = state_mean[:D]
    prior_state_mean_b = state_mean[D:]
    prior_state_cov_a = state_cov[:D, :D]
    prior_state_cov_b = state_cov[D:, D:]
    prior_state_cov_b_a = state_cov[D:, :D]
    prior_state_cov_a_sqrt = state_cov_sqrt[:D, :D]

    K = prior_state_cov_b_a.dot(np.linalg.inv(prior_state_cov_a))
    A = K.dot(updated_state_cov_sqrt_a)
    B = K.dot(prior_state_cov_a_sqrt)

    updated_state_mean_b = prior_state_mean_b + K.dot(
        updated_state_mean_a - prior_state_mean_a
    )
    updated_state_cov_b = prior_state_cov_b + A.dot(A.T) - B.dot(B.T)
    updated_state_cov_b_a = K.dot(updated_state_cov_a)

    updated_state_mean = np.vstack([updated_state_mean_a, updated_state_mean_b])
    updated_state_cov = np.block(
        [
            [updated_state_cov_a, updated_state_cov_b_a.T],
            [updated_state_cov_b_a, updated_state_cov_b],
        ]
    )
    return updated_state_mean, updated_state_cov


def get_sample_mean_and_covariance(
    samples: ColumnVectors, weights: Union[float, ColumnVectors] = None
) -> Tuple[ColumnVectors, np.ndarray]:
    """
    Computes mean and covariance of a set of weighted samples.
    :param samples: matrix containing one sample per column
    :param weights: vector containing weights for each sample
    :return: mean vector and covariance matrix
    """
    assert checks.is_mat(samples), "samples need to be given as a matrix"
    if weights is None or np.isscalar(weights):
        weights = (
            ColumnVectors(np.ones((1, np.shape(samples)[1]))) / np.shape(samples)[1]
        )
    else:
        assert checks.is_single_row_vec(
            weights, dim=samples.shape[1]
        ), "Weights must be a row vector of shape 1 × num_samples."

    mean = samples.dot(weights.T)
    zero_mean_samples = samples - mean
    # special case for negative weights is not necessary
    cov = (zero_mean_samples * weights).dot(zero_mean_samples.transpose())

    return mean, cov


def get_gaussian_mixture_mean_and_covariance(
    means: ColumnVectors, covariances: np.ndarray, weights: np.ndarray = None
) -> Tuple[ColumnVectors, np.ndarray]:
    """
    Computes mean and covariance of a Gaussian mixture
    :param means: mean of each component
    :param covariances: covariance of each component
    :param weights: weight of each component
    :return: mean vector and covariance matrix
    """

    num_components = means.shape[1]

    if weights is None:
        weights = np.ones((1, num_components)) / num_components

    mean, cov_means = get_sample_mean_and_covariance(means, weights)
    weighted_covs = covariances * weights[None, ...]
    cov = cov_means + np.sum(weighted_covs, axis=2)

    return mean, cov


# uses weights, not cumWeights unlike matlab version due to builtin np.random.choice function
def random_resampling(
    samples: ColumnVectors, weights: ColumnVectors, num_new_samples: int
) -> Tuple[ColumnVectors, np.ndarray]:
    n = samples.shape[1]
    a = np.arange(n)
    ind = np.random.choice(a, num_new_samples, p=weights.flatten())

    return samples[:, ind], ind


def systematic_resampling(
    samples: np.ndarray, weights: np.ndarray, num_new_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a systematic resampling.

    Implements the systematic resampling algorithm from:
        Branko Ristic, Sanjeev Arulampalam, and Neil Gordon,
        Beyond the Kalman Filter: Particle filters for Tracking Applications,
        Artech House Publishers, 2004,
        Section 3.3

    Also described here:
        Hol, Jeroen D., Thomas B. Schon, and Fredrik Gustafsson.
        "On resampling algorithms for particle filters."
        2006 IEEE nonlinear statistical signal processing workshop.
        IEEE, 2006.
        Section 2.3

    :param samples: Set of column-wise arranged sample positions to resample from.
    :param weights: Vector containing the sample weights.
    :param num_new_samples: Number of samples to draw from the given sample distribution.
    :return:
        - Column-wise arranged samples drawn from the given sample distribution.
        - Corresponding indices of the samples that were resampled from.
    """
    csw = np.cumsum(weights) * num_new_samples
    idx = np.zeros(num_new_samples, dtype=int)

    u1 = random.random()
    i = 0
    for j in range(num_new_samples):
        uj = u1 + j
        while uj > csw[i]:
            i = i + 1

        idx[j] = i

    rnd_samples = samples[:, idx]
    return rnd_samples, idx


def get_jacobian_samples(dim: int, nominal_vecs: ColumnVectors, step: float):
    assert checks.is_array_of_vecs(
        nominal_vecs
    ), "utils.getJacobianSamples:InvalidVector, nominalVec must be a vector."
    samples = (
        np.hstack((step * np.eye(dim), -step * np.eye(dim), np.zeros((dim, 1))))
        + nominal_vecs
    )
    return samples


def get_jacobian(dim: int, values: ColumnVectors, step: float) -> ColumnVectors:
    idx = np.arange(start=0, stop=dim)
    jacobian = values[:, idx] - values[:, dim + idx]
    jacobian /= 2 * step
    return jacobian


def kalman_update(
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_mean: np.ndarray,
    measurement_covariance: np.ndarray,
    state_measurement_cross_covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform the kalman filter/update step.
    """
    measurement_covariance_sqrt = checks.compute_cholesky_if_valid_covariance(
        measurement_covariance
    )

    # The kalman update in this function is reformulated
    # using K @ (z - H @ x_p) = A @ t
    #   --> x_e = x_p + A @ t
    #   --> C_e = C_p - A @ A.T

    # A is related to the Kalman gain K:
    # K =            C_p @ H^T @ (H @ C_p @ H^T + C_v)^(-1)
    #   = state_meas_cross_cov @ meas_cov^(-1)
    #   = state_meas_cross_cov @ sqrt(meas_cov^(-1))^T @ sqrt(meas_cov^(-1))
    #   = state_meas_cross_cov @ sqrt(meas_cov^T)^(-1) @ sqrt(meas_cov)^(-1)
    #   =                                            A @ sqrt(meas_cov)^(-1)
    measurement_covariance_sqrt_inv = np.linalg.inv(measurement_covariance_sqrt)
    A = state_measurement_cross_covariance.dot(measurement_covariance_sqrt_inv.T)

    # innovation = z - H_k @ x_p
    #            = z - meas_mean
    innovation = measurement - measurement_mean

    # t = sqrt((H @ C_p @ H^T + C_v)^(-1)) @ innovation
    #   =              sqrt(meas_cov)^(-1) @ innovation
    t = measurement_covariance_sqrt_inv.dot(innovation)

    updated_state_mean = state_mean + A.dot(t)
    updated_state_cov = state_covariance - A.dot(A.T)

    sq_meas_mahal_dist = float(t.T.dot(t))

    return updated_state_mean, updated_state_cov, sq_meas_mahal_dist


def rnd_orthogonal_matrix(dim: int) -> np.ndarray:
    # generate random orthogonal matrix

    mat = np.random.standard_normal([dim, dim])
    [Q, R] = np.linalg.qr(mat)
    D = np.diag(np.sign(np.diag(R)))

    rnd_mat = Q.dot(D)
    return rnd_mat
