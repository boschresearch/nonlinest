from typing import Callable, Tuple

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.filters.linear_gaussian import LinearGaussianFilter


class IterativeKalmanFilter(LinearGaussianFilter):
    # TODO: Add doc string
    """
    Toolbox/Interfaces/Filters/IterativeKalmanFilter.m
    """

    def __init__(self, name: str = ""):
        super(IterativeKalmanFilter, self).__init__(name)
        self.max_num_iterations = 1
        self.num_iterations = 0
        self.convergence_check_func = None

    def set_max_num_iterations(self, max_num_iterations: int):
        if not checks.is_pos_scalar(max_num_iterations):
            raise ValueError(
                "InvalidNumberOfIterations: maxNumIterations must be a positive scalar."
            )
        self.max_num_iterations = np.ceil(max_num_iterations)

    def get_max_num_iterations(self) -> int:
        return self.max_num_iterations

    def get_num_iterations(self) -> int:
        return self.num_iterations

    def set_convergence_check_func(self, convergenc_check_func: Callable):
        if convergenc_check_func is not None and not isinstance(
            convergenc_check_func, Callable
        ):
            raise TypeError(
                "InvalidConvergenceCheck: convergenceCheck must be a function handle or None."
            )
        self.convergence_check_func = convergenc_check_func

    def get_convergence_check_func(self) -> Callable:
        return self.convergence_check_func

    def update_nonlinear(
        self,
        measurement: np.ndarray,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Initialize iteration counter
        self.num_iterations = 1

        # Perform first measurement update
        meas_mean, meas_cov, state_meas_cross_cov = self.get_meas_moments(
            prior_state_mean, prior_state_cov, prior_state_cov_sqrt
        )

        try:
            if self._is_measurement_gating_enabled():
                # Perform measurement gating before any measurement information is processed
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

        except Exception:
            raise RuntimeError("Ignoring measurement - implement ignoring algorithm")

        # More updates to perform (iterative update)?
        if self.max_num_iterations > 1:
            # Initially, the last estimate is the prior one.
            last_state_mean = prior_state_mean
            last_state_cov = prior_state_cov
            last_state_cov_sqrt = prior_state_cov_sqrt

            while self.num_iterations < self.max_num_iterations:
                # Check if intermediate state covariance matrix is valid
                updated_state_cov_sqrt = checks.compute_cholesky_if_valid_covariance(
                    updated_state_cov, "Intermediate state"
                )

                # Check for convergence
                if self._is_convergence_reached(
                    last_state_mean,
                    last_state_cov,
                    last_state_cov_sqrt,
                    updated_state_mean,
                    updated_state_cov,
                    updated_state_cov_sqrt,
                ):
                    # No more iteration is required according to a possibly set convergence check
                    return updated_state_mean, updated_state_cov

                # Save estimate from last iteration
                last_state_mean = updated_state_mean
                last_state_cov = updated_state_cov
                last_state_cov_sqrt = updated_state_cov_sqrt

                # Increment iteration counter
                self.num_iterations += 1

                # Perform measurement update for the current iteration
                (
                    meas_mean,
                    meas_cov,
                    state_meas_cross_cov,
                ) = self.get_meas_moments_iteration(
                    prior_state_mean,
                    prior_state_cov,
                    prior_state_cov_sqrt,
                    updated_state_mean,
                    updated_state_cov,
                    updated_state_cov_sqrt,
                )

                try:
                    updated_state_mean, updated_state_cov, _ = utils.kalman_update(
                        prior_state_mean,
                        prior_state_cov,
                        measurement,
                        meas_mean,
                        meas_cov,
                        state_meas_cross_cov,
                    )
                except Exception:
                    raise RuntimeError(
                        "Ignoring measurement - implement ignoring algorithm"
                    )
        return updated_state_mean, updated_state_cov

    def get_meas_moments(
        self,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_meas_moments_iteration(
        self,
        prior_state_mean: np.ndarray,
        prior_state_cov: np.ndarray,
        prior_state_cov_sqrt: np.ndarray,
        updated_state_mean: np.ndarray,
        updated_state_cov: np.ndarray,
        updated_state_cov_sqrt: np.ndarray,
    ):
        raise NotImplementedError

    def _is_convergence_reached(
        self,
        last_state_mean: np.ndarray,
        last_state_cov: np.ndarray,
        last_state_cov_sqrt: np.ndarray,
        updated_state_mean: np.ndarray,
        updated_state_cov: np.ndarray,
        updated_state_cov_sqrt: np.ndarray,
    ) -> bool:
        if self.convergence_check_func is not None:
            return self.convergence_check_func(
                last_state_mean,
                last_state_cov,
                last_state_cov_sqrt,
                updated_state_mean,
                updated_state_cov,
                updated_state_cov_sqrt,
            )
        else:
            return False
