from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.distributions import Distribution
from nonlinear_estimation_toolbox.utils import ColumnVectors


class SystemModel:
    def __init__(self):
        """
        Constructor
        """
        self.noise: Distribution = Distribution()

    def set_noise(self, noise: Distribution):
        """
        :param noise: Subclass of Distribution.
        """
        assert isinstance(
            noise, Distribution
        ), "SystemModel:InvalidNoise, noise must be a subclass of Distribution."
        self.noise = noise

    def derivative(
        self, nominal_state: ColumnVectors, nominal_noise: ColumnVectors
    ) -> Tuple[ColumnVectors, ColumnVectors]:
        """
        :param nominal_state: ColumnVectors of shape (dimState,) which contains the nominal system state vector.
        :param nominal_noise: ColumnVectors of shape (dimNoise,) which contains the nominal system noise vector.
        """
        # TODO: dropped calculation of Hessian, for now...

        (
            state_jacobian,
            noise_jacobian,
        ) = utils.get_numeric_jacobians_for_state_and_noise(
            func=self.system_equation,
            nominal_state=nominal_state,
            nominal_noise=nominal_noise,
        )

        return state_jacobian, noise_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate the temporal evolution for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """

        assert checks.is_array_of_vecs(v=state)

        noise_sample = self.noise.draw_random_samples(1)

        predicted_state = self.system_equation(
            state_samples=state, noise_samples=noise_sample
        )

        return predicted_state

    def system_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )


class AdditiveNoiseSystemModel:
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.noise: Distribution = Distribution

    def set_noise(self, noise: Distribution):
        """
        :param noise: Subclass of Distribution.
        """
        assert isinstance(
            noise, Distribution
        ), "AdditiveNoiseSystemModel:InvalidNoise, noise must be a subclass of Distribution."
        self.noise = noise

    def derivative(self, nominal_state: ColumnVectors) -> ColumnVectors:
        """
        :param nominal_state: ColumnVectors of shape (dimState,) which contains the nominal system state vector.
        """
        # TODO: dropped calculation of Hessian, for now...

        state_jacobian = utils.get_numeric_jacobian_for_state(
            func=self.system_equation, nominal_state=nominal_state
        )

        return state_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate the temporal evolution for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """

        noise_sample = self.noise.draw_random_samples(1)

        predicted_state = self.system_equation(state_samples=state) + noise_sample

        return predicted_state

    def system_equation(
        self,
        state_samples: ColumnVectors,
    ) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )


class LinearSystemModel(SystemModel):
    def __init__(
        self, sys_matrix: np.ndarray = None, sys_noise_matrix: np.ndarray = None
    ):

        super(LinearSystemModel, self).__init__()
        self.sys_matrix = None
        self.sys_noise_matrix = None
        self.sys_input = None
        self.set_system_matrix(sys_matrix=sys_matrix)
        self.set_system_noise_matrix(sys_noise_matrix=sys_noise_matrix)
        self.set_system_input(sys_input=None)

    def set_system_matrix(self, sys_matrix: np.ndarray = None):
        if sys_matrix is not None:
            assert checks.is_square_mat(
                sys_matrix
            ), "LinearSystemModel:InvalidSystemMatrix, sysMatrix must be a square matrix or None."
        self.sys_matrix = sys_matrix

    def set_system_noise_matrix(self, sys_noise_matrix: np.ndarray = None):
        if sys_noise_matrix is not None:
            assert checks.is_mat(
                sys_noise_matrix
            ), "LinearSystemModel:InvalidSystemNoiseMatrix, sysMatrix must be a matrix or None."
        self.sys_noise_matrix = sys_noise_matrix

    def set_system_input(self, sys_input: ColumnVectors = None):
        if sys_input is not None:
            assert checks.is_single_vec(
                sys_input
            ), "LinearSystemModel:InvalidSystemInput, sysMatrix must be a vector or None."
        self.sys_input = sys_input

    def derivative(
        self, nominal_state: ColumnVectors, nominal_noise: ColumnVectors
    ) -> Tuple[np.ndarray, np.ndarray]:
        dim_state = nominal_state.shape[0]
        dim_noise = nominal_noise.shape[0]

        if self.sys_matrix is not None:
            self._check_sys_matrix(dim_state=dim_state)
            state_jacobian = self.sys_matrix
        else:
            state_jacobian = np.eye(dim_state)

        if self.sys_noise_matrix is not None:
            self._check_sys_noise_matrix(dim_state=dim_state, dim_noise=dim_noise)
            noise_jacobian = self.sys_noise_matrix
        else:
            self._check_sys_noise(dim_state=dim_state, dim_noise=dim_noise)
            noise_jacobian = np.eye(dim_state)

        return state_jacobian, noise_jacobian

    def system_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:

        dim_state = state_samples.shape[0]
        dim_noise = noise_samples.shape[0]

        if self.sys_matrix is None:
            predicted_states = state_samples
        else:
            self._check_sys_matrix(dim_state)
            predicted_states = np.matmul(self.sys_matrix, state_samples)

        if self.sys_noise_matrix is None:
            self._check_sys_noise(dim_state, dim_noise)
            predicted_states = predicted_states + noise_samples
        else:
            self._check_sys_noise_matrix(dim_state, dim_noise)
            predicted_states = predicted_states + np.matmul(
                self.sys_noise_matrix, noise_samples
            )

        if self.sys_input is not None:
            self._check_sys_input(dim_state)
            predicted_states = predicted_states + self.sys_input

        return predicted_states

    def analytic_moments(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[ColumnVectors, np.ndarray]:
        noise_mean, noise_cov, noise_cov_sqrt = self.noise.get_mean_and_covariance()
        dim_state = state_mean.shape[0]
        dim_noise = noise_mean.shape[0]

        if self.sys_matrix is None:
            predicted_state_mean = state_mean
            predicted_state_cov = state_cov
        else:
            self._check_sys_matrix(dim_state=dim_state)

            matrix_g = np.matmul(self.sys_matrix, state_cov_sqrt)

            predicted_state_mean = np.matmul(self.sys_matrix, state_mean)
            predicted_state_cov = np.matmul(matrix_g, matrix_g.T)

        if self.sys_noise_matrix is None:
            self._check_sys_noise(dim_state=dim_state, dim_noise=dim_noise)

            predicted_state_mean = predicted_state_mean + noise_mean
            predicted_state_cov = predicted_state_cov + noise_cov
        else:
            self._check_sys_noise_matrix(dim_state=dim_state, dim_noise=dim_noise)

            matrix_g = np.matmul(self.sys_noise_matrix, noise_cov_sqrt)

            predicted_state_mean = predicted_state_mean + np.matmul(
                self.sys_noise_matrix, noise_mean
            )
            predicted_state_cov = predicted_state_cov + np.matmul(matrix_g, matrix_g.T)

        if self.sys_input is not None:
            self._check_sys_input(dim_state=dim_state)

            predicted_state_mean = predicted_state_mean + self.sys_input

        return predicted_state_mean, predicted_state_cov

    def _check_sys_matrix(self, dim_state: int):
        dim_sys_matrix_rows = self.sys_matrix.shape[0]

        assert (
            dim_state == dim_sys_matrix_rows
        ), "LinearSystemModel:IncompatibleSystemMatrix, System state and system matrix with incompatible dimensions."

    def _check_sys_noise_matrix(self, dim_state: int, dim_noise: int):
        sys_noise_matrix_rows, sys_noise_matrix_cols = self.sys_noise_matrix.shape

        assert (
            dim_state == sys_noise_matrix_rows
        ), "LinearSystemModel:IncompatibleSystemNoiseMatrix, System state and system noise matrix with incompatible dimensions."

        assert (
            dim_noise == sys_noise_matrix_cols
        ), "LinearSystemModel:IncompatibleSystemNoiseMatrix, System noise and system noise matrix with incompatible dimensions."

    def _check_sys_input(self, dim_state: int):
        dim_sys_input = self.sys_input.shape[0]

        assert (
            dim_state == dim_sys_input
        ), "LinearSystemModel:IncompatibleSystemInput, System state and system input with incompatible dimensions."

    @staticmethod
    def _check_sys_noise(dim_state: int, dim_noise: int):
        assert (
            dim_state == dim_noise
        ), "LinearSystemModel:IncompatibleSystemNoise, System state and system noise with incompatible dimensions."


class MixedNoiseSystemModel:
    def __init__(self):
        self.additive_noise: Distribution = Distribution()
        self.noise: Distribution = Distribution()

    def set_noise(self, noise: Distribution):
        """
        :param noise: Subclass of Distribution.
        """
        assert isinstance(
            noise, Distribution
        ), "AdditiveNoiseSystemModel:InvalidNoise, noise must be a subclass of Distribution."
        self.noise = noise

    def set_additive_noise(self, additive_noise: Distribution):
        """
        :param additive_noise: Subclass of Distribution.
        """
        assert isinstance(
            additive_noise, Distribution
        ), "AdditiveNoiseSystemModel:InvalidNoise, noise must be a subclass of Distribution."
        self.additive_noise = additive_noise

    def derivative(
        self, nominal_state: ColumnVectors, nominal_noise: ColumnVectors
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param nominal_state: ColumnVectors of shape (dimState,) which contains the nominal system state vector.
        :param nominal_noise: ColumnVectors of shape (dimNoise,) which contains the nominal system noise vector.
        """
        # TODO: dropped calculation of Hessian, for now...

        (
            state_jacobian,
            noise_jacobian,
        ) = utils.get_numeric_jacobians_for_state_and_noise(
            func=self.system_equation,
            nominal_state=nominal_state,
            nominal_noise=nominal_noise,
        )

        return state_jacobian, noise_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate the temporal evolution for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """
        noise_sample = self.noise.draw_random_samples(1)
        add_noise_sample = self.additive_noise.draw_random_samples(1)

        predicted_state = (
            self.system_equation(state_samples=state, noise_samples=noise_sample)
            + add_noise_sample
        )

        return predicted_state

    def system_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )
