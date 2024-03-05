from typing import Tuple

import numpy as np

from nonlinear_estimation_toolbox import checks, utils
from nonlinear_estimation_toolbox.distributions import Distribution
from nonlinear_estimation_toolbox.utils import ColumnVectors


class MeasurementModel:
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
        ), "MeasurementModel:InvalidNoise, noise must be a subclass of Distribution."
        self.noise = noise

    def derivative(
        self, nominal_state: ColumnVectors, nominal_noise: np.array
    ) -> Tuple[ColumnVectors, np.ndarray]:
        """Compute the first-order derivative of the implemented measurement equation.

        This function can be overwritten to implement an analytic Jacobian calculation.

        By default, the derivative is computed using difference quotients.
        Mainly used by EKF.

        :param nominal_state: ColumnVectors of shape (dim_state,) which contains the nominal system state vector.
        :param nominal_noise: numpy ndarray of shape (dim_state, dim_state) which contains the nominal system noise matrix.
        :return:
            - Jacobian of state variables
            - Jacobian of noise variables
        """
        # TODO: dropped calculation of Hessian, for now...

        (
            state_jacobian,
            noise_jacobian,
        ) = utils.get_numeric_jacobians_for_state_and_noise(
            func=self.measurement_equation,
            nominal_state=nominal_state,
            nominal_noise=nominal_noise,
        )

        return state_jacobian, noise_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate a measurement for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """

        assert checks.is_array_of_vecs(v=state)

        noise_sample = self.noise.draw_random_samples(1)

        measurement = self.measurement_equation(
            state_samples=state, noise_samples=noise_sample
        )

        return measurement

    # Abstract Methods
    def measurement_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )


class Likelihood:
    def log_likelihood(
        self, state_samples: ColumnVectors, measurement: ColumnVectors
    ) -> ColumnVectors:
        raise NotImplementedError


class AdditiveNoiseMeasurementModel(Likelihood):
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
        ), "AdditiveNoiseMeasurementModel:InvalidNoise, noise must be a subclass of Distribution."
        self.noise = noise

    def log_likelihood(
        self, state_samples: ColumnVectors, measurement: ColumnVectors
    ) -> ColumnVectors:

        assert checks.is_array_of_vecs(v=measurement)

        dim_noise = self.noise.get_dimension()
        dim_meas = measurement.shape[0]
        num_samples = state_samples.shape[1]

        assert (
            dim_meas == dim_noise
        ), "AdditiveNoiseMeasurementModel:InvalidMeasurementNoise, measurement and additive measurement noise with different dimensions."

        deterministic_meas = self.measurement_equation(state_samples)

        checks.is_mat(deterministic_meas, rows=dim_meas, cols=num_samples)

        values = measurement - deterministic_meas

        log_values = self.noise.log_pdf(values)

        return log_values

    def derivative(self, nominal_state: ColumnVectors) -> ColumnVectors:
        """Compute the first-order derivative of the implemented measurement equation.

        This function can be overwritten to implement an analytic Jacobian calculation.

        By default, the derivative is computed using difference quotients.
        Mainly used by EKF.

        :param nominal_state: ColumnVectors of shape (dim_state,) which contains the nominal system state vector.
        :param nominal_noise: numpy ndarray of shape (dim_state, dim_state) which contains the nominal system noise matrix.
        :return: Jacobian of state variables
        """
        # TODO: dropped calculation of Hessian, for now...

        state_jacobian = utils.get_numeric_jacobian_for_state(
            func=self.measurement_equation, nominal_state=nominal_state
        )

        return state_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate a measurement for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """

        assert checks.is_array_of_vecs(
            v=state
        ), "AdditiveNoiseMeasurementModel:InvalidSystemState, state must be a column vector."

        noise_sample = self.noise.draw_random_samples(1)

        measurement = self.measurement_equation(state_samples=state) + noise_sample

        return measurement

    # Abstract Methods
    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )


class LinearMeasurementModel(AdditiveNoiseMeasurementModel):
    def __init__(self, meas_matrix: np.ndarray = None):
        super(LinearMeasurementModel, self).__init__()
        self.set_measurement_matrix(meas_matrix=meas_matrix)

    def set_measurement_matrix(self, meas_matrix: np.ndarray = None):
        if meas_matrix is not None:
            assert checks.is_mat(
                meas_matrix
            ), "LinearMeasurementModel:InvalidMeasurementMatrix, measMatrix must be a matrix or None."
        self.meas_matrix = meas_matrix

    def derivative(self, nominal_state: ColumnVectors) -> np.ndarray:
        dim_state = nominal_state.shape[0]

        if self.meas_matrix is not None:
            self._check_meas_matrix(dim_state=dim_state)
            state_jacobian = self.meas_matrix
        else:
            state_jacobian = np.eye(dim_state)

        return state_jacobian

    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        if self.meas_matrix is None:
            measurements = state_samples
        else:
            dim_state = state_samples.shape[0]

            self._check_meas_matrix(dim_state=dim_state)

            measurements = np.matmul(self.meas_matrix, state_samples)

        return measurements

    def analytic_moments(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> Tuple[ColumnVectors, np.ndarray, np.ndarray]:

        noise_mean, noise_cov, _ = self.noise.get_mean_and_covariance()
        dim_noise = noise_mean.shape[0]
        dim_state = state_mean.shape[0]

        if self.meas_matrix is None:
            self._check_meas_noise(dim_state, dim_noise)

            # Measurement mean
            meas_mean = state_mean + noise_mean

            # Measurement covariance
            meas_cov = state_cov + noise_cov

            # State measurement-cross-covariance
            state_meas_cross_cov = state_cov

        else:
            self._check_meas_matrix(dim_state, dim_noise)

            # Measurement mean
            meas_mean = np.matmul(self.meas_matrix, state_mean) + noise_mean

            # Measurement covariance
            G = np.matmul(self.meas_matrix, state_cov_sqrt)
            meas_cov = np.matmul(G, G.T) + noise_cov

            # State measurement-cross-covariance
            state_meas_cross_cov = np.matmul(state_cov, self.meas_matrix.T)

        return meas_mean, meas_cov, state_meas_cross_cov

    def _check_meas_matrix(self, dim_state: int, dim_noise: int = None):
        dim_meas_matrix_meas, dim_meas_matrix_state = self.meas_matrix.shape

        assert (
            dim_state == dim_meas_matrix_state
        ), "LinearMeasurementModel:IncompatibleMeasurementMatrix, System state and measurement matrix with incompatible dimensions."

        if dim_noise is not None:
            assert (
                dim_noise == dim_meas_matrix_meas
            ), "LinearMeasurementModel:IncompatibleMeasurementMatrix, Measurement noise and measurement matrix with incompatible dimensions."

    def _check_meas_noise(self, dim_state: int, dim_noise: int):
        assert (
            dim_noise == dim_state
        ), "LinearMeasurementModel:IncompatibleMeasurementNoise, System state and measurement noise with incompatible dimensions."


class MixedNoiseMeasurementModel:
    """Abstract base class for system models corrupted by a mixture of additive and arbitrary noise."""

    def __init__(self):
        """
        Constructor
        """
        self.noise: Distribution = Distribution()
        self.additive_noise: Distribution = Distribution()

    def set_noise(self, noise: Distribution) -> None:
        """Set the measurement noise.

        :param noise: Subclass of Distribution.
        """
        assert isinstance(
            noise, Distribution
        ), "Noise must be a subclass of Distribution."
        self.noise = noise

    def set_additive_noise(self, noise: Distribution) -> None:
        """Set the pure additive measurement noise.

        :param noise: The new pure additive measurement noise.
        """
        assert isinstance(
            noise, Distribution
        ), "Noise must be a subclass of Distribution."
        self.additive_noise = noise

    def derivative(
        self, nominal_state: ColumnVectors, nominal_noise: ColumnVectors
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the first-order derivatives of the implemented measurement equation.

        This function can be overwritten to implement an analytic Jacobian calculation.

        By default, the derivative is computed using difference quotients.
        Mainly used by EKF.

        :param nominal_state: The nominal system state vector.
        :param nominal_noise: The nominal system noise vector.
        :return:
            - Jacobian of state variables
            - Jacobian of noise variables
        """

        (
            state_jacobian,
            noise_jacobian,
        ) = utils.get_numeric_jacobians_for_state_and_noise(
            func=self.measurement_equation,
            nominal_state=nominal_state,
            nominal_noise=nominal_noise,
        )

        return state_jacobian, noise_jacobian

    def simulate(self, state: ColumnVectors) -> ColumnVectors:
        """
        Simulate a measurement for a given system state.
        :param state: ColumnVectors of shape (dimState,) which contains a state vector to predict.
        """

        assert checks.is_array_of_vecs(
            v=state
        ), "AdditiveNoiseMeasurementModel:InvalidSystemState, state must be a column vector."

        add_noise_samples = self.additive_noise.draw_random_samples(1)
        noise_sample = self.noise.draw_random_samples(1)

        measurement = (
            self.measurement_equation(state_samples=state, noise_samples=noise_sample)
            + add_noise_samples
        )

        return measurement

    def measurement_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        raise NotImplementedError(
            "This abstract method needs to be overwritten by the Subclass"
        )
