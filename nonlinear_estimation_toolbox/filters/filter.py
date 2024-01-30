import copy
from time import perf_counter
from typing import Callable, Tuple, TypeVar, Union

import numpy as np

from nonlinear_estimation_toolbox import checks
from nonlinear_estimation_toolbox.distributions import Distribution
from nonlinear_estimation_toolbox.measurement_models import Likelihood, MeasurementModel
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors

TFilter = TypeVar("TFilter", bound="Filter")


class Filter(object):
    """Abstract base class for a filter."""

    def __init__(self, name: str = "") -> None:
        """
        Constructor

        :param name: Name to distinguish different filter instances
        """
        if not isinstance(name, str):
            raise TypeError("Filter: InvalidFilterName: name must be a string.")
        self.name = name
        self.color = (1.0, 0.0, 0.0)
        self.dim_state = 0

    def copy(self) -> TFilter:
        """
        Copy the filter instance.
        Note: Use copy_with_name() instead to assign a new name to the current filter.

        :return: Copied filter
        """
        return copy.deepcopy(self)

    def copy_with_name(self, new_name: str) -> TFilter:
        """
        Copy the filter instance and assign a new name to the current filter.

        :param new_name: New name for current filter.
        :return: Copied filter with old name.
        """
        if not isinstance(new_name, str):
            raise TypeError("Filter: InvalidFilterName: name must be a string.")
        copied_object = self.copy()
        self.name = new_name
        return copied_object

    def get_name(self) -> str:
        """
        Getter for name attribute

        :return: Name which is set in the constructor
        """
        return self.name

    def set_color(self, color: Tuple[float, float, float]) -> None:
        """
        Setter for color attribute

        :param color: Color attribute which can be used for e.g. plotting
        """
        self.color = color

    def get_color(self) -> Tuple[float, float, float]:
        """
        Getter for color attribute

        :return: Color which can be set by set_color().
        """
        return self.color

    def get_state_dimension(self) -> int:
        """
        Getter for state dimension attribute

        :return: Dimension of the state space.
        """
        return self.dim_state

    def set_state(self, state: Distribution) -> None:
        """
        Setter for state distribution.
        This function is mainly used to set an initial state.

        :param state: New state distribution.
        """
        if not isinstance(state, Distribution):
            raise TypeError(
                "InvalidSystemState: state must be a subclass of Distribution."
            )
        self.dim_state = state.get_dimension()
        self._perform_set_state(state)

    def set_state_mean_and_covariance(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray = None,
    ) -> None:
        """
        Setter for state distribution using mean and covariance.
        It is intended to directly modify the existing distribution without
        creating an intermediate Distribution instance.

        :param state_mean: New state mean vector.
        :param state_cov: New positive definite state covariance.
        :param state_cov_sqrt: New lower Cholesky decomposition of state covariance.
        """
        if state_cov_sqrt is None:
            state_cov_sqrt = np.linalg.cholesky(state_cov)
        self._perform_set_state_mean_and_covariance(
            state_mean, state_cov, state_cov_sqrt
        )
        self.dim_state = len(state_mean)

    def predict(self, sys_model: SystemModel) -> None:
        """
        Perform a state prediction.

        :param sys_model: System model that describes the temporal behaviour of the system state.
        """
        self._perform_prediction(sys_model)
        # TODO: error handling. See Toolbox/Interfaces/Filters/Filter.m

    def predict_timed(self, sys_model: SystemModel) -> float:
        """
        Perform a timed state prediction.

        :param sys_model: System model that describes the temporal behaviour of the system state.
        :return: Time taken for prediction in seconds.
        """
        return self._call_timed(self.predict, sys_model)

    def update(
        self, meas_model: Union[MeasurementModel, Likelihood], measurement: np.ndarray
    ) -> None:
        """
        Perform an update step.

        :param meas_model: Measurement model which describes the relation between system state and measurement.
        :param measurement: Measurement data to be processed.
        """
        self._perform_update(meas_model, measurement)

    def update_timed(
        self, meas_model: Union[MeasurementModel, Likelihood], measurement: np.ndarray
    ) -> float:
        """
        Perform a timed state update.

        :param meas_model: Measurement model which describes the relation between system state and measurement.
        :param measurement: Measurement data to be processed.
        :return: Time taken for update in seconds.
        """
        return self._call_timed(self.update, meas_model, measurement)

    def step(
        self,
        sys_model: SystemModel,
        meas_model: MeasurementModel,
        measurement: np.ndarray,
    ) -> None:
        """
        Perform state prediction and update.

        :param sys_model: System model that describes the temporal behaviour of the system state.
        :param meas_model: Measurement model which describes the relation between system state and measurement.
        :param measurement: Measurement data to be processed.
        """
        self._perform_step(sys_model, meas_model, measurement)

    def step_timed(
        self,
        sys_model: SystemModel,
        meas_model: MeasurementModel,
        measurement: np.ndarray,
    ) -> float:
        """
        Perform state prediction and update.

        :param sys_model: System model that describes the temporal behaviour of the system state.
        :param meas_model: Measurement model which describes the relation between system state and measurement.
        :param measurement: Measurement data to be processed.
        :return: Time taken for whole step in seconds.
        """
        return self._call_timed(self.step, sys_model, meas_model, measurement)

    def get_state(self) -> Distribution:
        """
        Abstract getter for state distribution.
        """
        raise NotImplementedError

    def get_state_mean_and_covariance(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Abstract getter for state distribution mean and covariance

        :return: Tuple of state mean vector, covariance matrix and lower Cholesky decomposition of covariance matrix
        """
        raise NotImplementedError

    def _call_timed(self, func: Callable[..., None], *args) -> float:
        tic = perf_counter()
        func(*args)
        toc = perf_counter()
        return toc - tic

    def _perform_set_state(self, state: Distribution) -> None:
        raise NotImplementedError

    def _perform_set_state_mean_and_covariance(
        self,
        state_mean: ColumnVectors,
        state_cov: np.ndarray,
        state_cov_sqrt: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _perform_prediction(self, sys_model: SystemModel) -> None:
        raise NotImplementedError

    def _perform_update(
        self, meas_model: MeasurementModel, measurement: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _perform_step(
        self,
        sys_model: SystemModel,
        meas_model: MeasurementModel,
        measurement: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _check_measurement_vector(measurement: np.ndarray) -> None:
        # TODO: replace with check: Toolbox/Misc/Checks.m
        if not isinstance(measurement, np.ndarray):
            raise TypeError("InvalidMeasurement: measurement must be a numpy array.")

    def _check_predicted_state_samples(
        self, samples: np.ndarray, num_samples: int
    ) -> None:
        # TODO: replace with check: Toolbox/Misc/Checks.m
        if not isinstance(samples, np.ndarray):
            raise TypeError(
                "InvalidPredictedStateSamples: samples must be a numpy array."
            )
        if not len(samples.shape) == 2:
            raise TypeError("InvalidPredictedStateSamples: samples must be a matrix.")
        if not samples.shape[0] == self.dim_state:
            raise ValueError(
                "InvalidPredictedStateSamples: incorrect samples dimension."
            )
        if not samples.shape[1] == num_samples:
            raise ValueError(
                "InvalidPredictedStateSamples: incorrect number of samples."
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError(
                "InvalidPredictedStateSamples: At least one predicted state sample contains NaN or Inf."
            )

    @staticmethod
    def _check_computed_measurements(
        measurements: np.ndarray, dim_meas: int, num_measurements: int
    ) -> None:
        # TODO: replace with check: Toolbox/Misc/Checks.m
        if not isinstance(measurements, np.ndarray):
            raise TypeError("InvalidMeasurements: measurements must be a numpy array.")
        if not len(measurements.shape) == 2:
            raise TypeError("InvalidMeasurements: measurements must be a matrix.")
        if not measurements.shape[0] == dim_meas:
            raise ValueError("InvalidMeasurements: incorrect measurements dimension.")
        if not measurements.shape[1] == num_measurements:
            raise ValueError("InvalidMeasurements: incorrect number of measurements.")
        if not np.all(np.isfinite(measurements)):
            raise ValueError(
                "InvalidMeasurements: At least one measurement contains NaN or Inf."
            )

    @staticmethod
    def _check_log_likelihood_evaluations(
        values: np.ndarray, num_evaluations: int
    ) -> None:
        # TODO: use checks:  Toolbox/Misc/Checks.m
        if not isinstance(values, np.ndarray):
            raise TypeError(
                "InvalidLogLikelihoodEvaluations: Logarithmic likelihood evaluations must be a numpy array."
            )

        if not checks.is_single_row_vec(values, num_evaluations):
            raise ValueError(
                "InvalidLogLikelihoodEvaluations: Logarithmic likelihood evaluations must be a row vector of shape 1x{}".format(
                    num_evaluations
                )
            )

        if np.isnan(values).any() or np.isinf(values[values > 0.0]).any():
            raise ValueError(
                "InvalidLogLikelihoodEvaluations: At least one logarithmic likelihood evaluation is NaN or +Inf."
            )

    def _check_additive_system_noise(self, dim_noise: int) -> None:
        if not dim_noise == self.dim_state:
            raise ValueError(
                "InvalidAdditiveSystemNoise: System state and additive system noise with incompatible dimensions."
            )

    @staticmethod
    def _check_additive_measurement_noise(dim_meas: int, dim_noise: int) -> None:
        if not dim_noise == dim_meas:
            raise ValueError(
                "InvalidAdditiveMeasurementNoise: Measurement and additive measurement noise with incompatible dimensions."
            )

    @staticmethod
    def _check_state_jacobian(
        state_jacobian: np.ndarray, dim_output: int, dim_state: int
    ) -> None:
        if not checks.is_mat(state_jacobian, dim_output, dim_state):
            raise ValueError(
                "InvalidStateJacobian: State Jacobian must be a matrix of dimension {}x{}.".format(
                    dim_output, dim_state
                )
            )

    @staticmethod
    def _check_noise_jacobian(
        noise_jacobian: np.ndarray, dim_output: int, dim_noise: int
    ) -> None:
        if not checks.is_mat(noise_jacobian, dim_output, dim_noise):
            raise ValueError(
                "InvalidNoiseJacobian: Noise Jacobian has to be a matrix of dimension {}x{}.".format(
                    dim_output, dim_noise
                )
            )
