import numpy as np

from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    MeasurementModel,
    MixedNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import (
    AdditiveNoiseSystemModel,
    MixedNoiseSystemModel,
    SystemModel,
)
from nonlinear_estimation_toolbox.utils import ColumnVectors

"""
System Models
"""


class SysModel(SystemModel):
    def __init__(self):
        super(SysModel, self).__init__()
        self.sysMatrix = np.array([[3, -4], [0, 2]])

    def system_equation(self, state_samples, noise_samples):
        """
        Calculating this formula is desired:
        predictedStates = np.matmul(self.sysMatrix, stateSamples) + noiseSamples
        given 1 state sample and 1 noise sample, the first term (after matmul) has the shape (2,) and the second term noiseSamples has shape (2,1)
        if we add both, thanks to broadcasting, we get (2,2) (which is a totally incorrect calculation) where we desire (2,).
        Of course, we can fix it, but this requires ugly if-statements for handling multiple samples.
        Ideas on deal with this issue?
        - By design, stateSamples and noiseSamples should have the same dimension!
        """

        predicted_states = np.matmul(self.sysMatrix, state_samples) + noise_samples

        return predicted_states


class AddNoiseSysModel(AdditiveNoiseSystemModel):
    def __init__(self):
        super(AddNoiseSysModel, self).__init__()
        self.sysMatrix = np.array([[3, -4], [0, 2]])

    def system_equation(self, state_samples):
        predicted_states = np.matmul(self.sysMatrix, state_samples)
        return predicted_states


class MixedNoiseSysModel(MixedNoiseSystemModel):
    def __init__(self):
        super(MixedNoiseSysModel, self).__init__()
        self.sys_matrix = np.array([[3, -4], [0, 2]])

    def system_equation(self, state_samples, noise_samples=None):
        predicted_states = np.matmul(self.sys_matrix, state_samples) + noise_samples
        return predicted_states


"""
Measurement Models
"""


def _get_meas_matrix(state_decomp: bool):
    if state_decomp:
        return np.array([[3], [-0.5], [3]])
    else:
        return np.array([[3, -4], [0, 2], [3, 0]])


class MeasModel(MeasurementModel):
    def __init__(self, state_decomp=False):
        super(MeasModel, self).__init__()
        self.meas_matrix = _get_meas_matrix(state_decomp)

    def measurement_equation(self, state_samples, noise_samples):
        """
        Calculating this formula is desired:
        predictedStates = np.matmul(self.measMatrix, stateSamples) + noiseSamples
        given 1 state sample and 1 noise sample, the first term (after matmul) has the shape (2,) and the second term noiseSamples has shape (2,1)
        if we add both, thanks to broadcasting, we get (2,2) (which is a totally incorrect calculation) where we desire (2,).
        Of course, we can fix it, but this requires ugly if-statements for handling multiple samples.
        Ideas on deal with this issue?
        - By design, stateSamples and noiseSamples should have the same dimension!
        """

        measurements = np.matmul(self.meas_matrix, state_samples) + noise_samples

        return measurements


class AddNoiseMeasModel(AdditiveNoiseMeasurementModel):
    def __init__(self, state_decomp: bool = False):
        super(AddNoiseMeasModel, self).__init__()
        self.meas_matrix = _get_meas_matrix(state_decomp)

    def measurement_equation(self, state_samples):
        measurements = np.matmul(self.meas_matrix, state_samples)
        return measurements


class MixedNoiseMeasModel(MixedNoiseMeasurementModel):
    def __init__(self, state_decomp: bool = False):
        super(MixedNoiseMeasModel, self).__init__()
        self.meas_matrix = _get_meas_matrix(state_decomp)

    def measurement_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        return np.matmul(self.meas_matrix, state_samples) + noise_samples
