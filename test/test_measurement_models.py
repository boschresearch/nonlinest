import numpy as np
import numpy.testing as testing

from nonlinear_estimation_toolbox.distributions import Distribution, Uniform
from nonlinear_estimation_toolbox.measurement_models import LinearMeasurementModel
from nonlinear_estimation_toolbox.utils import ColumnVectors

from .models import AddNoiseMeasModel, MeasModel, MixedNoiseMeasModel


class TestMeasurementModel:
    def test_simulate(self):
        meas_model = MeasModel()
        a = ColumnVectors([0, 0, 0])
        b = ColumnVectors([1, 1, 1])
        meas_model.set_noise(Uniform(a=a, b=b))

        state = ColumnVectors([0.3, -np.pi])

        det_meas = np.matmul(meas_model.meas_matrix, state)

        measurement = meas_model.simulate(state)

        assert measurement.shape == (3, 1)
        assert np.all(measurement >= det_meas)
        assert np.all(measurement <= det_meas + 1)

    def test_derivative(self):
        meas_model = MeasModel()

        nominal_state = ColumnVectors([-3, 0.5])
        nominal_noise = ColumnVectors([1, -0.7, 2.3])

        state_jacobian, noise_jacobian = meas_model.derivative(
            nominal_state=nominal_state, nominal_noise=nominal_noise
        )

        testing.assert_almost_equal(
            actual=state_jacobian, desired=meas_model.meas_matrix, decimal=8
        )
        testing.assert_almost_equal(actual=noise_jacobian, desired=np.eye(3), decimal=8)


class TestAdditiveNoiseMeasurementModel:
    def test_simulate(self):
        meas_model = AddNoiseMeasModel()
        a = ColumnVectors([0, 0, 0])
        b = ColumnVectors([1, 1, 1])
        meas_model.set_noise(Uniform(a=a, b=b))

        state = ColumnVectors([0.3, -np.pi])

        det_meas = np.matmul(meas_model.meas_matrix, state)

        measurement = meas_model.simulate(state)

        assert measurement.shape == (3, 1)
        assert np.all(measurement >= det_meas)
        assert np.all(measurement <= det_meas + 1)

    def test_derivative(self):
        meas_model = AddNoiseMeasModel()

        nominal_state = ColumnVectors([-3, 0.5])

        state_jacobian = meas_model.derivative(nominal_state=nominal_state)

        testing.assert_almost_equal(
            actual=state_jacobian, desired=meas_model.meas_matrix, decimal=8
        )


class TestLinearMeasurementModel:
    def test_constructor_default(self):
        meas_model = LinearMeasurementModel()

        assert meas_model.meas_matrix is None
        assert isinstance(meas_model.noise, Distribution)

    def test_constructor_sys_matrix(self):
        meas_matrix = np.ones(shape=(2, 3))
        meas_model = LinearMeasurementModel(meas_matrix=meas_matrix)

        testing.assert_array_equal(x=meas_model.meas_matrix, y=meas_matrix)
        assert isinstance(meas_model.noise, Distribution)

    def test_set_measurement_matrix(self):
        meas_matrix = np.eye(3)
        meas_model = LinearMeasurementModel()

        meas_model.set_measurement_matrix(meas_matrix=meas_matrix)

        testing.assert_array_equal(x=meas_model.meas_matrix, y=meas_matrix)
        assert isinstance(meas_model.noise, Distribution)

    def test_set_measurement_matrix_empty(self):
        meas_matrix = None
        meas_model = LinearMeasurementModel()

        meas_model.set_measurement_matrix(meas_matrix=meas_matrix)

        testing.assert_array_equal(x=meas_model.meas_matrix, y=None)
        assert isinstance(meas_model.noise, Distribution)

    def test_simulate(self):
        meas_matrix = np.array([[3, -4], [np.pi / 4, 0], [0.5, 2]])
        state = ColumnVectors([0.3, -np.pi])

        meas_model = LinearMeasurementModel(meas_matrix=meas_matrix)
        a = ColumnVectors([0, 0, 0])
        b = ColumnVectors([1, 1, 1])
        meas_model.set_noise(Uniform(a=a, b=b))

        det_meas = np.matmul(meas_matrix, state)

        measurement = meas_model.simulate(state)

        assert measurement.shape == (3, 1)
        assert np.all(measurement >= det_meas)
        assert np.all(measurement <= det_meas + 1)

    def test_derivative(self):
        meas_matrix = np.array([[1, 1, 0], [0, 1, 2]])

        meas_model = LinearMeasurementModel(meas_matrix=meas_matrix)
        state_jacobian = meas_model.derivative(np.array([3, -2, 1]))

        testing.assert_array_equal(state_jacobian, meas_matrix)
        testing.assert_raises_regex(
            AssertionError,
            "LinearMeasurementModel:IncompatibleMeasurementMatrix",
            meas_model.derivative,
            np.array([3, -2]),
        )

        meas_model = LinearMeasurementModel()
        state_jacobian = meas_model.derivative(np.array([3, -2, 1]))
        testing.assert_array_equal(state_jacobian, np.eye(3))


class TestMixedNoiseMeasurementModel:
    def test_simulate(self):
        meas_model = MixedNoiseMeasModel()
        meas_model.set_additive_noise(
            Uniform(ColumnVectors([0, 0, 0]), ColumnVectors([1, 1, 1]))
        )
        meas_model.set_noise(
            Uniform(ColumnVectors([0, 0, 0]), ColumnVectors([1, 1, 1]))
        )

        state = ColumnVectors([0.3, -np.pi])
        det_meas = np.matmul(meas_model.meas_matrix, state)

        measurement = meas_model.simulate(state)

        assert measurement.shape == (3, 1)
        assert np.all(measurement >= det_meas)
        assert np.all(measurement <= det_meas + 2)

    def test_derivative(self):
        meas_model = MixedNoiseMeasModel()

        nominal_state = ColumnVectors([-3, 0.5])
        nominal_noise = ColumnVectors([1, -0.7, 2.3])

        state_jacobian, noise_jacobian = meas_model.derivative(
            nominal_state, nominal_noise
        )

        assert np.allclose(state_jacobian, meas_model.meas_matrix, atol=1e-8)
        assert np.allclose(noise_jacobian, np.eye(3), atol=1e-8)
