import numpy as np
import numpy.testing as testing

from nonlinear_estimation_toolbox.distributions import Distribution, Uniform
from nonlinear_estimation_toolbox.system_models import LinearSystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors

from .models import AddNoiseSysModel, MixedNoiseSysModel, SysModel


class TestSystemModel:
    def test_simulate(self):
        sys_model = SysModel()
        a = ColumnVectors([0, 0])
        b = ColumnVectors([1, 1])
        sys_model.set_noise(Uniform(a=a, b=b))

        state = ColumnVectors([0.3, -np.pi])

        det_sim_state = np.matmul(sys_model.sysMatrix, state)

        sim_state = sys_model.simulate(state)

        assert sim_state.shape == (2, 1)
        assert np.all(sim_state >= det_sim_state)
        assert np.all(sim_state <= det_sim_state + 1)

    def test_derivative(self):
        sys_model = SysModel()

        nominal_state = ColumnVectors([-3, 0.5])
        nominal_noise = ColumnVectors([1, -0.7])

        state_jacobian, noise_jacobian = sys_model.derivative(
            nominal_state=nominal_state, nominal_noise=nominal_noise
        )

        testing.assert_almost_equal(
            actual=state_jacobian, desired=sys_model.sysMatrix, decimal=8
        )
        testing.assert_almost_equal(actual=noise_jacobian, desired=np.eye(2), decimal=8)


class TestAdditiveNoiseSystemModel:
    def test_simulate(self):
        sys_model = AddNoiseSysModel()
        a = ColumnVectors([0, 0])
        b = ColumnVectors([1, 1])
        sys_model.set_noise(Uniform(a=a, b=b))

        state = ColumnVectors([0.3, -np.pi])

        det_sim_state = np.matmul(sys_model.sysMatrix, state)

        sim_state = sys_model.simulate(state)

        assert sim_state.shape == (2, 1)
        assert np.all(sim_state >= det_sim_state)
        assert np.all(sim_state <= det_sim_state + 1)

    def test_derivative(self):
        sys_model = AddNoiseSysModel()

        nominal_state = ColumnVectors([-3, 0.5])

        state_jacobian = sys_model.derivative(nominal_state=nominal_state)

        testing.assert_almost_equal(
            actual=state_jacobian, desired=sys_model.sysMatrix, decimal=8
        )


class TestLinearSystemModel:
    def test_constructor_default(self):
        sys_model = LinearSystemModel()

        assert sys_model.sys_matrix is None
        assert sys_model.sys_input is None
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_constructor_sys_matrix(self):
        sys_matrix = np.eye(2)
        sys_model = LinearSystemModel(sys_matrix=sys_matrix)

        testing.assert_array_equal(x=sys_model.sys_matrix, y=sys_matrix)
        assert sys_model.sys_input is None
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_constructor_sys_matrix_sys_noise_matrix(self):
        sys_matrix = np.eye(2)
        sys_noise_matrix = 2 * np.ones(shape=(2, 3))
        sys_model = LinearSystemModel(
            sys_matrix=sys_matrix, sys_noise_matrix=sys_noise_matrix
        )

        testing.assert_array_equal(x=sys_model.sys_matrix, y=sys_matrix)
        assert sys_model.sys_input is None
        testing.assert_array_equal(x=sys_model.sys_noise_matrix, y=sys_noise_matrix)
        assert isinstance(sys_model.noise, Distribution)

    def test_set_system_matrix(self):
        sys_matrix = np.eye(3)
        sys_model = LinearSystemModel()

        sys_model.set_system_matrix(sys_matrix=sys_matrix)

        testing.assert_array_equal(x=sys_model.sys_matrix, y=sys_matrix)
        assert sys_model.sys_input is None
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_set_system_matrix_empty(self):
        sys_matrix = None
        sys_model = LinearSystemModel()

        sys_model.set_system_matrix(sys_matrix=sys_matrix)

        assert sys_model.sys_matrix is None
        assert sys_model.sys_input is None
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_set_system_input(self):
        sys_input = ColumnVectors([1, 2, -3])
        sys_model = LinearSystemModel()

        sys_model.set_system_input(sys_input=sys_input)

        assert sys_model.sys_matrix is None
        testing.assert_array_equal(x=sys_model.sys_input, y=sys_input)
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_set_system_input_empty(self):
        sys_input = None
        sys_model = LinearSystemModel()

        sys_model.set_system_input(sys_input=sys_input)

        assert sys_model.sys_matrix is None
        assert sys_model.sys_input is None
        assert sys_model.sys_noise_matrix is None
        assert isinstance(sys_model.noise, Distribution)

    def test_set_system_noise_matrix(self):
        sys_noise_matrix = np.ones(shape=(3, 2))
        sys_model = LinearSystemModel()

        sys_model.set_system_noise_matrix(sys_noise_matrix=sys_noise_matrix)

        assert sys_model.sys_matrix is None
        assert sys_model.sys_input is None
        testing.assert_array_equal(x=sys_model.sys_noise_matrix, y=sys_noise_matrix)
        assert isinstance(sys_model.noise, Distribution)

    def test_simulate(self):
        sys_matrix = np.array([[3, -4], [0, 2]])
        state = ColumnVectors([0.3, -np.pi])

        sys_model = LinearSystemModel(sys_matrix=sys_matrix)
        a = ColumnVectors([0, 0])
        b = ColumnVectors([1, 1])
        sys_model.set_noise(Uniform(a=a, b=b))

        det_sim_state = np.matmul(sys_matrix, state)

        sim_state = sys_model.simulate(state)

        assert sim_state.shape == (2, 1)
        assert np.all(sim_state >= det_sim_state)
        assert np.all(sim_state <= det_sim_state + 1)

    def test_derivative(self):
        sys_matrix = np.array([[1, 1, 2], [0, 1, 0], [2, 1, 2]])

        sys_noise_matrix = np.array([[1, 0], [0, 1], [0, 1]])

        sys_model = LinearSystemModel(
            sys_matrix=sys_matrix, sys_noise_matrix=sys_noise_matrix
        )

        state_jacobian, noise_jacobian = sys_model.derivative(
            ColumnVectors([3, -2, 1]), ColumnVectors([0, 0])
        )

        testing.assert_array_equal(state_jacobian, sys_matrix)
        testing.assert_array_equal(noise_jacobian, sys_noise_matrix)
        testing.assert_raises_regex(
            AssertionError,
            "LinearSystemModel:IncompatibleSystemMatrix",
            sys_model.derivative,
            ColumnVectors([3, -2]),
            ColumnVectors([0, 0]),
        )
        testing.assert_raises_regex(
            AssertionError,
            "LinearSystemModel:IncompatibleSystemNoiseMatrix",
            sys_model.derivative,
            ColumnVectors([3, -2, 1]),
            ColumnVectors([0]),
        )

        sys_model = LinearSystemModel(
            sys_matrix=None, sys_noise_matrix=sys_noise_matrix
        )
        state_jacobian, noise_jacobian = sys_model.derivative(
            ColumnVectors([3, -2, 1]), ColumnVectors([0, 0])
        )
        testing.assert_array_equal(state_jacobian, np.eye(3))
        testing.assert_array_equal(noise_jacobian, sys_noise_matrix)
        testing.assert_raises_regex(
            AssertionError,
            "LinearSystemModel:IncompatibleSystemNoiseMatrix",
            sys_model.derivative,
            ColumnVectors([3, -2]),
            ColumnVectors([0, 0]),
        )

        sys_model = LinearSystemModel()
        state_jacobian, noise_jacobian = sys_model.derivative(
            ColumnVectors([3, -2, 1]), ColumnVectors([0, 0, 0])
        )
        testing.assert_array_equal(state_jacobian, np.eye(3))
        testing.assert_array_equal(noise_jacobian, np.eye(3))
        testing.assert_raises_regex(
            AssertionError,
            "LinearSystemModel:IncompatibleSystemNoise",
            sys_model.derivative,
            np.array([3, -2, 1]),
            np.array([0]),
        )


class TestMixedNoiseSystemModel:
    def test_simulate(self):
        sys_model = MixedNoiseSysModel()
        a = ColumnVectors([0, 0])
        b = ColumnVectors([1, 1])
        sys_model.set_noise(Uniform(a=a, b=b))
        sys_model.set_additive_noise(Uniform(a=a, b=b))

        state = ColumnVectors([0.3, -np.pi])

        det_sim_state = np.matmul(sys_model.sys_matrix, state)

        sim_state = sys_model.simulate(state)

        assert sim_state.shape == (2, 1)
        assert np.all(sim_state >= det_sim_state)
        assert np.all(sim_state <= det_sim_state + 2)

    def test_derivative(self):
        sys_model = MixedNoiseSysModel()

        nominal_state = ColumnVectors([-3, 0.5])
        nominal_noise = ColumnVectors([1, -0.7])

        state_jacobian, noise_jacobian = sys_model.derivative(
            nominal_state=nominal_state, nominal_noise=nominal_noise
        )

        testing.assert_almost_equal(
            actual=state_jacobian, desired=sys_model.sys_matrix, decimal=8
        )
        testing.assert_almost_equal(actual=noise_jacobian, desired=np.eye(2), decimal=8)
