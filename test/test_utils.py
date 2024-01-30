import numpy as np
import pytest

from nonlinear_estimation_toolbox import utils
from nonlinear_estimation_toolbox.utils import ColumnVectors


class TestUtilsColumnVectorsClass:
    list_vector = [1, 2, 3]
    numpy_row_vector = np.array(list_vector)
    numpy_column_vector = numpy_row_vector.T
    vectors_list = [2.4, list_vector, numpy_row_vector, numpy_row_vector]

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_initialization_ok(value):
        assert isinstance(ColumnVectors(value), np.ndarray)

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_dimension_ok(value):
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape((1, 1))
        assert ColumnVectors(value).shape == (len(value), 1)

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_copy(value):
        v = ColumnVectors(value)
        v_copy = v.copy()
        v[0] -= 1
        assert v[0] != v_copy[0]

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_inner_outer_product(value):
        v = ColumnVectors(value)
        assert (v + v).shape == v.shape
        assert (v * v).shape == v.shape
        assert (v.T.dot(v)).shape == (1, 1)
        assert (v.dot(v.T)).shape == (v.shape[0], v.shape[0])

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_matrix_product(value):
        v = ColumnVectors(value)
        A = np.random.randn(24, v.shape[0])
        assert (A.dot(v)).shape == (A.shape[0], v.shape[1])

    @staticmethod
    @pytest.mark.parametrize("value", vectors_list)
    def test_returns(value):
        v = ColumnVectors(value)
        assert isinstance(v.as_list(), list)
        assert isinstance(v.as_numpy(), np.ndarray)


class TestResampling:
    @pytest.fixture
    def samples(self):
        samples = np.hstack((np.zeros((3, 1)), 2 * np.eye(3), -2 * np.eye(3)))
        samples = samples - 4
        return samples

    @pytest.fixture
    def weights(self):
        return np.array([[2, 1, 1, 1, 1, 1, 1]]) / 8

    @pytest.fixture
    def cum_weights(self, weights):
        return np.cumsum(weights)

    def test_random_resampling(self, samples, weights):
        num_samples = 13
        rnd_samples, _ = utils.random_resampling(samples, weights, num_samples)
        assert rnd_samples.shape == (3, num_samples)

    def test_systematic_resampling(self, samples, cum_weights):
        num_samples = 13
        rnd_samples, _ = utils.systematic_resampling(samples, cum_weights, num_samples)
        assert rnd_samples.shape == (3, num_samples)


class TestUtilsRndOrthogonalMatrix:
    @staticmethod
    @pytest.mark.parametrize("dim", list(range(1, 12)))
    def test_rnd_orthogonal_matrix(dim: int):
        rnd_mat = utils.rnd_orthogonal_matrix(dim)
        identity = rnd_mat.dot(rnd_mat.T)

        np.testing.assert_allclose(identity, np.eye(dim), atol=1e-7)
