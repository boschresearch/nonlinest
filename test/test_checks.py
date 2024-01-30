import numpy as np
import pytest

import nonlinear_estimation_toolbox.checks as checks
from nonlinear_estimation_toolbox.utils import ColumnVectors


class TestChecks:
    def test_is_vec(self):
        assert checks.is_single_vec(ColumnVectors([1, 2, 3]))
        assert checks.is_single_vec(ColumnVectors([3]))
        # assert (not checks.isVec(np.array([])))
        assert not checks.is_single_vec(ColumnVectors([[1, 2]]))
        assert checks.is_single_vec(ColumnVectors([[1], [2]]))
        assert not checks.is_single_vec(ColumnVectors([[1, 2], [3, 4]]))

    def test_is_mat(self):
        pass

    def test_is_fixed_row_mat(self):
        pass

    def is_fixed_col_mat(self):
        pass

    def test_is_square_mat(self):
        pass

    def test_is_valid_covariance_matrix(self):
        m = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

        m_sqrt = checks.compute_cholesky_if_valid_covariance(m)
        m_sqrt_true = np.array([[2, 0, 0], [6, 1, 0], [-8, 5, 3]])

        assert m_sqrt.shape == m_sqrt_true.shape
        assert np.allclose(m_sqrt, m_sqrt_true)

    def test_is_valid_covariance_matrix_not_square(self):
        m = np.array([[1, 2]])
        msg = "test-message-location"

        with pytest.raises(ValueError) as error_info:
            checks.compute_cholesky_if_valid_covariance(m, msg)

        assert msg in str(error_info.value)
        assert "square" in str(error_info.value)

    def test_is_valid_covariance_matrix_not_pos_def(self):
        m = np.array([[4, 12, -16], [12, 37, -43], [-16, -42, 98]])
        msg = "test-message-location"

        with pytest.raises(ValueError) as error_info:
            checks.compute_cholesky_if_valid_covariance(m, msg)

        assert msg in str(error_info.value)
        assert "positive definite" in str(error_info.value)

    def test_compute_choleksy_3d(self):
        m = np.array([[[0.1, 2], [0, 0.5]], [[0, 0.5], [3, 1.2]]])

        true_sqrt_0 = np.linalg.cholesky(m[..., 0])[..., None]
        true_sqrt_1 = np.linalg.cholesky(m[..., 1])[..., None]
        true_sqrt = np.concatenate([true_sqrt_0, true_sqrt_1], axis=2)

        cov_sqrt = checks.compute_cholesky_if_valid_3d_covariance(m)

        assert np.allclose(true_sqrt, cov_sqrt)
