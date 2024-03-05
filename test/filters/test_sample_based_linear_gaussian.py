import numpy as np
import pytest

from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)


class TestSampleBasedLinearGaussian:
    @pytest.fixture
    def zero_mean_state_samples(self):
        zero_mean_state_samples = np.array(
            [
                [-1, -2, 2, 1],
                [-1, -2, 2, 1],
                [-1, -2, 2, 1],
            ]
        )
        return zero_mean_state_samples

    @pytest.fixture
    def zero_mean_noise_samples(self):
        zero_mean_noise_samples = np.array([[-0.5, -1, 1, 0.5]])
        return zero_mean_noise_samples

    def test_get_meas_model_moments_negative(
        self, zero_mean_state_samples, zero_mean_noise_samples
    ):
        """Test if negative weights are handled correctly"""

        h_samples = np.array([[-2, 2, -2, 2], [2, -2, 2, -2]])

        weights = np.array([[-0.5, 0.5, -0.5, 0.5]])

        filter = SampleBasedLinearGaussianFilter()
        (
            h_mean,
            h_cov,
            state_h_cross_cov,
            h_noise_cross_cov,
        ) = filter.get_meas_model_moments(
            h_samples, weights, zero_mean_state_samples, zero_mean_noise_samples
        )

        assert h_mean.shape == (2, 1)
        assert h_cov.shape == (2, 2)
        assert state_h_cross_cov.shape == (3, 2)
        assert h_noise_cross_cov.shape == (2, 1)

        true_h_mean = np.array([[4], [-4]])
        assert np.allclose(h_mean, true_h_mean)

    def test_get_meas_model_moments_scalar(
        self, zero_mean_state_samples, zero_mean_noise_samples
    ):
        """Test if scalar weights are handled correctly"""

        h_samples = np.array([[2, 2, 2, 2], [-2, -2, -2, -2]])

        weights = 0.5

        filter = SampleBasedLinearGaussianFilter()
        (
            h_mean,
            h_cov,
            state_h_cross_cov,
            h_noise_cross_cov,
        ) = filter.get_meas_model_moments(
            h_samples, weights, zero_mean_state_samples, zero_mean_noise_samples
        )

        assert h_mean.shape == (2, 1)
        assert h_cov.shape == (2, 2)
        assert state_h_cross_cov.shape == (3, 2)
        assert h_noise_cross_cov.shape == (2, 1)

        true_h_mean = np.array([[4], [-4]])
        assert np.allclose(h_mean, true_h_mean)
