import numpy as np
import pytest

import nonlinear_estimation_toolbox.distributions as distributions
import nonlinear_estimation_toolbox.gaussian_sampling as gaussian_sampling
import nonlinear_estimation_toolbox.utils as utils


@pytest.fixture
def gaussian_1d():
    return distributions.Gaussian(utils.ColumnVectors(-5), utils.ColumnVectors(3))


@pytest.fixture
def gaussian_3d():
    return distributions.Gaussian(
        utils.ColumnVectors([2, -1, 0.5]),
        10 * np.array([[2, -0.5, 0], [-0.5, 1.3, 0.5], [0, 0.5, np.sqrt(2)]]),
    )


class TestGaussianSampling:
    def _test_get_std_normal_samples(
        self, g: gaussian_sampling.GaussianSampling, dim, true_num_samples, tol
    ):
        samples, weights, num_samples = g.get_std_normal_samples(dim)

        assert num_samples == true_num_samples
        assert samples.shape == (dim, num_samples)
        assert weights.shape == (1, num_samples)
        assert np.allclose(np.sum(weights), 1, atol=tol)

        mean, cov = utils.get_sample_mean_and_covariance(samples, weights)

        assert np.allclose(mean, np.zeros([dim, 1]), atol=tol)
        assert np.allclose(cov, np.eye(dim), atol=tol)

    def _test_get_std_normal_samples_equally_weighted(
        self, g: gaussian_sampling.GaussianSampling, dim, trueNumSamples, tol
    ):
        samples, weights, numSamples = g.get_std_normal_samples(dim)

        assert numSamples == trueNumSamples
        assert samples.shape == (dim, numSamples)
        assert weights == 1 / numSamples

        mean, cov = utils.get_sample_mean_and_covariance(
            samples
        )  # todo should we pass the weights?

        np.testing.assert_allclose(mean, np.zeros([dim, 1]), atol=tol)
        np.testing.assert_allclose(cov, np.eye(dim), atol=tol)

    def _testGetSamples(
        self, g: gaussian_sampling.GaussianSampling, gaussian, true_num_samples, tol
    ):
        true_mean, true_cov, _ = gaussian.get_mean_and_covariance()
        dim = gaussian.get_dimension()

        samples, weights, num_samples = g.get_samples(gaussian)

        assert num_samples == true_num_samples
        assert samples.shape == (dim, num_samples)
        assert weights.shape == (1, num_samples)
        np.testing.assert_allclose(
            np.sum(weights), 1, atol=tol
        )  # this is hard coded as 1E-8 by Jannik

        mean, cov = utils.get_sample_mean_and_covariance(samples, weights)

        np.testing.assert_allclose(mean, true_mean, atol=tol)
        np.testing.assert_allclose(cov, true_cov, atol=tol)

    def _test_get_samples_equally_weighted(
        self, g: gaussian_sampling.GaussianSampling, gaussian, true_num_samples, tol
    ):
        true_mean, true_cov, _ = gaussian.get_mean_and_covariance()
        dim = gaussian.get_dimension()

        samples, weights, num_samples = g.get_samples(gaussian)

        assert num_samples == true_num_samples
        assert samples.shape == (dim, num_samples)
        assert weights == 1 / num_samples

        mean, cov = utils.get_sample_mean_and_covariance(
            samples
        )  # todo should we pass the weights?

        np.testing.assert_allclose(mean, true_mean, atol=tol)
        np.testing.assert_allclose(cov, true_cov, atol=tol)


class TestGaussianSamplingUKF(TestGaussianSampling):
    def test_constructor(self):
        g = gaussian_sampling.GaussianSamplingUKF()
        assert g.get_sample_scaling() == 0.5

    def test_set_sample_scaling(self):
        g = gaussian_sampling.GaussianSamplingUKF()
        g.set_sample_scaling(1)
        assert g.get_sample_scaling() == 1

    def test_default_config(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingUKF()
        tol = 1e-12

        self._test_get_std_normal_samples_equally_weighted(g, 1, 3, tol)
        self._test_get_std_normal_samples_equally_weighted(g, 5, 11, tol)
        self._test_get_std_normal_samples_equally_weighted(g, 10, 21, tol)

        self._test_get_samples_equally_weighted(g, gaussian_1d, 3, tol)
        self._test_get_samples_equally_weighted(g, gaussian_3d, 7, tol)

    def test_sample_scaling_one(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingUKF()
        tol = 1e-12
        g.set_sample_scaling(1)

        self._test_get_std_normal_samples(g, 1, 3, tol)
        self._test_get_std_normal_samples(g, 5, 11, tol)
        self._test_get_std_normal_samples(g, 10, 21, tol)

        self._testGetSamples(g, gaussian_1d, 3, tol)
        self._testGetSamples(g, gaussian_3d, 7, tol)

    def test_sample_scaling_zero(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingUKF()
        tol = 1e-12
        g.set_sample_scaling(0)

        self._test_get_std_normal_samples(g, 1, 3, tol)
        self._test_get_std_normal_samples(g, 5, 11, tol)
        self._test_get_std_normal_samples(g, 10, 21, tol)

        self._testGetSamples(g, gaussian_1d, 3, tol)
        self._testGetSamples(g, gaussian_3d, 7, tol)


class TestGaussianSamplingRUKF(TestGaussianSampling):
    def test_constructor(self):
        g = gaussian_sampling.GaussianSamplingRUKF()
        assert g.get_numiterations() == 5

    def test_set_num_iterations(self):
        g = gaussian_sampling.GaussianSamplingRUKF()
        g.set_numiterations(10)
        assert g.get_numiterations() == 10

    def test_default_config(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingRUKF()
        tol = 1e-12

        self._test_get_std_normal_samples(g, 1, 11, tol)
        self._test_get_std_normal_samples(g, 5, 51, tol)
        self._test_get_std_normal_samples(g, 10, 101, tol)

        self._testGetSamples(g, gaussian_1d, 11, tol)
        self._testGetSamples(g, gaussian_3d, 31, tol)

    def test_num_iterations(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingRUKF()
        tol = 1e-12
        g.set_numiterations(10)

        self._test_get_std_normal_samples(g, 1, 21, tol)
        self._test_get_std_normal_samples(g, 5, 101, tol)
        self._test_get_std_normal_samples(g, 10, 201, tol)

        self._testGetSamples(g, gaussian_1d, 21, tol)
        self._testGetSamples(g, gaussian_3d, 61, tol)


class TestGaussianSamplingCKF(TestGaussianSampling):
    def test_default_config(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingCKF()
        tol = 1e-12

        self._test_get_std_normal_samples(g, 1, 3, tol)
        self._test_get_std_normal_samples(g, 5, 51, tol)
        self._test_get_std_normal_samples(g, 10, 201, tol)

        self._testGetSamples(g, gaussian_1d, 3, tol)
        self._testGetSamples(g, gaussian_3d, 19, tol)


class TestGaussianSamplingGHQ(TestGaussianSampling):
    def test_constructor(self):
        g = gaussian_sampling.GaussianSamplingGHQ()
        assert g.get_num_quadature_points() == 2

    def test_set_num_quadature_points(self):
        g = gaussian_sampling.GaussianSamplingGHQ()
        g.set_num_quadrature_points(4)
        assert g.get_num_quadature_points() == 4

    def test_default_config(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingGHQ()
        tol = 1e-12

        self._test_get_std_normal_samples(g, 1, 2, tol)
        self._test_get_std_normal_samples(g, 5, 32, tol)
        self._test_get_std_normal_samples(g, 10, 1024, tol)

        self._testGetSamples(g, gaussian_1d, 2, tol)
        self._testGetSamples(g, gaussian_3d, 8, tol)

    def test_num_quadature_points_three(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingGHQ()
        g.set_num_quadrature_points(3)
        tol = 1e-12

        self._test_get_std_normal_samples(g, 1, 3, tol)
        self._test_get_std_normal_samples(g, 5, 243, tol)
        self._test_get_std_normal_samples(g, 10, 59049, tol)

        self._testGetSamples(g, gaussian_1d, 3, tol)
        self._testGetSamples(g, gaussian_3d, 27, tol)

    def test_num_quadature_points_four(self, gaussian_1d, gaussian_3d):
        g = gaussian_sampling.GaussianSamplingGHQ()
        g.set_num_quadrature_points(4)
        tol = 1e-12

        self._test_get_std_normal_samples(g, 1, 4, tol)
        self._test_get_std_normal_samples(g, 5, 1024, tol)
        self._test_get_std_normal_samples(g, 10, 1048576, tol)

        self._testGetSamples(g, gaussian_1d, 4, tol)
        self._testGetSamples(g, gaussian_3d, 64, tol)
