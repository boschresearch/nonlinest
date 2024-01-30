import numpy as np
import pytest

import nonlinear_estimation_toolbox.distributions as distributions
from nonlinear_estimation_toolbox.utils import ColumnVectors


def verify_gaussian(g, dim, mean, cov, cov_sqrt):
    assert g.dim == dim
    assert (g.mean == mean).all()
    assert (g.covariance == cov).all()
    assert (g.covariance_sqrt == cov_sqrt).all()


class TestGaussian:
    def test_constructor_default(self):
        g = distributions.Gaussian()
        verify_gaussian(g, 1, ColumnVectors(0), np.array([[1]]), np.array([[1]]))

    def test_constructor_uncorrelated(self):
        pass

    def test_constructor_correlated(self):
        mean = ColumnVectors([1, 2, 3])
        cov = np.array([[19, 2, 3], [2, 49, 5], [3, 5, 99]])
        g = distributions.Gaussian(mean, cov)

        verify_gaussian(g, 3, mean, cov, np.linalg.cholesky(cov))

    def test_draw_rnd_samples(self):
        mean = ColumnVectors([4, 5])
        cov = np.array([[1, 0.1], [0.1, 2]])
        g = distributions.Gaussian(mean, cov)

        samples = g.draw_random_samples(11)
        assert (np.shape(samples) == np.array([2, 11])).all()

    def test_log_pdf_uncorrelated(self):
        pass

    def test_log_pdf_correlated(self):
        mean = ColumnVectors([4, 5])
        cov = np.array([[1, 0.1], [0.1, 2]])
        g = distributions.Gaussian(mean, cov)
        values = ColumnVectors([[1, 2, 3], [0, 7, 11]])

        logpdf = g.log_pdf(values)
        assert (np.shape(logpdf) == np.array([1, 3])).all()


def verify_uniform(u, dim, a, b, mean, cov):
    assert u.dim == dim
    assert (u.a == a).all()
    assert (u.b == b).all()
    mean, cov, cov_sqrt = u.get_mean_and_covariance()
    assert (mean == mean).all()
    assert (cov == cov).all()
    assert (cov_sqrt == np.sqrt(cov)).all()


class TestUniform:
    def test_constructor_default(self):
        u = distributions.Uniform()
        verify_uniform(
            u,
            1,
            ColumnVectors(0),
            ColumnVectors(1),
            ColumnVectors(0.5),
            np.array([[1 / 12]]),
        )

    def test_constructor_scalar(self):
        a = ColumnVectors(7)
        b = ColumnVectors(13)
        u = distributions.Uniform(a, b)
        verify_uniform(u, 1, a, b, ColumnVectors(10), np.array([[3]]))

    def test_constructor_multivariate(self):
        a = ColumnVectors([7, 8])
        b = ColumnVectors([13, 20])
        u = distributions.Uniform(a, b)
        verify_uniform(u, 2, a, b, ColumnVectors([10, 14]), np.array([[3, 0], [0, 12]]))

    def test_draw_rnd_samples(self):
        a = ColumnVectors([7, 8])
        b = ColumnVectors([13, 20])
        u = distributions.Uniform(a, b)

        samples = u.draw_random_samples(11)
        assert (np.shape(samples) == np.array([2, 11])).all()
        assert (samples >= a).all()
        assert (samples <= b).all()

    def test_log_pdf(self):
        a = ColumnVectors([7, 8])
        b = ColumnVectors([13, 20])
        u = distributions.Uniform(a, b)

        c = np.log(1 / (6 * 12))
        values = ColumnVectors([[1, 2, 3, 4, 7, 13, 9, 2], [0, 1, 2, 3, 13, 20, 25, 9]])
        log_pdf = u.log_pdf(values)
        assert (log_pdf == ColumnVectors([0, 0, 0, 0, c, c, 0, 0]).T).all()


def verify_dm(
    dm: distributions.DiracMixture,
    dim: int,
    samples: np.array,
    weights: np.array,
    mean: np.array,
    cov: np.array,
    cov_sqrt: np.array,
) -> None:
    assert dm.dim == dim
    assert (dm.samples == samples).all()
    if weights is not None:
        assert (dm.weights == weights).all()
    m, c, csqrt = dm.get_mean_and_covariance()
    if dim > 0:
        assert np.allclose(m, mean, 1e-10)
        assert np.allclose(c, cov, 1e-10)
        assert np.allclose(csqrt, cov_sqrt, 1e-10)


class TestDiracMixture:
    def test_constructor_default(self):
        dm = distributions.DiracMixture()
        verify_dm(dm, 0, [], None, [], [], [])

    def test_constructor_samples(self):
        samples = 2 + ColumnVectors([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1]]) * np.sqrt(
            5 / 2
        )
        dm = distributions.DiracMixture(samples)
        mean = ColumnVectors([[2], [2]])
        cov = np.array([[1, 0], [0, 1]])
        cov_sqrt = cov

        verify_dm(dm, 2, samples, ColumnVectors(np.ones(5)).T / 5, mean, cov, cov_sqrt)

    def test_constructor_samples_weights(self):
        samples = 2 + ColumnVectors([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1]]) * np.sqrt(3)
        weights = ColumnVectors([[2, 1, 1, 1, 1]])
        dm = distributions.DiracMixture(samples, weights)
        mean = np.array([[2], [2]])
        cov = np.array([[1, 0], [0, 1]])
        cov_sqrt = cov

        verify_dm(dm, 2, samples, weights / np.sum(weights), mean, cov, cov_sqrt)

    def test_set_samples(self):
        dm = distributions.DiracMixture()
        samples = 4 + ColumnVectors([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1]]) * np.sqrt(5)
        dm.set(samples)
        mean = ColumnVectors([[4], [4]])
        cov = np.array([[2, 0], [0, 2]])
        cov_sqrt = np.sqrt(cov)

        verify_dm(dm, 2, samples, ColumnVectors(np.ones(5) / 5).T, mean, cov, cov_sqrt)

    def test_set_samples_weights(self):
        dm = distributions.DiracMixture()
        samples = 2 + ColumnVectors([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1]]) * np.sqrt(3)
        weights = ColumnVectors([[2, 1, 1, 1, 1]])
        dm.set(samples, weights)
        mean = ColumnVectors([[2], [2]])
        cov = np.array([[1, 0], [0, 1]])
        cov_sqrt = cov

        verify_dm(dm, 2, samples, weights / np.sum(weights), mean, cov, cov_sqrt)

    def test_draw_rnd_samples(self):
        samples = 2 + ColumnVectors([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1]]) * np.sqrt(3)
        weights = ColumnVectors([[2, 1, 1, 1, 1]])
        dm = distributions.DiracMixture(samples, weights)

        s = dm.draw_random_samples(10)
        assert s.shape[0] == 2
        assert s.shape[1] == 10

    def test_log_pdf(self):
        dm = distributions.DiracMixture(ColumnVectors([[1, 4], [5, 6]]))
        values = ColumnVectors([[1, 2, 3], [1, 2, 3]])
        with pytest.raises(Exception):
            dm.pdf(values)


class TestGaussianMixture:
    def _gaussian2d(self, mean, cov, values):
        det = cov[0, 0] * cov[1, 1] - cov[1, 0] * cov[0, 1]
        inv_cov = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det
        c = 1 / (2 * np.pi * np.sqrt(np.abs(det)))
        v = values - mean

        values = c * np.exp(-0.5 * np.diag(v.T.dot(inv_cov.dot(v))).T)
        return values

    @pytest.fixture
    def mean(self):
        return ColumnVectors([1, 0.4])

    @pytest.fixture
    def means(self, mean):
        return np.concatenate([mean, ColumnVectors([-2, 3.4])], axis=1)

    @pytest.fixture
    def cov(self):
        return np.array([[2, 0.5], [0.5, 1.2]])

    @pytest.fixture
    def covs(self, cov):
        return np.stack([np.diag([0.1, 3]), cov], axis=-1)

    @pytest.fixture
    def values(self):
        return ColumnVectors([[0.1, -0.5, 13.4], [0.9, -10, 5]])

    @pytest.fixture
    def weights(self):
        return np.array([[0.79, 0.21]])

    def test_set_means_covs(self, means, covs):
        gm = distributions.GaussianMixture(means, covs)
        assert np.all(gm.means == means)
        assert np.all(gm.covs == covs)
        assert np.all(gm.weights == 0.5)

    def test_set_means_covs_weights(self, means, covs, weights):
        gm = distributions.GaussianMixture(means, covs, weights)
        assert np.all(gm.means == means)
        assert np.all(gm.covs == covs)
        assert np.all(gm.weights == weights)

    def test_get_means_and_covs(self, means, covs, weights):
        gm = distributions.GaussianMixture(means, covs, weights)

        m1, m2 = means[..., None].transpose(1, 0, 2)
        w1, w2 = weights.flatten() / np.sum(weights)
        c1, c2 = covs.transpose(2, 0, 1)

        mean = w1 * m1 + w2 * m2
        cov = (
            w1 * c1
            + w2 * c2
            + w1 * (m1 - mean) * (m1 - mean).T
            + w2 * (m2 - mean) * (m2 - mean).T
        )
        cov_sqrt = np.linalg.cholesky(cov)

        mean_get, cov_get, cov_sqrt_get = gm.get_mean_and_covariance()
        assert np.allclose(mean_get.flatten(), mean.flatten())
        assert np.allclose(cov_get, cov)
        assert np.all(cov_sqrt_get == cov_sqrt)

    def test_draw_rnd_samples(self, means, covs, weights):
        gm = distributions.GaussianMixture(means, covs, weights)

        num_samples = 10
        samples = gm.draw_random_samples(num_samples)

        assert samples.shape == (2, num_samples)

    def test_draw_rnd_samples_single_with_comp_ids(self, means, covs, weights):
        gm = distributions.GaussianMixture(means, covs, weights)

        num_samples = 1
        samples, comp_ids = gm.draw_random_samples_with_comp_ids(num_samples)

        assert samples.shape == (2, num_samples)
        assert comp_ids.shape == (1, num_samples)
        assert np.all(comp_ids >= 0)
        assert np.all(comp_ids <= 1)

    def test_draw_rnd_samples_multiple_with_comp_ids(self, means, covs, weights):
        gm = distributions.GaussianMixture(means, covs, weights)

        num_samples = 19
        samples, comp_ids = gm.draw_random_samples_with_comp_ids(num_samples)

        assert samples.shape == (2, num_samples)
        assert comp_ids.shape == (1, num_samples)
        assert np.all(comp_ids >= 0)
        assert np.all(comp_ids <= 1)

        for i in range(1, num_samples):
            assert comp_ids[:, i] >= comp_ids[:, i - 1]

    def test_log_pdf_one_component(self, mean, cov, values):
        weight = np.array([[1]])

        gm = distributions.GaussianMixture(mean, cov, weight)

        log_values = gm.log_pdf(values)
        assert log_values.shape == (1, 3)

        true_log_values = np.log(self._gaussian2d(mean, cov, values))

        assert np.allclose(log_values, true_log_values, atol=1e-10)

    def test_log_pdf_two_components(self, means, covs, values):
        weights = np.array([[0.3, 1.2]])

        gm = distributions.GaussianMixture(means, covs, weights)
        log_values = gm.log_pdf(values)

        assert log_values.shape == (1, 3)

        v0 = self._gaussian2d(means[..., 0:1], covs[..., 0], values)
        v1 = self._gaussian2d(means[..., 1:2], covs[..., 1], values)
        w0 = weights[..., 0] / np.sum(weights)
        w1 = weights[..., 1] / np.sum(weights)
        true_log_values = np.log(w0 * v0 + w1 * v1)

        assert np.allclose(log_values, true_log_values, atol=1e-10)
