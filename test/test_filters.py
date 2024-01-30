import numpy as np
import pytest
from scipy.stats import chi2

from nonlinear_estimation_toolbox.distributions import (
    DiracMixture,
    Gaussian,
    GaussianMixture,
    Uniform,
)
from nonlinear_estimation_toolbox.filters.ckf import CKF, CubatureLinearGaussianFilter
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.filters.filter import Filter
from nonlinear_estimation_toolbox.filters.first_order_taylor import (
    FirstOrderTaylorLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.filters.gaussian import GaussianFilter
from nonlinear_estimation_toolbox.filters.ghkf import (
    GHKF,
    GaussianHermiteLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.filters.gpf import GPF
from nonlinear_estimation_toolbox.filters.iterative_kalman import IterativeKalmanFilter
from nonlinear_estimation_toolbox.filters.linear_gaussian import LinearGaussianFilter
from nonlinear_estimation_toolbox.filters.rukf import (
    RUKF,
    RandomizedUnscentedLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_iterative_kalman import (
    SampleBasedIterativeKalmanFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)
from nonlinear_estimation_toolbox.filters.ukf import UKF, UnscentedLinearGaussianFilter
from nonlinear_estimation_toolbox.utils import ColumnVectors

from .utils import (
    DummyPolarAdditiveNoiseMeasurementModel,
    DummyPolarMeasurementModel,
    TestUtilsAdditiveNoiseMeasurementModel,
    TestUtilsAdditiveNoiseSystemModel,
    TestUtilsLinearMeasurementModel,
    TestUtilsLinearSystemModel,
    TestUtilsMeasurementModel,
    TestUtilsMixedNoiseMeasurementModel,
    TestUtilsMixedNoiseSystemModel,
    TestUtilsSystemModel,
)

all_filters = [
    Filter,
    GaussianFilter,
    LinearGaussianFilter,
    FirstOrderTaylorLinearGaussianFilter,
    IterativeKalmanFilter,
    EKF,
    SampleBasedLinearGaussianFilter,
    SampleBasedIterativeKalmanFilter,
    UnscentedLinearGaussianFilter,
    RandomizedUnscentedLinearGaussianFilter,
    CubatureLinearGaussianFilter,
    GaussianHermiteLinearGaussianFilter,
    UKF,
    RUKF,
    CKF,
    GHKF,
]
gaussian_filters = [
    GaussianFilter,
    LinearGaussianFilter,
    FirstOrderTaylorLinearGaussianFilter,
    IterativeKalmanFilter,
    EKF,
    SampleBasedLinearGaussianFilter,
    SampleBasedIterativeKalmanFilter,
    UnscentedLinearGaussianFilter,
    RandomizedUnscentedLinearGaussianFilter,
    CubatureLinearGaussianFilter,
    GaussianHermiteLinearGaussianFilter,
    UKF,
    RUKF,
    CKF,
    GHKF,
    GPF,
]
linear_gaussian_filters = [
    LinearGaussianFilter,
    FirstOrderTaylorLinearGaussianFilter,
    IterativeKalmanFilter,
    EKF,
    SampleBasedLinearGaussianFilter,
    SampleBasedIterativeKalmanFilter,
    UnscentedLinearGaussianFilter,
    RandomizedUnscentedLinearGaussianFilter,
    CubatureLinearGaussianFilter,
    GaussianHermiteLinearGaussianFilter,
    UKF,
    RUKF,
    CKF,
    GHKF,
]

support_additive_system_model_filters = [
    CKF,
    CubatureLinearGaussianFilter,
    EKF,
    GaussianHermiteLinearGaussianFilter,
    GHKF,
    RandomizedUnscentedLinearGaussianFilter,
    RUKF,
    UnscentedLinearGaussianFilter,
    UKF,
]

support_additive_measurement_model_filters = [
    CKF,
    EKF,
    GHKF,
    RUKF,
    UKF,
]

support_measurement_model_filters = support_additive_measurement_model_filters

iterative_kalman_filters = [
    CKF,
    EKF,
    GHKF,
    RUKF,
    UKF,
]

iterative_kalman_filters_conv = [
    CKF,
    EKF,
    GHKF,
    pytest.param(RUKF, marks=pytest.mark.xfail(reason="convergence randomly fails")),
    UKF,
]

nonlinear_get_sys_models_data = [
    TestUtilsAdditiveNoiseSystemModel.get_sys_model_data,
    TestUtilsMixedNoiseSystemModel.get_sys_model_data,
    TestUtilsSystemModel.get_sys_model_data,
]

###########################
# all filters
###########################


@pytest.mark.parametrize("filter_cls", all_filters)
def test_default_constructor(filter_cls):
    f = filter_cls()
    assert f.get_state_dimension() == 0


@pytest.mark.parametrize("filter_cls", all_filters)
def test_copy(filter_cls):
    f = filter_cls()
    g = f.copy()
    assert g.get_name() == f.get_name() and "Filter copy must equal the original filter"


@pytest.mark.parametrize("filter_cls", all_filters)
def test_set_color(filter_cls):
    f = filter_cls()
    color = {"Color", "r", "LineStyle", "-"}
    f.set_color(color)
    assert f.get_color() == color and "Setting color does not work"


###########################
# GaussianFilter
###########################


def _test_mean_cov_cov_sqrt(f):
    state_mean, state_cov, state_cov_sqrt = f.get_state_mean_and_covariance()

    assert state_mean is None and "Mean of empty filter must be None"
    assert state_cov is None and "Covariance of empty filter must be None"
    assert (
        state_cov_sqrt is None
        and "Square root of covariance of empty filter must be None"
    )


@pytest.mark.parametrize("filter_cls", all_filters)
@pytest.mark.xfail
def test_mean_cov_sqrt_access_fail(filter_cls):
    f = filter_cls()
    _test_mean_cov_cov_sqrt(f)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_state_gaussian(filter_cls):
    f = filter_cls()
    mean = ColumnVectors(np.zeros((2,)))
    cov = np.diag([1.5, 2])
    d = Gaussian(mean, cov)
    f.set_state(d)
    _check_state(f, mean, cov, 2)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def testSetStateGaussianMixture(filter_cls):
    f = filter_cls()

    m1 = ColumnVectors([1, 0.4])
    m2 = ColumnVectors([-2, 3.4])
    c1 = np.diag([0.1, 3])
    c2 = np.array([[2, 0.5], [0.5, 1.2]])
    w1 = 0.31
    w2 = 0.967

    means = np.concatenate([m1, m2], axis=1)
    covs = np.stack([c1, c2], axis=-1)
    weights = np.stack([w1, w2], axis=-1)[None, :]

    d = GaussianMixture(means, covs, weights)

    weights = weights / (w1 + w2)
    w1 = weights[:, 0]
    w2 = weights[:, 1]
    mean = w1 * m1 + w2 * m2
    cov = (
        w1 * c1
        + w2 * c2
        + w1 * (m1 - mean) * (m1 - mean).T
        + w2 * (m2 - mean) * (m2 - mean).T
    )

    f.set_state(d)

    _check_state(f, mean, cov, 2)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_state_uniform(filter_cls):
    f = filter_cls()

    a = ColumnVectors(np.array([-1, 3]))
    b = ColumnVectors(np.array([-0.5, 5]))
    d = Uniform(a, b)

    mean = ColumnVectors(np.array([-0.75, 4]))
    cov = np.diag([0.25, 4]) / 12

    f.set_state(d)

    _check_state(f, mean, cov, 2)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_state_dirac_mixture(filter_cls):
    f = filter_cls()

    samples = [np.zeros((3,)), *(2 * np.identity(3)), *(-2 * np.identity(3))]
    samples = ColumnVectors(
        np.array([sample + -4 * np.ones((3,)) for sample in samples]).T
    )
    weights = ColumnVectors([[2, 1, 1, 1, 1, 1, 1]])

    d = DiracMixture(samples, weights)

    mean = ColumnVectors(np.array([-4, -4, -4]))
    cov = np.identity(3)

    f.set_state(d)

    _check_state(f, mean, cov, 3)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_state_mean_and_cov(filter_cls):
    f = filter_cls()

    mean = ColumnVectors(np.array([-0.75, 4]))
    cov = np.diag([0.25, 4]) / 12
    cov_sqrt = np.linalg.cholesky(cov)

    f.set_state_mean_and_covariance(mean, cov, cov_sqrt)

    _check_state(f, mean, cov, 2)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_state_mean_and_cov_no_cov_sqrt(filter_cls):
    f = filter_cls()

    mean = ColumnVectors(np.array([-0.75, 4]))
    cov = np.diag([0.25, 4]) / 12

    f.set_state_mean_and_covariance(mean, cov)

    _check_state(f, mean, cov, 2)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_get_state_mean_and_cov(filter_cls):
    f = filter_cls()

    a = ColumnVectors(np.array([-1, 3]))
    b = ColumnVectors(np.array([-0.5, 5]))
    d = Uniform(a, b)

    mean = ColumnVectors(np.array([-0.75, 4]))
    cov = np.diag([0.25, 4]) / 12
    cov_sqrt = np.linalg.cholesky(cov)

    f.set_state(d)

    state_mean, state_cov, state_cov_sqrt = f.get_state_mean_and_covariance()

    assert np.allclose(state_mean, mean)
    assert np.allclose(state_cov, state_cov)
    assert np.allclose(state_cov, cov)
    assert np.allclose(state_cov_sqrt, cov_sqrt)


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_prediction_post_processing(filter_cls):
    f = filter_cls()
    func = _post_processing_scale
    f.set_prediction_post_processing(func)
    assert f.get_prediction_post_processing() == func


@pytest.mark.parametrize("filter_cls", gaussian_filters)
def test_set_update_post_processing(filter_cls):
    f = filter_cls()
    func = _post_processing_scale
    f.set_update_post_processing(func)
    assert f.get_update_post_processing() == func


def _post_processing_scale(m, cov, *argv):
    return 2.0 * m, 0.5 * cov


class TestFilterPredict:
    @pytest.mark.parametrize("filter_cls", gaussian_filters)
    @pytest.mark.parametrize("has_sys_mat", [True, False])
    @pytest.mark.parametrize("has_input", [True, False])
    @pytest.mark.parametrize("has_sys_noise_mat", [True, False])
    @pytest.mark.parametrize("post_processing", [True, False])
    def test_predict_linear_system_model_configuration(
        self, filter_cls, has_sys_mat, has_input, has_sys_noise_mat, post_processing
    ):
        (
            initState,
            sysModel,
            true_state_mean,
            true_state_cov,
        ) = TestUtilsLinearSystemModel.get_sys_model_data(
            has_sys_mat, has_sys_noise_mat, has_input
        )

        f = filter_cls()
        f.set_state(initState)

        if post_processing:
            f.set_prediction_post_processing(_post_processing_scale)

        f.predict(sysModel)

        state_mean, state_cov, _ = f.get_state_mean_and_covariance()

        if post_processing:
            true_state_mean, true_state_cov = _post_processing_scale(
                true_state_mean, true_state_cov
            )

        assert state_mean == pytest.approx(true_state_mean, rel=1e-10)
        assert np.allclose(state_cov, state_cov.T)
        assert np.allclose(state_cov, true_state_cov, rtol=1e-10)

    @pytest.mark.parametrize("filter_cls", support_additive_system_model_filters)
    @pytest.mark.parametrize("post_processing", [True, False])
    @pytest.mark.parametrize("get_sys_model_data", nonlinear_get_sys_models_data)
    def test_predict_nonlinear_configuration(
        self, filter_cls, post_processing, get_sys_model_data
    ):
        init_state, sys_model, true_state_mean, true_state_cov = get_sys_model_data()

        f = filter_cls()
        tol = 1e-10

        f.set_state(init_state)

        if post_processing:
            f.set_prediction_post_processing(_post_processing_scale)

        f.predict(sys_model)

        state_mean, state_cov, _ = f.get_state_mean_and_covariance()

        if post_processing:
            true_state_mean, true_state_cov = _post_processing_scale(
                true_state_mean, true_state_cov
            )

        assert np.allclose(state_mean, true_state_mean, rtol=tol)
        assert np.allclose(state_cov, state_cov.T, rtol=tol)
        assert np.allclose(state_cov, true_state_cov, rtol=tol)

    @pytest.mark.parametrize("filter_cls", support_additive_system_model_filters)
    @pytest.mark.parametrize("get_sys_model_data", nonlinear_get_sys_models_data)
    def test_timed_predict(self, filter_cls, get_sys_model_data):
        init_state, sys_model, true_state_mean, true_state_cov = get_sys_model_data()
        f = filter_cls()
        f.set_state(init_state)
        assert f.predict_timed(sys_model) > 0.0


class TestFilterUpdate:
    @pytest.fixture
    def init_state(self) -> Gaussian:
        init_mean = ColumnVectors([1, 1, 0, 0, 0])
        init_cov = np.diag([10, 10, 1e-1, 1, 1e-1])
        return Gaussian(mean=init_mean, covariance=init_cov)

    @pytest.fixture
    def measurement(self) -> ColumnVectors:
        return ColumnVectors([3, np.pi / 5])

    @pytest.mark.parametrize("filter_cls", support_measurement_model_filters)
    def test_update_measurement_model(self, filter_cls, init_state, measurement):
        """Check if filter completes a single update step"""
        polar_meas_model = DummyPolarMeasurementModel()

        filter = filter_cls()
        filter.set_state(init_state)

        filter.update(meas_model=polar_meas_model, measurement=measurement)

        mean, cov, _ = filter.get_state_mean_and_covariance()
        assert mean.shape == init_state.mean.shape
        assert cov.shape == init_state.covariance.shape

    @pytest.mark.parametrize("filter_cls", support_additive_measurement_model_filters)
    def test_update_additive_noise_measurement_model(
        self, filter_cls, init_state, measurement
    ):
        """Check if filter completes a single update step"""
        polar_meas_model = DummyPolarAdditiveNoiseMeasurementModel()

        filter = filter_cls()
        filter.set_state(init_state)

        filter.update(meas_model=polar_meas_model, measurement=measurement)

        mean, cov, _ = filter.get_state_mean_and_covariance()
        assert mean.shape == init_state.mean.shape
        assert cov.shape == init_state.covariance.shape

    @pytest.mark.parametrize("filter_cls", iterative_kalman_filters_conv)
    def test_iterative_update_end_on_max_iters(
        self, filter_cls, init_state, measurement
    ):
        polar_meas_model = DummyPolarAdditiveNoiseMeasurementModel()
        filter = filter_cls()
        filter.set_state(init_state)
        filter.set_max_num_iterations(2)
        filter.update(meas_model=polar_meas_model, measurement=measurement)

        mean, cov, _ = filter.get_state_mean_and_covariance()
        assert mean.shape == init_state.mean.shape
        assert cov.shape == init_state.covariance.shape

    @pytest.mark.parametrize("filter_cls", iterative_kalman_filters_conv)
    def test_iterative_update_end_on_convergence(
        self, filter_cls, init_state, measurement
    ):
        polar_meas_model = DummyPolarAdditiveNoiseMeasurementModel()
        filter = filter_cls()
        filter.set_state(init_state)
        eps = 1e-8
        max_iters = 1000

        def euclidian_dist_convergence_check(
            last_state_mean,
            last_state_cov,
            last_state_cov_sqrt,
            updated_state_mean,
            updated_state_cov,
            updated_state_cov_sqrt,
        ) -> bool:
            diff = last_state_mean - updated_state_mean
            return np.sqrt(np.sum(diff.T.dot(diff))) < eps

        filter.set_convergence_check_func(euclidian_dist_convergence_check)
        filter.set_max_num_iterations(max_iters)
        filter.update(meas_model=polar_meas_model, measurement=measurement)

        mean, cov, _ = filter.get_state_mean_and_covariance()
        assert mean.shape == init_state.mean.shape
        assert cov.shape == init_state.covariance.shape
        assert filter.get_num_iterations() < max_iters, "Filter did not converge!"

    @pytest.mark.parametrize("filter_cls", support_measurement_model_filters)
    def test_timed_update(self, filter_cls, init_state):
        polar_meas_model = DummyPolarMeasurementModel()
        filter = filter_cls()
        filter.set_state(init_state)

        assert (
            filter.update_timed(
                meas_model=polar_meas_model, measurement=ColumnVectors([3, np.pi / 5])
            )
            > 0.0
        )

    @pytest.mark.parametrize("filter_cls", iterative_kalman_filters)
    @pytest.mark.parametrize(
        "get_meas_model_data",
        [
            TestUtilsMeasurementModel.get_meas_model_data,
            TestUtilsAdditiveNoiseMeasurementModel.get_meas_model_data,
            TestUtilsMixedNoiseMeasurementModel.get_meas_model_data,
        ],
    )
    @pytest.mark.parametrize("state_decomp", [True, False])
    @pytest.mark.parametrize("post_processing", [True, False])
    @pytest.mark.parametrize("meas_gating", [True, False])
    @pytest.mark.parametrize("multi_iterations", [True, False])
    @pytest.mark.parametrize("convergence_check", [True, False])
    def test_update_nonlinear_configurations(
        self,
        filter_cls,
        get_meas_model_data,
        state_decomp,
        post_processing,
        meas_gating,
        multi_iterations,
        convergence_check,
    ):

        (
            init_state,
            meas_model,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        ) = get_meas_model_data(state_decomp)

        f = filter_cls()
        tol = 1e-10

        f.set_state(init_state)

        if state_decomp:
            f.set_state_decomposition_dimension(state_decomp_dim)

        if post_processing:
            f.set_update_post_processing(_post_processing_scale)

        if multi_iterations:
            f.set_max_num_iterations(3)
        else:
            pass

        if convergence_check:

            def conv_check_true(*args) -> bool:
                return True

            f.set_convergence_check_func(conv_check_true)

        if meas_gating:
            dim_meas = true_meas_cov.shape[0]
            meas_dist = (
                (true_meas_mean - measurement)
                .T.dot(np.linalg.inv(true_meas_cov))
                .dot(true_meas_mean - measurement)
            )
            normalized_dist = chi2.cdf(float(meas_dist), dim_meas)

            f.set_measurement_gating_threshold(normalized_dist * 0.5)

            with pytest.raises(RuntimeError):
                f.update(meas_model, measurement)

            state_mean, state_cov, _ = f.get_state_mean_and_covariance()
            init_state_mean, init_state_cov, _ = init_state.get_mean_and_covariance()

            assert np.all(state_mean == init_state_mean)
            assert np.all(state_cov == state_cov.T)
            assert np.all(state_cov == init_state_cov)
        else:
            f.update(meas_model, measurement)

            state_mean, state_cov, _ = f.get_state_mean_and_covariance()

            if post_processing:
                true_state_mean, true_state_cov = _post_processing_scale(
                    true_state_mean, true_state_cov
                )

            np.testing.assert_allclose(state_mean, true_state_mean, rtol=tol)
            assert np.all(state_cov == state_cov.T)
            np.testing.assert_allclose(state_cov, true_state_cov, rtol=tol)

    @pytest.mark.parametrize("filter_cls", [EKF, UKF])
    @pytest.mark.parametrize("has_meas_matrix", [True, False])
    @pytest.mark.parametrize("state_decomp", [True, False])
    @pytest.mark.parametrize("post_processing", [True, False])
    @pytest.mark.parametrize("meas_gating", [True, False])
    def test_update_linear_measurement_model_configuration(
        self, filter_cls, has_meas_matrix, state_decomp, post_processing, meas_gating
    ):

        (
            initState,
            measModel,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        ) = TestUtilsLinearMeasurementModel.get_meas_model_data(
            has_meas_matrix, state_decomp
        )

        f = filter_cls()
        tol = 1e-10

        f.set_state(initState)

        if state_decomp:
            f.set_state_decomposition_dimension(state_decomp_dim)

        if post_processing:
            f.set_update_post_processing(_post_processing_scale)

        if meas_gating:
            dim_meas = true_meas_cov.shape[0]
            meas_dist = (
                (true_meas_mean - measurement)
                .T.dot(np.linalg.inv(true_meas_cov))
                .dot(true_meas_mean - measurement)
            )
            normalized_dist = chi2.cdf(float(meas_dist), dim_meas)

            f.set_measurement_gating_threshold(normalized_dist * 0.5)

            with pytest.raises(RuntimeError):
                f.update(measModel, measurement)

            state_mean, state_cov, _ = f.get_state_mean_and_covariance()

            init_state_mean, init_state_cov, _ = initState.get_mean_and_covariance()

            assert np.all(state_mean == init_state_mean)
            assert np.all(state_cov == state_cov.T)
            assert np.all(state_cov == init_state_cov)
        else:
            f.update(measModel, measurement)

            state_mean, state_cov, _ = f.get_state_mean_and_covariance()

            if post_processing:
                true_state_mean, true_state_cov = _post_processing_scale(
                    true_state_mean, true_state_cov
                )

            np.testing.assert_allclose(state_mean, true_state_mean, rtol=tol)
            assert np.all(state_cov == state_cov.T)
            np.testing.assert_allclose(state_cov, true_state_cov, rtol=tol)


def _check_state(f, mean, cov, dim):
    state = f.get_state()

    assert isinstance(state, Gaussian)

    state_mean, state_cov, _ = state.get_mean_and_covariance()

    assert state_mean == pytest.approx(mean, abs=1e-8)
    assert state_cov == pytest.approx(cov, abs=1e-8)

    assert f.get_state_dimension() == dim


###########################
# LinearGaussianFilter
###########################


@pytest.mark.parametrize("filter_cls", linear_gaussian_filters)
def test_set_meas_gating_threshold(filter_cls):
    f = filter_cls()
    f.set_measurement_gating_threshold(0.5)
    assert f.get_measurement_gating_threshold() == 0.5


@pytest.mark.parametrize("filter_cls", linear_gaussian_filters)
@pytest.mark.parametrize(
    "invalidMeas",
    [np.ones((2, 4)), Gaussian(ColumnVectors(np.array([0])), np.array([[1]]))],
)
@pytest.mark.xfail
def test_update_invalid_measurement(filter_cls, invalidMeas):
    f = filter_cls()
    f.set_state(Gaussian(np.array([0]), np.array([[1]])))
    meas_model = []
    f.update(meas_model, invalidMeas)
