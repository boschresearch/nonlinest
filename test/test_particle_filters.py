from test.test_filters import _post_processing_scale
from test.utils import (
    TestUtilsAdditiveNoiseMeasurementModel,
    TestUtilsAdditiveNoiseSystemModel,
    TestUtilsLinearMeasurementModel,
    TestUtilsLinearSystemModel,
    TestUtilsMixedNoiseSystemModel,
    TestUtilsStep,
    TestUtilsSystemModel,
)
from typing import Tuple

import numpy as np
import pytest

from nonlinear_estimation_toolbox.distributions import (
    DiracMixture,
    Gaussian,
    GaussianMixture,
    Uniform,
)
from nonlinear_estimation_toolbox.filters.asirpf import ASIRPF
from nonlinear_estimation_toolbox.filters.cgpf import CGPF
from nonlinear_estimation_toolbox.filters.enkf import EnKF
from nonlinear_estimation_toolbox.filters.filter import Filter
from nonlinear_estimation_toolbox.filters.gpf import GPF
from nonlinear_estimation_toolbox.filters.rpf import RPF
from nonlinear_estimation_toolbox.filters.sirpf import SIRPF
from nonlinear_estimation_toolbox.utils import ColumnVectors

particle_filters = [SIRPF, ASIRPF, RPF]

nonlinear_get_sys_models_data = [
    TestUtilsAdditiveNoiseSystemModel.get_sys_model_data,
    TestUtilsMixedNoiseSystemModel.get_sys_model_data,
    TestUtilsSystemModel.get_sys_model_data,
]


def _check_state(f: Filter, dim, num_samples):
    dm = f.get_state()

    assert isinstance(dm, DiracMixture)

    samples, weights = dm.get_components()
    assert samples.shape == (dim, num_samples)
    assert weights.shape == (1, num_samples)
    assert np.allclose(weights, np.ones((1, num_samples)) / num_samples, atol=1e-14)


@pytest.fixture
def gaussian() -> Gaussian:
    return Gaussian(ColumnVectors(np.zeros(2)), np.diag([1.5, 2]))


@pytest.fixture
def dirac_mixture() -> Tuple[DiracMixture, np.ndarray, np.ndarray]:
    s = np.concatenate(
        (np.zeros((3, 1)), np.ones((3, 1)), ColumnVectors([-2, 5, 0.7])), axis=1
    )
    w = np.array([[1, 2, 3]])

    return DiracMixture(s, w), s, w


@pytest.mark.parametrize("filter_cls", particle_filters)
def test_set_state_gaussian(filter_cls, gaussian):
    f = filter_cls()
    d = gaussian
    f.set_state(d)
    _check_state(f, 2, 1000)


@pytest.mark.parametrize("filter_cls", [*particle_filters, EnKF])
def test_set_state_gaussian_mixture(filter_cls):
    f = filter_cls()
    d = GaussianMixture(
        np.ones((2, 2)),
        np.stack([np.diag([1.5, 2]), 3 * np.eye(2)], axis=-1),
        np.array([[1, 5]]),
    )
    f.set_state(d)
    _check_state(f, 2, 1000)


@pytest.mark.parametrize("filter_cls", [*particle_filters, EnKF])
def test_set_state_uniform(filter_cls):
    f = filter_cls()
    d = Uniform(ColumnVectors([-2, 3, 0]), ColumnVectors([5, 10, 1]))
    f.set_state(d)
    _check_state(f, 3, 1000)


@pytest.mark.parametrize("filter_cls", particle_filters)
def test_set_state_dirac_mixture(filter_cls, dirac_mixture):
    f = filter_cls()

    d, s, w = dirac_mixture
    f.set_state(d)

    dm = f.get_state()
    assert isinstance(dm, DiracMixture)

    samples, weights = dm.get_components()
    assert np.array_equal(s, samples)
    assert np.array_equal(weights, w / np.sum(w))
    assert f.get_num_particles() == 3


def test_set_state_dirac_mixture_enkf(dirac_mixture):
    f = EnKF()

    d, s, w = dirac_mixture
    f.set_state(d)

    _check_state(f, 3, 1000)


@pytest.mark.parametrize("filter_cls", [*particle_filters, EnKF])
def test_set_state_mean_and_cov(filter_cls):
    f = filter_cls()
    mean = ColumnVectors([-0.75, 4])
    cov = np.diag([0.25, 4]) / 12
    cov_sqrt = np.sqrt(cov)

    f.set_state_mean_and_covariance(mean, cov, cov_sqrt)
    _check_state(f, 2, 1000)


@pytest.mark.parametrize("filter_cls", [*particle_filters, EnKF])
def test_set_state_mean_and_cov_no_cov_sqrt(filter_cls):
    f = filter_cls()
    mean = ColumnVectors([-0.75, 4])
    cov = np.diag([0.25, 4]) / 12

    f.set_state_mean_and_covariance(mean, cov)
    _check_state(f, 2, 1000)


@pytest.mark.parametrize("filter_cls", [*particle_filters, GPF, CGPF])
def test_set_num_particles(filter_cls):
    f = filter_cls()
    f.set_num_particles(2000)
    assert f.get_num_particles() == 2000


def test_set_ensemble_size():
    f = EnKF()
    f.set_ensemble_size(2000)
    assert f.get_ensemble_size() == 2000


@pytest.mark.parametrize("filter_cls", particle_filters)
def test_set_num_particles_with_resampling(filter_cls, gaussian):
    f = filter_cls()
    f.set_state(gaussian)

    f.set_num_particles(2000)

    assert f.get_num_particles() == 2000

    dm = f.get_state()
    samples, weights = dm.get_components()
    assert samples.shape == (2, 2000)
    assert weights.shape == (1, 2000)
    assert np.allclose(weights, np.ones((1, 2000)) / 2000, atol=1e-14)


def test_set_ensemble_size_with_resampling(gaussian):
    f = EnKF()
    f.set_ensemble_size(10**7)

    d = gaussian
    f.set_state(gaussian)

    state_mean, state_cov, state_cov_sqrt = f.get_state_mean_and_covariance()
    mean, cov, cov_sqrt = d.get_mean_and_covariance()

    tol = 1e-1
    assert np.allclose(state_mean, mean, atol=tol)
    assert np.allclose(state_cov, cov, atol=tol)
    assert np.allclose(state_cov_sqrt, cov_sqrt, atol=tol)


@pytest.mark.parametrize("filter_cls", particle_filters)
def test_set_min_allowed_normalized_ess(filter_cls):
    f = filter_cls()
    f.set_min_allowed_normalized_ess(0.3)
    assert f.get_min_allowed_normalized_ess() == 0.3


@pytest.mark.parametrize("filter_cls", particle_filters)
def test_get_state_mean_and_cov(filter_cls):
    f = filter_cls()

    samples = np.hstack((np.zeros((3, 1)), 2 * np.eye(3), -2 * np.eye(3)))
    samples = samples - 4
    weights = np.array([[2, 1, 1, 1, 1, 1, 1]]) / 8
    d = DiracMixture(samples, weights)
    f.set_state(d)

    state_mean, state_cov, state_cov_sqrt = f.get_state_mean_and_covariance()
    mean, cov, cov_sqrt = d.get_mean_and_covariance()

    assert np.array_equal(state_mean, mean)
    assert np.array_equal(state_cov, cov)
    assert np.array_equal(state_cov_sqrt, cov_sqrt)


@pytest.mark.parametrize("filter_cls", particle_filters)
@pytest.mark.parametrize("has_sys_mat", [True, False])
@pytest.mark.parametrize("has_input", [True, False])
@pytest.mark.parametrize("has_sys_noise_mat", [True, False])
def test_predict_linear_system_model(
    filter_cls, has_sys_mat, has_input, has_sys_noise_mat
):
    (
        init_state,
        sys_model,
        true_state_mean,
        true_state_cov,
    ) = TestUtilsLinearSystemModel.get_sys_model_data(
        has_sys_mat, has_sys_noise_mat, has_input
    )

    f = filter_cls()
    f.set_num_particles(int(1e7))
    tol = 1e-2

    f.set_state(init_state)
    f.predict(sys_model)
    state_mean, state_cov, _ = f.get_state_mean_and_covariance()

    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("has_sys_mat", [True, False])
@pytest.mark.parametrize("has_input", [True, False])
@pytest.mark.parametrize("has_sys_noise_mat", [True, False])
def test_predict_linear_system_model_enkf(has_sys_mat, has_input, has_sys_noise_mat):
    (
        init_state,
        sys_model,
        true_state_mean,
        true_state_cov,
    ) = TestUtilsLinearSystemModel.get_sys_model_data(
        has_sys_mat, has_sys_noise_mat, has_input
    )

    f = EnKF()
    f.set_ensemble_size(10**7)
    tol = 1e-2

    f.set_state(init_state)
    f.predict(sys_model)
    state_mean, state_cov, _ = f.get_state_mean_and_covariance()

    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("filter_cls", particle_filters)
@pytest.mark.parametrize("get_sys_model_data", nonlinear_get_sys_models_data)
def test_predict_nonlinear(filter_cls, get_sys_model_data):
    init_state, sys_model, true_state_mean, true_state_cov = get_sys_model_data()

    f = filter_cls()
    f.set_num_particles(int(1e6))
    tol = 1e-2

    f.set_state(init_state)
    f.predict(sys_model)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()
    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("get_sys_model_data", nonlinear_get_sys_models_data)
def test_predict_nonlinear_enkf(get_sys_model_data):
    init_state, sys_model, true_state_mean, true_state_cov = get_sys_model_data()

    f = EnKF()
    f.set_ensemble_size(10**6)
    tol = 1e-2

    f.set_state(init_state)
    f.predict(sys_model)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()
    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("filter_cls", particle_filters)
@pytest.mark.parametrize("has_meas_matrix", [True, False])
def test_update_linear_measurement_model_configuration(filter_cls, has_meas_matrix):
    (
        init_state,
        meas_model,
        measurement,
        _,
        true_state_mean,
        true_state_cov,
        _,
        _,
    ) = TestUtilsLinearMeasurementModel.get_meas_model_data(has_meas_matrix)

    f = filter_cls()
    f.set_num_particles(int(1e7))
    tol = 1e-2

    f.set_state(init_state)
    f.update(meas_model, measurement)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()
    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("filter_cls", [GPF, CGPF])
@pytest.mark.parametrize("has_meas_matrix", [True, False])
@pytest.mark.parametrize("state_decomp", [True, False])
@pytest.mark.parametrize("post_processing", [True, False])
def test_update_linear_measurement_model_configuration_2(
    filter_cls, has_meas_matrix, state_decomp, post_processing
):
    (
        init_state,
        meas_model,
        measurement,
        state_decomp_dim,
        true_state_mean,
        true_state_cov,
        _,
        _,
    ) = TestUtilsLinearMeasurementModel.get_meas_model_data(
        has_meas_matrix, state_decomp
    )

    f = filter_cls()
    f.set_num_particles(10**7)
    f.set_state(init_state)

    tol = 1e-2

    if state_decomp:
        f.set_state_decomposition_dimension(state_decomp_dim)

    if post_processing:
        f.set_update_post_processing(_post_processing_scale)

    f.update(meas_model, measurement)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()

    if post_processing:
        true_state_mean, true_state_cov = _post_processing_scale(
            true_state_mean, true_state_cov
        )

    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("filter_cls", [GPF, CGPF])
@pytest.mark.parametrize("state_decomp", [True, False])
def test_update_additive_noise_measurement_model_configuration(
    filter_cls, state_decomp
):
    (
        init_state,
        meas_model,
        measurement,
        state_decomp_dim,
        true_state_mean,
        true_state_cov,
        _,
        _,
    ) = TestUtilsAdditiveNoiseMeasurementModel.get_meas_model_data(state_decomp)

    f = filter_cls()
    f.set_num_particles(10**7)
    f.set_state(init_state)

    tol = 1e-2

    if state_decomp:
        f.set_state_decomposition_dimension(state_decomp_dim)

    f.update(meas_model, measurement)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()

    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)


@pytest.mark.parametrize("filter_cls", [ASIRPF, CGPF])
@pytest.mark.parametrize(
    "get_step_data",
    [
        TestUtilsStep.get_system_model_data,
        TestUtilsStep.get_mixed_noise_system_model_data,
        TestUtilsStep.get_additive_noise_system_model_data,
    ],
)
def test_step(filter_cls, get_step_data):
    (
        init_state,
        sys_model,
        meas_model,
        measurement,
        true_state_mean,
        true_state_cov,
    ) = get_step_data()

    f = filter_cls()
    f.set_num_particles(10**7)

    tol = 0.5

    f.set_state(init_state)
    f.step(sys_model, meas_model, measurement)

    state_mean, state_cov, _ = f.get_state_mean_and_covariance()

    assert np.allclose(state_mean, true_state_mean, rtol=tol)
    assert np.allclose(state_cov, state_cov.T)
    assert np.allclose(state_cov, true_state_cov, rtol=tol)
