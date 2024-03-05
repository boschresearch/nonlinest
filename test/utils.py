from test.models import (
    AddNoiseMeasModel,
    AddNoiseSysModel,
    MeasModel,
    MixedNoiseMeasModel,
    MixedNoiseSysModel,
    SysModel,
)

import numpy as np

from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
    LinearMeasurementModel,
    MeasurementModel,
)
from nonlinear_estimation_toolbox.system_models import LinearSystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


class TestUtilsLinearSystemModel:
    __test__ = False
    # TODO: Add doc string

    initMean = ColumnVectors(np.array([0.3, -np.pi]))
    initCov = np.array([[0.5, 0.1], [0.1, 3]])
    sysMatrix = np.array([[3, -4], [0, 2]])
    sysInput = ColumnVectors(np.array([2.5, -0.1]))
    sysNoiseMatrix = np.array([[0.01, 2], [-1.2, 0]])
    sysNoise = Gaussian(
        ColumnVectors(np.array([2, -1])), np.array([[2, -0.5], [-0.5, 1.3]])
    )

    @staticmethod
    def get_sys_model_data(has_sys_mat, has_sys_noise_mat, has_input):
        init_state = Gaussian(
            TestUtilsLinearSystemModel.initMean, TestUtilsLinearSystemModel.initCov
        )

        sys_model = LinearSystemModel()
        sys_model.set_noise(TestUtilsLinearSystemModel.sysNoise)

        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsLinearSystemModel.sysNoise.get_mean_and_covariance()

        if has_sys_mat:
            sys_model.set_system_matrix(TestUtilsLinearSystemModel.sysMatrix)

            true_state_mean = TestUtilsLinearSystemModel.sysMatrix.dot(
                TestUtilsLinearSystemModel.initMean
            )
            true_state_cov = TestUtilsLinearSystemModel.sysMatrix.dot(
                TestUtilsLinearSystemModel.initCov
            ).dot(TestUtilsLinearSystemModel.sysMatrix.T)
        else:
            true_state_mean = TestUtilsLinearSystemModel.initMean
            true_state_cov = TestUtilsLinearSystemModel.initCov

        if has_input:
            sys_model.set_system_input(TestUtilsLinearSystemModel.sysInput)

            true_state_mean = true_state_mean + TestUtilsLinearSystemModel.sysInput

        if has_sys_noise_mat:
            sys_model.set_system_noise_matrix(TestUtilsLinearSystemModel.sysNoiseMatrix)

            true_state_mean = (
                true_state_mean
                + TestUtilsLinearSystemModel.sysNoiseMatrix.dot(noise_mean)
            )
            true_state_cov = (
                true_state_cov
                + TestUtilsLinearSystemModel.sysNoiseMatrix.dot(noise_cov).dot(
                    TestUtilsLinearSystemModel.sysNoiseMatrix.T
                )
            )
        else:
            true_state_mean = true_state_mean + noise_mean
            true_state_cov = true_state_cov + noise_cov

        return init_state, sys_model, true_state_mean, true_state_cov


class TestUtilsSystemModelsData:
    """
    Hold the data for all test utils system models (except linear)
    """

    __test__ = False
    init_mean = ColumnVectors(np.array([0.3, -np.pi]))
    init_cov = np.array([[0.5, 0.1], [0.1, 3]])
    add_sys_noise = Gaussian(
        ColumnVectors(np.array([2, -1])), 0.01 * np.array([[2, -0.5], [-0.5, 1.3]])
    )
    sys_noise = Gaussian(ColumnVectors([0, 3]), 0.1 * np.diag([0.5, 0.1]))


class TestUtilsAdditiveNoiseSystemModel:
    __test__ = False

    @staticmethod
    def get_sys_model_data():
        init_state = Gaussian(
            TestUtilsSystemModelsData.init_mean,
            TestUtilsSystemModelsData.init_cov,
        )

        sys_model = AddNoiseSysModel()
        sys_model.set_noise(TestUtilsSystemModelsData.add_sys_noise)

        mat = sys_model.sysMatrix
        (
            noiseMean,
            noiseCov,
            _,
        ) = TestUtilsSystemModelsData.add_sys_noise.get_mean_and_covariance()

        true_state_mean = mat.dot(TestUtilsSystemModelsData.init_mean) + noiseMean
        true_state_cov = (
            mat.dot(TestUtilsSystemModelsData.init_cov).dot(mat.T) + noiseCov
        )

        return init_state, sys_model, true_state_mean, true_state_cov


class TestUtilsMixedNoiseSystemModel:
    __test__ = False

    @staticmethod
    def get_sys_model_data():
        init_state = Gaussian(
            TestUtilsSystemModelsData.init_mean,
            TestUtilsSystemModelsData.init_cov,
        )

        sys_model = MixedNoiseSysModel()
        sys_model.set_additive_noise(TestUtilsSystemModelsData.add_sys_noise)
        sys_model.set_noise(TestUtilsSystemModelsData.sys_noise)

        mat = sys_model.sys_matrix
        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = TestUtilsSystemModelsData.add_sys_noise.get_mean_and_covariance()

        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsSystemModelsData.sys_noise.get_mean_and_covariance()

        true_state_mean = (
            mat.dot(TestUtilsSystemModelsData.init_mean) + noise_mean + add_noise_mean
        )
        true_state_cov = (
            mat.dot(TestUtilsSystemModelsData.init_cov).dot(mat.T)
            + noise_cov
            + add_noise_cov
        )

        return init_state, sys_model, true_state_mean, true_state_cov


class TestUtilsSystemModel:
    __test__ = False

    @staticmethod
    def get_sys_model_data():
        init_state = Gaussian(
            TestUtilsSystemModelsData.init_mean,
            TestUtilsSystemModelsData.init_cov,
        )

        sys_model = SysModel()
        sys_model.set_noise(TestUtilsSystemModelsData.add_sys_noise)

        mat = sys_model.sysMatrix
        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsSystemModelsData.add_sys_noise.get_mean_and_covariance()

        true_state_mean = mat.dot(TestUtilsSystemModelsData.init_mean) + noise_mean
        true_state_cov = (
            mat.dot(TestUtilsSystemModelsData.init_cov).dot(mat.T) + noise_cov
        )

        return init_state, sys_model, true_state_mean, true_state_cov


class TestUtilsLinearMeasurementModel:
    __test__ = False
    initMean = ColumnVectors([0.3, -np.pi])
    initCov = np.array([[0.5, 0.1], [0.1, 3.0]])
    measMatrix = np.array([[3, -4], [np.pi / 4, 0], [0.5, 2]])
    measMatrixStateDecomp = ColumnVectors([3, -0.5, 3])
    measNoise2D = Gaussian(ColumnVectors([2, -1]), np.array([[2, -0.5], [-0.5, 1.3]]))
    measNoise3D = Gaussian(
        ColumnVectors([2, -1, 3]),
        np.array([[2.0, -0.5, 0.2], [-0.5, 1.3, 0], [0.2, 0, 3.0]]),
    )
    meas2D = ColumnVectors([3, -4])
    meas3D = ColumnVectors([15, -0.9, -3])

    @staticmethod
    def get_meas_model_data(has_meas_matrix, state_decomp=False):
        init_state = Gaussian(
            TestUtilsLinearMeasurementModel.initMean,
            TestUtilsLinearMeasurementModel.initCov,
        )
        meas_model = LinearMeasurementModel()

        if has_meas_matrix:
            if state_decomp:
                state_decomp_dim = 1

                meas_model_mat = TestUtilsLinearMeasurementModel.measMatrixStateDecomp
                meas_model.set_measurement_matrix(meas_model_mat)
                mat = np.hstack([meas_model_mat, np.zeros([3, 1])])
            else:
                state_decomp_dim = 0
                meas_model_mat = TestUtilsLinearMeasurementModel.measMatrix
                meas_model.set_measurement_matrix(meas_model_mat)
                mat = meas_model_mat

            meas_model.set_noise(TestUtilsLinearMeasurementModel.measNoise3D)
            (
                noise_mean,
                noise_cov,
                _,
            ) = TestUtilsLinearMeasurementModel.measNoise3D.get_mean_and_covariance()

            true_meas_mean = (
                mat.dot(TestUtilsLinearMeasurementModel.initMean) + noise_mean
            )
            true_meas_cov = (
                mat.dot(TestUtilsLinearMeasurementModel.initCov).dot(mat.T) + noise_cov
            )

            inv_true_meas_cov = np.linalg.inv(true_meas_cov)
            cross_cov = TestUtilsLinearMeasurementModel.initCov.dot(mat.T)

            measurement = TestUtilsLinearMeasurementModel.meas3D
        else:
            # no state decomposition possible due to identity measurement matrix
            state_decomp_dim = 0

            mat = np.eye(2)

            meas_model.set_noise(TestUtilsLinearMeasurementModel.measNoise2D)
            (
                noise_mean,
                noise_cov,
                _,
            ) = TestUtilsLinearMeasurementModel.measNoise2D.get_mean_and_covariance()

            true_meas_mean = (
                mat.dot(TestUtilsLinearMeasurementModel.initMean) + noise_mean
            )
            true_meas_cov = (
                mat.dot(TestUtilsLinearMeasurementModel.initCov).dot(mat.T) + noise_cov
            )

            inv_true_meas_cov = np.linalg.inv(true_meas_cov)
            cross_cov = TestUtilsLinearMeasurementModel.initCov.dot(mat.T)

            measurement = TestUtilsLinearMeasurementModel.meas2D

        K = cross_cov.dot(inv_true_meas_cov)

        true_state_mean = TestUtilsLinearMeasurementModel.initMean + K.dot(
            measurement - true_meas_mean
        )
        true_state_cov = TestUtilsLinearMeasurementModel.initCov - K.dot(cross_cov.T)

        return (
            init_state,
            meas_model,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        )


class TestUtilsMeasurementModelsData:
    """
    Holds the data for test utils measurement models (except linear)
    """

    __test__ = False
    init_mean = ColumnVectors([0.3, -np.pi])
    init_cov = np.array([[0.5, 0.1], [0.1, 3.0]])
    add_meas_noise = Gaussian(
        ColumnVectors([2, -1, 0.5]),
        np.array([[2, -0.5, 0], [-0.5, 1.3, 0.5], [0, 0.5, np.sqrt(2)]]),
    )
    meas_noise = Gaussian(ColumnVectors([1, 2.5, 0]), np.diag([1, 2, 3]))
    meas = ColumnVectors([1, -2, 5])


class TestUtilsMeasurementModel:
    """
    todo: This, TestUtilsAdditiveNoiseMeasurementModel,
        and TestUtilsMixedNoiseMeasurementModel have a lot of duplicate code
    """

    __test__ = False

    @staticmethod
    def get_meas_model_data(state_decomp=False):
        init_state = Gaussian(
            TestUtilsMeasurementModelsData.init_mean,
            TestUtilsMeasurementModelsData.init_cov,
        )

        meas_model = MeasModel(state_decomp)
        meas_model.set_noise(TestUtilsMeasurementModelsData.add_meas_noise)

        if state_decomp:
            state_decomp_dim = 1
            mat = np.hstack([meas_model.meas_matrix, np.zeros([3, 1])])
        else:
            state_decomp_dim = 0
            mat = meas_model.meas_matrix

        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsMeasurementModelsData.add_meas_noise.get_mean_and_covariance()

        measurement = TestUtilsMeasurementModelsData.meas

        true_meas_mean = mat.dot(TestUtilsMeasurementModelsData.init_mean) + noise_mean
        true_meas_cov = (
            mat.dot(TestUtilsMeasurementModelsData.init_cov).dot(mat.T) + noise_cov
        )
        cross_cov = TestUtilsMeasurementModelsData.init_cov.dot(mat.T)

        inv_true_meas_cov = np.linalg.inv(true_meas_cov)
        K = cross_cov.dot(inv_true_meas_cov)
        true_state_mean = TestUtilsMeasurementModelsData.init_mean + K.dot(
            measurement - true_meas_mean
        )
        true_state_cov = TestUtilsMeasurementModelsData.init_cov - K.dot(cross_cov.T)

        return (
            init_state,
            meas_model,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        )


class TestUtilsAdditiveNoiseMeasurementModel:
    __test__ = False

    @staticmethod
    def get_meas_model_data(state_decomp=False):
        init_state = Gaussian(
            TestUtilsMeasurementModelsData.init_mean,
            TestUtilsMeasurementModelsData.init_cov,
        )

        meas_model = AddNoiseMeasModel(state_decomp)
        meas_model.set_noise(TestUtilsMeasurementModelsData.add_meas_noise)

        if state_decomp:
            state_decomp_dim = 1
            mat = np.hstack([meas_model.meas_matrix, np.zeros([3, 1])])
        else:
            state_decomp_dim = 0
            mat = meas_model.meas_matrix

        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsMeasurementModelsData.add_meas_noise.get_mean_and_covariance()

        measurement = TestUtilsMeasurementModelsData.meas

        true_meas_mean = mat.dot(TestUtilsMeasurementModelsData.init_mean) + noise_mean
        true_meas_cov = (
            mat.dot(TestUtilsMeasurementModelsData.init_cov).dot(mat.T) + noise_cov
        )
        cross_cov = TestUtilsMeasurementModelsData.init_cov.dot(mat.T)

        inv_true_meas_cov = np.linalg.inv(true_meas_cov)
        K = cross_cov.dot(inv_true_meas_cov)
        true_state_mean = TestUtilsMeasurementModelsData.init_mean + K.dot(
            measurement - true_meas_mean
        )
        true_state_cov = TestUtilsMeasurementModelsData.init_cov - K.dot(cross_cov.T)

        return (
            init_state,
            meas_model,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        )


class TestUtilsMixedNoiseMeasurementModel:
    __test__ = False

    @staticmethod
    def get_meas_model_data(state_decomp=False):
        init_state = Gaussian(
            TestUtilsMeasurementModelsData.init_mean,
            TestUtilsMeasurementModelsData.init_cov,
        )

        meas_model = MixedNoiseMeasModel(state_decomp)
        meas_model.set_additive_noise(TestUtilsMeasurementModelsData.add_meas_noise)
        meas_model.set_noise(TestUtilsMeasurementModelsData.meas_noise)

        if state_decomp:
            state_decomp_dim = 1
            mat = np.hstack([meas_model.meas_matrix, np.zeros([3, 1])])
        else:
            state_decomp_dim = 0
            mat = meas_model.meas_matrix

        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = TestUtilsMeasurementModelsData.add_meas_noise.get_mean_and_covariance()
        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsMeasurementModelsData.meas_noise.get_mean_and_covariance()

        measurement = TestUtilsMeasurementModelsData.meas

        true_meas_mean = (
            mat.dot(TestUtilsMeasurementModelsData.init_mean)
            + add_noise_mean
            + noise_mean
        )
        true_meas_cov = (
            mat.dot(TestUtilsMeasurementModelsData.init_cov).dot(mat.T)
            + add_noise_cov
            + noise_cov
        )
        cross_cov = TestUtilsMeasurementModelsData.init_cov.dot(mat.T)

        inv_true_meas_cov = np.linalg.inv(true_meas_cov)
        K = cross_cov.dot(inv_true_meas_cov)
        true_state_mean = TestUtilsMeasurementModelsData.init_mean + K.dot(
            measurement - true_meas_mean
        )
        true_state_cov = TestUtilsMeasurementModelsData.init_cov - K.dot(cross_cov.T)

        return (
            init_state,
            meas_model,
            measurement,
            state_decomp_dim,
            true_state_mean,
            true_state_cov,
            true_meas_mean,
            true_meas_cov,
        )


class TestUtilsStep:
    __test__ = False
    init_mean = ColumnVectors([0.3, -np.pi])
    init_cov = np.array([[0.5, 0.1], [0.1, 3]])
    sys_noise = Gaussian(
        ColumnVectors([2, -1]), 0.01 * np.array([[2, -0.5], [-0.5, 1.3]])
    )
    sys_noise_2 = Gaussian(ColumnVectors([0, 3]), 0.01 * np.diag([5, 1]))
    meas_noise = Gaussian(
        ColumnVectors([2, -1, 0.5]),
        10 * np.array([[2, -0.5, 0], [-0.5, 1.3, 0.5], [0, 0.5, np.sqrt(2)]]),
    )
    meas = ColumnVectors([1, -2, 5])

    @staticmethod
    def get_additive_noise_system_model_data():
        init_state = Gaussian(TestUtilsStep.init_mean, TestUtilsStep.init_cov)
        (
            sys_model,
            true_pred_mean,
            true_pred_cov,
        ) = TestUtilsStep.pred_add_noise_sys_model()
        meas_model, measurement, true_state_mean, true_state_cov = TestUtilsStep.update(
            true_pred_mean, true_pred_cov
        )
        return (
            init_state,
            sys_model,
            meas_model,
            measurement,
            true_state_mean,
            true_state_cov,
        )

    @staticmethod
    def get_system_model_data():
        init_state = Gaussian(TestUtilsStep.init_mean, TestUtilsStep.init_cov)
        sys_model, true_pred_mean, true_pred_cov = TestUtilsStep.pred_sys_model()
        meas_model, measurement, true_state_mean, true_state_cov = TestUtilsStep.update(
            true_pred_mean, true_pred_cov
        )
        return (
            init_state,
            sys_model,
            meas_model,
            measurement,
            true_state_mean,
            true_state_cov,
        )

    @staticmethod
    def get_mixed_noise_system_model_data():
        init_state = Gaussian(TestUtilsStep.init_mean, TestUtilsStep.init_cov)
        (
            sys_model,
            true_pred_mean,
            true_pred_cov,
        ) = TestUtilsStep.pred_mixed_noise_sys_model()
        meas_model, measurement, true_state_mean, true_state_cov = TestUtilsStep.update(
            true_pred_mean, true_pred_cov
        )
        return (
            init_state,
            sys_model,
            meas_model,
            measurement,
            true_state_mean,
            true_state_cov,
        )

    @staticmethod
    def pred_add_noise_sys_model():
        sys_model = AddNoiseSysModel()
        sys_model.set_noise(TestUtilsStep.sys_noise)

        mat = sys_model.sysMatrix
        noise_mean, noise_cov, _ = TestUtilsStep.sys_noise.get_mean_and_covariance()
        true_pred_mean = mat.dot(TestUtilsStep.init_mean) + noise_mean
        true_pred_cov = mat.dot(TestUtilsStep.init_cov).dot(mat.T) + noise_cov

        return sys_model, true_pred_mean, true_pred_cov

    @staticmethod
    def pred_sys_model():
        sys_model = SysModel()
        sys_model.set_noise(TestUtilsStep.sys_noise)

        mat = sys_model.sysMatrix
        noise_mean, noise_cov, _ = TestUtilsStep.sys_noise.get_mean_and_covariance()

        true_pred_mean = mat.dot(TestUtilsStep.init_mean) + noise_mean
        true_pred_cov = mat.dot(TestUtilsStep.init_cov).dot(mat.T) + noise_cov

        return sys_model, true_pred_mean, true_pred_cov

    @staticmethod
    def pred_mixed_noise_sys_model():
        sys_model = MixedNoiseSysModel()
        sys_model.set_additive_noise(TestUtilsStep.sys_noise)
        sys_model.set_noise(TestUtilsStep.sys_noise_2)

        mat = sys_model.sys_matrix
        (
            add_noise_mean,
            add_noise_cov,
            _,
        ) = TestUtilsStep.sys_noise.get_mean_and_covariance()
        noise_mean, noise_cov, _ = TestUtilsStep.sys_noise_2.get_mean_and_covariance()

        true_pred_mean = mat.dot(TestUtilsStep.init_mean) + add_noise_mean + noise_mean
        true_pred_cov = (
            mat.dot(TestUtilsStep.init_cov).dot(mat.T) + add_noise_cov + noise_cov
        )

        return sys_model, true_pred_mean, true_pred_cov

    @staticmethod
    def update(true_pred_mean: np.ndarray, true_pred_cov: np.ndarray):
        _ = Gaussian(
            TestUtilsStep.init_mean,
            TestUtilsStep.init_cov,
        )

        meas_model = AddNoiseMeasModel()
        meas_model.set_noise(TestUtilsStep.meas_noise)

        mat = meas_model.meas_matrix
        (
            noise_mean,
            noise_cov,
            _,
        ) = TestUtilsStep.meas_noise.get_mean_and_covariance()

        measurement = TestUtilsStep.meas

        true_meas_mean = mat.dot(true_pred_mean) + noise_mean
        true_meas_cov = mat.dot(true_pred_cov).dot(mat.T) + noise_cov
        cross_cov = true_pred_cov.dot(mat.T)

        inv_true_meas_cov = np.linalg.inv(true_meas_cov)
        K = cross_cov.dot(inv_true_meas_cov)
        true_state_mean = true_pred_mean + K.dot(measurement - true_meas_mean)
        true_state_cov = true_pred_cov - K.dot(cross_cov.T)
        return meas_model, measurement, true_state_mean, true_state_cov


class DummyPolarMeasurementModel(MeasurementModel):
    def __init__(self):
        noise = Gaussian(mean=ColumnVectors([0, 0]), covariance=np.diag([1e-2, 1e-4]))
        self.set_noise(noise)

    def measurement_equation(
        self, state_samples: ColumnVectors, noise_samples: ColumnVectors
    ) -> ColumnVectors:
        px, py, _, _, _ = state_samples
        measurements = np.stack([np.sqrt(px**2 + py**2), np.arctan2(py, px)])
        return measurements + noise_samples


class DummyPolarAdditiveNoiseMeasurementModel(AdditiveNoiseMeasurementModel):
    def __init__(self):
        noise = Gaussian(mean=ColumnVectors([0, 0]), covariance=np.diag([1e-2, 1e-4]))
        self.set_noise(noise)

    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        px, py, _, _, _ = state_samples
        measurements = np.stack([np.sqrt(px**2 + py**2), np.arctan2(py, px)])
        return measurements
