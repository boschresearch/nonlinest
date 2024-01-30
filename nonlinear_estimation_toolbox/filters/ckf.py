import numpy as np

from nonlinear_estimation_toolbox import gaussian_sampling
from nonlinear_estimation_toolbox.filters.sample_based_iterative_kalman import (
    SampleBasedIterativeKalmanFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)


class CubatureLinearGaussianFilter(SampleBasedLinearGaussianFilter):
    #   Bin Jia, Ming Xin, and Yang Cheng,
    #   High-Degree Cubature Kalman Filter,
    #   Automatica, vol. 49, no. 2, pp. 510-518, Feb. 2013.
    def __init__(self, name: str = ""):
        self.sampling = gaussian_sampling.GaussianSamplingCKF()

        super(SampleBasedLinearGaussianFilter, self).__init__(name)

    def get_std_normal_samples_prediction(
        self, dim: int
    ) -> (np.ndarray, np.ndarray, int):
        return self.sampling.get_std_normal_samples(dim)

    def get_std_normal_samples_update(self, dim: int) -> (np.ndarray, np.ndarray, int):
        return self.sampling.get_std_normal_samples(dim)


class CKF(SampleBasedIterativeKalmanFilter, CubatureLinearGaussianFilter):
    def __init__(self, name: str = ""):
        if name == "":
            name = "CKF"
        super(CKF, self).__init__(name)
