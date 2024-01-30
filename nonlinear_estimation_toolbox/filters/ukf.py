import numpy as np

from nonlinear_estimation_toolbox import gaussian_sampling
from nonlinear_estimation_toolbox.filters.sample_based_iterative_kalman import (
    SampleBasedIterativeKalmanFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)


class UnscentedLinearGaussianFilter(SampleBasedLinearGaussianFilter):
    def __init__(self, name: str = ""):
        self.sampling_prediction = gaussian_sampling.GaussianSamplingUKF()
        self.sampling_update = gaussian_sampling.GaussianSamplingUKF()

        # not really needed, is the default anyway
        self.sampling_prediction.set_sample_scaling(0.5)
        self.sampling_update.set_sample_scaling(0.5)

        super(UnscentedLinearGaussianFilter, self).__init__(name)

    def set_sample_scalings(self, scaling_prediction: float, scaling_update: float):
        # did not implement fallback if only one is given
        self.sampling_prediction.set_sample_scaling(scaling_prediction)
        self.sampling_update.set_sample_scaling(scaling_update)

    def get_sample_scalings(self) -> (float, float):
        return (
            self.sampling_prediction.get_sample_scaling(),
            self.sampling_update.get_sample_scaling(),
        )

    def get_std_normal_samples_prediction(
        self, dim: int
    ) -> (np.ndarray, np.ndarray, int):
        return self.sampling_prediction.get_std_normal_samples(dim)

    def get_std_normal_samples_update(self, dim: int) -> (np.ndarray, np.ndarray, int):
        return self.sampling_update.get_std_normal_samples(dim)


class UKF(SampleBasedIterativeKalmanFilter, UnscentedLinearGaussianFilter):
    def __init__(self, name: str = ""):
        if name == "":
            name = "UKF"
        super(UKF, self).__init__(name)
