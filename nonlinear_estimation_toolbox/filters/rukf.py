import numpy as np

from nonlinear_estimation_toolbox import gaussian_sampling
from nonlinear_estimation_toolbox.filters.sample_based_iterative_kalman import (
    SampleBasedIterativeKalmanFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)


class RandomizedUnscentedLinearGaussianFilter(SampleBasedLinearGaussianFilter):
    #   Jindrich Dunik, Ondrej Straka, and Miroslav Simandl,
    #   The Development of a Randomised Unscented Kalman Filter,
    #   Proceedings of the 18th IFAC World Congress, Milano, Italy, pp. 8-13, Aug. 2011.
    def __init__(self, name: str = ""):
        self.sampling_prediction = gaussian_sampling.GaussianSamplingRUKF()
        self.sampling_update = gaussian_sampling.GaussianSamplingRUKF()

        # not really needed, is the default anyway
        self.sampling_prediction.set_numiterations(5)
        self.sampling_update.set_numiterations(5)

        super(SampleBasedLinearGaussianFilter, self).__init__(name)

    def set_num_samples_factor(self, factor_prediction: int, factor_update: int):
        # did not implement fallback if only one is given
        self.sampling_prediction.set_numiterations(factor_prediction)
        self.sampling_update.set_numiterations(factor_update)

    def get_num_samples_factors(self) -> (float, float):
        return (
            self.sampling_prediction.get_numiterations(),
            self.sampling_update.get_numiterations(),
        )

    def get_std_normal_samples_prediction(
        self, dim: int
    ) -> (np.ndarray, np.ndarray, int):
        return self.sampling_prediction.get_std_normal_samples(dim)

    def get_std_normal_samples_update(self, dim: int) -> (np.ndarray, np.ndarray, int):
        return self.sampling_update.get_std_normal_samples(dim)


class RUKF(SampleBasedIterativeKalmanFilter, RandomizedUnscentedLinearGaussianFilter):
    def __init__(self, name: str = ""):
        if name == "":
            name = "RUKF"
        super(RUKF, self).__init__(name)
