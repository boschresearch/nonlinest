import numpy as np

from nonlinear_estimation_toolbox import gaussian_sampling
from nonlinear_estimation_toolbox.filters.sample_based_iterative_kalman import (
    SampleBasedIterativeKalmanFilter,
)
from nonlinear_estimation_toolbox.filters.sample_based_linear_gaussian import (
    SampleBasedLinearGaussianFilter,
)


class GaussianHermiteLinearGaussianFilter(SampleBasedLinearGaussianFilter):
    #   Kazufumi Ito and Kaiqi Xiong,
    #   Gaussian Filters for Nonlinear Filtering Problems,
    #   IEEE Transactions on Automatic Control, vol. 45, no. 5, pp. 910-927, May 2000.
    def __init__(self, name: str = ""):
        self.sampling_prediction = gaussian_sampling.GaussianSamplingGHQ()
        self.sampling_update = gaussian_sampling.GaussianSamplingGHQ()

        # not really needed, is the default anyway
        self.sampling_prediction.set_num_quadrature_points(2)
        self.sampling_update.set_num_quadrature_points(2)

        super(SampleBasedLinearGaussianFilter, self).__init__(name)

    def set_num_quadature_points(
        self, num_points_prediction: int, num_points_update: int
    ):
        self.sampling_prediction.set_num_quadrature_points(num_points_prediction)
        self.sampling_update.set_num_quadrature_points(num_points_update)

    def get_num_quadrature_points(self) -> (int, int):
        return (
            self.sampling_prediction.get_num_quadature_points(),
            self.sampling_update.get_num_quadature_points(),
        )

    def get_std_normal_samples_prediction(
        self, dim: int
    ) -> (np.ndarray, np.ndarray, int):
        return self.sampling_prediction.get_std_normal_samples(dim)

    def get_std_normal_samples_update(self, dim: int) -> (np.ndarray, np.ndarray, int):
        return self.sampling_update.get_std_normal_samples(dim)


class GHKF(SampleBasedIterativeKalmanFilter, GaussianHermiteLinearGaussianFilter):
    #   Kazufumi Ito and Kaiqi Xiong,
    #   Gaussian Filters for Nonlinear Filtering Problems,
    #   IEEE Transactions on Automatic Control, vol. 45, no. 5, pp. 910-927, May 2000.
    #
    #   Ángel F. Garcia-Fernández, Lennart Svensson, Mark Morelande, and Simo Särkkä,
    #   Posterior Linearisation Filter: Principles and Implementation Using Sigma Points,
    #   IEEE Transactions on Signal Processing, vol. 63, no. 20, pp. 5561-5573, Oct. 2015.
    def __init__(self, name: str = ""):
        if name == "":
            name = "GHKF"
        super(GHKF, self).__init__(name)
