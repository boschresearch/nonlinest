"""
A particle filter example.

We consider the same problem as in the previous example and reuse
its constant turn rate / constant velocity system model.

To use estimators with a nonlinear update, a likelihood function is required.
In this library, we simply have to implement a Likelihood class and overwrite
the log_likelihood() function, which returns the logarithmic likelihood of the
given measurements vector for each state sample.
"""

import numpy as np

from examples.ex04_nonlinear import ConstantTurnRateVelocitySystemModel
from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.sirpf import SIRPF
from nonlinear_estimation_toolbox.measurement_models import Likelihood
from nonlinear_estimation_toolbox.utils import ColumnVectors


class PolarMeasLikelihood(Likelihood):
    def __init__(self):
        # The Likelihood class is an abstract class which only declares the
        # log_likelihood() function. Hence, there is no set_noise function
        # and you have to define everything else yourself.
        self.meas_noise = Gaussian(
            mean=ColumnVectors([0, 0]), covariance=np.diag([1e-2, 1e-4])
        )

    def log_likelihood(
        self, state_samples: ColumnVectors, measurement: ColumnVectors
    ) -> ColumnVectors:
        px, py, _, _, _ = state_samples

        h = ColumnVectors([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

        # In our example the log-likelihood is the logarithm of the
        # Gaussian noise probability density function evaluated at
        # the differences of measurement and measurement function h(x)
        diffs = measurement - h
        log_values = self.meas_noise.log_pdf(diffs)

        return log_values


def run_example():
    np.set_printoptions(precision=4)

    # System model
    delta_t = 0.1  # Step size
    sys_model = ConstantTurnRateVelocitySystemModel(delta_t)

    # Set system model noise
    system_model_noise = Gaussian(
        mean=ColumnVectors([0, 0]), covariance=np.diag([1e-1, 1e-3])
    )
    sys_model.set_noise(system_model_noise)

    # When implementing an AdditiveNoiseMeasurementModel, you get the log_likelihood() function for free.
    # Hence, the measurement model from the previous example would yield the same results:
    #
    #   from examples.ex02_nonlinear import NaivePolarMeasurementModel
    #   meas_model = NaivePolarMeasurementModel()
    #
    # Measurement model:
    meas_model = PolarMeasLikelihood()

    # Estimator
    # Only change compared to previous example, when using the additive noise measurement model.
    filter = SIRPF()
    filter.set_num_particles(10**5)

    # Set initial state estimate
    initial_state_estimate = Gaussian(
        mean=ColumnVectors([1, 1, 0, 0, 0]), covariance=np.diag([10, 10, 1e-1, 1, 1e-1])
    )
    filter.set_state(initial_state_estimate)

    # Perform the prediction step
    filter.predict(sys_model=sys_model)

    # Print state after prediction
    mean, cov, _ = filter.get_state().get_mean_and_covariance()
    print(f"Predicted mean:\n {mean}")
    print(f"Predicted covariance:\n {cov}")

    # Perform the update step
    filter.update(meas_model=meas_model, measurement=ColumnVectors([3, np.pi / 5]))

    # Print state after update
    mean, cov, _ = filter.get_state().get_mean_and_covariance()
    print(f"Updated mean:\n {mean}")
    print(f"Updated covariance:\n {cov}")


if __name__ == "__main__":
    run_example()
