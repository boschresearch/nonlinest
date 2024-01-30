"""
A complete example with multiple prediction and update steps.

We consider the same problem as in the previous example and reuse
its constant turn rate / constant velocity system model.
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils_complete
from examples.ex04_nonlinear import (
    ConstantTurnRateVelocitySystemModel,
    NaivePolarMeasurementModel,
)
from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.filters.sirpf import SIRPF
from nonlinear_estimation_toolbox.filters.ukf import UKF
from nonlinear_estimation_toolbox.utils import ColumnVectors


def run_example():
    np.random.seed(2002)

    # Reuse system model from previous examples
    sys_model = ConstantTurnRateVelocitySystemModel(delta_t=1)

    # Set system model noise
    system_model_noise = Gaussian(
        mean=ColumnVectors([0, 0]), covariance=np.diag([1e-1, 1e-3])
    )
    sys_model.set_noise(system_model_noise)

    # Reuse measurement model from previous examples
    meas_model = NaivePolarMeasurementModel()

    # Initialize some filters
    filters = []

    ekf = EKF("EKF")
    filters.append(ekf)

    ukf = UKF("UKF")
    filters.append(ukf)

    ukf2 = UKF("Iterative UKF")
    ukf2.set_max_num_iterations(5)
    filters.append(ukf2)

    sirpf = SIRPF("SIRPF")
    sirpf.set_num_particles(10**5)
    # filters.append(sirpf)

    initial_state = Gaussian(
        mean=ColumnVectors([1, 1, 0, 0, 0]), covariance=np.diag([5, 5, 1e-1, 1, 1e-1])
    )

    # Set initial state
    for filter in filters:
        filter.set_state(initial_state)

    # Number of simulated steps
    num_steps = 30

    # Empty arrays for simulated data
    system_states = np.empty((num_steps, 5, 1))
    measurements = np.empty((num_steps, 2, 1))

    # Empty arrays for prediction and update values
    updated_state_means = np.empty((len(filters), num_steps, 5, 1))
    updated_state_covs = np.empty((len(filters), num_steps, 5, 5))
    predicted_state_means = np.empty((len(filters), num_steps, 5, 1))
    predicted_state_covs = np.empty((len(filters), num_steps, 5, 5))

    # Empty arrays for execution times
    runtimes_update = np.empty((len(filters), num_steps))
    runtimes_prediction = np.empty((len(filters), num_steps))

    # Initial system state sample
    sys_state = initial_state.draw_random_samples(1)

    for k in range(num_steps):
        # Simulate a measurement
        measurement = meas_model.simulate(sys_state)

        system_states[k] = sys_state
        measurements[k] = measurement

        # Perform measurement updates on each filter
        for i, filter in enumerate(filters):
            # Here, we use the update_timed function to execute the
            # update step and measure its execution time in seconds.
            # There are also _timed functions for predict() and step().
            runtimes_update[i, k] = filter.update_timed(
                meas_model=meas_model, measurement=measurement
            )

            # Set intermediate values
            (
                updated_state_means[i, k],
                updated_state_covs[i, k],
                _,
            ) = filter.get_state_mean_and_covariance()

        # Simulate a state transition
        sys_state = sys_model.simulate(sys_state)

        # Perform predictions
        for i, filter in enumerate(filters):
            runtimes_prediction[i, k] = filter.predict_timed(sys_model=sys_model)

            # Set intermediate values
            (
                predicted_state_means[i, k],
                predicted_state_covs[i, k],
                _,
            ) = filter.get_state_mean_and_covariance()

    # Measure milliseconds
    runtimes_update_ms = 1000 * runtimes_update
    runtimes_prediction_ms = 1000 * runtimes_prediction

    # The example ends here, from now on we only draw plots
    utils_complete.plot_trajectories(
        "complete_example.png",
        filters,
        system_states,
        updated_state_means,
        updated_state_covs,
        predicted_state_means,
        predicted_state_covs,
    )

    colors = plt.get_cmap("jet", len(filters))
    utils_complete.plot_runtimes(
        "complete_example_runtimes.png",
        filters,
        runtimes_update_ms,
        runtimes_prediction_ms,
        colors,
    )

    utils_complete.plot_estimation_errors(
        "complete_example_errors.png",
        filters,
        system_states,
        updated_state_means,
        predicted_state_means,
        colors,
    )

    utils_complete.render_video_frames(
        filters,
        system_states,
        measurements,
        updated_state_means,
        updated_state_covs,
        predicted_state_means,
        predicted_state_covs,
        filter_idx=[0, 1],
    )


if __name__ == "__main__":
    run_example()
