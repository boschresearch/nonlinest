"""
This example shows how to recursively estimate the parameters p=[a,b,c] of a parabola
f_p(x) = a*x**2 + b*x + c
from measurements y = f_p(x) + v,
where v is zero-mean Gaussian noise with covariance Cv, i.e. v ~ N(0, Cv).

It only uses the kalman update step for estimation, see the example 1 and 2 for
an introduction with both update and prediction step.
"""

import numpy as np

from examples import utils_parabola
from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.utils import ColumnVectors


def run_example(plot_each_step=False):
    # number of steps for the demo
    NUM_STEPS = 100
    # varying number of measurements per step [min, max]
    NUM_MEASUREMENTS_PER_STEP = [1, 3]
    # interval, where measurements can occur on the x-axis [min, max]
    MEASUREMENT_INTERVAL_X_AXIS = [-3, 3]
    # magnitude of measurement noise
    MEASUREMENT_VARIANCE_Y_AXIS = 10.0
    # ground truth state, to be estimated
    parabola_state = ColumnVectors(
        np.random.normal(loc=np.array([0, 0, 0]), scale=10.0)
    )
    gt_parabola = utils_parabola.Parabola(
        state=parabola_state,
        x_interval=MEASUREMENT_INTERVAL_X_AXIS,
        y_variance=MEASUREMENT_VARIANCE_Y_AXIS,
    )

    # set up the filter
    initial_state_mean = ColumnVectors([0, 0, 0])
    initial_state_covariance = 10.0 * np.eye(3)
    initial_state = Gaussian(
        mean=initial_state_mean, covariance=initial_state_covariance
    )

    # you can switch between EKF and UKF just by changing the filter
    parabola_ekf = EKF(name="parabola")
    # setting the initial state
    parabola_ekf.set_state(state=initial_state)
    # setting the measurement model
    parabola_meas_model = utils_parabola.ParabolaMeasModel()
    # note, we do not set a system model for this experiment because the underlying system - the parabola - is constant.

    # a manager, which is used for plotting
    plot_manager = utils_parabola.ParabolaPlotManager()

    # recursion loop
    for i in range(NUM_STEPS):
        # sample number of measurements
        num_measurements_i = np.random.choice(
            np.arange(
                start=NUM_MEASUREMENTS_PER_STEP[0],
                stop=NUM_MEASUREMENTS_PER_STEP[1] + 1,
            )
        )
        print(f"INFO: Step {i}. Number of measurements: {num_measurements_i}")
        # sample measurements for step i
        x_points, y_points = gt_parabola.get_measurements_xy(
            num_samples=num_measurements_i
        )

        # set keypoints on x-axis as parameter in the measurement equation
        parabola_meas_model.key_points_on_x_axis = x_points

        # set the measurement noise according to the number of keypoints
        Nv = Gaussian(
            mean=ColumnVectors(np.zeros(num_measurements_i)),
            covariance=MEASUREMENT_VARIANCE_Y_AXIS * np.eye(num_measurements_i),
        )
        parabola_meas_model.set_noise(noise=Nv)
        parabola_ekf.setup_additive_noise_measurement_model(
            measurement_model=parabola_meas_model, dim_meas=num_measurements_i
        )

        # update parabola with measurements
        parabola_ekf.update(meas_model=parabola_meas_model, measurement=y_points)

        # plot intermediate result...
        if plot_each_step:
            points = ColumnVectors(np.hstack([x_points, y_points]).T)
            plot_manager.create_plot(
                gt_parabola_state=gt_parabola.state,
                est_parabola_state=parabola_ekf.state_mean,
                points=points,
            )
            # ... and save it to disk
            filename = f"parabola_{str(i).zfill(4)}.png"
            plot_manager.save(filename=filename)


if __name__ == "__main__":
    run_example(plot_each_step=True)
