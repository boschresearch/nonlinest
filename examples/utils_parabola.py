import os
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from nonlinear_estimation_toolbox.checks import is_single_vec
from nonlinear_estimation_toolbox.distributions import Gaussian, Uniform
from nonlinear_estimation_toolbox.measurement_models import (
    AdditiveNoiseMeasurementModel,
)
from nonlinear_estimation_toolbox.utils import ColumnVectors


class ParabolaMeasModel(AdditiveNoiseMeasurementModel):
    def __init__(self):
        super(ParabolaMeasModel, self).__init__()
        self._key_points_on_x_axis = None

    @property
    def key_points_on_x_axis(self):
        return self._key_points_on_x_axis

    @key_points_on_x_axis.setter
    def key_points_on_x_axis(self, value: ColumnVectors):
        assert is_single_vec(value)
        self._key_points_on_x_axis = value

    def measurement_equation(self, state_samples: ColumnVectors):
        assert not (self.key_points_on_x_axis is None)

        # state_samples are column vectors
        measurements = (
            state_samples[0, :] * self.key_points_on_x_axis**2
            + state_samples[1, :] * self.key_points_on_x_axis
            + state_samples[2, :]
        )
        return measurements


class Parabola:
    def __init__(
        self, state: ColumnVectors, x_interval: List[float], y_variance: float
    ):
        # state = [a, b, c]
        self.state = state

        self.meas_model = ParabolaMeasModel()

        Nv = Gaussian(mean=ColumnVectors(0), covariance=np.array([[y_variance]]))
        self.meas_model.set_noise(noise=Nv)

        self.Ux = Uniform(
            a=ColumnVectors(x_interval[0]), b=ColumnVectors(x_interval[1])
        )

    def get_measurements_xy(self, num_samples: int):

        key_points_on_x_axis = self.Ux.draw_random_samples(num_samples=num_samples).T

        self.meas_model.key_points_on_x_axis = key_points_on_x_axis

        values_on_y_axis = self.meas_model.simulate(state=self.state)

        return key_points_on_x_axis, values_on_y_axis


class ParabolaPlotManager:
    def __init__(self, plot_folder: str = None):
        self.fig, self.ax = plt.subplots(1, 1)
        self.parabola_measmodel = ParabolaMeasModel()
        key_points_on_x_axis = ColumnVectors(np.linspace(-4.0, 4.0, 401))
        self.parabola_measmodel.key_points_on_x_axis = key_points_on_x_axis

        if plot_folder is None:
            self.plot_folder = os.path.join(str(time()))
        else:
            self.plot_folder = plot_folder

        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def create_plot(
        self,
        gt_parabola_state: ColumnVectors,
        est_parabola_state: ColumnVectors,
        points: ColumnVectors,
    ):

        self.ax.cla()
        gt_key_point_values_on_y_axis = self.parabola_measmodel.measurement_equation(
            state_samples=gt_parabola_state
        )
        self.ax.plot(
            self.parabola_measmodel.key_points_on_x_axis,
            gt_key_point_values_on_y_axis,
            "k",
            label="groundtruth",
        )

        est_key_point_values_on_y_axis = self.parabola_measmodel.measurement_equation(
            state_samples=est_parabola_state
        )
        self.ax.plot(
            self.parabola_measmodel.key_points_on_x_axis,
            est_key_point_values_on_y_axis,
            "r",
            label="estimate",
        )

        self.ax.scatter(points[0, :], points[1, :], c="b")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(
            [
                min(self.parabola_measmodel.key_points_on_x_axis),
                max(self.parabola_measmodel.key_points_on_x_axis),
            ]
        )
        self.ax.set_ylim(
            [
                min(gt_key_point_values_on_y_axis) - 10.0,
                max(gt_key_point_values_on_y_axis) + 10.0,
            ]
        )
        self.ax.legend()

    def save(self, filename: str):
        self.fig.savefig(os.path.join(self.plot_folder, filename), dpi=300)
