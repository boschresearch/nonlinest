"""
This file provides plotting utilities for the complete state estimation example.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

from nonlinear_estimation_toolbox.filters.filter import Filter
from nonlinear_estimation_toolbox.utils import ColumnVectors


def generate_ellipse(
    mean: ColumnVectors, covariance: np.ndarray, color: str, n_std=3, label=""
):
    # cov @ V = W * V
    W, V = np.linalg.eig(covariance)
    sigma = np.sqrt(W)

    if covariance[0, 0] > covariance[1, 1]:
        phi = np.arctan2(V[1, 1], V[0, 1])
        extend = n_std * np.flip(sigma)
    else:
        phi = np.arctan2(V[1, 0], V[0, 0])
        extend = n_std * sigma

    ellipse = Ellipse(
        xy=(mean[0], mean[1]),
        width=extend[0] * 2,
        height=extend[1] * 2,
        angle=phi / np.pi * 180,
        facecolor="none",
        edgecolor=color,
        label="",
    )
    return ellipse


def plot_trajectories(
    filename: str,
    filters: List[Filter],
    system_states: np.ndarray,
    updated_state_means: np.ndarray,
    updated_state_covs: np.ndarray,
    predicted_state_means: np.ndarray,
    predicted_state_covs: np.ndarray,
) -> None:
    # Plot (x, y) estimates and their covariance
    # Because we scale this plot dynamically with the number of filters, there are some calculations to be done
    num_filters = len(filters)
    num_rows = int(np.round(np.sqrt(num_filters)))
    num_cols = int(np.ceil(num_filters / num_rows))

    # Generate an axis for each filter
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    # Plot data
    for i, filter in enumerate(filters):
        div, mod = divmod(i, num_cols)
        ax = axs[div, mod]

        ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(filter.get_name())
        ax.plot(system_states[:, 0, 0], system_states[:, 1, 0], color="black")

        trace = []

        for update_mean, update_cov, pred_mean, pred_cov in zip(
            updated_state_means[i, :, :2, :],
            updated_state_covs[i, :, :2, :2],
            predicted_state_means[i, :, :2, :],
            predicted_state_covs[i, :, :2, :2],
        ):
            ax.add_patch(generate_ellipse(update_mean, update_cov, color="blue"))
            trace.append(update_mean)

            ax.add_patch(generate_ellipse(pred_mean, pred_cov, color="red"))
            trace.append(pred_mean)

        trace = np.array(trace)
        ax.plot(trace[:, 0, 0], trace[:, 1, 0], color="green")

    # Remove the remaining axes
    for i in range(len(filters), num_cols * num_rows):
        div, mod = divmod(i, num_cols)
        axs[div, mod].set_axis_off()

    # Manually create the legend for all four plots
    legend_elements = [
        Patch(facecolor="none", edgecolor="blue"),
        Patch(facecolor="none", edgecolor="red"),
        Line2D([0], [0], color="black"),
        Line2D([0], [0], color="green"),
    ]
    legend_labels = [
        "Updated covariance (3σ)",
        "Predicted covariance (3σ)",
        "Ground truth path",
        "Estimated path",
    ]
    fig.legend(
        legend_elements,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.025),
        ncol=2,
    )

    # Apply tight layout and adjust the render rectangle such that there is some space for the legend
    fig.tight_layout(rect=(0, 0.125, 1, 1))
    fig.savefig(filename, dpi=300)


def plot_runtimes(
    filename: str,
    filters: List[Filter],
    runtimes_update_ms: np.ndarray,
    runtimes_prediction_ms: np.ndarray,
    colors: Colormap,
) -> None:
    _, num_steps = runtimes_update_ms.shape

    # Plot execution times of update and predict steps
    fig, ax = plt.subplots()
    ax.set_yscale("log")

    xs = np.arange(num_steps) + 1

    for i, filter in enumerate(filters):
        ax.plot(
            xs,
            runtimes_update_ms[i],
            color=colors(i),
            label=f"Update {filter.get_name()}",
        )
        ax.plot(
            xs,
            runtimes_prediction_ms[i],
            color=colors(i),
            label=f"Prediction {filter.get_name()}",
            linestyle="dashed",
        )

    ax.legend()
    ax.grid()
    ax.set_ylabel("Execution time [ms]")
    ax.set_xlabel("Step")
    ax.set_title("Execution times")
    fig.savefig("complete_example_runtimes.png", dpi=300)


def plot_estimation_errors(
    filename: str,
    filters: List[Filter],
    system_states: np.ndarray,
    updated_state_means: np.ndarray,
    predicted_state_means: np.ndarray,
    colors: Colormap,
) -> None:
    num_steps = system_states.shape[0]

    # Plot filter errors
    update_errors = np.linalg.norm(
        system_states[None, :, :2, 0] - updated_state_means[:, :, :2, 0], axis=-1
    )
    prediction_errors = np.linalg.norm(
        system_states[None, 1:, :2, 0] - predicted_state_means[:, :-1, :2, 0], axis=-1
    )

    fig, ax = plt.subplots()
    xs = np.arange(num_steps)
    for i, filter in enumerate(filters):
        ax.plot(
            xs,
            update_errors[i],
            color=colors(i),
            label=f"Update error {filter.get_name()}",
        )
        ax.plot(
            xs[1:],
            prediction_errors[i],
            color=colors(i),
            label=f"Prediction error {filter.get_name()}",
            linestyle="dashed",
        )

    ax.legend()
    ax.grid()
    ax.set_ylabel("Error")
    ax.set_xlabel("Step")
    ax.set_title("Prediction and update errors")
    fig.savefig(filename, dpi=300)


def render_video_frames(
    filters: List[Filter],
    system_states: np.ndarray,
    measurements: np.ndarray,
    updated_state_means: np.ndarray,
    updated_state_covs: np.ndarray,
    predicted_state_means: np.ndarray,
    predicted_state_covs: np.ndarray,
    filter_idx: List[int],
    output_directory: Path = Path("complete_example_video_frames"),
) -> None:
    """
    This function does the same plotting like plot_trajectory, but renders a frame for each step.
    With these frames an animation can be put together using for example ffmpeg:

        ffmpeg  -framerate 5 -i EKF/%05d.jpg -loop -1 output.gif

    The parameter filter_idx specifies which of the filters in the given list are rendered.
    """
    num_steps = system_states.shape[0]

    if max(filter_idx) >= len(filters) or min(filter_idx) < 0:
        raise ValueError(
            f'The parameter "filter_idx" does contain invalid entries. '
            f"It must contain the indices of the filters to render, "
            f"in this case they can be between 0 and {len(filters)-1} (inclusive)."
        )

    for i in filter_idx:
        filter = filters[i]

        # Create output directories
        directory = output_directory / filter.get_name()
        output_directory.mkdir(exist_ok=True)
        directory.mkdir(exist_ok=True)

        # Create figure, set labels and title
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(filter.get_name())

        # Set plot range
        margin = 3
        min_x = np.min(system_states[:, 0, 0])
        max_x = np.max(system_states[:, 0, 0])
        min_y = np.min(system_states[:, 1, 0])
        max_y = np.max(system_states[:, 1, 0])
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        # Plot ground truth trajectory
        ax.plot(system_states[:, 0, 0], system_states[:, 1, 0], color="black")

        # Save initial image, only gt trajectory
        fig.savefig(directory / f"{0:05d}.jpg", dpi=300)

        trace = np.empty((2 * num_steps, 2, 1))
        for s, (update_mean, update_cov, pred_mean, pred_cov) in enumerate(
            zip(
                updated_state_means[i, :, :2, :],
                updated_state_covs[i, :, :2, :2],
                predicted_state_means[i, :, :2, :],
                predicted_state_covs[i, :, :2, :2],
            )
        ):
            # Plot measurements
            x = measurements[s, 0, 0] * np.cos(measurements[s, 1, 0])
            y = measurements[s, 0, 0] * np.sin(measurements[s, 1, 0])
            ax.scatter(x, y, color="black", marker="x")
            fig.savefig(directory / f"{3*s+1:05d}.jpg", dpi=300)

            # Plot updated state (mean and covariance)
            ax.add_patch(generate_ellipse(update_mean, update_cov, color="blue"))
            trace[2 * s] = update_mean
            ax.plot(trace[: 2 * s + 1, 0, 0], trace[: 2 * s + 1, 1, 0], color="green")
            fig.savefig(directory / f"{3*s+2:05d}.jpg", dpi=300)

            # Plot predicted state (mean and covariance)
            ax.add_patch(generate_ellipse(pred_mean, pred_cov, color="red"))
            trace[2 * s + 1] = pred_mean
            ax.plot(trace[: 2 * s + 2, 0, 0], trace[: 2 * s + 2, 1, 0], color="green")
            fig.savefig(directory / f"{3*s+3:05d}.jpg", dpi=300)
