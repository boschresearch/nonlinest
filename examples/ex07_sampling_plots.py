"""
This example creates plots of different Gaussian sampling methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from nonlinear_estimation_toolbox.gaussian_sampling import (
    GaussianSamplingCKF,
    GaussianSamplingGHQ,
    GaussianSamplingRUKF,
    GaussianSamplingUKF,
)

np.random.seed(2)

samplings = [
    GaussianSamplingUKF(),
    GaussianSamplingCKF(),
    GaussianSamplingGHQ(),
    GaussianSamplingRUKF(),
]
names = ["UKF", "CKF", "GHKF", "RUKF"]

for s, name in zip(samplings, names):
    dim = 2
    samples, weight, _ = s.get_std_normal_samples(dim)

    print(f"plotting {name}")
    plt.figure()
    plt.gca().add_patch(
        plt.Circle(
            [0, 0], radius=2, facecolor="none", edgecolor="black", label="$2 |si"
        )
    )
    plt.plot(samples[0, :], samples[1, :], "o", label="samples")
    plt.legend()

    plt.title(name)
    plt.axis("equal")
    plt.ylim([-2.2, 2.2])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.savefig(f"samples-{name}.png")
