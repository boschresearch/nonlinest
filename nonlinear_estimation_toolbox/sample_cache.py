import os
import warnings
from tempfile import TemporaryDirectory

import numpy as np

from nonlinear_estimation_toolbox import checks


class SampleCache:
    def __init__(self, sample_cache_path: str):
        self.data = {}
        base_dir = os.path.join(os.path.expanduser("~"), ".cache", "nonlinest")
        self.full_sample_cache_path = os.path.join(base_dir, sample_cache_path)
        os.makedirs(self.full_sample_cache_path, exist_ok=True)

    def get_samples(
        self, dimension: int, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_request(dimension, num_samples)

        if dimension not in self.data or num_samples not in self.data[dimension]:
            if not self.load_samples_from_file(dimension, num_samples):
                print(
                    "The requested samples (%d %dD) are not in the sample cache yet and will be computed"
                    % (num_samples, dimension)
                )

                self.generate_samples(dimension, num_samples)
                self.load_samples_from_file(dimension, num_samples)

        return self.data[dimension][num_samples]

    def generate_samples(
        self, dimension: int, num_samples: int, force_overwrite: bool = False
    ):
        # todo not sure if this method is used anywhere
        self._check_request(dimension, num_samples)

        filename = self.get_sample_filename(dimension, num_samples)
        if os.path.isfile(filename) and not force_overwrite:
            print(
                "The requested samples (%d %dD) already exist in the sample cache. Set force_overwrite=True to recompute them."
                % (num_samples, dimension)
            )
            return

        self.generate_samples_and_save_to_file(dimension, num_samples)

    # todo abstract methods using abc?
    def check_dim_and_num_samples_combination(self, dimension: int, num_samples: int):
        raise NotImplementedError("implement in child class")

    def compute_samples(self, dimension: int, num_samples: int):
        raise NotImplementedError("implement in child class")

    def _check_request(self, dimension: int, num_samples: int):
        if not checks.is_pos_scalar(dimension):
            raise ValueError("dimension must be a positive scalar")

        if not checks.is_pos_scalar(num_samples):
            raise ValueError("num_samples must be a positive scalar")

        self.check_dim_and_num_samples_combination(dimension, num_samples)

    def load_samples_from_file(self, dimension: int, num_samples: int) -> bool:
        filename = self.get_sample_filename(dimension, num_samples)

        try:
            samples, weights = self._load_samples_and_weights(filename)
            if dimension not in self.data:
                self.data[dimension] = {}
            self.data[dimension][num_samples] = (samples, weights)
        except Exception:
            return False

        if samples.shape[0] != dimension or samples.shape[1] != num_samples:
            raise RuntimeError("invalid samples in file %s" % filename)

        if weights.shape[0] != 1 or weights.shape[1] != num_samples:
            raise RuntimeError("invalid weights in file %s" % filename)

        return True

    def generate_samples_and_save_to_file(self, dimension: int, num_samples: int):
        samples, weights = self.compute_samples(dimension, num_samples)

        filename = self.get_sample_filename(dimension, num_samples)

        self._save_samples_and_weights(filename, samples, weights)

    def get_sample_filename(self, dimension: int, num_samples: int) -> str:
        return os.path.join(
            self.full_sample_cache_path,
            f"{dimension}D-{num_samples}S.samples.npz",
        )

    @staticmethod
    def _save_samples_and_weights(filename, samples, weights):
        try:
            np.savez(file=filename, samples=samples, weights=weights)
        except RuntimeError:
            raise RuntimeError(f"saving samples and weights to {filename} failed")

    @staticmethod
    def _load_samples_and_weights(filename):
        try:
            npzfile = np.load(file=filename)
            samples = npzfile["samples"]
            weights = npzfile["weights"]
        except RuntimeError:
            raise RuntimeError(f"loading samples and weights from {filename} failed")

        return samples, weights


class SampleCacheCKF(SampleCache):
    # sample cache for fifth order CKF

    def __init__(self):
        # todo package might not be installed in a writable location?
        sample_cache_path = "SampleCacheCKF"

        super(SampleCacheCKF, self).__init__(sample_cache_path)

    def check_dim_and_num_samples_combination(self, dimension: int, num_samples: int):
        expected_num_samples = 2 * dimension * dimension + 1

        if num_samples != expected_num_samples:
            raise ValueError(
                "invalid number of samples, got %i but expected %i"
                % (num_samples, expected_num_samples)
            )

    def compute_samples(
        self, dimension: int, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        axes_samples, axes_weights = self._compute_axes_samples(dimension)
        off_axes_samples, off_axes_weights = self._compute_off_axes_samples(dimension)

        samples = np.hstack([axes_samples, off_axes_samples])
        weights = np.hstack([axes_weights, off_axes_weights])

        return samples, weights

    @staticmethod
    def _compute_axes_samples(dimension: int) -> tuple[np.ndarray, np.ndarray]:
        samples = np.sqrt(dimension + 2) * np.hstack(
            [np.zeros([dimension, 1]), -np.eye(dimension), np.eye(dimension)]
        )
        weights = (
            np.ones([1, 2 * dimension + 1])
            * (4 - dimension)
            / (2 * (dimension + 2) ** 2)
        )
        weights[0, 0] = 2 / (dimension + 2)
        return samples, weights

    @staticmethod
    def _compute_off_axes_samples(dim: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(dim * (dim - 1) / 2)
        if n > 0:
            s1 = np.zeros([dim, n])
            s2 = np.zeros([dim, n])

            e = np.eye(dim, dim)
            i = 0

            for dim_idx in range(dim):
                for k in range(dim_idx):
                    s1[:, i] = e[:, k] + e[:, dim_idx]
                    s2[:, i] = e[:, k] - e[:, dim_idx]
                    i += 1

            samples = np.sqrt((dim + 2) / 2) * np.hstack([s1, s2, -s1, -s2])
            weights = np.ones([1, 4 * n]) / (dim + 2) ** 2
            return samples, weights
        else:
            return np.zeros([dim, 0]), np.zeros([1, 0])


class SampleCacheGHQ(SampleCache):
    #   Kazufumi Ito and Kaiqi Xiong,
    #   Gaussian Filters for Nonlinear Filtering Problems,
    #   IEEE Transactions on Automatic Control, vol. 45, no. 5, pp. 910-927, May 2000.
    def __init__(self):
        # todo package might not be installed in a writable location?
        sample_cache_path = "SampleCacheGHQ"

        self.one_dim_samples = [-1, 1]
        self.one_dim_weights = [0.5, 0.5]

        super(SampleCacheGHQ, self).__init__(sample_cache_path)

    def set_num_quadrature_points(self, num_points: int):
        if num_points not in [2, 3, 4]:
            raise ValueError("Invalid number of quadrature points (must be 2,3, or 4).")

        if num_points != self.get_num_quadrature_points():
            if num_points == 2:
                a = 1
                self.one_dim_samples = [a, -a]
                a = 0.5
                self.one_dim_weights = [a, a]
            elif num_points == 3:
                a = 0
                b = np.sqrt(3)
                self.one_dim_samples = [a, b, -b]

                a = 2 / 3
                b = 1 / 6
                self.one_dim_weights = [a, b, b]
            elif num_points == 4:
                a = np.sqrt(3 - np.sqrt(6))
                b = np.sqrt(3 + np.sqrt(6))
                self.one_dim_samples = [a, -a, b, -b]

                a = 1 / (4 * (3 - np.sqrt(6)))
                b = 1 / (4 * (3 + np.sqrt(6)))
                self.one_dim_weights = [a, a, b, b]

    def get_num_quadrature_points(self) -> int:
        return len(self.one_dim_samples)

    def check_dim_and_num_samples_combination(self, dimension: int, num_samples: int):
        num_points = self.get_num_quadrature_points()
        expected_num_samples = num_points**dimension

        if num_samples != expected_num_samples:
            raise ValueError(
                "invalid number of samples, got %i but expected %i"
                % (num_samples, expected_num_samples)
            )

    def compute_samples(
        self, dimension: int, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = self._cartesian_product(self.one_dim_samples, dimension)
        weights = self._cartesian_product(self.one_dim_weights, dimension)

        weights = np.prod(weights, 0)
        weights = weights.reshape([1, num_samples])

        return samples, weights

    @staticmethod
    def _cartesian_product(data: list, dim: int) -> np.ndarray:
        # compute Cartesian product data x data x ... x data

        # create n-d grid based on data
        g = np.meshgrid(*[data for j in range(0, dim)])

        cart_prod = np.zeros([dim, len(data) ** dim])

        for i in range(0, dim):
            cart_prod[i, :] = g[i].reshape([1, -1])

        return cart_prod
