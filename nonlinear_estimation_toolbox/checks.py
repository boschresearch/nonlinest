from typing import Union

import numpy as np


def is_scalar(s: Union[int, float]) -> bool:
    return np.isscalar(s)


def is_pos_scalar(s: Union[int, float]) -> bool:
    if not np.isscalar(s):
        return False
    if s > 0.0:
        return True
    return False


def is_single_vec(v: np.ndarray, dim: Union[int, None] = None) -> bool:
    # todo still need to decide whether [] counts as a vector
    if dim is None:
        return len(v.shape) == 2 and v.shape[1] == 1
    else:
        return len(v.shape) == 2 and v.shape[1] == 1 and v.shape[0] == dim


def is_single_row_vec(v: np.ndarray, dim: Union[int, None] = None) -> bool:
    if dim is None:
        return len(v.shape) == 2 and v.shape[0] == 1
    else:
        return len(v.shape) == 2 and v.shape[0] == 1 and v.shape[1] == dim


def is_array_of_vecs(
    v: np.ndarray, dim: Union[int, None] = None, num: Union[int, None] = None
) -> bool:
    is_vec_valid = len(v.shape) == 2 and v.shape[1] >= 1
    if dim is not None:
        is_vec_valid &= v.shape[0] == dim
    if num is not None:
        is_vec_valid &= v.shape[1] == num
    return is_vec_valid


def is_mat(
    m: np.ndarray, rows: Union[int, None] = None, cols: Union[int, None] = None
) -> bool:
    if rows is None and cols is None:
        return len(m.shape) == 2
    elif rows is None and cols is not None:
        return is_fixed_col_mat(m, cols)
    elif cols is None and rows is not None:
        return is_fixed_row_mat(m, rows)
    else:
        return len(m.shape) == 2 and m.shape[0] == rows and m.shape[1] == cols


def is_fixed_row_mat(m: np.ndarray, num_rows: int) -> bool:
    if not len(m.shape) == 2:
        return False
    return m.shape[0] == num_rows


def is_fixed_col_mat(m: np.ndarray, n_cols: int) -> bool:
    if not len(m.shape) == 2:
        return False
    return m.shape[1] == n_cols


def is_square_mat(m: np.ndarray) -> bool:
    if not is_mat(m):
        return False
    return m.shape[0] == m.shape[1]


def compute_cholesky_if_valid_covariance(
    m: np.ndarray, location: str = ""
) -> np.ndarray:
    """
    Checks if given matrix is a valid covariance matrix and computes its square root.
    :param m: matrix to check
    :param location: Optional string for error message, indicating where matrix is not valid
    :return: Cholesky decomposition if matrix is valid covariance
    """
    if not is_square_mat(m):
        raise ValueError(
            f"{location} Invalid covariance matrix: Matrix must be square."
        )

    try:
        covariance_sqrt = np.linalg.cholesky(m)
        return covariance_sqrt
    except np.linalg.LinAlgError:
        raise ValueError(
            f"{location} Invalid covariance matrix: Matrix must be positive definite."
        )


def compute_cholesky_if_valid_3d_covariance(
    m: np.ndarray, dim: Union[int, None] = None, num_covs: Union[int, None] = None
) -> np.ndarray:
    """Compute the Cholesky decomposition of a list of covariance matrices, if the matrices are valid.

    The covariance matrices are in shape (num_variables, num_variables, num_matrices).

    :param m: List of covariance matrices to compute Cholesky decomposition from.
    :param dim: Require that the matrices have dim number of variables.
    :param num_covs: Require that there are num_covs number of covariance matrices.
    :return:
    """
    if len(m.shape) != 3:
        raise ValueError(
            f"3d covariance matrix must have three dimensions (of shape (num_dims, num_dims, num_covs)), got {len(m.shape)} dimensions."
        )

    dim_a, dim_b, N = m.shape

    if dim_a != dim_b:
        raise ValueError(
            "3d covariance matrices must be of form num_dims × num_dims × num_covs. "
            f"The first two dimensions of the given {m.shape} matrix are not equal: {m.shape[0]} != {m.shape[1]}."
        )
    elif dim is not None and dim_a != dim:
        raise ValueError(
            f"The present covariance matrices do not have the requested number of variables: {dim_a} does not equal the requested number {dim}."
        )
    elif num_covs is not None and N != num_covs:
        raise ValueError(
            f"The present 3d covariance matrix does not contain the requested number of covariance matrices {num_covs}. "
            f"Got {N} number of covariance matrices instead."
        )

    cov_sqrts = np.linalg.cholesky(m.transpose(2, 0, 1)).transpose(1, 2, 0)
    return cov_sqrts
