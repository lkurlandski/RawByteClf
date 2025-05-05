"""
Tools for performing input attribution with captum.
"""

from functools import wraps
from typing import Optional
import warnings

import numpy as np
from scipy import stats


def is_proper_rank_matrix(R: np.ndarray, *, method: str = "average", tol: float = 1e-12) -> bool:
    for col in R.T:
        if not np.allclose(col, stats.rankdata(col, method=method), atol=tol, rtol=0):
            return False
    return True


def topk_rank_matrix(R: np.ndarray, k: Optional[np.ndarray] = None, lower_is_higher: bool = False) -> np.ndarray:
    """
    Extracts the highest ranked items by each judge from the rank matrix.

    Args:
        R (np.ndarray): Rank matrix, as if produced by scipy.stats.rankdata, of shape (n, m).
        k (np.ndarray, optional): Number of top items to retain for each judge, of shape (m,).
            If None, all items are retained. Note that if the k-th item is not unique,
            all items with the same rank as the k-th item are retained, so the judge may
            return more than just k items.
        lower_is_higher (bool, optional): If True, smaller ranks are indicate a more malicious object.
            This is just the way I encoded the rank matrix (maybe it would have been better the other way).

    Returns:
        np.ndarray: Submatrix of R containing only the rows (items) that are in the
                    top-k for at least one judge, shape (n_, m) with n_ ≤ n.
    """
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError(f"R must be 2-dimensional, got shape {R.shape}")
    n, m = R.shape

    if k is None:
        return R.copy()
    if isinstance(k, int):
        k = np.full(m, k, dtype=int)
    else:
        k = np.asarray(k, dtype=int)
    if k.shape != (m,):
        raise ValueError(f"k must be a 1D array of length {m}, got shape {k.shape}")
    k = np.clip(k, 1, n)

    # For each judge j, mark all rows whose which should be kept.
    mask = np.zeros((n, m), dtype=bool)
    for j in range(m):
        if lower_is_higher:  # smaller R is better → find the kᵗʰ‐smallest rank
            thresh = np.partition(R[:, j], k[j] - 1)[k[j] - 1]
            mask[:, j] = (R[:, j] <= thresh)
        else:                # larger R is better → find the kᵗʰ‐largest value
            rev = np.partition(-R[:, j], k[j] - 1)[k[j] - 1]
            thresh = -rev
            mask[:, j] = (R[:, j] >= thresh)

        threshold = np.partition(R[:, j], k[j] - 1)[k[j] - 1]
        mask[:, j] = R[:, j] <= threshold

    # Keep any item that is selected by at least one judge
    idx = mask.any(axis=1)
    R = R[idx, :]

    # Re-rank the data, as otherwise, it won't be a proper rank matrix
    if lower_is_higher:
        R = stats.rankdata(R, axis=0)
    else:
        R = stats.rankdata(-R, axis=0)

    return R


def np_argtopk(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    if k < 1:
        raise ValueError("k must be a positive integer")

    flat = arr.ravel()
    n    = flat.size
    if k >= n:
        return np.argwhere(np.ones_like(arr, dtype=bool))

    kth_val = np.partition(flat, n - k)[n - k]
    mask = arr >= kth_val
    return np.argwhere(mask)


def np_topk(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    idx = np_argtopk(arr, k)
    return arr[idx]


def try_to_clip(x: float, tolerance: float, min_: float = float("inf"), max_: float = -float("inf")) -> float:
    """
    Tries to round/clip the value `x` to be within the range [min_, max_] by adding/subtracting `tolerance`.
    """
    if x < min_:
        x = min(x + tolerance, min_)
        if x < min_:
            raise ValueError(f"Value {x} is less than minimum {min_} at tolerance {tolerance}")

    if x > max_:
        x = max(x - tolerance, max_)
        if x > max_:
            raise ValueError(f"Value {x} is greater than maximum {max_} at tolerance {tolerance}")

    return x


def ignore_warnings_decorator(*filter_args, **filter_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*filter_args, **filter_kwargs)
                return func(*args, **kwargs)

        return wrapper

    return decorator
