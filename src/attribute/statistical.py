"""
Math and statistical functions for attribute analysis.
"""

from collections import namedtuple, Counter
from typing import Optional
import sys
import warnings

import numpy as np
from scipy import stats

from src.attribute.utils import is_proper_rank_matrix


ALTERNATIVE = "greater"


SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])


def kendalltau(R: np.ndarray) -> SignificanceResult:
    res = stats.kendalltau(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


def spearmanr(R: np.ndarray) -> SignificanceResult:
    res = stats.spearmanr(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


def kendallw(R: np.ndarray) -> SignificanceResult:
    # Number of items (n) and number of judges (m)
    n, m = R.shape

    # Correction for ties.
    T = 0
    for j in range(m):
        rank_counts = Counter(R[:, j])
        for count in rank_counts.values():
            if count > 1:
                T += (count**3 - count)

    # Kendall's W.
    sum_of_ranks = np.sum(R, axis=1)
    mean_ranks = np.mean(sum_of_ranks)
    S = np.sum((sum_of_ranks - mean_ranks)**2)
    denom = m**2 * (n**3 - n) - m * T
    w = (12.0 * S) / denom

    # P-value via chi-square distribution.
    chi2_val = m * (n - 1) * w
    p_val = 1.0 - stats.chi2.cdf(chi2_val, df=n - 1)

    return SignificanceResult(w, p_val)


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


def compute_agreement(R: np.ndarray, tolerance: float = 0.0, top_k: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the agreement and p-value using Kendall's Tau or Kendall's W, verifying the input and output.
    """
    # Verify the rank matrix is valid (we don't need to do this, usually, but it may be helpful later).
    # if not is_proper_rank_matrix(R):
    #     file = "/tmp/R.npy"
    #     np.save(file, R)
    #     raise ValueError(f"Rank matrix is not valid. Rank matrix has been saved to {file}")

    # If Top-K is specified, then we need to filter the rank matrix to only include the top-k items for each judge.
    if top_k is not None:
        R = topk_rank_matrix(R.copy(), k=top_k)

    num_judges = R.shape[1]
    if num_judges > 2:
        agreement_function = kendallw
    elif num_judges == 2:
        agreement_function = kendalltau
    else:
        raise ValueError(f"Correlation tests requires at least two judges, but only {num_judges} were provided.")

    # If only a single interpretable feature exists, then the agreement is not well-defined.
    if R.shape[0] == 1:
        return np.nan, np.nan

    # If any judge cannot rank any item higher or lower than any other then the agreement is not well-defined.
    # It is the responsibility of the caller to select annotators that are not unopinionated.
    # In practice, when unopinionated judges are present, `kendallw_with_ties` can return negative values.
    for k in range(num_judges):
        if len(np.unique(R[:,k])) == 1:
            return np.nan, np.nan

    # Compute the correlation test.
    w, p = agreement_function(R)

    # Raise exceptions for invalid values.
    try:
        if np.isinf(w) or np.isinf(p):
            raise ValueError(f"Correlation is InF ({w=} {p=})")
        if np.isnan(w) or np.isnan(p):
            raise ValueError(f"Correlation is NaN ({w=} {p=})")

        p = try_to_clip(p, tolerance, min_=0, max_=1)
        if num_judges == 2:
            w = try_to_clip(w, tolerance, min_=-1, max_=1)
        else:
            w = try_to_clip(w, tolerance, min_=0, max_=1)

    except ValueError as err:
        file = "/tmp/R.npy"
        np.save(file, R)
        raise ValueError(f"Correlation test failed. Rank matrix has been saved to {file}") from err

    return w, p


def descriptive_sparsity(
    relevances: np.ndarray,
    *,
    r_grid: np.ndarray | tuple = tuple(np.linspace(0.0, 1.0, 201).tolist()),
    num_bins: int = 500,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the descriptive sparsity curve (MAZ) for a 1-D relevance vector.

    Parameters
    ----------
    relevances : np.ndarray
        Raw relevance scores for a single explanation (shape: N,).
    r_grid : np.ndarray, optional
        Points in [0,1] at which to evaluate MAZ.  Defaults to 201
        uniformly-spaced points.
    num_bins : int, default=500
        Number of histogram bins used to approximate the pdf h(x).
    eps : float, default=1e-12
        Small constant to avoid divide-by-zero.

    Returns
    -------
    r_grid : np.ndarray
        Grid of r values in [0,1].
    maz : np.ndarray
        MAZ(r) evaluated at each r in `r_grid`.
    auc : float
        Area under the MAZ curve.
    """

    r_grid = np.asarray(r_grid)
    if not np.all((r_grid >= 0.0) & (r_grid <= 1.0)):
        raise ValueError("r_grid must be in [0, 1]")

    # 1. Rescale to [-1, 1]
    max_abs = np.maximum(np.max(np.abs(relevances)), eps)
    rel_scaled = relevances / max_abs

    # 2. Normalized histogram → pdf h(x)
    hist, bin_edges = np.histogram(
        rel_scaled, bins=num_bins, range=(-1.0, 1.0), density=True
    )
    bin_width = bin_edges[1] - bin_edges[0]
    cdf = np.cumsum(hist) * bin_width  # F(x) = ∫_{‑1}^x h(t) dt

    # helper to map a value x∈[‑1,1] to CDF(x)
    def _cdf_at(x: float) -> float:
        idx = np.clip(((x + 1.0) / 2.0) * num_bins, 0, num_bins - 1e-9).astype(int)
        return cdf[idx]

    # 3. Evaluate MAZ(r) = F(r) – F(‑r)
    maz = np.array([_cdf_at(r) - _cdf_at(-r) for r in r_grid])

    auc = np.trapz(maz, r_grid)

    return r_grid, maz, auc
