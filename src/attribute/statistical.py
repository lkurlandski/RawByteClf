"""
Math and statistical functions for attribute analysis.
"""

from collections import namedtuple, Counter
from enum import Enum
from typing import Optional, Protocol
import sys
import warnings

import numpy as np
from scipy import stats

from src.attribute.utils import try_to_clip, topk_rank_matrix


ALTERNATIVE = "greater"


SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])


class AgreementMethod(Enum):
    KENDALL  = "kendall"
    JACCARD  = "jaccard"
    DICE     = "dice"
    SPEARMAN = "spearman"


class AgreementFunction(Protocol):

    def __call__(self, R: np.ndarray, top_k: Optional[int] = None, tolerance: float = 0.0) -> SignificanceResult:
        pass


def get_agreement_function(method: AgreementMethod) -> AgreementFunction:
    if method == AgreementMethod.KENDALL:
        return compute_agreement_kendall
    if method == AgreementMethod.JACCARD:
        return compute_agreement_jaccard
    if method == AgreementMethod.DICE:
        return compute_agreement_dice
    raise ValueError(f"Unknown agreement method: {method}")


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


def compute_agreement_kendall(R: np.ndarray, top_k: Optional[int] = None, tolerance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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
