"""
Math and statistical functions for attribute analysis.
"""

from collections import namedtuple
from typing import Optional
import warnings

import numpy as np
from scipy import stats


ALTERNATIVE = "greater"


SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])


def kendalltau(R: np.ndarray) -> SignificanceResult:
    assert R.shape[1] == 2
    res = stats.kendalltau(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


def kendallw_without_ties(R: np.ndarray) -> SignificanceResult:
    """
    Implementation of Kendall's Coefficient for Concordance (Kendall's W).

    See Wikipedia (https://en.wikipedia.org/wiki/Kendall%27s_W) for details.
    """

    n = R.shape[0]  # number of samples
    m = R.shape[1]  # number of judges

    # Edge cases match the behavior of scipy.stats.kendalltau
    if n < 2:
        return SignificanceResult(np.nan, np.nan)
    if m < 2:
        raise ValueError("Kendall's W requires at least two judges.")
    if m == 2:
        warnings.warn("Kendall's W is less appropriate for only two judges. Use `kendalltau` instead.")

    # Compute Kendall's W
    R_sum  = np.sum(R, axis=1)              # (n,)
    R_mean = np.mean(R_sum)                 # (,)
    S = np.sum(np.square(R_sum - R_mean))   # (,)
    W = 12 * S / (m ** 2 * (n ** 3 - n))

    # Compute the significance
    s = m * (n - 1) * W         # Chi-squared statistic
    f = n - 1                   # Degrees of freedom
    p = stats.chi2.sf(s, f)

    return SignificanceResult(W, p)


def kendallw_with_ties(R: np.ndarray) -> SignificanceResult:
    """
    Implementation of Kendall's Coefficient for Concordance (Kendall's W) accounting for ties.

    See Wikipedia (https://en.wikipedia.org/wiki/Kendall%27s_W) for details.
    """

    n = R.shape[0]  # number of samples
    m = R.shape[1]  # number of judges

    # Edge cases match the behavior of scipy.stats.kendalltau
    if n < 2:
        return SignificanceResult(np.nan, np.nan)
    if m < 2:
        raise ValueError("Kendall's W requires at least two judges.")
    if m == 2:
        warnings.warn("Kendall's W is less appropriate for only two judges. Use `kendalltau` instead.")

    # Correction for ties
    T = np.zeros((m,))
    for j in range(m):
        _, counts = np.unique(R[:,j], return_counts=True)
        counts = counts[counts > 1]
        T[j] = np.sum(np.power(counts, 3) - counts)

    # Compute Kendall's W
    R_sum     = np.sum(R, axis=1)         # (n,)
    R_sqr_sum = np.sum(np.square(R_sum))  # (,)
    W = (
        ( (12 * R_sqr_sum) - (3 * m ** 2 * n * (n + 1) ** 2) )
        /
        ( m ** 2 * n * (n ** 2 - 1) - (m * np.sum(T)))
    )

    # Compute the significance
    s = m * (n - 1) * W         # Chi-squared statistic
    f = n - 1                   # Degrees of freedom
    p = stats.chi2.sf(s, f)

    return SignificanceResult(W, p)


def compute_agreement(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the agreement and p-value using Kendall's Tau or Kendall's W, verifying the input and output.
    """
    num_judges = R.shape[1]
    if num_judges > 2:
        agreement_function = kendallw_with_ties
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
        if p < 0 or p > 1:
            raise ValueError(f"Correlation p-value is outside [0, 1] ({w=} {p=})")
        if w > 1:
            raise ValueError(f"Correlation statistic test is greater than 1 ({w=} {p=})")
        if num_judges == 2 and w < -1:
            raise ValueError(f"Correlation statistic is less than -1 ({w=} {p=})")
        if num_judges > 2 and w < 0:
            raise ValueError(f"Correlation statistic is less than 0 ({w=} {p=})")
    except ValueError as err:
        file = "/tmp/R.npy"
        np.save(file, R)
        raise ValueError(f"Correlation test failed. Rank matrix has been saved to {file}") from err

    return w, p
