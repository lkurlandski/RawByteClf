"""
Math and statistical functions for attribute analysis.
"""

from collections import namedtuple, Counter
from typing import Optional
import sys
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


def kendallw_with_ties_me(R: np.ndarray) -> SignificanceResult:
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


def kendallw_with_ties_gpt(R: np.ndarray) -> SignificanceResult:
    """
    Implementation of Kendall's Coefficient of Concordance (Kendall's W) accounting for ties.

    Produced by ChatGPT-4.
    """
    # Number of items (n) and number of judges (m)
    n, m = R.shape

    # 1) Sum of ranks across judges for each item
    sum_of_ranks = np.sum(R, axis=1)

    # 2) Compute the "variance" term S = sum( (R_i - mean_R)^2 )
    mean_ranks = np.mean(sum_of_ranks)
    S = np.sum((sum_of_ranks - mean_ranks)**2)

    # 3) Correction for ties: T = sum_j sum_g(t_{jg}^3 - t_{jg})
    #    where t_{jg} = size of a tie group g under judge j.
    T = 0
    for j in range(m):
        # Count how many items share the same rank under judge j
        rank_counts = Counter(R[:, j])
        for count in rank_counts.values():
            if count > 1:
                T += (count**3 - count)

    # 4) Kendall's W with tie correction
    #    W = [12 * S] / [m^2 * (n^3 - n) - m * T]
    denom = m**2 * (n**3 - n) - m * T
    w = (12.0 * S) / denom

    # 5) p-value via chi-square distribution with (n-1) degrees of freedom
    #    chi^2 = m * (n - 1) * W
    chi2_val = m * (n - 1) * w
    p_val = 1.0 - stats.chi2.cdf(chi2_val, df=n - 1)

    return SignificanceResult(w, p_val)


# Basically, it looks like my implementation of Kendall's W may not work when the data is not centered properly.
kendallw_with_ties = kendallw_with_ties_gpt


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


def compute_agreement(R: np.ndarray, tolerance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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
