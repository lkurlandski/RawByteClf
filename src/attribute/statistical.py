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
    res = stats.kendalltau(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


def spearmanr(R: np.ndarray) -> SignificanceResult:
    res = stats.spearmanr(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


def kendallw(R: np.ndarray) -> SignificanceResult:
    # Number of items (n) and number of judges (m)
    n, m = R.shape

    # Sum of ranks across judges for each item
    sum_of_ranks = np.sum(R, axis=1)

    # Compute the "variance" term S
    mean_ranks = np.mean(sum_of_ranks)
    S = np.sum((sum_of_ranks - mean_ranks)**2)

    # Correction for ties (how many items share the same rank under judge j)
    T = 0
    for j in range(m):
        rank_counts = Counter(R[:, j])
        for count in rank_counts.values():
            if count > 1:
                T += (count**3 - count)

    # Kendall's W with tie correction
    denom = m**2 * (n**3 - n) - m * T
    w = (12.0 * S) / denom

    # P-value via chi-square distribution with (n-1) degrees of freedom
    chi2_val = m * (n - 1) * w
    p_val = 1.0 - stats.chi2.cdf(chi2_val, df=n - 1)

    return SignificanceResult(w, p_val)


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


class DescriptiveSparsity:

    def __init__(self, n_bins: int = 100, n_points: int = 100):
        self.n_bins   = n_bins
        self.n_points = n_points

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        self.n_bins = min(self.n_bins, len(scores))

        scores = self.squeeze(scores)

        hist, bin_edges = np.histogram(scores, bins=self.n_bins, density=True)
        cdf_vals = self.build_cdf(hist, bin_edges)

        maz = np.empty(self.n_points)
        for i, r in enumerate(np.linspace(0, 1, self.n_points)):
            m = self.mass_around_zero(r, hist, bin_edges, cdf_vals)
            maz[i] = m

        return maz

    @staticmethod
    def build_cdf(hist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        cdf_vals = np.zeros(len(bin_edges))
        for i in range(1, len(bin_edges)):
            bin_width = bin_edges[i] - bin_edges[i-1]
            cdf_vals[i] = cdf_vals[i-1] + hist[i-1] * bin_width
        return cdf_vals

    @staticmethod
    def squeeze(arr: np.ndarray) -> np.ndarray:
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        scaled = 2.0 * (arr - min_val) / (max_val - min_val) - 1.0
        return scaled

    @staticmethod
    def mass_around_zero(r: float, hist: np.ndarray, bin_edges: np.ndarray, cdf_vals: np.ndarray) -> float:

        def cdf_from_hist(x: float) -> float:
            x = max(x, bin_edges[0])
            x = min(x, bin_edges[-1])
            i = np.searchsorted(bin_edges, x, side="right") - 1
            i = max(i, 0)
            i = min(i, len(hist) - 1)
            dx = x - bin_edges[i]
            partial_area = hist[i] * dx
            return cdf_vals[i] + partial_area

        cdf_pos_r = cdf_from_hist(r)
        cdf_neg_r = cdf_from_hist(-r)
        return cdf_pos_r - cdf_neg_r
