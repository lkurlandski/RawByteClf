"""
Reweighting schemes for imbalanced class distributions.
"""

from collections import Counter
import math


def inverse_class_frequency(dist: Counter) -> dict[str, float]:
    weights: dict[str, float] = {}
    for c in dist:
        weights[c] = 1 / dist[c]
    return weights


def sample_reweighting(dist: Counter, beta: float) -> dict[str, float]:
    if beta < 0 or beta >= 1:
        raise ValueError("Beta must be in the range [0, 1)")
    weights: dict[str, float] = {}
    for c in dist:
        weights[c] = 1 / ((1 - math.pow(beta, dist[c])) / (1 - beta))
    return weights
