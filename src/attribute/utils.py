"""
Tools for performing input attribution with captum.
"""

from functools import wraps
import warnings

import numpy as np
from scipy import stats


def is_proper_rank_matrix(R: np.ndarray, *, method: str = "average", tol: float = 1e-12) -> bool:
    for col in R.T:
        if not np.allclose(col, stats.rankdata(col, method=method), atol=tol, rtol=0):
            return False
    return True


def ignore_warnings_decorator(*filter_args, **filter_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*filter_args, **filter_kwargs)
                return func(*args, **kwargs)

        return wrapper

    return decorator
