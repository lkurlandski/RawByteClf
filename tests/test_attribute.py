"""
Test the attribution methods.
"""

import os
import sys
import unittest

import torch

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

from src.attribute.utils import apply_feature_mask_slow, apply_feature_mask_fast


torch.random.manual_seed(0)

apply_feature_mask_1 = apply_feature_mask_slow
apply_feature_mask_2 = apply_feature_mask_fast


class TestApplyFeatureMask(unittest.TestCase):

    def test_1(self):
        X = torch.rand(4, 8, dtype=torch.float32)

        M = torch.tensor(
            [
            #    0  1  2  3  4  5  6  7
                [0, 0, 0, 1, 1, 1, 2, 2],
                [0, 0, 1, 1, 1, 1, 1, 2],
                [0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 2, 2],
            ],
            dtype=torch.int64,
        )

        Y = apply_feature_mask_1(X, M)

        assert tuple(Y.shape) == tuple(X.shape)
        assert Y.dtype == X.dtype

        assert torch.all(Y[0][0:3] == X[0][0:3].sum()).item(), f"{Y[0][0:3]} != {X[0][0:3].sum()}"
        assert torch.all(Y[0][3:6] == X[0][3:6].sum()).item(), f"{Y[0][3:6]} != {X[0][3:6]}"
        assert torch.all(Y[0][6:8] == X[0][6:8].sum()).item(), f"{Y[0][6:8]} != {X[0][6:8].sum()}"

        assert torch.all(Y[1][0:2] == X[1][0:2].sum()).item(), f"{Y[1][0:2]} != {X[1][0:2].sum()}"
        assert torch.all(Y[1][2:7] == X[1][2:7].sum()).item(), f"{Y[1][2:7]} != {X[1][2:7].sum()}"
        assert torch.all(Y[1][7:8] == X[1][7:8].sum()).item(), f"{Y[1][7:8]} != {X[1][7:8].sum()}"

        assert torch.all(Y[2][0:1] == X[2][0:1].sum()).item(), f"{Y[2][0:1]} != {X[2][0:1].sum()}"
        assert torch.all(Y[2][1:8] == X[2][1:8].sum()).item(), f"{Y[2][1:8]} != {X[2][1:8].sum()}"
        assert torch.all(Y[2][8:8] == X[2][8:8].sum()).item(), f"{Y[2][8:8]} != {X[2][8:8].sum()}"

        assert torch.all(Y[3][0:4] == X[3][0:4].sum()).item(), f"{Y[3][0:4]} != {X[3][0:4].sum()}"
        assert torch.all(Y[3][4:6] == X[3][4:6].sum()).item(), f"{Y[3][4:6]} != {X[3][4:6].sum()}"
        assert torch.all(Y[3][6:8] == X[3][6:8].sum()).item(), f"{Y[3][6:8]} != {X[3][6:8].sum()}"

        assert torch.equal(Y, apply_feature_mask_2(X, M))
