"""
Test the attribution methods.
"""

import os
from pprint import pformat, pprint
import sys
import unittest

import numpy as np
import torch

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

from src.attribute.utils import (
    apply_feature_mask_slow,
    apply_feature_mask_fast,
    Masker,
    ChunkFeatureMasker,
    AutoLenChunkFeatureMasker,
    AutoNumChunkFeatureMasker,
    FunctionFeatureMasker,
)


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


class TestMaskers(unittest.TestCase):

    def setUp(self):
        self.input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 42],
                [1, 3, 2, 4, 5, 6, 7, 42, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8, 42, 0],
            ],
            dtype=torch.int64,
        )

        self.boundaries = {
            "sha1": np.array([[1, 3], [4, 6], [6, 8]], dtype=np.uint32),
            "sha2": np.array([[0, 3], [5, 6],], dtype=np.uint32),
            "sha3": np.array([], dtype=np.uint32).reshape(0, 2),
        }

        self.bos_token_id = 1
        self.eos_token_id = 42
        self.pad_token_id = 0
        self.special_token_ids = (self.bos_token_id, self.eos_token_id, self.pad_token_id)

    def test_masker(self):
        m = Masker()
        assert m.special_token_ids == tuple()
        mask = m(self.input_ids)
        correct = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

        m = Masker(self.bos_token_id, self.eos_token_id, self.pad_token_id)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids)
        correct = torch.tensor([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_fix(self):
        m = ChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, chunk_size=2)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids)
        correct = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 0],
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct)

    def test_chunk_feature_masker_len(self):
        stats = {s: np.mean(b[:,1] - b[:,0]) for s, b in self.boundaries.items()}  # {'sha1': 2.0, 'sha2': 2.0, 'sha3': nan}
        m = AutoLenChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, stats=stats)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids, list(self.boundaries.keys()))
        correct = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_num(self):
        stats = {s: len(b) for s, b in self.boundaries.items()}  # {'sha1': 3, 'sha2': 2, 'sha3': 0}
        m = AutoNumChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, stats=stats)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids, list(self.boundaries.keys()))
        correct = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_fun(self):
        m = FunctionFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, boundaries=self.boundaries)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids, list(self.boundaries.keys()))
        correct = torch.tensor([
            [0, 1, 2, 2, 1, 3, 3, 4, 4, 0],
            [0, 2, 2, 2, 1, 1, 3, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"
