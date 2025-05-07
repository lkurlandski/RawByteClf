"""
Test the attribution methods.
"""

from collections import defaultdict
import gc
import io
import math
import os
from pathlib import Path
from pprint import pformat, pprint
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from scipy.stats import rankdata
import torch
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

from src.enums import ExplanationMethod
from src.utils import rglob, print_context
from src.attribute.masking import (
    apply_feature_mask_slow,
    apply_feature_mask_fast,
    Masker,
    ChunkFeatureMasker,
    AutoLenChunkFeatureMasker,
    AutoNumChunkFeatureMasker,
    AutoNumLenChunkFeatureMasker,
    FunctionFeatureMasker,
    get_masker,
    infer_chunk_sizes,
    assert_feature_mask_indices_are_consecutive,
)
from src.attribute.utils import ignore_warnings_decorator, is_proper_rank_matrix
from src.attribute.segtensor import SegmentedTensor
from src.attribute.statistical import (
    kendalltau,
    kendallw,
    spearmanr,
    topk_rank_matrix,
    descriptive_sparsity,
    compute_agreement_dice,
    compute_agreement_jaccard,
)
from src.data.function_boundaries import bounds_contain_totally_overlapping_functions


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

    def test_chunk_feature_masker_fix_1(self):
        m = ChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, chunk_size=2)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids)
        correct = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 0],
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_fix_2(self):
        with self.assertRaises(ValueError):
            ChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, chunk_size=0)

    def test_chunk_feature_masker_fix_3(self):
        m = ChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, chunk_size=1)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids)
        correct = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
            [0, 1, 2, 3, 4, 5, 6, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_len(self):
        m = AutoLenChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, boundaries=self.boundaries)
        assert m.special_token_ids == self.special_token_ids
        mask = m(self.input_ids, list(self.boundaries.keys()))
        correct = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.int64)
        assert torch.equal(mask, correct), f"Got:\n{pformat(mask.tolist())}\nExpected:\n{pformat(correct.tolist())}"

    def test_chunk_feature_masker_num(self):
        m = AutoNumChunkFeatureMasker(self.bos_token_id, self.eos_token_id, self.pad_token_id, boundaries=self.boundaries)
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


class TestMaskersWithRealData(unittest.TestCase):

    # sbatch --account=admalware --job-name=test_attribute --partition=debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G --time=02:00:00 --output=./logs/test_attribute.log --error=./logs/test_attribute.log --wrap="python -u -m unittest tests.test_attribute.TestMaskersWithRealData"

    def setUp(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.special_token_ids = (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        self.chunk_size = 4096
        self.max_length = int(os.environ.get("MASKERS_MAX_LENGTH", "1048576"))
        self.total = 16384
        self.apply_padding = True

    def should_include(self, s: str) -> bool:  # pylint: disable=unused-argument
        # return s in (
        #     "0000ef6904a7b01d154585d6c06f9f2a7a5ab2b1900fedfcf8b1ccf48e916046",
        # )
        return True

    @ignore_warnings_decorator("ignore", category=UserWarning, message=r"^The given buffer is not writable*")
    def get_data(self):

        files = []
        for f in rglob("./data/", "*.zip"):
            if "nop" in f:
                files.append(f)

        data = []
        shas = []
        pbar = tqdm(files)
        for f in pbar:
            pbar.set_description(f"Reading {f}. Progress {len(shas)} / {int(self.total * 1.25)}")
            with zipfile.ZipFile(f, "r") as zp:
                for name in zp.namelist():
                    s = name.split(".")[0]
                    if not self.should_include(s):
                        continue
                    b = zp.read(name)[0:self.max_length - 2]
                    data.append(b)
                    shas.append(s)

            if len(shas) > self.total * 1.25:
                break

        for i in tqdm(range(len(data)), total=len(data), desc="Converting to tensors..."):
            t = torch.frombuffer(data[i], dtype=torch.uint8).to(torch.int16)
            t = t + len(self.special_token_ids)
            t = torch.cat([torch.tensor([self.bos_token_id]), t, torch.tensor([self.eos_token_id])])
            data[i] = t
            if (i + 1) % 100 == 0:
                gc.collect()

        return shas, data

    def test(self):
        shas, data = self.get_data()

        args = (self.bos_token_id, self.eos_token_id, self.pad_token_id, self.chunk_size, shas, True, "pass")
        print("Building the maskers (this can take a minute or two)...")
        with print_context(suppress=True):
            masker_num: AutoNumChunkFeatureMasker = get_masker(ExplanationMethod.NUM, *args)
            masker_len: AutoLenChunkFeatureMasker = get_masker(ExplanationMethod.LEN, *args)
            masker_nml: AutoNumLenChunkFeatureMasker = get_masker(ExplanationMethod.NML, *args)
            masker_fun: FunctionFeatureMasker = get_masker(ExplanationMethod.FUN, *args)
            masker_chk: ChunkFeatureMasker = get_masker(ExplanationMethod.CHK, *args)

        present = set(masker_fun.boundaries.keys())
        remove  = []
        for i, (s, t) in enumerate(zip(shas, data)):
            if s not in present:
                remove.append(i)
        remove = set(remove)
        shas = [s for i, s in enumerate(shas) if i not in remove]
        data = [t for i, t in enumerate(data) if i not in remove]

        idx = np.argsort(np.array(shas, dtype=str))
        shas = [shas[i] for i in idx]
        data = [data[i] for i in idx]

        if len(shas) > self.total:
            shas = shas[:self.total]
            data = data[:self.total]

        gc.collect()

        num_errors = 0
        error_logs = defaultdict(int)
        for i, (s, t) in tqdm(enumerate(zip(shas, data)), total=len(shas), desc="Testing..."):
            errors = []

            names     = [s]

            input_ids = t
            if self.apply_padding:
                input_ids = torch.cat([t, torch.zeros(self.max_length - len(t), dtype=torch.int64)])
            input_ids = input_ids.unsqueeze(0).to(torch.int64)

            special_idx = torch.full_like(input_ids[0], False)
            for t in self.special_token_ids:
                special_idx |= input_ids[0] == t

            mask_chk = masker_chk(input_ids, names)
            mask_num = masker_num(input_ids, names)
            mask_len = masker_len(input_ids, names)
            mask_nml = masker_nml(input_ids, names)
            mask_fun = masker_fun(input_ids, names)

            unq_num = len(torch.unique(mask_num))
            unq_len = len(torch.unique(mask_len))  # pylint: disable=unused-variable
            unq_fun = len(torch.unique(mask_fun))
            unq_nml = len(torch.unique(mask_nml))
            unq_chk = len(torch.unique(mask_chk))  # pylint: disable=unused-variable

            for mask, name in ((mask_chk, "chk"), (mask_num, "num"), (mask_len, "len"), (mask_nml, "nml"), (mask_fun, "fun")):
                try:
                    assert_feature_mask_indices_are_consecutive(mask[0])
                except RuntimeError:
                    msg = f"{name} mask indices are not consecutive"
                    errors.append(msg)
                    error_logs[msg] += 1

            totally_overlapping_functions = False
            if set(torch.unique(mask_fun).tolist()) == {0, 1} and len(masker_fun.boundaries[s]) > 0:
                max_length = masker_fun.get_last_idx(input_ids[0]) - 1
                bounds = masker_fun.select_valid_bounds(masker_fun.boundaries[s], max_length)
                totally_overlapping_functions = bounds_contain_totally_overlapping_functions(bounds)
                if totally_overlapping_functions:
                    msg = "totally overlapping functions"
                    errors.append(f"{msg}")
                    error_logs[msg] += 1

            num_function_within_max_length = len(masker_fun.boundaries[s]) - masker_fun.number_of_functions_outside_input(input_ids[0], s)
            if unq_fun != num_function_within_max_length + 2 and not totally_overlapping_functions:
                msg = "unq_fun != num_function_within_max_length + 2"
                errors.append(f"{msg} ({unq_fun} != {num_function_within_max_length + 2})")
                error_logs[msg] += 1

            if unq_num != unq_fun and not totally_overlapping_functions:
                msg = "unq_num != unq_fun"
                errors.append(f"{msg} ({unq_num} != {unq_fun})")
                error_logs[msg] += 1

            if unq_nml != unq_fun and not totally_overlapping_functions:
                msg = "unq_nml != unq_fun"
                errors.append(f"{msg} ({unq_nml} != {unq_fun})")
                error_logs[msg] += 1

            num_chunk_sizes = infer_chunk_sizes(mask_num[0][~special_idx])
            if len(set(num_chunk_sizes)) not in (1, 2):
                msg = "len(set(num_chunk_sizes)) not in (1, 2)"
                errors.append(f"{msg} ({len(set(num_chunk_sizes))} not in (1, 2))")
                error_logs[msg] += 1

            nml_chunk_sizes = infer_chunk_sizes(mask_nml[0][~special_idx])
            if len(set(nml_chunk_sizes)) not in (1, 2):
                msg = "len(set(nml_chunk_sizes)) not in (1, 2)"
                errors.append(f"{msg} ({len(set(nml_chunk_sizes))} not in (1, 2))")
                error_logs[msg] += 1

            len_chunk_sizes = infer_chunk_sizes(mask_len[0][~special_idx])
            if len(set(len_chunk_sizes)) not in (1, 2):
                msg = "len(set(len_chunk_sizes)) not in (1, 2)"
                errors.append(f"{msg} ({len(set(len_chunk_sizes))} not in (1, 2))")
                error_logs[msg] += 1

            if set(nml_chunk_sizes) != set(len_chunk_sizes):
                msg = "set(nml_chunk_sizes) != set(len_chunk_sizes)"
                errors.append(f"{msg} ({set(nml_chunk_sizes)} != {set(len_chunk_sizes)})")
                error_logs[msg] += 1

            if errors:
                num_errors += 1
                print(s)
                for e in errors:
                    print(f"\t{e}")

        assert num_errors == 0, f"Errors Occurred: {num_errors}\n{pformat(dict(error_logs))}"


class TestTopkRankMatrix(unittest.TestCase):
    # ---------- correctness for lower_is_higher = True -----------------
    def test_lower_is_higher_selection_and_rerank(self):
        """
        Col-0 ascending ranks, col-1 descending ranks just to make
        the selection non-trivial.  k = [2, 3] should keep rows
        (0, 1) for col-0 and (2, 3, 4) for col-1 → every row once.
        """
        S = np.array([
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1],
        ])
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        out = topk_rank_matrix(R, k=[2, 3], lower_is_higher=True)

        # Same *set* of rows as input, but re-ranked
        self.assertEqual(out.shape, R.shape)
        self.assertTrue(is_proper_rank_matrix(out))

        # Ordering relation preserved: row 0 better than row 1 for judge-0
        # (rank 1 < rank 2), etc.
        self.assertLess(out[0, 0], out[1, 0])
        self.assertLess(out[4, 1], out[3, 1])

    # ---------- correctness for lower_is_higher = False ----------------
    def test_higher_is_better_selection_and_rerank(self):
        """
        Largest values are best.  Keep global k = 2.  We expect rows
        with original values 5 and 4 only, re-ranked to 1 and 2.
        """
        S = np.array([
            [5, 5],
            [4, 4],
            [3, 3],
            [2, 2],
            [1, 1],
        ])
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        out = topk_rank_matrix(R, k=2, lower_is_higher=False)

        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(is_proper_rank_matrix(out))
        # The row with the original value 5 should now have rank 1
        self.assertTrue(np.all(out[0] == 1))

    # ---------- tie handling ------------------------------------------
    def test_ties_are_kept(self):
        """
        Two items tie for rank 1; k = 1 must keep *both*.
        """
        S = np.array([
            [1, 1],
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        out = topk_rank_matrix(R, k=1, lower_is_higher=True)

        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(is_proper_rank_matrix(out))

    # ---------- k = None returns an identical copy --------------------
    def test_k_none_returns_copy(self):
        S = np.random.rand(6, 4)
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        out = topk_rank_matrix(R, k=None)

        self.assertTrue(np.array_equal(out, R))
        self.assertIsNot(out, R)          # must be a *copy*

    # ---------- per-judge k vector accepted and validated -------------
    def test_per_judge_k_vector(self):
        S = np.random.rand(10, 3)
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        out = topk_rank_matrix(R, k=[1, 2, 3])

        self.assertTrue(is_proper_rank_matrix(out))
        # Each judge should have ≤ k[j] items with rank 1 … k[j]
        self.assertLessEqual((out[:, 0] <= 1).sum(), 1)
        self.assertLessEqual((out[:, 1] <= 2).sum(), 2)
        self.assertLessEqual((out[:, 2] <= 3).sum(), 3)

    # ---------- invalid inputs raise ValueError -----------------------
    def test_bad_k_shape_raises(self):
        S = np.random.rand(5, 2)
        R = rankdata(S, axis=0)
        self.assertTrue(is_proper_rank_matrix(R))
        with self.assertRaises(ValueError):
            topk_rank_matrix(R, k=[1, 2, 3])   # wrong length

    def test_non_2d_input_raises(self):
        S = np.arange(5)          # 1-D
        R = rankdata(S)
        with self.assertRaises(ValueError):
            topk_rank_matrix(R, k=1)


class TestStatisticalFunctions(unittest.TestCase):

    def setUp(self):
        self.J = 5
        self.I = 101
        self.S = np.random.rand(self.I, self.J)
        self.R = rankdata(self.S, axis=0)

    def test_spearmanr(self):
        res = spearmanr(self.R[:,0:2])
        assert -1.0 <= res.statistic <= 1.0, f"Got: {res.statistic}, Expected: -1.0 <= {res.statistic} <= 1.0"
        assert 0.0 <= res.pvalue <= 1.0, f"Got: {res.pvalue}, Expected: 0.0 <= {res.pvalue} <= 1.0"

    def test_kendalltau(self):
        res = kendalltau(self.R[:,0:2])
        assert -1.0 <= res.statistic <= 1.0, f"Got: {res.statistic}, Expected: -1.0 <= {res.statistic} <= 1.0"
        assert 0.0 <= res.pvalue <= 1.0, f"Got: {res.pvalue}, Expected: 0.0 <= {res.pvalue} <= 1.0"

    def test_kendallw(self):
        res = kendallw(self.R)
        assert 0.0 <= res.statistic <= 1.0, f"Got: {res.statistic}, Expected: 0.0 <= {res.statistic} <= 1.0"
        assert 0.0 <= res.pvalue <= 1.0, f"Got: {res.pvalue}, Expected: 0.0 <= {res.pvalue} <= 1.0"


class TestDescriptiveSparsity(unittest.TestCase):
    """
    Each test checks an intuitive mathematical property that must hold if the
    code reproduces Definition 3 in the paper.
    """

    # ---------- 1.  Uniform distribution on [-1,1]  --------------------------
    def test_uniform_linear_curve(self):
        """
        For a *uniform* pdf on [-1,1] we have  h(x)=½,  so

            MAZ(r) = ∫_{-r}^{r} ½ dx = r      (for r∈[0,1])

        Hence the MAZ curve should be (close to) the identity line and the area
        under the curve should be ≈0.5.
        """
        rng = np.random.default_rng(0)
        rel = rng.uniform(-100, 100, size=100_000)   # any scale, we rescale inside
        r_grid = np.linspace(0, 1, 21)               # 0.0, 0.05, …, 1.0

        r, maz, auc = descriptive_sparsity(rel, r_grid=r_grid, num_bins=400)

        # Curve ~ r  (allow small numerical error)
        np.testing.assert_allclose(maz, r, atol=3e-2)

        # Area under curve ~ 0.5
        self.assertAlmostEqual(auc, 0.5, delta=3e-2)

    # ---------- 2.  “Sparse” signal: many zeros, few large values ------------
    def test_sparse_signal_high_mass_near_zero(self):
        """
        If 90 % of the relevance scores are exactly zero, almost all probability
        mass sits at 0; MAZ(r) should jump steeply near the origin.
        Concretely, by r = 0.05 we expect ≥80 % of the mass to be captured.
        """
        rng = np.random.default_rng(1)
        n = 10_000
        rel = np.zeros(n)
        rel[:1_000] = rng.normal(scale=10.0, size=1_000)   # ± large values

        _, maz, auc = descriptive_sparsity(rel, r_grid=np.array([0.01, 0.05, 0.1]),
                                      num_bins=400)

        self.assertGreater(maz[1], 0.80)   # MAZ(0.05) > 0.8

    # ---------- 3.  Monotonicity & bounds  -----------------------------------
    def test_monotone_and_bounds(self):
        """
        MAZ(r) is a cumulative probability → must be non‑decreasing, between
        0 and 1, and equal to 1 when r = 1.
        """
        rng = np.random.default_rng(2)
        rel = rng.standard_normal(5_000)

        r, maz, auc = descriptive_sparsity(rel)

        # Non‑decreasing
        self.assertTrue(np.all(np.diff(maz) >= -1e-6))
        # Bounds
        self.assertGreaterEqual(maz.min(), 0.0)
        self.assertLessEqual(maz.max(), 1.0 + 1e-6)
        # MAZ(1) ≈ 1
        self.assertAlmostEqual(maz[-1], 1.0, places=3)

    # ---------- 4.  Scale invariance  ----------------------------------------
    def test_scale_invariance(self):
        """
        Because we rescale by max|r_i|, multiplying the input by any constant
        factor must leave the MAZ curve unchanged.
        """
        rng = np.random.default_rng(3)
        rel = rng.normal(size=2_000)

        r1, maz1, auc1 = descriptive_sparsity(rel)
        r2, maz2, auc2 = descriptive_sparsity(rel * 42.0)

        np.testing.assert_allclose(r1, r2, rtol=0, atol=0)   # grids identical
        np.testing.assert_allclose(maz1, maz2, rtol=1e-6)

    def test_constant_relevances(self):
        """
        If all relevance scores are zero, the MAZ curve is flat at 0.
        """
        rel = np.zeros(1000)
        with self.assertRaises(ValueError):
            r, maz, auc = descriptive_sparsity(rel, constant_relevances="raise")
        with self.assertWarns(Warning):
            r, maz, auc = descriptive_sparsity(rel, constant_relevances="warn")

        r, maz, auc = descriptive_sparsity(rel, constant_relevances="ignore")
        self.assertEqual(auc, 0.9975)
        r, maz, auc = descriptive_sparsity(rel, constant_relevances="auto")
        self.assertEqual(auc, 0.0)


class TestDiceAndJaccard(unittest.TestCase):

    def setUp(self):
        self.S = np.random.rand(100, 5)
        self.R = rankdata(self.S, axis=0)

    def test_dice(self):
        result = compute_agreement_dice(self.R, 10)

    def test_jaccard(self):
        result = compute_agreement_jaccard(self.R, 10)


def mequal(*args):
    """Check if all tensors are equal."""
    for i in range(1, len(args)):
        if not torch.equal(args[0], args[i]):
            return False
    return True


class SegmentedTensorTest(unittest.TestCase):

    @staticmethod
    def dense_sample():
        """0–1023=0, 1024–16383=1, 16384–24575=17 (float32)"""
        return torch.cat(
            [
                torch.zeros(1024),
                torch.ones(15360),
                torch.full((8192,), 17.0),
            ]
        ).float()

    # ---------- construction / round-trip -----------------------------------
    def test_from_dense_roundtrip(self):
        x = SegmentedTensorTest.dense_sample()
        seg = SegmentedTensor.from_dense(x)
        self.assertEqual(len(seg), x.numel())
        self.assertListEqual(seg.lengths.tolist(), [1024, 15360, 8192])
        self.assertTrue(torch.equal(seg.values, torch.tensor([0.0, 1.0, 17.0])))
        self.assertTrue(torch.equal(seg.to_dense(), x))

    def test_single_segment(self):
        x = torch.full((5000,), 7, dtype=torch.int32)
        seg = SegmentedTensor.from_dense(x)
        self.assertListEqual(seg.lengths.tolist(), [5000])
        self.assertListEqual(seg.values.tolist(), [7])
        self.assertTrue(torch.equal(seg.to_dense(), x))

    def test_empty(self):
        x = torch.empty(0)
        seg = SegmentedTensor.from_dense(x)
        self.assertEqual(len(seg), 0)
        self.assertEqual(seg.lengths.numel(), 0)
        self.assertEqual(seg.values.numel(), 0)
        self.assertTrue(torch.equal(seg.to_dense(), x))

    # ---------- scalar indexing ---------------------------------------------
    def test_scalar_indexing(self):
        seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        self.assertEqual(seg[0].item(), 0.0)
        self.assertEqual(seg[2048].item(), 1.0)
        self.assertEqual(seg[len(seg) - 1].item(), 17.0)
        self.assertEqual(seg[-1].item(), 17.0)
        with self.assertRaises(IndexError):
            _ = seg[len(seg)]
        with self.assertRaises(IndexError):
            _ = seg[-len(seg) - 1]

    # ---------- slicing ------------------------------------------------------
    def test_slice_variants(self):
        den = SegmentedTensorTest.dense_sample()
        seg = SegmentedTensor.from_dense(den)

        # full slice
        sub = seg[:]
        self.assertIsInstance(sub, SegmentedTensor)
        self.assertEqual(len(sub), len(seg))
        self.assertTrue(mequal(sub.to_dense(), seg.to_dense(), den))

        # aligned slice
        sub = seg[1024:17408]
        self.assertListEqual(sub.lengths.tolist(), SegmentedTensor.from_dense(den[1024:17408]).lengths.tolist())
        self.assertTrue(mequal(sub.to_dense(), seg.to_dense()[1024:17408], den[1024:17408]))

        # mid-segment slice
        sub = seg[512:1536]
        self.assertTrue(torch.equal(sub.to_dense(), seg.to_dense()[512:1536]))
        self.assertTrue(torch.all(sub.lengths > 0))

    def test_step_slice_fallback(self):
        seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        dense_sub = seg[::2]
        self.assertTrue(torch.is_tensor(dense_sub))
        self.assertTrue(torch.equal(dense_sub, seg.to_dense()[::2]))

    # ---------- IO (path & file object) --------------------------------------
    def test_save_load_path(self):
        seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "seg.pt")
            seg.save(fname)
            re = SegmentedTensor.load(fname)
            self.assertTrue(torch.equal(re.to_dense(), seg.to_dense()))

    def test_save_load_fileobj(self):
        seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        buf = io.BytesIO()
        seg.save(buf)
        buf.seek(0)
        re = SegmentedTensor.load(buf)
        self.assertTrue(torch.equal(re.to_dense(), seg.to_dense()))

    def test_save_cpu_load_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        cpu_seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "gpu.pt")
            cpu_seg.save(fname)
            gpu_seg = SegmentedTensor.load(fname, map_location="cuda:0")
            self.assertEqual(gpu_seg.device.type, "cuda")
            self.assertTrue(
                torch.equal(gpu_seg.to_dense().cpu(), cpu_seg.to_dense())
            )

    def test_private_slice_helpers(self):
        seg = SegmentedTensor.from_dense(SegmentedTensorTest.dense_sample())
        left = seg._slice(512, len(seg))
        right = seg._slice(0, len(seg) - 512)
        self.assertEqual(len(left) + 512, len(seg))
        self.assertEqual(len(right) + 512, len(seg))
