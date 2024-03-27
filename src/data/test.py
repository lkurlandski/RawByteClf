"""
Some tests for the loaders_core module.
"""

from collections import Counter
import os
import sys
import unittest

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from loaders_core import (
    compute_integer_sizes,
    compute_float_sizes,
    tr_vl_ts_split_idx,
    tr_vl_ts_split,
    tr_vl_ts_split_idx_guarentee,
    tr_vl_ts_split_guarentee,
)

class TestSplitFunctions(unittest.TestCase):
    def test_compute_integer_sizes(self):
        total = 100
        self.assertEqual(compute_integer_sizes(total, 0.8, 0.1, 0.1), (80, 10, 10))
        self.assertEqual(compute_integer_sizes(total, 80, 10, 10), (80, 10, 10))

    def test_compute_float_sizes(self):
        total = 100
        self.assertEqual(compute_float_sizes(total, 0.8, 0.1, 0.1), (0.8, 0.1, 0.1))
        self.assertEqual(compute_float_sizes(total, 80, 10, 10), (0.8, 0.1, 0.1))

    def test_tr_vl_ts_split_idx(self):
        total = 100
        split_idx = tr_vl_ts_split_idx(total, 0.8, 0.1, 0.1)
        self.assertEqual(len(split_idx["tr"]), 80)
        self.assertEqual(len(split_idx["vl"]), 10)
        self.assertEqual(len(split_idx["ts"]), 10)

    def test_tr_vl_ts_split(self):
        collection = list(range(100))
        split = tr_vl_ts_split(collection, 0.8, 0.1, 0.1)
        self.assertEqual(len(split["tr"]), 80)
        self.assertEqual(len(split["vl"]), 10)
        self.assertEqual(len(split["ts"]), 10)

    def test_tr_vl_ts_split_idx_guarentee(self):
        labels = [0] * 50 + [1] * 50
        split_idx = tr_vl_ts_split_idx_guarentee(labels, 0.8, 0.1, 0.1, samples_per_class=5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["tr"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["tr"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["vl"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["vl"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["ts"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["ts"])[1], 5)

    def test_tr_vl_ts_split_guarentee(self):
        collection = list(range(100))
        labels = [0] * 50 + [1] * 50
        split = tr_vl_ts_split_guarentee(collection, labels, 0.8, 0.1, 0.1, samples_per_class=5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["tr"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["tr"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["vl"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["vl"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["ts"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split["ts"])[1], 5)


if __name__ == "__main__":
    unittest.main()
