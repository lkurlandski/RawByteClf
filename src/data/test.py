"""
Some tests for the loaders_core module.
"""

import bz2
from collections import Counter
import gzip
from io import BytesIO
import lzma
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import zlib

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import py7zr

from src.data.loaders_core import (
    compute_integer_sizes,
    compute_float_sizes,
    tr_vl_ts_split_idx,
    tr_vl_ts_split,
    tr_vl_ts_split_idx_guarentee,
    tr_vl_ts_split_guarentee,
)
from src.data.utils import Decompressor


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


class TestDecompressor(unittest.TestCase):
    def setUp(self):
        self.test_data = b'This is a test string.'
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.test_dir.name) / "test.bytes"
        with open(self.test_file, 'wb') as f:
            f.write(self.test_data)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_none_decompression(self):
        compressed_data = self.test_data
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.NONE)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.NONE)
            self.assertEqual(b, self.test_data)

    def test_gzip_decompression(self):
        compressed_data = gzip.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.GZIP)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.GZIP)
            self.assertEqual(b, self.test_data)

    def test_bzip2_decompression(self):
        compressed_data = bz2.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.BZIP2)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.BZIP2)
            self.assertEqual(b, self.test_data)

    def test_lzma_decompression(self):
        compressed_data = lzma.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.LZMA)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.LZMA)
            self.assertEqual(b, self.test_data)

    def test_zlib_decompression(self):
        compressed_data = zlib.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.ZLIB)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.ZLIB)
            self.assertEqual(b, self.test_data)

    # def test_py7zr_decompression(self):
    #     fp = BytesIO()
    #     with py7zr.SevenZipFile(fp, 'w') as archive:
    #         archive.writef(BytesIO(self.test_data), "tmp")
    #     fp.seek(0)
    #     compressed_data = fp.read()
    #     decompressor = Decompressor(Decompressor.S7Z)
    #     alg, b = decompressor(BytesIO(compressed_data))
    #     self.assertEqual(alg, Decompressor.S7Z)
    #     self.assertEqual(b, self.test_data)


if __name__ == "__main__":
    unittest.main()
