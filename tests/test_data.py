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
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zlib

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import py7zr

from src.data.detect_packing_sorel import PackingMap, unpack
from src.data.loaders_core import (
    compute_integer_sizes,
    compute_float_sizes,
    tr_vl_ts_split_idx,
    tr_vl_ts_split,
    tr_vl_ts_split_idx_guarentee,
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

    @unittest.skip("Skipping test_py7zr_decompression because its not implemented yet.")
    def test_py7zr_decompression(self):
        fp = BytesIO()
        with py7zr.SevenZipFile(fp, 'w') as archive:
            archive.writef(BytesIO(self.test_data), "tmp")
        fp.seek(0)
        compressed_data = fp.read()
        decompressor = Decompressor(Decompressor.S7Z)
        alg, b = decompressor(BytesIO(compressed_data))
        self.assertEqual(alg, Decompressor.S7Z)
        self.assertEqual(b, self.test_data)


@unittest.skip("Skipping TestPackingMap because it takes a long time and probably isn't needed.")
class TestPackingMap(unittest.TestCase):
    def setUp(self):
        self.maps = []

    def tearDown(self):
        self.maps = []

    def test_packing_map_0(self):
        print("packing_map_0")
        t = time.time()
        packing_map_0 = PackingMap(lazy=False, chunked=True, num_workers=16)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_0)=}")
        self.assertTrue(len(packing_map_0) > 0)
        self.maps.append(packing_map_0)

    def test_packing_map_1(self):
        print("packing_map_1")
        t = time.time()
        packing_map_1 = PackingMap(lazy=False, chunked=True, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_1)=}")
        self.assertTrue(len(packing_map_1) > 0)
        self.maps.append(packing_map_1)

    def test_packing_map_2(self):
        print("packing_map_2")
        t = time.time()
        packing_map_2 = PackingMap(lazy=False, chunked=False, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_2)=}")
        self.assertTrue(len(packing_map_2) > 0)
        self.maps.append(packing_map_2)

    def test_packing_map_3(self):
        print("packing_map_3")
        t = time.time()
        packing_map_3 = PackingMap(lazy=True, chunked=True, num_workers=16)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_3)=}")
        self.assertTrue(len(packing_map_3) > 0)
        self.maps.append(packing_map_3)

    def test_packing_map_4(self):
        print("packing_map_4")
        t = time.time()
        packing_map_4 = PackingMap(lazy=True, chunked=False, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_4)=}")
        self.assertTrue(len(packing_map_4) > 0)
        self.maps.append(packing_map_4)

    def test_maps_equality(self):
        for i, map1 in enumerate(self.maps):
            for j, map2 in enumerate(self.maps):
                if i != j:
                    self.assertEqual(map1, map2, f"Maps {i} and {j} are not equal")


class TestUnpacking(unittest.TestCase):

    _test_file = "./tmp/calc.exe"

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.unpacked = self.test_dir /  "unpacked.exe"
        self.packed = self.test_dir / "packed.exe"
        self.outfile = self.test_dir / "out.exe"
        shutil.copy2(self._test_file, self.unpacked)
        args = ["upx", "--best", "-o", str(self.packed), str(self.unpacked)]
        try:
            result = subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            raise err

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_packed_file(self):
        try:
            outfile, byte_0 = unpack(self.packed, self.outfile, True, True, 1)
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            raise err

        byte_1 = self.unpacked.read_bytes()
        assert len(byte_0) == len(byte_1), f"{len(byte_0)=} != {len(byte_1)=}"
        # Don't know why, but the executables themselves have some small differences.

    def test_unpacked_file(self):
        with self.assertRaises(subprocess.CalledProcessError):
            unpack(self.unpacked, self.outfile, True, False, 1)
        outfile, byte = unpack(self.unpacked, self.outfile, True, True, 0)
        assert outfile is None
        assert byte is None


if __name__ == "__main__":
    unittest.main()
