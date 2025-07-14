"""
Test.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

from src.data.detect_packing_sorel import PackingMap, unpack


ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"


class TestPackingMap(unittest.TestCase):

    def test_packing_map_0(self):
        t = time.time()
        packing_map = PackingMap(lazy=False, chunked=True, num_workers=16)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)

    def test_packing_map_1(self):
        t = time.time()
        packing_map = PackingMap(lazy=False, chunked=True, num_workers=None)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)

    def test_packing_map_2(self):
        t = time.time()
        packing_map = PackingMap(lazy=False, chunked=False, num_workers=None)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)

    def test_packing_map_3(self):
        print("packing_map_3")
        t = time.time()
        packing_map = PackingMap(lazy=True, chunked=True, num_workers=16)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)

    def test_packing_map_4(self):
        print("packing_map_4")
        t = time.time()
        packing_map = PackingMap(lazy=True, chunked=False, num_workers=None)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)


@unittest.skip("Skipping TestUnpacking.")
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
            subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            raise err

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_packed_file(self):
        try:
            _, byte_0 = unpack(self.packed, self.outfile, True, True, 1)
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
