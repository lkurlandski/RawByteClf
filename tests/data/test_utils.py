"""
Test.
"""

import bz2
import gzip
from io import BytesIO
import lzma
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

try:
    import py7zr
except (ModuleNotFoundError, ImportError) as _err:
    print(f"{_err.__class__.__name__}: py7zr")

from src.data.utils import Decompressor


ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"


class TestDecompressor(unittest.TestCase):

    def setUp(self):
        self.test_data = b'This is a test string.'
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
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

    @unittest.skip("Skipping TestDecompressor.test_py7zr_decompression")
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
