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

from src.data.cfg import PACKING_ROOTS
from src.data.detect_packing_sorel import PackingMap, pack, unpack


ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"


class TestPackingMap(unittest.TestCase):

    root = PACKING_ROOTS["bodmas_pe"]
    maps = []

    @classmethod
    def setUpClass(cls):
        cls.maps = []

    @classmethod
    def tearDownClass(cls):
        if ENABLE_UNITTEST_LOGGING:
            print(f"TestPackingMap: {len(cls.maps)} maps created.")
        for map_a in cls.maps:
            for map_b in cls.maps:
                assert map_a == map_b
        if ENABLE_UNITTEST_LOGGING:
            print("All maps are equal.")

    def test_packing_map_0(self):
        t = time.time()
        packing_map = PackingMap(root=self.root, lazy=False, chunked=True, num_workers=16, cache_load=False, cache_save=False)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)
        self.maps.append(packing_map)

    def test_packing_map_1(self):
        t = time.time()
        packing_map = PackingMap(root=self.root, lazy=False, chunked=True, num_workers=None, cache_load=False, cache_save=False)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)
        self.maps.append(packing_map)

    def test_packing_map_2(self):
        t = time.time()
        packing_map = PackingMap(root=self.root, lazy=False, chunked=False, num_workers=None, cache_load=False, cache_save=False)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)
        self.maps.append(packing_map)

    def test_packing_map_3(self):
        t = time.time()
        packing_map = PackingMap(root=self.root, lazy=True, chunked=True, num_workers=16, cache_load=False, cache_save=False)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)
        self.maps.append(packing_map)

    def test_packing_map_4(self):
        t = time.time()
        packing_map = PackingMap(root=self.root, lazy=True, chunked=False, num_workers=None, cache_load=False, cache_save=False)
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_packing_map_0: {len(packing_map)} samples {time.time() - t:.2f} seconds")
        self.assertTrue(len(packing_map) > 0)
        self.maps.append(packing_map)


class TestPack(unittest.TestCase):

    @classmethod
    def setUpClass(clf):
        clf.tempdir = tempfile.mkdtemp(prefix="TestPack_")
        clf.tmpcpp = os.path.join(clf.tempdir, "hello.cpp")
        clf.tmpbin = os.path.join(clf.tempdir, "hello.bin")

        with open(clf.tmpcpp, "w") as fp:
            fp.write("#include <iostream>\n")
            fp.write("int main() {\n")
            fp.write(f"std::cout << \"Hello World!\";")
            fp.write("return 0;\n")
            fp.write("}\n")

        args = ["g++", "-o", clf.tmpbin, clf.tmpcpp]
        subprocess.run(args, check=True)
        TestPack.check_binary(clf.tmpbin)

    @classmethod
    def tearDownClass(clf):
        shutil.rmtree(clf.tempdir, ignore_errors=True)

    @classmethod
    def check_binary(cls, binfile: str) -> bool:
        result = subprocess.run([binfile], check=True, capture_output=True)
        assert result.returncode == 0, f"Binary {binfile} did not run successfully."
        assert result.stdout.decode().strip() == "Hello World!", f"Binary {binfile} did not produce expected output."

    def _pack(self, data, outfile, return_bytes, errors):
        try:
            return pack(data, outfile, return_bytes, errors)
        except subprocess.CalledProcessError as err:
            print("stdout:", err.stdout.decode() if err.stdout else "None")
            print("stderr:", err.stderr.decode() if err.stderr else "None")
            raise

    def test_1(self):
        b = self._pack(self.tmpbin, None, True, "raise")
        assert isinstance(b, bytes), type(b)

    def test_2(self):
        with open(self.tmpbin, "rb") as fp:
            data = fp.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmpfile.bin")
            b = self._pack(data, tmpfile, True, "raise")
            self.check_binary(tmpfile)
        assert isinstance(b, bytes), type(b)

    def test_3(self):
        with open(self.tmpbin, "rb") as fp:
            data = fp.read()
        b = self._pack(data, None, True, "raise")
        assert isinstance(b, bytes), type(b)

    def test_4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmpfile.bin")
            b = self._pack(self.tmpbin, tmpfile, True, "raise")
            self.check_binary(tmpfile)
        assert isinstance(b, bytes), type(b)


class TestUnpack(unittest.TestCase):
    ...
