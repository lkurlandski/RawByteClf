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


def write_hello_world_cpp(file_path: str):
    with open(file_path, "w") as fp:
        fp.write("#include <iostream>\n")
        fp.write("int main() {\n")
        fp.write('std::cout << "Hello World!";\n')
        fp.write("return 0;\n")
        fp.write("}\n")


def compile_hello_world_cpp(file_path: str, output_bin: str):
    args = ["g++", "-o", output_bin, file_path]
    subprocess.run(args, check=True)


def check_hello_world_out(output_bin: str):
    result = subprocess.run([output_bin], check=True, capture_output=True)
    if result.returncode == 0 and result.stdout is not None and result.stdout.decode().strip() == "Hello World!":
        return None
    raise RuntimeError(f"Binary {output_bin} returned exit code {result.returncode} and printed output: {result.stdout}")


class TestPack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix="TestPack_")
        cls.tmpcpp = os.path.join(cls.tempdir, "hello.cpp")
        cls.tmpbin = os.path.join(cls.tempdir, "hello.bin")
        write_hello_world_cpp(cls.tmpcpp)
        compile_hello_world_cpp(cls.tmpcpp, cls.tmpbin)
        check_hello_world_out(cls.tmpbin)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

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
            check_hello_world_out(tmpfile)
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
            check_hello_world_out(tmpfile)
        assert isinstance(b, bytes), type(b)


class TestUnpack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix="TestUnpack_")
        cls.tmpcpp = os.path.join(cls.tempdir, "hello.cpp")
        cls.tmpbin = os.path.join(cls.tempdir, "hello.bin")
        cls.tmppck = os.path.join(cls.tempdir, "hello.pck")
        write_hello_world_cpp(cls.tmpcpp)
        compile_hello_world_cpp(cls.tmpcpp, cls.tmpbin)
        check_hello_world_out(cls.tmpbin)
        pack(cls.tmpbin, cls.tmppck, False, "raise")
        check_hello_world_out(cls.tmppck)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def _unpack(self, data, outfile, return_bytes, errors):
        try:
            return unpack(data, outfile, return_bytes, errors)
        except subprocess.CalledProcessError as err:
            print("stdout:", err.stdout.decode() if err.stdout else "None")
            print("stderr:", err.stderr.decode() if err.stderr else "None")
            raise

    def test_1(self):
        b = self._unpack(self.tmppck, None, True, "raise")
        assert isinstance(b, bytes), type(b)

    def test_2(self):
        with open(self.tmppck, "rb") as fp:
            data = fp.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmpfile.bin")
            b = self._unpack(data, tmpfile, True, "raise")
            check_hello_world_out(tmpfile)
        assert isinstance(b, bytes), type(b)

    def test_3(self):
        with open(self.tmppck, "rb") as fp:
            data = fp.read()
        b = self._unpack(data, None, True, "raise")
        assert isinstance(b, bytes), type(b)

    def test_4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmpfile.bin")
            b = self._unpack(self.tmppck, tmpfile, True, "raise")
            check_hello_world_out(tmpfile)
        assert isinstance(b, bytes), type(b)

    def test_5(self):
        with self.assertRaises(subprocess.CalledProcessError):
            unpack(self.tmpbin, None, True, "raise", return_original_if_not_packed=False)
        b = unpack(self.tmpbin, None, True, "raise", return_original_if_not_packed=True)
        assert isinstance(b, bytes), type(b)
        assert b == Path(self.tmpbin).read_bytes(), "Unpacked bytes do not match original binary."

    def test_6(self):
        with open(self.tmpbin, "rb") as fp:
            data = fp.read()
        with self.assertRaises(subprocess.CalledProcessError):
            unpack(data, None, True, "raise", return_original_if_not_packed=False)
        b = unpack(data, None, True, "raise", return_original_if_not_packed=True)
        assert isinstance(b, bytes), type(b)
        assert b == Path(self.tmpbin).read_bytes(), "Unpacked bytes do not match original binary."
