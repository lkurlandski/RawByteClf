"""
Test.
"""

import os
from pathlib import Path
import random
import shutil
import unittest

from src.data.detect_packing_sorel import pack, unpack
from src.data.executable_sections import get_executable_section
from src.data.pe_utils import rearm_disarmed_binary
from src.learn.preprocessing import hf_pack_bytes, hf_unpack_bytes, hf_rearm_bytes, hf_get_exe_bytes


class TestHFFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.file_1 = Path("./tests/fixtures/000015552e927279b35bec687c5a1840c0dbe2f2b7b03c417cbd70645a0cc073.exe")
        cls.file_2 = Path("./tests/fixtures/000dd231f1ce36036682e578843bd352d8482ad47e0e26d6ef84c07d2cca6c7f.exe")
        cls.file_3 = Path("./tests/fixtures/0010e43084953a9f78eb01ea2e9184bed7ae558a92e7ac926b9892cbfb39f3ac.exe")
        cls.file_4 = Path("./tests/fixtures/1233d51ceede157adf6dd696af7e3901f55b256b35087f20c86e04d77092957e.exe")

        cls.raw_1 = cls.file_1.read_bytes()
        cls.raw_2 = cls.file_2.read_bytes()
        cls.raw_3 = cls.file_3.read_bytes()
        cls.raw_4 = cls.file_4.read_bytes()

        cls.bin_1 = rearm_disarmed_binary(cls.file_1, cls.file_1.stem)
        cls.bin_2 = rearm_disarmed_binary(cls.file_2, cls.file_2.stem)
        cls.bin_3 = rearm_disarmed_binary(cls.file_3, cls.file_3.stem)
        cls.bin_4 = rearm_disarmed_binary(cls.file_4, cls.file_4.stem)

        cls.pck_1 = pack(cls.bin_1, return_bytes=True)
        cls.pck_2 = pack(cls.bin_2, return_bytes=True)
        cls.pck_3 = pack(cls.bin_3, return_bytes=True)
        cls.pck_4 = pack(cls.bin_4, return_bytes=True)

        cls.unp_1 = unpack(cls.pck_1, return_bytes=True)
        cls.unp_2 = unpack(cls.pck_2, return_bytes=True)
        cls.unp_3 = unpack(cls.pck_3, return_bytes=True)
        cls.unp_4 = unpack(cls.pck_4, return_bytes=True)

        cls.exe_1 = get_executable_section(file=cls.file_1)
        cls.exe_2 = get_executable_section(file=cls.file_2)
        cls.exe_3 = get_executable_section(file=cls.file_3)
        cls.exe_4 = get_executable_section(file=cls.file_4)

    def setUp(self):
        self.examples = {
            "bytes": [self.bin_1, self.bin_2, self.bin_3, self.bin_4],
            "name":  [self.file_1.stem, self.file_2.stem, self.file_3.stem, self.file_4.stem],
        }
        self.examples_packed = {
            "bytes": [self.pck_1, self.pck_2, self.pck_3, self.pck_4],
            "name":  [self.file_1.stem, self.file_2.stem, self.file_3.stem, self.file_4.stem],
        }
        self.examples_raw = {
            "bytes": [self.raw_1, self.raw_2, self.raw_3, self.raw_4], 
            "name":  [self.file_1.stem, self.file_2.stem, self.file_3.stem, self.file_4.stem],
        }

    def _test_hf_core(self, d):
        assert isinstance(d, dict)
        assert "bytes" in d
        assert isinstance(d["bytes"], list)
        assert isinstance(d["bytes"][0], bytes)
        assert isinstance(d["bytes"][1], bytes)
        assert isinstance(d["bytes"][2], bytes)
        assert isinstance(d["bytes"][3], bytes)

    def test_hf_pack_bytes_1(self):
        d = hf_pack_bytes(self.examples, probability=1.0)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.pck_1
        assert d["bytes"][1] == self.pck_2
        assert d["bytes"][2] == self.pck_3
        assert d["bytes"][3] == self.pck_4

    def test_hf_pack_bytes_2(self):
        d = hf_pack_bytes(self.examples, probability=0.0)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.bin_1
        assert d["bytes"][1] == self.bin_2
        assert d["bytes"][2] == self.bin_3
        assert d["bytes"][3] == self.bin_4

    def test_hf_unpack_bytes_1(self):
        d = hf_unpack_bytes(self.examples_packed, probability=1.0)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.unp_1
        assert d["bytes"][1] == self.unp_2
        assert d["bytes"][2] == self.unp_3
        assert d["bytes"][3] == self.unp_4

    def test_hf_unpack_bytes_2(self):
        d = hf_unpack_bytes(self.examples_packed, probability=0.0)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.pck_1
        assert d["bytes"][1] == self.pck_2
        assert d["bytes"][2] == self.pck_3
        assert d["bytes"][3] == self.pck_4

    def test_hf_rearm_bytes_1(self):
        d = hf_rearm_bytes(self.examples_raw)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.bin_1
        assert d["bytes"][1] == self.bin_2
        assert d["bytes"][2] == self.bin_3
        assert d["bytes"][3] == self.bin_4

    def test_hf_rearm_bytes_2(self):
        d = hf_rearm_bytes(self.examples)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.bin_1
        assert d["bytes"][1] == self.bin_2
        assert d["bytes"][2] == self.bin_3
        assert d["bytes"][3] == self.bin_4

    def test_hf_get_exe_bytes_1(self):
        d = hf_get_exe_bytes(self.examples)
        self._test_hf_core(d)
        assert d["bytes"][0] == self.exe_1
        assert d["bytes"][1] == self.exe_2
        assert d["bytes"][2] == self.exe_3
        assert d["bytes"][3] == self.exe_4

    def test_hf_get_exe_bytes_2(self):
        d = hf_get_exe_bytes(self.examples_packed)
        self._test_hf_core(d)
        assert d["bytes"][0] != self.exe_1
        assert d["bytes"][1] != self.exe_2
        assert d["bytes"][2] != self.exe_3
        assert d["bytes"][3] != self.exe_4
