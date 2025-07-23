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

import lief
import numpy as np

from src.data.cfg import PACKING_ROOTS
from src.data.pe_utils import patch_binary


# pylint: disable=c-extension-no-member


lief.logging.disable()


ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"


class TestPatchPEFile(unittest.TestCase):

    file_1 = Path("./tests/fixtures/000015552e927279b35bec687c5a1840c0dbe2f2b7b03c417cbd70645a0cc073.exe")
    file_2 = Path("./tests/fixtures/000dd231f1ce36036682e578843bd352d8482ad47e0e26d6ef84c07d2cca6c7f.exe")
    file_3 = Path("./tests/fixtures/0010e43084953a9f78eb01ea2e9184bed7ae558a92e7ac926b9892cbfb39f3ac.exe")
    file_4 = Path("./tests/fixtures/1233d51ceede157adf6dd696af7e3901f55b256b35087f20c86e04d77092957e.exe")

    @staticmethod
    def _test_patch_none(file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        new = patch_binary(org, machine=None, subsystem=None)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == org_subsystem
        assert new == org

    def test_patch_none_1(self):
        self._test_patch_none(self.file_1)

    def test_patch_none_2(self):
        self._test_patch_none(self.file_2)

    def test_patch_none_3(self):
        self._test_patch_none(self.file_3)

    def test_patch_none_4(self):
        self._test_patch_none(self.file_4)

    @staticmethod
    def _test_patch_same(file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        new = patch_binary(org, machine=org_machine, subsystem=org_subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == org_subsystem
        assert new == org

    def test_patch_same_1(self):
        self._test_patch_same(self.file_1)

    def test_patch_same_2(self):
        self._test_patch_same(self.file_2)

    def test_patch_same_3(self):
        self._test_patch_same(self.file_3)

    def test_patch_same_4(self):
        self._test_patch_same(self.file_4)

    @staticmethod
    def _test_patch_machine(file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd machine
        if org_machine == lief.PE.Header.MACHINE_TYPES.THUMB:
            machine = lief.PE.Header.MACHINE_TYPES.POWERPC
        else:
            machine = lief.PE.Header.MACHINE_TYPES.THUMB
        new = patch_binary(org, machine=machine, subsystem=None)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == machine
        assert new_subsystem == org_subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most two bytes should differ
        assert np.sum(equal) >= len(equal) - 2, f"{np.sum(equal)} {len(equal)}"

    def test_patch_machine_1(self):
        self._test_patch_machine(self.file_1)

    def test_patch_machine_2(self):
        self._test_patch_machine(self.file_2)

    def test_patch_machine_3(self):
        self._test_patch_machine(self.file_3)

    def test_patch_machine_4(self):
        self._test_patch_machine(self.file_4)

    @staticmethod
    def _test_patch_subsystem(file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd subsystem
        if org_subsystem == lief.PE.OptionalHeader.SUBSYSTEM.XBOX:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.OS2_CUI
        else:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.XBOX
        new = patch_binary(org, machine=None, subsystem=subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most one byte should differ
        assert np.sum(equal) >= len(equal) - 1, f"{np.sum(equal)} {len(equal)}"

    def test_patch_subsystem_1(self):
        self._test_patch_subsystem(self.file_1)

    def test_patch_subsystem_2(self):
        self._test_patch_subsystem(self.file_2)

    def test_patch_subsystem_3(self):
        self._test_patch_subsystem(self.file_3)

    def test_patch_subsystem_4(self):
        self._test_patch_subsystem(self.file_4)

    @staticmethod
    def _test_patch_machine_subsystem(file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd machine and subsystem
        if org_machine == lief.PE.Header.MACHINE_TYPES.THUMB:
            machine = lief.PE.Header.MACHINE_TYPES.POWERPC
        else:
            machine = lief.PE.Header.MACHINE_TYPES.THUMB
        if org_subsystem == lief.PE.OptionalHeader.SUBSYSTEM.XBOX:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.OS2_CUI
        else:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.XBOX
        new = patch_binary(org, machine=machine, subsystem=subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == machine
        assert new_subsystem == subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most three bytes should differ
        assert np.sum(equal) >= len(equal) - 3, f"{np.sum(equal)} {len(equal)}"

    def test_patch_machine_subsystem_1(self):
        self._test_patch_machine_subsystem(self.file_1)

    def test_patch_machine_subsystem_2(self):
        self._test_patch_machine_subsystem(self.file_2)

    def test_patch_machine_subsystem_3(self):
        self._test_patch_machine_subsystem(self.file_3)

    def test_patch_machine_subsystem_4(self):
        self._test_patch_machine_subsystem(self.file_4)

    @staticmethod
    def _test_files(files: list[Path]):

        c = 0
        i = 0
        for i, f in enumerate(files):
            try:
                TestPatchPEFile._test_patch_none(f)
            except AssertionError as e:
                if ENABLE_UNITTEST_LOGGING:
                    print(f"{f.name} _test_patch_none {e}")
                c += 1
            try:
                TestPatchPEFile._test_patch_same(f)
            except AssertionError as e:
                if ENABLE_UNITTEST_LOGGING:
                    print(f"{f.name} _test_patch_same {e}")
                c += 1
            try:
                TestPatchPEFile._test_patch_machine(f)
            except AssertionError as e:
                if ENABLE_UNITTEST_LOGGING:
                    print(f"{f.name} _test_patch_machine {e}")
                c += 1
            try:
                TestPatchPEFile._test_patch_subsystem(f)
            except AssertionError as e:
                if ENABLE_UNITTEST_LOGGING:
                    print(f"{f.name} _test_patch_subsystem {e}")
                c += 1

        if ENABLE_UNITTEST_LOGGING:
            print(f"Tested: {i * 4}. Failed: {c}")

    @unittest.skipIf(not Path("./tmp/original").exists(), "Skipping TestPatchPEFile.test_original.")
    def test_original(self):
        self._test_files(sorted(Path("./tmp/original").iterdir()))

    @unittest.skipIf(not Path("./tmp/packed").exists(), "Skipping TestPatchPEFile.test_original.")
    def test_packed(self):
        self._test_files(sorted(Path("./tmp/packed").iterdir()))

    @unittest.skipIf(not Path("./tmp/unpacked").exists(), "Skipping TestPatchPEFile.test_original.")
    def test_unpacked(self):
        self._test_files(sorted(Path("./tmp/unpacked").iterdir()))
