"""

"""

import os
import random
from pathlib import Path
import sys
import unittest

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import rglob
from src.data.function_boundaries import *


ASM = (
"""
undefined __stdcall FUN_004013d0(void)
CODE        \t000007d0    \t004013d0    \t89 00                                               \tMOV dword ptr [EAX],EAX
CODE        \t000007d2    \t004013d2    \t89 40 04                                            \tMOV dword ptr [EAX + 0x4],EAX
CODE        \t000007d5    \t004013d5    \tc3                                                  \tRET

int __stdcall MessageBoxA(HWND hWnd, LPCSTR lpText, LPCSTR lpCaption, UINT uType)
CODE        \t0000063c    \t0040123c    \tff 25 ac 31 48 00                                   \tJMP dword ptr [0x004831ac]    
"""
)


class TestDisFileToExeFuncBounds(unittest.TestCase):

    def test_snippet_typical(self):
        correct = [("000007d0", "000007d6"), ("0000063c", "00000642")]
        correct = np.array([(int(a, 16), int(b, 16)) for a, b in correct])
        bounds = dis_file_to_exe_func_bounds(ASM)
        assert (bounds == correct).all(), f"bounds: {bounds.tolist()}, correct: {correct.tolist()}"

    def test_snippet_atypical(self):
        raise NotImplementedError("Find a snippet that has a line that does not have five parts.")

    def test_snippets(self):
        root_a = Path("/home/lk3591/Documents/datasets")
        root_b = Path("/shared/rc/admalware")
        archives = []
        for dnm in ["Assemblage", "BODMAS", "Sorel", "Windows"]:
            p = root_a / dnm
            p = root_b / dnm if not p.exists() else p
            p = p / "ghidra/disassembled"
            if not p.exists():
                continue
            archives.extend(list(rglob(p, "*.zip")))
        archives = random.sample(archives, min(16, len(archives)))
        bounds = dis_files_archives_to_exe_func_bounds_map(archives, 1)
        assert isinstance(bounds, dict)
        assert isinstance(next(iter(bounds.keys())), str)
        assert isinstance(next(iter(bounds.values())), np.ndarray)
