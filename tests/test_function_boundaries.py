"""

"""

import os
import sys
import unittest

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    def test_1(self):
        correct = [("000007d0", "000007d6"), ("0000063c", "00000642")]
        correct = [(int(a, 16), int(b, 16)) for a, b in correct]
        bounds = dis_file_to_exe_func_bounds(ASM)
        assert bounds == correct, f"bounds: {bounds}, correct: {correct}"
