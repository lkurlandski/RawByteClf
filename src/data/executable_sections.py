"""
Identify the regions of a PE executable that are executable.
"""

from collections import Counter, namedtuple
from enum import IntFlag, auto
from itertools import islice
import os
from pathlib import Path
from pprint import pformat, pprint
import sys
import tempfile
import time
from typing import Literal, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

try:
    import lief
    import pefile
except (ModuleNotFoundError, ImportError):
    print("lief and pefile are not available. Binary analysis is disabled but you can still use caches and their readers.")
from tqdm import tqdm

from src.data.utils import get_data_from_archives


SectionSummary = namedtuple("SectionSummary", ("offset", "size", "is_executable"))
Boundaries = list[tuple[int, int]]
Toolkit = Literal["lief", "pefile"]


IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_CNT_CODE    = 0x00000020


def set_pefile_flags():
    global IMAGE_SCN_MEM_EXECUTE
    global IMAGE_SCN_CNT_CODE

    for char, code in pefile.section_characteristics:
        if char == "IMAGE_SCN_MEM_EXECUTE":
            IMAGE_SCN_MEM_EXECUTE = code
        if char == "IMAGE_SCN_CNT_CODE":
            IMAGE_SCN_CNT_CODE = code


class ExitCode(IntFlag):
    SUCCESS                     = auto()
    COULD_NOT_PARSE             = auto()
    NO_SECTIONS_FOUND           = auto()
    NO_NONEMPTY_SECTIONS_FOUND  = auto()
    NO_EXECUTABLE_SECTION_FOUND = auto()
    SECTION_OVER_FILE_BOUNDARY  = auto()
    SECTION_OVER_NEXT_SECTION   = auto()
    SECTION_EMPTY               = auto()
    SECTION_INVALID             = auto()


class GetExecutableSectionBounds:
    """
    Extract section boundaries of a PE file that are executable.

    Arguments:
     (Toolkit): The toolkit ("lief" or "pefile") to analysze the binary with.
      The values returned are identicail regardless of the toolkit used. The
      primary difference is the fact that lief is ~50x faster.

    Raises:
     (FileNotFoundError): If the input file does not exist.
     (ValueError): If an invalid `toolkit` is provided.
     (ValueError): If something really wonky happens when computing the bounds.

    Returns:
     (Boundaries): A list of (upper, lower) tuples indicating the boundaries of
      the PE file that are marked as executable or containing code. Note that
      the lower bound is inclusive whereas the upper is exclusive (for slicing).
     (ExitCode): Flag indicating the issues (if any) encountered during analysis.
    
    Usage:
     >>> bounds, error = GetExecutableSectionBounds(file)(toolkit)
    """

    def __init__(self, file: Optional[str] = None, content: Optional[bytes] = None) -> None:
        if file is not None:
            if not os.path.exists(file):
                raise FileNotFoundError(file)
        self.file = file
        self.content = content
        self.length = os.path.getsize(self.file) if self.file is not None else len(self.content)

        if self.file is not None and self.content is not None:
            raise ValueError("Only one of `file` or `content` can be provided.")

    def __call__(self, toolkit: Toolkit) -> tuple[Boundaries, ExitCode]:
        if self.file is None:
            self.file = tempfile.NamedTemporaryFile(delete=False).name
            with open(self.file, "wb") as fp:
                fp.write(self.content)

        try:
            if toolkit == "lief":
                return self._get_boundaries_lief()
            if toolkit == "pefile":
                return self._get_boundaries_pefile()
        except BaseException:
            if self.content is not None:
                os.unlink(self.file)

        raise ValueError(f"Invalid: {toolkit=}")

    def _get_boundaries_lief(self) -> tuple[Boundaries, ExitCode]:
        binary = lief.parse(self.file)
        if binary is None:
            return [], ExitCode.COULD_NOT_PARSE

        summaries = self._get_summaries_lief(binary)
        if not summaries:
            return [], ExitCode.NO_SECTIONS_FOUND

        summaries = [s for s in summaries if s.size > 0]
        if not summaries:
            return [], ExitCode.NO_NONEMPTY_SECTIONS_FOUND

        if not any(summary.is_executable for summary in summaries):
            return [], ExitCode.NO_EXECUTABLE_SECTION_FOUND

        return self._analyze_section_summaries(summaries)

    def _get_boundaries_pefile(self) -> tuple[Boundaries, ExitCode]:
        try:
            binary = pefile.PE(self.file)
        except pefile.PEFormatError:
            return [], ExitCode.COULD_NOT_PARSE

        summaries = self._get_summaries_pefile(binary)
        if not summaries:
            return [], ExitCode.NO_SECTIONS_FOUND

        summaries = [s for s in summaries if s.size > 0]
        if not summaries:
            return [], ExitCode.NO_NONEMPTY_SECTIONS_FOUND

        if not any(summary.is_executable for summary in summaries):
            return [], ExitCode.NO_EXECUTABLE_SECTION_FOUND

        return self._analyze_section_summaries(summaries)

    @staticmethod
    def _get_summaries_lief(binary: lief.PE.Binary) -> list[SectionSummary]:
        summaries: list[SectionSummary] = []
        for section in binary.sections:
            offset = section.offset
            size = section.size
            is_executable = GetExecutableSectionBounds._is_executable_section_lief(section)
            summary = SectionSummary(offset, size, is_executable)
            summaries.append(summary)
        return summaries

    @staticmethod
    def _get_summaries_pefile(binary: pefile.PE) -> list[SectionSummary]:
        summaries: list[SectionSummary] = []
        for section in binary.sections:
            offset = section.PointerToRawData
            size = section.SizeOfRawData
            is_executable = GetExecutableSectionBounds._is_executable_section_pefile(section)
            summary = SectionSummary(offset, size, is_executable)
            summaries.append(summary)
        return summaries

    @staticmethod
    def _is_executable_section_lief(section: lief.PE.Section) -> bool:
        for c in section.characteristics_lists:
            c = str(c)
            if "MEM_EXECUTE" in c:
                return True
            if "CNT_CODE" in c:
                return True
        return False

    @staticmethod
    def _is_executable_section_pefile(section: pefile.SectionStructure) -> bool:
        characteristics = section.Characteristics
        if characteristics & IMAGE_SCN_MEM_EXECUTE:
            return True
        if characteristics & IMAGE_SCN_CNT_CODE:
            return True
        return False

    def _analyze_section_summaries(self, summaries: list[SectionSummary]) -> tuple[Boundaries, ExitCode]:
        exit_code = ExitCode(0)
        boundary: Boundaries = []
        for prv, cur, nxt in zip([None] + summaries[:-1], summaries, summaries[1:] + [None]):
            if not cur.is_executable:
                continue

            (lower, upper), code = self._get_section_bounds(prv, cur, nxt, self.length)
            exit_code = exit_code | code

            # Do not add empty or invalid sections to the list.
            if ExitCode.SECTION_EMPTY & code or ExitCode.SECTION_INVALID & code:
                continue

            boundary.append((lower, upper))

        if exit_code == ExitCode(0):
            exit_code = ExitCode.SUCCESS

        return boundary, exit_code

    @staticmethod
    def _get_section_bounds(
        prv: Optional[SectionSummary],
        cur: SectionSummary,
        nxt: Optional[SectionSummary],
        length: int,
    ) -> tuple[tuple[int, int], ExitCode]:
        code = ExitCode(0)
        lower = cur.offset
        upper = cur.offset + cur.size

        if upper > length:
            upper = length
            code = code | ExitCode.SECTION_OVER_FILE_BOUNDARY

        if nxt is not None and upper > nxt.offset:
            upper = nxt.offset
            code = code | ExitCode.SECTION_OVER_NEXT_SECTION

        if lower == upper:
            code = code | ExitCode.SECTION_EMPTY

        if lower > upper:
            code = code | ExitCode.SECTION_INVALID

        return (lower, upper), code


def test(toolkit: Toolkit):
    total = 10000

    archives = sorted(Path("/media/lk3591/easystore/datasets/Sorel/ghidra/archived").rglob("*.zip"))
    files_and_contents = islice(get_data_from_archives(archives), total)

    results = {}
    for file, contents in files_and_contents:
        stem = file.strip().split(".")[0]
        extractor = GetExecutableSectionBounds(content=contents)
        bounds, error = extractor(toolkit)
        results[stem] = (bounds, error)
        print(stem, error, bounds)

    return

    t_f = time.time()

    t_d = t_f - t_i
    bounds = [b for b, _ in results.values()]
    print(f"bounds={pformat(bounds)}")
    errors = dict(Counter([e.name for _, e in results.values()]))
    print(f"errors={pformat(errors)}")
    print(f"runtime={round(t_d)} seconds")
    print(f"time/sample={round(t_d / total, 5)} seconds")


def main():
    lief.logging.disable()
    test("lief")
    # test("pefile")


if __name__ == "__main__":
    main()
