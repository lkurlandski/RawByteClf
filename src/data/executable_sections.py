"""
Identify the regions of a PE executable that are executable.
"""

from collections import Counter
import os
from pathlib import Path
from pprint import pprint
import sys
import time

try:
    import lief
except ModuleNotFoundError:
    print("pefile is not available. Scripting disabled, but utilities are still accessible.")
from tqdm import tqdm


def get_executable_section_boundaries(file: str) -> tuple[list[tuple[int, int]], int]:
    """
    Get the boundaries of the exectuable functions of a PE file.

    Error Codes:
      - 0: No errors
      - 1: File does not exist
      - 2: Error parsing binary with lief
      - 3: Bounds are outside of PE file
      - 4: No executable sections found
    """

    if not os.path.exists(file):
        return [], 1
    size = os.path.getsize(file)

    binary = lief.parse(file)
    if binary is None:
        return [], 2

    code = 0
    boundaries = []
    for section in binary.sections:

        is_executable = False
        for c in section.characteristics_lists:
            if "MEM_EXECUTE" in str(c):
                is_executable = True
                break

        if is_executable:
            lower, upper = section.offset, section.offset + section.size
            if lower > upper or lower > size or upper > size:
                code = 2
            boundaries.append((lower, upper))

    if not boundaries:
        code = 4

    return boundaries, code


def main():

    lief.logging.disable()

    

    # t_i = time.time()

    # root = Path("/home/lk3591/Documents/datasets/Sorel/binariesLabeled/00/")
    # files = root.rglob("*.exe")
    # total = sum(1 for _ in root.rglob("*.exe"))
    # results = []
    # for i, file in tqdm(enumerate(files), total=total):
    #     boundaries, error = get_executable_section_boundaries(file.as_posix())
    #     if error != 0:
    #         results.append((file.stem, error))

    # t_f = time.time()
    # t_t = t_f - t_i

    # print("Errors:")
    # pprint(results)
    # print("Error Types:")
    # counts = Counter([error for _, error in results])
    # pprint(counts)
    # print(f"Number of files: {i}")
    # print(f"Total ime: {round(t_t)} seconds.")
    # print(f"Time per file: {t_t / i}")


if __name__ == "__main__":
    main()
