"""
Prepare for the packing experiments.
"""

from collections import Counter
import hashlib
import random
import os
from pathlib import Path
from pprint import pformat
import subprocess
import sys

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: enable=wrong-import-position

import lief

from src.enums import LiftLevel
from src.utils import print_context
from src.data.detect_packing_sorel import pack, unpack
from src.data.executable_sections import GetExecutableSectionBounds, Boundaries, ExitCode
from src.data.loaders_core import get_materials_esp_det, get_materials_esp_fam, get_materials_esp_beh
from src.data.loaders_hf import generator_from_zipfiles
from src.data.pe_utils import is_dotnet, rearm_disarmed_binary


lief.logging.disable()  # pylint: disable=c-extension-no-member


random.seed(0)


def func(content: bytes) -> tuple[Boundaries, ExitCode, bytes]:
    bounds, ecode = GetExecutableSectionBounds(content=content)()
    exe = b"".join([content[l:u] for l, u in bounds])
    return bounds, ecode, exe


def get_upx_error_message(err: subprocess.CalledProcessError) -> str:
    try:
        # Common format looks like: 'upx: /tmp/tmpg7_1qn4y.bin: NotCompressibleException\n'
        msg: str = err.stderr.decode().strip()
        if msg.startswith("upx: "):
            msg = msg[len("upx: "):]
        if msg.startswith("/tmp/tmp"):
            msg = msg[msg.index(" ") + 1:]
    except Exception:
        msg = str(err.stderr)
    return msg


def main():
    allfiles = set()

    for get_materials in (get_materials_esp_det, get_materials_esp_fam, get_materials_esp_beh):
        with print_context(suppress=True):
            materials = get_materials(LiftLevel.NOP)
        for split, files in materials.files.items():
            allfiles.update(files)
            print(f"{get_materials.__name__}: adding {len(files)} files from the {split} split.")

    allfiles = list(allfiles)
    sorted(allfiles, key=lambda x: x.name)
    print(f"Total unique files: {len(allfiles)}")

    results = {
        "total":    len(allfiles),
        "skipped":  0,
        "rearmed":  0,
        "packed":   0,
        "unpacked": 0,
    }
    errors = Counter()

    generator = generator_from_zipfiles(allfiles, preserve_order=True, use_fast_storage=False)
    for i, d in enumerate(generator):
        name     = d["name"]
        original = d["bytes"]

        print(f"Processing ({i} / {len(allfiles)}) {name} ", end="")

        if is_dotnet(original):
            print("Skippped (.NET)")
            results["skipped"] += 1
            continue

        if hashlib.sha256(original).hexdigest() != name:
            original = rearm_disarmed_binary(original, name)
            results["rearmed"] += 1

        original_bounds, original_code, original_exe = func(original)

        try:
            packed = pack(original, return_bytes=True, errors="raise")
        except subprocess.CalledProcessError as err:
            msg = get_upx_error_message(err)
            print(f"Failed (pack {msg})")
            errors.update([msg])
            continue
        results["packed"] += 1
        packed_bounds, packed_code, packed_exe = func(packed)

        try:
            unpacked = unpack(packed, return_bytes=True, errors="raise")
        except subprocess.CalledProcessError as err:
            msg = get_upx_error_message(err)
            print(f"Failed (unpack {msg})")
            errors.update([msg])
            continue
        results["unpacked"] += 1
        unpacked_bounds, unpacked_code, unpacked_exe = func(unpacked)

        print("Success")

        # print(f"  Original: {original_code} {original_bounds} {len(original_exe)}")
        # print(f"  Packed:   {packed_code} {packed_bounds} {len(packed_exe)}")
        # print(f"  Unpacked: {unpacked_code} {unpacked_bounds} {len(unpacked_exe)}")


    print(f"Results:\n{pformat(results)}")
    print(f"Errors:\n{pformat(dict(errors))}")


if __name__ == "__main__":
    main()
