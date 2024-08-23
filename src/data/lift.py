"""
Lift binaries into disassembly and decompiled representations.

rm -rf /media/lk3591/easystore/datasets/Sorel/decompiled/*
rm -rf /media/lk3591/easystore/datasets/Sorel/disassembled/*
rm -rf /media/lk3591/easystore/datasets/Sorel/ghidra/*
"""

from argparse import ArgumentParser
import gc
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pprint
import re
import subprocess
import sys
import tempfile
import zlib

from tqdm import tqdm


p_ghi = None
p_bin = None
p_dis = None
p_dec = None


def run_ghidra(
    p_location: str | Path,
    p_input: str | Path,
    p_disassembled: str | Path,
    p_decompiled: str | Path,
    recursive: bool = False,
):
    """
    Raises:
        subprocess.CalledProcessError
        subprocess.TimeoutExpired
    """

    args = [
        "analyzeHeadless",
        str(p_location), "lift",
        "-import", str(p_input),
        "-scriptPath", "./ghidra_scripts",
        "-postScript", "disassembler.py", str(p_disassembled),
        "-postScript", "decompiler.py", str(p_decompiled),
    ]
    if recursive:
        args.insert(3, "-recursive")
        timeout = None
    else:
        timeout = 30

    subprocess.run(args, check=True, capture_output=True, timeout=timeout)


def run_ghidra_safe(*args, **kwds) -> int:
    try:
        run_ghidra(*args, **kwds)
    except subprocess.CalledProcessError as err:
        return 3
    except subprocess.TimeoutExpired as err:
        return 2
    except Exception as err:
        return 1

    return 0



def run_ghidra_parallel(f: str | Path) -> int:
    s = os.path.basename(f)[0:2]
    return run_ghidra_safe(
        p_location=p_ghi / s,
        p_input=p_bin / s,
        p_disassembled=p_dis / s,
        p_decompiled=p_dec / s,
        recursive=False,
    )


def main():

    global p_ghi, p_bin, p_dis, p_dec

    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ignore_complete", action="store_true")
    args = parser.parse_args()

    print(args)

    root = args.root

    p_ghi = root / "ghidra"
    p_bin = root / "binaries"
    p_dis = root / "disassembled"
    p_dec = root / "decompiled"

    for p in (p_ghi, p_bin, p_dis, p_dec):
        for h in tuple(hex(i)[2:] for i in range(256)):
            h = "0" + h if len(h) == 1 else h
            (p / h).mkdir(exist_ok=True, parents=True)

    files = list(p_bin.rglob("*.exe"))
    print(f"Found {len(files)} binaries to lift.")

    if args.ignore_complete:
        complete = set(f.stem for f in p_dis.rglob("*.asm"))
        complete &= set(f.stem for f in p_dec.rglob("*.c"))
        print(f"Found {len(complete)} ASM/C files already complete.")
    else:
        complete = set()

    files = sorted([str(f) for f in files if f.stem not in complete])
    print(f"Will proceed with lifting {len(files)} files.")

    gc.collect()

    # FIXME
    files = files[0:32]

    if args.num_workers < 2:
        for f in tqdm(files):
            print(f)
            s = run_ghidra_parallel(f)
            if s != 0:
                print(os.path.basename(f), s)
        return

    with mp.Pool(args.num_workers) as pool:
        status = pool.map(run_ghidra_parallel, files)

    print("Errors:")
    for f, s in zip(iterable, status):
        if s != 0:
            print(os.path.basename(f), s)


if __name__ == "__main__":
    main()
