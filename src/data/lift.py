"""
Lift binaries into disassembly and decompiled representations.

Performance:
    lift.py (--jobs=1): 32 samples / 340 seconds = Z samples/second or W seconds/sample
    lift.py (--jobs=4): 90 samples / 340 seconds = Z samples/second or W seconds/sample
    lift.sh (--jobs=1): X samples / Y seconds = Z samples/second or W seconds/sample
    lift.sh (--jobs=4): 864 samples / 340 seconds = Z samples/second or W seconds/sample
"""

from argparse import ArgumentParser
import gc
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

from tqdm import tqdm


timeout = 300
p_ghi = None
p_bin = None
p_dis = None
p_dec = None


def hexes(n: int) -> tuple[str]:
    arr = []
    for i in range(16 ** n):
        h = hex(i)[2:]
        p = "0" * (n - len(h))
        arr.append(p + h)
    return arr


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
        "-analysisTimeoutPerFile", str(timeout),
        "-import", str(p_input),
        "-scriptPath", "./ghidra_scripts",
        "-postScript", "disassembler.py", str(p_disassembled),
        "-postScript", "decompiler.py", str(p_decompiled),
    ]
    if recursive:
        args.insert(3, "-recursive")

    subprocess.run(args, check=True, capture_output=True, timeout=3600)


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
        p_input=p_bin / s / f,
        p_disassembled=p_dis / s,
        p_decompiled=p_dec / s,
        recursive=False,
    )


def run_ghidra_parallel_batched(files: list[str | Path]) -> list[int]:
    print(f"BEG: worker {os.getpid()} processing {len(files)} items.")

    verbose = len(files) > 0 and files[0] == "00"

    status = []
    for i, f in enumerate(files):
        s = run_ghidra_parallel(f)
        status.append(s)
        if verbose and i % 10 == 0: 
            print(f"PRG: Worker {os.getpid()} processing {i} / {len(files)} items.")
    print(f"END: worker {os.getpid()} processing {len(files)} items.")
    return status


def main():

    global p_ghi, p_bin, p_dis, p_dec, timeout

    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=timeout)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--ignore_complete", action="store_true")
    args = parser.parse_args()

    print(args)

    timeout = args.timeout

    root = args.root

    p_ghi = root / "ghidra"
    p_bin = root / "binaries"
    p_dis = root / "disassembled"
    p_dec = root / "decompiled"

    if False:
        shutil.rmtree(p_ghi)
        shutil.rmtree(p_dis)
        shutil.rmtree(p_dec)

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


    # Run single worker.
    if args.jobs is None or args.jobs < 2:
        t_i = time.time()

        for f in tqdm(files):
            s = run_ghidra_parallel(f)
            if s != 0:
                print(os.path.basename(f), s)

        t_f = time.time() 
        print(f"Elapsed time: {t_f - t_i:.2f} seconds.")

        sys.exit(0)


    # Run multiple workers. Each Ghidra process needs its own directory.
    chunks = {h: [] for h in hexes(2)}
    for f in files:
        h = os.path.basename(f)[0:2]
        chunks[h].append(f)
    chunks = tuple(tuple(fs) for fs in chunks.values())
    del files

    gc.collect()


    t_i = time.time()
    with mp.Pool(args.jobs) as pool:
        statuses = pool.map(run_ghidra_parallel_batched, chunks)
    t_f = time.time()

    print(f"Errors:\n{'-' * 88}")
    for files, status in zip(chunks, statuses):
        for f, s in zip(files, status):
            if s != 0:
                print(os.path.basename(f), s)

    print(f"Elapsed time: {t_f - t_i:.2f} seconds.")

if __name__ == "__main__":
    main()
