"""
Lift binaries into disassembly and decompiled representations.

Run with gnu-parallel, e.g.,

    parallel --bar -j $JOBS
     'python src/data/lift.py --root /home/lk3591/Documents/datasets/Sorel/ --filter_idx {1} > ./logs/lift_{1}.txt 2>&1' \
     ::: $(printf "%02x\n" {4095..0})

"""

from argparse import ArgumentParser
import gc
import os
from pathlib import Path
from pprint import pprint
import shutil
import subprocess
import sys
import time
from typing import Optional
import zlib

from tqdm import tqdm


PROCESSOR = "x86:LE:32:default"
TIMEOUT_ANALYSIS = 180
TIMEOUT_DISASSEMBLE = 180
TIMEOUT_DECOMPILE = 180


def get_file_type(path: Path) -> dict[str, str]:
    args = ["find", str(path), "-type", "l", "-exec", "file", "-L", "{}", "+"]
    result = subprocess.run(args, check=True, capture_output=True)
    lines = result.stdout.decode()
    out = {}
    for line in lines.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            file, result = line.split(": ")
            out[file] = result
        except Exception:
            print(f"{line=}")
    return out


def hexes(n: int) -> tuple[str]:
    arr = []
    for i in range(16 ** n):
        h = hex(i)[2:]
        p = "0" * (n - len(h))
        arr.append(p + h)
    return arr


def run_ghidra(
    p_location: str,
    p_input: str,
    p_disassembled: str,
    p_decompiled: str,
    p_log: str,
):
    """
    Raises:
        subprocess.CalledProcessError
        subprocess.TimeoutExpired
    """

    # FIXME: decompiler disabled
    # TODO: figure out a way to kill the decompiler's subprocesses.

    n = sum(1 for _ in Path(p_input).iterdir())
    # timeout = n * (TIMEOUT_ANALYSIS + TIMEOUT_DISASSEMBLE + TIMEOUT_DECOMPILE)
    timeout = n * (TIMEOUT_ANALYSIS + TIMEOUT_DISASSEMBLE)
    timeout += 900  # 15 minutes additional buffer
    timeout = None  # TODO: temporary to prevent unhandled subprocesses.

    args = [
        "analyzeHeadless",
        p_location,
        "lift",
        "-recursive",
        "-log", p_log, 
        "-processor", PROCESSOR,
        "-analysisTimeoutPerFile", str(TIMEOUT_ANALYSIS),
        "-import", p_input,
        "-scriptPath", "./ghidra_scripts",
        "-postScript", "disassembler.py", p_disassembled,
        # "-postScript", "decompiler.py", p_decompiled, str(TIMEOUT_DECOMPILE),
    ]

    subprocess.run(args, check=True, capture_output=True, timeout=timeout)


def main():

    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--filter_idx", type=str, required=True)
    args = parser.parse_args()

    p_log = Path("./logs") / f"lift_ghidra_{args.filter_idx}"

    # Define the directories
    root = args.root
    p_ghi = root / "ghidra" / args.filter_idx
    p_sym = root / "symBinaries" / args.filter_idx
    p_bin = root / "binaries" / args.filter_idx[0:2]
    p_dis = root / "disassembled" / args.filter_idx[0:2]
    p_dec = root / "decompiled" / args.filter_idx[0:2]

    # Set up the directories
    if not p_bin.exists():
        raise FileNotFoundError()
    shutil.rmtree(p_ghi, ignore_errors=True)
    shutil.rmtree(p_sym, ignore_errors=True)
    p_ghi.mkdir(exist_ok=True, parents=True)
    p_sym.mkdir(exist_ok=True, parents=True)
    p_dis.mkdir(exist_ok=True, parents=True)
    p_dec.mkdir(exist_ok=True, parents=True)

    # Create symbolic links
    files = [
        f for f in p_bin.iterdir()
        if f.stem[0:len(args.filter_idx)] == args.filter_idx
    ]
    for f in tqdm(files, desc="Creating symlinks..."):
        symlink = p_sym / f.name
        symlink.symlink_to(f)

    types = get_file_type(p_sym)
    types = {Path(f).stem: s for f, s in types.items()}
    for f in p_sym.iterdir():
        s = types[f.stem]
        if "PE32" in s and "PE32+" not in s:
            continue
        # print(f"Unlinking f={f.stem} because it is {s}")
        f.unlink()

    del f
    del s
    del symlink
    del files
    del types
    gc.collect()

    print("Starting Ghidra!")
    t_0 = time.time()
    try:
        run_ghidra(
            p_ghi.as_posix(),
            p_sym.as_posix(),
            p_dis.as_posix(),
            p_dec.as_posix(),
            p_log.as_posix(),
        )
    except subprocess.CalledProcessError:
        print(f"{subprocess.CalledProcessError}.")
    except subprocess.TimeoutExpired:
        print(f"{subprocess.TimeoutExpired}")

    n = sum(1 for _ in p_sym.iterdir())
    t = int(round(time.time() - t_0))
    for f in p_sym.iterdir():
        f.unlink()
    print(f"Finished {n} files in {t} seconds.")


if __name__ == "__main__":
    main()
