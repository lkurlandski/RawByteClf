"""
Compare the time required to lift binaries using different methods.
"""

from argparse import ArgumentParser
import asyncio
from collections import Counter, defaultdict
import functools
from itertools import islice, repeat
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pprint
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Literal, Optional
import warnings

# Importing lief here, then importing it again in src.data.ember
# or in src.data.executable_sections will cause a Segmentation fault (lief==0.12.0).
# import lief
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: enable=wrong-import-position

from src.enums import LiftLevel
from src.utils import rglob, batched
from src.data.cfg import ASSEMBLAGE_PATH, BODMAS_PATH, SOREL_PATH, WINDOWS_PATH
from src.data.ember import PEFeatureExtractor
from src.data.executable_sections import get_executable_section_bounds
from src.data.utils import get_data_from_archives, read_binary_files_asynch


random.seed(0)


ROOT = Path("./tmp/time-lifting")
PATH_BINARIES = ROOT / "binaries"
PATH_LOCATION = ROOT / "ghidra"
PATH_LOGFILES = ROOT / "logs"
PATH_DISASSEMBLED = ROOT / "disassembled"
PATH_DECOMPILED   = ROOT / "decompiled"

TIMEOUT_PER_FILE_ANALYSIS = 300
TIMEOUT_PER_FILE_DISASSEMBLY = 60
TIMEOUT_PER_FILE_DECOMPILATION = 300
TIMEOUT_PER_FUNC_DISASSEMBLY = 30
TIMEOUT_PER_FUNC_DECOMPILATION = 60


def run_tab(files: list[Path], num_workers: int = 1):

    import lief  # pylint: disable=import-outside-toplevel

    if lief.__version__[0:len("0.12.0")] != "0.12.0":
        raise RuntimeError(f"Expected lief version 0.12.0, but got {lief.__version__}. Please install the correct version.")

    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    future = read_binary_files_asynch(files, disable_tqdm=False)
    data = loop.run_until_complete(future)

    extractor = PEFeatureExtractor()
    func = extractor.feature_vector
    if num_workers > 1:
        with mp.Pool(20) as pool:
            out = pool.map(func, data)
    else:
        out = list(map(func, data))

    end = time.perf_counter()

    report = Counter([o.shape for o in out])
    print(f"Status: {report}")

    print(f"Processed {len(files)} files in {end - start:.3f}s")


def run_raw(files: list[Path], num_workers: int = 1):

    import lief  # pylint: disable=import-outside-toplevel

    if lief.__version__[0:len("0.15.1")] != "0.15.1":
        raise RuntimeError(f"Expected lief version 0.15.1, but got {lief.__version__}. Please install the correct version.")

    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    future = read_binary_files_asynch(files, disable_tqdm=False)
    data = loop.run_until_complete(future)

    func = functools.partial(get_executable_section_bounds, None)
    if num_workers > 1:
        with mp.Pool(20) as pool:
            out = pool.map(func, data)
    else:
        out = list(map(func, data))

    end = time.perf_counter()

    report = Counter([o[1].name for o in out])
    print(f"Status: {report}")

    print(f"Processed {len(files)} files in {end - start:.3f}s")


def run_dis(files: list[Path], num_workers: int = 1, max_cpu: int = 1):
    return run_dis_or_dec("dis", files, num_workers, max_cpu)


def run_dec(files: list[Path], num_workers: int = 1, max_cpu: int = 1):
    return run_dis_or_dec("dec", files, num_workers, max_cpu)


def _run_dis_or_dec(dis_or_dec: Literal["dis", "dec"], files: list[Path], location: Path, logfile: Path, max_cpu: int = 1):

    print(f"{os.getpid()}: Running {dis_or_dec} on {len(files)} files with {max_cpu} max CPU cores.")

    shutil.rmtree(location, ignore_errors=True)
    location.mkdir(exist_ok=True)

    # Create a temporary directory to hold the files (symlinks) to use batch processing
    directory = Path(tempfile.TemporaryDirectory().name)
    directory.mkdir()
    for f in files:
        os.symlink(f, directory / f.name)

    args = [
        "analyzeHeadless",
        str(location), "PROJECT",
        "-overwrite", "-recursive",
        "-processor", "x86:LE:32:default",
        "-loader", "PeLoader",
        "-import", str(directory),
        "-max-cpu", str(max_cpu),
        "-analysisTimeoutPerFile", str(TIMEOUT_PER_FILE_ANALYSIS),
    ]
    if dis_or_dec == "dis":
        args += [
            "-preScript", "SetAnalysisOptionsForDisassembly.java",
            "-postScript", "Disassembler.java", str(PATH_DISASSEMBLED), str(TIMEOUT_PER_FILE_DISASSEMBLY), str(TIMEOUT_PER_FUNC_DISASSEMBLY),
        ]
    if dis_or_dec == "dec":
        args += [
            # "-preScript", "SetAnalysisOptionsForDecompilation.java",
            "-postScript", "Decompiler.java", str(PATH_DECOMPILED), str(TIMEOUT_PER_FILE_DECOMPILATION), str(TIMEOUT_PER_FUNC_DECOMPILATION),
        ]

    if dis_or_dec == "dis":
        timeout = len(files) * (TIMEOUT_PER_FILE_ANALYSIS + TIMEOUT_PER_FILE_DISASSEMBLY + 60)
    if dis_or_dec == "dec":
        timeout = len(files) * (TIMEOUT_PER_FILE_ANALYSIS + TIMEOUT_PER_FILE_DECOMPILATION + 60)

    try:
        print(f"{os.getpid()}: Executing subprocess with timeout {timeout} and command {' '.join(args)}")
        with open(logfile, "w") as fp:
            subprocess.run(args, check=True, stdout=fp, stderr=fp, timeout=timeout)
    except subprocess.CalledProcessError:
        print("analyzeHeadless command: ", " ".join(args))
        raise
    except subprocess.TimeoutExpired:
        print(f"analyzeHeadless timed out after {timeout} seconds for {len(files)} files.")

    shutil.rmtree(directory, ignore_errors=True)


def run_dis_or_dec(dis_or_dec: Literal["dis", "dec"], files: list[Path], num_workers: int = 1, max_cpu: int = 1):

    if dis_or_dec not in ("dis", "dec"):
        raise ValueError(f"Invalid dis_or_dec value: {dis_or_dec}. Must be 'dis' or 'dec'.")

    print(f"{os.getpid()}: Running {dis_or_dec} on {len(files)} files with {num_workers} workers and {max_cpu} max CPU cores.")

    start = time.perf_counter()

    shutil.rmtree(PATH_LOCATION, ignore_errors=True)
    PATH_LOCATION.mkdir()

    if dis_or_dec == "dis":
        shutil.rmtree(PATH_DISASSEMBLED, ignore_errors=True)
        PATH_DISASSEMBLED.mkdir(exist_ok=True)
    if dis_or_dec == "dec":
        shutil.rmtree(PATH_DECOMPILED, ignore_errors=True)
        PATH_DECOMPILED.mkdir(exist_ok=True)

    PATH_LOGFILES.mkdir(exist_ok=True)

    files: list[list[Path]] = list(batched(files, int(math.ceil(len(files) / num_workers))))
    locations: list[Path]   = [PATH_LOCATION / f"{i}" for i in range(len(files))]
    logfiles: list[Path]    = [PATH_LOGFILES / f"{dis_or_dec}-{i}.log" for i in range(len(files))]

    func = functools.partial(_run_dis_or_dec, max_cpu=max_cpu)
    iterable = list(zip(repeat(dis_or_dec, len(files)), files, locations, logfiles))
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            out = pool.starmap(func, iterable)  # pylint: disable=unused-variable
    else:
        out = [func(*args) for args in iterable]  # pylint: disable=unused-variable

    end = time.perf_counter()

    report = defaultdict(int)
    for logfile in logfiles:
        text = logfile.read_text(encoding="utf-8")
        report["analysis-timeout"] += text.count("Analysis timed out")
        report["script-timeout"] += text.count("run: finished (timeout)")
        report["script-crash"] += text.count("run: finished (crash)")
        report["script-success"] += text.count("run: finished (success)")
    report = dict(report)
    print(f"Status: {report}")

    print(f"Processed {sum(len(f) for f in files)} files in {end - start:.3f}s")


def prepare(num_files: int, dnm: Optional[Literal["ass", "bod", "sor", "win"]] = None) -> None:

    print(f"Preparing {num_files} file from {dnm if dnm is not None else 'all'} dataset{'s' if dnm is None else ''}.")

    shutil.rmtree(PATH_BINARIES, ignore_errors=True)
    PATH_BINARIES.mkdir()

    with open(f"data/{dnm}/dis/digests.json") as fp:
        d = json.load(fp)
    valid = set(d.keys())
    print(f"Found {len(valid)} valid digests.")

    archives = []
    for d in map(Path, rglob("./data/", "nop")):
        if not d.is_dir():
            continue
        if dnm is not None and d.parent.name != dnm:
            continue
        archives.extend(list(map(Path, rglob(d, "*.zip"))))
    if not archives:
        raise ValueError("No archives found in ./data/")
    archives.sort()
    print(f"Found {len(archives)} valid archives: ")
    for a in archives:
        print(f"  - {a}")

    iterable = get_data_from_archives(archives)
    i = 0
    skipped = 0
    for n, b in tqdm(iterable, desc="Reading archives...", total=num_files, leave=False):
        if i == num_files:
            break
        if n.split(".")[0] not in valid:
            skipped += 1
            continue
        f: Path = PATH_BINARIES / n
        f.write_bytes(b)
        i += 1
    print(f"Skipped {skipped} files that were not among in the valid digests.")
    print(f"Last sha: {n}")

    if i != num_files:
        warnings.warn("Not enough files found in the archives.")
    print(f"Prepared {i} files in {PATH_BINARIES}")


def check_and_set_max_mem(new: Optional[int] = None) -> None:
    path = Path(shutil.which("analyzeHeadless")).resolve()

    cur = None
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("MAXMEM"):
                cur = int(line.split("=")[1].strip().replace("G", ""))
                break
    if cur is None:
        raise RuntimeError("Could not find MAXMEM in analyzeHeadless script.")

    if new is not None and cur != new:
        with open(path, "r") as fp:
            lines = fp.readlines()
        with open(path, "w") as fp:
            for line in lines:
                if line.startswith("MAXMEM"):
                    line = f"MAXMEM={new}G\n"
                fp.write(line)
        print(f"Current max memory (analyzeHeadless): {cur}G")
        cur = new

    print(f"Current max memory for (analyzeHeadless): {cur}G")


def main():
    parser = ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--representation", type=str, choices=["tab", "raw", "dis", "dec"], default=None)
    parser.add_argument("--dnm", type=str, choices=["ass", "bod", "sor", "win"], default=None)
    parser.add_argument("--num_files", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_cpu", type=int, default=1)
    parser.add_argument("--max_mem", type=int, default=None)
    args = parser.parse_args()

    print("Command line arguments:")
    print(f"  - prepare: {args.prepare}")
    print(f"  - representation: {args.representation}")
    print(f"  - dataset: {args.dnm}")
    print(f"  - num_files: {args.num_files}")
    print(f"  - num_workers: {args.num_workers}")
    print(f"  - max_cpu: {args.max_cpu}")
    print(f"  - max_mem: {args.max_mem}")

    ROOT.mkdir(exist_ok=True)

    if args.prepare:
        if args.num_files is None:
            raise ValueError("Please specify the number of files to prepare with --num_files.")
        prepare(args.num_files, args.dnm)
    files: list[Path] = sorted(PATH_BINARIES.iterdir())
    files = [f.resolve() for f in files]
    if not files:
        raise ValueError(f"No files found in {PATH_BINARIES}. Please prepare the data first.")

    if args.representation in ("dis", "dec"):
        check_and_set_max_mem(args.max_mem)

    if args.representation == "tab":
        run_tab(files, args.num_workers)
    if args.representation == "raw":
        run_raw(files, args.num_workers)
    if args.representation == "dis":
        run_dis(files, args.num_workers, args.max_cpu)
    if args.representation == "dec":
        run_dec(files, args.num_workers, args.max_cpu)

    print("FINISHED")
    print("-" * 80)


if __name__ == "__main__":
    main()
