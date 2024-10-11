"""
Prepare raw, disassembled, and decompiled data caches. This is essentially normalization.
"""

from argparse import ArgumentParser
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat
import re
import shutil
import sys
from tempfile import NamedTemporaryFile
import time
from typing import Callable
from zipfile import ZipFile, ZIP_DEFLATED

from tqdm import tqdm
from unidecode import unidecode

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import SYSTEM
from src.enums import LiftLevel, System
from src.utils import rglob
from src.data.utils import get_data_from_archives


if SYSTEM == System.GCCIS:
    ROOT = Path("/media/lk3591/easystore/datasets")
elif SYSTEM == System.SPORC:
    ROOT = Path("/shared/rc/admalware/")
elif SYSTEM == System.ARMITAGE:
    ROOT = Path("/home/lk3591/Documents/datasets")
else:
    raise NotImplementedError()

ROOTS = {
    "sorel_pe": ROOT / "Sorel",
    "bodmas_pe": ROOT / "BODMAS",
    "assemblage_pe": ROOT / "Assemblage",
    "windows_pe": ROOT / "Windows",
}
IN = "ghidra"
OUT = "processed"

NUM_WORKERS: int = None


def _run(f: Path, f_out: Path, func: Callable[[bytes, str], bytes], disable_tqdm: bool = True) -> tuple[int, int]:
    s_org = 0
    s_new = 0
    with ZipFile(f, "r") as zp, ZipFile(f_out, "w") as zp_out:
        for n in tqdm(zp.namelist(), leave=False, disable=disable_tqdm):
            b = zp.read(n)
            b_out = func(b, n.split(".")[0])
            zp_out.writestr(n, b_out, ZIP_DEFLATED, 9)

            s_org += len(b)
            s_new += len(b_out)

    return s_org, s_new


def run(path: Path, out: Path, func: Callable[[bytes, str], bytes]) -> None:
    files = sorted(map(Path, rglob(path, "*.zip")))
    outfiles = [out / f.name for f in files]

    if NUM_WORKERS is not None and NUM_WORKERS > 1:
        with mp.Pool(NUM_WORKERS) as pool:
            info = pool.starmap(_run, list(zip(files, outfiles, [func] * len(files))))
    else:
        info = [_run(f, f_out, func, disable_tqdm=False) for f, f_out in tqdm(list(zip(files, outfiles)))]

    for f, f_out, (s_org, s_new) in zip(files, outfiles, info):
        print(f"Processed {f} -> {f_out} ({s_org} -> {s_new})")

    delta = sum((s_org - s_new for s_org, s_new in info))
    print(f"Total Delta: {delta / 1e9:.1f}GB")


def raw_func(b: bytes, n: str, d: dict[str, list[tuple[int, int]]]) -> bytes:
    if n in d:
        bounds = d[n]
    elif n + ".exe" in d:
        bounds = d[n + ".exe"]
    else:
        raise KeyError(f"Could not find bounds for {n}")
    b_new =  b"".join([b[l:u] for l, u in bounds])
    return b_new

def raw(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "archived"
    out = ROOTS[dataset] / OUT / "raw"
    out.mkdir(parents=True, exist_ok=True)

    f = ROOTS[dataset] / "executableSections.json"
    with open(f, "r") as fp:
        d = json.load(fp)
    d = {name: data["bounds"] for name, data in d.items()}

    run(path, out, partial(raw_func, d=d))


def dis_func(b: bytes, n: str) -> bytes:  # pylint: disable=unused-argument
    s = b.decode()
    t = []
    for l in s.split("\n"):
        p = l.split("\t")
        if len(p) > 1:
            t.append(p[-1].strip())
    t = "\n".join([unidecode(l) for l in t])
    return t.encode(encoding="ascii")

def dis(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "disassembled"
    out = ROOTS[dataset] / OUT / "dis"
    out.mkdir(parents=True, exist_ok=True)

    run(path, out, dis_func)


def dec_func(b: bytes, n: str) -> bytes:  # pylint: disable=unused-argument
    s = b.decode()
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    t = []
    for l in s.split("\n"):
        t.append(l)
    t = "\n".join([unidecode(l) for l in t])
    return t.encode(encoding="ascii")

def dec(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "decompiled"
    out = ROOTS[dataset] / OUT / "dec"
    out.mkdir(parents=True, exist_ok=True)

    run(path, out, dec_func)


def _purge(file: Path, shas: list[str]) -> int:
    shas = set(shas)
    with NamedTemporaryFile() as file_tmp:
        with ZipFile(file, "r") as zp_in, ZipFile(file_tmp.name, "w") as zp_out:
            count = len(zp_in.namelist())
            for name in zp_in.namelist():
                if name.split(".")[0] not in shas:
                    buffer = zp_in.read(name)
                    zp_out.writestr(name, buffer, ZIP_DEFLATED, 9)
                    count -= 1

        file.unlink()
        shutil.copy2(file_tmp.name, file)

    return count


def purge(dataset: str, lift_level: str, shas: list[str]) -> None:
    shas = sorted(shas)
    path = ROOTS[dataset] / OUT / lift_level
    files = sorted(map(Path, rglob(path, "*.zip")))
    iterable = [(f, [s for s in shas if s.startswith(f.stem)]) for f in files]
    if NUM_WORKERS is not None and NUM_WORKERS > 1:
        with mp.Pool(NUM_WORKERS) as pool:
            counts = pool.starmap(_purge, iterable)
    else:
        counts = []
        for f, s in tqdm(iterable):
            count = _purge(f, s)
            counts.append(count)

    print(f"{dataset} {lift_level}: purged {sum(counts)} files.")


def sync(dataset: str) -> None:
    sha_inter = set()
    sha_union = set()
    for lift_level in ["raw", "dis", "dec"]:
        path = ROOTS[dataset] / OUT / lift_level
        files = sorted(map(Path, rglob(path, "*.zip")))
        shas = [name.split(".")[0] for name, _ in get_data_from_archives(files, contents=False)]
        sha_union.update(shas)
        sha_inter = sha_inter.intersection(shas) if sha_inter else set(shas)

    shas_to_purge = list(sha_union - sha_inter)
    for lift_level in ["raw", "dis", "dec"]:
        purge(dataset, lift_level, shas_to_purge)


def main():
    parser = ArgumentParser()
    parser.add_argument("--process", action="store_true", help="Process the data.")
    parser.add_argument("--purge", action="store_true", help="Purge the data.")
    parser.add_argument("--sync", action="store_true", help="Syncronize the data.")
    parser.add_argument("--dataset", type=str, choices=["sorel_pe", "bodmas_pe", "assemblage_pe", "windows_pe"], required=True)
    parser.add_argument("--lift_level", type=str, default=None, choices=["raw", "dis", "dec"])
    parser.add_argument("--purge_file", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    print(f"args={pformat(args.__dict__)}")

    if not any([args.process, args.purge, args.sync]):
        raise ValueError("No action specified.")
    if sum(bool(a) for a in [args.process, args.purge, args.sync]) != 1:
        raise ValueError("Only one action can be specified.")
    if args.purge and not args.purge_file:
        args.purge_file = ROOTS[args.dataset] / "purge.txt"
        print(f"Using default {args.purge_file=}")
    if not args.lift_level and not args.sync:
        raise ValueError("Lift level must be specified for this action.")

    global NUM_WORKERS
    NUM_WORKERS = args.num_workers

    t_i = time.time()

    if args.process:
        if args.lift_level == "raw":
            raw(args.dataset)
        if args.lift_level == "dis":
            dis(args.dataset)
        if args.lift_level == "dec":
            dec(args.dataset)

    if args.purge:
        purge(args.dataset, args.lift_level, args.purge_file.read_text().splitlines())
    if args.sync:
        sync(args.dataset)

    t_f = time.time()

    print(f"Finished. Time Elpased: {t_f - t_i:.2f}s")


if __name__ == "__main__":
    main()
