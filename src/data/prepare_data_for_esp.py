"""
Prepare raw, disassembled, and decompiled data caches.
"""

from argparse import ArgumentParser
import multiprocessing as mp
from pathlib import Path
import time
from typing import Callable
from zipfile import ZipFile, ZIP_DEFLATED

from tqdm import tqdm


NUM_WORKERS = 16
ROOT = Path("/media/lk3591/easystore/datasets")
ROOTS = {
    "sorel_pe": ROOT / "Sorel",
    "bodmas_pe": ROOT / "BODMAS",
    "assemblage_pe": ROOT / "Assemblage",
}
IN = "ghidra"
OUT = "processed"


def _run(f: Path, f_out: Path, func: Callable[[bytes, str], bytes]) -> tuple[int, int]:
    s_org = 0
    s_new = 0
    with ZipFile(f, "r") as zp, ZipFile(f_out, "w") as zp_out:
        for n in zp.namelist():
            b = zp.read(n)
            b_out = func(b, n.split(".")[0])
            zp_out.writestr(n, b_out, ZIP_DEFLATED, 9)

            s_org += len(b)
            s_new += len(b_out)

    return s_org, s_new


def run(path: Path, out: Path, func: Callable[[bytes, str], bytes]) -> None:
    files = sorted(path.rglob("*.zip"))
    outfiles = [out / f.name for f in files]

    with mp.Pool(NUM_WORKERS) as pool:
        info = pool.starmap(_run, list(zip(files, outfiles, [func] * len(files))))

    for f, f_out, (s_org, s_new) in zip(files, outfiles, info):
        print(f"Processed {f} -> {f_out} ({s_org} -> {s_new})")

    delta = sum([s_org - s_new for s_org, s_new in info])
    print(f"Total Delta: {delta / 1e9:.1f}GB")


d: dict[str, list[tuple[int, int]]] = dict()  # TODO
def raw_func(b: bytes, n: str) -> bytes:
    return b"".join([b[l:u] for l, u in d[n]])

def raw(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "archived"
    out = ROOTS[dataset] / OUT / "raw"
    out.mkdir(parents=True, exist_ok=True)

    run(path, out, raw_func)


def dis_func(b: bytes, n: str) -> bytes:  # pylint: disable=unused-argument
    s = b.decode()
    s = "\n".join([l.split("\t")[-1] for l in s.split("\n")])
    return s.encode()

def dis(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "disassembled"
    out = ROOTS[dataset] / OUT / "dis"
    out.mkdir(parents=True, exist_ok=True)

    run(path, out, dis_func)


def dec(dataset: str) -> None:

    path = ROOTS[dataset] / IN / "decompiled"
    out = ROOTS[dataset] / OUT / "dec"
    out.mkdir(parents=True, exist_ok=True)

    files = path.rglob("*.zip")

    for f in files:
        f_out = out / f.name
        f.symlink_to(f_out)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["sorel_pe", "bodmas_pe", "assemblage_pe"], required=True)
    parser.add_argument("--lift_level", type=str, choices=["raw", "dis", "dec"], required=True)
    args = parser.parse_args()

    t_i = time.time()

    if args.lift_level == "raw":
        raw(args.dataset)
    if args.lift_level == "dis":
        dis(args.dataset)
    if args.lift_level == "dec":
        dec(args.dataset)

    t_f = time.time()

    print(f"Finished. Time Elpased: {t_f - t_i:.2f}s")


if __name__ == "__main__":
    main()

