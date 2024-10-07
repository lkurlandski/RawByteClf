"""
Test the ESP endpoints and record their outputs. Run this program with a lot of memory.
"""

from argparse import ArgumentParser
from functools import partial
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat, pprint
from random import shuffle, seed
import sys
import time
from typing import Callable

import numpy as np
from tqdm import tqdm

#pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#pylint: enable=wrong-import-position

from src.utils import print_context
from src.data.loaders_core import (
    Materials,
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
)

seed(0)


NUM_WORKERS = None
SUPPRESS    = True
DEBUG       = True


def round_2(number: float | int) -> float:
    return round(number, 2)


def analyze(materials: Materials, binary: bool = False) -> dict[str, float]:

    def f(t: float, b: float) -> float:
        if b == 0:
            return float("nan")
        return round(100 * t / b)

    n_tr = len(materials.files["tr"])
    n_vl = len(materials.files["vl"])
    n_to = n_tr + n_vl

    info = {
        "n_to": n_to,
        "n_tr": n_tr,
        "n_vl": n_vl,
        "p_tr": f(n_tr, n_to),
        "p_vl": f(n_vl, n_to),
    }

    if not binary:
        return info

    n_tr_mal = np.sum(materials.labels["tr"] == materials.label2id["mal"])
    n_tr_ben = np.sum(materials.labels["tr"] == materials.label2id["ben"])
    n_vl_mal = np.sum(materials.labels["vl"] == materials.label2id["mal"])
    n_vl_ben = np.sum(materials.labels["vl"] == materials.label2id["ben"])
    info.update({
        "p_tr_mal": f(n_tr_mal, n_tr),
        "p_tr_ben": f(n_tr_ben, n_tr),
        "p_vl_mal": f(n_vl_mal, n_vl),
        "p_vl_ben": f(n_vl_ben, n_vl),
    })

    return info


def get_materials_and_log_info(kwds: dict, outdir: Path, get_materials: Callable) -> bool:
    outfile = outdir / f"{os.getpid()}.jsonl"

    if not outdir.exists:
        raise FileNotFoundError(outdir)

    with print_context(suppress=SUPPRESS):
        materials, error = None, None
        try:
            materials = get_materials(**kwds)
        except Exception as err:
            error = err

    result = analyze(materials, True) if materials is not None else str(error)
    d = {"kwds": kwds, "results": result}
    s = json.dumps(d)
    with open(outfile, "a") as fp:
        fp.write(s + "\n")

    if DEBUG:
        status = "success" if materials is not None else "failure"
        print(f"Processed ({status}) {list(kwds.values())} --> {outfile}")

    return materials is not None


def det():
    lift_level = "raw"
    ts_size    = 0.00

    outdir = Path("./output/fuzz/det/")
    outdir.mkdir(exist_ok=True, parents=True)
    for f in outdir.iterdir():
        f.unlink()

    ratios_pre_split = list(map(round_2, np.arange(0.00, 1.00,  0.05).tolist()))
    ratios_pos_split = list(map(round_2, np.arange(0.00, 1.00,  0.05).tolist()))
    tr_sizes         = list(map(round_2, np.arange(0.95, 0.50, -0.05).tolist()))
    vl_sizes         = list(map(round_2, np.arange(0.05, 0.55,  0.05).tolist()))
    sizes            = list(zip(tr_sizes, vl_sizes))

    iterable = []
    for (tr_size, vl_size), r_pre, r_pos in product(sizes, ratios_pre_split, ratios_pos_split):
        kwds = {
            "lift_level": lift_level,
            "tr_size": tr_size,
            "vl_size": vl_size,
            "ts_size": ts_size,
            "ratio_pre_split": r_pre,
            "ratio_pos_split": r_pos,
        }
        iterable.append(kwds)
    shuffle(iterable)
    if DEBUG:
        iterable = iterable[0:16]
        print(f"iterable={pformat(iterable)}")

    runner = partial(
        get_materials_and_log_info,
        get_materials=get_materials_esp_det,
        outdir=outdir,
    )

    print(f"Running get_materials_esp_det with {len(iterable)} times.")
    if NUM_WORKERS is not None and NUM_WORKERS > 1:
        with mp.Pool(NUM_WORKERS) as pool:
            overall = pool.map(runner, iterable)
    else:
        overall = [runner(i) for i in tqdm(iterable)]

    outfile = outdir / "result.jsonl"
    with open(outfile, "w") as fp:
        for f in outdir.iterdir():
            if f.name == outfile.name:
                continue
            t = f.read_text()
            fp.write(t)
            f.unlink()

    print(f"get_materials_esp_det finished without error on {sum(overall)} / {len(overall)} runs.")


def fam():
    ...


def beh():
    ...


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["det", "fam", "beh"])
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    global NUM_WORKERS
    NUM_WORKERS = args.num_workers

    if args.task == "det":
        det()
    if args.task == "fam":
        fam()
    if args.task == "beh":
        beh()


if __name__ == "__main__":
    main()
