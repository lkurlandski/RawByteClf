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
import random
from statistics import mean
import sys
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
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


SUPPRESS = True
DEBUG    = False
SHUFFLE  = True


def round_2(number: float | int) -> float:
    return round(number, 2)


def concatenate_files(files: list[Path], outfile: Path, unlink: bool = False) -> None:
    with open(outfile, "w") as fp:
        for f in files:
            if f.as_posix() == outfile.as_posix():
                continue
            t = f.read_text()
            fp.write(t)
            if unlink:
                f.unlink()


def fromoutput_extract_errors(f: Path) -> list[dict]:
    with open(f) as fp:
        data = [json.loads(l) for l in fp]
    data = [d for d in data if isinstance(d["results"], str)]
    return data


def fromoutput_to_dataframe(f: Path) -> pd.DataFrame:
    with open(f) as fp:
        data = [json.loads(l) for l in fp]
    data = [d for d in data if not isinstance(d["results"], str)]
    columns = list(data[0]["kwds"].keys()) + list(data[0]["results"].keys())
    flat = [d["kwds"] | d["results"] for d in data]
    return pd.DataFrame(flat)


def analyze(materials: Materials) -> dict[str, float]:

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

    if len(materials.id2label) > 2:
        info["dist_tr"] = dict(materials.dist_tr)
        info["dist_vl"] = dict(materials.dist_vl)
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


def get_materials_and_log_info(kwds: dict, outdir: Path, get_materials: Callable) -> tuple[bool, float]:
    t_i = time.time()
    outfile = outdir / f"{os.getpid()}.jsonl"

    if not outdir.exists:
        raise FileNotFoundError(outdir)

    with print_context(suppress=SUPPRESS):
        materials, error = None, None
        try:
            materials = get_materials(**kwds)
        except Exception as err:  # pylint: disable=broad-exception-caught
            error = err

    result = analyze(materials) if materials is not None else str(error)
    d = {"kwds": kwds, "results": result}
    s = json.dumps(d)
    with open(outfile, "a") as fp:
        fp.write(s + "\n")

    if DEBUG:
        status = "success" if materials is not None else "failure"
        print(f"Processed ({status}) {list(kwds.values())} --> {outfile}")

    return materials is not None, time.time() - t_i


def run(
    iterable: list,
    outdir: Path,
    get_materials: Callable,
    num_workers: Optional[int] = None,
    subset: Optional[int] = None,
) -> None:
    if SHUFFLE:
        random.shuffle(iterable)
    iterable = iterable[0:subset]

    outdir.mkdir(exist_ok=True, parents=True)
    for f in outdir.iterdir():
        f.unlink()

    runner = partial(
        get_materials_and_log_info,
        get_materials=get_materials,
        outdir=outdir,
    )

    print(f"Running {get_materials.__name__} {len(iterable)} times with {num_workers} workers.")
    if num_workers is not None and num_workers > 1:
        with mp.Pool(num_workers) as pool:
            overall = pool.map(runner, iterable)
    else:
        overall = [runner(i) for i in tqdm(iterable)]

    status = [b for b, t in overall]
    times  = [t for b, t in overall]
    print(f"Finished without error on {sum(status)} / {len(overall)} runs.")
    print(f"Mean runtime: {round(mean(times[1:]))} seconds.")

    concatenate_files(list(outdir.iterdir()), outdir / "result.jsonl", unlink=True)
    return overall


def get_det_iterable() -> list[dict[str, Any]]:
    lift_level     = "raw"
    lift_level_ddp = "dec"
    ts_size        = 0.00

    # ratios_pre_split = list(map(round_2, np.arange(0.00, 1.00,  0.05).tolist()))
    # ratios_pos_split = list(map(round_2, np.arange(0.00, 1.00,  0.05).tolist()))
    ratios_pre_split = list(map(round_2, np.arange(0.00, 1.00,  0.10).tolist()))
    ratios_pos_split = list(map(round_2, np.arange(0.00, 0.60,  0.10).tolist()))
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
            "lift_level_ddp": lift_level_ddp,
        }
        iterable.append(kwds)

    return iterable


def _get_clf_iterable() -> list[dict[str, Any]]:

    lift_level     = "raw"
    lift_level_ddp = "dec"
    ts_size        = 0.00

    min_freqs            = [10, 20, 30, 40, 50]
    max_imbalance_ratios = [2, 5, 10, 25, 50]
    tr_sizes             = [0.90, 0.85, 0.80, 0.75, 0.70]
    vl_sizes             = [0.10, 0.15, 0.20, 0.25, 0.30]
    sizes                = list(zip(tr_sizes, vl_sizes))

    iterable = []
    for (tr_size, vl_size), min_freq, max_imbalance_ratio in product(sizes, min_freqs, max_imbalance_ratios):
        kwds = {
            "lift_level": lift_level,
            "tr_size": tr_size,
            "vl_size": vl_size,
            "ts_size": ts_size,
            "lift_level_ddp": lift_level_ddp,
            "min_freq": min_freq,
            "max_imbalance_ratio": max_imbalance_ratio,
        }
        iterable.append(kwds)

    return iterable


def get_fam_iterable() -> list[dict[str, Any]]:
    return _get_clf_iterable()


def get_beh_iterable() -> list[dict[str, Any]]:
    return _get_clf_iterable()


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["det", "fam", "beh"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("./output/fuzz"))
    args = parser.parse_args()

    random.seed(args.seed)

    if args.task == "det":
        run(get_det_iterable(), args.root / "det", get_materials_esp_det, args.num_workers, args.subset)
    if args.task == "fam":
        run(get_fam_iterable(), args.root / "fam", get_materials_esp_fam, args.num_workers, args.subset)
    if args.task == "beh":
        run(get_beh_iterable(), args.root / "beh", get_materials_esp_beh, args.num_workers, args.subset)


if __name__ == "__main__":
    main()
