"""
Quick running and testing.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from pprint import pformat, pprint
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders_core import (
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
    spacially_bias,
)


RATIOS_PRE = np.arange(0.00, 1.00,  0.05).tolist()
RATIOS_POS = np.arange(0.00, 1.00,  0.05).tolist()
TR_SIZES   = np.arange(0.95, 0.50, -0.05).tolist()
VL_SIZES   = np.arange(0.05, 0.55,  0.05).tolist()
TS_SIZES   = np.repeat(0.00, len(TR_SIZES)).tolist()
SIZES      = list(zip(TR_SIZES, VL_SIZES, TS_SIZES))


def analyze(materials) -> tuple:

    def f(t: float, b: float) -> float:
        if b == 0:
            return float("nan")
        return round(100 * t / b)

    n_tr = len(materials.files["tr"])
    n_vl = len(materials.files["vl"])
    n_to = n_tr + n_vl
    n_tr_mal = np.sum(materials.labels["tr"] == materials.label2id["mal"])
    n_vl_mal = np.sum(materials.labels["vl"] == materials.label2id["mal"])

    return {
        "total": n_tr + n_vl,
        "p_tr": f(n_tr, n_to),
        "p_vl": f(n_vl, n_to),
        "p_tr_mal": f(n_tr_mal, n_tr),
        "p_vl_mal": f(n_vl_mal, n_vl),
    }


outfile = Path("./tmp/ratio.jsonl")
if outfile.exists():
    outfile.unlink()


for ratio_pre in RATIOS_PRE:
    for ratio_pos in RATIOS_POS:
        for tr_size, vl_size, ts_size in SIZES:
            ratio_pre = round(ratio_pre, 2)
            ratio_pos = round(ratio_pos, 2)
            tr_size   = round(tr_size, 2)
            vl_size   = round(vl_size, 2)
            ts_size   = round(ts_size, 2)

            args = ["raw", tr_size, vl_size, ts_size, ratio_pre, ratio_pos]
            try:
                materials = get_materials_esp_det(*args)
                results   = analyze(materials)
            except Exception as err:
                materials = None
                results   = str(err)

            d = {"args": args, "results": results}
            s = json.dumps(d)
            with open(outfile, "a") as fp:
                fp.write(s + "\n")
