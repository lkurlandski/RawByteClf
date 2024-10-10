"""
"""

from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Callable

from tqdm import tqdm

#pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#pylint: enable=wrong-import-position

from src.enums import LiftLevel
from src.utils import print_context
from src.data.loaders_core import (
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
)


parser = ArgumentParser()
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--suppress", action="store_true")
args = parser.parse_args()


GET_MATERIALS = [
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
]


def func(get_materials: Callable, lift_level: LiftLevel):
    print(f"{os.getpid()} running {get_materials.__name__} {lift_level.value}")
    with print_context(args.suppress):
        get_materials(lift_level)


for f in Path("./cache/materials").iterdir():
    f.unlink()


iterable = list(product(GET_MATERIALS, LiftLevel))


if args.num_workers > 1:
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(func, iterable)
else:
    for get_materials, lift_level in tqdm(iterable):
        print(f"{os.getpid()} running {get_materials.__name__} {lift_level}")
        with print_context(args.suppress):
            get_materials(lift_level)
