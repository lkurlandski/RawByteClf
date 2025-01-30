"""
Generate the materials cache files.
"""

from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Callable

#pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#pylint: enable=wrong-import-position

from src.enums import LiftLevel, Task
from src.data.loaders_core import (
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
)


parser = ArgumentParser()
parser.add_argument("--task", type=Task)
parser.add_argument("--lift_level", type=LiftLevel)
parser.add_argument("--lift_level_ddp", type=str, default="dec")
args = parser.parse_args()

args.lift_level_ddp = LiftLevel(args.lift_level_ddp) if args.lift_level_ddp != "none" else None


if args.task == Task.CLM:
    get_materials = get_materials_esp_clm
elif args.task == Task.MLM:
    get_materials = get_materials_esp_mlm
elif args.task == Task.DET:
    get_materials = get_materials_esp_det
elif args.task == Task.FAM:
    get_materials = get_materials_esp_fam
elif args.task == Task.BEH:
    get_materials = get_materials_esp_beh
else:
    raise RuntimeError()


materials = get_materials(args.lift_level, lift_level_ddp=args.lift_level_ddp)
print(materials)

sys.exit(0)
