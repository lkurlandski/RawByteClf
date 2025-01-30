"""
Verify the samples in the caches are valid.
"""

from collections import namedtuple
import os
import sys

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.enums import LiftLevel, TokenizationAlgorithm
from src.data.loaders_core import (
    get_materials_esp_beh,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_clm,
    get_materials_esp_mlm,
    Materials,
)
from src.data.loaders_hf import merge_raw_dis_dec_datasets
from src.learn.train import get_processed_dataset_hf
from src.tokenization.api import get_fast_tokenizer


LIFT_LEVEL_DDP = LiftLevel.DECOMPILED

GET_MATERIALS = [
    get_materials_esp_det,
    get_materials_esp_beh,
    get_materials_esp_fam,
    get_materials_esp_clm,
    get_materials_esp_mlm,
]

LIFT_LEVELS = [
    LiftLevel.RAW,
    LiftLevel.DISASSEMBLED,
    LiftLevel.DECOMPILED,
]

LIFT_LEVEL_PAIRS = [
    (LiftLevel.RAW, LiftLevel.DISASSEMBLED),
    (LiftLevel.RAW, LiftLevel.DECOMPILED),
    (LiftLevel.DECOMPILED, LiftLevel.DISASSEMBLED),
]

def get_files(materials: Materials, split: str) -> set[str]:
    return set(af.name.split(".")[0] for af in materials.files[split])

for get_materials in GET_MATERIALS:

    errors_detected = False

    multimaterials = {}
    for lift_level in LIFT_LEVELS:
        materials = get_materials(lift_level, lift_level_ddp=LIFT_LEVEL_DDP)
        multimaterials[lift_level] = materials

    multifiles = {}
    for lift_level in LIFT_LEVELS:
        files = {}
        for split in ("tr", "vl"):
            files[split] = get_files(multimaterials[lift_level], split)
        multifiles[lift_level] = files

    for lift_level in LIFT_LEVELS:
        s = multifiles[lift_level]["tr"].intersection(multifiles[lift_level]["vl"])
        if s:
            errors_detected = True
            print(f"Warning: train-test leakage detected! ({get_materials.__name__} {lift_level.value} {len(s)})")

    for lift_level_1, lift_level_2 in LIFT_LEVEL_PAIRS:
        s_1 = multifiles[lift_level_1]["tr"].symmetric_difference(multifiles[lift_level_2]["tr"])
        s_2 = multifiles[lift_level_1]["vl"].symmetric_difference(multifiles[lift_level_2]["vl"])
        s_3 = (multifiles[lift_level_1]["tr"] | multifiles[lift_level_1]["vl"]).symmetric_difference(
            multifiles[lift_level_2]["tr"] | multifiles[lift_level_2]["vl"])
        if s_1:
            errors_detected = True
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level.value} tr {len(s_1)})")
        if s_2:
            errors_detected = True
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level.value} vl {len(s_2)})")
        if s_3:
            errors_detected = True
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level.value} al {len(s_3)})")

    if not errors_detected:
        print(f"Success: no problems were detected! ({get_materials.__name__})")
