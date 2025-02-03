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


LIFT_LEVEL_DDP = LiftLevel.DEC

GET_MATERIALS = [
    get_materials_esp_det,
    get_materials_esp_beh,
    get_materials_esp_fam,
    get_materials_esp_clm,
    get_materials_esp_mlm,
]

LIFT_LEVELS = [
    LiftLevel.RAW,
    LiftLevel.DIS,
    LiftLevel.DEC,
]

LIFT_LEVEL_PAIRS = [
    (LiftLevel.RAW, LiftLevel.DIS),
    (LiftLevel.RAW, LiftLevel.DEC),
    (LiftLevel.DEC, LiftLevel.DIS),
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
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level_1.value} {lift_level_2.value} tr {len(s_1)})")
        if s_2:
            errors_detected = True
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level_1.value} {lift_level_2.value} vl {len(s_2)})")
        if s_3:
            errors_detected = True
            print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level_1.value} {lift_level_2.value} al {len(s_3)})")

        if get_materials in (get_materials_esp_clm, get_materials_esp_mlm):
            continue

        for s in ["tr", "vl"]:
            materials_1 = multimaterials[lift_level_1]
            materials_2 = multimaterials[lift_level_2]
            iterable_1 = zip(
                [af.name.split(".")[0] for af in materials_1.files[s]],
                list(materials_1.labels[s]),
                strict=True,
            )
            iterable_2 = zip(
                [af.name.split(".")[0] for af in materials_2.files[s]],
                list(materials_2.labels[s]),
                strict=True,
            )
            names_mismatch = 0
            label_mismatch = 0
            for (n_1, l_1), (n_2, l_2) in zip(iterable_1, iterable_2, strict=True):
                if n_1 != n_2:
                    if names_mismatch == 0:
                        print(f"Warning: dissimilar files detected! ({get_materials.__name__} {lift_level_1.value} {lift_level_2.value} {s} {n_1} != {n_2}")
                    names_mismatch += 1
                if l_1 != l_2:
                    if label_mismatch == 0:
                        print(f"Warning: dissimilar labels detected! ({get_materials.__name__} {lift_level_1.value} {lift_level_2.value} {s} {l_1} != {l_2}")
                    label_mismatch += 1
            errors_detected = errors_detected or names_mismatch > 0 or label_mismatch > 0


    if not errors_detected:
        print(f"Success: no problems were detected! ({get_materials.__name__})")
