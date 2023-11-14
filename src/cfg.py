"""
Global configurations.
"""

from pathlib import Path


SPECIALS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "cls_token": "<cls>",
}

BR = "|" + "-" * 88 + "|"

INPUT_PATH = Path("./input")
OUTPUT_PATH = Path("./output")

TMP_DIR = Path("/scratch.local") if Path("/scratch.local").exists() else Path(".")
