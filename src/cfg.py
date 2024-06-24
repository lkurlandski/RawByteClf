"""
Global configurations.
"""

from enum import Enum
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
TOKENIZERS_OUTPUT_PATH = OUTPUT_PATH / "tokenizers"

TMP_DIR = Path("/scratch.local") if Path("/scratch.local").exists() else Path(".")


class System(Enum):
    ARMITAGE = "ARMITAGE"
    RC = "RC"


SYSTEM = System(Path("./config/.system").read_text().strip())
