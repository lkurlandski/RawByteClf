"""
Global configurations.
"""

from enum import Enum
from pathlib import Path

from src.enums import System

BR = "|" + "-" * 88 + "|"

INPUT_PATH = Path("./input")
OUTPUT_PATH = Path("./output")
TOKENIZERS_OUTPUT_PATH = OUTPUT_PATH / "tokenizers"
TMP_DIR = Path("/scratch.local") if Path("/scratch.local").exists() else Path(".")
SYSTEM = System(Path("./config/.system").read_text().strip())
