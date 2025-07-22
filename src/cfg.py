"""
Global configurations.
"""

import json
import os
from pathlib import Path

from src.enums import System


BR = "|" + "-" * 88 + "|"


# Determine which configuration file to use
file = Path("./config/config.json")
if not file.exists():
    file = Path("./config/default.json")
if not file.exists():
    raise FileNotFoundError("Configuration file not found. Please create a config.json file in the config directory.")

# Load the configuration from the JSON file
with open(file, "r") as fp:
    config = json.load(fp)

# Set global constants based on the configuration
AVCLASS_EXE = Path(config["AVCLASS_EXE"])
CLARAVY_EXE = Path(config["CLARAVY_EXE"])
DATASETS_PATH = Path(config["DATASETS_PATH"])
OUTPUT_PATH = Path(config["OUTPUT_PATH"])
SYSTEM = System(config["SYSTEM"])
TMP_DIR = Path(config["TMP_DIR"])
TOKENIZERS_OUTPUT_PATH = Path(config["TOKENIZERS_OUTPUT_PATH"])

# Set environment variables based on the configuration
if SYSTEM == System.ARMITAGE:
    os.environ["LMLM_SYNC_ENSEMBLE_MATERIALS"] = "1"
if SYSTEM == System.SPORC:
    os.environ["LMLM_CAN_PRECOPY_ZIPFILES"] = "1"
