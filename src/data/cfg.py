"""
Globals and configuration for the data module.
"""

from collections.abc import Callable, Iterable
from enum import Enum
from itertools import chain
import os
from pathlib import Path
import sys
from typing import Literal

from src.cfg import SYSTEM
from src.enums import DatasetName, LiftLevel, System


if SYSTEM == System.ARMITAGE:
    DATASETS_PATH = Path("/home/lk3591/Documents/datasets")
elif SYSTEM == System.GCCIS:
    DATASETS_PATH = Path("/media/lk3591/easystore/datasets/")
elif SYSTEM == System.SPORC:
    DATASETS_PATH = Path("/shared/rc/admalware")

ASSEMBLAGE_PATH = DATASETS_PATH / "Assemblage"
BODMAS_PATH = DATASETS_PATH / "BODMAS"
MALWARE_BAZAAR_PATH = DATASETS_PATH / "MalwareBazaar"
SOREL_PATH = DATASETS_PATH / "Sorel"
VIRUS_SHARE_PATH = DATASETS_PATH / "VirusShare"
VIRUS_TOTAL_PATH = DATASETS_PATH / "VirusTotal" / "extracted"

DARWIN_PATH = DATASETS_PATH / "Darwin"
LINUX_PATH = DATASETS_PATH / "Linux"
WINDOWS_PATH = DATASETS_PATH / "Windows"

BODMAS_LABELS_FILE = BODMAS_PATH / "bodmas_metadata.csv"
BODMAS_DIST_FILE = BODMAS_PATH / "distribution.json"
MALWARE_BAZAAR_FILE_LISTS = {
    "malware_bazaar_dll": MALWARE_BAZAAR_PATH / "dll.txt",
    "malware_bazaar_elf": MALWARE_BAZAAR_PATH / "elf.txt",
    "malware_bazaar_exe": MALWARE_BAZAAR_PATH / "exe.txt",
    "malware_bazaar_macho": MALWARE_BAZAAR_PATH / "macho.txt",
}
VIRUS_SHARE_ELF_COLLECTION_PATHS = {
    "VirusShare_ELF_20140617": VIRUS_SHARE_PATH / "VirusShare_ELF_20140617",
    "VirusShare_ELF_20200405": VIRUS_SHARE_PATH / "VirusShare_ELF_20200405",
    "VirusShare_ELF_20190212": VIRUS_SHARE_PATH / "VirusShare_ELF_20190212",
    # "VirusShare_Linux_20160715": VIRUS_SHARE_PATH / "VirusShare_Linux_20160715",  # THIS IS NOT ELF-only malware
}

SOREL_META_CSV = SOREL_PATH / "meta.csv"
SOREL_LABEL_CACHE_DIR = SOREL_PATH / "labels"
SOREL_CLARAVY_CACHE = SOREL_PATH / "claravy_cache.txt"
SOREL_AVCLASS_CACHE = SOREL_PATH / "avclass_cache.txt"
SOREL_AVCLASS_FAMILY_CACHE = SOREL_PATH / "avclass_family_cache.txt"

ELF_CLASSIFICATION_DATASETS = ("malware_bazaar_elf", "virus_share_elf", "virus_total_elf")
ELF_LABEL_CACHE_DIR = Path("./cache") / "labels" / "elf"

TIMESTAMPS_FILES = {
    DatasetName.ASSEMBLAGE: ASSEMBLAGE_PATH / "timestamps.json",
    DatasetName.BODMAS:     BODMAS_PATH     / "timestamps.json",
    DatasetName.SOREL:      SOREL_PATH      / "timestamps.json",
    DatasetName.WINDOWS:    WINDOWS_PATH    / "timestamps.json",
}

FUNCTION_BOUNDARIES_FILES = {
    DatasetName.ASSEMBLAGE: ASSEMBLAGE_PATH / "function_boundaries.npz",
    DatasetName.BODMAS:     BODMAS_PATH     / "function_boundaries.npz",
    DatasetName.SOREL:      SOREL_PATH      / "function_boundaries.npz",
    DatasetName.WINDOWS:    WINDOWS_PATH    / "function_boundaries.npz",
}


DIGESTS_FILES = {
    dnm: {
        lift_level: Path(f"./data/{dnm.value}/{lift_level.value}") / "digests.json"
        for lift_level in (LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC)
    }
    for dnm in DatasetName
}

DATASET_NAMES = [
    "bodmas_pe",
    "local_pe",
    "local_elf",
    "local_macho",
    "malware_bazaar_elf",
    "malware_bazaar_macho",
    "sorel_pe",
    "virus_share_dll",
    "virus_share_elf",
    "virus_share_exe",
    "virus_share_macho",
    "virus_total_dll",
    "virus_total_elf",
    "virus_total_exe",
    "virus_total_macho",
]

PACKING_ROOTS = {
    "assemblage_pe": ASSEMBLAGE_PATH / "diec",
    "windows_pe": WINDOWS_PATH / "diec",
    "bodmas_pe": BODMAS_PATH / "diec",
    "malware_bazaar_elf": MALWARE_BAZAAR_PATH / "elf" / "diec",
    "sorel_pe": SOREL_PATH / "diec",
    "virus_share_elf": VIRUS_SHARE_PATH / "diec",
    "virus_total_elf": VIRUS_TOTAL_PATH.parent / "diec",
}


def _dataset_to_report_files_and_binaries(
    reports: bool = False,
    binaries: bool = False,
    disassembled: bool = False,
    decompiled: bool = False,
) -> dict[str, Callable[[], Iterable[Path]]]:
    assert sum([reports, binaries, disassembled, decompiled]) == 1

    if reports:
        s = "reports"
        ext = "json"
    elif binaries:
        s = "binaries"
        ext = "exe"
    elif disassembled:
        s = "disassembled"
        ext = "asm"
    elif decompiled:
        s = "decompiled"
        ext = "c"


    def vt(p: Path, t: Literal["DLL", "ELF", "EXE", "Mach-O"]) -> bool:
        return s in p.as_posix() and t in p.as_posix() and p.is_file()


    # pylint: disable=unnecessary-lambda
    return {
        "assemblage_pe": lambda: (p for p in (ASSEMBLAGE_PATH / s).iterdir()),
        "windows_pe": lambda: (p for p in (WINDOWS_PATH / s).iterdir()),
        "bodmas_pe": lambda: (BODMAS_PATH / s).iterdir(),
        "local_pe": lambda: (WINDOWS_PATH / s).iterdir(),
        "local_elf": lambda: (LINUX_PATH / s).iterdir(),
        "local_macho": lambda: (DARWIN_PATH / s).iterdir(),
        "malware_bazaar_elf": lambda: (MALWARE_BAZAAR_PATH / "elf" / s).iterdir(),
        "malware_bazaar_macho": lambda: (MALWARE_BAZAAR_PATH / "macho" / s).iterdir(),
        "sorel_pe": lambda: (SOREL_PATH / s).rglob(f"*.{ext}"),
        "virus_share_dll": lambda: [],
        "virus_share_elf": lambda: chain.from_iterable((p / s).iterdir() for p in VIRUS_SHARE_ELF_COLLECTION_PATHS.values()),
        "virus_share_exe": lambda: [],
        "virus_share_macho": lambda: [],
        "virus_total_dll": lambda: (p for p in VIRUS_TOTAL_PATH.rglob("*") if vt(p, "DLL")),
        "virus_total_elf": lambda: (p for p in VIRUS_TOTAL_PATH.rglob("*") if vt(p, "ELF")),
        "virus_total_exe": lambda: (p for p in VIRUS_TOTAL_PATH.rglob("*") if vt(p, "EXE")),
        "virus_total_macho": lambda: (p for p in VIRUS_TOTAL_PATH.rglob("*") if vt(p, "Mach-O")),
    }
    # pylint: enable=unnecessary-lambda


# {"reports" | "binaries" : {dataset_name: [Path]}}
DATASET_TO_FILES: dict[str, dict[str, Callable[[], Iterable[Path]]]] = {
    "reports": _dataset_to_report_files_and_binaries(reports=True),
    "binaries": _dataset_to_report_files_and_binaries(binaries=True),
    "disassembled": _dataset_to_report_files_and_binaries(disassembled=True),
    "decompiled": _dataset_to_report_files_and_binaries(decompiled=True),
}

SOREL_BUCKET = "sorel-20m"
SOREL_PREFIX = "09-DEC-2020/binaries/"

MALWARE_BAZAAR_URL = "https://mb-api.abuse.ch/api/v1/"

MAX_SHARD_SIZE = "1GB"  # MUST BE AN INTEGER FOLLOWED BY A UNIT (e.g. 1GB, 500MB, 100KB)

VALID_TIMESTAMP_RANGES = {
    DatasetName.ASSEMBLAGE : (0, 1713153600), # VALID=(01-01-1970 - 04-14-2024) -- PRESENT=(09-13-2012 - 10-06-2023)
    DatasetName.BODMAS     : (0, 1601524800), # VALID=(01-01-1970 - 10-01-2020) -- PRESENT=(08-29-2019 - 09-30-2020)
    DatasetName.SOREL      : (0, 1554955200), # VALID=(01-01-1970 - 04-11-2019) -- PRESENT=(01-01-1970 - 02-07-2106)
    DatasetName.WINDOWS    : (0, 1672549200), # VALID=(01-01-1970 - 01-01-2023) -- PRESENT=(01-01-1970 - 02-04-2106)
}

LIFT_LEVEL_EXTENSIONS = {
    LiftLevel.DEC: ".c",
    LiftLevel.DIS: ".asm",
    LiftLevel.RAW: ".exe",
    LiftLevel.NOP: ".exe",
}
