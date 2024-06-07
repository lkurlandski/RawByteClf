"""
Globals and configuration for the data module.
"""

from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path
from typing import Literal


DATASETS_PATH = Path("/home/lk3591/Documents/datasets")

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

ELF_CLASSIFICATION_DATASETS = ("malware_bazaar_elf", "virus_share_elf", "virus_total_elf")
ELF_LABEL_CACHE_DIR = Path("./cache") / "labels" / "elf"

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
    "bodmas_pe": BODMAS_PATH / "diec",
    "malware_bazaar_elf": MALWARE_BAZAAR_PATH / "elf" / "diec",
    "sorel_pe": SOREL_PATH / "diec",
    "virus_share_elf": VIRUS_SHARE_PATH / "diec",
    "virus_total_elf": VIRUS_TOTAL_PATH.parent / "diec",
}


def _dataset_to_report_files_and_binaries(
    reports: bool = False,
    binaries: bool = False,
) -> dict[str, Callable[[], Iterable[Path]]]:
    assert (reports or binaries) and not (reports and binaries)

    s = "reports" if reports else "binaries"

    def vt(p: Path, t: Literal["DLL", "ELF", "EXE", "Mach-O"]) -> bool:
        return s in p.as_posix() and t in p.as_posix() and p.is_file()

    # pylint: disable=unnecessary-lambda
    return {
        "bodmas_pe": lambda: (BODMAS_PATH / s).iterdir(),
        "local_pe": lambda: (WINDOWS_PATH / s).iterdir(),
        "local_elf": lambda: (LINUX_PATH / s).iterdir(),
        "local_macho": lambda: (DARWIN_PATH / s).iterdir(),
        "malware_bazaar_elf": lambda: (MALWARE_BAZAAR_PATH / "elf" / s).iterdir(),
        "malware_bazaar_macho": lambda: (MALWARE_BAZAAR_PATH / "macho" / s).iterdir(),
        "sorel_pe": lambda: (SOREL_PATH / s).iterdir(),
        "virus_share_dll": lambda: [],
        "virus_share_elf": lambda: chain.from_iterable(
            (p / s).iterdir() for p in VIRUS_SHARE_ELF_COLLECTION_PATHS.values()
        ),
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
}

SOREL_BUCKET = "sorel-20m"
SOREL_PREFIX = "09-DEC-2020/binaries/"

MALWARE_BAZAAR_URL = "https://mb-api.abuse.ch/api/v1/"

MAX_SHARD_SIZE = "1GB"  # MUST BE AN INTEGER FOLLOWED BY A UNIT (e.g. 1GB, 500MB, 100KB)
