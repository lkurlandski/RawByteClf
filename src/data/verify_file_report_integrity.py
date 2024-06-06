"""
Verify that VirusTotal reports and malware are correctly organized.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import shutil
import subprocess
import sys
from typing import Literal, Optional

from tqdm import tqdm


def survey_possible_output_of_file_command(files: list[Path]) -> set[str]:
    out = set()
    for f in tqdm(files):
        try:
            result = subprocess.run(["file", f], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if "ERROR: error reading (Invalid argument)" in str(e.output):
                continue
            print(e.output)
            raise e

        t = result.stdout.decode("utf-8").split(" ")[1]
        out.add(t)
    return out


def get_file_type(f: str) -> Literal["PE", "ELF", "MACHO", "IGNORE"]:
    try:
        result = subprocess.run(["file", f], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if "ERROR: error reading (Invalid argument)" in str(e.output):
            return "IGNORE"
        print(e.output)
        raise e

    t = result.stdout.decode("utf-8").split(" ")[1]
    if t in ("PE32", "MS-DOS", "PE32+"):
        return "PE"
    if t in ("ELF",):
        return "ELF"
    if t in ():
        return "MACHO"

    # stdout has the following format:
      # path/to/file: ELF|PE32

    for s in "PE", "ELF", "MACHO":
        if s in result.stdout.decode("utf-8"):
            return s

    return None


def get_sha256_from_report(f: Path) -> str:
    with open(f, "r") as fp:
        d = json.load(fp)
    return d["data"]["attributes"]["sha256"]


def rename_sha_based_files(files_and_new_shas: list[tuple[Path, str]], dry_run: bool = True) -> None:
    """Rename a series of incorrectly named files with the correct SHA.

    The files are intended to all be in the same directory. The renaming
      will only change the stem of the file and leave the suffix untouched.

    For each SHA_OLD.json report that needs to be renamed, we:
      - create a tmp_SHA_OLD.json file
      - move the tmp_SHA_OLD.json to SHA_NEW.json
      - delete the SHA_OLD.json file

    This is nessecary because the renaming of one report could
      collide with another, thereby deleting a report accidently.
    """

    pbar = tqdm(files_and_new_shas)
    for f, s in pbar:
        f_tmp = f.with_name("tmp_" + f.name)
        pbar.set_description(f"Copying: {f.name} --> {f_tmp.name}")
        if not dry_run:
            shutil.copy2(f, f_tmp)

    pbar = tqdm(files_and_new_shas)
    for f, s in pbar:
        f_new = f.with_stem(s)
        f_tmp = f.with_name("tmp_" + f.name)
        pbar.set_description(f"Renaming {f_tmp.name} --> {f_new.name}")
        if not dry_run:
            f_tmp.rename(f_new)

    pbar = tqdm(files_and_new_shas)
    for f, s in pbar:
        pbar.set_description(f"Unlinking: {f.name}")
        if not dry_run:
            f.unlink(missing_ok=True)

    print(f"Renamed {len(files_and_new_shas)} reports.")


def rename_reports(files: list[Path], dry_run: bool = True) -> None:

    if any(f.name.startswith(".") for f in files):
        raise ValueError("Hidden files detected.")

    incorrect = []
    pbar = tqdm(files)
    for f in pbar:
        pbar.set_description(f"Scanning: {f.name}")
        if f.suffix != ".json":
            raise ValueError(f"Expected a JSON file. Got {f.name=}")

        try:
            s = get_sha256_from_report(f)
        except Exception:
            print(f"{f.name=}")
            raise

        if s != f.stem:
            incorrect.append((f, s))

    rename_sha_based_files(incorrect, dry_run)


def get_file_sha(f: Path) -> str:
    with open(f, "rb") as fp:
        b = fp.read()
    return sha256(b).hexdigest()


def verify_binaries(
    files: list[Path],
    platform: Optional[Literal["PE", "ELF", "MACHO"]] = None,
    dry_run: bool = True,
) -> None:

    incorrect = []
    if platform is not None:
        with mp.Pool(16) as pool:
            file_types = pool.map(get_file_type, files)
        incorrect = [(f, t) for f, t in zip(files, file_types) if t not in ("IGNORE", platform)]

    for f, t in incorrect:
        print(f"{f} {t}")
    print(f"{len(incorrect)=} / {len(files)=} bad files.")
    if incorrect:
        sys.exit(1)

    with mp.Pool(16) as pool:
        shas = pool.map(get_file_sha, files)
    incorrect = [(f, s) for f, s in zip(files, shas) if f.stem != s]

    rename_sha_based_files(incorrect, dry_run)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir_reports", required=False, type=Path)
    parser.add_argument("--dir_binaries", required=False, type=Path)
    parser.add_argument("--no_dry_run", action="store_true")
    parser.add_argument("--platform", required=False, default=None)
    args = parser.parse_args()

    pprint(args)

    if args.dir_reports is not None:
        rename_reports(sorted(args.dir_reports.iterdir()), not args.no_dry_run)

    if args.dir_binaries is not None:
        verify_binaries(sorted(args.dir_binaries.iterdir()), args.platform, not args.no_dry_run)


if __name__ == "__main__":
    main()
