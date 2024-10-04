"""
Compute md5 digests from files.
"""

from argparse import ArgumentParser
from hashlib import md5
import json
from multiprocessing import Pool
import os
from pathlib import Path
from pprint import pformat
import sys
import time
from typing import Optional

from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.utils import rglob
from src.data.utils import get_data_from_archives


def compute_digest(b: bytes) -> str:
    m = md5()
    m.update(b)
    h = m.hexdigest()
    return h


def get_digests_from_zipfile(file: Path) -> dict[str, str]:
    print(f"{os.getpid()=} computing digests on {file.name=}")
    return {n.split(".")[0] : compute_digest(b) for n, b in get_data_from_archives([file])}


def get_digests_from_zipfiles(files: list[Path], num_workers: Optional[int] = None) -> dict[str, str]:
    if num_workers is None or num_workers < 2:
        d = {}
        for file in tqdm(files):
            d.update(get_digests_from_zipfile(file))
        return d

    with Pool(num_workers) as pool:
        ds = pool.map(get_digests_from_zipfile, files)
    d = {}
    for m in ds:
        d.update(m)
    return d


def main():

    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    print(f"args={pformat(args.__dict__)}")

    if args.input.is_dir():
        archives = sorted(map(Path, rglob(args.input, "*.zip")))
    else:
        archives = [args.input]
    print(f"{len(archives)=}")

    t_i = time.time()

    d = get_digests_from_zipfiles(archives, args.num_workers)
    with open(args.output, "w") as fp:
        json.dump(d, fp, sort_keys=True, indent=4)

    t_f = time.time()

    print(f"Finished in {round(t_f - t_i)} seconds.")


if __name__ == "__main__":
    main()
