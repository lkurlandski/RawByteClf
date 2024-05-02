"""
Download the SOREL dataset from s3, run detect-it-easy program, and save outputs to disk.
"""

import asyncio
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterable, Generator
from copy import deepcopy
from functools import singledispatch
import gc
from itertools import islice
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import subprocess
import sys
import time
import tempfile

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import pandas as pd
from tqdm import tqdm

from src.utils import batched
from src.data.prepare_datasets import s3_dataset_generator, s3_dataset_generator_async
from src.data.cfg import (
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta, Decompressor


OUTPUT = Path("/home/lk3591/Documents/datasets/Sorel/diec")
OUTPUT = Path("./diec")  # FIXME: place in correct directory after verifying this works
DOWNLOAD = Path("/tmp/sorel")
DIEC_TIMEOUT = 5
CACHE = Path("./tmp/detect_packing")


DECOMPRESSOR = Decompressor(Decompressor.ZLIB, must_decompress=True)


def analyze_sample(b: bytes, sha: str) -> None:

    def args(mode: str) -> list[str]:
        return ["diec", f"--{mode}scan", "--json", str(file)]

    file = (DOWNLOAD / sha).with_suffix(".exe")
    with open(file, "wb") as fp:
        fp.write(b)

    for mode in ("recursive", "deep", "heuristic"):
        try:
            subprocess.run(
                args(mode),
                stdout=open((OUTPUT / mode / sha).with_suffix(".txt"), "w"),
                timeout=DIEC_TIMEOUT,
                check=True,
                capture_output=False,
            )
        except subprocess.TimeoutExpired:
            print(f"TimeoutExpired: {mode} {sha}")

    file.unlink()


async def analyze_samples_async(files: Iterable[str]) -> None:
    generator = s3_dataset_generator_async(
        files=files,
        num_bytes=sys.maxsize,
        max_length=sys.maxsize,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=2,
        decompress=DECOMPRESSOR,
    )

    async for sample in generator:
        analyze_sample(sample["bytes"], sample["name"])


def analyze_samples(files: Iterable[str]) -> None:

    generator = s3_dataset_generator(
        files=files,
        num_bytes=sys.maxsize,
        max_length=sys.maxsize,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=2,
        decompress=DECOMPRESSOR,
    )

    pbar = tqdm(generator, total=len(files))
    for sample in pbar:
        pbar.set_description(f"Processing: {sample['name']}")
        analyze_sample(sample["bytes"], sample["name"])


async def run_async(files: Iterable[str]):
    await analyze_samples_async(files)


def run(files: Iterable[str]):
    analyze_samples(files)


def main():

    parser = ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--filter_mode", type=int, default=None)
    parser.add_argument("--filter_idx", type=str, default=None)
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--run_async", action="store_true")
    args = parser.parse_args()

    # Prepare the shard files for embarassingly parallel execution
    if args.prepare:
        CACHE.mkdir(exist_ok=True)
        for f in CACHE.iterdir():
            f.unlink()
        files = (s.sha256 for s in stream_sorel_meta() if s.is_malware)
        files = sorted(islice(files, None))
        print(f"{len(files)=}")
        print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

        if args.num_shards:
            shard_size = (len(files) // args.num_shards) + 1
            print(f"{shard_size=}") 
            for shard_idx in range(args.num_shards):
                idx_start = shard_idx * shard_size
                idx_end = (shard_idx + 1) * shard_size
                shard_file = CACHE / f"packingPrep_{shard_idx}.txt"
                with open(shard_file, "w") as fp:
                    for i in range(idx_start, min(idx_end, len(files) - 1)):
                        fp.write(f"{files[i]}\n")
            return

        if args.filter_mode:
            num_filters = 16 ** args.filter_mode
            print(f"{num_filters=}") 
            start = 0
            finish = len(files)
            for filter_idx in range(16 ** args.filter_mode):
                filter_ = hex(filter_idx)[2:]
                filter_ = ("0" * (args.filter_mode - len(filter_))) + filter_
                filter_file = CACHE / f"packingPrep_{filter_}.txt"
                print(str(filter_file))
                for i in range(start, len(files)):
                    if files[i][0:args.filter_mode] != filter_:
                        if i == start:
                            raise RuntimeError()
                        finish = i
                        break

                with open(filter_file, "w") as fp:
                    for f in files[start:finish]:
                        fp.write(f"{f}\n")

                start = finish
                finish = len(files)

            return


    # Get the files for this shard (or all files)
    if isinstance(args.shard_idx, int) or isinstance(args.filter_idx, str):
        file = CACHE / f"packingPrep_{args.shard_idx if args.shard_idx else args.filter_idx}.txt"
        with open(file, "r") as fp:
            files = [l.strip() for l in fp.readlines()]
    else:
        files = (s.sha256 for s in stream_sorel_meta() if s.is_malware)
        files = sorted(islice(files, None))
    print(f"{len(files)=}")
    print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

    DOWNLOAD.mkdir(exist_ok=True)

    t_0 = time.time()
    if args.run_async:
        asyncio.run(run_async(files))
    else:
        run(files)

    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


if __name__ == "__main__":
    main()
