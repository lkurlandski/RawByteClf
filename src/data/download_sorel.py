"""
Download the SOREL dataset from s3 and store to disk.
"""

import asyncio
from argparse import ArgumentParser
from copy import deepcopy
import gc
from itertools import islice
import os
from pathlib import Path
import sys
import time

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from tqdm import tqdm

from src.data.prepare_datasets import s3_dataset_generator, s3_dataset_generator_async
from src.data.cfg import (
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta, Decompressor
from src.data.detect_packing_sorel import PackingMap
from src.data.loaders_core import filter_packed_files


DECOMPRESSOR = Decompressor(Decompressor.ZLIB, must_decompress=True)
SUFFIX = "" if DECOMPRESSOR.alg == Decompressor.NONE else ".exe"


async def download_samples_async(files, output_root: Path, num_bytes: int, max_length: int, errors: int):
    async for sample in s3_dataset_generator_async(
        files=files,
        num_bytes=num_bytes,
        max_length=max_length,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=errors,
        decompress=DECOMPRESSOR,
    ):
        outfile = (output_root / sample["name"][0:2] / sample["name"]).with_suffix(SUFFIX)
        with open(outfile, "wb") as fp:
            fp.write(sample["bytes"])


async def run_async(files, output_root: Path, num_bytes: int, max_length: int, errors: int):
    await download_samples_async(files, output_root, num_bytes, max_length, errors)


def download_samples(files, output_root: Path, num_bytes: int, max_length: int, errors: int):
    generator = s3_dataset_generator(
        files=files,
        num_bytes=num_bytes,
        max_length=max_length,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=errors,
        decompress=DECOMPRESSOR,
    )

    for sample in tqdm(generator, total=len(files)):
        outfile = (output_root / sample["name"][0:2] / sample["name"]).with_suffix(SUFFIX)
        with open(outfile, "wb") as fp:
            fp.write(sample["bytes"])


def run(files, output_root: Path, num_bytes: int, max_length: int, errors: int):
    download_samples(files, output_root, num_bytes, max_length, errors)


def main():

    parser = ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=sys.maxsize)
    parser.add_argument("--num_bytes", type=int, default=sys.maxsize)
    parser.add_argument("--max_length", type=int, default=sys.maxsize)
    parser.add_argument("--packing_protocol", default="any", choices=["yes", "no", "unk", "any"])
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument(
        "--errors",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "How to handle specific errors. "
            "0 raises exceptions. "
            "1 returns empty samples when errors are encountered. "
            "2 skips errd samples entirely."
        ),
    )
    parser.add_argument("--run_async", action="store_true")
    parser.add_argument("--include_shas", type=str, default=None)
    parser.add_argument("--exclude_shas", type=str, default=None)
    args = parser.parse_args()
    print(f"{args=}")


    if args.include_shas is not None:
        with open(args.include_shas, "r") as fp:
            files = [f.strip() for f in fp]
    else:
        files = [s.sha256 for s in tqdm(stream_sorel_meta(), total=10000000, desc="Gathering Sorel SHAs...") if s.is_malware]
        files = filter_packed_files(files, args.packing_protocol)
    print(f"Eligible shas: {len(files)=}")

    if args.exclude_shas is not None:
        with open(args.exclude_shas, "r") as fp:
            exclude = set([f.strip() for f in fp])
        print(f"Removing {len(exclude)=} shas.")
        files = [f for f in files if f not in exclude]

    files = sorted(files)[0:args.num_samples]
    print(f"Eligible shas: {len(files)=}")


    if args.shard_idx > -1:
        shard_size = (len(files) // args.num_shards) + 1
        idx_start = args.shard_idx * shard_size
        idx_end = (args.shard_idx + 1) * shard_size
        print(f"{shard_size=}")
        print(f"{idx_start=}")
        print(f"{idx_end=}")
        files = deepcopy(files[idx_start:idx_end])
        gc.collect()

    print(f"{len(files)=}")

    for h in tuple(hex(i)[2:] for i in range(256)):
        h = "0" + h if len(h) == 1 else h
        (args.output_root / h).mkdir(exist_ok=True)

    t_0 = time.time()
    if args.run_async:
        asyncio.run(run_async(files, args.output_root, args.num_bytes, args.max_length, args.errors))
    else:
        run(files, args.output_root, args.num_bytes, args.max_length, args.errors)

    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


if __name__ == "__main__":
    main()
