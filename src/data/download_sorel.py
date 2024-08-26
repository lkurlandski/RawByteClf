"""
Download the SOREL dataset from s3 and store to disk.

parallel -j 0 python src/data/download_sorel.py --filter_idx={} ::: {00..ff}

USAGE
-----

Example 1:

    The following will download the corpus using FILTER mode. N parallel workers managed by gnu-parallel
      will download their own slice of the dataset. The slice for each worker is determined by the first
      two characters of the samples' SHA hash. The process will finish once all 256 workers are done.

    parallel --bar -j 8 'python src/data/download_sorel.py --output_root=path/to/binaries/ --packing_protocol=no --errors=2 --filter_mode=2 --filter_idx={1} > ./logs/download_{1}.txt 2>&1' ::: $(printf "%02x\n" {0..255})

    In this example, the memory spikes to around 2GB when the program is gathering the SHAs to download,
      then drops to less than 1GB during the actual downloading process.

Example 2:

    The following will download the corpus using IDX mode. N parallel workers (not managed by gnu-parallel)
      will download an equal portion of the dataset. The process will finish once all N workers are done.

    for i in {0..7}; do python src/data/download_sorel.py --output_root=/path/to/binaries --packing_protocol=no --errors=2 --num_shards=8 --shard_idx=$i > logs/download_$i.txt 2>&1 &; done

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


def main():

    parser = ArgumentParser()
    # Control how files are downloaded
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=sys.maxsize)
    parser.add_argument("--num_bytes", type=int, default=sys.maxsize)
    parser.add_argument("--max_length", type=int, default=sys.maxsize)
    parser.add_argument("--packing_protocol", default="any", choices=["yes", "no", "unk", "any"])
    parser.add_argument("--include_shas", type=str, default=None)
    parser.add_argument("--exclude_shas", type=str, default=None)
    # Control parallelization
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--filter_idx", type=str, default=None)
    parser.add_argument("--filter_mode", type=int, default=None)
    # Error handling
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
    args = parser.parse_args()
    print(f"{args=}")

    # Get eligible files for downloading
    if args.include_shas is not None:
        with open(args.include_shas, "r") as fp:
            files = [f.strip() for f in fp]
    else:
        files = [s.sha256 for s in tqdm(stream_sorel_meta(), total=20000000, desc="Gathering Sorel SHAs...") if s.is_malware]
        files = filter_packed_files(files, args.packing_protocol)
    print(f"Eligible shas: {len(files)=}")

    # Remove shas if needed
    if args.exclude_shas is not None:
        with open(args.exclude_shas, "r") as fp:
            exclude = set([f.strip() for f in fp])
        print(f"Removing {len(exclude)=} shas.")
        files = [f for f in files if f not in exclude]

    # Sort, then trim to the requested number
    files = sorted(files)[0:args.num_samples]
    print(f"Eligible shas: {len(files)=}")

    # Handle parallelization if using IDX mode
    if args.shard_idx is not None:
        shard_size = (len(files) // args.num_shards) + 1
        idx_start = args.shard_idx * shard_size
        idx_end = (args.shard_idx + 1) * shard_size

    # Handle parallelization if using FILTER mode
    if args.filter_idx is not None:
        if args.filter_mode != 2:
            raise ValueError(f"{args.filter_mode=}")
        if len(args.filter_idx) != 2:
            raise ValueError(f"{args.filter_idx}")

        idx_start = None
        idx_end = len(files)
        for i, f in enumerate(files):
            s = f[0:2]
            if idx_start is None and s == args.filter_idx:
                idx_start = i
                print(f"Set {idx_start=}")
            if idx_start is not None and s != args.filter_idx:
                idx_end = i
                print(f"Set {idx_end=}")
                break # exit loop to avoid re-setting idx_end

        if idx_start is None:
            raise RuntimeError("idx_start should not be None.")

    # Slice the files according to the parallelization mode
    if args.shard_idx is not None or args.filter_idx is not None:
        print(f"{idx_start=}")
        print(f"{idx_end=}")
        files = deepcopy(files[idx_start:idx_end])
        gc.collect()

    # Final length of the files
    print(f"{len(files)=}")

    # Create sub directories: 00, 01, ..., ff
    for h in tuple(hex(i)[2:] for i in range(256)):
        h = "0" + h if len(h) == 1 else h
        (args.output_root / h).mkdir(exist_ok=True)

    # Execute process
    t_0 = time.time()
    download_samples(files, args.output_root, args.num_bytes, args.max_length, args.errors)
    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


if __name__ == "__main__":
    main()
