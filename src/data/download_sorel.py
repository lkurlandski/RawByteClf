"""
Download the SOREL dataset from s3 and store to disk.
"""

from argparse import ArgumentParser
from itertools import islice
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

from src.data.prepare_datasets import s3_dataset_generator
from src.data.cfg import (
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta


parser = ArgumentParser()
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--num_samples", type=int, default=sys.maxsize)
parser.add_argument("--num_bytes", type=int, default=sys.maxsize)
parser.add_argument("--max_length", type=int, default=sys.maxsize)
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
args = parser.parse_args()

print(f"{args=}")


files = list(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num_samples))
files = sorted(files)

print(f"{len(files)=}")

if args.shard_idx > -1:
    shard_size = (len(files) // args.num_shards) + 1
    idx_start = args.shard_idx * shard_size
    idx_end = (args.shard_idx + 1) * shard_size
    print(f"{shard_size=}")
    print(f"{idx_start=}")
    print(f"{idx_end=}")
    files = files[idx_start:idx_end]

print(f"{len(files)=}")

generator = s3_dataset_generator(
    files=files,
    num_bytes=args.num_bytes,
    max_length=args.max_length,
    bucket=SOREL_BUCKET,
    prefix=SOREL_PREFIX,
    errors=args.errors,
)

for sample in tqdm(generator, total=len(files)):

    with open(args.output_root / sample["name"], "wb") as fp:
        fp.write(sample["bytes"])
