"""

"""

from argparse import ArgumentParser
from collections.abc import Callable, Generator, Iterable
from itertools import islice, repeat
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
from typing import Literal, Optional, Protocol

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
parser.add_argument("--num", type=int, default=sys.maxsize, help="Number of samples to use.")
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path("/home/lk3591/Documents/datasets/Sorel/binaries"),
    help="Where to save the datasets.",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=sys.maxsize,
    help="Keep the first `max_length` bytes; discard the rest.",
)
parser.add_argument(
    "--errors",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="""How to handle specific errors. 0 raises exceptions.
            1 returns empty samples when errors are encountered.
            2 skips errd samples entirely.""",
)
parser.add_argument(
    "--shard",
    type=int,
    default=None,
    help="If sharding, refers to the shard idx. Use -1 to merge shards.",
)
parser.add_argument(
    "--n_shards", type=int, default=None, help="Number of shards when sharding."
)
args = parser.parse_args()

print(f"{args=}")


files = list(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num))
files = sorted(files)

print(f"{len(files)=}")

if args.shard > -1:
    shard_size = (len(files) // args.n_shards) + 1
    idx_start = args.shard * shard_size
    idx_end = (args.shard + 1) * shard_size
    print(f"{shard_size=}")
    print(f"{idx_start=}")
    print(f"{idx_end=}")
    files = files[idx_start:idx_end]

print(f"{len(files)=}")

generator = s3_dataset_generator(
    files=files,
    max_length=args.max_length,
    bucket=SOREL_BUCKET,
    prefix=SOREL_PREFIX,
    errors=args.errors,
)

for sample in tqdm(generator, total=len(files)):

    with open(args.output_root / sample["name"], "wb") as fp:
        fp.write(sample["bytes"])
