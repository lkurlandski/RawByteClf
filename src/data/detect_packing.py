"""
Download the SOREL dataset from s3, run detect-it-easy program, and save outputs to disk.
"""

import asyncio
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterable, Generator
from copy import deepcopy
import gc
from itertools import islice
import json
import os
from pathlib import Path
from pprint import pprint
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import pandas as pd
from tqdm import tqdm

from src.data.prepare_datasets import s3_dataset_generator, s3_dataset_generator_async
from src.data.cfg import (
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta, Decompressor


DECOMPRESSOR = Decompressor(Decompressor.ZLIB, must_decompress=True)


# def process_report(text: str) -> dict:
#     d = {}
#     lines = text.split("\n")
#     d["Platform"] = lines[0].strip()
#     for line in lines[1:]:
#         if line.strip() == "":
#             continue
#         parts = line.split(":")
#         d[parts[0].strip()] = parts[1].strip()
#     return d


# def consolidate_reports(output_root: Path, meta_file: Path):
#     full = {}
#     for f in output_root.iterdir():
#         sha = f.stem
#         text = f.read_text()
#         d = process_report(text)
#         full[sha] = d

#     with open(meta_file, "w") as fp:
#         json.dump(full, fp, indent=2)


def analyze_sample(file_or_bytes: os.PathLike | bytes, del_file: bool = False) -> str:
    if isinstance(file_or_bytes, bytes):
        with NamedTemporaryFile("wb", delete=False) as fp:
            fp.write(file_or_bytes)
        file = Path(fp.name)
        del_file = True

    else:
        file = Path(file_or_bytes)

    result = subprocess.run(["diec", "--entropy", "--json", str(file)], check=True, capture_output=True)

    if del_file:
        file.unlink()

    output = result.stdout.decode("utf-8")
    analysis = json.loads(output)
    return analysis["status"]


async def analyze_samples_async(files: Iterable[str], output_file: Path, num_bytes: int, max_length: int, errors: int):

    if not output_file.exists():
        output_file.write_text("name,status\n")

    generator = s3_dataset_generator_async(
        files=files,
        num_bytes=num_bytes,
        max_length=max_length,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=errors,
        decompress=DECOMPRESSOR,
    )

    with open(output_file, "a") as fp:
        async for sample in generator:
            r = analyze_sample(sample["bytes"])
            fp.write(f"{sample['name']},{r}\n")

async def run_async(files: Iterable[str], output_file: Path, num_bytes: int, max_length: int, errors: int):
    await analyze_samples_async(files, output_file, num_bytes, max_length, errors)


def analyze_samples(files: Iterable[str], output_file: Path, num_bytes: int, max_length: int, errors: int):

    if not output_file.exists():
        output_file.write_text("name,status\n")

    generator = s3_dataset_generator(
        files=files,
        num_bytes=num_bytes,
        max_length=max_length,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=errors,
        decompress=DECOMPRESSOR,
    )

    with open(output_file, "a") as fp:
        for sample in tqdm(generator, total=len(files)):
            r = analyze_sample(sample["bytes"])
            fp.write(f"{sample['name']},{r}\n")


def run(files: Iterable[str], output_root: Path, num_bytes: int, max_length: int, errors: int):
    analyze_samples(files, output_root, num_bytes, max_length, errors)


def main():

    parser = ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finish", action="store_true")
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
    parser.add_argument("--run_async", action="store_true")
    args = parser.parse_args()
    print(f"{args=}")

    # Merge the csv files and clean up other files.
    if args.finish:
        finished_files = [f for f in args.output_root.glob("packingPartial_*.csv")]
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(args.output_root / "packing.csv", index=False)
        for f in finished_files:
            f.unlink()
        return

    # Prepare the shard files for embarassingly parallel execution
    if args.prepare:
        files = sorted(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num_samples))
        shard_size = (len(files) // args.num_shards) + 1
        print(f"{len(files)=}")
        print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")
        print(f"{shard_size=}") 
        for shard_idx in range(args.num_shards):
            idx_start = shard_idx * shard_size
            idx_end = (shard_idx + 1) * shard_size
            shard_file = args.output_root / f"packingPrep_{shard_idx}.txt"
            with open(shard_file, "w") as fp:
                for i in range(idx_start, min(idx_end, len(files) - 1)):
                    fp.write(f"{files[i]}\n")
        return

    if isinstance(args.shard_idx, int):
        shard_file = args.output_root / f"packingPrep_{args.shard_idx}.txt"
        with open(shard_file, "r") as fp:
            files = [l.strip() for l in fp.readlines()]
    else:
        files = sorted(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num_samples))
    print(f"{len(files)=}")
    print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

    output_file = args.output_root / f"packingPartial_{args.shard_idx}.csv" if args.shard_idx is not None else args.output_root / "packing.csv"
    t_0 = time.time()
    if args.run_async:
        asyncio.run(run_async(files, output_file, args.num_bytes, args.max_length, args.errors))
    else:
        run(files, output_file, args.num_bytes, args.max_length, args.errors)

    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


def test_prepare():
    output_root = Path("/home/lk3591/Documents/datasets/Sorel/")
    num_shards = 16
    prep_files = [output_root / f"packingPrep_{i}.txt" for i in range(num_shards)]
    files = []
    for f in prep_files:
        with open(f, "r") as fp:
            files.extend([line.strip() for line in fp])

    print(f"{len(files)=}")
    print(f"{len(set(files))=}")
    print(f"{files[0:4]=}")
    # print(f"{[(f, i) for f, i in Counter(files) if i > 1]}")

    baseline = sorted(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), None))

    print(f"{len(baseline)=}")
    print(f"{len(set(baseline))=}")
    print(f"{baseline[0:4]=}")
    print(f"{files == baseline}")


def test():

    sha = "00160eda1e5b66ea1f819aa3fd52f15342c9d6ced37d66354d86abde59ecd08a"
    run([sha], Path("./tmp"), 2 ** 20, 2 ** 20, 2)
    return

    d = Path("/home/lk3591/Documents/datasets/Windows/processed/train/")
    f_0 = d / "003eb5054c31c7b2d00e8dd03de42322e4b791ec6ae5b378e7e5655a.exe"
    f_1 = d / "001258fe331d0041d587f0b05fefe9ef2293e0eef058cf71d057c1b7.exe"

    # print(analyze_sample(f_0))
    # print(analyze_sample(f_0.read_bytes()))

    r_0 = analyze_sample(f_0)
    r_1 = analyze_sample(f_1)
    pprint(process_report(r_0))
    pprint(process_report(r_1))



if __name__ == "__main__":
    # test()
    # sys.exit()
    main()
