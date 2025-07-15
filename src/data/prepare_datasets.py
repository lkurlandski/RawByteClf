"""
Download and save datasets.

TODO:
    - Add fundamental metadata, such as `file_type`, etc.
"""

import asyncio
from collections.abc import Callable, Generator, Iterable
from itertools import islice, repeat
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
from typing import Any, AsyncGenerator, Literal, Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config as BotocoreConfig
from datasets import Dataset, Features, Value, concatenate_datasets

from src.data.cfg import (
    MAX_SHARD_SIZE,
    DATASET_TO_FILES,
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta, Decompressor, PerDatasetArgumentParser


class ErrorStream:
    def __init__(self, file: Path) -> None:
        self.file = Path(file)
        self.file.write_text("")

    def write(self, message: str):
        with open(self.file, "a") as log_file:
            log_file.write(message)


ERRORS = ErrorStream("err.log")


def sample(
    name: str,
    bytes_: bytes,
    labels: Optional[tuple[str]] = None,
    num_bytes: Optional[int] = None,
) -> dict[str, str | bytes | int]:
    num_bytes = sys.maxsize if num_bytes is None else num_bytes
    return {
        "name": name,
        "bytes": bytes_[0:num_bytes],
        "size": len(bytes_),
        "labels": labels,
        "length": min(len(bytes_), num_bytes),
    }


class DatasetGenerator(Protocol):
    def __next__(self) -> dict[str, str | bytes | int]:
        ...


def disk_dataset_generator(
    files: Iterable[Path],
    labels: Iterable[Optional[str]] = repeat(None),
    num_bytes: Optional[int] = None,
    max_length: Optional[int] = None,
    errors: int = 0,
    decompress: Optional[Decompressor] = None,
) -> Generator[dict[str, str | bytes | int], None, None]:
    decompress = Decompressor(None, False) if decompress is None else decompress
    for f, l in zip(files, labels):
        try:
            _, b = decompress(f)
        except Exception as err:
            msg = f"{f} {str(err)}"
            if errors == 0:
                raise type(err)(msg)
            if errors == 1:
                print(msg, file=ERRORS)
                yield sample(f.stem, bytes(), l, num_bytes)
                continue
            if errors == 2:
                print(msg, file=ERRORS)
                continue
            raise RuntimeError() from err

        if len(b) > max_length:
            continue

        yield sample(f.stem, b, l, num_bytes)


async def s3_download_file(s3, bucket, key, buffer):
    await asyncio.to_thread(s3.download_fileobj, bucket, key, buffer)


async def s3_decompress(
    buffer: BytesIO,
    decompress: Optional[Decompressor] = None,
) -> tuple[int, bytes]:
    decompress = Decompressor(None, False) if decompress is None else decompress
    return await asyncio.to_thread(decompress, buffer)


async def s3_dataset_generator_async(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    num_bytes: Optional[int] = None,
    max_length: Optional[int] = None,
    bucket: str = SOREL_BUCKET,
    prefix: str = SOREL_BUCKET,
    errors: int = 0,
    decompress: Optional[Decompressor] = None,
) -> AsyncGenerator[dict[str, str | bytes | int], None]:
    """
    files and labels must be pickle-able.
    """
    s3 = boto3.client("s3", config=BotocoreConfig(signature_version=UNSIGNED))
    decompress = Decompressor(None, False) if decompress is None else decompress
    for f, l in zip(files, labels):
        h = f.stem if isinstance(f, Path) else f
        key = prefix + h
        buffer = BytesIO()

        try:
            await s3_download_file(s3, bucket, key, buffer)
        except botocore.exceptions.ClientError as err:
            msg = f"{h} {str(err)}"
            if errors == 0:
                print(msg)
                raise err
            if errors == 1:
                print(msg, file=ERRORS)
                yield sample(h, bytes(), l, num_bytes)
                continue
            if errors == 2:
                print(msg, file=ERRORS)
                continue
            raise RuntimeError() from err

        try:
            _, b = await s3_decompress(buffer, decompress)
        except Exception as err:
            msg = f"{h} {str(err)}"
            if errors == 0:
                print(msg)
                raise err
            if errors == 1:
                print(msg, file=ERRORS)
                yield sample(h, bytes(), l, num_bytes)
                continue
            if errors == 2:
                print(msg, file=ERRORS)
                continue
            raise RuntimeError() from err

        if len(b) > max_length:
            continue

        yield sample(h, b, l, num_bytes)


def s3_dataset_generator(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    num_bytes: Optional[int] = None,
    max_length: Optional[int] = None,
    bucket: str = SOREL_BUCKET,
    prefix: str = SOREL_PREFIX,
    errors: int = 0,
    decompress: Optional[Decompressor] = None,
) -> Generator[dict[str, str | bytes | int], None, None]:
    """
    files and labels must be pickle-able.
    """
    decompress = Decompressor(None, False) if decompress is None else decompress
    s3: boto3.Session = boto3.client("s3", config=BotocoreConfig(signature_version=UNSIGNED))
    for f, l in zip(files, labels):
        h: str = f.stem if isinstance(f, Path) else f
        key = prefix + h
        buffer = BytesIO()

        try:
            s3.download_fileobj(bucket, key, buffer)
        except botocore.exceptions.ClientError as err:
            msg = f"{h} {str(err)}"
            if errors == 0:
                print(msg)
                raise err
            if errors == 1:
                print(msg, file=ERRORS)
                yield sample(h, bytes(), l, num_bytes)
                continue
            if errors == 2:
                print(msg, file=ERRORS)
                continue
            raise RuntimeError() from err

        try:
            _, b = decompress(buffer)
        except Exception as err:
            msg = f"{h} {str(err)}"
            if errors == 0:
                print(msg)
                raise err
            if errors == 1:
                print(msg, file=ERRORS)
                yield sample(h, bytes(), l, num_bytes)
                continue
            if errors == 2:
                print(msg, file=ERRORS)
                continue
            raise RuntimeError() from err

        if len(b) > max_length:
            continue

        yield sample(h, b, l, num_bytes)


class CallableDatasetGenerator(Protocol):
    def __call__(self) -> DatasetGenerator:
        ...


def callable_generator(func: Callable, **kwargs) -> Callable:
    return lambda: func(**kwargs)  # pylint: disable=unnecessary-lambda


def main() -> None:
    parser = PerDatasetArgumentParser()
    parser.add_argument("--num", type=int, default=sys.maxsize, help="Number of samples to use.")
    parser.add_argument("--keep_cache", action="store_true", help="Do not delete cache files.")
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

    if bool(args.shard_idx) != bool(args.num_shards):
        raise ValueError("Both a shard idx and the number of shards required.")

    features = Features(
        {
            "name": Value("string"),
            "bytes": Value("binary"),
            "labels": Value("string"),
            "size": Value("int64"),
            "length": Value("int64"),
        }
    )

    for d in args.datasets:
        print(f"Processing {d} ...")
        if "sorel" in d and args.shard_idx != -1:
            files = list(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num_samples))
            files = sorted(files)
            if args.shard_idx > -1:
                shard_size = (len(files) // args.num_shards) + 1
                files = files[args.shard_idx * shard_size : (args.shard_idx + 1) * shard_size]
            generator = callable_generator(
                s3_dataset_generator,
                files=files,
                num_bytes=args.num_bytes,
                max_length=args.max_length,
                bucket=SOREL_BUCKET,
                prefix=SOREL_PREFIX,
                errors=args.errors,
            )
        elif args.shard_idx != -1:
            files = list(islice(DATASET_TO_FILES["binaries"][d](), args.num_samples))
            generator = callable_generator(
                disk_dataset_generator,
                files=sorted(files),
                max_length=args.max_length,
                errors=args.errors,
            )

        if args.shard_idx == -1:
            dsets = [args.output_root / f"{d}_{i}" for i in range(args.num_shards)]
            print(f"Merging {d} {len(dsets)} shards...")
            dsets = [Dataset.load_from_disk(p.as_posix()) for p in dsets]
            dataset = concatenate_datasets(dsets)
        else:
            dataset = Dataset.from_generator(generator=generator, features=features)

        if dataset.num_rows == 0:
            print(f"Empty dataset for {d}. Skipping.")
            continue

        outpath = args.output_root / d
        if args.shard_idx and args.shard_idx != -1:
            outpath = args.output_root / f"{d}_{args.shard_idx}"
        dataset.save_to_disk(outpath.as_posix(), max_shard_size=MAX_SHARD_SIZE)
        if not args.keep_cache:
            shutil.rmtree(Path(dataset.cache_files[0]["filename"]).parent)


if __name__ == "__main__":
    main()
