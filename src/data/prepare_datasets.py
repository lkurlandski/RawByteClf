"""
Download and save datasets.

TODO:
    - Add fundamental metadata, such as `file_type`, etc.
"""

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

# pylint: disable=wrong-import-position

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotocoreConfig
from datasets import Dataset, Features, Value
import requests

from src.cfg import INPUT_PATH
from src.data.cfg import (
    BYTE_TO_UTF8,
    MAX_SHARD_SIZE,
    DATASET_TO_FILES,
    MALWARE_BAZAAR_FILE_LISTS,
    MALWARE_BAZAAR_URL,
    SOREL_BUCKET,
    SOREL_PREFIX,
)
from src.data.utils import stream_sorel_meta, decompress, PerDatasetArgumentParser


def sample(
    name: str,
    bytes_: bytes,
    labels: Optional[tuple[str]] = None,
    max_length: Optional[int] = None,
) -> dict[str, str | bytes | int]:
    max_length = sys.maxsize if max_length is None else max_length
    return {
        "name": name,
        "bytes": bytes_[0:max_length],
        "size": len(bytes_),
        "labels": labels,
        "length": min(len(bytes_), max_length),
    }


class DatasetGenerator(Protocol):
    def __next__(self) -> dict[str, str | bytes | int]:
        ...


def disk_dataset_generator(
    files: Iterable[Path],
    labels: Iterable[Optional[str]] = repeat(None),
    max_length: Optional[int] = None,
) -> Generator[dict[str, str | bytes | int], None, None]:
    for f, l in zip(files, labels):
        b: bytes = decompress(f)
        yield sample(f.stem, b, l, max_length)


def s3_dataset_generator(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    max_length: Optional[int] = None,
    bucket: str = SOREL_BUCKET,
    prefix: str = SOREL_PREFIX,
) -> Generator[dict[str, str | bytes | int], None, None]:
    """
    files and labels must be pickle-able.
    """
    s3: boto3.Session = boto3.client("s3", config=BotocoreConfig(signature_version=UNSIGNED))
    for f, l in zip(files, labels):
        h: str = f.stem if isinstance(f, Path) else f
        buffer = BytesIO()
        s3.download_fileobj(bucket, prefix + h, buffer)
        b: bytes = decompress(buffer)
        yield sample(h, b, l, max_length)


def malware_bazaar_dataset_generator(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    max_length: Optional[int] = None,
) -> Generator[dict[str, str | bytes | int], None, None]:
    def data(h: str) -> dict[str, str]:
        return {"query": "get_file", "sha256_hash": h}

    for h, l in zip(files, labels):
        response = requests.post(MALWARE_BAZAAR_URL, data=data(h), timeout=60)

        if response.status_code != 200:
            print(f"HTTPErrpr: {response.status_code} for {h}")
            continue

        b = response.content
        b = decompress(BytesIO(b))
        yield sample(h, b, l, max_length)


class CallableDatasetGenerator(Protocol):
    def __call__(self) -> DatasetGenerator:
        ...


def callable_generator(func: Callable, **kwargs) -> Callable:
    return lambda: func(**kwargs)  # pylint: disable=unnecessary-lambda


def main() -> None:
    parser = PerDatasetArgumentParser()
    parser.add_argument("--num", type=int, default=sys.maxsize)
    parser.add_argument("--keep_cache", action="store_true")
    parser.add_argument("--output_root", type=Path, default=INPUT_PATH)
    parser.add_argument("--max_length", type=int, default=sys.maxsize)
    args = parser.parse_args()

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
        if "sorel" in d:
            files = list(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num))
            generator = callable_generator(
                s3_dataset_generator,
                files=files,
                max_length=args.max_length,
                bucket=SOREL_BUCKET,
                prefix=SOREL_PREFIX,
            )
        elif "malware_bazaar" in d:
            with open(MALWARE_BAZAAR_FILE_LISTS[d]) as fp:
                files = list(islice((l.strip() for l in fp), args.num))
            generator = callable_generator(
                malware_bazaar_dataset_generator,
                files=files,
                max_length=args.max_length,
            )
        else:
            files = list(islice(DATASET_TO_FILES["binaries"][d](), args.num))
            generator = callable_generator(
                disk_dataset_generator,
                files=files,
                max_length=args.max_length,
            )

        dataset = Dataset.from_generator(generator=generator, features=features)
        if dataset.num_rows == 0:
            print(f"Empty dataset for {d}. Skipping.")
            continue

        outpath = args.output_root / d
        dataset.save_to_disk(outpath.as_posix(), max_shard_size=MAX_SHARD_SIZE)
        if not args.keep_cache:
            shutil.rmtree(Path(dataset.cache_files[0]["filename"]).parent)


def test():
    for f in INPUT_PATH.iterdir():
        d = Dataset.load_from_disk(f)
        print(f)
        print(d)


if __name__ == "__main__":
    main()
