"""
Download and save datasets.
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


INCLUDE = ("name", "bytes", "text", "label")


def _filtered_dict(
    n: str,
    b: bytes,
    t: str,
    l: Literal,
    include: tuple[str] = INCLUDE,
) -> dict[str, str | bytes | int]:
    r = {}
    if "name" in include:
        r["name"] = n
    if "bytes" in include:
        r["bytes"] = b
    if "text" in include:
        r["text"] = t
    if "label" in include:
        r["label"] = l
    return r


class DatasetGenerator(Protocol):
    def __next__(self) -> dict[str, str | bytes | int]:
        ...


def disk_dataset_generator(
    files: Iterable[Path],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    include: tuple[str] = INCLUDE,
) -> Generator[dict[str, str | bytes | int], None, None]:
    for f, l in zip(files, labels):
        b: bytes = decompress(f)
        t: str = "".join(BYTE_TO_UTF8[i] for i in b)
        yield _filtered_dict(f.stem, b, t, l, include)


def s3_dataset_generator(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    bucket: str = SOREL_BUCKET,
    prefix: str = SOREL_PREFIX,
    include: tuple[str] = INCLUDE,
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
        t: str = "".join(BYTE_TO_UTF8[i] for i in b)
        yield _filtered_dict(h, b, t, l, include)


def malware_bazaar_dataset_generator(
    files: Iterable[str],
    labels: Optional[Iterable[Optional[str]]] = repeat(None),
    include: tuple[str] = INCLUDE,
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
        t: str = "".join(BYTE_TO_UTF8[i] for i in b)
        yield _filtered_dict(h, b, t, l, include)


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
    parser.add_argument("--features", nargs="+", choices=INCLUDE, default=["name", "bytes"])
    args = parser.parse_args()

    features = Features(
        {
            k: v
            for k, v in {
                "name": Value("string"),
                "bytes": Value("binary"),
                "text": Value("string"),
                "label": Value("string"),
            }.items()
            if k in args.features
        }
    )

    for d in args.datasets:
        print(f"Processing {d} ...")
        if "sorel" in d:
            files = list(islice((s.sha256 for s in stream_sorel_meta() if s.is_malware), args.num))
            generator = callable_generator(
                s3_dataset_generator,
                files=files,
                bucket=SOREL_BUCKET,
                prefix=SOREL_PREFIX,
                include=features.keys(),
            )
        elif "malware_bazaar" in d:
            with open(MALWARE_BAZAAR_FILE_LISTS[d]) as fp:
                files = list(islice((l.strip() for l in fp), args.num))
            generator = callable_generator(
                malware_bazaar_dataset_generator,
                files=files,
                include=features.keys(),
            )
        else:
            files = list(islice(DATASET_TO_FILES["binaries"][d](), args.num))
            generator = callable_generator(
                disk_dataset_generator,
                files=files,
                include=features.keys(),
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
