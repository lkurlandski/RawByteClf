"""
Stream data from disk or S3.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import sys
from typing import Any, ClassVar, Iterable, Literal, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotocoreConfig

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.pre import decompress


BYTE_TO_UTF8: dict[int, str] = {i: chr(i + 10752) for i in range(256)}
SOREL_BUCKET = "sorel-20m"
SOREL_PREFIX = "09-DEC-2020/binaries/"


@dataclass
class Sample:
    name: Optional[str] = None
    bytes: Optional[bytes] = None
    text: Optional[str] = None
    label: Optional[str] = None


class DatasetGenerator(ABC):

    def __init__(self, files: Iterable[Path], labels: Optional[Iterable[str]] = None) -> None:
        self.iteration = 0
        labels = [None for _ in range(len(files))]
        files_and_labels = zip(files, labels, strict=True)
        files_and_labels = sorted(list(files_and_labels), key=lambda x: x[0])
        self.files = [f for f, _ in files_and_labels]
        self.labels = [l for _, l in files_and_labels]

    def __call__(self) -> Iterable:
        return iter(self)

    def __iter__(self) -> DiskDatasetGenerator:
        return self

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def __next__(self) -> dict[str, str | bytes, tuple[str]]:
        ...


class DiskDatasetGenerator(DatasetGenerator):

    def __next__(self) -> dict:
        if self.iteration >= len(self):
            raise StopIteration

        f: Path = self.files[self.iteration]
        l: str = self.labels[self.iteration]
        b: bytes = decompress(f)
        t: str = "".join(BYTE_TO_UTF8[i] for i in b)

        self.iteration += 1
        return Sample(f.as_posix(), b, t, l)


class S3DatasetGenerator(DatasetGenerator):

    def __init__(
        self,
        files: Iterable[str],
        labels: Optional[Iterable[str]] = None,
        bucket: str = SOREL_BUCKET,
        prefix: str = SOREL_PREFIX,
    ) -> None:
        super().__init__(files, labels)

        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3", config=BotocoreConfig(signature_version=UNSIGNED))
        self.iteration = 0

    def __next__(self) -> dict:
        if self.iteration >= len(self):
            raise StopIteration()

        f = self.files[self.iteration]
        f: str = f.stem if isinstance(f, Path) else f
        l: str = self.labels[self.iteration]

        buffer = BytesIO()
        self.s3.download_fileobj(self.bucket, self.prefix + f, buffer)
        b: bytes = decompress(buffer)
        t: str = "".join(BYTE_TO_UTF8[i] for i in b)

        self.iteration += 1
        return Sample(f, b, t, l)


def test() -> None:
    from itertools import islice
    from tqdm import tqdm
    from src.data.utils import stream_sorel_meta

    samples = list(islice(filter(lambda s: s.is_malware, stream_sorel_meta()), None))
    files = [s.sha256 for s in samples]

    generator = S3DatasetGenerator(files)
    for s in tqdm(generator):
        print(f"{s.name=}, {s.label=}, {len(s.bytes)=}, {len(s.text)=}, {s.text[0:16]=}")


if __name__ == "__main__":
    test()
