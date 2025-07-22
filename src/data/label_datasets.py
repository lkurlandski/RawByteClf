"""
Apply labels to the malware datasets.
"""

from __future__ import annotations
import asyncio
from collections.abc import Callable, Generator, Iterable
from itertools import chain
from functools import partial
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import sys
import tempfile
import time
from typing import Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

from datasets import Dataset, Features, Value
from tqdm import tqdm

from src.cfg import TMP_DIR
from src.utils import batched
from src.data.cfg import (
    MAX_SHARD_SIZE,
    BODMAS_LABELS_FILE,
    DATASET_TO_FILES,
    SOREL_LABEL_CACHE_DIR,
    ELF_LABEL_CACHE_DIR,
    ELF_CLASSIFICATION_DATASETS,
)
from src.data.utils import PerDatasetArgumentParser


FEATURES = Features({"name": Value("string"), "bytes": Value("binary"), "labels": list("str")})


################################################################################
# Boilerplate code to read parts of the VirusTotal reports.
################################################################################


def sorted_list_of_dicts(l: list[dict[str, int | str]]) -> list[str, int]:
    r = [(d["value"], d["count"]) for d in l]
    r.sort(key=lambda x: x[1], reverse=True)
    return r


def threat_classification(d: dict) -> dict:
    r = d["data"]["attributes"]["popular_threat_classification"]
    return r


################################################################################
# Virus total reports contain several fields that can be used to label the data:
# - popular_threat_name
# - popular_threat_category
# - suggested_threat_label
# It is unclear at the moment what exactly these fields represent.
# The ThreatLabelExtractor extracts the labels from the report.
################################################################################


def threat_name(report: dict) -> list[tuple[str, int]]:
    r = threat_classification(report)["popular_threat_name"]
    r = sorted_list_of_dicts(r)
    return r


def threat_category(report: dict) -> list[tuple[str, int]]:
    r = threat_classification(report)["popular_threat_category"]
    r = sorted_list_of_dicts(r)
    return r


def threat_label(report: dict) -> list[tuple[str, int]]:
    r = threat_classification(report)["suggested_threat_label"]
    return [(r, sys.maxsize)]


class ThreatLabelExtractor(Protocol):

    str_func_map = {
        "name": threat_name,
        "category": threat_category,
        "label": threat_label,
    }
    func_str_map = {
        threat_name: "name",
        threat_category: "category",
        threat_label: "label",
    }

    def __call__(self, report: dict) -> tuple[str, int]:
        ...

    @staticmethod
    def build(extractor: str | ThreatLabelExtractor, *args, **kwds) -> ThreatLabelExtractor:
        if callable(extractor):
            return extractor
        if extractor == "name":
            return partial(threat_name, *args, **kwds)
        if extractor == "category":
            return partial(threat_category, *args, **kwds)
        if extractor == "label":
            return partial(threat_label, *args, **kwds)
        raise ValueError(f"Unknown extractor: {extractor}")

    @staticmethod
    def name(extractor: str | partial | ThreatLabelExtractor) -> str:
        return _extractor_and_refiner_name(extractor, ThreatLabelExtractor.func_str_map)

    @staticmethod
    def descriptor(extractor: str | partial | ThreatLabelExtractor) -> str:
        return _extractor_and_refiner_descriptor(extractor)


def top_labels(labels: list[tuple[str, int]], k: int = sys.maxsize) -> tuple[str]:
    """Returns the top k most popular labels."""
    return tuple(labels[i][0] for i in range(min(k, len(labels))))


def vote_labels(labels: list[tuple[str, int]], k: int = -sys.maxsize) -> tuple[str]:
    """Returns labels with at least k votes."""
    return tuple(labels[i][0] for i in range(len(labels)) if labels[i][1] >= k)


################################################################################
# After extracting a set of labels from the report, the labels can be refined
# for single-label multiclass classification or multi-label multiclass
# classification. The refinement can be done according to two policies:
# - top: select the top k most popular labels
# - vote: select labels with at least k votes
# The ThreatLabelRefiner refines the labels according to these policies.
################################################################################


class ThreatLabelRefiner(Protocol):

    str_func_map = {
        "top": top_labels,
        "vote": vote_labels,
    }
    func_str_map = {
        top_labels: "top",
        vote_labels: "vote",
    }

    def __call__(self, labels: list[tuple[str, int]]) -> tuple[str]:
        ...

    @staticmethod
    def build(refiner: Optional[str | ThreatLabelRefiner], *args, **kwds) -> ThreatLabelRefiner:
        if callable(refiner):
            return refiner
        if refiner is None:
            return partial(top_labels, k=sys.maxsize)
        if refiner == "top":
            return partial(top_labels, *args, **kwds)
        if refiner == "vote":
            return partial(vote_labels, *args, **kwds)
        raise ValueError(f"Unknown refiner: {refiner}")

    @staticmethod
    def name(refiner: str | partial | ThreatLabelRefiner) -> str:
        return _extractor_and_refiner_name(refiner, ThreatLabelRefiner.func_str_map)

    @staticmethod
    def descriptor(refiner: str | partial | ThreatLabelRefiner) -> str:
        return _extractor_and_refiner_descriptor(refiner)


################################################################################
# This is some boilerplate code to help printing and debugging.
################################################################################


def _extractor_and_refiner_args_and_kwds(
    f: str | partial | ThreatLabelExtractor | ThreatLabelRefiner,
) -> tuple[tuple, dict]:
    if isinstance(f, str):
        return tuple(), {}
    if isinstance(f, partial):
        return f.args, f.keywords
    return tuple(), {}


def _extractor_and_refiner_descriptor(
    f: str | partial | ThreatLabelExtractor | ThreatLabelRefiner,
) -> str:
    args, kwds = _extractor_and_refiner_args_and_kwds(f)
    return "/".join([f"{a}" for a in args] + [f"{k}--{v}" for k, v in kwds.items()])


def _extractor_and_refiner_name(
    f: str | partial | ThreatLabelExtractor | ThreatLabelRefiner,
    func_str_map: dict[Callable, str],
) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, partial):
        return func_str_map[f.func]
    return func_str_map[f.__name__]


################################################################################
# The low-level API to extract labels from a report file.
################################################################################


def get_label(
    f: Path,
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
) -> Optional[tuple[str]]:
    """Extract a label from a report file.

    Returns None if the file is not a valid JSON or if the report contains no labels.
    """
    try:
        with open(f) as fp:
            d = json.load(fp)
    except json.decoder.JSONDecodeError:
        return None

    try:
        labels = extractor(d)
    except KeyError:
        return None

    labels = refiner(labels)
    return labels


async def get_label_asynch(
    f: Path,
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
) -> Optional[tuple[str]]:
    """Extract a label from a report file, asynchronously.

    Returns None if the file is not a valid JSON or if the report contains no labels.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_label, f, extractor, refiner)


def get_labels(
    report_files: Iterable[Path],
    extractor: ThreatLabelExtractor | str,
    refiner: ThreatLabelRefiner | str,
) -> Generator[Optional[tuple[str]], None, None]:
    """Lazily extract labels from report files.
    """
    extractor = ThreatLabelExtractor.build(extractor)
    refiner = ThreatLabelRefiner.build(refiner)

    for i, f in enumerate(report_files):  # pylint: disable=unused-variable
        yield get_label(f, extractor, refiner)


async def get_labels_asynch(
    report_files: Iterable[Path],
    extractor: ThreatLabelExtractor | str = "label",
    refiner: ThreatLabelRefiner | str = "top",
    asynch_chunk_size: int = 500000
) -> list[Optional[tuple[str]], None, None]:
    """Extract labels from report files, asynchronously.
    TODO: make lazy.
    """
    extractor = ThreatLabelExtractor.build(extractor)
    refiner = ThreatLabelRefiner.build(refiner)

    chunks = list(batched(report_files, asynch_chunk_size))
    iterable = tqdm(
        chunks,
        total=len(report_files) // asynch_chunk_size + 1,
        desc=f"Reading {len(report_files)} files asynchronously in {len(chunks)} chunks..."
    )

    labels = []
    for files in iterable:
        tasks = [get_label_asynch(f, extractor, refiner) for f in files]
        l = await asyncio.gather(*tasks)
        labels.extend(l)
    return labels


class ApplyLabelsFn(Protocol):
    def __call__(self, dataset: Dataset) -> Dataset:
        ...


def apply_labels_bodmas(dataset: Dataset) -> Dataset:
    labels = {}
    with open(BODMAS_LABELS_FILE) as fp:
        for line in fp:
            sha, _, family = line.strip().split(",")
            if family:
                labels[sha] = family

    labels = [labels[d["name"]] for d in dataset]
    return dataset.add_column("labels", labels)


################################################################################
# The high-level API to extract labels for a dataset.
################################################################################


def get_label_mapping_virus_total_reports(
    report_files: Iterable[os.PathLike],
    extractor: ThreatLabelExtractor | str,
    refiner: ThreatLabelRefiner | str,
    asynch: bool = False,
) -> dict[str, Optional[tuple[str]]]:
    """Get a label mapping for a set of report files.
    """
    report_files = list(report_files)

    if asynch:
        loop = asyncio.get_event_loop()
        future = get_labels_asynch(report_files, extractor, refiner)
        labels: Iterable = loop.run_until_complete(future)
    else:
        labels: Iterable = get_labels(report_files, extractor, refiner)

    iterable = zip(report_files, labels)
    if not asynch:  # Wrap in tqdm if performing sequentially
        iterable = tqdm(iterable, total=len(report_files))

    labels: dict[str, Optional[tuple[str]]] = {Path(file).stem: label for file, label in iterable}
    return labels


def get_label_mapping_virus_total_reports_with_cache(
    cache_file: Path,
    report_files: Iterable[os.PathLike],
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
    asynch: bool = False,
    use_cache: bool = True,
    create_cache: bool = True,
    overwrite_cache: bool = False,
) -> dict[str, Optional[tuple[str]]]:
    if cache_file.exists() and not use_cache and not overwrite_cache:
        raise ValueError(f"{use_cache=}, {overwrite_cache=} and file exists ({cache_file=}).")

    if cache_file.exists() and use_cache:
        print(f"Getting labels from cache: {cache_file=}", flush=True)
        t = time.time()
        with open(cache_file, "r") as fp:
            files_and_labels = json.load(fp)
        print(f"Acquired labels in {time.time() - t}")
        files_and_labels = {k: tuple(v) if isinstance(v, list) else None for k, v in files_and_labels.items()}
    else:
        print("Getting labels from Virus Total reports...")
        t = time.time()
        files_and_labels = get_label_mapping_virus_total_reports(
            report_files,
            extractor,
            refiner,
            asynch,
        )
        print(f"Acquired labels in {time.time() - t}")
        if create_cache:
            if overwrite_cache or not cache_file.exists():
                cache_file.parent.mkdir(exist_ok=True, parents=True)
                print(f"Saving labels to cache: {cache_file=}", flush=True)
                with open(cache_file, "w") as fp:
                    json.dump(files_and_labels, fp, indent=4, sort_keys=True)
    return files_and_labels


def get_label_mapping_virus_total_reports_sorel(
    report_files: Iterable[os.PathLike],
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
    asynch: bool = False,
    use_cache: bool = True,
    create_cache: bool = True,
    overwrite_cache: bool = False,
) -> dict[str, Optional[tuple[str]]]:
    """Get a label mapping for the Sorel dataset.
    """

    cache_file = Path(
        SOREL_LABEL_CACHE_DIR,
        f"extractor--{ThreatLabelExtractor.name(extractor)}",
        ThreatLabelExtractor.descriptor(extractor),
        f"refiner--{ThreatLabelRefiner.name(refiner)}",
        ThreatLabelRefiner.descriptor(refiner),
        "file_label_map.json"
    )

    return get_label_mapping_virus_total_reports_with_cache(
        cache_file,
        report_files,
        extractor,
        refiner,
        asynch,
        use_cache,
        create_cache,
        overwrite_cache,
    )


def get_label_mapping_virus_total_reports_elf(
    report_files: Iterable[os.PathLike],
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
    asynch: bool = False,
    use_cache: bool = True,
    create_cache: bool = True,
    overwrite_cache: bool = False,
) -> dict[str, Optional[tuple[str]]]:
    """Get a label mapping for the ELF dataset.
    """

    cache_file = Path(
        ELF_LABEL_CACHE_DIR,
        f"extractor--{ThreatLabelExtractor.name(extractor)}",
        ThreatLabelExtractor.descriptor(extractor),
        f"refiner--{ThreatLabelRefiner.name(refiner)}",
        ThreatLabelRefiner.descriptor(refiner),
        "file_label_map.json"
    )

    return get_label_mapping_virus_total_reports_with_cache(
        cache_file,
        report_files,
        extractor,
        refiner,
        asynch,
        use_cache,
        create_cache,
        overwrite_cache,
    )


def apply_labels_virus_total_reports(
    dataset: Dataset,
    report_files: Iterable[Path],
    extractor: ThreatLabelExtractor | str,
    refiner: ThreatLabelRefiner | str,
) -> Dataset:
    labels = get_label_mapping_virus_total_reports(report_files, extractor, refiner)
    labels = [labels.get(d["name"], None) for d in dataset]
    dataset = dataset.add_column("labels", labels)
    return dataset


def label_dataset(
    path: Path,
    apply_labels_fn: ApplyLabelsFn,
    override: bool = False,
    save: bool = False,
    verbose: bool = False,
) -> Dataset:
    dataset = Dataset.load_from_disk(path)
    if "labels" in dataset.column_names:
        if not override:
            raise ValueError("Dataset already labeled.")
        dataset = dataset.remove_columns("labels")
    if verbose:
        print(f"Unlabeled: {dataset}")

    dataset = apply_labels_fn(dataset)

    if verbose:
        print(f"Labeled: {dataset}")
        for i, d in enumerate(dataset):
            print(d["name"], d["labels"])
            if i == 16:
                break

    if save:
        temp_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
        dataset.save_to_disk(temp_dir.as_posix(), max_shard_size=MAX_SHARD_SIZE)
        shutil.rmtree(path)
        temp_dir.rename(path)

    return dataset


def main() -> None:
    raise NotImplementedError()

    parser = PerDatasetArgumentParser()
    parser.add_argument(
        "--extractor",
        choices=["name", "category", "label"],
        required=False,
        help="`name` refers to family labels, `category` to threat type, and `label` to the name.",
    )
    parser.add_argument(
        "--refiner",
        choices=["top", "vote"],
        required=False,
        help="""`top` select the top `k` most popular decisions,
             `vote` selects decisions with at least `k` votes.""",
    )
    parser.add_argument("--k", type=int)
    parser.add_argument("--override", action="store_true", help="Override existing labels.")
    parser.add_argument("--save", action="store_true", help="Save the labeled dataset.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.extractor:
        extractor = ThreatLabelExtractor.build(args.extractor)
    if args.refiner:
        refiner = ThreatLabelRefiner.build(args.refiner, k=args.k)

    if "bodmas_pe" in args.datasets:
        path = INPUT_PATH / "bodmas_pe"
        label_dataset(
            path,
            apply_labels_bodmas,
            args.override,
            args.save,
            verbose=args.verbose,
        )

    for d in [d for d in args.datasets if all(s not in d for s in ["bodmas", "local"])]:
        try:
            report_files = list(sorted(DATASET_TO_FILES["reports"][d]()))
            if len(report_files) == 0:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"No report files found for {d}. Skipping.")
            continue
        print(f"{len(report_files)} report files found for {d}.")
        path = INPUT_PATH / d
        apply_labels = partial(
            apply_labels_virus_total_reports,
            report_files=report_files,
            extractor=extractor,
            refiner=refiner,
        )
        label_dataset(
            path,
            apply_labels,
            args.override,
            args.save,
            verbose=args.verbose,
        )


def generate_label_caches():
    files_sorel = sorted(list(map(str, DATASET_TO_FILES["reports"]["sorel_pe"]())))
    files_elf = sorted(chain.from_iterable(
        (DATASET_TO_FILES["reports"][d]() for d in ELF_CLASSIFICATION_DATASETS)
    ))
    extractors = [
        ThreatLabelExtractor.build("category"),
        ThreatLabelExtractor.build("name"),
        ThreatLabelExtractor.build("label"),
    ]
    refiners = [
        ThreatLabelRefiner.build("top", k=1),
        ThreatLabelRefiner.build("top", k=2),
        ThreatLabelRefiner.build("vote", k=8),
    ]
    for extractor in extractors:
        for refiner in refiners:
            get_label_mapping_virus_total_reports_sorel(
                files_sorel, extractor, refiner, use_cache=False, overwrite_cache=True, asynch=False
            )
            get_label_mapping_virus_total_reports_elf(
                files_elf, extractor, refiner, use_cache=False, overwrite_cache=True, asynch=False,
            )


if __name__ == "__main__":
    generate_label_caches()
    # main()
    # generate_sorel_label_caches()
