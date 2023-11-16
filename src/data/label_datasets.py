"""
Apply labels to the malware datasets.
"""

from __future__ import annotations
from collections.abc import Generator, Iterable
from functools import partial
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import sys
import tempfile
from typing import Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

from datasets import Dataset, Features, Value

from src.cfg import INPUT_PATH, TMP_DIR
from src.data.cfg import MAX_SHARD_SIZE, BODMAS_LABELS_FILE, DATASET_TO_FILES
from src.data.utils import PerDatasetArgumentParser


FEATURES = Features({"name": Value("string"), "bytes": Value("binary"), "labels": list("str")})


def sorted_list_of_dicts(l: list[dict[str, int | str]]) -> list[str, int]:
    r = [(d["value"], d["count"]) for d in l]
    r.sort(key=lambda x: x[1], reverse=True)
    return r


def threat_classification(d: dict) -> dict:
    r = d["data"]["attributes"]["popular_threat_classification"]
    return r


class ThreatLabelExtractor(Protocol):
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


class ThreatLabelRefiner(Protocol):
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


def top_labels(labels: list[tuple[str, int]], k: int = sys.maxsize) -> tuple[str]:
    """Returns the top k most popular labels."""
    return tuple(labels[i][0] for i in range(min(k, len(labels))))


def vote_labels(labels: list[tuple[str, int]], k: int = -sys.maxsize) -> tuple[str]:
    """Returns labels with at least k votes."""
    return tuple(labels[i][0] for i in range(len(labels)) if labels[i][1] >= k)


def get_label(
    f: Path,
    extractor: ThreatLabelExtractor,
    refiner: ThreatLabelRefiner,
) -> tuple[str]:
    with open(f) as fp:
        d = json.load(fp)
    labels = extractor(d)
    labels = refiner(labels)
    return labels


def get_labels(
    report_files: Iterable[Path],
    extractor: ThreatLabelExtractor | str = "label",
    refiner: ThreatLabelRefiner | str = "top",
) -> Generator[Optional[tuple[str]], None, None]:
    extractor = ThreatLabelExtractor.build(extractor)
    refiner = ThreatLabelRefiner.build(refiner)

    for i, f in enumerate(report_files):  # pylint: disable=unused-variable
        try:
            labels = get_label(f, extractor, refiner)
        except KeyError:
            labels = None
        except json.decoder.JSONDecodeError:
            labels = None

        yield labels


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

    labels = [[labels[d["name"]]] for d in dataset]
    return dataset.add_column("labels", labels)


def apply_labels_virus_total_reports(
    dataset: Dataset,
    report_files: Iterable[Path],
    extractor: ThreatLabelExtractor | str,
    refiner: ThreatLabelRefiner | str,
) -> Dataset:
    report_files = list(report_files)
    iterable = zip(report_files, get_labels(report_files, extractor, refiner))
    labels: dict[str, tuple[str]] = {file.stem: label for file, label in iterable}
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


if __name__ == "__main__":
    main()
