"""
Assist with labeling process.
"""

from collections.abc import Generator, Iterable
import json
from pathlib import Path
import sys
from typing import Literal, Optional, Protocol


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


def top_labels(labels: list[tuple[str, int]], k: int = 1) -> tuple[str]:
    return tuple(labels[i][0] for i in range(max(k, len(labels))))


def vote_labels(labels: list[tuple[str, int]], k: int = 2) -> tuple[str]:
    return tuple(labels[i][0] for i in range(len(labels)) if labels[i][1] >= k)


def _get_labels(
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
    report_dir: Path,
    files: Iterable[Path | str],
    extractor: ThreatLabelExtractor | Literal["name", "category", "label"] = "label",
    refiner: ThreatLabelRefiner = lambda x: x,
    logfile: str = "./labeling.log",
) -> Generator[Optional[tuple[str]], None, None]:

    if extractor == "name":
        extractor = threat_name
    elif extractor == "category":
        extractor = threat_category
    elif extractor == "label":
        extractor = threat_label

    print(f"Logging failures in {logfile=}")
    with open(logfile, "a") as fp:
        fp.write("-" * 88 + "\n")

    failures = 0
    report_files = map(lambda f: report_dir / f"{Path(f).stem}.json", files)
    for i, f in enumerate(report_files):
        try:
            labels = _get_labels(f, extractor, refiner)
        except KeyError as err:
            failures += 1
            with open(logfile, "a") as fp:
                fp.write(f.name + ": KeyError: " + err.args[0] + "\n")
            labels = None
        
        yield labels

    print(f"Retrieved labels for {i - failures} / {i} files")


def test() -> None:
    report_dir = Path("/home/lk3591/Documents/datasets/Sorel/reports")
    itr = get_labels(
        report_dir,
        sorted(list(f.name for f in report_dir.iterdir())),
        extractor="label",
    )

    count = 0
    for l in itr:
        if l:
            count += 1
            print(l)


if __name__ == "__main__":
    test()
