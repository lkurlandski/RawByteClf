"""
Labeling with AVClass and ClarAVy.

TODO
----
- Implement an embarassingly parallel system for labeling with AVClass.
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections import Counter, defaultdict, UserDict
from dataclasses import dataclass
import hashlib
from itertools import chain
import os
from pathlib import Path
import pickle
from pprint import pprint
import re
import subprocess
import sys
import time
from typing import Optional
import warnings

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import AVCLASS_EXE, CLARAVY_EXE


NUM_WORKERS = 16


KEYS = [
    "class_",  # avclass
    "file",  # claravy
    "fam",  # avclass
    "beh",  # claravy
    "unk",  # ?
    "pack",  # claravy
    "vuln",  # claravy
]


@dataclass
class Label:
    class_: tuple[str] = None
    file: tuple[str]  = None
    fam: tuple[str] = None
    beh: tuple[str] = None
    unk: tuple[str] = None
    pack: tuple[str] = None
    vuln: tuple[str] = None

    @property
    def is_labeled(self) -> bool:
        return any(getattr(self, k) for k in KEYS)


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class FilterArgs:
    """
    The filtering protocol (top_k, min_freq) for each label.
    """
    class_: tuple[int, int] = (1, 5)
    file: tuple[int, int] = (1, 5)
    fam: tuple[int, int] = (1, 2)
    beh: tuple[int, int] = (1, 5)
    unk: tuple[int, int] = (1, 2)
    pack: tuple[int, int] = (1, 1)
    vuln: tuple[int, int] = (1, 1)

    def byte_identifier(self) -> bytes:
        # pickle.dumps(self) seems to be nondeterministic, so we'll use something cruder.
        s = f"{self.class_}{self.file}{self.fam}{self.beh}{self.unk}{self.pack}{self.vuln}"
        return s.encode()


@dataclass
class Item:
    sha: str
    flagged: int
    class_: list[tuple[str, int]] = None
    file: list[tuple[str, int]] = None
    fam: list[tuple[str, int]] = None
    beh: list[tuple[str, int]] = None
    unk: list[tuple[str, int]] = None
    pack: list[tuple[str, int]] = None
    vuln: list[tuple[str, int]] = None

    @classmethod
    def from_tool_output_line(cls, s: str) -> Item:
        # Extract the sha, the number of AVs that flagged the sample,
        # and the information extracted from all of the AVs.
        args = s.split()
        sha = args[0]
        flagged = int(args[1]) if args[1].isdigit() else 0
        if len(args) == 3:
            information = args[2]
        else:
            information = ""

        # If there was no information about the samples, return None for all fields.
        if information in ("[]", "") or "SINGLETON" in information:
            return Item(sha, flagged)

        # Parse the information extracted from the AVs.
        # The information string looks roughly like: "{FIELD}:{VALUE}|{COUNT},...,"
        labels = defaultdict(list)
        for piece in information.split(","):
            field = piece[0:piece.index(":")]
            value = piece[piece.index(":") + 1:piece.index("|")]
            count = int(piece[piece.index("|") + 1:])
            labels[field].append((value, count))

        # Replace `class` with `class_` to avoid conflicts with the reserved keyword.
        labels = {k.lower(): v for k, v in labels.items()}
        if "class" in labels:
            labels["class_"] = labels.pop("class")

            # class_ can get polluted with some poorly formatted values.
            # Figure out which values are actually multiple values separated by a colon.
            append = []
            remove = []
            for value, count in labels["class_"]:
                if ":" not in value:
                    continue
                remove.append((value, count))
                append.extend([(v, count) for v in value.split(":")])

            if remove or append:
                # These need to be removed and their constituent values need to be added.
                for r in remove:
                    labels["class_"].remove(r)
                for a in append:
                    labels["class_"].append(a)
                # The net counts then need to be summed.
                d = defaultdict(int)
                for value, count in labels["class_"]:
                    d[value] += count
                labels["class_"] = list(d.items())

        return Item(sha, flagged, **labels)

    def filter(self, args: FilterArgs) -> Label:

        def _filter(
            items: Optional[list[tuple[str, int]]],
            top_k: Optional[int],
            vote_k: Optional[int],
        ) -> Optional[tuple[str]]:
            if items is None:
                return None

            items = list(sorted(items, key=lambda x: x[1], reverse=True))
            if vote_k:
                items = [(l, c) for l, c in items if c >= vote_k]
            if top_k:
                items = items[:top_k]

            l = tuple(l for l, _ in items)
            return None if not l else l

        class_ = _filter(self.class_, *args.class_)
        file = _filter(self.file, *args.file)
        fam = _filter(self.fam, *args.fam)
        beh = _filter(self.beh, *args.beh)
        unk = _filter(self.unk, *args.unk)
        pack = _filter(self.pack, *args.pack)

        return Label(class_, file, fam, beh, unk, pack)


class ToolRunner:
    """
    TODO:
      - Implement an embarassingly parallel system for the AVClass labeling.
    """

    def __init__(
        self,
        reports_dir: Path,
        claravy_cache: Path,
        avclass_cache: Path,
        avclass_family_cache: Path,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.avclass_cache = Path(avclass_cache)
        self.avclass_family_cache = Path(avclass_family_cache)
        self.claravy_cache = Path(claravy_cache)

    def __call__(self) -> Labeler:

        print("Running CLARAVY...", end="", flush=True)
        t_0 = time.time()
        if not self.claravy_cache.exists():
            self.run_claravy()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        print("Running AVCLASS...", end="", flush=True)
        t_0 = time.time()
        if not self.avclass_cache.exists():
            self.run_avclass()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        print("Running AVCLASS-family...", end="", flush=True)
        t_0 = time.time()
        if not self.avclass_family_cache.exists():
            self.run_avclass_family()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        return self

    def run_claravy(self) -> None:

        args = [
            f"{CLARAVY_EXE.as_posix()}",
            f"{'-d' if self.reports_dir.is_dir() else '-f'}={self.reports_dir.as_posix()}",
            f"-o={self.claravy_cache.as_posix()}",
            "-hash=sha256",
            "-bt=1",
            "-ft=1",
            "-vt=1",
            "-pt=1",
            f"--num-processes={NUM_WORKERS}"
        ]
        print(f"args='{' '.join(args)}'")
        try:
            subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(f"{err.stdout=}\n{err.stderr=}")
            raise

    def run_avclass(self) -> None:

        args = [
            f"{AVCLASS_EXE.as_posix()}",
            f"{'-d' if self.reports_dir.is_dir() else '-f'}={self.reports_dir.as_posix()}",
            f"-o={self.avclass_cache.as_posix()}",
            "-hash=sha256",
            "-t",
        ]
        print(f"args='{' '.join(args)}'")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr.decode())
            raise

    def run_avclass_family(self) -> None:

        args = [
            f"{AVCLASS_EXE.as_posix()}",
            f"{'-d' if self.reports_dir.is_dir() else '-f'}={self.reports_dir.as_posix()}",
            f"-o={self.avclass_family_cache.as_posix()}",
            "-hash=sha256",
        ]
        print(f"args='{' '.join(args)}'")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as err:
            print(f"{err.stdout=}\n{err.stderr=}")
            raise


class Labeler:
    """
    Provides a simple interface for using simplified malware labels.

    This class parses the output of AVClass and ClarAVy (.txt) cache files and
    extracts labels based upon the a filter to control how accurate a label must be.
    It also creates a unique cache file based on the contents of the claravy, avclass,
    and avclass-family cache files as well as the `filter_args`. Subsequent usage
    reuses the cache to avoid reparsing the data, which is slow.

    There were some difficulties dealing with import errors when using pickle to
    serialize a custom object, so we serialize things as dict, not Label, then
    convert back to Label when loading the cache.

    Usage
    -----

    Suppose we want to consider the five most popular labels that were flagged by
     at least two AVs. In other words, we set:
    >>> top_k = 5
    >>> min_freq = 2
    >>> rule = (top_k, min_freq)

    We can create a special FilterArgs object representing this desire. If we wanted to,
     we could control the filtering rules for each label type individually, but for now
     we will just use the same rule for every type of label:
    >>> filter_args = FilterArgs(
    ...     class_=rule,
    ...     file=rule,
    ...     fam=rule,
    ...     beh=rule,
    ...     unk=rule,
    ...     pack=rule,
    ...     vuln=rule,
    ... )

    Although there is an option to control the labeling for "fam" (family labels),
    note that this is in fact inert in other parts of the program because AVClass2
    implements its own family-label-consolidation method.

    To apply to filter rules to the saved plaintext cache files, we can use the Labeler.
    This parses the text files, applies the filtering, and serializes the results:
    >>> labeler = Labeler(
    ...     "claravy_cache.txt",
    ...     "avclass_cache.txt",
    ...     "avclass_family_cache.txt",
    ...     filter_args,
    ...     use_cache=True,
    ... )

    The labeler contains all of the information we need. We can a specific type of label,
    e.g., family, class, or behavioral labels with the `view` method:
    >>> family_labels = labeler.view("fam")
    >>> class_labels = labeler.view("class")
    >>> behavioral_labels = labeler.view("beh")

    `Labeler.view` returns a dictionary where the keys are the SHA256 hashes of the samples
    and the values are a tuple of string labels produced by the AVClass and ClarAVy tools.

    The family labels are always going to have one element:
    >>> family_labels["0000000000000000000000000000000000000000000000000000000000000000"]
    ... ('upatre',)
    Other types of labels can have between zero and `top_k` elements:
    >>> class_labels["0000000000000000000000000000000000000000000000000000000000000000"]
    ... ('trojan', 'downloader', 'generic', 'malware', 'trojandownloader')
    """

    def __init__(
        self,
        claravy_cache: Path,
        avclass_cache: Path,
        avclass_family_cache: Path,
        filter_args: Optional[FilterArgs] = FilterArgs(),
        use_cache: bool = True,
    ) -> None:
        self.avclass_cache = Path(avclass_cache)
        self.avclass_family_cache = Path(avclass_family_cache)
        self.claravy_cache = Path(claravy_cache)
        self.filter_args = filter_args
        self.data: dict[str, Label] = {}
        self._cache_file = None
        self.use_cache = use_cache

    def __call__(self) -> Labeler:
        if self.cache_file.exists() and self.use_cache:
            print(f"Loading data from {self.cache_file=}...", end="", flush=True)
            t_0 = time.time()
            with open(self.cache_file, "rb") as fp:
                data: dict[str, Optional[tuple[str]]] = pickle.load(fp)
                self.data = {k: Label(**v) for k, v in data.items()}
            print(f"Done. Took {time.time() - t_0:.2f} seconds")
            return self

        print(f"Cache file {self.cache_file=} does not exist. Running the labeler.")

        print("Parsing CLARAVY...", end="", flush=True)
        t_0 = time.time()
        self.parse_claravy()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        t_0 = time.time()
        print("Parsing AVCLASS...", end="", flush=True)
        self.parse_avclass()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        t_0 = time.time()
        print("Parsing AVCLASS-family...", end="", flush=True)
        self.parse_avclass_family()
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        print(f"Dumping data to {self.cache_file=}...", end="", flush=True)
        t_0 = time.time()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as fp:
            data: dict[str, Optional[tuple[str]]] = {k: v.__dict__ for k, v in self.data.items()}
            pickle.dump(data, fp)
        print(f"Done. Took {time.time() - t_0:.2f} seconds")

        return self

    @property
    def cache_file(self) -> Path:
        if self._cache_file is not None:
            return self._cache_file

        b_1 = self.claravy_cache.read_bytes()
        b_2 = self.avclass_cache.read_bytes()
        b_3 = self.avclass_family_cache.read_bytes()
        b_4 = self.filter_args.byte_identifier()
        b = b_1 + b_2 + b_3 + b_4

        h = hashlib.sha256(b).hexdigest()
        self._cache_file = Path("./cache") / "labeling" / f"{h}.pkl"
        return self._cache_file

    def parse_claravy(self) -> None:

        with open(self.claravy_cache, "r") as fp:
            for line in fp:
                item = Item.from_tool_output_line(line)
                label = item.filter(self.filter_args)
                self.data[item.sha] = label

    def parse_avclass(self) -> None:

        with open(self.avclass_cache, "r") as fp:
            for line in fp:
                item = Item.from_tool_output_line(line)
                label = item.filter(self.filter_args)
                if item.sha not in self.data:
                    self.data[item.sha] = Label()
                self.data[item.sha].class_ = label.class_

    def parse_avclass_family(self) -> None:

        # TODO: There is a bug here which results in some samples being
        # unlabeled when they do in fact have a label. Concretely, the first
        # branch does not set the sha argument, so the sha from the previous
        # iteration of the for loop is used and its family is set to None.
        if os.environ.get("USE_UPDATED_AVCLASS_FAMILY_PARSER") != "1":
            warnings.warn(
                "Using a variant of parse_avclass_family that has a small bug in it for backwards compatibility. "
                f"{os.environ.get('USE_UPDATED_AVCLASS_FAMILY_PARSER')=}."
                "Use USE_UPDATED_AVCLASS_FAMILY_PARSER=1 to use the updated version."
            )

        with open(self.avclass_family_cache, "r") as fp:
            for line in fp:
                line = re.sub(r'\s+', ' ', line).strip()
                args = line.split()
                if len(args) != 2:
                    if args[1:] == ["-", "[]"]:
                        if os.environ.get("USE_UPDATED_AVCLASS_FAMILY_PARSER") == "1":
                            sha = args[0]
                        fam = None
                    else:
                        raise ValueError(f"Unexpected line: {line=}")

                else:
                    sha, fam = line.split()
                    if "SINGLETON" in fam:
                        fam = None

                if sha not in self.data:
                    self.data[sha] = Label()
                if fam is not None:
                    self.data[sha].fam = (fam,)

    def view(self, name: str) -> dict[str, tuple[str]]:
        return {sha: getattr(self.data[sha], name) for sha in self.data.keys()}


def main():

    parser = ArgumentParser()
    parser.add_argument("--reports_dir", type=str, required=False, help="Directory containing the (.jsonl) reports.")
    parser.add_argument("--claravy_cache", type=str, required=True, help="File (.txt) to store the output of Claravy.")
    parser.add_argument("--avclass_cache", type=str, required=True, help="File (.txt) to store the output of AVClass.")
    parser.add_argument("--avclass_family_cache", type=str, required=True, help="File (.txt) to store the output of AVClass-family.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--run_tool_runner", action="store_true")
    parser.add_argument("--run_labeler", action="store_true")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir) if args.reports_dir is not None else None
    claravy_cache = Path(args.claravy_cache)
    avclass_cache = Path(args.avclass_cache)
    avclass_family_cache = Path(args.avclass_family_cache)

    if args.clean:
        print("To remove old caches, run:\n")
        print(f"\trm {claravy_cache.as_posix()}")
        print(f"\trm {avclass_cache.as_posix()}")
        print(f"\trm {avclass_family_cache.as_posix()}")
        sys.exit(0)

    if args.run_tool_runner:
        runner = ToolRunner(
            reports_dir,
            claravy_cache,
            avclass_cache,
            avclass_family_cache,
        )
        runner = runner()

    if args.run_labeler:
        filter_args = FilterArgs(
            class_=(None, 2),
            file=(1, 2),
            fam=(1, 2),
            beh=(None, 2),
            unk=(None, 2),
            pack=(None, 1),
            vuln=(None, 1),
        )
        labeler = Labeler(
            claravy_cache,
            avclass_cache,
            avclass_family_cache,
            filter_args,
            args.use_cache,
        )
        labeler = labeler()


if __name__ == "__main__":
    main()
