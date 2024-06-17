"""
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
import re
import subprocess
import sys
import time
from typing import Optional


AVCLASS_EXE = Path("/home/lk3591/anaconda3/envs/MalwareLabeler/bin/avclass")
CLARAVY_EXE = Path("/home/lk3591/anaconda3/envs/MalwareLabeler/bin/claravy")


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


@dataclass(frozen=True)
class FilterArgs:
    class_: tuple[int, int] = (1, 5)
    file: tuple[int, int] = (1, 5)
    fam: tuple[int, int] = (1, 2)
    beh: tuple[int, int] = (1, 5)
    unk: tuple[int, int] = (1, 2)
    pack: tuple[int, int] = (1, 1)
    vuln: tuple[int, int] = (1, 1)


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
        args = s.split()
        sha = args[0]
        flagged = int(args[1]) if args[1].isdigit() else 0
        if len(args) == 3:
            information = args[2]
        else:
            information = ""

        if information in ("[]", "") or "SINGLETON" in information:
            return Item(sha, flagged)

        labels = defaultdict(list)
        for piece in information.split(","):

            field = piece[0:piece.index(":")]
            value = piece[piece.index(":") + 1:piece.index("|")]
            count = int(piece[piece.index("|") + 1:])
            labels[field].append((value, count))

        labels = {k.lower(): v for k, v in labels.items()}
        if "class" in labels:
            labels["class_"] = labels.pop("class")
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


class Labeler:

    def __init__(
        self,
        reports_dir: Path,
        claravy_cache: Path,
        avclass_cache: Path,
        avclass_family_cache: Path,
        filter_args: Optional[FilterArgs] = FilterArgs(),
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.avclass_cache = Path(avclass_cache)
        self.avclass_family_cache = Path(avclass_family_cache)
        self.claravy_cache = Path(claravy_cache)
        self.filter_args = filter_args
        self.data: dict[str, Label] = {}

    def __call__(self) -> Labeler:
        print(f"Labeler for {self.reports_dir.as_posix()}")

        print("Running CLARAVY...")
        t_0 = time.time()
        if not self.claravy_cache.exists():
            self.run_claravy()
        print(f"Running CLARAVY took {time.time() - t_0:.2f} seconds")

        print("Running AVCLASS...")
        t_0 = time.time()
        if not self.avclass_cache.exists():
            self.run_avclass()
        print(f"Running AVCLASS took {time.time() - t_0:.2f} seconds")

        print("Running AVCLASS-family...")
        t_0 = time.time()
        if not self.avclass_family_cache.exists():
            self.run_avclass_family()
        print(f"Running AVCLASS took {time.time() - t_0:.2f} seconds")

        print("Parsing CLARAVY...")
        t_0 = time.time()
        self.parse_claravy()
        print(f"Parsing CLARAVY took {time.time() - t_0:.2f} seconds")

        t_0 = time.time()
        print("Parsing AVCLASS...")
        self.parse_avclass()
        print(f"Parsing AVCLASS took {time.time() - t_0:.2f} seconds")

        t_0 = time.time()
        print("Parsing AVCLASS-family...")
        self.parse_avclass_family()
        print(f"Parsing AVCLASS-family took {time.time() - t_0:.2f} seconds")

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


        # def get_args(f: Path, o: Path):
        #     return [
        #         f"{AVCLASS_EXE.as_posix()}",
        #         f"-f={f}",
        #         f"-o={o}",
        #         "-hash=sha256",
        #         "-t",
        #     ]

        # processes: list[subprocess.Popen] = []

        # for f in sorted(self.reports_dir.iterdir()):
        #     while len(processes) > NUM_WORKERS:
        #         for i, p in enumerate(processes):
        #             if p.poll() is not None:
        #                 if p != 0:
        #                     raise subprocess.CalledProcessError(p.returncode, p.args)
        #                 processes[i] = None
        #         processes = [p for p in processes if p is not None]

        #     f_out = Path("/tmp/avclass/") / f.name

        #     args = get_args(f, f_out)
        #     p = subprocess.Popen(args)
        #     r = None
        #     processes.append([p, r])

        # self.avclass_cache.as_posix()


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

        with open(self.avclass_family_cache, "r") as fp:
            for line in fp:
                line = re.sub(r'\s+', ' ', line).strip()
                args = line.split()
                if len(args) != 2:
                    if args[1:] == ["-", "[]"]:
                        fam = None
                    else:
                        raise ValueError(f"Unexpected line: {line=}")

                else:
                    sha, fam = line.split()
                    if "SINGLETON" in fam:
                        fam = None

                if sha not in self.data:
                    self.data[sha] = Label()
                self.data[sha].fam = fam

    def normalize_whitespace(text):
        # Replace all sequences of whitespace with a single space
        return re.sub(r'\s+', ' ', text).strip()

    def view(self, name: str, shas: Optional[list[str]] = None) -> tuple[list[str], list[str]]:
        shas = list(self.data.keys()) if shas is None else shas
        return shas, [getattr(self.data[sha], name) for sha in shas]


def main():

    parser = ArgumentParser()
    parser.add_argument("--reports_dir", type=str, required=True)
    parser.add_argument("--claravy_cache", type=str, required=True)
    parser.add_argument("--avclass_cache", type=str, required=True)
    parser.add_argument("--avclass_family_cache", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--do_anal", action="store_true")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    claravy_cache = Path(args.claravy_cache)
    avclass_cache = Path(args.avclass_cache)
    avclass_family_cache = Path(args.avclass_family_cache)

    if args.clean:
        print("To remove old caches, run:\n")
        print(f"\trm {claravy_cache.as_posix()}")
        print(f"\trm {avclass_cache.as_posix()}")
        print(f"\trm {avclass_family_cache.as_posix()}")
        sys.exit(0)
        # claravy_cache.unlink(missing_ok=True)
        # avclass_cache.unlink(missing_ok=True)
        # avclass_family_cache.unlink(missing_ok=True)

    filter_args = FilterArgs()

    labeler = Labeler(
        reports_dir,
        claravy_cache,
        avclass_cache,
        avclass_family_cache,
        filter_args,
    )()

    if args.do_anal:
        for k in KEYS:
            print("-" * 40 + f" {k} " + "-" * 40)
            shas, values = labeler.view(k)
            counter = Counter([v[0] if isinstance(v, tuple) else v for v in values])
            pprint(counter)


if __name__ == "__main__":
    main()
