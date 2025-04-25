"""
Conduct analysis to compare the attributions of different XAI methods and algorithms.
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable, Generator
from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pprint, pformat
import queue
import sys
import threading
import time
from typing import Optional, NamedTuple, Callable, Literal
import warnings

import numpy as np
from scipy import stats
import torch
from torch import Tensor
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.enums import ExplanationMethod, ExplanationAlgorithm, Task
from src.utils import torch_safe_downcast
from src.learn.helpers import OutputHelper
from src.attribute.statistical import compute_agreement, DescriptiveSparsity


class Annotations(NamedTuple):
    """
    Container to score an XAI method's attributions for each interpretable feature in a sample.

    Args:
        name: The name of the sample, i.e., its sha256 hash.
        label: The label of the sample, i.e., its class(es).
        scores: The score assigned to each interpretable feature in the sample.
        ranks: The rank assigned to each interpretable feature in the sample.

    The index of the score/rank corresponds to the interpretable feature's index, e.g.,
    if interpretable features correspond to functions, scores[0] is the score of the first function
    in the sample and scores[-1] is the score of the last function in the sample.

    Higher scores and higher ranks indicate greater importance.

    We chose to name the variable `scores` instead of `attribs` because there is a single score
    for each interpretable feature, whereas the `attribs` contains a score for each feature.
    """

    name: str
    label: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray


AnnotationStream = Generator[Annotations, None, None]


class AnnotationStreamConstructor:
    """
    Contains staticmethods to construct an `AnnotationStream` from various sources.
    """

    @staticmethod
    def from_iterables(names: Iterable[str], labels: Iterable[Tensor], attribs: Iterable[Tensor], masks: Iterable[Tensor]) -> AnnotationStream:
        for name, label, attrib, mask in zip(names, labels, attribs, masks):
            idx    = mask != 0
            attrib = attrib[idx]
            mask   = mask[idx]

            unq, idx = torch.unique(mask, return_inverse=True)
            scores = torch.zeros_like(unq, dtype=attrib.dtype)
            scores.scatter_reduce_(0, idx, attrib, reduce="amax")

            scores = scores.numpy(force=True)
            ranks  = stats.rankdata(scores)
            label  = label.numpy(force=True)

            yield Annotations(name, label, scores, ranks)

    @staticmethod
    def from_files(names: Path | Iterable[Path], labels: Path | Iterable[Path], attribs: Path | Iterable[Path], masks: Path | Iterable[Path]) -> AnnotationStream:
        return AnnotationStreamConstructor.from_files_multithread(names, labels, attribs, masks)

    @staticmethod
    def from_files_multithread(names: Path | Iterable[Path], labels: Path | Iterable[Path], attribs: Path | Iterable[Path], masks: Path | Iterable[Path]) -> AnnotationStream:
        names = [names] if isinstance(names, Path) else names
        labels = [labels] if isinstance(labels, Path) else labels
        attribs = [attribs] if isinstance(attribs, Path) else attribs
        masks = [masks] if isinstance(masks, Path) else masks

        q = queue.Queue(maxsize=8)
        done_event = threading.Event()

        def producer():
            for p_name, p_label, p_attrib, p_mask in zip(names, labels, attribs, masks, strict=True):
                text_names = p_name.read_text().splitlines()
                data_labels = torch.load(p_label, map_location="cpu")
                data_attribs = torch.load(p_attrib, map_location="cpu")
                data_masks = torch.load(p_mask, map_location="cpu")
                q.put((text_names, data_labels, data_attribs, data_masks))

            done_event.set()

        threading.Thread(target=producer, daemon=True).start()

        while not done_event.is_set() or not q.empty():
            try:
                text_names, data_labels, data_attribs, data_masks = q.get(timeout=0.1)
                yield from AnnotationStreamConstructor.from_iterables(text_names, data_labels, data_attribs, data_masks)
                q.task_done()
            except queue.Empty:
                continue

    @staticmethod
    def from_files_singlethread(names: Path | Iterable[Path], labels: Path | Iterable[Path], attribs: Path | Iterable[Path], masks: Path | Iterable[Path]) -> AnnotationStream:
        names = [names] if isinstance(names, Path) else names
        labels = [labels] if isinstance(labels, Path) else labels
        attribs = [attribs] if isinstance(attribs, Path) else attribs
        masks = [masks] if isinstance(masks, Path) else masks

        for p_name, p_label, p_attrib, p_mask in zip(names, labels, attribs, masks, strict=True):
            names = p_name.read_text().splitlines()
            labels = torch.load(p_label, map_location="cpu")
            attribs = torch.load(p_attrib, map_location="cpu")
            masks = torch.load(p_mask, map_location="cpu")

            yield from AnnotationStreamConstructor.from_iterables(names, labels, attribs, masks)

    @staticmethod
    def from_attribution_path(path: Path) -> AnnotationStream:
        names = path / "names.txt"
        names = OutputHelper.get_attribution_data_files(names, None) if not names.exists() else names
        labels = path / "labels.pt"
        labels = OutputHelper.get_attribution_data_files(labels, None) if not labels.exists() else labels
        attribs = path / "attribs.pt"
        attribs = OutputHelper.get_attribution_data_files(attribs, None) if not attribs.exists() else attribs
        masks = path / "masks.pt"
        masks = OutputHelper.get_attribution_data_files(masks, None) if not masks.exists() else masks

        for i, f in enumerate([names, labels, attribs, masks]):
            if isinstance(f, list) and len(f) == 0:
                print(f"{i=}")
                raise FileNotFoundError(path)
            if isinstance(f, Path) and not f.exists():
                raise FileNotFoundError(f)

        yield from AnnotationStreamConstructor.from_files(names, labels, attribs, masks)


class Attributions(NamedTuple):
    """
    Container to score an XAI method's raw attributions for each interpretable feature in a sample.

    Unlike `Annotations`, this container does not contain the ranks or the scores. Instead, this contains
      the raw attributions and the raw mask used to produce the attributions. Each of these structures
      will have the same length as the input_ids passed to the network, whereas the ranks and scores from
        `Annotations` will have the same length as the number of interpretable features.
    """
    name:   str
    label:  Tensor
    attrib: Tensor
    mask:   Tensor


class AttributionPathManager:
    """
    Manages the paths to the attribution data files and provides methods to access the data.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

        self.names_files   = self._attain_files(path / "names.txt")
        self.labels_files  = self._attain_files(path / "labels.pt")
        self.attribs_files = self._attain_files(path / "attribs.pt")
        self.masks_files   = self._attain_files(path / "masks.pt")
        self.scores_files  = self._attain_files(path / "scores.pt")
        self.ranks_files   = self._attain_files(path / "ranks.pt")

        assert len(self.names_files) == len(self.labels_files) == len(self.attribs_files) == len(self.masks_files)
        assert len(self.scores_files) == len(self.ranks_files)

        self.file_id_to_ids   = defaultdict(set)
        self.file_id_to_names = defaultdict(set)
        self.id_to_name = {}
        self.name_to_id = {}
        i = 0
        for j, names_file in enumerate(self.names_files):
            for name in names_file.read_text().splitlines():
                self.file_id_to_ids[j].add(i)
                self.file_id_to_names[j].add(name)
                self.id_to_name[i] = name
                self.name_to_id[name] = i
                i += 1

    def __len__(self):
        return len(self.id_to_name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        msg = ""
        msg += "-" * 80 + "\n"
        msg += "AttributionPathManager\n"
        msg += f"  path: {self.path}\n"
        msg += f"  groups: {len(self.names_files)}\n"
        msg += f"  samples: {len(self.id_to_name)}\n"
        msg += "-" * 80
        return msg

    def _attain_files(self, file: Path) -> list[Path]:
        if file.exists():
            return [file]
        return OutputHelper.get_attribution_data_files(file, None)

    def _name_and_id_from_name_or_id(self, name_or_id: str | int) -> tuple[str, int]:
        if isinstance(name_or_id, str):
            name = name_or_id
            id_   = self.name_to_id[name_or_id]
        else:
            name = self.id_to_name[name_or_id]
            id_   = name_or_id
        return name, id_

    def _id_to_file_id(self, i: int) -> int:
        for j, ids in self.file_id_to_ids.items():
            if i in ids:
                return j
        raise ValueError(f"Could not find the file id for {i}.")

    def get_attributions(self, name_or_id: str | int) -> Attributions:
        name, id_ = self._name_and_id_from_name_or_id(name_or_id)
        file_id  = self._id_to_file_id(id_)

        names   = self.names_files[file_id].read_text().splitlines()
        labels  = torch.load(self.labels_files[file_id], map_location="cpu")
        attribs = torch.load(self.attribs_files[file_id], map_location="cpu")
        masks   = torch.load(self.masks_files[file_id], map_location="cpu")

        idx = names.index(name)
        return Attributions(names[idx], labels[idx], attribs[idx], masks[idx])

    def get_annotations(self, name_or_id: str | int) -> Annotations:
        if len(self.ranks_files) == 0:
            raise ValueError("Ranks have not been generated. Use `generate_ranks` to generate them first.")

        name, id_ = self._name_and_id_from_name_or_id(name_or_id)
        file_id  = self._id_to_file_id(id_)

        names   = self.names_files[file_id].read_text().splitlines()
        labels  = torch.load(self.labels_files[file_id], map_location="cpu")
        scores  = torch.load(self.scores_files[file_id], map_location="cpu")
        ranks   = torch.load(self.ranks_files[file_id], map_location="cpu")

        idx = names.index(name)
        labels = labels[idx].numpy(force=True)
        scores = scores[idx].numpy(force=True)
        ranks  = ranks[idx].numpy(force=True)

        return Annotations(name, labels, scores, ranks)

    def generate_ranks(self, disable_tqdm: bool = False, verbose: bool = True) -> AttributionPathManager:

        def save_batch_and_clear_structures_(file_id: int, names: list[str], ranks: list[Tensor], scores: list[Tensor]):
            # Make sure that every sample expected to be in this group is accounted for. Then save the ranks
            # and scores to the appropriate files. Finally, clear the lists to prepare for the next group.
            if set(names) != self.file_id_to_names[file_id]:
                raise ValueError(f"Names do not match for file {file_id}.")
            name_file  = self.names_files[file_id]
            rank_file  = name_file.parent / name_file.name.replace("names", "ranks").replace(".txt", ".pt")
            score_file = name_file.parent / name_file.name.replace("names", "scores").replace(".txt", ".pt")
            torch.save(ranks, rank_file)
            torch.save(scores, score_file)
            names.clear()
            ranks.clear()
            scores.clear()

        names  = []
        ranks  = []
        scores = []
        file_id = 0
        iterable = AnnotationStreamConstructor.from_attribution_path(self.path)
        iterable = iterable if disable_tqdm else tqdm(iterable, total=len(self.id_to_name), desc="Generating ranks",  leave=False)

        if verbose:
            print(f"{os.getpid()} generating ranks...", flush=True)
            t_i = time.time()

        for annotation in iterable:
            name  = annotation.name
            rank  = torch_safe_downcast(torch.from_numpy(annotation.ranks))
            score = torch_safe_downcast(torch.from_numpy(annotation.scores))

            # Checks if the next name belongs to a different group of outputs,
            # in which case, we save and move on to the next group.
            if (fid := self._id_to_file_id(self.name_to_id[name])) != file_id:
                save_batch_and_clear_structures_(file_id, names, ranks, scores)
                file_id = fid

            names.append(name)
            ranks.append(rank)
            scores.append(score)

        save_batch_and_clear_structures_(file_id, names, ranks, scores)

        if verbose:
            t_f = time.time()
            print(f"{os.getpid()} finished generating ranks ({round(t_f - t_i)} seconds).", flush=True)

        # The object now has access to the ranks and scores files.
        self.ranks_files  = self._attain_files(self.path / "ranks.pt")
        self.scores_files = self._attain_files(self.path / "scores.pt")

        return self

    def safe_downcast(self, num_workers: Optional[int] = 0, disable_tqdm: bool = False, verbose: bool = True) -> AttributionPathManager:
        files = self.labels_files + self.attribs_files + self.masks_files + self.scores_files + self.ranks_files

        if verbose:
            print(f"{os.getpid()} safe downcasting with {num_workers} workers...", flush=True)
            t_i = time.time()

        safe_downcast_files(files, num_workers=num_workers, disable_tqdm=disable_tqdm, verbose=verbose)

        if verbose:
            t_f = time.time()
            print(f"{os.getpid()} finished safe downcasting ({round(t_f - t_i)} seconds).", flush=True)

        return self

    @property
    def names(self) -> list[str]:
        names = []
        for f in self.names_files:
            data = f.read_text().splitlines()
            names.extend(data)
        return np.array(names)

    @property
    def ranks(self) -> list[np.ndarray]:
        ranks = []
        for f in self.ranks_files:
            data: list[Tensor] = torch.load(f, map_location="cpu")
            data = [d.numpy(force=True).astype(np.float32) for d in data]
            ranks.extend(data)
        return ranks

    @property
    def scores(self) -> list[np.ndarray]:
        scores = []
        for f in self.scores_files:
            data: list[Tensor] = torch.load(f, map_location="cpu")
            data = [d.numpy(force=True).astype(np.float32) for d in data]
            scores.extend(data)
        return scores

    @property
    def has_names_files(self) -> bool:
        return len(self.names_files) > 0

    @property
    def has_ranks_files(self) -> bool:
        return len(self.ranks) == len(self.names)

    @property
    def has_scores_files(self) -> bool:
        return len(self.scores) == len(self.names)

    def descriptive_sparsity(self, n_bins: int = 100, n_points: int = 100, num_workers: int = 0) -> np.ndarray:
        # M = np.full((len(self), n_points), np.nan, dtype=np.float32)
        # for i, s in enumerate(self.scores):
        #     n_bins = min(n_bins, len(s))
        #     m = DescriptiveSparsity(n_bins=n_bins, n_points=n_points)(s)
        #     M[i] = m
        # return M
        func = DescriptiveSparsity(n_bins=n_bins, n_points=n_points)
        if num_workers == 0:
            M = [func(s) for s in self.scores]
        else:
            with mp.Pool(num_workers) as pool:
                M = pool.map(func, self.scores)
        M = np.stack(M, axis=0)
        return M


def safe_downcast_file(f: Path) -> tuple[bool, int, float]:
    t_i = time.time()
    s_i = f.stat().st_size

    data: list[Tensor] = torch.load(f, map_location="cpu")
    if not isinstance(data, list) or not isinstance(data[0], Tensor) or data[0].ndim != 1:
        raise ValueError(f"Expected a list of 1-D tensors, but got {type(data)=} {data=}.")

    newdata = [torch_safe_downcast(d) for d in data]

    downcast = False
    if any(d1.dtype != d2.dtype for d1, d2 in zip(data, newdata)):
        downcast = True
        torch.save(newdata, f)

    s_f = f.stat().st_size
    t_f = time.time()

    return downcast, s_i - s_f, t_f - t_i

def safe_downcast_files(files: list[Path], num_workers: Optional[int] = 0, disable_tqdm: bool = False, verbose: bool = False) -> None:
    iterable = tqdm(files, desc="Safe Downcasting...") if not disable_tqdm else files

    iterable = files

    t_i = time.time()

    if num_workers == 0:
        results = []
        for f in iterable:
            r = safe_downcast_file(f)
            results.append(r)
    else:
        with mp.Pool(num_workers) as pool:
            results = list(pool.imap(safe_downcast_file, iterable))

    t_f = time.time()

    if verbose:
        n_downcast = sum(1 for r in results if r[0])
        n_bytes = sum(r[1] for r in results)
        n_mbytes = round(n_bytes / (1024 * 1024))
        n_time = round(t_f - t_i)
        print(f"Downcasted {n_downcast} / {len(files)} files, saved ({n_mbytes}MB), took {n_time} seconds.", flush=True)


class StatisticalSummary(NamedTuple):
    mean: float
    error: float
    medn: float
    stdv: float
    min_: float
    max_: float
    support: int


def compute_statistical_summary(x: np.ndarray, alpha: float = 0.05) -> StatisticalSummary:
    support = len(x)
    mean = np.mean(x)
    medn = np.median(x)
    stdv = np.std(x, ddof=1)
    min_  = np.min(x)
    max_  = np.max(x)
    t_crit = stats.t.ppf(1 - alpha / 2, support - 1)
    error  = t_crit * (stdv / np.sqrt(support))
    return StatisticalSummary(mean, error, medn, stdv, min_, max_, support)


class AgreementCoordinator:
    """
    Manages the agreement computations between several different annotators.

    Attributes:
      ranks: A list of rank matrices gathered from the input paths. Each rank matrix
        corresponds to a sample. Each row in the matrix corresponds to an interpretable
        feature. Each column corresponds to a judge. In other words, ranks[k,i,j] is the
        rank of the i-th interpretable feature in the k-th sample by the j-th judge.
    """

    def __init__(self, paths: list[Path], judge_names: list[str], remove_incongruent_samples: bool = False, tolerance: float = 0.0) -> None:
        self.paths = paths
        self.judge_names = judge_names
        self.remove_incongruent_samples = remove_incongruent_samples
        self.tolerance = tolerance
        self.managers = [AttributionPathManager(p) for p in paths]
        self.name_to_idx: dict[str, int] = None
        self.idx_to_name: dict[int, str] = None
        self.I: int = None                   # Number of samples.
        self.J: int = len(self.managers)     # Number of judges.
        self.K: np.ndarray = None            # Number of interpretable features per sample.
        self.ranks: list[np.ndarray] = None  # Shape: (I, K, J)

        if len(paths) == 0:
            raise ValueError("No paths provided.")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Path {p} does not exist.")

        if len(judge_names) != len(paths):
            raise ValueError("Number of judge names must match the number of paths.")
        if not all(isinstance(name, str) for name in judge_names):
            raise ValueError("Judge names must be strings.")

    def __call__(self, remove_cachefiles: bool = False) -> AgreementCoordinator:
        self.determine_samples()
        print(f"Determined {self.I} samples for which all {self.J} judges have ranked.")
        if remove_cachefiles:
            self.remove_cachefiles()
        self.create_rank_matrices()
        print(f"Created rank matrices with between {self.K.min()} and {self.K.max()} interpretable features.")
        for judges in self.judge_groups():
            _, _ = self.compute_per_sample_agreement(judges)
            stat = self.compute_agreement_statistic(judges)
            print(f"{stat.mean:.5f} +/ {stat.error:.5f} N={stat.support} ({self.judge_subset(judges)})")
        return self

    def judge_subset(self, judges: np.ndarray) -> str:
        return " ".join(sorted(np.array(self.judge_names)[judges].tolist()))

    def judge_groups(self) -> list[np.ndarray]:
        groups = product([False, True], repeat=len(self.managers))
        groups = filter(lambda judges: sum(judges) > 1, groups)
        groups = sorted(groups, key=lambda judges: sum(judges), reverse=False)  # pylint: disable=unnecessary-lambda
        groups = map(np.array, groups)
        return list(groups)

    def get_cachefile(self, judges: np.ndarray) -> Path:
        """
        Returns a unique cachefile considering the names of samples and the experiment paths indicated by judges.
        """
        data_1 = "".join(self.name_to_idx.keys())
        hash_1 = hashlib.blake2s(data_1.encode()).hexdigest()
        data_2 = "".join(manager.path.as_posix() for judge, manager in zip(judges, self.managers) if judge)
        hash_2 = hashlib.blake2s(data_2.encode()).hexdigest()
        cachefile = Path(f"./cache/attribute/agreement--{hash_1}--{hash_2}.npy")
        return cachefile

    def remove_cachefiles(self) -> None:
        for judges in self.judge_groups():
            cachefile = self.get_cachefile(judges)
            if cachefile.exists():
                os.remove(cachefile)
                print("Removed cachefile:", cachefile)

    def determine_samples(self) -> AgreementCoordinator:
        for j, manager in enumerate(self.managers):
            names = set(manager.names)
            if j == 0:
                allnames = names
            if names != allnames:
                allnames = allnames.intersection(names)
                warnings.warn("Not all experiments contain data for the same samples.")

        allnames = sorted(allnames)
        self.idx_to_name = {i: name for i, name in enumerate(allnames)}  # pylint: disable=unnecessary-comprehension
        self.name_to_idx = {name: i for i, name in enumerate(allnames)}
        self.I = len(allnames)
        self.K = np.full((self.I,), -1, dtype=np.int32)

    def create_rank_matrices(self) -> AgreementCoordinator:
        self.ranks = [None for _ in range(self.I)]

        incongruent = []
        for j, manager in enumerate(self.managers):
            for name, rank in zip(manager.names, manager.ranks):
                if (i := self.name_to_idx.get(name)) is not None:
                    if j == 0:
                        self.K[i] = len(rank)
                        self.ranks[i] = np.full((self.K[i], self.J), np.nan, np.float32)
                        self.ranks[i][:,j] = rank
                    else:
                        if self.K[i] != len(rank):
                            incongruent.append((i, j, len(rank), name))
                            continue
                        self.ranks[i][:,j] = rank

        if incongruent:
            for i, j, k, name in incongruent:
                print(f"Sample {i} ({name}) was found to be incongruent with judge {j} who ranked {k} interpretable features when {self.K[i]} were expected.")
            if not self.remove_incongruent_samples:
                raise RuntimeError("Incongruent samples were detected. Set `remove_incongruent_samples` to True to remove them.")
            remove = tuple(set(i for i, _, _, _ in incongruent))
            print(f"Removing incongruent {len(remove)} samples.")
            self.ranks = [r for i, r in enumerate(self.ranks) if i not in remove]
            self.I = len(self.ranks)
            self.K = np.delete(self.K, np.array(remove))

        for i in range(self.I):
            if np.any(np.isnan(self.ranks[i])):
                raise RuntimeError(f"Rank matrix {i} contains NaN values.")
            if np.any(np.isinf(self.ranks[i])):
                raise RuntimeError(f"Rank matrix {i} contains inf values.")
            if self.ranks[i].shape != (self.K[i], self.J):
                raise RuntimeError(f"Rank matrix {i} has shape {self.ranks[i].shape}, expected ({self.K[i]}, {self.J}).")

    def compute_per_sample_agreement(self, judges: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        if judges is None:
            judges = np.full((len(self.managers),), True)

        cachefile = self.get_cachefile(judges)
        if cachefile.exists():
            W, P = np.load(cachefile)
            assert W.shape == (self.I,) and P.shape == (self.I,)

        W = np.empty((self.I,))
        P = np.empty((self.I,))
        for i in range(self.I):
            R = self.ranks[i]
            R = R[:,judges]
            w, p = compute_agreement(R, self.tolerance)

            W[i] = w
            P[i] = p

        np.save(cachefile, (W, P))

        return W, P

    def compute_agreement_statistic(self, judges: Optional[np.ndarray] = None) -> StatisticalSummary:
        W, P = self.compute_per_sample_agreement(judges)
        idx = np.isnan(W)
        W = W[~idx]
        P = P[~idx]
        return compute_statistical_summary(W)


class AttributionConfiguration:

    last_path_attribute: str = "bf16"

    def __init__(self, root: Optional[Path], seed: int, xai_method: ExplanationMethod, xai_algorithm: ExplanationAlgorithm, xai_chunk_size: Optional[int], xai_seed: int) -> None:
        self.root = root
        self.seed = seed
        self.xai_method = xai_method
        self.xai_algorithm = xai_algorithm
        self.xai_chunk_size = xai_chunk_size
        self.xai_seed = xai_seed

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(None, {self.seed}, {self.xai_method.value}, {self.xai_algorithm.value}, {self.xai_chunk_size}, {self.xai_seed})"

    def ensure_root_valid(self) -> None:
        if self.root is None:
            raise RuntimeError("Root path is not set.")
        if self.root.name.split("--")[0] != AttributionConfiguration.last_path_attribute:
            raise ValueError(f"Root directory {self.root.name} does not match the expected name {AttributionConfiguration.last_path_attribute}.")

    @property
    def path(self) -> Path:
        self.ensure_root_valid()
        return (
            self.root /
            f"seed--{self.seed}" /
            "results" /
            "attributions" /
            f"xai_method--{self.xai_method.value}" /
            f"xai_algorithm--{self.xai_algorithm.value}" /
            f"xai_chunk_size--{self.xai_chunk_size if self.xai_chunk_size is not None else 'none'}" /
            f"xai_seed--{self.xai_seed}"
        )

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()

    @property
    def empty(self) -> bool:
        if not self.exists:
            return FileNotFoundError(self.path)
        if not self.is_dir:
            return NotADirectoryError(self.path)
        try:
            next(self.path.glob("*"))
        except StopIteration:
            return True
        return False

    @classmethod
    def from_path(cls, path: Path) -> AttributionConfiguration:
        root_parts = []
        parse_root = True
        for p in path.parts:
            if parse_root:
                root_parts.append(p)
                if p.split("--")[0] == AttributionConfiguration.last_path_attribute:
                    parse_root = False
                continue

            if p.split("--")[0] == "seed":
                seed = int(p.split("--")[1])
            elif p.split("--")[0] == "xai_method":
                xai_method = ExplanationMethod(p.split("--")[1])
            elif p.split("--")[0] == "xai_algorithm":
                xai_algorithm = ExplanationAlgorithm(p.split("--")[1])
            elif p.split("--")[0] == "xai_chunk_size":
                xai_chunk_size = int(p.split("--")[1]) if p.split("--")[1] != "none" else None
            elif p.split("--")[0] == "xai_seed":
                xai_seed = int(p.split("--")[1])

        root = Path(*root_parts)
        if not root.exists():
            raise RuntimeError("Incorrectly parsed root path.")

        try:
            return cls(root, seed, xai_method, xai_algorithm, xai_chunk_size, xai_seed)
        except NameError as err:
            raise ValueError(f"Could not parse the {err.name} attribute from the path {path}.") from err  # pylint: disable=no-member


ConfigurationPredicate = Callable[[AttributionConfiguration], bool]


class ConfigurationFilter:

    def __init__(
        self,
        f_seed: ConfigurationPredicate = lambda c: True,
        f_method: ConfigurationPredicate = lambda c: True,
        f_algorithm: ConfigurationPredicate = lambda c: True,
        f_chunk_size: ConfigurationPredicate = lambda c: True,
        f_xai_seed: ConfigurationPredicate = lambda c: True,
    ) -> None:
        self.f_seed = f_seed
        self.f_method = f_method
        self.f_algorithm = f_algorithm
        self.f_chunk_size = f_chunk_size
        self.f_xai_seed = f_xai_seed

    def __call__(self, config: AttributionConfiguration) -> bool:
        return (
            self.f_seed(config) and
            self.f_method(config) and
            self.f_algorithm(config) and
            self.f_chunk_size(config) and
            self.f_xai_seed(config)
        )


def main():

    base = Path("/home/lk3591/Documents/code/RawByteClf/output/esp-exe/")
    experiments: list[AttributionConfiguration] = []
    for root in sorted(base.rglob("xai_seed--*")):
        config = AttributionConfiguration.from_path(root)
        if not config.exists:
            print("Skipping non-existing path:", root)
            continue
        if config.empty:
            print("Skipping empty path:", root)
            continue
        experiments.append(AttributionConfiguration.from_path(root))

    # Generate ranks.
    # roots = [e.path for e in experiments]
    # for root in tqdm(roots, desc="Generating Ranks..."):
    #     manager = AttributionPathManager(root)
    #     if manager.has_ranks_files and manager.has_scores_files:
    #         continue
    #     manager = manager.generate_ranks(disable_tqdm=False, verbose=False)

    # Generate masks.
    # masks_files = []
    # roots = [e.path for e in experiments]
    # for root in roots:
    #     manager = AttributionPathManager(root)
    #     masks_files.extend(manager.masks_files)
    # safe_downcast_files(masks_files, num_workers=0, disable_tqdm=False, verbose=True)

    # Descriptive Sparsity.
    for e in experiments:
        manager = AttributionPathManager(e.path)
        if not (manager.has_ranks_files and manager.has_scores_files):
            continue
        M = manager.descriptive_sparsity(n_bins=100, n_points=100)
        stat = compute_statistical_summary(np.mean(M, axis=1))
        print(f"{e}: {stat.mean:.5f} +/ {stat.error:.5f} N={stat.support}")

    # Compute the agreement.
    # configutation_filter = ConfigurationFilter(
    #     lambda c: c.seed in [0, 1, 2, 3, 4],
    #     lambda c: c.xai_method == ExplanationMethod.FUN,
    #     lambda c: c.xai_algorithm in [ExplanationAlgorithm.GSHP],
    #     lambda c: c.xai_chunk_size is None,
    #     lambda c: c.xai_seed == 0,
    # )
    # exps: list[AttributionConfiguration] = list(filter(configutation_filter, experiments))
    # for i, e in enumerate(exps):
    #     print(f"{i}: {e}")
    # roots = [e.path for e in exps]
    # judge_names = [str(e.seed) for e in exps]
    # coordinator = AgreementCoordinator(roots, judge_names, True)
    # coordinator = coordinator()


if __name__ == "__main__":
    main()
