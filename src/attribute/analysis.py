"""
Conduct analysis to compare the attributions of different XAI methods and algorithms.
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable, Generator
from collections import defaultdict, Counter
from copy import deepcopy
from functools import partial
import hashlib
from itertools import chain, product
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
from src.attribute.masking import apply_feature_mask
from src.attribute.segtensor import SegmentedTensor
from src.attribute.statistical import AgreementMethod, AgreementFunction, get_agreement_function, descriptive_sparsity
from src.attribute.utils import ignore_warnings_decorator


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
    def from_iterables(names: Iterable[str], labels: Iterable[Tensor | SegmentedTensor], attribs: Iterable[Tensor | SegmentedTensor], masks: Iterable[Tensor | SegmentedTensor]) -> AnnotationStream:
        for name, label, attrib, mask in zip(names, labels, attribs, masks):
            if isinstance(label, SegmentedTensor):
                raise RuntimeError("Label is a SegmentedTensor, but we were expected a Tensor.")
            if isinstance(attrib, SegmentedTensor):
                attrib = attrib.to_dense()
            if isinstance(mask, SegmentedTensor):
                mask = mask.to_dense()
            if mask.shape != attrib.shape:
                raise RuntimeError(f"Mask shape {mask.shape} does not match attribution shape {attrib.shape}.")

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

        def check(f: Path | list[Path]):
            if isinstance(f, list) and len(f) == 0:
                raise FileNotFoundError(path)
            if isinstance(f, Path) and not f.exists():
                raise FileNotFoundError(f)

        check(names)
        check(labels)
        check(attribs)
        check(masks)

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

    def __post_init__(self):
        if isinstance(self.attrib, SegmentedTensor):
            self.attrib = self.attrib.to_dense()
        if isinstance(self.mask, SegmentedTensor):
            self.mask = self.mask.to_dense()
        assert isinstance(self.label, Tensor), f"Expected a Tensor, but got {type(self.label)=} {self.label=}"
        assert isinstance(self.attrib, Tensor), f"Expected a Tensor, but got {type(self.attrib)=} {self.attrib=}"
        assert isinstance(self.mask, Tensor), f"Expected a Tensor, but got {type(self.mask)=} {self.mask=}"


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

        assert len(self.names_files) == len(self.labels_files) == len(self.attribs_files)
        assert len(self.masks_files) == 0 or len(self.masks_files) == len(self.names_files)
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

        for i, annotation in enumerate(iterable):  # pylint: disable=unused-variable
            name  = annotation.name
            rank  = torch.from_numpy(annotation.ranks)
            score = torch.from_numpy(annotation.scores)
            # NOTE: rank is a floating point tensor with large values, so downcasting
            # signficantly reduces the precision of the larger-valued ranks. Downcasting
            # score seems safe, as its already essentially saved from training in fp16 anyway.
            # rank  = torch_safe_downcast(rank)
            score = torch_safe_downcast(score)

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

    def degenerates(self, num_feature: tuple[int, int] = (1, sys.maxsize),) -> np.ndarray:
        """
        Returns the indices of the elements for which the relevance score is constant.
        """
        rows = []
        for i, s in enumerate(self.scores):
            if num_feature[0] <= len(s) <= num_feature[1]:
                if np.unique(s).size == 1:
                    rows.append(i)
        return np.array(rows)

    @property
    def has_names_files(self) -> bool:
        return len(self.names_files) > 0

    @property
    def has_ranks_files(self) -> bool:
        return len(self.ranks) == len(self.names)

    @property
    def has_scores_files(self) -> bool:
        return len(self.scores) == len(self.names)

    def descriptive_sparsity(self, num_feature: tuple[int, int] = (1, sys.maxsize), num_workers: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the descriptive sparsity for each sample in the dataset.

        Arguments:
            num_feature (tuple[int, int]): The range of interpretable features to consider.
                Samples outside of this range will be ignored and the output arrays will not correspond to samples.

        Returns:
          R (np.ndarray): For each sample, grid of r values. Shape (I, 200).
          M (np.ndarray): For each sample, MAZ(r) evaluated at each r. Shape (I, 200).
          A (np.ndarray): Area under the MAZ curve for each sample. Shape (I,).
            For samples with constant relevance score, A is set to 0.0.
        """
        scores = self.scores
        scores = [s for s in scores if num_feature[0] <= len(s) <= num_feature[1]]

        if num_workers is None or num_workers < 2:
            R_M_A = [descriptive_sparsity(s, constant_relevances="auto") for s in scores]
        else:
            with mp.Pool(num_workers) as pool:
                R_M_A = list(pool.imap(partial(descriptive_sparsity, constant_relevances="auto"), scores))

        R = np.stack([r for r, m, a in R_M_A], axis=0)
        M = np.stack([m for r, m, a in R_M_A], axis=0)
        A = np.stack([a for r, m, a in R_M_A], axis=0)

        # rows = []
        # for i in range(len(scores)):
        #     if np.unique(scores[i]).size == 1:
        #         rows.append(i)
        # A[np.array(rows)] = 0.0
        # print(f"Cannot rank any item higher or lower than any other for {len(rows)} / {len(self.scores)} samples.")

        return R, M, A


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
    """

    def __init__(
        self,
        paths: list[Path],
        judge_names: list[str],
        method: AgreementMethod,
        top_k: Optional[int] = None,
        act_judges: Optional[list[np.ndarray]] = None,
        num_feature: tuple[int, int] = (1, sys.maxsize),
        incongruent: bool = False,
        num_workers: int = 0,
        load_cache: bool = True,
        save_cache: bool = True,
    ) -> None:
        """
        Initialization.
        """
        self.paths       = paths
        self.judge_names = judge_names
        self._method     = AgreementMethod(method)
        self.aggfunction = get_agreement_function(self.method)
        self._top_k      = top_k
        self._act_judges = np.full(len(paths), True) if act_judges is None else np.asarray(act_judges)
        self.num_feature = num_feature
        self.incongruent = incongruent
        self.num_workers = num_workers
        self.load_cache  = load_cache
        self.save_cache  = save_cache

        self.name_to_idx: dict[str, int] = None
        self.idx_to_name: dict[int, str] = None
        self.managers = [AttributionPathManager(p) for p in paths]
        self.I: int = None                    # Number of samples.
        self.J: int = len(self.managers)      # Number of judges.
        self.K: np.ndarray = None             # Number of interpretable features per sample.
        self.ranks: list[np.ndarray]  = None  # Shape: (I, K, J)
        self.scores: list[np.ndarray] = None  # Shape: (I, K, J)

        self.W: np.ndarray = None  # Agreement matrix (I,)
        self.P: np.ndarray = None  # P-values matrix (I,)

        if len(self.paths) == 0:
            raise ValueError("No paths provided.")
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"Path {p} does not exist.")
        if len(self.judge_names) != len(self.paths):
            raise ValueError("Number of judge names must match the number of paths.")
        if not all(isinstance(name, str) for name in self.judge_names):
            raise ValueError("Judge names must be strings.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("Top-K must be greater than 0.")
        if self.act_judges.shape != (self.J,):
            raise ValueError(f"Judges shape {self.act_judges.shape} does not match number of judges {self.J}.")
        if self.num_feature[0] < 1 or self.num_feature[1] < 1:
            raise ValueError("Number of features must be greater than 0.")

    @property
    def method(self) -> AgreementMethod:
        return self._method

    @method.setter
    def method(self, method: AgreementMethod) -> None:
        self._method = AgreementMethod(method)

    @property
    def act_judges(self) -> np.ndarray:
        return self._act_judges

    @act_judges.setter
    def act_judges(self, act_judges: np.ndarray) -> None:
        if act_judges is not None and act_judges.shape != (self.J,):
            raise ValueError(f"Judges shape {act_judges.shape} does not match number of judges {self.J}.")
        self._act_judges = np.full(self.J, True) if act_judges is None else np.asarray(act_judges)

    @property
    def top_k(self) -> Optional[int]:
        return self._top_k

    @top_k.setter
    def top_k(self, top_k: Optional[int]) -> None:
        if top_k is not None and top_k <= 0:
            raise ValueError("Top-K must be greater than 0.")
        self._top_k = top_k

    def __call__(self) -> AgreementCoordinator:
        """
        Preparation.
        """
        self.determine_samples()
        print(f"Determined {self.I} samples for which all {self.J} judges have ranked.")
        self.create_matrices()
        print(f"Created rank matrices with between {self.K.min()} and {self.K.max()} features.")
        self.compute_agreement()
        print(f"Computed a well-defined agreement for {self.I - np.isnan(self.W).sum()} / {self.I} samples.")
        return self

    def determine_samples(self) -> AgreementCoordinator:
        """
        Determines the samples that are present in all the paths.
        """
        for j, manager in enumerate(self.managers):
            names = set(manager.names)
            if j == 0:
                allnames = names
            if names != allnames:
                allnames = allnames.intersection(names)
                warnings.warn("Not all experiments contain data for the same samples.")

        lengths  = {name: len(rank) for name, rank in zip(self.managers[0].names, self.managers[0].ranks) if name in allnames}
        allnames = sorted(filter(lambda name: self.num_feature[0] <= lengths[name] <= self.num_feature[1], allnames))
        print(f"Removed {len(lengths) - len(allnames)} samples with num interpretable features outside of [{self.num_feature}].")

        self.idx_to_name = {i: name for i, name in enumerate(allnames)}  # pylint: disable=unnecessary-comprehension
        self.name_to_idx = {name: i for i, name in enumerate(allnames)}
        self.I = len(allnames)
        self.K = np.full((self.I,), -1, dtype=np.int32)

        return self

    def create_matrices(self) -> AgreementCoordinator:
        """
        Creates the rank and score matrices for each sample.
        """

        # Create the rank and score matrices for each sample.
        self.ranks  = [None for _ in range(self.I)]
        self.scores = [None for _ in range(self.I)]
        incongruent = []
        for j, manager in enumerate(self.managers):
            for name, rank, score in zip(manager.names, manager.ranks, manager.scores):
                if rank.shape != score.shape:
                    raise RuntimeError(f"Rank shape {rank.shape} does not match score shape {score.shape}.")
                i = self.name_to_idx.get(name)
                if i is None:
                    continue
                if j == 0:
                    self.K[i] = len(rank)
                    self.ranks[i]  = np.full((self.K[i], self.J), np.nan, np.float32)
                    self.scores[i] = np.full((self.K[i], self.J), np.nan, np.float32)
                    self.ranks[i][:,j]  = rank
                    self.scores[i][:,j] = score
                    continue
                if self.K[i] != len(rank):
                    incongruent.append((i, j, len(rank), name))
                    continue
                self.ranks[i][:,j]  = rank
                self.scores[i][:,j] = score

        # Address incongruences (samples with different number of interpretable features).
        if len(incongruent) > 0:
            for i, j, k, name in incongruent:
                print(f"Incongruence: judge {j} ({self.judge_names[j]}) ranked {k} features when {self.K[i]} were expected on sample {i} ({name}).")
            if not self.incongruent:
                raise RuntimeError("Incongruent samples were detected. Set `incongruent=True` to remove them.")
            remove = tuple(set(i for i, _, _, _ in incongruent))
            self.ranks = [r for i, r in enumerate(self.ranks) if i not in remove]
            self.scores = [s for i, s in enumerate(self.scores) if i not in remove]
            self.I = len(self.ranks)
            self.K = np.delete(self.K, np.array(remove))
            print(f"Incongruent samples were detected. Removed {len(remove)} samples.")

        # Check for NaN and inf values in the ranks and scores matrices.
        def _validate(matrix: list[np.ndarray], which: Literal["rank", "score"]) -> None:
            if len(matrix) != self.I:
                raise RuntimeError(f"Expected {self.I} {which} matrices, but got {len(matrix)}.")
            for i, m in enumerate(matrix):
                if np.any(np.isnan(m)):
                    raise RuntimeError(f"{which} matrix {i} contains NaN values.")
                if np.any(np.isinf(m)):
                    raise RuntimeError(f"{which} matrix {i} contains inf values.")
                if m.shape != (self.K[i], self.J):
                    raise RuntimeError(f"{which} matrix {i} has shape {m.shape}, expected ({self.K[i]}, {self.J}).")

        _validate(self.ranks,  "rank")
        _validate(self.scores, "score")

        return self

    @ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Judge cannot rank any item higher or lower than any other*")
    def compute_agreement(self) -> AgreementCoordinator:
        """
        Computes the agreement between every sample.
        """
        def _validate(x: np.ndarray, which: Literal["W", "P"]) -> None:
            if x.shape != (self.I,):
                raise RuntimeError(f"Expected {which} to have shape ({self.I},) but got {x.shape}.")

        ranks = (r[:,self.act_judges] for r in self.ranks)
        aggfunction = partial(self.aggfunction, top_k=self.top_k)

        if self.load_cache and self.cachefile.exists():
            W, P = np.load(self.cachefile)

        elif self.num_workers < 2:
            W = np.empty((self.I,))
            P = np.empty((self.I,))
            for i, r in enumerate(ranks):
                w, p = aggfunction(r)
                W[i] = w
                P[i] = p

        else:
            with mp.Pool(self.num_workers) as pool:
                W_P = list(pool.imap(aggfunction, ranks))
                W = np.array([w for w, _ in W_P])
                P = np.array([p for _, p in W_P])

        _validate(W, "W")
        _validate(P, "P")
        self.W = W
        self.P = P
        if self.save_cache:
            np.save(self.cachefile, (self.W, self.P))

        return self

    @property
    def agreement_statistic(self) -> StatisticalSummary:
        """
        Computes the statistical summary of the agreement statistic W.
        """
        idx = np.isnan(self.W)
        W = self.W[~idx]
        P = self.P[~idx]  # pylint: disable=unused-variable
        return compute_statistical_summary(W)

    @property
    def cachefile(self) -> Path:
        """
        Returns a unique cachefile for the agreement scores between samples.
        """
        data_1 = "".join(self.name_to_idx.keys()).encode()
        data_2 = "".join(manager.path.as_posix() for manager in self.managers).encode()
        data_3 = self.method.value.encode()
        data_4 = str(self.top_k).encode()
        data_5 = self.act_judges.tobytes()
        data_6 = str(self.num_feature).encode()
        data = data_1 + data_2 + data_3 + data_4 + data_5 + data_6
        hash_ = hashlib.blake2s(data).hexdigest()
        cachefile = Path(f"./cache/attribute/agreement--{hash_}.npy")
        return cachefile

    @property
    def act_judge_names(self) -> list[str]:
        return [j for j, b in zip(self.judge_names, self.act_judges) if b]

    def judge_groups(self) -> list[np.ndarray]:
        groups = product([False, True], repeat=len(self.managers))
        groups = filter(lambda judges: sum(judges) > 1, groups)
        groups = sorted(groups, key=sum, reverse=False)
        groups = map(np.array, groups)
        return list(groups)


class AttributionConfiguration:

    last_path_attribute: str  = "bf16"
    final_path_attribute: str = "xai_seed"

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

    @staticmethod
    def collect(path: Path, filter_fn: Optional[ConfigurationFilter] = None, verbose: bool = False) -> list[AttributionConfiguration]:
        configs: list[AttributionConfiguration] = []
        for root in sorted(path.rglob(f"{AttributionConfiguration.final_path_attribute}--*")):
            try:
                config = AttributionConfiguration.from_path(root)
            except ValueError as err:
                if "is not a valid" in str(err):
                    if verbose:
                        print("Skipping invalid algorithm:", str(err))
                    continue
            if not config.exists:
                if verbose:
                    print("Skipping non-existing path:", config)
                continue
            if config.empty:
                if verbose:
                    print("Skipping empty path:", config)
                continue
            if filter_fn is not None and not filter_fn(config):
                if verbose:
                    print("Skipping filtered path:", config)
                continue
            configs.append(config)
        return configs


class TOK2AnyAttributionConverter:
    """
    Takes the output of a experiment with feature-level explanations and applies new feature masks.

    Example
    -------
    >>> root = Path("/path/to/root/bf16--False")
    >>> # Exists: TOK + IGRD
    >>> cfg_1 = AttributionConfiguration.from_path(root / "seed--0/results/attributions/xai_method--tok/xai_algorithm--igrd/xai_chunk_size--256/xai_seed--0/")
    >>> # Exists: FUN + LIME
    >>> cfg_2 = AttributionConfiguration.from_path(root / "seed--0/results/attributions/xai_method--fun/xai_algorithm--lime/xai_chunk_size--256/xai_seed--0/")
    >>> # Create: FUN + IGRD
    >>> cfg_3 = AttributionConfiguration(root, cfg_1.seed, ExplanationMethod.FUN, ExplanationAlgorithm.IGRD, None, cfg_1.xai_seed)
    >>> TOK2AnyAttributionConverter(cfg_1, cfg_2)(cfg_3.path)
    """

    MAX_ATTRIBUTION_MEMORY = 2 ** 32  # 4 GB

    def __init__(self, tok_cfg: AttributionConfiguration, any_cfg: AttributionConfiguration) -> None:
        """
        Core initializer.
        """
        if not tok_cfg.exists:
            raise FileNotFoundError(tok_cfg.path)
        if not any_cfg.exists:
            raise FileNotFoundError(any_cfg.path)

        self.tok_cfg = tok_cfg
        self.any_cfg = any_cfg

        self.tok_mng = AttributionPathManager(tok_cfg.path)
        self.any_mng = AttributionPathManager(any_cfg.path)
        if not np.array_equal(self.tok_mng.names, self.any_mng.names):
            raise ValueError("The names of the two configurations do not match. The streamed samples will not correspond to each other.")

    def __call__(self, outdir: Path) -> None:

        def get_tensors(f: Path | str) -> Generator[Tensor, None, None]:
            l = torch.load(f, map_location="cpu")
            for t in l:
                yield t.to_dense() if isinstance(t, SegmentedTensor) else t

        # Attributions for every feature; feature masks for every interpretable feature.
        names   = chain.from_iterable(f.read_text().split("\n") for f in self.tok_mng.names_files)
        labels  = chain.from_iterable(get_tensors(f) for f in self.tok_mng.labels_files)
        attribs = chain.from_iterable(get_tensors(f) for f in self.tok_mng.attribs_files)
        masks   = chain.from_iterable(get_tensors(f) for f in self.any_mng.masks_files)

        outdir.mkdir(parents=True, exist_ok=True)
        io_iteration = 0

        N: list[str]    = []
        L: list[Tensor] = []
        A: list[Tensor] = []
        M: list[Tensor] = []
        for i, (n, l, r, m) in tqdm(enumerate(zip(names, labels, attribs, masks, strict=True)), total=len(self.tok_mng), desc="Converting attributions"):  # pylint: disable=unused-variable
            if l.ndim not in (0, 1):
                raise ValueError(tuple(l.shape))
            if r.ndim not in (1,):
                raise ValueError(tuple(r.shape))
            if m.ndim not in (1,):
                raise ValueError(tuple(m.shape))

            # The lengths of the attributions and the masks can differ from padding so we need to take the minimum length.
            length = min(r.shape[0], m.shape[0])
            r = r[:length].to(torch.float32)
            m = m[:length].to(torch.int64)
            a = apply_feature_mask(r.unsqueeze(0), m.unsqueeze(0)).squeeze(0)
            N.append(n)
            L.append(l)
            A.append(torch_safe_downcast(a))
            M.append(torch_safe_downcast(m))
            mem_N = 0
            mem_L = sum(x.numel() * x.element_size() for x in L)
            mem_A = sum(x.numel() * x.element_size() for x in A)
            mem_M = sum(x.numel() * x.element_size() for x in M)
            if mem_N + mem_L + mem_A + mem_M > TOK2AnyAttributionConverter.MAX_ATTRIBUTION_MEMORY:
                suffix = f".{'0' * (3 - len(str(io_iteration)))}{io_iteration}"
                (outdir / f"names{suffix}.txt").write_text("\n".join(N))
                torch.save(L, outdir / f"labels{suffix}.pt")
                torch.save([SegmentedTensor.from_dense(x) for x in A], outdir / f"attribs{suffix}.pt")
                torch.save([SegmentedTensor.from_dense(x) for x in M], outdir / f"masks{suffix}.pt")
                N.clear()
                L.clear()
                A.clear()
                M.clear()
                io_iteration += 1


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
    configuration_filter = ConfigurationFilter(f_method=lambda c: c.xai_method != ExplanationMethod.TOK)
    configs = AttributionConfiguration.collect(base, configuration_filter, verbose=True)

    # Generate ranks.
    # pbar = tqdm(configs)
    # for config in pbar:
    #     manager = AttributionPathManager(config.path)
    #     if manager.has_ranks_files and manager.has_scores_files:
    #         pbar.set_description(f"Skip: {config}")
    #         continue
    #     pbar.set_description(f"Generate: {config}")
    #     manager = manager.generate_ranks(disable_tqdm=False, verbose=False)

    # Check ranks.
    # for config in configs:
    #     manager = AttributionPathManager(config.path)
    #     if not manager.has_ranks_files or not manager.has_scores_files:
    #         print(f"Skiping {config}")
    #         continue
    #     print(f"Checking {config} ... ", end="")
    #     ranks = manager.ranks
    #     nequal = np.full(len(ranks), True)
    #     for i, r in enumerate(ranks):
    #         r_ = stats.rankdata(r)
    #         nequal[i] = not np.array_equal(r, r_)
    #     print(f"Found {nequal.sum()} differences.")

    # Downsize the masks.
    # masks_files = []
    # roots = [e.path for e in experiments]
    # for root in roots:
    #     manager = AttributionPathManager(root)
    #     masks_files.extend(manager.masks_files)
    # safe_downcast_files(masks_files, num_workers=0, disable_tqdm=False, verbose=True)

    # Compute Descriptive Sparsity.
    for config in configs:
        manager = AttributionPathManager(config.path)
        if not (manager.has_ranks_files and manager.has_scores_files):
            print(f"Skiping {config}")
            continue
        t_i = time.time()
        R, M, A = manager.descriptive_sparsity(num_feature=(2, sys.maxsize), num_workers=16)  # pylint: disable=unused-variable
        t_f = time.time()
        stat = compute_statistical_summary(A)
        print(f"{str(config).replace('AttributionConfiguration', '')}: {stat.mean:.5f} +/ {stat.error:.5f} N={stat.support} F={(A==0.0).sum()} T={round(t_f - t_i, 1)}")

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
