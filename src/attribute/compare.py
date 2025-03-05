"""
Conduct analysis to compare the attributions of different XAI methods and algorithms.
"""

from collections import namedtuple
from collections.abc import Iterable, Generator
from copy import deepcopy
import hashlib
from itertools import chain
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Optional, NamedTuple
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

from src.enums import ExplanationMethod, ExplanationAlgorithm
from src.attribute.utils import Masker
from src.learn.helpers import OutputHelper


SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])


ALTERNATIVE = "greater"


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


def get_function_annotations(
    names: Iterable[str],
    labels: Iterable[Tensor],
    attribs: Iterable[Tensor],
    masks: Iterable[Tensor],
) -> Generator[Annotations, None, None]:

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


def get_function_annotations_from_files(
    names: Path | Iterable[Path],
    labels: Path | Iterable[Path],
    attribs: Path | Iterable[Path],
    masks: Path | Iterable[Path],
) -> Generator[Annotations, None, None]:
    names = [names] if isinstance(names, Path) else names
    labels = [labels] if isinstance(labels, Path) else labels
    attribs = [attribs] if isinstance(attribs, Path) else attribs
    masks = [masks] if isinstance(masks, Path) else masks

    for p_name, p_label, p_attrib, p_mask in zip(names, labels, attribs, masks, strict=True):
        names = p_name.read_text().splitlines()
        labels = torch.load(p_label, map_location="cpu")
        attribs = torch.load(p_attrib, map_location="cpu")
        masks = torch.load(p_mask, map_location="cpu")

        yield from get_function_annotations(names, labels, attribs, masks)


def get_function_annotations_from_attribution_path(path: Path) -> Generator[Annotations, None, None]:
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

    yield from get_function_annotations_from_files(names, labels, attribs, masks)


def get_function_annotations_from_attribution_paths(paths: list[Path]) -> Generator[list[Annotations], None, None]:
    iterables = [get_function_annotations_from_attribution_path(path) for path in paths]
    while True:
        try:
            out = [next(it) for it in iterables]
            yield out
        except StopIteration:
            break


def kendallw_without_ties(R: np.ndarray) -> SignificanceResult:
    """
    Implementation of Kendall's Coefficient for Concordance (Kendall's W).

    See Wikipedia (https://en.wikipedia.org/wiki/Kendall%27s_W) for details.
    """

    n = R.shape[0]  # number of samples
    m = R.shape[1]  # number of judges

    # Edge cases match the behavior of scipy.stats.kendalltau
    if n < 2:
        return SignificanceResult(np.nan, np.nan)
    if m < 2:
        raise ValueError("Kendall's W requires at least two judges.")
    if m == 2:
        warnings.warn("Kendall's W is less appropriate for only two judges. Use `kendalltau` instead.")

    # Compute Kendall's W
    R_sum  = np.sum(R, axis=1)              # (n,)
    R_mean = np.mean(R_sum)                 # (,)
    S = np.sum(np.square(R_sum - R_mean))   # (,)
    W = 12 * S / (m ** 2 * (n ** 3 - n))

    # Compute the significance
    s = m * (n - 1) * W         # Chi-squared statistic
    f = n - 1                   # Degrees of freedom
    p = stats.chi2.sf(s, f)

    return SignificanceResult(W, p)


def kendallw_with_ties(R: np.ndarray) -> SignificanceResult:
    """
    Implementation of Kendall's Coefficient for Concordance (Kendall's W) accounting for ties.

    See Wikipedia (https://en.wikipedia.org/wiki/Kendall%27s_W) for details.
    """

    n = R.shape[0]  # number of samples
    m = R.shape[1]  # number of judges

    # Edge cases match the behavior of scipy.stats.kendalltau
    if n < 2:
        return SignificanceResult(np.nan, np.nan)
    if m < 2:
        raise ValueError("Kendall's W requires at least two judges.")
    if m == 2:
        warnings.warn("Kendall's W is less appropriate for only two judges. Use `kendalltau` instead.")

    # Correction for ties
    T = np.zeros((m,))
    for j in range(m):
        _, counts = np.unique(R[:,j], return_counts=True)
        counts = counts[counts > 1]
        T[j] = np.sum(np.power(counts, 3) - counts)

    # Compute Kendall's W
    R_sum     = np.sum(R, axis=1)         # (n,)
    R_sqr_sum = np.sum(np.square(R_sum))  # (,)
    W = (
        ( (12 * R_sqr_sum) - (3 * m ** 2 * n * (n + 1) ** 2) )
        /
        ( m ** 2 * n * (n ** 2 - 1) - (m * np.sum(T)))
    )

    # Compute the significance
    s = m * (n - 1) * W         # Chi-squared statistic
    f = n - 1                   # Degrees of freedom
    p = stats.chi2.sf(s, f)

    return SignificanceResult(W, p)


kendallw = kendallw_with_ties


def kendalltau(R: np.ndarray) -> SignificanceResult:
    assert R.shape[1] == 2
    res = stats.kendalltau(R[:,0], R[:,1], alternative=ALTERNATIVE)
    return SignificanceResult(res[0], res[1])


####################################################################################################
# Main
####################################################################################################


def verify_names_are_consistent(xai_methods: Iterable[ExplanationMethod], xai_algorithms: Iterable[ExplanationAlgorithm], root: Path, verbose: bool = True) -> list[str]:
    names = None
    index = None
    for xai_method in xai_methods:
        for xai_algorithm in xai_algorithms:
            index = (xai_method, xai_algorithm) if index is None else index
            path = root / f"xai_method--{xai_method.value}/xai_algorithm--{xai_algorithm.value}/xai_chunk_size--none"
            if not path.exists():
                raise FileNotFoundError(path)
            files = [path / "names.txt"]
            files = OutputHelper.get_attribution_data_files(files[0], None) if not files[0].exists() else files
            if len(files) == 0 or any(not f.exists() for f in files):
                raise FileNotFoundError(path)
            names_ = []
            for f in files:
                names_ += f.read_text().splitlines()
            names = deepcopy(names_) if names is None else names
            if names != names_:
                raise ValueError(f"Samples analized in {(xai_method.value, xai_algorithm.value)} do not match those from {index}.")
            if verbose:
                print(f"Method={xai_method.value}, Algorithm={xai_algorithm.value}, Samples={len(names)}")
    return names


def _create_rank_matrices(
    xai_method: ExplanationMethod,
    xai_algorithm: ExplanationAlgorithm,
    root: Path,
    num_samples: int,
    num_judges: int,
    verbose: bool,
    subset: Optional[int],
    skip: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    # Logging.
    if verbose:
        print(f"{os.getpid()} creating rank matrices for {xai_method.value} {xai_algorithm.value}.")

    # The path to read attributions from.
    path = root / f"xai_method--{xai_method.value}/xai_algorithm--{xai_algorithm.value}/xai_chunk_size--none"

    # Defines the initial presumed number of interpretable features and
    # the increment to add when the number of features is exceeded.
    feature_incr = 2 ** 16
    num_features = feature_incr

    # Initialize the rank matrix and length matrix. Note the difference in shape
    # between this rank matrix and the one containing ranks for all judges.
    L = np.empty((num_samples,), dtype=np.int64)
    A = np.full((num_samples, num_features), np.nan, dtype=np.float16)

    # Loop over every annotation in the path and extract ranks.
    t_i = time.time()
    i   = 0
    for annotation in get_function_annotations_from_attribution_path(path):
        # Skip files that have been identified as problematic.
        if annotation.name in skip:
            continue

        # Logging.
        if i % 1000 == 0 and verbose:
            t_f = time.time()
            spaces = " " * (len(str(num_samples)) - len(str(i)))
            print(f"{os.getpid()} %={spaces}{i} / {num_samples} Δ={round(t_f - t_i)} ({xai_method.value} {xai_algorithm.value} {annotation.name})", flush=True)
            t_i = time.time()

        # If the ranks are too large, rescale it to fit within the float16 range.
        r = annotation.ranks
        if (overflow := np.finfo(np.float16).max - r.max()) < 0:
            if verbose:
                print(f"Downscaling sample due to overflow {r.max()} ({xai_method.value} {xai_algorithm.value} {annotation.name})")
            factor = np.ceil(-overflow / np.finfo(np.float16).max)
            r = r / factor
        r = r.astype(np.float16)

        # If the ranks are too long, resize the cumulative matrix to fit the new length.
        l = len(r)
        if l > num_features:
            while l > num_features:
                num_features += feature_incr
            if verbose:
                print(f"Increasing feature dimensionality {A.shape[1]} --> {num_features} to acomidate {l} features ({xai_method.value} {xai_algorithm.value} {annotation.name})")
            P = np.full((num_samples, feature_incr), np.nan, dtype=np.float16)
            A = np.concatenate((A, P), axis=1)

        # Save the ranks and length.
        A[i, 0:l] = r
        L[i] = l

        # If we are only processing a subset of the samples, break early, otherwise, increment the counter.
        if subset is not None and i == subset - 1:
            break
        i += 1

    # Verify that we processed the correct number of samples.
    if i != num_samples - 1:
        raise ValueError(f"Expected {num_samples} samples, but only found {i + 1}.")

    return A, L


def create_rank_matrices(
    xai_method: ExplanationMethod,
    xai_algorithms: Iterable[ExplanationAlgorithm],
    root: Path,
    num_samples: int,
    num_judges: int,
    verbose: bool = True,
    subset: Optional[int] = None,
    cache_load: bool = True,
    cache_save: bool = True,
    skip: Optional[set[str]] = tuple(),
) -> tuple[np.ndarray, np.ndarray]:

    # If a previous cache exists, load it and return.
    for p in root.parts:
        if p.split("--")[0] == "task":
            task = p.split("--")[1]
            break
    else:
        raise ValueError(f"Could not find the task in {root}.")
    s = f"{task}--{xai_method.value}--"
    s += "--".join(sorted([xai_algorithm.value for xai_algorithm in xai_algorithms]))
    s += f"--{num_samples}"
    cache_file = Path(f"./cache/attribute/{s}")
    cache_file_A = cache_file.with_suffix(".A.npy")
    cache_file_L = cache_file.with_suffix(".L.npy")
    if cache_load and os.path.exists(cache_file_A) and os.path.exists(cache_file_L):
        A = np.load(cache_file_A)
        L = np.load(cache_file_L)
        return A, L

    if subset is not None:
        assert subset == num_samples

    skip = set(skip)

    with mp.Pool(len(xai_algorithms)) as pool:
        results = pool.starmap(
            _create_rank_matrices,
            [(xai_method, xai_algorithm, root, num_samples, num_judges, verbose, subset, skip) for xai_algorithm in xai_algorithms]
        )

    A = np.concatenate([np.expand_dims(r[0], 1) for r in results], axis=1)
    L = results[0][1]
    for _, L_ in results[1:]:
        if not np.all(L == L_):
            raise ValueError("Lengths of samples do not match.")

    num_features = A.shape[2]
    for k in range(num_features - 1, -1, -1):
        if not np.all(np.isnan(A[:, :, k])):
            break
    num_features = k + 1
    if verbose:
        print(f"Determined that the maximum number of features is {num_features}. Removing {A.shape[2] - num_features} empty columns.")
    A = A[:, :, 0:num_features]

    if verbose:
        print(f"Saving rank matrices and lengths to {cache_file.name}.")

    if cache_save:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file_A, A)
        np.save(cache_file_L, L)

    return A, L


def auto_compute_agreement(A: np.ndarray, L: np.ndarray, judges: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the agreement scores and p-value for every sample.
    If the number of interpretable is less than 2, then the agreement score is NaN, so the output will have NaNs in it!
    If the number of judges is two, use Kendall's tau-b, otherwise use Kendall's W.
    xai_algorithms allows for the selection of a subset of algorithms to compute agreement.
     Unlike in previous functions, this is in fact a 1D array of booleans indicating which columns to keep.
    """
    if A.dtype == np.float16:
        warnings.warn("Performing correlation tests in low-precision mode can lead to numerical instability.")

    if judges is not None:
        assert judges.ndim == 1, judges.shape
        assert judges.shape[0] == A.shape[1], judges.shape
        assert judges.dtype == bool, judges.dtype

    num_judges = A.shape[1]
    if judges is not None:
        num_judges = np.sum(judges)
    if num_judges < 2:
        raise ValueError(f"Correlation tests requires at least two judges, but only {num_judges} were provided.")
    kendall = kendalltau if num_judges == 2 else kendallw

    w = np.empty((A.shape[0],))
    p = np.empty((A.shape[0],))
    for j in range(A.shape[0]):
        l = L[j]

        R = A[j].transpose()
        assert np.all(np.isnan(R[l:,]))
        assert not np.any(np.isnan(R[:l,]))
        R = R[:l,]
        if judges is not None:
            R = R[:,judges]

        # If no functions were detected, we wind up with only a single annotation and we cannot compute agreement.
        # This isn't an error, so we just set the agreement to NaN and move on.
        if R.shape[0] == 1:
            w_, p_ = np.nan, np.nan
        else:
            w_, p_ = kendall(R)
            if any(np.isnan(z) or np.isinf(z) for z in (w_, p_)):
                should_raise = True
                for k in range(num_judges):
                    if len(np.unique(R[:,k])) == 1:
                        warnings.warn(f"Sample {j} has only one unique rank for judge {k}.")
                        if num_judges == 2:
                            w_, p_ = 0, 1
                            should_raise = False

                if should_raise:
                    raise ValueError(f"Correlation test is NaN/InF for sample {j}. ({w_=} {p_=})")

        w[j] = w_
        p[j] = p_

    return w, p


def main():
    root = Path("/home/lk3591/Documents/code/RawByteClf/output/esp-exe/"
        "lift_level--nop/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--wdl/vocab_size--256/"
        "max_length--1048576/"
        "model_name--malconv/channels--256/stride--64/kernel_size--64/head_num_hidden_layers--1/head_hidden_size--128/embedding_size--8/"
        "task--det/").rglob("attributions")
    root = list(root)
    assert len(root) == 1
    root = root[0]

    XAI_METHODS = (
        ExplanationMethod.NUM,
        ExplanationMethod.FUN,
    )
    XAI_ALGORITHMS = (
        ExplanationAlgorithm.LIME,
        ExplanationAlgorithm.KSHP,
        ExplanationAlgorithm.IGRD,
        ExplanationAlgorithm.GSHP,
        ExplanationAlgorithm.FABL,
        ExplanationAlgorithm.SSHP,
    )

    SUBSET    = None
    VERBOSE   = True
    CACHELOAD = False
    CACHESAVE = True
    SKIP = [
        "94430ac65ede0bd6562674339a24daf507ed3004b41e910ee5ed3a163403f16d",  # Has 2 ** 19 interpretable features.
    ]

    print("Verifying that the names are consistent accross experiments.")
    names = verify_names_are_consistent(XAI_METHODS, XAI_ALGORITHMS, root, VERBOSE)
    names = [n for n in names if n not in SKIP]
    num_samples = len(names) if SUBSET is None else SUBSET
    num_judges = len(XAI_ALGORITHMS)

    for xai_method in XAI_METHODS:
        print("Generating rank matrices.")
        A, L = create_rank_matrices(xai_method, XAI_ALGORITHMS, root, num_samples, num_judges, VERBOSE, SUBSET, CACHELOAD, CACHESAVE, skip=SKIP)
        # w, p = auto_compute_agreement(A.astype(np.float32), L, judges=None)
        # w = w[~np.isnan(w)]
        # p = p[~np.isnan(p)]
        # mean = np.mean(w)
        # medn = np.median(w)
        # stdv = np.std(w)
        # print(f"\nMethod={xai_method.value} Mean={mean:.4f} Median={medn:.4f} StdDev={stdv:.4f}")


if __name__ == "__main__":
    main()
