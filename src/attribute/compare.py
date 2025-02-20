"""
Conduct analysis to compare the attributions of different XAI methods and algorithms.
"""

from collections import namedtuple
from collections.abc import Iterable, Generator
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
import os
from pathlib import Path
import sys
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

    Higher scores and lower ranks indicate greater importance. The scores may
    be any real value, positive or negative, whereas the ranks are non-negative
    integers with 0 being given to the most important function.

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
        ranks = torch.argsort(torch.argsort(scores, descending=True))

        scores = scores.numpy(force=True)
        ranks  = ranks.numpy(force=True)
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


def kendallw(R: np.ndarray) -> SignificanceResult:
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


def main():
    root = Path("./output/test/lift_level--nop/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--wdl/vocab_size--256/max_length--65536/model_name--malconv/channels--256/stride--64/kernel_size--64/embedding_size--8/head_num_hidden_layers--1/head_hidden_size--128/task--det/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.01/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--5.0/world_size--1/per_device_train_batch_size--64/gradient_accumulation_steps--1/tf32--False/fp16--False/bf16--False/seed--0/results/attributions/")

    # Ensure that every XAI method and algorithm has the same names in the same order.
    print("Checking integrity of samples across XAI methods and algorithms.")
    names = None
    index = None
    for xai_method in (ExplanationMethod.NUM, ExplanationMethod.FUN):
        for xai_algorithm in ExplanationAlgorithm:
            index = (xai_method, xai_algorithm) if index is None else index
            path = root / f"xai_method--{xai_method.value}/xai_algorithm--{xai_algorithm.value}/xai_chunk_size--4096"
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
            print(f"\tMethod={xai_method.value}, Algorithm={xai_algorithm.value}, Samples={len(names)}")


    for xai_method in tqdm((ExplanationMethod.NUM, ExplanationMethod.FUN)):
        # Array of (n_samples, n_algorithms, n_interpretable_features) with the rank of each feature.
        A = np.full((len(names), len(ExplanationAlgorithm), 2 ** 20), np.nan)
        lengths = np.empty(len(names), dtype=int)
        for i, xai_algorithm in tqdm(enumerate(ExplanationAlgorithm), leave=False):
            path = root / f"xai_method--{xai_method.value}/xai_algorithm--{xai_algorithm.value}/xai_chunk_size--4096"
            iterator = get_function_annotations_from_attribution_path(path)
            for j, annotation in tqdm(enumerate(iterator), leave=False, total=len(names)):
                ranks = annotation.ranks
                l = len(ranks)
                lengths[j] = l
                A[j, i, 0:l] = ranks

        w = np.empty((len(names),))
        p = np.empty((len(names),))
        for j in range(A.shape[0]):
            l = lengths[j]
            R = A[j].transpose()
            assert np.all(np.isnan(R[l:,]))
            R = R[:l,]
            assert not np.any(np.isnan(R))

            # If no functions were detected, we wind up with only a single annotation,
            # in which case, we cannot compute agreement, so we skip it.
            if R.shape[0] == 1 and xai_method in (ExplanationMethod.NUM, ExplanationMethod.FUN):
                w_, p_ = np.nan, np.nan
            else:
                w_, p_ = kendallw(R)
                if np.isnan(w_):
                    raise ValueError(f"Kendall's W is NaN for sample {j}.")
                if np.isnan(p_):
                    raise ValueError(f"Kendall's W p-value is NaN for sample {j}.")

            w[j] = w_
            p[j] = p_

        # NOTE: once removing the NaNs, the list no longer corresponds to each sample.
        w = w[~np.isnan(w)]
        p = p[~np.isnan(p)]
        mean = np.mean(w)
        medn = np.median(w)
        stdv = np.std(w)
        print(f"\nMethod={xai_method.value} Mean={mean:.4f} Median={medn:.4f} StdDev={stdv:.4f}")


if __name__ == "__main__":
    main()
