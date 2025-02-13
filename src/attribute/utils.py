"""
Tools for performing input attribution with captum.
"""

from __future__ import annotations
from functools import wraps
import json
import sys
from typing import Optional, Protocol
import warnings

from captum.attr import (
    Attribution,
    Lime,
    IntegratedGradients,
    GradientShap,
)
import torch
from torch import Tensor
from torch.nn import Module
from torch.functional import F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.enums import ExplanationMethod, ExplanationAlgorithm


def chunk_mask(x: Tensor, size: int) -> Tensor:
    length = x.shape[0]
    if length < size:
        return torch.full((length,), 0, dtype=torch.int64)
    q, r = divmod(length, size)
    mask = torch.cat([torch.full((size,), i) for i in range(q)])
    mask = torch.cat([mask, torch.full((r,), q)])
    return mask.to(torch.int64)


class Masker(Protocol):

    def __call__(self, length: int, shas: Optional[list[str]] = None) -> Tensor:
        ...


class FunctionFeatureMasker:

    def __init__(self, boundaries: dict[str, list[tuple[int, int]]]) -> None:
        self.boundaries = boundaries

    @classmethod
    def from_file(cls, file: str) -> FunctionFeatureMasker:
        with open(file, "r") as fp:
            boundaries = json.load(fp)
        return cls(boundaries)

    def __call__(self, length: int, shas: list[str]) -> Tensor:

        mask = torch.zeros(len(shas), length, dtype=torch.int64)

        for i, s in enumerate(shas):
            for j, (start, end) in enumerate(self.boundaries[s], 1):
                mask[i, start:end] = j

        return mask


class ChunkFeatureMasker:

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def __call__(self, length: int, shas: Optional[list[str]] = None, num: Optional[int] = None) -> Tensor:

        num = len(shas) if num is None else num

        mask = torch.zeros(length, dtype=torch.int64)

        i = 0
        while i < length:
            mask[i:i + self.chunk_size] = i // self.chunk_size
            i += self.chunk_size

        mask = torch.stack([mask.clone() for _ in range(num)])

        return mask


def get_masker(method: ExplanationMethod, chunk_size: Optional[int] = None) -> FunctionFeatureMasker | ChunkFeatureMasker:
    if method == ExplanationMethod.CHK:
        return ChunkFeatureMasker(chunk_size)
    if method == ExplanationMethod.FUN:
        return FunctionFeatureMasker.from_file(file=None)

    raise ValueError(f"Explanation method {method} not supported.")


def apply_feature_mask_slow(X: Tensor, M: Tensor) -> Tensor:
    """
    Not really sure how this works, but it passes the tests.
    """
    assert X.dim() == 2
    assert M.dim() == 2

    Y = torch.zeros_like(X)

    for i, (x, m) in enumerate(zip(X, M)):
        n = m.unique().numel()
        s = torch.zeros(n, dtype=x.dtype, device=x.device)
        s.scatter_add_(0, m, x)
        y = s[m]
        Y[i] = y

    return Y


def apply_feature_mask_fast(X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Not really sure how this works, but it passes the tests.
    """
    assert X.dim() == 2
    assert M.dim() == 2

    b = X.shape[0]
    n = M.unique().numel()

    S = torch.zeros(b, n, dtype=X.dtype, device=X.device)
    O = torch.arange(b, device=X.device).unsqueeze(1) * n
    i = (O + M).view(-1)
    Xf = X.view(-1)
    Sf = S.view(-1)
    Sf.scatter_add_(0, i, Xf)
    return S.gather(1, M)


apply_feature_mask = apply_feature_mask_fast


def ignore_warnings_decorator(*filter_args, **filter_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*filter_args, **filter_kwargs)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def forward_func_with_input_ids(input_ids: Tensor, model: PreTrainedModel) -> Tensor:
    output: SequenceClassifierOutput = model.forward(input_ids)
    logits = output.logits
    probas = F.softmax(logits, dim=1)
    return probas


def forward_func_with_inputs_embeds(inputs_embeds: Tensor, model: PreTrainedModel) -> Tensor:
    output: SequenceClassifierOutput = model.forward(inputs_embeds=inputs_embeds)
    logits = output.logits
    probas = F.softmax(logits, dim=1)
    return probas


def get_attributor(xai_algorithm: ExplanationAlgorithm) -> Attribution:

    if xai_algorithm == ExplanationAlgorithm.SHAP:
        return GradientShap(forward_func_with_inputs_embeds)
    if xai_algorithm == ExplanationAlgorithm.LIME:
        return Lime(forward_func_with_input_ids)
    if xai_algorithm == ExplanationAlgorithm.IGRD:
        return IntegratedGradients(forward_func_with_inputs_embeds)

    raise TypeError(f"Explanation algorithm {xai_algorithm} not supported.")


@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^You are providing multiple inputs for Lime / Kernel SHAP attributions.*")
@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Attempting to construct interpretable model with > 10000 features.*")
def get_attribution(
    alg: Attribution,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    labels: Tensor,
    model: PreTrainedModel,
    feature_mask: Optional[Tensor] = None,
) -> Tensor:

    attribs = None

    if isinstance(alg, GradientShap):
        apply_pooling = True
        apply_masking = feature_mask is not None
        attribs = alg.attribute(
            inputs_embeds, baselines=torch.zeros_like(inputs_embeds),
            target=labels, additional_forward_args=(model,),
        )

    if isinstance(alg, IntegratedGradients):
        apply_pooling = True
        apply_masking = feature_mask is not None
        attribs = alg.attribute(
            inputs_embeds, baselines=torch.zeros_like(inputs_embeds),
            target=labels, additional_forward_args=(model,),
        )

    if isinstance(alg, Lime):
        apply_pooling = False
        apply_masking = False
        attribs = alg.attribute(
            input_ids, baselines=torch.zeros_like(input_ids),
            target=labels, additional_forward_args=(model,), feature_mask=feature_mask,
        )

    if attribs is None:
        raise TypeError(f"Attribution algorithm {alg} not supported.")

    if apply_pooling:
        attribs = attribs.mean(dim=2)
    if apply_masking:
        attribs = apply_feature_mask(attribs, feature_mask)

    return attribs
