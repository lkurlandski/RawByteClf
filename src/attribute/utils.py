"""
Tools for performing input attribution with captum.
"""

from __future__ import annotations
from collections import namedtuple
from functools import wraps
import json
import math
import os
import sys
from typing import Optional
import warnings

from captum.attr import (
    Attribution,
    Lime,
    IntegratedGradients,
    GradientShap,
    KernelShap,
    FeatureAblation,
    DeepLift,
    ShapleyValueSampling,
)
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.functional import F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.enums import ExplanationMethod, ExplanationAlgorithm
from src.data.function_boundaries import EXEFuncBoundsMap


REQUIRES_INTERPRETABLE_EMBEDDINGS = (
    ExplanationAlgorithm.IGRD,
    ExplanationAlgorithm.GSHP,
    ExplanationAlgorithm.DLFT,
)


def chunk_mask(x: Tensor, size: int) -> Tensor:
    length = x.shape[0]
    if length < size:
        return torch.full((length,), 0, dtype=torch.int64)
    q, r = divmod(length, size)
    mask = torch.cat([torch.full((size,), i) for i in range(q)])
    mask = torch.cat([mask, torch.full((r,), q)])
    return mask.to(torch.int64)


class Masker:
    """
    Base class for masking features in input data.

    Special tokens are masked with 0, if present.
    All other tokens are masked with integers starting at 1.
    """

    special_token_ids: tuple[int]

    def __init__(self, bos_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None) -> None:
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = tuple(filter(lambda x: x is not None, (self.bos_token_id, self.eos_token_id, self.pad_token_id)))

    def __call__(self, input_ids: Tensor, shas: Optional[list[str]] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        mask = torch.full_like(input_ids, 1, dtype=torch.int64)
        for t in self.special_token_ids:
            mask[input_ids == t] = 0
        return mask


class ChunkFeatureMasker(Masker):

    chunk_size: int

    def __init__(self, *args, chunk_size: int) -> None:
        super().__init__(*args)
        self.chunk_size = chunk_size
        if chunk_size < 1:
            raise ValueError("Chunk size must be greater than 0.")

    def __call__(self, input_ids: Tensor, shas: Optional[list[str]] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        i = 1 if self.bos_token_id is not None else 0  # Skip the BOS token.
        c = 0 if self.chunk_size == 1 else 1
        while i < mask.shape[1]:
            mask[:, i:i + self.chunk_size] = (i // self.chunk_size) + c  # Start at 1.
            i += self.chunk_size

        for t in self.special_token_ids:
            mask[input_ids == t] = 0

        assert torch.all(mask >= 0), "Negative values were detected in the mask, which might break `apply_feature_mask`."

        return mask


class AutoChunkFeatureMasker(Masker):

    def __init__(self, *args, stats: dict[str, float]) -> None:
        super().__init__(*args)
        self.stats = stats

    def __call__(self, input_ids: Tensor, shas: list[str] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        masks = []
        for i, s in zip(input_ids, shas):
            chunk_size = self.get_chunk_size(i, s)
            mask = self.chunk_mask_for_one_input(i, chunk_size)
            masks.append(mask)
        return torch.stack(masks)

    def chunk_mask_for_one_input(self, input_ids: Tensor, chunk_size: int) -> Tensor:
        if chunk_size < 1:
            raise ValueError("Chunk size must be greater than 0.")

        if input_ids.dim() != 1:
            raise RuntimeError()

        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        i = 1 if self.bos_token_id is not None else 0  # Skip the BOS token.
        c = 0 if chunk_size == 1 else 1
        while i < mask.shape[0]:
            mask[i:i + chunk_size] = (i // chunk_size) + c  # Start at 1.
            i += chunk_size

        for t in self.special_token_ids:
            mask[input_ids == t] = 0

        assert torch.all(mask >= 0), "Negative values were detected in the mask, which might break `apply_feature_mask`."

        return mask

    def get_chunk_size(self, input_ids: Tensor, sha: str) -> int:
        raise NotImplementedError()


class AutoLenChunkFeatureMasker(AutoChunkFeatureMasker):

    def get_chunk_size(self, input_ids: Tensor, sha: str) -> int:
        v = self.stats[sha]
        if math.isnan(v):
            return len(input_ids)
        return int(v)


class AutoNumChunkFeatureMasker(AutoChunkFeatureMasker):

    def get_chunk_size(self, input_ids: Tensor, sha: str) -> int:
        if self.eos_token_id is not None:
            n = torch.argmax((input_ids == self.eos_token_id).int()).item()
        else:
            n = len(input_ids)
        if self.bos_token_id is not None:
            n -= 1

        v = self.stats[sha]
        if v == 0:
            return n
        return int(math.ceil(n / self.stats[sha])) - 1


class FunctionFeatureMasker(Masker):

    boundaries: EXEFuncBoundsMap

    def __init__(self, *args, boundaries: dict[str, np.ndarray], allow_missing_shas: bool = False) -> None:
        super().__init__(*args)
        self.boundaries = EXEFuncBoundsMap(boundaries) if not isinstance(boundaries, EXEFuncBoundsMap) else boundaries
        self.allow_missing_shas = allow_missing_shas

    def __call__(self, input_ids: Tensor, shas: list[str] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        offset = 1 if self.bos_token_id is not None else 0

        for i, s in enumerate(shas):
            mask[i,:] = 1
            if self.allow_missing_shas and s not in self.boundaries:
                continue
            for j, (start, end) in enumerate(self.boundaries[s], 2):
                # Figure out which positions have not yet been set. If any have been set,
                # then there are overlapping functions and we need to handle them accordingly.
                # If some positions have already been set, we just set the unset ones and move on.
                # If all positions have already been set, we could just skip this function, decrement j,
                # then move on, but other parts of the codebase aren't going to know about this situation,
                # which could cause issues. So for now, we'll just consider these samples as completely broken.
                # An example of a problematic sample is 91e06aa60176b1a5506e3f875eb7d80329240e34de36bacb8a5606f9f3c4bdb
                # at j=2516 and j=2534.
                unset = mask[i, start + offset : end + offset] == 1
                if not torch.all(unset):
                    if torch.any(unset):
                        warnings.warn(f"Overlapping (partial) function boundaries detected ({s=} {j=} {start=} {end=})!")
                        mask[i, start + offset : end + offset][unset] = j
                    else:
                        warnings.warn(f"Overlapping (total) function boundaries detected ({s=} {j=} {start=} {end=})!")
                        mask[i,:] = 1
                        break
                else:
                    mask[i, start + offset : end + offset] = j

        for t in self.special_token_ids:
            mask[input_ids == t] = 0

        for i in range(len(shas)):
            u = mask[i,:].unique().tolist()
            if set(u) != set(range(len(u))):
                raise RuntimeError("The mask indices are not consecutive.")

        assert torch.all(mask >= 0), "Negative values were detected in the mask, which might break `apply_feature_mask`."

        return mask


def get_masker(
    method: ExplanationMethod,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    chunk_size: Optional[int] = None,
    shas: Optional[list[str]] = None,
) -> FunctionFeatureMasker | ChunkFeatureMasker:
    if any(x is None for x in (bos_token_id, eos_token_id, pad_token_id)):
        raise ValueError(
            "This may not be nessecary in the future, but for now, every model needs three special tokens. "
            f"Got bos_token_id={bos_token_id}, eos_token_id={eos_token_id}, pad_token_id={pad_token_id}."
        )

    if method == ExplanationMethod.CHK:
        return ChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, chunk_size=chunk_size)

    bounds_map = EXEFuncBoundsMap.from_dataset_name(shas=shas)
    len_stats = {s: np.mean(v[:,1] - v[:,0]) for s, v in bounds_map.items()}
    num_stats = {s: len(v) for s, v in bounds_map.items()}

    if method == ExplanationMethod.LEN:
        return AutoLenChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, stats=len_stats)
    if method == ExplanationMethod.NUM:
        return AutoNumChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, stats=num_stats)
    if method == ExplanationMethod.FUN:
        return FunctionFeatureMasker(bos_token_id, eos_token_id, pad_token_id, boundaries=bounds_map)

    raise ValueError(f"Explanation method {method} not supported.")


def apply_feature_mask_slow(X: Tensor, M: Tensor) -> Tensor:
    """
    Not really sure how this works, but it passes the tests.
    """
    assert X.dim() == 2
    assert M.dim() == 2

    Y = torch.zeros_like(X)

    for i, (x, m) in enumerate(zip(X, M)):
        u = m.unique()
        n = u.numel()

        if set(u.tolist()) != set(range(len(u))):
            raise RuntimeError("The mask indices are not consecutive.")

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
    u = M.unique()
    n = u.numel()

    if set(u.tolist()) != set(range(len(u))):
        raise RuntimeError("The mask indices are not consecutive.")

    S = torch.zeros(b, n, dtype=X.dtype, device=X.device)
    O = torch.arange(b, device=X.device).unsqueeze(1) * n
    i = (O + M).view(-1)
    Xf = X.view(-1)
    Sf = S.view(-1)
    Sf.scatter_add_(0, i, Xf)
    return S.gather(1, M)


apply_feature_mask = apply_feature_mask_fast


def convert_to_overlapping_feature_mask(mask: Tensor) -> Tensor:
    assert mask.dim() == 2

    mask = mask.clone()

    offset = 0
    for i in range(len(mask)):
        mask[i,:] += offset
        offset = mask[i].max().item() + 1

    return mask


def ignore_warnings_decorator(*filter_args, **filter_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*filter_args, **filter_kwargs)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def forward_func_with_input_ids(input_ids: Tensor, model: PreTrainedModel, targets: Optional[Tensor] = None) -> Tensor:
    """
    Forward function for captum algorithms that take input IDs.

    Args:
        input_ids (torch.Tensor): The input IDs. Shape (B, T).
        model (transformers.PreTrainedModel): The model.
        targets (torch.Tensor, optional): The target tensor. Defaults to None. Shape (B, C).

    Returns:
        torch.Tensor: The output tensor. Shape (B, C) if targets is None, else (B,).

    If targets is None, the Attribution class should be used as follows:
        >>> alg = AttributionClass(forward_func_with_input_ids)
        >>> attribs = alg.attribute(input_ids, baselines=torch.zeros_like(input_ids), target=labels, additional_forward_args=(model,))

    If performing multilabel classification, the targets should be an indicator tensor,
    and the Attribution class should be used a little differently:
        >>> alg = AttributionClass(forward_func_with_input_ids)
        >>> attribs = alg.attribute(input_ids, baselines=torch.zeros_like(input_ids), target=None, additional_forward_args=(model, labels))
    """
    output: SequenceClassifierOutput = model.forward(input_ids)
    logits = output.logits
    probas = F.softmax(logits, dim=1)
    if targets is not None:
        return (probas * targets).sum(dim=1)
    return probas


def forward_func_with_inputs_embeds(inputs_embeds: Tensor, model: PreTrainedModel, targets: Optional[Tensor] = None) -> Tensor:
    """
    Forward function for captum algorithms that take input embeddings.

    Args:
        inputs_embeds (torch.Tensor): The input embeddings. Shape (B, T, H).
        model (transformers.PreTrainedModel): The model.
        targets (torch.Tensor, optional): The target tensor. Defaults to None. Shape (B, C).

    Returns:
        torch.Tensor: The output tensor. Shape (B, C) if targets is None, else (B,).

    For use with Attribution algorithms that require differentiable functions. Use as follows:
        >>> embedding = configure_interpretable_embedding_layer(model, "backbone.embeddings")
        >>> alg = AttributionClass(forward_func_with_inputs_embeds)
        >>> attribs = alg.attribute(embedding(input_ids), baselines=torch.zeros_like(embedding), target=labels, additional_forward_args=(model,))

    See `forward_func_with_input_ids` for additional documentation.
    """
    output: SequenceClassifierOutput = model.forward(inputs_embeds=inputs_embeds)
    logits = output.logits
    probas = F.softmax(logits, dim=1)
    if targets is not None:
        return (probas * targets).sum(dim=1)
    return probas


class WrapperWithInputIDs(Module):
    """
    Wrapper class for captum algorithms that take input IDs and require an instance of `torch.nn.Module`.

    See `forward_func_with_input_ids` for more information.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        return forward_func_with_input_ids(input_ids, self.model, targets)


class WrapperWithInputEmbeds(Module):
    """
    Wrapper class for captum algorithms that take input embeddings and require an instance of `torch.nn.Module`.

    See `forward_func_with_inputs_embeds` for more information.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        return forward_func_with_inputs_embeds(inputs_embeds, self.model, targets)


def get_attributor(xai_algorithm: ExplanationAlgorithm, model: Optional[PreTrainedModel]) -> Attribution:

    # Surrogate methods.
    if xai_algorithm == ExplanationAlgorithm.LIME:
        return Lime(forward_func_with_input_ids)
    if xai_algorithm == ExplanationAlgorithm.KSHP:
        return KernelShap(forward_func_with_input_ids)

    # Gradient methods.
    if xai_algorithm == ExplanationAlgorithm.IGRD:
        return IntegratedGradients(forward_func_with_inputs_embeds)
    if xai_algorithm == ExplanationAlgorithm.GSHP:
        return GradientShap(forward_func_with_inputs_embeds)
    if xai_algorithm == ExplanationAlgorithm.DLFT:
        return DeepLift(WrapperWithInputEmbeds(model))

    # Perturbation methods.
    if xai_algorithm == ExplanationAlgorithm.FABL:
        return FeatureAblation(forward_func_with_input_ids)
    if xai_algorithm == ExplanationAlgorithm.SSHP:
        return ShapleyValueSampling(forward_func_with_input_ids)

    raise TypeError(f"Explanation algorithm {xai_algorithm} not supported.")


@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^You are providing multiple inputs for Lime / Kernel SHAP attributions.*")
@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Attempting to construct interpretable model with > 10000 features.*")
@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Setting forward, backward hooks and attributes on non-linear*")
def get_attribution(
    alg: Attribution,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    labels: Tensor,
    model: PreTrainedModel,
    feature_mask: Optional[Tensor] = None,
) -> Tensor:

    is_multilabel = labels.dim() == 2

    # Surrogate methods.
    if isinstance(alg, Lime):
        apply_pooling = False
        apply_masking = False
        attribs = alg.attribute(
            input_ids,
            baselines=torch.zeros_like(input_ids),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
            feature_mask=feature_mask,
        )
    if isinstance(alg, KernelShap):
        apply_pooling = False
        apply_masking = False
        attribs = alg.attribute(
            input_ids,
            baselines=torch.zeros_like(input_ids),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
            feature_mask=feature_mask,
        )

    # Gradient methods.
    if isinstance(alg, IntegratedGradients):
        apply_pooling = True
        apply_masking = feature_mask is not None
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
        )
    if isinstance(alg, GradientShap):
        apply_pooling = True
        apply_masking = feature_mask is not None
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
        )
    if isinstance(alg, DeepLift):
        apply_pooling = True
        apply_masking = feature_mask is not None
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(labels,) if is_multilabel else None,
        )

    # Perturbation methods.
    if isinstance(alg, FeatureAblation):
        apply_pooling = False
        apply_masking = False
        attribs = alg.attribute(
            input_ids,
            baselines=0,
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
            feature_mask=feature_mask,
        )
    if isinstance(alg, ShapleyValueSampling):
        # This algorithm has complexity O(num_features * n_samples * batch_size / perturbations_per_eval).
        # The number of features is the number of unique indices in the feature_mask.
        # If we convert the feature_mask into a non-overlapping one with convert_to_overlapping_feature_mask,
        # we wind up with num_features=batch_size * num_features_per_sample. This is a problem because
        # then the complexity is O(batch_size^2 * num_features_per_sample * n_samples / perturbations_per_eval).
        # In other words, increasing the batch_size to improve performance will actually make it performance worse.
        # Upon deeper investigation, I'm fairly confident that converting to a non-overlapping feature mask is
        # not necessary. The features indicated by the same mask value will be perturbed together, but since
        # the model has no interaction between samples of a batch (dropout and batch norm are disabled during eval),
        # this ultimately does not matter.
        apply_pooling = False
        apply_masking = False
        n_samples = 32
        perturbations_per_eval = 32
        if os.environ.get("BATCH_SHAPLEY_VALUE_SAMPLING", "1") == "0":
            attribs = []
            for i in range(input_ids.shape[0]):
                attrib = alg.attribute(
                    input_ids[i].unsqueeze(0),
                    baselines=0,
                    target=None if is_multilabel else labels[i].unsqueeze(0),
                    additional_forward_args=(model, labels[i].unsqueeze(0)) if is_multilabel else (model,),
                    feature_mask=feature_mask[i].unsqueeze(0) if feature_mask is not None else None,
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                )
                attribs.append(attrib)
            attribs = torch.cat(attribs, axis=0)
        else:
            attribs = alg.attribute(
                input_ids,
                baselines=0,
                target=None if is_multilabel else labels,
                additional_forward_args=(model, labels) if is_multilabel else (model,),
                feature_mask=feature_mask,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

    if apply_pooling:
        attribs = attribs.mean(dim=2)
    if apply_masking:
        attribs = apply_feature_mask(attribs, feature_mask)

    if tuple(attribs.shape) != (tuple(input_ids.shape)):
        raise RuntimeError(attribs.shape)

    return attribs
