"""
Core attribution interface.
"""

from __future__ import annotations
import math
import os
import sys
from typing import Optional

from captum.attr import (
    Attribution,
    Lime,
    IntegratedGradients,
    GradientShap,
    KernelShap,
    FeatureAblation,
    FeaturePermutation,
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

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# pylint: enable=wrong-import-position

from src.enums import ExplanationMethod, ExplanationAlgorithm
from src.attribute.masking import apply_feature_mask, convert_to_overlapping_feature_mask
from src.attribute.utils import ignore_warnings_decorator


REQUIRES_INTERPRETABLE_EMBEDDINGS = (
    ExplanationAlgorithm.IGRD,
    ExplanationAlgorithm.GSHP,
    ExplanationAlgorithm.DLFT,
)

SINGLE_SAMPLE_ATTRIBUTORS = (
    ExplanationAlgorithm.KSHP,
    ExplanationAlgorithm.LIME,
    ExplanationAlgorithm.SSHP,
)


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


def get_attribution_fabl(alg: FeatureAblation, input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
    is_multilabel = labels.dim() == 2
    baselines     = torch.zeros_like(input_ids)
    target        = None if is_multilabel else labels
    forward_args  = (model, labels) if is_multilabel else (model,)
    attribs       = alg.attribute(input_ids, baselines, target, forward_args, feature_mask)
    return attribs


def get_attribution_fprm(alg: FeaturePermutation, input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
    is_multilabel = labels.dim() == 2
    target        = None if is_multilabel else labels
    forward_args  = (model, labels) if is_multilabel else (model,)
    attribs       = alg.attribute(input_ids, target, forward_args, feature_mask)
    return attribs


def get_attribution_kshp(alg: KernelShap, input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor], batch_size: int = 1) -> Tensor:
    is_multilabel = labels.dim() == 2

    def get_attrib(input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
        input_ids    = input_ids.unsqueeze(0)
        labels       = labels.unsqueeze(0)
        baselines    = torch.zeros_like(input_ids)
        feature_mask = feature_mask.unsqueeze(0) if feature_mask is not None else None
        target       = None if is_multilabel else labels
        forward_args = (model, labels) if is_multilabel else (model,)
        n_features   = len(feature_mask.unique()) if feature_mask is not None else input_ids.shape[1]
        n_samples    = 64 * int(math.log2(n_features + 1))
        attribs      = alg.attribute(input_ids, baselines, target, forward_args, feature_mask, n_samples, batch_size)
        return attribs[0]

    attribs = torch.zeros_like(input_ids, dtype=torch.float32)
    for i in range(input_ids.shape[0]):
        attribs[i] = get_attrib(input_ids[i], labels[i], model, feature_mask[i] if feature_mask is not None else None)
    return attribs


def get_attribution_lime(alg: Lime, input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor], batch_size: int = 1) -> Tensor:
    is_multilabel = labels.dim() == 2

    def get_attrib(input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
        input_ids    = input_ids.unsqueeze(0)
        labels       = labels.unsqueeze(0)
        baselines    = torch.zeros_like(input_ids)
        feature_mask = feature_mask.unsqueeze(0) if feature_mask is not None else None
        target       = None if is_multilabel else labels
        forward_args = (model, labels) if is_multilabel else (model,)
        n_features   = len(feature_mask.unique()) if feature_mask is not None else input_ids.shape[1]
        n_samples    = 64 * int(math.log2(n_features + 1))
        attribs      = alg.attribute(input_ids, baselines, target, forward_args, feature_mask, n_samples, batch_size)
        return attribs[0]

    attribs = torch.zeros_like(input_ids, dtype=torch.float32)
    for i in range(input_ids.shape[0]):
        attribs[i] = get_attrib(input_ids[i], labels[i], model, feature_mask[i] if feature_mask is not None else None)
    return attribs


def get_attribution_sshp(alg: KernelShap, input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor], batch_size: int = 1) -> Tensor:
    is_multilabel = labels.dim() == 2

    def get_attrib(input_ids: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
        input_ids    = input_ids.unsqueeze(0)
        labels       = labels.unsqueeze(0)
        baselines    = torch.zeros_like(input_ids)
        feature_mask = feature_mask.unsqueeze(0) if feature_mask is not None else None
        target       = None if is_multilabel else labels
        forward_args = (model, labels) if is_multilabel else (model,)
        n_features   = len(feature_mask.unique()) if feature_mask is not None else input_ids.shape[1]
        n_samples    = 64 * int(math.log2(n_features + 1))
        attribs      = alg.attribute(input_ids, baselines, target, forward_args, feature_mask, n_samples, batch_size)
        return attribs[0]

    attribs = torch.zeros_like(input_ids, dtype=torch.float32)
    for i in range(input_ids.shape[0]):
        attribs[i] = get_attrib(input_ids[i], labels[i], model, feature_mask[i] if feature_mask is not None else None)
    return attribs


def get_attribution_dlft(alg: DeepLift, inputs_embeds: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:  # pylint: disable=unused-argument
    is_multilabel = labels.dim() == 2
    baselines     = torch.zeros_like(inputs_embeds)
    target        = None if is_multilabel else labels
    forward_args  = (labels,) if is_multilabel else None
    attribs       = alg.attribute(inputs_embeds, baselines, target, forward_args)
    attribs       = attribs.mean(dim=2)
    attribs       = apply_feature_mask(attribs, feature_mask) if feature_mask is not None else attribs
    return attribs


def get_attribution_igrd(alg: IntegratedGradients, inputs_embeds: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
    is_multilabel = labels.dim() == 2
    baselines     = torch.zeros_like(inputs_embeds)
    target        = None if is_multilabel else labels
    forward_args  = (model, labels) if is_multilabel else (model,)
    attribs       = alg.attribute(inputs_embeds, baselines, target, forward_args)
    attribs       = attribs.mean(dim=2)
    attribs       = attribs.to(torch.float32)
    attribs       = apply_feature_mask(attribs, feature_mask) if feature_mask is not None else attribs
    return attribs


def get_attribution_gshp(alg: GradientShap, inputs_embeds: Tensor, labels: Tensor, model: PreTrainedModel, feature_mask: Optional[Tensor]) -> Tensor:
    is_multilabel = labels.dim() == 2
    baselines     = torch.zeros_like(inputs_embeds)
    n_samples     = 5
    stdevs        = 0.0
    target        = None if is_multilabel else labels
    forward_args  = (model, labels) if is_multilabel else (model,)
    attribs       = alg.attribute(inputs_embeds, baselines, n_samples, stdevs, target, forward_args)
    attribs       = attribs.mean(dim=2)
    attribs       = apply_feature_mask(attribs, feature_mask) if feature_mask is not None else attribs
    return attribs


@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Attempting to construct interpretable model with > 10000 features.*")
@ignore_warnings_decorator("ignore", category=UserWarning, message=r"^Setting forward, backward hooks and attributes on non-linear*")
def get_attribution(
    alg: Attribution,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    labels: Tensor,
    model: PreTrainedModel,
    feature_mask: Optional[Tensor] = None,
    batch_size: int = 1,
) -> Tensor:
    if isinstance(alg, FeatureAblation):
        attribs = get_attribution_fabl(alg, input_ids, labels, model, feature_mask)
    elif isinstance(alg, FeaturePermutation):
        attribs = get_attribution_fprm(alg, input_ids, labels, model, feature_mask)
    elif isinstance(alg, KernelShap):
        attribs = get_attribution_kshp(alg, input_ids, labels, model, feature_mask, batch_size)
    elif isinstance(alg, Lime):
        attribs = get_attribution_lime(alg, input_ids, labels, model, feature_mask, batch_size)
    elif isinstance(alg, ShapleyValueSampling):
        attribs = get_attribution_sshp(alg, input_ids, labels, model, feature_mask, batch_size)
    elif isinstance(alg, IntegratedGradients):
        attribs = get_attribution_igrd(alg, inputs_embeds, labels, model, feature_mask)
    elif isinstance(alg, GradientShap):
        attribs = get_attribution_gshp(alg, inputs_embeds, labels, model, feature_mask)
    elif isinstance(alg, DeepLift):
        attribs = get_attribution_dlft(alg, inputs_embeds, labels, model, feature_mask)
    else:
        raise TypeError(f"Attribution algorithm {alg} not supported.")

    if attribs.dtype != torch.float32:
        raise TypeError(f"Expected float32, got {attribs.dtype}.")
    if input_ids is not None and tuple(attribs.shape) != (tuple(input_ids.shape)):
        raise RuntimeError(attribs.shape)
    if inputs_embeds is not None and tuple(attribs.shape) != (tuple(inputs_embeds.shape)[0:2]):
        raise RuntimeError(attribs.shape)

    return attribs
