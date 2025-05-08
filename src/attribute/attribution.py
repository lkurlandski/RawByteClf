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


def get_n_samples(num_features: int) -> int:
    assert num_features > 0, "Number of features must be greater than 0."
    return 64 * int(math.log2(num_features + 1))


def get_attribution_with_input_ids(
    alg: FeatureAblation | Lime | KernelShap | ShapleyValueSampling,
    input_ids: Tensor,
    labels: Tensor,
    model: PreTrainedModel,
    feature_mask: Optional[Tensor],
    batch_size: int = 1,
    n_samples: Optional[int] = None,
) -> Tensor:
    is_multilabel = labels.dim() == 2

    attribs       = torch.zeros_like(input_ids, dtype=torch.float32)
    for i in range(input_ids.shape[0]):
        _input_ids    = input_ids[i].unsqueeze(0)
        _baselines    = torch.zeros_like(_input_ids)
        _labels       = labels[i].unsqueeze(0)
        _feature_mask = feature_mask[i].unsqueeze(0) if feature_mask is not None else None
        _target       = None if is_multilabel else _labels
        _forward_args = (model, _labels) if is_multilabel else (model,)
        _n_features   = _feature_mask.unique().shape[0] if _feature_mask is not None else _input_ids.shape[1]
        _n_samples    = get_n_samples(_n_features) if n_samples is None else n_samples
        attribs[i]    = alg.attribute(
            _input_ids,
            baselines=_baselines,
            target=_target,
            additional_forward_args=_forward_args,
            feature_mask=_feature_mask,
            perturbations_per_eval=batch_size,
            n_samples=_n_samples,
        )

    return attribs


def get_attribution_with_inputs_embeds(
    alg: IntegratedGradients | GradientShap | DeepLift,
    inputs_embeds: Tensor,
    labels: Tensor,
    model: PreTrainedModel,
    feature_mask: Optional[Tensor],
) -> Tensor:
    is_multilabel = labels.dim() == 2

    if isinstance(alg, IntegratedGradients):
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
        )
    elif isinstance(alg, GradientShap):
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(model, labels) if is_multilabel else (model,),
        )
    elif isinstance(alg, DeepLift):
        attribs = alg.attribute(
            inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            target=None if is_multilabel else labels,
            additional_forward_args=(labels,) if is_multilabel else None,
        )
    else:
        raise TypeError(f"Explanation algorithm {alg} not supported.")

    attribs = attribs.mean(dim=2)
    if feature_mask is not None:
        attribs = apply_feature_mask(attribs, feature_mask)

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
    n_samples: Optional[int] = None,
) -> Tensor:
    if isinstance(alg, (FeatureAblation, Lime, KernelShap, ShapleyValueSampling)):
        attribs =  get_attribution_with_input_ids(alg, input_ids, labels, model, feature_mask, batch_size, n_samples)
    elif isinstance(alg, (IntegratedGradients, GradientShap, DeepLift)):
        attribs = get_attribution_with_inputs_embeds(alg, inputs_embeds, labels, model, feature_mask)
    else:
        raise TypeError(f"Explanation algorithm {alg} not supported.")

    if isinstance(alg, IntegratedGradients):  # For some reason, IGRD returns float64.
        attribs = attribs.to(torch.float32)

    if attribs.dtype != torch.float32:
        raise TypeError(f"Expected float32, got {attribs.dtype}.")
    if input_ids is not None and tuple(attribs.shape) != (tuple(input_ids.shape)):
        raise RuntimeError(attribs.shape)
    if inputs_embeds is not None and tuple(attribs.shape) != (tuple(inputs_embeds.shape)[0:2]):
        raise RuntimeError(attribs.shape)

    return attribs
