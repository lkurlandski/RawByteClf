"""
Core attribution interface.
"""

from __future__ import annotations
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
            if os.environ.get("NOOVERLAP_SHAPLEY_VALUE_SAMPLING", "0") == "1":
                feature_mask = convert_to_overlapping_feature_mask(feature_mask)
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
