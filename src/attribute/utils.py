"""
Tools for performing input attribution with captum.
"""

from functools import wraps
from typing import Optional
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
) -> Tensor:

    if isinstance(alg, GradientShap):
        return alg.attribute(
            inputs_embeds, baselines=torch.zeros_like(inputs_embeds),
            target=labels, additional_forward_args=(model,),
        ).mean(dim=-1)
    if isinstance(alg, IntegratedGradients):
        return alg.attribute(
            inputs_embeds, baselines=torch.zeros_like(inputs_embeds),
            target=labels, additional_forward_args=(model,),
        ).mean(dim=-1)
    if isinstance(alg, Lime):
        return alg.attribute(
            input_ids, baselines=torch.zeros_like(input_ids),
            target=labels, additional_forward_args=(model,),
        )

    raise TypeError(f"Attribution algorithm {alg} not supported.")
