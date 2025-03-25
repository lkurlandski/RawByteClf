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
from typing import Optional, Literal
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


def infer_chunk_sizes(mask: torch.Tensor) -> list[int]:
    if mask.dim() != 1:
        raise ValueError("The mask must be 1D.")

    # Compute where consecutive elements differ.
    diff = mask[1:] != mask[:-1]
    # Get indices where changes occur (adjust indices by +1).
    change_indices = diff.nonzero(as_tuple=True)[0] + 1

    # Concatenate the start and end boundaries.
    indices = torch.cat((
        torch.tensor([0], dtype=change_indices.dtype, device=mask.device),
        change_indices,
        torch.tensor([mask.size(0)], dtype=change_indices.dtype, device=mask.device)
    ))

    # Compute sizes by taking differences between consecutive boundaries.
    sizes = indices[1:] - indices[:-1]
    return sizes.tolist()


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

    def get_last_idx(self, input_ids: Tensor) -> int:
        """
        Returns the last index of the input. This is either the position of the EOS token or the PAD token,
        if present, or the length of the input, in which case, input_ids[get_last_idx(input_ids)] will
        raise an IndexError.
        """
        if self.eos_token_id is not None:
            idx = torch.nonzero(input_ids == self.eos_token_id)
            last_idx = idx[0].item()
        elif self.pad_token_id is not None:
            idx = torch.nonzero(input_ids == self.pad_token_id)
            if len(idx) == 0:
                last_idx = len(input_ids)
            else:
                last_idx = idx[0].item()
        else:
            last_idx = len(input_ids)
        return last_idx

    @staticmethod
    def select_valid_bounds(bounds: np.ndarray, max_length: int) -> np.ndarray:
        idx = bounds[:,0] < max_length
        return bounds[idx]


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

    boundaries: EXEFuncBoundsMap

    def __init__(self, *args, boundaries: dict[str, np.ndarray]) -> None:
        super().__init__(*args)
        self.boundaries = boundaries

    def __call__(self, input_ids: Tensor, shas: list[str] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        masks = []
        for i, s in zip(input_ids, shas):
            mask = self.chunk_mask_for_one_input(i, s)
            for t in self.special_token_ids:
                mask[i == t] = 0
            assert torch.all(mask >= 0), "Negative values were detected in the mask."
            masks.append(mask)
        return torch.stack(masks)

    def chunk_mask_for_one_input(self, input_ids: Tensor, sha: str) -> Tensor:  # pylint: disable=unused-argument
        raise NotImplementedError()

    def compute_stat(self, input_ids: Tensor, sha: str) -> float:  # pylint: disable=unused-argument
        raise NotImplementedError()

    @staticmethod
    def compute_stats_map_from_bounds_map(bounds_map: dict[str, np.ndarray], max_length: Optional[int] = None) -> dict[str, float]:  # pylint: disable=unused-argument
        """
        NOTE: this function cannot dynamically account for the possibility of out-of-bounds functions!
        """
        raise NotImplementedError()


class AutoLenChunkFeatureMasker(AutoChunkFeatureMasker):
    """
    Creates fixed-size chunk masks such that the size of the interpretable features is equal to the average function length in the file.
    """

    def chunk_mask_for_one_input(self, input_ids: Tensor, sha: str) -> Tensor:
        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        chunk_size = self.get_chunk_size(input_ids, sha)

        i = 1 if self.bos_token_id is not None else 0  # Skip the BOS token.
        c = 0 if chunk_size == 1 else 1
        while i < mask.shape[0]:
            mask[i:i + chunk_size] = (i // chunk_size) + c  # Start at 1.
            i += chunk_size

        return mask

    def compute_stat(self, input_ids: Tensor, sha: str) -> float:
        """
        Returns the average function length (possibly NaN), out of all functions within the length of the input, in the file.
        """
        return self.compute_average_function_length(input_ids, sha)

    def compute_average_function_length(self, input_ids: Tensor, sha: str) -> float:
        max_length = self.get_last_idx(input_ids)
        if self.bos_token_id:
            max_length -= 1
        v = self.select_valid_bounds(self.boundaries[sha], max_length)
        if len(v) == 0:
            return np.NaN
        k = np.mean(v[:,1] - v[:,0])
        return k

    def get_chunk_size(self, input_ids: Tensor, sha: str) -> int:
        v = self.compute_average_function_length(input_ids, sha)
        if math.isnan(v):
            return len(input_ids)
        return int(v)

    @staticmethod
    def compute_stats_map_from_bounds_map(bounds_map: dict[str, np.ndarray], max_length: Optional[int] = None) -> dict[str, float]:
        stats = {}
        for s, v in bounds_map.items():
            if max_length is not None:
                idx = v[:,0] < max_length
                v = v[idx]
            if len(v) == 0:
                stats[s] = np.NaN
            else:
                stats[s] = np.mean(v[:,1] - v[:,0])
        return stats


class AutoNumChunkFeatureMasker(AutoChunkFeatureMasker):
    """
    Creates fixed-size chunk masks such that the number of interpretable features is equal to the number of functions in the file.
    """

    def chunk_mask_for_one_input(self, input_ids: Tensor, sha: str) -> Tensor:
        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        num_chunks = self.get_num_chunks(input_ids, sha)
        num_tok    = self.get_last_idx(input_ids) - 1
        num_tok    = num_tok - 1 if self.bos_token_id is not None else num_tok

        chunk_size_1 = num_tok // num_chunks
        chunk_size_2 = chunk_size_1 + 1
        change_idx = num_tok % num_chunks

        i = 1 if self.bos_token_id is not None else 0  # Skip the BOS token.
        j = 0
        v = 1 if self.special_token_ids else 0
        while i < num_tok + (1 if self.bos_token_id is not None else 0) + (1 if self.eos_token_id is not None else 0):
            chunk_size = chunk_size_2 if j <= change_idx else chunk_size_1
            mask[i:i + chunk_size] = v
            i += chunk_size
            j += 1
            v += 1

        return mask

    def compute_stat(self, input_ids: Tensor, sha: str) -> float:
        """
        Returns the number of functions, within the length of the input, in the file.
        """
        return self.compute_number_of_functions(input_ids, sha)

    def compute_number_of_functions(self, input_ids: Tensor, sha: str) -> int:
        max_length = self.get_last_idx(input_ids)
        if self.bos_token_id:
            max_length -= 1
        v = self.select_valid_bounds(self.boundaries[sha], max_length)
        k = len(v)
        return k

    def get_num_chunks(self, input_ids: Tensor, sha: str) -> int:
        num_fun = self.compute_number_of_functions(input_ids, sha)
        if num_fun == 0:
            return 1
        return num_fun + 1

    @staticmethod
    def compute_stats_map_from_bounds_map(bounds_map: dict[str, np.ndarray], max_length: Optional[int] = None) -> dict[str, float]:
        stats = {}
        for s, v in bounds_map.items():
            if max_length is not None:
                idx = v[:,0] < max_length
                v = v[idx]
            stats[s] = len(v)
        return stats


class AutoNumLenChunkFeatureMasker(AutoNumChunkFeatureMasker, AutoLenChunkFeatureMasker):
    """
    Creates fixed-size chunk masks such that the number of interpretable features is equal to the number of functions in the file
       and the size of the interpretable features is equal to the average function length in the file. The masks are created
       over the region of the input that contains functions.
    """

    def chunk_mask_for_one_input(self, input_ids: Tensor, sha: str) -> Tensor:
        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        start, end = self.get_chunk_mask_region(input_ids, sha)
        num_chunks = self.get_num_chunks(input_ids, sha) - 1
        chunk_size = self.compute_average_function_length(input_ids, sha)
        if math.isnan(chunk_size):
            chunk_size = end - start
        else:
            chunk_size = int(round(chunk_size))

        last_idx = self.get_last_idx(input_ids)
        num_tok  = end - start

        diff = chunk_size * num_chunks - num_tok
        if diff > 0:  # Move the start index back.
            buff  = start - (1 if self.bos_token_id is not None else 0)
            subt  = min(diff, buff)
            start -= subt
            diff  -= subt
        if diff > 0:  # Move the end index forward.
            buff = last_idx - end
            subt = min(diff, buff)
            end  += subt
            diff -= subt
        if diff > 0:
            warnings.warn(f"The function region is too small to fit {num_chunks} chunks with size {chunk_size}. ({sha} {self.get_chunk_mask_region(input_ids, sha)=} {last_idx=})")

        i = start
        j = 0
        v = 1 if self.special_token_ids else 0
        while i < end and j < num_chunks:
            mask[i:i + chunk_size] = v
            i += chunk_size
            j += 1
            v += 1

        mask[0:start] = v
        mask[i:]      = v

        return mask

    def compute_stat(self, input_ids: Tensor, sha: str) -> float:  # pylint: disable=unused-argument
        raise NotImplementedError("Cannot compute a single statistic for both the number of functions and the average function length.")

    def get_chunk_mask_region(self, input_ids: Tensor, sha: str) -> tuple[int, int]:
        last_idx = self.get_last_idx(input_ids)
        if self.eos_token_id is not None:
            if input_ids[last_idx] != self.eos_token_id:
                raise RuntimeError(f"The last token in the function is not the EOS token. {last_idx=} {input_ids[last_idx-1:last_idx+2]=}.")
        elif self.pad_token_id is not None:
            if last_idx < len(input_ids) and input_ids[last_idx] != self.pad_token_id:
                raise RuntimeError(f"The last token in the function is not the PAD token. {last_idx=} {input_ids[last_idx-1:last_idx+2]=}.")

        bounds: np.ndarray = self.boundaries[sha]
        bounds = bounds + 1 if self.bos_token_id is not None else bounds

        if len(bounds) == 0:
            start = 1 if self.bos_token_id is not None else 0
            end   = last_idx
        else:
            start = max(bounds[:,0].min(), 0 + (1 if self.bos_token_id is not None else 0))
            end   = min(bounds[:,1].max(), last_idx)

        if start > last_idx:
            start = 1 if self.bos_token_id is not None else 0
            end   = last_idx

        if start == end:
            raise RuntimeError(f"Lower bound is equal to upper bound {sha=} {last_idx=} {start=} {end=} {bounds=}")
        if start > end:
            raise RuntimeError(f"Lower bound is greater than upper bound {start=} {end=}.")

        return start, end

    @staticmethod
    def compute_stats_map_from_bounds_map(bounds_map: dict[str, np.ndarray], max_length: Optional[int] = None) -> dict[str, float]:  # pylint: disable=unused-argument
        raise NotImplementedError("Cannot compute a single statistic for both the number of functions and the average function length.")


class FunctionFeatureMasker(Masker):

    boundaries: EXEFuncBoundsMap

    def __init__(self, *args, boundaries: dict[str, np.ndarray], allow_missing_shas: bool = False, function_out_of_bounds: Literal["warn", "raise", "pass"] = "raise") -> None:
        super().__init__(*args)
        self.boundaries = EXEFuncBoundsMap(boundaries) if not isinstance(boundaries, EXEFuncBoundsMap) else boundaries
        self.allow_missing_shas = allow_missing_shas
        self.function_out_of_bounds = function_out_of_bounds

    def __call__(self, input_ids: Tensor, shas: list[str] = None) -> Tensor:
        if shas is not None and len(shas) != input_ids.shape[0]:
            raise ValueError("The number of SHAs must match the number of input IDs.")

        mask = torch.full_like(input_ids, -1, dtype=torch.int64)

        offset = 1 if self.bos_token_id is not None else 0

        for i, s in enumerate(shas):
            mask[i,:] = 1

            if self.allow_missing_shas and s not in self.boundaries:
                continue

            if self.number_of_functions_outside_input(input_ids[i], s) > 0:
                if self.function_out_of_bounds == "warn":
                    warnings.warn(f"Function boundary past length of the file detected ({s})!")
                elif self.function_out_of_bounds == "raise":
                    raise RuntimeError(f"Function boundary past length of the file detected ({s})!")

            # This can fix an issue with the boundaries being out of order, resulting in non-consecutive mask indices.
            bounds = self.boundaries[s]
            idx = np.argsort(bounds[:,0])
            bounds = bounds[idx]
            for j, (start, end) in enumerate(bounds, 2):
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

        for i, s in enumerate(shas):
            assert_feature_mask_indices_are_consecutive(mask[i,:])

        assert torch.all(mask >= 0), "Negative values were detected in the mask, which might break `apply_feature_mask`."

        return mask

    def number_of_functions_outside_input(self, input_ids: Tensor, sha: str) -> int:
        max_length = self.get_last_idx(input_ids)
        if self.bos_token_id is not None:
            max_length -= 1
        v = self.select_valid_bounds(self.boundaries[sha], max_length)
        return len(self.boundaries[sha]) - len(v)


def get_masker(
    method: ExplanationMethod,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    chunk_size: Optional[int] = None,
    shas: Optional[list[str]] = None,
    allow_missing_shas: bool = False,
    function_out_of_bounds: Literal["warn", "raise", "pass"] = "pass",
) -> Masker:
    if any(x is None for x in (bos_token_id, eos_token_id, pad_token_id)):
        raise ValueError(
            "This may not be nessecary in the future, but for now, every model needs three special tokens. "
            f"Got bos_token_id={bos_token_id}, eos_token_id={eos_token_id}, pad_token_id={pad_token_id}."
        )

    if method == ExplanationMethod.CHK:
        return ChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, chunk_size=chunk_size)

    bounds_map = EXEFuncBoundsMap.from_dataset_name(shas=shas, allow_missing_shas=allow_missing_shas)

    if method == ExplanationMethod.LEN:
        return AutoLenChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, boundaries=bounds_map)
    if method == ExplanationMethod.NUM:
        return AutoNumChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, boundaries=bounds_map)
    if method == ExplanationMethod.NML:
        return AutoNumLenChunkFeatureMasker(bos_token_id, eos_token_id, pad_token_id, boundaries=bounds_map)
    if method == ExplanationMethod.FUN:
        return FunctionFeatureMasker(bos_token_id, eos_token_id, pad_token_id, boundaries=bounds_map, function_out_of_bounds=function_out_of_bounds)

    raise ValueError(f"Explanation method {method} not supported.")


def assert_feature_mask_indices_are_consecutive(mask: Tensor) -> int:
    """
    Asserts that the feature mask indices are consecutive. Returns the number of unique indices.
    """
    u = mask.unique()
    if set(u.tolist()) != set(range(len(u))):
        raise RuntimeError("The mask indices are not consecutive.")
    return len(u)


def apply_feature_mask_slow(X: Tensor, M: Tensor) -> Tensor:
    """
    Not really sure how this works, but it passes the tests.
    """
    assert X.dim() == 2
    assert M.dim() == 2

    Y = torch.zeros_like(X)

    for i, (x, m) in enumerate(zip(X, M)):
        n = assert_feature_mask_indices_are_consecutive(m)
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

    n = assert_feature_mask_indices_are_consecutive(M)
    b = X.shape[0]
    S = torch.zeros(b, n, dtype=X.dtype, device=X.device)
    O = torch.arange(b, device=X.device).unsqueeze(1) * n
    i = (O + M).view(-1)
    Xf = X.view(-1)
    Sf = S.view(-1)
    Sf.scatter_add_(0, i, Xf)
    Z = S.gather(1, M)

    return Z


apply_feature_mask = apply_feature_mask_fast


def convert_to_overlapping_feature_mask(mask: Tensor) -> Tensor:
    assert mask.dim() == 2

    mask = mask.clone()

    offset = 0
    for i in range(len(mask)):  # pylint: disable=consider-using-enumerate
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
