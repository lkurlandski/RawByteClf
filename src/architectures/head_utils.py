"""
Utilities for the classification and language modeling heads.
"""

from __future__ import annotations
import math
from typing import Literal, Optional
import warnings

import torch
from torch import nn, Tensor


class ShapeError(ValueError):

    def __init__(self, actual_shape: tuple, expected_shape: Optional[tuple] = None):
        self.expected_shape = tuple(expected_shape)
        self.actual_shape = tuple(actual_shape) if actual_shape else None
        super().__init__(f"Recieved: {self.actual_shape}. Expected: {self.expected_shape}")


def check_for_anomalous_weights(
    module: nn.Linear | nn.Embedding | Head,
    errors: Literal["raise", "ignore", "warn"] = "raise",
) -> None:
    if not isinstance(module, (nn.Linear, nn.Embedding, Head)):
        raise TypeError(f"{type(module)=}")

    if isinstance(module, Head):
        modules = [l for l in module.layers if isinstance(l, nn.Linear)] + [module.final_layer]
    else:
        modules = [module]

    for l in modules:
        w: Tensor = l.weight.to(torch.float64)
        m: float  = w.mean().cpu().item()
        s: float  = w.std().cpu().item()
        anomalous_mean = any([math.isnan(m), math.isinf(m), m <= -1, m >= 1])
        anomalous_std  = any([math.isnan(s), math.isinf(s), s <= 0, s >= 0.1])
        if anomalous_mean or anomalous_std:
            message = f"Detected anamalous weights (mean={m}, std={s})"
            if errors == "raise":
                raise RuntimeError(message)
            if errors == "warn":
                warnings.warn(message)


class Head(nn.Module):

    """
    A generic head with one of two structures:

        Linear(in_size, out_size)

        Linear(in_size, hidden_size)
        ReLU()
        Dropout(dropout_rate)
        ...
        Linear(hidden_size, out_size)
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = -1,
        num_hidden_layers: int = 0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        if num_hidden_layers == 0:
            self.final_layer = nn.Linear(in_size, out_size)
            return

        self.add_layer(in_size, hidden_size, dropout_rate)
        for _ in range(1, num_hidden_layers):
            self.add_layer(hidden_size, hidden_size, dropout_rate)
        self.final_layer = nn.Linear(hidden_size, out_size)

    def add_layer(self, in_size: int, out_size: int, dropout_rate: float) -> list[nn.Module]:
        self.layers.append(nn.Linear(in_size, out_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, last_hidden_state: Tensor) -> Tensor:
        if last_hidden_state.dim() != 3:
            raise ShapeError(last_hidden_state.shape, ("B", "T", "H"))

        x = last_hidden_state
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

    def get_output_embeddings(self) -> nn.Linear:
        return self.final_layer

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.final_layer = new_embeddings


def pool_logits(pooling: str, logits: Tensor, input_ids: Tensor, pad_token_id: int) -> Tensor:
    if logits.dim() != 3:
        raise ShapeError(logits.shape, ("B", "T", "H"))

    if pooling == "none":
        return logits

    if pooling == "mean":
        pooled_logits = torch.mean(logits, dim=1)
        return pooled_logits

    if pooling == "last":
        batch_size = logits.shape[0]
        sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

    raise ValueError(f"Invalid pooling: {pooling}")


def get_clm_loss(logits: Tensor, labels: Tensor, num_labels: int) -> Tensor:
    if logits.dim() != 3:
        raise ShapeError(logits.shape, ("B", "T", "V"))
    if labels.dim() != 2:
        raise ShapeError(labels.shape, ("B", "T"))

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shifted_logits.view(-1, num_labels), shifted_labels.view(-1))
    return loss


def get_mlm_loss(logits: Tensor, labels: Tensor, num_labels: int) -> Tensor:
    if logits.dim() != 3:
        raise ShapeError(logits.shape, ("B", "T", "V"))
    if labels.dim() != 2:
        raise ShapeError(labels.shape, ("B", "T"))

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


def get_clf_loss(logits: Tensor, labels: Tensor, num_labels: int, problem_type: str, ) -> Tensor:

    if problem_type == "regression":
        if logits.dim() != 1:
            raise ShapeError(logits.shape, ("B",))
        if labels.dim() != 1:
            raise ShapeError(labels.shape, ("B",))

        loss_fct = nn.MSELoss()
        if num_labels == 1:
            loss = loss_fct(logits.squeeze(), labels.squeeze())
            return loss
        loss = loss_fct(logits, labels)
        return loss

    if problem_type == "single_label_classification":
        if logits.dim() != 2:
            raise ShapeError(logits.shape, ("B", "C"))
        if labels.dim() != 1:
            raise ShapeError(labels.shape, ("B",))

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss

    if problem_type == "multi_label_classification":
        if logits.dim() != 2:
            raise ShapeError(logits.shape, ("B", "C"))
        if labels.dim() != 2:
            raise ShapeError(labels.shape, ("B", "C"))
        if not torch.is_floating_point(labels):
            raise TypeError(f"Expected labels to be a floating point, got {labels.dtype}")

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss

    raise ValueError(f"Invalid problem type: {problem_type}")
