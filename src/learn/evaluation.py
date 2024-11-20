"""
Evaluation of models.
"""

from abc import ABC, abstractmethod
import math
import sys
import time
from typing import Literal, Optional

import numpy as np
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
from transformers import EvalPrediction
import torch
from torch import tensor


def compute_roc_auc(labels: np.ndarray, probabilities: np.ndarray, multi_class: str, average: str) -> float:
    try:
        return metrics.roc_auc_score(labels, probabilities, multi_class=multi_class, average=average)
    except ValueError as err:
        if "Only one class present in y_true." in str(err):
            return np.nan
        raise err


def compute_softmax(z: np.ndarray, axis: int) -> np.ndarray:
    # torch is about twice as fast as numpy.
    if z.size < 100_000:
        return softmax(z, axis=axis)
    return torch.nn.functional.softmax(torch.from_numpy(z), dim=axis).numpy()


def compute_perplexity(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    e^(
        -1/N sum(
            log P(x_i | x_0, x_1, ..., x_{i-1})
        )
    )
    """

    assert logits.ndim == 3, "Logits must have shape (B, T, V)."
    assert labels.ndim == 2, "Labels must have shape (B, T)."

    V = logits.shape[2]

    logits = logits.reshape(-1, V)  # (B * T, V)
    labels = labels.reshape(-1)     # (B * T,)

    mask   = labels != -100
    logits = logits[mask]
    labels = labels[mask]

    probs = compute_softmax(logits, axis=1)

    x = probs[np.arange(probs.shape[0]), labels]
    x = np.log(x)
    x = -np.mean(x)
    x = np.exp(x)

    return float(x)


class ComputeMetrics(ABC):
    """
    Compute metrics.

    Args:
      eval_pred (EvalPrediction): The evaluation predictions.
      compute_result (bool): Whether to compute the result.
    """

    include_for_metrics = []

    def __init__(self) -> None:
        self.results: list[dict]  = []
        self.weights: list[float] = []

    @abstractmethod
    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:
        ...

    def compute_result(self) -> dict[str, float]:
        if len(self.results) == 0:
            return {}
        if len(self.results) != len(self.weights):
            raise RuntimeError(f"{len(self.results)=} != {len(self.weights)=}")

        r = {}
        w = np.array(self.weights)
        for k in self.results[0].keys():
            a = np.array([d[k] for d in self.results])
            if a.ndim > 1:
                raise ValueError(f"Expected 1D array. Got {a.dim()=}.")
            m = np.isnan(a) | np.isinf(a)
            if len(a) == 0 or np.all(m):
                v = np.nan
            else:
                v = np.average(a[~m], weights=w[~m])
            r[k] = float(v)

        return r

    def update_and_return(self, result: dict[str, float], weight: float, compute_result: bool = True) -> dict[str, float]:
        result = {k: float(f) for k, f in result.items()}
        self.results.append(result)
        self.weights.append(weight)
        if compute_result:
            return self.compute_result()
        return result


class CLFComputeMetrics(ComputeMetrics):
    """
    Compute classification metrics.
    """

    def __init__(self, pos_label: int = 1, threshold: float = 0.5) -> None:
        self.pos_label = pos_label
        self.threshold = threshold
        super().__init__()


class CLFComputeMetricsBinary(CLFComputeMetrics):
    """
    Compute binary classification metrics.
    """

    averages    = ["binary"]
    multi_class = "raise"

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:

        labels  = eval_pred.label_ids  # (N,)
        support = labels.shape[0]
        if type_of_target(labels) != "binary":
            raise TypeError(f"Expected binary labels. Got {type_of_target(labels)}.")

        probabilities = eval_pred.predictions             # (N, 2)
        probabilities = softmax(probabilities, axis=1)    # (N, 2)
        predictions   = np.argmax(probabilities, axis=1)  # (N,)
        probabilities = probabilities[:, 1]               # (N,)

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average="binary", pos_label=self.pos_label)
        report = {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "hamming_loss": metrics.hamming_loss(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc-auc": compute_roc_auc(labels, probabilities, self.multi_class, None),
        }

        return self.update_and_return(report, support, compute_result)


class CLFComputeMetricsSingleLabel(CLFComputeMetrics):
    """
    Compute multiclass classification metrics.
    """

    averages = ["macro", "weighted", "micro"]
    multi_class = "ovr"

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:

        labels  = eval_pred.label_ids  # (N,)
        support = labels.shape[0]
        if type_of_target(labels) != "multiclass":
            raise TypeError(f"Expected binary labels. Got {type_of_target(labels)}.")

        probabilities = eval_pred.predictions             # (N, C)
        probabilities = softmax(probabilities, axis=1)    # (N, C) | (N,)
        predictions   = np.argmax(probabilities, axis=1)  # (N,)

        report = {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "hamming_loss": metrics.hamming_loss(labels, predictions),
        }
        for average in self.averages:
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average=average, pos_label=self.pos_label)
            roc_auc = compute_roc_auc(labels, probabilities, self.multi_class, average)
            report |= {
                f"precision-{average}": precision,
                f"recall-{average}": recall,
                f"f1-{average}": f1,
                f"roc-auc-{average}": roc_auc,
            }

        return self.update_and_return(report, support, compute_result)


class CLFComputeMetricsMultiLabel(CLFComputeMetrics):
    """
    Compute multilabel classification metrics.
    """

    averages = ["macro", "weighted", "micro"]
    multi_class = "ovr"

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:

        labels  = eval_pred.label_ids  # (N, C)
        support = labels.shape[0]
        if type_of_target(labels) != "multilabel-indicator":
            raise TypeError(f"Expected binary labels. Got {type_of_target(labels)}.")
        probabilities = expit(eval_pred.predictions)                           # (N, C)
        predictions   = (probabilities > self.threshold).astype(labels.dtype)  # (N, C)

        report = {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "hamming_loss": metrics.hamming_loss(labels, predictions),
        }

        for average in self.averages:
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average=average, pos_label=self.pos_label)
            roc_auc = compute_roc_auc(labels, probabilities, self.multi_class, average)
            avg_precision = metrics.average_precision_score(labels, probabilities, average=average, pos_label=self.pos_label)
            report |= {
                f"precision-{average}": precision,
                f"recall-{average}": recall,
                f"f1-{average}": f1,
                f"roc-auc-{average}": roc_auc,
                f"average_precision-{average}": avg_precision,
            }

        report["coverage_error"] = metrics.coverage_error(labels, probabilities)
        report["label_ranking_average_precision_score"] = metrics.label_ranking_average_precision_score(labels, probabilities)
        report["label_ranking_loss"] = metrics.label_ranking_loss(labels, probabilities)

        return self.update_and_return(report, support, compute_result)


class LMComputeMetrics(ComputeMetrics):
    """
    Compute language modeling metrics.
    """

    include_for_metrics = ["losses"]

    def __init__(self, unigrams: Optional[np.ndarray] = None, raise_if_loss_not_present: bool = True) -> None:
        super().__init__()
        self.unigrams = unigrams
        self.raise_if_loss_not_present = raise_if_loss_not_present

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[float, str]:

        y_true, y_pred = LMComputeMetrics.get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        support = eval_pred.label_ids[eval_pred.label_ids != -100].size

        if getattr(eval_pred, "losses", None) is None:
            if self.raise_if_loss_not_present:
                raise ValueError("Expected loss to be present.")
            ppl = compute_perplexity(eval_pred.predictions, eval_pred.label_ids)
        else:
            if eval_pred.losses.ndim != 0:
                raise ValueError(f"Expected scalar loss. Got {eval_pred.losses.shape}.")
            ppl = np.exp(eval_pred.losses)

        report = {
            "ppl": ppl,
            "accuracy": metrics.accuracy_score(y_true, y_pred),
            "f1-macro": metrics.f1_score(y_true, y_pred, average="macro"),
        }
        if self.unigrams is not None:
            labels = eval_pred.label_ids[eval_pred.label_ids != -100]
            scale = np.mean(np.log(self.unigrams[labels]))
            report["nppl"] = report["ppl"] - scale

        return self.update_and_return(report, support, compute_result)

    @staticmethod
    def get_y_true_y_pred(predictions: np.ndarray, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        assert predictions.ndim == 3, f"Got shape={tuple(predictions.shape)}. Expected (B, T, V)."
        assert label_ids.ndim == 2, f"Got shape={tuple(label_ids.shape)}. Expected (B, T)."

        def get_y_pred(predictions: np.ndarray) -> np.ndarray:
            predictions = tensor(predictions, dtype=torch.float32)
            predictions = predictions.view(-1, predictions.shape[2])  # (B * L, M)
            probas = torch.softmax(predictions, dim=1).numpy()
            y_pred = np.argmax(probas, axis=1).astype(np.int32)
            return y_pred

        def get_y_true(label_ids: np.ndarray) -> np.ndarray:
            y_true = tensor(label_ids, dtype=torch.float32).view(-1)  # (B * L,)
            return y_true.numpy().astype(np.int32)

        y_pred = get_y_pred(predictions)
        y_true = get_y_true(label_ids)
        mask = y_true == -100
        y_pred = y_pred[~mask]
        y_true = y_true[~mask]
        return y_true, y_pred


class MLMComputeMetrics(LMComputeMetrics):
    """
    Compute masked language model metrics.
    """


class CLMComputeMetrics(LMComputeMetrics):
    """
    Compute causal language model metrics.
    """
