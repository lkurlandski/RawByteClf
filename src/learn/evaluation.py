"""
Evaluation of models.
"""

from abc import ABC, abstractmethod
import sys
import time
from typing import Literal

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

    probs = softmax(logits, axis=1)
    loss  = log_loss(labels, probs, labels=np.array(list(range(vocab_size))), normalize=True)
    ppl   = np.exp(loss)

    return float(ppl)


def compute_pseudo_perplexity(logits: np.ndarray, labels: np.ndarray, inputs: np.ndarray, mask_token_id: int) -> float:
    """
    This is basically perplexity, except only for the tokens that were masked.
    The context is already factored in since the logits were computed under this context.
    """

    assert logits.ndim == 3, "Logits must have shape (B, T, V)."
    assert labels.ndim == 2, "Labels must have shape (B, T)."

    V = logits.shape[2]

    logits = logits.reshape(-1, V)  # (B * T, V)
    labels = labels.reshape(-1)     # (B * T,)
    inputs = inputs.reshape(-1)     # (B * T,)

    mask   = (labels != -100) & (inputs == mask_token_id)
    logits = logits[mask]
    labels = labels[mask]

    probs = softmax(logits, axis=1)
    loss  = log_loss(labels, probs, labels=np.array(list(range(vocab_size))), normalize=True)
    ppl   = np.exp(loss)

    return float(ppl)


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
            m = np.isnan(a) | np.isinf(a)
            v = float(np.average(a[~m], weights=w[~m]))
            r[k] = v

        return r

    def update_and_return(self, result: dict[str, float], weight: float, compute_result: bool = True) -> dict[str, float]:
        result = {k: float(f) for k, f in result.items()}
        self.results.append(result)
        self.weights.append(weight)
        if compute_result:
            return self.compute_result()
        return result


class CLFSingleLabelCompute(ComputeMetrics):
    """
    Compute classification metrics.
    """

    def __init__(self, threshold: float = 0.5, pos_label: int = 1) -> None:
        self.threshold = threshold
        self.pos_label = pos_label
        super().__init__()

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:
        report  = {}
        support = eval_pred.label_ids.shape[0]

        labels        = eval_pred.label_ids               # (N,)
        probabilities = eval_pred.predictions             # (N, C) 
        probabilities = softmax(probabilities, axis=1)    # (N, C) | (N,)
        predictions   = np.argmax(probabilities, axis=1)  # (N,)
 
        if type_of_target(labels) == "binary":
            probabilities = probabilities[:, 1]
            averages = [None]
            multi_class = "raise"
        else:
            averages = ["macro", "weighted", "micro"]
            multi_class = "ovr"

        report["accuracy"]     = metrics.accuracy_score(labels, predictions)
        report["hamming_loss"] = metrics.hamming_loss(labels, predictions)

        for average in averages:
            _average = "binary" if average is None else average
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average=_average, pos_label=self.pos_label)
            roc_auc = compute_roc_auc(labels, probabilities, multi_class, average)
            report |= {
                f"precision-{average}": precision,
                f"recall-{average}": recall,
                f"f1-{average}": f1,
                f"roc-auc-{average}": roc_auc,
            }

        return self.update_and_return(report, support, compute_result)


class CLFComputeMetricsMultiLabel(ComputeMetrics):
    """
    Compute classification metrics.
    """

    def __init__(self, threshold: float = 0.5, pos_label: int = 1) -> None:
        self.threshold = threshold
        self.pos_label = pos_label
        super().__init__()

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:
        report  = {}
        support = eval_pred.label_ids.shape[0]

        labels        = eval_pred.label_ids               # (N, C)
        probabilities = eval_pred.predictions             # (N, C)
        probabilities = expit(probabilities)              # (N, C)
        predictions   = probabilities > self.threshold    # (N, C)

        averages = ["macro", "weighted", "micro", "samples"]
        multi_class = "ovr"

        report["accuracy"]     = metrics.accuracy_score(labels, predictions)
        report["hamming_loss"] = metrics.hamming_loss(labels, predictions)

        for average in averages:
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average=average, pos_label=self.pos_label)
            roc_auc = compute_roc_auc(labels, probabilities, multi_class, average)
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


def get_y_true_y_pred(predictions: np.ndarray, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    assert predictions.ndim == 3, f"Got shape={tuple(predictions.shape)}. Expected (B, T, V)."
    assert label_ids.ndim == 3, f"Got shape={tuple(label_ids.shape)}. Expected (B, T, V)."

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


class MLMComputeMetrics(ComputeMetrics):
    """
    Compute masked language model metrics.
    """

    include_for_metrics = ["inputs"]

    def __init__(self, mask_token_id: int) -> None:
        super().__init__()

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[float, str]:
        report  = {}
        support = eval_pred.label_ids.shape[0]

        pseudo_perplexity = compute_pseudo_perplexity(eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs, mask_token_id)
        y_true, y_pred = get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro")

        report |= {
            "pseudo_perplexity": pseudo_perplexity,
            "accuracy": accuracy,
            "f1-macro": f1_macro,
        }

        return self.update_and_return(report, support, compute_result)


class CLMComputeMetrics(ComputeMetrics):

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[float, str]:
        report  = {}
        support = eval_pred.label_ids.shape[0]

        perplexity = compute_perplexity(eval_pred.predictions, eval_pred.label_ids)
        y_true, y_pred = get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro")

        report |= {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "f1-macro": f1_macro,
        }

        return self.update_and_return(report, support, compute_result)
