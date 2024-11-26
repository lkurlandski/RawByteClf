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
from torch import tensor, Tensor


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


# TODO: filtering out labels whose value is -100 is not sufficient.
# Modify to simply expect the nessecary perprocessing to take place before hand.
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


def numpify_eval_prediction(eval_pred: EvalPrediction) -> EvalPrediction:
    d = {}
    for k in ["predictions", "label_ids", "inputs", "losses"]:
        if (v := getattr(eval_pred, k, None)) is None:
            continue
        if isinstance(v, Tensor):
            v = v.numpy(force=True)
        d[k] = v
    return EvalPrediction(**d)


def eval_prediction_to_cpu(eval_pred: EvalPrediction) -> EvalPrediction:
    d = {}
    for k in ["predictions", "label_ids", "inputs", "losses"]:
        if (v := getattr(eval_pred, k, None)) is None:
            continue
        if isinstance(v, Tensor) and v.device.type != "cpu":
            v = v.to("cpu")
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v).to("cpu")
        d[k] = v
    return EvalPrediction(**d)


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
                raise ValueError(f"Expected 1D array. Got {a.ndim=}.")
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
            "balanced_accuracy": metrics.balanced_accuracy_score(labels, predictions),
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
            "balanced_accuracy": metrics.balanced_accuracy_score(labels, predictions),
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


def are_any_nan(x: Tensor | np.ndarray) -> bool:
    if isinstance(x, Tensor):
        return bool(torch.isnan(x).any())
    return bool(np.isnan(x).any())


def are_any_inf(x: Tensor | np.ndarray) -> bool:
    if isinstance(x, Tensor):
        return bool(torch.isinf(x).any())
    return bool(np.isinf(x).any())


class LMComputeMetrics(ComputeMetrics):
    """
    Compute language modeling metrics.
    """

    # The value here needs to be "loss" while the name in EvalPrediction is "losses".
    include_for_metrics = ["loss"]

    def __init__(self,
        unigrams: Optional[np.ndarray] = None,
        raise_if_loss_not_present: bool = True,
        basic_metrics: bool = True,
        check: bool = True,
        special_token_ids: tuple[int] = (-100,),
        cpu: bool = True,
    ) -> None:
        super().__init__()
        self.unigrams = unigrams
        self.raise_if_loss_not_present = raise_if_loss_not_present
        self.basic_metrics = basic_metrics
        self.check = check
        self.special_token_ids = np.array(special_token_ids, dtype=np.int64)
        self.cpu = cpu

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = True) -> dict[float, str]:
        if self.cpu:
            eval_pred = eval_prediction_to_cpu(eval_pred)

        if isinstance(eval_pred.label_ids, Tensor) and isinstance(self.unigrams, np.ndarray):
            self.unigrams = torch.from_numpy(self.unigrams).to(eval_pred.label_ids.device)
            self.special_token_ids = torch.from_numpy(self.special_token_ids).to(eval_pred.label_ids.device)
        elif isinstance(eval_pred.label_ids, np.ndarray) and isinstance(self.unigrams, Tensor):
            self.unigrams = self.unigrams.numpy(force=True)
            self.special_token_ids = self.special_token_ids.numpy(force=True)

        # Determine the indices corresponding to special tokens.
        if isinstance(eval_pred.label_ids, np.ndarray):
            ignore = np.isin(eval_pred.label_ids, self.special_token_ids)
        elif isinstance(eval_pred.label_ids, Tensor):
            ignore = torch.isin(eval_pred.label_ids, self.special_token_ids)

        # Get the relevant data from the EvalPrediction.
        labels      = eval_pred.label_ids[~ignore]
        predictions = eval_pred.predictions[~ignore]
        support = labels.size if isinstance(labels, np.ndarray) else labels.numel()

        # Perplexity
        if getattr(eval_pred, "losses", None) is None:
            if self.raise_if_loss_not_present:
                raise ValueError("Expected losses to be present.")
            ppl = compute_perplexity(predictions, labels)
        else:
            loss = eval_pred.losses.mean()
            loss = loss.detach().cpu().item() if isinstance(loss, Tensor) else float(loss)
            ppl = math.exp(loss)
        report = {"ppl": ppl}

        # Normalized Perplexity
        if self.unigrams is not None:
            word_probs = self.unigrams[labels]
            if self.check:
                if are_any_nan(word_probs):
                    raise ValueError(f"Detected NAN in word_probs.\n{self.unigrams.tolist()=}\n{labels.tolist()=}\n{word_probs.tolist()=}")
                if are_any_inf(word_probs):
                    raise ValueError(f"Detected INF in word_probs.\n{self.unigrams.tolist()=}\n{labels.tolist()=}\n{word_probs.tolist()=}")
                if bool((word_probs > 1.0).any()):
                    raise ValueError(f"Detected value > 1.0 in word_probs.\n{self.unigrams.tolist()=}\n{labels.tolist()=}\n{word_probs.tolist()=}")
                if bool((word_probs < 0.0).any()):
                    raise ValueError(f"Detected value < 0.0 in word_probs.\n{self.unigrams.tolist()=}\n{labels.tolist()=}\n{word_probs.tolist()=}")

            if isinstance(word_probs, np.ndarray):
                word_probs = np.clip(word_probs, a_min=1e-10, a_max=None)
                word_probs = np.log(word_probs)
            else:
                word_probs = torch.clamp(word_probs, min=1e-10)
                word_probs = torch.log(word_probs)

            scale = word_probs.mean()
            scale = scale.detach().cpu().item() if isinstance(scale, Tensor) else float(scale)
            report["nppl"] = report["ppl"] - scale

        # Basic metrics
        if self.basic_metrics:
            y_true = labels.numpy(force=True) if isinstance(labels, Tensor) else labels
            y_pred = predictions.numpy(force=True) if isinstance(predictions, Tensor) else labels
            report |= {
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "f1-macro": metrics.f1_score(y_true, y_pred, average="macro"),
            }

        return self.update_and_return(report, support, compute_result)


class MLMComputeMetrics(LMComputeMetrics):
    """
    Compute masked language model metrics.
    """


class CLMComputeMetrics(LMComputeMetrics):
    """
    Compute causal language model metrics.
    """
