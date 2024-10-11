"""
Evaluation of models.

TODO: using the object-oriented ComputeMetrics seems to cause memory leaks.
TODO: refactor out the use of huggingface's evaluate module.
"""

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


class ComputeMetrics:
    def __init__(self, detailed: bool = False) -> None:
        self.detailed = detailed

    def set_detailed(self, detailed: bool) -> None:
        self.detailed = detailed

    def return_report(self, report: dict[str, float | dict]) -> dict[str, float | dict]:
        if self.detailed:
            return report
        return {
            "accuracy": report["accuracy"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }


class CLFComputeMetrics(ComputeMetrics):
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float | dict]:
        # predictions (B, M)
        # label_ids (B,)
        y_true, y_pred = self.get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
        return super().return_report(report)

    @staticmethod
    def get_y_true_y_pred(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return CLFComputeMetrics.get_y_true(label_ids), CLFComputeMetrics.get_y_pred(predictions)

    @staticmethod
    def get_y_pred(predictions: np.ndarray) -> np.ndarray:
        predictions = tensor(predictions, dtype=torch.float32)
        probas = torch.softmax(predictions, dim=1).numpy()
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    @staticmethod
    def get_y_true(label_ids: np.ndarray) -> np.ndarray:
        return label_ids.astype(np.int64)


class MLMComputeMetrics(ComputeMetrics):
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float | dict]:
        # predictions (B, L, M)
        # label_ids (B, M)
        y_true, y_pred = self.get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
        return super().return_report(report)

    @staticmethod
    def get_y_true_y_pred(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = MLMComputeMetrics.get_y_pred(predictions)
        y_true = MLMComputeMetrics.get_y_true(label_ids)
        mask = y_true == -100
        y_pred = y_pred[~mask]
        y_true = y_true[~mask]
        return y_true, y_pred

    @staticmethod
    def get_y_pred(predictions: np.ndarray) -> np.ndarray:
        predictions = tensor(predictions, dtype=torch.float32)
        predictions = predictions.view(-1, predictions.shape[2])  # (B * L, M)
        probas = torch.softmax(predictions, dim=1).numpy()
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    @staticmethod
    def get_y_true(label_ids: np.ndarray) -> np.ndarray:
        y_true = tensor(label_ids, dtype=torch.float32).view(-1)  # (B * L,)
        return y_true.numpy().astype(np.int64)


def clf_compute_metrics(
    eval_pred: EvalPrediction,
    problem_type: Literal["single_label_classification", "multi_label_classification"],
) -> dict[str, float]:
    """
    Compute classification metrics.

    For binary classification, it is critical that the positive label is 1!

    Args:
      eval_pred: EvalPrediction object.
      problem_type: Type of classification problem.

    Returns:
      dict: Classification metrics, containing the following keys:
        - accuracy
        - precision-macro
        - precision-weighted
        - precision-micro
        - precision-samples (only for multi-label classification)
        - recall-macro
        - recall-weighted
        - recall-micro
        - recall-samples (only for multi-label classification)
        - f1-macro
        - f1-weighted
        - f1-micro
        - f1-samples (only for multi-label classification)
        - roc-auc-macro
        - roc-auc-weighted
        - roc-auc-micro
        - roc-auc-samples (only for multi-label classification)
        - coverage_error (only for multi-label classification)
        - label_ranking_average_precision_score (only for multi-label classification)
        - label_ranking_loss (only for multi-label classification)
    """
    probabilities, labels = eval_pred.predictions, eval_pred.label_ids

    print("Computing metrics...", end="")
    t_0 = time.time()

    if problem_type not in ("single_label_classification", "multi_label_classification"):
        raise ValueError(f"Invalid problem type: {problem_type=}.")


    multi_class = "ovr"

    if problem_type == "single_label_classification":
        # probabilities, predictions (N, C)
        # labels (N,)
        probabilities = softmax(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        if type_of_target(labels) == "binary":
            # probabilities (N,)
            probabilities = probabilities[:, 1]
            averages = [None]
            multi_class = "raise"
        else:
            averages = ["macro", "weighted", "micro"]

    if problem_type == "multi_label_classification":
        # probabilities, predictions (N, C)
        # labels (N, C)
        probabilities = expit(probabilities)
        predictions = probabilities > 0.5
        averages = ["macro", "weighted", "micro", "samples"]

    report = {}

    report["accuracy"] = metrics.accuracy_score(labels, predictions)
    report["hamming_loss"] = metrics.hamming_loss(labels, predictions)

    for average in averages:
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average=average if average else "binary")

        try:
            roc_auc = metrics.roc_auc_score(labels, probabilities, multi_class=multi_class, average=average)
        except ValueError as err:
            if "Only one class present in y_true." in str(err):
                roc_auc = np.nan
            else:
                raise err

        r = {
            f"precision-{average}": precision,
            f"recall-{average}": recall,
            f"f1-{average}": f1,
            f"roc-auc-{average}": roc_auc,
        }
        if problem_type == "multi_label_classification":
            report[f"average_precision-{average}"] = metrics.average_precision_score(labels, probabilities, average=average)
        report.update(r)

    if problem_type == "multi_label_classification":
        report["coverage_error"] = metrics.coverage_error(labels, probabilities)
        report["label_ranking_average_precision_score"] = metrics.label_ranking_average_precision_score(labels, probabilities)
        report["label_ranking_loss"] = metrics.label_ranking_loss(labels, probabilities)

    print(f"Done. Took {time.time() - t_0:.2f} seconds.")

    return report


def mlm_get_y_true_y_pred(predictions: np.ndarray, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pred = mlm_get_y_pred(predictions)
    y_true = mlm_get_y_true(label_ids)
    mask = y_true == -100
    y_pred = y_pred[~mask]
    y_true = y_true[~mask]
    return y_true, y_pred


def mlm_get_y_pred(predictions: np.ndarray) -> np.ndarray:
    predictions = tensor(predictions, dtype=torch.float32)
    predictions = predictions.view(-1, predictions.shape[2])  # (B * L, M)
    probas = torch.softmax(predictions, dim=1).numpy()
    y_pred = np.argmax(probas, axis=1).astype(np.int32)
    return y_pred


def mlm_get_y_true(label_ids: np.ndarray) -> np.ndarray:
    y_true = tensor(label_ids, dtype=torch.float32).view(-1)  # (B * L,)
    return y_true.numpy().astype(np.int32)


def mlm_compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    y_true, y_pred = mlm_get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1-macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "f1-weighted": metrics.f1_score(y_true, y_pred, average="weighted"),
        "f1-micro": metrics.f1_score(y_true, y_pred, average="micro"),
    }
