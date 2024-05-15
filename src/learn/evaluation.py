"""
Evaluation of models.

TODO: using the object-oriented ComputeMetrics seems to cause memory leaks.
TODO: refactor out the use of huggingface's evaluate module.
"""

from typing import Optional

import evaluate
import numpy as np
from sklearn.metrics import classification_report
from transformers import EvalPrediction
import torch
from torch import tensor


ACCURACY = evaluate.load("accuracy")
F1 = evaluate.load("f1")


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
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
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
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
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
    eval_pred: EvalPrediction, single_shot_classes: Optional[list[int]] = None
) -> dict[str, float]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    metrics = {
        "accuracy": ACCURACY.compute(predictions=predictions, references=labels)["accuracy"],
        "f1-macro": F1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1-weighted": F1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "f1-micro": F1.compute(predictions=predictions, references=labels, average="micro")["f1"],
    }
    if single_shot_classes is None or single_shot_classes == []:
        return metrics

    include = np.array([i for i, l in enumerate(labels) if l in single_shot_classes])
    predictions = predictions[include]
    labels = labels[include]
    ss_metrics = {
        "ss_accuracy": ACCURACY.compute(predictions=predictions, references=labels)["accuracy"],
        "ss_f1-macro": F1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "ss_f1-weighted": F1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "ss_f1-micro": F1.compute(predictions=predictions, references=labels, average="micro")["f1"],
    }
    metrics.update(ss_metrics)
    return metrics


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
        "accuracy": ACCURACY.compute(predictions=y_pred, references=y_true)["accuracy"],
        "f1-macro": F1.compute(predictions=y_pred, references=y_true, average="macro")["f1"],
        "f1-micro": F1.compute(predictions=y_pred, references=y_true, average="micro")["f1"],
    }
