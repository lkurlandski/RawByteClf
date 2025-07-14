"""
Test.
"""

import random
import unittest

import numpy as np
from scipy.special import softmax
from sklearn.datasets import make_classification, make_multilabel_classification
from transformers import EvalPrediction
from src.learn.evaluation import (
    ComputeMetrics,
    CLFComputeMetricsBinary,
    CLFComputeMetricsSingleLabel,
    CLFComputeMetricsMultiLabel,
    LMComputeMetrics,
    CLMComputeMetrics,
    MLMComputeMetrics,
)


N = 10
V = 256
T = 16384


random.seed(0)
np.random.seed(0)


class TestComputeMetrics(unittest.TestCase):

    def setUp(self):
        self.lm_predictions = 100 * (np.random.rand(N, T, V) - 0.5)
        self.lm_label_ids = np.random.randint(0, V, size=(N, T))

    def check(self, metric_computer: ComputeMetrics, eval_pred: EvalPrediction):
        result = metric_computer(eval_pred)
        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, float)

    def test_clf_1(self):
        _, y = make_classification(n_samples=N, n_classes=2)
        predictions = 100 * (np.random.rand(N, 2) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsBinary()
        self.check(metric_computer, eval_pred)

    def test_clf_2(self):
        _, y = make_classification(n_samples=N, n_classes=3, n_informative=3)
        predictions = 100 * (np.random.rand(N, 3) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsSingleLabel()
        self.check(metric_computer, eval_pred)

    def test_clf_3(self):
        _, y = make_multilabel_classification(n_samples=N, n_classes=3)  # pylint: disable=unbalanced-tuple-unpacking
        predictions = 100 * (np.random.rand(N, 3) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsMultiLabel()
        self.check(metric_computer, eval_pred)

    def test_lm_1(self):
        eval_pred = EvalPrediction(predictions=self.lm_predictions, label_ids=self.lm_label_ids, losses=np.array(12.0))  # pylint: disable=unexpected-keyword-arg
        metric_computer = LMComputeMetrics(unigrams=None, raise_if_loss_not_present=True, basic_metrics=False)
        self.check(metric_computer, eval_pred)

    def test_lm_2(self):
        eval_pred = EvalPrediction(predictions=self.lm_predictions, label_ids=self.lm_label_ids, losses=None)  # pylint: disable=unexpected-keyword-arg
        metric_computer = LMComputeMetrics(unigrams=None, raise_if_loss_not_present=False, basic_metrics=False)
        self.check(metric_computer, eval_pred)

    def test_lm_3(self):
        eval_pred = EvalPrediction(predictions=self.lm_predictions, label_ids=self.lm_label_ids, losses=np.array(12.0))  # pylint: disable=unexpected-keyword-arg
        metric_computer = LMComputeMetrics(unigrams=softmax(np.random.rand(V)), raise_if_loss_not_present=True, basic_metrics=False)
        self.check(metric_computer, eval_pred)

    def test_lm_4(self):
        eval_pred = EvalPrediction(predictions=self.lm_predictions, label_ids=self.lm_label_ids, losses=None)  # pylint: disable=unexpected-keyword-arg
        metric_computer = LMComputeMetrics(unigrams=softmax(np.random.rand(V)), raise_if_loss_not_present=False, basic_metrics=False)
        self.check(metric_computer, eval_pred)
