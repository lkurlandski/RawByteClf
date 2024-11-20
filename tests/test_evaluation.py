import random
import unittest

import numpy as np
from scipy.special import softmax
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.utils import shuffle
from transformers import EvalPrediction
from src.learn.evaluation import *


N = 10
V = 256
T = 16384


random.seed(0)
np.random.seed(0)


class TestComputeMetrics(unittest.TestCase):

    def check(self, metric_computer: ComputeMetrics, eval_pred: EvalPrediction):
        result = metric_computer(eval_pred)
        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, float)

    def test_clf_1(self):
        X, y = make_classification(n_samples=N, n_classes=2)
        predictions = 100 * (np.random.rand(N, 2) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsBinary()
        self.check(metric_computer, eval_pred)

    def test_clf_2(self):
        X, y = make_classification(n_samples=N, n_classes=3, n_informative=3)
        predictions = 100 * (np.random.rand(N, 3) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsSingleLabel()
        self.check(metric_computer, eval_pred)

    def test_clf_3(self):
        X, y = make_multilabel_classification(n_samples=N, n_classes=3)
        predictions = 100 * (np.random.rand(N, 3) - 0.5)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=y)
        metric_computer = CLFComputeMetricsMultiLabel()
        self.check(metric_computer, eval_pred)

    def test_lm_1(self):
        predictions = 100 * (np.random.rand(N, T, V) - 0.5)
        label_ids = np.random.randint(0, V, size=(N, T))
        losses = np.array(12.0)
        eval_pred = EvalPrediction(predictions=np.random.rand(N, T, V), label_ids=np.random.randint(0, V, size=(N, T)), losses=losses)
        metric_computer = LMComputeMetrics(unigrams=None, raise_if_loss_not_present=True)
        self.check(metric_computer, eval_pred)

    def test_lm_2(self):
        predictions = 100 * (np.random.rand(N, T, V) - 0.5)
        label_ids = np.random.randint(0, V, size=(N, T))
        eval_pred = EvalPrediction(predictions=np.random.rand(N, T, V), label_ids=np.random.randint(0, V, size=(N, T)))
        metric_computer = LMComputeMetrics(unigrams=None, raise_if_loss_not_present=False)
        self.check(metric_computer, eval_pred)

    def test_lm_3(self):
        predictions = 100 * (np.random.rand(N, T, V) - 0.5)
        label_ids = np.random.randint(0, V, size=(N, T))
        losses = np.array(12.0)
        unigrams = softmax(np.random.rand(V))
        eval_pred = EvalPrediction(predictions=np.random.rand(N, T, V), label_ids=np.random.randint(0, V, size=(N, T)), losses=losses)
        metric_computer = LMComputeMetrics(unigrams=None, raise_if_loss_not_present=True)
        self.check(metric_computer, eval_pred)
