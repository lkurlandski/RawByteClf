"""
Tests for the learn module.
"""

from functools import partial
import math
import os
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
import torch
from torch import Tensor
from transformers import PreTrainedModel, TrainingArguments, EvalPrediction

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.architectures.mamba_hf import MambaConfig, MambaForCausalLM, MambaForSequenceClassification
from src.learn.helpers import OutputHelper
from src.learn.evaluation import clf_compute_metrics
from src.learn.train import get_model


class TestGetModel(unittest.TestCase):
    """
    Mamba weights have been observed to be anomalous in the past.
    This test checks that the weights are within some normal bounds.
    """

    num_labels = 999
    num_hidden_layers = 8
    hidden_size = 256
    embedding_size = 256
    vocab_size = 256
    initializer_range = 0.1

    tmp_clm_dir = "/tmp/mamba_clm"
    tmp_clf_dir = "/tmp/mamba_clf"

    def setUp(self) -> None:
        self.id2label = {i: f"label_{i}" for i in range(self.num_labels)}
        self.label2id = {f"label_{i}": i for i in range(self.num_labels)}
        self.clf_kwds = {
            "num_labels": self.num_labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }

        config = self.get_config()
        model = MambaForCausalLM(config)
        model.save_pretrained(self.tmp_clm_dir)

        config = self.get_config(**self.clf_kwds)
        model = MambaForSequenceClassification(config)
        model.save_pretrained(self.tmp_clf_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_clm_dir)
        shutil.rmtree(self.tmp_clf_dir)

    def get_config(self, **kwds):
        return MambaConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            vocab_size=self.vocab_size,
            initializer_range=self.initializer_range,
            **kwds,
        )

    def check_clf_head(self, model: PreTrainedModel, head_names: tuple[str] = ("clf_head",)) -> None:
        for h in head_names:
            l: torch.nn.Linear = getattr(model, h)
            if not isinstance(l, torch.nn.Linear):
                raise TypeError(f"Expected torch.nn.Linear, got {type(l)}")
            w: Tensor = l.weight.to(torch.float64)
            m: float = w.mean().cpu().item()
            s: float = w.std().cpu().item()
            anomalous_mean = any([math.isnan(m), math.isinf(m), m <= -1, m >= 1])
            anomalous_std = any([math.isnan(s), math.isinf(s), s <= 0, s >= 2 * self.initializer_range])
            assert not anomalous_mean, f"Anomalous mean in {h} weights: {m}"
            assert not anomalous_std, f"Anomalous std in {h} weights: {s}"
            print(f"{h} weights: mean={m}, std={s}")

    def test_from_config(self) -> None:
        print("Testing model from config...")
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task="clf",
            model_name_or_path=None,
            config=config,
            **self.clf_kwds,
        )
        # print(model)
        self.check_clf_head(model)

    def test_from_clf_checkpoint(self) -> None:
        print("Testing model from CLF checkpoint...")
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task="clf",
            model_name_or_path=self.tmp_clf_dir,
            config=config,
            **self.clf_kwds,
        )
        # print(model)
        self.check_clf_head(model)

    def test_from_clm_checkpoint(self) -> None:
        print("Testing model from CLM checkpoint...")
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task="clf",
            model_name_or_path=self.tmp_clm_dir,
            config=config,
            **self.clf_kwds,
        )
        # print(model)
        self.check_clf_head(model)


class TestOutputHelper(unittest.TestCase):

    def setUp(self) -> None:
        self.root = Path("/tmp/output/")
        self.root.mkdir(parents=True, exist_ok=True)

        training_arguments = TrainingArguments(self.root)
        self.kwds = {
            "root": self.root,
            "packing_protocol": "any",
            "representation": 8,
            "algorithm": "Raw",
            "vocab_size": 256,
            "max_length": 65536,
            "arch_config": {"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256},
            "tr_size": 0.85,
            "depth": 1,
            "min_freq": None,
            "top_k": 10,
            "tr_samples_per_class": None,
            "tr_length_cutoff": None,
            "trainer_config": training_arguments.__dict__ | {"world_size": training_arguments.world_size},
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_from_path(self) -> None:

        oh_clm = OutputHelper(**self.kwds | {"task": "clm", "model_name_or_path": "mamba"})
        oh_clm.path.mkdir(parents=True, exist_ok=True)
        oh = OutputHelper.from_path(oh_clm.path)
        assert oh_clm == oh, f"Expected \"\"\"\n{oh_clm}\n\"\"\", got \"\"\"\n{oh}\n\"\"\""

        oh_clf = OutputHelper(**self.kwds | {"task": "clf-bod", "model_name_or_path": "mamba"})
        oh_clf.path.mkdir(parents=True, exist_ok=True)
        oh = OutputHelper.from_path(oh_clf.path)
        assert oh_clf == oh, f"Expected \"\"\"\n{oh_clf}\n\"\"\", got \"\"\"\n{oh}\n\"\"\""

        model_name_or_path = oh_clm.checkpoints_dir / "checkpoint-666"
        model_name_or_path.mkdir(parents=True, exist_ok=True)
        oh_ft = OutputHelper(**self.kwds | {"task": "clf-bod", "model_name_or_path": model_name_or_path})
        oh_ft.path.mkdir(parents=True, exist_ok=True)
        oh = OutputHelper.from_path(oh_ft.path)
        assert oh_ft == oh, f"Expected \"\"\"\n{oh_ft}\n\"\"\", got \"\"\"\n{oh}\n\"\"\""

    def test_get_finetuning_model_name_or_path(self) -> None:
        oh = OutputHelper(**self.kwds | {"task": "clm", "model_name_or_path": "mamba"})
        oh.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        for i in [100, 200, 300]:
            (oh.checkpoints_dir / f"checkpoint-{i}").mkdir()

        p = OutputHelper.get_finetuning_model_name_or_path(
            "clm", **self.kwds | {"task": "clf-bod", "model_name_or_path": "mamba"}
        )
        highest = oh.checkpoints_dir / "checkpoint-300"
        assert p == highest.as_posix(), f"Expected \"{highest}\", got \"{p}\""


class TestComputeMetrics(unittest.TestCase):

    seeds = [0, 1, 2, 3, 4]

    # def _test_clf_compute_metrics_binary(self, seed: int) -> None:
    #     ...

    def _test_clf_compute_metrics_multiclass_singlelabel(self, seed: int) -> None:
        predictions = np.random.rand(1000, 100)
        label_ids = make_classification(n_samples=1000, n_classes=100, n_informative=8, random_state=seed)[1]
        eval_pred = EvalPrediction(predictions, label_ids)
        report = clf_compute_metrics(eval_pred, problem_type="single_label_classification")
        print(report)

    def _test_clf_compute_metrics_multiclass_multilabel(self, seed: int) -> None:
        predictions = np.random.rand(1000, 100)
        label_ids = make_multilabel_classification(n_samples=1000, n_classes=100, n_labels=5, random_state=seed)[1]
        eval_pred = EvalPrediction(predictions, label_ids)
        report = clf_compute_metrics(eval_pred, problem_type="multi_label_classification")
        print(report)

    def test_clf_compute_metrics_multiclass_multilabel(self) -> None:
        for seed in self.seeds:
            self._test_clf_compute_metrics_multiclass_multilabel(seed)

    def test_clf_compute_metrics_multiclass_singlelabel(self) -> None:
        for seed in self.seeds:
            self._test_clf_compute_metrics_multiclass_singlelabel(seed)

    # def test_clf_compute_metrics_binary(self) -> None:
    #     for seed in self.seeds:
    #         self._test_clf_compute_metrics_binary(seed)


if __name__ == "__main__":
    unittest.main()
