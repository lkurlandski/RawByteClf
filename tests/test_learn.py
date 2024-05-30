"""
Tests for the learn module.
"""

import math
import os
import shutil
import sys
import unittest

import torch
from torch import Tensor
from transformers import PreTrainedModel

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.architectures.mamba_hf import MambaConfig, MambaForCausalLM, MambaForSequenceClassification
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


if __name__ == "__main__":
    unittest.main()
