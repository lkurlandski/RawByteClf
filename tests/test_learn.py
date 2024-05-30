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

from src.architectures.mamba_hf import (
    MambaConfig,
    MambaModel,
    MambaForCausalLM,
    MambaForMaskedLM,
    MambaForSequenceClassification,
)
from src.learn.train import get_model


class TestMambaWeightsInitializedCorrect(unittest.TestCase):

    num_labels = 999
    num_hidden_layers = 8
    hidden_size = 256
    embedding_size = 8
    vocab_size = 256
    initializer_range = 0.1

    layer_names = ("clf_head", "embedding_projection")

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
        model = MambaModel(config)
        print(model)

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

    def check_linear_layers(self, model: MambaForCausalLM | MambaForMaskedLM | MambaForSequenceClassification) -> None:
        layers: list[tuple[str, torch.nn.Linear]]
        if isinstance(model, MambaForSequenceClassification):
            layers = [("clf_head", model.clf_head)]
        elif isinstance(model, (MambaForCausalLM, MambaForMaskedLM)):
            layers = [("lm_head", model.lm_head)]
        if model.backbone.embedding_projection is not None:
            layers.append(("embedding_projection", model.backbone.embedding_projection))

        for h, l in layers:
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
        self.check_linear_layers(model)

    def test_from_clf_checkpoint(self) -> None:
        print("Testing model from CLF checkpoint...")
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task="clf",
            model_name_or_path=self.tmp_clf_dir,
            config=config,
            **self.clf_kwds,
        )
        self.check_linear_layers(model)

    def test_from_clm_checkpoint(self) -> None:
        print("Testing model from CLM checkpoint...")
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task="clf",
            model_name_or_path=self.tmp_clm_dir,
            config=config,
            **self.clf_kwds,
        )
        self.check_linear_layers(model)


if __name__ == "__main__":
    unittest.main()
