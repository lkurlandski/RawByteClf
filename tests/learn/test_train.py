"""
Test.
"""

import math
import os
import shutil
import unittest

import torch
from torch import Tensor
from transformers import PreTrainedModel

from src.enums import Task
from src.utils import get_array_shape, get_array_datatype, get_array_dim
from src.architectures.head_utils import check_for_anomalous_weights
from src.architectures.mamba_hf import MambaConfig, MambaForCausalLM, MambaForSequenceClassification
from src.data.loaders_core import get_materials_clf_sorel, get_materials_clf_bodmas
from src.tokenization.api import get_fast_tokenizer
from src.learn.train import get_model, get_processed_dataset_hf


ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"


class TestGetModel(unittest.TestCase):
    """
    Mamba weights have been observed to be anomalous in the past.
    This test checks that the weights are within some normal bounds.
    """

    vocab_size = 256
    hidden_size = 256
    num_hidden_layers = 8
    num_labels = 999

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
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            is_decoder=True,
            **kwds,
        )

    def test_from_config(self) -> None:
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task=Task.DET,
            model_name_or_path=None,
            config=config,
            **self.clf_kwds,
        )
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_from_config: {model.__class__.__name__}\n{str(model)}")
        check_for_anomalous_weights(model.head_clf, errors="raise", mean_tolerance=0.001, std_tolerance=2 * config.initializer_range)

    def test_from_clf_checkpoint(self) -> None:
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task=Task.DET,
            model_name_or_path=self.tmp_clf_dir,
            config=config,
            **self.clf_kwds,
        )
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_from_clf_checkpoint: {model.__class__.__name__}\n{str(model)}")
        check_for_anomalous_weights(model.head_clf, errors="raise", mean_tolerance=0.001, std_tolerance=2 * config.initializer_range)

    def test_from_clm_checkpoint(self) -> None:
        config = self.get_config(**self.clf_kwds)
        model = get_model(
            task=Task.DET,
            model_name_or_path=self.tmp_clm_dir,
            config=config,
            **self.clf_kwds,
        )
        if ENABLE_UNITTEST_LOGGING:
            print(f"test_from_clm_checkpoint: {model.__class__.__name__}\n{str(model)}")
        check_for_anomalous_weights(model.head_clf, errors="raise", mean_tolerance=0.001, std_tolerance=2 * config.initializer_range)


if __name__ == "__main__":
    unittest.main()
