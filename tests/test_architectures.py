"""
Tests for the architectures module.
"""

import gc
import os
from pprint import pformat
import sys
import unittest

import torch
from torch import nn
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.architectures.malconv_hf import MalConvForSequenceClassification, MalConvConfig
from src.architectures.malconv2 import MalConv2ForSequenceClassification, MalConv2Config
from src.architectures.head_utils import Head, pool_logits, get_clm_loss, get_mlm_loss, get_clf_loss


class TestMalConvForSequenceClassification(unittest.TestCase):

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    max_length_pow = 16
    batch_size = 2

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_normal = MalConvConfig(
            vocab_size=256,
            embedding_size=8,
            channels=128,
            stride=256,
            kernel_size=512,
            pad_token_id=0,
        )
        self.config_weird = MalConvConfig(
            vocab_size=79,
            embedding_size=17,
            channels=97,
            stride=269,
            kernel_size=271,
            pad_token_id=0,
        )

        self.lengths = list(range(1024, 2049))
        self.lengths += [2 ** i - 1 for i in range(1024, self.max_length_pow + 1)]
        self.lengths += [2 ** i for i in range(1024, self.max_length_pow + 1)]
        self.lengths += [2 ** i + 1 for i in range(1024, self.max_length_pow + 1)]
        self.lengths = sorted(self.lengths)

        if not torch.cuda.is_available():
            self.lengths = self.lengths[0:10] + self.lengths[-10:None]

        print(f"Testing {len(self.lengths)=} from {min(self.lengths)=} to {max(self.lengths)=}.")

    def _get_input_tensor(self, vocab_size: int, length: int) -> torch.Tensor:
        bos = torch.tensor([self.bos_token_id], dtype=torch.long).repeat(self.batch_size, 1)
        eos = torch.tensor([self.eos_token_id], dtype=torch.long).repeat(self.batch_size, 1)
        inputs = torch.randint(3, vocab_size, (self.batch_size, length - 2))
        x = torch.cat([bos, inputs, eos], dim=1)
        return x

    def _test_input_length(self, model: MalConvForSequenceClassification, length: int):
        x = self._get_input_tensor(model.config.vocab_size, length).to(self.device)
        model(x)
        torch.cuda.empty_cache()
        gc.collect()

    def _test_model(self, config: MalConvConfig):
        model = MalConvForSequenceClassification(config).to(self.device)
        errors = {}
        pbar = tqdm(self.lengths)
        for l in pbar:
            pbar.set_description(f"Testing input size: {l=}")
            try:
                self._test_input_length(model, l)
            except Exception as e:
                errors[l] = e
                pbar.set_description(f"Testing input size: {l=} -- FAILED")
            else:
                pbar.set_description(f"Testing input size: {l=} -- PASSED")

        msg = f"Failed {len(errors)} times for inputs sized:\n{pformat(errors)}"
        self.assertEqual(errors, {}, msg)

    def test_normal(self):
        self._test_model(self.config_normal)

    def test_wierd(self):
        self._test_model(self.config_weird)


class TestMalConv2ForSequenceClassification(unittest.TestCase):

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    max_length_pow = 16
    batch_size = 2

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_normal_base = MalConv2Config(
            mode="base",
            vocab_size=256,
            embedding_size=8,
            channels=128,
            stride=256,
            kernel_size=512,
            pad_token_id=0,
        )
        self.config_weird_base = MalConv2Config(
            mode="base",
            vocab_size=79,
            embedding_size=17,
            channels=97,
            stride=269,
            kernel_size=271,
            pad_token_id=0,
        )
        self.config_normal_gcg = MalConv2Config(
            mode="gcg",
            vocab_size=256,
            embedding_size=8,
            channels=128,
            stride=256,
            kernel_size=512,
            pad_token_id=0,
        )
        self.config_weird_gcg = MalConv2Config(
            mode="gcg",
            vocab_size=79,
            embedding_size=17,
            channels=97,
            stride=269,
            kernel_size=271,
            pad_token_id=0,
        )

        self.lengths = list(range(1024, 2049))
        self.lengths += [2 ** i - 1 for i in range(1024, self.max_length_pow + 1)]
        self.lengths += [2 ** i for i in range(1024, self.max_length_pow + 1)]
        self.lengths += [2 ** i + 1 for i in range(1024, self.max_length_pow + 1)]
        self.lengths = sorted(self.lengths)

        if not torch.cuda.is_available():
            self.lengths = self.lengths[0:10] + self.lengths[-10:None]

        print(f"Testing {len(self.lengths)=} from {min(self.lengths)=} to {max(self.lengths)=}.")

    def _get_input_tensor(self, vocab_size: int, length: int) -> torch.Tensor:
        bos = torch.tensor([self.bos_token_id], dtype=torch.long).repeat(self.batch_size, 1)
        eos = torch.tensor([self.eos_token_id], dtype=torch.long).repeat(self.batch_size, 1)
        inputs = torch.randint(3, vocab_size, (self.batch_size, length - 2))
        x = torch.cat([bos, inputs, eos], dim=1)
        return x

    def _test_input_length(self, model: MalConv2ForSequenceClassification, length: int):
        x = self._get_input_tensor(model.config.vocab_size, length).to(self.device)
        model(x)
        torch.cuda.empty_cache()
        gc.collect()

    def _test_model(self, config: MalConv2Config):
        model = MalConv2ForSequenceClassification(config).to(self.device)
        errors = {}
        pbar = tqdm(self.lengths)
        for l in pbar:
            pbar.set_description(f"Testing input size: {l=}")
            try:
                self._test_input_length(model, l)
            except Exception as e:
                errors[l] = e
                pbar.set_description(f"Testing input size: {l=} -- FAILED")
            else:
                pbar.set_description(f"Testing input size: {l=} -- PASSED")

        msg = f"Failed {len(errors)} times for inputs sized:\n{pformat(errors)}"
        self.assertEqual(errors, {}, msg)

    def test_normal_base(self):
        self._test_model(self.config_normal_base)

    def test_wierd_base(self):
        self._test_model(self.config_weird_base)

    def test_normal_gcg(self):
        self._test_model(self.config_normal_gcg)

    def test_weird_gcg(self):
        self._test_model(self.config_weird_gcg)


class TestHeadUtils(unittest.TestCase):

    B = 4       # Batch size
    H = 64      # Hidden size
    T = 1024    # Sequence length
    C = 10      # Number of classes
    L = 2       # Number of hidden layers
    V = 256     # Vocabulary size
    P = 0       # Pad token

    def test_head_simple(self):
        model = Head(in_size=self.H, out_size=self.C)
        input_tensor = torch.randn(self.B, self.T, self.H)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.B, self.T, self.C))

    def test_head_hidden(self):
        model = Head(in_size=self.H, out_size=self.C, hidden_size=self.H, num_hidden_layers=self.L)
        input_tensor = torch.randn(self.B, self.T, self.H)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.B, self.T, self.C))

    def test_pool_logits_none(self):
        logits = torch.randn(self.B, self.T, self.C)
        input_ids = torch.randint(0, self.V, (self.B, self.T))
        output = pool_logits("none", logits, input_ids, self.P)
        self.assertEqual(output.shape, (self.B, self.T, self.C))
        self.assertEqual(torch.equal(output, logits), True)

    def test_pool_logits_mean(self):
        logits = torch.randn(self.B, self.T, self.C)
        input_ids = torch.randint(0, self.V, (self.B, self.T))
        output = pool_logits("mean", logits, input_ids, self.P)
        self.assertEqual(output.shape, (self.B, self.C))

    def test_pool_logits_last(self):
        logits = torch.randn(self.B, self.T, self.C)
        input_ids = torch.randint(0, self.V, (self.B, self.T))
        output = pool_logits("last", logits, input_ids, self.P)
        self.assertEqual(output.shape, (self.B, self.C))

    def test_get_clm_loss(self):
        logits = torch.randn(self.B, self.T, self.V)
        labels = torch.randint(0, self.V, (self.B, self.T))
        loss = get_clm_loss(logits, labels, self.V)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_get_mlm_loss(self):
        logits = torch.randn(self.B, self.T, self.V)
        labels = torch.randint(0, self.V, (self.B, self.T))
        loss = get_mlm_loss(logits, labels, self.V)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_get_clf_loss_regression(self):
        logits = torch.randn(self.B)
        labels = torch.randn(self.B)
        loss = get_clf_loss(logits, labels, num_labels=1, problem_type="regression")
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_get_clf_loss_single_label_classification(self):
        logits = torch.randn(self.B, self.C)
        labels = torch.randint(0, self.C, (self.B,))
        loss = get_clf_loss(logits, labels, num_labels=self.C, problem_type="single_label_classification")
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_get_clf_loss_multi_label_classification(self):
        logits = torch.randn(self.B, self.C)
        labels = torch.randint(0, 2, (self.B, self.C), dtype=torch.float32)
        loss = get_clf_loss(logits, labels, num_labels=self.C, problem_type="multi_label_classification")
        self.assertTrue(isinstance(loss, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
