"""
Tests for the architectures module.
"""

import gc
import os
from pprint import pformat
import sys
import unittest

import torch
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.architectures.malconv_hf import MalConvForSequenceClassification, MalConvConfig


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
            pad_token_id=0,
            channels=128,
            stride=256,
            kernel_size=512,
        )
        self.config_weird = MalConvConfig(
            vocab_size=79,
            embedding_size=17,
            pad_token_id=0,
            channels=97,
            stride=269,
            kernel_size=271,
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


if __name__ == "__main__":
    unittest.main()
