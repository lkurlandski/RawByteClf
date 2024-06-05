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

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length_pow = 16
        self.config = MalConvConfig(
            model_max_length=2**self.max_length_pow,
            vocab_size=256,
            embedding_size=8,
            pad_token_id=0,
            window_size=512,
            channels=128,
            stride=512,
        )
        self.model = MalConvForSequenceClassification(self.config).to(self.device)
        self.bos = torch.tensor([self.bos_token_id])
        self.eos = torch.tensor([self.eos_token_id])

    def _test_input_length(self, length: int):
        x = torch.randint(3, self.config.vocab_size, (length - 2,))
        x = torch.cat([self.bos, x, self.eos], dim=0)
        x = x.unsqueeze(0).to(self.device)
        self.model(x)
        torch.cuda.empty_cache()
        gc.collect()

    def test_input_length_too_short(self):

        for l in range(3, self.config.window_size):
            with self.assertRaises(ValueError, msg=f"Failed for input size: {l}"):
                self._test_input_length(l)

    def test_input_length_just_right(self):

        lengths = [self.config.window_size + i for i in range(self.config.window_size)]
        lengths += [2 ** i - 1 for i in range(2, self.max_length_pow + 2)]
        lengths += [2 ** i for i in range(2, self.max_length_pow + 2)]
        lengths += [2 ** i + 1 for i in range(2, self.max_length_pow + 2)]
        lengths = [l for l in lengths if l >= self.config.window_size]
        lengths = sorted(lengths)

        print(f"Testing {len(lengths)=} from {min(lengths)=} to {max(lengths)=}.")

        errors = {}
        pbar = tqdm(lengths)
        for l in pbar:
            pbar.set_description(f"Testing input size: {l=}")
            try:
                self._test_input_length(l)
            except Exception as e:
                errors[l] = e
                pbar.set_description(f"Testing input size: {l=} -- FAILED")
            else:
                pbar.set_description(f"Testing input size: {l=} -- PASSED")

        msg = f"Failed for inputs sized: {pformat(errors)}"
        self.assertEqual(errors, {}, msg)


if __name__ == "__main__":
    unittest.main()
