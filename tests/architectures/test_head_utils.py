"""
Test.
"""

import os
import sys
import unittest

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

import torch

from src.architectures.head_utils import Head, pool_logits, get_clm_loss, get_mlm_loss, get_clf_loss


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
