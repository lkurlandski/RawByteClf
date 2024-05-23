"""
Naughty stuff happpening here.
"""

from collections.abc import Iterable
import os
import sys
from typing import Dict, List, Optional, Tuple
import unittest
import warnings

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import Tensor, inf
from torch.nn.utils import clip_grad_norm_
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from src.utils import (
    detect_anomalous_gradients,
    detect_anomalous_parameters,
    compute_gradient_norm,
)


def my_clip_grad_norm_(
        parameters: Tensor | Iterable[Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        resist_overflow: bool = True,
    ) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters even if it is infinite.
    """
    warnings.warn("Using patched version of torch.nn.utils.clip_grad_norm_")

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]

    # <\CHANGED>
    DTYPE = grads[0].dtype
    # <CHANGED/>

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

    if norm_type == inf:
        norms = [g.detach().abs().max().to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.norm(g, norm_type) for g in grads])

        # <\CHANGED>
        if resist_overflow:
            total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type, dtype=torch.float64)
        else:
            total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
        # <CHANGED/>

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')

    # <\CHANGED>
    if resist_overflow and total_norm > torch.finfo(torch.float32).max:
        clip_coef = max_norm / torch.tensor(torch.finfo(torch.float32).max, dtype=DTYPE, device=total_norm.device)
    else:
        clip_coef = max_norm / (total_norm + 1e-6)
    # <CHANGED/>

    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), [grads]) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))  # type: ignore[call-overload]
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)

    return total_norm


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestMyClipGradNorm(unittest.TestCase):

    def setUp(self):
        self.max_norm = 10.0
        self.model = SimpleNet()
        self.inputs = torch.randn(2, 10)
        self.targets = torch.randint(0, 2, (2,))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def prepare(self):
        self.optimizer.zero_grad()
        outputs: Tensor = self.model(self.inputs)
        loss: Tensor = self.criterion(outputs, self.targets)
        loss.backward()
        assert not any(detect_anomalous_gradients(self.model)), "Anamalous gradients detected."
        assert not any(detect_anomalous_parameters(self.model)), "Anamalous parameters detected."

    def test_normal_gradients(self):
        self.prepare()

        original_norm = compute_gradient_norm(self.model)
        self.assertGreater(original_norm, 0, f"{original_norm=}")
        self.assertLess(original_norm, float("inf"), f"{original_norm=}")

        total_norm = my_clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.assertGreater(total_norm, 0, f"{total_norm=}")
        self.assertLess(total_norm, float("inf"), f"{total_norm=}")

        clipped_norm = compute_gradient_norm(self.model)
        self.assertGreater(clipped_norm, 0, f"{clipped_norm=}")
        self.assertLessEqual(clipped_norm, self.max_norm, f"{clipped_norm=}")

    def test_infinite_gradients(self):
        self.prepare()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float('inf'))

        original_norm = compute_gradient_norm(self.model)
        self.assertEqual(original_norm, float("inf"), f"{original_norm=}")

        total_norm = my_clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False, resist_overflow=False)
        self.assertEqual(total_norm, float("inf"), f"{total_norm=}")

        clipped_norm = compute_gradient_norm(self.model)
        self.assertEqual(torch.isnan(torch.tensor(clipped_norm)).item(), True, f"{clipped_norm=}")

    def test_nan_gradients(self):
        self.prepare()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float('nan'))

        original_norm = compute_gradient_norm(self.model)
        self.assertEqual(torch.isnan(torch.tensor(original_norm)).item(), True, f"{original_norm=}")

        total_norm = my_clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False, resist_overflow=False)
        self.assertEqual(torch.isnan(torch.tensor(total_norm)).item(), True, f"{total_norm=}")

        clipped_norm = compute_gradient_norm(self.model)
        self.assertEqual(torch.isnan(torch.tensor(clipped_norm)).item(), True, f"{clipped_norm=}")

    def test_overflowing_gradients(self):
        self.prepare()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(1e20)

        original_norm = compute_gradient_norm(self.model)
        self.assertEqual(original_norm, float("inf"), f"{original_norm=}")

        total_norm = my_clip_grad_norm_(self.model.parameters(), self.max_norm, resist_overflow=True)
        self.assertEqual(total_norm, float("inf"), f"{total_norm=}")

        clipped_norm = compute_gradient_norm(self.model)
        self.assertGreater(clipped_norm, 0, f"{clipped_norm=}")
        self.assertLessEqual(clipped_norm, self.max_norm, f"{clipped_norm=}")


def main():
    torch.nn.utils.clip_grad_norm_ = my_clip_grad_norm_


if __name__ == "__main__":
    unittest.main()
