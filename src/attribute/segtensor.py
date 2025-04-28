"""
An efficient tensor representation for storing tensors with repeated values.

WARNING: Any modification to this script (or its location) may break the ability
to load the SegmentedTensor class from a saved file using torch.load()
"""

from __future__ import annotations
from itertools import repeat
import multiprocessing as mp
import os
from pathlib import Path
from typing import Literal
import warnings

import torch
from torch import Tensor
from tqdm import tqdm


class SegmentedTensor:
    """
    Encode a Tensor as a list of segments, ideal for compressing tensors with repeated values.

    I/O is a little clumsy. You can use torch.save/load to save/load the SegmentedTensor,
    but doing so requires the SegmentedTensor class to be defined in the same module as the code
    that saves/loads the tensor. The SegmentedTensor.save/load methods are more robust, as
    they will save/load the SegmentedTensor in an implementation-agnostic way, but are harder to use.
    """

    def __init__(self, values: Tensor, lengths: Tensor):
        if values.ndim != 1 or lengths.ndim != 1:
            raise ValueError("`values` and `lengths` must be 1D tensors.")
        if values.numel() != lengths.numel():
            raise ValueError("`values` and `lengths` must have the same length.")
        self.values = values
        self.lengths = lengths.to(torch.int64)
        self.device = values.device
        self.dtype = values.dtype

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("index out of range")
            cum = torch.cumsum(self.lengths, 0)
            seg = torch.searchsorted(cum, torch.tensor(idx, device=self.device))
            return self.values[seg]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                return self.to_dense()[idx]
            return self._slice(start, stop)

        raise TypeError("Index must be int or slice.")

    def __len__(self) -> int:
        return int(self.lengths.sum())

    def __repr__(self) -> str:
        return f"SegmentedTensor({repr(self.to_dense())})"

    def __str__(self) -> str:
        return f"SegmentedTensor({str(self.to_dense())})"

    @classmethod
    def from_dense(cls, t: Tensor) -> SegmentedTensor:
        if t.ndim != 1:
            raise ValueError("Only 1D tensors are supported.")
        if t.numel() == 0:
            return cls(torch.empty_like(t), torch.empty_like(t, dtype=torch.int64))
        diff_idx = torch.nonzero(t[1:] != t[:-1], as_tuple=False).flatten() + 1
        boundaries = torch.cat([torch.tensor([0]), diff_idx, torch.tensor([t.numel()])])
        lengths = (boundaries[1:] - boundaries[:-1]).long()
        values = t[boundaries[:-1]]
        return cls(values, lengths)

    def to_dense(self) -> Tensor:
        return torch.repeat_interleave(self.values, self.lengths)

    def _slice(self, start: int, stop: int) -> SegmentedTensor:
        if start == 0 and stop == len(self):
            return self
        cum = torch.cumsum(self.lengths, 0)

        left_seg = torch.searchsorted(cum, torch.tensor(start, device=self.device))
        left_offset = start - (cum[left_seg - 1] if left_seg > 0 else 0)

        right_seg = torch.searchsorted(cum, torch.tensor(stop - 1, device=self.device))
        right_offset = cum[right_seg] - stop

        vals = self.values[left_seg:right_seg + 1].clone()
        lens = self.lengths[left_seg:right_seg + 1].clone()
        lens[0] -= left_offset
        lens[-1] -= right_offset

        if lens[0] == 0:
            vals, lens = vals[1:], lens[1:]
        if lens.numel() and lens[-1] == 0:
            vals, lens = vals[:-1], lens[:-1]

        return SegmentedTensor(vals, lens)

    def save(self, path_or_file: str | Path, **kwds) -> None:
        state = {"values": self.values, "lengths": self.lengths}
        torch.save(state, path_or_file, **kwds)

    @classmethod
    def load(cls, path_or_file: str | Path, **kwds) -> SegmentedTensor:
        state = torch.load(path_or_file, **kwds)
        return cls(state["values"], state["lengths"])


class ConvertSavedTensorsToSegmentedTensors:

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers

    def __call__(self, inputs: list[Path], outputs: list[Path]) -> list[bool]:
        if len(inputs) != len(outputs):
            raise ValueError("`inputs` and `outputs` must have the same length.")

        total    = len(inputs)
        desc     = "Converting tensors to segmented tensors"
        iterable = zip(inputs, outputs, repeat("warn", total), strict=True)

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._convert, iterable),
                    total=len(inputs),
                    desc="Converting tensors to segmented tensors"
                ))

        else:
            results  = []
            iterable = tqdm(iterable, total=total, desc=desc)
            for inp, out, err in iterable:
                ok = self.convert(inp, out, err)
                results.append(ok)

        return results

    def convert(self, input_path: Path, output_path: Path, errors: Literal["raise", "warn", "ignore"] = "raise") -> bool:
        if errors not in ["raise", "warn", "ignore"]:
            raise ValueError("Invalid value for `errors`. Must be 'raise' or 'warn'.")

        t = torch.load(input_path)

        if isinstance(t, SegmentedTensor):
            return True

        if isinstance(t, Tensor):
            if t.ndim == 1:
                s = SegmentedTensor.from_dense(t)
                torch.save(s, output_path)
                return True
            if t.ndim == 2:
                s = [SegmentedTensor.from_dense(t_i) for t_i in t]
                torch.save(s, output_path)
                return True

        if isinstance(t, list) and all(isinstance(t_i, Tensor) for t_i in t) and all(t_i.ndim == 1 for t_i in t):
            s = [SegmentedTensor.from_dense(t_i) for t_i in t]
            torch.save(s, output_path)
            return True

        message = f"Expected 1D or 2D Tensor or list of 1D Tensors ({input_path})."
        if errors == "raise":
            raise TypeError(message)
        if errors == "warn":
            warnings.warn(message)

        return False

    def _convert(self, args: tuple[Path, Path, Literal["raise", "warn", "ignore"]]) -> bool:
        input_path, output_path, errors = args
        return self.convert(input_path, output_path, errors)
