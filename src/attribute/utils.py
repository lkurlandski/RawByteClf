"""
Tools for performing input attribution with captum.
"""

from __future__ import annotations
from functools import wraps
from pathlib import Path
from typing import Optional
import warnings

import torch
from torch import Tensor


def ignore_warnings_decorator(*filter_args, **filter_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*filter_args, **filter_kwargs)
                return func(*args, **kwargs)

        return wrapper

    return decorator


class SegmentedTensor:
    """
    Run–length encoded 1‑D tensor.
    Stores only `values` and `lengths`, giving O(#segments) memory.

    Parameters
    ----------
    values  : 1‑D Tensor   constant value for each segment
    lengths : 1‑D torch.LongTensor  length of each segment
    """

    __slots__ = ("values", "lengths", "device", "dtype")

    def __init__(self, values: Tensor, lengths: Tensor):
        if values.ndim != 1 or lengths.ndim != 1:
            raise ValueError("`values` and `lengths` must be 1‑D tensors.")
        if values.numel() != lengths.numel():
            raise ValueError("`values` and `lengths` must have the same length.")
        self.values = values
        self.lengths = lengths.long()
        self.device = values.device
        self.dtype = values.dtype

    def __len__(self) -> int:
        return int(self.lengths.sum())

    def __repr__(self) -> str:
        return (
            f"SegmentedTensor(num_elems={len(self)}, "
            f"num_segments={self.lengths.numel()}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    @classmethod
    def from_tensor(cls, t: Tensor) -> SegmentedTensor:
        """Compress a dense 1‑D tensor into segment form."""
        if t.ndim != 1:
            raise ValueError("Only 1‑D tensors are supported.")
        if t.numel() == 0:
            empty = torch.empty(0, dtype=t.dtype, device=t.device)
            return cls(empty, empty.long())
        diff_idx = torch.nonzero(t[1:] != t[:-1], as_tuple=False).flatten() + 1
        boundaries = torch.cat([t.new_tensor([0]), diff_idx, t.new_tensor([t.numel()])])
        lengths = (boundaries[1:] - boundaries[:-1]).long()
        values = t[boundaries[:-1]]
        return cls(values, lengths)

    def to_tensor(self) -> Tensor:
        return torch.repeat_interleave(self.values, self.lengths)

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
            if step != 1:                 # exotic slicing → dense fallback
                return self.to_tensor()[idx]
            return self._slice(start, stop)

        raise TypeError("Index must be int or slice.")

    # ---------- slicing helpers ---------------------------------------------
    def _slice(self, start: int, stop: int) -> SegmentedTensor:
        """Return a slice [start:stop] as a new SegmentedTensor."""
        if start == 0 and stop == len(self):
            return self                         # full slice
        cum = torch.cumsum(self.lengths, 0)

        # left boundary
        left_seg = torch.searchsorted(cum, torch.tensor(start, device=self.device))
        left_offset = start - (cum[left_seg - 1] if left_seg > 0 else 0)

        # right boundary (exclusive)
        right_seg = torch.searchsorted(cum, torch.tensor(stop - 1, device=self.device))
        right_offset = cum[right_seg] - stop

        vals = self.values[left_seg:right_seg + 1].clone()
        lens = self.lengths[left_seg:right_seg + 1].clone()
        lens[0] -= left_offset
        lens[-1] -= right_offset
        # prune zero‑length ends that can appear for exact boundaries
        if lens[0] == 0:
            vals, lens = vals[1:], lens[1:]
        if lens.numel() and lens[-1] == 0:
            vals, lens = vals[:-1], lens[:-1]
        return SegmentedTensor(vals, lens)

    # ---------- IO -----------------------------------------------------------
    def save(self, path_or_file: str | Path, **torch_save_kwargs):
        """
        Persist to disk using `torch.save`.

        Parameters
        ----------
        path_or_file : str | Path | file‑like
            File name or open file object.
        torch_save_kwargs : any
            Extra kwargs forwarded to `torch.save` (e.g., `_use_new_zipfile_serialization`).
        """
        state = {"values": self.values, "lengths": self.lengths}
        torch.save(state, path_or_file, **torch_save_kwargs)

    @classmethod
    def load(
        cls,
        path_or_file: str | Path,
        map_location: Optional[str | torch.device] = None,
        **torch_load_kwargs
    ) -> "SegmentedTensor":
        """
        Load a SegmentedTensor previously saved with `.save`.

        Parameters
        ----------
        path_or_file : str | Path | file‑like
        map_location : torch.device | str | None
            Passed through to `torch.load` to relocate tensors.
        torch_load_kwargs : any
            Extra kwargs forwarded to `torch.load`.
        """
        state = torch.load(path_or_file, map_location=map_location, **torch_load_kwargs)
        return cls(state["values"], state["lengths"])
