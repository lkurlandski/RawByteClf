"""
Useful functions for the project.
"""

import asyncio
import bz2
from collections.abc import Collection, Iterable
from concurrent.futures import ThreadPoolExecutor
import gzip
import inspect
from io import BytesIO
from itertools import islice
import json
import lzma
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Literal
import zlib

import numpy as np
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import py7zr
import torch
from torch import nn, ByteTensor, LongTensor, Tensor


def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def process_mem(fmt: str = "G") -> str:
    if fmt == "B":
        d = 1
    elif fmt == "M":
        d = 2
    elif fmt == "G":
        d = 3
    else:
        raise ValueError()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024**d
    return f"{round(m, 2)}{fmt}"


def gig(b: int) -> str:
    return f"{round(b / (1024 ** 3), 2)}G"


def mem() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def output_root(vocab_size: int, n_sorel: int, n_windows: int) -> Path:
    return Path(f"{vocab_size}/{n_windows}/{n_sorel}")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


def get_paths_sorted_numerically(
    path: Collection[Path] | Path,
    lstrip: str = "",
    rstrip: str = "",
    reverse: bool = False,
) -> list[Path]:
    def key(p: Path) -> int:
        return int(p.stem.lstrip(lstrip).rstrip(rstrip))

    files = Path(path).iterdir() if isinstance(path, (Path, str)) else path
    return list(sorted(files, key=key, reverse=reverse))


def get_highest_path(
    path: Collection[Path] | Path,
    lstrip: str = "",
    rstrip: str = "",
    lowest: bool = False,
) -> Path:
    """
    Get the highest/lowest numerically indexed path from a directory or a collection of paths.

    Note that lstrip and rstrip are applied to the stem of the path and that they do not align
    with the typical API for str.lstrip and str.rstrip.
    """

    def key(p: Path) -> int:
        s = p.stem
        if s.startswith(lstrip):
            s = s[len(lstrip):]
        if s.endswith(rstrip):
            s = s[:-len(rstrip)]
        return int(s)

    files = list(Path(path).iterdir()) if isinstance(path, (Path, str)) else path
    idx = 0 if lowest else -1
    return list(sorted(files, key=key))[idx]


def is_dataset_path(path: Path) -> bool:
    REQUIRED = ("dataset_info.json", "state.json")
    ALLOWED = (".arrow",)

    paths = [p.name for p in path.iterdir()]
    if not all(p in paths for p in REQUIRED):
        return False

    for p in paths:
        if p in REQUIRED:
            continue
        if Path(p).suffix not in ALLOWED:
            return False
    return True


def is_dataset_path_completed(path: Path) -> bool:
    tr_path = path / "tr"
    vl_path = path / "vl"
    ts_path = path / "ts"
    return all(p.exists() for p in (tr_path, vl_path, ts_path)) and all(
        is_dataset_path(p) for p in (tr_path, vl_path, ts_path)
    )


def get_scale_fn(scale: float) -> Callable[[int], float]:
    return lambda x: int(round(x * scale))


def object_from_superset_of_constructor_kwds(cls, **kwds) -> Any:
    kwds = {k: v for k, v in kwds.items() if k in inspect.getfullargspec(cls.__init__).args}
    return cls(**kwds)


def bash_file_to_vscode_debug_str(file: Path) -> str:
    args = []
    add = False
    with open(file, "r") as fp:
        for line in fp:
            if add:
                args.append(line)
            if line.startswith("python"):
                add = True

    args = [a for a in args if not a.startswith("#")]
    args = [a.replace('"', "").replace("'", "").replace("\\", "").rstrip() for a in args]
    args = [f'"{a}"' for a in args]
    return ", ".join(args)


def log_tensor(path: str | Path, x: Tensor, name: str) -> None:
    """
    Log statistics of a tensor to a CSV file for debugging.

    Args:
        path: The path to the directory where the CSV file is stored.
        x: The tensor to log.
        name: The stem of the csv file.

    If the CSV file does not exist, it will be created. If it does exist, it will be appended to.
        The csv file will contain the following fields:
            pos_min: The minimum value of the positive elements of the tensor.
            pos_max: The maximum value of the positive elements of the tensor.
            pos_mean: The mean value of the positive elements of the tensor.
            pos_stdev: The standard deviation of the positive elements of the tensor.
            neg_min: The minimum value of the negative elements of the tensor.
            neg_max: The maximum value of the negative elements of the tensor.
            neg_mean: The mean value of the negative elements of the tensor.
            neg_stdev: The standard deviation of the negative elements of the tensor.
            dtype: The dtype of the tensor.
            shape: The shape of the tensor.
        If the tensor is all zero, each field will be 0.
        If the tensor contains NaN, each field will be NaN.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    p = path / f"{name}.csv"

    if not p.exists():
        with open(p, "w") as fp:
            fp.write("pos_min,pos_max,pos_mean,pos_stdev,neg_min,neg_max,neg_mean,neg_stdev,dtype,shape\n")

    dtype = str(x.dtype)
    shape = "_".join(str(s) for s in x.shape)

    if torch.any(torch.isnan(x)):
        with open(p, "a") as fp:
            fp.write(f"NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,{dtype},{shape}\n")
        return

    if torch.all(x == 0):
        with open(p, "a") as fp:
            fp.write(f"0,0,0,0,0,0,0,0,{dtype},{shape}\n")
        return

    pos: Tensor = x[(x > 0) & (x != 0)]
    with open(p, "a") as fp:
        if pos.numel() == 0:
            fp.write(",,,,")
        else:
            fp.write(
                f"{pos.min().item()},"
                f"{pos.max().item()},"
                f"{pos.mean().item()},"
                f"{pos.std().item()},"
            )

    neg: Tensor = x[(x < 0) & (x != 0)]
    with open(p, "a") as fp:
        if neg.numel() == 0:
            fp.write(",,,,")
        else:
            fp.write(
                f"{neg.min().item()},"
                f"{neg.max().item()},"
                f"{neg.mean().item()},"
                f"{neg.std().item()},"
            )

    with open(p, "a") as fp:
        fp.write(f"{dtype},{shape}\n")


def stable_softmax(x: Tensor, dim: int = 0):
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    exp_scores = torch.exp(x - max_values)
    sum_exp_scores = torch.sum(exp_scores, dim=dim, keepdim=True)
    softmax_result = exp_scores / sum_exp_scores
    return softmax_result


async def _process_files_asynch(files: list[Path], fn: Callable[[Path], Any], *args) -> list[Any]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        tasks = [loop.run_in_executor(pool, fn, file, *args) for file in files]
    return await asyncio.gather(*tasks)


def process_files_asynch(files: list[Path], fn: Callable[[Path], Any], *args) -> list[Any]:
    return asyncio.run(_process_files_asynch(files, fn, *args))


def test_process_files_async():

    # For 10000 BODMAS binaries & LENGTH=None
    # ASYNC: 2.79, 2.69, 2.57
    # NO ASYNC: 5.94, 5.49, 5.49

    # For 10000 BODMAS binaries & LENGTH=1024
    # ASYNC: 1.20, 1.16, 1.16
    # NO ASYNC: 0.14, 0.13, 0.14,

    # For 15-30GB cache files
        #    29.1 GiB [##############]  cache-6c21e812d4159e7f.arrow
        #    29.1 GiB [############# ]  cache-cba03356c8bf06ef.arrow
        #    29.1 GiB [############# ]  cache-218c715b5470901d.arrow
        #    26.6 GiB [############  ]  cache-ebafa8e80277b572.arrow
        #    23.2 GiB [###########   ]  cache-c0a754467bb8bb59.arrow
        #    23.1 GiB [###########   ]  cache-55c2f24a66955904.arrow
        #    23.0 GiB [###########   ]  cache-1f6cdf7dabe09db3.arrow
        #    15.6 GiB [#######       ]  cache-26956f6b1ff4c778.arrow

    ASYNC = True
    LENGTH = None
    BODMAS = True

    def fn(f: Path, s: int):
        return f.open("rb").read(s)

    if BODMAS:
        files = list(Path("/home/lk3591/Documents/datasets/BODMAS/binaries").iterdir())[0:10000]
    else:
        files = [
            Path("./input/bodmas_pe") / p for p in [
                "cache-6c21e812d4159e7f.arrow",
                "cache-cba03356c8bf06ef.arrow",
                "cache-218c715b5470901d.arrow",
                "cache-ebafa8e80277b572.arrow",
                # "cache-c0a754467bb8bb59.arrow",
                # "cache-55c2f24a66955904.arrow",
                # "cache-1f6cdf7dabe09db3.arrow",
                # "cache-26956f6b1ff4c778.arrow",
            ]
        ]

    t = time.time()

    if ASYNC:
        _ = process_files_asynch(files, fn, LENGTH)
    else:
        _ = [fn(f, LENGTH) for f in files]

    print(time.time() - t)


def to_long_tensor(x: bytes | list | np.ndarray | Tensor) -> LongTensor:
    if isinstance(x, bytes):
        return torch.frombuffer(x, dtype=torch.uint8).to(torch.long)
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.int64)).to(torch.long)
    if isinstance(x, Tensor):
        return x.to(torch.long)
    raise TypeError(f"Unexpected type: {type(x)=}")


def compose_functions(*funcs):
    def inner(arg):
        result = arg
        for func in funcs:
            result = func(result)
        return result
    return inner


def get_max_keys_from_dict(d: dict[str, int]) -> tuple[str]:
    keys = []
    val = -1
    for k, v in d.items():
        if v >= val:
            if v > val:
                keys = []
            keys.append(k)
            val = v

    return tuple(keys)


def is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


COMPRESSION_TYPES = ("gzip", "bzip2", "lzma", "zlib", "7z")


def compress(
    bs: bytes,
    compression_type: Literal["gzip", "bzip2", "lzma", "zlib", "7z"],
    compression_level: int = 9,
    **kwds,
) -> bytes:
    if compression_type == "gzip":
        if kwds.get("compresslevel", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `compresslevel` and `compression_level`.")
        kwds["compresslevel"] = compression_level
        return gzip.compress(bs, **kwds)

    if compression_type == "bzip2":
        if kwds.get("compresslevel", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `compresslevel` and `compression_level`.")
        kwds["compresslevel"] = compression_level
        return bz2.compress(bs, **kwds)

    if compression_type == "lzma":
        if kwds.get("preset", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `preset` and `compression_level`.")
        kwds["preset"] = compression_level
        return lzma.compress(bs, **kwds)

    if compression_type == "zlib":
        if kwds.get("level", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `level` and `compression_level`.")
        kwds["level"] = compression_level
        return zlib.compress(bs, **kwds)

    if compression_type == "7z":
        fp = BytesIO()
        with py7zr.SevenZipFile(fp, "w", **kwds) as archive:
            archive.writef(BytesIO(bs), "tmp")
        fp.seek(0)
        return fp.read()

    raise RuntimeError(f"Unknown compression type: {compression_type}")


if __name__ == "__main__":
    print(bash_file_to_vscode_debug_str(Path(sys.argv[1])))
