"""
Useful functions for the project.
"""

import asyncio
import bz2
from collections.abc import Collection, Iterable
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fnmatch
import gzip
import inspect
from io import BytesIO
from itertools import islice
import json
import lzma
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Generator, Literal, Optional
import warnings
import zlib

from Crypto.Cipher import AES
import numpy as np
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import py7zr
import torch
from torch import nn, ByteTensor, LongTensor, Tensor

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: disable=wrong-import-position

from src.enums import CompressionAlgorithm, EncryptionAlgorithm


def check_model_parameters(model: nn.Module, min_: float = -float("inf"), max_: float = float("inf")) -> list[tuple[str, tuple[Literal["nan", "inf", ">", "<"]]]]:
    # For example, check_model_parameters(model, -torch.finfo(torch.float32).max, torch.finfo(torch.float32).max)
    all_issues = []
    for name, param in model.named_parameters():
        param_data = param.data
        issues = []
        if torch.isnan(param_data).any():
            issues.append("NaN")
        if torch.isinf(param_data).any():
            issues.append("Inf")
        if (param_data > max_).any():
            issues.append(f"> {param_data.max().item()}")
        if (param_data < min_).any():
            issues.append(f"< {param_data.min().item()}")
        if issues:
            all_issues.append((name, tuple(issues)))
    return all_issues


@contextlib.contextmanager
def print_context(suppress: bool = False):
    if not suppress:
        yield
    else:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


def rglob(top: str, pattern: str, followlinks: bool = True) -> Generator[str, None, None]:
    for root, dirs, files in os.walk(top, followlinks=followlinks):  # pylint: disable=unused-variable
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                yield os.path.join(root, file)


def unique_value(iterable):
    values = set(iterable)
    if len(values) != 1:
        raise ValueError("The iterable does not contain a unique value")
    return values.pop()


def getattr_recursively(obj: Any, attr: str) -> Any:
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def get_array_datatype(x: Tensor | np.ndarray | list| int | float) -> Literal["int", "float"]:

    def datatype_of_list(x: list) -> Literal["int", "float"]:
        if isinstance(x[0], list):
            return datatype_of_list(x[0])
        return type(x[0]).__name__

    if isinstance(x, Tensor):
        return str(x.dtype).split(".")[1][:-2]
    if isinstance(x, np.ndarray):
        return str(x.dtype)[:-2]
    if isinstance(x, list):
        return datatype_of_list(x)
    if isinstance(x, (int, float)):
        return type(x).__name__

    raise ValueError(f"Unexpected type: {type(x)=}")


def get_array_shape(x: Tensor | np.ndarray | list | int | float) -> tuple[int]:

    def shape_of_list(x: list) -> tuple[int]:
        if isinstance(x[0], list):
            return (len(x),) + shape_of_list(x[0])
        return (len(x),)

    if isinstance(x, Tensor):
        return tuple(x.shape)
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    if isinstance(x, list):
        return shape_of_list(x)
    if isinstance(x, (int, float)):
        return tuple()

    raise ValueError(f"Unexpected type: {type(x)=}")


def get_array_dim(x: Tensor | np.ndarray | list | int | float) -> tuple[int]:
    return len(get_array_shape(x))


def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def get_unique_files(files: list[os.PathLike | Path]) -> list[os.PathLike | Path]:
    shas, remove = set(), set()
    for i, f in enumerate(files):
        sha = Path(f).stem
        if sha in shas:
            remove.add(i)
        shas.add(sha)
    return [f for i, f in enumerate(files) if i not in remove]


def count_lines_big_file(file: os.PathLike) -> int:
    args = ["wc", "-l", file]
    result = subprocess.run(args, check=True, capture_output=True)
    total = int(result.stdout.split()[0])
    return total


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


def detect_anomalous_parameters(model: nn.Module) -> tuple[bool, bool]:
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    has_inf = any(torch.isinf(p).any() for p in model.parameters())
    return has_nan, has_inf


def detect_anomalous_gradients(model: nn.Module) -> tuple[bool, bool]:
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
    has_inf = any(p.grad is not None and torch.isinf(p.grad).any() for p in model.parameters())
    return has_nan, has_inf


def compute_gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0,
    dtype: torch.dtype = None,
) -> float:
    norm = 0
    for p in model.parameters():
        norm += p.grad.data.norm(norm_type, dtype=dtype) ** 2
    norm = norm ** .5
    return norm.detach().cpu().item()


def remove_empty_directories(directory: str, missing_ok: bool = False) -> None:
    if missing_ok and not os.path.exists(directory):
        return

    for root, dirs, files in os.walk(directory, topdown=False):  # pylint: disable=unused-variable
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


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
    lstrip: Optional[str] = None,
    rstrip: Optional[str] = None,
    idx: int = -1,
) -> Path:
    """
    Get the highest/lowest numerically indexed path from a directory or a collection of paths.

    Note that lstrip and rstrip are applied to the stem of the path and that they do not align
    with the typical API for str.lstrip and str.rstrip.
    """

    def key(p: Path) -> int:
        s = p.stem
        if lstrip and s.startswith(lstrip):
            s = s[len(lstrip):]
        if rstrip and s.endswith(rstrip):
            s = s[:-len(rstrip)]
        return int(s)

    files = list(Path(path).iterdir()) if isinstance(path, (Path, str)) else path
    if len(files) == 0:
        raise FileNotFoundError(f"{path=}")
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

    # Special processing for the arch_config and pretraining_checkpoint

    def encapsulate_string(s: str) -> str:
        r = ""
        r += '\\'         # add backlash
        r += '"'          # add quote
        r += s            # add key
        r += '\\'         # add backlash
        r += '"'          # add quote
        return r

    def isdigit(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    idx_1 = None
    idx_2 = None
    for i, a in enumerate(args):
        if "--arch_config" in a:
            if idx_1 is not None:
                raise RuntimeError()
            idx_1 = i
        if "--pretraining_checkpoint" in a:
            if idx_2 is not None:
                raise RuntimeError()
            idx_2 = i

    if idx_1 is not None:
        s = args[idx_1]
        s = s[len('"--arch_config={'):-len('}"')]
        iterator = [x.split(":") for x in s.split(",")]
        s = ""
        for k, v in iterator:
            k = k.strip()
            v = v.strip()
            s += encapsulate_string(k)
            s += ": "

            if v in ("true", "false", "null") or isdigit(v):
                s += v
            else:
                s += encapsulate_string(v)

            s += ", "

        s = s[:-len(", ")]
        s = '"--arch_config={' + s + '}"'
        args[idx_1] = s

    if idx_2 is not None and all(c in args[idx_2] for c in ("{", "}")):
        s = args[idx_2]
        s = s[len('"--pretraining_checkpoint={'):-len('}"')]
        iterator = [x.split(":") for x in s.split(",")]
        s = ""
        for k, v in iterator:
            k = k.strip()
            v = v.strip()
            s += encapsulate_string(k)
            s += ": "

            if v in ("true", "false", "null") or isdigit(v):
                s += v
            else:
                s += encapsulate_string(v)

            s += ", "

        s = s[:-len(", ")]
        s = '"--pretraining_checkpoint={' + s + '}"'
        args[idx_2] = s

    return ", ".join(args)


UPCAST_TENSOR = {
    torch.int8: torch.int16,
    torch.int16: torch.int32,
    torch.int32: torch.int64,

    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.float32: torch.float64,
}


def basic_tensor_stats(x: Tensor) -> tuple[float, float, float, float]:
    _mean = x.mean().cpu().item()
    _std = x.std().cpu().item()
    _min = x.min().cpu().item()
    _max = x.max().cpu().item()
    r = (_mean, _std, _min, _max)

    upcast = UPCAST_TENSOR.get(x.dtype, None)

    if upcast is not None and any(math.isnan(v) or math.isinf(v) for v in r):
        return basic_tensor_stats(x.to(upcast))

    return r


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


def get_memory_usage(obj, seen=None):
    """
    Recursively calculate the memory usage of a nested dictionary.
    
    Args:
    - obj: The dictionary or nested structure to analyze.
    - seen: A set to track objects already visited to avoid infinite recursion (optional).
    
    Returns:
    - Memory usage in bytes.
    """
    # Initialize seen set if not provided
    if seen is None:
        seen = set()

    # Check if object has already been visited to avoid infinite recursion
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Calculate memory usage of current object
    memory_usage = sys.getsizeof(obj)

    # If obj is a dictionary, recursively calculate memory usage of its values
    if isinstance(obj, dict):
        for key, value in obj.items():
            memory_usage += sys.getsizeof(key)
            memory_usage += get_memory_usage(value, seen)

    # If obj is a list or tuple, recursively calculate memory usage of its elements
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            memory_usage += get_memory_usage(item, seen)

    return memory_usage


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


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def compress(
    bs: bytes,
    compression_type: CompressionAlgorithm,
    compression_level: int = 9,
    **kwds,
) -> bytes:

    compression_type = CompressionAlgorithm(compression_type)

    if compression_type == CompressionAlgorithm.GZIP:
        if kwds.get("compresslevel", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `compresslevel` and `compression_level`.")
        kwds["compresslevel"] = compression_level
        return gzip.compress(bs, **kwds)

    if compression_type == CompressionAlgorithm.BZ2:
        if kwds.get("compresslevel", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `compresslevel` and `compression_level`.")
        kwds["compresslevel"] = compression_level
        return bz2.compress(bs, **kwds)

    if compression_type == CompressionAlgorithm.LZMA:
        if kwds.get("preset", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `preset` and `compression_level`.")
        kwds["preset"] = compression_level
        return lzma.compress(bs, **kwds)

    if compression_type == CompressionAlgorithm.ZLIB:
        if kwds.get("level", compression_level) != compression_level:
            raise RuntimeError("Cannot specify both `level` and `compression_level`.")
        kwds["level"] = compression_level
        return zlib.compress(bs, **kwds)

    if compression_type == CompressionAlgorithm.S7Z:
        fp = BytesIO()
        with py7zr.SevenZipFile(fp, "w", **kwds) as archive:
            archive.writef(BytesIO(bs), "tmp")
        fp.seek(0)
        return fp.read()

    raise ValueError(f"Unknown compression type: {compression_type}")


def encrypt(bs: bytes, encryption_type: EncryptionAlgorithm, key: Optional[bytes] = None, **kwds) -> bytes:

    encryption_type = EncryptionAlgorithm(encryption_type)

    key = np.random.randint(0, 256, 16, dtype=np.uint8).tobytes() if key is None else key

    if encryption_type == EncryptionAlgorithm.AES:
        kwds["mode"] = kwds.pop("mode", AES.MODE_CTR)
        cipher = AES.new(key, **kwds)
        return key + cipher.encrypt(bs)

    raise ValueError(f"Unknown encryption type: {encryption_type}")


if __name__ == "__main__":
    print(bash_file_to_vscode_debug_str(Path(sys.argv[1])))
