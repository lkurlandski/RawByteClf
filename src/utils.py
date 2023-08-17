"""
Useful functions for the project.
"""

from collections.abc import Collection
from pathlib import Path
import psutil
import os

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch import nn


def process_mem(fmt: "G") -> str:
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


def get_highest_path(
    path_or_files: Collection[Path] | Path, lstrip: str = "", rstrip: str = ""
) -> Path:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip(rstrip))))[-1]


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
