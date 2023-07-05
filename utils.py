"""
"""

from collections import namedtuple
from pathlib import Path
import psutil
import os

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


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
