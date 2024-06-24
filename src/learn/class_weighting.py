"""
Reweighting schemes for imbalanced class distributions.
"""

from collections import Counter
from itertools import islice
import json
import math
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional

from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.cfg import DATASET_TO_FILES


def inverse_class_frequency(dist: Counter) -> dict[str, float]:
    weights: dict[str, float] = {}
    for c in dist:
        weights[c] = 1 / dist[c]
    return weights


def sample_reweighting(dist: Counter, beta: float) -> dict[str, float]:
    if beta < 0 or beta >= 1:
        raise ValueError("Beta must be in the range [0, 1)")
    weights: dict[str, float] = {}
    for c in dist:
        weights[c] = 1 / ((1 - math.pow(beta, dist[c])) / (1 - beta))
    return weights


def get_byte_distribution(num_files: Optional[int] = 50000) -> Counter:
    file = Path(f"./cache/byte_distribution--num_files={num_files}.json")
    if file.exists():
        with open(file, "r") as f:
            return Counter(json.load(f))

    dist = Counter()
    for f in tqdm(sorted(islice(DATASET_TO_FILES["binaries"]["bodmas_pe"](), num_files)), total=num_files):
        with open(f, "rb") as fp:
            dist.update(fp.read())

    with open(file, "w") as fp:
        json.dump(dist, fp, indent=4)
    return dist


if __name__ == "__main__":
    c = get_byte_distribution(50000)
    pprint(c)
