"""
Quick script to get the lengths of files based on the class their in.
"""

from collections import defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from tqdm import tqdm

from src.data.cfg import DATASET_TO_FILES, SOREL_LABEL_CACHE_DIR
from src.utils import batched


N_WORKERS = 32


cache = (SOREL_LABEL_CACHE_DIR / f"extractor--category/refiner--top/k--{1}/file_label_map").with_suffix(".json")
file_label_map = json.load(cache.open())
file_label_map = {k : v[0] for k, v in file_label_map.items() if v is not None}


files = list(f for f in DATASET_TO_FILES["binaries"]["sorel_pe"]() if f.name in file_label_map)


def get_file_lengths_by_class(_files: list[Path]) -> dict[str, list[int]]:
    _file_lengths_by_class = defaultdict(list)
    for f in _files:
        c = file_label_map[f.name]
        _file_lengths_by_class[c].append(f.stat().st_size)
    return _file_lengths_by_class


file_chunks = list(batched(files, len(files) // N_WORKERS + 1))

with mp.Pool(N_WORKERS) as pool:
    file_lengths_by_class_for_each_chunk = pool.map(get_file_lengths_by_class, file_chunks)


keys = set()
for d in file_lengths_by_class_for_each_chunk:
    keys.update(d.keys())

file_lengths_by_class = {k: [] for k in keys}
for k in keys:
    for d in file_lengths_by_class_for_each_chunk:
        file_lengths_by_class[k].extend(list(d[k]))

with open("file_lengths_by_class.json", "w") as fp:
    json.dump(file_lengths_by_class, fp)
