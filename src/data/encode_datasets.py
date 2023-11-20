"""
Encode the datasets categorically.
"""

from collections import defaultdict
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

from datasets import Dataset, Features, Value, ClassLabel

from src.cfg import INPUT_PATH, TMP_DIR
from src.data.cfg import MAX_SHARD_SIZE
from src.data.utils import PerDatasetArgumentParser


ITER_SIZE = 4
PRE = "is_"


class ApplyEncodeFn(Protocol):
    def __call__(self, dataset: Dataset) -> Dataset:
        ...


def encode_single_label_dataset(dataset: Dataset) -> Dataset:
    return dataset.class_encode_column("labels")


def encode_multi_label_dataset(dataset: Dataset) -> Dataset:
    """Expects `labels` column to consist of lists of str or None.
    Returns a dataset with a boolean column for each unique label named `is_<label>`.
    """

    columns = defaultdict(lambda: [False] * len(dataset))
    for i, d in enumerate(dataset.select_columns("labels").iter(ITER_SIZE)):
        for j, per_sample_labels in enumerate(d["labels"]):
            if per_sample_labels is None:
                continue
            for l in per_sample_labels:
                columns[f"{PRE}{l}"][i * ITER_SIZE + j] = True

    for name, column in columns.items():
        dataset = dataset.add_column(name, column)

    features = Features(
        {
            "name": Value("string"),
            "bytes": Value("binary"),
            "size": Value("int64"),
            "length": Value("int64"),
        }
        | {k: Value("bool") for k in columns}
    )
    dataset = dataset.remove_columns("labels").cast(features)
    return dataset


def encode_dataset(
    path: Path,
    apply_encode_fn: ApplyEncodeFn,
    override: bool = False,
    save: bool = False,
    verbose: bool = False,
) -> Dataset:
    dataset = Dataset.load_from_disk(path)

    # TODO: this may or may not make sense...
    if "labels" in dataset.features and isinstance(dataset.features["labels"], ClassLabel):
        if not override:
            raise ValueError("Dataset already encoded.")
    elif any(PRE in c for c in dataset.features):
        if not override:
            raise ValueError("Dataset already encoded.")

    if verbose:
        print(f"Unencoded: {dataset}")
        for i, d in enumerate(dataset):
            print(d["name"], d["labels"])
            if i == 16:
                break

    dataset = apply_encode_fn(dataset)

    if verbose:
        print(f"Encoded: {dataset}")
        for i, d in enumerate(dataset):
            print(d["name"], d["labels"])
            if i == 16:
                break

    if save:
        temp_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
        dataset.save_to_disk(temp_dir.as_posix(), max_shard_size=MAX_SHARD_SIZE)
        shutil.rmtree(path)
        temp_dir.rename(path)

    return dataset


def main(datasets: list[str], override: bool, save: bool, verbose: bool) -> None:
    if "bodmas_pe" in datasets:
        path = INPUT_PATH / "bodmas_pe"

    for d in [d for d in datasets if all(s not in d for s in ["local"])]:
        path = INPUT_PATH / d
        apply_encode_fn = encode_multi_label_dataset
        if d == "bodmas_pe":
            apply_encode_fn = encode_single_label_dataset
        dataset = encode_dataset(path, apply_encode_fn, override, save, verbose)
        dataset.cleanup_cache_files()


def cli() -> None:
    parser = PerDatasetArgumentParser()
    parser.add_argument("--override", action="store_true", help="Override existing labels.")
    parser.add_argument("--save", action="store_true", help="Save the labeled dataset.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.datasets, args.override, args.save, args.verbose)


def debug() -> None:
    main(["malware_bazaar_macho"], True, False, True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and "debug" in sys.argv[1].lower():
        debug()
    else:
        cli()
