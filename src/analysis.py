"""
Various codes for data analysis.
"""

from collections import defaultdict, Counter, OrderedDict
import json
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Literal

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd


def process_tuning_dataframe(
    path: Path, objective_col: str = "eval_loss", ascending: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df[[objective_col] + [c for c in df.columns if "config/" in c]]
    df = df.rename(columns={c : c.replace("config/", "") for c in df.columns})
    df = df.sort_values(by=objective_col, ascending=ascending)
    return df


def process_trainer_state(path: Path) -> tuple[list[dict], list[dict]]:
    with open(path, "r") as fp:
        trainer_state = json.load(fp)
    log_history = trainer_state["log_history"]
    validation_reports = [d for d in log_history if "eval_loss" in d]
    train_reports = [d for d in log_history if "loss" in d]
    return validation_reports, train_reports


def process_validation_reports(
    reports: list[dict],
    metrics: tuple[str] = ("eval_loss", "eval_accuracy", "eval_f1-macro"),
    lower_is_betters: tuple[bool] = (True, False, False),
) -> dict[tuple[float, int]]:
    """
    Returns a dict for each metric containing a tuple indicating the best value
    along with the first index at which the value was found.
    """
    if len(metrics) != len(lower_is_betters):
        raise ValueError()

    results = {}
    for metric, lower_is_better in zip(metrics, lower_is_betters):
        values = np.array([r[metric] for r in reports])
        if lower_is_better:
            loc = np.argmin(values)
        else:
            loc = np.argmax(values)
        best = values[loc]
        results[metric] = (best, loc)

    return results


def overflow_analysis(path: Path) -> dict[str, pd.DataFrame]:
    data: dict[tuple] = defaultdict(dict)
    for f in path.iterdir():
        df = pd.read_csv(f, index_col=False)
        data[f.stem] = df
    return data


def tokenizer_vocab_analysis(
    path: os.PathLike,
    alg: Literal["BPE", "Unigram"],
    num_special_tokens: int = 0,
) -> Counter:
    with open(path) as fp:
        d = json.load(fp)

    if alg == "Unigram":
        tokens = [k for k, _ in d["model"]["vocab"][num_special_tokens:]]
    elif alg == "BPE":
        tokens = [k for k, _ in list(d["model"]["vocab"].items())[num_special_tokens:]]
    else:
        raise ValueError(f"Invalid alg: {alg}")

    return Counter([len(k) for k in tokens]), tokens


def main():

    ALGORITHMS = ["BPE", "Unigram"]
    VOCAB_SIZES = [2 ** i for i in range(9, 21)] + [2 ** i + 2 ** (i - 1) for i in range(9, 21)]
    VOCAB_SIZES.remove(2 ** 20 + 2 ** 19)
    VOCAB_SIZES.sort()
    NUM_FILES = 2000
    TOKEN_LENGTHS = list(range(1, 17))
    NUM_SPECIAL_TOKENS = 7

    data: dict[str, dict[str, Counter]] = {a: {v: None for v in VOCAB_SIZES} for a in ALGORITHMS}
    for a in ALGORITHMS:
        for v in VOCAB_SIZES:
            path = Path(f"./output/tokenizers/{a}_{v}_{NUM_FILES}.json")
            if not path.exists():
                print(f"File not found: {path}")
                print(f"Train a {a} tokenizer with vocab size {v}")
                continue
            c, tokens = tokenizer_vocab_analysis(path, a, NUM_SPECIAL_TOKENS)
            l = len(set(tokens))
            if l != v:
                print(f"Expected to find {v} tokens, but found {l} tokens for {a} tokenizer!")
            data[a][v] = c
            c = OrderedDict(sorted(c.items(), key=lambda x: x[0]))
            pprint(c)

    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ALGORITHMS), len(VOCAB_SIZES), figsize=(15, 10))

    for i, algorithm in enumerate(ALGORITHMS):
        for j, vocab_size in enumerate(VOCAB_SIZES):
            if data[algorithm][vocab_size] is None:
                continue
            counts = [data[algorithm][vocab_size][length] for length in TOKEN_LENGTHS]
            ax: Axes = axes[i][j]
            ax.bar(TOKEN_LENGTHS, counts, color='skyblue')
            ax.set_title(f'{algorithm[0:3]}@{vocab_size}')
            ax.set_xticks(TOKEN_LENGTHS, [str(TOKEN_LENGTHS[0])] + ["" for _ in range(len(TOKEN_LENGTHS) - 2)] + [str(TOKEN_LENGTHS[-1])])
            # ax.set_xlabel('Token Length')
            # ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()
    plt.savefig("tmp/fig.png", dpi=800)


# File not found: output/tokenizers/BPE_12288_2000.json
# Train a BPE tokenizer with vocab size 12288

# File not found: output/tokenizers/BPE_24576_2000.json
# Train a BPE tokenizer with vocab size 24576


if __name__ == "__main__":
    main()
