"""
Various codes for data analysis.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def overflow_analysis(path: Path) -> dict[str, pd.DataFrame]:
    data: dict[tuple] = defaultdict(dict)
    for f in path.iterdir():
        df = pd.read_csv(f, index_col=False)
        data[f.stem] = df
    return data


def main():
    path = ".output/mymalconv/65536/clf/1/False/False/False/tuning_results/dataframe.csv"
    df = process_tuning_dataframe(path)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)
    


if __name__ == "__main__":
    main()
