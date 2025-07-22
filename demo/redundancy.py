"""
"""

from collections import Counter
import json
from typing import Literal


print("redundancy.py: Measuring the redundancy within the dataset.")


def get_digests(files: list[str]) -> dict[Literal["raw", "dis", "dec"], dict[str, str]]:
    digests = {}
    for f in files:
        with open(f, "r") as fp:
            digests[f.split("/")[-2]] = json.load(fp)
    return digests


def core_stats(digests: dict[Literal["raw", "dis", "dec"], dict[str, str]]) -> None:
    for l, d in digests.items():
        c = Counter(d.values())
        num_nonunique = sum(1 for v in c.values() if v > 1)
        print(f"{l}: Total samples: {len(d)} Unique species: {len(c)} Non-unique: {num_nonunique} Percent Unique: {round(100 * len(c) / len(d))}%")


for dnm in ["ass", "bod", "sor", "win"]:
    print(f"Analyzing redundancy for {dnm}")
    files = [f"./data/{dnm}/{lll}/digests.json" for lll in ["raw", "dis", "dec"]]
    digests = get_digests(files)
    core_stats(digests)
