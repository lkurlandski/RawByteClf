"""
Process the raw json file output of the detection script.
"""

import csv
from itertools import chain, islice
import json
from pathlib import Path
from pprint import pprint
import sys

from tqdm import tqdm


ALGS = ("recursive", "deep", "heuristic")
ROOT = Path("/home/lk3591/Documents/datasets/Sorel/diec")
ROOT = Path("/home/lk3591/Documents/code/RawByteClf/diec")
INPUT = ROOT
INPUTS = {alg : INPUT / alg for alg in ALGS}
OUTPUT = ROOT / "merged/"
PARSED = ROOT / "parsed"


def merge():

    OUTPUT.mkdir(exist_ok=True)
    
    
    files = islice(chain.from_iterable(d.iterdir() for d in INPUTS.values()), None)
    files = list(tqdm(files, desc="Initial Scan..."))
    shas = set(file.stem for file in files)
    print(f"{len(files)} reports from {len(shas)} unique files.")
    
    
    errors = {name: 0 for name in (0, 1, 2)}
    pbar = tqdm(shas)
    for sha in pbar:
        pbar.set_description(f"Processing: {sha}")
        files = {alg: (path / sha).with_suffix(".txt") for alg, path in INPUTS.items()}
        data = {}
        for alg, file in files.items():
            s = str(Path(file.parent.name) / file.name)
            if not file.exists():
                print(f"File not found: {s}")
                d = None
                errors[0] += 1
            elif file.stat().st_size == 0:
                print(f"File is empty: {s}")
                d = None
                errors[1] += 1
            else:
    
                with open(file, "r") as fp:
                    raw = fp.read()
                content = raw[raw.find("{"):raw.rfind("}") + 1].strip()
    
                try:
                    d = json.loads(content)
                except json.JSONDecodeError:
                    print(f"JSONDecodeError: {s}")
                    print(f"*****{content}*****")
                    errors[2] += 1
                    d = None
    
            data[alg] = d
    
        outfile = (OUTPUT / sha).with_suffix(".json")
        with open(outfile, "w") as fp:
            json.dump(data, fp, indent=4)
    
    print("ERRORS\n------")
    print(f"\tfile not found: {errors[0]}")
    print(f"\tFile is empty: {errors[1]}")
    print(f"\tJSONDecodeError: {errors[2]}")


def parse():

    def packeds_decision(packeds: list[bool]) -> bool:
        return any(packeds)

    def packers_decision(packers: list[str]) -> str:
        packers = [p if p != "Packer detected" else "Heuristic" for p in packers]
        packers = [p for p in packers if p != ""]
        return "|".join(packers) if packers != "" else ""

    def parse_values_blob(values: list[dict]) -> tuple[bool, str]:
        packeds: list[bool] = []
        packers: list[str] = []
        for value in values:
            if "values" in value:
                packed, packer = parse_values_blob(value.get("values"))
            elif value.get("type") == "Packer":
                packed = True
                packer = value.get("name", "")
            else:
                packed = False
                packer = ""

            packeds.append(packed)
            packers.append(packer)

        return packeds_decision(packeds), packers_decision(packers)

    def parse_detects_blob(detects: list[dict]) -> tuple[bool, str]:
        packeds: list[bool] = []
        packers: list[str] = []

        for detect in detects:
            packed, packer = parse_values_blob(detect.get("values", []))
            packeds.append(packed)
            packers.append(packer)

        return packeds_decision(packeds), packers_decision(packers)


    PARSED.mkdir(exist_ok=True)

    output = {}
    pbar = tqdm(OUTPUT.iterdir(), total=sum(1 for _ in OUTPUT.iterdir()))
    for i, file in enumerate(pbar):
        sha = file.stem
        pbar.set_description(f"Processing: {sha}")
        with open(file, "r") as f:
            data = json.load(f)

        output[sha] = {}
        for mode, d in data.items():
            if d is None:
                output[sha][mode] = None
                continue
            packed, packer = parse_detects_blob(d.get("detects", []))
            output[sha][mode] = {"packed": packed, "packer": packer}

    with open(PARSED / "output.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    merge()
    parse()
