"""
"""

# TODO: do this for the timestamps?

from pathlib import Path

from tqdm import tqdm

keep = set(Path("/home/lk3591/Documents/code/RawByteClf/demo/shas.txt").read_text().split("\n")[:-1])
print(f"Looking for {len(keep)} shas, e.g., {next(iter(keep))}")

src = Path("/home/lk3591/Documents/datasets/Sorel")
dst = Path("/home/lk3591/Documents/code/RawByteClf/demo/datasets/Sorel")

def process(name: str) -> None:
    f_i = src / name
    f_o = dst / name

    print(f"Processing {name}: {f_i.as_posix()} --> {f_o.as_posix()}")

    matches = 0
    with open(f_i, "r") as fp_i, open(f_o, "w") as fp_o:
        for line in tqdm(fp_i, total=2455209):
            sha = line.split()[0]
            if sha in keep:
                print(f"Matched: {sha}")
                fp_o.write(f"{line.strip()}\n")
                matches += 1
    print(f"Copied {matches} lines. Done.")

process("avclass_cache.txt")
process("avclass_family_cache.txt")
process("claravy_cache.txt")
