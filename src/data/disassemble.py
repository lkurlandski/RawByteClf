"""
Disassemble binaries.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zlib

from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import SYSTEM, System


if SYSTEM == System.ARMITAGE:
    SCRIPT_PATH = "/home/lk3591/lib/ghidra_10.3_PUBLIC/Ghidra/Features/Base/ghidra_scripts"
elif SYSTEM == System.LAB:
    SCRIPT_PATH = "/home/lk3591/lib/ghidra_11.1.2_PUBLIC/Ghidra/Features/Base/ghidra_scripts"

POST_SCRIPT = "DisassembleScript.java"


def run(f_in: Path) -> list[str]:

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["analyzeHeadless", tmpdir, "NONE", "-import", str(f_in)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["analyzeHeadless", tmpdir, "NONE", "-process", "-scriptPath", SCRIPT_PATH, "-postScript", POST_SCRIPT],
            check=True,
            capture_output=True,
        )

    return result.stdout.decode().split("\n")


def parse(lines: str) -> list[str]:

    output = []
    parse = False
    for line in lines:
        if "DISASSEMBLING - START" in line:
            parse = True
        if not parse:
            continue
        if "DISASSEMBLING - END" in line:
            break
        line = re.sub(r'INFO  DisassembleScript.java> ', '', line)
        line = re.sub(r'\s\(GhidraScript\)', '', line)
        output.append(line)

    return output


def save(f_out: Path, lines: list[str], compress: bool) -> None:

    output = "\n".join(lines) + "\n"
    mode = "w"

    if compress:
        output = zlib.compress(output.encode(), level=9)
        mode = "wb"

    with open(f_out, mode) as fp:
        fp.write(output)
    

def disassemble(f_in: Path, f_out: Path, compress: bool) -> None:
    lines = run(f_in)
    lines = parse(lines)
    save(f_out, lines, compress)


def main():

    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    if not args.input.exists() or not args.output.exists():
        raise FileNotFoundError()

    infiles = [args.input]
    if args.input.is_dir():
        infiles = list(args.input.iterdir())

    suffix = ".asm.zlib" if args.compress else ".asm"
    outfiles = [(args.output / f.name).with_suffix(suffix) for f in infiles]

    for f_in, f_out in tqdm(zip(infiles, outfiles), total=len(infiles)):
        try:
            disassemble(f_in, f_out, args.compress)
        except Exception as err:
            print(f"{f_in=} {err=}")


if __name__ == "__main__":
    main()

