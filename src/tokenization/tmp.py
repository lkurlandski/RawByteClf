"""
Handles preprocessing of malware bytes.

# FIXME: there are some issues with trained tokenizers, e.g., Added Token 16391
# FIXME: the SPECIALS dict is duplicated; it is also in the cfg module
"""

import os
from pathlib import Path
from pprint import pprint
import sys

from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.tokenization.decompiled import _get_dec_normalizer, _get_dec_pretokenizer
from src.tokenization.disassembled import _get_dis_normalizer, _get_dis_pretokenizer
from src.tokenization.train import TokenizationTrainingIterator


root = Path("/media/lk3591/easystore/datasets/Sorel/tmp")
dis = root / "dis"
dec = root / "dec"
stem = "001779de7d1c12500d86b2e5ab55e31dd1e2c5db7dfb3d242878f0d0ac3a5f97"
dis_file = dis / f"{stem}.asm"
dec_file = dec / f"{stem}.c"


def pretokenizer_to_str(d) -> str:
    return "\n".join([f"|{s}|" for s, _ in d])


def test_dis():
    text = dis_file.read_text()

    normalizer = _get_dis_normalizer(False, False)
    text = normalizer.normalize_str(text)
    pretokenizer = _get_dis_pretokenizer()
    text = pretokenizer.pre_tokenize_str(text)
    print(f"{text}\n{'-' * 88}")


def test_dec():
    text = dec_file.read_text()

    breaks = [85, 147, 444, 445, 7078222, 7078223, 7078306, 7078354]
    # breaks = []
    # for i, (t_1, t_2) in enumerate(tqdm(zip(text, text[1:]), total=len(text))):
    #     if t_1 == "\n" and t_2 == "\n":
    #         breaks.append(i)
    #     if len(breaks) < -10:
    #         break
    # print(breaks[0:4] + breaks[-4:])

    text = text[:breaks[2]] + text[breaks[-4]:]
    print(f"{text}\n{'-' * 88}")

    normalizer = _get_dec_normalizer()
    text = normalizer.normalize_str(text)
    print(f"{text}\n{'-' * 88}")

    pretokenizer = _get_dec_pretokenizer()
    text = pretokenizer.pre_tokenize_str(text)
    text = pretokenizer_to_str(text)
    print(f"{text}\n{'-' * 88}")


def test_gen():

    files = sorted(dis.iterdir())[0:4]
    iterable = TokenizationTrainingIterator("dis", files, 2, bytes.decode)()
    print(iterable)
    for batch in iterable:
        print(f"{type(batch)=}")
        print(f"{len(batch)=}")
        for item in batch:
            print(f"{type(item)=}")
            print(f"{len(item)=}")
            print(f"{item[0:32]=}")


if __name__ == "__main__":
    test_dis()
