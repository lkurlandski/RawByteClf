"""
Handles preprocessing of malware bytes.

# FIXME: there are some issues with trained tokenizers, e.g., Added Token 16391
# FIXME: the SPECIALS dict is duplicated; it is also in the cfg module
"""

from pathlib import Path


from src.tokenization import *


root = Path("/home/lk3591/Documents/datasets/Sorel")
dis = root / "disassembled"
dec = root / "decompiled"
stem = "00027e60efe5482ffb707e3a332e1cffb28ffef12b7d95dcc7d044d85c40a44e"

file = dis / f"{stem}.asm"
with open(file) as fp:
    text = fp.read()
text = text.split("\n\n")[3]
print(f"{text}\n{'-' * 88}")

text = DIS_NORMALIZER.normalize_str(text)
print(f"{text}\n{'-' * 88}")

text = DIS_PRETOKENIZER.pre_tokenize_str(text)
print(f"{text}\n{'-' * 88}")
