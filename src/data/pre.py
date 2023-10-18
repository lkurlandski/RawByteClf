"""
Prepocessing files.
"""

import bz2
import gzip
from io import BufferedReader
import lzma
from pathlib import Path
from typing import Optional
import zlib

import py7zr


SIG_GZIP = b'\x1f\x8b\x08'
SIG_BZIP2 = b'\x42\x5a\x68'
SIG_LZMA = b'\xfd7zXZ\x00'
SIG_ZLIB = b'\x78\x01'
SIG_7Z = b'7z'


def decompress_fp(fp: BufferedReader) -> bytes:
    fp.seek(0)
    signature = fp.read(10)

    if signature.startswith(SIG_GZIP):
        with gzip.open(fp, 'rb') as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_BZIP2):
        with bz2.BZ2File(fp, 'rb') as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_LZMA):
        with lzma.open(fp, 'rb') as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_ZLIB):
        return zlib.decompress(fp.read())

    if signature.startswith(SIG_7Z):
        with py7zr.SevenZipFile(fp, mode='r') as archive:
            file_list = archive.getnames()
            if len(file_list) != 1:
                raise ValueError("The 7zip archive does not contain a single file.")
            return archive.read(file_list[0])

    fp.seek(0)
    return fp.read()


def decompress(file_or_file_pointer: str | Path | bytes | BufferedReader, outfile: Optional[Path] = None) -> bytes:
    if isinstance(file_or_file_pointer, (str, Path, bytes)):
        with open(file_or_file_pointer, 'rb') as fp:
            b = decompress_fp(fp)
    else:
        b = decompress_fp(file_or_file_pointer)

    if outfile:
        with open(outfile, "rb") as fp:
            fp.write(b)

    return b
