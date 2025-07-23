"""
Process PE files.
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable
from collections import Counter, namedtuple, OrderedDict
from enum import Enum, IntFlag, auto
from itertools import islice, repeat, tee
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat, pprint
import struct
import sys
import tempfile
import time
from typing import Literal, Optional
import zipfile

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import lief
import numpy as np


# pylint: disable=c-extension-no-member


def is_dotnet(raw: str | Path | bytes) -> bool:
    pe: lief.PE.Binary = lief.parse(raw)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    dd: lief.PE.DataDirectory = pe.data_directory(lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER)
    if not isinstance(dd, lief.PE.DataDirectory):
        raise RuntimeError(f"Expected lief.PE.DataDirectory, got {type(pe)}")
    return dd.rva != 0 and dd.size != 0


def get_machine_and_subsystem(data: str | Path | bytes) -> tuple[lief.PE.Header.MACHINE_TYPES, lief.PE.OptionalHeader.SUBSYSTEM]:
    pe = lief.parse(data)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    return pe.header.machine, pe.optional_header.subsystem


def patch_binary(
    src: str | Path | bytes,
    machine: Optional[lief.PE.Header.MACHINE_TYPES] = None,
    subsystem: Optional[lief.PE.OptionalHeader.SUBSYSTEM] = None,
) -> bytes:
    if isinstance(src, (str, Path)):
        data = bytearray(Path(src).read_bytes())
    elif isinstance(src, bytes):
        data = bytearray(src)
    else:
        raise TypeError(f"Unsupported type for src: {type(src)}. Expected str, Path, or bytes.")

    # ---- Locate headers ----
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    pe_sig_off   = e_lfanew
    filehdr_off  = pe_sig_off + 4          # after "PE\0\0"
    opthdr_off   = filehdr_off + 20        # IMAGE_FILE_HEADER is 20 bytes

    # ---- Patch Machine (IMAGE_FILE_HEADER.Machine) ----
    if machine is not None:
        if hasattr(machine, "value"):  # LIEF enum
            machine = machine.value
        struct.pack_into("<H", data, filehdr_off + 0, machine)

    # ---- Find Subsystem offset in Optional Header ----
    magic = struct.unpack_from("<H", data, opthdr_off)[0]
    # Subsystem offset depends on PE32 vs PE32+
    subsys_rel = 68 if magic == 0x10B else 72  # 0x10B=PE32, 0x20B=PE32+
    if subsystem is not None:
        if hasattr(subsystem, "value"):
            subsystem = subsystem.value
        struct.pack_into("<H", data, opthdr_off + subsys_rel, subsystem)

    return bytes(data)


def rearm_disarmed_binary(src: str | Path | bytes, sha: str, check: bool = True, verbose: bool = False) -> bytes:
    """
    Rearm a Sorel binary by patching its machine and subsystem until the target SHA-256 matches.
    """
    if isinstance(src, (str, Path)):
        if not Path(src).exists():
            raise FileNotFoundError(f"Source file {src} does not exist.")
        data = Path(src).read_bytes()
    elif isinstance(src, bytes):
        data = src
    else:
        raise TypeError(f"Unsupported type for src: {type(src)}. Expected str, Path, or bytes.")

    machines = []
    for v in lief.PE.Header.MACHINE_TYPES.__dict__.values():
        if type(v).__name__ == "MACHINE_TYPES":
            machines.append(v)

    subsystems = []
    for v in lief.PE.OptionalHeader.SUBSYSTEM.__dict__.values():
        if type(v).__name__ == "SUBSYSTEM":
            subsystems.append(v)

    if verbose:
        machine, subsystem = get_machine_and_subsystem(data)
        print(f"Target SHA-256: {sha}")
        print(f"Current Machine: {machine}")
        print(f"Current Subsystem: {subsystem}")

    for machine in machines:
        for subsystem in subsystems:
            patched = patch_binary(data, machine=machine, subsystem=subsystem)
            s = hashlib.sha256(patched).hexdigest()
            if verbose:
                print(f"Ran Machine: {machine}, Subsystem: {subsystem} -> SHA-256: {s}")
            if s == sha:
                if check:
                    machine_, subsystem_ = get_machine_and_subsystem(patched)
                    if machine_ != machine or subsystem_ != subsystem:
                        raise ValueError(f"Patched binary does not match expected machine and subsystem: {machine_}, {subsystem_}")
                    equal = np.equal(np.frombuffer(data, dtype=np.uint8), np.frombuffer(patched, dtype=np.uint8))
                    if len(equal) - np.sum(equal) > 3:
                        raise ValueError(f"Patched binary differs from original by more than 3 bytes: {len(equal) - np.sum(equal)} differences.")
                return patched

    raise ValueError(f"Could not find matching machine and subsystem for SHA-256: {sha}")
