#!/bin/bash


export ROOT="/media/lk3591/easystore/datasets/Sorel/"
export SCRIPTS="/home/lk3591/Documents/code/RawByteClf/ghidra_scripts"


run() {

  location="$ROOT/ghidra/$1"
  input="$ROOT/binaries/$1"
  disassembled="$ROOT/disassembled/$1"
  decompiled="$ROOT/decompiled/$1"

  mkdir -p "$location"
  mkdir -p "$disassembled"
  mkdir -p "$decompiled"

  analyzeHeadless \
    "$location" "lift" \
    -recursive \
    -import "$input" \
    -scriptPath "$SCRIPTS" \
    -postScript "disassembler.py" "$disassembled" \
    -postScript "decompiler.py" "$decompiled"

}


export -f run


parallel --bar -j 8 \
  'run {1} > ./logs/lift_{1}.txt 2>&1' \
  ::: $(printf "%02x\n" {0..255})

