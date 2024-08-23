#!/bin/bash


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
    -analysisTimeoutPerFile $TIMEOUT \
    -import "$input" \
    -scriptPath "$SCRIPTS" \
    -postScript "disassembler.py" "$disassembled" \
    -postScript "decompiler.py" "$decompiled"

}
export -f run


# Command line interface
while [[ $# -gt 0 ]]; do
  case $1 in
    --root) export ROOT="${2%/}"; shift 2;;
    --timeout) export TIMEOUT="$2"; shift 2;;
    --jobs) export JOBS="$2"; shift 2;;
  *) echo "Usage: $0 --root <path> --timeout <seconds> --jobs <number>"; exit 1;;
  esac
done

[[ -z "$ROOT" || -z "$TIMEOUT" || -z "$JOBS" ]] && { 
  echo "Usage: $0 --root <path> --timeout <seconds> --jobs <number>";
  exit 1; 
}

echo "ROOT: $ROOT"
echo "TIMEOUT: $TIMEOUT"
echo "JOBS: $JOBS"

# rm -rf "logs/lift_"*
# rm -rf "$ROOT/ghidra/"*
# rm -rf "$ROOT/disassembled/"*
# rm -rf "$ROOT/decompiled/"*

parallel --bar -j $JOBS \
  'run {1} > ./logs/lift_{1}.txt 2>&1' \
  ::: $(printf "%02x\n" {0..255})

