#!/bin/bash


#
#
#


export SCRIPTS="/home/lk3591/Documents/code/RawByteClf/ghidra_scripts"


run_ghidra() {

  # Run Ghidra recursively.
 
  # Parameters:
    # $1 (location)
    # $2 (input)
    # $3 (disassembled)
    # $4 (decompiled)

  local location=$1
  local input=$2
  local disassembled=$3
  local decompiled=$4

  mkdir -p $location
  mkdir -p $disassembled
  mkdir -p $decompiled

  analyzeHeadless \
    "$location" "lift" \
    -recursive \
    -analysisTimeoutPerFile $TIMEOUT \
    -import "$input" \
    -scriptPath "$SCRIPTS" \
    -postScript "disassembler.py" "$disassembled" \
    -postScript "decompiler.py" "$decompiled"

}
export -f run_ghidra


run() {

  # Lift binaries that begin with `hhh`.

  # Parameters:
    # $1 (hhh)

  local hhh=$1
  local hh="${hhh:0:2}"

  local location="$ROOT/ghidra/$hhh"  # Must be a unique directory.
  local disassembled="$ROOT/disassembled/$hh"
  local decompiled="$ROOT/decompiled/$hh"

  local source="$ROOT/binariesTmp/$hh" # FIXME: change before committing.
  local input="$ROOT/.liftTmp/$hhh"
  mkdir -p "$input"

  for file in "$source"/"$hhh"*; do
    if [ -e "$file" ]; then
      filename=$(basename "$file")
        ln -s "$file" "$input/$filename"
    fi
  done

  # FIXME: remove
  echo "hhh: --$hhh--"
  echo "hh: --$hh--"
  echo "location: --$location--"
  echo "disassembled: --$disassembled--"
  echo "decompiled: --$decompiled--"
  echo "source: --$source--"
  echo "input: --$input--"
  # exit

  run_ghidra "$location" "$input" "$disassembled" "$decompiled"

  rm -rf "$input" 
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

rm -rf "logs/lift_"*
rm -rf "$ROOT/.liftTmp/"
rm -rf "$ROOT/ghidra/"
rm -rf "$ROOT/disassembled/"
rm -rf "$ROOT/decompiled/"

run "fff"

# parallel --bar -j $JOBS \
#    'run {1} > ./logs/lift_{1}.txt 2>&1' \
#    ::: $(printf "%02x\n" {255..0})

