#!/bin/bash -l

#SBATCH --job-name=lift
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-12:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2


#
# CLEANUP
# -------
#   ALL:
#     rm /home/lk3591/Documents/code/RawByteClf/logs/lift_*.out
#     rm /home/lk3591/Documents/code/RawByteClf/logsGhidra/*.log
#   RC:
#     INT: rm -rf /scratch/lk3591/disassembled/*
#     INT: rm -rf /scratch/lk3591/decompiled/*
#     FIN: rm -rf /shared/rc/admalware/Sorel/disassembled/*
#     FIN: rm -rf /shared/rc/admalware/Sorel/decompiled/*
#   ARMITAGE:
#     INT:
#     INT:
#     FIN: rm -rf /home/lk3591/Documents/datasets/Sorel/disassembled/*
#     FIN: rm -rf /home/lk3591/Documents/datasets/Sorel/decompiled/*
#   LAB:
#     INT: rm -rf /home/lk3591/Documents/datasets/Sorel/disassembled/*
#     INT: rm -rf /home/lk3591/Documents/datasets/Sorel/decompiled/*
#     FIN: rm -rf /media/lk3591/easystore/datasets/Sorel/disassembled/*
#     FIN: rm -rf /media/lk3591/easystore/datasets/Sorel/decompiled/*
#
# NOTES
# -----
# - Don't forget to read the ~/lib/ghidra_11.1.2_PUBLIC/support/analyzeHeadless
#   file. There are several critical configurations defined there, e.g., MAXMEM.
#
# RETURN CODES
# ------------
# - 0: command completes without error
#     Things that don't error: analysisTimeoutPerFile, out of memory (from within
#       a script ONLY)
# - 1: various errors
#     Tested by reducing MAXMEM to small values. At very small values (20M), it looks
#       like the JVM cannot even start up properly and analyzeHeadless returns 1. At
#       modestly small values (100M) the analysis begins, but crashes partially through
#       with an error "OutOfMemoryError: Java heap space" and analyzeHeadless returns 1.
# - 130: keyboard interrupt
#
# PERFORMANCE
# -----------
#   ALL: --time=00-06:00:00, --mem=16G, --nodes=1
#   ---------------------------------------------
#   A: --ntasks=1, --cpus-per-task=1, MAXMEM=2G
#   B: --ntasks=2, --cpus-per-task=2, MAXMEM=2G
#   C: --ntasks=1, --cpus-per-task=4, MAXMEM=2G
#   D: --ntasks=4, --cpus-per-task=1, MAXMEM=2G
#   E: --ntasks=2, --cpus-per-task=4, MAXMEM=2G
#   F: --ntasks=4, --cpus-per-task=2, MAXMEM=2G
#   G: --ntasks=3, --cpus-per-task=4, MAXMEM=2G
#   H: --ntasks=4, --cpus-per-task=3, MAXMEM=2G
#   -------------------------------------------
#   I: --ntasks=1, --cpus-per-task=1, MAXMEM=4G
#   J: --ntasks=2, --cpus-per-task=2, MAXMEM=4G
#   K: --ntasks=1, --cpus-per-task=4, MAXMEM=4G
#   L: --ntasks=4, --cpus-per-task=1, MAXMEM=4G
#   M: --ntasks=2, --cpus-per-task=4, MAXMEM=4G
#   N: --ntasks=4, --cpus-per-task=2, MAXMEM=4G
#   O: --ntasks=3, --cpus-per-task=4, MAXMEM=4G
#   P: --ntasks=4, --cpus-per-task=3, MAXMEM=4G
#   -------------------------------------------
#   Q: --ntasks=1, --cpus-per-task=1, MAXMEM=100M
#
#   Based on the experiments above, it seems that Ghidra seems to perform SLIGHTLY
#   better when --cpus-per-task > 1.


echo "lift.sh: ($1)"

hh="${1:0:2}"
t_i=$(date +%s.%N)

# Configuration for Ghidra's headless analyzer.
TIMEOUT_ANALY="120"
TIMEOUT_DECOM="120"
PROCESSOR="x86:LE:32:default"
LOADER="PeLoader"
SCRIPT_PATH="/home/lk3591/Documents/code/RawByteClf/ghidra_scripts"

echo "TIMEOUT_ANALY: $TIMEOUT_ANALY"
echo "TIMEOUT_DECOM: $TIMEOUT_DECOM"
echo "PROCESSOR: $PROCESSOR"
echo "LOADER: $LOADER"

P_LOG="./logsGhidra"
if [[ ! -d "$P_LOG" ]]; then
  echo "Error: Directory $P_LOG does not exist. Exiting."
  exit 1
fi
echo "P_LOG: $P_LOG"

# Determine which computer we're running on and set base paths accordingly.
#   P_FIN: long-term storage with possibly slow I/O
#   P_INT: medium-term storage with fast I/O
#   P_TMP: short-term storage with fast I/O

SYSTEM=$(<./config/.system)

if [[ "$SYSTEM" != "RC" && "$SYSTEM" != "LAB" && "$SYSTEM" != "ARMITAGE" ]]; then
  echo "Error: Invalid SYSTEM value '$SYSTEM'. Exiting."
  exit 1
fi
echo "SYSTEM: $SYSTEM"

if [[ "$SYSTEM" == "RC" ]]; then
  P_FIN="/shared/rc/admalware/Sorel"
  P_INT="/scratch/lk3591"
  P_TMP="/tmp"
elif [[ "$SYSTEM" == "ARMITAGE" ]]; then
  P_FIN="/home/lk3591/Documents/datasets/Sorel"
  P_INT="$P_FIN"
  P_TMP="$P_INT/tmp"
elif [[ "$SYSTEM" == "LAB" ]]; then
  P_FIN="/media/lk3591/easystore/datasets/Sorel"
  P_INT="/home/lk3591/Documents/datasets/Sorel"
  P_TMP="$P_INT/tmp"
fi

if [[ ! -d "$P_FIN" ]]; then
  echo "Error: Directory P_FIN $P_FIN does not exist. Exiting."
  exit 1
fi
echo "P_FIN: $P_FIN"

if [[ ! -d "$P_INT" ]]; then
  echo "Error: Directory P_INT $P_INT does not exist. Exiting."
  exit 1
fi
echo "P_INT: $P_INT"

if [[ ! -d "$P_TMP" ]]; then
  echo "Error: Directory P_TMP $P_TMP does not exist. Exiting."
  exit 1
fi
echo "P_TMP: $P_TMP"

# Define and create FIN directories.
p_fin_arc="$P_FIN/archived/$hh"
p_fin_dis="$P_FIN/disassembled/$hh"
p_fin_dec="$P_FIN/decompiled/$hh"
mkdir -p "$p_fin_arc"
mkdir -p "$p_fin_dis"
mkdir -p "$p_fin_dec"
echo "p_fin_arc: $p_fin_arc"
echo "p_fin_dis: $p_fin_dis"
echo "p_fin_dec: $p_fin_dec"

# Define and create INT directories.
p_int_dis="$P_INT/disassembled/$hh"
p_int_dec="$P_INT/decompiled/$hh"
mkdir -p "$p_int_dis"
mkdir -p "$p_int_dec"
echo "p_int_dis: $p_int_dis"
echo "p_int_dec: $p_int_dec"

# Define and create TMP directories.
p_tmp_arc="$P_TMP/archived/$hh"
p_tmp_bin="$P_TMP/binaries/$hh"
p_tmp_ghi="$P_TMP/ghidra/$hh"
rm -rf "$p_tmp_arc"
rm -rf "$p_tmp_bin"
rm -rf "$p_tmp_ghi"
mkdir -p "$p_tmp_arc"
mkdir -p "$p_tmp_bin"
mkdir -p "$p_tmp_ghi"
echo "p_tmp_arc: $p_tmp_arc"
echo "p_tmp_bin: $p_tmp_bin"
echo "p_tmp_ghi: $p_tmp_ghi"

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Set up time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)

# Copy and extract binaries into the temporary directory.
cp "$p_fin_arc/$1.zip" "$p_tmp_arc/$1.zip"
unzip -q -j "$p_tmp_arc/$1.zip" -d "$p_tmp_bin/"
rm "$p_tmp_arc/$1.zip"

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Extraction time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)


counter=0
while true; do


# Create lists of file stems that have completed.
fs_fin_dis=$(find "$p_fin_dis" -type f -exec basename {} \; | sed 's/\.[^.]*$//')
fs_int_dis=$(find "$p_int_dis" -type f -exec basename {} \; | sed 's/\.[^.]*$//')
fs_fin_dec=$(find "$p_fin_dec" -type f -exec basename {} \; | sed 's/\.[^.]*$//')
fs_int_dec=$(find "$p_int_dec" -type f -exec basename {} \; | sed 's/\.[^.]*$//')

# Iterate over files in p_tmp_bin and remove if its already been processed.
for f in "$p_tmp_bin"/*; do
  s=$(basename "$f" | sed 's/\.[^.]*$//')
  in_fs_fin_dis=$(echo "$fs_fin_dis" | grep -w "$s")
  in_fs_int_dis=$(echo "$fs_int_dis" | grep -w "$s")
  in_fs_fin_dec=$(echo "$fs_fin_dec" | grep -w "$s")
  in_fs_int_dec=$(echo "$fs_int_dec" | grep -w "$s")
  if { [ -n "$in_fs_fin_dis" ] || [ -n "$in_fs_int_dis" ]; } && \
     { [ -n "$in_fs_fin_dec" ] || [ -n "$in_fs_int_dec" ]; }; then
    echo "Skipping: $s"
    rm "$f"
  fi
done

cnt=$(find "$p_tmp_bin" -type f | wc -l)
siz=$(du -shc "$p_tmp_bin"/* | grep total | awk '{print $1}')
echo "Lifting $cnt files totaling $siz."

# Redirect stdout and stderr to keep the main log file clean.
p_log="$P_LOG/$1.$counter.log"
echo "Logging headlessAnalysis $p_log"

# Run Ghidra to disassemble and decompile the files.
analyzeHeadless \
  "$p_tmp_ghi" \
  "lift" \
  -recursive \
  -log $p_log \
  -processor $PROCESSOR \
  -loader $LOADER \
  -analysisTimeoutPerFile $TIMEOUT_ANALY \
  -import "$p_tmp_bin" \
  -scriptPath "$SCRIPT_PATH" \
  -postScript "disassembler.py" "$p_int_dis" \
  -postScript "decompiler.py" "$p_int_dec" $TIMEOUT_DECOM \
  &> "$p_log"

code=$?
echo "analyzeHeadless returned $code"
counter=$((counter+1))

if [ $code -eq 0 ]; then
  break
fi

# Otherwise, we locate the file that caused an error in the Ghidra log
# and ensure it is skipped on the next loop iteration.
fail_line=$(grep "IMPORTING: file" "$p_log" | tail -n 1)
fail_stem=$(echo "$fail_line" | sed -n 's|.*IMPORTING: file://.*/\(.*\)\.exe.*|\1|p')
fail_file="$p_tmp_bin/$fail_stem.exe"
echo "fail_file: $fail_file"
rm "$fail_file"


done


t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Ghidra time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)

cnt=$(find "$p_int_dis" -type f | wc -l)
siz=$(du -shc "$p_int_dis"/* | grep total | awk '{print $1}')
echo "Disassembled $cnt files totaling $siz (some may already have been present)."

cnt=$(find "$p_int_dec" -type f | wc -l)
siz=$(du -shc "$p_int_dec"/* | grep total | awk '{print $1}')
echo "Decompiled $cnt files totaling $siz (some may already have been present)."

# Move everything from intermediate storage to final storage.
if [[ $p_int_dis != $p_fin_dis ]]; then
  rsync --archive --remove-source-files "$p_int_dis/" "$p_fin_dis/"
fi
if [[ $p_int_dec != $p_fin_dec ]]; then
  rsync --archive --remove-source-files "$p_int_dec/" "$p_fin_dec/"
fi

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Clean up: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)
