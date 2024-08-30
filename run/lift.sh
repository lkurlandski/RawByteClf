#!/bin/bash -l

#SBATCH --job-name=lift
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --time=00-12:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# USAGE
# -----
#   > sbatch --array=0-4095%512 run/lift.sh  # Everything
#   > sbatch --array=0-511 run/lift.sh       # Debug 000-0ff
#   > sbatch --array=0,4095 run/lift.sh      # Debug 000,00f
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
# OPTIMIZATIONS
#  - write Ghidra log file in /scratch instead of /home
#  - archive output files in /scratch before transferring to /shared
#    - adjust checkpointing system to read contents of archives
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

HHH=$(printf "%03x" "$SLURM_ARRAY_TASK_ID")
HH="${HHH:0:2}"
echo "HHH: $HHH"
echo "HH: $HH"

# Establish some timing variables.
t_start=$(date +%s.%N)
HOUR=12
TIME=$(( HOUR * 60 * 60 ))
TIME_FOR_CLEANUP=3600
if [ "$HOUR" -le 0 ] || [ "$TIME" -le 0 ]; then
  echo "Error: HOURS and TIME must both be greater than 0."
  exit 1
fi
echo "HOUR: $HOUR"
echo "TIME: $TIME"
echo "TIME_FOR_CLEANUP: $TIME_FOR_CLEANUP"

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
echo "SCRIPT_PATH: $SCRIPT_PATH"

# Determine which computer we're running on and set base paths accordingly.
#   P_FIN: long-term storage with possibly slow I/O
#   P_TMP: short-term storage with fast I/O

SYSTEM=$(<./config/.system)

if [[ "$SYSTEM" != "RC" && "$SYSTEM" != "LAB" && "$SYSTEM" != "ARMITAGE" ]]; then
  echo "Error: Invalid SYSTEM value '$SYSTEM'. Exiting."
  exit 1
fi
echo "SYSTEM: $SYSTEM"

if [[ "$SYSTEM" == "RC" ]]; then
  P_FIN="/shared/rc/admalware/Sorel"
  P_TMP="/tmp"
elif [[ "$SYSTEM" == "ARMITAGE" ]]; then
  P_FIN="/home/lk3591/Documents/datasets/Sorel"
  P_TMP="$P_FIN/tmp"
elif [[ "$SYSTEM" == "LAB" ]]; then
  P_FIN="/media/lk3591/easystore/datasets/Sorel"
  P_TMP="/home/lk3591/Documents/datasets/Sorel/tmp"
fi

if [[ ! -d "$P_FIN" ]]; then
  echo "Error: Directory P_FIN $P_FIN does not exist. Exiting."
  exit 1
fi
echo "P_FIN: $P_FIN"

if [[ ! -d "$P_TMP" ]]; then
  echo "Error: Directory P_TMP $P_TMP does not exist. Exiting."
  exit 1
fi
echo "P_TMP: $P_TMP"

# Define and create FIN directories.
p_fin_arc="$P_FIN/archived/$HH"
p_fin_dis="$P_FIN/disassembled/$HH"
p_fin_dec="$P_FIN/decompiled/$HH"
p_fin_log="$P_FIN/ghidraLogs/$HH"
mkdir -p "$p_fin_arc"
mkdir -p "$p_fin_dis"
mkdir -p "$p_fin_dec"
mkdir -p "$p_fin_log"
echo "p_fin_arc: $p_fin_arc"
echo "p_fin_dis: $p_fin_dis"
echo "p_fin_dec: $p_fin_dec"
echo "p_fin_log: $p_fin_log"

# Define and create TMP directories.
p_tmp_arc="$P_TMP/archived/$HHH"
p_tmp_dis="$P_TMP/disassembled/$HHH"
p_tmp_dec="$P_TMP/decompiled/$HHH"
p_tmp_log="$P_TMP/ghidraLogs/$HHH"
p_tmp_bin="$P_TMP/binaries/$HHH"
p_tmp_ghi="$P_TMP/ghidra/$HHH"
rm -rf "$p_tmp_arc"
rm -rf "$p_tmp_dis"
rm -rf "$p_tmp_dec"
rm -rf "$p_tmp_log"
rm -rf "$p_tmp_bin"
rm -rf "$p_tmp_ghi"
mkdir -p "$p_tmp_arc"
mkdir -p "$p_tmp_dis"
mkdir -p "$p_tmp_dec"
mkdir -p "$p_tmp_log"
mkdir -p "$p_tmp_bin"
mkdir -p "$p_tmp_ghi"
echo "p_tmp_arc: $p_tmp_arc"
echo "p_tmp_dis: $p_tmp_dis"
echo "p_tmp_dec: $p_tmp_dec"
echo "p_tmp_log: $p_tmp_log"
echo "p_tmp_bin: $p_tmp_bin"
echo "p_tmp_ghi: $p_tmp_ghi"

t_setup=$(date +%s.%N)
t_d=$(echo "$t_setup - $t_start" | bc)
printf "Set up time: %.6f seconds\n" $t_d

# Copy and extract binaries into the temporary directory.
cp "$p_fin_arc/$HHH.zip" "$p_tmp_arc/$HHH.zip"
unzip -q -j "$p_tmp_arc/$HHH.zip" -d "$p_tmp_bin/"
rm "$p_tmp_arc/$HHH.zip"

t_extract=$(date +%s.%N)
t_d=$(echo "$t_extract - $t_setup" | bc)
printf "Extraction time: %.6f seconds\n" $t_d


counter=0
while true; do


# Store files that have already successfully been disassembled.
# Search the final archive.
if [[ -f "$p_fin_dis/$HHH.zip" ]]; then
  fs_fin_dis=$(unzip -Z1 "$p_fin_dis/$HHH.zip" | sed 's/\.[^.]*$//')
else
  fs_fin_dis=""
fi
# Search the tmp directory.
for file in "$p_tmp_dis"/*; do
  if [[ -f "$file" ]]; then
    stem=$(basename "$file" | sed 's/\.[^.]*$//')
    fs_fin_dis=$(printf "%s\n%s" "$fs_fin_dis" "$stem")
  fi
done
echo "Num disassembled files: $(echo "$fs_fin_dis" | grep -c -v '^$')"

# Store files that have already successfully been decompiled.
# Search the final archive.
if [[ -f "$p_fin_dec/$HHH.zip" ]]; then
  fs_fin_dec=$(unzip -Z1 "$p_fin_dec/$HHH.zip" | sed 's/\.[^.]*$//')
else
  fs_fin_dec=""
fi
# Search the tmp directory.
for file in "$p_tmp_dec"/*; do
  if [[ -f "$file" ]]; then
    stem=$(basename "$file" | sed 's/\.[^.]*$//')
    fs_fin_dec=$(printf "%s\n%s" "$fs_fin_dec" "$stem")
  fi
done
echo "Num decompiled files: $(echo "$fs_fin_dec" | grep -c -v '^$')"

# Iterate over files in p_tmp_bin and remove if its already been processed.
for f in "$p_tmp_bin"/*; do
  s=$(basename "$f" | sed 's/\.[^.]*$//')
  in_fs_fin_dis=$(echo "$fs_fin_dis" | grep -w "$s")
  in_fs_fin_dec=$(echo "$fs_fin_dec" | grep -w "$s")
  if [ -n "$in_fs_fin_dis" ] && [ -n "$in_fs_fin_dec" ]; then
    echo "Skipping: $s"
    rm "$f"
  fi
done

cnt=$(find "$p_tmp_bin" -type f | wc -l)
siz=$(du -shc "$p_tmp_bin"/* | grep total | awk '{print $1}')
echo "Lifting $cnt files totaling $siz."

p_log="$p_tmp_log/$HHH.$counter.log"
t_ghidra=$(date +%s.%N)
timeout=$(echo "$TIME - ($t_start - $t_ghidra) - $TIME_FOR_CLEANUP" | bc)
echo "Running analyzeHeadless for $timeout seconds and logging to $p_log"

# Run Ghidra to disassemble and decompile the files.
timeout $timeout analyzeHeadless \
  "$p_tmp_ghi" \
  "lift" \
  -recursive \
  -log "$p_log" \
  -processor $PROCESSOR \
  -loader $LOADER \
  -analysisTimeoutPerFile $TIMEOUT_ANALY \
  -import "$p_tmp_bin" \
  -scriptPath "$SCRIPT_PATH" \
  -postScript "disassembler.py" "$p_tmp_dis" \
  -postScript "Decompiler.java" "$p_tmp_dec" $TIMEOUT_DECOM \
  &> "$p_log"

code=$?
echo "analyzeHeadless returned $code"
counter=$((counter+1))

# If analyzeHeadless completes or fails, its code will be returned by timeout.
# If analyzeHeadless times out, timeout returns 124.
# If analyzeHeadless returns 0 or timeout returns 124, we should exit the loop.
if [ $code -eq 0 ]; then
  break
fi
if [ $code -eq 124 ]; then
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

# Print time take during lifting process.
t_cleanup=$(date +%s.%N)
t_d=$(echo "$t_cleanup - $t_extract" | bc)
printf "Ghidra time: %.6f seconds\n" $t_d

# Print number of files disassembled and decompiled.
cnt=$(find "$p_tmp_dis" -type f | wc -l)
siz=$(du -shc "$p_tmp_dis"/* | grep total | awk '{print $1}')
echo "Disassembled $cnt files totaling $siz."
cnt=$(find "$p_tmp_dec" -type f | wc -l)
siz=$(du -shc "$p_tmp_dec"/* | grep total | awk '{print $1}')
echo "Decompiled $cnt files totaling $siz."

# Copy the log files to perminant storage.
mv "$p_tmp_log/"* "$p_fin_log"

# Compress the lifted files and move to final storage.
# If the archive is already present, zip will simply add new files to the archive.
zip -9 -r -j -q -u "$p_fin_dis/$HHH.zip" "$p_tmp_dis"
zip -9 -r -j -q -u "$p_fin_dec/$HHH.zip" "$p_tmp_dec"

t_transfer=$(date +%s.%N)
t_d=$(echo "$t_transfer - $t_cleanup" | bc)
printf "Transfer time: %.6f seconds\n" $t_d
