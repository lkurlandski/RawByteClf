#!/bin/bash -l

#SBATCH --job-name=lift
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=16G


#
# sshpass -p "RITPassword1!" scp \
  # "lk3591@armitage.csec.rit.edu:/home/lk3591/transfer/zip/$1.zip" \
  # "$p_arc/$1.zip"
#

echo "lift.rc.sh: ($1)"

t_i=$(date +%s.%N)

# Configuration for Ghidra's headless analyzer.
TIMEOUT_ANALY="60"
TIMEOUT_DECOM="60"
PROCESSOR="x86:LE:32:default"
SCRIPT_PATH="/home/lk3591/Documents/code/RawByteClf/ghidra_scripts"

echo "TIMEOUT_ANALY: $TIMEOUT_ANALY"
echo "TIMEOUT_DECOM: $TIMEOUT_DECOM"
echo "PROCESSOR: $PROCESSOR"

P_LOG="./logsGhidra"
if [[ ! -d "$P_LOG" ]]; then
  echo "Error: Directory $P_LOG does not exist. Exiting."
  exit 1
fi
echo "P_LOG: $P_LOG"

echo "---------------------------------------------------------------------------"

# Determine which computer we're running on and set paths accordingly.
SYSTEM=$(<./config/.system)

if [[ "$SYSTEM" != "RC" && "$SYSTEM" != "LAB" && "$SYSTEM" != "ARMITAGE" ]]; then
  echo "Error: Invalid SYSTEM value '$SYSTEM'. Exiting."
  exit 1
fi
echo "SYSTEM: $SYSTEM"

if [[ "$SYSTEM" == "RC" ]]; then
  P_STABLE="/shared/rc/admalware/Sorel"
  P_TMP="/tmp"
elif [[ "$SYSTEM" == "ARMITAGE" ]]; then
  P_STABLE="/home/lk3591/Documents/datasets/Sorel"
  P_TMP="$P_STABLE/tmp"
elif [[ "$SYSTEM" == "LAB" ]]; then
  P_STABLE="/media/lk3591/easystore/datasets/Sorel"
  P_TMP="$P_STABLE/tmp"
fi

if [[ ! -d "$P_STABLE" ]]; then
  echo "Error: Directory $P_STABLE does not exist. Exiting."
  exit 1
fi
echo "_P_STABLE: $P_STABLE"

if [[ ! -d "$P_TMP" ]]; then
  echo "Error: Directory $P_TMP does not exist. Exiting."
  exit 1
fi
echo "_P_TMP: $P_TMP"

echo "---------------------------------------------------------------------------"

# Define and create stable directories for storage.
P_ARC="$P_STABLE/archived"
P_DIS="$P_STABLE/disassembled"
P_DEC="$P_STABLE/decompiled"

mkdir -p "$P_ARC"
mkdir -p "$P_DIS"
mkdir -p "$P_DEC"

echo "P_ARC: $P_ARC"
echo "P_DIS: $P_DIS"
echo "P_DEC: $P_DEC"

echo "---------------------------------------------------------------------------"

# Define and create temporary directories for fast read/writes.
p_root=$(mktemp -d "$P_TMP/tmpdir.XXXXXX")
p_arc="$p_root/archives"
p_bin="$p_root/binaries"
p_ghi="$p_root/ghidra"
p_dis="$p_root/disassembled"
p_dec="$p_root/decompiled"

mkdir $p_arc
mkdir $p_bin
mkdir $p_ghi
mkdir $p_dis
mkdir $p_dec

echo "Temporary Directories (Empty)"
tree "$p_root"

echo "---------------------------------------------------------------------------"

# Extract the raw binaries.
cp "$P_ARC/$1.zip" "$p_arc"
unzip -q -j "$p_arc/$1.zip" -d "$p_bin/"

# TEST: removes all files that do not match the regex.
# -------------------------------------------------------------------------------- #
# find "$p_bin" -type f ! -name "00000*" -delete
# echo "Populated Directories:"
# tree "$p_root"
# echo "---------------------------------------------------------------------------"
# -------------------------------------------------------------------------------- #

echo "Temporary Directories (Summary)"
cnt=$(find "$p_bin" -type f | wc -l)
siz=$(du -shc "$p_bin"/* | grep total | awk '{print $1}')
echo "Lifting $cnt files totaling $siz."

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Set up time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)

echo "---------------------------------------------------------------------------"

# Redirect stdout and stderr to keep the main log file clean.
p_log="$P_LOG/$1.log"
echo "Logging headlessAnalysis $p_log"

# Run Ghidra to disassemble and decompile the files.
analyzeHeadless \
  "$p_ghi" \
  "lift" \
  -recursive \
  -log $p_log \
  -processor $PROCESSOR \
  -analysisTimeoutPerFile $TIMEOUT_ANALY \
  -import "$p_bin" \
  -scriptPath "$SCRIPT_PATH" \
  -postScript "disassembler.py" "$p_dis" \
  -postScript "decompiler.py" "$p_dec" $TIMEOUT_DECOM \
  &> "$p_log"

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Ghidra time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)

echo "Data Generated (Summary):"
cnt=$(find "$p_dis" -type f | wc -l)
siz=$(du -shc "$p_dis"/* | grep total | awk '{print $1}')
echo "Disassembled $cnt files totaling $siz."

cnt=$(find "$p_dec" -type f | wc -l)
siz=$(du -shc "$p_dec"/* | grep total | awk '{print $1}')
echo "Decompiled $cnt files totaling $siz."
echo "---------------------------------------------------------------------------"


# Compress the output of Ghidra.
zip -q -9 "$p_root/dis_$1.zip" "$p_dis/"*
zip -q -9 "$p_root/dec_$1.zip" "$p_dec/"*
mv "$p_root/dis_$1.zip" "$P_DIS/$1.zip"
mv "$p_root/dec_$1.zip" "$P_DEC/$1.zip"

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Clean up: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)
