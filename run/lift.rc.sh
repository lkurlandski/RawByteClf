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
# Notes:
#

# t_i=$(date +%s.%N)
# t_f=$(date +%s.%N)
# t_d=$(echo "$t_f - $t_i" | bc)
# printf "Execution time: %.6f seconds\n" $t_d


echo "lift.rc.sh: ($1)"


t_i=$(date +%s.%N)


TIMEOUT_ANALY="60"
TIMEOUT_DECOM="60"
PROCESSOR="x86:LE:32:default"
SCRIPT_PATH="/home/lk3591/Documents/code/RawByteClf/ghidra_scripts"
ARM_PATH="/home/lk3591/transfer"

echo "TIMEOUT_ANALY: $TIMEOUT_ANALY"
echo "TIMEOUT_DECOM: $TIMEOUT_DECOM"
echo "PROCESSOR: $PROCESSOR"
echo "--------------------------------------------------------------------------------"


# Define and create stable directories for storage.
P_ARC="/shared/rc/admalware/Sorel/tmp"
P_DIS="/shared/rc/admalware/Sorel/disassembled"
P_DEC="/shared/rc/admalware/Sorel/decompiled"

mkdir -p "$P_DIS"
mkdir -p "$P_DEC"


# Define and create temporary directories for fast read/writes.
p_root=$(mktemp -d /tmp/tmpdir.XXXXXX)
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

echo "Empty Directory Structure:"
tree "$p_root"
echo "--------------------------------------------------------------------------------"


# Transfer the archives from armitage.
# sshpass -p "RITPassword1!" scp \
  # "lk3591@armitage.csec.rit.edu:/home/lk3591/transfer/zip/$1.zip" \
  # "$p_arc/$1.zip"

# Extract the raw binaries.
cp "$P_ARC/$1.zip" "$p_arc"
unzip -q -j "$p_arc/$1.zip" -d "$p_bin/"

echo "Populated Directories:"
# tree "$p_root"
# echo "--------------------------------------------------------------------------------"

# TEST: removes all files that do not match the regex.
# -------------------------------------------------------------------------------- #
# find "$p_bin" -type f ! -name "00000*" -delete
# echo "Populated Directories:"
# tree "$p_root"
# echo "--------------------------------------------------------------------------------"
# -------------------------------------------------------------------------------- #


# Print the number and size of the data to be operated upon.
cnt=$(find "$p_bin" -type f | wc -l)
siz=$(du -shc "$p_bin"/* | grep total | awk '{print $1}')
echo "Lifting $cnt files totaling $siz."
echo "--------------------------------------------------------------------------------"

# Redirect stdout and stderr to keep the main log file clean.
p_log="./logsGhidra/$1.log"
echo "Logging to $p_log"


t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Set up time: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)

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

echo "Data Generated:"
# tree "$p_root"
# echo "--------------------------------------------------------------------------------"

cnt=$(find "$p_dis" -type f | wc -l)
siz=$(du -shc "$p_dis"/* | grep total | awk '{print $1}')
echo "Disassembled $cnt files totaling $siz."

cnt=$(find "$p_dec" -type f | wc -l)
siz=$(du -shc "$p_dec"/* | grep total | awk '{print $1}')
echo "Decompiled $cnt files totaling $siz."
echo "--------------------------------------------------------------------------------"


# Compress the output of Ghidra.
# Transfer the archives to armitage.
zip -q -9 "$p_root/dis_$1.zip" "$p_dis/"*
zip -q -9 "$p_root/dec_$1.zip" "$p_dec/"*
mv "$p_root/dis_$1.zip" "$P_DIS/$1.zip"
mv "$p_root/dec_$1.zip" "$P_DEC/$1.zip"

# scp "$p_root/dis.zip" "lk3591@armitage.csec.rit.edu:/home/lk3591/transfer/dis"
# scp "$p_root/dec.zip" "lk3591@armitage.csec.rit.edu:/home/lk3591/transfer/dec"

t_f=$(date +%s.%N)
t_d=$(echo "$t_f - $t_i" | bc)
printf "Clean up: %.6f seconds\n" $t_d
t_i=$(date +%s.%N)
