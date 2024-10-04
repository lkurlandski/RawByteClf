#!/bin/bash -l

#SBATCH --job-name=liftMini-Windows-5
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --time=00-02:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


RERUN="1"  # Use 1 to detect and eliminate problematic files on re-runs.

DATASET=$1
if [[ $DATASET == "" ]]; then
   echo "Must specify a dataset, DATASET: $DATASET"
   exit
fi

HH=$(printf "%02x" "$SLURM_ARRAY_TASK_ID")  # use 01 or 02
echo "HH: $HH"

# Establish some timing variables.
t_start=$(date +%s.%N)
TIME=$((2 * 3600))
if ! [[ "$TIME" =~ ^[0-9]+$ ]]; then
  echo "Error: Invalid TIME: $TIME"
  exit 1
fi
TIME_FOR_CLEANUP=900
if [ "$TIME" -le 0 ] || [ "$TIME_FOR_CLEANUP" -le 0 ] || [ "$TIME" -le "$TIME_FOR_CLEANUP" ]; then
  echo "Error: TIME and TIME_FOR_CLEANUP must be greater than 0 and TIME greater than TIME_FOR_CLEANUP."
  exit 1
fi
echo "TIME: $TIME"
echo "TIME_FOR_CLEANUP: $TIME_FOR_CLEANUP"


# Configuration for Ghidra's headless analyzer and timeout values for the scripts.
TIMEOUT_PER_FILE_ANALYSIS="300"
TIMEOUT_PER_FILE_REGIONING="60"
TIMEOUT_PER_FILE_DISASSEMBLY="60"
TIMEOUT_PER_FILE_DECOMPILATION="300"
TIMEOUT_PER_FUNC_DISASSEMBLY="30"
TIMEOUT_PER_FUNC_DECOMPILATION="60"
PROCESSOR="x86:LE:32:default"
LOADER="PeLoader"
MAXMEM=$(grep "MAXMEM=" ~/lib/ghidra_11.1.2_PUBLIC/support/analyzeHeadless | sed 's/.*MAXMEM=\(.*\)/\1/')
echo "TIMEOUT_PER_FILE_ANALYSIS: $TIMEOUT_PER_FILE_ANALYSIS"
echo "TIMEOUT_PER_FILE_REGIONING: $TIMEOUT_PER_FILE_REGIONING"
echo "TIMEOUT_PER_FILE_DISASSEMBLY: $TIMEOUT_PER_FILE_DISASSEMBLY"
echo "TIMEOUT_PER_FILE_DECOMPILATION: $TIMEOUT_PER_FILE_DECOMPILATION"
echo "TIMEOUT_PER_FUNC_DISASSEMBLY: $TIMEOUT_PER_FUNC_DISASSEMBLY"
echo "TIMEOUT_PER_FUNC_DECOMPILATION: $TIMEOUT_PER_FUNC_DECOMPILATION"
echo "PROCESSOR: $PROCESSOR"
echo "LOADER: $LOADER"
echo "MAXMEM: $MAXMEM"

# Determine which computer we're running on and set base paths accordingly.
#   P_FIN: long-term storage with possibly slow I/O
#   P_TMP: short-term storage with fast I/O
P_FIN="/shared/rc/admalware/$DATASET/ghidra"
P_TMP="/tmp"

if [[ ! -d "$P_FIN" ]]; then
  echo "Error: Directory P_FIN $P_FIN does not exist. Exiting."
  exit 1
fi
echo "P_FIN: $P_FIN"

if [[ ! -d "$P_TMP" ]]; then
  echo "Error: Directory P_TMP $P_TMP does not exist. Exiting."
  exit 1
fi
P_TMP="$P_TMP/lk3591/$DATASET"
echo "P_TMP: $P_TMP"

# Define and create FIN directories.
p_fin_arc="$P_FIN/archived"
p_fin_dis="$P_FIN/disassembled"
p_fin_dec="$P_FIN/decompiled"
p_fin_reg="$P_FIN/regions"
p_fin_log="$P_FIN/ghidraLogs"
p_fin_ghi="$P_FIN/ghidraLocation"
mkdir -p "$p_fin_arc"
mkdir -p "$p_fin_dis"
mkdir -p "$p_fin_dec"
mkdir -p "$p_fin_reg"
mkdir -p "$p_fin_log"
mkdir -p "$p_fin_ghi"
echo "p_fin_arc: $p_fin_arc"
echo "p_fin_dis: $p_fin_dis"
echo "p_fin_dec: $p_fin_dec"
echo "p_fin_reg: $p_fin_reg"
echo "p_fin_log: $p_fin_log"
echo "p_fin_ghi: $p_fin_ghi"

# Define and create TMP directories.
p_tmp_arc="$P_TMP/archived/$HH"
p_tmp_dis="$P_TMP/disassembled/$HH"
p_tmp_dec="$P_TMP/decompiled/$HH"
p_tmp_reg="$P_TMP/regions/$HH"
p_tmp_log="$P_TMP/ghidraLogs/$HH"
p_tmp_bin="$P_TMP/binaries/$HH"
p_tmp_ghi="$P_TMP/ghidraLocation/$HH"
rm -rf "$p_tmp_arc"
rm -rf "$p_tmp_dis"
rm -rf "$p_tmp_dec"
rm -rf "$p_tmp_reg"
rm -rf "$p_tmp_log"
rm -rf "$p_tmp_bin"
rm -rf "$p_tmp_ghi"
mkdir -p "$p_tmp_arc"
mkdir -p "$p_tmp_dis"
mkdir -p "$p_tmp_dec"
mkdir -p "$p_tmp_reg"
mkdir -p "$p_tmp_log"
mkdir -p "$p_tmp_bin"
mkdir -p "$p_tmp_ghi"
echo "p_tmp_arc: $p_tmp_arc"
echo "p_tmp_dis: $p_tmp_dis"
echo "p_tmp_dec: $p_tmp_dec"
echo "p_tmp_reg: $p_tmp_reg"
echo "p_tmp_log: $p_tmp_log"
echo "p_tmp_bin: $p_tmp_bin"
echo "p_tmp_ghi: $p_tmp_ghi"

t_setup=$(date +%s.%N)
t_d=$(echo "$t_setup - $t_start" | bc)
printf "Set up time: %.6f seconds\n" $t_d

# Copy and extract binaries into the temporary directory.
echo "Copying and extracting binaries."
cp "$p_fin_arc/$HH.zip" "$p_tmp_arc/$HH.zip"
unzip -q -j "$p_tmp_arc/$HH.zip" -d "$p_tmp_bin/"
rm "$p_tmp_arc/$HH.zip"

# Copy and extract the ghidra project into the temporary directory
if [ -f "$p_fin_ghi/$HH.zip" ]; then
  echo "Copying and extracting an existing ghidra project."
  cp "$p_fin_ghi/$HH.zip" "$p_tmp_ghi/$HH.zip"
  unzip -q "$p_tmp_ghi/$HH.zip" -d "$p_tmp_ghi/"
  rm "$p_tmp_ghi/$HH.zip"
fi

t_extract=$(date +%s.%N)
t_d=$(echo "$t_extract - $t_setup" | bc)
printf "Extraction time: %.6f seconds\n" $t_d


if [[ $RERUN -eq 1 ]]; then

# If a log file exists, then we're probably rerunning the script.
# Its likely that we're rerunning it because the most recently attempted file
# caused a SLURM timeout, so we identify that file and remove it from the queue.
if [[ -f "$p_fin_log/$HH.log" ]]; then

  p_log="$p_fin_log/$HH.log"

  if [[ -f "$p_log" ]] && grep --text -q "IMPORTING: file" "$p_log" ]]; then
    echo "Inferring fail_from p_log: $p_log"
    fail_line=$(grep --text "IMPORTING: file" "$p_log" | tail -n 1)
    fail_stem=$(echo "$fail_line" | sed -n 's|.*IMPORTING: file://.*/\(.*\)\.exe.*|\1|p')
    fail_file="$p_tmp_bin/$fail_stem.exe"
  else
    echo "Inferring fail_from p_tmp_bin: $p_tmp_bin"
    fail_file=$(ls "$p_tmp_bin" | sort | head -n 1)
    fail_file="$p_tmp_bin/$fail_file"
  fi
  echo "fail_file: $fail_file"
  rm "$fail_file"

  p_log=""
  fail_line=""
  fail_stem=""
  fail_file=""

fi

fi

counter=0
while true; do

############################################################
# Determine which files have been processed and skip them. #
############################################################

# Store files that have already successfully been disassembled.
# Search the final archive.
if [[ -f "$p_fin_dis/$HH.zip" ]]; then
  fs_fin_dis=$(unzip -Z1 "$p_fin_dis/$HH.zip" | sed 's/\.[^.]*$//')
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
if [[ -f "$p_fin_dec/$HH.zip" ]]; then
  fs_fin_dec=$(unzip -Z1 "$p_fin_dec/$HH.zip" | sed 's/\.[^.]*$//')
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

# Store files that have already successfully been regioned.
# Search the final archive.
if [[ -f "$p_fin_reg/$HH.jsonl" ]]; then
  fs_fin_reg=$(grep -o '"sha": "[^"]*"' "$p_fin_reg/$HH.jsonl" | awk -F'"' '{print $4}')
else
  fs_fin_reg=""
fi	
# Search the tmp directory.
if [[ -f "$p_tmp_reg/$HH.jsonl" ]]; then
  for stem in $(grep -o '"sha": "[^"]*"' "$p_tmp_reg/$HH.jsonl" | awk -F'"' '{print $4}'); do
    fs_fin_reg=$(printf "%s\n%s" "$fs_fin_reg" "$stem")
  done
fi
echo "Num regioned files: $(echo "$fs_fin_reg" | grep -c -v '^$')"

# Iterate over files in p_tmp_bin and remove if its already been processed.
for f in "$p_tmp_bin"/*; do
  s=$(basename "$f" | sed 's/\.[^.]*$//')
  in_fs_fin_dis=$(echo "$fs_fin_dis" | grep -w "$s")
  in_fs_fin_dec=$(echo "$fs_fin_dec" | grep -w "$s")
  in_fs_fin_reg=$(echo "$fs_fin_reg" | grep -w "$s")
  if [ -n "$in_fs_fin_dis" ] && [ -n "$in_fs_fin_dec" ] && [ "$in_fs_fin_reg"  ]; then
    echo "Skipping: $s"
    rm "$f"
  fi
done

############################################################
######### Run headless analysis and handle errors. #########
############################################################

cnt=$(find "$p_tmp_bin" -type f | wc -l)
siz=$(du -shc "$p_tmp_bin"/* | grep total | awk '{print $1}')
echo "Lifting $cnt files totaling $siz."

p_log="$p_tmp_log/$HH.$counter.log"
t_ghidra=$(date +%s.%N)
timeout=$(echo "$TIME - ($t_ghidra - $t_start) - $TIME_FOR_CLEANUP" | bc)
if (( $(echo "$timeout < 1" | bc -l) )); then
  echo "Not enough time to run analyzeHeadless, so moving to cleanup; timeout: $timeout"
  break
fi
echo "Running analyzeHeadless for $timeout seconds and logging to $p_log"

# Run Ghidra to disassemble and decompile the files.
timeout $timeout analyzeHeadless \
  "$p_tmp_ghi" \
  "$HH" \
  -overwrite \
  -recursive \
  -log "$p_log" \
  -processor $PROCESSOR \
  -loader $LOADER \
  -analysisTimeoutPerFile $TIMEOUT_PER_FILE_ANALYSIS \
  -import "$p_tmp_bin" \
  -postScript "ExtractExecutableRegions.java" "$p_tmp_reg/$HH.jsonl" $TIMEOUT_PER_FILE_REGIONING \
  -postScript "Disassembler.java" "$p_tmp_dis" $TIMEOUT_PER_FILE_DISASSEMBLY $TIMEOUT_PER_FUNC_DISASSEMBLY \
  -postScript "Decompiler.java" "$p_tmp_dec" $TIMEOUT_PER_FILE_DECOMPILATION $TIMEOUT_PER_FUNC_DECOMPILATION \
  &> "$p_log"

code=$?
echo "analyzeHeadless returned $code"
counter=$((counter+1))

# If 0, then analyzeHeadless has finished correctly.
if [ $code -eq 0 ]; then
  break
fi
# If 124, then analyzeHeadless timed out
if [ $code -eq 124 ]; then
  break
fi
# If 125, 126, 127, something unexpected happened with the timeout command that should be
# investigated. Nonetheless, we can still try and recover the analysis results.
if [ "$code" -eq 125 ] || [ "$code" -eq 126 ] || [ "$code" -eq 127 ] ; then
  break
fi
# If 137, then analyzeHeadless was sent a hard kill signal, probably due to a OOM.
# Otherwise, analyzeHeadless returned with a non-zero exit code, which we handle the same way.
# We locate a likly "fail file" and remove it from the analysis.
if [[ -f "$p_log" ]] && grep --text -q "IMPORTING: file" "$p_log" ]]; then
  echo "Inferring fail_from p_log: $p_log"
  fail_line=$(grep --text "IMPORTING: file" "$p_log" | tail -n 1)
  fail_stem=$(echo "$fail_line" | sed -n 's|.*IMPORTING: file://.*/\(.*\)\.exe.*|\1|p')
  fail_file="$p_tmp_bin/$fail_stem.exe"
else
  echo "Inferring fail_from p_tmp_bin: $p_tmp_bin"
  fail_file=$(ls "$p_tmp_bin" | sort | head -n 1)
  fail_file="$p_tmp_bin/$fail_file"
fi
echo "fail_file: $fail_file"
rm "$fail_file"


done

# Print time take during lifting process.
t_cleanup=$(date +%s.%N)
t_d=$(echo "$t_cleanup - $t_extract" | bc)
printf "Ghidra time: %.6f seconds\n" $t_d

# Print number of files disassembled and decompiled.

if [ "$(find "$p_tmp_dis" -mindepth 1 -print -quit)" ]; then
  cnt=$(find "$p_tmp_dis" -type f | wc -l)
  siz=$(du -shc "$p_tmp_dis"/* | grep total | awk '{print $1}')
else
  cnt=0
  siz=0
fi
echo "Disassembled $cnt files totaling $siz."

if [ "$(find "$p_tmp_dec" -mindepth 1 -print -quit)" ]; then
  cnt=$(find "$p_tmp_dec" -type f | wc -l)
  siz=$(du -shc "$p_tmp_dec"/* | grep total | awk '{print $1}')
else
  cnt=0
  siz=0
fi
echo "Decompiled $cnt files totaling $siz."

if [ -f "$p_tmp_reg/$HH.jsonl" ]; then
  cnt=$(wc -l "$p_tmp_reg/$HH.jsonl" | awk '{print $1}')
  siz=$(du -shc "$p_tmp_reg/$HH.jsonl" | grep total | awk '{print $1}')
else
  cnt=0
  siz=0
fi
echo "Regioned $cnt files totaling $siz."

# Compress the lifted files and move to final storage.
# If the archive is already present, zip will simply add new files to the archive.
zip -9 -r -j -q -u "$p_fin_dis/$HH.zip" "$p_tmp_dis"
zip -9 -r -j -q -u "$p_fin_dec/$HH.zip" "$p_tmp_dec"

# Append the tmp log to the fin log file.
# Append the tmp reg to the fin reg file.
for f in "$p_tmp_log"/*; do
    cat $f >> "$p_fin_log/$HH.log"
done
cat "$p_tmp_reg/$HH.jsonl" >> "$p_fin_reg/$HH.jsonl"

# Update the existing ghidra location cache thing.
pushd "$p_tmp_ghi"
zip -9 -r -q -u "$p_fin_ghi/$HH.zip" "."
popd

t_transfer=$(date +%s.%N)
t_d=$(echo "$t_transfer - $t_cleanup" | bc)
printf "Transfer time: %.6f seconds\n" $t_d
