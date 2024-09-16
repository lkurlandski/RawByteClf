#

log_slurm=$1
echo "log_slurm: $log_slurm"
HHH=$(grep "HHH" "$log_slurm" | awk '{print $2}')
echo "HHH: $HHH"
log_ghidra="$(grep "p_fin_log" "$log_slurm" | awk '{print $2}')/$HHH.log"
echo "log_ghidra: $log_ghidra"

echo "analyzeHeadless (exit)"
grep "analyzeHeadless returned" $log_slurm | tail -n 1 | awk '{print $3}'
echo "----------------------------------------------------------------"

echo "analyzeHeadless (crash)"
if grep -q "fail_file" "$log_slurm"; then
  grep "fail_file" "$log_slurm" | while read -r line; do
    exit_code=$(grep -B1 "$line" "$log_slurm" | head -n 1 | awk '{print $3}')
    sha=$(echo "$line" | awk '{print $2}' | xargs -n 1 basename | cut -f 1 -d '.')
    echo "$exit_code $sha"
  done
fi
echo "----------------------------------------------------------------"

echo "analyzeHeadless (timeout)"
grep --text "Analysis timed out" "$log_ghidra" | grep -oP '/[0-9a-f]{64}(?=\.exe)' | sed 's|/||'
echo "----------------------------------------------------------------"

echo "script (timeout):"
grep --text "run: finished (timeout)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)'
echo "----------------------------------------------------------------"

echo "script (crash):"
grep --text "run: finished (crash)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)'
echo "----------------------------------------------------------------"

# echo "script (success):"
# grep --text "run: finished (success)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)'
# echo "----------------------------------------------------------------"
