#

VERBOSE=$2

log_slurm=$1
HHH=$(grep "HHH" "$log_slurm" | awk '{print $2}')
log_ghidra="$(grep "p_fin_log" "$log_slurm" | awk '{print $2}')/$HHH.log"

echo "HHH: $HHH"

if [[ ! -f "$log_ghidra" ]]; then
  echo "INCOMPLETE"
  exit
fi


files=$(grep "fail_file" "$log_slurm" | grep -oP '/[0-9a-f]{64}(?=\.exe)' | sed 's|/||')
echo "analyzeHeadless (crash): $(echo "$files" | sed '/^$/d' | wc -l)"
if [[ $VERBOSE == "1"  ]]; then
  echo "$files"
  echo "----------------------------------------------------------------"
fi

files=$(grep --text "Analysis timed out" "$log_ghidra" | grep -oP '/[0-9a-f]{64}(?=\.exe)' | sed 's|/||')
echo "analyzeHeadless (timeout): $(echo "$files" | sed '/^$/d' | wc -l)"
if [[ $VERBOSE == "1"  ]]; then
  echo "$files"
  echo "----------------------------------------------------------------"
fi

files=$(grep --text "run: finished (timeout)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)')
echo "script (timeout): $(echo "$files" | sed '/^$/d' | wc -l)"
if [[ $VERBOSE == "1"  ]]; then
  echo "$files"
  echo "----------------------------------------------------------------"
fi

files=$(grep --text "run: finished (crash)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)')
echo "script (crash): $(echo "$files" | sed '/^$/d' | wc -l)"
if [[ $VERBOSE == "1"  ]]; then
  echo "$files"
  echo "----------------------------------------------------------------"
fi

files=$(grep --text "run: finished (success)" "$log_ghidra" | grep -oP '<\K[0-9a-f]{64}(?=>)')
echo "script (success): $(echo "$files" | sed '/^$/d' | wc -l)"
if [[ $VERBOSE == "1"  ]]; then
  echo "$files"
  echo "----------------------------------------------------------------"
fi
