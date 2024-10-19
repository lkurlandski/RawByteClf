#!/bin/bash -l

#SBATCH --job-name=debug
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2

# Run shortened versions of the experiments.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
# conda activate RawByteClf2
# module unload blindfold

python ./run/esp/create.py --action=dbg --armitage

runpath="./run/esp/sbatch/dbg"
logpath="./logs/esp-dbg"
outpath="./output/esp-dbg"

rm -rf "$outpath"
rm -rf "$logpath"
mkdir -p "$logpath"

get_eighth_component() {
    echo "$(basename $1 .sh)" | cut -d'-' -f8
}

get_ninth_component() {
    echo "$(basename $1 .sh)" | cut -d'-' -f9
}

nop_files=()
non_nop_files=()

for file in "$runpath/"*; do
    eighth_component=$(get_eighth_component $file)
    if [[ "$eighth_component" == "nop" ]]; then
        nop_files+=("$file")
    else
        non_nop_files+=("$file")
    fi
done

sorted_files=("${nop_files[@]}" "${non_nop_files[@]}")

for file in "${sorted_files[@]}"; do

    stem=$(basename -s .sh $file)
    logfile="$logpath/$stem.log"

    # task=$(get_ninth_component $file)
    # if [[ "$task" == "clm" || "$task" == "mlm" ]]; then
    #     devices="0"
    # elif [[ "$task" == "det" || "$task" == "fam" || "$task" == "beh" ]]; then
    #     devices="0"
    # else
    #     echo "task: $task"
    #     exit 1
    # fi

    devices="0"
    echo -n "Running $stem..."
    CUDA_VISIBLE_DEVICES="$devices" bash "$file" &> "$logfile"
    echo "Done. Status: $?."

done
