
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
conda activate RawByteClf2
module unload blindfold

# rm -rf ./output/esp-tmp
# rm -rf ./logs/test
# mkdir -p ./logs/test

get_eighth_component() {
    echo "$(basename $1 .sh)" | cut -d'-' -f8
}

get_ninth_component() {
    echo "$(basename $1 .sh)" | cut -d'-' -f9
}

mean_samples_per_second() {
    local log_file=$1
    local search_string=$2

    grep -o "'${search_string}': [0-9]*\.[0-9]*" "$log_file" | \
    sed "s/'${search_string}': //" | \
    awk '{sum += $1; count += 1} END {if (count > 0) print sum / count; else print 0}'
}


nop_files=()
non_nop_files=()

for file in "./run/esp/test/"*; do
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
    logfile="./logs/test/$stem.log"
    task=$(get_ninth_component $file)

    if [[ "$task" == "clm" || "$task" == "mlm" ]]; then
        devices="0"
    elif [[ "$task" == "det" || "$task" == "fam" || "$task" == "beh" ]]; then
        devices="0"
    else
        echo "task: $task"
        exit 1
    fi

    echo -n "Running $stem..."
    CUDA_VISIBLE_DEVICES="$devices" bash "$file" &> "$logfile"
    echo "Done. Status: $?."

done

for file in "${sorted_files[@]}"; do

    stem=$(basename -s .sh $file)
    logfile="./logs/test/$stem.log"
    task=$(get_ninth_component $file)

    eval_samples_per_second=$(mean_samples_per_second $logfile "eval_sample_per_second")
    train_samples_per_second=$(mean_samples_per_second $logfile "train_sample_per_second")
    echo "$stem: eval_samples_per_second: $eval_samples_per_second, train_samples_per_second: $train_samples_per_second"

done
