#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

declare -a exit_codes

test="longformer"
bash run/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

test="hrrformer"
bash run/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

test="mamba"
bash run/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

echo "Exit codes: ${exit_codes[@]}"
