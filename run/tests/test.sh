#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

declare -a exit_codes

module unload blindfold  # Let's mamba use gcc correctly on RC

test="mamba"
bash run/tests/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

test="longformer"
bash run/tests/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

test="hrrformer"
bash run/tests/test_$test.sh
exit_codes+=($?)
if [ $? -ne 0 ]; then
    echo "Error: $test $?"
fi

echo "Exit codes: ${exit_codes[@]}"
