#!/bin/bash -l

#SBATCH --job-name=unittest
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf2


python -m unittest discover -s tests
