#!/bin/bash -l

#SBATCH --job-name=tokenize
#SBATCH --comment="train tokenizers and preprocess datasets for learning"

#SBATCH --account=admalware
#SBATCH --partition=debug

#SBATCH --output=./slurm/%x_%j.out

#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python main.py --vocab_size=1024 --algorithm="BPE" --block_size=256 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 12)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 16)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 20)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 24)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 28)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
# python main.py --vocab_size=$((2 ** 32)) --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"

