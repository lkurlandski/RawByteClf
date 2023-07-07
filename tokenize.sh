#!/bin/bash -l

#SBATCH --job-name=tokenize
#SBATCH --comment="Train byte-level tokenizers on 1000 files"

#SBATCH --account=admalware
#SBATCH --partition=tier3

#SBATCH --output=./slurm/%x_%j.out

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python main.py --vocab_size=524288 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=131072 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=32768 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=8192 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=2048 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=512 --max_token_length=128 --algorithm="SentencePieceUnigram" --n_tok=10 --tok_batch_size=4096 --block_size=4096 --do_preprocess=false --overwrite_saved_tokenizer=true --use_saved_tokenizer=false --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
