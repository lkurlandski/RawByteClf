"""
Benchmark the speed of using a tokenizer with different batch sizes and max lengths.
"""

from argparse import ArgumentParser
import os
from statistics import mean, median
import sys
import time

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from tqdm import tqdm

from src.utils import batched
from src.data.cfg import DATASET_TO_FILES
from src.learn.preprocessing import bytes_to_str_ascii
from src.learn.tokenization import get_tokenizer


# parser = ArgumentParser()
# parser.add_argument("--batch_size", type=int)
# parser.add_argument("--num_files", type=int)
# parser.add_argument("--max_length", type=int)
# parser.add_argument("--num_trials", type=int)
# args = parser.parse_args()


outfile = "./tmp/tokenizer_benchmarks.csv"
num_files = 256
num_trials = 5
batch_sizes = [1, 4, 16, 64, 256]
max_lengths = [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]


tokenizer = get_tokenizer(False, 8, "SentencePieceBPE", 16390)
files = list(DATASET_TO_FILES["binaries"]["sorel_pe"]())[:num_files]


with open(outfile, "w") as fp:
    fp.write("num_files,num_trials,max_length,batch_size,time\n")


for max_length in tqdm(max_lengths, desc="max_lengths", leave=True):
    data = (open(f, "rb").read(max_length) for f in files)
    data = (bytes_to_str_ascii(bs) for bs in tqdm(data, total=num_files, desc="Reading..."))
    for batch_size in batch_sizes:
        batches = list(batched(data, batch_size))
        times = [None for _ in range(num_trials)]
        for i in tqdm(range(num_trials), total=num_trials, leave=False):
            t_i = time.time()
            for batch in batches:
                tokenizer(batch)
            t_f = time.time()
            times[i] = t_f - t_i

        with open(outfile, "a") as fp:
            fp.write(f"{num_files},{num_trials},{max_length},{batch_size},{mean(times)}\n")
