import os
from pathlib import Path
import time


BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-HH:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/tokenization.py \\
--algorithm=ALGORITHM \\
--vocab_size=VOCAB_SIZE \\
--num_files=2000 \\
--block_size=1024 \\
--batch_size=1024 \\
--max_token_length=16
"""


OUTPUT = Path(os.path.realpath(__file__)).parent
ALGORITHMS = ["BPE", "Unigram"]
VOCAB_SIZES = [
    512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144,
]


outfiles = []
for alg in ALGORITHMS:
    for vs in VOCAB_SIZES:
        jobname = f"tok_{alg}_{vs}"
        text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("ALGORITHM", alg) \
            .replace("VOCAB_SIZE", str(vs)) \
            .replace("HH", "01")

        outfile = OUTPUT / f"{jobname}.sh"
        outfiles.append(outfile)
        with open(outfile, "w") as fp:
            fp.write(text)

outfiles = sorted(outfiles, key=lambda x: x.stem.split("_")[1])
with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in outfiles:
        fp.write(f"bash {outfile}\n")
