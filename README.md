# RawByteClf

## Preprocessing

- Ensure all raw binaries and VirusTotal reports are in the correct directory on disk.
- Run `prepare` script
- Run `label` script
- Run `encode` script
- Run `split` script

## TODO

- Encode datasets during preprocessing.
- Craft train, test, and validation sets.
- Save predictions on each epoch (for evaluating accuracy of low-resource classes)

pip install "ray[tune]"==2.6.3
pip install bayesian-optimization
pip install hyperopt

pip install ninja

## Useful

Memory analysis:
	- mprof run python {SCRIPT.py}
	- mprof plot --output={PLOT.png}

Time analysis
	- python -m cProfile -o {STATS.pstats} {SCRIPT.py}
	- gprof2dot --colour-nodes-by-selftime -f pstats {STATS.pstats} | dot -Tpng -o {PLOT.png}

## Environment

environment_0.yml - environment from some time ago...
environment_1.yml - environment before integrating mamba into codebase
environment_2.yml - environment after integrating mamba into codebase

Create the environment and install cuda and torch with conda (cuda-nvcc is needed for mamba)

```
conda create -n RawByteClf python=3.10 pytorch=2.0.1 torchtext=0.15.2 pytorch-cuda=11.8 cuda-nvcc -c pytorch -c nvidia
```

Install everything else with pip (unless you want to wait 10 years for conda to resolve this). A very specific of ray tune is required for compatibility with transformers.

```
pip install \
transformers==4.35 \
datasets==2.14 \
tokenizers==0.14 \
accelerate==0.22 \
safetensors==0.3.1 \
boto3==1.28 \
psutil \
pandas \
scipy \
scikit-learn \
matplotlib \
requests \
evaluate==0.4 \
memory-profiler \
ninja==1.11.1.1 \
"ray[tune]"==2.6.3 \
bayesian-optimization \
hyperopt \
pynvml \
einops \
py7zr
```

Install mamba locally

```
pip install packaging
pip install /path/to/mamba/repository
```
pip install pycryptodome



find /path/to/directory -type d -path "*/None/10/*" -regex '.*/[0-9]+/None/10/.*'


cythonize -i src/learn/bytes_to_str_utf8.pyx



Weird BUG

  File "/home/lk3591/anaconda3/envs/RawByteClf2/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 1366, in __del__
    if hasattr(self, "_indices"):
  File "/home/lk3591/anaconda3/envs/RawByteClf2/lib/python3.10/site-packages/ray/_private/worker.py", line 1723, in sigterm_handler
    sys.exit(signum)
SystemExit: 15

https://github.com/huggingface/datasets/issues/3172 
