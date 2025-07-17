# RawByteClf

Repository for the paper "Beyond Raw Bytes: Towards Large Malware Language Models".

## Demo

To verify the functionality of our codebase, we include a small demonstration.

### Prerequisites

This demo requires [Docker](https://docs.docker.com/engine/install/) (tested with 28.0.4) and [Zstandard](https://github.com/facebook/zstd) (tested with 1.5.5). The shell commands are for a Linux machine running CentOS and will need to be adjusted slightly if using a different OS.

### Setup

To get the docker image, combine the files from our anonymous [Drive](https://drive.google.com/drive/folders/13cZ8Jd0jIkuWevUHbG-ilhs-OmWIfYwH?usp=sharing):
```
cat demo.tar.zst.part-* > demo.tar.zst
```

Set up the docker image:
```
zstd -dc demo.tar.zst | sudo docker load
```

Check the docker image:
```
sudo docker run --rm demo:latest
```

### Preparation

Start up the docker daemon:
```
sudo docker run -d --name demo_state demo:latest tail -f /dev/null
```

Generate dataset caches:
```
sudo docker exec demo_state bash demo/gencaches.sh
```

Generate experiments scripts:
```
sudo docker exec demo_state python demo/create.py
```

Create a directory to inspect outputs:
```
mkdir ./workdir
```

### Exploration

Copy preprocessed data archives:
```
sudo docker cp demo_state:/home/appuser/app/data/bod/nop/0.zip ./workdir/nop.zip
sudo docker cp demo_state:/home/appuser/app/data/bod/raw/0.zip ./workdir/raw.zip
sudo docker cp demo_state:/home/appuser/app/data/bod/dis/0.zip ./workdir/dis.zip
sudo docker cp demo_state:/home/appuser/app/data/bod/dec/0.zip ./workdir/dec.zip
```

Extract preprocessed samples:
```
unzip -p ./workdir/nop.zip 000782e505d3927adcbdc5701f9e3f08c14006a1547f552a03ce1e477e54c82b.exe > ./workdir/0007.exe
unzip -p ./workdir/raw.zip 000782e505d3927adcbdc5701f9e3f08c14006a1547f552a03ce1e477e54c82b.exe > ./workdir/0007.bytes
unzip -p ./workdir/dis.zip 000782e505d3927adcbdc5701f9e3f08c14006a1547f552a03ce1e477e54c82b.asm > ./workdir/0007.asm
unzip -p ./workdir/dec.zip 000782e505d3927adcbdc5701f9e3f08c14006a1547f552a03ce1e477e54c82b.c > ./workdir/0007.c
```

Examine preprocessed samples:
```
file ./workdir/0007.exe
file ./workdir/0007.bytes
file ./workdir/0007.asm
file ./workdir/0007.c
```

### Tokenization

Train tokenizers:
```
sudo docker exec demo_state bash demo/tokenize.sh
```

Copy pretrained tokenizers:
```
sudo docker cp demo_state:/home/appuser/app/output/tokenizers ./workdir/
```

Inspect tokenizer vocabularies:
```
head -n 1128 ./workdir/tokenizers/raw/bpe/1024/100/tokenizer.json | tail -n 1033
head -n 1127 ./workdir/tokenizers/dis/bpe/1024/100/tokenizer.json | tail -n 1033
head -n 1127 ./workdir/tokenizers/dec/bpe/1024/100/tokenizer.json | tail -n 1033
```

### Redundancy

Assess redundant samples:
```
sudo docker exec demo_state python demo/redundancy.py
```

### Pretraining

Pretrain unidirectional HRRFormer on EXE inputs for causal language modeling:
```
sudo docker exec demo_state bash demo/sbatch/hrr-tn-un-000512-raw-bpe-16384-nop-clm-dec-0.sh
```

Pretrain bidirectional HRRFormer on DIS inputs for masked language modeling:
```
sudo docker exec demo_state bash demo/sbatch/hrr-tn-bi-000512-dis-bpe-16384-nop-mlm-dec-0.sh
```

Pretrain unidirectional Mamba on DEC inputs for causal language modeling:
```
sudo docker exec demo_state bash demo/sbatch/mam-tn-un-000512-dec-bpe-16384-nop-clm-dec-0.sh
```

### Classification

Train/finetune unidirectional HRRFormer on EXE inputs for malware detection:
```
sudo docker exec demo_state bash demo/sbatch/hrr-tn-un-000512-raw-bpe-16384-nop-det-dec-0.sh
sudo docker exec demo_state bash demo/sbatch/hrr-tn-un-000512-raw-bpe-16384-clm-det-dec-0.sh
```

Train/finetune bidirectional HRRFormer on DIS inputs for family classification:
```
sudo docker exec demo_state bash demo/sbatch/hrr-tn-bi-000512-dis-bpe-16384-nop-fam-dec-0.sh
sudo docker exec demo_state bash demo/sbatch/hrr-tn-bi-000512-dis-bpe-16384-mlm-fam-dec-0.sh
```

Train/finetune unidirectional Mamba on DEC inputs for behavioral tagging:
```
sudo docker exec demo_state bash demo/sbatch/mam-tn-un-000512-dec-bpe-16384-nop-beh-dec-0.sh
sudo docker exec demo_state bash demo/sbatch/mam-tn-un-000512-dec-bpe-16384-clm-beh-dec-0.sh
```

Train the byte-based MalConvGCT on RAW inputs for all classification tasks:
```
sudo docker exec demo_state bash demo/sbatch/mal-tn-bi-1048576-nop-wdl-00256-nop-det-dec-0.sh
sudo docker exec demo_state bash demo/sbatch/mal-tn-bi-1048576-nop-wdl-00256-nop-fam-dec-0.sh
sudo docker exec demo_state bash demo/sbatch/mal-tn-bi-1048576-nop-wdl-00256-nop-beh-dec-0.sh
```

### Cleanup

Kill the daemon:
```
sudo docker stop demo_state && sudo docker rm demo_state
```

Remove the output data:
```
sudo rm -rf ./workdir
```

### Appendix

To clean up docker artifacts:
```
sudo docker container prune -f
sudo docker image prune -f
sudo docker system prune -a --volumes -f
```

To change Docker's storage device:
```
sudo systemctl stop docker
sudo rsync -aP /var/lib/docker/ /home/docker-data
sudo sh -c 'printf "{\n\"data-root\": \"/home/docker-data\"\n}" > /etc/docker/daemon.json'
sudo systemctl start docker
sudo docker info | grep "Docker Root Dir"
```

To create the docker image:
```
sudo docker build -t demo:latest .
```

To save the docker image:
```
sudo docker save demo:latest | zstd --ultra -T0 -22 -o demo.tar.zst
```

To split the docker image:
```
split -b 1G demo.tar.zst demo.tar.zst.part-
```

Examine data in the image:
```
sudo docker cp demo_state:/home/appuser/app/[OUT] ./workdir/[OUT]
```

## Environment

To run locally an environment with the relevant dependencies must be created.

Create conda environment:
```
conda env create -f environment.yml
conda activate LMLM
```

Install core dependencies:
```
pip install -r requirements.txt
```

Install optional dependencies:
```
pip install -r requirementsComplete.txt
```

## Documentation

The system is a bit complicated at the moment. We are working on simplifying it.

### Environment Variables

All environment variables for this system are prefaced with "LMLM_".

- `LMLM_ENABLE_UNITTEST_LOGGING`
- `LMLM_ENABLE_UNITTEST_WARNING`
- `LMLM_SYNC_ENSEMBLE_MATERIALS`
- `LMLM_CAN_PRECOPY_ZIPFILES`
- `LMLM_GET_MATERIALS_ESP_FAM_MIN_FREQ`
- `LMLM_GET_MATERIALS_ESP_FAM_MAX_IMBALANCE_RATIO`
- `LMLM_GET_MATERIALS_ESP_FAM_TOP_K`
- `LMLM_GET_MATERIALS_ESP_BEH_MIN_FREQ`
- `LMLM_GET_MATERIALS_ESP_BEH_MAX_IMBALANCE_RATIO`
- `LMLM_GET_MATERIALS_ESP_BEH_TOP_K`
- `LMLM_GET_MATERIALS_ESP_CLM_VL_SIZE`
- `LMLM_GET_MATERIALS_ESP_MLM_VL_SIZE`
- `LMLM_GET_MATERIALS_ESP_LM_VL_SIZE`
- `LMLM_EXIT_AFTER_UNIGRAM_COMPUTATION`

### System

```
├── build:            home for Cython kernels.
├── cache:            serialized data structures for fast retrieval.
├── config:           fundamental configurations for the project.
├── data:             processed data in zip-archives (long story).
├── demo:             materials for the demonstration.
├── environment:      documentation of software dependencies.
├── export:           figures and files to transfer from servers.
├── ghidra_scripts:   custom scripts for disassembly and decompilation.
├── logs:             logfiles for SLURM batch processing.
├── notebooks:        jupyter notebooks for data analysis.
├── output:           saved models and training logs.
├── run:              scripts for executing batch experiments.
├── scripts:          quick useful scripts.
├── src:              source code for project.
├── tests:            tests for the source code.
├── tmp:              temporary directory.
```

### Usage

We are working on documenting how to use the system more thoroughly.

## Citation

```
@inproceedings{
  authors={Anomynous Author(s)},
  title={Beyond Raw Bytes: Towards Large Malware Language Models},
  booktitle={The Network and Distributed System Security (NDSS) Symposium},
  year={2026},
}
```
