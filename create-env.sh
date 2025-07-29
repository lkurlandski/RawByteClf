#!/bin/bash

source ~/.bashrc

manager=$1
name=$2
remove=$3

if [[ "$manager" != "conda" && "$manager" != "venv" ]]; then
  echo "manager must be 'conda' or 'venv' not '$manager'."
  exit 1
fi

if [[ "$name" == "" ]]; then
  echo "name must be provided, not '$name'."
  exit 1
fi

if [[ "$manager" == "venv" ]]; then
  if [[ -d "$name" ]]; then
    echo "venv environment '$name' already exists."
    exit 1
  fi
else
  conda list --name "$name" > /dev/null 2>&1
  if [[ "$?" != "1" ]]; then
    echo "conda environment '$name' already exists."
    exit 1 
  fi
fi

if [[ "$remove" != "" && "$remove" != "0" && "$remove" != "1" ]]; then
  echo "remove must be '', '0', or '1'."
fi

if [[ "$manager" == "venv" ]]; then
  nvcc --version > /dev/null 2>&1
  if [[ "$?" == "1" ]]; then
    echo "nvcc not found and cannot be installed with venv."
  fi
  python3.10 -m venv "$name"
  source "./$name/bin/activate"
else
  conda create --yes --name "$name" python=3.10 cuda-nvcc -c nvidia
  conda activate "$name"
fi

pip install -r requirements.txt -r requirements-opt.txt 
pip install -r requirements-mamba.txt
bash check-deps.sh
source ~/.bashrc

if [[ "$remove" == "1" ]]; then
  if [[ "$manager" == "venv" ]]; then
    rm -rf "./$name"
  else
    conda env remove --yes --name "$name" > /dev/null
  fi 
fi
