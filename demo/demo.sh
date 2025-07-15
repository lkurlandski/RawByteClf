#!/bin/bash -l

# Set up software environment
echo "Setting up software..."
echo "--------------------------------------------------------------------------------"

conda create -n MLLM python=3.10 pytorch=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate MLLM

pip install \
transformers==4.46.3 \
datasets==3.0.1 \
tokenizers==0.20.1 \
accelerate==1.0.1 \
safetensors==0.4.5
scikit-learn==1.3.0 \
evaluate==0.4.0

echo "Done setting up software!"
echo "--------------------------------------------------------------------------------"

echo "Running python tests..."
echo "--------------------------------------------------------------------------------"

python -m unittest discover tests

echo "Done running python tests!"
echo "--------------------------------------------------------------------------------"


