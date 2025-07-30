#!/bin/bash

echo "Checking pytorch..."
python -c "import torch" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Success."
else
  echo "Failed."
fi

echo "Checking pytorch-CUDA..."
python -c "import torch; assert torch.cuda.is_available()" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Success."
else
  echo "Failed."
fi

echo "Checking NVIDIA's CUDA Compiler (NVCC)..."
nvcc --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Success."
else
  echo "Failed."
fi

echo "Checking causal_conv1d..."
python -c "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Success."
else
  echo "Failed."
fi

echo "Checking mamba_ssm..."
python -c "from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Success."
else
  echo "Failed."
fi
