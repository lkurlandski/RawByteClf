#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

while :
do
  bash ./run/preprocessing.sh $1
  if [ $? -eq 0 ]; then
    exit 0
  fi
done

