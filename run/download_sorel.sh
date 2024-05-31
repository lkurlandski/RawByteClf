#!/bin/bash -l

find /home/lk3591/Documents/datasets/Sorel/binaries -type f -print | awk -F/ '{print $NF}' | awk -F. '{print $1}' > /home/lk3591/Documents/code/RawByteClf/tmp/sorel_exclude.txt
for i in {0..499}; do sbatch run/download_sorel_labeled.sh $i; done
for i in {0..999}; do sbatch run/download_sorel_unlabeled.sh $i; done

