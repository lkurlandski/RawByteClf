#!/bin/bash -l

find /shared/rc/admalware/Sorel/binaries/ -type f | awk -F/ '{print $NF}' | sed 's/\.[^.]*$//' > /home/lk3591/Documents/RawByteClf/tmp/exclude.txt

for i in {0..99}; do sbatch run/_download_sorel.sh $i; done

