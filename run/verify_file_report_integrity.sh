#!/bin/bash -l

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


directories=(
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20140617/reports
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20190212/reports
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20200405/reports
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_Linux_20160715/reports
  /home/lk3591/Documents/datasets/MalwareBazaar/elf/reports/
)
for directory in "${directories[@]}"; do
  echo "Verifying the integrity of the reports in $directory"
  python \
  src/data/verify_file_report_integrity.py \
  --dir_reports="$directory"
  echo "--------------------------------------------------"
done


platform=ELF
directories=(
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20140617/binaries
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20190212/binaries
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_ELF_20200405/binaries
  /home/lk3591/Documents/datasets/VirusShare/VirusShare_Linux_20160715/binaries
  /home/lk3591/Documents/datasets/MalwareBazaar/elf/binaries/
)
for directory in "${directories[@]}"; do
  echo "Verifying the integrity of the binaries in $directory"
  python \
  src/data/verify_file_report_integrity.py \
  --dir_binaries="$directory" \
  --platform="$platform"
  echo "--------------------------------------------------"
done




