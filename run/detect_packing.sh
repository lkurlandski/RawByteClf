#!/bin/bash -l

#SBATCH --job-name=download
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


S="src/data/detect_packing.py"
O="/home/lk3591/Documents/datasets/Sorel/"
N=32
L=$((N - 1))
P=$((N - 2))


echo "Parameters:"
echo $S
echo $O
echo $N
echo $L
echo $P
echo "--------------------------------------------------"


echo "Preparation..."
python -u $S --output_root=$O --prepare --num_shards=$N
echo "--------------------------------------------------"


echo "Processing..."
for ((i=0; i < $P; i++)); do
    echo "Shard $i --> logs/detect_packing_$i.out"
    python -u $S --output_root=$O --shard_idx=$i --num_shards=$N --errors=2 > "logs/detect_packing_$i.out" 2>&1 &
done
echo "Shard $L --> logs/detect_packing_$L.out"
python -u $S --output_root=$O --shard_idx=$L --num_shards=$N --errors=2 > "logs/detect_packing_$L.out"
echo "--------------------------------------------------"


echo "Finishing..."
python -u $S --output_root=$O --finish --num_shards=$N
echo "--------------------------------------------------"
