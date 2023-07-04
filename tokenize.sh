source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
python main.py --vocab_size=$((2 ** 8)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 12)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 16)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 20)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 24)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 28)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"
python main.py --vocab_size=$((2 ** 32)) --max_size="-1" --model="go-fuck-yourself" --output_dir="go-fuck-yourself"

