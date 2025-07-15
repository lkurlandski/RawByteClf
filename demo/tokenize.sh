for l in raw dis dec; do

    for a in bpe uni; do
        
        for v in 1024 4096 16384; do

            logfile="./logs/tok-$l-$a-$v.log"

            echo "Running $l-$a-$v..."

            # systemd-run --scope -p MemoryLimit=96G /home/lk3591/anaconda3/envs/RawByteClf/bin/python \
            python \
                src/tokenization/train.py \
                --lift_level "$l" \
                --algorithm "$a" \
                --vocab_size "$v" \
                --num_files=100 \
                --batch_size=1024 \
                --block_size=1024 \
                --max_token_length=16 # &> "$logfile"

            # echo "Done. Status: $?"

        done

    done

done
