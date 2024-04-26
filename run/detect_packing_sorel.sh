#!/bin/bash -l
#
#
#


run() {
    local filter="$1"

    echo "$filter: Creating..."
    directory="/tmp/sorel/$filter"
    mkdir $directory

    echo "$filter: Downloading..."
    aws s3 cp s3://sorel-20m/09-DEC-2020/binaries/ $directory --no-sign-request --exclude="*" --include="$filter*" --recursive > "logs/packing_aws_$filter.log" 2>&1 &
    aws_pid=$!
    sleep 15

    # Busy wait until the log file size remains unchanges,
    # indicating that the AWS process has completed but is hanging.
    initial_size=$(wc -l "logs/packing_aws_$filter.log")
    while :
    do
        current_size=$(wc -l %s "logs/packing_aws_$filter.log")
        if [ "$current_size" -eq "$initial_size" ]; then
            echo "Log file size remains unchanged. AWS process has completed."
            break
        fi
        initial_size=$current_size
        sleep 2
    done
    kill $aws_pid

    num_downloaded=$(ls $directory | wc -l)
    echo "$filter: Extracting $num_downloaded..."
    for file in "$directory"/*; do
        if [[ $file == *"."* ]]; then
            continue  # Skip files with a suffix
        fi
        zlib-flate -uncompress < "$file" > "${file}.exe"
        rm $file
    done

    echo "$filter: Analyzing..."
    diec --entropy -j $directory > "/home/lk3591/Documents/datasets/Sorel/diec/$filter.txt"

    sleep 1

    echo "$filter: Cleaning..."
    rm -rf $directory
    rm $directory/* > /dev/null 2>&1
    rmdir $directory > /dev/null 2>&1

    kill -9 $aws_pid  > /dev/null 2>&1
}

test() {
    local filter="$1"
    echo "$filter"
}

export -f run
export -f test

pkill aws > /dev/null 2>&1
pkill diec > /dev/null 2>&1
pkill zlib-flate > /dev/null 2>&1

rm /home/lk3591/Documents/datasets/Sorel/diec/* > /dev/null 2>&1
rm /home/lk3591/Documents/code/RawByteClf/logs/packing_aws_* > /dev/null 2>&1
rm -rf /tmp/sorel/* > /dev/null 2>&1

JOBS=1024
ULIMIT=16384

currentLimit=$(ulimit -n)
echo "Current ULimit: $currentLimit"
ulimit -n $ULIMIT
currentLimit=$(ulimit -n)
echo "New ulimit: $currentLimit"

# 
# for ((i=0; i<256; i++)); do
#     filter=$(printf "%02x" "$i")
#     echo "$filter"
# done | parallel --jobs $JOBS run

# Takes ~10 seconds to connect plus ~100 seconds per aws batch for ~2450 files (2.1GiB)
# for ((i=0; i<4096; i++)); do
#     filter=$(printf "%03x" "$i")
#     echo "$filter"
# done | parallel --jobs $JOBS run

# Takes ~10 seconds to connect plus ~10 seconds per aws batch for ~175 files (0.2GiB)
for ((i=0; i<65536; i++)); do
    filter=$(printf "%04x" "$i")
    echo "$filter"
done | parallel --jobs $JOBS run


# echo "sha,status" > /home/lk3591/Documents/datasets/Sorel/packing.csv


# for file in /home/lk3591/Documents/datasets/Sorel/diec/*.txt; do
#     sha=$(basename "$file" | cut -d '.' -f 1)
#     echo "$sha"
#     status=$(jq -r '.status' "$file")
#     echo "$status"
#     echo "$sha,$status" >> output.csv
# done

