#!/bin/bash -l
#
#
#

# Global variables must be exported to be available to parallel subprocesses
export OUTPUT="/home/lk3591/Documents/datasets/Sorel/diec"
export LOG="/home/lk3591/Documents/code/RawByteClf/logs"
export DOWNLOAD="/tmp/sorel"

export AWS_WAIT=5
export DIEC_TIMEOUT=5

run() {
    t_start_all=$(date +%s)

    local filter="$1"
    logfile="$LOG/packing_$filter.log"
    awslogfile="$LOG/packing_aws_$filter.log"
    directory="$DOWNLOAD/$filter"
    mkdir $directory

    #echo "logfile: $logfile"
    #echo "awslogile: $awslogfile"
    #echo "directory: $directory"
    # exit

    echo "Downloading..." >> $logfile 2>&1
    t_start=$(date +%s)
    aws s3 cp s3://sorel-20m/09-DEC-2020/binaries/ $directory --no-sign-request --exclude="*" --include="$filter*" --recursive > $awslogfile 2>&1 &
    aws_pid=$!

    # Wait for aws to connect.
    num=$(ls $directory | wc -l)
    echo "Starting AWS ($num)..." >> $logfile 2>&1
    while [ $num -eq 0 ]; do
        sleep $AWS_WAIT
	num=$(ls $directory | wc -l)
	echo "Starting AWS ($num)..." >> $logfile 2>&1
    done

    # Wait until aws hangs.
    sizePrev="-1"
    sizeCurr=$(wc -l < "$awslogfile")
    echo "Runnning AWS ($sizeCurr, $sizePrev)..." >> $logfile 2>&1
    while [ $sizeCurr -ne $sizePrev ]; do
        sleep $AWS_WAIT
        sizePrev=$sizeCurr
        sizeCurr=$(wc -l < "$awslogfile")
	echo "Runnning AWS ($sizeCurr, $sizePrev)..." >> $logfile 2>&1
    done
    sleep 5
    kill $aws_pid
    t_end=$(date +%s)
    t_download=$((t_end-t_start))

    num=$(ls $directory | wc -l)
    echo "Extracting ($num)..." >> $logfile 2>&1
    t_start=$(date +%s)
    for file in $directory/*; do
        if [[ $file == *"."* ]]; then
	    rm -rf $file
            continue
        fi
        zlib-flate -uncompress < "$file" > "${file}.exe"
        rm $file
    done
    t_end=$(date +%s)
    t_extract=$((t_end-t_start))

    num=$(ls $directory | wc -l)
    echo "Scanning ($num)..." >> $logfile 2>&1
    t_start=$(date +%s)
    for file in $directory/*; do
        if [[ $file != *".exe" ]]; then
            continue
        fi
	outfile=$(basename "$file")
	outfile="${outfile%.*}.txt"
	timeout $DIEC_TIMEOUT diec --recursivescan -j $file >> $OUTPUT/recursive/$outfile
	timeout $DIEC_TIMEOUT diec --deepscan -j $file >> $OUTPUT/deep/$outfile
	timeout $DIEC_TIMEOUT diec --heuristicscan -j $file >> $OUTPUT/heuristic/$outfile
    done
    t_end=$(date +%s)
    t_scan=$((t_end-t_start))

    num=$(ls $directory | wc -l)
    rm $directory/* > /dev/null 2>&1
    rmdir $directory > /dev/null 2>&1
    rm -rf $directory > /dev/null 2>&1
    kill -9 $aws_pid  > /dev/null 2>&1

    t_end_all=$(date +%s)
    t_all=$((t_end_all-t_start_all))
    echo "Process,Time,Throughput:" >> $logfile 2>&1
    echo "Downloading,$t_download,$((t_download / num))" >> $logfile 2>&1
    echo "Extracting,$t_extract,$((t_extract / num))" >> $logfile 2>&1
    echo "Scanning,$t_scan,$((t_downscan / num))" >> $logfile 2>&1
    echo "All,$t_all,$((t_all / num))" >> $logfile 2>&1

    echo "Finished $filter in $t_all seconds"
}

test() {
    local filter="$1"
    echo "$filter"
}

export -f run
export -f test

MODE=$1
JOBS=$2
ULIMIT=$((JOBS * 8))

if [ "$MODE" -eq 1 ]; then
    TOTAL=16
    FORMAT="%01x"
elif [ "$MODE" -eq 2 ]; then
    TOTAL=256
    FORMAT="%02x"
elif [ "$MODE" -eq 3 ]; then
    TOTAL=4096
    FORMAT="%03x"
elif [ "$MODE" -eq 4 ]; then
    TOTAL=65536
    FORMAT="%04x"
else
    echo "Error: Invalid MODE"
    exit 1
fi

currentLimit=$(ulimit -n)
if [ $currentLimit -lt $ULIMIT ]; then
    echo "CHANGING ULIMIT: $currentLimit"
    ulimit -n $ULIMIT
    currentLimit=$(ulimit -n)
fi

echo "MODE: $MODE"
echo "JOBS: $JOBS"
echo "FORMAT: $FORMAT"
echo "ULIMIT: $ULIMIT"

pkill aws > /dev/null 2>&1
pkill diec > /dev/null 2>&1
pkill zlib-flate > /dev/null 2>&1

echo "OUTPUT: $OUTPUT"
echo "LOG: $LOG"
echo "DOWNLOAD: $DOWNLOAD"

rm -rf $DOWNLOAD > /dev/null 2>&1
rm -rf $OUTPUT > /dev/null 2>&1
rm $LOG/packing_* > /dev/null 2>&1

mkdir $DOWNLOAD
mkdir $OUTPUT
mkdir $OUTPUT/recursive
mkdir $OUTPUT/deep
mkdir $OUTPUT/heuristic

for ((i=0; i<$TOTAL; i++)); do
    filter=$(printf $FORMAT "$i")
    echo "$filter"
done | parallel --jobs $JOBS run

# for file in /home/lk3591/Documents/datasets/Sorel/diec/*.txt; do
#     sha=$(basename "$file" | cut -d '.' -f 1)
#     echo "$sha"
#     status=$(jq -r '.status' "$file")
#     echo "$status"
#     echo "$sha,$status" >> output.csv
# done

