
run() {
    local filter="$1"

    directory="/tmp/sorel/$filter"
    mkdir -p $directory

    echo "$filter: Downloading..."
    aws s3 cp s3://sorel-20m/09-DEC-2020/binaries/ $directory --no-sign-request --exclude="*" --include="$filter*" --recursive > "logs/packing_aws_$filter.log" 2>&1 &
    aws_pid=$!

    sleep 1

    while ! grep -q "Completed" "logs/packing_aws_$filter.log"; do
        # Check if the aws process is still running
        if ! ps -p $aws_pid > /dev/null; then
            echo "The aws s3 cp command terminated unexpectedly."
            exit 1
        fi
        # Sleep for a short duration before checking again
        sleep 1
    done
    kill $aws_pid

    sleep 1

    echo "$filter: Extracting..."
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
    rm "$directory/*" > /dev/null 2>&1
    rmdir $directory > /dev/null 2>&1

    kill -9 $aws_pid  > /dev/null 2>&1
}

export -f run


for ((i=0; i<4096; i++)); do
    filter=$(printf "%03x" "$i")
    echo "$filter"
done | parallel --max-procs 252 run

