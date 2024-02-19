#!/bin/bash -l

# USAGE
# rm -rf tmp
# rm -rf "/tmp/MOVE_CONTENTS_INTO_NEW_DIRECTORY"
# mkdir tmp
# mkdir tmp/tmp1
# touch tmp/tmp1/tmp.txt
# mkdir tmp/tmp2
# touch tmp/tmp2/tmp.txt
# move_contents_into_new_directory "/home/lk3591/tmp" "tmp"

move_contents_into_new_directory() {
	local src=$1
	local dest=$2

    # Make a temporary directory, move everything there
    tmp="./tmp/MOVE_CONTENTS_INTO_NEW_DIRECTORY"
    mkdir -p $tmp
    mv $src/* $tmp
	# Make the new destination, move everything there
	mkdir $src/$dest
	mv $tmp/* $src/$dest

    # remove the temporary directory
	rmdir $tmp
}

move_contents_into_new_directory /home/lk3591/Documents/code/RawByteClf/output/scaling/mamba/512/clf "None"
move_contents_into_new_directory /home/lk3591/Documents/code/RawByteClf/output/scaling/mamba/512/clm "3000000"
