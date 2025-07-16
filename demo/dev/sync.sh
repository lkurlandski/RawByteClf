#!/bin/bash

# Get the shalist

root="/home/lk3591/Documents/code/RawByteClf"

ll="dis"
for dn in "ass" "bod" "sor" "win"; do
    for f in $root/demo/data/$dn/$ll/*.zip; do
        for l in $(unzip -Z1 $f); do
            echo "${l%.*}"
        done
    done
done

exit 0

# Download the digests

# root="/home/lk3591/Documents/code/RawByteClf"

# for dn in "ass" "bod" "sor" "win"; do
#     for ll in "raw" "dis" "dec"; do
#         echo -n "Downloading $dn/$ll/digests.json ... "
#         scp lk3591@sporcsubmit.rc.rit.edu:$root/data/$dn/$ll/digests.json $root/demo/data/$dn/$ll/digests.json > /dev/null 2>&1
#         echo "Status: $?"
#     done
# done

# exit 0

# Download the dataset from RC

# root="/home/lk3591/Documents/code/RawByteClf"

# dn="ass"
# for ll in "nop" "raw" "dis" "dec"; do 
#     for i in "00" "01"; do
#         echo -n "Downloading $dn/$ll/$i.zip ... "
#         scp lk3591@sporcsubmit.rc.rit.edu:$root/data/$dn/$ll/$i.zip $root/demo/data/$dn/$ll/$i.zip > /dev/null 2>&1
#         echo "Status: $?"
#         chmod 777 $root/demo/data/$dn/$ll/$i.zip
#     done
# done

# dn="bod"
# for ll in "nop" "raw" "dis" "dec"; do
#     for i in "0" "1"; do
#         echo -n "Downloading $dn/$ll/$i.zip ... "
#         scp lk3591@sporcsubmit.rc.rit.edu:$root/data/$dn/$ll/$i.zip $root/demo/data/$dn/$ll/$i.zip > /dev/null 2>&1
#         echo "Status: $?"
#         chmod 777 $root/demo/data/$dn/$ll/$i.zip
#     done
# done

# dn="sor"
# for ll in "nop" "raw" "dis" "dec"; do 
#     for i in "000" "001"; do
#         echo -n "Downloading $dn/$ll/$i.zip ... "
#         scp lk3591@sporcsubmit.rc.rit.edu:$root/data/$dn/$ll/$i.zip $root/demo/data/$dn/$ll/$i.zip > /dev/null 2>&1
#         echo "Status: $?"
#         chmod 777 $root/demo/data/$dn/$ll/$i.zip
#     done
# done

# dn="win"
# for ll in "nop" "raw" "dis" "dec"; do 
#     for i in "00" "01"; do
#         echo -n "Downloading $dn/$ll/$i.zip ... "
#         scp lk3591@sporcsubmit.rc.rit.edu:$root/data/$dn/$ll/$i.zip $root/demo/data/$dn/$ll/$i.zip > /dev/null 2>&1
#         echo "Status: $?"
#         chmod 777 $root/demo/data/$dn/$ll/$i.zip
#     done
# done

# exit 0

# Selectonly the first files from each archive

root="/home/lk3591/Documents/code/RawByteClf"

for dn in "ass" "bod" "sor" "win"; do
    for ll in "nop" "raw" "dis" "dec"; do
        for f in $root/demo/data/$dn/$ll/*.zip; do
            echo -n "$f: "
            echo -n "$(unzip -Z1 $f | wc -l) --> "
            zipinfo -1 $f | tail -n +101 | xargs -d '\n' -r zip -qd $f
            echo $(unzip -Z1 $f | wc -l)
        done
    done
done

exit 0
