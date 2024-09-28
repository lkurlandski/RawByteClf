#!/bin/bash


# PACKINGROOT="/home/lk3591/Documents/datasets/BODMAS/diec"
# NOTPACKEDFILE="/home/lk3591/Documents/code/RawByteClf/tmp/notpackedBODMAS.txt"
# SOURCEDIR="/home/lk3591/Documents/datasets/BODMAS/binaries"
# TARGETDIR="/media/lk3591/easystore/datasets/BODMAS/ghidra/archived"

# PACKINGROOT="/home/lk3591/Documents/datasets/Assemblage/diec"
# NOTPACKEDFILE="/home/lk3591/Documents/code/RawByteClf/tmp/notpackedAssemblage.txt"
# SOURCEDIR="/home/lk3591/Documents/datasets/Assemblage/binaries"
# TARGETDIR="/media/lk3591/easystore/datasets/Assemblage/ghidra/archived"

PACKINGROOT="/home/lk3591/Documents/datasets/Windows/diec"
NOTPACKEDFILE="/home/lk3591/Documents/code/RawByteClf/tmp/notpackedWindows.txt"
SOURCEDIR="/home/lk3591/Documents/datasets/Windows/binaries"
TARGETDIR="/media/lk3591/easystore/datasets/Windows/ghidra/archived"

echo "PACKINGROOT: $PACKINGROOT"
echo "NOTPACKEDFILE: $NOTPACKEDFILE"
echo "SOURCEDIR: $SOURCEDIR"
echo "TARGETDIR: $TARGETDIR"

if [[ ! -d "$PACKINGROOT" ]]; then
    echo "Packing root directory not found: $PACKINGROOT"
    exit 1
fi
if [[ ! -d "$SOURCEDIR" ]]; then
    echo "Source directory not found: $SOURCEDIR"
    exit 1
fi
if [[ ! -d "$TARGETDIR" ]]; then
    echo "Target directory not found: $TARGETDIR"
    exit 1
fi


python -c "from src.data.detect_packing_sorel import not_packed_list; not_packed_list('$PACKINGROOT', '$NOTPACKEDFILE')"


declare -A files
total_lines=$(wc -l < "$NOTPACKEDFILE")
current_line=0
while IFS= read -r stem; do

    # if [[ $current_line -eq 10000 ]]; then
    #     break
    # fi

    f="$SOURCEDIR/$stem.exe"
    t=$(file "$f")

    if [[ -f "$f" && "$t" == *"PE32 executable"* ]]; then
        # echo "Adding   $f ($(echo $t | awk '{print $2}'))"
        h="${stem:0:2}"  # 0:1 for 0-f, 0:2 for 00-ff, etc.
        files["$h"]+="$f "
    #else
        # echo "Skipping $f ($(echo $t | awk '{print $2}'))"
    fi

    current_line=$((current_line + 1))
    percentage=$((100 * current_line / total_lines))
    echo -ne "Running file: $percentage%\r"

done < "$NOTPACKEDFILE"

for h in "${!files[@]}"; do
    file_count=$(echo "${files[$h]}" | wc -w)
    echo "Bin '$h' has $file_count files."
done

for i in {0..255}; do  # 0..15 for 0-f, 0..255 for 00-ff, etc.
    h=$(printf "%02x" "$i")  # %01x for 0-f, %02x for 00-ff, etc.
    echo "Processing $h"
    outfile="$TARGETDIR/$h.zip"
    zip -9 -j -q "$outfile" ${files[$h]}
done
