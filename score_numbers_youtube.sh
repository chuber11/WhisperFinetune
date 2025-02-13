
temp_file_label1=$(mktemp --suffix=".txt")
temp_file_label2=$(mktemp --suffix=".txt")

cat /export/data2/chuber/2024/YTData/test.seg.aligned | cut -d" " -f1 > $temp_file_label1
paste $temp_file_label1 /export/data2/chuber/2024/YTData/test.cased > $temp_file_label2

rm $temp_file_label1

temp_file_label3=$(mktemp --suffix=".txt")
#echo ${temp_file_label3%.txt}_lc.txt

python scripts/clean_lc.py -inf $temp_file_label2 -i 1 -lc -splitter space -o $temp_file_label3
rm $temp_file_label2
rm ${temp_file_label3}

for hypofile in `ls hypos/hypo_*youtube.txt`
do
    echo "$hypofile"

    model_name=`echo ${hypofile:11:-26}`

    echo $model_name

    temp_file1=$(mktemp --suffix=".txt")
    cat $hypofile > $temp_file1

    temp_file2=$(mktemp --suffix=".txt")
    #echo ${temp_file2%.txt}_lc.txt

    python scripts/clean_lc.py -inf $temp_file1 -i 1 -lc -splitter space -o $temp_file2
    rm "$temp_file1"
    rm "$temp_file2"

    outfile="scores/saves_${model_name}_beam4.wer"

    python scripts/wer.py --hypo ${temp_file2%.txt}_lc.txt --ref ${temp_file_label3%.txt}_lc.txt --ref-field 1 --ignore-segmentation > $outfile

    rm ${temp_file2%.txt}_lc.txt
done

rm ${temp_file_label3%.txt}_lc.txt

