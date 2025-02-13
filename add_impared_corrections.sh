
version=${1:-2}

dir="/project/OML/chuber/2024/data/Doehring_corrections"

segfile="data_impairedSpeech_new$version/corrections.seg.aligned"
labelfile="data_impairedSpeech_new$version/corrections.cased"

rm -rf $segfile $labelfile

last=""
while IFS= read -r line; do
    id=`echo $line | cut -d" " -f1`
    file="$dir/$id.mp3"
    if [ -e "$file" ]; then
        echo "$id $file" >> $segfile
        text=`echo $line | cut -d" " -f2-`
        if [ -n "$last" ]; then
            echo "" >> $labelfile
        fi
        echo -n "$text" >> $labelfile
        last="\n"
    else
        echo -n $line >> $labelfile
    fi
done < "$dir/labels.cased"

#cat $dir/labels.cased | cut -d" " -f1 | awk -v dir="$dir" '{print $1, dir"/"$1 ".mp3"}' > data_impairedSpeech_new$version/corrections.seg.aligned
#cat $dir/labels.cased | cut -d" " -f2- > data_impairedSpeech_new$version/corrections.cased

#cat data_impairedSpeech_new$version/corrections.seg.aligned

