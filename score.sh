
if [ -z "$1" ]; then
    echo "Error: No input path given"
    exit 1  # Exit with a non-zero status to indicate an error
fi

path=`echo $1 | sed "s/\//_/g"`
e_scripts="../WhisperE+Phi2/scripts"

mkdir -p hypos/$path

python $e_scripts/clean_lc.py -inf hypos/hypo_$path.txt -i 1 -lc -splitter space -o hypos/$path/hypo_postpr.txt

for lang in DE
do
	echo Language: $lang

    if [ ! -e "data/impairedSpeech.DE.test.id+lower" ]; then
        temp_file=$(mktemp)
        cat data/impairedSpeech.DE.test.cased | tr '[:upper:]' '[:lower:]' | sed 's/[[:punct:]]//g' > $temp_file
        temp_file2=$(mktemp)
        cat data/impairedSpeech.DE.test.seg.aligned | cut -d" " -f1 > $temp_file2
        paste $temp_file2 $temp_file > data/impairedSpeech.DE.test.id+lower
        rm $temp_file $temp_file2
    fi

	python $e_scripts/wer.py --hypo hypos/$path/hypo_postpr_lc.txt --ref data/impairedSpeech.DE.test.id+lower --ref-field 1 --word-stats-file hypos/$path/stats_txt_AE_$lang.txt > hypos/$path/eval_$lang
	tail -n4 hypos/$path/eval_$lang
done

