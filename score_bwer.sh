
file=${1:-hypos_memory/saves_model_newwords8_checkpoint-199000.EN.data_filtered_test_cv_memory.EN.test.allwords.0.hyp}

lang=`echo $file | cut -d"." -f2`

tempfile=$(mktemp)
cat data_filtered_test/cv_memory.$lang.test.seg.aligned | cut -d" " -f1 > $tempfile

refs=$(mktemp)
paste -d"\t" $tempfile data_filtered_test/cv_memory.$lang.test.cased > $refs
rm $tempfile $tempfile2

tempfile=$(mktemp)
cat $file | cut -d" " -f1 > $tempfile
tempfile2=$(mktemp)
cat $file | cut -d" " -f2- > $tempfile2

hyps=$(mktemp)
paste -d"\t" $tempfile $tempfile2 > $hyps
rm $tempfile $tempfile2

python score_bwer.py --refs $refs --hyps $hyps --biasing_list data_filtered_test/cv_memory.$lang.test.allwords --lenient

rm $refs $hyps

