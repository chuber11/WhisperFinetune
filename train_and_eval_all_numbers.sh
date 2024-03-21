
clear

GPU=$1

if [ $GPU -eq 0 ]; then
    all_args=(
        "0 0 0 0 1e-4 0"
        "0 0 1 0 1e-4 0"
        "0 0 2 0 1e-4 0"
    )
else
    all_args=(
        "0 16 0 0 1e-4 1"
        "0 16 1 0 1e-4 1"
    )
fi

for args in "${all_args[@]}"
do

read batch_weighting factorization freeze real_dev_data lr train_emb <<< "$args"

model_name="numbers_batchweighting${batch_weighting}_fact${factorization}_freeze${freeze}_real_dev_data${real_dev_data}_lr${lr}_train_emb${train_emb}"
echo $model_name

if [ ! -e "logs/log_$model_name.txt" ]; then
    bash train_numbers.sh $args 
fi

checkpoint=`ls -d saves/model_$model_name/checkpoint-*`
pattern=`echo hypo_$checkpoint/* | sed "s/\//_/g"`
#file_count=`ls hypos/$pattern | wc -l`
file_count=`find hypos/$pattern -type f -size +0c -exec basename {} \; | wc -l`

if [ $file_count -lt 6 ]; then
    if [ $factorization -eq 0 ]; then
        bash decode_numbers_human.sh $checkpoint
        bash decode_numbers.sh $checkpoint
        bash decode_cv_filtered.sh $checkpoint
    else
        bash decode_numbers_human.sh $checkpoint $checkpoint
        bash decode_numbers.sh $checkpoint $checkpoint
        bash decode_cv_filtered.sh $checkpoint $checkpoint
    fi
elif [ $file_count -gt 6 ]; then
    echo "ERROR: Found to many hypo files"
    exit
fi

for lang in EN DE
do
    file=`ls hypos/$pattern | grep cv_filtered_beam4.$lang`
    outfile=scores/${file:11:-4}.score
    if [ ! -e $outfile ]; then
        bash score_CV.sh ${file:11:-4} 2>/dev/null > $outfile
    fi
done

file=`ls hypos/$pattern | grep -v cv | grep human | grep EN | sed 's/EN/*/g'`
outfile=scores/${file:11:-12}.human.score
if [ ! -e $outfile ]; then
    python eval_numbers.py "$file" test "data_numbers_human/llm_augment_human.*.test.seg.aligned" > $outfile
fi

file=`ls hypos/$pattern | grep -v cv | grep -v human | grep EN | sed 's/EN/*/g'`
outfile=scores/${file:11:-12}.score
if [ ! -e $outfile ]; then
    python eval_numbers.py "$file" test > $outfile
fi

done

