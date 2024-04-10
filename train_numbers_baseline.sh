
factorization=${1:-0} # 0: no factorization, >0: factorization rank
freeze=${2:-1} # 0: no freeze, 1: freeze encoder, 2: freeze all except embedding / proj. layer
real_dev_data=${3:-0} # 0: tts dev data, 1: real dev data
lr=${4:-1e-5}
train_emb=${5:-0} # 1: train embedding (only used when factorizing)

model_name="numbers_baseline_fact${factorization}_freeze${freeze}_real_dev_data${real_dev_data}_lr${lr}_train_emb${train_emb}"

args=""

args+=" --segfiles data_filtered/cv_filtered.*.train.seg.aligned"

if [ $factorization -gt 0 ]; then
    args+=" --factorization_rank $factorization"
    if [ $train_emb -eq 1 ]; then
        args+=" --train_embedding"
    fi
fi

if [ $freeze -eq 1 ]; then
    args+=" --freeze_encoder --factorization_only_decoder"
elif [ $freeze -eq 2 ]; then
    args+=" --only_train_embedding"
fi

if [ $real_dev_data -eq 0 ]; then
    args+=" --segfiles_dev data_numbers/llm_augment.*.dev.seg.aligned"
else
    read -p "Not implemented, waiting..." response
fi

logfile="logs/log_$model_name.txt"
echo $logfile

if [ ! -e "$logfile" ]; then
    python -u train.py \
        `#--load saves/model_numbers1_lr1e-4_df10` \
        --model_path ./saves/model_$model_name \
        --warmup_steps 100 --learning_rate $lr \
        --log_steps 25 \
        --eval_steps 50 \
        `#--gradient_checkpointing` \
        --batch_size 4 --gradient_accumulation_steps 8 \
        $args \
        | tee $logfile
fi

checkpoint=`ls -d saves/model_$model_name/checkpoint-* | head -n1`
pattern=`echo hypo_$checkpoint/* | sed "s/\//_/g"`
file_count=`find hypos/$pattern -type f -size +0c -exec  basename {} \; 2>/dev/null | wc -l`

if [ $file_count -lt 8 ]; then
    if [ $factorization -eq 0 ]; then
        bash decode_numbers_human.sh $checkpoint
        bash decode_numbers_human_train.sh $checkpoint
        bash decode_numbers.sh $checkpoint
        bash decode_cv_filtered.sh $checkpoint
    else
        bash decode_numbers_human.sh $checkpoint $checkpoint
        bash decode_numbers_human_train.sh $checkpoint $checkpoint
        bash decode_numbers.sh $checkpoint $checkpoint
        bash decode_cv_filtered.sh $checkpoint $checkpoint
    fi
elif [ $file_count -gt 8 ]; then
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

file=`ls hypos/$pattern | grep -v cv | grep human | grep -v human_train | grep EN | sed 's/EN/*/g'`
outfile=scores/${file:11:-12}.human.score
if [ ! -e $outfile ]; then
    python eval_numbers.py "$file" test "data_numbers_human/llm_augment_human.*.test.seg.aligned" > $outfile
fi

file=`ls hypos/$pattern | grep -v cv | grep -v human | grep EN | sed 's/EN/*/g'`
outfile=scores/${file:11:-6}.score
if [ ! -e $outfile ]; then
    python eval_numbers.py "$file" test > $outfile
fi

file=`ls hypos/$pattern | grep -v cv | grep human_train | grep EN | sed 's/EN/*/g'`
outfile=scores/${file:11:-17}human_train.score
if [ ! -e $outfile ]; then
    python eval_numbers.py "$file" test "data_numbers_human_train/llm_augment_human_train.*.test.seg.aligned" > $outfile
fi

