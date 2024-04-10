
batch_weighting=${1:-0} # 0: finetuning, >0: batch weighting factor for new data
factorization=${2:-0} # 0: no factorization, >0: factorization rank
freeze=${3:-0} # 0: no freeze, 1: freeze encoder, 2: freeze all except embedding / proj. layer
real_dev_data=${4:-0} # 0: tts dev data, 1: real dev data
lr=${5:-1e-5}
train_emb=${6:-0} # 1: train embedding (only used when factorizing)

model_name="numbers_batchweighting${batch_weighting}_fact${factorization}_freeze${freeze}_real_dev_data${real_dev_data}_lr${lr}_train_emb${train_emb}"

args=""

if [ $batch_weighting -eq 0 ]; then
    args+=" --segfiles data_numbers/llm_augment.*.train.seg.aligned"
else
    args+=" --segfiles data_filtered/cv_filtered.*.train.seg.aligned data_numbers/llm_augment.*.train.seg.aligned --dataset_factors 1 1 $batch_weighting $batch_weighting"
fi

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

if [ -e "$logfile" ]; then
    echo "The file $logfile already exists."
    read -p "Do you want to continue and overwrite it? (y/n): " response
    if [ "$response" = "y" ]; then
        echo "Continuing..."
        # Your code to handle the continuation here
    else
        echo "Exiting..."
        exit 1
    fi
fi

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

