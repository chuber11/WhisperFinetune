
model_name="${2:-numbers16}"

clear

#export CUDA_VISIBLE_DEVICES=1

logfile="logs/log_$model_name.txt"
echo $logfile

if [ -e "$logfile" ] && [ "$1" != "-y" ]; then
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
    --segfiles "data/cv.*.train.seg.aligned" "data_numbers/llm_augment.*.train.seg.aligned" \
    --dataset_factors 1 10 \
    --segfiles_dev "data/cv.*.dev.seg.aligned" "data_numbers/llm_augment.*.dev.seg.aligned" \
    --warmup_steps 100 --learning_rate 1e-5 \
    --log_steps 10 \
    --eval_steps 200 \
    `#--gradient_checkpointing` \
    --factorization_rank 1 `#--factorization_only_decoder` \
    --train_embedding \
    --batch_size 4 --gradient_accumulation_steps 8 \
    | tee -a $logfile

