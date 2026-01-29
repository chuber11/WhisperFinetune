
model_name="${2:-baseline_adapt_yodas}"

#export CUDA_VISIBLE_DEVICES=1

logfile="logs/log_$model_name.txt"

if [ -e "$logfile" ] && [ "$1" != "-y" ]; then
    echo "The file $file already exists."
    read -p "Do you want to continue and overwrite it? (y/n): " response
    if [ "$response" = "y" ]; then
        echo "Continuing..."
        # Your code to handle the continuation here
    else
        echo "Exiting..."
        exit 1
    fi
fi

python -u train.py --model_path ./saves/model_$model_name \
    --segfiles "data_yodas/EN.train.seg.aligned" \
    --segfiles_dev "data_yodas/EN.dev.seg.aligned" \
    --warmup_steps 500 --learning_rate 5e-5 \
    --log_steps 10 \
    --eval_steps 500 \
    `#--gradient_checkpointing` \
    --factorization_rank 8 `#--factorization_only_decoder` \
    --batch_size 4 --gradient_accumulation_steps 4 \
    | tee $logfile

