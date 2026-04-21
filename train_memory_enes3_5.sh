
model_name="${2:-newwords18_2_enes3_5}"

#clear

#export CUDA_VISIBLE_DEVICES=1

logfile="logs/log_$model_name.txt"

if [ -e "$logfile" ] && [ "$1" != "-y" ]; then
    echo "The file $file already exists."
    read -p "Do you want to continue and overwrite it? (y/N): " response
    if [ "$response" = "y" ]; then
        echo "Continuing..."
        # Your code to handle the continuation here
    else
        echo "Exiting..."
        exit 1
    fi
fi

python -u train.py --model_path ./saves/model_$model_name \
    --segfiles data_cs2/*.train.seg.aligned data_cs/*.train.seg.aligned data_cs3/*.train.seg.aligned data_cs_ar/*.AR.train.seg.aligned \
    --segfiles_dev data_cs2/*.dev.seg.aligned data_cs/*.dev.seg.aligned data_cs3/*.dev.seg.aligned data_cs_ar/*.AR.dev.seg.aligned \
    --load saves/model_newwords18_2 \
    --use_memory \
    --warmup_steps 500 --learning_rate 1e-5 \
    --log_steps 10 \
    --eval_steps 1000 --use_early_stopping 1000 \
    `#--gradient_checkpointing` \
    --batch_size 16 --gradient_accumulation_steps 2 \
    --metric_for_best_model ppl_ntp_mem \
    | tee -a $logfile

