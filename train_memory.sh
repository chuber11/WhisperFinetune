
model_name="${2:-newwords13}"

clear

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
    `#--load saves/model_newwords10` \
    --use_memory \
    --warmup_steps 500 --learning_rate 1e-4 \
    --log_steps 10 \
    --eval_steps 1000 --use_early_stopping 1000 \
    `#--gradient_checkpointing` \
    --batch_size 8 --gradient_accumulation_steps 4 \
    --metric_for_best_model ppl_ntp_mem \
    | tee -a $logfile

