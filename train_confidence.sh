
model_name="confidence1"
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
    --train_confidence \
    --segfiles "confidence/output_conf/train.txt" \
    --segfiles_dev "confidence/output_conf/dev.txt" \
    --warmup_steps 100 --learning_rate 1e-4 \
    --log_steps 10 \
    --eval_steps 500 \
    `#--gradient_checkpointing` \
    --batch_size 32 --gradient_accumulation_steps 1 #\
    #| tee $logfile

