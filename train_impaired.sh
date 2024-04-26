
model_name="${2:-impairedFull}"

clear

#export CUDA_VISIBLE_DEVICES=1

logfile="logs/log_$model_name.txt"

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

python -u train.py --model_path ./saves/model_$model_name \
    --segfiles "data_impairedSpeech_new/impairedSpeech.DE.train.seg.aligned" \
    --segfiles_dev "data_impairedSpeech_new/impairedSpeech.DE.dev.seg.aligned" \
    --warmup_steps 100 --learning_rate 1e-5 \
    --log_steps 10 \
    --eval_steps 10 \
    `#--gradient_checkpointing` \
    `#--factorization_rank 16` `#--factorization_only_decoder` \
    --batch_size 4 --gradient_accumulation_steps 8 \
    | tee $logfile

