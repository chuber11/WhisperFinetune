
model_name="${2:-segmenter3}"

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

python -u train_mt.py --model_path ./saves/model_$model_name \
    --segfiles data_textseg/textseg_filtered.train..tts \
    --segfiles_dev data_textseg/textseg_filtered.dev..tts \
    --eval_steps 500 \
    | tee $logfile

