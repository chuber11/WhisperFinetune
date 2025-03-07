
model_name="${2:-segmenter2_moredata}"

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
    `#--segfiles "hypos_whisper_cv/step2/llm_augment.*.train.seg.aligned"` \
    --segfiles_dev "hypos_whisper_cv/step2/llm_augment.*.dev.seg.aligned" \
    --segfiles "hypos_whisper_cv/step2/llm_augment.*.train.seg.aligned" "hypos_whisper_cv/step2_more/llm_augment.*.seg.aligned" \
    --load saves/model_segmenter1 \
    --learning_rate 1e-5 `#--train_embedding` \
    --eval_steps 10 \
    | tee $logfile

