
version=4
model_name=5_new${version}_v3

clear
set -e

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
    --segfiles "data_impairedSpeech_new$version/impairedSpeech.DE.train.seg.aligned" "data_impairedSpeech_new$version/corrections.seg.aligned" \
    --dataset_factors 1 4 \
    --segfiles_dev "data_impairedSpeech_new$version/impairedSpeech.DE.dev.seg.aligned" \
    --warmup_steps 100 --learning_rate 1e-5 \
    --log_steps 10 --model_name data_impairedSpeech \
    --eval_steps 10 --use_early_stopping 20 \
    `#--gradient_checkpointing` \
    `#--factorization_rank 16` `#--factorization_only_decoder` \
    --batch_size 4 --gradient_accumulation_steps 64 \
    | tee -a $logfile

cp data_impairedSpeech/*.json saves/model_$model_name/checkpoint-*
ct2-transformers-converter --model saves/model_$model_name/checkpoint-* --output_dir saves/model_$model_name/ct2 --copy_files tokenizer.json preprocessor_config.json

