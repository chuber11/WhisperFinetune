
model_name="${2:-bw}"
load="${3:-./saves/model7/checkpoint-7300}"
dataset_factor=10000

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
    --load $load \
    --segfiles "../WhisperE+Phi2/data/cv.*.train.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.train.*.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.test.train.*.seg.aligned" \
	--dataset_factors 1 $dataset_factor $dataset_factor \
    --segfiles_dev "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.seg.aligned"\
    --warmup_steps 0 --learning_rate 1e-5 \
    --log_steps 10 \
    --eval_steps 10 \
    `#--gradient_checkpointing` \
    --factorization_rank 16 `#--factorization_only_decoder` \
    --batch_size 4 --gradient_accumulation_steps 8 \
    | tee $logfile

