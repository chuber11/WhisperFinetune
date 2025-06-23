
model_name="${2:-baseline_adapt}"
load="${3:-./saves/model7}"
dataset_factor=${4:-10000}
user="${5:-admin@example.com}"

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
    `#--load $load` \
    --segfiles "../WhisperE+Phi2/data/cv.EN.train.seg.aligned" "data/voxpopuli.EN.train.seg.aligned" `#"/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.train.*.seg.aligned" "LT_CL/data_processed_$user/*.train.seg.aligned"` \
	--dataset_factors 1 1 `#$dataset_factor $dataset_factor` \
    --segfiles_dev "../WhisperE+Phi2/data/cv.EN.dev.seg.aligned" "data/voxpopuli.EN.validation.seg.aligned" `#"/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.seg.aligned" "LT_CL/data_processed_$user/*.dev.seg.aligned"` \
    --warmup_steps 0 --learning_rate 1e-5 \
    --log_steps 10 \
    --eval_steps 1000 \
    `#--gradient_checkpointing` \
    --factorization_rank 4 --factorization_only_decoder \
    --batch_size 8 --gradient_accumulation_steps 4 \
    | tee $logfile

